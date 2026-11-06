from typing import Dict, Optional, Tuple, Literal
import csv
import math
import os
import random
import warnings

import numpy
import torch
from torch.utils import data

import yaml

from yolo_1d.utils.constants import INPUT_LENGTH

# форматы датасета (ключ `format` в yaml):
#   "yolo_pt" (по умолчанию) — тензоры .pt, разметка YOLO "cls x_center width" в долях длины;
#   "ecr_npy" — ECRv1: сигналы .npy (int16, 15 x L), разметка "type begin end" в отсчётах,
#               классы перекодируются по `class_map` (см. yolo_1d.utils.datasets.ecr)
FORMAT_YOLO_PT = "yolo_pt"
FORMAT_ECR_NPY = "ecr_npy"
_ECR_CACHE_VERSION = 1


class Dataset(data.Dataset):
    
    def __init__(self, filename, transform, type: Literal["train", "val", "test"],
                 mosaic_prob=0,
                 mix_up_prob=0,
                 mix_up_alpha=20,
                 edge_mask=None,
                 splice_prob=0,
                 beat_splice_prob=0):
        """
            edge_mask
                Callable (signal, annotated) -> signal, например `transforms.EdgeMask`: маскирует
                неразмеченные края записи при загрузке. Применяется, только если в yaml задан ключ
                `zones` (CSV с колонками sample_id, annot_start, annot_end); иначе сигнал не меняется.
            splice_prob, beat_splice_prob
                Вероятности склейки двух записей (ЭКГ-аугментации X1 и X2, см. `load_splice`).
        """
        super().__init__()
        self.transform = transform
        self.edge_mask = edge_mask
        self.mosaic_prob = mosaic_prob
        self.splice_prob = splice_prob
        self.beat_splice_prob = beat_splice_prob
        self.mix_up_prob = mix_up_prob
        self.mix_up_alpha = mix_up_alpha
        self.filename = filename
        self.type = type
        # Загружаем конфигурацию датасета
        self._parse_yaml()
        # Загружаем пути к данным
        self._load_data()
        # Загружаем отметки
        self._load_labels()
        
        self.n = len(self.filenames)    # Количество данных
        self.indices = range(self.n)    # Индексы данных
        
    def _parse_yaml(self):
        """ 
            Считывает их файла настроек:
            - путь к данным и разметке
            - Информацию о названиях классов и их количестве
            - (необязательно) `<type>_list` — файл со списком ID примеров сплита
            - (необязательно) `path` — базовая папка для относительных путей, задаётся
              относительно самого yaml-файла. Без `path` относительные пути, как и раньше,
              отсчитываются от текущей рабочей папки.
            - (необязательно) `format` — "yolo_pt" (по умолчанию) или "ecr_npy"; для "ecr_npy" —
              `class_map` (id класса в разметке → номер класса сети) и `preprocess_cache`
              (папка для кэша разметки; папка датасета может быть только для чтения).
        """
        with open(self.filename, encoding="utf-8") as f:
            data = yaml.safe_load(f) # Парсим yaml - файл
        self.format = data.get("format", FORMAT_YOLO_PT)
        if self.format not in (FORMAT_YOLO_PT, FORMAT_ECR_NPY):
            raise ValueError(f"{self.filename}: неизвестный format {self.format!r}")
        self.class_map: Optional[Dict[int, int]] = None
        self.cache_dir: Optional[str] = None
        if self.format == FORMAT_ECR_NPY:
            if "class_map" not in data:
                raise ValueError(f"{self.filename}: для format ecr_npy нужен class_map")
            self.class_map = {int(k): int(v) for k, v in data["class_map"].items()}
            self.cache_dir = data.get("preprocess_cache")

        # Загружаем названия классов
        self.class_names = [i for i in data["names"].values()]
        # Считаем количество классов
        self.num_classes = len(self.class_names)
        # базовая папка для относительных путей
        base = data.get("path")
        if base is not None and not os.path.isabs(base):
            base = os.path.join(os.path.dirname(os.path.abspath(self.filename)), base)
        resolve = lambda p: p if (p is None or base is None or os.path.isabs(p)) else os.path.normpath(os.path.join(base, p))
        # Считываем путь к данным
        self.data_path = resolve(data[f"{self.type}_data"])
        # Считываем путь к разметке
        self.labels_path = resolve(data[f"{self.type}_labels"])
        # Список ID примеров сплита (имена файлов без расширения); None — вся папка
        self.list_path = resolve(data.get(f"{self.type}_list"))
        # Границы размеченной зоны (ID примера -> (первый onset, последний offset)) для edge_mask
        self.zones = self._read_zones(resolve(data.get("zones")))

    @staticmethod
    def _read_zones(path):
        """CSV (sample_id, annot_start, annot_end) -> {ID: (annot_start, annot_end)}; None — зон нет."""
        if path is None:
            return None
        with open(path, encoding="utf-8", newline="") as f:
            return {row["sample_id"]: (int(row["annot_start"]), int(row["annot_end"])) for row in csv.DictReader(f)}
        
    def _load_data(self):
        """
            Saves filenames of files with data.

            Без списка берутся все файлы папки данных; со списком `<type>_list` — только
            перечисленные ID (имя файла без расширения), в порядке списка.
        """
        files = sorted(os.listdir(self.data_path))
        if self.format == FORMAT_ECR_NPY:
            files = [f for f in files if f.endswith(".npy")]
        if self.list_path is None:
            self.filenames = [os.path.join(self.data_path, file) for file in files]
            return
        by_id = {file.rsplit(".", 1)[0]: file for file in files}
        with open(self.list_path, encoding="utf-8") as f:
            ids = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise FileNotFoundError(f"{len(missing)} примеров из {self.list_path} нет в {self.data_path}: {missing[:5]}")
        self.filenames = [os.path.join(self.data_path, by_id[i]) for i in ids]

    @staticmethod
    def _read_label_file(label_file):
        """
            Reads labels of one sample. Returns array (n, 3) or None if labels are invalid.
        """
        try:
            # read labels
            if os.path.isfile(label_file):
                with open(label_file) as f:
                    label = [x.split() for x in f.read().strip().splitlines() if len(x)]
                    label = numpy.array(label, dtype=numpy.float32)
                nl = len(label)
                if nl:
                    # simple asserts,
                    # TODO: better no to raise error but fix problem and show message
                    assert (label >= 0).all()   # all parts of label must be positive
                    assert label.shape[1] == 3  # 3 parts: class, center, width
                    assert (label[:, 1:] <= 1).all() # check labels not exceed corners
                    _, i = numpy.unique(label, axis=0, return_index=True)
                    if len(i) < nl:  # check for duplicates
                        label = label[i]  # delete duplicates
                        # TODO: show amount of duplicates deleted
                else:# no labels
                    label = numpy.zeros((0, 3),dtype=numpy.float32)
            else: # labels file not found (not a file)
                label = numpy.zeros((0, 3),dtype=numpy.float32)
        except FileNotFoundError: # labels file not found
            label = numpy.zeros((0, 3), dtype=numpy.float32)
        except AssertionError:
            return None
        return label

    @staticmethod
    def _read_ecr_label_file(label_file, length: int, class_map: Dict[int, int]):
        """Разметка ECRv1 ("type begin end" в отсчётах) → (array (n, 3) [cls, x_center, width], список проблем,
        число удалённых дубликатов).

        Класс перекодируется по `class_map`, координаты нормируются на фактическую длину сигнала `length`.
        Отметки с неизвестным классом, `end <= begin` или выходом за [0, length] отбрасываются и
        попадают в список проблем; точные дубликаты удаляются. Нет файла — пустая разметка.
        """
        empty = numpy.zeros((0, 3), dtype=numpy.float32)
        if not os.path.isfile(label_file):
            return empty, [], 0
        rows, problems = [], []
        with open(label_file, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                parts = line.split()
                if not parts:
                    continue
                try:
                    cls, begin, end = int(parts[0]), float(parts[1]), float(parts[2])
                except (ValueError, IndexError):
                    problems.append(f"строка {n}: не разобрана {line.strip()!r}")
                    continue
                if cls not in class_map:
                    problems.append(f"строка {n}: неизвестный класс {cls}")
                elif not (0 <= begin < end <= length):
                    problems.append(f"строка {n}: границы ({begin:g}, {end:g}) вне [0, {length}] или пустые")
                else:
                    rows.append((class_map[cls], (begin + end) / 2 / length, (end - begin) / length))
        if not rows:
            return empty, problems, 0
        label = numpy.unique(numpy.array(rows, dtype=numpy.float32), axis=0)
        return label, problems, len(rows) - len(label)

    def _ecr_labels_cache_path(self) -> str:
        """Кэш разметки ECRv1: в `preprocess_cache` (если задан), иначе в папке разметки."""
        if self.list_path is not None:
            return self.list_path + ".ecr_labels.cache"
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            return os.path.join(self.cache_dir, f"{self.type}_ecr_labels.cache")
        return os.path.join(self.labels_path, "ecr_labels.cache")

    def _load_ecr_labels(self):
        """Загружает разметку ECRv1 с кэшем.

        Кэш хранит {"meta": {...}, "labels": {путь сигнала: (n, 3)}} и пересоздаётся, если изменились
        класс-карта, версия формата кэша, набор файлов или папка разметки изменена позже кэша.
        Длина сигнала берётся из заголовка .npy (без чтения данных). Сводка отброшенных отметок —
        одним предупреждением.
        """
        path = self._ecr_labels_cache_path()
        meta = {"version": _ECR_CACHE_VERSION, "class_map": self.class_map}
        fresh = os.path.exists(path) and os.path.getmtime(path) >= os.path.getmtime(self.labels_path)
        cache = torch.load(path, weights_only=False) if fresh else {}
        labels = cache.get("labels", {}) if cache.get("meta") == meta else {}
        if not all(f in labels for f in self.filenames):
            labels, problems, duplicates = {}, [], 0
            for filename in self.filenames:
                length = numpy.load(filename, mmap_mode="r").shape[-1]
                name = os.path.basename(filename).rsplit(".", 1)[0] + ".txt"
                label, file_problems, file_dups = self._read_ecr_label_file(
                    os.path.join(self.labels_path, name), length, self.class_map)
                labels[filename] = label
                problems += [f"{name}: {p}" for p in file_problems]
                duplicates += file_dups
            if problems or duplicates:
                msg = f"{self.type}: отброшено некорректных отметок: {len(problems)}, удалено дубликатов: {duplicates}"
                warnings.warn(msg + ("; например: " + "; ".join(problems[:3]) if problems else ""))
            torch.save({"meta": meta, "labels": labels}, path)
        self.labels = {f: labels[f] for f in self.filenames}

    def _load_labels(self):
        """
            Loads labels from cache and creates cache if it does not exist.

            Кэш хранится рядом со списком сплита (`<type>_list` + ".labels.cache") или, без
            списка, в папке разметки (`labels.cache`). Кэш пересоздаётся, если в нём нет
            какого-либо из текущих файлов или папка разметки изменена позже кэша
            (например, данные подготовлены заново). Примеры с некорректной разметкой
            (значение None в кэше) исключаются из `filenames`, чтобы `labels` и `filenames` совпадали.
            Для format ecr_npy — отдельная ветка `_load_ecr_labels`.
        """
        if self.format == FORMAT_ECR_NPY:
            return self._load_ecr_labels()
        if self.list_path is not None:
            path = self.list_path + ".labels.cache"
        else:
            path = os.path.join(self.labels_path, "labels.cache")
        fresh = os.path.exists(path) and os.path.getmtime(path) >= os.path.getmtime(self.labels_path)
        cache = torch.load(path, weights_only=False) if fresh else {}
        if not all(filename in cache for filename in self.filenames):
            # cache not found or stale
            cache = {}
            for filename in self.filenames:
                # get filenames
                name = os.path.basename(filename).rsplit(".", 1)[0] + ".txt"
                cache[filename] = self._read_label_file(os.path.join(self.labels_path, name))
            # save cache
            torch.save(cache, path)
        # порядок labels совпадает с filenames (важно для весов сэмплера в datasets.sampling)
        self.labels = {f: cache[f] for f in self.filenames if cache[f] is not None}
        self.filenames = [f for f in self.filenames if f in self.labels]
    
    def load_signal(self, i) -> torch.Tensor:
        """
            Reads signal with index i from dataset.

            Returns
            ---
            signal
                Signal from dataset with index i
        """
        if self.format == FORMAT_ECR_NPY:
            # ECRv1: (15, L) int16 → float32; предобработка (ICA, пороги, нормировка) — в трансформе
            return torch.from_numpy(numpy.load(self.filenames[i]).astype(numpy.float32))
        with open(self.filenames[i], "rb") as f:
            data = torch.load(f, map_location="cpu", weights_only=False)
        data = data.float()
        if self.edge_mask is not None and self.zones is not None:
            sample_id = os.path.basename(self.filenames[i]).rsplit(".", 1)[0]
            data = self.edge_mask(data, self.zones.get(sample_id))
        return data
    
    def load_mosaic(self, index):
        # resulting labels
        labels = []
        # resulting signal
        signals = []
        # select second signal
        indices = [index] + random.choices(self.indices, k=1)
        # shuffle signals
        random.shuffle(indices)
        
        length = []
        for i, index in enumerate(indices):
            signal = self.load_signal(index)
            # save signal
            signals.append(signal)
            # save length of signal
            length.append(signal.size(1))
            # save labels
            labels.append(self.labels[self.filenames[index]].copy())
        
        length = numpy.array(length)
        full_length = length.sum()
        offset = numpy.concatenate([[0], length.cumsum()[:-1]])
        
        for index in range(0, length.shape[0]):
            # calculate a coefficient to convert labels
            scale = length[index] / full_length
            # calculate center of labels
            labels[index][:, 1] = offset[index] / full_length + labels[index][:, 1] * scale
            # calculate width of labels
            labels[index][:, 2] = labels[index][:, 2] * scale
            
        labels = numpy.concatenate(labels, axis=0)
        signals = torch.concat(signals, dim=1)
        
        return signals, labels
    
    # ------------------------------------------------------------------ X1/X2: склейка двух записей (ЭКГ, A1)
    @staticmethod
    def _cut_points(label: numpy.ndarray, length: int, aligned: bool, span) -> list:
        """Допустимые точки разреза записи (в отсчётах).

        aligned=True (X2): середины пауз после T-волны (между её концом и началом следующей волны,
        пауза ≥ 10 отсчётов) — разрез не проходит через волну. aligned=False (X1): любая точка
        активной части.
        """
        if not aligned:
            return list(range(span[0] + 1, span[1] - 1))
        if len(label) < 2:
            return []
        left = (label[:, 1] - label[:, 2] / 2) * length
        right = (label[:, 1] + label[:, 2] / 2) * length
        order = numpy.argsort(left)
        cls, left, right = label[order, 0], left[order], right[order]
        cuts = []
        for k in range(len(order) - 1):
            if int(cls[k]) == 2 and left[k + 1] - right[k] >= 10:
                cuts.append(int((right[k] + left[k + 1]) / 2))
        return cuts

    @staticmethod
    def _crop_labels(label: numpy.ndarray, length: int, start: float, end: float, shift: float, new_length: int):
        """Разметка части [start, end) записи, сдвинутой на shift, в долях new_length. Волны, видимые
        на отрезке меньше чем наполовину, отбрасываются, остальные обрезаются по его краям."""
        if len(label) == 0:
            return numpy.zeros((0, 3), dtype=numpy.float32)
        left = (label[:, 1] - label[:, 2] / 2) * length
        right = (label[:, 1] + label[:, 2] / 2) * length
        new_l, new_r = left.clip(start, end), right.clip(start, end)
        keep = (new_r - new_l) >= 0.5 * (right - left)
        new_l, new_r = new_l[keep] + shift, new_r[keep] + shift
        out = numpy.stack([label[keep, 0], (new_l + new_r) / 2 / new_length, (new_r - new_l) / new_length], axis=1)
        return out.astype(numpy.float32)

    def load_splice(self, index, aligned: bool):
        """Склейка начала записи `index` с концом другой случайной записи.

        aligned=False (X1, «наивная мозаика»): разрезы в случайных точках активной части, без
        выравнивания — на стыке возможны разрыв ритма и ступенька изолинии.
        aligned=True (X2, beat-splice): разрезы в паузах после T-волн, изолиния второй части
        сдвигается к уровню первой на стыке, обнулённые края (маска) не трогаются.
        Разрез второй записи выбирается так, чтобы длина результата была близка к исходной.
        Если допустимых разрезов нет, возвращается исходная запись.
        """
        other = random.choice(self.indices)
        a, b = self.load_signal(index), self.load_signal(other)
        label_a = self.labels[self.filenames[index]].copy()
        label_b = self.labels[self.filenames[other]].copy()
        la, lb = a.shape[-1], b.shape[-1]
        span_a = self._active_span(a)
        span_b = self._active_span(b)
        cuts_a = self._cut_points(label_a, la, aligned, span_a)
        cuts_b = self._cut_points(label_b, lb, aligned, span_b)
        if not cuts_a or not cuts_b or other == index:
            return a, label_a
        cut_a = random.choice(cuts_a)
        # разрез второй записи — ближайший к позиции, при которой длина результата = la
        target = cut_a + lb - la
        cut_b = min(cuts_b, key=lambda c: abs(c - target)) if aligned else int(numpy.clip(target, *span_b))
        head, tail = a[:, :cut_a], b[:, cut_b:].clone()
        if aligned:
            # выравнивание изолинии: медиана 25 отсчётов слева и справа от стыка (в паузе после T)
            offset = head[:, -25:].median(dim=-1, keepdim=True).values - tail[:, :25].median(dim=-1, keepdim=True).values
            active = tail != 0
            tail = torch.where(active, tail + offset, tail)
        data = torch.cat([head, tail], dim=-1)
        new_length = data.shape[-1]
        label = numpy.concatenate([
            self._crop_labels(label_a, la, 0, cut_a, 0, new_length),
            self._crop_labels(label_b, lb, cut_b, lb, cut_a - cut_b, new_length),
        ], axis=0)
        return data, label

    @staticmethod
    def _active_span(signal: torch.Tensor):
        nz = torch.nonzero(signal.abs().sum(0) > 0).flatten()
        return (0, signal.shape[-1]) if len(nz) == 0 else (int(nz[0]), int(nz[-1]) + 1)

    def load_mix_up(self, data1, target_box1, target_cls1):
        # load another signal
        index = random.choices(self.indices, k=1)[0]
        # load signal
        data2 = self.load_signal(index)
            # load labels
        label = self.labels[self.filenames[index]].copy()

        nl = len(label)  # number of labels
        cls = label[:, 0:1]
        box = label[:, 1:3]
        # convert labels to torch tensors
        target_cls2 = torch.zeros((nl, 1))
        target_box2 = torch.zeros((nl, 2))
        if nl:
            target_cls2 = torch.from_numpy(cls)
            target_box2 = torch.from_numpy(box)
        # perform transform
        data2, target_box2, target_cls2 = self.transform.forward(data2, target_box2, target_cls2)
        
        lamb = float(numpy.random.beta(self.mix_up_alpha, self.mix_up_alpha, 1))
        # mix the signals
        data1 = data1 * lamb + data2 * (1 - lamb)
        target_box1 = torch.concat([target_box1, target_box2], axis=0)
        target_cls1 = torch.concat([target_cls1, target_cls2], axis=0)
        return data1, target_box1, target_cls1
        
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
            Extracts signal and labels from dataset.
        """
        index = self.indices[index]
        
        # склейка записей (X1/X2) или mosaic
        # случайное число тянется только при включённой склейке: поток случайных чисел прежних конфигов не меняется
        r = random.random() if (self.splice_prob or self.beat_splice_prob) else 1.0
        if r < self.splice_prob:
            data, label = self.load_splice(index, aligned=False)
        elif r < self.splice_prob + self.beat_splice_prob:
            data, label = self.load_splice(index, aligned=True)
        elif self.mosaic_prob and random.random() < self.mosaic_prob:
            data, label = self.load_mosaic(index)
        else:
            # load signal
            data = self.load_signal(index)
            # load labels
            label = self.labels[self.filenames[index]].copy()

        nl = len(label)  # number of labels
        cls = label[:, 0:1]
        box = label[:, 1:3]
        # convert labels to torch tensors
        target_cls = torch.zeros((nl, 1))
        target_box = torch.zeros((nl, 2))
        if nl:
            target_cls = torch.from_numpy(cls)
            target_box = torch.from_numpy(box)
        # perform transform
        data, target_box, target_cls = self.transform.forward(data, target_box, target_cls)
        
        if self.mix_up_prob and random.random() < self.mix_up_prob:
            data, target_box, target_cls = self.load_mix_up(data, target_box, target_cls)
        
        nl = target_box.size(0)

        return data, target_cls, target_box, torch.zeros(nl)

    def __len__(self):
        return self.n

    @staticmethod
    def collate_fn(batch):
        samples, cls, box, indices = zip(*batch)

        cls = torch.cat(cls, dim=0)
        box = torch.cat(box, dim=0)

        new_indices = list(indices)
        for i in range(len(indices)):
            new_indices[i] += i
        indices = torch.cat(new_indices, dim=0)

        targets = {'cls': cls,
                   'box': box,
                   'idx': indices}
        return torch.stack(samples, dim=0), targets
