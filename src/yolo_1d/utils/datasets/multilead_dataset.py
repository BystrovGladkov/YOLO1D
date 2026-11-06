"""Датасет режима MC (M6): пример — вся запись LUDB, 12 отведений, разметка каждого отведения.

Формат (data/prepare_ludb_mc.py, data/make_splits_mc.py, конфиг `data/splits/fold_<k>_mc.yaml`, `mode: multi_lead`):
    <split>_data/<запись>.pt     torch.float32 (12, 5000), Z-score по отведению (как в режиме S)
    <split>_labels/<запись>.txt  строки "class lead center width" (lead — индекс 0..11)
    zones (leads.csv)            запись × отведение: annot_start, annot_end, usable

Решения R5 (docs/research/R5_multilead.md §2.1–2.2):
- маска неразмеченных краёв (`EdgeMask`) применяется к **каждому отведению по его зоне**; запас δ
  тянется **независимо для каждого отведения** (на train он случайный, на eval фиксированный);
- непригодные отведения (usable = 0) остаются во входе, но помечаются в `valid` — их нет в лоссе и метриках.

Выход `__getitem__`: (signal (12, L), cls (n, 1), box (n, 2), lead (n, 1), valid (12,) bool).
`collate_fn` добавляет к целям поля `lead` (n,) и `valid` (B, 12).

Аугментации (A3): сигнальные — `transform` (pipeline `ecg_multi`, метки (n, 2) [class, lead]); склейка
двух записей X1/X2 (`splice_prob`, `beat_splice_prob`) — с общими для всех отведений точками разреза;
группа L (`lead_transform`, меняет и `valid`) — после сигнальных. Мозаика и mix-up в MC не поддерживаются.
"""
import csv
import os
import random
from typing import List, Optional, Tuple

import numpy
import torch
import yaml

from yolo_1d.utils.datasets.dataset import Dataset

N_LEADS = 12


class MultiLeadDataset(Dataset):
    """Записи LUDB 12 × 5000 с поотведенческой разметкой. Интерфейс как у `Dataset` (режим S)."""

    n_leads = N_LEADS

    def __init__(self, filename, transform, type, edge_mask=None, splice_prob=0, beat_splice_prob=0,
                 lead_transform=None, **augmentations):
        # мозаика (склейка с ресайзом) и mix-up в MC не поддерживаются (параметры вроде mix_up_alpha не важны)
        unsupported = {k: v for k, v in augmentations.items() if k.endswith("_prob") and v}
        if unsupported:
            raise NotImplementedError(f"в режиме MC не поддерживаются: {sorted(unsupported)}")
        super().__init__(filename, transform, type, edge_mask=edge_mask, splice_prob=splice_prob,
                         beat_splice_prob=beat_splice_prob)
        self.lead_transform = lead_transform

    # ------------------------------------------------------------------ чтение
    @staticmethod
    def _read_zones(path) -> Optional[dict]:
        """leads.csv → {запись: {"zones": [(start, end) | None] * 12, "valid": [bool] * 12}}."""
        if path is None:
            return None
        out = {}
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rec = out.setdefault(row["record"], {"zones": [None] * N_LEADS, "valid": [False] * N_LEADS})
                k = int(row["lead_index"])
                rec["zones"][k] = (int(row["annot_start"]), int(row["annot_end"]))
                rec["valid"][k] = row["usable"] == "1"
        return out

    @staticmethod
    def _read_label_file(label_file):
        """Разметка записи (n, 4) [class, lead, center, width] или None, если она некорректна."""
        if not os.path.isfile(label_file):
            return numpy.zeros((0, 4), dtype=numpy.float32)
        with open(label_file) as f:
            rows = [x.split() for x in f.read().strip().splitlines() if x.strip()]
        if not rows:
            return numpy.zeros((0, 4), dtype=numpy.float32)
        label = numpy.array(rows, dtype=numpy.float32)
        if label.shape[1] != 4 or (label < 0).any() or (label[:, 2:] > 1).any() or (label[:, 1] >= N_LEADS).any():
            return None
        return numpy.unique(label, axis=0)

    def sample_id(self, i: int) -> str:
        return os.path.basename(self.filenames[i]).rsplit(".", 1)[0]

    def valid(self, i: int) -> torch.Tensor:
        """(12,) bool: отведения записи, которые входят в лосс и метрики."""
        if self.zones is None:
            return torch.ones(N_LEADS, dtype=torch.bool)
        return torch.tensor(self.zones[self.sample_id(i)]["valid"], dtype=torch.bool)

    def load_signal(self, i) -> torch.Tensor:
        """Сигнал записи (12, 5000); каждое отведение маскируется по своей зоне с независимым δ."""
        with open(self.filenames[i], "rb") as f:
            data = torch.load(f, map_location="cpu", weights_only=False).float()
        if self.edge_mask is not None and self.zones is not None:
            zones = self.zones[self.sample_id(i)]["zones"]
            data = torch.cat([self.edge_mask(data[k:k + 1], zones[k]) for k in range(data.shape[0])], dim=0)
        return data

    # ------------------------------------------------------------------ пример и батч
    def __getitem__(self, index) -> Tuple[torch.Tensor, ...]:
        index = self.indices[index]
        # случайное число тянется только при включённой склейке (как в режиме S)
        r = random.random() if (self.splice_prob or self.beat_splice_prob) else 1.0
        if r < self.splice_prob:
            data, label, valid = self.load_splice(index, aligned=False)
        elif r < self.splice_prob + self.beat_splice_prob:
            data, label, valid = self.load_splice(index, aligned=True)
        else:
            data, label, valid = self.load_signal(index), self.labels[self.filenames[index]], self.valid(index)
        nl = len(label)
        box = torch.from_numpy(label[:, 2:4].copy()) if nl else torch.zeros((0, 2))
        # трансформации получают метки (n, 2) [class, lead]: отбор/перестановка волн сохраняет отведение
        meta = torch.from_numpy(label[:, 0:2].copy()) if nl else torch.zeros((0, 2))
        data, box, meta = self.transform.forward(data, box, meta)
        if self.lead_transform is not None:
            data, box, meta, valid = self.lead_transform(data, box, meta, valid)
        return data, meta[:, 0:1], box, meta[:, 1:2], valid

    @staticmethod
    def collate_fn(batch):
        samples, cls, box, lead, valid = zip(*batch)
        idx = torch.cat([torch.full((len(c),), i, dtype=torch.float32) for i, c in enumerate(cls)])
        targets = {"cls": torch.cat(cls, dim=0),
                   "box": torch.cat(box, dim=0),
                   "idx": idx,
                   "lead": torch.cat(lead, dim=0).view(-1),
                   "valid": torch.stack(valid, dim=0)}
        return torch.stack(samples, dim=0), targets

    # ------------------------------------------------------------------ X1/X2: склейка двух записей
    def load_splice(self, index, aligned: bool):
        """Склейка начала записи `index` с концом другой записи; разрез **один для всех 12 отведений**.

        Точки разреза — как в режиме S (`Dataset._cut_points`), но по кластерам волн всех отведений
        (объединение границ волны по отведениям): X2 режет только в паузах после T, общих для всех отведений.
        Изолиния выравнивается по каждому отведению отдельно. valid результата — отведения, пригодные в обеих
        записях (у непригодного отведения в одной из частей были бы неразмеченные волны).
        Returns: (signal, label (m, 4), valid (12,)); если разрезов нет — исходная запись.
        """
        other = random.choice(self.indices)
        a, b = self.load_signal(index), self.load_signal(other)
        label_a = self.labels[self.filenames[index]].copy()
        label_b = self.labels[self.filenames[other]].copy()
        valid = self.valid(index) & self.valid(other)
        la, lb = a.shape[-1], b.shape[-1]
        span_a, span_b = self._active_span(a), self._active_span(b)
        cuts_a = self._cut_points(self._merged(label_a, la), la, aligned, span_a)
        cuts_b = self._cut_points(self._merged(label_b, lb), lb, aligned, span_b)
        if not cuts_a or not cuts_b or other == index:
            return a, label_a, self.valid(index)
        cut_a = random.choice(cuts_a)
        target = cut_a + lb - la
        cut_b = min(cuts_b, key=lambda c: abs(c - target)) if aligned else int(numpy.clip(target, *span_b))
        head, tail = a[:, :cut_a], b[:, cut_b:].clone()
        if aligned:
            offset = head[:, -25:].median(dim=-1, keepdim=True).values - tail[:, :25].median(dim=-1, keepdim=True).values
            tail = torch.where(tail != 0, tail + offset, tail)
        data = torch.cat([head, tail], dim=-1)
        new_length = data.shape[-1]
        label = numpy.concatenate([self._crop_leads(label_a, la, 0, cut_a, 0, new_length),
                                   self._crop_leads(label_b, lb, cut_b, lb, cut_a - cut_b, new_length)], axis=0)
        label = label[valid.numpy()[label[:, 1].astype(int)]] if len(label) else label
        return data, label, valid

    @staticmethod
    def _merged(label: numpy.ndarray, length: int) -> numpy.ndarray:
        """Разметка записи (n, 4) → кластеры волн по отведениям (m, 3) [class, center, width]."""
        if len(label) == 0:
            return numpy.zeros((0, 3), dtype=numpy.float32)
        from yolo_1d.utils.transforms.ecg_multilead_transforms import wave_clusters
        left, right, cls, _ = wave_clusters(torch.from_numpy(label[:, 2:4]), torch.from_numpy(label[:, 0:2]), length)
        return numpy.stack([cls, (left + right) / 2 / length, (right - left) / length], axis=1).astype(numpy.float32)

    @staticmethod
    def _crop_leads(label: numpy.ndarray, length: int, start: float, end: float, shift: float, new_length: int):
        """`Dataset._crop_labels` для разметки (n, 4) [class, lead, center, width]: отведение сохраняется."""
        if len(label) == 0:
            return numpy.zeros((0, 4), dtype=numpy.float32)
        left = (label[:, 2] - label[:, 3] / 2) * length
        right = (label[:, 2] + label[:, 3] / 2) * length
        new_l, new_r = left.clip(start, end), right.clip(start, end)
        keep = (new_r - new_l) >= 0.5 * (right - left)
        new_l, new_r = new_l[keep] + shift, new_r[keep] + shift
        out = numpy.stack([label[keep, 0], label[keep, 1], (new_l + new_r) / 2 / new_length,
                           (new_r - new_l) / new_length], axis=1)
        return out.astype(numpy.float32)

    # ------------------------------------------------------------------ для метрик
    def lead_ids(self) -> List[str]:
        """ID пригодных отведений `<запись>_<отведение>` в порядке строк выхода модели при последовательном
        проходе датасета (запись за записью, отведения 0..11, непригодные пропущены) — как ID режима S."""
        from yolo_1d.utils.datasets.ludb import LEADS
        return [f"{self.sample_id(i)}_{LEADS[k]}" for i in range(self.n) for k in range(N_LEADS) if self.valid(i)[k]]

    def lead_labels(self, i: int, k: int) -> numpy.ndarray:
        """Разметка отведения k записи i в формате режима S (m, 3) [class, center, width]."""
        label = self.labels[self.filenames[i]]
        return label[label[:, 1] == k][:, [0, 2, 3]]


def dataset_class(filename) -> type:
    """Класс датасета по ключу `mode` yaml-конфига: `multi_lead` → MultiLeadDataset, иначе Dataset."""
    with open(filename, encoding="utf-8") as f:
        mode = (yaml.safe_load(f) or {}).get("mode", "single_lead")
    return MultiLeadDataset if mode == "multi_lead" else Dataset
