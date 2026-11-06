"""Абляция аугментаций на ECRv1 (вихретоковые дефектограммы): конфиги, обучение, сводка (TASKS.md, D7).

Использование:
    python run_ec_ablation.py make-configs                    # experiments/ec_ablation/configs/{A0..A8,IN_*}.yaml
    python run_ec_ablation.py make-configs --in-base A6       # база для IN_raw / IN_filtered (после E4)
    python run_ec_ablation.py train A3 --seeds 0 1 2          # обучение + оценка val/test
    python run_ec_ablation.py train A8 --seeds 0 --epochs 50  # пилот бюджета (E1b) → runs_budget/A8_e50_s0
    python run_ec_ablation.py noisy-test A0 A8 --seeds 0 1 2  # E5: оценка на test + SquareNoises
    python run_ec_ablation.py summary                         # summary.csv, summary_agg.csv, summary.md
    python run_ec_ablation.py summary --budget                # сводка пилота: experiments/ec_ablation/budget/
    python run_ec_ablation.py postprocess-eval A0 A8          # переоценка best.pt с постобработкой (metrics_pp.json)

Конфиги: кумулятивные A0…A8 (таблица `GROUPS`; источник истины — этот файл, yaml генерируются из него) и
IN_raw / IN_filtered (итоговый набор с подмножеством входных каналов, E6; поле `input_subset` — D8). Каждый
yaml — полный и явный: `name`, `description`, `dataset`, `seeds`, `training` (поля TrainingSettings без
`dataset`, `results_path`, `project`, `seed`). Неизвестные поля — ошибка (pydantic их молча игнорирует).

Результаты прогона `experiments/ec_ablation/runs/<ID>_s<seed>/`:
    train1/              step.csv, args.yaml, best.pt, last.pt (Trainer)
    config.yaml          конфиг прогона; environment.json — git hash, версии, GPU, команда
    metrics.json         val/test: mAP@0.5, mAP@0.5:0.95, P, R + AP@0.5, AP@0.5:0.95, P, R по классам
    predictions_test.npz детекции и разметка каждого тестового сигнала (офлайн-bootstrap, R1 §3)
    noisy_test.json      (E5) метрики на test + SquareNoises
    metrics_pp.json      val / test / test + шум с постобработкой POSTPROCESS (вложенные объекты после NMS);
                         predictions_test_pp.npz — детекции test после неё
    run.json             чекпоинт, длительность
Готовый прогон (есть metrics.json) повторно не обучается; незавершённый — продолжается (`resume`); --rerun — заново.

Бюджет (E1b, D9): `--epochs N` переопределяет `training.epochs`; такие прогоны идут в отдельный каталог
`experiments/ec_ablation/runs_budget/<ID>_e<N>_s<seed>/` и не пересекаются с основной абляцией (`runs/<ID>_s<seed>`).
Сводка группирует прогоны по (конфиг, epochs); колонки `epochs`, `stopped_epoch` (последняя эпоха в step.csv),
`best_epoch` (эпоха best.pt — первый максимум val mAP@0.5:0.95).
Seed задаёт инициализацию весов (Trainer вызывает `setup_seed` до создания модели), порядок батчей и аугментации.
"""
import argparse
import copy
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

# E1: каждый воркер DataLoader считает FastICA сам; без ограничения BLAS 15 воркеров делят ядра потоками
# друг друга (−10 % скорости). Переменные наследуются воркерами при запуске (spawn).
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
# E3: входы, на которых FastICA не сошлась в NaN/Inf, сохраняются для разбора (пример идёт без подавления)
os.environ.setdefault("EDDY_ICA_FAILURE_DIR", str(Path(__file__).resolve().parent / "experiments" / "ec_ablation"
                                                  / "ica_failures"))
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
from run_experiment import environment  # noqa: E402
from yolo_1d.utils.settings import AugmentationSettings, TrainingSettings  # noqa: E402

EXP_DIR = REPO / "experiments" / "ec_ablation"
CONFIG_DIR = EXP_DIR / "configs"
RUNS_DIR = EXP_DIR / "runs"
RUNS_BUDGET_DIR = EXP_DIR / "runs_budget"
BUDGET_DIR = EXP_DIR / "budget"
DATASET = "data/ecr_v1.yaml"
# один seed (решение пользователя 2026-10-17: срок статьи — неделя); статистика — парный bootstrap по сигналам test
SEEDS = [0]
NAMES = ["flashbutt", "aluminothermic", "rail end", "axle counter", "crossing", "manual arc"]

# базовые настройки обучения (start_example.py); epochs = 50 — решение пользователя 2026-10-17 (E1b)
BASE_TRAINING = {
    "names": NAMES,
    "model_size": "large",
    "batch_size": 50,
    "epochs": 50,
    "enable_amp": True,
    # E1b: одинаковый бюджет для всех конфигов, расписание LinearLR доходит до конца — без ранней остановки
    "early_stopping_epochs": 0,
    "weight_box": 10,
    "weight_dfl": 3,
    # E1: предобработка (ICA шага 7 на каждый пример) — узкое место; 15 воркеров на i5-13400F (16 потоков);
    # val/test из кэша — 4 воркера (воркеры трёх загрузчиков живут одновременно, ≈ 1.4 ГБ каждый)
    "num_workers": 15,
    "eval_num_workers": 4,
    # выбор best.pt — по mAP@0.5:0.95 на val (решение пользователя 2026-10-17; Trainer: r["mAP"])
    "best_metric": "mAP",
}
# A0: все аугментации выключены, длина приводится как на валидации
AUG_OFF = {
    "pipeline": "eddy", "regular_length": 50000, "input_length": 49152, "validation_resize": True,
    "random_resize": 0, "random_crop": False, "random_reverse": 0, "phase_spectrum_noise": 0, "random_jitter": 0,
    "random_invert": 0, "channel_scaling": 0, "time_masking": 0, "channel_masking": 0, "mosaic": 0, "mix_up": 0,
    "sine_noise": 0, "square_noises": 0, "adversarial": 0, "augment_apply_p": 0.5,
}
# кумулятивные группы (порядок SPEC); каждая аугментация, кроме RandomResize/RandomCrop/RandomPadding, — под
# RandomApply(augment_apply_p × собственная вероятность)
GROUPS = [
    ("A0", "без аугментаций", {}),
    ("A1", "+ геометрия (RandomResize, RandomCrop, RandomPadding, RandomReverse)",
     {"random_resize": 0.1, "random_crop": True, "random_reverse": 0.5}),
    ("A2", "+ спектральные (PhaseSpectrumNoise)", {"phase_spectrum_noise": 0.5}),
    ("A3", "+ аддитивный шум (RandomJitter)", {"random_jitter": 0.6}),
    ("A4", "+ амплитудные (RandomInvert, ChannelScaling)", {"random_invert": 0.5, "channel_scaling": 0.2}),
    ("A5", "+ маскирование (TimeMasking, ChannelMasking)", {"time_masking": 0.2, "channel_masking": 0.2}),
    ("A6", "+ композитные (mosaic, mix-up)", {"mosaic": 0.5, "mix_up": 0.5}),
    ("A7", "+ синусоидальные помехи (SineNoise)", {"sine_noise": 0.5}),
    ("A8", "+ прямоугольные шумы (SquareNoises) = полный набор", {"square_noises": 0.1}),
]
INPUT_CONFIGS = [("IN_raw", "raw", "только «грязная» половина (15 каналов)"),
                 ("IN_filtered", "filtered", "только «чистая» половина (15 каналов)")]
# «A8 без одной группы» (решение пользователя 2026-10-18 по итогам E3): база A8, перечисленные параметры обнуляются
LEAVE_OUT_CONFIGS = [
    ("A8_noRev", "A8 без RandomReverse (нереалистичные «обратные» сигналы; падение axle counter на A0→A1)",
     {"random_reverse": 0}),
    ("A8_noMask", "A8 без маскирования A5 (TimeMasking, ChannelMasking; единственный отрицательный шаг)",
     {"time_masking": 0, "channel_masking": 0}),
]
METRIC_KEYS = ["mAP", "mAP@50", "Precision", "Recall"]
# постобработка после NMS (решение пользователя 2026-10-21): главный класс — стрелочный перевод (crossing, 4);
# вторичные — счётчик осей (3) и стрелочный перевод; вторичный объект удаляется, если > 0.8 его длины лежит внутри
# главного с не меньшей уверенностью (metrics.detection.suppress_contained) — дубли стрелочных переводов, пропущенные NMS
POSTPROCESS = {"contain_main_classes": [4], "contain_secondary_classes": [3, 4], "contain_threshold": 0.8}
# конфиги, для которых summary.md показывает метрики по классам
DETAIL_CONFIGS = ["A8", "A8_noRev"]
# порог уверенности для P / R в сводке (решение пользователя 2026-10-21): один и тот же для всех конфигов и классов,
# вместо точки максимума среднего F1 из compute_ap (она зависит от всех классов сразу)
CONF_THRESHOLD = 0.5


def all_config_ids() -> List[str]:
    """Все ID конфигов в порядке сводки: A0…A8, IN_*, A8 без групп."""
    return [g[0] for g in GROUPS] + [c[0] for c in INPUT_CONFIGS] + [c[0] for c in LEAVE_OUT_CONFIGS]


# ------------------------------------------------------------------ конфиги

def ablation_augmentations(group_id: str) -> dict:
    """Параметры аугментаций кумулятивного конфига `group_id` (A0…A8)."""
    ids = [g[0] for g in GROUPS]
    if group_id not in ids:
        raise KeyError(f"нет группы {group_id!r}, есть {ids}")
    aug = dict(AUG_OFF)
    for gid, _, delta in GROUPS[:ids.index(group_id) + 1]:
        aug.update(delta)
    return aug


def make_config(config_id: str, in_base: str = "A8") -> dict:
    """Полный конфиг A0…A8, IN_raw / IN_filtered (на базе `in_base`) или A8 без группы (`LEAVE_OUT_CONFIGS`)."""
    training = copy.deepcopy(BASE_TRAINING)
    inputs = {c[0]: c for c in INPUT_CONFIGS}
    leave_out = {c[0]: c for c in LEAVE_OUT_CONFIGS}
    if config_id in leave_out:
        _, text, off = leave_out[config_id]
        training["augmentations"] = {**ablation_augmentations("A8"), **off}
        description = f"Абляция «без группы»: {text}."
    elif config_id in inputs:
        _, subset, text = inputs[config_id]
        training["augmentations"] = ablation_augmentations(in_base)
        training["input_subset"] = subset
        description = f"Абляция входа (E6): {text}; аугментации — {in_base}."
    else:
        training["augmentations"] = ablation_augmentations(config_id)
        description = f"Кумулятивная абляция: {dict((g[0], g[1]) for g in GROUPS)[config_id]}."
    return {"name": config_id, "description": description, "dataset": DATASET, "seeds": list(SEEDS),
            "training": training}


def write_configs(out_dir: Path = CONFIG_DIR, in_base: str = "A8") -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for cid in all_config_ids():
        path = out_dir / f"{cid}.yaml"
        header = "# Сгенерировано: python run_ec_ablation.py make-configs (таблица GROUPS). Правки — в run_ec_ablation.py.\n"
        path.write_text(header + yaml.safe_dump(make_config(cid, in_base), allow_unicode=True, sort_keys=False),
                        encoding="utf-8")
        paths.append(path)
    return paths


def check_training_keys(training: dict) -> None:
    """Неизвестные поля TrainingSettings / AugmentationSettings — ошибка (pydantic молча игнорирует их)."""
    unknown = set(training) - set(TrainingSettings.model_fields)
    unknown |= {f"augmentations.{k}" for k in set(training.get("augmentations", {})) - set(AugmentationSettings.model_fields)}
    forbidden = {"dataset", "results_path", "project", "seed"} & set(training)
    if unknown:
        raise ValueError(f"неизвестные поля конфига: {sorted(unknown)}")
    if forbidden:
        raise ValueError(f"поля {sorted(forbidden)} задаёт раннер, уберите их из training")


def load_config(config_id: str, config_dir: Path = CONFIG_DIR, epochs: Optional[int] = None) -> dict:
    """Конфиг из yaml. `epochs` переопределяет бюджет и помечает конфиг полем `budget` (→ имя `<ID>_e<N>`)."""
    path = config_dir / f"{config_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"{path}: нет конфига, выполните `python run_ec_ablation.py make-configs`")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if epochs is not None:
        config["training"]["epochs"] = epochs
        config["budget"] = epochs
    check_training_keys(config["training"])
    return config


def run_name(config: dict, seed: int) -> str:
    """Имя каталога прогона: `<ID>_s<seed>`, при переопределённом бюджете — `<ID>_e<N>_s<seed>`."""
    budget = config.get("budget")
    return f"{config['name']}_e{budget}_s{seed}" if budget is not None else f"{config['name']}_s{seed}"


def default_runs_dir(config: dict) -> Path:
    """Прогоны с переопределённым бюджетом — в `runs_budget/`, остальные — в `runs/`."""
    return RUNS_BUDGET_DIR if config.get("budget") is not None else RUNS_DIR


# ------------------------------------------------------------------ обучение и оценка

def _metrics(r: dict, per_class, names: List[str]) -> dict:
    """Результат `Trainer._evaluate_loader` → JSON-совместимый словарь."""
    out = {k: float(r.get(k, 0.0)) for k in METRIC_KEYS}
    pre, rec, ap50, ap = (np.asarray(x, float) for x in per_class)
    if len(ap) == len(names):
        out["per_class"] = {n: {"AP": float(a), "AP@50": float(a50), "Precision": float(p), "Recall": float(q)}
                            for n, p, q, a50, a in zip(names, pre, rec, ap50, ap)}
    else:   # в сплите есть не все классы — compute_ap возвращает только присутствующие
        out["per_class_unnamed"] = {"AP": ap.tolist(), "AP@50": ap50.tolist()}
    return out


def save_predictions(path: Path, ids: List[str], per_sample) -> None:
    """Детекции [x1, x2, conf, cls] (отсчёты входа) и разметка [cls], [center, width] по каждому сигналу."""
    def ragged(arrays):   # массив объектов: np.asarray склеил бы массивы одинаковой формы в один
        out = np.empty(len(arrays), dtype=object)
        out[:] = arrays
        return out

    det = [np.asarray(d, np.float32).reshape(-1, 4) for d, _, _ in per_sample]
    cls = [np.asarray(c, np.float32).reshape(-1) for _, c, _ in per_sample]
    box = [np.asarray(b, np.float32).reshape(-1, 2) for _, _, b in per_sample]
    np.savez_compressed(path, ids=np.asarray(ids), det=ragged(det), cls=ragged(cls), box=ragged(box))


def load_predictions(path: Path) -> dict:
    """Обратное к `save_predictions`: {"ids", "det", "cls", "box"} (списки по сигналам)."""
    with np.load(path, allow_pickle=True) as z:
        return {k: list(z[k]) for k in ("ids", "det", "cls", "box")}


def make_trainer(config: dict, run_dir: Path, seed: int, resume: bool):
    from yolo_1d.yolo_1d import Trainer
    settings = TrainingSettings(**config["training"], dataset=str(REPO / config["dataset"]),
                                results_path=str(run_dir), project="train", seed=seed, resume=resume)
    return Trainer(settings)


def _load_best(trainer) -> str:
    best = Path(trainer._best_path)
    if best.is_file():
        trainer.load_checkpoint(str(best))
        return best.name
    return "last (best.pt не создан)"


def run_one(config: dict, seed: int, runs_dir: Optional[Path] = None, rerun: bool = False) -> Path:
    """Обучение и оценка одного конфига с одним seed (каталог — `default_runs_dir`, если не задан)."""
    run_dir = (runs_dir or default_runs_dir(config)) / run_name(config, seed)
    if (run_dir / "metrics.json").is_file() and not rerun:
        print(f"[skip] {run_dir.name}: уже есть metrics.json")
        return run_dir
    if rerun and run_dir.exists():
        shutil.rmtree(run_dir)
    resume = (run_dir / "train1" / "last.pt").is_file()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (run_dir / "environment.json").write_text(json.dumps(environment(), indent=2), encoding="utf-8")

    start = time.time()
    trainer = make_trainer(config, run_dir, seed, resume)
    trainer.train()
    checkpoint = _load_best(trainer)
    names = config["training"]["names"]
    result = {}
    for split in ("val", "test"):
        trainer._init_dataloaders()
        loader = {"val": trainer._val_loader, "test": trainer._test_loader}[split]
        with torch.no_grad():
            r, per_class, per_sample = trainer._evaluate_loader(loader, desc=split)
        result[split] = _metrics(r, per_class, names)
        if split == "test":
            ids = [Path(f).stem for f in loader.dataset.filenames]
            save_predictions(run_dir / "predictions_test.npz", ids, per_sample)
    (run_dir / "metrics.json").write_text(json.dumps({"config": config["name"], "seed": seed, **result}, indent=2,
                                                     ensure_ascii=False), encoding="utf-8")
    (run_dir / "run.json").write_text(json.dumps({"checkpoint": checkpoint, "resumed": resume,
                                                  "duration_s": round(time.time() - start, 1)}, indent=2),
                                      encoding="utf-8")
    torch.cuda.empty_cache()
    return run_dir


# ------------------------------------------------------------------ E5: test + SquareNoises

class SeededSquareNoises(torch.nn.Module):
    """SquareNoises (p = 1) с детерминированной помехой: seed — от содержимого сигнала и `noise_seed`.

    Класс модульного уровня — трансформ передаётся в воркеры DataLoader через pickle (spawn на Windows).
    """

    def __init__(self, noise_seed: int = 0):
        super().__init__()
        from yolo_1d.utils.transforms import SquareNoises
        self.noise_seed = noise_seed
        self.noise = SquareNoises()

    def forward(self, signals, bboxes, labels):
        from yolo_1d.utils.transforms import ReferenceCache
        key = ReferenceCache.key(signals.numpy()) + str(self.noise_seed)
        state = np.random.get_state()
        np.random.seed(int(key[:8], 16) ^ self.noise_seed)
        try:
            return self.noise(signals, bboxes, labels)
        finally:
            np.random.set_state(state)


def noisy_test_transform(settings: AugmentationSettings, cache_dir: Optional[str], noise_seed: int = 0,
                         input_subset: str = "all"):
    """Валидационный конвейер с прямоугольной помехой на шаге 6 (p = 1).

    Помеха детерминирована: seed — от содержимого сигнала и `noise_seed` (не зависит от порядка и воркеров).
    Полный результат кэшируется отдельно от чистого test (`<cache>/noisy_test_s<noise_seed>/`).
    """
    from torchvision.transforms.v2 import Compose
    from yolo_1d.utils.transforms import EddyPreprocessing, PreprocessedCache, SelectChannels
    from yolo_1d.utils.transforms.transforms import Padding, Resize

    pre = EddyPreprocessing(cache_dir=cache_dir)
    ops = [pre.reference, SeededSquareNoises(noise_seed), pre.ica, pre.destandardize, pre.fourier, pre.normalize]
    if cache_dir:
        ops = [PreprocessedCache(ops, str(Path(cache_dir) / f"noisy_test_s{noise_seed}"))]
    if input_subset != "all":
        ops.append(SelectChannels(input_subset))
    return Compose(ops + [Padding(settings.regular_length), Resize(settings.input_length)])


def run_noisy_test(config: dict, seed: int, runs_dir: Optional[Path] = None, noise_seed: int = 0) -> Path:
    from torch.utils.data import DataLoader
    from yolo_1d.utils.datasets.dataset import Dataset

    run_dir = (runs_dir or default_runs_dir(config)) / run_name(config, seed)
    if not (run_dir / "metrics.json").is_file():
        raise FileNotFoundError(f"{run_dir}: прогон не завершён")
    trainer = make_trainer(config, run_dir, seed, resume=True)
    _load_best(trainer)
    with torch.no_grad():
        r, per_class, _ = trainer._evaluate_loader(_noisy_loader(trainer, config, noise_seed), desc="noisy test")
    out = {"config": config["name"], "seed": seed, "noise_seed": noise_seed,
           "test_noisy": _metrics(r, per_class, config["training"]["names"])}
    (run_dir / "noisy_test.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    torch.cuda.empty_cache()
    return run_dir


def _noisy_loader(trainer, config: dict, noise_seed: int = 0):
    """Загрузчик зашумлённой копии test (SquareNoises p = 1 на шаге 6, кэшируется)."""
    from torch.utils.data import DataLoader
    from yolo_1d.utils.datasets.dataset import Dataset
    cache = yaml.safe_load((REPO / config["dataset"]).read_text(encoding="utf-8")).get("preprocess_cache")
    transform = noisy_test_transform(trainer._settings.augmentations, cache, noise_seed, trainer._settings.input_subset)
    dataset = Dataset(str(REPO / config["dataset"]), transform, "test")
    return DataLoader(dataset, trainer._settings.batch_size, shuffle=False, collate_fn=Dataset.collate_fn,
                      num_workers=trainer._settings.num_workers)


def run_postprocess_eval(config: dict, seed: int, runs_dir: Optional[Path] = None, noise_seed: int = 0,
                         postprocess: Optional[dict] = None) -> Path:
    """Переоценка best.pt готового прогона с постобработкой после NMS → metrics_pp.json.

    Обучение и выбор best.pt не меняются (best.pt выбран по val без постобработки); `metrics.json` не трогается.
    """
    postprocess = dict(POSTPROCESS if postprocess is None else postprocess)
    run_dir = (runs_dir or default_runs_dir(config)) / run_name(config, seed)
    if not (run_dir / "metrics.json").is_file():
        raise FileNotFoundError(f"{run_dir}: прогон не завершён")
    trainer = make_trainer(config, run_dir, seed, resume=True)
    checkpoint = _load_best(trainer)
    for key, value in postprocess.items():
        setattr(trainer._settings, key, value)
    trainer._init_dataloaders()
    names = config["training"]["names"]
    result = {"config": config["name"], "seed": seed, "checkpoint": checkpoint, "postprocess": postprocess,
              "noise_seed": noise_seed}
    loaders = {"val": trainer._val_loader, "test": trainer._test_loader,
               "test_noisy": _noisy_loader(trainer, config, noise_seed)}
    for split, loader in loaders.items():
        with torch.no_grad():
            r, per_class, per_sample = trainer._evaluate_loader(loader, desc=f"{split} pp")
        result[split] = _metrics(r, per_class, names)
        if split == "test":
            ids = [Path(f).stem for f in loader.dataset.filenames]
            save_predictions(run_dir / "predictions_test_pp.npz", ids, per_sample)
    (run_dir / "metrics_pp.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    torch.cuda.empty_cache()
    return run_dir


# ------------------------------------------------------------------ сводка

def training_progress(run_dir: Path) -> dict:
    """`stopped_epoch` — последняя эпоха в step.csv; `best_epoch` — первый максимум val mAP (эпоха best.pt)."""
    step = run_dir / "train1" / "step.csv"
    if not step.is_file():
        return {}
    with open(step, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    out = {"stopped_epoch": int(rows[-1]["epoch"])}
    scored = [(float(r["mAP"]), int(r["epoch"])) for r in rows if r.get("mAP")]
    if scored:
        best = max(s for s, _ in scored)
        out["best_epoch"] = next(e for s, e in scored if s == best)
    return out


def fixed_conf_metrics(run_dir: Path, predictions: str = "predictions_test.npz",
                       conf: float = CONF_THRESHOLD) -> Optional[dict]:
    """P и R (IoU ≥ 0.5) по классам при пороге уверенности `conf` по сохранённым детекциям test.

    Детекции ниже порога отбрасываются до сопоставления с разметкой (`metrics.detection.compute_metric`, как в Trainer).
    Возвращает {"P", "R"} — макро-средние по классам разметки, и {"per_class": {name: {"P", "R", "n"}}}.
    """
    from yolo_1d.utils.metrics import detection
    path = run_dir / predictions
    config_path = run_dir / "config.yaml"
    if not path.is_file() or not config_path.is_file():
        return None
    w = yaml.safe_load(config_path.read_text(encoding="utf-8"))["training"]["augmentations"]["input_length"]
    iou = torch.tensor([0.5])
    tp, n_det, n_gt = np.zeros(len(NAMES)), np.zeros(len(NAMES)), np.zeros(len(NAMES))
    p = load_predictions(path)
    for det, cls, box in zip(p["det"], p["cls"], p["box"]):
        det = torch.as_tensor(np.asarray(det, np.float32).reshape(-1, 4))
        det = det[det[:, 2] >= conf]
        cls = torch.as_tensor(np.asarray(cls, np.float32).reshape(-1, 1))
        box = torch.as_tensor(np.asarray(box, np.float32).reshape(-1, 2))
        np.add.at(n_gt, cls[:, 0].long().numpy(), 1)
        np.add.at(n_det, det[:, 3].long().numpy(), 1)
        if det.shape[0] and cls.shape[0]:
            hit = detection.compute_metric(
                det[:, :4], torch.cat((cls, detection.xw_to_x1x2(box) * w), dim=1), iou
            )[:, 0]
            np.add.at(tp, det[hit, 3].long().numpy(), 1)
    present = n_gt > 0
    prec = np.divide(tp, n_det, out=np.zeros_like(tp), where=n_det > 0)
    rec = np.divide(tp, n_gt, out=np.zeros_like(tp), where=n_gt > 0)
    return {"P": float(prec[present].mean()), "R": float(rec[present].mean()),
            "per_class": {name: {"P": float(prec[k]), "R": float(rec[k]), "n": int(n_gt[k])}
                          for k, name in enumerate(NAMES)}}


def collect(runs_dir: Path = RUNS_DIR) -> List[dict]:
    """Строки по прогонам: конфиг, seed, бюджет, метрики val/test(/test_noisy), AP по классам на test."""
    rows = []
    for metrics_path in sorted(runs_dir.glob("*_s*/metrics.json")):
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = {"config": m["config"], "seed": m["seed"]}
        config_path = metrics_path.parent / "config.yaml"
        if config_path.is_file():
            row["epochs"] = yaml.safe_load(config_path.read_text(encoding="utf-8"))["training"]["epochs"]
        row.update(training_progress(metrics_path.parent))
        for split in ("val", "test"):
            row.update({f"{split}_{k}": m[split][k] for k in METRIC_KEYS})
        for name, v in m["test"].get("per_class", {}).items():
            row[f"test_AP_{name}"] = v["AP"]
            row[f"test_AP50_{name}"] = v["AP@50"]
        noisy = metrics_path.parent / "noisy_test.json"
        if noisy.is_file():
            n = json.loads(noisy.read_text(encoding="utf-8"))["test_noisy"]
            row.update({f"test_noisy_{k}": n[k] for k in METRIC_KEYS})
        pp_path = metrics_path.parent / "metrics_pp.json"
        if pp_path.is_file():
            pp = json.loads(pp_path.read_text(encoding="utf-8"))
            for split in ("val", "test", "test_noisy"):
                row.update({f"pp_{split}_{k}": pp[split][k] for k in METRIC_KEYS})
            for name, v in pp["test"].get("per_class", {}).items():
                row[f"pp_test_AP_{name}"] = v["AP"]
        for prefix, predictions in (("", "predictions_test.npz"), ("pp_", "predictions_test_pp.npz")):
            fixed = fixed_conf_metrics(metrics_path.parent, predictions)
            if fixed is not None:
                row[f"{prefix}test_P_conf"], row[f"{prefix}test_R_conf"] = fixed["P"], fixed["R"]
        rows.append(row)
    return rows


def _order(config: str) -> tuple:
    ids = all_config_ids()
    return (ids.index(config) if config in ids else len(ids), config)


_KEY_FIELDS = ("config", "epochs", "seed")


def summarize(runs_dir: Path = RUNS_DIR, out_dir: Path = EXP_DIR) -> List[dict]:
    """summary.csv (по прогонам), summary_agg.csv (mean ± SD по seed для каждой пары конфиг × epochs), summary.md."""
    rows = sorted(collect(runs_dir), key=lambda r: (_order(r["config"]), r.get("epochs") or 0, r["seed"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r},
                    key=lambda k: (_KEY_FIELDS.index(k) if k in _KEY_FIELDS else len(_KEY_FIELDS), k))
    with open(out_dir / "summary.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    agg = []
    groups = sorted({(r["config"], r.get("epochs")) for r in rows}, key=lambda g: (_order(g[0]), g[1] or 0))
    for config, epochs in groups:
        runs = [r for r in rows if r["config"] == config and r.get("epochs") == epochs]
        a = {"config": config, "epochs": epochs, "n_seeds": len(runs)}
        for k in [k for k in fields if k not in _KEY_FIELDS]:
            vals = [r[k] for r in runs if k in r]
            if vals:
                a[f"{k}_mean"], a[f"{k}_sd"] = float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        agg.append(a)
    head = ("config", "epochs", "n_seeds")
    agg_fields = sorted({k for a in agg for k in a}, key=lambda k: (head.index(k) if k in head else len(head), k))
    with open(out_dir / "summary_agg.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fields)
        writer.writeheader()
        writer.writerows(agg)
    lines = ["# Абляция аугментаций ECRv1", "",
             f"Test (294 сигнала), seed 0. mAP — mAP@0.5:0.95, макро по 6 классам; P и R — при пороге уверенности "
             f"{CONF_THRESHOLD} (IoU ≥ 0.5), макро по классам. «С постобработкой» — удаление вложенных объектов после "
             f"NMS: главный класс стрелочный перевод, вторичные — счётчик осей и стрелочный перевод, порог доли "
             f"{POSTPROCESS['contain_threshold']}, главный не менее уверен (тот же best.pt).", "",
             "| конфиг | epochs | эпоха best.pt | mAP | mAP@50 | P | R | mAP, с постобр. | mAP@50, с постобр. "
             "| P, с постобр. | R, с постобр. |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda a, k: f"{100 * a[k + '_mean']:.2f}" if k + "_mean" in a else "—"
    for a in agg:
        best = f"{a['best_epoch_mean']:.0f}" if "best_epoch_mean" in a else "—"
        epochs = a["epochs"] if a["epochs"] is not None else "—"
        lines.append(f"| {a['config']} | {epochs} | {best} | {fmt(a, 'test_mAP')} | {fmt(a, 'test_mAP@50')} | "
                     f"{fmt(a, 'test_P_conf')} | {fmt(a, 'test_R_conf')} | {fmt(a, 'pp_test_mAP')} | "
                     f"{fmt(a, 'pp_test_mAP@50')} | {fmt(a, 'pp_test_P_conf')} | {fmt(a, 'pp_test_R_conf')} |")
    lines += _detail_section(runs_dir)
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return agg


def _detail_section(runs_dir: Path) -> List[str]:
    """Метрики test по классам для DETAIL_CONFIGS, без и с постобработкой (P, R — при CONF_THRESHOLD)."""
    lines = []
    for config in DETAIL_CONFIGS:
        for run_dir in sorted(runs_dir.glob(f"{config}_s*")):
            if not (run_dir / "metrics.json").is_file():
                continue
            sources = [("без постобработки", json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")),
                        "predictions_test.npz")]
            if (run_dir / "metrics_pp.json").is_file():
                sources.append(("с постобработкой", json.loads((run_dir / "metrics_pp.json").read_text(encoding="utf-8")),
                                "predictions_test_pp.npz"))
            lines += ["", f"## {run_dir.name}: test по классам", "",
                      f"P и R — при пороге уверенности {CONF_THRESHOLD}, IoU ≥ 0.5."]
            for title, src, predictions in sources:
                fixed = fixed_conf_metrics(run_dir, predictions)
                if fixed is None or "per_class" not in src["test"]:
                    continue
                t = src["test"]
                lines += ["", f"**{title}**", "", "| класс | объектов | AP@0.5:0.95 | AP@50 | P | R |",
                          "|---|---|---|---|---|---|"]
                for name, v in t["per_class"].items():
                    f = fixed["per_class"][name]
                    lines.append(f"| {name} | {f['n']} | {v['AP']:.3f} | {v['AP@50']:.3f} | {f['P']:.3f} | {f['R']:.3f} |")
                n = sum(f["n"] for f in fixed["per_class"].values())
                lines.append(f"| **среднее** | {n} | **{t['mAP']:.3f}** | **{t['mAP@50']:.3f}** | {fixed['P']:.3f} | "
                             f"{fixed['R']:.3f} |")
    return lines


# ------------------------------------------------------------------ CLI

def main(argv=None) -> int:
    # при перенаправлении в файл Windows по умолчанию пишет в cp1251, а torchinfo печатает псевдографику
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("make-configs", help="сгенерировать yaml-конфиги A0…A8, IN_raw, IN_filtered")
    p.add_argument("--in-base", default="A8", help="база аугментаций для IN_raw / IN_filtered (по умолчанию A8)")
    for name in ("train", "noisy-test", "postprocess-eval"):
        p = sub.add_parser(name)
        p.add_argument("configs", nargs="+", help="ID конфигов (A0 … A8, IN_raw, IN_filtered)")
        p.add_argument("--seeds", type=int, nargs="+", help="переопределить seeds конфига")
        p.add_argument("--epochs", type=int, help="переопределить training.epochs (пилот бюджета; → runs_budget/)")
        if name == "train":
            p.add_argument("--rerun", action="store_true", help="обучить заново, удалив прежний прогон")
        else:
            p.add_argument("--noise-seed", type=int, default=0, help="seed зашумлённой копии test")
    p = sub.add_parser("summary", help="пересобрать сводку по готовым прогонам")
    p.add_argument("--budget", action="store_true", help="сводка пилота бюджета (runs_budget/ → budget/)")
    args = parser.parse_args(argv)

    if args.command == "make-configs":
        for path in write_configs(in_base=args.in_base):
            print(path.relative_to(REPO))
        return 0
    if args.command == "summary":
        runs_dir, out_dir = (RUNS_BUDGET_DIR, BUDGET_DIR) if args.budget else (RUNS_DIR, EXP_DIR)
        summarize(runs_dir, out_dir)
        print((out_dir / "summary.md").read_text(encoding="utf-8"))
        return 0
    for cid in args.configs:
        config = load_config(cid, epochs=args.epochs)
        for seed in args.seeds if args.seeds is not None else config["seeds"]:
            print(f"\n===== {cid}, seed {seed} ({args.command}) =====")
            if args.command == "train":
                run_one(config, seed, rerun=args.rerun)
            elif args.command == "postprocess-eval":
                run_postprocess_eval(config, seed, noise_seed=args.noise_seed)
            else:
                run_noisy_test(config, seed, noise_seed=args.noise_seed)
    if args.epochs is not None:
        summarize(RUNS_BUDGET_DIR, BUDGET_DIR)
    else:
        summarize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
