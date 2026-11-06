"""Датасет ECRv1 (Eddy Current Rail): источник данных и манифест `dataset.yaml`.

Структура датасета (README ECRv1):

    {train,val,test}/data/passage{K}_{side}_{NNNN}.npy    int16, (15, L)
    {train,val,test}/labels/passage{K}_{side}_{NNNN}.txt  строки "type begin end" (отсчёты)
    annotations/passage{K}.json

Источник — локальная папка или репозиторий Hugging Face (`Nasochec/ECRv1`). Сигналы не
преобразуются заранее: `Dataset` читает `.npy` и исходную разметку, предобработка идёт в трансформах.
Манифест помечается `format: ecr_npy` и содержит карту классов исходной разметки → 0…5.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml

# репозиторий на Hugging Face и локальная копия по умолчанию
HF_REPO_ID = "Nasochec/ECRv1"
DEFAULT_LOCAL_ROOT = Path(r"E:\rails\public_dataset\dataset")

SPLITS = ("train", "val", "test")
DATA_EXT = ".npy"
LABEL_EXT = ".txt"

# id класса в разметке ECRv1 (не подряд) → номер класса сети
CLASS_MAP: Dict[int, int] = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 12: 5}
CLASS_NAMES: List[str] = ["flashbutt", "aluminothermic", "rail end", "axle counter", "crossing", "manual arc"]

# значение ключа `format` в манифесте; Dataset по нему выбирает чтение .npy + "type begin end"
MANIFEST_FORMAT = "ecr_npy"


def _hf_repo_id(source: str) -> Optional[str]:
    """`hf://owner/name` или `owner/name` → id репозитория; иначе None."""
    if source.startswith("hf://"):
        return source[len("hf://"):].strip("/")
    parts = source.split("/")
    if len(parts) == 2 and all(parts) and ":" not in source and "\\" not in source:
        return source
    return None


def validate_layout(root: Union[str, Path], strict: bool = True) -> Dict[str, Dict[str, int]]:
    """Проверяет структуру `{split}/{data,labels}` и попарное совпадение имён сигналов и разметки.

    Возвращает статистику {split: {"data": n, "labels": n, "pairs": n}}. При `strict=True` любое
    расхождение (нет папки, сигнал без разметки или разметка без сигнала) — `ValueError`;
    иначе расхождения только отражаются в статистике.
    """
    root = Path(root)
    stats, problems = {}, []
    for split in SPLITS:
        data_dir, labels_dir = root / split / "data", root / split / "labels"
        if not data_dir.is_dir() or not labels_dir.is_dir():
            problems.append(f"{split}: нет папки {data_dir if not data_dir.is_dir() else labels_dir}")
            continue
        data = {p.stem for p in data_dir.iterdir() if p.suffix == DATA_EXT}
        labels = {p.stem for p in labels_dir.iterdir() if p.suffix == LABEL_EXT}
        stats[split] = {"data": len(data), "labels": len(labels), "pairs": len(data & labels)}
        if data - labels:
            problems.append(f"{split}: {len(data - labels)} сигналов без разметки, например {sorted(data - labels)[:3]}")
        if labels - data:
            problems.append(f"{split}: {len(labels - data)} файлов разметки без сигнала, например {sorted(labels - data)[:3]}")
        if not data:
            problems.append(f"{split}: нет файлов {DATA_EXT}")
    if problems and strict:
        raise ValueError(f"Неверная структура ECRv1 в {root}:\n  " + "\n  ".join(problems))
    return stats


def get_ecr_dataset(source: Union[str, Path, None] = None, cache_dir: Union[str, Path, None] = None,
                    revision: Optional[str] = None, validate: bool = True) -> Path:
    """Корень датасета ECRv1.

    source
        Существующая локальная папка — используется как есть (по умолчанию `DEFAULT_LOCAL_ROOT`).
        `hf://Nasochec/ECRv1` или `Nasochec/ECRv1` — скачивание `huggingface_hub.snapshot_download`
        (`repo_type="dataset"`); повторный вызов берёт уже скачанные файлы.
    cache_dir
        Папка для скачанной копии (`local_dir`); None — стандартный кэш Hugging Face.
    revision
        Ветка, тег или коммит репозитория (для воспроизводимости); None — последняя версия.
    validate
        Проверить структуру (`validate_layout`, strict).
    """
    source = DEFAULT_LOCAL_ROOT if source is None else source
    path = Path(source)
    if path.is_dir():
        root = path
    else:
        repo_id = _hf_repo_id(str(source))
        if repo_id is None:
            raise FileNotFoundError(f"{source}: не папка и не id репозитория Hugging Face")
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:  # опциональная зависимость
            raise ImportError("для загрузки с Hugging Face установите `pip install yolo1d[hf]` "
                              "(пакет huggingface_hub)") from e
        root = Path(snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision,
                                      local_dir=None if cache_dir is None else str(cache_dir)))
    if validate:
        validate_layout(root, strict=True)
    return root


def write_manifest(root: Union[str, Path], out_path: Union[str, Path],
                   preprocess_cache: Union[str, Path, None] = None) -> Path:
    """Создаёт `dataset.yaml` для ECRv1 в формате, который читает `Dataset`.

    Пути сплитов абсолютные: корень датасета может быть в кэше Hugging Face (только чтение), поэтому
    манифест пишется в отдельное место. `preprocess_cache` — папка кэша предобработки (D4/D6);
    None — `cache/` рядом с манифестом.
    """
    root, out_path = Path(root).resolve(), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache = Path(preprocess_cache) if preprocess_cache is not None else out_path.parent.resolve() / "cache"
    manifest = {
        "format": MANIFEST_FORMAT,
        "source": str(root),
        **{f"{s}_{kind}": str(root / s / kind) for s in SPLITS for kind in ("data", "labels")},
        "names": dict(enumerate(CLASS_NAMES)),
        "class_map": dict(CLASS_MAP),
        "preprocess_cache": str(cache),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# ECRv1: сигналы .npy (int16, 15 x L), разметка \"type begin end\" в отсчётах.\n"
                "# Создано yolo_1d.utils.datasets.ecr.write_manifest; class_map: id класса в разметке -> номер класса сети.\n")
        yaml.safe_dump(manifest, f, allow_unicode=True, sort_keys=False)
    return out_path
