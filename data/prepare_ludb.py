"""Подготовка LUDB для обучения MyNet-1d: поотведенческие примеры в формате `Dataset`.

Для каждого из 200 × 12 отведений:
- Z-score по всему отведению (опционально — полосовой фильтр до него); сигнал сохраняется без маски;
- границы размеченной зоны (первый onset, последний offset) — в манифест (`annot_start`, `annot_end`);
  неразмеченные края обнуляются при загрузке (`EdgeMask`, D5: случайный запас на train, фиксированный на val);
- разметка в формате `class center width` (доли длины 5000), классы P = 0, QRS = 1, T = 2.

Отведения с дефектной разметкой внутри зоны исключаются
из всех выборок по умолчанию; они остаются в манифесте с причиной (`--keep-defective` — оставить).
Зона для этой проверки расширяется на максимальный запас маски (`--max-margin`, 24 отсчёта).

Выход (`data/processed/ludb/`, в git не хранится):
    signals/<запись>_<отведение>.pt   torch.float32, форма (1, 5000)
    labels/<запись>_<отведение>.txt   строки "class center width"
    manifest.csv                      все 2400 отведений: annot_start/annot_end, число волн, группа ритма,
                                      usable/причина (служит и файлом зон для Dataset, ключ `zones`)
    preprocessing.json                параметры и итоговые счётчики
и конфиг `data/ludb.yaml` (имена классов и пути).

Использование:
    python data/prepare_ludb.py [--max-margin 24] [--bandpass 0.5 100] [--keep-defective]
"""
import argparse
import csv
import json
import os
import shutil
import sys
from datetime import date
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))  # модули этого репозитория, а не установленной копии пакета
from yolo_1d.utils.datasets.ludb import CLASS_NAMES, LEADS, SIGNAL_LENGTH, iter_records, rhythm_group  # noqa: E402
from yolo_1d.utils.datasets.ludb_preprocess import annotated_zone, prepare_lead, unusable_reason  # noqa: E402

MANIFEST_FIELDS = ["sample_id", "record", "lead", "rhythm_group", "annot_start", "annot_end",
                   "n_P", "n_QRS", "n_T", "issues", "usable", "reason"]


def write_labels(path: Path, labels) -> None:
    lines = [f"{int(c)} {x:.8f} {w:.8f}" for c, x, w in labels]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_dataset_yaml(path: Path, out: Path) -> None:
    """Конфиг датасета: классы и пути. Ключи `<split>_data/<split>_labels` добавляет D4."""
    # пути относительно `path`, который сам задан относительно yaml-файла (см. Dataset._parse_yaml)
    base = path.parent.resolve()
    rel = lambda p: Path(os.path.relpath(p.resolve(), base)).as_posix()  # noqa: E731
    config = {
        "path": ".",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
        "signal_length": SIGNAL_LENGTH,
        "mode": "single_lead",
        "data": rel(out / "signals"),
        "labels": rel(out / "labels"),
        "manifest": rel(out / "manifest.csv"),
    }
    header = ("# LUDB для MyNet-1d: сгенерировано data/prepare_ludb.py.\n"
              "# Поотведенческие примеры (вход 1 × 5000, паддинг до 5120 в трансформациях).\n"
              "# Разбиения train/val/test — data/splits/fold_<k>.yaml (data/make_splits.py).\n")
    path.write_text(header + yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=REPO / "data" / "raw" / "ludb")
    parser.add_argument("--out", type=Path, default=REPO / "data" / "processed" / "ludb")
    parser.add_argument("--max-margin", type=int, default=24,
                        help="максимальный запас маски краёв, отсчётов: зона проверки дефектной разметки")
    parser.add_argument("--bandpass", type=float, nargs=2, metavar=("LOW", "HIGH"), default=None,
                        help="полосовой фильтр до Z-score, Гц (по умолчанию выключен)")
    parser.add_argument("--keep-defective", action="store_true",
                        help="не исключать отведения с дефектной разметкой внутри зоны")
    parser.add_argument("--config", type=Path, default=REPO / "data" / "ludb.yaml")
    args = parser.parse_args()
    if not (args.root / "RECORDS").is_file():
        print(f"Нет данных в {args.root}. Сначала: python data/download_ludb.py")
        return 1

    # полная перегенерация: удаляем старые примеры и кэш меток Dataset (labels.cache)
    if args.out.exists():
        shutil.rmtree(args.out)
    signals_dir, labels_dir = args.out / "signals", args.out / "labels"
    signals_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    band = tuple(args.bandpass) if args.bandpass else None
    rows, written, excluded = [], 0, 0
    waves_written = {name: 0 for name in CLASS_NAMES}
    for rec in iter_records(args.root):
        group = rhythm_group(rec.meta)
        for lead_index, lead in enumerate(LEADS):
            sample_id = f"{rec.name}_{lead}"
            sample = prepare_lead(rec.signal[lead_index], rec.waves[lead], band)
            check_zone = annotated_zone(rec.waves[lead], SIGNAL_LENGTH, args.max_margin)
            reason = unusable_reason([i.sample for i in rec.dropped(lead)], check_zone)
            usable = reason is None or args.keep_defective
            counts = {name: sum(w.cls == name for w in rec.waves[lead]) for name in CLASS_NAMES}
            rows.append({
                "sample_id": sample_id, "record": rec.name, "lead": lead, "rhythm_group": group,
                "annot_start": sample.annotated[0], "annot_end": sample.annotated[1],
                **{f"n_{name}": n for name, n in counts.items()},
                "issues": ";".join(sorted({i.kind for i in rec.issues[lead]})),
                "usable": int(usable), "reason": reason or "",
            })
            if not usable:
                excluded += 1
                continue
            torch.save(torch.from_numpy(sample.signal), signals_dir / f"{sample_id}.pt")
            write_labels(labels_dir / f"{sample_id}.txt", sample.labels)
            written += 1
            for name, n in counts.items():
                waves_written[name] += n

    with open(args.out / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "date": date.today().isoformat(),
        "source": str(args.root.relative_to(REPO)) if args.root.is_relative_to(REPO) else str(args.root),
        "mode": "single_lead",
        "signal_length": SIGNAL_LENGTH,
        "max_mask_margin_samples": args.max_margin,
        "bandpass_hz": list(band) if band else None,
        "normalization": "z-score per lead over the whole record; edge mask applied on load (EdgeMask, D5)",
        "label_format": "class center width, normalized by signal_length; left = onset, right = offset",
        "classes": CLASS_NAMES,
        "keep_defective": args.keep_defective,
        "leads_total": len(rows),
        "leads_written": written,
        "leads_excluded": excluded,
        "excluded": [r["sample_id"] for r in rows if not r["usable"]],
        "waves_written": waves_written,
    }
    (args.out / "preprocessing.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_dataset_yaml(args.config, args.out)

    print(f"Отведений: {len(rows)}, записано {written}, исключено {excluded} (дефектная разметка внутри зоны)")
    print(f"Волн в записанных примерах: {waves_written}")
    print(f"Данные: {args.out}\nКонфиг: {args.config}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
