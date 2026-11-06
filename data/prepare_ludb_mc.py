"""Подготовка LUDB для режима MC (M6): пример = вся запись, 12 отведений.

Пригодность отведения берётся из манифеста режима S (`data/processed/ludb/manifest.csv`): непригодные
отведения (дефектная разметка внутри зоны, 24 отведения в 7 записях) **остаются во входе** (контекст для
соседей), но их разметка не пишется, а в `leads.csv` они помечены `usable = 0` — `MultiLeadDataset`
исключает их из лосса и метрик (R5 §2.2).

Выход (`data/processed/ludb_mc/`, в git не хранится):
    signals/<запись>.pt   torch.float32, форма (12, 5000), порядок отведений `ludb.LEADS`
    labels/<запись>.txt   строки "class lead center width" (lead — индекс 0..11; center/width — доли 5000)
    leads.csv             запись × отведение: annot_start, annot_end (размеченная зона отведения), usable
    preprocessing.json    параметры и счётчики

Использование (после data/prepare_ludb.py):
    python data/prepare_ludb_mc.py
"""
import argparse
import csv
import json
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))  # модули этого репозитория, а не установленной копии пакета
from yolo_1d.utils.datasets.ludb import CLASS_NAMES, LEADS, SIGNAL_LENGTH, iter_records  # noqa: E402
from yolo_1d.utils.datasets.ludb_preprocess import prepare_lead  # noqa: E402

LEADS_FIELDS = ["record", "lead_index", "lead", "annot_start", "annot_end", "usable"]


def read_manifest(path: Path) -> dict:
    """sample_id → строка манифеста режима S."""
    with open(path, encoding="utf-8", newline="") as f:
        return {row["sample_id"]: row for row in csv.DictReader(f)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=REPO / "data" / "raw" / "ludb")
    parser.add_argument("--manifest", type=Path, default=REPO / "data" / "processed" / "ludb" / "manifest.csv")
    parser.add_argument("--out", type=Path, default=REPO / "data" / "processed" / "ludb_mc")
    args = parser.parse_args()
    if not args.manifest.is_file():
        print(f"Нет {args.manifest}. Сначала: python data/prepare_ludb.py")
        return 1
    manifest = read_manifest(args.manifest)
    s_config = json.loads((args.manifest.parent / "preprocessing.json").read_text(encoding="utf-8"))
    band = tuple(s_config["bandpass_hz"]) if s_config.get("bandpass_hz") else None

    if args.out.exists():
        shutil.rmtree(args.out)
    (args.out / "signals").mkdir(parents=True)
    (args.out / "labels").mkdir(parents=True)

    rows, n_records, n_ignored = [], 0, 0
    waves = {name: 0 for name in CLASS_NAMES}
    for rec in iter_records(args.root):
        signals, lines = [], []
        for k, lead in enumerate(LEADS):
            row = manifest[f"{rec.name}_{lead}"]
            sample = prepare_lead(rec.signal[k], rec.waves[lead], band)
            signals.append(sample.signal)
            usable = row["usable"] == "1"
            rows.append({"record": rec.name, "lead_index": k, "lead": lead, "annot_start": sample.annotated[0],
                         "annot_end": sample.annotated[1], "usable": int(usable)})
            if not usable:
                n_ignored += 1
                continue
            for c, x, w in sample.labels:
                lines.append(f"{int(c)} {k} {x:.8f} {w:.8f}")
                waves[CLASS_NAMES[int(c)]] += 1
        torch.save(torch.from_numpy(np.concatenate(signals, axis=0)), args.out / "signals" / f"{rec.name}.pt")
        (args.out / "labels" / f"{rec.name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        n_records += 1

    with open(args.out / "leads.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEADS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "date": date.today().isoformat(),
        "mode": "multi_lead",
        "signal_length": SIGNAL_LENGTH,
        "leads": LEADS,
        "bandpass_hz": list(band) if band else None,
        "normalization": "z-score per lead (как в режиме S); маска краёв — при загрузке, по зоне каждого отведения",
        "label_format": "class lead center width; center/width — доли signal_length",
        "records": n_records,
        "leads_ignored": n_ignored,
        "ignored": [f"{r['record']}_{r['lead']}" for r in rows if not r["usable"]],
        "waves_written": waves,
    }
    (args.out / "preprocessing.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Записей: {n_records}, игнорируемых отведений: {n_ignored}; волн: {waves}\nДанные: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
