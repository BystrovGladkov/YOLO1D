"""Разбиения режима MC: те же фолды, что в режиме S, но пример — запись.

Списки записей выводятся из существующих списков отведений `data/splits/fold_<k>/{train,val,test}.txt`
(data/make_splits.py), поэтому разбиение по записям совпадает с режимом S буквально — новое случайное
разбиение не создаётся. Запись попадает в список, если в нём есть хотя бы одно её отведение
(у каждой записи LUDB есть пригодные отведения).

Выход (`data/splits/`, хранится в git):
    fold_<k>/{train,val,test}_mc.txt   ID записей, по одному в строке
    fold_<k>_mc.yaml                   конфиг MultiLeadDataset (`mode: multi_lead`)

Использование (после data/make_splits.py и data/prepare_ludb_mc.py):
    python data/make_splits_mc.py [--folds 5]
"""
import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def records_of(list_path: Path) -> list:
    """Записи из списка отведений <запись>_<отведение>, в порядке первого появления."""
    out = []
    for line in list_path.read_text(encoding="utf-8").split():
        rec = line.rsplit("_", 1)[0]
        if rec not in out:
            out.append(rec)
    return out


def fold_config(k: int, names: dict) -> dict:
    """Конфиг MultiLeadDataset: пути относительно data/ (ключ `path` — от yaml-файла)."""
    config = {"path": "..", "names": names, "mode": "multi_lead", "fold": k,
              # зоны и пригодность каждого отведения (маска краёв по отведениям, игнор непригодных)
              "zones": "processed/ludb_mc/leads.csv"}
    for split in ("train", "val", "test"):
        config[f"{split}_data"] = "processed/ludb_mc/signals"
        config[f"{split}_labels"] = "processed/ludb_mc/labels"
        config[f"{split}_list"] = f"splits/fold_{k}/{split}_mc.txt"
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--splits", type=Path, default=REPO / "data" / "splits")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    seen = []
    for k in range(args.folds):
        s_yaml = yaml.safe_load((args.splits / f"fold_{k}.yaml").read_text(encoding="utf-8"))
        sizes = {}
        for split in ("train", "val", "test"):
            records = records_of(args.splits / f"fold_{k}" / f"{split}.txt")
            (args.splits / f"fold_{k}" / f"{split}_mc.txt").write_text("\n".join(records) + "\n", encoding="utf-8")
            sizes[split] = len(records)
            if split == "test":
                seen += records
        (args.splits / f"fold_{k}_mc.yaml").write_text(
            f"# LUDB, режим MC (12 отведений), фолд {k}: сгенерировано data/make_splits_mc.py из fold_{k}.yaml.\n"
            + yaml.safe_dump(fold_config(k, s_yaml["names"]), allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(f"fold {k}: train {sizes['train']}, val {sizes['val']}, test {sizes['test']} записей")
    ok = len(seen) == len(set(seen)) == 200
    print("Проверка (каждая запись ровно один раз в test):", "OK" if ok else "ОШИБКА")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
