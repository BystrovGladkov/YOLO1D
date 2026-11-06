"""Разбиения LUDB: 5-fold кросс-валидация по записям со стратификацией по ритму (D4).

Протокол:
- единица разбиения — запись (1 запись = 1 человек): все отведения записи попадают в один сплит;
- 5 фолдов, стратифицированных по группе ритма (`rhythm_group`: sinus / af_flutter / paced);
  каждая запись ровно один раз оказывается в test → итоговые метрики считаются по всем записям
  (объединение тестов 5 фолдов), разброс — mean ± SD по фолдам;
- в фолде k: test = записи фолда k; из остальных ~10 % (по группам) уходит в val
  (early stopping, подбор порогов), остальное — train.
Распределение детерминировано (`--seed`).

Вход: `data/processed/ludb/manifest.csv` (data/prepare_ludb.py). В сплиты попадают только
отведения с usable = 1.

Выход (`data/splits/`, хранится в git):
    folds.csv                       запись, группа ритма, номер тестового фолда
    fold_<k>/{train,val,test}.txt   ID примеров (<запись>_<отведение>), по одному в строке
    fold_<k>.yaml                   конфиг Dataset для фолда k
    summary.json                    размеры сплитов по группам и проверки

Использование:
    python data/make_splits.py [--folds 5] [--val-fraction 0.1] [--seed 2026]
"""
import argparse
import collections
import csv
import json
import random
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
GROUPS = ["sinus", "af_flutter", "paced"]


def read_manifest(path: Path):
    """Записи → группа ритма и список пригодных отведений (ID примеров)."""
    groups, samples = {}, collections.defaultdict(list)
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            groups[row["record"]] = row["rhythm_group"]
            if row["usable"] == "1":
                samples[row["record"]].append(row["sample_id"])
    return groups, samples


def stratified_folds(groups: dict, n_folds: int, seed: int) -> dict:
    """Запись → номер фолда. В каждой группе записи перемешиваются и раздаются по кругу."""
    rng = random.Random(seed)
    fold_of, offset = {}, 0
    for group in GROUPS:
        records = sorted((r for r, g in groups.items() if g == group), key=int)
        rng.shuffle(records)
        for i, record in enumerate(records):
            # сдвиг, чтобы малые группы не скапливались в первых фолдах
            fold_of[record] = (i + offset) % n_folds
        offset += len(records)
    return fold_of


def split_val(records: list, groups: dict, fraction: float, rng: random.Random):
    """Делит записи на (train, val): в val по каждой группе max(1, round(fraction · n))."""
    val = []
    for group in GROUPS:
        members = sorted((r for r in records if groups[r] == group), key=int)
        if not members:
            continue
        rng.shuffle(members)
        val += members[:max(1, round(fraction * len(members)))]
    val_set = set(val)
    return [r for r in records if r not in val_set], val


def write_list(path: Path, records: list, samples: dict) -> int:
    ids = [sid for r in sorted(records, key=int) for sid in samples[r]]
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")
    return len(ids)


def fold_config(k: int, names: dict) -> dict:
    """Конфиг Dataset: пути относительно папки data/ (ключ `path` задаётся от yaml-файла)."""
    config = {"path": "..", "names": names, "mode": "single_lead", "fold": k,
              # границы размеченной зоны для маскирования краёв при загрузке (EdgeMask, D5)
              "zones": "processed/ludb/manifest.csv"}
    for split in ("train", "val", "test"):
        config[f"{split}_data"] = "processed/ludb/signals"
        config[f"{split}_labels"] = "processed/ludb/labels"
        config[f"{split}_list"] = f"splits/fold_{k}/{split}.txt"
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--manifest", type=Path, default=REPO / "data" / "processed" / "ludb" / "manifest.csv")
    parser.add_argument("--dataset-config", type=Path, default=REPO / "data" / "ludb.yaml")
    parser.add_argument("--out", type=Path, default=REPO / "data" / "splits")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if not args.manifest.is_file():
        print(f"Нет {args.manifest}. Сначала: python data/prepare_ludb.py")
        return 1

    groups, samples = read_manifest(args.manifest)
    names = yaml.safe_load(args.dataset_config.read_text(encoding="utf-8"))["names"]
    fold_of = stratified_folds(groups, args.folds, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    with open(args.out / "folds.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["record", "rhythm_group", "fold"])
        for record in sorted(fold_of, key=int):
            writer.writerow([record, groups[record], fold_of[record]])

    rng = random.Random(args.seed + 1)
    summary = {"seed": args.seed, "folds": args.folds, "val_fraction": args.val_fraction, "per_fold": []}
    test_seen = collections.Counter()
    problems = []
    for k in range(args.folds):
        test = [r for r, f in fold_of.items() if f == k]
        train, val = split_val([r for r, f in fold_of.items() if f != k], groups, args.val_fraction, rng)
        fold_dir = args.out / f"fold_{k}"
        fold_dir.mkdir(exist_ok=True)
        info = {"fold": k}
        for split, records in (("train", train), ("val", val), ("test", test)):
            n_samples = write_list(fold_dir / f"{split}.txt", records, samples)
            info[split] = {"records": len(records), "samples": n_samples,
                           "groups": dict(collections.Counter(groups[r] for r in records))}
        test_seen.update(test)
        if set(train) & set(val) or set(train) & set(test) or set(val) & set(test):
            problems.append(f"fold {k}: пересечение записей между сплитами")
        (args.out / f"fold_{k}.yaml").write_text(
            f"# LUDB, фолд {k} из {args.folds}: сгенерировано data/make_splits.py (seed {args.seed}).\n"
            + yaml.safe_dump(fold_config(k, names), allow_unicode=True, sort_keys=False), encoding="utf-8")
        summary["per_fold"].append(info)

    if set(test_seen) != set(groups) or any(n != 1 for n in test_seen.values()):
        problems.append("не каждая запись ровно один раз в test")
    summary["problems"] = problems
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("fold | train зап./прим. | val зап./прим. | test зап./прим. | test по группам")
    for info in summary["per_fold"]:
        print(f"{info['fold']:4d} | {info['train']['records']:3d} / {info['train']['samples']:4d} | "
              f"{info['val']['records']:3d} / {info['val']['samples']:3d} | "
              f"{info['test']['records']:3d} / {info['test']['samples']:3d} | {info['test']['groups']}")
    print("Проверки:", "OK" if not problems else problems)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
