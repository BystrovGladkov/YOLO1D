"""Запуск эксперимента MyNet-1d на LUDB по YAML-конфигу (k-fold × сиды).

Использование:
    python run_experiment.py experiments/configs/ludb_baseline_A.yaml
    python run_experiment.py <config> --folds 0 1 --seeds 0 --epochs 3   # переопределения из CLI
    python run_experiment.py <config> --summary-only                     # только пересобрать сводку

Конфиг (пример — experiments/configs/ludb_baseline_A.yaml):
    name: ludb_baseline_A               # папка experiments/<name>/
    description: ...
    dataset: data/splits/fold_{fold}.yaml
    manifest: data/processed/ludb/manifest.csv   # для разбивки метрик по группам ритма
    folds: [0, 1, 2, 3, 4]
    seeds: [0]
    training: {...}                     # поля TrainingSettings (кроме dataset, results_path, project, seed)

Результаты (`experiments/<name>/`):
    config.yaml                  копия конфига с учётом переопределений CLI
    environment.json             git hash и состояние рабочей копии, версии, GPU, команда, время
    fold<k>_seed<s>/train<n>/    step.csv (метрики по эпохам), args.yaml (все гиперпараметры), best.pt, last.pt
    fold<k>_seed<s>/             metrics.json, metrics.md, thresholds.json, predictions_{val,test}.json
    summary.json, summary.md     по каждому прогону + пулинг тестов всех фолдов (для каждого сида)
                                 + mean ± SD по фолдам; разбивка по группам ритма
и строка в experiments/results.csv (реестр всех экспериментов).

Протокол: docs/research/R2_metrics_benchmarks.md §3 (основная метрика — F1_mean6, вторичная — sd_mean6_ms).
Готовый прогон (есть metrics.json) повторно не запускается; --rerun — запустить заново.
"""
import argparse
import csv
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))  # модули этого репозитория, а не установленной копии пакета
from yolo_1d.nets.training import set_seed  # noqa: E402
from yolo_1d.utils.metrics.delineation import DelineationMetrics, apply_thresholds, format_table  # noqa: E402
from yolo_1d.utils.settings import TrainingSettings  # noqa: E402

REGISTRY_FIELDS = ["date", "name", "git_hash", "git_dirty", "folds", "seeds", "F1_mean6_pooled",
                   "sd_mean6_ms_pooled", "F1_mean6_folds_mean", "F1_mean6_folds_sd", "config"]


def git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def environment() -> dict:
    """Версия кода и окружения для воспроизводимости."""
    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "git_hash": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform": platform.platform(),
    }


def load_config(path: Path, args) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if args.folds is not None:
        config["folds"] = args.folds
    if args.seeds is not None:
        config["seeds"] = args.seeds
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    for key in ("name", "dataset", "folds", "seeds", "training"):
        if key not in config:
            raise ValueError(f"в конфиге нет обязательного поля {key!r}")
    forbidden = {"dataset", "results_path", "project", "seed"} & set(config["training"])
    if forbidden:
        raise ValueError(f"поля {sorted(forbidden)} задаются run_experiment.py, уберите их из training")
    return config


def load_groups(manifest) -> dict:
    """ID примера → группа ритма (из manifest.csv data/prepare_ludb.py)."""
    if not manifest:
        return {}
    with open(REPO / manifest, encoding="utf-8", newline="") as f:
        return {row["sample_id"]: row["rhythm_group"] for row in csv.DictReader(f)}


def run_one(config: dict, exp_dir: Path, fold: int, seed: int, groups: dict, rerun: bool) -> Path:
    """Обучение и оценка на одном фолде с одним сидом."""
    from yolo_1d.yolo_1d import Trainer

    run_dir = exp_dir / f"fold{fold}_seed{seed}"
    if (run_dir / "metrics.json").is_file() and not rerun:
        print(f"[skip] {run_dir.name}: уже есть metrics.json")
        return run_dir
    if rerun and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    settings = TrainingSettings(**config["training"], dataset=str(REPO / config["dataset"].format(fold=fold)),
                                results_path=str(run_dir), project="train", seed=seed)
    set_seed(seed)
    start = time.time()
    trainer = Trainer(settings)
    trainer.train()
    best = Path(trainer._best_path)
    if best.is_file():
        trainer.load_checkpoint(str(best))
    result = trainer.evaluate_delineation(groups=groups, save_dir=str(run_dir))
    with open(run_dir / "run.json", "w", encoding="utf-8") as f:
        json.dump({"fold": fold, "seed": seed, "train_dir": Path(trainer._project_path).name,
                   "checkpoint": best.name if best.is_file() else "last (best.pt не создан)",
                   "duration_s": round(time.time() - start, 1),
                   "summary": result["test"]["summary"]}, f, indent=2)
    torch.cuda.empty_cache()
    return run_dir


def metrics_from_run(run_dir: Path, groups: dict) -> DelineationMetrics:
    """Пересчёт метрик теста прогона из сохранённых предсказаний и порогов (для пулинга фолдов)."""
    thresholds = json.loads((run_dir / "thresholds.json").read_text(encoding="utf-8"))
    thresholds = {k: (v["conf"], v["val_F1"]) for k, v in thresholds.items()}
    names = list(thresholds)
    # тот же режим граничных точек, что и в прогоне (в прогонах до D5 поля нет — они считались без исключения)
    run_metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    exclude = bool(run_metrics.get("exclude_boundary_points", False))
    metrics = DelineationMetrics(class_names=names, exclude_boundary_points=exclude)
    for s in json.loads((run_dir / "predictions_test.json").read_text(encoding="utf-8")):
        pred = np.asarray(s["pred"], float).reshape(-1, 4)
        metrics.add(apply_thresholds(pred, thresholds, names), np.asarray(s["ref"], float).reshape(-1, 3),
                    group=groups.get(s["id"]))
    return metrics


def summarize(config: dict, exp_dir: Path, groups: dict) -> dict:
    """Сводка: каждый прогон, пулинг тестов всех фолдов для каждого сида, mean ± SD по фолдам."""
    runs, pooled = [], {}
    for seed in config["seeds"]:
        merged = None
        for fold in config["folds"]:
            run_dir = exp_dir / f"fold{fold}_seed{seed}"
            if not (run_dir / "metrics.json").is_file():
                continue
            m = metrics_from_run(run_dir, groups)
            runs.append({"fold": fold, "seed": seed, **m.compute()["summary"]})
            merged = m if merged is None else merged.merge(m)
        if merged is not None:
            pooled[seed] = {"all": merged.compute(), **{g: merged.compute(g) for g in merged.groups}}
    f1 = [r["F1_mean6"] for r in runs]
    sd = [r["sd_mean6_ms"] for r in runs]
    summary = {
        "name": config["name"], "runs": runs,
        "folds_mean_sd": {"F1_mean6": [float(np.mean(f1)), float(np.std(f1))] if f1 else None,
                          "sd_mean6_ms": [float(np.mean(sd)), float(np.std(sd))] if sd else None},
        "pooled": pooled,
    }
    (exp_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float),
                                          encoding="utf-8")
    lines = [f"# {config['name']}", "", config.get("description", ""), "",
             "## Прогоны (test каждого фолда)", "", "| fold | seed | F1_mean6, % | sd_mean6, мс | |m|_mean6, мс |",
             "|---|---|---|---|---|"]
    lines += [f"| {r['fold']} | {r['seed']} | {100 * r['F1_mean6']:.2f} | {r['sd_mean6_ms']:.1f} | "
              f"{r['mae_mean6_ms']:.1f} |" for r in runs]
    if f1:
        lines += ["", f"**mean ± SD по прогонам:** F1_mean6 = {100 * np.mean(f1):.2f} ± {100 * np.std(f1):.2f} %, "
                      f"sd_mean6 = {np.mean(sd):.1f} ± {np.std(sd):.1f} мс"]
    for seed, res in pooled.items():
        lines += ["", f"## Пулинг тестов всех фолдов (seed {seed})", "", format_table(res["all"])]
        for g, r in res.items():
            if g != "all":
                lines += ["", f"### Группа ритма: {g}", "", format_table(r)]
    (exp_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def append_registry(config: dict, env: dict, summary: dict, config_path: Path, out: Path) -> None:
    """Строка эксперимента в <out>/results.csv (реестр всех запусков, по умолчанию experiments/results.csv)."""
    registry = out / "results.csv"
    pooled = next(iter(summary["pooled"].values()), None)
    fm = summary["folds_mean_sd"]["F1_mean6"] or [float("nan")] * 2
    row = {"date": env["date"], "name": config["name"], "git_hash": env["git_hash"][:10],
           "git_dirty": env["git_dirty"], "folds": " ".join(map(str, config["folds"])),
           "seeds": " ".join(map(str, config["seeds"])),
           "F1_mean6_pooled": f"{pooled['all']['summary']['F1_mean6']:.4f}" if pooled else "",
           "sd_mean6_ms_pooled": f"{pooled['all']['summary']['sd_mean6_ms']:.2f}" if pooled else "",
           "F1_mean6_folds_mean": f"{fm[0]:.4f}", "F1_mean6_folds_sd": f"{fm[1]:.4f}",
           "config": config_path.as_posix()}
    # одна строка на эксперимент: прежняя строка с тем же именем заменяется
    rows = []
    if registry.is_file():
        with open(registry, encoding="utf-8", newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["name"] != config["name"]]
    registry.parent.mkdir(parents=True, exist_ok=True)
    with open(registry, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        writer.writerows(rows + [row])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--folds", type=int, nargs="+", help="переопределить список фолдов")
    parser.add_argument("--seeds", type=int, nargs="+", help="переопределить список сидов")
    parser.add_argument("--epochs", type=int, help="переопределить training.epochs")
    parser.add_argument("--out", type=Path, default=REPO / "experiments", help="корневая папка экспериментов")
    parser.add_argument("--rerun", action="store_true", help="перезапустить уже готовые прогоны")
    parser.add_argument("--summary-only", action="store_true", help="не обучать, только пересобрать сводку")
    args = parser.parse_args()

    config = load_config(args.config, args)
    exp_dir = args.out / config["name"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    groups = load_groups(config.get("manifest"))
    env = environment()
    if not args.summary_only:
        (exp_dir / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
                                             encoding="utf-8")
        (exp_dir / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
        if env["git_dirty"]:
            print("ВНИМАНИЕ: в рабочей копии есть незакоммиченные изменения — git hash не описывает код полностью.")
        for fold in config["folds"]:
            for seed in config["seeds"]:
                print(f"\n===== {config['name']}: fold {fold}, seed {seed} =====")
                run_one(config, exp_dir, fold, seed, groups, args.rerun)
    summary = summarize(config, exp_dir, groups)
    if not args.summary_only:
        append_registry(config, env, summary, args.config, args.out)
    print((exp_dir / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
