"""Отчёт о датасете LUDB: сверка с публикацией, статистика разметки, примеры визуализаций.

Читает `data/raw/ludb/` модулем `yolo_1d.utils.datasets.ludb` и пишет в `data/reports/`:
- `ludb_report.md` — проверки (PASS/FAIL) против чисел из статьи и README LUDB, статистика;
- `ludb_stats.json` — те же числа в машиночитаемом виде;
- `figures/*.png` — распределения длительностей и краёв, примеры записей.

Использование:
    python data/download_ludb.py     # если данные ещё не скачаны
    python data/ludb_report.py       # [--root data/raw/ludb] [--out data/reports]

Код возврата 1, если хотя бы одна проверка не прошла.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import wfdb  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))  # модули этого репозитория, а не установленной копии пакета
from yolo_1d.utils.datasets.ludb import (CLASS_NAMES, FS, LEADS, PEAK_SYMBOLS, SIGNAL_LENGTH,  # noqa: E402
                                iter_records, rhythm_group)

# Числа из публикации (Kalyakulina et al., IEEE Access 2020) и README LUDB v1.0.1
PAPER = {
    "records": 200,
    "waves": {"P": 16797, "QRS": 21966, "T": 19666},
    "waves_total": 58429,
    "sex": {"F": 85, "M": 115},
    "age_min": 11, "age_max": 90, "age_mean": 52,
}
README_RHYTHMS = {
    "Sinus rhythm": 143, "Sinus tachycardia": 4, "Sinus bradycardia": 25, "Sinus arrhythmia": 8,
    "Irregular sinus rhythm": 2, "Atrial fibrillation": 15, "Atrial flutter, typical": 3,
}
# допуск сверки числа волн с публикацией (расхождение в самих файлах v1.0.1 — 6 из 58 429)
WAVE_TOLERANCE = 0.001
COLORS = {"P": "tab:green", "QRS": "tab:red", "T": "tab:blue"}


def count_peak_symbols(root: Path, records) -> collections.Counter:
    """Число символов пиков по классам — так считаются волны в публикации LUDB."""
    counts = collections.Counter()
    for rec in records:
        for lead in LEADS:
            for sym in wfdb.rdann(str(root / "data" / rec.name), lead).symbol:
                if sym in PEAK_SYMBOLS:
                    counts[PEAK_SYMBOLS[sym]] += 1
    return counts


def parse_age(values) -> float:
    """Возраст из ludb.csv; значения вида '>89' трактуются как 90."""
    text = values[0] if values else ""
    return float(text.lstrip(">")) + (1 if text.startswith(">") else 0)


def describe(values) -> dict:
    v = np.asarray(values, dtype=float)
    return {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std()), "min": float(v.min()),
            "p1": float(np.percentile(v, 1)), "p50": float(np.median(v)),
            "p99": float(np.percentile(v, 99)), "max": float(v.max())}


def collect(records) -> dict:
    """Вся статистика по разобранной разметке."""
    waves = collections.Counter()
    durations = collections.defaultdict(list)
    issues = collections.Counter()
    issue_records = collections.defaultdict(set)
    first_onset, last_offset, zone_len, unannotated = [], [], [], []
    count_differs = collections.Counter()
    no_p_lead_ii = collections.Counter()
    groups = collections.Counter()
    leads_with_dropped, leads_dropped_outside = [], []
    for rec in records:
        groups[rhythm_group(rec.meta)] += 1
        for lead in LEADS:
            for w in rec.waves[lead]:
                waves[w.cls] += 1
                durations[w.cls].append((w.offset - w.onset) * 1000 / FS)
            if rec.dropped(lead):
                start, end = rec.zone(lead)
                inside = any(start <= issue.sample <= end for issue in rec.dropped(lead))
                (leads_with_dropped if inside else leads_dropped_outside).append(f"{rec.name}.{lead}")
            for issue in rec.issues[lead]:
                issues[(issue.kind, issue.kept)] += 1
                issue_records[issue.kind].add(rec.name)
            start, end = rec.zone(lead)
            first_onset.append(start / FS)
            last_offset.append(end / FS)
            zone_len.append((end - start) / FS)
            unannotated.append(1 - (end - start) / SIGNAL_LENGTH)
        for cls in CLASS_NAMES:
            n = [sum(w.cls == cls for w in rec.waves[lead]) for lead in LEADS]
            count_differs[cls] += max(n) != min(n)
        if not any(w.cls == "P" for w in rec.waves["ii"]):
            no_p_lead_ii[rhythm_group(rec.meta)] += 1
    return {
        "waves": dict(waves),
        "durations_ms": {c: describe(v) for c, v in durations.items()},
        "durations_raw": durations,
        "issues": {f"{k}{'' if kept else ' (отброшены)'}": n for (k, kept), n in issues.items()},
        "issue_records": {k: sorted(v, key=int) for k, v in issue_records.items()},
        "first_onset_s": describe(first_onset), "last_offset_s": describe(last_offset),
        "zone_s": describe(zone_len), "unannotated_fraction": describe(unannotated),
        "edges_raw": (first_onset, last_offset),
        "records_with_lead_count_mismatch": dict(count_differs),
        "records_without_p_in_lead_ii": dict(no_p_lead_ii),
        "rhythm_groups": dict(groups),
        "leads_with_dropped_waves": leads_with_dropped,
        "leads_with_dropped_waves_outside_zone": leads_dropped_outside,
    }


def checks(records, stats, peak_symbols) -> list:
    """Сверка с публикацией и README. Возвращает список (проверка, ожидание, факт, ok)."""
    rows = []

    def add(name, expected, actual, ok=None):
        rows.append((name, expected, actual, expected == actual if ok is None else ok))

    add("Число записей", PAPER["records"], len(records))
    shapes = {rec.signal.shape for rec in records}
    add("Форма сигнала (отведения × отсчёты)", {(len(LEADS), SIGNAL_LENGTH)}, shapes)
    add("У каждого отведения есть разметка", True,
        all(rec.waves[lead] for rec in records for lead in LEADS))
    # В файлах v1.0.1 на 6 символов пика меньше, чем заявлено в README/статье (QRS −1, T −5);
    # это свойство исходных данных, поэтому число волн сверяется с допуском WAVE_TOLERANCE.
    for cls in CLASS_NAMES:
        add(f"Волн {cls} (символов пика, допуск {WAVE_TOLERANCE:.1%})", PAPER["waves"][cls], peak_symbols[cls],
            abs(peak_symbols[cls] - PAPER["waves"][cls]) <= WAVE_TOLERANCE * PAPER["waves"][cls])
    total = sum(peak_symbols.values())
    add(f"Волн всего (символов пика, допуск {WAVE_TOLERANCE:.1%})", PAPER["waves_total"], total,
        abs(total - PAPER["waves_total"]) <= WAVE_TOLERANCE * PAPER["waves_total"])
    sex = collections.Counter(rec.meta.get("Sex", ["?"])[0] for rec in records)
    add("Пол Ж / М", PAPER["sex"], {"F": sex["F"], "M": sex["M"]})
    ages = [parse_age(rec.meta.get("Age", [])) for rec in records]
    add("Возраст min–max, среднее",
        f"{PAPER['age_min']}–{PAPER['age_max']}, ≈{PAPER['age_mean']}",
        f"{min(ages):.0f}–{max(ages):.0f}, {np.mean(ages):.1f}",
        min(ages) == PAPER["age_min"] and max(ages) == PAPER["age_max"]
        and round(np.mean(ages)) == PAPER["age_mean"])
    primary = collections.Counter(rec.meta.get("Rhythms", ["?"])[0] for rec in records)
    add("Ритмы (первый в списке) = таблица README", README_RHYTHMS, dict(primary))
    return rows


def plot_distributions(stats, out: Path) -> list:
    """Гистограммы длительностей волн и положения краёв размеченной зоны."""
    files = []
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(0, 420, 8)
    for cls in CLASS_NAMES:
        ax.hist(stats["durations_raw"][cls], bins=bins, alpha=0.55, label=cls, color=COLORS[cls])
    ax.set_xlabel("Длительность волны, мс")
    ax.set_ylabel("Число волн")
    ax.set_title("LUDB: длительности волн (все отведения)")
    ax.legend()
    fig.tight_layout()
    files.append(out / "durations.png")
    fig.savefig(files[-1], dpi=120)
    plt.close(fig)

    first, last = stats["edges_raw"]
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(0, 10.05, 0.1)
    ax.hist(first, bins=bins, alpha=0.6, label="первый onset")
    ax.hist(last, bins=bins, alpha=0.6, label="последний offset")
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Число отведений")
    ax.set_title("LUDB: границы размеченной зоны (2400 отведений)")
    ax.legend()
    fig.tight_layout()
    files.append(out / "edges.png")
    fig.savefig(files[-1], dpi=120)
    plt.close(fig)
    return files


def plot_record(rec, leads, title: str, path: Path) -> None:
    """Сигнал выбранных отведений с размеченными волнами, неразмеченными краями и дефектами."""
    t = np.arange(SIGNAL_LENGTH) / FS
    fig, axes = plt.subplots(len(leads), 1, figsize=(14, 2.3 * len(leads)), sharex=True)
    for ax, lead in zip(np.atleast_1d(axes), leads):
        ax.plot(t, rec.signal[LEADS.index(lead)], color="black", linewidth=0.8)
        for w in rec.waves[lead]:
            ax.axvspan(w.onset / FS, w.offset / FS, color=COLORS[w.cls], alpha=0.25, linewidth=0)
        start, end = rec.zone(lead)
        ax.axvspan(0, start / FS, color="gray", alpha=0.2, hatch="//", linewidth=0)
        ax.axvspan(end / FS, SIGNAL_LENGTH / FS, color="gray", alpha=0.2, hatch="//", linewidth=0)
        for issue in rec.issues[lead]:
            ax.axvline(issue.sample / FS, color="magenta" if not issue.kept else "orange",
                       linestyle="--", linewidth=1)
        ax.set_ylabel(f"{lead}, мВ")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[c], alpha=0.4) for c in CLASS_NAMES]
    handles += [plt.Rectangle((0, 0), 1, 1, color="gray", alpha=0.3, hatch="//"),
                plt.Line2D([], [], color="orange", linestyle="--"),
                plt.Line2D([], [], color="magenta", linestyle="--")]
    labels = CLASS_NAMES + ["не размечено", "дефект (исправлен)", "дефект (волна отброшена)"]
    np.atleast_1d(axes)[0].legend(handles, labels, loc="upper right", ncol=6, fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel("Время, с")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_examples(records, out: Path) -> list:
    """Примеры: синусовый ритм, ФП, ЭКС с дефектами, дубли разметки."""
    by_name = {rec.name: rec for rec in records}
    af = next(rec for rec in records if rhythm_group(rec.meta) == "af_flutter")
    examples = [
        (by_name["1"], ["i", "ii", "v1"], "Запись 1: синусовый ритм"),
        (af, ["i", "ii", "v1"], f"Запись {af.name}: {', '.join(af.meta.get('Rhythms', []))} (нет P-волн)"),
        (by_name["104"], ["ii", "v1"], "Запись 104: ЭКС — QRS размечены без onset (отброшены)"),
        (by_name["116"], ["v1", "v2", "v4"], "Запись 116: вложенные дубли T (слиты) и T без offset"),
    ]
    files = []
    for rec, leads, title in examples:
        path = out / f"record_{rec.name}.png"
        plot_record(rec, leads, title, path)
        files.append(path)
    return files


def fmt(d: dict, digits=0) -> str:
    return " | ".join(f"{d[k]:.{digits}f}" for k in ["mean", "sd", "min", "p1", "p50", "p99", "max"])


def write_markdown(path: Path, rows, stats, peak_symbols, figures) -> None:
    rel = lambda p: p.relative_to(path.parent).as_posix()  # noqa: E731
    lines = ["# LUDB: отчёт о данных", "",
             "Сгенерировано `data/ludb_report.py`. Особенности разметки и выводы — "
             "[docs/research/R1_ludb.md](../../docs/research/R1_ludb.md).", "",
             "## Сверка с публикацией и README", "",
             "| Проверка | Ожидание | Факт | |", "|---|---|---|---|"]
    lines += [f"| {n} | {e} | {a} | {'PASS' if ok else 'FAIL'} |" for n, e, a, ok in rows]
    parsed = stats["waves"]
    lines += ["", "## Разобранные волны", "",
              "Публикация считает волны по символам пиков. Модуль `yolo_1d.utils.datasets.ludb` возвращает только волны, "
              "из которых строится отрезок [onset, offset]: дефектные исправляются или отбрасываются.", "",
              "В файлах разметки LUDB v1.0.1 символов пика **58 423**, тогда как README и статья указывают 58 429 "
              "(QRS 21 966 → 21 965, T 19 666 → 19 661, P совпадает). Расхождение есть в исходных файлах, "
              "оно не связано с чтением: символы считаются напрямую через `wfdb.rdann`.", "",
              "| Класс | Символов пика | Волн после разбора | Разница |", "|---|---|---|---|"]
    for cls in CLASS_NAMES:
        lines.append(f"| {cls} | {peak_symbols[cls]} | {parsed[cls]} | {parsed[cls] - peak_symbols[cls]} |")
    lines.append(f"| Всего | {sum(peak_symbols.values())} | {sum(parsed.values())} | "
                 f"{sum(parsed.values()) - sum(peak_symbols.values())} |")
    lines += ["", "Разница складывается из дефектов ниже: отброшенные волны (`отброшены`) уменьшают счёт на 1 каждая, "
              "`merged_duplicates` сливает k вложенных копий одной волны в одну (−(k−1)).",
              "", "### Дефекты разметки", "", "| Вид | Число | Записи |", "|---|---|---|"]
    for kind, n in sorted(stats["issues"].items()):
        recs = stats["issue_records"][kind.split(" ")[0]]
        lines.append(f"| {kind} | {n} | {', '.join(recs)} |")
    dropped = stats["leads_with_dropped_waves"]
    lines += ["", f"**Отведений с отброшенными волнами: {len(dropped)}** — в сигнале внутри размеченной зоны "
              "есть волна без метки; для обучения такие отведения нужно исключить или обработать (D3): "
              + ", ".join(dropped) + ".",
              "", "Отведения, где отброшенные волны лежат только вне размеченной зоны (они будут обнулены "
              "маскированием краёв, проблем нет): " + ", ".join(stats["leads_with_dropped_waves_outside_zone"]) + "."]
    lines += ["", "## Длительности волн, мс", "",
              "| Класс | среднее | SD | min | p1 | p50 | p99 | max |", "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {c} | {fmt(stats['durations_ms'][c])} |" for c in CLASS_NAMES]
    lines += ["", "## Неразмеченные края (по 2400 отведениям)", "",
              "| Величина | среднее | SD | min | p1 | p50 | p99 | max |", "|---|---|---|---|---|---|---|---|",
              f"| Первый onset, с | {fmt(stats['first_onset_s'], 2)} |",
              f"| Последний offset, с | {fmt(stats['last_offset_s'], 2)} |",
              f"| Размеченная зона, с | {fmt(stats['zone_s'], 2)} |",
              f"| Доля неразмеченного сигнала | {fmt(stats['unannotated_fraction'], 3)} |",
              "", "## Прочее", "",
              f"- Группы ритма (для стратификации D4): {stats['rhythm_groups']}",
              f"- Записей без P в отведении II, по группам: {stats['records_without_p_in_lead_ii']}",
              f"- Записей, где число волн класса различается между отведениями: "
              f"{stats['records_with_lead_count_mismatch']}",
              "", "## Рисунки", ""]
    lines += [f"![{p.stem}]({rel(p)})" for p in figures]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=REPO / "data" / "raw" / "ludb")
    parser.add_argument("--out", type=Path, default=REPO / "data" / "reports")
    args = parser.parse_args()
    if not (args.root / "RECORDS").is_file():
        print(f"Нет данных в {args.root}. Сначала: python data/download_ludb.py")
        return 1
    fig_dir = args.out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Чтение записей...")
    records = list(iter_records(args.root))
    peak_symbols = count_peak_symbols(args.root, records)
    stats = collect(records)
    rows = checks(records, stats, peak_symbols)
    figures = plot_distributions(stats, fig_dir) + plot_examples(records, fig_dir)

    write_markdown(args.out / "ludb_report.md", rows, stats, peak_symbols, figures)
    json_stats = {k: v for k, v in stats.items() if not k.endswith("_raw")}
    json_stats["peak_symbols"] = dict(peak_symbols)
    json_stats["checks"] = [{"check": n, "expected": str(e), "actual": str(a), "ok": bool(ok)}
                            for n, e, a, ok in rows]
    (args.out / "ludb_stats.json").write_text(json.dumps(json_stats, ensure_ascii=False, indent=2),
                                              encoding="utf-8")

    for name, expected, actual, ok in rows:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {actual}" + ("" if ok else f" (ожидалось {expected})"))
    print(f"Отчёт: {args.out / 'ludb_report.md'}")
    return 0 if all(ok for *_, ok in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
