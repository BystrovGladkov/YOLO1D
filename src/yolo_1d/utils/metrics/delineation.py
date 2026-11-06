from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

FS = 500
TOLERANCE_MS = 150.0
CLASS_NAMES = ("P", "QRS", "T")
FIDUCIALS = tuple(f"{c}_{e}" for c in CLASS_NAMES for e in ("on", "off"))
# 2σ_CSE, мс (CSE Working Party 1985, цит. по Kalyakulina et al. 2020, табл. 6); для T_on не задан
CSE_2SIGMA_MS = {"P_on": 10.2, "P_off": 12.7, "QRS_on": 6.5, "QRS_off": 11.6, "T_off": 30.6}


def _nanmean(values) -> float:
    values = np.asarray(values, float)
    return float(np.nanmean(values)) if np.any(~np.isnan(values)) else float("nan")


def match_points(pred: np.ndarray, ref: np.ndarray, tolerance: float) -> Tuple[np.ndarray, np.ndarray]:
    """Жадное сопоставление один-к-одному по возрастанию |pred − ref| в пределах tolerance.

    Returns
    -------
    pairs
        (k, 2) индексы сопоставленных (pred, ref).
    errors
        (k,) pred − ref для каждой пары.
    """
    pred, ref = np.asarray(pred, float), np.asarray(ref, float)
    if pred.size == 0 or ref.size == 0:
        return np.zeros((0, 2), int), np.zeros(0)
    diff = pred[:, None] - ref[None, :]
    i, j = np.nonzero(np.abs(diff) <= tolerance)
    order = np.argsort(np.abs(diff[i, j]), kind="stable")
    used_p, used_r, pairs = set(), set(), []
    for k in order:
        p, r = i[k], j[k]
        if p not in used_p and r not in used_r:
            used_p.add(p)
            used_r.add(r)
            pairs.append((p, r))
    pairs = np.asarray(pairs, int).reshape(-1, 2)
    return pairs, diff[pairs[:, 0], pairs[:, 1]] if len(pairs) else np.zeros(0)


def class_zones(ref: np.ndarray, num_classes: int = len(CLASS_NAMES)) -> Dict[int, Optional[Tuple[float, float]]]:
    """Зона разметки каждого класса: [первый onset_c, последний offset_c]; для класса без волн —
    зона по всем классам; если разметки нет вовсе — None (исключения нет)."""
    ref = np.asarray(ref, float).reshape(-1, 3)
    whole = (ref[:, 1].min(), ref[:, 2].max()) if len(ref) else None
    zones = {}
    for c in range(num_classes):
        rc = ref[ref[:, 0] == c]
        zones[c] = (rc[:, 1].min(), rc[:, 2].max()) if len(rc) else whole
    return zones


def filter_to_zone(pred: np.ndarray, ref: np.ndarray, num_classes: int = len(CLASS_NAMES)) -> np.ndarray:
    """Убирает детекции, центр которых вне зоны разметки своего класса (R2 §3.2)."""
    pred = np.asarray(pred, float)
    if len(pred) == 0:
        return pred
    zones = class_zones(ref, num_classes)
    keep = np.ones(len(pred), bool)
    centers = (pred[:, 1] + pred[:, 2]) / 2
    for c, zone in zones.items():
        if zone is not None:
            mask = pred[:, 0] == c
            keep[mask] = (centers[mask] >= zone[0]) & (centers[mask] <= zone[1])
    return pred[keep]


@dataclass
class _Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    errors: List[float] = field(default_factory=list)  # мс

    def add(self, other: "_Counts") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.errors.extend(other.errors)

    def summary(self, n_leads: int = 0) -> Dict[str, float]:
        """Se, PPV, F1, счётчики и ошибки; FP_per_lead — ложные срабатывания на отведение.

        Если в разметке нет ни одной точки (TP + FN = 0, например P при ФП), Se и F1 не определены
        (NaN): качество такой точки описывает только FP_per_lead. Если разметка есть, но
        предсказаний нет, F1 = 0.
        """
        se = self.tp / (self.tp + self.fn) if self.tp + self.fn else float("nan")
        ppv = self.tp / (self.tp + self.fp) if self.tp + self.fp else float("nan")
        if self.tp + self.fn == 0:
            f1 = float("nan")
        else:
            f1 = 2 * se * ppv / (se + ppv) if se + ppv > 0 else 0.0
        err = np.asarray(self.errors, float)
        return {"Se": se, "PPV": ppv, "F1": f1, "TP": self.tp, "FP": self.fp, "FN": self.fn,
                "FP_per_lead": self.fp / n_leads if n_leads else float("nan"),
                "mean_ms": float(err.mean()) if err.size else float("nan"),
                "sd_ms": float(err.std()) if err.size else float("nan"),
                "mae_ms": float(np.abs(err).mean()) if err.size else float("nan")}


class DelineationMetrics:
    """Накопитель метрик делинеации с micro-пулингом по отведениям, группам и фолдам.

    Пример:
        metrics = DelineationMetrics()
        for pred, ref, group in samples:
            metrics.add(pred, ref, group=group)
        result = metrics.compute()          # 6 точек + сводка ⟨F1⟩₆, ⟨σ⟩₆
        by_rhythm = metrics.compute("af_flutter")
    """

    def __init__(self, fs: int = FS, tolerance_ms: float = TOLERANCE_MS, exclude_edges: bool = True,
                 class_names: Sequence[str] = CLASS_NAMES, exclude_boundary_points: bool = False):
        self.fs = fs
        self.tolerance = tolerance_ms * fs / 1000  # в отсчётах
        self.exclude_edges = exclude_edges
        self.exclude_boundary_points = exclude_boundary_points
        self.class_names = tuple(class_names)
        self.fiducials = tuple(f"{c}_{e}" for c in self.class_names for e in ("on", "off"))
        self._counts: Dict[Optional[str], Dict[str, _Counts]] = defaultdict(
            lambda: {f: _Counts() for f in self.fiducials})
        self._waves: Dict[Optional[str], Dict[str, _Counts]] = defaultdict(
            lambda: {c: _Counts() for c in self.class_names})
        self._leads: Dict[Optional[str], int] = defaultdict(int)  # число отведений (для FP_per_lead)

    def add(self, pred: np.ndarray, ref: np.ndarray, group: Optional[str] = None) -> None:
        """Добавляет одно отведение: pred (n, 3|4) [class, left, right(, conf)], ref (m, 3) [class, onset, offset]."""
        pred = np.asarray(pred, float)
        pred = np.zeros((0, 3)) if pred.size == 0 else pred.reshape(-1, pred.shape[-1])[:, :3]
        ref = np.asarray(ref, float).reshape(-1, 3)
        if self.exclude_edges:
            pred = filter_to_zone(pred, ref, len(self.class_names))
        targets = [None] if group is None else [None, group]
        for g in targets:
            self._leads[g] += 1
        # граничные точки: onset первой и offset последней волны отведения (индексы в ref)
        boundary = {}
        if self.exclude_boundary_points and len(ref):
            boundary = {1: int(np.argmin(ref[:, 1])), 2: int(np.argmax(ref[:, 2]))}
        for c, name in enumerate(self.class_names):
            idx = np.nonzero(ref[:, 0] == c)[0]
            pc, rc = pred[pred[:, 0] == c], ref[idx]
            for col, edge in ((1, "on"), (2, "off")):
                pairs, err = match_points(pc[:, col], rc[:, col], self.tolerance)
                excluded = np.nonzero(idx == boundary[col])[0] if col in boundary else np.zeros(0, int)
                counts = self._count(pc[:, col], rc[:, col], pairs, err, excluded)
                for g in targets:
                    self._counts[g][f"{name}_{edge}"].add(counts)
            wave = self._match_waves(pc, rc)
            for g in targets:
                self._waves[g][name].add(wave)

    def _count(self, pred_pts, ref_pts, pairs, err, excluded) -> _Counts:
        """TP/FP/FN и ошибки точки; точки разметки `excluded` и связанные с ними предсказания
        (сопоставленные или лежащие в пределах допуска) не учитываются."""
        if len(excluded) == 0:
            return _Counts(len(pairs), len(pred_pts) - len(pairs), len(ref_pts) - len(pairs),
                           list(err * 1000 / self.fs))
        excluded = set(int(e) for e in excluded)
        keep = np.array([int(r) not in excluded for r in pairs[:, 1]], bool) if len(pairs) else np.zeros(0, bool)
        matched_pred = set(int(p) for p in pairs[:, 0])
        near = np.abs(np.asarray(pred_pts)[:, None] - np.asarray(ref_pts)[sorted(excluded)][None, :]) <= self.tolerance
        ignored_pred = {i for i in range(len(pred_pts)) if i not in matched_pred and near[i].any()}
        tp = int(keep.sum())
        fp = len(pred_pts) - len(matched_pred) - len(ignored_pred)
        fn = len(ref_pts) - len(excluded) - tp
        return _Counts(tp, fp, fn, list(err[keep] * 1000 / self.fs))

    @staticmethod
    def _match_waves(pc: np.ndarray, rc: np.ndarray) -> _Counts:
        """Уровень волн (по Jiménez-Pérez 2024): пара совпадает, если центр одной волны лежит внутри
        другой; один-к-одному по возрастанию расстояния между центрами."""
        if len(pc) == 0 or len(rc) == 0:
            return _Counts(0, len(pc), len(rc))
        cp, cr = (pc[:, 1] + pc[:, 2]) / 2, (rc[:, 1] + rc[:, 2]) / 2
        ok = (((cp[:, None] >= rc[None, :, 1]) & (cp[:, None] <= rc[None, :, 2]))
              | ((cr[None, :] >= pc[:, None, 1]) & (cr[None, :] <= pc[:, None, 2])))
        dist = np.where(ok, np.abs(cp[:, None] - cr[None, :]), np.inf)
        i, j = np.nonzero(np.isfinite(dist))
        used_p, used_r, tp = set(), set(), 0
        for k in np.argsort(dist[i, j], kind="stable"):
            if i[k] not in used_p and j[k] not in used_r:
                used_p.add(i[k])
                used_r.add(j[k])
                tp += 1
        return _Counts(tp, len(pc) - tp, len(rc) - tp)

    def merge(self, other: "DelineationMetrics") -> "DelineationMetrics":
        """Добавляет счётчики другого накопителя (пулинг фолдов)."""
        for g, counts in other._counts.items():
            for f, c in counts.items():
                self._counts[g][f].add(c)
        for g, counts in other._waves.items():
            for name, c in counts.items():
                self._waves[g][name].add(c)
        for g, n in other._leads.items():
            self._leads[g] += n
        return self

    @property
    def groups(self) -> List[str]:
        return sorted(g for g in self._counts if g is not None)

    def compute(self, group: Optional[str] = None) -> Dict[str, object]:
        """Метрики по 6 точкам, по волнам и сводка (для всех отведений или одной группы).

        Сводные средние (F1_mean6, sd_mean6_ms, mae_mean6_ms, cse_fraction) берутся по точкам, для
        которых метрика определена; n_defined — число таких точек (F1 определён), n_leads — отведений.
        """
        n_leads = self._leads[group]
        points = {f: c.summary(n_leads) for f, c in self._counts[group].items()}
        waves = {name: c.summary(n_leads) for name, c in self._waves[group].items()}
        f1 = [points[f]["F1"] for f in self.fiducials]
        sd = [points[f]["sd_ms"] for f in self.fiducials]
        mae = [points[f]["mae_ms"] for f in self.fiducials]
        cse = [points[f]["sd_ms"] <= limit for f, limit in CSE_2SIGMA_MS.items()
               if f in points and not np.isnan(points[f]["sd_ms"])]
        summary = {"F1_mean6": _nanmean(f1), "sd_mean6_ms": _nanmean(sd), "mae_mean6_ms": _nanmean(mae),
                   "cse_fraction": float(np.mean(cse)) if cse else float("nan"),
                   "n_defined": int(np.sum(~np.isnan(f1))), "n_leads": n_leads}
        return {"points": points, "waves": waves, "summary": summary}


def boxes_to_intervals(cls: np.ndarray, boxes: np.ndarray, input_length: int, pad_offset: int = 0,
                       signal_length: Optional[int] = None) -> np.ndarray:
    """Разметка Dataset → (m, 3) [class, onset, offset] в отсчётах исходного сигнала.

    cls (m,) или (m, 1); boxes (m, 2) — (center, width), нормированные на длину входа сети.
    """
    cls = np.asarray(cls, float).reshape(-1)
    boxes = np.asarray(boxes, float).reshape(-1, 2)
    left = (boxes[:, 0] - boxes[:, 1] / 2) * input_length - pad_offset
    right = (boxes[:, 0] + boxes[:, 1] / 2) * input_length - pad_offset
    out = np.stack([cls, left, right], axis=1)
    return clip_to_signal(out, signal_length) if signal_length else out


def detections_to_intervals(det: np.ndarray, pad_offset: int = 0, signal_length: Optional[int] = None) -> np.ndarray:
    """Выход `metrics.detection.non_max_suppression` (n, 4) [x1, x2, conf, cls] в отсчётах входа →
    (n, 4) [class, left, right, conf] в отсчётах исходного сигнала (вычитается смещение паддинга)."""
    det = np.asarray(det, float).reshape(-1, 4)
    out = np.stack([det[:, 3], det[:, 0] - pad_offset, det[:, 1] - pad_offset, det[:, 2]], axis=1)
    return clip_to_signal(out, signal_length) if signal_length else out


def clip_to_signal(intervals: np.ndarray, signal_length: int) -> np.ndarray:
    """Обрезает интервалы по [0, signal_length]; полностью выходящие за сигнал отбрасывает."""
    out = np.asarray(intervals, float).copy()
    out[:, 1:3] = out[:, 1:3].clip(0, signal_length)
    return out[out[:, 2] > out[:, 1]]


def best_class_thresholds(samples: Iterable[Tuple[np.ndarray, np.ndarray]],
                          grid: Sequence[float] = tuple(np.round(np.arange(0.05, 0.96, 0.05), 2)),
                          class_names: Sequence[str] = CLASS_NAMES, **metric_kwargs) -> Dict[str, Tuple[float, float]]:
    """Порог уверенности для каждого класса, максимизирующий F1 класса (среднее F1 onset/offset)
    на валидационных отведениях (R2 §3.1, п. 2).

    samples
        Пары (pred (n, 4) [class, left, right, conf], ref (m, 3)); pred — с низким порогом NMS.
    Returns
        {класс: (порог, F1)}.
    """
    samples = [(np.asarray(p, float).reshape(-1, 4), np.asarray(r, float).reshape(-1, 3)) for p, r in samples]
    best = {}
    for c, name in enumerate(class_names):
        scores = []
        for thr in grid:
            metrics = DelineationMetrics(class_names=class_names, **metric_kwargs)
            for pred, ref in samples:
                keep = (pred[:, 0] != c) | (pred[:, 3] >= thr)
                metrics.add(pred[keep], ref)
            points = metrics.compute()["points"]
            scores.append((points[f"{name}_on"]["F1"] + points[f"{name}_off"]["F1"]) / 2)
        # класс без разметки на val (F1 не определён): порог по минимуму ложных срабатываний — максимальный
        scores = np.nan_to_num(np.asarray(scores, float), nan=-1.0)
        k = int(np.argmax(scores)) if np.any(scores >= 0) else len(grid) - 1
        best[name] = (float(grid[k]), float(scores[k]))
    return best


def apply_thresholds(pred: np.ndarray, thresholds: Dict[str, Tuple[float, float]],
                     class_names: Sequence[str] = CLASS_NAMES) -> np.ndarray:
    """Оставляет детекции с уверенностью не ниже порога своего класса."""
    pred = np.asarray(pred, float).reshape(-1, 4)
    keep = np.zeros(len(pred), bool)
    for c, name in enumerate(class_names):
        keep |= (pred[:, 0] == c) & (pred[:, 3] >= thresholds[name][0])
    return pred[keep]


def format_table(result: Dict[str, object]) -> str:
    """Таблица Markdown: Se / PPV / F1 / m ± σ / FP на отведение по 6 точкам + сводка.

    Не определённые значения (нет разметки или нет совпадений) выводятся как «—».
    """
    pct = lambda v: "—" if np.isnan(v) else f"{100 * v:.2f}"  # noqa: E731
    lines = ["| Точка | Se, % | PPV, % | F1, % | m ± σ, мс | FP/отв. |", "|---|---|---|---|---|---|"]
    for f, p in result["points"].items():
        err = "—" if np.isnan(p["mean_ms"]) else f"{p['mean_ms']:.1f} ± {p['sd_ms']:.1f}"
        fp = "—" if np.isnan(p.get("FP_per_lead", float("nan"))) else f"{p['FP_per_lead']:.3f}"
        lines.append(f"| {f} | {pct(p['Se'])} | {pct(p['PPV'])} | {pct(p['F1'])} | {err} | {fp} |")
    s = result["summary"]
    n = s.get("n_defined", len(result["points"]))
    note = "" if n == len(result["points"]) else f" (по {n} определённым точкам)"
    cse = "—" if np.isnan(s["cse_fraction"]) else f"{100 * s['cse_fraction']:.0f} %"
    lines.append(f"\n⟨F1⟩₆ = {pct(s['F1_mean6'])} %{note}, ⟨σ⟩₆ = {s['sd_mean6_ms']:.1f} мс, "
                 f"⟨|m|⟩₆ = {s['mae_mean6_ms']:.1f} мс, σ ≤ 2σ_CSE: {cse} точек")
    return "\n".join(lines)
