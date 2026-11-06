import random
from typing import Sequence, Tuple

import numpy as np
import torch
from torch.nn import Module

from .ecg_transforms import (FS, QRS_CLASS, NSTDBNoise, RandomAug, active_span, add_in_span, boxes_to_edges,
                             hrv_knots, noise_scale, warp)

N_LEADS = 12


# --------------------------------------------------------------------------- общие функции
def wave_clusters(bboxes: torch.Tensor, labels: torch.Tensor, length: int) -> Tuple[np.ndarray, ...]:
    """Волны одного класса, перекрывающиеся между отведениями, → кластер (одна волна сердца).

    Returns: left, right, cls кластеров (по возрастанию left) и cluster_of — номер кластера каждой волны.
    """
    left, right = boxes_to_edges(bboxes, length)
    cls = labels[:, 0].detach().cpu().numpy().astype(int)
    order = np.argsort(left, kind="stable")
    c_left, c_right, c_cls, cluster_of = [], [], [], np.zeros(len(left), int)
    for i in order:
        if c_cls and c_cls[-1] == cls[i] and left[i] <= c_right[-1]:
            c_right[-1] = max(c_right[-1], right[i])
        else:
            c_left.append(left[i])
            c_right.append(right[i])
            c_cls.append(cls[i])
        cluster_of[i] = len(c_cls) - 1
    return np.asarray(c_left), np.asarray(c_right), np.asarray(c_cls, int), cluster_of


def lead_coefficients(n: int) -> torch.Tensor:
    """Коэффициенты общего источника по отведениям: U(0.5, 1.5) со случайным знаком, (n, 1)."""
    return torch.tensor([[random.uniform(0.5, 1.5) * random.choice((-1.0, 1.0))] for _ in range(n)])


class KeepZeros(Module):
    """Обёртка: после аугментации отсчёты, которые были ровно нулевыми (маска краёв отведения, паддинг),
    снова нулевые. Нужна, потому что маски краёв у отведений разные, а активная часть — общая."""

    def __init__(self, op: Module):
        super().__init__()
        self.op = op

    @property
    def p(self) -> float:
        return getattr(self.op, "p", 1.0)

    def forward(self, signals, bboxes, labels):
        zero = signals == 0
        out, bboxes, labels = self.op(signals, bboxes, labels)
        return out.masked_fill(zero, 0.0), bboxes, labels


# --------------------------------------------------------------------------- N: шумы с общим источником
class CoupledNoise(RandomAug):
    """N1/N2/N5 в MC: один источник шума (как в A1) во всех отведениях с коэффициентом U(0.5, 1.5)·(±1).

    kind
        "wander" — сумма 1–3 синусоид 0.05–0.5 Гц, амплитуда ≤ max_amp·σ отведения (N1);
        "powerline" — 50/60 Гц (± 0.5) с 2-й гармоникой (p 0.3), SNR из snr (N2);
        "nstdb" — фрагмент записи MIT-BIH NSTDB (bw/ma/em), SNR из snr, для em ≥ 12 дБ (N5).
    """

    def __init__(self, p, kind: str, snr: Sequence[float] = (15, 35), max_amp: float = 0.5,
                 freq: Sequence[float] = (0.05, 0.5), nstdb_root: str = "data/raw/nstdb", fs: int = FS):
        super().__init__(p)
        if kind not in ("wander", "powerline", "nstdb"):
            raise ValueError(f"неизвестный kind {kind!r}")
        self.kind, self.snr, self.max_amp, self.freq, self.fs = kind, tuple(snr), max_amp, tuple(freq), fs
        self._nstdb = NSTDBNoise(1.0, root=nstdb_root, snr=snr) if kind == "nstdb" else None

    def _source(self, n: int) -> Tuple[torch.Tensor, float]:
        """Источник (n,) с единичной SD и SNR (для wander — амплитуда в долях σ, SNR = None)."""
        t = torch.arange(n, dtype=torch.float32) / self.fs
        if self.kind == "wander":
            wave = torch.zeros(n)
            k = random.randint(1, 3)
            for _ in range(k):
                wave += random.uniform(0, self.max_amp) / k * torch.sin(
                    2 * np.pi * random.uniform(*self.freq) * t + random.uniform(0, 2 * np.pi))
            return wave, None
        if self.kind == "powerline":
            f = random.choice([50.0, 60.0]) + random.uniform(-0.5, 0.5)
            wave = torch.sin(2 * np.pi * f * t + random.uniform(0, 2 * np.pi))
            if random.random() < 0.3:
                wave = wave + 0.3 * torch.sin(2 * np.pi * 2 * f * t + random.uniform(0, 2 * np.pi))
            return wave / wave.std().clamp_min(1e-6), random.uniform(*self.snr)
        record = random.choice(self._nstdb.records)
        source = self._nstdb._noise(record)
        ch = random.randrange(source.shape[0])
        start = random.randint(0, source.shape[1] - n)
        seg = source[ch, start:start + n]
        seg = (seg - seg.mean()) / max(seg.std(), 1e-6)
        lo = max(self.snr[0], self._nstdb.em_min_snr) if record == "em" else self.snr[0]
        return torch.as_tensor(seg, dtype=torch.float32), random.uniform(lo, max(lo, self.snr[1]))

    def apply(self, signals, bboxes, labels):
        span = active_span(signals)
        wave, snr = self._source(span[1] - span[0])
        coef = lead_coefficients(signals.shape[0])
        if snr is None:  # амплитуда в долях σ отведения
            scale = signals[:, span[0]:span[1]].std(dim=-1, keepdim=True)
        else:
            scale = noise_scale(signals, span, snr)
        return add_in_span(signals, wave[None, :] * coef * scale, span), bboxes, labels

    def extra_repr(self) -> str:
        return f"p={self.p}, kind={self.kind}"


class LeadBaselineStep(RandomAug):
    """N6 в MC: ступенька изолинии ±U(0, max_amp)·σ на отрезке 10–50 % активной части в 1–3 отведениях
    (смещение контакта одного электрода)."""

    def __init__(self, p, max_amp: float = 1.0, fraction: Sequence[float] = (0.1, 0.5), max_leads: int = 3):
        super().__init__(p)
        self.max_amp, self.fraction, self.max_leads = max_amp, tuple(fraction), max_leads

    def apply(self, signals, bboxes, labels):
        span = active_span(signals)
        n = span[1] - span[0]
        m = max(1, int(random.uniform(*self.fraction) * n))
        start = random.randint(0, n - m)
        leads = random.sample(range(signals.shape[0]), random.randint(1, min(self.max_leads, signals.shape[0])))
        sd = signals[:, span[0]:span[1]].std(dim=-1, keepdim=True)
        noise = torch.zeros(signals.shape[0], n)
        for k in leads:
            noise[k, start:start + m] = random.choice([-1, 1]) * random.uniform(0, self.max_amp)
        return add_in_span(signals, noise * sd, span), bboxes, labels


# --------------------------------------------------------------------------- T2, A1, A2 в MC
class MultiLeadHRVWarp(RandomAug):
    """T2 в MC: одно отображение времени для всех отведений. Узлы — по кластерам волн (объединение границ
    волны по отведениям, `wave_clusters`), так что растягиваются паузы, общие для всех отведений."""

    def __init__(self, p, tp: Sequence[float] = (0.6, 1.5), segment: Sequence[float] = (0.85, 1.15),
                 wave: Sequence[float] = (0.95, 1.05)):
        super().__init__(p)
        self.tp, self.segment, self.wave = tuple(tp), tuple(segment), tuple(wave)

    def apply(self, signals, bboxes, labels):
        length = signals.shape[-1]
        if len(bboxes) < 2:
            return signals, bboxes, labels
        left, right, cls, _ = wave_clusters(bboxes, labels, length)
        src, dst = hrv_knots(left, right, cls, length, self.tp, self.segment, self.wave)
        signals, bboxes = warp(signals, bboxes, src, dst)
        return signals, bboxes, labels


class MultiLeadAmplitudeScale(RandomAug):
    """A1 в MC: общий множитель записи lognormal(0, σ²) × поканальный lognormal(0, σ_lead²)."""

    def __init__(self, p, sigma: float = 0.2, sigma_lead: float = 0.1):
        super().__init__(p)
        self.sigma, self.sigma_lead = sigma, sigma_lead

    def apply(self, signals, bboxes, labels):
        common = float(np.random.lognormal(0, self.sigma))
        per_lead = torch.as_tensor(np.random.lognormal(0, self.sigma_lead, (signals.shape[0], 1)), dtype=signals.dtype)
        return signals * common * per_lead, bboxes, labels


class MultiLeadWaveScale(RandomAug):
    """A2 в MC: каждая волна сердца (кластер волн по отведениям) с вероятностью 0.5 масштабируется общим
    множителем lognormal (P, T: σ_pt; QRS: σ_qrs) × поканальный lognormal(0, σ_lead²); масштаб применяется
    к волне в её собственном отведении (относительно локальной изолинии, окно Ханна — как в A1)."""

    def __init__(self, p, sigma_pt: float = 0.3, sigma_qrs: float = 0.1, sigma_lead: float = 0.1):
        super().__init__(p)
        self.sigma_pt, self.sigma_qrs, self.sigma_lead = sigma_pt, sigma_qrs, sigma_lead

    def apply(self, signals, bboxes, labels):
        length = signals.shape[-1]
        if len(bboxes) == 0:
            return signals, bboxes, labels
        left, right = boxes_to_edges(bboxes, length)
        _, _, c_cls, cluster_of = wave_clusters(bboxes, labels, length)
        factor = [None if random.random() < 0.5 else
                  float(np.random.lognormal(0, self.sigma_qrs if c == QRS_CLASS else self.sigma_pt)) for c in c_cls]
        leads = labels[:, 1].detach().cpu().numpy().astype(int)
        out = signals.clone()
        for l, r, k, j in zip(left, right, leads, cluster_of):
            a, b = max(0, int(round(l))), min(length, int(round(r)) + 1)
            if b - a < 4 or factor[j] is None:
                continue
            scale = factor[j] * float(np.random.lognormal(0, self.sigma_lead))
            seg = out[k, a:b]
            base = seg[0] + (seg[-1] - seg[0]) * torch.linspace(0, 1, b - a)
            window = torch.as_tensor(np.hanning(b - a), dtype=signals.dtype)
            out[k, a:b] = base + (seg - base) * (1 + (scale - 1) * window)
        return out, bboxes, labels


# --------------------------------------------------------------------------- L: отведения (уровень датасета)
def _relabel(bboxes, labels, valid, new_from_old: np.ndarray, keep_leads=None):
    """Перенумерация отведений волн: волна отведения i переходит в отведение new_from_old[i];
    keep_leads (bool по новым отведениям) — оставить разметку только этих отведений."""
    if len(labels):
        new_lead = torch.as_tensor(new_from_old, dtype=labels.dtype)[labels[:, 1].long()]
        labels = torch.stack([labels[:, 0], new_lead], dim=1)
        if keep_leads is not None:
            keep = torch.as_tensor(keep_leads)[new_lead.long()]
            labels, bboxes = labels[keep], bboxes[keep]
    return bboxes, labels


class LeadDropout(Module):
    """L1. Обнуление отведений (RLM, Oh 2022): с вероятностью p в примере обнуляется каждое отведение
    с вероятностью p_lead, не больше max_leads (хотя бы одно).

    labels_mode
        "ignore" — разметка обнулённых отведений удаляется и они исключаются из лосса (valid = False);
        "keep" — разметка остаётся (сеть учится находить волны отведения по соседним).
    """

    def __init__(self, p: float, p_lead: float = 0.15, max_leads: int = 4, labels_mode: str = "ignore"):
        super().__init__()
        if labels_mode not in ("ignore", "keep"):
            raise ValueError(f"labels_mode должен быть 'ignore' или 'keep', получено {labels_mode!r}")
        self.p, self.p_lead, self.max_leads, self.labels_mode = float(p), p_lead, max_leads, labels_mode

    def forward(self, signals, bboxes, labels, valid):
        if self.p <= 0 or random.random() >= self.p:
            return signals, bboxes, labels, valid
        n = signals.shape[0]
        leads = [k for k in range(n) if random.random() < self.p_lead] or [random.randrange(n)]
        leads = random.sample(leads, min(len(leads), self.max_leads))
        signals = signals.clone()
        signals[leads] = 0
        if self.labels_mode == "ignore":
            valid = valid.clone()
            valid[leads] = False
            bboxes, labels = _relabel(bboxes, labels, valid, np.arange(n), keep_leads=valid.numpy())
        return signals, bboxes, labels, valid

    def extra_repr(self) -> str:
        return f"p={self.p}, p_lead={self.p_lead}, max_leads={self.max_leads}, labels={self.labels_mode}"


class LeadShuffle(Module):
    """L2. Перестановка 2–4 случайных отведений вместе с разметкой и маской valid."""

    def __init__(self, p: float, n_leads: Sequence[int] = (2, 4)):
        super().__init__()
        self.p, self.n_leads = float(p), tuple(n_leads)

    def forward(self, signals, bboxes, labels, valid):
        if self.p <= 0 or random.random() >= self.p:
            return signals, bboxes, labels, valid
        n = signals.shape[0]
        chosen = random.sample(range(n), random.randint(*self.n_leads))
        target = chosen[:]
        while target == chosen:
            random.shuffle(target)
        src_of_new = np.arange(n)  # новый канал j берёт старый канал src_of_new[j]
        src_of_new[target] = chosen
        new_from_old = np.argsort(src_of_new)
        bboxes, labels = _relabel(bboxes, labels, valid, new_from_old)
        return signals[torch.as_tensor(src_of_new)], bboxes, labels, valid[torch.as_tensor(src_of_new)]

    def extra_repr(self) -> str:
        return f"p={self.p}, n_leads={self.n_leads}"


# перестановки электродов конечностей (I, II, III, aVR, aVL, aVF; грудные не меняются):
# новый канал j = sign[j] · старый канал src[j] (Эйнтховен/Гольдбергер, R5 §3.3)
ELECTRODE_SWAPS = {
    "LA-RA": ([0, 2, 1, 4, 3, 5], [-1, 1, 1, 1, 1, 1]),
    "LA-LL": ([1, 0, 2, 3, 5, 4], [1, 1, -1, 1, 1, 1]),
    "RA-LL": ([2, 1, 0, 5, 4, 3], [-1, -1, -1, 1, 1, 1]),
}


class ElectrodeSwap(Module):
    """L3. Перестановка электродов конечностей (LA↔RA, LA↔LL, RA↔LL) — реальная ошибка наложения.
    Точные формулы перестановки и смены знака отведений; коммутируют с Z-score по отведению. Разметка и
    valid переезжают вместе с отведениями."""

    def __init__(self, p: float, swaps: Sequence[str] = tuple(ELECTRODE_SWAPS)):
        super().__init__()
        self.p, self.swaps = float(p), tuple(swaps)

    def forward(self, signals, bboxes, labels, valid):
        if self.p <= 0 or random.random() >= self.p:
            return signals, bboxes, labels, valid
        return self.swap(random.choice(self.swaps), signals, bboxes, labels, valid)

    @staticmethod
    def swap(name, signals, bboxes, labels, valid):
        src6, sign6 = ELECTRODE_SWAPS[name]
        n = signals.shape[0]
        src_of_new = np.array(src6 + list(range(6, n)))
        sign = torch.tensor(sign6 + [1] * (n - 6), dtype=signals.dtype)[:, None]
        new_from_old = np.argsort(src_of_new)
        bboxes, labels = _relabel(bboxes, labels, valid, new_from_old)
        index = torch.as_tensor(src_of_new)
        return signals[index] * sign, bboxes, labels, valid[index]

    def extra_repr(self) -> str:
        return f"p={self.p}, swaps={self.swaps}"


class LeadCompose(Module):
    """Последовательность аугментаций группы L: (signals, bboxes, labels, valid) → то же."""

    def __init__(self, ops: Sequence[Module]):
        super().__init__()
        self.ops = torch.nn.ModuleList(ops)

    def forward(self, signals, bboxes, labels, valid):
        for op in self.ops:
            signals, bboxes, labels, valid = op(signals, bboxes, labels, valid)
        return signals, bboxes, labels, valid
