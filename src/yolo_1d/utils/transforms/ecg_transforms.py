import random
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.signal import butter, resample_poly, sosfiltfilt
from torch.nn import Module

FS = 500
P_CLASS, QRS_CLASS, T_CLASS = 0, 1, 2
REPO = Path(__file__).resolve().parents[4]


# --------------------------------------------------------------------------- общие функции
def active_span(signals: torch.Tensor) -> Tuple[int, int]:
    """[start, end) — от первого до последнего ненулевого отсчёта (по всем каналам)."""
    nz = torch.nonzero(signals.abs().sum(0) > 0).flatten()
    if len(nz) == 0:
        return 0, signals.shape[-1]
    return int(nz[0]), int(nz[-1]) + 1


def boxes_to_edges(bboxes: torch.Tensor, length: int) -> Tuple[np.ndarray, np.ndarray]:
    """(center, width) в долях длины → левые и правые границы в отсчётах."""
    b = bboxes.detach().cpu().numpy().astype(float).reshape(-1, 2)
    return (b[:, 0] - b[:, 1] / 2) * length, (b[:, 0] + b[:, 1] / 2) * length


def edges_to_boxes(left: np.ndarray, right: np.ndarray, length: int, like: torch.Tensor) -> torch.Tensor:
    out = np.stack([(left + right) / 2 / length, (right - left) / length], axis=1)
    return torch.as_tensor(out, dtype=like.dtype)


def noise_scale(signals: torch.Tensor, span: Tuple[int, int], snr_db: float) -> torch.Tensor:
    """σ шума по каналам для заданного SNR относительно активной части сигнала."""
    s, e = span
    sd = signals[:, s:e].std(dim=-1, keepdim=True).clamp_min(1e-6)
    return sd * 10 ** (-snr_db / 20)


def add_in_span(signals: torch.Tensor, noise: torch.Tensor, span: Tuple[int, int]) -> torch.Tensor:
    """Добавляет шум (каналы, e − s) только в активную часть."""
    s, e = span
    out = signals.clone()
    out[:, s:e] += noise.to(signals.dtype)
    return out


class RandomAug(Module):
    """База: аугментация применяется с вероятностью p."""

    def __init__(self, p: float):
        super().__init__()
        self.p = float(p)

    def forward(self, signals, bboxes, labels):
        if self.p <= 0 or random.random() >= self.p:
            return signals, bboxes, labels
        return self.apply(signals, bboxes, labels)

    def apply(self, signals, bboxes, labels):  # pragma: no cover - абстрактный метод
        raise NotImplementedError

    def extra_repr(self) -> str:
        return f"p={self.p}"


# --------------------------------------------------------------------------- N: шумы (разметка не меняется)
class BaselineWander(RandomAug):
    """N1. Дрейф изолинии: сумма 1–3 синусоид с частотой из `freq` Гц и общей амплитудой ≤ `max_amp`·σ."""

    def __init__(self, p, max_amp: float = 0.5, freq: Sequence[float] = (0.05, 0.5), fs: int = FS):
        super().__init__(p)
        self.max_amp, self.freq, self.fs = max_amp, tuple(freq), fs

    def apply(self, signals, bboxes, labels):
        span = active_span(signals)
        n = span[1] - span[0]
        t = torch.arange(n, dtype=torch.float32) / self.fs
        sd = signals[:, span[0]:span[1]].std(dim=-1, keepdim=True)
        noise = torch.zeros(signals.shape[0], n)
        k = random.randint(1, 3)
        for _ in range(k):
            f = random.uniform(*self.freq)
            amp = random.uniform(0, self.max_amp) / k
            noise += amp * torch.sin(2 * np.pi * f * t + random.uniform(0, 2 * np.pi))
        return add_in_span(signals, noise * sd, span), bboxes, labels


class PowerlineNoise(RandomAug):
    """N2. Сетевая наводка 50 или 60 Гц (±0.5 Гц), с вероятностью 0.3 — 2-я гармоника (×0.3)."""

    def __init__(self, p, snr: Sequence[float] = (15, 35), fs: int = FS):
        super().__init__(p)
        self.snr, self.fs = tuple(snr), fs

    def apply(self, signals, bboxes, labels):
        span = active_span(signals)
        t = torch.arange(span[1] - span[0], dtype=torch.float32) / self.fs
        f = random.choice([50.0, 60.0]) + random.uniform(-0.5, 0.5)
        wave = torch.sin(2 * np.pi * f * t + random.uniform(0, 2 * np.pi))
        if random.random() < 0.3:
            wave = wave + 0.3 * torch.sin(2 * np.pi * 2 * f * t + random.uniform(0, 2 * np.pi))
        wave = wave / wave.std().clamp_min(1e-6)
        return add_in_span(signals, wave * noise_scale(signals, span, random.uniform(*self.snr)), span), bboxes, labels


class EMGBursts(RandomAug):
    """N3. Мышечный шум всплесками: гауссов шум с ФНЧ 100–150 Гц на 1–3 окнах 0.2–2 с с огибающей Ханна."""

    def __init__(self, p, snr: Sequence[float] = (10, 25), duration_s: Sequence[float] = (0.2, 2.0), fs: int = FS):
        super().__init__(p)
        self.snr, self.duration, self.fs = tuple(snr), tuple(duration_s), fs

    def apply(self, signals, bboxes, labels):
        span = active_span(signals)
        n = span[1] - span[0]
        sigma = noise_scale(signals, span, random.uniform(*self.snr))
        noise = torch.zeros(signals.shape[0], n)
        for _ in range(random.randint(1, 3)):
            m = min(n, int(random.uniform(*self.duration) * self.fs))
            if m < 8:
                continue
            start = random.randint(0, n - m)
            sos = butter(4, random.uniform(100, 150), btype="lowpass", fs=self.fs, output="sos")
            burst = sosfiltfilt(sos, np.random.randn(signals.shape[0], m), axis=-1)
            burst = burst / max(burst.std(), 1e-6) * np.hanning(m)
            noise[:, start:start + m] += torch.as_tensor(burst, dtype=torch.float32)
        return add_in_span(signals, noise * sigma, span), bboxes, labels


class WhiteNoise(RandomAug):
    """N4. Белый гауссов шум на всей активной части (замена RandomJitter: SNR вместо абсолютной σ)."""

    def __init__(self, p, snr: Sequence[float] = (20, 35)):
        super().__init__(p)
        self.snr = tuple(snr)

    def apply(self, signals, bboxes, labels):
        span = active_span(signals)
        noise = torch.randn(signals.shape[0], span[1] - span[0])
        return add_in_span(signals, noise * noise_scale(signals, span, random.uniform(*self.snr)), span), bboxes, labels


class NSTDBNoise(RandomAug):
    """N5. Реальные шумы MIT-BIH NSTDB (bw, ma, em): случайный фрагмент случайной записи и канала,
    ресемплинг 360 → 500 Гц, нормировка на заданный SNR. Для `em` (артефакт электродов, может
    имитировать комплексы) SNR не ниже `em_min_snr`."""

    _cache = {}

    def __init__(self, p, root: str = "data/raw/nstdb", snr: Sequence[float] = (6, 24), em_min_snr: float = 12.0,
                 records: Sequence[str] = ("bw", "ma", "em")):
        super().__init__(p)
        self.root, self.snr, self.em_min_snr, self.records = str(root), tuple(snr), em_min_snr, tuple(records)

    def _noise(self, record: str) -> np.ndarray:
        """Запись шума (каналы, отсчёты) при 500 Гц; загружается один раз на процесс."""
        key = (self.root, record)
        if key not in self._cache:
            import wfdb
            root = Path(self.root)
            if not root.is_absolute() and not root.exists():
                root = REPO / root  # относительный путь — от корня репозитория
            rec = wfdb.rdrecord(str(root / record))
            self._cache[key] = resample_poly(rec.p_signal.T, 25, 18, axis=-1).astype(np.float32)  # 360 → 500 Гц
        return self._cache[key]

    def apply(self, signals, bboxes, labels):
        span = active_span(signals)
        n = span[1] - span[0]
        record = random.choice(self.records)
        source = self._noise(record)
        out = []
        for _ in range(signals.shape[0]):
            ch = random.randrange(source.shape[0])
            start = random.randint(0, source.shape[1] - n)
            seg = source[ch, start:start + n]
            seg = (seg - seg.mean()) / max(seg.std(), 1e-6)
            out.append(seg)
        lo = max(self.snr[0], self.em_min_snr) if record == "em" else self.snr[0]
        snr = random.uniform(lo, max(lo, self.snr[1]))
        noise = torch.as_tensor(np.stack(out), dtype=torch.float32)
        return add_in_span(signals, noise * noise_scale(signals, span, snr), span), bboxes, labels


class BaselineStep(RandomAug):
    """N6. Резкий сдвиг изолинии: ступенька ±U(0, max_amp)·σ на случайном отрезке 10–50 % активной части."""

    def __init__(self, p, max_amp: float = 1.0, fraction: Sequence[float] = (0.1, 0.5)):
        super().__init__(p)
        self.max_amp, self.fraction = max_amp, tuple(fraction)

    def apply(self, signals, bboxes, labels):
        span = active_span(signals)
        n = span[1] - span[0]
        m = max(1, int(random.uniform(*self.fraction) * n))
        start = random.randint(0, n - m)
        sd = signals[:, span[0]:span[1]].std(dim=-1, keepdim=True)
        noise = torch.zeros(signals.shape[0], n)
        noise[:, start:start + m] = random.choice([-1, 1]) * random.uniform(0, self.max_amp)
        return add_in_span(signals, noise * sd, span), bboxes, labels


class RandomFilter(RandomAug):
    """N7. Случайная АЧХ аппарата: ФВЧ (срез 0.05–0.7 Гц) и/или ФНЧ (срез 25–100 Гц), Баттерворт 2-го
    порядка, `sosfiltfilt` — нулевая фаза, границы волн не сдвигаются. Только активная часть."""

    def __init__(self, p, highpass: Sequence[float] = (0.05, 0.7), lowpass: Sequence[float] = (25, 100), fs: int = FS):
        super().__init__(p)
        self.highpass, self.lowpass, self.fs = tuple(highpass), tuple(lowpass), fs

    def apply(self, signals, bboxes, labels):
        s, e = active_span(signals)
        x = signals[:, s:e].detach().cpu().numpy().astype(np.float64)
        if x.shape[-1] < 50:
            return signals, bboxes, labels
        kind = random.choice(["high", "low", "both"])
        if kind in ("high", "both"):
            x = sosfiltfilt(butter(2, random.uniform(*self.highpass), "highpass", fs=self.fs, output="sos"), x, axis=-1)
        if kind in ("low", "both"):
            x = sosfiltfilt(butter(2, random.uniform(*self.lowpass), "lowpass", fs=self.fs, output="sos"), x, axis=-1)
        out = signals.clone()
        out[:, s:e] = torch.as_tensor(x.copy(), dtype=signals.dtype)
        return out, bboxes, labels


# --------------------------------------------------------------------------- T: время (боксы пересчитываются)
def warp(signals: torch.Tensor, bboxes: torch.Tensor, src_knots: np.ndarray, dst_knots: np.ndarray):
    """Применяет монотонное кусочно-линейное отображение времени src → dst.

    src_knots, dst_knots — возрастающие узлы, первый 0, последний — длина исходного/нового сигнала.
    Сигнал интерполируется линейно; границы боксов отображаются той же функцией.
    """
    length = signals.shape[-1]
    new_length = max(2, int(round(dst_knots[-1])))
    dst_knots = dst_knots * (new_length / dst_knots[-1])
    # для каждого отсчёта нового сигнала — позиция в исходном (обратное отображение)
    src_pos = np.interp(np.arange(new_length) + 0.5, dst_knots, src_knots) - 0.5
    src_pos = np.clip(src_pos, 0, length - 1)
    x = signals.detach().cpu().numpy()
    out = np.stack([np.interp(src_pos, np.arange(length), ch) for ch in x])
    left, right = boxes_to_edges(bboxes, length)
    new_left, new_right = np.interp(left, src_knots, dst_knots), np.interp(right, src_knots, dst_knots)
    return (torch.as_tensor(out, dtype=signals.dtype),
            edges_to_boxes(new_left, new_right, new_length, bboxes) if len(bboxes) else bboxes)


class TimeScale(RandomAug):
    """T1. Глобальное растяжение/сжатие по времени с коэффициентом k ∈ `scale`."""

    def __init__(self, p, scale: Sequence[float] = (0.85, 1.15)):
        super().__init__(p)
        self.scale = tuple(scale)

    def apply(self, signals, bboxes, labels):
        length = signals.shape[-1]
        k = random.uniform(*self.scale)
        signals, bboxes = warp(signals, bboxes, np.array([0.0, length]), np.array([0.0, length * k]))
        return signals, bboxes, labels


class HRVWarp(RandomAug):
    """T2. Физиологичный варп ЧСС: промежутки между волнами растягиваются по-разному.

    - диастола (промежуток после T до следующей волны) — общий коэффициент k_tp ∈ `tp`, у каждого
      промежутка дополнительный разброс ±5 %;
    - PR (P → QRS) и ST (QRS → T) — k ∈ `segment`;
    - сами волны — k ∈ `wave` (почти не меняются, как длительность QRS при смене ЧСС);
    - края до первой и после последней волны не меняются.
    """

    def __init__(self, p, tp: Sequence[float] = (0.6, 1.5), segment: Sequence[float] = (0.85, 1.15),
                 wave: Sequence[float] = (0.95, 1.05)):
        super().__init__(p)
        self.tp, self.segment, self.wave = tuple(tp), tuple(segment), tuple(wave)

    def apply(self, signals, bboxes, labels):
        length = signals.shape[-1]
        if len(bboxes) < 2:
            return signals, bboxes, labels
        left, right = boxes_to_edges(bboxes, length)
        cls = labels.detach().cpu().numpy().reshape(-1).astype(int)
        order = np.argsort(left)
        src, dst = hrv_knots(left[order], right[order], cls[order], length, self.tp, self.segment, self.wave)
        index = torch.as_tensor(order)
        signals, new_boxes = warp(signals, bboxes[index], src, dst)
        return signals, new_boxes, labels[index]


def hrv_knots(left: np.ndarray, right: np.ndarray, cls: np.ndarray, length: int, tp: Sequence[float],
              segment: Sequence[float], wave: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Узлы отображения времени HRV-варпа (T2) по волнам, упорядоченным по левому краю.

    Узлы: 0, l1, r1, l2, r2, ..., L; промежуток после T — общий k_tp ∈ tp (±5 %), остальные промежутки —
    k ∈ segment, волны — k ∈ wave, края до первой и после последней волны — 1.
    """
    k_tp = random.uniform(*tp)
    src, scale = [0.0], []
    prev_end, prev_cls = 0.0, None
    for l, r, c in zip(left, right, cls):
        l, r = max(l, prev_end), max(r, max(l, prev_end))
        if prev_cls is None:
            gap_k = 1.0
        elif prev_cls == T_CLASS:
            gap_k = k_tp * random.uniform(0.95, 1.05)
        else:
            gap_k = random.uniform(*segment)
        src += [l, r]
        scale += [gap_k, random.uniform(*wave)]
        prev_end, prev_cls = r, c
    src.append(float(length))
    scale.append(1.0)
    src = np.asarray(src)
    return src, np.concatenate([[0.0], np.cumsum(np.diff(src) * np.asarray(scale))])


class WindowWarp(RandomAug):
    """T3. Window warping (Le Guennec 2016): окно 10–30 % длины растягивается/сжимается в k ∈ `scale` раз."""

    def __init__(self, p, window: Sequence[float] = (0.1, 0.3), scale: Sequence[float] = (0.7, 1.4)):
        super().__init__(p)
        self.window, self.scale = tuple(window), tuple(scale)

    def apply(self, signals, bboxes, labels):
        length = signals.shape[-1]
        s, e = active_span(signals)
        w = random.uniform(*self.window) * (e - s)
        start = random.uniform(s, max(s, e - w))
        k = random.uniform(*self.scale)
        src = np.array([0.0, start, start + w, length])
        dst = np.array([0.0, start, start + w * k, length + w * (k - 1)])
        signals, bboxes = warp(signals, bboxes, src, dst)
        return signals, bboxes, labels


# --------------------------------------------------------------------------- A: амплитуда (разметка не меняется)
class AmplitudeScale(RandomAug):
    """A1. Масштаб амплитуды всего сигнала: lognormal(0, σ²)."""

    def __init__(self, p, sigma: float = 0.2):
        super().__init__(p)
        self.sigma = sigma

    def apply(self, signals, bboxes, labels):
        return signals * float(np.random.lognormal(0, self.sigma)), bboxes, labels


class WaveScale(RandomAug):
    """A2. Масштаб отдельных волн относительно локальной изолинии (линия между краями бокса),
    плавно — с окном Ханна. P и T: lognormal(0, σ_pt²), QRS: lognormal(0, σ_qrs²); каждая волна
    масштабируется с вероятностью 0.5."""

    def __init__(self, p, sigma_pt: float = 0.3, sigma_qrs: float = 0.1):
        super().__init__(p)
        self.sigma_pt, self.sigma_qrs = sigma_pt, sigma_qrs

    def apply(self, signals, bboxes, labels):
        length = signals.shape[-1]
        left, right = boxes_to_edges(bboxes, length)
        cls = labels.detach().cpu().numpy().reshape(-1).astype(int)
        out = signals.clone()
        for l, r, c in zip(left, right, cls):
            a, b = max(0, int(round(l))), min(length, int(round(r)) + 1)
            if b - a < 4 or random.random() < 0.5:
                continue
            scale = float(np.random.lognormal(0, self.sigma_qrs if c == QRS_CLASS else self.sigma_pt))
            seg = out[:, a:b]
            base = seg[:, :1] + (seg[:, -1:] - seg[:, :1]) * torch.linspace(0, 1, b - a)
            window = torch.as_tensor(np.hanning(b - a), dtype=signals.dtype)
            out[:, a:b] = base + (seg - base) * (1 + (scale - 1) * window)
        return out, bboxes, labels


class Invert(RandomAug):
    """A3. Инверсия полярности (в поотведенческом режиме: другая проекция той же волны)."""

    def apply(self, signals, bboxes, labels):
        return -signals, bboxes, labels


# --------------------------------------------------------------------------- M: маскирование
class TimeMask(RandomAug):
    """M1. Обнуляет случайный отрезок до `max_fraction` активной части. Волны, закрытые больше чем
    на `drop_covered`, удаляются; частично закрытые с края обрезаются; закрытые изнутри (маска
    внутри волны) остаются."""

    def __init__(self, p, max_fraction: float = 0.1, drop_covered: float = 0.5):
        super().__init__(p)
        self.max_fraction, self.drop_covered = max_fraction, drop_covered

    def apply(self, signals, bboxes, labels):
        length = signals.shape[-1]
        s, e = active_span(signals)
        m = int(random.uniform(0, self.max_fraction) * (e - s))
        if m < 1:
            return signals, bboxes, labels
        a = random.randint(s, e - m)
        b = a + m
        out = signals.clone()
        out[:, a:b] = 0
        if len(bboxes) == 0:
            return out, bboxes, labels
        left, right = boxes_to_edges(bboxes, length)
        covered = (np.minimum(right, b) - np.maximum(left, a)).clip(0) / np.maximum(right - left, 1e-6)
        keep = covered <= self.drop_covered
        left = np.where((left >= a) & (left < b), b, left)      # маска закрывает левый край
        right = np.where((right > a) & (right <= b), a, right)  # маска закрывает правый край
        keep &= right > left
        return out, edges_to_boxes(left[keep], right[keep], length, bboxes), labels[torch.as_tensor(keep)]


# --------------------------------------------------------------------------- приведение к длине входа
class FitToLength(Module):
    """Приводит сигнал к длине `length`, не разрезая размеченные волны.

    Длиннее: кроп, содержащий все боксы (положение случайно в допустимых пределах); если все боксы
    не помещаются — кроп по центру боксов с обрезкой крайних (оставляются волны, видимые ≥ 50 %).
    Короче: нули с обеих сторон, случайный сдвиг (`random=True`) или нули справа (`random=False`).
    """

    def __init__(self, length: int, random_shift: bool = True):
        super().__init__()
        self.length, self.random_shift = length, random_shift

    def forward(self, signals, bboxes, labels):
        n = signals.shape[-1]
        left, right = boxes_to_edges(bboxes, n)
        if n > self.length:
            if len(bboxes):
                lo_box, hi_box = float(left.min()), float(right.max())
                lo, hi = max(0, int(np.ceil(hi_box)) - self.length), min(n - self.length, int(np.floor(lo_box)))
                if lo > hi:  # все волны не помещаются — кроп по центру размеченной части
                    lo = hi = int(np.clip((lo_box + hi_box) / 2 - self.length / 2, 0, n - self.length))
            else:
                lo, hi = 0, n - self.length
            start = random.randint(lo, hi) if self.random_shift else lo
            out = signals[:, start:start + self.length]
            left, right = left - start, right - start
            new_l, new_r = left.clip(0, self.length), right.clip(0, self.length)
            keep = (new_r - new_l) >= 0.5 * (right - left)
            return (out, edges_to_boxes(new_l[keep], new_r[keep], self.length, bboxes),
                    labels[torch.as_tensor(keep)])
        start = random.randint(0, self.length - n) if self.random_shift else 0
        out = torch.zeros(signals.shape[0], self.length, dtype=signals.dtype)
        out[:, start:start + n] = signals
        if len(bboxes):
            bboxes = edges_to_boxes(left + start, right + start, self.length, bboxes)
        return out, bboxes, labels
