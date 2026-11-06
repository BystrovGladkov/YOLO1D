"""Предобработка LUDB для детекции: Z-score, границы размеченной зоны, перевод разметки в боксы.

Принятые решения (TASKS.md, «Принятые решения»; docs/research/R1_ludb.md §4.1–4.2):
- детекция **поотведенческая** (режим S): пример = одно отведение одной записи, вход 1 × 5000;
- каждый пример — полная запись, без кропов; паддинг до 5120 делают трансформации;
- Z-score по всему отведению; сигнал сохраняется **без маски**, а рядом — границы размеченной
  зоны (первый onset, последний offset). Значения вне `[first_onset − δ, last_offset + δ]` обнуляются
  уже при загрузке (`yolo_1d.utils.transforms.EdgeMask`, D5): на обучении δ случайный, чтобы граница
  маски не служила подсказкой (E1 §3), на валидации — фиксированный; лосс не меняется;
- боксы в формате репозитория: `class center width`, нормированные на длину сигнала (5000),
  классы P = 0, QRS = 1, T = 2.
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import butter, sosfiltfilt

from yolo_1d.utils.datasets.ludb import FS, Wave

# запас вокруг размеченной зоны, отсчётов (20 мс при 500 Гц); используется в проверках и тестах
DEFAULT_ZONE_MARGIN = 10


@dataclass(frozen=True)
class LeadSample:
    """Готовый пример одного отведения."""
    signal: np.ndarray            # (1, length) float32: Z-score, без маски
    labels: np.ndarray            # (n, 3) float64: class, center, width (доли длины)
    annotated: Tuple[int, int]    # (первый onset, последний offset) разметки, в отсчётах (включительно)


def bandpass(signal: np.ndarray, low: float, high: float, fs: int = FS, order: int = 2) -> np.ndarray:
    """Полосовой фильтр Баттерворта с нулевой фазой (`sosfiltfilt`): границы волн не сдвигаются."""
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, signal, axis=-1).astype(signal.dtype)


def zscore(signal: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Z-score вдоль последней оси; для почти постоянного сигнала — только центрирование."""
    mean = signal.mean(axis=-1, keepdims=True)
    std = signal.std(axis=-1, keepdims=True)
    return ((signal - mean) / np.maximum(std, eps)).astype(np.float32)


def annotated_zone(waves: Sequence[Wave], length: int, margin: int = DEFAULT_ZONE_MARGIN) -> Tuple[int, int]:
    """Размеченная зона `[first_onset − margin, last_offset + margin]` в виде полуинтервала [start, end)."""
    if not waves:
        raise ValueError("нет размеченных волн — зону определить нельзя")
    start = max(0, min(w.onset for w in waves) - margin)
    end = min(length, max(w.offset for w in waves) + margin + 1)
    return start, end


def mask_outside(signal: np.ndarray, zone: Tuple[int, int], fill: str = "zero") -> np.ndarray:
    """Заменяет сигнал вне зоны [start, end) (последняя ось) постоянным значением.

    fill
        "zero" — нули (принятое решение, R1 §4.2);
        "median" — медиана сигнала внутри зоны, то есть оценка изолинии. После Z-score изолиния
        лежит ниже 0 (среднее тянут вверх пики QRS), и заполнение нулями даёт на границе зоны
        ступеньку, которую сеть может использовать как подсказку (R4 §5.4); медиана её убирает.
    """
    start, end = zone
    if fill == "zero":
        value = np.zeros(signal.shape[:-1] + (1,), dtype=signal.dtype)
    elif fill == "median":
        value = np.median(signal[..., start:end], axis=-1, keepdims=True).astype(signal.dtype)
    else:
        raise ValueError(f"fill должен быть 'zero' или 'median', получено {fill!r}")
    masked = np.broadcast_to(value, signal.shape).copy()
    masked[..., start:end] = signal[..., start:end]
    return masked


def waves_to_labels(waves: Sequence[Wave], length: int) -> np.ndarray:
    """Волны → массив (n, 3): `class, center, width`, нормированные на длину сигнала.

    Левая граница бокса = onset, правая = offset (R2 §3.1): onset/offset детекции читаются
    напрямую как left/right без поправок.
    """
    if not waves:
        return np.zeros((0, 3), dtype=np.float64)
    rows = [(w.cls_id, (w.onset + w.offset) / 2 / length, (w.offset - w.onset) / length) for w in waves]
    return np.asarray(rows, dtype=np.float64)


def labels_to_intervals(labels: np.ndarray, length: int) -> np.ndarray:
    """Обратное к `waves_to_labels`: (n, 3) → (n, 3) `class, onset, offset` в отсчётах."""
    cls, center, width = labels[:, 0], labels[:, 1] * length, labels[:, 2] * length
    return np.stack([cls, center - width / 2, center + width / 2], axis=1)


def prepare_lead(signal: np.ndarray, waves: Sequence[Wave],
                 band: Optional[Tuple[float, float]] = None) -> LeadSample:
    """Полная предобработка одного отведения.

    Parameters
    ----------
    signal
        Сигнал отведения (length,) в мВ.
    waves
        Волны отведения (yolo_1d.utils.datasets.ludb.LudbRecord.waves[lead]).
    band
        (low, high) Гц для полосового фильтра до Z-score или None — без фильтрации.
    """
    if not waves:
        raise ValueError("нет размеченных волн — зону определить нельзя")
    x = np.asarray(signal, dtype=np.float32)[None, :]
    if band is not None:
        x = bandpass(x, *band)
    x = zscore(x)
    annotated = (min(w.onset for w in waves), max(w.offset for w in waves))
    return LeadSample(x, waves_to_labels(waves, x.shape[-1]), annotated)


def unusable_reason(dropped_samples: List[int], zone: Tuple[int, int]) -> Optional[str]:
    """Причина исключить отведение: отброшенная (дефектная) волна внутри размеченной зоны.

    Такая волна есть в сигнале, но не имеет метки — сеть училась бы считать её фоном (D2).
    Отброшенные волны вне зоны безвредны: они обнуляются маскированием. Проверять с зоной,
    расширенной на максимальный запас маски (`annotated_zone(..., margin=max δ)`).
    """
    start, end = zone
    inside = [s for s in dropped_samples if start <= s < end]
    if inside:
        return f"дефектная разметка внутри зоны: {len(inside)} волн(ы) без метки"
    return None
