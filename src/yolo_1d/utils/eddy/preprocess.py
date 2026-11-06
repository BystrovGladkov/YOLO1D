"""Предобработка вихретоковой дефектограммы (15 каналов) для сети YOLO-1d.

Шаги (TASKS.md, таблица конвейера; функции используются трансформами `transforms/eddy_preprocess.py`):

* шаг 1, `reference_stats` — опорные статистики по исходному сигналу:
  1. признак битого канала — первый проход `Get_Signal_Thresholds` по **сырым** данным
     (`detect_bad_channels`; пороги этого прохода не используются);
  2. выровненная копия: ICA **только по исправным каналам** (`ica_good`; битые в FastICA не подаются —
     иначе алгоритм не сходится) + Фурье-выравнивание (`fourier_align`);
  3. второй проход `Get_Signal_Thresholds` по выровненной копии — только пороговые уровни:
     σ_опорн = min(|thplus|, |thmins|) / 3 по `CorrectedThreshold1` (Кузьмин и др., МАИС 2021);
     σ битого канала — σ его канала-замены;
* шаг 2 — стандартизация «грязного» сигнала (x − mean) / σ_опорн;
* шаг 7, `ica_suppress` — 15 → 30 каналов: `[0:15]` вход, `[15:30]` — ICA по исправным каналам (маска шага 1);
* шаг 8 — обратно в единицы АЦП: × σ_опорн + mean;
* шаг 9 — `fourier_align` «чистой» половины;
* шаг 11, `threshold_normalize` — замена битых каналов по карте шага 1 в обеих половинах, пороги по
  «чистой» половине (`CorrectedThreshold1`, признак битости этого прохода игнорируется), нормировка
  обеих половин на σ.

`preprocess(raw)` — те же шаги без аугментаций (= валидационный конвейер, побитно), эталон для тестов.
Все шаги — во float32, как в конвейере (гистограмма порогов округляет отсчёты, поэтому ошибки float32
порогов не сдвигают).
`prepare_signal` — исходный алгоритм `EC_preprocess/preprare.py` (ICA по всем каналам, битость по
выровненному сигналу); оставлен для сравнения, в конвейере не используется.

ICA — `ICA_filtration(median=True, EMA=True, method='hyvarinen')` из `transforms/square_noises.py`;
Фурье-выравнивание — вычитание гармоник ниже `convert_freq(0.03, L)` (Быстров, Гладков, Кузьмин, МАИС 2024).
"""
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch
from scipy.fft import irfft, rfft

from yolo_1d.utils.transforms import square_noises
from .kuzmin_levels import Get_Signal_Thresholds, New_Channel_Substitution, find_closest
from .signal import Signal

# число каналов дефектограммы (алгоритм порогов рассчитан ровно на 15)
N_CHANNELS = 15
# число независимых компонент в ICA_filtration; при меньшем числе исправных каналов ICA пропускается
ICA_COMPONENTS = 5
# частота среза Фурье-выравнивания, рад/отсчёт
FOURIER_FREQ = 0.03
# версия алгоритма предобработки: входит в ключ дискового кэша, менять при изменении логики
PREPROCESS_VERSION = "eddy-2026-10-17"


def fftine_fast(data, to=None):
    channels, length = data.shape
    to = 0 if to is None else to

    # Используем векторные операции вместо цикла
    ff = rfft(data, axis=1)  # Прямое преобразование для всех каналов сразу
    ff[:, to:] = 0  # Обрезание фурье коэффициентов для всех каналов
    rez = np.round(irfft(ff, n=length, axis=1)).astype(np.int32)  # Обратное преобразование
    return rez


def convert_freq(freq: float, x: int) -> int:
    return np.ceil(freq / 2 / np.pi * x).astype("int")


def prepare_signal(data: np.ndarray) -> Signal:
    """Исходный алгоритм `EC_preprocess/preprare.py` (для сравнения; в конвейере не используется).

    data: дефектограмма (15, L). Возвращает `Signal` с полями `data` (очищенный сигнал), `sigma` и
    `substitute` (индекс канала-замены для каждого канала; −1 — заменить нечем).
    """
    data = square_noises.ICA_filtration(data, median=True, EMA=True, method='hyvarinen')
    # Фурье выравнивание
    data = data - fftine_fast(data, convert_freq(0.03, data.shape[1]))
    # вычисление пороговых уровней
    _Threshold1, _Threshold2, _CorrectedThreshold1, _CorrectedThreshold2 = [], [], [], []
    Get_Signal_Thresholds(
            data.T,
            _Threshold1,
            _Threshold2,
            _CorrectedThreshold1,
            _CorrectedThreshold2,
        )
    # карта замен — до New_Channel_Substitution (она перезаписывает badchl битых каналов)
    substitute = find_closest(_Threshold1)
    # Замена битых каналов
    New_Channel_Substitution(data.T, _Threshold1, _Threshold2, _CorrectedThreshold1, _CorrectedThreshold2)
    # Сохранение нужного порогового уровня
    signal = Signal(data)
    signal.sigma = np.array([min(abs(item1.thplus), abs(item1.thmins)) for item1 in _CorrectedThreshold1]) / 3
    signal.substitute = substitute
    return signal


# ------------------------------------------------------------------ элементарные шаги

def _thresholds(data: np.ndarray):
    """Четыре набора порогов `Get_Signal_Thresholds` для (15, L)."""
    t1, t2, c1, c2 = [], [], [], []
    Get_Signal_Thresholds(np.asarray(data).T, t1, t2, c1, c2)
    return t1, t2, c1, c2


def detect_bad_channels(raw: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """Шаг 1.1: признак битого канала по сырым данным → (маска bad (15,) bool, карта замен `find_closest`).

    Пороги этого прохода не используются. Карта: substitute[i] == i — канал исправен, иначе индекс
    ближайшего исправного канала (−1 — заменить нечем).
    """
    t1, _, _, _ = _thresholds(raw)
    bad = np.array([t.badchl != 0 for t in t1])
    return bad, find_closest(t1)


def substitute_channels(data: np.ndarray, substitute: List[int]) -> np.ndarray:
    """Копия `data` (15, L), где битые каналы заменены каналами из карты `substitute` (−1 — без замены)."""
    out = data.copy()
    for line, src in enumerate(substitute):
        if src != line and src != -1:
            out[line] = data[src]
    return out


def ica_good(data: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """ICA-подавление только по исправным каналам; битые каналы возвращаются без изменений.

    Если исправных каналов меньше `ICA_COMPONENTS`, ICA пропускается (предупреждение) — возвращается копия.
    """
    data = np.asarray(data, dtype=np.float32)
    good = ~np.asarray(bad, dtype=bool)
    out = data.copy()
    if good.sum() < ICA_COMPONENTS:
        warnings.warn(f"исправных каналов {int(good.sum())} < {ICA_COMPONENTS}: ICA-подавление пропущено")
        return out
    try:
        filtered = square_noises.ICA_filtration(data[good], median=True, EMA=True, method='hyvarinen')
    except (ValueError, np.linalg.LinAlgError) as e:     # E3: на редких обучающих примерах W FastICA → NaN
        filtered, reason = None, f"{type(e).__name__}: {e}"
    else:
        reason = "результат содержит NaN/Inf"
    if filtered is None or not np.isfinite(filtered).all():
        _save_ica_failure(data, bad)
        warnings.warn(f"ICA-подавление не удалось ({reason}): пример оставлен без подавления")
        return out
    out[good] = filtered
    return out


def _save_ica_failure(data: np.ndarray, bad: np.ndarray) -> None:
    """Сохраняет вход неудавшейся ICA в каталог `EDDY_ICA_FAILURE_DIR` (если задан) — для разбора."""
    root = os.environ.get("EDDY_ICA_FAILURE_DIR")
    if not root:
        return
    os.makedirs(root, exist_ok=True)
    np.savez(os.path.join(root, f"ica_failure_{os.getpid()}_{time.time_ns()}.npz"), data=data, bad=bad)


def fourier_align(data: np.ndarray) -> np.ndarray:
    """Вычитание низкочастотных гармоник (ниже `convert_freq(FOURIER_FREQ, L)`) по каждому каналу."""
    data = np.asarray(data, dtype=np.float32)
    return data - fftine_fast(data, convert_freq(FOURIER_FREQ, data.shape[1])).astype(np.float32)


def corrected_sigma(aligned: np.ndarray, substitute: List[int]) -> Tuple[np.ndarray, List[int]]:
    """σ = min(|thplus|, |thmins|) / 3 по `CorrectedThreshold1`; σ битого канала — σ канала-замены.

    Признак битости этого прохода игнорируется. σ ≤ 0 заменяется на 1; возвращает (σ (15,), такие каналы).
    """
    _, _, c1, _ = _thresholds(aligned)
    sigma = np.array([min(abs(t.thplus), abs(t.thmins)) for t in c1], dtype=np.float32) / 3
    for line, src in enumerate(substitute):
        if src != line and src != -1:
            sigma[line] = sigma[src]
    fallback = [int(i) for i in np.flatnonzero(sigma <= 0)]
    sigma[fallback] = 1.0
    return sigma, fallback


# ------------------------------------------------------------------ шаги конвейера

@dataclass
class ReferenceStats:
    """Опорные статистики шага 1 (детерминированы по исходному сигналу, кэшируются).

    mean: (15,) среднее «грязного» канала; sigma: (15,) σ_опорн; bad: (15,) маска битых каналов;
    substitute: карта замен; sigma_fallback: каналы с σ_опорн = 0 → 1.
    """
    mean: np.ndarray
    sigma: np.ndarray
    bad: np.ndarray
    substitute: List[int]
    sigma_fallback: List[int] = field(default_factory=list)


def reference_stats(raw: np.ndarray) -> ReferenceStats:
    """Шаг 1: битость по сырым данным → ICA по исправным + Фурье → σ_опорн по скорректированным порогам."""
    raw = np.asarray(raw, dtype=np.float32)
    bad, substitute = detect_bad_channels(raw)
    aligned = fourier_align(ica_good(raw, bad))
    sigma, fallback = corrected_sigma(aligned, substitute)
    return ReferenceStats(mean=raw.mean(axis=1), sigma=sigma, bad=bad, substitute=substitute,
                          sigma_fallback=fallback)


def ica_suppress(z: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """Шаг 7: (15, L) → (30, L): `[0:15]` — вход, `[15:30]` — ICA по исправным каналам (битые — копия входа)."""
    z = np.asarray(z, dtype=np.float32)
    return np.concatenate([z, ica_good(z, bad)], axis=0)


def threshold_normalize(data30: np.ndarray, substitute: List[int]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Шаг 11: замена битых каналов по карте шага 1 в обеих половинах, σ по «чистой» половине, нормировка.

    Возвращает (данные (30, L), σ (15,), каналы с σ = 0 → 1).
    """
    data30 = np.asarray(data30, dtype=np.float32)
    dirty = substitute_channels(data30[:N_CHANNELS], substitute)
    clean = substitute_channels(data30[N_CHANNELS:], substitute)
    sigma, fallback = corrected_sigma(clean, substitute)
    return np.concatenate([dirty, clean], axis=0) / np.tile(sigma, 2)[:, None], sigma, fallback


@dataclass
class PreprocessResult:
    """Результат предобработки одного фрагмента (валидационный конвейер без аугментаций).

    raw:        (15, L) float32 — без подавления помех, битые каналы заменены, делён на σ;
    filtered:   (15, L) float32 — с подавлением помех (ICA + Фурье), битые каналы заменены, делён на σ;
    sigma:      (15,) — финальный нормирующий уровень (шаг 11);
    reference:  опорные статистики шага 1;
    substitute, bad_channels — карта замен и битые каналы (шаг 1);
    sigma_fallback — каналы, у которых финальный σ = 0 заменён на 1.
    """
    raw: np.ndarray
    filtered: np.ndarray
    sigma: np.ndarray
    reference: ReferenceStats
    substitute: List[int]
    bad_channels: List[int] = field(default_factory=list)
    sigma_fallback: List[int] = field(default_factory=list)

    def to_tensor(self) -> torch.Tensor:
        """Тензор входа сети (30, L) float16: `[0:15]` — без подавления помех, `[15:30]` — с подавлением."""
        return torch.from_numpy(np.concatenate([self.raw, self.filtered], axis=0).astype(np.float16))


def preprocess(raw: np.ndarray) -> PreprocessResult:
    """Полная предобработка без аугментаций: шаги 1, 2, 7, 8, 9, 11 (= валидационный конвейер).

    raw: (15, L) — исходные отсчёты (int16 из ECRv1 или любой числовой тип).
    """
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.shape[0] != N_CHANNELS:
        raise ValueError(f"expected array of shape ({N_CHANNELS}, L), got {raw.shape}")
    raw = raw.astype(np.float32)

    ref = reference_stats(raw)                                         # шаг 1
    z = (raw - ref.mean[:, None]) / ref.sigma[:, None]                 # шаг 2
    x30 = ica_suppress(z, ref.bad)                                     # шаг 7
    x30 = x30 * np.tile(ref.sigma, 2)[:, None] + np.tile(ref.mean, 2)[:, None]   # шаг 8
    x30[N_CHANNELS:] = fourier_align(x30[N_CHANNELS:])                 # шаг 9
    out, sigma, fallback = threshold_normalize(x30, ref.substitute)    # шаг 11
    return PreprocessResult(
        raw=out[:N_CHANNELS], filtered=out[N_CHANNELS:], sigma=sigma, reference=ref,
        substitute=list(ref.substitute), bad_channels=[int(i) for i in np.flatnonzero(ref.bad)],
        sigma_fallback=fallback,
    )
