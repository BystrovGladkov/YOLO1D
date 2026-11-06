import hashlib
import os
import tempfile
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.nn import Module

from yolo_1d.utils.eddy.preprocess import (N_CHANNELS, PREPROCESS_VERSION, ReferenceStats, fourier_align,
                                           ica_suppress, reference_stats, threshold_normalize)


class EddyContext:
    """Состояние одного примера между шагами предобработки (см. docstring модуля)."""

    def __init__(self):
        self.ref: Optional[ReferenceStats] = None

    def require(self, step: str) -> ReferenceStats:
        if self.ref is None:
            raise RuntimeError(f"{step}: нет опорных статистик — в конвейере перед ним должен быть ReferenceStandardize")
        return self.ref


class ReferenceCache:
    """Дисковый кэш опорных статистик шага 1: один `.npz` на сигнал в `<root>/reference/`.

    Ключ — BLAKE2b от версии алгоритма, формы и байтов сигнала (float32). Запись атомарная (временный
    файл + `os.replace`), поэтому несколько воркеров могут писать одновременно.
    """

    def __init__(self, root: str, max_length: Optional[int] = 50000):
        self.dir = os.path.join(str(root), "reference")
        self.max_length = max_length

    @staticmethod
    def key(raw: np.ndarray) -> str:
        raw = np.ascontiguousarray(raw, dtype=np.float32)
        h = hashlib.blake2b(digest_size=20)
        h.update(PREPROCESS_VERSION.encode())
        h.update(str(raw.shape).encode())
        h.update(raw.tobytes())
        return h.hexdigest()

    def _path(self, key: str) -> str:
        return os.path.join(self.dir, key[:2], key + ".npz")

    def get(self, raw: np.ndarray) -> Optional[ReferenceStats]:
        path = self._path(self.key(raw))
        if not os.path.exists(path):
            return None
        with np.load(path) as z:
            return ReferenceStats(mean=z["mean"], sigma=z["sigma"], bad=z["bad"].astype(bool),
                                  substitute=[int(s) for s in z["substitute"]],
                                  sigma_fallback=[int(s) for s in z["sigma_fallback"]])

    def put(self, raw: np.ndarray, ref: ReferenceStats) -> None:
        if self.max_length is not None and raw.shape[-1] > self.max_length:
            return
        path = self._path(self.key(raw))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
        with os.fdopen(fd, "wb") as f:
            np.savez(f, mean=ref.mean, sigma=ref.sigma, bad=ref.bad,
                     substitute=np.asarray(ref.substitute, dtype=np.int64),
                     sigma_fallback=np.asarray(ref.sigma_fallback, dtype=np.int64))
        os.replace(tmp, path)

    def __call__(self, raw: np.ndarray) -> ReferenceStats:
        ref = self.get(raw)
        if ref is None:
            ref = reference_stats(raw)
            self.put(raw, ref)
        return ref


def _check_channels(signals: torch.Tensor, n: int, step: str) -> None:
    if signals.dim() != 2 or signals.shape[0] != n:
        raise ValueError(f"{step}: ожидался сигнал ({n}, L), получен {tuple(signals.shape)}")


def _to_tensor(x: np.ndarray, dtype=np.float32) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x, dtype=dtype))


class ReferenceStandardize(Module):
    """Шаги 1–2: опорные статистики по исходному сигналу (15, L) и стандартизация (x − mean) / σ_опорн."""

    def __init__(self, context: EddyContext, cache: Optional[ReferenceCache] = None):
        super().__init__()
        self.context = context
        self.cache = cache

    def forward(self, signals, bboxes, labels):
        _check_channels(signals, N_CHANNELS, "ReferenceStandardize")
        raw = signals.detach().cpu().numpy().astype(np.float32)
        ref = self.cache(raw) if self.cache is not None else reference_stats(raw)
        self.context.ref = ref
        z = (raw - ref.mean[:, None]) / ref.sigma[:, None]
        return _to_tensor(z), bboxes, labels


class ICASuppression(Module):
    """Шаг 7: 15 → 30 каналов; ICA только по исправным каналам (маска шага 1), битые — копия входа."""

    def __init__(self, context: EddyContext):
        super().__init__()
        self.context = context

    def forward(self, signals, bboxes, labels):
        _check_channels(signals, N_CHANNELS, "ICASuppression")
        ref = self.context.require("ICASuppression")
        return _to_tensor(ica_suppress(signals.detach().cpu().numpy(), ref.bad)), bboxes, labels


class Destandardize(Module):
    """Шаг 8: обратно в единицы АЦП для обеих половин: × σ_опорн + mean."""

    def __init__(self, context: EddyContext):
        super().__init__()
        self.context = context

    def forward(self, signals, bboxes, labels):
        _check_channels(signals, 2 * N_CHANNELS, "Destandardize")
        ref = self.context.require("Destandardize")
        sigma = torch.from_numpy(np.tile(ref.sigma, 2)).to(signals.dtype)[:, None]
        mean = torch.from_numpy(np.tile(ref.mean, 2)).to(signals.dtype)[:, None]
        return signals * sigma + mean, bboxes, labels


class FourierAlign(Module):
    """Шаг 9: вычитание НЧ-гармоник (`fourier_align`) только в «чистой» половине [15:30]."""

    def forward(self, signals, bboxes, labels):
        _check_channels(signals, 2 * N_CHANNELS, "FourierAlign")
        out = signals.clone()
        out[N_CHANNELS:] = _to_tensor(fourier_align(signals[N_CHANNELS:].detach().cpu().numpy()))
        return out, bboxes, labels


class ThresholdNormalize(Module):
    """Шаг 11: замена битых каналов по карте шага 1 (обе половины), σ по «чистой» половине, нормировка.

    После шага контекст очищается: следующий пример обязан снова пройти `ReferenceStandardize`.
    """

    def __init__(self, context: EddyContext):
        super().__init__()
        self.context = context

    def forward(self, signals, bboxes, labels):
        _check_channels(signals, 2 * N_CHANNELS, "ThresholdNormalize")
        ref = self.context.require("ThresholdNormalize")
        out, _, _ = threshold_normalize(signals.detach().cpu().numpy(), ref.substitute)
        self.context.ref = None
        return _to_tensor(out), bboxes, labels


class SelectChannels(Module):
    """Срез каналов после шага 11 (абляция входа E6): 'all' — все 30, 'raw' — [0:15], 'filtered' — [15:30]."""

    SLICES = {"all": slice(0, 2 * N_CHANNELS), "raw": slice(0, N_CHANNELS), "filtered": slice(N_CHANNELS, 2 * N_CHANNELS)}

    def __init__(self, subset: str):
        super().__init__()
        if subset not in self.SLICES:
            raise ValueError(f"SelectChannels: неизвестный режим {subset!r}, есть {sorted(self.SLICES)}")
        self.subset = subset

    def forward(self, signals, bboxes, labels):
        _check_channels(signals, 2 * N_CHANNELS, "SelectChannels")
        return signals[self.SLICES[self.subset]].contiguous(), bboxes, labels


class PreprocessedCache(Module):
    """Шаги предобработки целиком (val/test, без аугментаций) с дисковым кэшем результата.

    Результат хранится в float16 (`<root>/preprocessed/`, ключ — как у `ReferenceCache`): 30 × 50 000
    отсчётов ≈ 3 МБ на сигнал (как в прежнем датасете .pt); сеть всё равно работает в AMP. Чтобы выход не
    зависел от того, был ли кэш заполнен, результат **всегда** округляется до float16 (и при промахе).
    Сигналы длиннее `max_length` считаются без записи в кэш.
    """

    def __init__(self, ops: List[Module], root: str, max_length: Optional[int] = 50000):
        super().__init__()
        self.ops = list(ops)
        self.dir = os.path.join(str(root), "preprocessed")
        self.max_length = max_length

    def _path(self, key: str) -> str:
        return os.path.join(self.dir, key[:2], key + ".npy")

    def forward(self, signals, bboxes, labels):
        raw = signals.detach().cpu().numpy().astype(np.float32)
        path = self._path(ReferenceCache.key(raw))
        if os.path.exists(path):
            return torch.from_numpy(np.load(path).astype(np.float32)), bboxes, labels
        out = signals
        for op in self.ops:
            out, bboxes, labels = op(out, bboxes, labels)
        out16 = out.detach().cpu().numpy().astype(np.float16)
        if self.max_length is None or raw.shape[-1] <= self.max_length:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
            with os.fdopen(fd, "wb") as f:
                np.save(f, out16)
            os.replace(tmp, path)
        return torch.from_numpy(out16.astype(np.float32)), bboxes, labels


class EddyPreprocessing:
    """Набор шагов предобработки с общим контекстом и (необязательным) кэшем опорных статистик.

    Пример (валидация, без аугментаций):
        pre = EddyPreprocessing(cache_dir="data/cache")
        ops = pre.validation_ops() + [Padding(regular_length), Resize(input_length)]
    В D6 обучающий конвейер собирается из `pre.reference`, `pre.ica`, `pre.destandardize`, `pre.fourier`,
    `pre.normalize` с аугментациями между ними.
    """

    def __init__(self, cache_dir: Optional[str] = None, cache_max_length: Optional[int] = 50000):
        self.cache_dir = cache_dir
        self.cache_max_length = cache_max_length
        self.context = EddyContext()
        cache = ReferenceCache(cache_dir, cache_max_length) if cache_dir else None
        self.reference = ReferenceStandardize(self.context, cache)
        self.ica = ICASuppression(self.context)
        self.destandardize = Destandardize(self.context)
        self.fourier = FourierAlign()
        self.normalize = ThresholdNormalize(self.context)

    def validation_ops(self) -> List[Module]:
        """Шаги 1–2, 7, 8, 9, 11 подряд (без аугментаций)."""
        return [self.reference, self.ica, self.destandardize, self.fourier, self.normalize]

    def cached_validation_ops(self) -> List[Module]:
        """Как `validation_ops`, но с кэшем полного результата (`PreprocessedCache`), если задан `cache_dir`."""
        if not self.cache_dir:
            return self.validation_ops()
        return [PreprocessedCache(self.validation_ops(), self.cache_dir, self.cache_max_length)]
