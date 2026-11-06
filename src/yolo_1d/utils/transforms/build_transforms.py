from torchvision.transforms.v2 import Compose

from typing import List, Optional

import yaml

from .transforms import Padding, Resize, RandomResize, RandomCrop, RandomPadding, RandomReverse, RandomJitter, RandomInvert, TimeMasking, ChannelMasking, ChannelScaling, MarkScaling, PhaseSpectrumNoise, RandomApply, SineNoise
from yolo_1d.utils.settings import AugmentationSettings
from .square_noises import SquareNoises
from .eddy_preprocess import EddyPreprocessing, SelectChannels
from .ecg_transforms import (AmplitudeScale, BaselineStep, BaselineWander, EMGBursts, FitToLength, HRVWarp,
                             Invert, NSTDBNoise, PowerlineNoise, RandomFilter, TimeMask, TimeScale, WaveScale,
                             WhiteNoise, WindowWarp)
from .ecg_multilead_transforms import (CoupledNoise, ElectrodeSwap, KeepZeros, LeadBaselineStep, LeadCompose,
                                       LeadDropout, LeadShuffle, MultiLeadAmplitudeScale, MultiLeadHRVWarp,
                                       MultiLeadWaveScale)

def build_ecg_training_transforms(settings: AugmentationSettings):
    """Аугментации обучения для ЭКГ (A1). Порядок — docs/research/R3_augmentations.md §4.1:
    время (T2, T3, T1) → амплитуда (A2, A1, A3) → шумы (N1, N6, N5, N3, N2, N4) → фильтрация (N7)
    → маскирование (M1) → приведение к длине входа со случайным сдвигом. Маска краёв (EdgeMask)
    применяется раньше, при загрузке; склейка записей X1/X2 — в Dataset."""
    e = settings.ecg
    ops = [
        HRVWarp(e.hrv_warp_p, tp=e.hrv_tp),
        WindowWarp(e.window_warp_p, scale=e.window_warp_scale),
        TimeScale(e.time_scale_p, scale=e.time_scale),
        WaveScale(e.wave_scale_p, sigma_pt=e.wave_scale_sigma_pt),
        AmplitudeScale(e.amp_scale_p, sigma=e.amp_scale_sigma),
        Invert(e.invert_p),
        BaselineWander(e.wander_p, max_amp=e.wander_amp, freq=e.wander_freq),
        BaselineStep(e.step_p, max_amp=e.step_amp),
        NSTDBNoise(e.nstdb_p, root=e.nstdb_root, snr=e.nstdb_snr),
        EMGBursts(e.emg_p, snr=e.emg_snr),
        PowerlineNoise(e.powerline_p, snr=e.powerline_snr),
        WhiteNoise(e.white_p, snr=e.white_snr),
        RandomFilter(e.filter_p),
        TimeMask(e.time_mask_p, max_fraction=e.time_mask_max),
    ]
    ops = [op for op in ops if op.p > 0]
    ops.append(FitToLength(settings.input_length, random_shift=True))
    return Compose(ops)


def build_ecg_multilead_training_transforms(settings: AugmentationSettings):
    """Аугментации обучения режима MC (A3, docs/research/R5_multilead.md §3.2): вероятности из `settings.ecg`,
    порядок как в A1. Время, маска и фильтр — общие для отведений; N1/N2/N5 — общий источник с коэффициентом
    по отведению; N3/N4 — поканально; N6 — в 1–3 отведениях; A1/A2 — общий × поканальный множитель;
    A3 (инверсия) не используется. Шумы и фильтр обёрнуты в KeepZeros (маска краёв каждого отведения).
    Группа L (отведения) и склейка X1/X2 — в MultiLeadDataset (`build_lead_transforms`)."""
    e, sl = settings.ecg, settings.leads.amp_lead_sigma
    noise = [
        CoupledNoise(e.wander_p, "wander", max_amp=e.wander_amp, freq=e.wander_freq),
        LeadBaselineStep(e.step_p, max_amp=e.step_amp),
        CoupledNoise(e.nstdb_p, "nstdb", snr=e.nstdb_snr, nstdb_root=e.nstdb_root),
        EMGBursts(e.emg_p, snr=e.emg_snr),
        CoupledNoise(e.powerline_p, "powerline", snr=e.powerline_snr),
        WhiteNoise(e.white_p, snr=e.white_snr),
        RandomFilter(e.filter_p),
    ]
    ops = [
        MultiLeadHRVWarp(e.hrv_warp_p, tp=e.hrv_tp),
        WindowWarp(e.window_warp_p, scale=e.window_warp_scale),
        TimeScale(e.time_scale_p, scale=e.time_scale),
        MultiLeadWaveScale(e.wave_scale_p, sigma_pt=e.wave_scale_sigma_pt, sigma_lead=sl),
        MultiLeadAmplitudeScale(e.amp_scale_p, sigma=e.amp_scale_sigma, sigma_lead=sl),
        *[KeepZeros(op) for op in noise],
        TimeMask(e.time_mask_p, max_fraction=e.time_mask_max),
    ]
    ops = [op for op in ops if op.p > 0]
    ops.append(FitToLength(settings.input_length, random_shift=True))
    return Compose(ops)


def build_lead_transforms(settings: AugmentationSettings):
    """Группа L режима MC (A3): L1 lead dropout, L2 lead shuffling, L3 перестановка электродов
    (порядок: перестановки, затем обнуление). None — если все выключены."""
    l = settings.leads
    ops = [op for op in (ElectrodeSwap(l.electrode_swap_p),
                         LeadShuffle(l.shuffle_p, n_leads=l.shuffle_n),
                         LeadDropout(l.dropout_p, p_lead=l.dropout_lead_p, max_leads=l.dropout_max,
                                     labels_mode=l.dropout_labels)) if op.p > 0]
    return LeadCompose(ops) if ops else None


def _fit_length_ops(settings: AugmentationSettings) -> List:
    """Детерминированное приведение длины, как на валидации: паддинг до regular_length и ресайз до input_length."""
    return [Padding(settings.regular_length), Resize(settings.input_length)]


def build_ecr_training_transforms(settings: AugmentationSettings, pre: EddyPreprocessing, input_subset: str = "all"):
    """Обучающий конвейер ECRv1: предобработка на лету + аугментации (SPEC п. 3; TASKS.md, таблица шагов 1–13).

    Порядок задан пользователем и не меняется; группы включаются только параметрами. Каждая аугментация,
    кроме RandomResize, RandomCrop и RandomPadding, применяется с вероятностью `augment_apply_p`, умноженной на
    её собственную вероятность (`sine_noise`, `square_noises`). Вход — сырой сигнал (15, L), выход — (30, input_length).
    `input_subset` ("raw" / "filtered", абляция входа E6) — после шага 11 остаются 15 каналов (`SelectChannels`),
    шаги 12–13 работают с ними.
    """
    if settings.mosaic > 0 and not settings.random_crop:
        raise ValueError("mosaic склеивает сигналы длиннее regular_length: нужен random_crop=True")
    p = settings.augment_apply_p

    def maybe(op, own=1.0):
        return RandomApply(op, p=p * own)

    # 1–2: опорные статистики по «выровненной» копии (ICA + Фурье, CorrectedThreshold1) и стандартизация
    #      «грязного» (невыровненного) сигнала — дальше работаем с ним
    ops = [pre.reference]
    # 3: геометрия (без RandomApply)
    if settings.random_resize > 0:
        ops.append(RandomResize(settings.random_resize))
    if settings.random_crop:
        ops.append(RandomCrop(settings.input_length))
    # 4: спектральные, аддитивный шум, амплитудные, инверсия по времени
    if settings.phase_spectrum_noise > 0:
        ops.append(maybe(PhaseSpectrumNoise(settings.phase_spectrum_noise)))
    if settings.random_jitter > 0:
        ops.append(maybe(RandomJitter(settings.random_jitter)))
    if settings.random_invert > 0:
        ops.append(maybe(RandomInvert(settings.random_invert)))
    if settings.random_reverse > 0:
        ops.append(maybe(RandomReverse(settings.random_reverse)))
    # 5: низкочастотная синусоида
    if settings.sine_noise > 0:
        ops.append(maybe(SineNoise(settings.sine_low, settings.sine_high, settings.sine_amplitude,
                                   settings.sine_sigma, settings.sine_freq_jitter), settings.sine_noise))
    # 6: прямоугольная помеха (без встроенного подавления)
    if settings.square_noises > 0:
        ops.append(maybe(SquareNoises(), settings.square_noises))
    # 7: подавление прямоугольных помех ICA (всегда): 15 «грязных» + 15 «чистых» каналов
    # 8: обратно в единицы АЦП (× σ_опорн + mean)
    # 9: Фурье-подавление помех в «чистой» половине
    ops += [pre.ica, pre.destandardize, pre.fourier]
    # 10: маскирование по времени (до порогов — намеренно: в реальных данных куски тоже бывают пустыми)
    if settings.time_masking > 0:
        ops.append(maybe(TimeMasking(settings.time_masking)))
    # 11: новые пороговые уровни, замена битых каналов по карте шага 1, нормировка обеих половин
    ops.append(pre.normalize)
    if input_subset != "all":
        ops.append(SelectChannels(input_subset))
    # 12: амплитуды и маскирование каналов
    if settings.channel_scaling > 0:
        ops.append(maybe(ChannelScaling(settings.channel_scaling)))
    if settings.channel_masking > 0:
        ops.append(maybe(ChannelMasking(settings.channel_masking)))
    # 13: приведение длины ко входу сети — последним, чтобы шумы не испортили добавленные нули
    ops += [RandomPadding(settings.input_length)] if settings.random_crop else _fit_length_ops(settings)
    return Compose(ops)


def build_training_transforms(settings: AugmentationSettings, preprocessing: Optional[EddyPreprocessing] = None,
                              input_subset: str = "all"):
    """Обучающий трансформ. `preprocessing` задан — конвейер ECRv1 с предобработкой на лету
    (`build_ecr_training_transforms`); иначе — по `settings.pipeline` (ЭКГ или прежний вихретоковый для .pt)."""
    if settings.pipeline == "ecg":
        return build_ecg_training_transforms(settings)
    if settings.pipeline == "ecg_multi":
        return build_ecg_multilead_training_transforms(settings)
    if preprocessing is not None:
        return build_ecr_training_transforms(settings, preprocessing, input_subset)
    return _build_eddy_legacy_training_transforms(settings)


def _build_eddy_legacy_training_transforms(settings: AugmentationSettings):
    """Прежний вихретоковый конвейер для заранее подготовленных 30-канальных .pt (без предобработки на лету).

    Оставлен для старых датасетов; для ECRv1 используется `build_ecr_training_transforms`.
    """
    ops = [
        # MarkScaling(settings.mark_scaling),
        RandomResize(settings.random_resize),
        RandomCrop(settings.input_length),
    ]
    if settings.phase_spectrum_noise > 0:
        ops.append(PhaseSpectrumNoise(settings.phase_spectrum_noise))
    if settings.random_jitter > 0:
        ops.append(RandomJitter(settings.random_jitter))
    if settings.random_invert > 0:
        ops.append(RandomInvert(settings.random_invert))
    if settings.random_reverse > 0:
        ops.append(RandomReverse(settings.random_reverse))
    if settings.square_noises > 0:
        ops.append(RandomApply(SquareNoises(), p=settings.square_noises))
    if settings.time_masking > 0:
        ops.append(TimeMasking(settings.time_masking))
    if settings.channel_scaling > 0:
        ops.append(ChannelScaling())
    if settings.channel_masking > 0:
        ops.append(ChannelMasking(settings.channel_masking))
    ops.append(RandomPadding(settings.input_length))
    return Compose(ops)

def build_validation_transform(settings: AugmentationSettings, preprocessing: Optional[EddyPreprocessing] = None,
                               input_subset: str = "all"):
    """Трансформ val/test. С `preprocessing` (ECRv1): шаги 1–2, 7, 8, 9, 11 (с кэшем полного результата, если
    у `preprocessing` задан `cache_dir`) + паддинг до regular_length и ресайз до input_length."""
    ops = []
    if preprocessing is not None:
        ops += preprocessing.cached_validation_ops()   # кэш хранит все 30 каналов; срез — после него
        if input_subset != "all":
            ops.append(SelectChannels(input_subset))
    if settings.validation_resize:
        # вихретоковые данные: паддинг до regular_length и ресайз до input_length
        ops += _fit_length_ops(settings)
    else:
        # ЭКГ: только паддинг до input_length, временной масштаб не меняется
        ops.append(Padding(settings.input_length))
    return Compose(ops)


def build_transforms_for_dataset(settings: AugmentationSettings, dataset: Optional[str], input_subset: str = "all"):
    """(train, val/test) трансформы для датасета `dataset` (yaml).

    `format: ecr_npy` — предобработка на лету (`pipeline` должен быть "eddy"); кэш — `preprocess_cache` из yaml.
    У train и val/test — разные экземпляры `EddyPreprocessing` (свой контекст), дисковый кэш общий.
    Остальные форматы — как раньше. `input_subset` ("raw" / "filtered") — только для ECRv1.
    """
    fmt, cache = None, None
    if dataset is not None:
        with open(dataset, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        fmt, cache = data.get("format"), data.get("preprocess_cache")
    if fmt != "ecr_npy":
        if input_subset != "all":
            raise ValueError(f"input_subset={input_subset!r} поддерживается только для датасетов ECRv1 (format: ecr_npy)")
        return build_training_transforms(settings), build_validation_transform(settings)
    if settings.pipeline != "eddy":
        raise ValueError(f"датасет ECRv1 ({dataset}) требует pipeline='eddy', задан {settings.pipeline!r}")
    if not settings.validation_resize:
        raise ValueError("датасет ECRv1 требует validation_resize=True")
    train = build_training_transforms(settings, EddyPreprocessing(cache_dir=cache), input_subset)
    test = build_validation_transform(settings, EddyPreprocessing(cache_dir=cache), input_subset)
    return train, test
