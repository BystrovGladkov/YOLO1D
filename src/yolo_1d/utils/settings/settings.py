from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field, model_validator

# каналы входа сети для каждого режима TrainingSettings.input_subset (абляция входа E6, TASKS.md D8)
SUBSET_CHANNELS = {"all": 30, "raw": 15, "filtered": 15}


class ECGAugmentationSettings(BaseModel):
    """ЭКГ-аугментации (A1, docs/research/R3_augmentations.md §3); работают при `pipeline: ecg`.

    Вероятность `*_p` = 0 выключает аугментацию. SNR — в дБ относительно активной части сигнала,
    диапазоны — [min, max], длительности — в секундах при 500 Гц.
    """
    # N — шумы
    wander_p: float = Field(0.0, ge=0, le=1, description="N1 baseline wander probability.")
    wander_amp: float = Field(0.5, ge=0, description="N1 max amplitude, in SD of signal.")
    wander_freq: List[float] = Field([0.05, 0.5], min_length=2, max_length=2, description="N1 frequency range, Hz.")
    powerline_p: float = Field(0.0, ge=0, le=1, description="N2 powerline 50/60 Hz probability.")
    powerline_snr: List[float] = Field([15, 35], min_length=2, max_length=2)
    emg_p: float = Field(0.0, ge=0, le=1, description="N3 EMG bursts probability.")
    emg_snr: List[float] = Field([10, 25], min_length=2, max_length=2)
    white_p: float = Field(0.0, ge=0, le=1, description="N4 white noise probability.")
    white_snr: List[float] = Field([20, 35], min_length=2, max_length=2)
    nstdb_p: float = Field(0.0, ge=0, le=1, description="N5 real noise (MIT-BIH NSTDB bw/ma/em) probability.")
    nstdb_snr: List[float] = Field([6, 24], min_length=2, max_length=2)
    nstdb_root: str = Field("data/raw/nstdb", description="Folder with NSTDB records (data/download_nstdb.py).")
    step_p: float = Field(0.0, ge=0, le=1, description="N6 baseline step probability.")
    step_amp: float = Field(1.0, ge=0, description="N6 max step, in SD of signal.")
    filter_p: float = Field(0.0, ge=0, le=1, description="N7 random high/low-pass filter probability.")
    # T — время
    time_scale_p: float = Field(0.0, ge=0, le=1, description="T1 global time scaling probability.")
    time_scale: List[float] = Field([0.85, 1.15], min_length=2, max_length=2)
    hrv_warp_p: float = Field(0.0, ge=0, le=1, description="T2 physiological HRV warp probability.")
    hrv_tp: List[float] = Field([0.6, 1.5], min_length=2, max_length=2, description="T2 diastole (TP) scale range.")
    window_warp_p: float = Field(0.0, ge=0, le=1, description="T3 window warping probability.")
    window_warp_scale: List[float] = Field([0.7, 1.4], min_length=2, max_length=2)
    # A — амплитуда
    amp_scale_p: float = Field(0.0, ge=0, le=1, description="A1 amplitude scaling probability.")
    amp_scale_sigma: float = Field(0.2, ge=0, description="A1 lognormal sigma.")
    wave_scale_p: float = Field(0.0, ge=0, le=1, description="A2 per-wave amplitude scaling probability.")
    wave_scale_sigma_pt: float = Field(0.3, ge=0, description="A2 lognormal sigma for P and T (QRS: 0.1).")
    invert_p: float = Field(0.0, ge=0, le=1, description="A3 polarity inversion probability.")
    # M — маскирование
    time_mask_p: float = Field(0.0, ge=0, le=1, description="M1 time masking probability.")
    time_mask_max: float = Field(0.1, ge=0, le=1, description="M1 max masked fraction of active part.")
    # X — склейка записей (в Dataset)
    mosaic_p: float = Field(0.0, ge=0, le=1, description="X1 naive splice of two records at random points.")
    splice_p: float = Field(0.0, ge=0, le=1, description="X2 beat splice of two records at TP gaps with isoline alignment.")


class LeadAugmentationSettings(BaseModel):
    """Аугментации отведений режима MC (A3, группа L; docs/research/R5_multilead.md §3.3); работают при
    `pipeline: ecg_multi`. Вероятность 0 выключает аугментацию."""
    dropout_p: float = Field(0.0, ge=0, le=1, description="L1 lead dropout: probability per sample.")
    dropout_lead_p: float = Field(0.15, ge=0, le=1, description="L1: probability to zero each lead.")
    dropout_max: int = Field(4, ge=1, le=12, description="L1: max zeroed leads.")
    dropout_labels: Literal["ignore", "keep"] = Field(
        "ignore", description="L1: 'ignore' — zeroed leads excluded from loss; 'keep' — their labels are kept.")
    shuffle_p: float = Field(0.0, ge=0, le=1, description="L2 lead shuffling probability.")
    shuffle_n: List[int] = Field([2, 4], min_length=2, max_length=2, description="L2: number of permuted leads.")
    electrode_swap_p: float = Field(0.0, ge=0, le=1, description="L3 limb electrode swap (LA-RA, LA-LL, RA-LL) probability.")
    amp_lead_sigma: float = Field(0.1, ge=0, description="A1/A2 in MC: lognormal sigma of per-lead factor.")


class AugmentationSettings(BaseModel):
    # Значения по умолчанию — вихретоковые данные (сигнал 50000 → вход 49152 через ресайз).
    # ЭКГ (LUDB): regular_length=5000, input_length=5120, validation_resize=False — только паддинг.
    regular_length: int = Field(50000, description="regular length of input signal, used to save aspect in validation transforms.")
    input_length: int = Field(49152, description="Target input length of signal.")
    validation_resize: bool = Field(
        True, description="Validation transform: pad to regular_length and resize to input_length (True, "
                          "eddy-current data) or only pad to input_length without changing time scale (False, ECG).")
    # аугментации помех (вихретоковые данные, TASKS.md D5); по умолчанию выключены, включаются конфигом абляции
    square_noises: float = Field(0.0, ge=0.0, le=1.0,
                                 description="Probability of SquareNoises (adds square-wave interference, 15 channels).")
    sine_noise: float = Field(0.0, ge=0.0, le=1.0, description="Probability of SineNoise (low-frequency sinusoid).")
    sine_low: float = Field(0.002, gt=0.0, description="SineNoise: lower bound of angular frequency, rad/sample.")
    sine_high: float = Field(0.03, gt=0.0, description="SineNoise: upper bound of angular frequency, rad/sample.")
    sine_amplitude: List[float] = Field([0.5, 3.0], min_length=2, max_length=2,
                                        description="SineNoise: range of peak amplitude (units of standardised signal).")
    sine_sigma: List[float] = Field([0.05, 0.2], min_length=2, max_length=2,
                                    description="SineNoise: range of width of the gaussian channel profile.")
    sine_freq_jitter: float = Field(0.1, ge=0.0, lt=1.0, description="SineNoise: max relative drift of frequency.")
    # Маскирование неразмеченных краёв записи (D5; действует, только если в yaml датасета задан ключ `zones`)
    edge_margin_start: List[int] = Field([0, 12], min_length=2, max_length=2,
                                         description="Train: range (min, max) of mask margin before first onset, samples.")
    edge_margin_end: List[int] = Field([0, 24], min_length=2, max_length=2,
                                       description="Train: range (min, max) of mask margin after last offset, samples.")
    edge_margin_eval: List[int] = Field([6, 12], min_length=2, max_length=2,
                                        description="Validation/test: fixed mask margins (before first onset, after last offset).")

    # параметры для train transform
    mark_scaling: int = Field(300, ge=1, description="Scale used in MarkScaling.")
    random_resize: float = Field(0.1, ge=0.0, description="RandomResize strength.")
    random_crop: bool = Field(True, description="RandomCrop + RandomPadding to input_length (geometry group); False — "
                                                "deterministic Padding(regular_length) + Resize(input_length), as in validation.")
    augment_apply_p: float = Field(0.5, ge=0.0, le=1.0,
                                   description="ECRv1 pipeline: every augmentation except RandomResize/RandomCrop/"
                                               "RandomPadding is wrapped in RandomApply with this probability "
                                               "(multiplied by the augmentation's own probability, if any).")
    
    phase_spectrum_noise: float = Field(0.5, ge=0.0, description="Strength of phase spectrum noise.")
    random_jitter: float = Field(0.6, ge=0.0, description="Strength of jitter.")
    random_invert: float = Field(0.5, ge=0.0, le=1.0, description="Probability of sign inversion.")
    random_reverse: float = Field(0.5, ge=0.0, le=1.0, description="Probability of time reversal.")
    channel_scaling: float = Field(0.4, ge=0.0, description="Strength of channel scaling.")
    time_masking: float = Field(0.2, ge=0.0, le=1.0, description="Max length of time masking.")
    channel_masking: float = Field(0.1, ge=0.0, le=1.0, description="Probability of channel masking.")

    # набор аугментаций обучения: "eddy" — исходный (вихретоковые данные, поля выше),
    # "ecg" — ЭКГ-аугментации из `ecg` (A1, режим S); "ecg_multi" — те же вероятности `ecg` в многоканальном
    # варианте + группа L из `leads` (A3, режим MC); значения по умолчанию сохраняют прежнее поведение
    pipeline: Literal["eddy", "ecg", "ecg_multi"] = Field("eddy", description="Training augmentation pipeline.")
    ecg: ECGAugmentationSettings = Field(default_factory=ECGAugmentationSettings)
    leads: LeadAugmentationSettings = Field(default_factory=LeadAugmentationSettings)

    # настройки для mixup / mosaic / adversarial
    mosaic: float = Field(0.0, ge=0.0, le=1.0, description="Probability of mosaic augmentation.")
    mix_up: float = Field(0.0, ge=0.0, le=1.0, description="Probability of mixup augmentation.")
    mix_up_alpha: float = Field(20.0, ge=0.0, description="Beta distribution alpha for mixup.")
    adversarial: float = Field(0.0, ge=0.0, le=1.0, description="Probability of adversarial batch.")

class TrainingSettings(BaseModel):
    """
        Settings of model for training.
    """

    # path to dataset
    dataset: Optional[str] = Field(None, description="Path to dataset manifest file (yaml).")

    # settings for saving of results
    results_path: str = Field("runs", description="Path for results of training.")
    project: str = Field("detect", description="Name of project of training.")
    resume: bool = Field(False, description="Resumes training from last project if option is `True`.")
    starting_checkpoint: Optional[str] = Field(
        None, description="Path to starting checkpoint of model. If `resume=True`, then option is ignored. "
    )

    model_size: Literal["nano", "small", "medium", "large"] = Field(
        "nano", description="Size of model. One from next options: 'nano', 'small', 'medium', 'large'."
    )
    input_channels: int = Field(30, ge=1, description="Number of channels of input signal (30 — eddy-current data, 1 — ECG per lead).")
    input_subset: Literal["all", "raw", "filtered"] = Field(
        "all", description="ECRv1: channels fed to the network after preprocessing step 11 — 'all' (30: raw + "
                           "interference-suppressed), 'raw' ([0:15]) or 'filtered' ([15:30]); input_channels is set "
                           "automatically (30 / 15 / 15).")
    backbone_strides: Union[Literal["original", "A", "B", "C"], List[int]] = Field(
        "original", description="Strides of 8 downsampling convolutions of backbone: template name "
                                "('original' — heads {32,128,256}; ECG: 'A' — {4,16,32}, 'B' — {8,16,32}, "
                                "'C' — {4,8,16}) or list of 8 values from {1, 2}.")
    bias_init: Literal["legacy", "1d"] = Field(
        "legacy", description="Initialization of classification biases: 'legacy' (original 2D formula) "
                              "or '1d' (prior = expected_objects / classes / anchors).")
    expected_objects: float = Field(24.0, gt=0.0, description="Expected number of objects per input (for bias_init='1d').")
    # режим MC (M7, docs/research/R5_multilead.md): 12 отведений на вход, по набору предсказаний на отведение
    n_leads: int = Field(1, ge=1, description="Number of per-lead prediction sets of the head (1 — one set as before; "
                                              "12 — ECG mode MC: dataset with `mode: multi_lead`).")
    head_width_mult: float = Field(1.0, gt=0.0, description="Multiplier of hidden widths of head branches (1 — as before).")
    lead_loss_norm: Literal["joint", "per_lead"] = Field(
        "joint", description="MC loss: 'joint' — per-lead losses summed and normalized by the target score sum of the "
                             "whole batch (scale as in single-lead mode); 'per_lead' — each lead normalized separately, then summed.")

    # training settings
    enable_amp: bool = Field(False, description="Enable AMP (automatic mixed precision).")
    num_workers: int = Field(0, description="Number of threads used in data loaders.")
    eval_num_workers: Optional[int] = Field(None, ge=0, description="Workers of val/test loaders; None — as "
                                            "num_workers. Persistent workers of three loaders add up in RAM (E1).")
    epochs: int = Field(10, ge=1, description="Number of Epochs of training.")
    batch_size: int = Field(32, ge=1, description="Size of batch in training.")
    early_stopping_epochs: int = Field(0, ge=0, description="Number of epochs for early stop of training.")

    # loss settings
    weight_box: float = Field(7.5, ge=0.0, description="Weight of box loss (>= 0).")
    weight_cls: float = Field(0.5, ge=0.0, description="Weight of cls loss (>= 0).")
    weight_dfl: float = Field(1.5, ge=0.0, description="Weight of dfl loss (>= 0).")

    # optimizer settings
    weight_decay: float = Field(0.0005, ge=0.0, description="Weight decay (>= 0).")
    min_lr: float = Field(0.0001, ge=0.0, description="Minimal learning rate (>= 0).")
    max_lr: float = Field(0.01, ge=0.0, description="Maximal learning rate (>= 0).")
    momentum: float = Field(0.937, ge=0.0, le=1.0, description="Momentum in optimizer (0-1).")
    warmup_epochs: float = Field(3.0, ge=0.0, description="Number of warmup epochs (>= 0).")

    names: List[str] = Field(..., description="List of names of classes (labels).")

    # reproducibility and EMA
    seed: int = Field(0, description="Random seed (python, numpy, torch).")
    ema_decay: float = Field(0.9999, gt=0.0, lt=1.0, description="Final decay of EMA of model weights.")
    ema_tau: float = Field(2000.0, gt=0.0, description="EMA warm-up: decay = ema_decay * (1 - exp(-updates / ema_tau)).")

    # evaluation and model selection
    nms_iou: float = Field(0.65, gt=0.0, le=1.0, description="IoU threshold of NMS.")
    contain_main_classes: List[int] = Field(default_factory=list, description=(
        "Post-NMS suppression (metrics.detection.suppress_contained): main classes. Empty — step disabled."))
    contain_secondary_classes: List[int] = Field(default_factory=list, description=(
        "Classes removed when > contain_threshold of their length lies inside a main-class object "
        "with not lower confidence."))
    contain_threshold: float = Field(0.8, gt=0.0, le=1.0, description="Share of a secondary object inside a main one.")
    delineation_metrics: bool = Field(
        False, description="Compute ECG delineation metrics (Se/PPV/F1, error in ms for 6 fiducials, "
                           "yolo_1d.utils.metrics.delineation) on validation each epoch and log them to step.csv.")
    exclude_boundary_points: bool = Field(
        True, description="Delineation metrics: ignore fiducials adjacent to the edge mask (onset of the first and "
                          "offset of the last wave of a lead) and predictions matched to them (D5).")
    delineation_conf: float = Field(0.25, ge=0.0, le=1.0,
                                    description="Fixed confidence threshold for per-epoch delineation metrics.")
    best_metric: Literal["val_loss", "mAP", "F1_mean6"] = Field(
        "val_loss", description="Validation metric used to select best.pt: 'val_loss' (min), 'mAP' (max) or "
                                "'F1_mean6' (max, ties broken by lower sd_mean6_ms; needs delineation_metrics).")

    # augmentations settings
    augmentations: AugmentationSettings = Field(default_factory=AugmentationSettings)

    @model_validator(mode="before")
    @classmethod
    def _subset_channels(cls, data):
        """`input_subset` raw/filtered без явного `input_channels` → 15 каналов."""
        if isinstance(data, dict) and data.get("input_subset", "all") != "all" and "input_channels" not in data:
            data = {**data, "input_channels": SUBSET_CHANNELS[data["input_subset"]]}
        return data

    @model_validator(mode="after")
    def _check_subset_channels(self):
        if self.input_subset != "all" and self.input_channels != SUBSET_CHANNELS[self.input_subset]:
            raise ValueError(f"input_subset={self.input_subset!r} требует input_channels=15, "
                             f"задано {self.input_channels}")
        return self

    class Config:
        str_strip_whitespace = True
        validate_assignment = True  # валидация при присвоении после создания
