# YOLO-1D: Universal Model for Object Detection Task in One Dimensional Signals

Official repository for the paper **"Neural Network YOLO-1D for Useful Signals Detection on Eddy Current Rail Defectograms"** published in *Trudy ISP RAN/Proc. ISP RAS* (RSCI).

This repository provides the core neural network architecture, domain-specific pre- and post-processing pipelines, pre-trained weights, and evaluation scripts for two distinct signal/spatial processing domains:
1. **Biomedical Signal Processing (ECG):** Automated delineation on the Lobachevsky University Electrocardiography Database (**LUDB**).
2. **Non-Destructive Testing (Defectograms):** Automated constructive elements detection on our newly released dataset [**ECRv1**](https://huggingface.co/datasets/Nasochec/ECRv1) containing records of eddy current signals obtained from defectoscope during non-destructive testing of railway tracks.

---

## Released Artifacts & Licenses

All components of this research are openly available for academic and commercial use:

| Component | Task / Domain | Link | License | Rospatent Reg. No. |
| :--- | :--- | :--- | :--- | :--- |
| **Source Code** | Core backbone + ECG & Defectogram pipelines | This repository | [Apache 2.0](LICENSE) | *Pending* <!-- No. 20266XXXXXX, No. 20266YYYYYY --> |
| **Weights (Task 1)** | Pre-trained checkpoint for ECG (LUDB) | [HF Model Hub](https://huggingface.co/Nasochec/YOLO-1D_ECG_LUDB) | [Apache 2.0](LICENSE) | *Included in SW Reg.* |
| **Weights (Task 2)** | Pre-trained checkpoint for Defectograms | [HF Model Hub](https://huggingface.co/Nasochec/YOLO-1D_ECRv1) | [Apache 2.0](LICENSE) | *Included in SW Reg.* |
| **Defectogram Dataset** | New large-scale annotated dataset for Task 2 | [HF Datasets](https://huggingface.co/datasets/Nasochec/ECRv1) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | *Pending* <!-- No. 202662ZZZZZ --> |

---

## Overview of the Architecture & Pipelines

There is one Python package, `src/yolo_1d/`, and **one network, one loss and one post-processing stage for both tasks**. The two domains differ only in configuration (`TrainingSettings`, a pydantic model in `src/yolo_1d/utils/settings/settings.py`), in the dataset reader and in the transform pipeline that is built for the dataset.

**Shared core**

* `nets/nn.py` — a 1-D YOLO-style detector: `DarkNet` backbone (8 stride-2 convolutions interleaved with CSP blocks, SPP and PSA self-attention on the last level), `DarkFPN` neck and an anchor-free `Head` that regresses the distance from each anchor to the left/right boundary with Distribution Focal Loss (16 bins) and classifies the interval. Sizes `nano`/`small`/`medium`/`large`; the 8 backbone strides are configurable (`backbone_strides`), which sets the head strides.
* `nets/criterion.py` — task-aligned assigner (TAL) + IoU + DFL loss, unchanged between tasks.
* `nets/ops.py` and `nets/training.py` — shared model/loss operations (anchors and IoU) and training helpers (optimizer parameter groups, scheduler, EMA, gradient clipping and reproducibility).
* `yolo_1d.py` — `Trainer`: training loop with EMA, AMP and early stopping, `validate()` / `test()` (mAP, P, R), `predict()` for a single signal, `evaluate_delineation()` (ECG metrics), `collect_predictions()`, `export_onnx()`.
* `utils/metrics/detection.py` — 1-D NMS (`non_max_suppression`), optional removal of nested objects (`suppress_contained`), and AP computation.

**Task 1 — ECG delineation on LUDB** (per-lead detection of P, QRS and T as intervals `[onset, offset]`):

* *Data & preprocessing:* `utils/datasets/ludb.py` reads WFDB records and wave annotations; `utils/datasets/ludb_preprocess.py` z-scores each lead (optional band-pass, off by default). `data/prepare_ludb.py` writes one example per lead (input `1 × 5000` samples at 500 Hz, zero-padded to 5120 inside the transforms) and `data/make_splits.py` makes 5 patient-level folds. During training the unannotated edges of a lead are masked with a random margin (`transforms/edge_mask.py`).
* *Architecture settings:* `model_size: nano` (1.07 M parameters), `backbone_strides: B` (head strides 8/16/32 instead of the eddy-current 32/128/256), `bias_init: 1d`, `nms_iou: 0.2`. Optional multi-lead mode (`n_leads: 12`, `utils/datasets/multilead_dataset.py`, `transforms/ecg_multilead_transforms.py`) feeds all 12 leads and predicts a set of waves per lead.
* *Augmentations:* 16 ECG-specific augmentations (`transforms/ecg_transforms.py`: noises including real MIT-BIH NSTDB noise, time warps, amplitude scaling, masking, record splicing), see `docs/augmentations/README.md`.
* *Postprocessing & metrics:* NMS → intervals; per-class confidence thresholds picked on the validation fold; `utils/metrics/delineation.py` computes Se / PPV / F1 and mean ± SD error (ms) for the six fiducial points (P/QRS/T on- and offsets, ±150 ms tolerance). Model selection uses `F1_mean6`.

**Task 2 — constructive elements on eddy-current defectograms (ECRv1)** (6 classes: flashbutt, aluminothermic, rail end, axle counter, crossing, manual arc):

* *Data & preprocessing:* a record is an `int16` array `(15, L)`, L ≈ 50 000 samples, with labels `type begin end`. `utils/datasets/ecr.py` locates the dataset (local folder or `hf://Nasochec/ECRv1`) and writes the manifest `data/ecr_v1.yaml`; `utils/datasets/dataset.py` reads it (`format: ecr_npy`). Preprocessing runs **on the fly inside the transforms** (`transforms/eddy_preprocess.py`, numerics in `utils/eddy/`): reference standardization, detection and substitution of dead channels, FastICA interference suppression that produces 15 "clean" channels next to the 15 raw ones (→ 30 input channels), Fourier low-frequency alignment and threshold normalization; the result is padded to 50 000 and resized to the network input of 49 152. Full preprocessing results are cached on disk (`preprocess_cache` in the manifest).
* *Architecture settings:* `model_size: large`, `input_channels: 30`, original strides (head strides 32/128/256), `nms_iou: 0.65`. `input_subset: raw|filtered` trains on 15 of the 30 channels (input ablation).
* *Augmentations:* the eddy-current pipeline (`transforms/transforms.py`, `transforms/square_noises.py`): random resize/crop, phase-spectrum noise, jitter, inversion, reversal, channel scaling/masking, time masking, mosaic, mix-up, sinusoidal and square-wave interference; ablation configs A0 … A8 add them cumulatively.
* *Postprocessing:* NMS, rescaling of boxes to the original sample grid, optional suppression of objects nested inside a crossing (`contain_*` settings), mAP@0.5 / mAP@0.5:0.95 per class.

---

## Installation

Tested with Python 3.11.9, PyTorch 2.7.1 (CUDA 12.8) on Windows 10 with an RTX 5060 Ti; the code also runs on CPU. 

```bash
git clone https://github.com/BystrovGladkov/YOLO1D.git
cd YOLO1D

# 1. PyTorch + torchvision for your CUDA version: https://pytorch.org/get-started/locally/
pip install torch torchvision

# 2. The package itself (declares only `wfdb`; add `[hf]` to be able to download ECRv1 from Hugging Face)
pip install -e .[hf]

# 3. Remaining runtime dependencies (there is no requirements.txt yet)
pip install numpy scipy pyyaml pydantic tqdm pandas torchinfo scikit-learn numba matplotlib
```

`scikit-learn` and `numba` are needed only by the eddy-current preprocessing (FastICA, thresholds); `matplotlib` only by the report/plot scripts. The entry-point scripts (`run_experiment.py`, `run_ec_ablation.py`, `data/*.py`) add `src/` to `sys.path` themselves, so they work without `pip install -e .`.

---

## Quick Start (Out-of-the-Box Inference)

There are no bundled sample files or notebooks; the snippets below are the actual API and were run against the checkpoints of this repository.

**Download the weights.** The pre-trained checkpoints are published on the Hugging Face Model Hub together with the `args.yaml` of the run (the full `TrainingSettings` the model was trained with) and a model card with the data split, metrics and licensing details. Download each model into its own folder (`weights/` is git-ignored):

```bash
pip install huggingface_hub            # or: pip install -e .[hf]
huggingface-cli download Nasochec/YOLO-1D_ECG_LUDB --local-dir weights/ecg   # ECG, 8.5 MB
huggingface-cli download Nasochec/YOLO-1D_ECRv1    --local-dir weights/ecr   # Eddy Current Defectograms, 111 MB
```

| Task | Model Hub | Local folder | Model | Test result |
| :--- | :--- | :--- | :--- | :--- |
| ECG delineation (LUDB) | [Nasochec/YOLO-1D_ECG_LUDB](https://huggingface.co/Nasochec/YOLO-1D_ECG_LUDB) | `weights/ecg/` | `nano`, 1.07 M parameters, single lead; E4 configuration, fold 0 of the 5-fold split, seed 2 | mean F1 over the six fiducial points 99.05 %, mean boundary SD 14.2 ms (fold 0 test split, 40 records) |
| Eddy Current Defectograms (ECRv1) | [Nasochec/YOLO-1D_ECRv1](https://huggingface.co/Nasochec/YOLO-1D_ECRv1) | `weights/ecr/` | `large`, 14.5 M parameters, 30 input channels; full eddy-current augmentation set, 50 epochs, seed 0 | mAP@0.5:0.95 = 82.4 %, mAP@0.5 = 96.2 % (83.1 % / 96.7 % with nested-object post-processing; ECRv1 test split, 294 signals) |

Each folder then contains `best.pt` and `args.yaml`, which is all the snippets below need. Your own training runs produce the same two files next to each other in `experiments/<run>/train1/` (checkpoints are git-ignored), so a run folder can be used in place of `weights/<task>` everywhere.

A checkpoint is a `torch.save` dict with `model`, `ema` and `epoch`; `Trainer.load_checkpoint()` loads it and `Trainer.predict()` runs the EMA model. `predict()` takes **one** signal `(channels, length)` as a `torch.Tensor`, applies the validation transform of the task and returns a tensor `(n, 4)` with columns `[left, right, confidence, class]` in samples of the original signal.

### 1. Run ECG Pipeline (LUDB Sample)

Download the weights (above) and LUDB (13.8 MB from PhysioNet): `python data/download_ludb.py`. Any other lead works as well, as long as it is sampled at 500 Hz, is at most 5000 samples long and is z-scored.

```python
import sys, torch
sys.path.insert(0, "src")                                   # or `pip install -e .` once
from yolo_1d import Trainer, read_settings
from yolo_1d.utils.datasets.ludb import load_record, LEADS
from yolo_1d.utils.datasets.ludb_preprocess import zscore

run = "weights/ecg"                                         # downloaded from the Model Hub (best.pt + args.yaml)
settings = read_settings(f"{run}/args.yaml")                # full TrainingSettings of that run
settings.dataset = None                                     # no dataset needed for inference
trainer = Trainer(settings)                                 # builds the model and the test transform (pad 5000 -> 5120)
trainer.load_checkpoint(f"{run}/best.pt")                   # model + EMA weights; predict() uses the EMA model

record = load_record("data/raw/ludb", "1")                  # (12, 5000) in mV, 500 Hz (python data/download_ludb.py)
for lead, signal in zip(LEADS, record.signal):
    x = torch.from_numpy(zscore(signal))[None, :]           # (1, 5000): one lead, z-scored exactly as in training
    det = trainer.predict(x, conf=0.25)[0]                  # (n, 4): [onset, offset, confidence, class] in samples
    for onset, offset, conf, cls in det.tolist():
        print(f"{lead:>3s} {settings.names[int(cls)]:>3s} {onset * 2:7.1f} - {offset * 2:7.1f} ms  conf={conf:.2f}")
```

Output for lead II of record 1 (two consecutive beats; the full output lists all 12 leads):

```text
 ii   P  3823.0 -  3904.0 ms  conf=0.91
 ii QRS  3955.4 -  4058.2 ms  conf=0.89
 ii   T  4235.6 -  4439.7 ms  conf=0.94
 ii   P  5094.4 -  5197.8 ms  conf=0.91
 ii QRS  5244.6 -  5336.4 ms  conf=0.89
 ii   T  5523.8 -  5738.2 ms  conf=0.91
```

Notes: classes are `0 = P, 1 = QRS, 2 = T`; at 500 Hz one sample is 2 ms. The model sees the whole 10 s lead, so it also reports waves outside the expert-annotated zone of LUDB (including partially cut waves at the record edges, which can have negative onsets). To reproduce the paper's metrics on a fold (per-class thresholds tuned on `val`, Se/PPV/F1 and errors on `test`) keep `settings.dataset` as in `args.yaml` and call `trainer.evaluate_delineation(save_dir=...)`; this is what `run_experiment.py` does after training.

### 2. Run Defectogram Pipeline (New Dataset Sample)

Inference needs the dataset manifest, because it tells the `Trainer` to build the eddy-current preprocessing and where to cache it. Create it once; the first form downloads ECRv1 from Hugging Face (`pip install -e .[hf]`), the second uses a local copy:

```bash
python data/prepare_ecr.py --source hf://Nasochec/ECRv1 --cache-dir ./data/raw/ecr_v1   # -> data/ecr_v1.yaml
python data/prepare_ecr.py --source /path/to/ECRv1 --out data/ecr_v1.yaml
```

```python
import sys, numpy as np, torch
sys.path.insert(0, "src")
from yolo_1d import Trainer, read_settings

run = "weights/ecr"                                         # downloaded from the Model Hub (best.pt + args.yaml)
settings = read_settings(f"{run}/args.yaml")
settings.dataset = "data/ecr_v1.yaml"                       # manifest from data/prepare_ecr.py: switches the Trainer to
                                                            # the eddy-current preprocessing and sets its cache folder
trainer = Trainer(settings)
trainer.load_checkpoint(f"{run}/best.pt")

raw = np.load("data/raw/ecr_v1/test/data/passage24_left_0003.npy")   # int16, (15, 50000): 15 raw sensor channels
det = trainer.predict(torch.from_numpy(raw), conf=0.5)[0]   # (n, 4): [begin, end, confidence, class] in samples
for begin, end, conf, cls in det.tolist():
    print(f"{settings.names[int(cls)]:>14s} {begin:8.0f} - {end:8.0f}  conf={conf:.2f}")
```

Output for that test record, next to its ground truth (`test/labels/passage24_left_0003.txt`, format `type begin end` with the original class ids `2 = aluminothermic`, `5 = crossing`):

```text
      crossing    24628 -    27559  conf=0.99      5 24537 27687
aluminothermic    22141 -    22352  conf=0.93      2 22143 22343
aluminothermic     9223 -     9423  conf=0.92      2  9221  9421
aluminothermic    41411 -    41616  conf=0.92      2 41408 41608
aluminothermic    29233 -    29438  conf=0.91      2 29235 29435
aluminothermic    46268 -    46479  conf=0.90      2 46270 46470
aluminothermic    13431 -    13643  conf=0.90      2 13443 13643
aluminothermic    17132 -    17344  conf=0.89      2 17146 17346
```

Notes: Class ids of the ECRv1 labels (`1, 2, 3, 4, 5, 12`) are mapped to network classes `0 … 5` by `class_map` in the manifest. To evaluate a trained run on the whole `test` split use `python run_ec_ablation.py postprocess-eval A8` (writes `metrics_pp.json` next to the run).

---

## Full Datasets & Training

Every experiment is a YAML config whose `training:` block is a `TrainingSettings` object; a run folder keeps the resolved `args.yaml`, `environment.json`, `step.csv` with per-epoch metrics, `metrics.json` and the checkpoints. Finished runs are skipped on re-launch (`--rerun` to force).

### Task 1 — LUDB

```bash
python data/download_ludb.py        # LUDB v1.0.1 from PhysioNet -> data/raw/ludb/ 
python data/download_nstdb.py       # MIT-BIH NSTDB noise records bw/ma/em -> data/raw/nstdb/ (augmentation N5)
python data/prepare_ludb.py         # per-lead examples -> data/processed/ludb/ (+ manifest.csv, data/ludb.yaml)
python data/make_splits.py          # 5 patient-level folds -> data/splits/fold_<k>.yaml

# final single-lead model of the paper: nano, 16 augmentations, 250 epochs, 5 folds x seeds {2, 3}
python run_experiment.py experiments/configs/final.yaml
python run_experiment.py experiments/configs/final.yaml --folds 0 --seeds 2 --epochs 3   # smoke test
python run_experiment.py experiments/configs/final.yaml --summary-only                   # rebuild summary.md/json

# 12-lead mode (E7)
python data/prepare_ludb_mc.py && python data/make_splits_mc.py
python run_experiment.py experiments/configs/mc_full.yaml
```

Results go to `experiments/<name>/` (`summary.md` pools the test folds; one line per experiment is appended to `experiments/results.csv`). 

### Task 2 — ECRv1

ECRv1 has 2099 / 560 / 294 records in `train` / `val` / `test` (`{split}/data/*.npy`, `{split}/labels/*.txt`, `annotations/*.json`).

```bash
python data/prepare_ecr.py --source hf://Nasochec/ECRv1 --cache-dir ./data/raw/ecr_v1   # manifest data/ecr_v1.yaml

python run_ec_ablation.py make-configs                 # writes experiments/ec_ablation/configs/{A0..A8,IN_*}.yaml
python run_ec_ablation.py train A8 --seeds 0           # full augmentation set, large model, 50 epochs
python run_ec_ablation.py train A0 A8 --seeds 0 1 2    # several configs / seeds
python run_ec_ablation.py noisy-test A0 A8 --seeds 0   # robustness: test split + square-wave interference
python run_ec_ablation.py postprocess-eval A0 A8       # re-evaluate best.pt with nested-object suppression
python run_ec_ablation.py summary                      # summary.csv / summary_agg.csv / summary.md
```

Runs go to `experiments/ec_ablation/runs/<ID>_s<seed>/`. Preprocessing is computed by the DataLoader workers and cached in `data/cache/`. On the test split the A8 model reaches mAP@0.5:0.95 = 81.6 % and mAP@0.5 = 96.5 % (seed 0, `experiments/ec_ablation/summary.md`); the released checkpoint (`weights/ecr`, [Nasochec/YOLO-1D_ECRv1](https://huggingface.co/Nasochec/YOLO-1D_ECRv1)) reaches 82.4 % and 96.2 %, and 83.1 % / 96.7 % with nested-object post-processing (`postprocess-eval`).

---

## Repository Structure

```text
├── README.md
├── pyproject.toml               # package `yolo_1d` (src layout); extra `hf` = huggingface_hub
├── run_experiment.py            # ECG: k-fold x seeds training + delineation evaluation from a YAML config
├── run_ec_ablation.py           # ECRv1: make-configs / train / noisy-test / postprocess-eval / summary
├── start_example.py             # minimal Trainer usage without
├── src/yolo_1d/
│   ├── __init__.py              # exports Trainer, TrainingSettings, read_settings, save_settings
│   ├── yolo_1d.py               # Trainer: train / validate / test / predict / evaluate_delineation / export_onnx
│   ├── nets/
│   │   ├── nn.py                # 1-D YOLO: DarkNet backbone, DarkFPN, DFL head, nano…large, stride templates
│   │   ├── criterion.py         # TAL assigner, IoU + DFL loss (+ multi-lead wrapper)
│   │   ├── ops.py               # anchor generation and 1-D IoU operations
│   │   ├── training.py          # scheduler, EMA, optimizer groups, gradient clipping and seed setup
│   │   ├── checkpoint.py        # checkpoint loading and legacy checkpoint conversion
│   │   └── export.py            # ONNX export wrapper (pad/resize + box rescaling)
│   └── utils/
│       ├── settings/            # TrainingSettings / AugmentationSettings (pydantic) and YAML I/O
│       ├── datasets/            # Dataset readers and implementations for LUDB/ECRv1, multi-lead data and sampling
│       ├── metrics/             # detection (NMS/AP/matching) and ECG delineation metrics
│       ├── eddy/                # eddy-current preprocessing numerics (ICA, Fourier alignment, thresholds)
│       ├── transforms/          # transform pipelines: eddy_preprocess.py, transforms.py, square_noises.py,
│       │                        # ecg_transforms.py, ecg_multilead_transforms.py, edge_mask.py, build_transforms.py
│       └── logging.py, constants.py
├── data/
│   ├── README.md                # dataset notes (Russian)
│   ├── download_ludb.py, download_nstdb.py, prepare_ludb.py, prepare_ludb_mc.py, make_splits.py, make_splits_mc.py
│   ├── prepare_ecr.py           # ECRv1 manifest (local or hf://Nasochec/ECRv1)
│   ├── ludb.yaml, ecr_v1.yaml   # dataset manifests
│   ├── splits/                  # LUDB 5-fold patient-level splits (tracked)
│   ├── reports/                 # LUDB data report and figures
│   └── raw/, processed/, cache/ # downloaded data, prepared examples, preprocessing cache (git-ignored)
├── weights/                     # pre-trained checkpoints downloaded from the Model Hub (git-ignored)
│   ├── ecg/                     # Nasochec/YOLO-1D_ECG_LUDB: best.pt + args.yaml
│   └── ecr/                     # Nasochec/YOLO-1D_ECRv1:    best.pt + args.yaml
└── EC_preprocess/               # original (legacy) defectogram preprocessing code, kept for reference
```

---

## Third-Party Data Attribution

* **LUDB Dataset:** The ECG experiments use the [Lobachevsky University Electrocardiography Database (LUDB)](https://physionet.org/content/ludb/1.0.1/) by Kalyakulina et al., distributed via PhysioNet under the **Open Data Commons Attribution License (ODC-BY) v1.0**. It is downloaded by `data/download_ludb.py` into `data/raw/ludb/` and is not redistributed in this repository.
* **MIT-BIH Noise Stress Test Database:** the noise records `bw`, `ma`, `em` (Moody, Muldrow, Mark, 1984; PhysioNet, ODC-BY v1.0) are used as a training augmentation only, downloaded by `data/download_nstdb.py`.

---

## License and Intellectual Property

This project was developed at **P.G. Demidov Yaroslavl State University** (*Department of Theoretical Computer Sciences*).

* **Source Code & Pre-trained Weights:** Licensed under the [Apache License 2.0](LICENSE) (see [NOTICE](NOTICE) for attribution details).
* **Defectogram Dataset ECRv1:** Licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license<!-- and avaliable on [link](https://huggingface.co/datasets/Nasochec/ECRv1) -->.

Copyright (c) 2026 P.G. Demidov Yaroslavl State University.  
Developed by: Gladkov A. N., Bystrov L. Y., and Kuzmin E. V..

<!-- РАСКОММЕНТИРОВАТЬ ПОСЛЕ ПОЛУЧЕНИЯ СВИДЕТЕЛЬСТВ РОСПАТЕНТА:
### State Registration of Intellectual Property (Rospatent / FIPS)
* **Computer Program (ECG Analysis Pipeline):** Certificate of State Registration No. 20266XXXXXX (dated DD.MM.2026).
* **Computer Program (Defectogram Analysis Pipeline):** Certificate of State Registration No. 20266YYYYYY (dated DD.MM.2026).
* **Database (Defectogram Dataset):** Certificate of State Registration No. 202662ZZZZZ (dated DD.MM.2026).
-->

---

## Citation

If you use our neural network architecture, processing pipelines, pre-trained weights, or dataset in your research, please cite our paper:

```bibtex
@article{Gladkov2027,
  title   = {Neural Network YOLO-1D for Useful Signals Detection on Eddy Current Rail Defectograms},
  author  = {Gladkov A.N. and Bystrov L.Y. and Kuzmin E.V.},
  note    = {Submitted to Trudy ISP RAN/Proc. ISP RAS},
}
```
  <!--
  year    = {2027},
  journal = {[Trudy ISP RAN/Proc. ISP RAS]},
  volume  = {[XX]},
  number  = {[X]},
  pages   = {[XX--XX]},
  doi     = {[10.XXXX/XXXXXX]},
  note    = {RSCI. EDN: [XXXXXX]}-->


<!-- **Для цитирования в русскоязычных изданиях (РИНЦ / ГОСТ):**
> *Фамилия И. О., Соавтор И. О.* Название вашей статьи на русском языке // Название журнала. — 2026. — Т. XX, № X. — С. XX–XX. — DOI: 10.XXXX/XXXXXX. — EDN: XXXXXX. -->
