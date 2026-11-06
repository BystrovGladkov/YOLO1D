from typing import List, Dict

import csv
import json
import os
from pathlib import Path
import warnings
import shutil

import torch
import tqdm
import yaml
from torch.utils import data
from torchvision.transforms.v2 import Compose
from torchinfo import summary
import numpy as np
import pandas as pd

from yolo_1d.nets import nn
from yolo_1d.nets import training
from yolo_1d.nets.criterion import CalculateLoss, MultiLeadLoss, flatten_lead_targets
from yolo_1d.utils.datasets.dataset import Dataset
from yolo_1d.utils.datasets.multilead_dataset import MultiLeadDataset, dataset_class
from yolo_1d.utils.datasets.sampling import get_data_weights
from yolo_1d.utils.metrics.delineation import (DelineationMetrics, apply_thresholds, best_class_thresholds,
                                       boxes_to_intervals, detections_to_intervals, format_table)
from yolo_1d.utils.metrics import detection
from yolo_1d.utils.transforms import EdgeMask, build_lead_transforms, build_transforms_for_dataset
from yolo_1d.utils.settings import TrainingSettings, get_default_settings, save_settings, read_settings
from yolo_1d.utils.constants import *

warnings.filterwarnings("ignore")

def first_available(dir_path:str, subdir:str) -> int:
    """
        Finds first free number of run.
    """
    counter = 1
    while True:
        # create potential path
        potential_subdir_name = f"{subdir}{counter}"
        full_path = os.path.join(dir_path, potential_subdir_name)

        # check path exists
        if not os.path.exists(full_path):
            return counter  # return free number
        
        counter += 1

def _seed_worker(worker_id: int) -> None:
    """Сид numpy и random воркера DataLoader из его torch-сида (torch сидирует воркеры сам)."""
    import random
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


class Trainer:
    
    def __init__(self, settings: TrainingSettings=None):
        self._settings = settings
        
        self._chkpt_path = self._settings.starting_checkpoint 
        # parameters of 
        self._results_path = os.path.abspath(self._settings.results_path)
        self._project = self._settings.project
        self._num = first_available(self._results_path, self._project)
        # is resume available
        self._resume = self._settings.resume and self._num > 1
        # if need to resume training - load last session
        if self._resume:
            self._num = max(self._num - 1, 1)
        # path for results    
        self._project_path = os.path.join(self._results_path, f"{self._project}{self._num}")
        self._num_classes = len(self._settings.names)
        self._epoch_results = dict()
        # reproducibility: same seed for model init, sampling and augmentations
        training.set_seed(self._settings.seed)
        # load settings
        self._init_model()  
        self._prepare_transforms()
        self._load_model_weights()
        
    def _get_model(self):
        """ Creates a model of desired size."""
        return nn.get_model(self._settings.model_size, self._num_classes,
                            input_channels=self._settings.input_channels,
                            input_length=self._settings.augmentations.input_length,
                            strides=self._settings.backbone_strides,
                            bias_init=self._settings.bias_init,
                            expected_objects=self._settings.expected_objects,
                            n_leads=self._settings.n_leads,
                            head_width_mult=self._settings.head_width_mult)

    @property
    def _n_leads(self) -> int:
        return self._settings.n_leads

    @property
    def _rows_per_batch(self) -> int:
        """Эффективный батч в «отведениях»: в режиме MC пример даёт n_leads строк лосса (масштаб лосса,
        накопление градиента и weight decay считаются от него; в режиме S равен batch_size)."""
        return self._settings.batch_size * self._n_leads

    def _make_criterion(self, model):
        """Исходный ComputeLoss; в режиме MC — он же по отведениям с суммированием (MultiLeadLoss, M7)."""
        criterion = CalculateLoss(model, self._settings.weight_box, self._settings.weight_cls, self._settings.weight_dfl)
        if self._n_leads == 1:
            return criterion
        return MultiLeadLoss(criterion, self._n_leads, self._settings.lead_loss_norm)
        
    def _init_model(self):
        """
            Initializes a model.
        """
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # create model
        self._model = self._get_model()
        self._model.to(self._device)
        
        # Exponential moving average for weights of model
        self._ema = training.EMA(self._model, decay=self._settings.ema_decay, tau=self._settings.ema_tau)
    
    def _init_dataloaders(self):
        # if fields already initialized - exit
        if hasattr(self, "_train_dataset"):
            return
        # маскирование неразмеченных краёв (только для датасетов с ключом `zones`): случайный запас на train,
        # фиксированный на val/test (D5)
        aug = self._settings.augmentations
        train_mask = EdgeMask(aug.edge_margin_start, aug.edge_margin_end)
        start, end = aug.edge_margin_eval
        eval_mask = EdgeMask((start, start), (end, end))
        # load dataset: `mode: multi_lead` в yaml — режим MC (MultiLeadDataset), иначе — поотведенческий Dataset
        dataset_cls = dataset_class(self._settings.dataset)
        if (dataset_cls is MultiLeadDataset) != (self._n_leads > 1) or                 (dataset_cls is MultiLeadDataset and self._n_leads != MultiLeadDataset.n_leads):
            raise ValueError(f"датасет {self._settings.dataset} ({dataset_cls.__name__}) не согласован с n_leads = "
                             f"{self._n_leads}: для режима MC нужны n_leads = 12 и yaml с `mode: multi_lead`")
        multi = dataset_cls is MultiLeadDataset
        if (aug.pipeline == "ecg" and multi) or (aug.pipeline == "ecg_multi" and not multi):
            raise ValueError(f"pipeline '{aug.pipeline}' не подходит для {dataset_cls.__name__}: "
                             "режим S — 'ecg', режим MC — 'ecg_multi'")
        ecg = aug.pipeline in ("ecg", "ecg_multi")
        extra = {"lead_transform": build_lead_transforms(aug)} if multi and aug.pipeline == "ecg_multi" else {}
        self._train_dataset = dataset_cls(self._settings.dataset, transform=self._train_transform, type="train", 
                                      mosaic_prob=self._settings.augmentations.mosaic,
                                      mix_up_prob=self._settings.augmentations.mix_up,
                                      mix_up_alpha=self._settings.augmentations.mix_up_alpha,
                                      edge_mask=train_mask,
                                      splice_prob=aug.ecg.mosaic_p if ecg else 0,
                                      beat_splice_prob=aug.ecg.splice_p if ecg else 0, **extra)
        self._val_dataset = dataset_cls(self._settings.dataset, transform=self._test_transform, type="val",
                                        edge_mask=eval_mask)
        self._test_dataset = dataset_cls(self._settings.dataset, transform=self._test_transform, type="test",
                                         edge_mask=eval_mask)
        
        weights = get_data_weights(self._train_dataset)
        sampler = torch.utils.data.WeightedRandomSampler(weights=weights,
                                    num_samples=self._train_dataset.n, replacement=True)
        
        # create dataloaders
        # при num_workers > 0: воркеры живут всё обучение (на Windows запуск воркера дорогой), а numpy и
        # random в каждом воркере сидируются от torch-сида воркера — аугментации воспроизводимы
        workers = self._settings.num_workers
        loader_kwargs = dict(num_workers=workers, pin_memory=True, collate_fn=dataset_cls.collate_fn)
        if workers > 0:
            loader_kwargs.update(persistent_workers=True, worker_init_fn=_seed_worker)
        self._train_loader = data.DataLoader(self._train_dataset, self._settings.batch_size, sampler=sampler,
                                             **loader_kwargs)
        # val/test: отдельное число воркеров (eval_num_workers); воркеры трёх загрузчиков живут одновременно
        eval_workers = self._settings.eval_num_workers
        eval_kwargs = dict(loader_kwargs)
        if eval_workers is not None:
            eval_kwargs = dict(num_workers=eval_workers, pin_memory=True, collate_fn=dataset_cls.collate_fn)
            if eval_workers > 0:
                eval_kwargs.update(persistent_workers=True, worker_init_fn=_seed_worker)
        self._val_loader = data.DataLoader(self._val_dataset, self._settings.batch_size, shuffle=False, **eval_kwargs)
        self._test_loader = data.DataLoader(self._test_dataset, self._settings.batch_size, shuffle=False, **eval_kwargs)

    def _init_training(self):
        """
            Initializes everything needed for training.
        """

        # scaling of Regularization to make it independent of batch_size (в режиме MC — батч в отведениях)
        self._accumulate = max(round(64 / self._rows_per_batch), 1)
        self._settings.weight_decay *= self._rows_per_batch * self._accumulate / 64
        # create optimizer
        self._optimizer = torch.optim.SGD(training.set_weight_decay(self._model, self._settings.weight_decay),
                                self._settings.min_lr, self._settings.momentum, nesterov=True)
        # if need to resume training
        if self._resume and self._checkpoint and "optimizer" in self._checkpoint.keys():
            self._optimizer.load_state_dict(self._checkpoint["optimizer"]) # load optimizer state
        if not self._resume:
            self._start_epoch = 0
        
        # Scheduler
        self._num_steps = len(self._train_loader)
        self._scheduler = training.LinearLR(self._settings.min_lr, self._settings.max_lr,
                                        self._settings.warmup_epochs, self._settings.epochs, self._num_steps)
        
        # automatic mixed precision scaling of gradients
        self._amp_scale = torch.amp.GradScaler() #if self._settings.enable_amp else None
        # loss function
        self._criterion = self._make_criterion(self._model)
        
    def _prepare_transforms(self):
        """
            Creates transforms for dataLoaders.
        """
        # ECRv1 (`format: ecr_npy` в yaml) — предобработка на лету в трансформах; остальные датасеты — как раньше
        self._train_transform, self._test_transform = build_transforms_for_dataset(
            self._settings.augmentations, self._settings.dataset, self._settings.input_subset)
        
    def _load_model_weights(self):
        # paths to resulting files
        self._best_path = os.path.join(self._project_path, 'best.pt')
        self._last_path = os.path.join(self._project_path, 'last.pt')
        
        self._checkpoint: Dict = None
        # load state of model for resume
        if self._resume and os.path.exists(self._last_path):
            self._checkpoint = torch.load(self._last_path, weights_only=False)
        # start from checkpoint
        if not self._resume and self._chkpt_path:
            self._checkpoint = torch.load(self._chkpt_path, weights_only=False)
        # load checkpoint
        if self._checkpoint is not None:
            self._model.load_state_dict(self._checkpoint['model'], strict=False)   # load model
            self._start_epoch = self._checkpoint['epoch']                  # load number of last epoch
            self._ema.load_state_dict(self._checkpoint['ema'])             # load Averaged weights of model (for testing)
             
    def _strip_optimizer(self, name):
        chkpt = torch.load(name, weights_only=False)
        if "optimizer" in chkpt.keys():
            del chkpt["optimizer"]
        torch.save(chkpt, f=name)
    
    # ------------------------------------------------------------------ выбор лучшей модели
    def _delineation_log_keys(self) -> List[str]:
        """Колонки делинеации: сводка и F1 каждой точки (имена точек — из `names`)."""
        return ["F1_mean6", "sd_mean6_ms"] + [f"F1_{c}_{e}" for c in self._settings.names for e in ("on", "off")]

    def _log_fieldnames(self) -> List[str]:
        """Колонки step.csv. Колонки делинеации добавляются только при delineation_metrics=True,
        поэтому формат логов прежних (вихретоковых) экспериментов не меняется."""
        fields = ['epoch', 'box_train', 'cls_train', 'dfl_train', 'box_val', 'cls_val', 'dfl_val',
                  'Recall', 'Precision', 'mAP@50', 'mAP']
        if self._settings.delineation_metrics:
            fields += self._delineation_log_keys()
        return fields

    def _score(self, results: Dict) -> tuple:
        """Оценка эпохи для выбора best.pt (больше — лучше) по `best_metric`."""
        metric = self._settings.best_metric
        if metric == "val_loss":
            return (-(float(results["box_val"]) + float(results["cls_val"]) + float(results["dfl_val"])),)
        if metric == "mAP":
            return (float(results.get("mAP", 0)),)
        # F1_mean6: при равенстве — меньшее sd_mean6_ms
        sd = float(results.get("sd_mean6_ms", np.nan))
        return (float(results.get("F1_mean6", 0)), -sd if np.isfinite(sd) else -np.inf)

    def _load_best(self):
        """Лучшая оценка и эпоха из step.csv при продолжении обучения."""
        if self._resume and os.path.exists(self._logger_path):
            df = pd.read_csv(self._logger_path)
            if len(df):
                scores = [self._score(row) for _, row in df.iterrows()]
                best_idx = max(range(len(scores)), key=lambda i: scores[i])
                return scores[best_idx], int(df.iloc[best_idx]["epoch"]) - 1
        return (-np.inf, -np.inf), -1

    def _load_best_map(self):
        if self._resume and os.path.exists(self._logger_path):
            df = pd.read_csv(self._logger_path)
            best_idx = df["mAP"].argmax()
            return df["mAP"].max(), df.iloc[best_idx]["epoch"]
        return 0.0, -1

    def _load_best_val_loss(self):
        if self._resume and os.path.exists(self._logger_path):
            df = pd.read_csv(self._logger_path)
            loss = df["box_val"] + df["cls_val"] + df["dfl_val"]
            best_idx = loss.argmin()
            return loss.min(), df.iloc[best_idx]["epoch"]
        return np.inf, -1

    def train(self):
        self._init_dataloaders()
        self._init_training()
        # create folder for results
        os.makedirs(self._project_path, exist_ok=True)
        # save training config
        save_settings(os.path.join(self._project_path, "args.yaml"), self._settings)

        self._adversarial = self._settings.augmentations.adversarial
        # show summary
        dummy_size = (self._settings.batch_size, self._settings.input_channels,
                      self._settings.augmentations.input_length)
        print(summary(self._model, dummy_size))

        self._logger_path = os.path.join(self._project_path, 'step.csv')
        best, best_epoch = self._load_best()
        use_cuda = self._device == "cuda"
        with open(self._logger_path, "a") as log:
            logger = csv.DictWriter(log, fieldnames=self._log_fieldnames())
            if not self._resume:
                logger.writeheader()

            for epoch in range(self._start_epoch, self._settings.epochs):
                self._epoch_results = dict()
                self._model.train()
                # adversarial-примеры выключаются на последних 10 эпохах. Mosaic не выключается (решение
                # пользователя 2026-10-17, E1): в SPEC этого нет, а при persistent_workers присваивание
                # dataset.mosaic_prob в главном процессе до воркеров всё равно не доходило
                if self._settings.epochs - epoch < 10:
                    self._adversarial = 0
                # size of progress bar
                p_bar = enumerate(self._train_loader)

                # init progress bar with header
                print(('\n' + '%10s' * 5) % ('epoch', 'memory', 'box', 'cls', 'dfl'))
                p_bar = tqdm.tqdm(p_bar, total=self._num_steps)

                self._optimizer.zero_grad()
                # averaging of loss on dataset
                avg_box_loss = training.AverageMeter()
                avg_cls_loss = training.AverageMeter()
                avg_dfl_loss = training.AverageMeter()
                for i, (samples, targets) in p_bar:

                    step = i + self._num_steps * epoch
                    self._scheduler.step(step, self._optimizer)
                    # convert training data
                    samples = samples.to(self._device).float()
                    if np.random.uniform(0,1) < self._adversarial:
                        samples = training.get_adversal_examples(self._model, self._criterion,
                                            self._amp_scale, samples, targets)

                    # Forward
                    # if self._settings.enable_amp: TODO: fix amp
                    with torch.amp.autocast('cuda', enabled=use_cuda):
                        outputs = self._model(samples)  # forward
                        loss_box, loss_cls, loss_dfl = self._criterion(outputs, targets)
                    # else:
                    #     loss_box, loss_cls, loss_dfl = self._criterion(outputs, targets)
                    # update average loss
                    avg_box_loss.update(loss_box.item(), samples.size(0))
                    avg_cls_loss.update(loss_cls.item(), samples.size(0))
                    avg_dfl_loss.update(loss_dfl.item(), samples.size(0))
                    # scale loss by batch_size (в режиме MC — батч в отведениях)
                    loss_box *= self._rows_per_batch
                    loss_cls *= self._rows_per_batch
                    loss_dfl *= self._rows_per_batch

                    # Backward
                    self._amp_scale.scale(loss_box + loss_cls + loss_dfl).backward()

                    # Optimize
                    if step % self._accumulate == 0:
                        self._amp_scale.unscale_(self._optimizer)  # unscale gradients
                        training.grad_clip(self._model)  # clip gradients
                        self._amp_scale.step(self._optimizer)
                        self._amp_scale.update()
                        self._optimizer.zero_grad()
                        if self._ema:
                            self._ema.update(self._model)

                    if use_cuda:
                        torch.cuda.synchronize()

                    # Logging
                    memory = f'{torch.cuda.memory_reserved() / 1E9:.4g}G' if use_cuda else 'cpu'  # (GB)
                    s = ('%10s' * 2 + '%10.3g' * 3) % (f'{epoch + 1}/{self._settings.epochs}', memory,
                                                    avg_box_loss.avg, avg_cls_loss.avg, avg_dfl_loss.avg)
                    p_bar.set_description(s)

                self._epoch_results["box_train"] = avg_box_loss.avg
                self._epoch_results["cls_train"] = avg_cls_loss.avg
                self._epoch_results["dfl_train"] = avg_dfl_loss.avg
                # validation
                self.validate()
                # logging results
                logger.writerow({
                        'epoch': str(epoch + 1).zfill(3),
                        **{k: f'{v:.3f}' for k, v in self._epoch_results.items()}
                    })
                log.flush()

                # create checkpoint
                save =  {
                            'epoch': epoch + 1,
                            'model': self._model.state_dict(),
                            'ema': self._ema.state_dict(),
                            'optimizer': self._optimizer.state_dict()
                        }
                # save last model
                torch.save(save, f=self._last_path)

                # Update best metric (best_metric) and save best model
                score = self._score(self._epoch_results)
                if score > best:
                    best = score
                    best_epoch = epoch
                    shutil.copy(self._last_path, self._best_path)
                # delete checkpoint
                del save
                if (self._settings.early_stopping_epochs > 0) and (epoch - best_epoch > self._settings.early_stopping_epochs):
                    print(f"Early stopping at epoch {epoch}.")
                    break
        # delete optimizer from checkpoints
        self._strip_optimizer(self._last_path)
        self._strip_optimizer(self._best_path)

    # ------------------------------------------------------------------ оценка
    def _signal_length(self):
        """Длина исходного сигнала для пересчёта координат: без ресайза (ЭКГ) — regular_length."""
        aug = self._settings.augmentations
        return None if aug.validation_resize else aug.regular_length

    def _evaluate_loader(self, loader, criterion=None, conf_threshold=0.001, desc=''):
        """Прогон загрузчика через EMA-модель.

        Returns
        -------
        results
            mAP@50, mAP, Precision, Recall (как раньше) и средние лоссы, если передан criterion.
        per_class
            (Precision, Recall, AP50, AP) по классам для печати.
        samples
            Список (детекции NMS (n, 4) [x1, x2, conf, cls] в отсчётах входа, cls (m, 1), box (m, 2))
            в порядке `loader.dataset.filenames` (загрузчики val/test не перемешиваются); в режиме MC — по
            пригодным отведениям в порядке `MultiLeadDataset.lead_ids()`, координаты — в отсчётах отведения.
        """
        device = self._device
        model = self._ema.ema
        model.eval()
        model.set_validate(criterion is not None)
        iou_v = torch.linspace(start=0.5, end=0.95, steps=10, device=device)  # iou vector for mAP@0.5:0.95
        n_iou = iou_v.numel()

        metrics, per_sample = [], []
        p_bar = tqdm.tqdm(loader, desc=('%10s' * 5) % (desc, 'precision', 'recall', 'mAP50', 'mAP'))
        # Sum of losses on validation
        avg_box_loss = training.AverageMeter()
        avg_cls_loss = training.AverageMeter()
        avg_dfl_loss = training.AverageMeter()
        for samples, targets in p_bar:
            samples = samples.to(device).float()
            w = samples.shape[2]  # length of signal
            # режим MC: строки выхода — отведения (b·n_leads + k); цели перенумеровываются так же,
            # непригодные отведения (valid = False) пропускаются в метриках
            row_valid = None
            if self._n_leads > 1:
                criterion_targets = targets
                targets, row_valid = flatten_lead_targets(targets, self._n_leads)
            scale = torch.tensor((w, w), device=device)
            # Inference
            if criterion is not None:
                raw, outputs = model(samples)
                # Calculate loss
                loss_box, loss_cls, loss_dfl = criterion(raw, criterion_targets if row_valid is not None else targets)
                # update average losses
                avg_box_loss.update(loss_box.item(), samples.size(0))
                avg_cls_loss.update(loss_cls.item(), samples.size(0))
                avg_dfl_loss.update(loss_dfl.item(), samples.size(0))
            else:
                outputs = model(samples)
            # NMS
            outputs = detection.non_max_suppression(outputs, conf_threshold, self._settings.nms_iou)
            # постобработка: вложенные вторичные объекты (выключена при пустом contain_main_classes)
            if self._settings.contain_main_classes:
                outputs = [detection.suppress_contained(o, self._settings.contain_main_classes,
                                                   self._settings.contain_secondary_classes,
                                                   self._settings.contain_threshold) for o in outputs]
            # Metrics
            for i, output in enumerate(outputs):
                if row_valid is not None and not row_valid[i]:
                    continue
                idx = targets['idx'] == i
                cls = targets['cls'][idx].to(device)
                box = targets['box'][idx].to(device)
                per_sample.append((output.cpu().numpy(), cls.cpu().numpy(), box.cpu().numpy()))

                metric = torch.zeros(output.shape[0], n_iou, dtype=torch.bool, device=device)

                if output.shape[0] == 0:
                    if cls.shape[0]:
                        metrics.append((metric, *torch.zeros((2, 0), device=device), cls.squeeze(-1)))
                    continue
                # Evaluate
                if cls.shape[0]:
                    target = torch.cat(tensors=(cls, detection.xw_to_x1x2(box) * scale), dim=1)
                    metric = detection.compute_metric(output[:, :4], target, iou_v)
                # Append
                metrics.append((metric, output[:, 2], output[:, 3], cls.squeeze(-1)))
        model.set_validate(False)
        # Compute metrics
        metrics = [torch.cat(x, dim=0).cpu().numpy() for x in zip(*metrics)]  # to numpy
        r = dict()
        if len(metrics) and metrics[0].any():
            # get the class-wise Precision and Recall (compute_ap returns precision, recall, ...)
            pre, rec, AP50, AP = detection.compute_ap(*metrics)
            r["Precision"], r["Recall"], r["mAP@50"], r["mAP"] = pre.mean(), rec.mean(), AP50.mean(), AP.mean()
        else:
            pre = np.zeros(self._num_classes)
            rec = np.zeros(self._num_classes)
            AP50 = np.zeros(self._num_classes)
            AP = np.zeros(self._num_classes)
        if criterion is not None:
            r["box_val"] = avg_box_loss.avg
            r["cls_val"] = avg_cls_loss.avg
            r["dfl_val"] = avg_dfl_loss.avg
        return r, (pre, rec, AP50, AP), per_sample

    def _print_detection_results(self, r, per_class):
        print(('%10s' + '%10.3g' * 4) % ('', r.get("Precision",0), r.get("Recall",0), r.get("mAP@50",0), r.get("mAP",0)))
        for name, prec, recl, ap50, ap in zip(self._settings.names, *per_class):
            print(('%10s' + '%10.3g' * 4) % (name, prec, recl, ap50, ap))

    def _to_intervals(self, per_sample, input_length):
        """Детекции и разметку примеров → (pred (n, 4) [class, left, right, conf], ref (m, 3)) в отсчётах сигнала."""
        length = self._signal_length()
        out = []
        for det, cls, box in per_sample:
            pred = detections_to_intervals(det, signal_length=length)
            ref = boxes_to_intervals(cls, box, input_length, signal_length=length)
            out.append((pred, ref))
        return out

    @torch.no_grad()
    def validate(self):
        self._init_dataloaders()
        criterion = self._make_criterion(self._ema.ema)
        r, per_class, per_sample = self._evaluate_loader(self._val_loader, criterion)
        # Print results
        self._print_detection_results(r, per_class)
        if self._settings.delineation_metrics:
            metrics = DelineationMetrics(class_names=self._settings.names,
                                         exclude_boundary_points=self._settings.exclude_boundary_points)
            for pred, ref in self._to_intervals(per_sample, self._settings.augmentations.input_length):
                metrics.add(pred[pred[:, 3] >= self._settings.delineation_conf], ref)
            result = metrics.compute()
            r["F1_mean6"] = result["summary"]["F1_mean6"]
            r["sd_mean6_ms"] = result["summary"]["sd_mean6_ms"]
            for f, p in result["points"].items():
                r[f"F1_{f}"] = p["F1"]
            print(f"   delineation: F1_mean6 = {r['F1_mean6']:.4f}, sd_mean6 = {r['sd_mean6_ms']:.1f} ms "
                  f"(conf >= {self._settings.delineation_conf})")
        # Return results
        self._epoch_results.update(r)
        return r

    @torch.no_grad()
    def test(self):
        self._init_dataloaders()
        r, per_class, _ = self._evaluate_loader(self._test_loader)
        # Print results
        self._print_detection_results(r, per_class)
        return r

    def load_checkpoint(self, path: str):
        """Загружает веса модели и EMA из чекпоинта (например, best.pt после обучения)."""
        checkpoint = torch.load(path, weights_only=False, map_location=self._device)
        self._model.load_state_dict(checkpoint['model'])
        self._ema.load_state_dict(checkpoint['ema'])
        self._ema.ema.to(self._device)

    @torch.no_grad()
    def collect_predictions(self, split: str):
        """Детекции EMA-модели на сплите ('val' или 'test') с низким порогом уверенности.

        Returns
        -------
        list of dict(id, pred, ref), detection_results
            pred — (n, 4) [class, left, right, conf], ref — (m, 3) [class, onset, offset], в отсчётах сигнала;
            id — имя файла примера без расширения; detection_results — mAP/P/R сплита.
        """
        self._init_dataloaders()
        loader = {"val": self._val_loader, "test": self._test_loader}[split]
        r, per_class, per_sample = self._evaluate_loader(loader, desc=split)
        if isinstance(loader.dataset, MultiLeadDataset):
            ids = loader.dataset.lead_ids()  # <запись>_<отведение> пригодных отведений, в порядке строк выхода
        else:
            ids = [os.path.basename(f).rsplit(".", 1)[0] for f in loader.dataset.filenames]
        pairs = self._to_intervals(per_sample, self._settings.augmentations.input_length)
        return [{"id": i, "pred": p, "ref": t} for i, (p, t) in zip(ids, pairs)], r

    def evaluate_delineation(self, groups: Dict[str, str] = None, save_dir: str = None) -> Dict:
        """Итоговая оценка делинеации (R2 §3.1–3.2).

        1. Пороги уверенности по классам подбираются на val (максимум F1 класса).
        2. На test детекции фильтруются этими порогами, метрики считаются по всем отведениям
           и по группам `groups` (id примера → группа, например группа ритма).
        3. В save_dir пишутся thresholds.json, predictions_{val,test}.json, metrics.json, metrics.md.
        """
        groups = groups or {}
        val, _ = self.collect_predictions("val")
        exclude = self._settings.exclude_boundary_points
        thresholds = best_class_thresholds([(s["pred"], s["ref"]) for s in val], class_names=self._settings.names,
                                           exclude_boundary_points=exclude)
        test, detection = self.collect_predictions("test")
        metrics = DelineationMetrics(class_names=self._settings.names, exclude_boundary_points=exclude)
        # вариант без исключения граничных точек — для прозрачности (D5)
        metrics_all = DelineationMetrics(class_names=self._settings.names, exclude_boundary_points=False)
        for s in test:
            pred = apply_thresholds(s["pred"], thresholds, self._settings.names)
            metrics.add(pred, s["ref"], group=groups.get(s["id"]))
            metrics_all.add(pred, s["ref"])
        result = {
            "thresholds": {k: {"conf": v[0], "val_F1": v[1]} for k, v in thresholds.items()},
            "exclude_boundary_points": exclude,
            "test": metrics.compute(),
            "test_with_boundary_points": metrics_all.compute(),
            "test_by_group": {g: metrics.compute(g) for g in metrics.groups},
            "test_detection": {k: float(v) for k, v in detection.items()},
            "n_test": len(test), "n_val": len(val),
        }
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            # сохраняются детекции с уверенностью не ниже минимального порога сетки best_class_thresholds (0.05):
            # пересчёт метрик по сохранённым предсказаниям (пулинг фолдов) даёт тот же результат
            def dump(items):
                out = []
                for s in items:
                    pred = s["pred"][s["pred"][:, 3] >= 0.05].copy()
                    pred[:, :3] = np.round(pred[:, :3], 3)  # уверенность не округляется: она сравнивается с порогами
                    out.append({"id": s["id"], "pred": pred.tolist(), "ref": np.round(s["ref"], 3).tolist()})
                return out
            for name, items in (("val", val), ("test", test)):
                with open(os.path.join(save_dir, f"predictions_{name}.json"), "w", encoding="utf-8") as f:
                    json.dump(dump(items), f)
            with open(os.path.join(save_dir, "thresholds.json"), "w", encoding="utf-8") as f:
                json.dump(result["thresholds"], f, indent=2)
            with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=float)
            lines = ["# Test: delineation", "", format_table(result["test"])]
            for g, res in result["test_by_group"].items():
                lines += ["", f"## Group: {g}", "", format_table(res)]
            with open(os.path.join(save_dir, "metrics.md"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        print(format_table(result["test"]))
        return result

    @torch.no_grad()
    def predict(self, data: torch.Tensor, conf: float = 0.1) -> torch.Tensor:
        """Детекции для сигнала (channels, length) или батча; в режиме MC — список по строкам
        «батча отведений» (b·n_leads + k — отведение k примера b)."""
        # create empty boxes and labels to pass into transform
        bboxes, labels = torch.zeros((0,2), device=data.device, dtype=torch.float), torch.zeros((0), device=data.device, dtype=torch.float)
        # convert data to float
        data = data.float()
        # apply transform
        data, _, _ = self._test_transform(data, bboxes, labels)
        # add batch
        if data.ndim == 2:
            data = data.unsqueeze(0)
        data = data.to(self._device)
        # get predictions
        outputs = self._ema.ema(data)
        outputs = detection.non_max_suppression(outputs, conf)
        aug = self._settings.augmentations
        if aug.validation_resize:
            # вход был дополнен до regular_length и сжат до input_length — возвращаем масштаб
            for out in outputs:
                out[:, :2] = out[:, :2] / aug.input_length * aug.regular_length
        # без ресайза (ЭКГ) паддинг добавлен справа: координаты уже в отсчётах исходного сигнала
        return outputs
    
    def export_onnx(self, filename, opset=17):
        from yolo_1d.nets.export import ExportWrapper
        aug = self._settings.augmentations
        example_input = (torch.randn((1, self._settings.input_channels, aug.regular_length), device=self._device),)
        self._ema.ema.eval()
        wrapper = ExportWrapper(self._ema.ema, input_length=aug.input_length, resize=aug.validation_resize)
        torch.onnx.export(wrapper, example_input,
            f=filename,   
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=opset)
        

