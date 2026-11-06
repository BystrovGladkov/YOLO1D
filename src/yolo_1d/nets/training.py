from typing import Any, List
import copy
import math
import random

import numpy as np
import torch


def set_seed(seed: int = 0):
    """
    Setup random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def grad_clip(model, max_norm=10.0):
    parameters = model.parameters()
    torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)


def set_weight_decay(model: Any, decay: float) -> List[dict]:
    """Split trainable parameters into zero-decay and decayed optimizer groups.

    Biases and BatchNorm parameters receive no weight decay. Other trainable
    parameters use the requested ``decay`` value.
    """
    no_decay = []
    with_decay = []
    assigned = set()
    batch_norm = torch.nn.modules.batchnorm._BatchNorm

    for module in model.modules():
        for name, parameter in module.named_parameters(recurse=False):
            parameter_id = id(parameter)
            if not parameter.requires_grad or parameter_id in assigned:
                continue
            assigned.add(parameter_id)

            if name == "bias" or isinstance(module, batch_norm):
                no_decay.append(parameter)
            else:
                with_decay.append(parameter)

    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": with_decay, "weight_decay": decay},
    ]


class LinearLR:
    def __init__(
        self,
        min_lr: float,
        max_lr: float,
        warmup_epochs: float,
        epochs: int,
        num_steps: int,
    ):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_steps = int(epochs * num_steps)
        if self.total_steps <= 0:
            raise ValueError("epochs * num_steps must produce at least one training step")

        self.warmup_steps = min(
            max(int(warmup_epochs * num_steps), 100),
            self.total_steps,
        )
        self.decay_steps = self.total_steps - self.warmup_steps

    def _learning_rate_at(self, step: int) -> float:
        if step < self.warmup_steps:
            # Warmup approaches max_lr without including it.
            fraction = step / self.warmup_steps
            return self.min_lr + (self.max_lr - self.min_lr) * fraction

        if self.decay_steps <= 1:
            return self.max_lr

        # Decay includes both endpoints, matching a linear max-to-min schedule.
        fraction = (step - self.warmup_steps) / (self.decay_steps - 1)
        return self.max_lr + (self.min_lr - self.max_lr) * fraction

    def step(self, step: int, optimizer: Any) -> None:
        learning_rate = self._learning_rate_at(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate


class EMA:
    """Maintain a warm-started exponential moving average of model state."""

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.9999,
        tau: int = 2000,
        updates: int = 0,
    ):
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}")
        if updates < 0:
            raise ValueError(f"updates must be non-negative, got {updates}")

        self.ema = copy.deepcopy(model).eval()
        self.decay = float(decay)
        self.tau = int(tau)
        self.updates = int(updates)
        for parameter in self.ema.parameters():
            parameter.requires_grad_(False)

    def _decay_at(self, update: int) -> float:
        """Ramp the averaging coefficient from near zero to its configured value."""
        return self.decay * (1.0 - math.exp(-update / self.tau))

    @torch.no_grad()
    def update(self, model: Any) -> None:
        if hasattr(model, "module"):
            model = model.module

        source_state = model.state_dict()
        ema_state = self.ema.state_dict()
        if source_state.keys() != ema_state.keys():
            raise ValueError("EMA and source models must have identical state keys")
        for name, ema_value in ema_state.items():
            if source_state[name].shape != ema_value.shape:
                raise ValueError(f"State shape mismatch for {name!r}")

        self.updates += 1
        decay = self._decay_at(self.updates)
        for name, ema_value in ema_state.items():
            source_value = source_state[name].detach()
            source_value = source_value.to(device=ema_value.device, dtype=ema_value.dtype)
            if ema_value.dtype.is_floating_point:
                # lerp_(source, weight) applies ema += (source - ema) * weight.
                ema_value.lerp_(source_value, 1.0 - decay)
            else:
                # Integer buffers (for example BatchNorm counters) are copied, not averaged.
                ema_value.copy_(source_value)

    def state_dict(self) -> dict:
        return {
            "updates": self.updates,
            "decay": self.decay,
            "tau": self.tau,
            "ema_model_state_dict": self.ema.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        decay = float(state_dict["decay"])
        tau = int(state_dict["tau"])
        updates = int(state_dict["updates"])
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}")
        if updates < 0:
            raise ValueError(f"updates must be non-negative, got {updates}")

        self.ema.load_state_dict(state_dict["ema_model_state_dict"])
        self.decay = decay
        self.tau = tau
        self.updates = updates


class AverageMeter:
    def __init__(self):
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(self, v, n):
        if not math.isnan(float(v)):
            self.num = self.num + n
            self.sum = self.sum + v * n
            self.avg = self.sum / self.num


def get_adversal_examples(model:torch.nn.Module, criterion, grad_scaler:torch.amp.GradScaler,
                          images: torch.Tensor, targets: torch.Tensor):
    
    # fraction of signals power to change
    abs_scale = 0.3
    power_scale = 0.1
    # add gradient to images
    images = images.clone().detach().requires_grad_(True)
    
    # turn off gradients
    prev_requires = []
    for p in model.parameters():
        prev_requires.append(p.requires_grad)
        p.requires_grad_(False)
    # compute loss
    with torch.amp.autocast('cuda'):
        outputs = model(images)
        loss_box, loss_cls, loss_dfl = criterion(outputs, targets)
    
    loss = (loss_box + loss_cls + loss_dfl)
    # calculate gradient
    grad_x = torch.autograd.grad(grad_scaler.scale(loss), images, retain_graph=False, create_graph=False)[0]
    # calculate powers of signal and gradient
    power = images.square().sum(dim=-1) / images.size(-1)
    max_abs = images.abs().max(dim=-1).values
    grad_power = grad_x.square().sum(dim=-1) / grad_x.size(-1)
    grad_max_abs = grad_x.abs().max(dim=-1).values
    # calculate a scale
    scale = torch.min(power_scale * power / grad_power, abs_scale * max_abs / grad_max_abs)
    # add gradient to images
    # images = images + grad_x.sign() * images * scale # * scale[..., None]
    images = images + grad_x * scale[..., None]
    
    # turn on_gradients
    for p, r in zip(model.parameters(), prev_requires):
        p.requires_grad_(r)
        
    return images.detach()
