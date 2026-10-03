"""Exact separately registered hidden-matrix Muon plus AdamW elsewhere.

The trainer owns global gradient synchronization, nonfinite-update rejection,
global clipping and schedule advancement. No matrix is orthogonalized per shard.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.optim import Optimizer

from .model import MAB


def newton_schulz(matrix, steps=5, eps=1e-7):
    if matrix.ndim != 2:
        raise ValueError("Muon requires original two-dimensional matrices")
    x = matrix.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (torch.linalg.vector_norm(x) + eps)
    a, b, c = 3.4445, -4.775, 2.0315
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x.T if transposed else x


class Muon(Optimizer):
    def __init__(self, params, lr=8e-4, momentum=.95, weight_decay=.01, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps))
        for group in self.param_groups:
            group.setdefault("initial_lr", group["lr"])
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError("Muon may only receive registered hidden matrices")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta, lr = group["momentum"], group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue  # Globally unused means no state advance and no decay.
                if p.grad.is_sparse:
                    raise ValueError("Muon does not support sparse gradients")
                gradient = p.grad.float()
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
                buffer = state["momentum_buffer"]
                buffer.mul_(beta).add_(gradient)
                direction = gradient + beta * buffer
                update = newton_schulz(direction, group["ns_steps"])
                p.mul_(1 - lr * group["weight_decay"])
                effective_lr = .2 * lr * math.sqrt(max(p.shape))
                p.add_(update.to(p.dtype), alpha=-effective_lr)
        return loss


def parameter_partition(model):
    """Return disjoint parameter lists plus auditable names, with no size heuristic."""
    hidden_ids = {id(p) for module in model.modules() if isinstance(module, MAB) for p in module.hidden_parameters()}
    linear_ids = {id(module.weight) for module in model.modules() if isinstance(module, nn.Linear)}
    groups = {"muon": [], "adamw_decay": [], "adamw_no_decay": []}
    names = {key: [] for key in groups}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in hidden_ids:
            key = "muon"
        elif id(parameter) in linear_ids:
            key = "adamw_decay"
        else:
            key = "adamw_no_decay"
        groups[key].append(parameter)
        names[key].append(name)
    combined = [p for values in groups.values() for p in values]
    expected = [p for p in model.parameters() if p.requires_grad]
    if len(combined) != len({id(p) for p in combined}) or {id(p) for p in combined} != {id(p) for p in expected}:
        raise AssertionError("optimizer partition is incomplete or overlapping")
    return groups, names


def build_optimizers(model, mode="muon", adam_lr=3e-4, muon_lr=8e-4, weight_decay=.01):
    if mode not in ("muon", "adamw"):
        raise ValueError("optimizer mode must be muon or adamw")
    groups, _ = parameter_partition(model)
    optimizers = []
    decay = list(groups["adamw_decay"])
    if mode == "muon":
        optimizers.append(Muon(groups["muon"], lr=muon_lr, weight_decay=weight_decay))
    else:
        decay += groups["muon"]
    adam_groups = [dict(params=decay, lr=adam_lr, initial_lr=adam_lr, weight_decay=weight_decay),
                   dict(params=groups["adamw_no_decay"], lr=adam_lr, initial_lr=adam_lr, weight_decay=0.)]
    optimizers.append(torch.optim.AdamW(adam_groups, betas=(.9, .95), eps=1e-8))
    return optimizers


def schedule_multiplier(update, total_updates):
    """One-based update schedule, fixed horizon, two-percent warmup then cosine."""
    if total_updates < 1 or not 1 <= update <= total_updates:
        raise ValueError("update must be in 1..total_updates")
    warmup = max(1, math.ceil(.02 * total_updates))
    if update <= warmup:
        return update / warmup
    progress = (update - warmup) / max(1, total_updates - warmup)
    return .1 + .9 * .5 * (1 + math.cos(math.pi * progress))


def set_learning_rates(optimizers, update, total_updates):
    multiplier = schedule_multiplier(update, total_updates)
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * multiplier
