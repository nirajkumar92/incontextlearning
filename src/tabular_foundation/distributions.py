"""Differentiable histogram/Lomax distributions and a genuine zero-atom head.

All likelihoods use original target units.  The positive hurdle component is
the base density conditioned on Y>0, not a continuous approximation to zero.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor
from torch.nn import functional as F


def _working(x: Tensor) -> Tensor:
    return x if x.dtype == torch.float64 else x.float()


def _log_positive(x: Tensor) -> Tensor:
    tiny = torch.finfo(x.dtype).tiny
    return torch.where(x > 0, x.clamp_min(tiny).log(), x.new_full((), -torch.inf))


def _log1p_ratio(nonnegative: Tensor, positive: Tensor) -> Tensor:
    """log(1+a/b) without overflowing a/b or cancelling tiny a/b."""
    small = torch.log1p(torch.minimum(nonnegative, positive) / positive)
    large_value = torch.maximum(nonnegative, positive)
    large = large_value.log() - positive.log() + torch.log1p(positive / large_value)
    return torch.where(nonnegative <= positive, small, large)


class ContinuousDistribution:
    """Configurable uniform bins in [-5,5], plus two shape-three Lomax tails.

    The default 128-bin geometry remains checkpoint compatible. A 1,025-bin
    grid places zero at a bin midpoint, but still has within-bin NLL blindness.
    """

    def __init__(self, params: Tensor, center=0.0, scale=1.0, bins=128):
        if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
            raise ValueError("bins must be a positive integer")
        if params.shape[-1] != bins + 4:
            raise ValueError(f"continuous head requires {bins + 4} outputs for {bins} bins")
        self.bins = bins
        self.inverse_width = bins / 10.
        self.params = _working(params)
        # Original-unit values can have a large offset and a small spread.
        # Preserve the affine transform independently of neural-head precision.
        self.center = torch.as_tensor(center, device=params.device, dtype=torch.float64)
        self.scale = torch.as_tensor(scale, device=params.device, dtype=torch.float64)
        if not torch.isfinite(self.center).all():
            raise ValueError("target center must be finite")
        if not torch.isfinite(self.scale).all() or not (self.scale > 0).all():
            raise ValueError("target scale must be finite and positive")
        self.log_weights = F.log_softmax(self.params[..., :bins + 2], dim=-1)
        self.tail_scales = .05 + F.softplus(self.params[..., bins + 2:])

    def _z(self, y) -> Tensor:
        values = torch.as_tensor(y, device=self.params.device, dtype=torch.float64)
        delta = values - self.center
        normalized = delta / self.scale
        # Opposite-sign finite values can overflow subtraction even when the
        # standardized difference is representable.
        normalized = torch.where(torch.isinf(delta) & torch.isfinite(values),
                                 values / self.scale - self.center / self.scale, normalized)
        return normalized.to(self.params.dtype)

    def _original_units(self, z) -> Tensor:
        # Explicitly promote z: a scalar FP64 tensor alone does not necessarily
        # promote a non-scalar FP32 tensor under PyTorch's scalar dtype rules.
        return self.center + self.scale * z.double()

    def log_prob(self, y) -> Tensor:
        z = self._z(y)
        z, _ = torch.broadcast_tensors(z, self.params[..., 0])
        index = ((z + 5) * self.inverse_width).floor().long().clamp(0, self.bins - 1) + 1
        index = torch.where(z < -5, 0, torch.where(z >= 5, self.bins + 1, index))
        weights = self.log_weights.expand(z.shape + (self.bins + 2,))
        log_mass = weights.gather(-1, index.unsqueeze(-1)).squeeze(-1)
        sl, sr = self.tail_scales.unbind(-1)
        left = math.log(3) - sl.log() - 4 * _log1p_ratio((-5 - z).clamp_min(0), sl)
        right = math.log(3) - sr.log() - 4 * _log1p_ratio((z - 5).clamp_min(0), sr)
        log_density = torch.where(z < -5, left, torch.where(z >= 5, right, z.new_full((), math.log(self.inverse_width))))
        result = log_mass + log_density - self.scale.log()
        return torch.where(torch.isnan(z), z, result)

    def _component_log_survival(self, z) -> Tensor:
        z = torch.as_tensor(z, device=self.params.device, dtype=self.params.dtype)
        z, _ = torch.broadcast_tensors(z, self.params[..., 0])
        sl, sr = self.tail_scales.unbind(-1)
        distance_left = (-5 - z).clamp_min(0)
        left_mass = -torch.expm1(-3 * _log1p_ratio(distance_left, sl))
        left = _log_positive(left_mass)
        lower = torch.arange(self.bins, dtype=z.dtype, device=z.device) / self.inverse_width - 5
        upper = lower + 1 / self.inverse_width
        finite_mass = ((upper - z.unsqueeze(-1)) * self.inverse_width).clamp(0, 1)
        finite = _log_positive(finite_mass)
        right = -3 * _log1p_ratio((z - 5).clamp_min(0), sr)
        return torch.cat((left.unsqueeze(-1), finite, right.unsqueeze(-1)), dim=-1)

    def log_survival(self, y) -> Tensor:
        return torch.logsumexp(self.log_weights + self._component_log_survival(self._z(y)), dim=-1)

    def cdf(self, y) -> Tensor:
        return -torch.expm1(self.log_survival(y).clamp_max(0))

    @property
    def mean(self) -> Tensor:
        finite = (torch.arange(self.bins, dtype=self.params.dtype, device=self.params.device) - (self.bins - 1) / 2) / self.inverse_width
        finite = finite.expand(self.params.shape[:-1] + (self.bins,))
        component_means = torch.cat(((-5 - self.tail_scales[..., :1] / 2), finite,
                                     (5 + self.tail_scales[..., 1:] / 2)), dim=-1)
        return self._original_units((self.log_weights.exp() * component_means).sum(-1))

    def mean_above_zero(self) -> Tensor:
        """E[Y|Y>0], using nonnegative component moments without cancellation."""
        t = -self.center / self.scale
        t, _ = torch.broadcast_tensors(t, self.params[..., 0])
        log_component = self.log_weights + self._component_log_survival(t)
        conditional_weights = F.softmax(log_component, dim=-1).double()
        # Keep a large original-unit offset from magnifying FP32 probability
        # summation error in the positive-truncated moment.
        conditional_weights = conditional_weights / conditional_weights.sum(-1, keepdim=True)
        sl, sr = self.tail_scales.unbind(-1)
        v = (-5 - t).clamp_min(0)
        inverse_one_plus_u = torch.exp(-_log1p_ratio(v, sl))
        # Exact rational expression; unlike subtracting tail moments, stable at v=0.
        left_mean = v * (1 + .5 * inverse_one_plus_u) / (1 + inverse_one_plus_u + inverse_one_plus_u.square())
        lower = torch.arange(self.bins, dtype=t.dtype, device=t.device) / self.inverse_width - 5
        upper = lower + 1 / self.inverse_width
        finite_mean = ((torch.maximum(lower, t.unsqueeze(-1)) - t.unsqueeze(-1)) +
                       (upper - t.unsqueeze(-1))) / 2
        finite_mean = finite_mean.clamp_min(0)
        right_mean = (5 - t).clamp_min(0) + (sr + (t - 5).clamp_min(0)) / 2
        means = torch.cat((left_mean.unsqueeze(-1), finite_mean, right_mean.unsqueeze(-1)), -1)
        return self.scale * (conditional_weights * means).sum(-1)

    def quantile(self, probability) -> Tensor:
        u = torch.as_tensor(probability, device=self.params.device, dtype=self.params.dtype)
        u, _ = torch.broadcast_tensors(u, self.params[..., 0])
        if not ((u >= 0) & (u <= 1)).all():
            raise ValueError("quantiles require probabilities in [0,1]")
        weights = self.log_weights.exp().expand(u.shape + (self.bins + 2,))
        cumulative = weights.cumsum(-1)
        index = (u.unsqueeze(-1) >= cumulative).sum(-1).clamp_max(self.bins + 1)
        lower_cdf = torch.cat((torch.zeros_like(cumulative[..., :1]), cumulative[..., :-1]), -1)
        prev = lower_cdf.gather(-1, index.unsqueeze(-1)).squeeze(-1)
        mass = weights.gather(-1, index.unsqueeze(-1)).squeeze(-1)
        fraction = ((u - prev) / mass.clamp_min(torch.finfo(mass.dtype).tiny)).clamp(0, 1)
        finite = -5 + (index - 1).to(u.dtype) / self.inverse_width + fraction / self.inverse_width
        left = -5 - self.tail_scales[..., 0] * (fraction.pow(-1 / 3) - 1)
        right = 5 + self.tail_scales[..., 1] * ((1 - fraction).pow(-1 / 3) - 1)
        z = torch.where(index == 0, left, torch.where(index == self.bins + 1, right, finite))
        z = torch.where(u == 0, z.new_full((), -torch.inf), torch.where(u == 1, z.new_full((), torch.inf), z))
        return self._original_units(z)


class HurdleDistribution:
    """p0 delta_0 + (1-p0) times a positive-truncated continuous distribution."""

    def __init__(self, base: ContinuousDistribution, atom_logit: Tensor):
        self.base = base
        self.atom_logit = _working(atom_logit).squeeze(-1) if atom_logit.ndim == base.params.ndim else _working(atom_logit)
        self.log_zero = -F.softplus(-self.atom_logit)
        self.log_positive = -F.softplus(self.atom_logit)
        self.log_normalizer = base.log_survival(0.)

    def log_prob(self, y) -> Tensor:
        y = torch.as_tensor(y, device=self.base.params.device, dtype=torch.float64)
        positive = self.log_positive + self.base.log_prob(y) - self.log_normalizer
        result = torch.where(y == 0, self.log_zero, torch.where(y > 0, positive, positive.new_full((), -torch.inf)))
        return torch.where(torch.isnan(y), y, result)

    @property
    def mean(self) -> Tensor:
        return self.log_positive.exp() * self.base.mean_above_zero()

    def cdf(self, y) -> Tensor:
        y = torch.as_tensor(y, device=self.base.params.device, dtype=torch.float64)
        log_s = self.log_positive + self.base.log_survival(y) - self.log_normalizer
        return torch.where(y < 0, torch.zeros_like(log_s), -torch.expm1(log_s.clamp_max(0)))


def regression_nll(output: dict, target: Tensor) -> Tensor:
    """Unreduced original-unit negative log likelihood for trainer-owned weighting."""
    if output.get("loss_kind") == "pinball":
        raise ValueError("quantile forecasts have a pinball score, not a density NLL; use regression_loss")
    return -output["distribution"].log_prob(target)


def regression_loss(output: dict, target: Tensor) -> Tensor:
    """Unreduced per-row head score; query/proposal weighting belongs to the trainer.

    Quantile training uses the mean pinball loss in support-standardized units,
    on raw (unsorted) outputs. Averaging avoids a 999-fold task multiplier. It
    is a different proper score from density NLL, with a different gradient
    scale. Finite quantiles do not define the extreme tails or a trained density.
    """
    if output.get("loss_kind", "nll") != "pinball":
        return regression_nll(output, target)
    params = output["regression_params"]
    target = torch.as_tensor(target, device=params.device, dtype=torch.float64)
    center = torch.as_tensor(output["target_center"], device=params.device, dtype=torch.float64)
    scale = torch.as_tensor(output["target_scale"], device=params.device, dtype=torch.float64)
    delta = target - center
    z = delta / scale
    z = torch.where(torch.isinf(delta) & torch.isfinite(target), target / scale - center / scale, z)
    residual = z.unsqueeze(-1) - params.double()
    levels = output["quantile_levels"].double()
    return torch.maximum(levels * residual, (levels - 1) * residual).mean(-1)
