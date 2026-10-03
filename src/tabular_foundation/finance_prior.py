"""Finite, lazy finance worlds with revealed-label support and population queries.

Default histories really contain 300k--3m finite rows. Counts and identities are
created first; requested feature rows are regenerated with counter-style keys.
Selecting the full-index branch really visits every eligible candidate. Tiny
overrides are explicit, recorded in audit metadata, and are not recipe defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Iterator

import numpy as np

from .schema import Episode
from .retrieval import (ReservoirCodec, assign_routes, farthest_first_centers,
                        mix64, stable_keys, stream_nearest_to_centers, union_context)
from .static_prior import _Column, _column, _Renderer


_DAYS = np.r_[np.arange(-90, 0), np.arange(1, 31)]
_STAGE_CAPS = {1: (512, 256, 1024), 2: (1024, 512, 2048), 3: (2048, 1024, 4096)}
_WIDTHS = {1: ([16, 32, 64, 128], [.45, .25, .15, .15]),
           2: ([64, 128, 256, 512], [.25, .35, .25, .15]),
           3: ([128, 256, 512], [.25, .375, .375])}
_ALLOWED_OVERRIDES = {
    "history_size", "future_size", "width", "prevalence", "mode_count",
    "complete_adjudication", "maturity_days", "audit_probability",
    "review_probability_h0", "review_probability_h1", "self_report_probability",
    "negative_h_probability", "missingness", "missing_rate", "risk_campaigns",
    "arrival_burst", "future_drift_multiplier", "reservoir_cap", "positive_cap",
    "local_cap", "candidate_cap", "max_centers", "max_positive_centers",
    "query_count", "index_dimension", "scan_chunk_size", "regression_kind",
    "separation", "indistinguishable_overlap", "signal_sigma", "calibration_rows",
    "feature_view", "loss_normalization", "positive_h_probability",
    "mechanism_family", "risk_features", "risk_strength", "risk_interaction_fraction",
    "risk_rotate",
}
_PAIRED_CONTROLS = {
    "feature_view", "loss_normalization", "reservoir_cap", "positive_cap",
    "local_cap", "candidate_cap", "max_centers", "max_positive_centers",
    "index_dimension", "scan_chunk_size",
}


def _rng(seed: int, namespace: str) -> np.random.Generator:
    digest = hashlib.blake2b(f"{int(seed)}:{namespace}".encode(), digest_size=16).digest()
    return np.random.default_rng(int.from_bytes(digest, "little"))


def _sigmoid(x: Any) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -700, 700)))


def _uniform(ids: np.ndarray, seed: int, channel: int) -> np.ndarray:
    key = mix64(np.uint64(channel)) ^ np.uint64(seed & ((1 << 64) - 1))
    bits = mix64(np.asarray(ids, dtype=np.uint64) ^ key)
    return ((bits >> np.uint64(11)).astype(np.float64) + .5) * (1.0 / (1 << 53))


def _normal(ids: np.ndarray, seed: int, channel: int) -> np.ndarray:
    return np.sqrt(-2 * np.log(_uniform(ids, seed, 2 * channel))) * np.cos(
        2 * np.pi * _uniform(ids, seed, 2 * channel + 1))


@dataclass(frozen=True)
class _Stratum:
    start: int
    count: int
    day: int
    truth_class: int
    mode: int
    policy_flag: int
    fraud_revealed: bool
    historical: bool
    risk_cell: int = -1


@dataclass
class FinanceRows:
    ids: np.ndarray
    x: np.ndarray
    y: np.ndarray
    truth_class: np.ndarray
    event_day: np.ndarray
    label_available_at: np.ndarray
    fraud_available_at: np.ndarray
    eligible: np.ndarray
    policy_flag: np.ndarray
    risk_cell: np.ndarray


@dataclass
class FinanceMacroepisode:
    routes: list[Episode]
    original_query_count: int
    reference_normalization_weight: float
    world_id: str
    audit: dict[str, Any] = field(default_factory=dict)

    def weighted_loss(self, route_losses: list[np.ndarray]) -> float:
        """Reference reduction; trainer should implement the same expression."""
        if len(route_losses) != len(self.routes):
            raise ValueError("One loss vector is required per route")
        result = 0.0
        for episode, losses in zip(self.routes, route_losses):
            losses = np.asarray(losses, dtype=np.float64)
            if losses.shape != episode.y_query.shape:
                raise ValueError("Route loss length mismatch")
            result += float(np.dot(episode.query_weights, losses))
        return self.reference_normalization_weight * result / self.original_query_count


@dataclass
class _SupportIndex:
    reservoir_ids: np.ndarray
    positive_ids: np.ndarray
    center_ids: np.ndarray
    centers: np.ndarray
    local_ids: list[np.ndarray]
    codec: ReservoirCodec
    candidate_count: int
    candidate_mode: str


class FinanceWorld:
    """Immutable population and replayable features, with cached support selectors."""

    def __init__(self, seed: int, stage: int = 1, task: str | None = None,
                 overrides: dict[str, Any] | None = None):
        if stage not in _STAGE_CAPS:
            raise ValueError("stage must be 1, 2, or 3")
        self.seed, self.stage = int(seed), int(stage)
        self.overrides = dict(overrides or {})
        unknown = set(self.overrides) - _ALLOWED_OVERRIDES
        if unknown:
            raise ValueError(f"Unknown finance overrides: {sorted(unknown)}")
        self.feature_view = str(self.overrides.get("feature_view", "all"))
        if self.feature_view not in {"all", "hide_h", "h_only"}:
            raise ValueError("feature_view must be all, hide_h, or h_only")
        self.loss_normalization = str(self.overrides.get("loss_normalization", "reference_entropy"))
        if self.loss_normalization not in {"unit", "reference_entropy"}:
            raise ValueError("loss_normalization must be unit or reference_entropy")
        self.mechanism_family = str(self.overrides.get("mechanism_family", "class_conditional"))
        if self.mechanism_family not in {"class_conditional", "risk_partition"}:
            raise ValueError("Unknown finance mechanism family")
        rng = _rng(seed, "world_parameters")
        self.task = task or str(rng.choice(["binary", "multiclass", "regression"], p=[.7, .1, .2]))
        if self.task not in {"binary", "multiclass", "regression"}:
            raise ValueError("Unknown finance task")
        self.regression_kind = str(self.overrides.get("regression_kind", rng.choice(["full_amount", "realized_fraud_loss"])))
        if self.regression_kind not in {"full_amount", "realized_fraud_loss"}:
            raise ValueError("Unknown regression kind")
        self.full_amount = self.task == "regression" and self.regression_kind == "full_amount"
        self.hurdle = self.task == "regression" and not self.full_amount
        self.n_history = int(self.overrides.get("history_size", rng.choice([300_000, 1_000_000, 3_000_000], p=[.2, .3, .5])))
        self.n_future = int(self.overrides.get("future_size", self.n_history // 3))
        if self.n_history < 1 or self.n_future < 1:
            raise ValueError("Finite historical and future populations must be nonempty")
        self.width = int(self.overrides.get("width", rng.choice(_WIDTHS[stage][0], p=_WIDTHS[stage][1])))
        if self.width < 16:
            raise ValueError("Finance width must be at least 16")
        interval = int(rng.choice(3, p=[.1, .7, .2]))
        low, high = [(-5, -4.5), (-4.5, -3.5), (-3.5, -2)][interval]
        self.prevalence = float(self.overrides.get("prevalence", 10 ** rng.uniform(low, high)))
        if not 0 <= self.prevalence <= 1:
            raise ValueError("Prevalence must be in [0,1]")
        self.modes = int(self.overrides.get("mode_count", rng.choice([2, 3, 4], p=[.5, .3, .2])))
        if self.modes < 1:
            raise ValueError("At least one fraud mode is required")
        self.mode_weights = rng.dirichlet(np.full(self.modes, .5))
        self.n_classes = (2 if self.task == "binary" else self.modes + 1) if self.task != "regression" else 0
        self.complete = bool(self.overrides.get("complete_adjudication", rng.random() < .4))
        self.maturity_days = int(self.overrides.get("maturity_days", rng.choice([7, 30, 60])))
        if not 1 <= self.maturity_days <= 60:
            raise ValueError("Maturity days must be within [1,60]")
        self.audit_probability = float(self.overrides.get("audit_probability", .001))
        self.review_probabilities = [float(self.overrides.get("review_probability_h0", .001)),
                                     float(self.overrides.get("review_probability_h1", .05))]
        self.report_probability = float(self.overrides.get("self_report_probability", rng.choice([.5, .8, 1.], p=[.2, .4, .4])))
        self.negative_h_probability = float(self.overrides.get("negative_h_probability", rng.choice([1e-6, 1e-4, .001, .01], p=[.1, .3, .3, .3])))
        self.positive_h_probability = float(self.overrides.get("positive_h_probability", .8))
        for value in [self.audit_probability, *self.review_probabilities, self.report_probability,
                      self.negative_h_probability, self.positive_h_probability]:
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Observation probabilities must be in [0,1]")
        # CDFs include exact integer delay atoms and a final never-observed atom.
        self._delay_cdf = {(fraud, h): self._make_delay_cdf(fraud, h) for fraud in (False, True) for h in (0, 1)}
        volume = np.ones(120)
        if bool(self.overrides.get("arrival_burst", rng.random() < .5)):
            duration = int(rng.choice([1, 3, 7], p=[.5, .3, .2]))
            start = int(rng.integers(0, 121 - duration))
            volume[start:start + duration] *= rng.choice([5., 20.])
        self.daily_counts = np.r_[rng.multinomial(self.n_history, volume[:90] / volume[:90].sum()),
                                  rng.multinomial(self.n_future, volume[90:] / volume[90:].sum())]
        self.campaign_kernels: list[np.ndarray] = []
        if bool(self.overrides.get("risk_campaigns", rng.random() < .5)):
            for _ in range(int(rng.integers(1, 5))):
                center = int(rng.choice(_DAYS))
                length = int(rng.choice([1, 3, 7], p=[.5, .3, .2]))
                amplitude = np.exp(rng.uniform(np.log(10), np.log(100)))
                self.campaign_kernels.append(amplitude * np.exp(-.5 * ((_DAYS - center) / length) ** 2))
        self.risk_multiplier = 1 + np.sum(self.campaign_kernels, axis=0) if self.campaign_kernels else np.ones(120)
        drift = float(self.overrides.get("future_drift_multiplier", 1.))
        if not np.isfinite(drift) or drift <= 0:
            raise ValueError("Future drift multiplier must be finite and positive")
        if self.mechanism_family == "risk_partition":
            self._initialize_risk_partition()
        elif self.prevalence in (0., 1.):
            self.daily_risk = np.full(120, self.prevalence)
        else:
            lo, hi = -50., 20.
            for _ in range(64):
                mid = (lo + hi) / 2
                achieved = np.dot(self.daily_counts[:90], _sigmoid(mid + np.log(self.risk_multiplier[:90]))) / self.n_history
                if achieved < self.prevalence:
                    lo = mid
                else:
                    hi = mid
            logits = (lo + hi) / 2 + np.log(self.risk_multiplier)
            logits[90:] += np.log(drift)
            self.daily_risk = _sigmoid(logits)
        self.strata: list[_Stratum] = []
        if self.mechanism_family == "risk_partition":
            self._generate_risk_strata()
        else:
            self._generate_strata(_rng(seed, "finite_counts"))
        self._ends = np.asarray([s.start + s.count for s in self.strata], dtype=np.int64)
        self._counts = np.asarray([s.count for s in self.strata], dtype=np.int64)
        self._historical = np.asarray([s.historical for s in self.strata], dtype=bool)
        self._truth_classes = np.asarray([s.truth_class for s in self.strata], dtype=np.int64)
        self._eligible_strata = np.asarray([s.historical and (self.full_amount or s.fraud_revealed) for s in self.strata])
        self.n_eligible = int(self._counts[self._eligible_strata].sum())
        event_mask = self._eligible_strata & (self._truth_classes > 0)
        self.m_eligible = int(self._counts[event_mask].sum()) if not self.full_amount else 0
        self._reference_mask = np.asarray([s.historical and self.complete and s.day <= -self.maturity_days for s in self.strata])
        self.n_reference = int(self._counts[self._reference_mask].sum()) if not self.full_amount else 0
        self.reference_probs: np.ndarray | None = None
        self.reference_kind = 0
        self.reference_prevalence: float | None = None
        if self.n_reference:
            event_reference = int(self._counts[self._reference_mask & (self._truth_classes > 0)].sum())
            self.reference_prevalence = (event_reference + .5) / (self.n_reference + 1.)
            self.reference_kind = 2 if self.n_reference == self.n_eligible else 1
            if self.task in {"binary", "multiclass"}:
                labels = np.minimum(self._truth_classes, 1) if self.task == "binary" else self._truth_classes
                counts = np.bincount(labels[self._reference_mask], weights=self._counts[self._reference_mask], minlength=self.n_classes)
                self.reference_probs = (counts + .5) / (self.n_reference + .5 * self.n_classes)
        self.reference_normalization_weight = 1.
        if self.loss_normalization == "reference_entropy" and not self.full_amount and self.reference_prevalence is not None:
            pi = self.reference_prevalence
            entropy = -pi * np.log(pi) - (1 - pi) * np.log1p(-pi)
            self.reference_normalization_weight = 1. / max(entropy, 1e-4)
        self._initialize_features(_rng(seed, "feature_parameters"))
        self._support_cache: dict[tuple[int, bool], _SupportIndex] = {}
        self.world_id = "finance-" + hashlib.sha256(json.dumps({"seed": self.seed, "stage": stage, "task": self.task,
                                                                "overrides": self.overrides}, sort_keys=True).encode()).hexdigest()[:20]
        # Feature views, support selection and loss multipliers do not alter
        # the finite population or the paired query proposal.
        # Preserve the old query namespace when neither control was supplied.
        population_overrides = {k: v for k, v in self.overrides.items() if k not in _PAIRED_CONTROLS}
        self.population_id = "finance-" + hashlib.sha256(json.dumps(
            {"seed": self.seed, "stage": stage, "task": self.task, "overrides": population_overrides},
            sort_keys=True).encode()).hexdigest()[:20]

    def _initialize_risk_partition(self) -> None:
        """Finite X-first threshold cells, with sparse logistic main/interactions.

        Covariate cells are sampled before outcomes. The intercept matches the
        requested expected historical rate conditional on realized cell counts;
        actual event counts remain binomial and are never forced or retried.
        This is a discrete threshold model, not a continuous GAM or entity-risk
        process. New parameters use namespaces separate from the legacy family.
        """
        rng = _rng(self.seed, "risk_partition_parameters_v1")
        self.risk_features = int(self.overrides.get("risk_features", rng.integers(2, 6)))
        self.risk_strength = float(self.overrides.get("risk_strength", rng.uniform(.5, 4.)))
        fraction = float(self.overrides.get("risk_interaction_fraction", .5))
        if not 1 <= self.risk_features <= 6:
            raise ValueError("risk_features must be within [1,6]")
        if not np.isfinite(self.risk_strength) or not 0 <= self.risk_strength <= 12:
            raise ValueError("risk_strength must be finite and within [0,12]")
        if not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("risk_interaction_fraction must be within [0,1]")
        self.risk_rotate = bool(self.overrides.get("risk_rotate", rng.random() < .5))
        count = 1 << self.risk_features
        self.risk_bits = ((np.arange(count)[:, None] >> np.arange(self.risk_features)) & 1)
        marginal = np.clip(rng.beta(2, 2, self.risk_features), .05, .95)
        self.risk_cell_weights = np.prod(np.where(self.risk_bits, marginal, 1 - marginal), axis=1)
        self.risk_cell_weights /= self.risk_cell_weights.sum()
        signed = 2 * self.risk_bits - 1
        coefficients = rng.normal(size=self.risk_features)
        coefficients[rng.random(self.risk_features) < .5] = 0.
        coefficients[int(rng.integers(self.risk_features))] = rng.choice([-1., 1.])
        effects = (1 - fraction) * (signed @ coefficients)
        if self.risk_features > 1:
            pair = rng.choice(self.risk_features, 2, replace=False)
            effects += fraction * rng.choice([-1., 1.]) * signed[:, pair[0]] * signed[:, pair[1]]
        else:
            effects = signed[:, 0].astype(float)
        effects -= np.dot(self.risk_cell_weights, effects)
        scale = np.sqrt(np.dot(self.risk_cell_weights, effects ** 2))
        self.risk_cell_effects = self.risk_strength * effects / max(scale, 1e-12)
        self.risk_mode_weights = rng.dirichlet(np.full(self.modes, .5), count)
        count_rng = _rng(self.seed, "risk_partition_cell_counts_v1")
        self.risk_cell_counts = np.asarray([count_rng.multinomial(int(n), self.risk_cell_weights)
                                           for n in self.daily_counts])
        offsets = np.log(self.risk_multiplier)[:, None] + self.risk_cell_effects[None, :]
        if self.prevalence in (0., 1.):
            self.risk_cell_probabilities = np.full_like(offsets, self.prevalence)
        else:
            lo, hi = -50. - float(np.max(offsets)), 50. - float(np.min(offsets))
            for _ in range(80):
                mid = (lo + hi) / 2
                achieved = np.sum(self.risk_cell_counts[:90] * _sigmoid(mid + offsets[:90])) / self.n_history
                if achieved < self.prevalence:
                    lo = mid
                else:
                    hi = mid
            offsets[90:] += np.log(float(self.overrides.get("future_drift_multiplier", 1.)))
            self.risk_cell_probabilities = _sigmoid((lo + hi) / 2 + offsets)
        self.daily_risk = np.divide(np.sum(self.risk_cell_counts * self.risk_cell_probabilities, axis=1),
                                    self.daily_counts, out=np.zeros(120), where=self.daily_counts > 0)

    def _generate_risk_strata(self) -> None:
        """Exact multinomial/binomial finite counts; policy draws are separate."""
        cursor = 0
        for day_index, day in enumerate(_DAYS):
            for cell, n in enumerate(self.risk_cell_counts[day_index]):
                outcomes = _rng(self.seed, f"risk_partition_outcomes_v1-{day_index}-{cell}")
                policy = _rng(self.seed, f"risk_partition_policy_v1-{day_index}-{cell}")
                reveal = _rng(self.seed, f"risk_partition_reveal_v1-{day_index}-{cell}")
                event_count = outcomes.binomial(int(n), self.risk_cell_probabilities[day_index, cell])
                class_counts = np.r_[n - event_count, outcomes.multinomial(event_count, self.risk_mode_weights[cell])]
                for truth, count in enumerate(class_counts):
                    flagged = int(policy.binomial(int(count), self.positive_h_probability if truth else self.negative_h_probability))
                    for h, amount in enumerate((int(count) - flagged, flagged)):
                        if day < 0:
                            cdf = self._delay_cdf[(truth > 0, h)]
                            observed = int(reveal.binomial(amount, cdf[min(-int(day), 60) - 1]))
                            parts = [(True, observed), (False, amount - observed)]
                        else:
                            parts = [(False, amount)]
                        for known, part in parts:
                            if part:
                                self.strata.append(_Stratum(cursor, part, int(day), truth, truth - 1,
                                                            h, known, bool(day < 0), cell))
                                cursor += part
        if cursor != self.n_history + self.n_future:
            raise RuntimeError("Finite risk-partition counts did not conserve mass")

    def _make_delay_cdf(self, fraud: bool, h: int) -> np.ndarray:
        t = np.arange(1, 61)
        survival = (1 - self.audit_probability * (t >= 2)) * (1 - self.review_probabilities[h] * (t >= 1))
        if fraud:
            lognormal_cdf = np.asarray([.5 * (1 + math.erf((math.log(float(v)) - math.log(7)) / math.sqrt(2))) for v in t])
            lognormal_cdf[-1] = 1.  # clamp delay at 60
            survival *= 1 - self.report_probability * lognormal_cdf
        if self.complete:
            survival *= t < self.maturity_days
        return np.r_[np.clip(1 - survival, 0, 1), 1.]

    def _generate_strata(self, rng: np.random.Generator) -> None:
        cursor = 0
        for day_index, day in enumerate(_DAYS):
            n = int(self.daily_counts[day_index])
            fraud = int(rng.binomial(n, self.daily_risk[day_index]))
            fraud_modes = rng.multinomial(fraud, self.mode_weights)
            components: list[tuple[int, int, int, int]] = []
            hard = int(rng.binomial(n - fraud, self.negative_h_probability))
            components.append((0, -1, 0, n - fraud - hard))
            for mode, count in enumerate(rng.multinomial(hard, self.mode_weights)):
                components.append((0, mode, 1, int(count)))
            for mode, count in enumerate(fraud_modes):
                flagged = int(rng.binomial(count, self.positive_h_probability))
                components.extend([(mode + 1, mode, 0, int(count) - flagged), (mode + 1, mode, 1, flagged)])
            for truth, mode, h, count in components:
                if not count:
                    continue
                if day < 0:
                    cdf = self._delay_cdf[(truth > 0, h)]
                    probability = cdf[min(-int(day), 60) - 1]
                    revealed = int(rng.binomial(count, probability))
                    parts = [(True, revealed), (False, count - revealed)]
                else:
                    parts = [(False, count)]
                for known, part in parts:
                    if part:
                        self.strata.append(_Stratum(cursor, part, int(day), truth, mode, h, known, bool(day < 0)))
                        cursor += part
        if cursor != self.n_history + self.n_future:
            raise RuntimeError("Finite population counts did not conserve mass")

    def _initialize_features(self, rng: np.random.Generator) -> None:
        self.mode_centers = rng.normal(0, 2, (self.modes, 4))
        self.sigma = float(self.overrides.get("signal_sigma", rng.choice([.25, .5, 1.], p=[.25, .5, .25])))
        self.separation = float(self.overrides.get("separation", rng.choice([0., 1., 2., 4., 6.], p=[.05, .15, .25, .35, .2])))
        self.overlap = float(self.overrides.get("indistinguishable_overlap", rng.choice([0., .001, .01, .1], p=[.4, .3, .2, .1])))
        if not np.isfinite(self.sigma) or self.sigma <= 0 or not np.isfinite(self.separation) or self.separation < 0:
            raise ValueError("Signal sigma must be positive and separation nonnegative, both finite")
        if not np.isfinite(self.overlap) or not 0 <= self.overlap <= 1:
            raise ValueError("Indistinguishable overlap must be within [0,1]")
        q, r = np.linalg.qr(rng.normal(size=(8, 8)))
        self.rotation = q * np.where(np.diag(r) < 0, -1., 1.)
        self.entity_count = min(self.n_history, 10000)
        weights = rng.permutation(np.arange(1, self.entity_count + 1)) ** -1.1
        self.entity_cdf = np.cumsum(weights / weights.sum())
        self.entity_cdf[-1] = 1.
        self.entity_offsets = rng.normal(0, .25, (self.entity_count, 8))
        self.campaign_offsets = rng.normal(0, .5, (len(self.campaign_kernels) + 1, 8))
        self.campaign_offsets[0] = 0.
        self.columns = [_Column(False) for _ in range(8)]
        self.columns.extend([_Column(True, np.full(2, .5)), _Column(True, np.full(4, .25)),
                             _Column(True, np.full(self.entity_count, 1 / self.entity_count)), _Column(False)])
        fraction = float(rng.choice([0., .25, .5, 1.], p=[.35, .3, .25, .1]))
        self.extras: list[tuple[int | None, float]] = []
        self.extra_heavy_tail = bool(rng.random() < .2)
        for j in range(12, self.width):
            if j == 12:
                self.columns.append(_Column(False))
                self.extras.append((None, 0.))
            elif rng.random() < .3:
                self.columns.append(_Column(False))
                self.extras.append((int(rng.integers(0, 8)), float(np.exp(rng.uniform(np.log(.01), np.log(.3))))))
            else:
                self.columns.append(_column(rng, rng.random() < fraction))
                self.extras.append((None, 0.))
        self.missingness = str(self.overrides.get("missingness", rng.choice(["MCAR", "channel_block", "value_dependent", "outage"], p=[.2, .4, .3, .1])))
        if self.missingness not in {"MCAR", "channel_block", "value_dependent", "outage"}:
            raise ValueError("Unknown missingness mode")
        self.missing_rate = float(self.overrides.get("missing_rate", np.exp(rng.uniform(np.log(.01), np.log(.2)))))
        if not 0 <= self.missing_rate <= 1:
            raise ValueError("Invalid missing rate")
        ordinary = np.r_[np.arange(8), np.arange(12, self.width)]
        shuffled = rng.permutation(ordinary)
        self.missing_blocks = [shuffled[j::4] for j in range(4)]
        self.channel_available = rng.random((4, 4)) < .8
        self.missing_coordinates = rng.integers(0, 8, 4)
        self.missing_targets = rng.choice([.05, .2, .5], 4)
        self.outage_block = int(rng.integers(0, 4))
        self.outage_start = int(rng.integers(0, 118))
        calibration_rng = _rng(self.seed, "independent_calibration")
        calibration_n = int(self.overrides.get("calibration_rows", 4096))
        if calibration_n < 1:
            raise ValueError("Calibration rows must be positive")
        day_index = calibration_rng.choice(90, calibration_n, p=self.daily_counts[:90] / self.n_history)
        risk_cells = None
        if self.mechanism_family == "risk_partition":
            # Independent process rows from the realized covariate-cell mixture,
            # never the enriched selected support or future query population.
            cell_rng = _rng(self.seed, "risk_partition_calibration_v1")
            risk_cells = np.empty(calibration_n, dtype=int)
            for index in np.unique(day_index):
                positions = np.flatnonzero(day_index == index)
                risk_cells[positions] = cell_rng.choice(len(self.risk_cell_weights), len(positions),
                    p=self.risk_cell_counts[index] / self.daily_counts[index])
            calibration_risk = self.risk_cell_probabilities[day_index, risk_cells]
        else:
            calibration_risk = self.daily_risk[day_index]
        truth = (calibration_rng.random(calibration_n) < calibration_risk).astype(int)
        modes = calibration_rng.choice(self.modes, calibration_n, p=self.mode_weights)
        h = (calibration_rng.random(calibration_n) < np.where(truth > 0, self.positive_h_probability, self.negative_h_probability)).astype(int)
        modes[(truth == 0) & (h == 0)] = -1
        calibration_ids = np.arange(calibration_n, dtype=np.uint64) + np.uint64(1 << 62)
        raw, latent, _, channels = self._raw_features(calibration_ids, _DAYS[day_index], truth, modes, h, risk_cells)
        self.renderer = _Renderer(rng, raw, self.columns)
        self.renderer.missing_rate = 0.
        self.categorical = self.renderer.categorical.copy()
        self.missing_intercepts = np.zeros(4)
        self.missing_achieved = np.zeros(4)
        for block in range(4):
            score = .75 * latent[:, self.missing_coordinates[block]] + .8 * (channels == 0)
            lo, hi = -20., 20.
            for _ in range(64):
                mid = (lo + hi) / 2
                if _sigmoid(mid + score).mean() < self.missing_targets[block]:
                    lo = mid
                else:
                    hi = mid
            self.missing_intercepts[block] = (lo + hi) / 2
            self.missing_achieved[block] = _sigmoid(self.missing_intercepts[block] + score).mean()

    def _raw_features(self, ids: np.ndarray, day: np.ndarray, truth: np.ndarray,
                      modes: np.ndarray, h: np.ndarray,
                      risk_cells: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(ids)
        latent = np.column_stack([_normal(ids, self.seed, 100 + j) for j in range(8)])
        entities = np.searchsorted(self.entity_cdf, _uniform(ids, self.seed, 301)).clip(0, self.entity_count - 1)
        if self.mechanism_family == "risk_partition":
            if risk_cells is None:
                raise ValueError("Risk-partition rows require their finite covariate cells")
            # Half normals lie in the chosen cell. Given cell/day, X (apart
            # from explicit H) never reads the outcome, reveal, or policy flag.
            d = self.risk_features
            latent[:, :d] = (2 * self.risk_bits[risk_cells] - 1) * np.abs(latent[:, :d])
            latent[:, d:] += self.entity_offsets[entities, d:]
            if self.risk_rotate:
                latent = latent @ self.rotation
        else:
            motif = (truth > 0) | (h > 0)
            safe_modes = np.maximum(modes, 0)
            latent[motif, :4] = self.mode_centers[safe_modes[motif]] + self.sigma * latent[motif, :4]
            latent[motif, 4] += self.separation
            latent[motif, 5] += np.where(safe_modes[motif] % 2 == 0, -1., 1.) * self.separation / 2
            hard = (truth == 0) & (h > 0)
            distinguishable = hard & (_uniform(ids, self.seed, 300) >= self.overlap)
            latent[distinguishable, 4] -= 2 * self.separation
            latent += self.entity_offsets[entities]
            if self.campaign_kernels:
                day_index = np.where(day < 0, day + 90, day + 89).astype(int)
                weights = np.column_stack([np.ones(n), *[kernel[day_index] for kernel in self.campaign_kernels]])
                cdf = np.cumsum(weights, axis=1) / weights.sum(1)[:, None]
                campaign = (cdf < _uniform(ids, self.seed, 302)[:, None]).sum(1)
                latent[motif] += self.campaign_offsets[campaign[motif]]
            latent = latent @ self.rotation
        channels = np.minimum((_uniform(ids, self.seed, 303) * 4).astype(int), 3)
        raw = np.zeros((n, self.width), dtype=np.float64)
        raw[:, :8] = latent
        raw[:, 8], raw[:, 9], raw[:, 10], raw[:, 11] = h, channels, entities, day
        amount = np.exp(np.clip(2 + .4 * latent[:, 0] + .2 * latent[:, 1] + _normal(ids, self.seed, 304), -700, 700))
        multiplier = np.where(_uniform(ids, self.seed, 610) < .05, (1 - _uniform(ids, self.seed, 611)) ** (-1 / 3), 1.)
        amount *= multiplier
        for j in range(12, self.width):
            source, noise = self.extras[j - 12]
            column = self.columns[j]
            if j == 12 and self.hurdle:
                raw[:, j] = amount
            elif source is not None:
                raw[:, j] = raw[:, source] + noise * _normal(ids, self.seed, 1000 + j)
            elif column.categorical:
                raw[:, j] = np.searchsorted(np.cumsum(column.probabilities), _uniform(ids, self.seed, 5000 + j)).clip(0, column.cardinality - 1)
            else:
                raw[:, j] = _normal(ids, self.seed, 1000 + j)
                if self.extra_heavy_tail:
                    chi_square = sum(_normal(ids, self.seed, 20000 + 5 * j + k) ** 2 for k in range(5))
                    raw[:, j] *= np.sqrt(5 / np.maximum(chi_square, 1e-12))
        return raw, latent, amount, channels

    def _lookup(self, ids: np.ndarray) -> tuple[np.ndarray, list[_Stratum]]:
        ids = np.asarray(ids, dtype=np.int64)
        if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= self.n_history + self.n_future):
            raise ValueError("IDs must belong to the finite world")
        indices = np.searchsorted(self._ends, ids, side="right")
        return indices, [self.strata[int(i)] for i in indices]

    def rows(self, ids: np.ndarray) -> FinanceRows:
        """Replay exact finite rows in any order or chunking, including audit state."""
        ids = np.asarray(ids, dtype=np.int64)
        indices, strata = self._lookup(ids)
        day = np.asarray([s.day for s in strata], dtype=np.int64)
        truth = np.asarray([s.truth_class for s in strata], dtype=np.int64)
        modes = np.asarray([s.mode for s in strata], dtype=np.int64)
        h = np.asarray([s.policy_flag for s in strata], dtype=np.int64)
        cells = np.asarray([s.risk_cell for s in strata], dtype=np.int64)
        raw, latent, amount, channels = self._raw_features(ids, day, truth, modes, h, cells)
        rendered = self.renderer._transform(raw)
        for j, grid in enumerate(self.renderer.grid):
            if grid > 0:
                rendered[:, j] = np.rint(rendered[:, j] / grid) * grid
        if self.missingness == "MCAR":
            for block in self.missing_blocks:
                for j in block:
                    rendered[_uniform(ids, self.seed, 80000 + int(j)) < self.missing_rate, j] = np.nan
        else:
            for b, block in enumerate(self.missing_blocks):
                if self.missingness == "channel_block":
                    probability = np.where(self.channel_available[channels, b], .02, .95)
                elif self.missingness == "value_dependent":
                    probability = _sigmoid(self.missing_intercepts[b] + .75 * latent[:, self.missing_coordinates[b]] + .8 * (channels == 0))
                else:
                    index = np.where(day < 0, day + 90, day + 89)
                    probability = np.where((index >= self.outage_start) & (index < self.outage_start + 3), .95, .02) if b == self.outage_block else np.zeros(len(ids))
                missing = _uniform(ids, self.seed, 90000 + b) < probability
                rendered[np.ix_(missing, block)] = np.nan
        rendered = rendered[:, self.renderer.permutation]
        # Apply the same view before any selector codec fitting, candidate
        # ranking, query routing, or model call. NaN keeps the feature shape and
        # contributes only a constant missing token, with no H information.
        if self.feature_view == "hide_h":
            rendered[:, self.renderer.permutation == 8] = np.nan
        elif self.feature_view == "h_only":
            rendered[:, self.renderer.permutation != 8] = np.nan
        delay = np.empty(len(ids), dtype=np.int64)
        draws = _uniform(ids, self.seed, 99990)
        for stratum_index in np.unique(indices):
            positions = np.flatnonzero(indices == stratum_index)
            stratum = self.strata[int(stratum_index)]
            cdf = self._delay_cdf[(stratum.truth_class > 0, stratum.policy_flag)]
            lower, upper = 0., 1.
            if stratum.historical:
                threshold = cdf[min(-stratum.day, 60) - 1]
                if stratum.fraud_revealed:
                    upper = threshold
                else:
                    lower = threshold
            if not upper > lower:
                raise RuntimeError("Impossible reveal-conditioned stratum")
            u = lower + (upper - lower) * draws[positions]
            sampled = np.searchsorted(cdf, np.minimum(u, np.nextafter(upper, lower)), side="right") + 1
            delay[positions] = np.where(sampled <= 60, sampled, 10 ** 9)
        fraud_available = day + delay
        available = day + 1 if self.full_amount else fraud_available
        if self.task == "binary":
            target = (truth > 0).astype(np.int64)
        elif self.task == "multiclass":
            target = truth.copy()
        else:
            target = amount if self.full_amount else amount * (truth > 0)
        return FinanceRows(ids.copy(), rendered, target, truth, day, available, fraud_available,
                           (day < 0) & (available <= 0), h, cells)

    def _pool_mask(self, pool: str, label: int | None = None) -> np.ndarray:
        if pool == "eligible":
            return self._eligible_strata.copy()
        if pool == "positive":
            return self._eligible_strata & (self._truth_classes > 0)
        if pool == "negative":
            return self._eligible_strata & (self._truth_classes == 0)
        if pool == "future":
            result = ~self._historical
            if label is not None:
                labels = self._truth_classes if self.task == "multiclass" else np.minimum(self._truth_classes, 1)
                result &= labels == label
            return result
        raise ValueError("Unknown finite pool")

    def pool_count(self, pool: str, label: int | None = None) -> int:
        return int(self._counts[self._pool_mask(pool, label)].sum())

    def sample_ids(self, pool: str, count: int, rng: np.random.Generator,
                   replace: bool = False, label: int | None = None) -> np.ndarray:
        """Uniform finite IDs; without replacement via exact hypergeometric strata."""
        mask = self._pool_mask(pool, label)
        strata_indices = np.flatnonzero(mask)
        counts = self._counts[mask]
        total = int(counts.sum())
        count = int(count)
        if count < 0 or (not replace and count > total) or (count and total == 0):
            raise ValueError("Requested impossible finite sample")
        if count == 0:
            return np.empty(0, dtype=np.int64)
        if replace:
            rank = rng.integers(0, total, count)
            ends = np.cumsum(counts)
            positions = np.searchsorted(ends, rank, side="right")
            starts = np.r_[0, ends[:-1]]
            return np.asarray([self.strata[int(strata_indices[p])].start for p in positions], dtype=np.int64) + rank - starts[positions]
        allocations = np.zeros(len(counts), dtype=np.int64)
        remaining_draws, remaining_total = count, total
        for j, size in enumerate(counts):
            if remaining_draws == 0:
                break
            draws = int(rng.hypergeometric(int(size), remaining_total - int(size), remaining_draws)) if remaining_draws < remaining_total else int(size)
            allocations[j] = draws
            remaining_total -= int(size)
            remaining_draws -= draws
        groups = [rng.choice(int(size), int(draw), replace=False).astype(np.int64) + self.strata[int(index)].start
                  for index, size, draw in zip(strata_indices, counts, allocations) if draw]
        output = np.concatenate(groups) if groups else np.empty(0, dtype=np.int64)
        rng.shuffle(output)
        return output

    def iter_ids(self, pool: str, chunk_size: int = 1024) -> Iterator[np.ndarray]:
        if chunk_size < 1:
            raise ValueError("Chunk size must be positive")
        for index in np.flatnonzero(self._pool_mask(pool)):
            stratum = self.strata[int(index)]
            for start in range(stratum.start, stratum.start + stratum.count, chunk_size):
                yield np.arange(start, min(start + chunk_size, stratum.start + stratum.count), dtype=np.int64)

    def prepare_support(self, stage: int | None = None, training: bool = True) -> _SupportIndex:
        stage = self.stage if stage is None else int(stage)
        if stage not in _STAGE_CAPS:
            raise ValueError("Unknown stage")
        cache_key = (stage, bool(training))
        if cache_key in self._support_cache:
            return self._support_cache[cache_key]
        caps = _STAGE_CAPS[stage]
        rcap, pcap, lcap = [int(self.overrides.get(name, default)) for name, default in zip(
            ["reservoir_cap", "positive_cap", "local_cap"], caps)]
        if min(rcap, pcap, lcap) < 0:
            raise ValueError("Context caps cannot be negative")
        reservoir = self.sample_ids("eligible", min(rcap, self.n_eligible), _rng(self.seed, f"reservoir-{stage}"))
        positive_pool = "eligible" if self.full_amount else "positive"
        positives = self.sample_ids(positive_pool, min(pcap, self.pool_count(positive_pool)), _rng(self.seed, f"positive-{stage}"))
        dimension = int(self.overrides.get("index_dimension", 128))
        codec = ReservoirCodec.fit(self.rows(reservoir).x, self.categorical, self.seed, dimension)
        positive_index = codec.transform(self.rows(positives).x)
        reservoir_index = codec.transform(self.rows(reservoir).x)
        center_ids, centers = farthest_first_centers(
            positives if not self.full_amount else np.empty(0, dtype=np.int64),
            positive_index if not self.full_amount else np.empty((0, dimension)), reservoir, reservoir_index,
            max_centers=int(self.overrides.get("max_centers", 64)),
            max_positive=int(self.overrides.get("max_positive_centers", 16)), tie_seed=self.seed)
        pool = "eligible" if self.full_amount else "negative"
        available = self.pool_count(pool)
        chosen_cap = _rng(self.seed, "training_candidate_cap").choice([65536, 262144, -1], p=[.7, .2, .1]) if training else -1
        chosen_cap = int(self.overrides.get("candidate_cap", chosen_cap))
        if chosen_cap < -1:
            raise ValueError("Candidate cap must be -1 for full or nonnegative")
        candidate_count = available if chosen_cap == -1 else min(available, chosen_cap)
        chunk_size = int(self.overrides.get("scan_chunk_size", 1024))
        if chunk_size < 1:
            raise ValueError("Scan chunk must be positive")
        if candidate_count == available:
            chunks = self.iter_ids(pool, chunk_size)
            mode = "full_eligible_pool"
        else:
            sampled = self.sample_ids(pool, candidate_count, _rng(self.seed, "candidate_finite_ids"))
            chunks = (sampled[start:start + chunk_size] for start in range(0, len(sampled), chunk_size))
            mode = "declared_uniform_candidate_sample"
        visited = 0

        def indexed() -> Iterator[tuple[np.ndarray, np.ndarray]]:
            nonlocal visited
            for ids in chunks:
                visited += len(ids)
                yield ids, codec.transform(self.rows(ids).x)

        if len(centers) and lcap:
            local = stream_nearest_to_centers(indexed(), centers, lcap, self.seed)
            if visited != candidate_count:
                raise RuntimeError("Candidate scan did not visit its declared population")
        else:
            # Empty contexts have no retrieval operation; do not claim a scan.
            local = [np.empty(0, dtype=np.int64)] if not len(centers) else [np.empty(0, dtype=np.int64) for _ in centers]
            candidate_count = 0
        result = _SupportIndex(reservoir, positives, center_ids, centers, local, codec, candidate_count, mode)
        self._support_cache[cache_key] = result
        return result

    def _metadata(self, support: _SupportIndex, local: np.ndarray, context: np.ndarray) -> np.ndarray:
        r, p, l, c = len(support.reservoir_ids), len(support.positive_ids), len(local), len(context)
        events_r = int(np.count_nonzero(self.rows(support.reservoir_ids).truth_class))
        fraction_r = np.log(r / self.n_eligible) if r and self.n_eligible else 0.
        fraction_p = np.log(p / self.m_eligible) if p and self.m_eligible else 0.
        logit = 0. if self.reference_prevalence is None else np.clip(np.log(self.reference_prevalence) - np.log1p(-self.reference_prevalence), -40, 40) / 20
        result = np.asarray([np.log1p(self.n_eligible), np.log1p(self.m_eligible), np.log1p(r), np.log1p(p),
                             np.log1p(l), np.log1p(c), np.log1p(events_r), fraction_r, fraction_p,
                             float(self.m_eligible == 0), self.reference_kind, logit, np.log1p(support.candidate_count),
                             np.log1p(self.n_reference)])
        if self.full_amount:
            result[[1, 6, 8, 9, 10, 11]] = 0.
        return result

    def sample_macroepisode(self, seed: int, stage: int | None = None, training: bool = True,
                            query_count: int | None = None,
                            natural_queries: bool | None = None) -> FinanceMacroepisode:
        """Global finite-class query proposal, then feature routing, then grouping."""
        support = self.prepare_support(stage, training)
        query_count = int(query_count if query_count is not None else self.overrides.get("query_count", 128))
        if query_count < 1:
            raise ValueError("Macro query count must be positive")
        natural_queries = (not training) if natural_queries is None else bool(natural_queries)
        rng = _rng(seed, f"query-{self.population_id}")
        if natural_queries or self.full_amount:
            ids = self.sample_ids("future", query_count, rng, replace=not natural_queries)
            weights = np.ones(query_count)
        else:
            classes = self.modes + 1 if self.task == "multiclass" else 2
            counts = np.asarray([self.pool_count("future", label) for label in range(classes)], dtype=np.int64)
            present = np.flatnonzero(counts > 0)
            proposal = rng.choice(present, query_count)
            ids = np.empty(query_count, dtype=np.int64)
            weights = np.empty(query_count)
            for label in present:
                positions = np.flatnonzero(proposal == label)
                ids[positions] = self.sample_ids("future", len(positions), rng, replace=True, label=int(label))
                weights[positions] = counts[label] / self.n_future * len(present)
        query = self.rows(ids)
        route_assignment = assign_routes(support.codec.transform(query.x), support.center_ids, support.centers, self.seed)
        routes: list[Episode] = []
        for route in np.unique(route_assignment):
            positions = np.flatnonzero(route_assignment == route)
            local = support.local_ids[int(route)]
            context_ids, bits, codec_indices = union_context(support.reservoir_ids, support.positive_ids, local)
            context = self.rows(context_ids)
            if not np.all(context.eligible):
                raise RuntimeError("Unrevealed outcome reached labeled support")
            audit = {"profile": "finance", "world_id": self.world_id, "source_population_size": self.n_history,
                     "population_id": self.population_id, "feature_view": self.feature_view,
                     "loss_normalization": self.loss_normalization, "mechanism_family": self.mechanism_family,
                     "eligible_population_size": self.n_eligible, "context_ids": context_ids,
                     "query_ids": ids[positions], "original_query_positions": positions,
                     "original_query_count": query_count, "route_id": int(route), "candidate_count": support.candidate_count,
                     "candidate_mode": support.candidate_mode, "overrides": dict(self.overrides),
                     "regression_kind": self.regression_kind if self.task == "regression" else None}
            routes.append(Episode(x_support=context.x, y_support=context.y, x_query=query.x[positions],
                                  y_query=query.y[positions], categorical=self.categorical.copy(), task=self.task,
                                  n_classes=self.n_classes, metadata=audit, query_weights=weights[positions],
                                  finance_metadata=self._metadata(support, local, context_ids), source_bits=bits,
                                  codec_indices=codec_indices, reference_probs=None if self.reference_probs is None else self.reference_probs.copy(),
                                  hurdle=self.hurdle, encoding_seed=self.seed))
        audit = {"source_population_size": self.n_history, "future_population_size": self.n_future,
                 "population_id": self.population_id, "feature_view": self.feature_view,
                 "loss_normalization": self.loss_normalization, "mechanism_family": self.mechanism_family,
                 "eligible_population_size": self.n_eligible, "observed_event_count": self.m_eligible,
                 "reference_population_size": self.n_reference, "reference_kind": self.reference_kind,
                 "route_group_count": len(routes), "context_rows_total": sum(len(e.y_support) for e in routes),
                 "finite_query_ids": ids, "query_weights": weights, "route_assignment": route_assignment,
                 "overrides": dict(self.overrides), "population_resampled_for_coverage": False,
                 "natural_unique_queries": natural_queries}
        return FinanceMacroepisode(routes, query_count, self.reference_normalization_weight, self.world_id, audit)


def generate_finance_world(seed: int, stage: int = 1, task: str | None = None,
                           overrides: dict[str, Any] | None = None) -> FinanceWorld:
    return FinanceWorld(seed=seed, stage=stage, task=task, overrides=overrides)
