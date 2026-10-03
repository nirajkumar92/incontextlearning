"""Response and observation laws for the standard-prior selection study.

Fit every state on the declared calibration pool once. Neither transform method
fits statistics, sees labels, nor changes state. Pass independent support/query
seeds; stable ``row_ids`` make random draws invariant to batching and row order.
Probabilities returned by the response compiler are loss/audit-only latent-world
quantities, not inputs to the predictor and not observed-input posteriors.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import numpy as np

_EPS = 1e-6
MODES = ("identity", "mcar", "mar", "mnar", "coarsen")
MODE_WEIGHTS = (.50, .15, .15, .10, .10)


def _rng(seed: int, *keys: object) -> np.random.Generator:
    value = int.from_bytes(hashlib.sha256(repr((int(seed), *keys)).encode()).digest()[:16], "little")
    return np.random.Generator(np.random.Philox(value))


def _row_ids(n: int, row_ids: np.ndarray | None) -> np.ndarray:
    if row_ids is None:
        return np.arange(n, dtype=np.uint64)
    ids = np.asarray(row_ids)
    if ids.shape != (n,) or ids.dtype.kind not in "iu" or np.any(ids < 0):
        raise ValueError("row_ids must be one nonnegative integer per row")
    return ids.astype(np.uint64, copy=False)


def _uniform_rows(seed: int, stream: str, row_ids: np.ndarray, coordinates: int = 1) -> np.ndarray:
    """SplitMix64 counter draws: row IDs, not call boundaries, identify outcomes."""
    key = int.from_bytes(hashlib.sha256(repr((int(seed), stream)).encode()).digest()[:8], "little")
    with np.errstate(over="ignore"):
        value = (row_ids[:, None] * np.uint64(0x9E3779B97F4A7C15)
                 + np.arange(coordinates, dtype=np.uint64)[None, :] * np.uint64(0xD2B74407B1CE6E93)
                 + np.uint64(key))
        value = value + np.uint64(0x9E3779B97F4A7C15)
        value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        value = value ^ (value >> np.uint64(31))
    # Midpoints avoid log(0), and all uniforms are strictly less than one.
    return ((value >> np.uint64(12)).astype(np.float64) + .5) * (1.0 / 2**52)


def _normal_rows(seed: int, stream: str, row_ids: np.ndarray, coordinates: int = 1) -> np.ndarray:
    uniforms = _uniform_rows(seed, stream, row_ids, coordinates * 2)
    return np.sqrt(-2 * np.log(uniforms[:, 0::2])) * np.cos(2 * np.pi * uniforms[:, 1::2])


def standardized_noise(kind: str, *, seed: int, row_ids: np.ndarray) -> np.ndarray:
    """Draw the specified zero-mean, unit-variance population noise law."""
    ids = _row_ids(len(row_ids), row_ids)
    z = _normal_rows(seed, "response_noise", ids, 6 if kind == "student_t5" else 1)
    if kind == "gaussian":
        return z[:, 0]
    if kind == "student_t5":
        # t5 * sqrt(3/5) has variance one; the six Gaussian draws are independent.
        return z[:, 0] * np.sqrt(3 / np.sum(z[:, 1:] ** 2, axis=1))
    if kind == "lognormal":
        return (np.exp(z[:, 0]) - math.exp(.5)) / math.sqrt(math.e * (math.e - 1))
    raise ValueError(f"Unknown noise law {kind!r}")


def _matrix(values: np.ndarray, name: str, *, allow_missing: bool = False) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] < 1:
        raise ValueError(f"{name} must be a nonempty-width matrix")
    valid = ~np.isinf(result) if allow_missing else np.isfinite(result)
    if not np.all(valid):
        raise FloatingPointError(f"{name} contains nonfinite values")
    return result


def _softmax(logits: np.ndarray) -> np.ndarray:
    if len(logits) == 0:
        return np.empty_like(logits)
    if not np.all(np.isfinite(logits)):
        raise FloatingPointError("Nonfinite response logits")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        value = np.exp(logits - logits.max(axis=1, keepdims=True))
        return value / value.sum(axis=1, keepdims=True)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1 / (1 + np.exp(-values[positive]))
    e = np.exp(values[~positive])
    result[~positive] = e / (1 + e)
    return result


class CalibrationError(FloatingPointError):
    """A world must be retried, retaining task/shape/source, with these diagnostics."""

    def __init__(self, message: str, diagnostics: dict[str, Any]):
        super().__init__(message)
        self.diagnostics = diagnostics


def fit_class_intercepts(logits: np.ndarray, target_masses: np.ndarray, *,
                         tolerance: float = 1e-3, max_iterations: int = 200
                         ) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit sum-zero softmax intercepts with safeguarded, damped Newton updates.

    The final coordinate is fixed during the Newton solve, then the entire
    vector is centered. This removes the additive-logit null direction. Every
    accepted update decreases the convex calibration objective.
    """
    logits = _matrix(logits, "calibration logits")
    masses = np.asarray(target_masses, dtype=np.float64)
    if (len(logits) == 0 or logits.shape[1] < 2 or masses.shape != (logits.shape[1],)
            or not np.all(np.isfinite(masses)) or np.any(masses <= 0)
            or not np.isclose(masses.sum(), 1, atol=1e-12, rtol=1e-12)):
        raise ValueError("Positive normalized class masses and nonempty logits are required")
    if tolerance <= 0 or max_iterations < 0:
        raise ValueError("Invalid calibration convergence settings")
    intercepts = np.log(masses)
    intercepts -= intercepts.mean()
    backtracks = 0

    def objective(b: np.ndarray) -> float:
        shifted = logits + b
        maxima = shifted.max(axis=1)
        return float(np.mean(maxima + np.log(np.exp(shifted - maxima[:, None]).sum(axis=1)))
                     - masses @ b)

    for iteration in range(max_iterations + 1):
        probabilities = _softmax(logits + intercepts)
        achieved = probabilities.mean(axis=0)
        gradient = achieved - masses
        maximum_error = float(np.abs(gradient).max())
        diagnostics = {
            "requested_masses": masses.tolist(), "achieved_masses": achieved.tolist(),
            "maximum_absolute_error": maximum_error,
            "total_variation": float(.5 * np.abs(gradient).sum()),
            "iterations": iteration, "backtracking_steps": backtracks,
            "tolerance": float(tolerance), "max_iterations": int(max_iterations),
            "converged": maximum_error <= tolerance,
        }
        if maximum_error <= tolerance:
            return intercepts, diagnostics
        if iteration == max_iterations:
            break
        hessian = np.diag(achieved) - probabilities.T @ probabilities / len(probabilities)
        reduced = hessian[:-1, :-1]
        ridge = max(1e-12, float(np.trace(reduced)) * 1e-10 / len(reduced))
        try:
            step = np.r_[np.linalg.solve(reduced + ridge * np.eye(len(reduced)), gradient[:-1]), 0.]
        except np.linalg.LinAlgError as exc:
            raise CalibrationError("Singular class-mass calibration", diagnostics) from exc
        step -= step.mean()
        step *= min(1., 20 / max(20., float(np.abs(step).max())))
        if not np.all(np.isfinite(step)):
            raise CalibrationError("Nonfinite class-mass Newton step", diagnostics)
        old_objective = objective(intercepts)
        descent = float(gradient @ step)
        fraction = 1.
        for _ in range(40):
            candidate = intercepts - fraction * step
            candidate -= candidate.mean()
            if objective(candidate) <= old_objective - 1e-4 * fraction * descent + 1e-13:
                intercepts = candidate
                break
            fraction *= .5
            backtracks += 1
        else:
            raise CalibrationError("Class-mass line search failed", diagnostics)
    raise CalibrationError("Class-mass calibration failed to converge", diagnostics)


@dataclass(frozen=True)
class ResponseBatch:
    y: np.ndarray
    probabilities: np.ndarray | None
    diagnostics: dict[str, Any]


class ResponseCompiler:
    """Calibration-fitted response sampler; latent probabilities stay loss-only."""

    @classmethod
    def fit(cls, calibration_scores: np.ndarray, calibration_mechanism: np.ndarray, *,
            task: str, seed: int, family: str) -> "ResponseCompiler":
        scores = _matrix(calibration_scores, "calibration scores")
        mechanism = _matrix(calibration_mechanism, "calibration mechanism")
        if len(scores) == 0 or len(scores) != len(mechanism):
            raise ValueError("Calibration scores and mechanism rows must match and be nonempty")
        if task not in {"binary", "multiclass", "regression"}:
            raise ValueError(f"Unknown task {task!r}")
        if ((task == "binary" and scores.shape[1] != 2)
                or (task == "multiclass" and scores.shape[1] < 2)
                or (task == "regression" and scores.shape[1] != 1)):
            raise ValueError("Each requested class needs its own score; regression needs one")
        self = cls()
        self.task, self.family, self.mechanism_width = task, family, mechanism.shape[1]
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            self.mean, self.sd = scores.mean(axis=0), scores.std(axis=0)
        if not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.sd)):
            raise FloatingPointError("Nonfinite response calibration moments")
        self.no_signal = self.sd < _EPS
        self.diagnostics = {"task": task, "family": family, "calibration_rows": len(scores),
                            "score_mean": self.mean.tolist(), "score_sd": self.sd.tolist(),
                            "no_signal": self.no_signal.tolist(), "fit_attempts": 1,
                            "conditional_supervision": "disabled; hard sampled labels only"}
        rng = _rng(seed, "response_world")
        calibrated = self._scores(scores)
        if task != "regression":
            self.temperature = float(np.exp(rng.uniform(np.log(.25), np.log(4))))
            concentration = float(np.exp(rng.uniform(np.log(.2), np.log(5))))
            self.masses = rng.dirichlet(np.full(scores.shape[1], concentration))
            self.intercepts, fitted = fit_class_intercepts(calibrated / self.temperature, self.masses)
            self.diagnostics.update({"temperature": self.temperature, "dirichlet_concentration": concentration,
                                     "class_calibration": fitted, "class_intercepts": self.intercepts.tolist()})
        else:
            self.noise_scale = float(np.exp(rng.uniform(np.log(.03), np.log(1))))
            self.noise_kind = str(rng.choice(["gaussian", "student_t5", "lognormal"], p=[.6, .3, .1]))
            self.heteroscedastic = bool(rng.random() < .5)
            self.noise_coordinates = np.array([], dtype=int)
            self.noise_coefficients = np.array([], dtype=float)
            self.noise_rms = 1.
            if self.heteroscedastic:
                count = int(rng.integers(1, min(4, mechanism.shape[1]) + 1))
                self.noise_coordinates = rng.choice(mechanism.shape[1], count, replace=False)
                coefficients = rng.normal(size=count)
                self.noise_coefficients = coefficients / max(np.linalg.norm(coefficients), 1e-12)
                multiplier = np.exp(np.clip(mechanism[:, self.noise_coordinates] @ self.noise_coefficients, -2, 2))
                self.noise_rms = float(np.sqrt(np.mean(multiplier ** 2)))
            hierarchy = family.lower() in {"h", "p1", "hierarchy"}
            self.target_transform = str(rng.choice(["identity", "asinh"] if hierarchy else
                                                   ["identity", "asinh", "signed_square"],
                                                   p=[.75, .25] if hierarchy else [.6, .2, .2]))
            self.diagnostics.update({"noise_kind": self.noise_kind, "noise_scale": self.noise_scale,
                                     "heteroscedastic": self.heteroscedastic,
                                     "noise_coordinates": self.noise_coordinates.tolist(),
                                     "noise_coefficients": self.noise_coefficients.tolist(),
                                     "noise_calibration_rms": self.noise_rms,
                                     "target_transform": self.target_transform})
        return self

    def _scores(self, scores: np.ndarray) -> np.ndarray:
        result = np.zeros_like(scores)
        active = ~self.no_signal
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result[:, active] = ((scores[:, active] - self.mean[active])
                                 / np.maximum(self.sd[active], _EPS))
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("Nonfinite standardized response score")
        return result

    def transform(self, scores: np.ndarray, mechanism: np.ndarray, *, seed: int,
                  row_ids: np.ndarray | None = None) -> ResponseBatch:
        scores = _matrix(scores, "response scores")
        mechanism = _matrix(mechanism, "response mechanism")
        if (scores.shape[1] != len(self.mean) or len(scores) != len(mechanism)
                or mechanism.shape[1] != self.mechanism_width):
            raise ValueError("Response transform shape differs from fitted world")
        ids = _row_ids(len(scores), row_ids)
        calibrated = self._scores(scores)
        if self.task != "regression":
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                probabilities = _softmax(calibrated / self.temperature + self.intercepts)
            u = _uniform_rows(seed, "response_labels", ids)[:, 0]
            y = np.sum(u[:, None] >= np.cumsum(probabilities, axis=1), axis=1)
            y = np.minimum(y, probabilities.shape[1] - 1).astype(np.int64)
            diagnostics = {"class_counts": np.bincount(y, minlength=probabilities.shape[1]).tolist(),
                           "absent_classes": np.flatnonzero(np.bincount(y, minlength=probabilities.shape[1]) == 0).tolist()}
            return ResponseBatch(y, probabilities, diagnostics)
        multiplier = np.ones(len(scores))
        if self.heteroscedastic:
            multiplier = np.exp(np.clip(mechanism[:, self.noise_coordinates] @ self.noise_coefficients, -2, 2))
            multiplier /= max(self.noise_rms, _EPS)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            y = (calibrated[:, 0] + self.noise_scale * multiplier
                 * standardized_noise(self.noise_kind, seed=seed, row_ids=ids))
        if self.target_transform == "asinh":
            y = np.arcsinh(y)
        elif self.target_transform == "signed_square":
            with np.errstate(over="raise", invalid="raise"):
                y = np.sign(y) * y ** 2
        if not np.all(np.isfinite(y)):
            raise FloatingPointError("Nonfinite regression target; retry this world without truncating tails")
        # Diagnostics do not influence acceptance or generation and are never model inputs.
        if len(y):
            moment_scale = max(1., float(np.max(np.abs(y))))
            diagnostics = {"target_mean": float(moment_scale * np.mean(y / moment_scale)),
                           "target_sd": float(moment_scale * np.std(y / moment_scale)),
                           "target_max_abs": float(np.max(np.abs(y))),
                           "target_abs_quantiles": np.quantile(np.abs(y), [.5, .9, .99]).tolist(),
                           "realized_noise_rms_multiplier": float(np.sqrt(np.mean(multiplier ** 2)))}
        else:
            diagnostics = {"target_mean": None, "target_sd": None, "target_max_abs": None,
                           "target_abs_quantiles": [], "realized_noise_rms_multiplier": None}
        return ResponseBatch(y, None, diagnostics)


def _calibrate_missing_rate(values: np.ndarray, rate: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Missing-rate calibration requires finite nonempty predictors")
    baseline = math.log(rate / (1 - rate))
    lo, hi = baseline - float(values.max()) - 40., baseline - float(values.min()) + 40.
    for _ in range(100):
        middle = .5 * lo + .5 * hi
        actual = float(_sigmoid(values + middle).mean())
        if abs(actual - rate) <= 1e-12:
            return middle
        if actual < rate:
            lo = middle
        else:
            hi = middle
    return .5 * lo + .5 * hi


def _audit_value(value: Any) -> Any:
    """Bound episode metadata even for the largest declared category catalogs."""
    if not isinstance(value, np.ndarray):
        return value
    if value.size <= 128:
        return value.tolist()
    contiguous = np.ascontiguousarray(value)
    return {"size": int(value.size), "shape": list(value.shape),
            "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
            "minimum": float(value.min()), "maximum": float(value.max()),
            "storage": "deterministic fitted state; summarized in episode audit"}


class ObservationModel:
    """One exclusive observation mode fitted independently of label outcomes.

    New generators provide an independent same-world pool and complete category
    catalogs. The unchanged numerical reference adapter provides clean support X
    and sets ``calibration_source='clean_support'``. Existing missing cells remain
    missing, including under identity. MAR drivers receive no *additional* mask.
    """

    @classmethod
    def fit(cls, calibration_x: np.ndarray, categorical_mask: np.ndarray, *, seed: int,
            mode: str | None = None, category_levels: list[np.ndarray | None] | None = None,
            calibration_source: str = "independent_world") -> "ObservationModel":
        x = _matrix(calibration_x, "observation calibration", allow_missing=True)
        categorical = np.asarray(categorical_mask, dtype=bool)
        if len(x) == 0 or categorical.shape != (x.shape[1],):
            raise ValueError("Observation calibration requires nonempty rows and one type flag per column")
        if calibration_source not in {"independent_world", "clean_support"}:
            raise ValueError("Unknown observation calibration source")
        if category_levels is not None and len(category_levels) != x.shape[1]:
            raise ValueError("category_levels must have one entry per column")
        rng = _rng(seed, "observation_world")
        requested_mode = str(rng.choice(MODES, p=MODE_WEIGHTS)) if mode is None or mode == "mixture" else mode
        if requested_mode not in MODES:
            raise ValueError(f"Unknown observation mode {requested_mode!r}")
        self = cls()
        self.width, self.categorical = x.shape[1], categorical.copy()
        self.mode = "mcar" if requested_mode == "mar" and self.width == 1 else requested_mode
        self.columns: dict[int, dict[str, Any]] = {}
        self.drivers = np.array([], dtype=int)
        self.driver_coefficients = np.array([], dtype=float)
        self.targets = np.array([], dtype=int)
        self.rate = None
        self.means, self.sds = np.zeros(self.width), np.ones(self.width)
        self.catalogs: dict[int, np.ndarray] = {}
        self.effects: dict[int, np.ndarray] = {}
        self.diagnostics: dict[str, Any] = {
            "requested_mode": requested_mode, "mode": self.mode,
            "fallback": "mar_width_one_to_mcar" if self.mode != requested_mode else None,
            "calibration_source": calibration_source, "calibration_rows": len(x),
            "width": self.width, "fit_attempts": 1,
            "half_column_rounding": "ceil", "categorical_drivers": "fixed Gaussian effects",
            "unknown_category_policy": "neutral effect; reserved unseen coarsening group",
        }
        for j in range(self.width):
            observed = x[np.isfinite(x[:, j]), j]
            if categorical[j]:
                if np.any(observed != np.floor(observed)):
                    raise ValueError("Nominal inputs must be integer IDs or NaN")
                declared = None if category_levels is None else category_levels[j]
                levels = np.unique(observed) if declared is None else np.asarray(declared, dtype=np.float64)
                if (levels.ndim != 1 or not np.all(np.isfinite(levels))
                        or np.any(levels != np.floor(levels)) or len(np.unique(levels)) != len(levels)):
                    raise ValueError("Category catalogs must contain distinct finite integer IDs")
                if not np.all(np.isin(observed, levels)):
                    raise ValueError("Calibration category absent from declared catalog")
                raw_effects = _rng(seed, "observation_nominal_effect", j).normal(size=len(levels))
                order = np.argsort(levels)
                self.catalogs[j], self.effects[j] = levels[order], raw_effects[order]
                observed_values = self._lookup(j, observed, self.effects[j])
            else:
                observed_values = observed
            if len(observed_values):
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    self.means[j] = observed_values.mean()
                    self.sds[j] = max(float(observed_values.std()), _EPS)
        if not np.all(np.isfinite(self.means)) or not np.all(np.isfinite(self.sds)):
            raise FloatingPointError("Nonfinite observation calibration moments")
        if self.mode != "identity":
            eligible = np.arange(self.width)
            if self.mode == "mar":
                driver_count = int(rng.integers(1, min(3, self.width - 1) + 1))
                self.drivers = np.sort(rng.choice(eligible, driver_count, replace=False))
                eligible = np.setdiff1d(eligible, self.drivers)
                coefficients = rng.normal(size=driver_count)
                self.driver_coefficients = coefficients / max(float(np.linalg.norm(coefficients)), 1e-12)
            self.targets = np.sort(rng.choice(eligible, max(1, math.ceil(len(eligible) / 2)), replace=False))
        if self.mode in {"mcar", "mar", "mnar"}:
            self.rate = float(rng.uniform(.01, .30))
        mar_values = None
        if self.mode == "mar":
            mar_values = self._standardized(x, self.drivers) @ self.driver_coefficients
        for j_raw in self.targets:
            j = int(j_raw)
            if self.mode == "mcar":
                self.columns[j] = {"rate": self.rate}
            elif self.mode in {"mar", "mnar"}:
                slope = int(rng.choice([-2, -1, 1, 2])) if self.mode == "mnar" else 1
                values = (slope * self._standardized(x, np.array([j]))[:, 0]
                          if self.mode == "mnar" else mar_values)
                intercept = _calibrate_missing_rate(values, self.rate)
                self.columns[j] = {"slope": slope, "intercept": intercept,
                                   "calibration_rate": float(_sigmoid(values + intercept).mean())}
            elif self.categorical[j]:
                cardinality = len(self.catalogs[j])
                groups = max(2, math.ceil(cardinality / 2)) if cardinality > 2 else cardinality
                # Assign in declared catalog order before sorting: arbitrary ID
                # relabeling keeps the same semantic random map when a catalog is supplied.
                declared = None if category_levels is None else category_levels[j]
                order = np.argsort(np.asarray(declared)) if declared is not None else np.arange(cardinality)
                mapping = rng.integers(groups, size=cardinality)[order] if cardinality > 2 else self.catalogs[j].copy()
                self.columns[j] = {"mapping": mapping, "groups": groups,
                                   "unseen_group": groups if cardinality > 2 else None,
                                   "unchanged": cardinality <= 2}
            else:
                observed = x[np.isfinite(x[:, j]), j]
                requested_bins = int(rng.choice([2, 4, 8, 16, 32]))
                if len(observed) == 0:
                    edges, representatives = np.array([], dtype=float), np.array([0.])
                    empty_bins = [0]
                elif np.all(observed == observed[0]):
                    edges, representatives = np.array([], dtype=float), np.array([observed[0]])
                    empty_bins = []
                else:
                    edges = np.unique(np.quantile(observed, np.arange(1, requested_bins) / requested_bins))
                    assignment = np.searchsorted(edges, observed, side="right")
                    representatives = np.full(len(edges) + 1, np.nan)
                    for index in range(len(representatives)):
                        members = observed[assignment == index]
                        if len(members):
                            representatives[index] = np.median(members)
                    empty_bins = np.flatnonzero(np.isnan(representatives)).tolist()
                    occupied = np.flatnonzero(np.isfinite(representatives))
                    for index in empty_bins:
                        nearest = occupied[np.argmin(np.abs(occupied - index))]
                        representatives[index] = representatives[nearest]
                self.columns[j] = {"edges": edges, "representatives": representatives,
                                   "requested_bins": requested_bins, "empty_bins": empty_bins}
        self.diagnostics.update({"target_columns": self.targets.tolist(), "driver_columns": self.drivers.tolist(),
                                 "driver_coefficients": self.driver_coefficients.tolist(), "requested_mask_rate": self.rate,
                                 "columns": {str(j): {k: _audit_value(v)
                                                       for k, v in settings.items()}
                                             for j, settings in self.columns.items()}})
        return self

    def _lookup(self, column: int, values: np.ndarray, lookup: np.ndarray, *, unknown: float = 0.) -> np.ndarray:
        catalog = self.catalogs[column]
        result = np.full(len(values), unknown, dtype=np.float64)
        if len(catalog):
            positions = np.searchsorted(catalog, values)
            in_bounds = positions < len(catalog)
            matched = np.zeros(len(values), dtype=bool)
            matched[in_bounds] = catalog[positions[in_bounds]] == values[in_bounds]
            result[matched] = lookup[positions[matched]]
        return result

    def _standardized(self, x: np.ndarray, indices: np.ndarray) -> np.ndarray:
        result = np.empty((len(x), len(indices)))
        for offset, j in enumerate(indices):
            values = (self._lookup(int(j), x[:, j], self.effects[int(j)], unknown=self.means[j])
                      if self.categorical[j] else x[:, j])
            result[:, offset] = (values - self.means[j]) / self.sds[j]
            result[~np.isfinite(x[:, j]), offset] = 0.
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("Nonfinite observation predictor")
        return result

    def transform(self, x: np.ndarray, *, seed: int, row_ids: np.ndarray | None = None) -> np.ndarray:
        x = _matrix(x, "observation inputs", allow_missing=True)
        if x.shape[1] != self.width:
            raise ValueError("Observation transform width differs from fitted world")
        for j in np.flatnonzero(self.categorical):
            values = x[np.isfinite(x[:, j]), j]
            if np.any(values != np.floor(values)):
                raise ValueError("Nominal inputs must be integer IDs or NaN")
        ids = _row_ids(len(x), row_ids)
        result = x.copy()
        mar_values = (self._standardized(x, self.drivers) @ self.driver_coefficients
                      if self.mode == "mar" else None)
        for j, settings in self.columns.items():
            if self.mode in {"mcar", "mar", "mnar"}:
                if self.mode == "mcar":
                    probability = self.rate
                elif self.mode == "mar":
                    probability = _sigmoid(mar_values + settings["intercept"])
                else:
                    values = self._standardized(x, np.array([j]))[:, 0]
                    probability = _sigmoid(settings["slope"] * values + settings["intercept"])
                mask = _uniform_rows(seed, f"observation_mask_{j}", ids)[:, 0] < probability
                result[mask, j] = np.nan
            elif self.categorical[j]:
                if not settings["unchanged"]:
                    result[:, j] = self._lookup(j, x[:, j], settings["mapping"], unknown=settings["unseen_group"])
                    result[~np.isfinite(x[:, j]), j] = np.nan
            else:
                assignment = np.searchsorted(settings["edges"], x[:, j], side="right")
                result[:, j] = settings["representatives"][assignment]
                result[~np.isfinite(x[:, j]), j] = np.nan
        return result
