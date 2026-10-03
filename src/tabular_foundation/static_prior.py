"""Executable static-v3 prior bank in float64 NumPy.

The implementation exposes all seven baseline arms. Calibration rows are internal
generator state, and never become model context. Each named stream uses Philox
with a stable key, so support/query sizes cannot advance the parameter stream.
Optional coverage rejection, teacher selection, and distribution-shift ablations
are deliberately not exposed by this baseline implementation.
"""
from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass

import numpy as np

from .schema import Episode

CALIBRATION_ROWS = 4096
FAMILIES = ("P0", "P1", "P4")
SUBFAMILIES = ("linear_gam", "tree", "heterogeneous_scm", "local_kernel", "regime")


def _rng(seed: int, *key: object) -> np.random.Generator:
    encoded = repr((int(seed), *key)).encode("utf8")
    value = int.from_bytes(hashlib.sha256(encoded).digest()[:16], "little")
    return np.random.Generator(np.random.Philox(value))


def _ru(rng, lo, hi):
    return int(rng.integers(lo, hi + 1))


def _lu(rng, lo, hi, size=None):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size=size))


def _dlu(rng, lo, hi):
    return min(hi, max(lo, int(np.floor(_lu(rng, lo, hi + 1)))))


def _choice(rng, values, probabilities=None):
    return values[int(rng.choice(len(values), p=probabilities))]


def _unit(x, axis=0):
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12)


def _softmax(x):
    value = np.exp(x - np.max(x, axis=1, keepdims=True))
    return value / value.sum(axis=1, keepdims=True)


def _standardizer(values):
    mean = values.mean(axis=0)
    sd = values.std(axis=0)
    active = sd * sd >= 1e-12
    return lambda x: np.where(active, (x - mean) / np.maximum(sd, 1e-6), 0.0)


@dataclass
class _Column:
    categorical: bool
    probabilities: np.ndarray | None = None
    effects: np.ndarray | None = None

    @property
    def cardinality(self):
        return len(self.probabilities) if self.categorical else 0


def _column(rng, categorical):
    if not categorical:
        return _Column(False)
    c = _choice(rng, [2, 3, 5, 10, 32, 128], [.25, .15, .2, .2, .15, .05])
    frequencies = rng.permutation(np.arange(1, c + 1)) ** (-rng.uniform(0, 1.5))
    frequencies /= frequencies.sum()
    effect = rng.normal(size=c)
    effect -= np.dot(effect, frequencies)
    effect /= np.sqrt(max(np.dot(effect * effect, frequencies), 1e-12))
    return _Column(True, frequencies, effect)


class _Factory:
    def __init__(self, rng, width, active, categorical_fraction=None):
        self.width, self.active = width, active
        self.fraction = (categorical_fraction if categorical_fraction is not None else
                         _choice(rng, [0, .25, .5, 1], [.35, .3, .25, .1]))
        self.columns = [_column(rng, rng.random() < self.fraction) for _ in range(active)]
        numerical = np.array([j for j, col in enumerate(self.columns) if not col.categorical], dtype=int)
        shuffled = rng.permutation(numerical)
        self.blocks = [shuffled[start:start + 4] for start in range(0, len(shuffled), 4)]
        self.correlation = _choice(rng, [0, .3, .8], [.4, .4, .2])
        self.heavy_tail = rng.random() < .2
        self.extras = []
        for _ in range(width - active):
            if rng.random() < .3:
                source = _ru(rng, 0, active - 1)
                self.columns.append(self.columns[source])
                self.extras.append((source, _lu(rng, .01, .3)))
            else:
                self.columns.append(_column(rng, rng.random() < self.fraction))
                self.extras.append((None, 0))

    def draw(self, seed, split, rows):
        latent = np.zeros((rows, self.active), dtype=np.float64)
        raw = np.zeros((rows, self.width), dtype=np.float64)
        tail = (np.sqrt(5 / _rng(seed, split, "feature_tail").chisquare(5, rows))
                if self.heavy_tail else np.ones(rows))
        for b, indices in enumerate(self.blocks):
            shared = _rng(seed, split, "feature_block", b).normal(size=rows)
            for j in indices:
                epsilon = _rng(seed, split, "feature_coordinate", int(j)).normal(size=rows)
                raw[:, j] = tail * (np.sqrt(self.correlation) * shared + np.sqrt(1 - self.correlation) * epsilon)
                latent[:, j] = raw[:, j]
        for j, col in enumerate(self.columns):
            rng = _rng(seed, split, "feature_coordinate", j)
            if j < self.active:
                if col.categorical:
                    raw[:, j] = rng.choice(col.cardinality, size=rows, p=col.probabilities)
                    latent[:, j] = col.effects[raw[:, j].astype(int)]
                continue
            source, noise = self.extras[j - self.active]
            if source is not None:
                raw[:, j] = raw[:, source]
                if col.categorical:
                    replace = rng.random(rows) < .05
                    raw[replace, j] = rng.choice(col.cardinality, replace.sum(), p=col.probabilities)
                else:
                    raw[:, j] += rng.normal(0, noise, rows)
            elif col.categorical:
                raw[:, j] = rng.choice(col.cardinality, rows, p=col.probabilities)
            else:
                raw[:, j] = rng.normal(size=rows)
                if self.heavy_tail:
                    raw[:, j] *= np.sqrt(5 / rng.chisquare(5, rows))
        return latent, raw


def _gam(rng, calibration, outputs, force_linear=False):
    d = calibration.shape[1]
    exact = force_linear or rng.random() < .4
    kinds = ([0] * d if exact else rng.choice(5, size=d, p=[.4, .15, .2, .15, .1]))
    knots = np.array([np.quantile(calibration[:, j], rng.uniform(.2, .8)) for j in range(d)])
    frequency, phase = _lu(rng, .5, 4, d), rng.uniform(0, 2 * np.pi, d)

    def bases(x):
        result = x.copy()
        for j, kind in enumerate(kinds):
            if kind == 1:
                result[:, j] = np.tanh(x[:, j])
            elif kind == 2:
                result[:, j] = np.maximum(0, x[:, j] - knots[j])
            elif kind == 3:
                result[:, j] = np.sin(frequency[j] * x[:, j] + phase[j])
            elif kind == 4:
                result[:, j] = np.clip(x[:, j], -5, 5) ** 2
        return result

    standardize = _standardizer(bases(calibration))
    exponent = rng.uniform(0, 2)
    rank = rng.permutation(np.arange(1, d + 1))
    coefficients = _unit(rng.normal(size=(d, outputs)) * rank[:, None] ** (-exponent))
    pairs = []
    if not exact and d >= 2 and rng.random() >= .5:
        possible = list(itertools.combinations(range(d), 2))
        count = _ru(rng, 1, min(6, len(possible)))
        pairs = [possible[i] for i in rng.choice(len(possible), count, replace=False)]
    interactions = rng.normal(0, .5, (len(pairs), outputs))

    def score(x, raw=None):
        b = standardize(bases(x))
        result = b @ coefficients
        for k, (i, j) in enumerate(pairs):
            result += (b[:, i] * b[:, j])[:, None] * interactions[k]
        return result
    return score


def _tree(rng, calibration, raw_calibration, columns, outputs, depth, oblivious=False, axis_only=False):
    d = calibration.shape[1]
    numerical = np.array([j for j in range(d) if not columns[j].categorical], dtype=int)

    def predicate(indices):
        j = _ru(rng, 0, d - 1)
        if columns[j].categorical:
            count = _ru(rng, 1, columns[j].cardinality - 1)
            subset = rng.choice(columns[j].cardinality, count, replace=False)
            return lambda x, raw: np.isin(raw[:, j].astype(int), subset)
        js, coefficients = np.array([j]), np.ones(1)
        if not axis_only and len(numerical) >= 2 and rng.random() < .3:
            js = rng.choice(numerical, min(3, len(numerical)), replace=False)
            coefficients = _unit(rng.normal(size=len(js)))
        threshold = np.quantile(calibration[indices][:, js] @ coefficients, rng.uniform(.1, .9))
        return lambda x, raw: x[:, js] @ coefficients <= threshold

    if oblivious:
        predicates = [predicate(np.arange(len(calibration))) for _ in range(depth)]
        leaves = rng.normal(size=(2 ** depth, outputs))

        def score(x, raw):
            codes = np.zeros(len(x), dtype=int)
            for pred in predicates:
                codes = 2 * codes + (~pred(x, raw)).astype(int)
            return leaves[codes]
        return score

    def build(indices, remaining):
        if remaining == 0 or len(indices) < 8:
            return rng.normal(size=outputs)
        pred = predicate(indices)
        left = pred(calibration[indices], raw_calibration[indices])
        return pred, build(indices[left], remaining - 1), build(indices[~left], remaining - 1)

    root = build(np.arange(len(calibration)), depth)

    def score(x, raw):
        result = np.zeros((len(x), outputs))

        def visit(node, indices):
            if len(indices) == 0:
                return
            if isinstance(node, np.ndarray):
                result[indices] = node
            else:
                pred, left_node, right_node = node
                left = pred(x[indices], raw[indices])
                visit(left_node, indices[left])
                visit(right_node, indices[~left])
        visit(root, np.arange(len(x)))
        return result
    return score


def _rff(rng, calibration, outputs, features=128, student=False, dimensions=None):
    d = calibration.shape[1]
    indices = rng.choice(d, min(8, d) if dimensions is None else dimensions, replace=False)
    ell = _lu(rng, .3, 3, len(indices)) * np.exp(rng.normal(0, .35, len(indices)))
    omega = rng.normal(size=(features, len(indices)))
    if student:
        omega /= np.sqrt(rng.chisquare(3, features)[:, None] / 3)
    omega /= ell[None, :]
    phase = rng.uniform(0, 2 * np.pi, features)
    coefficients = rng.normal(size=(features, outputs)) / np.sqrt(features)
    return lambda x, raw=None: np.cos(x[:, indices] @ omega.T + phase) @ coefficients


def _score(rng, family, calibration, raw, columns, outputs, depth_cap):
    if family == "linear_gam":
        return _gam(rng, calibration, outputs)
    if family == "tree":
        count = _choice(rng, [1, 4, 16, 64], [.25, .25, .35, .15])
        depth = min(depth_cap, _choice(rng, [1, 2, 3, 4, 6], [.1, .2, .3, .3, .1]))
        oblivious = rng.random() < .5
        trees = [_tree(rng, calibration, raw, columns, outputs, depth, oblivious) for _ in range(count)]
        coefficients = _unit(rng.normal(size=(count, outputs)))
        return lambda x, r: sum(tree(x, r) * coefficients[i] for i, tree in enumerate(trees))
    if family == "local_kernel":
        if rng.random() < .7:
            return _rff(rng, calibration, outputs, student=rng.random() >= .6)
        indices = rng.choice(calibration.shape[1], min(8, calibration.shape[1]), replace=False)
        centers = calibration[rng.choice(len(calibration), 16, replace=False)][:, indices]
        bandwidth = _lu(rng, .3, 3)
        values = rng.normal(size=(16, outputs))

        def score(x, r=None):
            distance = np.sum((x[:, indices, None] - centers.T[None, :, :]) ** 2, axis=1)
            return _softmax(-distance / (2 * bandwidth ** 2)) @ values
        return score
    if family == "regime":
        regimes = _ru(rng, 2, 4)
        numeric = [j for j in range(calibration.shape[1]) if not columns[j].categorical]
        if rng.random() < .7:
            if numeric:
                j = int(rng.choice(numeric))
                thresholds = np.quantile(calibration[:, j], np.arange(1, regimes) / regimes)
                gate = lambda x, r: np.eye(regimes)[np.searchsorted(thresholds, x[:, j], side="right")]
            else:
                mapping = np.empty(columns[0].cardinality, dtype=int)
                mapping[rng.permutation(len(mapping))] = np.arange(len(mapping)) % regimes
                gate = lambda x, r: np.eye(regimes)[mapping[r[:, 0].astype(int)]]
        else:
            indices = rng.choice(calibration.shape[1], min(3, calibration.shape[1]), replace=False)
            projection = rng.normal(size=(len(indices), regimes))
            temperature = _lu(rng, .2, 2)
            gate = lambda x, r: _softmax(x[:, indices] @ projection / temperature)
        base = _gam(rng, calibration, outputs)
        residuals = [(_gam(rng, calibration, outputs, force_linear=True) if rng.random() < .5 else
                      _tree(rng, calibration, raw, columns, outputs, min(2, depth_cap), axis_only=True))
                     for _ in range(regimes)]
        amplitude = _choice(rng, [0, .5, 1, 2], [.2, .3, .3, .2])

        def score(x, r):
            gates = gate(x, r)
            return base(x) + amplitude * sum(gates[:, k, None] * residuals[k](x, r) for k in range(regimes))
        return score
    raise ValueError(f"Unknown score family {family!r}")


class _Response:
    def __init__(self, rng, task, classes, n_support):
        self.task, self.classes = task, classes
        self.outputs = classes if task == "multiclass" else 1
        if task == "regression":
            level = _choice(rng, range(5), [.05, .2, .45, .25, .05])
            self.rho = [0, rng.uniform(.02, .2), rng.uniform(.2, .7), rng.uniform(.7, .98), 1][level]
            self.null = self.rho == 0
            self.noise = _choice(rng, range(4), [.6, .2, .1, .1])
            self.homoscedastic = rng.random() < .6
            self.transform = _choice(rng, range(3), [.75, .15, .1])
            self.scale = _lu(rng, .1, 100)
            self.location = rng.normal(0, 10 * self.scale)
        else:
            self.null = rng.random() < .05
            alpha = _choice(rng, [.2, 1, 5], [.35, .4, .25])
            delta = min(2 / n_support, 1 / (2 * classes))
            self.pi = delta + (1 - classes * delta) * rng.dirichlet(np.full(classes, alpha))
            self.logit_scale = _lu(rng, .25, 8)
            self.contamination = 0 if rng.random() < .6 else rng.uniform(0, .15)
            self.permutation = rng.permutation(classes)

    def fit(self, rng, scores, latent):
        self.standardize = _standardizer(scores)
        self.degenerate_scores = int(np.sum(scores.var(axis=0) < 1e-12))
        if self.task == "regression":
            self.hetero_coordinate = _ru(rng, 0, latent.shape[1] - 1)
            h = np.exp(.5 * np.tanh(latent[:, self.hetero_coordinate]))
            self.h_normalizer = np.sqrt(np.mean(h ** 2))
        else:
            g = self.logits(scores)
            self.intercept = np.log(self.pi)
            for _ in range(64):
                mean = _softmax(g + self.intercept).mean(axis=0)
                self.intercept = np.clip(self.intercept + .5 * np.log(self.pi / np.maximum(mean, 1e-8)), -12, 12)
                self.intercept -= self.intercept.mean()
            self.achieved_prevalence = self.probabilities(scores).mean(axis=0)

    def logits(self, scores):
        s = self.standardize(scores)
        g = np.c_[np.zeros(len(s)), s[:, 0]] if self.task == "binary" else s
        return g * (0 if self.null else self.logit_scale)

    def probabilities(self, scores):
        p = (1 - self.contamination) * _softmax(self.logits(scores) + self.intercept) + self.contamination * self.pi
        result = np.empty_like(p)
        result[:, self.permutation] = p
        return result

    def draw(self, rng, scores, latent):
        if self.task != "regression":
            p = self.probabilities(scores)
            y = np.sum(rng.random(len(p))[:, None] > np.cumsum(p, axis=1), axis=1)
            return np.minimum(y, self.classes - 1).astype(np.int64), p
        n = len(scores)
        if self.noise == 0:
            epsilon = rng.normal(size=n)
        elif self.noise == 1:
            epsilon = np.sqrt(3 / 5) * rng.standard_t(5, n)
        elif self.noise == 2:
            epsilon = rng.laplace(0, 1 / np.sqrt(2), n)
        else:
            epsilon = rng.normal(size=n) * np.where(rng.random(n) < .9, .5, 2) / np.sqrt(.625)
        standardized = self.standardize(scores)[:, 0]
        h = 1 if self.homoscedastic else np.exp(.5 * np.tanh(latent[:, self.hetero_coordinate])) / self.h_normalizer
        w = epsilon if self.null else standardized + np.sqrt((1 - self.rho) / self.rho) * h * epsilon
        if self.transform == 1:
            w = np.sign(w) * np.log1p(np.abs(w))
        elif self.transform == 2:
            w = np.exp(np.clip(w, -8, 8))
        mean = (self.location + self.scale * (np.zeros(n) if self.null else standardized)) if self.transform == 0 else None
        return self.location + self.scale * w, mean


class _Renderer:
    def __init__(self, rng, calibration, columns):
        self.columns = columns
        self.permutation = rng.permutation(len(columns))
        self.category_codes = [rng.permutation(col.cardinality) if col.categorical else None for col in columns]
        self.kinds = rng.choice(4, size=len(columns), p=[.5, .15, .2, .15])
        self.power = _lu(rng, .3, 3, len(columns))
        self.scale = _lu(rng, .1, 10, len(columns))
        self.location = rng.normal(0, 3, len(columns))
        self.grid = np.zeros(len(columns))
        numeric = np.array([j for j, col in enumerate(columns) if not col.categorical], dtype=int)
        if rng.random() < .25:
            selected = rng.choice(numeric, min(len(numeric), len(columns) // 4), replace=False)
            rendered = self._transform(calibration)
            self.grid[selected] = _lu(rng, .05, .5, len(selected)) * rendered[:, selected].std(axis=0)
        self.missing_rate = 0 if rng.random() < .7 else rng.uniform(.01, .2)
        self.categorical = np.array([c.categorical for c in columns])[self.permutation]

    def _transform(self, raw):
        out = raw.copy()
        for j, col in enumerate(self.columns):
            if col.categorical:
                out[:, j] = self.category_codes[j][raw[:, j].astype(int)]
                continue
            value = raw[:, j]
            if self.kinds[j] == 1:
                value = np.arcsinh(value)
            elif self.kinds[j] == 2:
                value = np.sign(value) * np.abs(value) ** self.power[j]
            elif self.kinds[j] == 3:
                value = np.exp(np.clip(value, -5, 5))
            out[:, j] = self.location[j] + self.scale[j] * value
        return out

    def draw(self, seed, split, raw):
        out = self._transform(raw)
        for j, grid in enumerate(self.grid):
            if grid > 0:
                out[:, j] = np.rint(out[:, j] / grid) * grid
            mask = _rng(seed, "rendering", split, j).random(len(out)) < self.missing_rate
            out[mask, j] = np.nan
        return out[:, self.permutation]


def sample_p4_outcomes(rng, mu, dispersion, zero_inflation, severity_mean=None, severity_shape=2):
    """Exact unbounded NB/Poisson + structural zero + compound Gamma sampler."""
    mu = np.asarray(mu, dtype=np.float64)
    if np.any(~np.isfinite(mu)) or np.any(mu < 0) or not 0 <= zero_inflation <= 1:
        raise ValueError("Invalid count law")
    if np.isinf(dispersion):
        count = rng.poisson(mu)
    elif dispersion > 0:
        count = rng.negative_binomial(dispersion, dispersion / (dispersion + mu))
    else:
        raise ValueError("Dispersion must be positive")
    count[rng.random(mu.shape) < zero_inflation] = 0
    if severity_mean is None:
        return count
    if severity_shape <= 0:
        raise ValueError("Severity shape must be positive")
    severity_mean = np.broadcast_to(np.asarray(severity_mean), mu.shape)
    if np.any(~np.isfinite(severity_mean)) or np.any(severity_mean <= 0):
        raise ValueError("Invalid severity mean")
    value = np.zeros_like(mu)
    positive = count > 0
    value[positive] = rng.gamma(count[positive] * severity_shape, severity_mean[positive] / severity_shape)
    return value


def _scm_mechanism(rng, parents, outputs, depth_cap):
    kind = _choice(rng, ["linear", "mlp", "tree", "rff"], [.25, .35, .25, .15])
    k = parents.shape[1]
    if kind == "linear":
        weight = rng.normal(0, 1 / np.sqrt(k), (k, outputs))
        return lambda x: x @ weight
    if kind == "mlp":
        width = _choice(rng, [8, 16, 32], [.3, .4, .3])
        layers = _choice(rng, [1, 2], [.75, .25])
        activation = _choice(rng, [np.tanh, lambda x: np.maximum(x, 0), np.sin], [.5, .3, .2])
        sizes = [k] + [width] * layers + [outputs]
        weights = [rng.normal(0, 1 / np.sqrt(a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        biases = [rng.normal(0, .5, b) for b in sizes[1:]]

        def mechanism(x):
            for i, (w, b) in enumerate(zip(weights, biases)):
                x = x @ w + b
                if i < len(weights) - 1:
                    x = activation(x)
            return x
        return mechanism
    if kind == "tree":
        tree = _tree(rng, parents, parents, [_Column(False)] * k, outputs, min(2, depth_cap), axis_only=True)
        return lambda x: tree(x, x)
    return _rff(rng, parents, outputs, features=64, dimensions=k)


def _scm(seed, rng, response, width, active, n_support, n_query, depth_cap):
    observed_count = min(active, 16)
    nodes = _ru(rng, max(8, observed_count + 2), max(16, 2 * observed_count + 8))
    parents = [[]]
    for i in range(1, nodes):
        count = min(i, _choice(rng, [1, 2, 3, 4], [.4, .3, .2, .1]))
        parents.append(list(rng.choice(i, count, replace=False)))
    target = _ru(rng, 1, nodes - 2)
    eligible = np.array([i for i in range(nodes) if i != target])
    if rng.random() < .7:
        mandatory = int(rng.choice(parents[target]))
        observed = np.r_[mandatory, rng.choice(eligible[eligible != mandatory], observed_count - 1, replace=False)]
    else:
        observed = rng.choice(eligible, observed_count, replace=False)
    descendants = set()
    for i in range(target + 1, nodes):
        if target in parents[i] or any(p in descendants for p in parents[i]):
            descendants.add(i)
    fraction = _choice(rng, [0, .25, .5, 1], [.35, .3, .25, .1])
    factory = _Factory(rng, width, observed_count, fraction)
    if response.null:
        # The null override has independent features, so target noise cannot leak
        # through descendants. It still uses the originally sampled response.
        pools = {split: factory.draw(seed, split, rows) for split, rows in
                 (("calibration", CALIBRATION_ROWS), ("support", n_support), ("query", n_query))}
        score = _gam(rng, pools["calibration"][0], response.outputs)
        response.fit(rng, score(pools["calibration"][0]), pools["calibration"][0])
        y = {s: response.draw(_rng(seed, "labels", s), score(v), v)[0] for s, (v, x) in pools.items()}
        return {s: x for s, (v, x) in pools.items()}, y, factory.columns, {
            "target_node": target, "observed_nodes": observed.tolist(),
            "observed_target_descendants": [], "null_override": True}

    c = np.zeros((CALIBRATION_ROWS, nodes))
    mechanisms, normalizers = {}, {}
    embedding = rng.normal(size=response.classes) if response.task != "regression" else None
    calibration_target = None
    target_normalize = None
    for i in range(nodes):
        if i == 0:
            c[:, i] = _rng(seed, "calibration", "scm", i).normal(size=len(c))
            continue
        x = c[:, parents[i]]
        mechanism = _scm_mechanism(_rng(seed, "parameters", "scm", i), x,
                                   response.outputs if i == target else 1, depth_cap)
        mechanisms[i] = mechanism
        f = mechanism(x)
        if i == target:
            response.fit(rng, f, x)
            calibration_target, _ = response.draw(_rng(seed, "labels", "calibration", i), f, x)
            if response.task == "regression":
                target_normalize = _standardizer(calibration_target[:, None])
                c[:, i] = target_normalize(calibration_target[:, None])[:, 0]
            else:
                c[:, i] = embedding[calibration_target]
        else:
            noisy = f + _rng(seed, "calibration", "scm", i).normal(0, .1, (len(c), 1))
            normalizers[i] = _standardizer(noisy)
            c[:, i] = normalizers[i](noisy)[:, 0]

    def graph_draw(split, rows):
        values = np.zeros((rows, nodes))
        y = None
        for i in range(nodes):
            if i == 0:
                values[:, i] = _rng(seed, split, "scm", i).normal(size=rows)
                continue
            x = values[:, parents[i]]
            f = mechanisms[i](x)
            if i == target:
                y, _ = response.draw(_rng(seed, "labels", split, i), f, x)
                values[:, i] = target_normalize(y[:, None])[:, 0] if response.task == "regression" else embedding[y]
            else:
                noisy = f + _rng(seed, split, "scm", i).normal(0, .1, (rows, 1))
                values[:, i] = normalizers[i](noisy)[:, 0]
        return values, y

    graph_pools = {"calibration": (c, calibration_target),
                   "support": graph_draw("support", n_support), "query": graph_draw("query", n_query)}
    edges = {}
    for j in range(observed_count):
        col = factory.columns[j]
        if col.categorical:
            edges[j] = np.quantile(c[:, observed[j]], np.cumsum(col.probabilities)[:-1])
    raw_pools, y_pools = {}, {}
    for split, (values, y) in graph_pools.items():
        _, raw = factory.draw(seed, split, len(values))
        for j, node in enumerate(observed):
            raw[:, j] = (np.searchsorted(edges[j], values[:, node], side="right") if j in edges else values[:, node])
        # Factory copies must use actual observed graph coordinates, not the
        # independent temporary coordinates used for its distractor draws.
        for j in range(observed_count, width):
            source, noise = factory.extras[j - observed_count]
            if source is None:
                continue
            stream = _rng(seed, split, "scm_extra", j)
            raw[:, j] = raw[:, source]
            col = factory.columns[j]
            if col.categorical:
                replace = stream.random(len(raw)) < .05
                raw[replace, j] = stream.choice(col.cardinality, replace.sum(), p=col.probabilities)
            else:
                raw[:, j] += stream.normal(0, noise, len(raw))
        raw_pools[split], y_pools[split] = raw, y
    return raw_pools, y_pools, factory.columns, {
        "target_node": target, "parents": parents, "observed_nodes": observed.tolist(),
        "observed_target_descendants": [int(i) for i in observed if i in descendants],
        "null_override": False}


def _p1(seed, rng, factory, pools, outputs, n_support):
    entities = _dlu(rng, 8, min(4096, max(16, 2 * n_support)))
    organizations = _ru(rng, 2, min(32, max(2, entities // 2)))
    group = np.empty(entities, dtype=int)
    group[rng.permutation(entities)] = np.arange(entities) % organizations
    metadata = rng.normal(size=(entities, 2))
    frequencies = np.ones(entities)
    if rng.random() >= .2:
        frequencies = rng.permutation(np.arange(1, entities + 1)) ** (-rng.uniform(.6, 1.8))
    frequencies /= frequencies.sum()
    effect, organization_effect = np.zeros((entities, outputs)), np.zeros((organizations, outputs))
    slopes = np.zeros((entities, min(2, factory.active), outputs))
    for k in range(outputs):
        omega = _choice(rng, [0, .5, 1], [.3, .4, .3])
        tau = _choice(rng, [0, .25, 1, 2], [.15, .25, .4, .2])
        beta = _unit(rng.normal(size=2))
        effect[:, k] = tau * (np.sqrt(omega) * (metadata @ beta) + np.sqrt(1 - omega) * rng.normal(size=entities))
        organization_effect[:, k] = rng.normal(0, _choice(rng, [0, .5, 1], [.5, .3, .2]), organizations)
        slopes[:, :, k] = rng.normal(0, _choice(rng, [0, .25, .75], [.6, .3, .1]), slopes.shape[:2])
    base = _gam(rng, pools["calibration"][0], outputs)
    scores, ids, raw_pools = {}, {}, {}
    for split, (v, raw) in pools.items():
        entity = _rng(seed, split, "p1_entity").choice(entities, len(v), p=frequencies)
        ids[split] = entity
        scores[split] = base(v) + effect[entity] + organization_effect[group[entity]] + np.einsum("nd,ndo->no", v[:, :slopes.shape[1]], slopes[entity])
        raw_pools[split] = np.column_stack([raw, entity, group[entity], metadata[entity]])
    group_probabilities = np.bincount(group, weights=frequencies, minlength=organizations)
    columns = factory.columns + [_Column(True, frequencies), _Column(True, group_probabilities), _Column(False), _Column(False)]
    diagnostics = {"entity_count": entities, "organization_count": organizations,
                   "support_unique_entities": int(len(np.unique(ids["support"]))),
                   "unseen_query_entity_fraction": float(np.mean(~np.isin(ids["query"], ids["support"])))}
    return raw_pools, scores, columns, diagnostics


def _p4(seed, rng, factory, pools, task):
    c = pools["calibration"][0]
    score, severity_score = _gam(rng, c, 1), _gam(rng, c, 1)
    standardize, standardize_v = _standardizer(score(c)), _standardizer(severity_score(c))
    sigma = _choice(rng, [0, .5, 1], [.2, .5, .3])
    mu0 = _choice(rng, [.2, 1, 5, 20], [.25, .35, .25, .15])
    signal = _lu(rng, .25, 2)
    dispersion = _choice(rng, [.5, 2, 10, np.inf], [.2, .35, .25, .2])
    zero = _choice(rng, [0, .2, .6], [.5, .35, .15])
    kind = _choice(rng, ["count", "compound_gamma_amount"], [.6, .4]) if task == "regression" else "binary"
    nu0, coupling = _lu(rng, 1, 100), _choice(rng, [-.5, 0, .5], [.2, .4, .4])
    severity_shape = _choice(rng, [.5, 2, 10])
    reverse = bool(rng.random() < .5)
    scores, exposures, unnormalized = {}, {}, {}
    for split, (v, raw) in pools.items():
        scores[split] = (standardize(score(v))[:, 0], standardize_v(severity_score(v))[:, 0])
        exposures[split] = np.clip(np.exp(_rng(seed, split, "p4_exposure").normal(-sigma ** 2 / 2, sigma, len(v))), .02, 50)
        unnormalized[split] = exposures[split] * np.exp(np.clip(signal * scores[split][0], -4, 4))
    normalizer = max(1e-6, unnormalized["calibration"].mean())
    raw_pools, targets, laws = {}, {}, {}
    for split, (v, raw) in pools.items():
        mu = mu0 * unnormalized[split] / normalizer
        s, s_v = scores[split]
        nu = nu0 * np.exp(np.clip(coupling * s + .5 * s_v, -4, 4))
        y = sample_p4_outcomes(_rng(seed, "labels", split), mu, dispersion, zero,
                               nu if kind == "compound_gamma_amount" else None, severity_shape)
        if task == "binary":
            y = (y > 0).astype(np.int64)
            p_zero = np.exp(-mu) if np.isinf(dispersion) else np.exp(-dispersion * np.log1p(mu / dispersion))
            positive = (1 - zero) * (1 - p_zero)
            if reverse:
                y, positive = 1 - y, 1 - positive
            laws[split] = np.column_stack([1 - positive, positive])
        else:
            laws[split] = (1 - zero) * mu * (nu if kind == "compound_gamma_amount" else 1)
        targets[split] = y
        raw_pools[split] = np.column_stack([raw, exposures[split]])
    diagnostics = {"kind": kind, "dispersion": float(dispersion), "zero_inflation": zero,
                   "calibration_target_mean": float(np.mean(targets["calibration"])),
                   "calibration_zero_rate": float(np.mean(targets["calibration"] == 0))}
    return raw_pools, targets, laws, factory.columns + [_Column(False)], diagnostics


def _shape(rng, family, task, stage, n_support, n_query, n_features):
    if stage not in (1, 2, 3):
        raise ValueError("stage must be 1, 2, or 3")
    bucket = "short" if stage == 1 else (_choice(rng, ["short", "medium"], [.4, .6]) if stage == 2 else
               _choice(rng, ["short", "medium", "long", "wide"], [.3, .3, .25, .15]))
    manual_support = n_support is not None
    if n_support is None:
        if bucket == "short":
            lo, hi = _choice(rng, [(32, 127), (128, 511), (512, 2048)])
            n_support = _ru(rng, lo, hi)
        else:
            bounds = {"medium": (2049, 8192), "long": (8193, 32768), "wide": (1024, 4096)}
            n_support = _dlu(rng, *bounds[bucket])
    if task == "multiclass":
        if stage == 1:
            # Preserve the original first-stage draw and its RNG consumption.
            n_classes = _ru(rng, 3, 10)
        else:
            ranges = [(3, 10), (11, 32), (33, 128)]
            probabilities = [.7, .2, .1]
            if stage == 3:
                ranges.append((129, 256))
                probabilities = [.7, .2, .08, .02]
            n_classes = _ru(rng, *_choice(rng, ranges, probabilities))
    else:
        n_classes = 2 if task == "binary" else 0
    if task == "multiclass" and not manual_support:
        n_support = max(n_support, 16 * n_classes)
    if n_query is None:
        n_query = min(128, max(16, n_support // 4))
    if n_features is None:
        choices = {"short": ([4, 8, 16, 32, 64, 128], [.1, .15, .2, .25, .15, .15]),
                   "medium": ([64, 128, 256, 512], [.25, .35, .25, .15]),
                   "long": ([128, 256, 512], [.25, .375, .375]), "wide": ([512, 1024], [.35, .65])}
        widths, p = choices[bucket]
        widths, p = np.asarray(widths), np.asarray(p, dtype=float)
        allowed = (n_support + n_query) * widths <= {1: 2 ** 19, 2: 2 ** 21, 3: 2 ** 23}[stage]
        if not np.any(allowed):
            raise ValueError("No width satisfies the curriculum cell budget")
        n_features = int(rng.choice(widths[allowed], p=p[allowed] / p[allowed].sum()))
        n_features = max(8 if family == "P1" else 4, n_features)
    if n_support < 1 or n_query < 1 or n_features < (8 if family == "P1" else 4):
        raise ValueError("Explicit shape violates positive-row or minimum-width requirements")
    return int(n_support), int(n_query), int(n_features), n_classes, bucket


def generate_episode(seed: int, family: str | None = None, subfamily: str | None = None,
                     task: str | None = None, n_support: int | None = None,
                     n_query: int | None = None, n_features: int | None = None,
                     stage: int = 1, context_conditioning: bool = False,
                     joint: bool = False, n_classes: int | None = None) -> Episode:
    """Generate one baseline episode, without any query-dependent acceptance.

    Explicit shape arguments are diagnostic overrides: they bypass the sampled
    multiclass support floor and curriculum cell caps. ``joint=True`` selects
    the standard component head mix that preserves overall one-third task mass
    under an outer 80% standard / 20% finance mixture. It does not draw finance.
    Unsupported family/head combinations raise, rather than silently changing
    the requested family. Invalid arithmetic raises with the world seed intact.
    """
    shape_rng, rng = _rng(seed, "shape"), _rng(seed, "parameters")
    if family is None:
        family = _choice(shape_rng, FAMILIES, [.85, .1, .05])
    if family not in FAMILIES:
        raise ValueError(f"Unknown family {family!r}")
    if family != "P0" and subfamily is not None:
        raise ValueError("subfamily is only valid for P0")
    if family == "P0":
        subfamily = subfamily or _choice(shape_rng, SUBFAMILIES, [.2, .25, .3, .15, .1])
        if subfamily not in SUBFAMILIES:
            raise ValueError(f"Unknown P0 subfamily {subfamily!r}")
    if task is None:
        if family == "P4":
            task = _choice(shape_rng, ["binary", "regression"], [.2, .8])
        else:
            probabilities = ([.243859649122807, .412280701754386, .343859649122807] if joint else
                             [.34035087719298246, .3508771929824561, .3087719298245614])
            task = _choice(shape_rng, ["binary", "multiclass", "regression"], probabilities)
    if task not in ("binary", "multiclass", "regression") or (family == "P4" and task == "multiclass"):
        raise ValueError(f"Unsupported task {task!r} for {family}")
    ns, nq, width, classes, bucket = _shape(shape_rng, family, task, stage, n_support, n_query, n_features)
    # Reference-mixture controls fix the class universe before choosing a family.
    # Keep the original shape draw above unchanged when this override is absent.
    if n_classes is not None:
        if isinstance(n_classes, bool) or not isinstance(n_classes, (int, np.integer)):
            raise ValueError("n_classes must be an integer")
        if ((task == "binary" and n_classes != 2) or
                (task == "regression" and n_classes != 0) or
                (task == "multiclass" and n_classes < 3)):
            raise ValueError("n_classes disagrees with the requested task")
        classes = int(n_classes)
    available = width - (4 if family == "P1" else 1 if family == "P4" else 0)
    active = _dlu(rng, 1, min(available, 16)) if rng.random() < .75 else _ru(rng, math.ceil(available / 3), available)
    capped = context_conditioning and _rng(seed, "context_conditioning").random() < .9
    if capped:
        active = min(active, max(1, ns // 8))
    depth_cap = max(1, int(np.floor(np.log2(max(2, ns / 4))))) if capped else 6
    diagnostics = {"seed": int(seed), "family": family, "subfamily": subfamily,
                   "version": "static-v3-context" if context_conditioning else "static-v3-base",
                   "active_dimensions": active, "context_capped": bool(capped),
                   "curriculum_bucket": bucket, "calibration_rows": CALIBRATION_ROWS,
                   "class_universe_declared_before_rows": True, "query_acceptance_used": False,
                   "explicit_shape_override": any(x is not None for x in (n_support, n_query, n_features))}
    laws = {}
    if family == "P0" and subfamily == "heterogeneous_scm":
        response = _Response(rng, task, classes, ns)
        raw_pools, targets, columns, info = _scm(seed, rng, response, width, active, ns, nq, depth_cap)
        diagnostics.update(info)
        diagnostics["null"] = bool(response.null)
    else:
        factory = _Factory(rng, available, active)
        pools = {split: factory.draw(seed, split, rows) for split, rows in
                 (("calibration", CALIBRATION_ROWS), ("support", ns), ("query", nq))}
        if family == "P4":
            raw_pools, targets, laws, columns, info = _p4(seed, rng, factory, pools, task)
            diagnostics.update(info)
        else:
            response = _Response(rng, task, classes, ns)
            if family == "P1":
                raw_pools, scores, columns, info = _p1(seed, rng, factory, pools, response.outputs, ns)
                diagnostics.update(info)
            else:
                mechanism = _score(rng, subfamily, pools["calibration"][0], pools["calibration"][1],
                                   factory.columns, response.outputs, depth_cap)
                scores = {split: mechanism(v, raw) for split, (v, raw) in pools.items()}
                raw_pools = {split: raw for split, (v, raw) in pools.items()}
                columns = factory.columns
            response.fit(rng, scores["calibration"], pools["calibration"][0])
            targets = {}
            for split, (v, raw) in pools.items():
                targets[split], laws[split] = response.draw(_rng(seed, "labels", split), scores[split], v)
            diagnostics["null"] = bool(response.null)
            diagnostics["degenerate_scores"] = response.degenerate_scores
            if task != "regression":
                diagnostics["achieved_calibration_prevalence"] = response.achieved_prevalence.tolist()
    renderer = _Renderer(_rng(seed, "rendering", "parameters"), raw_pools["calibration"], columns)
    support = renderer.draw(seed, "support", raw_pools["support"])
    query = renderer.draw(seed, "query", raw_pools["query"])
    diagnostics["latent_conditional_law"] = laws.get("query")
    diagnostics["missing_rate"] = renderer.missing_rate
    diagnostics["support_missing"] = np.isnan(support)
    diagnostics["query_missing"] = np.isnan(query)
    return Episode(support, targets["support"], query, targets["query"], renderer.categorical,
                   task, classes, metadata=diagnostics, encoding_seed=int(seed))
