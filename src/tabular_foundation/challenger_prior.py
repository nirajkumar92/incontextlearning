"""Executable selection-v1 challenger, using the pinned native shape sampler.

A world is sampled before its calibration, support and query rows. All new
mechanisms use independent keyed streams; only numerical-invalidity retries are
allowed. Native R preserves its own complete generation and acceptance rules.
The common response and observation laws live in :mod:`selection_laws`.
"""
from __future__ import annotations

import itertools
import math
from typing import Mapping

import numpy as np

from .reference_prior import ReferenceShape, generate_reference_episode
from .schema import Episode
from .static_prior import (_Column, _Factory, _choice, _column, _dlu, _gam, _lu,
                           _rng, _ru, _standardizer, _unit)
from .selection_laws import ObservationModel, ResponseCompiler

CALIBRATION_ROWS = 4096
MECHANISM_WEIGHTS = {"R": .50, "forest": .20, "hierarchy": .15,
                     "smooth_local": .10, "sparse_interaction": .05}
OBSERVATION_WEIGHTS = {"identity": .50, "mcar": .15, "mar": .15,
                      "mnar": .10, "coarsen": .10}


def _seed(seed, *keys):
    return int(_rng(seed, *keys).integers(0, 2**63))


def _weights(values, defaults, name):
    if values is None:
        return dict(defaults)
    if not isinstance(values, Mapping) or set(values) - set(defaults):
        raise ValueError(f"Unknown {name} weights")
    result = {key: float(values.get(key, 0)) for key in defaults}
    if not all(np.isfinite(v) and v >= 0 for v in result.values()) or not np.isclose(sum(result.values()), 1., atol=1e-10, rtol=0):
        raise ValueError(f"{name} weights must be nonnegative and sum to one")
    return result


def _draw_features(factory, seed, split, rows):
    """Draw coordinates with row-prefix-stable mask/value streams."""
    latent = np.zeros((rows, factory.active), dtype=np.float64)
    raw = np.zeros((rows, factory.width), dtype=np.float64)
    tail = (np.sqrt(5 / _rng(seed, split, "feature_tail").chisquare(5, rows))
            if factory.heavy_tail else np.ones(rows))
    for block, indices in enumerate(factory.blocks):
        shared = _rng(seed, split, "feature_block", block).normal(size=rows)
        for j in indices:
            epsilon = _rng(seed, split, "feature_coordinate", int(j)).normal(size=rows)
            raw[:, j] = tail * (np.sqrt(factory.correlation) * shared + np.sqrt(1 - factory.correlation) * epsilon)
            latent[:, j] = raw[:, j]
    for j, col in enumerate(factory.columns):
        rng = _rng(seed, split, "feature_coordinate", j)
        if j < factory.active:
            if col.categorical:
                raw[:, j] = rng.choice(col.cardinality, size=rows, p=col.probabilities)
                latent[:, j] = col.effects[raw[:, j].astype(int)]
            continue
        source, noise = factory.extras[j - factory.active]
        if source is not None:
            raw[:, j] = raw[:, source]
            if col.categorical:
                mask = _rng(seed, split, "replacement_mask", j).random(rows) < .05
                replacement = _rng(seed, split, "replacement_value", j).choice(col.cardinality, size=rows, p=col.probabilities)
                raw[mask, j] = replacement[mask]
            else:
                raw[:, j] += rng.normal(0, noise, rows)
        elif col.categorical:
            raw[:, j] = rng.choice(col.cardinality, rows, p=col.probabilities)
        else:
            extra_tail = tail
            if factory.heavy_tail and not factory.common_distractor_scale:
                extra_tail = np.sqrt(5 / _rng(seed, split, "distractor_tail", j).chisquare(5, rows))
            raw[:, j] = rng.normal(size=rows) * extra_tail
    return latent, raw


class _HierarchicalFactory(_Factory):
    """P1's original active-type law with separated row-noise streams."""
    common_distractor_scale = False

    def draw(self, seed, split, rows):
        return _draw_features(self, seed, split, rows)


class _FeatureFactory(_HierarchicalFactory):
    """Same marginals as the authored factory, with an exact nominal count."""
    common_distractor_scale = True

    def __init__(self, rng, width, active):
        self.width, self.active = width, active
        self.fraction = _choice(rng, [0., .25, .5, 1.], [.35, .30, .25, .10])
        nominal = set(rng.choice(active, int(np.rint(self.fraction * active)), replace=False).tolist())
        self.columns = [_column(rng, j in nominal) for j in range(active)]
        numerical = rng.permutation([j for j, col in enumerate(self.columns) if not col.categorical])
        self.blocks = [numerical[i:i + 4] for i in range(0, len(numerical), 4)]
        self.correlation = _choice(rng, [0., .3, .8], [.4, .4, .2])
        self.heavy_tail = bool(rng.random() < .2)
        self.extras = []
        for _ in range(width - active):
            if rng.random() < .30:
                source = _ru(rng, 0, active - 1)
                self.columns.append(self.columns[source])
                self.extras.append((source, float(_lu(rng, .01, .3))))
            else:
                self.columns.append(_column(rng, rng.random() < self.fraction))
                self.extras.append((None, 0.))


def _proper_subset(rng, levels):
    """Uniform over the 2**C-2 nonempty proper subsets, not over sizes."""
    levels = np.asarray(levels)
    if len(levels) < 2:
        return None
    while True:
        keep = rng.random(len(levels)) < .5
        if keep.any() and not keep.all():
            return levels[keep]


def _forest(rng, calibration, raw, columns, outputs, support_rows, conditioned):
    d = calibration.shape[1]
    oblique = bool(rng.random() < .5 and d >= 2)
    count = int(_choice(rng, [1, 4, 16, 64]))
    occupancy = float(_lu(rng, 8, 128)) if conditioned else None
    depth = int(np.clip(np.floor(np.log2(max(1, support_rows / occupancy))), 1, 10)) if conditioned else _ru(rng, 1, 10)
    fallback = {"one_level_nodes": 0, "constant_projection_nodes": 0,
                "small_nodes": 0, "leaves": 0}

    def build(indices, remaining):
        if remaining == 0 or len(indices) < 16:
            fallback["leaves"] += 1
            fallback["small_nodes"] += int(remaining > 0)
            return rng.normal(size=outputs)
        if oblique:
            coordinates = rng.choice(d, _ru(rng, 2, min(4, d)), replace=False)
            coefficients = _unit(rng.normal(size=len(coordinates)))
            values = calibration[indices][:, coordinates] @ coefficients
            threshold = np.quantile(values, rng.uniform(.1, .9))
            predicate = ("oblique", coordinates, coefficients, threshold)
            left = values <= threshold
        else:
            j = _ru(rng, 0, d - 1)
            if columns[j].categorical:
                subset = _proper_subset(rng, np.unique(raw[indices, j]))
                if subset is None:
                    fallback["one_level_nodes"] += 1
                    fallback["leaves"] += 1
                    return rng.normal(size=outputs)
                predicate = ("nominal", j, subset)
                left = np.isin(raw[indices, j], subset)
            else:
                values = calibration[indices, j]
                threshold = np.quantile(values, rng.uniform(.1, .9))
                predicate = ("numeric", j, threshold)
                left = values <= threshold
        if not left.any() or left.all():
            fallback["constant_projection_nodes"] += 1
            fallback["leaves"] += 1
            return rng.normal(size=outputs)
        return predicate, build(indices[left], remaining - 1), build(indices[~left], remaining - 1)

    trees = [build(np.arange(len(calibration)), depth) for _ in range(count)]

    def score(z, x):
        result = np.zeros((len(z), outputs))
        def visit(node, indices):
            if not len(indices):
                return
            if isinstance(node, np.ndarray):
                result[indices] += node
                return
            predicate, left_tree, right_tree = node
            if predicate[0] == "nominal":
                left = np.isin(x[indices, predicate[1]], predicate[2])
            elif predicate[0] == "numeric":
                left = z[indices, predicate[1]] <= predicate[2]
            else:
                left = z[indices][:, predicate[1]] @ predicate[2] <= predicate[3]
            visit(left_tree, indices[left])
            visit(right_tree, indices[~left])
        for tree in trees:
            visit(tree, np.arange(len(z)))
        return result / np.sqrt(count)
    return score, {"subfamily": "oblique" if oblique else "axis_aligned", "trees": count,
                   "depth": depth, "target_leaf_occupancy": occupancy, **fallback}


def _spline_score(rng, calibration, outputs):
    standardize = _standardizer(calibration)
    c = standardize(calibration)
    knots, heights = [], []
    for j in range(c.shape[1]):
        count = int(_choice(rng, [4, 8, 16]))
        candidates = np.unique(np.quantile(c[:, j], np.linspace(0, 1, count)))
        # Deduplication precedes height draws. Keep a new knot only after 1e-6.
        selected = [candidates[0]]
        for value in candidates[1:]:
            if value - selected[-1] >= 1e-6:
                selected.append(value)
        knots.append(np.asarray(selected))
        heights.append(rng.normal(size=(len(selected), outputs)))
    coefficients = _unit(rng.normal(size=(c.shape[1], outputs)), axis=0)

    def score(z, raw=None):
        values = standardize(z)
        result = np.zeros((len(values), outputs))
        for j, (knot, height) in enumerate(zip(knots, heights)):
            if len(knot) == 1:
                basis = np.broadcast_to(height[0], result.shape)
            else:
                interval = np.clip(np.searchsorted(knot, values[:, j], side="right") - 1, 0, len(knot) - 2)
                fraction = (values[:, j] - knot[interval]) / (knot[interval + 1] - knot[interval])
                basis = height[interval] + fraction[:, None] * (height[interval + 1] - height[interval])
            result += basis * coefficients[j]
        return result
    return score, {"subfamily": "splines", "knot_counts": [len(k) for k in knots],
                   "minimum_knot_spacing": 1e-6}


def _squared_distance(a, b):
    return np.maximum(0., np.sum(a * a, axis=1)[:, None] + np.sum(b * b, axis=1)[None, :] - 2 * a @ b.T)


def _smooth(seed, rng, factory, calibration, outputs, support_rows):
    if rng.random() < .5:
        return _spline_score(rng, calibration, outputs)
    d = calibration.shape[1]
    rank = _dlu(rng, 1, min(8, d))
    projection = _unit(rng.normal(size=(d, rank)), axis=0)
    count = int(_choice(rng, [16, 32, 64, 128]))
    # Independent pools for anchors and density estimation, never query rows.
    anchors = factory.draw(seed, "rbf_anchors", count)[0] @ projection
    density = factory.draw(seed, "rbf_density", 512)[0] @ projection
    neighbors = float(_lu(rng, 4, 64))
    neighbor_rank = int(np.clip(np.rint(512 * neighbors / support_rows), 1, 511))
    distances = _squared_distance(density, density)
    np.fill_diagonal(distances, np.inf)
    bandwidth = max(1e-6, float(np.median(np.sqrt(np.partition(distances, neighbor_rank - 1, axis=1)[:, neighbor_rank - 1]))))
    coefficients = rng.normal(size=(count, outputs)) / np.sqrt(count)
    def score(z, raw=None):
        # Chunk large shapes to avoid allocating rows*anchors intermediates.
        result = np.empty((len(z), outputs))
        for start in range(0, len(z), 2048):
            projected = z[start:start + 2048] @ projection
            kernel = np.exp(-_squared_distance(projected, anchors) / (2 * bandwidth * bandwidth))
            result[start:start + 2048] = kernel @ coefficients
        return result
    realized = float(np.mean(np.sum(distances <= bandwidth * bandwidth, axis=1)) * support_rows / 512)
    return score, {"subfamily": "local_rbf", "intrinsic_rank": rank, "anchors": count,
                   "target_neighbors": neighbors, "density_neighbor_rank": neighbor_rank,
                   "bandwidth": bandwidth, "estimated_support_neighbors": realized,
                   "density_calibration_rows": 512, "anchors_independent": True}


def _subsets(rng, dimensions, order, count):
    available = math.comb(dimensions, order)
    count = min(count, available)
    if available <= 8:
        catalog = list(itertools.combinations(range(dimensions), order))
        return [catalog[int(i)] for i in rng.choice(available, count, replace=False)]
    selected = set()
    while len(selected) < count:
        selected.add(tuple(sorted(rng.choice(dimensions, order, replace=False).tolist())))
    return sorted(selected)


def _interactions(rng, calibration, raw, columns, outputs):
    d = calibration.shape[1]
    order = min(d, int(_choice(rng, [2, 3, 4, 6], [.45, .30, .20, .05])))
    subsets = _subsets(rng, d, order, _ru(rng, 1, 8))
    terms, kinds = [], []
    for subset in subsets:
        if rng.random() < .5:
            terms.append(("product", subset, None))
            kinds.append("product")
        else:
            rules = []
            for j in subset:
                if columns[j].categorical:
                    rules.append(("nominal", j, _proper_subset(rng, np.arange(columns[j].cardinality))))
                else:
                    rules.append(("numeric", j, float(np.quantile(calibration[:, j], rng.uniform(.2, .8)))))
            terms.append(("parity", subset, rules))
            kinds.append("parity")
    weights = _unit(rng.normal(size=(len(terms), outputs)), axis=0)
    def interaction(z, x):
        basis = np.empty((len(z), len(terms)))
        for t, (kind, subset, rules) in enumerate(terms):
            if kind == "product":
                basis[:, t] = np.prod(np.tanh(z[:, subset]), axis=1)
            else:
                parity = np.zeros(len(z), dtype=bool)
                for typ, j, value in rules:
                    bit = np.isin(x[:, j], value) if typ == "nominal" else z[:, j] > value
                    parity ^= bit
                basis[:, t] = 2 * parity.astype(float) - 1
        return basis @ weights
    # Independent linear additive coordinate weights; no shared interaction coefficients.
    additive_weights = _unit(rng.normal(size=(d, outputs)), axis=0)
    normalize_h = _standardizer(interaction(calibration, raw))
    normalize_g = _standardizer(calibration @ additive_weights)
    alpha = float(rng.uniform(.25, 1.))
    def score(z, x):
        return np.sqrt(alpha) * normalize_h(interaction(z, x)) + np.sqrt(1 - alpha) * normalize_g(z @ additive_weights)
    return score, {"order": order, "subsets": [list(s) for s in subsets], "term_kinds": kinds,
                   "interaction_share": alpha, "additive_law": "independent_unit_normal_linear"}


def _hierarchy(seed, rng, factory, pools, outputs, support_rows):
    extended = bool(rng.random() < .5)
    occupancy = float(_lu(rng, .5, 64)) if extended else None
    entities = int(np.clip(np.rint(support_rows / occupancy), 8, 262144)) if extended else _dlu(rng, 8, min(4096, max(16, 2 * support_rows)))
    organizations = _ru(rng, 2, min(32, max(2, entities // 2)))
    group = np.empty(entities, dtype=np.int64)
    group[rng.permutation(entities)] = np.arange(entities) % organizations
    frequencies = np.ones(entities)
    exponent = None
    if rng.random() >= .2:
        exponent = float(rng.uniform(.6, 1.8))
        frequencies = rng.permutation(np.arange(1, entities + 1)).astype(float) ** (-exponent)
    frequencies /= frequencies.sum()
    omega = np.empty(outputs); tau = np.empty(outputs); beta = np.empty((2, outputs))
    organization_effect = np.empty((organizations, outputs)); slope_sd = np.empty(outputs)
    for k in range(outputs):
        omega[k] = _choice(rng, [0., .5, 1.], [.3, .4, .3])
        tau[k] = _choice(rng, [0., .25, 1., 2.], [.15, .25, .4, .2])
        beta[:, k] = _unit(rng.normal(size=2))
        organization_effect[:, k] = rng.normal(0, _choice(rng, [0., .5, 1.], [.5, .3, .2]), organizations)
        slope_sd[k] = _choice(rng, [0., .25, .75], [.6, .3, .1])
    base = _gam(rng, pools["calibration"][0], outputs)
    identities = {split: _rng(seed, split, "hierarchy_ids").choice(entities, len(v), p=frequencies)
                  for split, (v, raw) in pools.items()}
    encountered = np.unique(np.concatenate(list(identities.values())))
    dimensions = min(2, factory.active)
    # Catalog frequency and group assignment are finite. Effects are materialized
    # only for encountered identities and keyed by identity, independent of rows.
    metadata = np.empty((len(encountered), 2))
    effects = np.empty((len(encountered), outputs))
    slopes = np.empty((len(encountered), dimensions, outputs))
    for index, entity in enumerate(encountered):
        erng = _rng(seed, "hierarchy_identity", int(entity))
        metadata[index] = erng.normal(size=2)
        effects[index] = tau * (np.sqrt(omega) * (metadata[index] @ beta) + np.sqrt(1 - omega) * erng.normal(size=outputs))
        slopes[index] = erng.normal(size=(dimensions, outputs)) * slope_sd
    raw_pools, scores = {}, {}
    for split, (v, raw) in pools.items():
        entity = identities[split]
        position = np.searchsorted(encountered, entity)
        scores[split] = base(v) + effects[position] + organization_effect[group[entity]] + np.einsum("nd,ndo->no", v[:, :dimensions], slopes[position])
        raw_pools[split] = np.column_stack([raw, entity, group[entity], metadata[position]])
    group_probabilities = np.bincount(group, weights=frequencies, minlength=organizations)
    columns = factory.columns + [_Column(True, frequencies), _Column(True, group_probabilities), _Column(False), _Column(False)]
    info = {"catalog_law": "support_occupancy" if extended else "legacy_p1", "target_mean_occupancy": occupancy,
            "entity_count": entities, "organization_count": organizations, "frequency_exponent": exponent,
            "materialized_entities": len(encountered), "lazy_identity_effects": True,
            "support_unique_entities": len(np.unique(identities["support"])),
            "unseen_query_entity_fraction": float(np.mean(~np.isin(identities["query"], identities["support"]))),
            "legacy_corruption_disabled": True}
    return raw_pools, scores, columns, info


def _render(seed, raw_pools, columns):
    rng = _rng(seed, "invertible_rendering")
    permutation = rng.permutation(len(columns))
    scales = _lu(rng, .1, 10., len(columns)) * rng.choice([-1., 1.], len(columns))
    maps = [(_rng(seed, "category_relabeling", j).permutation(col.cardinality)
             if col.categorical else None) for j, col in enumerate(columns)]
    rendered = {}
    for split, raw in raw_pools.items():
        value = raw.copy()
        for j, col in enumerate(columns):
            if col.categorical:
                value[:, j] = maps[j][raw[:, j].astype(np.int64)]
            else:
                value[:, j] *= scales[j]
        rendered[split] = value[:, permutation]
    categorical = np.asarray([col.categorical for col in columns])[permutation]
    levels = [(maps[j].copy() if columns[j].categorical else None) for j in permutation]
    info = {"column_permutation": permutation.tolist(), "numeric_scale_range": [.1, 10.],
            "signed_numeric_scales": scales[permutation].tolist(),
            "categorical_cardinalities": [len(v) if v is not None else 0 for v in levels]}
    return rendered, categorical, levels, info


def _new_episode(seed, family, task, ns, nq, width, classes, conditioned_probability, observation):
    rng = _rng(seed, "world")
    conditioned = bool(rng.random() < conditioned_probability)
    available = width - 4 if family == "hierarchy" else width
    if family == "hierarchy":
        active = _dlu(rng, 1, min(available, 16)) if rng.random() < .75 else _ru(rng, math.ceil(available / 3), available)
        factory = _HierarchicalFactory(rng, available, active)
        rho = None
    else:
        rho = float(_lu(rng, 8, 128)) if conditioned else None
        active = min(available, max(1, int(ns // rho))) if conditioned else _dlu(rng, 1, available)
        factory = _FeatureFactory(rng, available, active)
    pools = {split: factory.draw(seed, split, rows) for split, rows in
             (("calibration", CALIBRATION_ROWS), ("support", ns), ("query", nq))}
    outputs = 1 if task == "regression" else classes
    if family == "hierarchy":
        raw_pools, scores, columns, details = _hierarchy(seed, rng, factory, pools, outputs, ns)
    else:
        c, raw_c = pools["calibration"]
        if family == "forest":
            score, details = _forest(rng, c, raw_c, factory.columns, outputs, ns, conditioned)
        elif family == "smooth_local":
            score, details = _smooth(seed, rng, factory, c, outputs, ns)
        else:
            score, details = _interactions(rng, c, raw_c, factory.columns, outputs)
        scores = {split: score(z, raw) for split, (z, raw) in pools.items()}
        raw_pools = {split: raw for split, (z, raw) in pools.items()}
        columns = factory.columns
    if any(not np.all(np.isfinite(s)) for s in scores.values()):
        raise FloatingPointError("Nonfinite mechanism score")
    response = ResponseCompiler.fit(scores["calibration"], pools["calibration"][0],
                                    task=task, seed=_seed(seed, "response_parameters"), family=family)
    targets = {split: response.transform(scores[split], pools[split][0], seed=_seed(seed, "outcomes", split))
               for split in ("support", "query")}
    rendered, categorical, levels, rendering_info = _render(seed, raw_pools, columns)
    law = ObservationModel.fit(rendered["calibration"], categorical, seed=_seed(seed, "observation_parameters"),
                               mode=observation, category_levels=levels)
    observed = {split: law.transform(rendered[split], seed=_seed(seed, "observation_rows", split))
                for split in ("support", "query")}
    details.update(active_dimensions=active, context_conditioned=conditioned if family != "hierarchy" else None,
                   context_ratio=rho, active_nominal_count=sum(c.categorical for c in factory.columns[:active]),
                   numerical_correlation=factory.correlation, common_student_t5_scale=factory.heavy_tail)
    return Episode(observed["support"], targets["support"].y, observed["query"], targets["query"].y,
                   categorical, task, classes, encoding_seed=seed,
                   metadata={"family": family, "version": "selection-v1", "class_universe_declared_before_rows": True,
                             "query_acceptance_used": False, "calibration_rows": CALIBRATION_ROWS,
                             "mechanism": details, "response": response.diagnostics,
                             "response_draws": {split: batch.diagnostics for split, batch in targets.items()},
                             "observation": law.diagnostics, "rendering": rendering_info})


def generate_challenger_episode(seed: int, *, task: str = "classification", stage: int = 1,
                                shape: ReferenceShape | None = None, envelope: dict | None = None,
                                mechanism_weights=None, observation_weights=None,
                                complexity_conditioned_probability: float = .8,
                                namespace: str = "train", max_attempts: int = 100) -> Episode:
    """Sample selection-v1 after native task/shape, with fixed-source retries.

    ``task`` is the package-level classification/regression choice; training
    samples these 50/50 before calling this API. Explicit ``shape`` overrides
    are diagnostics and recorded. No native table is generated for new families.
    Observation draws are independent of source, including hierarchy fallback.
    """
    weights = _weights(mechanism_weights, MECHANISM_WEIGHTS, "mechanism")
    obs_weights = _weights(observation_weights, OBSERVATION_WEIGHTS, "observation")
    if not isinstance(namespace, str) or not namespace or len(namespace) > 128:
        raise ValueError("namespace must be a nonempty string of at most 128 characters")
    if not np.isfinite(complexity_conditioned_probability) or not 0 <= complexity_conditioned_probability <= 1:
        raise ValueError("complexity_conditioned_probability must be in [0,1]")
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    root = _seed(seed, "selection-v1", namespace)
    record = {}
    def callback(params, event):
        requested = _choice(_rng(root, "mechanism_choice"), list(weights), list(weights.values()))
        selected = "R" if requested == "hierarchy" and params["num_features"] < 8 else requested
        observation = _choice(_rng(root, "observation_choice"), list(obs_weights), list(obs_weights.values()))
        record.update(requested_mechanism=requested, selected_mechanism=selected,
                      requested_observation=observation, hierarchy_fallback=requested != selected,
                      namespace=namespace, seed=int(seed), native_shape_seed=root,
                      mechanism_weights=weights, observation_weights=obs_weights,
                      calibration_source="clean_support_x" if selected == "R" else "independent_world_rows",
                      failed_worlds=[], failed_world_diagnostics=[], version="selection-v1")
        event.update(branch_draw=requested, selected_source=selected,
                     ineligible_mass_returned=requested != selected, eligible=requested == selected,
                     requested_addition_probability=1 - weights["R"],
                     conditional_addition_probability=1 - weights["R"] - (weights["hierarchy"] if params["num_features"] < 8 else 0),
                     failed_world_diagnostics=record["failed_world_diagnostics"])
        if selected == "R":
            return None
        for attempt in range(max_attempts):
            event["raw_dataset_attempts"] += 1
            try:
                ep = _new_episode(_seed(root, "world_attempt", attempt), selected, event["task"],
                                  event["n_support"], event["n_query"], event["requested_features"],
                                  event["n_classes"], complexity_conditioned_probability, observation)
                record["raw_attempts"] = attempt + 1
                return ep
            except (FloatingPointError, np.linalg.LinAlgError) as exc:
                record["failed_worlds"].append(f"{type(exc).__name__}: {exc}")
                record["failed_world_diagnostics"].append({"error": str(exc), "details": getattr(exc, "diagnostics", None)})
        raise FloatingPointError(f"{selected} failed {max_attempts} fixed-source world attempts: {record['failed_worlds'][-1]}")
    ep = generate_reference_episode(root, arm="R", task=task, stage=stage, shape=shape,
                                    envelope=envelope, source_callback=callback)
    if record["selected_mechanism"] == "R":
        law = ObservationModel.fit(ep.x_support, ep.categorical, seed=_seed(root, "reference_observation_parameters"),
                                   mode=record["requested_observation"], calibration_source="clean_support")
        ep.x_support = law.transform(ep.x_support, seed=_seed(root, "reference_observation_rows", "support"))
        ep.x_query = law.transform(ep.x_query, seed=_seed(root, "reference_observation_rows", "query"))
        ep.metadata["observation"] = law.diagnostics
        record["raw_attempts"] = ep.metadata["reference_control"]["raw_dataset_attempts"]
    record["effective_observation"] = ep.metadata["observation"].get("effective_mode", ep.metadata["observation"].get("mode", record["requested_observation"]))
    ep.metadata["selection"] = record
    return ep
