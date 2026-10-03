"""Selection-v1 mechanism and integration contracts; no benchmark claims."""
from contextlib import contextmanager

import numpy as np
import pytest

from tabular_foundation import challenger_prior as cp
from tabular_foundation.reference_prior import (ReferenceGenerationError, ReferenceShape,
                                               generate_reference_episode)
from tabular_foundation.static_prior import _rng


def only(name):
    return {name: 1.}


def generate(family, task="classification", seed=41, **kwargs):
    options = dict(task=task, shape=ReferenceShape(24, 12, 12, 4 if task == "classification" else None),
                   mechanism_weights=only(family), observation_weights=only("identity"))
    options.update(kwargs)
    return cp.generate_challenger_episode(seed, **options)


@pytest.mark.parametrize("family", ["forest", "hierarchy", "smooth_local", "sparse_interaction"])
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_families_keep_shape_and_legal_class_universe(family, task):
    ep = generate(family, task)
    assert ep.x_support.shape == (24, 12) and ep.x_query.shape == (12, 12)
    assert ep.n_classes == (4 if task == "classification" else 0)
    assert ep.metadata["selection"]["selected_mechanism"] == family
    assert ep.metadata["calibration_rows"] == 4096
    assert not ep.metadata["query_acceptance_used"]
    assert not ep.metadata["native_reference_filter"]
    assert ep.metadata["reference_shape_override"] == dict(n_support=24, n_query=12, n_features=12, n_classes=ep.n_classes or None)
    assert np.isfinite(ep.x_support).all() and np.isfinite(ep.y_query).all()
    assert "mechanism" not in ep.model_inputs() and "response" not in ep.model_inputs()
    assert "y_query" not in ep.model_inputs()
    if family == "hierarchy":
        assert ep.categorical.sum() >= 2
        assert ep.metadata["mechanism"]["legacy_corruption_disabled"]
        assert ep.metadata["mechanism"]["lazy_identity_effects"]


@pytest.mark.parametrize("family", ["forest", "hierarchy", "smooth_local", "sparse_interaction"])
def test_support_and_query_prefix_do_not_depend_on_query_length(family):
    a = generate(family, seed=52)
    b = generate(family, seed=52, shape=ReferenceShape(24, 21, 12, 4))
    np.testing.assert_array_equal(a.x_support, b.x_support)
    np.testing.assert_allclose(a.y_support, b.y_support, atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(a.x_query, b.x_query[:12])
    np.testing.assert_allclose(a.y_query, b.y_query[:12], atol=1e-12, rtol=1e-12)
    assert a.metadata["rendering"] == b.metadata["rendering"]


def test_new_family_never_generates_a_native_table(monkeypatch):
    from tabular_foundation import reference_prior
    @contextmanager
    def forbidden(*args, **kwargs):
        raise AssertionError("native R table was generated and discarded")
        yield
    monkeypatch.setattr(reference_prior, "_observe_native", forbidden)
    ep = generate("forest")
    assert ep.metadata["selection"]["raw_attempts"] == 1
    assert ep.metadata["reference_control"]["graph_proposals"] == 0


def test_r_identity_matches_unmodified_reference_at_recorded_shape_seed():
    ep = generate("R", task="regression", seed=131)
    native = generate_reference_episode(ep.metadata["selection"]["native_shape_seed"], task="regression",
                                        shape=ReferenceShape(24, 12, 12))
    for attr in ("x_support", "x_query", "y_support", "y_query"):
        np.testing.assert_array_equal(getattr(ep, attr), getattr(native, attr))
    assert ep.metadata["native_reference_filter"]
    assert ep.metadata["selection"]["calibration_source"] == "clean_support_x"


def test_hierarchy_ineligible_mass_returns_to_r_after_shape():
    ep = generate("hierarchy", task="regression", seed=171,
                  shape=ReferenceShape(32, 16, 3))
    selection = ep.metadata["selection"]
    assert selection["requested_mechanism"] == "hierarchy"
    assert selection["selected_mechanism"] == "R"
    assert selection["hierarchy_fallback"]
    assert ep.metadata["reference_control"]["requested_features"] == 3


def test_native_requested_shape_law_is_identical_between_new_sources():
    kwargs = dict(task="regression", observation_weights=only("identity"))
    a = cp.generate_challenger_episode(651, mechanism_weights=only("forest"), **kwargs)
    b = cp.generate_challenger_episode(651, mechanism_weights=only("smooth_local"), **kwargs)
    for field in ("n_support", "n_query", "requested_features", "n_classes"):
        assert a.metadata["reference_control"][field] == b.metadata["reference_control"][field]
    assert a.metadata["reference_shape_override"] is None


@pytest.mark.parametrize("mode", ["identity", "mcar", "mar", "mnar", "coarsen"])
def test_all_observation_modes_integrate_without_changing_outcomes(mode):
    baseline = generate("forest", seed=231)
    episode = generate("forest", seed=231, observation_weights=only(mode))
    np.testing.assert_array_equal(baseline.y_support, episode.y_support)
    np.testing.assert_array_equal(baseline.y_query, episode.y_query)
    assert episode.metadata["selection"]["requested_observation"] == mode
    assert episode.metadata["selection"]["effective_observation"] == mode
    if mode in ("mcar", "mar", "mnar"):
        assert np.isnan(episode.x_support).any() or np.isnan(episode.x_query).any()
    assert not np.isinf(episode.x_support).any()


def test_namespace_changes_world_but_same_namespace_reproduces():
    a = generate("sparse_interaction", namespace="development")
    b = generate("sparse_interaction", namespace="development")
    c = generate("sparse_interaction", namespace="confirmation")
    np.testing.assert_array_equal(a.x_support, b.x_support)
    np.testing.assert_array_equal(a.y_query, b.y_query)
    assert not np.array_equal(a.x_support, c.x_support)


def test_missing_support_classes_are_retained():
    ep = generate("smooth_local", shape=ReferenceShape(2, 3, 4, 16))
    assert ep.n_classes == 16
    assert len(np.unique(ep.y_support)) <= 2
    assert ep.metadata["reference_control"]["class_split_rejections"] == 0


def test_fixed_source_retries_and_abort(monkeypatch):
    original = cp._new_episode
    seen = []
    def flaky(seed, family, task, ns, nq, width, classes, probability, observation):
        seen.append((seed, family, task, ns, nq, width, classes))
        if len(seen) == 1:
            raise FloatingPointError("diagnostic invalid world")
        return original(seed, family, task, ns, nq, width, classes, probability, observation)
    monkeypatch.setattr(cp, "_new_episode", flaky)
    ep = generate("forest", max_attempts=2)
    assert ep.metadata["selection"]["raw_attempts"] == 2
    assert len(ep.metadata["selection"]["failed_worlds"]) == 1
    assert seen[0][1:] == seen[1][1:] and seen[0][0] != seen[1][0]
    def invalid(*args, **kwargs):
        raise FloatingPointError("diagnostic invalid world")
    monkeypatch.setattr(cp, "_new_episode", invalid)
    with pytest.raises(ReferenceGenerationError, match="forest failed 2 fixed-source") as error:
        generate("forest", max_attempts=2)
    assert error.value.audit["events"][0]["selected_source"] == "forest"
    assert error.value.audit["events"][0]["raw_dataset_attempts"] == 2


def test_exact_active_nominal_count_and_unordered_ids():
    for seed in range(25):
        factory = cp._FeatureFactory(_rng(seed, "test"), 40, 33)
        assert sum(c.categorical for c in factory.columns[:33]) == round(factory.fraction * 33)
        z, x = factory.draw(seed, "diagnostic", 20)
        for j, column in enumerate(factory.columns[:33]):
            if column.categorical:
                np.testing.assert_array_equal(z[:, j], column.effects[x[:, j].astype(int)])
    # Zero-weight families and observation arms are legal but malformed laws fail.
    with pytest.raises(ValueError, match="sum to one"):
        generate("forest", mechanism_weights={"forest": .5})
    with pytest.raises(ValueError, match="Unknown"):
        generate("forest", observation_weights={"not_a_mode": 1})


def test_splines_keep_zero_signal_and_extrapolate_linearly():
    score, info = cp._spline_score(_rng(8, "spline"), np.ones((30, 2)), 3)
    assert info["knot_counts"] == [1, 1]
    result = score(np.ones((7, 2)))
    np.testing.assert_array_equal(result, np.broadcast_to(result[0], result.shape))
    c = np.arange(30, dtype=float)[:, None]
    score, info = cp._spline_score(_rng(9, "spline"), c, 1)
    outside = score(np.array([[-10.], [-11.], [-12.]]))[:, 0]
    np.testing.assert_allclose(outside[0] - outside[1], outside[1] - outside[2])


def test_proper_subset_is_uniform_over_subsets_and_never_ordinal_cut():
    rng = _rng(55, "subsets")
    counts = {}
    for _ in range(6000):
        subset = tuple(cp._proper_subset(rng, [10, 20, 30]))
        counts[subset] = counts.get(subset, 0) + 1
    assert len(counts) == 6
    assert all(800 < value < 1200 for value in counts.values())
    assert (10, 30) in counts
    assert cp._proper_subset(rng, [42]) is None


def test_interaction_subsets_do_not_materialize_huge_catalog():
    result = cp._subsets(_rng(61, "subsets"), 1024, 6, 8)
    assert len(result) == len(set(result)) == 8
    assert all(len(set(s)) == 6 for s in result)
    assert cp._subsets(_rng(62, "subsets"), 1, 1, 8) == [(0,)]


def test_factory_large_dimension_has_no_16_feature_cap():
    # Force broad replay; find a seed by its world draw, without outcome filtering.
    for candidate in range(100):
        root = cp._seed(candidate, "selection-v1", "train")
        world = cp._seed(root, "world_attempt", 0)
        rng = _rng(world, "world")
        rng.random()
        if cp._dlu(rng, 1, 64) > 16:
            break
    ep = generate("sparse_interaction", task="regression", seed=candidate,
                  shape=ReferenceShape(24, 12, 64), complexity_conditioned_probability=0.)
    assert ep.metadata["mechanism"]["active_dimensions"] > 16
    assert not ep.metadata["mechanism"]["context_conditioned"]


@pytest.mark.parametrize("family", ["forest", "smooth_local", "sparse_interaction"])
def test_new_mechanisms_accept_one_feature_and_report_mar_fallback(family):
    ep = generate(family, task="regression", shape=ReferenceShape(16, 8, 1),
                  observation_weights=only("mar"))
    assert ep.x_support.shape == (16, 1)
    assert ep.metadata["selection"]["requested_observation"] == "mar"
    assert ep.metadata["selection"]["effective_observation"] == "mcar"
    assert "response_draws" in ep.metadata
    assert ep.metadata["response_draws"]["support"]["target_sd"] >= 0


def test_large_hierarchy_materializes_only_encountered_identity_effects():
    # Unit diagnostic isolates the catalog law; 262144 is the declared catalog cap.
    # This does not train with a falsely declared million-row context.
    factory = cp._FeatureFactory(_rng(930, "factory"), 4, 2)
    pools = {split: factory.draw(930, split, rows) for split, rows in
             (("calibration", 128), ("support", 4), ("query", 3))}
    selected = next(s for s in range(100) if _rng(s, "hierarchy").random() < .5)
    raw, scores, columns, info = cp._hierarchy(930, _rng(selected, "hierarchy"), factory,
                                              pools, 3, 262144)
    assert info["entity_count"] > 4096
    assert info["materialized_entities"] <= 135
    assert len(columns) == 8
    extended_pools = dict(pools, query=factory.draw(930, "query", 31))
    other, other_scores, _, more = cp._hierarchy(930, _rng(selected, "hierarchy"), factory,
                                                extended_pools, 3, 262144)
    np.testing.assert_array_equal(raw["support"], other["support"])
    np.testing.assert_array_equal(scores["support"], other_scores["support"])
    np.testing.assert_array_equal(raw["query"], other["query"][:3])
    assert info["entity_count"] == more["entity_count"]


def test_render_is_invertible_and_preserves_semantic_catalog_order():
    rng = _rng(14, "columns")
    columns = [cp._column(rng, False), cp._column(rng, True)]
    raw = np.column_stack([np.linspace(-2, 2, columns[1].cardinality), np.arange(columns[1].cardinality)])
    rendered, categorical, levels, info = cp._render(21, {"calibration": raw, "support": raw}, columns)
    permutation = info["column_permutation"]
    for j, source in enumerate(permutation):
        if categorical[j]:
            np.testing.assert_array_equal(rendered["support"][:, j], levels[j])
            assert set(levels[j]) == set(range(columns[source].cardinality))
        else:
            np.testing.assert_allclose(rendered["support"][:, j] / info["signed_numeric_scales"][j], raw[:, source])
    assert np.isfinite(rendered["support"]).all()


def test_native_observation_reports_label_blind_support_calibration():
    ep = generate("R", task="regression", seed=139, observation_weights=only("coarsen"))
    assert ep.metadata["observation"]["calibration_source"] == "clean_support"
    assert ep.metadata["observation"]["calibration_rows"] == 24
