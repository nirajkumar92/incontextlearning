"""Independent probability, calibration, information-flow and batching checks."""
import copy
import inspect

import numpy as np
import pytest

from tabular_foundation.schema import Episode
from tabular_foundation.selection_laws import (
    CalibrationError, ObservationModel, ResponseCompiler, fit_class_intercepts,
    standardized_noise,
)


def test_constant_logits_have_the_known_exact_intercept_solution():
    masses = np.array([.001, .039, .16, .3, .5])
    logits = np.zeros((256, 5))
    intercepts, audit = fit_class_intercepts(logits, masses, tolerance=1e-12)
    expected = np.log(masses) - np.log(masses).mean()
    np.testing.assert_allclose(intercepts, expected, atol=1e-12)
    assert audit["maximum_absolute_error"] < 1e-12
    assert audit["total_variation"] < 1e-12
    assert abs(intercepts.sum()) < 1e-12


@pytest.mark.parametrize("scale", [.05, 1, 4, 12])
def test_newton_matches_nonuniform_masses_under_difficult_score_scales(scale):
    rng = np.random.default_rng(21)
    logits = scale * rng.normal(size=(4096, 7))
    masses = np.array([1e-5, .00009, .0009, .009, .09, .2, .7])
    intercepts, audit = fit_class_intercepts(logits, masses)
    shifted = logits + intercepts
    exponential = np.exp(shifted - shifted.max(axis=1, keepdims=True))
    achieved = (exponential / exponential.sum(axis=1, keepdims=True)).mean(axis=0)
    np.testing.assert_allclose(achieved, masses, atol=1e-3, rtol=0)
    assert audit["iterations"] <= 200 and audit["converged"]
    assert abs(intercepts.sum()) < 1e-10


def test_failed_calibration_is_explicit_and_carries_diagnostics():
    logits = np.array([[20., 0.], [10., 0.]])
    with pytest.raises(CalibrationError) as caught:
        fit_class_intercepts(logits, np.array([.5, .5]), max_iterations=0)
    assert caught.value.diagnostics["iterations"] == 0
    assert not caught.value.diagnostics["converged"]


@pytest.mark.parametrize("task,width", [("binary", 2), ("multiclass", 7), ("regression", 1)])
def test_response_batching_and_permutation_do_not_change_draws(task, width):
    rng = np.random.default_rng(34)
    z = rng.normal(size=(1024, 4))
    scores = rng.normal(size=(1024, width))
    compiler = ResponseCompiler.fit(scores[:800], z[:800], task=task, seed=9, family="forest")
    diagnostics = copy.deepcopy(compiler.diagnostics)
    ids = np.arange(224) + 4500
    whole = compiler.transform(scores[800:], z[800:], seed=110, row_ids=ids)
    first = compiler.transform(scores[800:827], z[800:827], seed=110, row_ids=ids[:27])
    second = compiler.transform(scores[827:], z[827:], seed=110, row_ids=ids[27:])
    np.testing.assert_array_equal(whole.y, np.r_[first.y, second.y])
    order = rng.permutation(len(ids))
    shuffled = compiler.transform(scores[800:][order], z[800:][order], seed=110, row_ids=ids[order])
    np.testing.assert_array_equal(whole.y[order], shuffled.y)
    if task != "regression":
        np.testing.assert_array_equal(whole.probabilities, np.concatenate([first.probabilities, second.probabilities]))
    assert compiler.diagnostics == diagnostics


def test_constant_scores_are_retained_as_a_zero_signal_world_without_class_floor_repair():
    compiler = ResponseCompiler.fit(np.full((4096, 9), 4.), np.ones((4096, 2)),
                                    task="multiclass", seed=23, family="smooth_local")
    result = compiler.transform(np.full((1, 9), 12345.), np.ones((1, 2)), seed=90)
    assert compiler.diagnostics["no_signal"] == [True] * 9
    np.testing.assert_allclose(result.probabilities[0], compiler.masses, atol=1e-12)
    assert len(result.diagnostics["absent_classes"]) == 8
    assert result.y.shape == (1,)


def test_support_is_independent_of_query_scores_noise_and_query_labels():
    rng = np.random.default_rng(27)
    z = rng.normal(size=(4096, 3))
    scores = np.c_[z[:, 0], z[:, 1], z[:, 2]]
    compiler = ResponseCompiler.fit(scores, z, task="multiclass", seed=10, family="sparse_interaction")
    support = compiler.transform(scores[:32], z[:32], seed=31)
    before = copy.deepcopy(compiler.diagnostics)
    query = compiler.transform(scores[:400] * 1e3, z[:400] * 10, seed=33)
    query.y[:] = 0
    query.probabilities[:] = .333
    again = compiler.transform(scores[:32], z[:32], seed=31)
    np.testing.assert_array_equal(support.y, again.y)
    np.testing.assert_array_equal(support.probabilities, again.probabilities)
    assert compiler.diagnostics == before


def test_response_audit_probabilities_and_query_outcomes_never_become_model_inputs():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(4096, 3))
    compiler = ResponseCompiler.fit(x[:, :2], x, task="binary", seed=10, family="forest")
    support = compiler.transform(x[:20, :2], x[:20], seed=40)
    query = compiler.transform(x[20:32, :2], x[20:32], seed=41)
    episode = Episode(x[:20], support.y, x[20:32], query.y, np.zeros(3, dtype=bool), "binary", 2,
                      metadata={"latent_probabilities": query.probabilities, "response_audit": compiler.diagnostics})
    inputs = episode.model_inputs()
    assert not {"y_query", "metadata", "latent_probabilities", "response_audit"}.intersection(inputs)
    copied = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in inputs.items()}
    query.y[:] = 1 - query.y
    query.probabilities[:] = -99
    for k, value in copied.items():
        np.testing.assert_equal(value, episode.model_inputs()[k])


@pytest.mark.parametrize("kind,variance_tolerance", [("gaussian", .015), ("student_t5", .03), ("lognormal", .07)])
def test_noise_population_mean_and_variance_are_standardized(kind, variance_tolerance):
    # Constants here follow the analytic moments of N(0,1), t5 and LN(0,1).
    noise = standardized_noise(kind, seed=450, row_ids=np.arange(400_000))
    assert abs(noise.mean()) < .012
    assert abs(noise.var() - 1) < variance_tolerance
    prefix = standardized_noise(kind, seed=450, row_ids=np.arange(150))
    np.testing.assert_array_equal(prefix, noise[:150])


def test_regression_additive_noise_scale_and_calibration_rms_are_exact():
    rng = np.random.default_rng(555)
    z = rng.normal(size=(4096, 5))
    scores = z[:, :1] * 3 + 9
    compiler = ResponseCompiler.fit(scores, z, task="regression", seed=8, family="forest")
    compiler.target_transform = "identity"  # isolate the specified pre-render noise law
    ids = np.arange(len(z)) + 100
    output = compiler.transform(scores, z, seed=81, row_ids=ids)
    multiplier = np.ones(len(z))
    if compiler.heteroscedastic:
        multiplier = np.exp(np.clip(z[:, compiler.noise_coordinates] @ compiler.noise_coefficients, -2, 2))
        multiplier /= compiler.noise_rms
        assert abs(np.mean(multiplier ** 2) - 1) < 1e-14
    expected_signal = ((scores - scores.mean(axis=0)) / scores.std(axis=0))[:, 0]
    noise = standardized_noise(compiler.noise_kind, seed=81, row_ids=ids)
    np.testing.assert_allclose(output.y - expected_signal, compiler.noise_scale * multiplier * noise, atol=1e-14)
    assert .03 <= compiler.noise_scale <= 1.
    assert output.probabilities is None


def test_hierarchy_cannot_draw_a_signed_square_transform():
    z = np.arange(256., dtype=float).reshape(128, 2) / 128
    transforms = set()
    for seed in range(40):
        compiler = ResponseCompiler.fit(z[:, :1], z, task="regression", seed=seed, family="hierarchy")
        transforms.add(compiler.target_transform)
    assert transforms == {"identity", "asinh"}


@pytest.mark.parametrize("mode", ["identity", "mcar", "mar", "mnar", "coarsen"])
def test_observation_is_fitted_once_batch_invariant_and_never_label_dependent(mode):
    rng = np.random.default_rng(1)
    calibration = rng.normal(size=(4096, 6))
    calibration[:, 2] = rng.integers(0, 7, len(calibration))
    category = np.array([False, False, True, False, False, False])
    levels = [None, None, np.arange(7), None, None, None]
    model = ObservationModel.fit(calibration, category, seed=55, mode=mode, category_levels=levels)
    diagnostics = copy.deepcopy(model.diagnostics)
    x = calibration[:140].copy()
    x[1, 0] = np.nan
    x_before = x.copy()
    ids = np.arange(140) + 1000
    whole = model.transform(x, seed=44, row_ids=ids)
    parts = np.r_[model.transform(x[:19], seed=44, row_ids=ids[:19]),
                  model.transform(x[19:], seed=44, row_ids=ids[19:])]
    np.testing.assert_array_equal(whole, parts)
    order = rng.permutation(len(x))
    np.testing.assert_array_equal(whole[order], model.transform(x[order], seed=44, row_ids=ids[order]))
    np.testing.assert_array_equal(x, x_before)
    assert model.diagnostics == diagnostics
    assert np.isnan(whole[1, 0])
    assert "y" not in inspect.signature(ObservationModel.fit).parameters
    assert "y" not in inspect.signature(ObservationModel.transform).parameters
    # Extreme future/query values cannot update the already-fitted state.
    future = calibration[:20].copy()
    future[:, ~category] *= 1e8
    model.transform(future, seed=900)
    np.testing.assert_array_equal(whole, model.transform(x, seed=44, row_ids=ids))
    assert model.diagnostics == diagnostics


@pytest.mark.parametrize("mode", ["mcar", "mar", "mnar"])
def test_mask_rates_match_calibration_and_mar_drivers_remain_observed(mode):
    rng = np.random.default_rng(15)
    calibration = rng.normal(size=(4096, 6))
    model = ObservationModel.fit(calibration, np.zeros(6, dtype=bool), seed=16, mode=mode)
    # Independent row noise over repeated calibration covariates isolates the
    # exact fitted probability average rather than population approximation.
    x = np.tile(calibration, (30, 1))
    observed = model.transform(x, seed=18)
    actual = np.isnan(observed[:, model.targets]).mean(axis=0)
    np.testing.assert_allclose(actual, model.rate, atol=.004, rtol=0)
    if mode == "mar":
        assert len(model.drivers) > 0
        assert not set(model.targets).intersection(model.drivers)
        np.testing.assert_array_equal(observed[:, model.drivers], x[:, model.drivers])
    for settings in model.columns.values():
        if "calibration_rate" in settings:
            assert abs(settings["calibration_rate"] - model.rate) < 2e-12


def test_one_column_mar_falls_back_to_mcar_with_an_explicit_audit():
    x = np.arange(200., dtype=float)[:, None]
    model = ObservationModel.fit(x, np.array([False]), seed=1, mode="mar", calibration_source="clean_support")
    assert model.mode == "mcar"
    assert model.diagnostics["fallback"] == "mar_width_one_to_mcar"
    assert model.diagnostics["calibration_source"] == "clean_support"
    assert model.targets.tolist() == [0]


def test_coarsening_uses_calibration_medians_and_constant_and_tied_columns_are_finite():
    x = np.r_[np.zeros(200), np.ones(30), np.full(20, 8.)][:, None]
    model = ObservationModel.fit(x, np.array([False]), seed=12, mode="coarsen")
    settings = model.columns[0]
    assert np.all(np.diff(settings["edges"]) > 0)
    assignment = np.searchsorted(settings["edges"], x[:, 0], side="right")
    for j, representative in enumerate(settings["representatives"]):
        members = x[assignment == j, 0]
        if len(members):
            assert representative == np.median(members)
        else:
            assert j in settings["empty_bins"]
            assert representative in x[:, 0]
    future = np.array([[-1e12], [1e12], [.5], [np.nan]])
    observed = model.transform(future, seed=1)
    assert np.isfinite(observed[:3]).all() and np.isnan(observed[-1])
    constant = ObservationModel.fit(np.full((30, 1), 7.), np.array([False]), seed=12, mode="coarsen")
    np.testing.assert_array_equal(constant.transform(future[:3], seed=2), np.full((3, 1), 7.))


@pytest.mark.parametrize("mode", ["mar", "mnar", "coarsen"])
def test_declared_nominal_catalog_relabeling_preserves_semantic_observation_law(mode):
    rng = np.random.default_rng(18)
    x = rng.integers(0, 9, size=(4096, 4)).astype(float)
    permutations = [rng.permutation(np.arange(9)) + 40 for _ in range(4)]
    relabeled = np.column_stack([permutations[j][x[:, j].astype(int)] for j in range(4)])
    model = ObservationModel.fit(x, np.ones(4, dtype=bool), seed=10, mode=mode,
                                 category_levels=[np.arange(9)] * 4)
    renamed = ObservationModel.fit(relabeled, np.ones(4, dtype=bool), seed=10, mode=mode,
                                   category_levels=permutations)
    output = model.transform(x, seed=44)
    output_renamed = renamed.transform(relabeled, seed=44)
    np.testing.assert_array_equal(np.isnan(output), np.isnan(output_renamed))
    if mode == "coarsen":
        np.testing.assert_array_equal(output[:, model.targets], output_renamed[:, model.targets])


def test_unknown_category_uses_frozen_effect_and_reserved_group_without_query_fitting():
    x = np.tile(np.arange(5), 100)[:, None].astype(float)
    model = ObservationModel.fit(x, np.array([True]), seed=1, mode="coarsen", calibration_source="clean_support")
    result = model.transform(np.array([[999.], [np.nan]]), seed=2)
    assert result[0, 0] == model.columns[0]["unseen_group"]
    assert np.isnan(result[1, 0])


@pytest.mark.parametrize("bad", [np.array([[np.inf]]), np.array([[np.nan]])])
def test_nonfinite_response_values_are_rejected(bad):
    with pytest.raises(FloatingPointError):
        ResponseCompiler.fit(bad, np.ones_like(bad), task="regression", seed=1, family="forest")


def test_binary_requires_two_scores_and_nominal_types_require_integer_codes():
    with pytest.raises(ValueError, match="Each requested class"):
        ResponseCompiler.fit(np.ones((4, 1)), np.ones((4, 1)), task="binary", seed=1, family="forest")
    with pytest.raises(ValueError, match="integer IDs"):
        ObservationModel.fit(np.array([[1.5], [2.]]), np.array([True]), seed=1, mode="mcar")


def test_large_nominal_catalog_audit_is_bounded_and_replayable():
    import json
    x = (np.arange(4096) % 1000)[:, None].astype(float)
    law = ObservationModel.fit(x, np.array([True]), seed=40, mode="coarsen",
                               category_levels=[np.arange(262144)])
    assert len(json.dumps(law.diagnostics)) < 2000
    audit = law.diagnostics["columns"]["0"]["mapping"]
    assert audit["size"] == 262144 and len(audit["sha256"]) == 64
    again = ObservationModel.fit(x, np.array([True]), seed=40, mode="coarsen",
                                 category_levels=[np.arange(262144)])
    assert law.diagnostics == again.diagnostics
    np.testing.assert_array_equal(law.transform(x, seed=1), again.transform(x, seed=1))


def test_extreme_finite_scores_fail_explicitly_instead_of_emitting_invalid_probabilities():
    x = np.array([[-1.], [1.]])
    compiler = ResponseCompiler.fit(np.c_[x, -x], x, task="binary", seed=40, family="forest")
    compiler.temperature = .25
    with pytest.raises(FloatingPointError):
        compiler.transform(np.array([[1e308, -1e308]]), np.ones((1, 1)), seed=1)
    with pytest.raises(FloatingPointError):
        ResponseCompiler.fit(np.array([[-1e308], [1e308]]), x,
                             task="regression", seed=40, family="forest")


def test_empty_observation_columns_preserve_missingness_and_transform_is_not_an_imputer():
    x = np.full((128, 1), np.nan)
    for mode in ("identity", "mcar", "mar", "mnar", "coarsen"):
        law = ObservationModel.fit(x, np.array([False]), seed=41, mode=mode)
        assert np.isnan(law.transform(x, seed=4)).all()
