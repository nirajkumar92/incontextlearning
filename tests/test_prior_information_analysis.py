"""Independent identities for CPU information probes, not downstream tests."""

import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest


_PATH = Path(__file__).resolve().parents[1] / "research" / "probes" / "prior_information_analysis.py"
_SPEC = importlib.util.spec_from_file_location("prior_information_analysis", _PATH)
probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe)


def test_random_effects_matches_precision_addition_and_limiting_cases():
    for n in (0, 1, 4, 100):
        result = probe.normal_random_effects(n, tau2=2, sigma2=3)
        assert result["bayes_theta_mse"] == pytest.approx(1 / (1 / 2 + n / 3))
        a = result["shrinkage_toward_sample_mean"]
        # Compute estimator MSE directly from independent signal/noise terms.
        direct_risk = (1 - a) ** 2 * 2 + (a ** 2 * 3 / n if n else 0)
        assert result["bayes_theta_mse"] == pytest.approx(direct_risk)
        assert result["bayes_new_response_mse"] == pytest.approx(direct_risk + 3)
    assert probe.normal_random_effects(0)["unpooled_sample_mean_theta_mse"] is None
    assert probe.normal_random_effects(10 ** 9)["shrinkage_toward_sample_mean"] > 0.999999


def test_occupancy_matches_exhaustive_support_enumeration():
    p = np.array([0.6, 0.3, 0.1])
    n = 3
    unseen_mass = distinct = 0.0
    for support in itertools.product(range(3), repeat=n):
        probability = np.prod(p[list(support)])
        observed = set(support)
        unseen_mass += probability * sum(p[j] for j in range(3) if j not in observed)
        distinct += probability * len(observed)
    result = probe.category_occupancy(p, n)
    assert result["expected_unseen_query_mass"] == pytest.approx(unseen_mass)
    assert result["expected_distinct_observed_categories"] == pytest.approx(distinct)
    assert result["expected_unseen_query_mass"] != pytest.approx(result["expected_fraction_of_identities_unseen"])
    assert probe.category_occupancy([1, 0], 0)["expected_unseen_query_mass"] == 1
    assert probe.category_occupancy([1, 0], 5)["expected_unseen_query_mass"] == 0


def test_coarsening_preserves_population_balance_but_hides_latent_certainty():
    result = probe.coarsened_observation()
    joint = np.array(result["joint_X_Y"])
    assert joint.sum(axis=0) == pytest.approx([0.5, 0.5])
    assert result["positive_probability_given_observed_X"][1] == pytest.approx(0.5)
    assert joint == pytest.approx(np.array([[0.25, 0], [0.25, 0.25], [0, 0.25]]))
    assert result["bayes_classification_error"] == pytest.approx(0.25)
    assert result["bayes_binary_brier_risk"] == pytest.approx(0.125)
    assert 0 < result["bayes_log_loss_observing_noisy_unquantized_W_nats"] < result["bayes_log_loss_nats"]
    assert result["bayes_log_loss_nats"] < np.log(2)


def test_affine_risk_matches_enumeration_of_parameter_posteriors():
    # Independently enumerate all 2^(k+1) latent affine functions and retain
    # those consistent with each possible support/label outcome. No GF(2)
    # rank formula is used in this reference calculation.
    k = 2
    cells = list(itertools.product((0, 1), repeat=k))
    functions = []
    for coefficients in itertools.product((0, 1), repeat=k + 1):
        functions.append([sum(a * x for a, x in zip(coefficients[:-1], cell)) % 2 ^ coefficients[-1] for cell in cells])
    functions = np.array(functions)
    for n in (0, 1, 2, 3):
        risk = 0.0
        for support in itertools.product(range(len(cells)), repeat=n):
            for true_labels in functions:
                consistent = np.all(functions[:, list(support)] == true_labels[list(support)], axis=1)
                posterior = functions[consistent].mean(axis=0)
                risk += np.minimum(posterior, 1 - posterior).mean()
        risk /= len(cells) ** n * len(functions)
        result = probe.affine_parity_risk(k, n)
        assert result["affine_parity_bayes_classification_error"] == pytest.approx(risk)
        assert sum(result["augmented_rank_probabilities"]) == pytest.approx(1)


def test_occupancy_is_not_a_universal_interaction_hardness_bound():
    result = probe.affine_parity_risk(12, 32)
    assert result["affine_parity_bayes_classification_error"] < 1e-6
    assert result["independent_truth_table_bayes_classification_error"] > 0.49
    assert probe.affine_parity_risk(12, 0)["affine_parity_bayes_classification_error"] == 0.5
    assert probe.in_gf2_span(0b101, probe.gf2_basis([0b011, 0b110]))
    assert not probe.in_gf2_span(0b001, probe.gf2_basis([0b011, 0b110]))


def test_uncertainty_is_not_zero_after_zero_observed_bernoulli_events():
    result = probe.sample_summary(np.zeros(100), bernoulli=True)
    assert result["mean"] == 0
    assert result["wilson_95_percent_interval"][1] > 0


def test_invalid_inputs_fail_and_small_run_is_reproducible():
    with pytest.raises(ValueError):
        probe.normal_random_effects(-1)
    with pytest.raises(ValueError):
        probe.normal_random_effects(1, sigma2=0)
    with pytest.raises(ValueError):
        probe.category_occupancy([0.2, 0.3], 5)
    with pytest.raises(ValueError):
        probe.affine_parity_risk(31, 5)
    with pytest.raises(ValueError):
        probe.run(replicates=1)
    assert probe.run(seed=6, replicates=20, occupancy_replicates=10, parity_replicates=10) == probe.run(seed=6, replicates=20, occupancy_replicates=10, parity_replicates=10)
