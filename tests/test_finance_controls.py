"""Paired information controls and the finite X-first risk-partition law."""
import unittest

import numpy as np

from tabular_foundation.finance_prior import generate_finance_world
from tabular_foundation.retrieval import ReservoirCodec, squared_distances, stable_keys


def settings(**updates):
    result = dict(history_size=800, future_size=320, prevalence=.1, width=16,
                  complete_adjudication=True, reservoir_cap=24, positive_cap=12,
                  local_cap=24, max_centers=4, candidate_cap=-1, query_count=64,
                  calibration_rows=128, risk_campaigns=False, arrival_burst=False)
    result.update(updates)
    return result


class FinanceControlTests(unittest.TestCase):
    def test_views_pair_population_query_draws_and_mask_both_selector_and_model(self):
        for family in ("class_conditional", "risk_partition"):
            base = settings(mechanism_family=family)
            worlds = [generate_finance_world(32, task="binary", overrides=dict(base, feature_view=view))
                      for view in ("all", "hide_h", "h_only")]
            ids = np.arange(1120)
            full, hidden, only = [world.rows(ids) for world in worlds]
            hcolumn = worlds[0].renderer.permutation == 8
            self.assertEqual(len({w.population_id for w in worlds}), 1)
            self.assertEqual(len({w.world_id for w in worlds}), 3)
            self.assertEqual(worlds[0].strata, worlds[1].strata)
            self.assertEqual(worlds[0].strata, worlds[2].strata)
            np.testing.assert_allclose(hidden.x[:, ~hcolumn], full.x[:, ~hcolumn], equal_nan=True)
            self.assertTrue(np.isnan(hidden.x[:, hcolumn]).all())
            np.testing.assert_allclose(only.x[:, hcolumn], full.x[:, hcolumn])
            self.assertTrue(np.isnan(only.x[:, ~hcolumn]).all())
            macros = [world.sample_macroepisode(711) for world in worlds]
            for macro in macros[1:]:
                np.testing.assert_array_equal(macro.audit["finite_query_ids"], macros[0].audit["finite_query_ids"])
                np.testing.assert_array_equal(macro.audit["query_weights"], macros[0].audit["query_weights"])
            support = worlds[1].prepare_support()
            manual = full.x.copy()
            manual[:, hcolumn] = np.nan
            codec = ReservoirCodec.fit(manual[support.reservoir_ids], worlds[0].categorical, seed=32)
            np.testing.assert_allclose(codec.transform(manual), support.codec.transform(hidden.x))
            candidates = np.flatnonzero(hidden.eligible & (hidden.y == 0))
            distance = squared_distances(codec.transform(manual[candidates]), support.centers)
            for j, local in enumerate(support.local_ids):
                expected = candidates[np.lexsort((stable_keys(candidates, 32), distance[:, j]))[:24]]
                np.testing.assert_array_equal(local, expected)
            for route in macros[1].routes:
                self.assertTrue(np.isnan(route.x_support[:, hcolumn]).all())
                self.assertTrue(np.isnan(route.x_query[:, hcolumn]).all())
            # The H-only selector sees exactly two vectors, even though the
            # underlying worlds retain all other covariates and label delays.
            h_support = worlds[2].prepare_support()
            vectors = h_support.codec.transform(only.x)
            for value in (0, 1):
                subset = vectors[only.policy_flag == value]
                np.testing.assert_allclose(subset, np.broadcast_to(subset[0], subset.shape))

    def test_unit_entropy_pair_keeps_proposal_weights_and_population_risk(self):
        for family in ("class_conditional", "risk_partition"):
            base = settings(mechanism_family=family)
            unit = generate_finance_world(43, task="multiclass", overrides=dict(base, loss_normalization="unit"))
            entropy = generate_finance_world(43, task="multiclass", overrides=dict(base, loss_normalization="reference_entropy"))
            a, b = unit.sample_macroepisode(818), entropy.sample_macroepisode(818)
            self.assertEqual(unit.strata, entropy.strata)
            self.assertEqual(unit.population_id, entropy.population_id)
            np.testing.assert_array_equal(a.audit["finite_query_ids"], b.audit["finite_query_ids"])
            np.testing.assert_array_equal(a.audit["route_assignment"], b.audit["route_assignment"])
            np.testing.assert_array_equal(a.audit["query_weights"], b.audit["query_weights"])
            self.assertEqual(a.reference_normalization_weight, 1.)
            self.assertGreater(b.reference_normalization_weight, 1.)
            self.assertAlmostEqual(b.weighted_loss([np.ones_like(e.y_query) for e in b.routes]),
                a.weighted_loss([np.ones_like(e.y_query) for e in a.routes]) * b.reference_normalization_weight)
            # Exact integration of any ID-dependent loss over the finite class
            # proposal, including nonuniform class counts and missing classes.
            future = unit.rows(np.arange(unit.n_history, unit.n_history + unit.n_future))
            losses = 1 + np.sin(future.ids) ** 2 + future.y
            present = np.unique(future.y)
            expectation = sum((1 / len(present)) *
                (np.mean(future.y == label) * len(present)) * losses[future.y == label].mean()
                for label in present)
            self.assertAlmostEqual(expectation, losses.mean())

    def test_h_only_information_identity_and_nondiscriminative_control(self):
        pi, positive, negative = 1e-4, .8, .001
        joint = np.asarray([[(1 - pi) * (1 - negative), (1 - pi) * negative],
                            [pi * (1 - positive), pi * positive]])
        precision = joint[1, 1] / joint[:, 1].sum()
        self.assertAlmostEqual((1 + positive - negative) / 2, .8995)
        self.assertAlmostEqual(positive * precision + (1 - positive) * pi, .05928474636500325)
        world = generate_finance_world(61, task="binary", overrides=settings(
            mechanism_family="risk_partition", history_size=180000, future_size=60000,
            positive_h_probability=.15, negative_h_probability=.15))
        counts = np.zeros((2, 2), dtype=int)
        for stratum in world.strata:
            counts[int(stratum.truth_class > 0), stratum.policy_flag] += stratum.count
        rates = counts[:, 1] / counts.sum(1)
        np.testing.assert_allclose(rates, .15, atol=.006)
        self.assertAlmostEqual((1 + rates[1] - rates[0]) / 2, .5, delta=.004)
        # Different H laws are distinct generative worlds. In the new family
        # independent outcome namespaces keep day/cell/class counts matched.
        informative = generate_finance_world(61, task="binary", overrides=settings(
            mechanism_family="risk_partition", history_size=180000, future_size=60000,
            positive_h_probability=.8, negative_h_probability=.001))
        def outcome_counts(w):
            result = {}
            for s in w.strata:
                key = (s.day, s.risk_cell, s.truth_class)
                result[key] = result.get(key, 0) + s.count
            return result
        self.assertEqual(outcome_counts(world), outcome_counts(informative))
        self.assertNotEqual(world.population_id, informative.population_id)


class RiskPartitionTests(unittest.TestCase):
    def test_event_counts_fluctuate_with_binomial_not_fixed_count_variance(self):
        residuals, variances = [], []
        for seed in range(32):
            world = generate_finance_world(500 + seed, task="binary", overrides=settings(
                mechanism_family="risk_partition", history_size=3000, future_size=1000,
                prevalence=.003, risk_features=2, calibration_rows=16))
            n, p = world.risk_cell_counts[:90], world.risk_cell_probabilities[:90]
            actual = sum(s.count for s in world.strata if s.historical and s.truth_class > 0)
            residuals.append(actual - np.sum(n * p))
            variances.append(np.sum(n * p * (1 - p)))
        self.assertLess(abs(sum(residuals)), 4 * np.sqrt(sum(variances)))
        standardized_second_moment = np.mean(np.asarray(residuals) ** 2 / variances)
        self.assertGreater(standardized_second_moment, .35)
        self.assertLess(standardized_second_moment, 1.8)

    def test_large_population_exact_counts_expected_rarity_and_lazy_rows(self):
        world = generate_finance_world(72, task="binary", overrides=settings(
            mechanism_family="risk_partition", history_size=3_000_000, future_size=1_000_000,
            prevalence=1e-4, risk_features=4, risk_strength=3., candidate_cap=64))
        self.assertEqual(sum(s.count for s in world.strata), 4_000_000)
        np.testing.assert_array_equal(world.risk_cell_counts.sum(1), world.daily_counts)
        expectation = np.sum(world.risk_cell_counts[:90] * world.risk_cell_probabilities[:90])
        self.assertAlmostEqual(expectation, 300., places=9)
        counts = np.zeros_like(world.risk_cell_counts)
        for s in world.strata:
            counts[s.day + 90 if s.day < 0 else s.day + 89, s.risk_cell] += s.count
        np.testing.assert_array_equal(counts, world.risk_cell_counts)
        self.assertLess(len(world.strata), 20000)
        macro = world.sample_macroepisode(91)
        self.assertEqual(macro.audit["mechanism_family"], "risk_partition")
        self.assertFalse(macro.audit["population_resampled_for_coverage"])
        self.assertEqual(macro.routes[0].metadata["candidate_count"], 64)

    def test_given_cell_non_h_features_do_not_read_outcomes_or_policy(self):
        world = generate_finance_world(83, task="binary", overrides=settings(
            mechanism_family="risk_partition", risk_features=3, risk_rotate=False))
        ids = np.arange(120)
        day = np.full(120, -30)
        cells = ids % 8
        zero, one = np.zeros(120, dtype=int), np.ones(120, dtype=int)
        first = world._raw_features(ids, day, zero, zero, zero, cells)
        second = world._raw_features(ids, day, one, one, one, cells)
        non_h = np.arange(world.width) != 8
        np.testing.assert_array_equal(first[0][:, non_h], second[0][:, non_h])
        np.testing.assert_array_equal(first[1], second[1])
        np.testing.assert_array_equal(first[2], second[2])
        np.testing.assert_array_equal(first[1][:, :3] > 0, world.risk_bits[cells])
        self.assertGreater(np.ptp(world.risk_cell_probabilities[0]), .05)

    def test_missingness_reveal_and_keyed_replay_for_new_law(self):
        for missingness in ("MCAR", "channel_block", "value_dependent", "outage"):
            world = generate_finance_world(94, task="binary", overrides=settings(
                mechanism_family="risk_partition", missingness=missingness,
                feature_view="hide_h", complete_adjudication=False))
            ids = np.arange(1120)
            rows = world.rows(ids)
            declared = np.concatenate([np.full(s.count, s.historical and s.fraud_revealed) for s in world.strata])
            np.testing.assert_array_equal(rows.eligible, declared)
            np.testing.assert_array_equal(rows.eligible, (rows.event_day < 0) & (rows.label_available_at <= 0))
            permutation = np.random.default_rng(17).permutation(len(ids))
            replay = world.rows(ids[permutation])
            np.testing.assert_allclose(replay.x, rows.x[permutation], rtol=0, atol=0, equal_nan=True)
            np.testing.assert_array_equal(replay.label_available_at, rows.label_available_at[permutation])
            chunked = np.concatenate([world.rows(chunk).x for chunk in np.array_split(ids, 13)])
            np.testing.assert_allclose(chunked, rows.x, rtol=0, atol=0, equal_nan=True)
            reserved = np.isin(world.renderer.permutation, [9, 10, 11])
            self.assertTrue(np.isfinite(rows.x[:, reserved]).all())
            if missingness == "value_dependent":
                np.testing.assert_allclose(world.missing_achieved, world.missing_targets, atol=1e-12)

    def test_boundary_prevalence_and_strength_preserve_the_law(self):
        for pi in (0., .1, 1.):
            world = generate_finance_world(105, task="binary", overrides=settings(
                mechanism_family="risk_partition", prevalence=pi, risk_strength=0.,
                future_drift_multiplier=1.))
            np.testing.assert_allclose(world.risk_cell_probabilities, pi, atol=1e-15)
            if pi in (0., 1.):
                self.assertEqual(world.pool_count("future", int(pi)), world.n_future)
                macro = world.sample_macroepisode(2)
                np.testing.assert_array_equal(macro.audit["query_weights"], 1.)

    def test_invalid_controls_are_rejected(self):
        for override in ({"feature_view": "input_h_only"}, {"loss_normalization": "balanced"},
                         {"positive_h_probability": np.nan}, {"risk_features": 7},
                         {"risk_strength": -1.}, {"risk_interaction_fraction": 1.1}):
            with self.assertRaises(ValueError):
                generate_finance_world(2, task="binary", overrides=settings(mechanism_family="risk_partition", **override))


if __name__ == "__main__":
    unittest.main()
