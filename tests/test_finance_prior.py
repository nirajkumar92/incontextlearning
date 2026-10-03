"""Finite-population, delayed-label and loss-boundary finance tests."""
import unittest

import numpy as np

from tabular_foundation.finance_prior import generate_finance_world
from tabular_foundation.retrieval import squared_distances, stable_keys


def tiny(**updates):
    values = dict(history_size=360, future_size=120, width=16, prevalence=.05,
                  reservoir_cap=24, positive_cap=12, local_cap=32, max_centers=4,
                  query_count=48, calibration_rows=128, scan_chunk_size=17)
    values.update(updates)
    return values


class FinancePriorTests(unittest.TestCase):
    def test_population_mass_and_finite_ids_are_exact(self):
        world = generate_finance_world(1, task="binary", overrides=tiny())
        self.assertEqual(sum(s.count for s in world.strata if s.historical), 360)
        self.assertEqual(sum(s.count for s in world.strata if not s.historical), 120)
        all_ids = np.concatenate([np.arange(s.start, s.start + s.count) for s in world.strata])
        np.testing.assert_array_equal(all_ids, np.arange(480))
        sampled = world.sample_ids("future", 120, np.random.default_rng(2))
        np.testing.assert_array_equal(np.sort(sampled), np.arange(360, 480))
        self.assertEqual(len(np.unique(sampled)), 120)
        with self.assertRaises(ValueError):
            world.sample_ids("future", 121, np.random.default_rng(2))

    def test_rows_replay_across_order_chunks_and_repeated_ids(self):
        world = generate_finance_world(2, task="multiclass", overrides=tiny(missingness="value_dependent"))
        ids = np.asarray([1, 2, 70, 200, 201, 359, 360, 470, 2])
        a = world.rows(ids)
        b = np.concatenate([world.rows(ids[i:i + 2]).x for i in range(0, len(ids), 2)])
        np.testing.assert_allclose(a.x, b, rtol=0, atol=0, equal_nan=True)
        permutation = np.asarray([8, 5, 4, 3, 2, 1, 0, 6, 7])
        replay = world.rows(ids[permutation])
        np.testing.assert_allclose(replay.x, a.x[permutation], rtol=0, atol=0, equal_nan=True)
        np.testing.assert_array_equal(replay.y, a.y[permutation])
        np.testing.assert_array_equal(replay.label_available_at, a.label_available_at[permutation])

    def test_unknown_outcomes_never_become_negatives(self):
        world = generate_finance_world(3, task="binary", overrides=tiny(
            complete_adjudication=False, audit_probability=0., review_probability_h0=0.,
            review_probability_h1=0., self_report_probability=1.))
        self.assertGreater(world.n_eligible, 0)
        self.assertEqual(world.pool_count("negative"), 0)
        self.assertEqual(world.n_eligible, world.m_eligible)
        self.assertLess(world.n_eligible, world.n_history)
        macro = world.sample_macroepisode(4)
        for episode in macro.routes:
            self.assertTrue(np.all(episode.y_support == 1))
            self.assertIsNone(episode.reference_probs)
            self.assertEqual(episode.finance_metadata[10], 0)

    def test_zero_positives_not_retried_or_fabricated(self):
        world = generate_finance_world(4, task="binary", overrides=tiny(prevalence=0., complete_adjudication=True))
        self.assertEqual(world.m_eligible, 0)
        self.assertEqual(world.pool_count("future", 1), 0)
        macro = world.sample_macroepisode(5)
        self.assertFalse(macro.audit["population_resampled_for_coverage"])
        for episode in macro.routes:
            self.assertTrue(np.all(episode.y_support == 0))
            self.assertTrue(np.all(episode.y_query == 0))
            np.testing.assert_array_equal(episode.query_weights, 1.)
            self.assertEqual(episode.finance_metadata[9], 1.)

    def test_empty_eligible_history_emits_empty_support(self):
        world = generate_finance_world(5, task="binary", overrides=tiny(
            complete_adjudication=False, audit_probability=0., review_probability_h0=0.,
            review_probability_h1=0., self_report_probability=0.))
        self.assertEqual(world.n_eligible, 0)
        macro = world.sample_macroepisode(6)
        self.assertEqual(len(macro.routes), 1)
        route = macro.routes[0]
        self.assertEqual(route.x_support.shape, (0, 16))
        self.assertEqual(route.source_bits.shape, (0, 3))
        self.assertEqual(len(route.codec_indices), 0)
        self.assertTrue(np.isfinite(route.finance_metadata).all())
        self.assertIsNone(route.reference_probs)
        self.assertEqual(macro.reference_normalization_weight, 1.)

    def test_reveal_conditioned_strata_and_maturity_boundary(self):
        for complete in (False, True):
            world = generate_finance_world(6, task="multiclass", overrides=tiny(
                complete_adjudication=complete, maturity_days=30, prevalence=.2))
            rows = world.rows(np.arange(world.n_history))
            declared = np.concatenate([np.full(s.count, s.fraud_revealed) for s in world.strata if s.historical])
            np.testing.assert_array_equal(rows.fraud_available_at <= 0, declared)
            np.testing.assert_array_equal(rows.eligible, declared)
            if complete:
                self.assertTrue(np.all(rows.eligible[rows.event_day <= -30]))
            eligible_ids = world.sample_ids("eligible", world.n_eligible, np.random.default_rng(8))
            self.assertTrue(world.rows(eligible_ids).eligible.all())

    def test_reference_uses_complete_old_cohort_not_selected_feedback(self):
        world = generate_finance_world(7, task="multiclass", overrides=tiny(
            complete_adjudication=True, maturity_days=60, prevalence=.3,
            review_probability_h0=1., review_probability_h1=1.))
        rows = world.rows(np.arange(world.n_history))
        reference = rows.event_day <= -60
        counts = np.bincount(rows.y[reference], minlength=world.n_classes)
        np.testing.assert_allclose(world.reference_probs, (counts + .5) / (reference.sum() + .5 * world.n_classes))
        self.assertEqual(world.n_reference, reference.sum())
        self.assertEqual(world.reference_kind, 1)
        self.assertEqual(world.n_eligible, world.n_history)
        self.assertGreater(world.reference_normalization_weight, 1.)
        route = world.sample_macroepisode(72).routes[0]
        self.assertEqual(route.finance_metadata.shape, (14,))
        self.assertAlmostEqual(route.finance_metadata[13], np.log1p(reference.sum()))

    def test_all_task_normalization_is_legal_reference_function(self):
        for task, extra in [("binary", {}), ("multiclass", {}),
                            ("regression", {"regression_kind": "realized_fraud_loss"})]:
            world = generate_finance_world(8, task=task, overrides=tiny(complete_adjudication=True, **extra))
            pi = world.reference_prevalence
            expected = 1 / max(-pi * np.log(pi) - (1 - pi) * np.log1p(-pi), 1e-4)
            self.assertAlmostEqual(world.reference_normalization_weight, expected)
        full = generate_finance_world(8, task="regression", overrides=tiny(
            complete_adjudication=True, regression_kind="full_amount"))
        self.assertEqual(full.reference_normalization_weight, 1.)

    def test_query_weights_are_global_before_routing_and_reduction_is_not_group_mean(self):
        world = generate_finance_world(9, task="multiclass", overrides=tiny(prevalence=.35, complete_adjudication=True))
        macro = world.sample_macroepisode(10, query_count=113)
        counts = np.asarray([world.pool_count("future", c) for c in range(world.n_classes)])
        present = np.count_nonzero(counts)
        losses = []
        expected = 0.
        for route in macro.routes:
            expected_weights = counts[route.y_query] / world.n_future * present
            np.testing.assert_allclose(route.query_weights, expected_weights)
            loss = 1 + route.metadata["original_query_positions"] / 100.
            losses.append(loss)
            expected += np.dot(expected_weights, loss)
        self.assertEqual(sum(len(e.y_query) for e in macro.routes), 113)
        self.assertAlmostEqual(macro.weighted_loss(losses), expected / 113 * macro.reference_normalization_weight)
        # The weights integrate each finite class exactly in expectation.
        self.assertAlmostEqual(sum((1 / present) * (counts[c] / world.n_future * present)
                                   for c in np.flatnonzero(counts)), 1.)

    def test_full_candidate_search_matches_observable_bruteforce(self):
        world = generate_finance_world(10, task="binary", overrides=tiny(complete_adjudication=True, candidate_cap=-1))
        support = world.prepare_support(training=False)
        ids = world.sample_ids("negative", world.pool_count("negative"), np.random.default_rng(1))
        vectors = support.codec.transform(world.rows(ids).x)
        distances = squared_distances(vectors, support.centers)
        self.assertEqual(support.candidate_count, len(ids))
        self.assertEqual(support.candidate_mode, "full_eligible_pool")
        for j, local in enumerate(support.local_ids):
            expected = ids[np.lexsort((stable_keys(ids, world.seed), distances[:, j]))[:32]]
            np.testing.assert_array_equal(local, expected)

    def test_candidate_sample_is_declared_and_reuse_does_not_add_labels(self):
        world = generate_finance_world(11, task="binary", overrides=tiny(complete_adjudication=True, candidate_cap=19))
        support = world.prepare_support()
        self.assertEqual(support.candidate_count, 19)
        self.assertEqual(support.candidate_mode, "declared_uniform_candidate_sample")
        self.assertIs(support, world.prepare_support())
        snapshots = [world.sample_macroepisode(seed) for seed in (20, 21, 22, 23)]
        self.assertEqual(len({m.world_id for m in snapshots}), 1)
        self.assertTrue(all(m.audit["observed_event_count"] == world.m_eligible for m in snapshots))
        context_union = set()
        for macro in snapshots:
            for route in macro.routes:
                context_union.update(route.metadata["context_ids"].tolist())
        self.assertTrue(world.rows(np.asarray(sorted(context_union))).eligible.all())

    def test_missingness_keeps_reserved_decision_features_observable(self):
        for missingness in ("MCAR", "channel_block", "value_dependent", "outage"):
            world = generate_finance_world(12, task="binary", overrides=tiny(missingness=missingness, missing_rate=1.))
            rows = world.rows(np.arange(world.n_history + world.n_future))
            reserved = np.flatnonzero(np.isin(world.renderer.permutation, np.arange(8, 12)))
            self.assertTrue(np.isfinite(rows.x[:, reserved]).all())
            if missingness == "MCAR":
                ordinary = np.flatnonzero(~np.isin(world.renderer.permutation, np.arange(8, 12)))
                self.assertTrue(np.isnan(rows.x[:, ordinary]).all())
            if missingness == "value_dependent":
                np.testing.assert_allclose(world.missing_achieved, world.missing_targets, atol=1e-12)

    def test_full_amount_target_and_hurdle_observation_contracts(self):
        full = generate_finance_world(13, task="regression", overrides=tiny(regression_kind="full_amount"))
        self.assertEqual(full.n_eligible, full.n_history)
        m = full.sample_macroepisode(14)
        for route in m.routes:
            self.assertFalse(route.hurdle)
            self.assertTrue(np.all(route.y_support > 0))
            np.testing.assert_array_equal(route.finance_metadata[[1, 6, 8, 9, 10, 11]], 0.)
            self.assertEqual(route.finance_metadata[13], 0.)
            self.assertIsNone(route.reference_probs)
        hurdle = generate_finance_world(13, task="regression", overrides=tiny(
            regression_kind="realized_fraud_loss", complete_adjudication=False,
            audit_probability=0., review_probability_h0=0., review_probability_h1=0., self_report_probability=0.))
        h = hurdle.sample_macroepisode(15)
        self.assertEqual(hurdle.n_eligible, 0)
        self.assertTrue(h.routes[0].hurdle)
        self.assertTrue(np.all(h.routes[0].y_query >= 0))
        self.assertIsNone(h.routes[0].reference_probs)

    def test_forward_allowlist_excludes_query_labels_weights_and_latents(self):
        world = generate_finance_world(14, task="binary", overrides=tiny(complete_adjudication=True))
        route = world.sample_macroepisode(15).routes[0]
        before = route.model_inputs()
        forbidden = {"y_query", "query_weights", "metadata", "prevalence", "future_class_counts", "query_ids"}
        self.assertFalse(set(before) & forbidden)
        route.y_query[:] = 1 - route.y_query
        route.query_weights[:] = 999
        route.metadata["prevalence"] = .999
        after = route.model_inputs()
        self.assertEqual(set(before), set(after))
        for key in before:
            if isinstance(before[key], np.ndarray):
                np.testing.assert_allclose(before[key], after[key], equal_nan=True)

    def test_natural_evaluation_queries_are_unique_and_unweighted(self):
        world = generate_finance_world(15, task="binary", overrides=tiny(complete_adjudication=True))
        macro = world.sample_macroepisode(16, training=False, query_count=100)
        self.assertTrue(macro.audit["natural_unique_queries"])
        self.assertEqual(len(np.unique(macro.audit["finite_query_ids"])), 100)
        np.testing.assert_array_equal(macro.audit["query_weights"], 1.)
        with self.assertRaises(ValueError):
            world.sample_macroepisode(16, training=False, query_count=121)

    def test_explicit_subtarget_rarity_boundary_keeps_three_million_rows_lazy(self):
        world = generate_finance_world(1701, task="binary", overrides=tiny(
            history_size=3_000_000, future_size=1_000_000, prevalence=1e-6,
            candidate_cap=256, reservoir_cap=16, positive_cap=8, local_cap=16))
        self.assertEqual(sum(s.count for s in world.strata), 4_000_000)
        self.assertLess(len(world.strata), 10000)
        macro = world.sample_macroepisode(1702)
        self.assertLessEqual(world.m_eligible, sum(s.count for s in world.strata if s.historical and s.truth_class > 0))
        self.assertEqual(macro.routes[0].metadata["candidate_count"], 256)
        self.assertEqual(macro.audit["source_population_size"], 3_000_000)
        self.assertLessEqual(max(len(e.y_support) for e in macro.routes), 16 + 8 + 16)
        if world.pool_count("future", 1) == 0:
            self.assertTrue(all(np.all(e.y_query == 0) for e in macro.routes))

    def test_default_prevalence_intervals_center_on_one_in_ten_thousand(self):
        frequencies = np.zeros(3, dtype=int)
        for seed in range(120):
            # Small populations/calibration accelerate this parameter-law test;
            # prevalence itself is not overridden.
            settings = tiny(history_size=30, future_size=10, calibration_rows=16)
            del settings["prevalence"]
            world = generate_finance_world(seed, task="binary", overrides=settings)
            log_pi = np.log10(world.prevalence)
            self.assertGreaterEqual(log_pi, -5.)
            self.assertLess(log_pi, -2.)
            frequencies[0 if log_pi < -4.5 else (1 if log_pi < -3.5 else 2)] += 1
        self.assertGreater(frequencies[1], frequencies[0])
        self.assertGreater(frequencies[1], frequencies[2])


if __name__ == "__main__":
    unittest.main()
