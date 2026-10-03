"""Contract and distribution tests for the actual NumPy prior implementation."""
import unittest

import numpy as np

from tabular_foundation.schema import Episode
from tabular_foundation.static_prior import (
    SUBFAMILIES, _Column, _Renderer, _Response, _rng, _shape,
    generate_episode, sample_p4_outcomes,
)


class StaticPriorTests(unittest.TestCase):
    def test_all_selected_arms_and_task_heads_execute(self):
        arms = [("P0", subfamily) for subfamily in SUBFAMILIES] + [("P1", None), ("P4", None)]
        for family, subfamily in arms:
            for task in ("binary", "multiclass", "regression"):
                if family == "P4" and task == "multiclass":
                    continue
                with self.subTest(family=family, subfamily=subfamily, task=task):
                    episode = generate_episode(23, family, subfamily, task, 32, 16, 8)
                    self.assertEqual(episode.x_support.shape, (32, 8))
                    self.assertEqual(episode.x_query.shape, (16, 8))
                    self.assertTrue(np.isfinite(episode.y_query).all())
                    self.assertFalse(np.isinf(episode.x_support).any())
                    self.assertFalse(episode.metadata["query_acceptance_used"])

    def test_deterministic_replay_and_independent_query_stream(self):
        for family, subfamily in [("P0", "tree"), ("P0", "heterogeneous_scm"), ("P1", None), ("P4", None)]:
            args = dict(seed=77, family=family, subfamily=subfamily, task="binary", n_support=33, n_features=8)
            a, b = generate_episode(**args, n_query=16), generate_episode(**args, n_query=16)
            np.testing.assert_equal(a.x_support, b.x_support)
            np.testing.assert_equal(a.x_query, b.x_query)
            np.testing.assert_equal(a.y_support, b.y_support)
            np.testing.assert_equal(a.y_query, b.y_query)
            longer = generate_episode(**args, n_query=47)
            np.testing.assert_equal(a.x_support, longer.x_support)
            np.testing.assert_equal(a.y_support, longer.y_support)

    def test_loss_only_targets_and_hidden_laws_do_not_enter_forward(self):
        episode = generate_episode(9, "P0", "linear_gam", "binary", 32, 16, 8)
        before = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in episode.model_inputs().items()}
        episode.y_query[:] = 1 - episode.y_query
        episode.metadata["latent_conditional_law"] = np.full((16, 2), 12345.)
        episode.metadata["secret"] = "do not forward"
        after = episode.model_inputs()
        self.assertEqual(before.keys(), after.keys())
        self.assertFalse({"y_query", "metadata", "query_weights", "latent_conditional_law"} & after.keys())
        for key in before:
            np.testing.assert_equal(before[key], after[key])

    def test_scm_has_actual_observed_descendants_and_no_forward_label_law(self):
        episode = generate_episode(23, "P0", "heterogeneous_scm", "binary", 32, 16, 8)
        self.assertFalse(episode.metadata["null_override"])
        self.assertTrue(episode.metadata["observed_target_descendants"])
        self.assertIsNone(episode.metadata["latent_conditional_law"])
        target = episode.metadata["target_node"]
        parents = episode.metadata["parents"]
        reachable = {target}
        for i in range(target + 1, len(parents)):
            if any(p in reachable for p in parents[i]):
                reachable.add(i)
        self.assertTrue(set(episode.metadata["observed_target_descendants"]).issubset(reachable))

    def test_absent_classes_do_not_trigger_support_or_query_rejection(self):
        episode = generate_episode(104, "P0", "tree", "multiclass", 1, 128, 4)
        self.assertEqual(len(np.unique(episode.y_support)), 1)
        self.assertGreater(episode.n_classes, 2)
        self.assertEqual(len(episode.y_query), 128)
        self.assertTrue(episode.metadata["class_universe_declared_before_rows"])

    def test_categorical_codes_are_nominal_and_missing_values_are_explicit(self):
        found = False
        for seed in range(12):
            episode = generate_episode(seed, "P1", task="binary", n_support=32, n_query=16, n_features=8)
            self.assertGreaterEqual(episode.categorical.sum(), 2)
            values = episode.x_support[:, episode.categorical]
            observed = values[np.isfinite(values)]
            np.testing.assert_equal(observed, np.floor(observed))
            self.assertTrue(np.all(observed >= 0))
            np.testing.assert_equal(episode.metadata["support_missing"], np.isnan(episode.x_support))
            found |= bool(np.isnan(episode.x_support).any())
        self.assertTrue(found)

    def test_renderer_is_row_permutation_equivariant(self):
        calibration = np.arange(120, dtype=float).reshape(40, 3)
        calibration[:, 1] %= 3
        columns = [_Column(False), _Column(True, np.array([.2, .3, .5])), _Column(False)]
        renderer = _Renderer(_rng(10, "renderer"), calibration, columns)
        # Missingness is a row-index stochastic draw. Permutation equivariance
        # concerns the already sampled rows/masks; fixed deterministic maps are
        # independently checked here before that stochastic observation draw.
        permutation = np.random.default_rng(33).permutation(len(calibration))
        np.testing.assert_allclose(renderer._transform(calibration)[permutation], renderer._transform(calibration[permutation]))

    def test_response_probabilities_and_latent_law_are_consistent(self):
        rng = np.random.default_rng(1)
        calibration = rng.normal(size=(4096, 3))
        compiler = _Response(_rng(52, "params"), "multiclass", 3, 128)
        compiler.fit(_rng(52, "fit"), calibration, calibration)
        y, probabilities = compiler.draw(_rng(52, "labels"), calibration, calibration)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1., atol=1e-14)
        observed = np.bincount(y, minlength=3) / len(y)
        expected = probabilities.mean(axis=0)
        np.testing.assert_allclose(observed, expected, atol=.025)

    def test_p4_exact_distribution_moments(self):
        # Independent fixed-law Monte Carlo, rather than checking a generator
        # against its own returned labels. Tolerances are sampling-error scales.
        n, mu, kappa, zeta = 200000, 4., 2., .2
        rng = np.random.default_rng(551)
        count = sample_p4_outcomes(rng, np.full(n, mu), kappa, zeta)
        mean = (1 - zeta) * mu
        variance = (1 - zeta) * (mu + mu ** 2 / kappa) + zeta * (1 - zeta) * mu ** 2
        pzero = zeta + (1 - zeta) * (kappa / (kappa + mu)) ** kappa
        self.assertAlmostEqual(float(count.mean()), mean, delta=.04)
        self.assertAlmostEqual(float(count.var()), variance, delta=.3)
        self.assertAlmostEqual(float(np.mean(count == 0)), pzero, delta=.004)
        nu, shape = 3., 2.
        amount = sample_p4_outcomes(rng, np.full(n, mu), kappa, zeta, nu, shape)
        amount_variance = mean * nu ** 2 / shape + variance * nu ** 2
        self.assertAlmostEqual(float(amount.mean()), mean * nu, delta=.13)
        self.assertAlmostEqual(float(amount.var()), amount_variance, delta=3.5)
        self.assertAlmostEqual(float(np.mean(amount == 0)), pzero, delta=.004)
        poisson = sample_p4_outcomes(rng, np.full(n, mu), np.inf, 0)
        self.assertAlmostEqual(float(poisson.var()), mu, delta=.05)

    def test_curriculum_shapes_obey_cell_caps(self):
        for stage in (1, 2, 3):
            records = []
            for seed in range(512):
                ns, nq, width, classes, bucket = _shape(_rng(seed, "shape_test"), "P0", "multiclass", stage, None, None, None)
                records.append((ns, nq, width, classes, bucket))
                self.assertGreaterEqual(ns, 16 * classes)
                self.assertLessEqual((ns + nq) * width, {1: 2 ** 19, 2: 2 ** 21, 3: 2 ** 23}[stage])
            if stage == 3:
                # High class counts must lift even the short-row bucket's
                # support floor before feature widths are conditioned on cost.
                raised = [r for r in records if r[3] > 128 and r[4] == "short"]
                self.assertTrue(raised)
                self.assertTrue(all(r[0] == 16 * r[3] for r in raised))
                self.assertTrue(any(r[4] == "wide" and r[2] == 1024 for r in records))

    def test_stage_one_shape_replay_keeps_original_rng_consumption(self):
        # Frozen observations from the pre-curriculum implementation. Checking
        # the next raw RNG value also catches an unnecessary range-choice draw.
        snapshots = {
            0: ((1389, 128, 128, 9, "short"), 15018478757426936518),
            3: ((135, 33, 32, 4, "short"), 7048907006726445107),
            19: ((417, 104, 32, 3, "short"), 2006487260480983028),
            41: ((947, 128, 32, 4, "short"), 18397272051966023673),
        }
        for seed, (shape, next_raw) in snapshots.items():
            rng = _rng(seed, "shape_test")
            self.assertEqual(_shape(rng, "P0", "multiclass", 1, None, None, None), shape)
            self.assertEqual(int(rng.bit_generator.random_raw()), next_raw)

    def test_multiclass_curriculum_range_mass_and_within_range_uniformity(self):
        # Inspect emitted shape decisions rather than generating/calibrating
        # thousands of full datasets. Pearson checks assess all legal K values;
        # binomial checks separately protect the rare high-class range's mass.
        ranges = ((3, 10), (11, 32), (33, 128), (129, 256))
        count = 24000
        for stage, probabilities in ((2, (.70, .20, .10)), (3, (.70, .20, .08, .02))):
            rng = _rng(4109, "class_curriculum", stage)
            classes = np.array([_shape(rng, "P0", "multiclass", stage, 16, 8, 4)[3]
                                for _ in range(count)])
            self.assertGreaterEqual(classes.min(), 3)
            self.assertLessEqual(classes.max(), 128 if stage == 2 else 256)
            for (lo, hi), probability in zip(ranges, probabilities):
                selected = classes[(classes >= lo) & (classes <= hi)]
                expected_count = count * probability
                deviation = 6 * np.sqrt(count * probability * (1 - probability))
                self.assertAlmostEqual(len(selected), expected_count, delta=deviation)
                counts = np.bincount(selected - lo, minlength=hi - lo + 1)
                expected_per_class = len(selected) / len(counts)
                pearson = np.sum((counts - expected_per_class) ** 2 / expected_per_class)
                degrees = len(counts) - 1
                self.assertLess(pearson, degrees + 8 * np.sqrt(2 * degrees) + 16)

    def test_late_curriculum_replay_and_explicit_shape_overrides(self):
        for stage in (2, 3):
            for seed in range(20):
                arguments = ("P0", "multiclass", stage, None, None, None)
                self.assertEqual(_shape(_rng(seed, "late_replay"), *arguments),
                                 _shape(_rng(seed, "late_replay"), *arguments))
        # Explicit diagnostic overrides continue to bypass the sampled support
        # floor and cell cap; class selection still follows the stated stage.
        tiny = _shape(_rng(63, "shape"), "P0", "multiclass", 3, 4, 8, 4)
        self.assertGreater(tiny[3], 128)
        self.assertLess(tiny[0], 16 * tiny[3])
        large = _shape(_rng(63, "shape"), "P0", "multiclass", 3, 9000, 128, 1024)
        self.assertEqual(large[:3], (9000, 128, 1024))
        self.assertGreater((large[0] + large[1]) * large[2], 2 ** 23)

    def test_generated_many_class_episode_law_replay_and_loss_isolation(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is needed for the real forward/loss check")
        from tabular_foundation.model import build_model
        from tabular_foundation.train import episode_loss

        arguments = dict(seed=63, family="P0", subfamily="linear_gam", task="multiclass",
                         n_support=4, n_query=8, n_features=4, stage=3)
        episode = generate_episode(**arguments)
        replay = generate_episode(**arguments)
        self.assertGreater(episode.n_classes, 128)
        self.assertLessEqual(episode.n_classes, 256)
        self.assertEqual(episode.n_classes, replay.n_classes)
        for name in ("x_support", "y_support", "x_query", "y_query", "categorical"):
            np.testing.assert_equal(getattr(episode, name), getattr(replay, name))
        law = episode.metadata["latent_conditional_law"]
        self.assertEqual(law.shape, (8, episode.n_classes))
        self.assertTrue(np.isfinite(law).all() and np.all(law >= 0))
        np.testing.assert_allclose(law.sum(1), 1., atol=1e-13)
        for labels in (episode.y_support, episode.y_query):
            self.assertTrue(np.all((labels >= 0) & (labels < episode.n_classes)))

        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(63)
                model = build_model("tiny", finance=False).eval()
            with torch.no_grad():
                before = model(**episode.model_inputs())["logits"]
            self.assertEqual(tuple(before.shape), law.shape)
            logits = before.detach().clone().requires_grad_(True)
            loss, queries = episode_loss({"logits": logits}, episode, "cpu", conditional_labels=True)
            loss.backward()
            torch.testing.assert_close(logits.grad, logits.detach().softmax(-1) - torch.tensor(law, dtype=logits.dtype))
            self.assertEqual(queries, 8)

            # Permitted inputs are unchanged when scored outcomes and the
            # loss-only teacher law change, even with most support classes absent.
            episode.y_query[:] = (episode.y_query + 1) % episode.n_classes
            episode.metadata["latent_conditional_law"] = np.eye(episode.n_classes)[episode.y_query]
            with torch.no_grad():
                after = model(**episode.model_inputs())["logits"]
            torch.testing.assert_close(before, after, rtol=0, atol=0)
            hard, _ = episode_loss({"logits": after}, episode, "cpu")
            soft, _ = episode_loss({"logits": after}, episode, "cpu", conditional_labels=True)
            torch.testing.assert_close(hard, soft)
        finally:
            torch.set_num_threads(previous_threads)

    def test_context_candidate_preserves_exact_uncapped_replay_worlds(self):
        seed = next(s for s in range(100) if _rng(s, "context_conditioning").random() >= .9)
        for subfamily in SUBFAMILIES:
            base = generate_episode(seed, "P0", subfamily, "binary", 8, 16, 8)
            replay = generate_episode(seed, "P0", subfamily, "binary", 8, 16, 8, context_conditioning=True)
            self.assertFalse(replay.metadata["context_capped"])
            for name in ("x_support", "y_support", "x_query", "y_query", "categorical"):
                np.testing.assert_equal(getattr(base, name), getattr(replay, name))

    def test_invalid_requests_raise_without_family_substitution(self):
        for kwargs in ({"family": "P4", "task": "multiclass"}, {"family": "P1", "subfamily": "tree"},
                       {"family": "unknown"}, {"stage": 4}, {"family": "P1", "n_features": 4}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                generate_episode(1, **kwargs)

    def test_schema_rejects_invalid_class_or_importance_weights(self):
        args = dict(x_support=np.zeros((1, 4)), y_support=np.array([0]), x_query=np.zeros((1, 4)),
                    y_query=np.array([2]), categorical=np.zeros(4, bool), task="binary", n_classes=2)
        with self.assertRaises(ValueError):
            Episode(**args)
        args["y_query"] = np.array([1])
        with self.assertRaises(ValueError):
            Episode(**args, query_weights=np.array([-1.]))


if __name__ == "__main__":
    unittest.main()
