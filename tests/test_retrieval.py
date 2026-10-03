"""Exact observable retrieval invariants, independent of a trained model."""
import unittest

import numpy as np

from tabular_foundation.retrieval import (
    ReservoirCodec, assign_routes, farthest_first_centers, squared_distances,
    stable_keys, stream_nearest_to_centers, union_context,
)


class RetrievalTests(unittest.TestCase):
    def test_streaming_topk_matches_full_bruteforce_and_chunking(self):
        rng = np.random.default_rng(91)
        ids = rng.permutation(101).astype(np.int64)
        x = rng.normal(size=(101, 7))
        x[7:11] = x[0]  # exercise stable distance ties
        centers = np.vstack([x[0], rng.normal(size=(2, 7))])
        expected = [ids[np.lexsort((stable_keys(ids, 17), d))[:13]]
                    for d in squared_distances(x, centers).T]
        for chunk_size in (1, 8, 200):
            chunks = ((ids[k:k + chunk_size], x[k:k + chunk_size]) for k in range(0, len(ids), chunk_size))
            actual = stream_nearest_to_centers(chunks, centers, 13, tie_seed=17)
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a, b)

    def test_farthest_first_uses_capped_positive_set_then_all_existing_centers(self):
        p = np.asarray([70, 71, 72])
        px = np.asarray([[0., 0.], [10., 0.], [20., 0.]])
        r = np.asarray([70, 10, 20, 30])
        rx = np.asarray([[0., 0.], [2., 0.], [5., 0.], [30., 0.]])
        ids, centers = farthest_first_centers(p, px, r, rx, 4, 2, tie_seed=3)
        self.assertTrue(set(ids[:2]).issubset(set(p)))
        self.assertEqual(len(set(ids)), len(ids))
        self.assertTrue(set(ids[2:]).issubset(set(r)))
        for i in range(2, len(ids)):
            eligible = ~np.isin(r, ids[:i])
            distances = squared_distances(rx, centers[:i]).min(1)
            winner_distance = squared_distances(centers[i:i + 1], centers[:i]).min()
            self.assertAlmostEqual(winner_distance, distances[eligible].max(), places=12)

    def test_zero_distance_candidates_do_not_create_routes(self):
        ids, centers = farthest_first_centers(np.arange(3), np.ones((3, 4)),
                                              np.arange(3, 8), np.ones((5, 4)), 64, 16)
        self.assertEqual(len(ids), 1)
        self.assertEqual(centers.shape, (1, 4))

    def test_nearby_distinct_points_survive_distance_cancellation(self):
        x = np.asarray([[1., 1.], [1. + 1e-10, 1.]])
        distances = squared_distances(x, x)
        self.assertEqual(distances[0, 0], 0.)
        self.assertGreater(distances[0, 1], 0.)
        ids, _ = farthest_first_centers(np.empty(0, int), np.empty((0, 2)), np.arange(2), x, 2, 0)
        self.assertEqual(len(ids), 2)

    def test_routing_is_query_batch_and_permutation_invariant(self):
        rng = np.random.default_rng(7)
        centers, queries = rng.normal(size=(5, 8)), rng.normal(size=(23, 8))
        ids = np.asarray([9, 3, 20, 5, 7])
        full = assign_routes(queries, ids, centers, 41)
        single = np.concatenate([assign_routes(q[None], ids, centers, 41) for q in queries])
        np.testing.assert_array_equal(full, single)
        permutation = rng.permutation(len(queries))
        np.testing.assert_array_equal(assign_routes(queries[permutation], ids, centers, 41), full[permutation])

    def test_context_deduplicates_and_preserves_all_memberships(self):
        ids, bits, fit_indices = union_context(np.asarray([4, 1, 8]), np.asarray([8, 2]), np.asarray([4, 2, 9]))
        np.testing.assert_array_equal(ids, [4, 1, 8, 2, 9])
        np.testing.assert_array_equal(bits, [[1, 0, 1], [1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1]])
        np.testing.assert_array_equal(fit_indices, [0, 1, 2])

    def test_codec_uses_only_fitted_reservoir_and_exact_category_identity(self):
        r = np.asarray([[0., 100.], [1., 200.], [2., np.nan], [3., 100.]])
        codec = ReservoirCodec.fit(r, np.asarray([False, True]), seed=21)
        q = np.asarray([[1.5, 100.], [1.5, 101.], [np.nan, 100.]])
        full = codec.transform(q)
        self.assertFalse(np.allclose(full[0], full[1]))
        self.assertFalse(np.allclose(full[0], full[2]))
        np.testing.assert_allclose(codec.transform(q[:1]), full[:1], atol=0, rtol=0)
        r[:] = 999  # fitted reference owns its arrays
        np.testing.assert_allclose(codec.transform(q), full, atol=0, rtol=0)

    def test_empty_codec_context_and_zero_candidates_are_valid(self):
        codec = ReservoirCodec.fit(np.empty((0, 2)), np.asarray([False, True]))
        index = codec.transform(np.asarray([[12., 1.], [np.nan, np.nan]]))
        self.assertTrue(np.isfinite(index).all())
        np.testing.assert_array_equal(assign_routes(index, np.empty(0), np.empty((0, 128))), [0, 0])
        ids, bits, fit = union_context(np.empty(0, int), np.empty(0, int), np.empty(0, int))
        self.assertEqual(bits.shape, (0, 3))
        self.assertEqual(len(ids) + len(fit), 0)


if __name__ == "__main__":
    unittest.main()
