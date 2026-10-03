"""Actual tiny-model integration of observable finance selection and caches."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tabular_foundation.inference import FinancePredictor, Predictor
from tabular_foundation.model import build_model
from tabular_foundation.retrieval import squared_distances, stable_keys


def _data():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(70, 4))
    x[::7, 2] = np.nan
    y = np.zeros(len(x), dtype=np.int64)
    y[[5, 21, 59]] = 1
    query = rng.normal(size=(9, 4))
    return x, y, query, np.zeros(4, dtype=bool)


def test_finance_full_search_contexts_match_observable_brute_force():
    x, y, query, categorical = _data()
    predictor = FinancePredictor(build_model("tiny"), seed=12, reservoir=8, positive_cap=4, local=7,
                                 centers=4, positive_centers=2, candidate_chunk=9)
    predictor.fit_context(x, y, categorical, complete_cohort=True)
    negatives = np.flatnonzero(y == 0)
    distances = squared_distances(predictor.index_codec.transform(x[negatives]), predictor.centers)
    assert predictor.searched == len(negatives)
    assert set(predictor.P) == set(np.flatnonzero(y == 1))
    for route, local in enumerate(predictor.local_ids):
        order = np.lexsort((stable_keys(negatives, predictor.seed), distances[:, route]))[:7]
        np.testing.assert_array_equal(local, negatives[order])
        ids, bits, codec_indices = predictor.contexts[route]
        assert len(ids) == len(set(ids))
        assert set(ids[codec_indices]) == set(predictor.R)
        assert set(ids[bits[:, 1] == 1]) == set(predictor.P)
    np.testing.assert_allclose(predictor.reference_probs, np.array([67.5, 3.5]) / 71)


def test_finance_predictions_survive_cache_eviction_batching_and_sibling_changes():
    torch.manual_seed(1)
    torch.set_num_threads(1)
    x, y, query, categorical = _data()
    model = build_model("tiny")
    predictor = FinancePredictor(model, seed=5, reservoir=8, positive_cap=4, local=7,
                                 centers=4, positive_centers=2, cache_limit=1, query_batch_size=2)
    predictor.fit_context(x, y, categorical, complete_cohort=True)
    before = predictor.predict(query)
    assert len(predictor.caches) <= 1
    after = predictor.predict(query[::-1])[::-1]
    np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-6)
    predictor.query_batch_size = 1
    singleton = predictor.predict(query[:1])
    changed_siblings = query.copy()
    changed_siblings[1:] = 1000.
    modified = predictor.predict(changed_siblings)
    np.testing.assert_allclose(singleton[0], before[0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(modified[0], before[0], rtol=1e-5, atol=1e-6)


def test_full_amount_regression_does_not_define_every_nonzero_target_as_fraud():
    x, y, query, categorical = _data()
    amount = np.exp(np.nan_to_num(x[:, 0]))
    predictor = FinancePredictor(build_model("tiny"), reservoir=7, positive_cap=4, local=6, centers=3)
    predictor.fit_context(x, amount, categorical, task="regression", n_classes=0)
    assert predictor.M == 0
    assert predictor.searched == len(x)
    assert predictor.reference_probs is None
    np.testing.assert_equal(predictor._metadata(0)[[1, 6, 8, 9, 10, 11]], 0.)
    result = predictor.predict(query[:2])
    assert result.shape == (2,)
    assert np.isfinite(result).all()


def test_regression_serving_preserves_large_offset_small_spread_readout():
    torch.manual_seed(83)
    x, _, query, categorical = _data()
    y = 100000000. + .25*np.arange(len(x), dtype=np.float64)
    model = build_model("tiny").eval()
    predictor = Predictor(model, query_batch_size=2, bf16=False)
    predictor.fit_context(x, y, categorical, 'regression', n_classes=0)
    actual = predictor.predict(query[:3])
    with torch.inference_mode():
        expected = model.predict_cached(predictor.context, query[:3])['mean'].cpu().numpy()
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
    assert np.any(actual != actual.astype(np.float32).astype(np.float64))


def test_zero_positive_support_is_accepted_with_declared_binary_universe():
    x, y, query, categorical = _data()
    predictor = FinancePredictor(build_model("tiny"), reservoir=4, positive_cap=4, local=4, centers=2)
    predictor.fit_context(x[:10], np.zeros(10, dtype=int), categorical, complete_cohort=True)
    assert len(predictor.P) == 0
    result = predictor.predict(query[:2])
    assert result.shape == (2, 2)
    np.testing.assert_allclose(result.sum(axis=1), 1., atol=1e-6)
