"""Real-library sparse-class coverage and chronological validation contracts."""
import json
import sys

import numpy as np
import pytest

from tabular_foundation.baselines import (
    _frames, _parameters, _predict, _validate_snapshots, fit_candidate, main,
)
from tabular_foundation.metrics import multiclass_metrics


def table(y, seed=5):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(len(y), 3))
    x[:, 1] = rng.integers(0, 4, size=len(y))
    return {'X': x, 'y': np.asarray(y), 'categorical': np.array([False, True, False])}


@pytest.mark.parametrize('method', ['xgboost', 'lightgbm', 'catboost'])
@pytest.mark.parametrize('observed,validation', [([0, 2], [0, 1, 2]), ([1, 3, 4], [0, 1, 2, 3, 4])])
def test_real_trees_map_sparse_training_classes_and_keep_unseen_outcomes(method, observed, validation):
    pytest.importorskip(method)
    support = table(np.tile(observed, 32))
    valid = table(np.tile(validation, 4), seed=6)
    X, V = _frames(support, [valid], method)
    params = _parameters(method, np.random.default_rng(1), 0, 12, 3)
    model = fit_candidate(method, 'multiclass', params, X, support['y'], V, valid['y'], support['categorical'])
    probabilities = _predict(model, V, 'multiclass', 5)
    np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-6)
    unseen = np.setdiff1d(np.arange(5), observed)
    assert (probabilities[:, unseen] == 0).all()
    assert model.classes_.tolist() == observed
    missing_fraction = np.mean(~np.isin(valid['y'], observed))
    metric = multiclass_metrics(valid['y'], probabilities)
    assert metric['rows'] == len(valid['y'])
    assert metric['log_loss'] >= missing_fraction * -np.log(1e-15) - 1e-6


@pytest.mark.parametrize('method', ['xgboost', 'lightgbm', 'catboost'])
def test_no_validation_class_overlap_disables_early_stopping_without_dropping_rows(method):
    pytest.importorskip(method)
    support, valid = table(np.tile([0, 2], 32)), table(np.ones(8, dtype=int), seed=6)
    X, V = _frames(support, [valid], method)
    params = _parameters(method, np.random.default_rng(2), 0, 12, 7)
    model = fit_candidate(method, 'multiclass', params, X, support['y'], V, valid['y'], support['categorical'])
    probabilities = _predict(model, V, 'multiclass', 3)
    assert not model.validation_seen.any()
    assert (probabilities[:, 1] == 0).all()
    assert multiclass_metrics(valid['y'], probabilities)['log_loss'] == pytest.approx(-np.log(1e-15))


@pytest.mark.parametrize('method', ['xgboost', 'lightgbm', 'catboost'])
def test_regression_preserves_original_target_units(method):
    pytest.importorskip(method)
    support, valid = table(np.linspace(-10, 10, 64)), table(np.linspace(-5, 5, 8), seed=6)
    X, V = _frames(support, [valid], method)
    params = _parameters(method, np.random.default_rng(3), 0, 12, 8)
    model = fit_candidate(method, 'regression', params, X, support['y'], V, valid['y'], support['categorical'])
    predictions = _predict(model, V, 'regression', 0)
    assert predictions.shape == (8,)
    assert np.isfinite(predictions).all()


def test_constant_class_fallback_keeps_declared_class_slots():
    support, valid = table(np.full(6, 2)), table([0, 1, 2])
    model = fit_candidate('xgboost', 'multiclass', {}, support['X'], support['y'],
                          valid['X'], valid['y'], support['categorical'])
    probabilities = _predict(model, valid['X'], 'multiclass', 4)
    np.testing.assert_array_equal(probabilities, np.tile([0., 0., 1., 0.], (3, 1)))


def temporal_tables():
    snapshots = [table([0, 1], seed=i) for i in range(3)]
    for item, events, labels in zip(snapshots, ([0., 1.], [3., 4.], [7., 8.]),
                                   ([.5, 2.], [5., 6.], [9., 10.])):
        item['event_time'] = np.array(events)
        item['label_available_time'] = np.array(labels)
    return snapshots


def test_temporal_validation_labels_must_follow_events_and_precede_test():
    snapshots = temporal_tables()
    _validate_snapshots(*snapshots, 'binary', 2, 2.)
    for malformed in ([2.9, 6.], [5., 7.], [5., np.nan], [5.]):
        snapshots = temporal_tables()
        snapshots[1]['label_available_time'] = np.array(malformed)
        with pytest.raises(ValueError):
            _validate_snapshots(*snapshots, 'binary', 2, 2.)


def test_all_snapshot_labels_and_nonempty_windows_are_validated():
    snapshots = temporal_tables()
    snapshots[1]['y'][0] = 2
    with pytest.raises(ValueError, match='validation class'):
        _validate_snapshots(*snapshots, 'binary', 2, None)
    snapshots = temporal_tables()
    snapshots[2] = table([])
    with pytest.raises(ValueError, match='nonempty'):
        _validate_snapshots(*snapshots, 'binary', 2, None)
    with pytest.raises(ValueError, match='exactly two'):
        _validate_snapshots(*temporal_tables(), 'binary', 3, None)


def test_sparse_class_cli_reports_coverage_and_scores_full_test(tmp_path, monkeypatch):
    pytest.importorskip('xgboost')
    snapshots = [table(np.tile([0, 2], 32)), table([1, 3, 1, 3]), table([0, 1, 2, 3])]
    for name, data in zip(('train', 'validation', 'test'), snapshots):
        np.savez(tmp_path / (name + '.npz'), **data)
    output = tmp_path / 'result.json'
    monkeypatch.setattr(sys, 'argv', ['baselines', '--method', 'xgboost', '--task', 'multiclass',
        '--classes', '4', '--train', str(tmp_path / 'train.npz'),
        '--validation', str(tmp_path / 'validation.npz'), '--test', str(tmp_path / 'test.npz'),
        '--trials', '1', '--rounds', '12', '--output', str(output)])
    main()
    result = json.loads(output.read_text())
    assert result['class_coverage']['validation_unseen_class_rows'] == 4
    assert result['class_coverage']['test_unseen_class_rows'] == 2
    assert result['trials'][0]['early_stopping_validation_rows'] == 0
    assert result['test']['rows'] == 4
    assert result['test']['classes'] == 4
