import numpy as np
import pytest
from tabular_foundation.data import load_table, assert_time_contract, assert_validation_time_contract


def test_optional_target_and_timestamp_arrays_still_require_row_alignment(tmp_path):
    base = {'X': np.zeros((2, 3)), 'categorical': np.zeros(3, dtype=bool)}
    for extra in [{'y': np.zeros((2, 1))}, {'event_time': np.array([1])}, {'categorical': np.array([0, 1, 2])}]:
        path = tmp_path / 'invalid.npz'
        np.savez(path, **dict(base, **extra))
        with pytest.raises(ValueError):
            load_table(path, require_target=False)


def test_future_and_pre_event_support_labels_rejected():
    support = {'X': np.zeros((2, 1)), 'event_time': np.array([1., 2.]), 'label_available_time': np.array([2., 6.])}
    query = {'X': np.zeros((1, 1)), 'event_time': np.array([7.])}
    with pytest.raises(ValueError):
        assert_time_contract(support, query, 5)
    support['label_available_time'] = np.array([0., 3.])
    with pytest.raises(ValueError):
        assert_time_contract(support, query, 5)
    support['label_available_time'] = np.array([2., 3.])
    assert_time_contract(support, query, 5)


def test_validation_outcomes_must_be_known_at_test_decision():
    validation = {'X': np.zeros((2, 1)), 'event_time': np.array([3., 4.]), 'label_available_time': np.array([4., 5.])}
    query = {'X': np.zeros((1, 1)), 'event_time': np.array([6.])}
    assert_validation_time_contract(validation, query)
    for labels in [np.array([4., 6.]), np.array([2., 5.]), np.array([4., np.nan])]:
        validation['label_available_time'] = labels
        with pytest.raises(ValueError):
            assert_validation_time_contract(validation, query)
