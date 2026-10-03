import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from tabular_foundation.metrics import binary_metrics, select_threshold, blocked_bootstrap


def test_ap_and_auc_with_ties_match_reference():
    y = np.array([1, 0, 0, 1, 0, 1])
    p = np.array([.8, .8, .4, .3, .3, .1])
    m = binary_metrics(y, p)
    assert abs(m['average_precision'] - average_precision_score(y, p)) < 1e-12
    assert abs(m['roc_auc'] - roc_auc_score(y, p)) < 1e-12


def test_zero_positive_period_retained_and_threshold_validation_only():
    m = binary_metrics([0, 0], [.2, .1])
    assert m['empty_positive_period'] and m['recall'] is None
    assert select_threshold([0, 0], [.2, .1]) == float('inf')
    threshold = select_threshold([1, 0, 1, 0], [.9, .8, .7, .1], .6)
    assert threshold == .7
    assert binary_metrics([1, 0], [.6, .8], threshold)['TP'] == 0


def test_blocked_bootstrap_preserves_zero_outcome_blocks():
    result = blocked_bootstrap([0,0,1,1], [.1,.1,.9,.9], [0,0,1,1],
                               lambda y,p: float(np.mean(y)), repeats=100)
    assert result['valid_replicates'] == 100
    assert result['lower'] == 0 and result['upper'] == 1
