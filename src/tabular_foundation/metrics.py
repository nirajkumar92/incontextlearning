"""Metrics with explicit ties, natural prevalence and empty-positive behavior."""
from __future__ import annotations
import numpy as np


def binary_curve(y, scores):
    y, scores = np.asarray(y), np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or scores.shape != y.shape or len(y) == 0:
        raise ValueError("Nonempty one-dimensional labels and aligned scores required")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Finite scores and binary labels required")
    order = np.argsort(-scores, kind='stable')
    sy, ss = y[order], scores[order]
    ends = np.r_[np.flatnonzero(ss[1:] != ss[:-1]), len(ss)-1]
    tp = np.cumsum(sy, dtype=float)[ends]
    fp = 1 + ends - tp
    return ss[ends], tp, fp


def binary_metrics(y, probabilities, threshold=.5, review_budget=None):
    raw = np.asarray(y)
    if np.isnan(threshold):
        raise ValueError('NaN threshold is invalid')
    if raw.ndim != 1 or not np.isfinite(raw).all() or not np.isin(raw, [0, 1]).all():
        raise ValueError('Binary targets must be exactly zero or one')
    y, p = raw.astype(int), np.asarray(probabilities, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must lie in [0,1]")
    levels, tp, fp = binary_curve(y, p)
    positives, negatives = int(y.sum()), int((1-y).sum())
    recall = tp / positives if positives else np.zeros_like(tp)
    precision = tp / (tp + fp)
    ap = float(np.sum(np.diff(np.r_[0., recall]) * precision)) if positives else 0.
    if positives and negatives:
        fpr = np.r_[0., fp / negatives]
        tpr = np.r_[0., recall]
        auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    else:
        auc = None
    pred = p >= threshold
    TP, FP = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    FN, TN = positives - TP, negatives - FP
    clipped = np.clip(p, 1e-15, 1-1e-15)
    result = {'rows': len(y), 'positives': positives, 'negatives': negatives,
              'prevalence': positives/len(y), 'average_precision': ap, 'roc_auc': auc,
              'log_loss': float(-np.mean(y*np.log(clipped)+(1-y)*np.log1p(-clipped))),
              'brier': float(np.mean((p-y)**2)), 'threshold': float(threshold) if np.isfinite(threshold) else None,
              'decision_rule': 'no alerts' if threshold == float('inf') else 'probability >= threshold',
              'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
              'precision': TP/(TP+FP) if TP+FP else None,
              'recall': TP/positives if positives else None,
              'false_positives_per_million_negatives': FP/negatives*1e6 if negatives else None,
              'empty_positive_period': positives == 0}
    if review_budget is not None:
        k = max(0, min(int(review_budget), len(y)))
        top = np.argsort(-p, kind='stable')[:k]
        captured = int(y[top].sum())
        result.update({'review_count': k, 'precision_at_k': captured/k if k else None,
                       'recall_at_k': captured/positives if positives else None,
                       'review_ties': 'stable input order; budget can cut a score tie'})
    return result


def select_threshold(y_validation, p_validation, minimum_precision=.8, min_alerts=1):
    """Maximize empirical recall subject to a validation-only precision constraint.

    This is a point-estimate selector, not a confidence guarantee. Empty-positive
    or infeasible validation returns +infinity, meaning alert nobody.
    """
    if not 0 < minimum_precision <= 1 or min_alerts < 1:
        raise ValueError("Invalid operating point")
    levels, tp, fp = binary_curve(y_validation, p_validation)
    eligible = (tp > 0) & (tp/(tp+fp) >= minimum_precision) & (tp+fp >= min_alerts)
    if not eligible.any():
        return float('inf')
    indices = np.flatnonzero(eligible)
    best = indices[np.argmax(tp[indices])]
    return float(levels[best])


def regression_metrics(y, prediction):
    y, p = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if y.shape != p.shape or len(y) == 0 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Finite aligned nonempty regression arrays required")
    return {'rows': len(y), 'rmse': float(np.sqrt(np.mean((y-p)**2))),
            'mae': float(np.mean(np.abs(y-p)))}


def multiclass_metrics(y, probabilities):
    raw = np.asarray(y)
    if raw.ndim != 1 or not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all():
        raise ValueError('Multiclass labels must be finite integers')
    y, p = raw.astype(int), np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or len(y) != len(p) or len(y) == 0 or ((y < 0)|(y >= p.shape[1])).any():
        raise ValueError("Labels must use declared column indices")
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError("Invalid class probabilities")
    return {'rows': len(y), 'classes': p.shape[1],
            'log_loss': float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean()),
            'accuracy': float((p.argmax(1) == y).mean())}


def blocked_bootstrap(y, p, blocks, statistic, seed=0, repeats=1000):
    """Resample complete declared blocks; user selects meaningful time/entity units."""
    y, p, blocks = np.asarray(y), np.asarray(p), np.asarray(blocks)
    if len(y) != len(p) or len(y) != len(blocks):
        raise ValueError("Aligned blocks required")
    unique = np.unique(blocks)
    if len(unique) < 2:
        raise ValueError("At least two independent blocks required")
    indices = [np.flatnonzero(blocks == b) for b in unique]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        take = np.concatenate([indices[i] for i in rng.integers(len(indices), size=len(indices))])
        val = statistic(y[take], p[take])
        if val is not None and np.isfinite(val):
            values.append(val)
    return {'lower': float(np.quantile(values, .025)) if values else None,
            'upper': float(np.quantile(values, .975)) if values else None,
            'valid_replicates': len(values), 'requested_replicates': repeats,
            'blocks': len(unique)}
