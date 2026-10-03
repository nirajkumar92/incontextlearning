"""A small explicit NPZ interchange contract; query outcomes stay out of inputs."""
from __future__ import annotations
from pathlib import Path
import numpy as np


def load_table(path, require_target=True):
    with np.load(Path(path), allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    if 'X' not in data or 'categorical' not in data:
        raise ValueError("NPZ requires X and categorical arrays")
    x, cat = data['X'], data['categorical']
    if x.ndim != 2 or cat.shape != (x.shape[1],) or x.dtype.kind not in 'fi':
        raise ValueError("X must be a numeric 2D matrix; nominal categories are consistent numeric IDs")
    if not np.isin(cat, [0, 1]).all():
        raise ValueError('categorical flags must be Boolean or zero/one')
    data['categorical'] = cat.astype(bool)
    if (require_target and 'y' not in data) or ('y' in data and data['y'].shape != (len(x),)):
        raise ValueError("A target y per row is required")
    if 'y' in data and not np.isfinite(data['y']).all():
        raise ValueError("Only adjudicated targets belong in a labeled table; filter unknown labels explicitly")
    if 'ids' in data and (data['ids'].shape != (len(x),) or len(np.unique(data['ids'])) != len(x)):
        raise ValueError("Duplicate row IDs are not independent support records")
    for key in ['event_time', 'label_available_time']:
        if key in data and data[key].shape != (len(x),):
            raise ValueError(key + ' requires one timestamp per row')
    if 'feature_names' in data and (data['feature_names'].shape != (x.shape[1],) or len(np.unique(data['feature_names'])) != x.shape[1]):
        raise ValueError('Feature names must be unique and align with columns')
    return data


def validate_pair(support, query):
    if support['X'].shape[1] != query['X'].shape[1] or not np.array_equal(support['categorical'], query['categorical']):
        raise ValueError("Feature count/types differ between support and query")
    if 'feature_names' in support and 'feature_names' in query:
        if not np.array_equal(support['feature_names'], query['feature_names']):
            raise ValueError("Feature order differs")
    if 'ids' in support and 'ids' in query:
        if np.intersect1d(support['ids'], query['ids']).size:
            raise ValueError("Support and scored query identities overlap")


def assert_time_contract(support, query, cutoff):
    """For finance, timestamps use the same numeric unit and timezone upstream."""
    for k in ['event_time', 'label_available_time']:
        if k not in support or support[k].shape != (len(support['X']),):
            raise ValueError("Temporal support requires " + k)
    if 'event_time' not in query or query['event_time'].shape != (len(query['X']),):
        raise ValueError("Temporal queries require event_time")
    if cutoff is None or not np.isfinite(cutoff) or not np.isfinite(support['event_time']).all() or not np.isfinite(query['event_time']).all():
        raise ValueError('Finite event timestamps and cutoff are required')
    if not np.isfinite(support['label_available_time']).all():
        raise ValueError("Unknown labels cannot be in labeled support")
    if (support['label_available_time'] < support['event_time']).any():
        raise ValueError('An outcome label cannot precede its event in this snapshot schema')
    if (support['label_available_time'] > cutoff).any() or (support['event_time'] > cutoff).any():
        raise ValueError("Support uses future events or labels")
    if (query['event_time'] <= cutoff).any():
        raise ValueError("Queries must be after the declared snapshot")
    # Point-in-time feature correctness cannot be inferred from a matrix. The
    # upstream feature computation must retain and audit feature availability.


def assert_validation_time_contract(validation, query):
    """All threshold/HPO outcomes must exist before the first test decision."""
    if not len(validation['X']) or not len(query['X']):
        raise ValueError('Temporal evaluation requires nonempty validation and test periods')
    first_query = np.min(query['event_time'])
    labels = validation.get('label_available_time')
    if (labels is None or labels.shape != (len(validation['X']),)
            or not np.isfinite(labels).all()
            or (labels < validation['event_time']).any()
            or (labels >= first_query).any()):
        raise ValueError('Validation labels must follow their events and precede the first scored test prediction')
    if np.max(validation['event_time']) >= first_query:
        raise ValueError('Validation events must precede locked test events')
