"""The prior audit reports measurements and aborts honestly on failed worlds."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / 'scripts/audit_selection_priors.py'
SPEC = importlib.util.spec_from_file_location('selection_audit', PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_streaming_moments_match_independent_numpy_and_large_values():
    values = np.array([1., 2., -8., 25.])
    accumulator = audit.StreamingMoments()
    for value in values:
        accumulator.add(value)
    result = accumulator.result()
    assert result['mean'] == pytest.approx(values.mean())
    assert result['sample_sd'] == pytest.approx(values.std(ddof=1))
    assert result['min'] == -8 and result['max'] == 25
    large = audit.StreamingMoments()
    for value in [1e200, 1e200, 2e200]:
        large.add(value)
    assert large.result()['mean'] == pytest.approx(4e200 / 3)
    assert large.result()['sample_sd'] == pytest.approx(1e200 / np.sqrt(3))


def test_real_diagnostic_audit_records_shape_hashes_and_coverage():
    report = audit.run_audit(config={'standard_prior': 'selection'}, families=['forest'],
                             stages=[1], worlds=2, shape=[12, 6, 8], classes=4,
                             observation='mcar')
    assert report['status'] == 'passed'
    assert report['requested_worlds'] == report['accepted_worlds'] == 2
    assert report['settings']['diagnostic_shape_override'] == [12, 6, 8]
    assert not report['model_training_performed'] and not report['teacher_acceptance']
    assert len(report['source_sha256']) >= 5
    bucket = report['buckets']['forest/stage1']
    assert bucket['metrics']['support_rows']['min'] == 12
    assert bucket['counts']['effective_observation'] == {'mcar': 2}
    assert 'noise_scale' in bucket['metrics'] or 'class_calibration_maximum_absolute_error' in bucket['metrics']


def test_failed_worlds_are_not_replaced_or_counted_as_accepted(monkeypatch):
    def fail(*args, **kwargs):
        raise FloatingPointError('test invalid world')
    monkeypatch.setattr(audit, 'generate_challenger_episode', fail)
    report = audit.run_audit(config={'standard_prior': 'selection'}, families=['forest'],
                             stages=[1, 2], worlds=7, max_failures=2)
    assert report['status'] == 'failed'
    assert report['attempted_worlds'] == 2 and report['accepted_worlds'] == 0
    assert report['requested_worlds'] == 14
    assert len(report['failed_worlds']) == 2
    assert report['failed_worlds'][0]['seed'] != report['failed_worlds'][1]['seed']
    assert 'test invalid world' in report['failed_worlds'][0]['error']


@pytest.mark.parametrize('override', [{'worlds':0}, {'families':['R','R']}, {'stages':[4]},
                                    {'classes':8}, {'config':{'standard_prior':'R'}}])
def test_invalid_audit_laws_rejected(override):
    kwargs = dict(config={'standard_prior':'selection'}, families=['R'], stages=[1], worlds=1)
    kwargs.update(override)
    with pytest.raises(ValueError):
        audit.run_audit(**kwargs)
