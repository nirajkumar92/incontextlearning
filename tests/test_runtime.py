import json
from tabular_foundation.runtime import profile_schedule, lr_factor, ProjectLedger, charged_gpu_hours


def test_reuse_keeps_joint_episode_mass_and_independent_batch():
    for batch in [1, 2, 7, 256]:
        all_entries = [profile_schedule(42, t, batch, .2) for t in range(20)]
        assert sum(e['profile'] == 'finance' for es in all_entries for e in es) == 4 * batch
        for es in all_entries:
            worlds = [e['world_seed'] for e in es]
            assert len(worlds) == len(set(worlds))
        f0 = sorted(e['world_seed'] for e in all_entries[0] if e['profile'] == 'finance')
        f3 = sorted(e['world_seed'] for e in all_entries[3] if e['profile'] == 'finance')
        assert f0 == f3


def test_schedule_and_accounting(tmp_path):
    assert lr_factor(0, 1000) == .05
    assert abs(lr_factor(999, 1000) - .1) < 1e-12
    assert charged_gpu_hours(3600, 64) == 64
    ledger = ProjectLedger(tmp_path/'ledger.json', cap=100)
    assert ledger.update('a', 64)
    assert ledger.update('a', 70)
    assert not ledger.update('b', 40)
    assert json.loads((tmp_path/'ledger.json').read_text())['total_gpu_hours'] == 110
