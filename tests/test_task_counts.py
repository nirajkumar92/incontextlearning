"""Audit counters distinguish macroepisodes, reference proposals and class coverage."""
from collections import Counter

import numpy as np
import pytest

from tabular_foundation.runtime import profile_schedule
from tabular_foundation.schema import Episode
from tabular_foundation.train import macroepisode_task_counts


def _episode(labels=(0, 1), classes=2, metadata=None, width=4):
    y = np.asarray(labels)
    return Episode(np.zeros((len(y), width)), y, np.zeros((len(y), width)), y,
                   np.zeros(width, dtype=bool), 'binary' if classes == 2 else 'multiclass',
                   classes, metadata={} if metadata is None else metadata)


def test_reference_return_rejections_widths_and_observed_classes_are_separate():
    event = dict(selected_source='R', branch_draw='P1', eligible=False, ineligible_mass_returned=True,
                 raw_dataset_attempts=12, graph_proposals=14, graph_rejections=2,
                 empty_feature_rejections=2, class_split_rejections=8, predictability_rejections=1,
                 requested_features=6, accepted_features=5)
    ep = _episode(labels=(1, 7, 25), classes=64, width=5,
                  metadata={'reference_control': event, 'reference_envelope': {'max_classes': 64}})
    counts = macroepisode_task_counts({'profile': 'standard'}, [ep], 'R_P1_05')
    assert counts['R'] == 1 and counts['P1'] == 0
    assert counts['reference_branch_P1'] == 1 and counts['reference_ineligible_returns'] == 1
    assert counts['reference_p1_eligible_slots'] == 0
    assert counts['reference_raw_attempts'] == 12 and counts['reference_raw_rejections'] == 11
    assert counts['reference_graph_rejections'] == 2 and counts['reference_class_split_rejections'] == 8
    assert counts['reference_requested_features_sum'] == 6 and counts['reference_accepted_features_sum'] == 5
    assert counts['reference_removed_features_sum'] == 1
    assert counts['class_budget_33_128'] == counts['classes_33_128'] == 1
    assert counts['observed_classes_3_10'] == 1 and counts['observed_classes_33_128'] == 0
    assert counts['class_budget_sum'] == 64 and counts['observed_classes_sum'] == 3
    assert counts['envelope_extended'] == 1


def test_eligible_authored_branch_counts_no_native_retries():
    event = dict(selected_source='P1', branch_draw='P1', eligible=True, ineligible_mass_returned=False,
                 raw_dataset_attempts=1, requested_features=8, accepted_features=8)
    counts = macroepisode_task_counts({'profile': 'standard'}, [
        _episode(classes=7, width=8, metadata={'reference_control': event})], 'R_P1_05')
    assert counts['P1'] == counts['reference_branch_P1'] == counts['reference_p1_eligible_slots'] == 1
    assert counts['reference_p1_raw_attempts'] == 1 and counts['reference_raw_attempts'] == 0
    assert counts['multiclass'] == 1 and counts['observed_classes_2'] == 1


def test_finance_routes_count_once_and_union_observed_labels():
    audit = {'world_id': 'same-world', 'source_population_size': 3_000_000, 'eligible_population_size': 2_000_000}
    routes = [_episode(labels=(0,), metadata=dict(audit)), _episode(labels=(1,), metadata=dict(audit))]
    counts = macroepisode_task_counts({'profile': 'finance', 'reuse': 3}, routes)
    assert counts['finance'] == counts['binary'] == counts['finance_reuse_3'] == 1
    assert counts['finance_history_rows_macro_sum'] == 3_000_000
    assert counts['finance_eligible_rows_macro_sum'] == 2_000_000
    assert counts['observed_classes_2'] == 1 and counts['observed_classes_sum'] == 2
    routes[1].metadata['world_id'] = 'different-world'
    with pytest.raises(ValueError, match='world identity'):
        macroepisode_task_counts({'profile': 'finance', 'reuse': 3}, routes)


def test_64_rank_schedule_keeps_four_local_worlds_across_reuse_and_additive_counts():
    totals = Counter()
    first_rank_worlds = None
    all_worlds = set()
    for attempt in range(4):
        schedule = profile_schedule(1729, attempt, 256, 1)
        rank_worlds = []
        rank_totals = []
        for rank in range(64):
            local = schedule[rank::64]
            assert len(local) == 4
            rank_worlds.append([entry['world_seed'] for entry in local])
            local_counts = Counter()
            for entry in local:
                all_worlds.add(entry['world_seed'])
                ep = _episode(metadata={'world_id': str(entry['world_seed'])})
                local_counts.update(macroepisode_task_counts(entry, [ep]))
            rank_totals.append(local_counts)
        if first_rank_worlds is None:
            first_rank_worlds = rank_worlds
        assert rank_worlds == first_rank_worlds
        for local_counts in rank_totals:
            totals.update(local_counts)
    assert len(all_worlds) == 256
    assert totals['finance'] == totals['binary'] == 1024
    assert [totals[f'finance_reuse_{i}'] for i in range(4)] == [256] * 4
    assert totals['observed_classes_sum'] == 2048
