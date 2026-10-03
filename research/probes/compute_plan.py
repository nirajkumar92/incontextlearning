"""Budget arithmetic for the prior-bank project; never estimates a winning size.

A profile CSV must contain measured end-to-end job counters. Synthetic examples
are deliberately separate and labeled illustrative. Run --help for the schema.
Only the standard library is required.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

ALLOCATION = {
    'systems_and_reproduction': 0.10,
    'prior_screening': 0.30,
    'scaling_and_confirmation': 0.20,
    'final_pretraining': 0.24,
    'locked_evaluation': 0.10,
    'uncommitted_reserve': 0.06,
}
COUNTERS = ('rows', 'cells', 'query_targets', 'core_tokens')


def positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value


def project_budget(gpu_hours=50_000.0, gpus=64):
    hours = positive(gpu_hours, 'gpu_hours')
    devices = positive(gpus, 'gpus')
    if devices != int(devices):
        raise ValueError('gpus must be an integer')
    return {
        'total_gpu_hours': hours,
        'total_gpus': int(devices),
        'full_cluster_days': hours / devices / 24,
        'phase_gpu_hours': {k: hours * w for k, w in ALLOCATION.items()},
        'phase_full_cluster_days': {k: hours * w / devices / 24 for k, w in ALLOCATION.items()},
    }


def dense_row_train_flops(params_active, layers, width, rows, episodes=1):
    """Forward+backward proxy: 6*N*s + 12*L*d*s^2 per episode.

    Assumes all active dense projections process all row tokens, full quadratic
    attention, multiply-add counted as two FLOPs, and backward = 2x forward.
    Excludes encoders/heads/optimizer/rematerialization. Not a runtime estimate.
    Masked dense kernels still pay for the full attention matrix.
    """
    n, l, d, s, e = [positive(v, k) for v, k in zip(
        [params_active, layers, width, rows, episodes],
        ['params_active', 'layers', 'width', 'rows', 'episodes'])]
    return e * (6 * n * s + 12 * l * d * s * s)


def profile_to_plan(record, gpu_hours):
    """Extrapolate only the exact profiled configuration/topology/shape mixture."""
    g = positive(record['allocated_gpus'], 'allocated_gpus')
    if g != int(g):
        raise ValueError('allocated_gpus must be an integer')
    wall = positive(record['wall_seconds'], 'wall_seconds')
    accepted = positive(record['accepted_episodes'], 'accepted_episodes')
    if accepted != int(accepted):
        raise ValueError('accepted_episodes must be an integer')
    params = positive(record['params'], 'params')
    budget = positive(gpu_hours, 'gpu_hours')
    if not record.get('core_token_definition', '').strip():
        raise ValueError('core_token_definition is required')
    per_episode = g * wall / accepted
    episodes = math.floor(budget * 3600 / per_episode)
    counters = {}
    for key in COUNTERS:
        count = positive(record[key], key)
        counters[key] = episodes * count / accepted
    return {
        'config_id': record['config_id'],
        'params': params,
        'status': 'Extrapolation of supplied measurements; not an optimality claim',
        'profiled_allocated_gpus': int(g),
        'gpu_hours_allocated': budget,
        'wall_days_at_profiled_topology': budget / g / 24,
        'gpu_seconds_per_accepted_episode': per_episode,
        'projected_accepted_episodes': episodes,
        'projected_processed_counters': counters,
        'core_token_definition': record['core_token_definition'],
        'projected_core_tokens_per_parameter': counters['core_tokens'] / params,
        'caveat': 'Same shape mix, kernels, topology and data pipeline; reprofile after changes. Counters include repeats and are not independent world counts.',
    }


def illustrative_scenarios(final_hours):
    rows, features, queries = 1152, 64, 128
    out = []
    for seconds in (0.1, 1.0, 10.0):
        episodes = math.floor(final_hours * 3600 / seconds)
        out.append({
            'status': 'ILLUSTRATIVE ASSUMPTION, NOT MEASURED MI355X THROUGHPUT',
            'assumed_gpu_seconds_per_episode': seconds,
            'accepted_episodes': episodes,
            'row_tokens_if_one_core_token_per_row': episodes * rows,
            'raw_cells': episodes * rows * features,
            'supervised_query_targets': episodes * queries,
            'row_token_to_parameter_ratio_at_300m': episodes * rows / 300_000_000,
        })
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu-hours', type=float, default=50_000)
    p.add_argument('--gpus', type=int, default=64)
    p.add_argument('--measurements', type=Path, help='CSV fields: config_id,params,allocated_gpus,wall_seconds,accepted_episodes,rows,cells,query_targets,core_tokens,core_token_definition. Counters are totals over the measured interval, not per-episode averages. Each row must profile the representative mixture.')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    report = {'budget': project_budget(args.gpu_hours, args.gpus)}
    final_hours = report['budget']['phase_gpu_hours']['final_pretraining']
    if args.measurements:
        with args.measurements.open(newline='') as f:
            records = list(csv.DictReader(f))
        if not records:
            raise ValueError('measurement CSV has no profiles')
        report['candidate_plans'] = [profile_to_plan(r, final_hours) for r in records]
        report['candidate_plan_rule'] = 'Alternative uses of the same final budget; do not sum these plans.'
    else:
        report['illustrative_sensitivity'] = illustrative_scenarios(final_hours)
        report['measurement_status'] = 'No GPU measurements supplied; no throughput or optimal model-size estimate.'
    report['dense_proxy_example'] = {
        'status': 'ARITHMETIC EXAMPLE ONLY; core projections and attention, no encoder',
        'params_active': 300_000_000, 'layers': 24, 'width': 1024,
        'rows_per_episode': 1152,
        'flops_per_episode': dense_row_train_flops(300_000_000, 24, 1024, 1152),
    }
    result = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result)
    else:
        print(result, end='')


if __name__ == '__main__':
    main()
