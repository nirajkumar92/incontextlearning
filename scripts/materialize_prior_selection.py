"""Resolve an explicitly reviewed phase into immutable, runnable trainer configs.

No score is invented and no winner is chosen here. P can be materialized directly;
M/A/V/F require phase-specific reviewed decisions. Each config has a complete
curriculum horizon selected using measured throughput, with a separate time cap.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from plan_prior_selection import build_plan, mixture_with_anchor, MECHANISMS, OBSERVATIONS, _simplex
from tabular_foundation.train import validate_config


def _review(decisions, name):
    decision = decisions.get(name)
    if not isinstance(decision, dict) or decision.get('reviewed') is not True or not isinstance(decision.get('evidence'), str) or not decision['evidence'].strip():
        raise ValueError(f'{name} requires reviewed=true and a nonempty evidence path or assessment')
    return decision


def _prior(decision):
    return {name: _simplex(decision[name], names, name) for name, names in
            [('mechanism_weights', MECHANISMS), ('observation_weights', OBSERVATIONS)]}


def _model(decision):
    model = decision.get('model')
    if model not in ('base', 'small50', 'small100', 'wide500', 'large1000', 'persistent_small'):
        raise ValueError('Selected architecture must name an implemented research-size model')
    return model


def materialize(spec, phase, *, steps=None, output, decisions=None, allocated_gpus=64,
                horizons=None, evaluation_reserve_gpu_hours=0.):
    if horizons is not None and steps is not None:
        raise ValueError('Choose common steps or exhaustive per-trial horizons, not both')
    if (isinstance(evaluation_reserve_gpu_hours, bool)
            or not isinstance(evaluation_reserve_gpu_hours, (float,int))
            or not math.isfinite(evaluation_reserve_gpu_hours)
            or evaluation_reserve_gpu_hours < 0):
        raise ValueError('evaluation_reserve_gpu_hours must be finite and nonnegative')
    if isinstance(allocated_gpus, bool) or not isinstance(allocated_gpus, int) or allocated_gpus < 1 or allocated_gpus > 256:
        raise ValueError('allocated_gpus must be an integer in 1..256')
    if phase not in ('P', 'M', 'A', 'V', 'F', 'FINAL_STANDARD', 'FINAL_FRAUD'):
        raise ValueError('Unknown experiment phase')
    divisor = 400 if phase in ('F','FINAL_FRAUD') else 100
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new output directory to freeze each materialized phase')
    decisions = decisions or {}
    plan = build_plan(spec)
    base = json.loads((ROOT / 'configs' / 'selection_candidate_standard.json').read_text())
    base.update(allocated_gpus=allocated_gpus)
    trials = [t for t in plan['trials'] if t['phase'] == phase]
    if phase in ('FINAL_STANDARD', 'FINAL_FRAUD'):
        trials = [{'id':phase.lower(), 'arm':phase, 'seed':1729,
                   'gpu_hour_ceiling':9000 if phase == 'FINAL_STANDARD' else 3000}]
    trial_ids = {trial['id'] for trial in trials}
    if horizons is not None and (not isinstance(horizons, dict) or set(horizons) != trial_ids):
        raise ValueError('horizons must contain every selected phase trial ID exactly once and no extras')
    chosen_horizons = horizons if horizons is not None else {key:steps for key in trial_ids}
    for run_id, horizon in chosen_horizons.items():
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 3:
            raise ValueError(f'{run_id}: declare a positive integer full-schedule horizon')
        if horizon % divisor:
            raise ValueError(f'{phase} horizon must be a multiple of {divisor} for exact 90/9/1 stages and finance reuse alignment')
    if any(evaluation_reserve_gpu_hours >= trial['gpu_hour_ceiling'] for trial in trials):
        raise ValueError('Evaluation reserve must be smaller than each trial package ceiling')
    configs = []
    ref = {'mechanism_weights':{key:float(key == 'R') for key in MECHANISMS},
           'observation_weights':{key:float(key == 'identity') for key in OBSERVATIONS}}
    for trial in trials:
        cfg = copy.deepcopy(base)
        horizon = chosen_horizons[trial['id']]
        cfg.update(seed=trial['seed'], gpu_hour_cap=trial['gpu_hour_ceiling']-evaluation_reserve_gpu_hours,
                   steps=horizon, stage_end_steps=[horizon*90//100, horizon*99//100, horizon])
        # Pair task-shape schedules by phase/seed; arm identity never alters them.
        prior = None
        if phase == 'P':
            prior = {key:trial[key] for key in ('mechanism_weights','observation_weights')}
        elif phase == 'M':
            chosen = _review(decisions, 'after_P')
            prior = _prior(chosen)
            anchor = trial['proposed_mechanism_weights']['R']
            prior['mechanism_weights'] = mixture_with_anchor(prior['mechanism_weights'], anchor)
        elif phase == 'A':
            chosen = _review(decisions, 'after_M')
            prior = ref if trial['arm'].startswith('R_') else _prior(chosen)
            cfg['model'] = 'persistent_small' if trial['arm'].endswith('persistent_cell') else 'base'
        elif phase == 'V':
            chosen = _review(decisions, 'after_A')
            prior = ref if trial['arm'] == 'R' else _prior(chosen)
            cfg['model'] = 'base' if trial['arm'] == 'R' else _model(chosen)
        else:
            key = 'after_V' if phase == 'F' else 'final'
            chosen = _review(decisions, key)
            prior = _prior(chosen)
            cfg['model'] = _model(chosen)
            if phase in ('F','FINAL_FRAUD'):
                parent = Path(chosen.get('parent_checkpoint', '')).expanduser()
                if not parent.is_file():
                    raise ValueError(f'{key} requires an existing parent_checkpoint shared by all continuation arms')
                import torch
                checkpoint = torch.load(parent, map_location='cpu', weights_only=True)
                parent_config = checkpoint['config']
                for k in ('model','model_options','finance_adapter'):
                    if parent_config.get(k) != cfg.get(k):
                        raise ValueError('Parent checkpoint architecture mismatch: ' + k)
                cfg['initial_checkpoint'] = str(parent.resolve())
                cfg['adam_lr'] *= .1
                cfg['muon_lr'] *= .1
                if phase == 'FINAL_FRAUD' or trial['arm'] != 'standard_replay':
                    cfg.update(finance_share=1, finance_task='binary', finance_overrides={
                        'mechanism_family':'risk_partition', 'feature_view':'hide_h',
                        'loss_normalization':'unit', 'prevalence':.0001})
                    if trial['arm'] == 'fraud_reservoir_positives':
                        cfg['finance_overrides'].update(local_cap=0, max_centers=1, max_positive_centers=0)
        cfg['selection_options'] = {**prior, 'namespace':'train', 'complexity_conditioned_probability':.8}
        validate_config(cfg)
        configs.append((trial, cfg))
    # Resolve everything before writing any output; invalid decisions leave no half-phase.
    output.mkdir(parents=True)
    records = []
    for trial, cfg in configs:
        config_path = output / (trial['id'] + '.json')
        config_path.write_text(json.dumps(cfg, indent=2) + '\n')
        records.append({'id':trial['id'], 'config':str(config_path),
                        'output':str(output/'runs'/trial['id']),
                        'gpu_hour_ceiling':trial['gpu_hour_ceiling'],
                        'package_gpu_hour_ceiling':trial['gpu_hour_ceiling'],
                        'training_gpu_hour_ceiling':cfg['gpu_hour_cap'],
                        'evaluation_reserve_gpu_hours':evaluation_reserve_gpu_hours,
                        'evaluation_allocation_id':str(output/'evaluations'/trial['id']),
                        'steps':cfg['steps'],
                        'accepted_macroepisode_target':cfg['steps']*cfg['global_batch'],
                        'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
                        'submit_command':['python','scripts/submit_training.py','--config',str(config_path),
                                          '--output',str(output/'runs'/trial['id']),'--submit'],
                        'launched':False})
    manifest = {'phase':phase,'reviewed_decisions':decisions,'runs':records,
                'canonical_spec_sha256':hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest(),
                'total_gpu_hour_ceiling':sum(r['gpu_hour_ceiling'] for r in records),
                'training_gpu_hour_ceiling':sum(r['training_gpu_hour_ceiling'] for r in records),
                'evaluation_reserve_gpu_hours':sum(r['evaluation_reserve_gpu_hours'] for r in records),
                'evaluation_reserve_status':'explicitly_reserved' if evaluation_reserve_gpu_hours else 'unreserved',
                'automatic_evaluation_charging':False,
                'accounting_note':'Package ceilings include training and a declared evaluation reserve. Record actual cumulative scheduler-billed training/evaluation hours with record_allocation.py; reserves are not recorded as consumed work. The ledger does not reserve concurrent allocations.',
                'horizon_policy':'measured_per_trial' if horizons is not None else 'common_horizon',
                'performance_guarantee':False,
                'schedule_note':'Horizon is explicit. Validate all three stage costs before scheduling; a capped partial curriculum is not a completed comparison.'}
    (output/'materialization.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,default=ROOT/'research/specs/prior_selection_v1.json')
    p.add_argument('--phase',required=True,choices=['P','M','A','V','F','FINAL_STANDARD','FINAL_FRAUD'])
    horizon_group = p.add_mutually_exclusive_group(required=True)
    horizon_group.add_argument('--steps',type=int,help='One common full-schedule horizon for the phase')
    horizon_group.add_argument('--horizons',type=Path,help='JSON object mapping every phase trial ID to its measured full-schedule horizon')
    p.add_argument('--evaluation-reserve-gpu-hours',type=float,default=0.,help='Per-trial reserve subtracted from the package ceiling; zero means explicitly unreserved')
    p.add_argument('--allocated-gpus',type=int,default=64)
    p.add_argument('--decisions',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=materialize(json.loads(a.spec.read_text()),a.phase,steps=a.steps,output=a.output,
        decisions=json.loads(a.decisions.read_text()) if a.decisions else {},allocated_gpus=a.allocated_gpus,
        horizons=json.loads(a.horizons.read_text()) if a.horizons else None,
        evaluation_reserve_gpu_hours=a.evaluation_reserve_gpu_hours)
    print(json.dumps({'phase':a.phase,'runs':len(result['runs']),'gpu_hour_ceiling':result['total_gpu_hour_ceiling']}))

if __name__ == '__main__':
    main()
