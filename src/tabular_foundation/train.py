"""Ragged-episode reference trainer with replicated distributed gradient reduction.

This prioritizes explicit probability/accounting semantics. Profile the selected
shapes on ROCm before committing a long allocation; CPU tests do not certify a
64-GPU execution environment.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from .runtime import atomic_json, profile_schedule, seed_for, lr_factor, stage_for, charged_gpu_hours, ProjectLedger

_COUNT_KEYS = ('R','P0','P1','P4','finance','binary','multiclass','regression',
               'reference_raw_attempts','reference_filter_rejections','envelope_extended',
               'nominal_branch_macros','classes_11_32','classes_33_128','classes_129_256',
               'reference_p1_eligible_slots','reference_p4_eligible_slots',
               'reference_branch_R','reference_branch_P1','reference_branch_P4',
               'reference_ineligible_returns','reference_p1_raw_attempts','reference_p4_raw_attempts',
               'reference_graph_proposals','reference_graph_rejections',
               'reference_empty_feature_rejections','reference_class_split_rejections',
               'reference_predictability_rejections','reference_raw_rejections',
               'reference_requested_features_sum','reference_accepted_features_sum',
               'reference_removed_features_sum','class_budget_sum','observed_classes_sum',
               'class_budget_2','class_budget_3_10','class_budget_11_32',
               'class_budget_33_128','class_budget_129_256','class_budget_over_256',
               'observed_classes_0','observed_classes_1','observed_classes_2',
               'observed_classes_3_10','observed_classes_11_32','observed_classes_33_128',
               'observed_classes_129_256','observed_classes_over_256',
               'finance_reuse_0','finance_reuse_1','finance_reuse_2','finance_reuse_3',
               'finance_history_rows_macro_sum','finance_eligible_rows_macro_sum')


def macroepisode_task_counts(entry, routes, standard_prior='authored'):
    """Bounded additive audit counters, once per macroepisode, across its routes.

    These record consumed tasks, including an eventual skipped optimizer update.
    Class-budget counters and observed labels are separate. Finance population
    row sums count represented populations once per macroepisode, including
    repeated world uses; they are neither materialized rows nor unique worlds.
    Distinct world/lineage reconstruction uses attempt/stage plus manifest seed,
    not a growing set of world IDs in the trainer.
    """
    if not routes:
        raise ValueError('Cannot count a macroepisode without routes')
    counts = {key: 0 for key in _COUNT_KEYS}
    first = routes[0]
    reference = first.metadata.get('reference_control', {})
    family = ('finance' if entry['profile'] == 'finance' else
              reference.get('selected_source', first.metadata.get('family', 'P0')))
    if family not in ('R', 'P0', 'P1', 'P4', 'finance'):
        raise ValueError('Unknown consumed prior family: ' + str(family))
    counts[family] = 1
    task = ('regression' if first.task in ('regression', 'amount', 'fraud_loss') else
            'binary' if first.n_classes == 2 else 'multiclass')
    counts[task] = 1
    counts['envelope_extended'] = int(bool(first.metadata.get('reference_envelope')))
    counts['nominal_branch_macros'] = int(any(np.any(ep.categorical) for ep in routes))
    if reference:
        counts['reference_p1_eligible_slots'] = int(standard_prior == 'R_P1_05' and reference.get('eligible', False))
        counts['reference_p4_eligible_slots'] = int(standard_prior == 'R_P4_05' and reference.get('eligible', False))
        branch = reference.get('branch_draw', 'R')
        if branch not in ('R', 'P1', 'P4'):
            raise ValueError('Unknown reference branch draw')
        counts['reference_branch_' + branch] = 1
        counts['reference_ineligible_returns'] = int(reference.get('ineligible_mass_returned', False))
        attempts_key = {'R': 'reference_raw_attempts', 'P1': 'reference_p1_raw_attempts',
                        'P4': 'reference_p4_raw_attempts'}[family]
        counts[attempts_key] = int(reference.get('raw_dataset_attempts', 0))
        for field in ('graph_proposals', 'graph_rejections', 'empty_feature_rejections',
                      'class_split_rejections', 'predictability_rejections'):
            counts['reference_' + field] = int(reference.get(field, 0))
        # Retain the old name's specific ExtraTrees meaning for older readers.
        counts['reference_filter_rejections'] = counts['reference_predictability_rejections']
        counts['reference_raw_rejections'] = sum(counts['reference_' + field] for field in
            ('empty_feature_rejections', 'class_split_rejections', 'predictability_rejections'))
        requested = int(reference['requested_features'])
        accepted = int(reference['accepted_features'])
        counts['reference_requested_features_sum'] = requested
        counts['reference_accepted_features_sum'] = accepted
        counts['reference_removed_features_sum'] = requested - accepted
    if task != 'regression':
        observed = set()
        for ep in routes:
            observed.update(np.unique(ep.y_support).tolist())
            observed.update(np.unique(ep.y_query).tolist())
        counts['class_budget_sum'] = int(first.n_classes)
        counts['observed_classes_sum'] = len(observed)
        bins = ((2, 2, '2'), (3, 10, '3_10'), (11, 32, '11_32'),
                (33, 128, '33_128'), (129, 256, '129_256'), (257, float('inf'), 'over_256'))
        for low, high, label in bins:
            counts['class_budget_' + label] = int(low <= first.n_classes <= high)
            counts['observed_classes_' + label] = int(low <= len(observed) <= high)
        counts['observed_classes_0'] = int(len(observed) == 0)
        counts['observed_classes_1'] = int(len(observed) == 1)
        # Legacy counters explicitly remain class-budget bins, not observed bins.
        for label in ('11_32', '33_128', '129_256'):
            counts['classes_' + label] = counts['class_budget_' + label]
    if entry['profile'] == 'finance':
        reuse = entry.get('reuse')
        if isinstance(reuse, bool) or reuse not in (0, 1, 2, 3):
            raise ValueError('Finance reuse index must be 0..3')
        counts['finance_reuse_' + str(reuse)] = 1
        world = first.metadata.get('world_id')
        if any(ep.metadata.get('world_id') != world for ep in routes):
            raise ValueError('Finance routes disagree on world identity')
        counts['finance_history_rows_macro_sum'] = int(first.metadata.get('source_population_size', 0))
        counts['finance_eligible_rows_macro_sum'] = int(first.metadata.get('eligible_population_size', 0))
    return counts


def synchronize_gradients(model):
    """Average after ALL local macroepisodes, including variable route counts.

    Active flags distinguish globally unused parameters (no optimizer update)
    from locally unused ones (zero contribution to a globally active update).
    Manual reduction avoids mismatched per-forward DDP collectives for ragged
    route groups. Full original matrices remain replicated for Muon.
    """
    if not dist.is_initialized():
        return
    params = list(model.parameters())
    device = next(model.parameters()).device
    flags = torch.tensor([p.grad is not None for p in params], device=device, dtype=torch.int32)
    dist.all_reduce(flags, op=dist.ReduceOp.MAX)
    active = flags.cpu().tolist()
    bucket, count = [], 0
    def flush():
        if not bucket:
            return
        flat = torch.cat([p.grad.reshape(-1) for p in bucket])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(dist.get_world_size())
        offset = 0
        for p in bucket:
            p.grad.copy_(flat[offset:offset+p.numel()].view_as(p))
            offset += p.numel()
    for p, used in zip(params, active):
        if not used:
            p.grad = None
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        if bucket and count + p.numel() > 4*1024*1024:
            flush(); bucket, count = [], 0
        bucket.append(p); count += p.numel()
    flush()


def episode_loss(output, episode, device, conditional_labels=False):
    if 'logits' in output:
        target = torch.as_tensor(episode.y_query, device=device, dtype=torch.long)
        law = episode.metadata.get('latent_conditional_law') if conditional_labels else None
        if isinstance(law, np.ndarray) and law.shape == tuple(output['logits'].shape):
            probabilities = torch.as_tensor(law, device=device, dtype=torch.float32)
            if (not torch.isfinite(probabilities).all() or (probabilities < 0).any()
                    or not torch.allclose(probabilities.sum(-1), torch.ones(len(target), device=device), atol=1e-5)):
                raise ValueError('Invalid loss-only conditional classification law')
            per_query = -(probabilities * F.log_softmax(output['logits'].float(), dim=-1)).sum(-1)
        else:
            per_query = F.cross_entropy(output['logits'].float(), target, reduction='none')
    else:
        # The distribution standardizes in FP64 before entering neural units.
        target = torch.as_tensor(episode.y_query, device=device, dtype=torch.float64)
        from .distributions import regression_loss
        per_query = regression_loss(output, target)
    weights = getattr(episode, 'query_weights', None)
    if weights is not None:
        per_query = per_query * torch.as_tensor(weights, device=device, dtype=torch.float32)
    return per_query.sum(), len(target)


def _distributed_device(request):
    size, rank = int(os.environ.get('WORLD_SIZE', '1')), int(os.environ.get('RANK', '0'))
    if request == 'auto':
        request = 'cuda' if torch.cuda.is_available() else 'cpu'
    if request.startswith('cuda'):
        requested = torch.device(request)
        local = (int(os.environ.get('LOCAL_RANK', '0')) if size > 1
                 else requested.index if requested.index is not None else 0)
        torch.cuda.set_device(local)
        device = torch.device('cuda', local)
    else:
        device = torch.device(request)
    if size > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    return device, rank, size


def _save_checkpoint(path, model, optimizers, config, step, attempt, cumulative_hours):
    path = Path(path)
    payload = {'model': model.state_dict(), 'optimizers': [o.state_dict() for o in optimizers],
               'config': config, 'step': step, 'attempt': attempt,
               'cumulative_gpu_hours': cumulative_hours,
               'run_id': str(path.resolve().parent),
               'initialization': getattr(model, '_initialization_provenance', None),
               'torch_rng': torch.get_rng_state(), 'version': 1}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def validate_config(config):
    allowed = {'model','device','cpu_threads','seed','steps','global_batch','finance_share',
               'finance_adapter','optimizer','adam_lr','muon_lr','gradient_clip','bf16',
               'activation_checkpointing','checkpoint_every','gpu_hour_cap','project_ledger',
               'stage_end_steps','allocated_gpus','update_reserve_seconds','static_overrides',
               'finance_overrides','conditional_label_fraction','model_options',
               'standard_prior','reference_task','reference_shape','reference_envelope',
               'producer_workers','prefetch_tasks','weight_decay','reference_envelope_probability','finance_task'}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError('Unknown training configuration keys: ' + str(sorted(unknown)))
    for key, default in [('steps',0),('global_batch',256),('checkpoint_every',100),('cpu_threads',1)]:
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(key + ' must be a positive integer')
    if config.get('finance_share',.2) not in (0,.2,1):
        raise ValueError('finance_share must be 0,0.2 or1 for a declared controlled profile')
    if not 0 <= config.get('conditional_label_fraction',0) <= 1:
        raise ValueError('conditional_label_fraction must lie in[0,1]')
    if config.get('standard_prior', 'authored') not in ('authored','R','R_P1_05','R_P4_05'):
        raise ValueError('Unknown standard_prior')
    if config.get('reference_task', 'mixed') not in ('mixed','classification','regression'):
        raise ValueError('Unknown reference_task')
    workers = config.get('producer_workers', 0)
    if isinstance(workers, bool) or workers not in (0,1):
        raise ValueError('producer_workers must be 0 or 1 per training rank')
    depth = config.get('prefetch_tasks', 2)
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise ValueError('prefetch_tasks must be a positive integer')
    if config.get('standard_prior', 'authored') != 'authored' and config.get('static_overrides'):
        raise ValueError('Use reference_shape for reference diagnostics; static_overrides only applies to authored priors')
    if not np.isfinite(config.get('weight_decay', .01)) or config.get('weight_decay', .01) < 0:
        raise ValueError('weight_decay must be finite and nonnegative')
    if not 0 <= config.get('reference_envelope_probability', 1.) <= 1:
        raise ValueError('reference_envelope_probability must lie in[0,1]')
    if config.get('finance_task') not in (None,'binary','multiclass','regression'):
        raise ValueError('finance_task must be binary, multiclass, regression, or null')


def run(config, output_dir, resume=None, initialize_from=None):
    validate_config(config)
    if resume and initialize_from:
        raise ValueError('Choose exact resume or new-run initialization, not both')
    from .model import build_model
    from .optim import build_optimizers
    from .static_prior import generate_episode
    start = time.perf_counter()  # include model, optimizer and checkpoint setup
    out = Path(output_dir)
    run_id = str(out.resolve())
    if (out / 'manifest.json').exists() and not resume:
        raise FileExistsError('Output already contains a run; select a new directory or explicitly resume it')
    device, rank, world_size = _distributed_device(config.get('device', 'auto'))
    torch.set_num_threads(int(config.get('cpu_threads', 1)))
    seed = int(config.get('seed', 1729))
    torch.manual_seed(seed)
    model = build_model(config.get('model', 'tiny'), finance=bool(config.get('finance_adapter', True)), device=device,
                        model_options=config.get('model_options'))
    if initialize_from:
        parent_path = Path(initialize_from).resolve()
        parent = torch.load(parent_path, map_location='cpu', weights_only=True)
        parent_config = parent['config']
        for key, default in [('model','tiny'),('finance_adapter',True),('model_options',{})]:
            if parent_config.get(key, default) != config.get(key, default):
                raise ValueError('New-run initialization requires identical model configuration: ' + key)
        model.load_state_dict(parent['model'], strict=True)
        digest = hashlib.sha256()
        with parent_path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                digest.update(chunk)
        model._initialization_provenance = {'path':str(parent_path),'sha256':digest.hexdigest(),
            'parent_run_id':parent.get('run_id'), 'parent_step':parent['step'],
            'optimizer_reset':True,'schedule_reset':True,'charged_parent_hours_again':False}
    if config.get('activation_checkpointing', False):
        if not hasattr(model, 'activation_checkpointing'):
            raise ValueError('Selected model does not expose activation_checkpointing')
        model.activation_checkpointing = True
    optimizers = build_optimizers(model, mode=config.get('optimizer', 'adamw'),
                                  adam_lr=config.get('adam_lr', 3e-4), muon_lr=config.get('muon_lr', 8e-4),
                                  weight_decay=config.get('weight_decay', .01))
    steps = int(config['steps'])
    batch = int(config.get('global_batch', 256))
    if batch < world_size:
        raise ValueError('Global batch must cover every rank')
    share = float(config.get('finance_share', .2))
    if share and not config.get('finance_adapter', True):
        raise ValueError('Finance training requires the sampling metadata adapter')
    stages = config.get('stage_end_steps', [steps, steps, steps])
    if stages[-1] != steps:
        raise ValueError('Final stage endpoint must equal the full schedule horizon')
    stage_for(0, stages)
    step, attempt, prior_hours = 0, 0, 0.
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=True)
        if checkpoint['config'] != config:
            raise ValueError('Resume configuration differs; use explicit new-run continuation rather than silently changing horizon/prior')
        original_run = checkpoint.get('run_id', str(Path(resume).resolve().parent))
        if config.get('project_ledger') and original_run != run_id:
            raise ValueError('A ledger-backed resume must use its original output directory. Branching to another run identity requires separate accounting and is not supported by --resume')
        model.load_state_dict(checkpoint['model'])
        model._initialization_provenance = checkpoint.get('initialization')
        for o, state in zip(optimizers, checkpoint['optimizers']):
            o.load_state_dict(state)
        step, attempt = checkpoint['step'], checkpoint['attempt']
        prior_hours = checkpoint.get('cumulative_gpu_hours', 0.)
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(out/'manifest.json', {'config': config, 'parameters': sum(p.numel() for p in model.parameters()),
                    'torch': torch.__version__, 'hip': torch.version.hip, 'device': str(device),
                    'world_size': world_size, 'status': 'running',
                    'initialization': getattr(model, '_initialization_provenance', None),
                    'config_sha256': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()})
    if dist.is_initialized():
        dist.barrier()
    allocated = int(config.get('allocated_gpus', os.environ.get('TFM_ALLOCATED_GPUS', world_size if device.type == 'cuda' else 0)))
    if allocated < (world_size if device.type == 'cuda' else 0):
        raise ValueError('Allocated GPU count cannot be less than GPU worker count')
    cap = float(config.get('gpu_hour_cap', float('inf')))
    ledger_path = config.get('project_ledger')
    ledger = ProjectLedger(ledger_path) if rank == 0 and ledger_path else None
    if ledger:
        # Work after the saved checkpoint was still consumed. Keep that charge,
        # then charge replay/setup anew instead of rolling the project ledger back.
        prior_hours = max(prior_hours, ledger.recorded_hours(run_id))
    accounting_base = torch.tensor(prior_hours, device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.broadcast(accounting_base, src=0)
    prior_hours = float(accounting_base.item())
    from .producer import TaskProducer
    producer = TaskProducer(config)
    skips = 0
    model.train()
    initial_ok = True
    if ledger:
        initial_ok = ledger.update(run_id, prior_hours + charged_gpu_hours(time.perf_counter()-start, allocated))
    admission = torch.tensor(int(initial_ok), device=device)
    if dist.is_initialized():
        dist.broadcast(admission, src=0)
    try:
        while step < steps and admission.item():
            elapsed = time.perf_counter() - start
            hours = prior_hours + charged_gpu_hours(elapsed, allocated)
            reserve_hours = charged_gpu_hours(float(config.get('update_reserve_seconds', 0)), allocated)
            should_stop = torch.tensor(int(hours + reserve_hours >= cap) if rank == 0 else 0, device=device)
            if dist.is_initialized():
                dist.broadcast(should_stop, src=0)
            if should_stop.item():
                break
            stage = stage_for(step, stages)
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
                for group in opt.param_groups:
                    group.setdefault('initial_lr', group['lr'])
                    group['lr'] = group['initial_lr'] * lr_factor(step, steps)
            schedule = profile_schedule(seed, attempt, batch, share)
            totals = np.zeros(7, dtype=np.float64)  # weighted loss, macros, routes, support rows, query rows, cells, finance macros
            counts = {key:0 for key in _COUNT_KEYS}
            tick = time.perf_counter()
            producer_seconds, input_wait_seconds = 0., 0.
            entries = [schedule[index] for index in range(rank, batch, world_size)]
            for entry, prepared, waited in producer.ordered(entries, stage):
                routes, denominator, multiplier, generated_seconds = prepared
                producer_seconds += generated_seconds
                input_wait_seconds += waited
                if entry['profile'] == 'finance':
                    totals[6] += 1
                if denominator < 1:
                    raise ValueError('Macroepisode has no query targets')
                task_counts = macroepisode_task_counts(entry, routes, config.get('standard_prior', 'authored'))
                for key, value in task_counts.items():
                    counts[key] += value
                totals[1] += 1
                for episode in routes:
                    inputs = episode.model_inputs()
                    if episode.task in ('binary', 'multiclass', 'classification'):
                        rng = np.random.default_rng(seed_for(entry['query_seed'], 'slots'))
                        inputs['class_slots'] = rng.choice(model.config.classes,
                                                           episode.n_classes, replace=False)
                    autocast = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' and config.get('bf16', True) else nullcontext()
                    with autocast:
                        output = model(**inputs)
                        fraction = float(config.get('conditional_label_fraction', 0))
                        if not 0 <= fraction <= 1:
                            raise ValueError('conditional_label_fraction must lie in [0,1]')
                        soft = entry['profile'] == 'standard' and np.random.default_rng(seed_for(entry['query_seed'], 'conditional')).random() < fraction
                        summed, nq = episode_loss(output, episode, device, conditional_labels=soft)
                        # all-reduce later averages ranks: scale by W/B, even when
                        # ranks receive unequal macro counts or numbers of routes.
                        loss = summed * (multiplier / denominator) * world_size / batch
                    loss.backward()
                    totals[0] += float(summed.detach()) * multiplier / denominator
                    ns, nf = episode.x_support.shape
                    totals[2:6] += [1, ns, nq, (ns+nq)*nf]
            synchronize_gradients(model)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get('gradient_clip', 10)))
            finite = bool(torch.isfinite(norm).item())
            if finite:
                for opt in optimizers:
                    opt.step()
                step += 1; skips = 0
            else:
                skips += 1
                if skips >= 3:
                    raise FloatingPointError('Three consecutive nonfinite updates; no optimizer/schedule step was applied')
            attempt += 1
            if device.type == 'cuda':
                torch.cuda.synchronize()
            stats = torch.as_tensor(totals, device=device)
            if dist.is_initialized():
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            totals = stats.cpu().numpy()
            count_tensor = torch.tensor([counts[key] for key in _COUNT_KEYS],device=device,dtype=torch.int64)
            if dist.is_initialized():
                dist.all_reduce(count_tensor,op=dist.ReduceOp.SUM)
            counts = dict(zip(_COUNT_KEYS,count_tensor.cpu().tolist()))
            hours = prior_hours + charged_gpu_hours(time.perf_counter()-start, allocated)
            continue_project = True
            if rank == 0:
                record = {'step': step, 'attempt': attempt, 'stage': stage, 'loss': float(totals[0]/batch) if np.isfinite(totals[0]) else None,
                          'macroepisodes': int(totals[1]), 'finance_macroepisodes': int(totals[6]),
                          'routed_contexts': int(totals[2]), 'support_rows': int(totals[3]),
                          'query_targets': int(totals[4]), 'cells': int(totals[5]),
                          'update_seconds': time.perf_counter()-tick, 'cumulative_gpu_hours': hours,
                          'gradient_norm': float(norm) if finite else None, 'update_applied': finite,
                          'rank0_generation_seconds': producer_seconds,
                          'rank0_input_wait_seconds': input_wait_seconds,
                          'consumed_task_counts': counts}
                with open(out/'metrics.jsonl', 'a') as f:
                    f.write(json.dumps(record, allow_nan=False)+'\n')
                print(json.dumps(record, allow_nan=False), flush=True)
                if ledger:
                    continue_project = ledger.update(run_id, hours)
                if step % int(config.get('checkpoint_every', 100)) == 0 or step == steps or hours >= cap:
                    _save_checkpoint(out/'last.pt', model, optimizers, config, step, attempt, hours)
            decision = torch.tensor(int(continue_project), device=device)
            if dist.is_initialized():
                dist.broadcast(decision, src=0)
            if not decision.item():
                break
    finally:
        producer.close()
    if rank == 0:
        hours = prior_hours + charged_gpu_hours(time.perf_counter()-start, allocated)
        _save_checkpoint(out/'last.pt', model, optimizers, config, step, attempt, hours)
        hours = prior_hours + charged_gpu_hours(time.perf_counter()-start, allocated)
        if ledger:
            ledger.update(run_id, hours)
        atomic_json(out/'completion.json', {'steps': step, 'requested_steps': steps,
                    'cumulative_gpu_hours': hours, 'status': 'completed' if step == steps else 'budget_stopped'})
    if dist.is_initialized():
        dist.barrier(); dist.destroy_process_group()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--resume')
    source.add_argument('--initialize-from', help='New experiment with parent weights and fresh optimizer/schedule')
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()), args.output, args.resume, args.initialize_from)


if __name__ == '__main__':
    main()
