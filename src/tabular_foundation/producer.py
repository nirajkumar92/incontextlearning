"""Deterministic, ordered CPU task production with optional process prefetch.

One persistent producer per training rank preserves finance-world reuse and
isolates upstream global RNG use. Queue depth bounds task count, not bytes;
this is a portable overlap path, not a GPU utilization guarantee.
"""
from collections import OrderedDict, deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
import time

import numpy as np
import torch

from .runtime import seed_for

_WORLD_CACHE = OrderedDict()


def _initialize_worker():
    torch.set_num_threads(1)


def prepare_task(config, entry, stage):
    start = time.perf_counter()
    if entry['profile'] == 'finance':
        from .finance_prior import generate_finance_world
        # Include overrides to prevent accidental cross-experiment cache reuse.
        import json
        key = (entry['world_seed'], stage, config.get('finance_task'),
               json.dumps(config.get('finance_overrides', {}), sort_keys=True))
        if key not in _WORLD_CACHE:
            _WORLD_CACHE[key] = generate_finance_world(
                entry['world_seed'], stage=stage, task=config.get('finance_task'),
                overrides=config.get('finance_overrides', {}))
        _WORLD_CACHE.move_to_end(key)
        # Four local worlds cover the intended 256-macro/64-rank batch. A
        # smaller device count may regenerate evicted worlds without bias.
        while len(_WORLD_CACHE) > 4:
            _WORLD_CACHE.popitem(last=False)
        macro = _WORLD_CACHE[key].sample_macroepisode(entry['query_seed'], stage=stage)
        routes, denominator, multiplier = (macro.routes, macro.original_query_count,
                                           macro.reference_normalization_weight)
    else:
        arm = config.get('standard_prior', 'authored')
        if arm == 'authored':
            from .static_prior import generate_episode
            overrides = dict(config.get('static_overrides', {}))
            overrides.update(stage=stage, joint=config.get('finance_share', .2) == .2)
            episode = generate_episode(entry['world_seed'], **overrides)
        else:
            from .reference_prior import generate_reference_episode, ReferenceShape
            task = config.get('reference_task', 'mixed')
            if task == 'mixed':
                task = ('classification' if np.random.default_rng(
                    seed_for(entry['world_seed'], 'reference_task')).random() < .5 else 'regression')
            options = dict(arm=arm, task=task, stage=stage)
            if config.get('reference_shape'):
                options['shape'] = ReferenceShape(**config['reference_shape'])
            envelope = config.get('reference_envelope')
            if envelope and set(envelope).issubset({'1','2','3'}):
                envelope = envelope.get(str(stage))
            if envelope and np.random.default_rng(seed_for(
                    entry['world_seed'], 'reference_envelope')).random() < config.get('reference_envelope_probability', 1.):
                options['envelope'] = envelope
            episode = generate_reference_episode(entry['world_seed'], **options)
        routes, denominator, multiplier = [episode], len(episode.y_query), 1.
    return routes, denominator, multiplier, time.perf_counter() - start


class TaskProducer:
    def __init__(self, config):
        self.config = config
        self.depth = config.get('prefetch_tasks', 2)
        workers = config.get('producer_workers', 0)
        if config.get('standard_prior', 'authored') != 'authored' and os.environ.get('PYTHONHASHSEED') != '0':
            raise ValueError('Start Python with PYTHONHASHSEED=0 for the pinned reference generator')
        self.pool = (ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context('spawn'),
                                        initializer=_initialize_worker) if workers else None)

    def ordered(self, entries, stage):
        """Yield scheduled tasks in order; faster generators never change mix."""
        if self.pool is None:
            for entry in entries:
                start = time.perf_counter()
                value = prepare_task(self.config, entry, stage)
                yield entry, value, time.perf_counter() - start
            return
        iterator, pending = iter(entries), deque()
        def submit_one():
            entry = next(iterator, None)
            if entry is not None:
                pending.append((entry, self.pool.submit(prepare_task, self.config, entry, stage)))
        for _ in range(self.depth):
            submit_one()
        while pending:
            entry, future = pending.popleft()
            start = time.perf_counter()
            result = future.result()  # Propagate generator errors; never replace draws.
            wait = time.perf_counter() - start
            submit_one()
            yield entry, result, wait

    def close(self):
        if self.pool:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None
