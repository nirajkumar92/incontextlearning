"""Predict from a checkpoint and explicit support/query NPZ tables."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .data import load_table, validate_pair, assert_time_contract
from .inference import load_checkpoint, Predictor, FinancePredictor
from .runtime import atomic_json


def add_prediction_arguments(parser):
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--support', required=True)
    parser.add_argument('--query', required=True)
    parser.add_argument('--task', choices=['binary','multiclass','regression'], required=True)
    parser.add_argument('--classes', type=int, default=2)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--fp32', action='store_true', help='Disable default GPU BF16 autocast')
    parser.add_argument('--query-batch-size', type=int, default=512)
    parser.add_argument('--finance', action='store_true')
    parser.add_argument('--cutoff', type=float, help='Finance snapshot in the same timestamp unit as NPZ metadata')
    parser.add_argument('--complete-cohort', action='store_true', help='Assert every outcome in support is adjudicated, not merely reviewed alerts')
    parser.add_argument('--reference-counts', help='JSON array of past complete-cohort class counts')
    parser.add_argument('--hurdle', action='store_true', help='Target schema is nonnegative realized loss with exact zero atom')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--context-limit', type=int, default=32768)
    parser.add_argument('--search-cap', type=int, help='Optional finance negative-pool latency control; omitted searches every eligible negative')
    parser.add_argument('--cache-limit', type=int, default=64)


def construct_predictor(args, support):
    load_start = time.perf_counter()
    model, _ = load_checkpoint(args.checkpoint, args.device)
    load_seconds = time.perf_counter()-load_start
    if args.finance:
        if args.cutoff is None:
            raise ValueError('Finance prediction requires an explicit label-availability cutoff')
        predictor = FinancePredictor(model, seed=args.seed, search_cap=args.search_cap,
                     cache_limit=args.cache_limit, query_batch_size=args.query_batch_size, bf16=not args.fp32)
        ref = json.loads(args.reference_counts) if args.reference_counts else None
        predictor.fit_context(support['X'],support['y'],support['categorical'],args.task,
                     n_classes=args.classes, complete_cohort=args.complete_cohort,
                     reference_counts=ref,hurdle=args.hurdle)
    else:
        if len(support['X']) > args.context_limit:
            raise ValueError('Support exceeds explicit dense context limit; choose a declared selection policy instead of silent truncation')
        predictor = Predictor(model,args.query_batch_size,bf16=not args.fp32).fit_context(support['X'],support['y'],support['categorical'],
                     args.task,n_classes=args.classes,hurdle=args.hurdle,encoding_seed=args.seed)
    predictor.model_load_seconds = load_seconds
    return predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_prediction_arguments(parser)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    support, query = load_table(args.support), load_table(args.query, require_target=False)
    validate_pair(support, query)
    if args.finance:
        assert_time_contract(support, query, args.cutoff)
    start = time.perf_counter()
    predictor = construct_predictor(args, support)
    predictions = predictor.predict(query['X'])
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    path = Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path, prediction=predictions)
    atomic_json(str(path)+'.json', {'rows':len(predictions),'device':args.device,
                'load_context_and_query_seconds':elapsed,'finance':args.finance,
                'checkpoint':args.checkpoint,'no_query_targets_used':True})


if __name__ == '__main__':
    main()
