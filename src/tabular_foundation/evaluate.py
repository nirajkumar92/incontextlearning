"""Evaluate one predeclared split; operating thresholds use validation only."""
from __future__ import annotations
import argparse
import math
import time
import numpy as np
from .predict import add_prediction_arguments, construct_predictor
from .data import load_table, validate_pair, assert_time_contract, assert_validation_time_contract
from .metrics import binary_metrics, multiclass_metrics, regression_metrics, select_threshold
from .runtime import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_prediction_arguments(p)
    p.add_argument('--validation', help='Optional later development period for threshold selection')
    p.add_argument('--minimum-precision', type=float, default=.8)
    p.add_argument('--review-budget', type=int)
    p.add_argument('--threshold', type=float, default=.5)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    support, query = load_table(args.support), load_table(args.query)
    validate_pair(support,query)
    if args.finance:
        assert_time_contract(support,query,args.cutoff)
    validation = load_table(args.validation) if args.validation else None
    if validation is not None:
        validate_pair(support,validation); validate_pair(validation,query)
        if args.finance:
            assert_time_contract(support,validation,args.cutoff)
            assert_validation_time_contract(validation, query)
    start=time.perf_counter()
    predictor=construct_predictor(args,support)
    threshold=args.threshold
    if validation is not None and args.task=='binary':
        pv=predictor.predict(validation['X'])[:,1]
        threshold=select_threshold(validation['y'],pv,args.minimum_precision)
    prediction=predictor.predict(query['X'])
    if args.task=='binary':
        result=binary_metrics(query['y'],prediction[:,1],threshold,args.review_budget)
    elif args.task=='multiclass':
        result=multiclass_metrics(query['y'],prediction)
    else:
        result=regression_metrics(query['y'],prediction)
    result.update({'elapsed_seconds':time.perf_counter()-start,'finance':args.finance,
                   'threshold_selected_on_validation':validation is not None and args.task=='binary',
                   'inference_track':'frozen_weights','device':args.device,
                   'scope':'one supplied split; not the official TabArena aggregate'})
    atomic_json(args.output,result)
    print(result)


if __name__=='__main__':
    main()
