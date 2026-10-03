"""Measure cold context setup and repeated warm predictions on the actual device."""
from __future__ import annotations
import argparse
import time
import numpy as np
import torch
from .predict import add_prediction_arguments, construct_predictor
from .data import load_table, validate_pair, assert_time_contract
from .runtime import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    add_prediction_arguments(p)
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    if args.repeats<1:
        raise ValueError('At least one measured repeat required')
    support,query=load_table(args.support),load_table(args.query,False)
    validate_pair(support,query)
    if args.finance:
        assert_time_contract(support,query,args.cutoff)
    def sync():
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
    sync(); start=time.perf_counter()
    predictor=construct_predictor(args,support)
    sync(); setup=time.perf_counter()-start
    start=time.perf_counter(); cold_prediction=predictor.predict(query['X']); sync()
    first=time.perf_counter()-start
    timings=[]
    for _ in range(args.repeats):
        start=time.perf_counter(); prediction=predictor.predict(query['X']); sync()
        timings.append(time.perf_counter()-start)
        if not np.allclose(prediction,cold_prediction,rtol=1e-5,atol=1e-6):
            raise AssertionError('Repeated predictions changed')
    peak=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None
    result={'device':args.device,'torch':torch.__version__,'hip':torch.version.hip,
            'support_rows':len(support['X']),'query_rows':len(query['X']),
            'load_and_setup_seconds':setup,'model_load_seconds':predictor.model_load_seconds,
            'context_setup_seconds':setup-predictor.model_load_seconds,'precision':'fp32' if args.fp32 or not args.device.startswith('cuda') else 'bf16','first_query_seconds':first,'cold_total_seconds':setup+first,
            'warm_seconds':timings,'warm_median_seconds':float(np.median(timings)),
            'warm_p95_seconds':float(np.quantile(timings,.95)),
            'peak_allocated_device_bytes':peak,'finance':args.finance,
            'note':'CPU timings are not GPU latency. First finance query includes lazily activated route prefills; warm timings include cache misses if cache_limit is too small.'}
    atomic_json(args.output,result); print(result)


if __name__=='__main__':
    main()
