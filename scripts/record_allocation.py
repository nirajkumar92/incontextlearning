"""Reconcile actual cumulative scheduler-billed GPU-hours into the project ledger.

Use the exact absolute training output directory as its run ID. Evaluation gets
a separate stable allocation ID (emitted by the phase materializer). Submit
cumulative totals across attempts, not each attempt's incremental cost. Repeating
an identical total is idempotent; reducing a previously recorded total is rejected.
This records consumed allocation and never reserves future jobs or invokes Slurm.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tabular_foundation.runtime import ProjectLedger


def record(ledger,run_id,cumulative_gpu_hours,*,cap_gpu_hours=None):
    if not isinstance(run_id,str) or not run_id.strip():
        raise ValueError('run_id must be a nonempty stable allocation identity')
    if (isinstance(cumulative_gpu_hours,bool) or not isinstance(cumulative_gpu_hours,(int,float))
            or not math.isfinite(cumulative_gpu_hours) or cumulative_gpu_hours < 0):
        raise ValueError('cumulative_gpu_hours must be finite and nonnegative')
    path=Path(ledger)
    existing=json.loads(path.read_text()) if path.exists() else {}
    cap=existing.get('cap_gpu_hours',50000.) if cap_gpu_hours is None else cap_gpu_hours
    if (isinstance(cap,bool) or not isinstance(cap,(int,float)) or not math.isfinite(cap) or cap <= 0):
        raise ValueError('cap_gpu_hours must be finite and positive')
    if existing.get('cap_gpu_hours',cap) != cap:
        raise ValueError('Reconciliation cannot change the existing project cap')
    within=ProjectLedger(path,cap=cap).update(run_id,cumulative_gpu_hours)
    current=json.loads(path.read_text())
    return {'ledger':str(path.resolve()),'run_id':run_id,
            'cumulative_gpu_hours':current['runs'][run_id],
            'total_gpu_hours':current['total_gpu_hours'],'cap_gpu_hours':current['cap_gpu_hours'],
            'within_project_cap':within,'future_allocation_reserved':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger',required=True,type=Path)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--cumulative-gpu-hours',required=True,type=float)
    args=parser.parse_args()
    result=record(args.ledger,args.run_id,args.cumulative_gpu_hours)
    print(json.dumps(result,indent=2))
    if not result['within_project_cap']:
        raise SystemExit(2)  # Keep the actual spend recorded, even if over budget.

if __name__=='__main__':main()
