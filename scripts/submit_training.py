"""Prepare or submit one Slurm allocation with a scheduler-enforced run ceiling.

Prints a dry-run command unless --submit is supplied. This bounds the allocation
wall time; the trainer's between-update check alone cannot enforce a hard cap.
Reconcile killed allocations in the project ledger before resubmitting. This is
not a concurrent project reservation service.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tabular_foundation.train import validate_config


def command(config_path, output, *, gpus_per_node=8, resume=None, initialize_from=None, sbatch_args=()):
    config_path=Path(config_path).resolve()
    output=Path(output).resolve()
    cfg=json.loads(config_path.read_text())
    validate_config(cfg)
    gpus=cfg.get('allocated_gpus')
    if isinstance(gpus, bool) or not isinstance(gpus,int) or gpus < 1:
        raise ValueError('GPU submission requires positive allocated_gpus in the config')
    if isinstance(gpus_per_node,bool) or not isinstance(gpus_per_node,int) or gpus_per_node < 1 or gpus % gpus_per_node:
        raise ValueError('allocated_gpus must be divisible by gpus_per_node')
    if resume and initialize_from:
        raise ValueError('Choose resume or initialization')
    cap=cfg.get('gpu_hour_cap')
    if cap is None or not math.isfinite(cap):
        raise ValueError('A finite gpu_hour_cap is required for submission')
    charged=0.
    if resume:
        import torch
        saved=torch.load(resume,map_location='cpu',weights_only=True)
        if saved['config'] != cfg:
            raise ValueError('Resume configuration differs from checkpoint')
        if saved.get('run_id',str(Path(resume).resolve().parent)) != str(output):
            raise ValueError('Resume must use the original output directory')
        charged=float(saved.get('cumulative_gpu_hours',0.))
    elif (output/'manifest.json').exists():
        raise FileExistsError('Existing output requires explicit --resume')
    ledger=cfg.get('project_ledger')
    if ledger and Path(ledger).exists():
        data=json.loads(Path(ledger).read_text())
        charged=max(charged,float(data['runs'].get(str(output),0.)))
        if data.get('total_gpu_hours',sum(data['runs'].values())) >= data.get('cap_gpu_hours',50000):
            raise ValueError('Project ledger has exhausted its allocation')
    remaining=cap-charged
    if ledger and Path(ledger).exists():
        project_remaining = data.get('cap_gpu_hours',50000) - data.get('total_gpu_hours',sum(data['runs'].values()))
        remaining = min(remaining, project_remaining)
    seconds=math.floor(remaining*3600/gpus)
    if seconds < 1:
        raise ValueError('Run has exhausted its GPU-hour ceiling')
    days,remainder=divmod(seconds,86400)
    hours,remainder=divmod(remainder,3600)
    minutes,seconds=divmod(remainder,60)
    duration=f'{days}-{hours:02d}:{minutes:02d}:{seconds:02d}'
    for value in sbatch_args:
        if value.startswith(('--time','--nodes','--ntasks','--gpus','--gres','--export','--chdir','--wrap','--array')) or value.startswith(('-t','-N','-n','-G','-a')):
            raise ValueError('Site arguments cannot override resource, time, directory or launch contract')
    env=dict(os.environ, TFM_CONFIG=str(config_path), TFM_OUTPUT=str(output),
             TFM_GPUS_PER_NODE=str(gpus_per_node),TFM_ALLOCATED_GPUS=str(gpus), TFM_PYTHON=sys.executable,
             TFM_RESUME=str(Path(resume).resolve()) if resume else '',
             TFM_INITIALIZE_FROM=str(Path(initialize_from).resolve()) if initialize_from else '')
    parent=initialize_from or cfg.get('initial_checkpoint')
    if cfg.get('initial_checkpoint_required') and not (resume or parent):
        raise ValueError('This continuation requires --initialize-from or initial_checkpoint')
    if parent and not Path(parent).is_file():
        raise FileNotFoundError('Parent checkpoint does not exist: '+str(parent))
    args=['sbatch',*sbatch_args,'--chdir',str(ROOT),'--nodes',str(gpus//gpus_per_node),
          '--ntasks-per-node','1','--gpus-per-node',str(gpus_per_node),
          '--cpus-per-task',str(gpus_per_node*(cfg.get('producer_workers',0)+cfg.get('cpu_threads',1))),
          '--time',duration,'--export=ALL',str(ROOT/'scripts/launch_slurm.sh')]
    return args,env


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--gpus-per-node',type=int,default=8)
    p.add_argument('--resume')
    p.add_argument('--initialize-from')
    p.add_argument('--sbatch-arg',action='append',default=[],help='Site option, e.g. --sbatch-arg=--partition=gpu')
    p.add_argument('--submit',action='store_true')
    a=p.parse_args()
    args,env=command(a.config,a.output,gpus_per_node=a.gpus_per_node,resume=a.resume,initialize_from=a.initialize_from,sbatch_args=a.sbatch_arg)
    print(shlex.join(args))
    if a.submit:
        subprocess.run(args,env=env,check=True)

if __name__=='__main__':main()
