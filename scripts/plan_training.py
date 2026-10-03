"""Turn measured per-stage update costs into a fixed, budgeted training horizon."""
import argparse
import json
from pathlib import Path
import math
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--template',default='configs/pilot.json')
p.add_argument('--measurements',required=True,help='JSON: allocated_gpus and stage_seconds_per_update[3]')
p.add_argument('--gpu-hours',type=float,required=True)
p.add_argument('--output',required=True)
a=p.parse_args();cfg=json.loads(Path(a.template).read_text());m=json.loads(Path(a.measurements).read_text())
gpus=int(m['allocated_gpus']);costs=m['stage_seconds_per_update']
if gpus<1 or len(costs)!=3 or any(not math.isfinite(c) or c<=0 for c in costs) or a.gpu_hours<=0:
    raise ValueError('Positive measured allocation, three costs, and budget required')
# Leave 5% for setup/checkpointing/cost uncertainty, all still charged by trainer.
usable=.95*a.gpu_hours
counts=[math.floor(usable*f*3600/(gpus*c)) for f,c in zip([.6,.25,.15],costs)]
if min(counts)<1:
    raise ValueError('Measured costs exceed available stage allocation')
ends=[sum(counts[:i+1]) for i in range(3)]
cfg.update(steps=ends[-1],stage_end_steps=ends,gpu_hour_cap=a.gpu_hours,allocated_gpus=gpus,
           update_reserve_seconds=1.25*max(costs))
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(cfg,indent=2)+'\n')
print(json.dumps({'output':str(out),'stage_updates':counts,'horizon':ends[-1],
                  'projected_training_gpu_hours':sum(n*c*gpus/3600 for n,c in zip(counts,costs)),
                  'note':'Projection from measurements, not a compute-optimal accuracy claim.'},indent=2))
