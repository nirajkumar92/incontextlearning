"""Create a small reproducible tabular split for CLI smoke checks, not a benchmark."""
import argparse
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--output',default='runs/example_data');args=p.parse_args()
out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(42)
for split,n,offset in [('support',32,0),('validation',16,32),('query',16,48)]:
    X=rng.normal(size=(n,8)); X[:,2]=rng.integers(0,4,size=n)
    y=(X[:,0]+.7*X[:,1]+rng.normal(size=n)*.2>0).astype(int)
    X[rng.random(X.shape)<.05]=np.nan
    np.savez_compressed(out/(split+'.npz'),X=X,y=y,categorical=np.array([0,0,1,0,0,0,0,0],bool),ids=np.arange(offset,offset+n))
print(out)
