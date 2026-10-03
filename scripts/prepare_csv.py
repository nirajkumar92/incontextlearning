"""Convert a selected table snapshot to the explicit numeric NPZ interchange."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--csv',required=True);p.add_argument('--schema',required=True);p.add_argument('--output',required=True)
p.add_argument('--drop-unlabeled',action='store_true')
a=p.parse_args();schema=json.loads(Path(a.schema).read_text())
features=schema['features'];categorical=set(schema.get('categorical',[]));target=schema.get('target')
classes=schema.get('classes');mapping={str(v):i for i,v in enumerate(classes)} if classes else None
metadata=schema.get('metadata',{})
reserved={target,*metadata.values()}
if any(f in reserved for f in features) or len(set(features))!=len(features):
    raise ValueError('Features must be distinct and exclude target/metadata columns')
missing=set(schema.get('missing_values',['','NA','NaN','null','None']))
X=[];Y=[];meta={k:[] for k in metadata};dropped=0
hash_values={f:{} for f in categorical}
with open(a.csv,newline='') as file:
    reader=csv.DictReader(file)
    required=set(features)|set(metadata.values())|({target} if target else set())
    if not required.issubset(set(reader.fieldnames or [])):
        raise ValueError('CSV is missing schema columns: '+str(required-set(reader.fieldnames or [])))
    for row in reader:
        if target and row[target] in missing:
            if a.drop_unlabeled:
                dropped+=1;continue
            raise ValueError('Unknown outcome; --drop-unlabeled filters it explicitly instead of labeling it negative')
        values=[]
        for name in features:
            value=row[name]
            if value in missing:
                values.append(np.nan)
            elif name in categorical:
                # Fixed nominal ID, independent of split vocabulary or query labels.
                raw=json.dumps([name,value],ensure_ascii=False,separators=(',',':')).encode()
                code=int.from_bytes(hashlib.sha256(raw).digest()[:8],'big')>>12
                previous=hash_values[name].setdefault(code,value)
                if previous!=value:
                    raise ValueError('Nominal ID hash collision; use a persisted collision-free mapping upstream')
                values.append(float(code))
            else:
                values.append(float(value))
        X.append(values)
        if target:
            if mapping is not None:
                if row[target] not in mapping:
                    raise ValueError('Target not in the explicitly declared class vocabulary')
                Y.append(mapping[row[target]])
            else:
                Y.append(float(row[target]))
        for key,column in metadata.items():
            meta[key].append(row[column] if key=='ids' else float(row[column]))
arrays={'X':np.asarray(X,dtype=float).reshape(-1,len(features)),
        'categorical':np.array([f in categorical for f in features]),'feature_names':np.array(features)}
if target:
    arrays['y']=np.asarray(Y,dtype=np.int64 if mapping is not None else float)
    if not np.isfinite(arrays['y']).all():
        raise ValueError('Nonfinite target')
arrays.update({k:np.asarray(v) for k,v in meta.items()})
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**arrays)
print(json.dumps({'output':str(out),'rows':len(X),'dropped_unlabeled':dropped,'features':features}))
