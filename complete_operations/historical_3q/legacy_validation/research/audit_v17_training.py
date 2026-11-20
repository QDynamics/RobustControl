"""Independent physical-register readback of the frozen training objectives."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
from pathlib import Path
import json,time,hashlib,sys
import numpy as np
import v12_model as v
import v11_full8 as f
HERE=Path(__file__).resolve().parent
OUT=HERE/'runs/v17_independent_training_v1'; OUT.mkdir(exist_ok=False)
start=time.perf_counter(); lib=HERE/'runs/v17_library_v1'
selected=json.loads((lib/'result.json').read_text())['selected']; rows=[]
for kind,record in selected.items():
    p=np.load(lib/(kind+'.npz'))['params']; losses=[]
    for row in v.TRAIN:
        u,_=f.propagate(f.schedule(p),errors=v.errors(row))
        overlap=np.trace((f.P if kind in ('projected','noise_projected') else np.eye(8))@f.TARGET.conj().T@u)
        dim=4 if kind in ('projected','noise_projected') else 8
        losses.append(1-abs(overlap/dim)**2)
    value=float(np.mean(losses)+1.5e-5*v.gate_time(p))
    rows.append({'kind':kind,'physical_training_objective':value,'stored_training_objective':record['training_objective'],'difference':abs(value-record['training_objective'])})
result={'pass':all(r['difference']<1e-12 for r in rows),'checks':rows,'elapsed_s':time.perf_counter()-start,'python':sys.version,'numpy':np.__version__,'bindings':{str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),HERE/'v11_full8.py',HERE/'v12_model.py',lib/'result.json',*lib.glob('*.npz')]}}
with (OUT/'result.json').open('x',encoding='utf-8') as h:json.dump(result,h,indent=2)
print(json.dumps(result))
if not result['pass']:raise RuntimeError('Physical training objective mismatch')
