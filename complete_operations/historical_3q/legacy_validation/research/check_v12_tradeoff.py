"""Independent analytic-coefficient check, with a fresh evidence directory."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
from pathlib import Path
import json,time,hashlib,sys
import numpy as np
import v12_model as v
import v12_channel as c
import v11_full8 as f
ROOT=Path(__file__).resolve().parent; out=ROOT/'runs/v12_tradeoff_check_v1'; out.mkdir(exist_ok=False)
start=time.perf_counter(); rows=[]
for name in ('whole','module_response','modular','nominal'):
    p=np.load(ROOT/('runs/v12_library_v1/'+name+'.npz'))['params']; ctrl=c.compile_controls(p)
    base=c.metrics(c.propagate(ctrl,np.zeros((3,2)),0,64))[1]
    h=1e-8; y1=c.metrics(c.propagate(ctrl,np.zeros((3,2)),h,64))[1]; y2=c.metrics(c.propagate(ctrl,np.zeros((3,2)),2*h,64))[1]
    derivative=(-3*base+4*y1-y2)/(2*h); predicted=1.5*v.gate_time(p)
    _,k=f.propagate(f.schedule(p),responses=True)
    rows.append({'kind':name,'full8_dephasing_slope':float(derivative),'predicted_slope':predicted,'relative_error':abs(float(derivative/predicted-1)),'module_drift_quadratic_coefficient':float(np.sum(abs(k)**2)/24),'source_sha256':hashlib.sha256((ROOT/('runs/v12_library_v1/'+name+'.npz')).read_bytes()).hexdigest()})
record={'pass':all(r['relative_error']<1e-5 for r in rows),'checks':rows,'gamma_step':h,'elapsed_s':time.perf_counter()-start,'python':sys.version,'numpy':np.__version__,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
with (out/'result.json').open('x',encoding='utf-8') as stream: json.dump(record,stream,indent=2)
print(json.dumps(record))
if not record['pass']: raise RuntimeError('Analytic coefficient check failed')
