from __future__ import annotations
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import sys,time,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.linalg import null_space
import cnot_model_v1 as m
import run_cnot_optimization_v1 as io
import run_cnot_smoothing_v1 as smooth
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(io.ROOT/'src'))
import waveform_framework_v1 as wf
import waveform_rank_v1 as wr
OUT=HERE/'runs'/'cnot_smooth_rank_20260926_v1'

def main():
    OUT.mkdir(parents=True,exist_ok=False); start=time.perf_counter(); rows=[]
    source=HERE/'runs'/'cnot_smoothing_20260926_v1'/'width_0.45.npz'
    a=np.load(source); p=a['params']; p0=a['source_params']; width=float(a['width']); ss=smooth.schedules(p0,width)
    n=null_space(m.combined(p)[1],rcond=1e-10)
    model=SimpleNamespace(schedules=ss,source_params=p0,source_target_gate4=m.GTARGET,fiber_basis=n,z=np.zeros(6))
    io.write(OUT/'bindings.json',{str(f.relative_to(io.ROOT)):io.sha(f) for f in [Path(__file__),source,io.ROOT/'src'/'waveform_rank_v1.py',io.ROOT/'src'/'waveform_framework_v1.py']})
    result=wr.evaluate_rank_matrices(model,p,wf.WORKING_INTEGRATION)
    j=result.jacobian126x132; s=np.linalg.svd(j,compute_uv=False); sr=np.linalg.svd(j[:111],compute_uv=False)
    target_null=j[111:]@null_space(j[:111],rcond=1e-10); st=np.linalg.svd(target_null,compute_uv=False)
    checks=[]; rng=np.random.default_rng(20260926)
    for _ in range(2):
        v=rng.normal(size=132); v/=np.linalg.norm(v); h=2e-6
        rp=smooth.evaluate(p.ravel()+h*v,ss,tight=True,full=True)[0]
        rm=smooth.evaluate(p.ravel()-h*v,ss,tight=True,full=True)[0]
        checks.append(float(np.max(np.abs((rp-rm)/(2*h)-j@v))))
    data={'width_t0':width,'rank_R':int(np.sum(sr>1e-8)),'rank_augmented':int(np.sum(s>1e-8)),'rank_target_on_null':int(np.sum(st>1e-8)),'sigma_min_R':float(sr[-1]),'sigma_min_augmented':float(s[-1]),'sigma_min_target_on_null':float(st[-1]),'condition_augmented':float(s[0]/s[-1]),'full8_directional_fd_max_errors':checks,'elapsed_s':time.perf_counter()-start,'interval_certified':False}
    np.savez(OUT/'jacobians.npz',augmented=j,singular_values=s,target_on_null=target_null)
    io.write(OUT/'result.json',data); print(json.dumps(data),flush=True)

if __name__=='__main__': main()
