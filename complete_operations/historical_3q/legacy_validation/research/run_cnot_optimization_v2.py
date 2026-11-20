"""Successor: bounded projected descent avoids leaving the hidden-trace chart."""
from __future__ import annotations
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import time,json
from pathlib import Path
import numpy as np
from scipy.linalg import lstsq,null_space
import cnot_model_v1 as m
import run_cnot_optimization_v1 as audit
HERE=Path(__file__).resolve().parent
OUT=HERE/'runs'/'cnot_optimization_20260926_v2'

def main():
    OUT.mkdir(parents=True,exist_ok=False); audit.OUT=OUT
    source=HERE/'runs'/'cnot_continuation_20260926_v1'/'final.npz'
    p=np.load(source)['params'].ravel().copy(); start=time.perf_counter(); initial=m.resources(p)
    audit.write(OUT/'protocol_addendum.json',{'reason':'v1 unconstrained SLSQP trial left the local hidden-trace chart before returning a candidate. Preserve v1 and use projected time descent with local Newton retraction. Same fixed target, resources, 400 iterations and 1200 s limit. No result or acceptance domain changed.','maximum_tangent_step':.2,'retraction_iterations':10,'line_search_halvings':14})
    audit.write(OUT/'bindings.json',{str(f.relative_to(audit.ROOT)):audit.sha(f) for f in [Path(__file__),HERE/'run_cnot_optimization_v1.py',HERE/'cnot_model_v1.py',source,HERE/'V10_PROTOCOL_20260926_v1.md']})
    history=[]; reason='iteration_budget'; scale=.2
    for iteration in range(400):
        if time.perf_counter()-start>1200: reason='time_budget'; break
        r,j,_,_=m.combined(p); n=null_space(j,rcond=1e-10)
        gradient=m.time_gradient(p); projected=n@(n.T@gradient); gn=np.linalg.norm(projected)
        if gn<1e-7: reason='stationary'; break
        direction=-projected/gn; accepted=False
        for back in range(14):
            step=scale*2**(-back); x=p+step*direction
            try:
                for correction in range(10):
                    rr,jj,gg,_=m.combined(x)
                    if np.max(np.abs(rr))<2e-11: break
                    dx=lstsq(jj,-rr,cond=1e-12,lapack_driver='gelsy')[0]
                    if np.linalg.norm(dx)>.5: raise ValueError('retraction too far')
                    x+=dx
                feasible=bool(np.max(np.abs(rr))<1e-9 and np.linalg.norm(gg-m.GTARGET)<1e-9)
                if feasible and m.resources(x)<m.resources(p)-1e-5*step*gn:
                    accepted=True; break
            except ValueError: continue
        row={'iteration':iteration,'accepted':accepted,'time_before':m.resources(p),'time_after':m.resources(x),'projected_gradient_norm':float(gn),'step':step,'halvings':back,'elapsed_s':time.perf_counter()-start}
        history.append(row)
        if not accepted: reason='line_search_limit'; break
        p=x
        np.savez(OUT/f'robust_step_{iteration:04d}.npz',params=p)
        if iteration%20==0: print('ROBUST',json.dumps(row),flush=True)
    np.savez(OUT/'robust_time.npz',params=p.reshape(3,44))
    rs=audit.full8_audit(p); rs.update(stopping_reason=reason,iterations=len(history),initial_time_t0=initial,fractional_time_reduction=1-m.resources(p)/initial,elapsed_s=time.perf_counter()-start,global_optimality_claim=False)
    audit.write(OUT/'robust_time_trace.json',history); audit.write(OUT/'robust_time_result.json',rs)
    print('ROBUST_RESULT',json.dumps(rs),flush=True)
    baseline,bs=audit.optimize(p,'nominal','nominal_time')
    audit.write(OUT/'result.json',{'robust':rs,'nominal':bs,'note':'Numerical local improvements, not global optima. v1 failure retained.'})

if __name__=='__main__': main()
