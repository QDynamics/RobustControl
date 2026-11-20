"""Larger-family baselines with exact nominal CNOT and explicit time accounting."""
from __future__ import annotations
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import time,json,sys
from pathlib import Path
import numpy as np
from scipy.linalg import lstsq,null_space
from scipy.optimize import linprog
import cnot_model_v1 as m
import run_cnot_optimization_v1 as io
HERE=Path(__file__).resolve().parent
OUT=HERE/'runs'/'cnot_baselines_20260926_v1'
CHARGED=np.array([44*k+i for k in range(3) for i in list(range(15))+list(range(29,44))])
WEIGHTS=np.tile(np.r_[np.ones(15)/10,np.ones(15)],3)

def endpoint_audit(p):
    sys.path.insert(0,str(io.ROOT/'src'))
    import manifold_full8_v1 as f
    es=[f.module_propagator(q) for q in p.reshape(3,44)]
    u,kr,kt=f.compose_full8(es)
    target=-(f.VISIBLE_ISOMETRY@m.GTARGET@f.VISIBLE_ISOMETRY.conj().T+f.COMPLEMENT_PROJECTOR)
    return {'full8_target_fro':float(np.linalg.norm(u-target)),'full8_KR_fro':float(np.linalg.norm(kr)),'full8_KT_fro':float(np.linalg.norm(kt)),'time_t0':m.resources(p),'nominal_jacobian_rank':int(np.linalg.matrix_rank(m.nominal(p)[1],tol=1e-8))}

def retract(x):
    for _ in range(10):
        r,j,u=m.nominal(x)
        if np.max(np.abs(r))<2e-12: break
        delta=lstsq(j,-r,cond=1e-12,lapack_driver='gelsy')[0]
        if np.linalg.norm(delta)>1: return None
        x=x+delta
    if np.linalg.norm(m.total(x,False)[0]-m.UTARGET)>1e-9: return None
    return x

def time_baseline(p):
    start=time.perf_counter(); initial=m.resources(p); history=[]; trust=.1; reason='iteration_budget'
    # Local linear L1 minimization handles zero-amplitude slots without a false gradient stall.
    c=np.r_[np.zeros(132),WEIGHTS]
    aub=np.zeros((180,222)); bub=np.zeros(180)
    for a,idx in enumerate(CHARGED):
        aub[a,idx]=1; aub[a,132+a]=-1
        aub[90+a,idx]=-1; aub[90+a,132+a]=-1
    for it in range(400):
        if time.perf_counter()-start>1200: reason='time_budget'; break
        r,j,_=m.nominal(p); bub[:90]=-p[CHARGED]; bub[90:]=p[CHARGED]
        bounds=[(-trust,trust)]*132+[(0,None)]*90
        lp=linprog(c,A_ub=aub,b_ub=bub,A_eq=np.c_[j,np.zeros((19,90))],b_eq=-r,bounds=bounds,method='highs')
        if not lp.success: reason='linear_subproblem_failed'; break
        delta=lp.x[:132]; predicted=m.resources(p)-lp.fun
        if predicted<1e-8: reason='local_linear_stationarity'; break
        accepted=False
        for back in range(12):
            x=retract(p+delta*2**(-back))
            if x is not None and m.resources(x)<m.resources(p)-1e-4*predicted*2**(-back): accepted=True; break
        row={'iteration':it,'accepted':accepted,'time_before':m.resources(p),'time_after':m.resources(x) if x is not None else None,'predicted_decrease':float(predicted),'halvings':back,'elapsed_s':time.perf_counter()-start}; history.append(row)
        if not accepted:
            trust/=2
            if trust<1e-5: reason='trust_limit'; break
            continue
        p=x
        if it%25==0: print('NOMINAL',json.dumps(row),flush=True)
    record=endpoint_audit(p); record.update(initial_time_t0=initial,iterations=len(history),stopping_reason=reason,elapsed_s=time.perf_counter()-start,global_optimum=False)
    np.savez(OUT/'nominal_time.npz',params=p.reshape(3,44)); io.write(OUT/'nominal_time_trace.json',history); io.write(OUT/'nominal_time_result.json',record)
    print('NOMINAL_RESULT',json.dumps(record),flush=True)
    return p

TRAIN=np.array([(x,y) for x in [-.01,0,.01] for y in [-.01,0,.01] if x or y])
def noise_objective(p,jac=True):
    val=0.; grad=np.zeros(132)
    for error in TRAIN:
        u,du=m.total(p,jac,errors=error); w=m.UTARGET.conj().T@u
        trace=np.trace(w[:4,:4])+2*np.trace(w[4:,4:])
        # Stable phase-aligned Frobenius metric; equivalent to 2-2|Tr|/8.
        phase=trace/abs(trace); diff=u-phase*m.UTARGET
        val+=(np.linalg.norm(diff[:4,:4])**2+2*np.linalg.norm(diff[4:,4:])**2)/8
        if jac:
            dt=m.UTARGET.conj().T@du
            dtrace=np.trace(dt[:,:4,:4],axis1=1,axis2=2)+2*np.trace(dt[:,4:,4:],axis1=1,axis2=2)
            grad+=-np.real(np.conj(phase)*dtrace)/4
    return float(val/len(TRAIN)),grad/len(TRAIN)

def noise_baseline(p):
    cap=m.resources(p); start=time.perf_counter(); history=[]; initial,_=noise_objective(p)
    reason='iteration_budget'
    # Check the independent directional derivative before using this objective.
    v=np.random.default_rng(20260926).normal(size=132); v/=np.linalg.norm(v); h=2e-6
    f,g=noise_objective(p); fd=(noise_objective(p+h*v,False)[0]-noise_objective(p-h*v,False)[0])/(2*h)
    io.write(OUT/'noise_gradient_check.json',{'analytic':float(g@v),'finite_difference':fd,'absolute_difference':float(abs(g@v-fd))})
    if abs(g@v-fd)>1e-8: raise RuntimeError('noise derivative check failed')
    for it in range(300):
        if time.perf_counter()-start>1200: reason='time_budget'; break
        f,g=noise_objective(p); r,j,_=m.nominal(p); n=null_space(j,rcond=1e-10)
        pg=n@(n.T@g); tg=n@(n.T@m.time_gradient(p)); d=-pg
        if m.resources(p)>cap-.05 and np.dot(tg,d)>0:
            d-=tg*(np.dot(tg,d)/np.dot(tg,tg)+.01*np.linalg.norm(pg)/np.linalg.norm(tg))
        dn=np.linalg.norm(d)
        if dn<1e-12: reason='stationary'; break
        d/=dn; accepted=False
        for back in range(14):
            step=.05*2**(-back); x=retract(p+step*d)
            if x is None or m.resources(x)>cap+1e-10: continue
            fn,_=noise_objective(x,False)
            if fn<f-1e-5*step*abs(np.dot(g,d)): accepted=True; break
        row={'iteration':it,'accepted':accepted,'objective_before':f,'objective_after':fn if x is not None and m.resources(x)<=cap+1e-10 else None,'time_t0':m.resources(p),'halvings':back,'elapsed_s':time.perf_counter()-start}; history.append(row)
        if not accepted: reason='line_search_limit'; break
        p=x
        if it%25==0: print('NOISE',json.dumps(row),flush=True)
    record=endpoint_audit(p); record.update(initial_training_objective=initial,final_training_objective=noise_objective(p,False)[0],time_cap=cap,iterations=len(history),stopping_reason=reason,elapsed_s=time.perf_counter()-start,global_optimum=False,training_points=TRAIN.tolist())
    np.savez(OUT/'noise_aware.npz',params=p.reshape(3,44)); io.write(OUT/'noise_aware_trace.json',history); io.write(OUT/'noise_aware_result.json',record)
    print('NOISE_RESULT',json.dumps(record),flush=True)

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    source=HERE/'runs'/'cnot_optimization_20260926_v2'/'robust_time.npz'
    p=np.load(source)['params'].ravel()
    io.write(OUT/'protocol_addendum.json',{'reason':'The nominal SLSQP comparison exhausted its 400-iteration budget without a feasible improvement. Preserve that negative record; use exact-target sequential linear programming with 400 iterations/1200 s. Noise-aware projected descent is limited to 300 iterations/1200 s. Neither changes target, errors, or physical cost.','nominal_trust_box':.1,'noise_tangent_step':.05,'noise_training_objective':'mean phase-aligned full8 Frobenius error squared divided by 8','noise_time_cap':m.resources(p),'extra_random_starts':0})
    io.write(OUT/'bindings.json',{str(f.relative_to(io.ROOT)):io.sha(f) for f in [Path(__file__),source,HERE/'cnot_model_v1.py',HERE/'V10_PROTOCOL_20260926_v1.md']})
    time_baseline(p.copy()); noise_baseline(p.copy())

if __name__=='__main__': main()
