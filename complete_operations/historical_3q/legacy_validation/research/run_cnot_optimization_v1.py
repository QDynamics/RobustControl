from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import sys,time,json,hashlib,traceback
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import lstsq,block_diag
import cnot_model_v1 as m
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=HERE/'runs'/'cnot_optimization_20260926_v1'

def write(p,v):
    text=json.dumps(v,indent=2,allow_nan=False)
    with p.open('x',encoding='utf-8') as f: f.write(text+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def full8_audit(p):
    sys.path.insert(0,str(ROOT/'src'))
    import manifold_full8_v1 as full
    import manifold_finite_fiber_v1 as ff
    es=[full.module_propagator(v) for v in p.reshape(3,44)]
    u,kr,kt=full.compose_full8(es)
    target=-(full.VISIBLE_ISOMETRY@m.GTARGET@full.VISIBLE_ISOMETRY.conj().T+full.COMPLEMENT_PROJECTOR)
    rr,jj=ff.block_module_constraints(p,True)
    tr,tj,_,_=ff.target_local_chart(p,m.GTARGET,True)
    sv=np.linalg.svd(np.vstack((jj,tj)),compute_uv=False)
    out={'full_target_fro':float(np.linalg.norm(u-target)),'K_rotation_fro':float(np.linalg.norm(kr)),'K_twist_fro':float(np.linalg.norm(kt)),'module_max_K_fro':float(max(np.linalg.norm(e.k_error_rotation) for e in es)),'module_max_twist_K_fro':float(max(np.linalg.norm(e.k_error_twist) for e in es)),'module_hidden_minus_I_max':float(max(np.linalg.norm(full.block(e.unitary,full.HIDDEN_ISOMETRY)+np.eye(2)) for e in es)),'jacobian_rank':int(np.sum(sv>1e-8)),'sigma_min':float(sv[-1]),'condition':float(sv[0]/sv[-1]),'time_t0_eta10':m.resources(p),'eta_diagnostics':{str(e):m.resources(p,e) for e in [1,10,100]}}
    out['endpoint_pass']=bool(out['full_target_fro']<1e-8)
    out['robust_pass']=bool(out['endpoint_pass'] and max(out['K_rotation_fro'],out['K_twist_fro'],out['module_max_K_fro'],out['module_max_twist_K_fro'],out['module_hidden_minus_I_max'])<1e-8)
    return out

def optimize(p0,kind,name):
    start=time.perf_counter(); history=[]; best=p0.copy(); best_time=m.resources(best)
    cache={'p':None}; cbcount=0
    target_model=m.combined if kind=='robust' else m.nominal
    def evaluate(x):
        if time.perf_counter()-start>1200: raise TimeoutError('1200 s optimization limit')
        if cache['p'] is None or not np.array_equal(x,cache['p']):
            data=target_model(x); cache.update(p=x.copy(),r=data[0],j=data[1])
        return cache['r'],cache['j']
    def fun(x):
        q=x.reshape(3,44)
        return float(np.sqrt(q[:,:15]**2+1e-14).sum()/10+np.sqrt(q[:,29:]**2+1e-14).sum())
    def grad(x):
        q=x.reshape(3,44); z=np.zeros_like(q); z[:,:15]=q[:,:15]/np.sqrt(q[:,:15]**2+1e-14)/10; z[:,29:]=q[:,29:]/np.sqrt(q[:,29:]**2+1e-14)
        return z.ravel()
    def callback(x):
        nonlocal best,best_time,cbcount
        r,_=evaluate(x); err=float(np.max(np.abs(r))); cost=m.resources(x); cbcount+=1
        actual=m.total(x,False)[0]; endpoint=float(np.linalg.norm(actual-m.UTARGET))
        feasible=err<1e-8 and endpoint<1e-8
        if feasible and cost<best_time: best=x.copy(); best_time=cost
        row={'iteration':cbcount,'time_t0':cost,'max_residual':err,'target_block_fro':endpoint,'feasible':bool(feasible),'elapsed_s':time.perf_counter()-start}; history.append(row)
        if cbcount%25==0: print(name,json.dumps(row),flush=True)
    bounds=[(-4*np.pi,4*np.pi) if k<15 or k>=29 else (-8*np.pi,8*np.pi) for _ in range(3) for k in range(44)]
    try:
        res=minimize(fun,p0,jac=grad,method='SLSQP',bounds=bounds,constraints={'type':'eq','fun':lambda x:evaluate(x)[0],'jac':lambda x:evaluate(x)[1]},callback=callback,options={'maxiter':400,'ftol':2e-11,'disp':False})
        callback(res.x)
        status={'scipy_success':bool(res.success),'message':str(res.message),'nit':int(res.nit),'nfev':int(res.nfev)}
        np.savez(OUT/(name+'_last_iterate.npz'),params=res.x)
    except TimeoutError as exc: status={'scipy_success':False,'message':str(exc)}
    # Polish the best feasible point, retaining its target branch and reporting actual cost.
    for _ in range(6):
        r,j=evaluate(best)
        if np.max(np.abs(r))<1e-12: break
        best+=lstsq(j,-r,cond=1e-12,lapack_driver='gelsy')[0]
    np.savez(OUT/(name+'.npz'),params=best.reshape(3,44))
    write(OUT/(name+'_trace.json'),history)
    audit=full8_audit(best); status.update(audit)
    status.update(initial_time_t0=m.resources(p0),fractional_time_reduction=1-m.resources(best)/m.resources(p0),elapsed_s=time.perf_counter()-start,global_optimality_claim=False)
    write(OUT/(name+'_result.json'),status)
    print('RESULT',name,json.dumps(status),flush=True)
    return best,status

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    source=HERE/'runs'/'cnot_continuation_20260926_v1'/'final.npz'
    p=np.load(source)['params'].ravel()
    write(OUT/'bindings.json',{str(x.relative_to(ROOT)):sha(x) for x in [source,Path(__file__),HERE/'cnot_model_v1.py',HERE/'V10_PROTOCOL_20260926_v1.md',ROOT/'src'/'manifold_full8_v1.py',ROOT/'src'/'manifold_finite_fiber_v1.py']})
    audit=full8_audit(p)
    audit['continuation_record_note']='Continuation completed and final.npz saved; original result.json serialization failed on a numpy boolean. No failed-run file was changed. This independent audit is the acceptance record.'
    write(OUT/'continuation_independent_acceptance.json',audit)
    print('INDEPENDENT',json.dumps(audit),flush=True)
    if not audit['robust_pass']: raise RuntimeError('CNOT source fails independent full8 audit')
    robust,rs=optimize(p,'robust','robust_time')
    baseline,bs=optimize(robust,'nominal','nominal_time')
    write(OUT/'result.json',{'robust':rs,'nominal':bs,'comparison_note':'The nominal feasible set contains the robust one. Both are local, budget-limited numerical optimizations.'})

if __name__=='__main__':
    try: main()
    except Exception:
        if OUT.exists(): (OUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
