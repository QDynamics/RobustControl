"""Predeclared matched L1 searches. Every output directory must be new."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
from pathlib import Path
import time,json,hashlib,sys,traceback,argparse
import numpy as np
import scipy
from scipy.linalg import lstsq,block_diag
from scipy.optimize import linprog
import v11_global_model as g
import v11_full8 as f
import cnot_model_v1 as m

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'runs/cnot_optimization_20260926_v2/robust_time.npz'
PROTOCOL=HERE/'V11_PROTOCOL_20260927_v1.md'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):
    with p.open('x',encoding='utf-8') as stream: json.dump(obj,stream,indent=2,allow_nan=False)

def preflight(p,out):
    r,j,u,k,ds=g.global_constraints(p)
    u8,k8=f.propagate(f.schedule(p),responses=True)
    compact=block_diag(f.VD.conj().T@u8@f.VD,f.VH.conj().T@u8@f.VH)
    shared=k8.sum(axis=0)
    response_diff=[]
    for a in range(2):
        compact_k=block_diag(f.VD.conj().T@shared[a]@f.VD,f.VH.conj().T@shared[a]@f.VH)
        response_diff.append(float(np.linalg.norm(compact_k-k[a])))
    rng=np.random.default_rng(20260927); checks=[]; h=2e-6
    for _ in range(3):
        v=rng.normal(size=132); v/=np.linalg.norm(v)
        fd=(g.global_constraints(p+h*v,False)[0]-g.global_constraints(p-h*v,False)[0])/(2*h)
        checks.append(float(np.max(np.abs(fd-j@v))))
    sv=np.linalg.svd(j,compute_uv=False)
    record={'physical_audit':f.audit(p),'block_propagator_difference':float(np.linalg.norm(u-compact)),
            'block_response_differences':response_diff,'jacobian_directional_max_differences':checks,
            'whole_gate_rank':int(np.sum(sv>1e-8)),'whole_gate_singular_values':sv.tolist(),
            'initial_residual':float(np.max(np.abs(r))),'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__}
    record['pass']=bool(max(checks)<1e-6 and max(response_diff)<1e-9 and record['block_propagator_difference']<1e-9 and record['whole_gate_rank']==52 and record['physical_audit']['target_fro']<1e-9 and max(record['physical_audit']['shared_response_fro'])<1e-9)
    write(out/'preflight.json',record)
    if not record['pass']: raise RuntimeError('Preflight failed')
    print('PREFLIGHT',json.dumps({k:record[k] for k in ('pass','whole_gate_rank','jacobian_directional_max_differences')}),flush=True)

def feasible(x,kind,tol=2e-10):
    if np.max(np.abs(x)-g.LIMITS)>1e-10: return False
    r,_,u,k,ds=g.evaluation(x,kind,False)
    if np.max(np.abs(r))>tol or np.linalg.norm(u-m.UTARGET)>1e-8: return False
    if kind=='whole_gate': return max(np.linalg.norm(a) for a in k)<1e-8
    return max(np.linalg.norm(a) for d in ds for a in d[2])<1e-8

def retract(x,kind):
    for _ in range(12):
        try: r,j,*_=g.evaluation(x,kind)
        except ValueError: return None
        if np.max(np.abs(r))<2e-12: break
        delta=lstsq(j,-r,cond=1e-12,lapack_driver='gelsy')[0]
        if np.linalg.norm(delta)>1: return None
        accepted=False
        for b in range(8):
            y=x+delta*2**(-b)
            if np.max(np.abs(y)-g.LIMITS)>1e-10: continue
            try: rr=g.evaluation(y,kind,False)[0]
            except ValueError: continue
            if np.linalg.norm(rr)<np.linalg.norm(r): x=y; accepted=True; break
        if not accepted: return None
    try: return x if feasible(x,kind) else None
    except ValueError: return None

def optimize(p,kind,out):
    start=time.perf_counter(); history=[]; iterates=[p.copy()]; trust=.1; reason='iteration_budget'
    initial=m.resources(p); cost=np.r_[np.zeros(132),g.WEIGHTS]
    aub=np.zeros((180,222))
    for a,idx in enumerate(g.CHARGED):
        aub[a,idx]=1; aub[a,132+a]=-1
        aub[90+a,idx]=-1; aub[90+a,132+a]=-1
    with (out/'trace.jsonl').open('x',encoding='utf-8') as log:
        for it in range(400):
            if time.perf_counter()-start>1200: reason='time_budget'; break
            r,j,*_=g.evaluation(p,kind)
            lower=np.maximum(-trust,-g.LIMITS-p); upper=np.minimum(trust,g.LIMITS-p)
            bounds=list(zip(lower,upper))+[(0,None)]*90
            lp=linprog(cost,A_ub=aub,b_ub=np.r_[-p[g.CHARGED],p[g.CHARGED]],
                       A_eq=np.c_[j,np.zeros((len(r),90))],b_eq=-r,bounds=bounds,method='highs')
            if not lp.success:
                reason='linear_subproblem_failed'; write(out/'linear_failure.json',{'iteration':it,'message':lp.message}); break
            predicted=m.resources(p)-lp.fun
            if predicted<1e-8: reason='local_linear_stationarity'; break
            delta=lp.x[:132]; accepted=False; x=None
            for back in range(12):
                x=retract(p+delta*2**(-back),kind)
                if x is not None and m.resources(x)<m.resources(p)-1e-4*predicted*2**(-back):
                    accepted=True; break
            row={'iteration':it,'accepted':accepted,'time_before':m.resources(p),
                 'time_after':m.resources(x) if x is not None else None,'predicted_decrease':float(predicted),
                 'halvings':back,'trust':trust,'elapsed_s':time.perf_counter()-start}
            history.append(row); log.write(json.dumps(row)+'\n'); log.flush()
            if accepted:
                p=x; iterates.append(p.copy())
            else:
                trust/=2
                if trust<1e-5: reason='trust_limit'; break
            if it%25==0: print(kind.upper(),json.dumps(row),flush=True)
            if (it+1)%50==0: np.savez(out/('checkpoint_%04d.npz'%(it+1)),params=p)
    np.savez(out/'final.npz',params=p.reshape(3,44),accepted_iterates=np.array(iterates))
    audit=g.audit(p,kind); physical=f.audit(p)
    record={'kind':kind,'initial_time_t0':initial,'result':audit,'physical_acceptance':physical,
            'elapsed_s':time.perf_counter()-start,'iterations':len(history),'accepted_steps':len(iterates)-1,
            'stopping_reason':reason,'global_optimum':False,
            'accepted':bool(feasible(p,kind) and physical['target_fro']<1e-8 and max(physical['shared_response_fro'])<1e-8),
            'final_sha256':sha(out/'final.npz')}
    if kind=='modular': record['accepted']=bool(record['accepted'] and np.max(physical['module_response_fro'])<1e-8)
    write(out/'result.json',record)
    print('RESULT',json.dumps(record),flush=True)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('kind',choices=['preflight','modular','whole_gate']); args=parser.parse_args()
    out=HERE/'runs'/('v11_'+args.kind+'_20260927_v1'); out.mkdir(parents=True,exist_ok=False)
    try:
        write(out/'bindings.json',{p.name:sha(p) for p in [Path(__file__),PROTOCOL,SOURCE,HERE/'v11_global_model.py',HERE/'v11_full8.py',HERE/'cnot_model_v1.py']})
        p=np.load(SOURCE,allow_pickle=False)['params'].ravel()
        if args.kind=='preflight': preflight(p,out)
        else:
            prereq=HERE/'runs/v11_preflight_20260927_v1/preflight.json'
            if not json.loads(prereq.read_text())['pass']: raise RuntimeError('Missing passing preflight')
            optimize(p,args.kind,out)
    except Exception:
        with (out/'failure.txt').open('x',encoding='utf-8') as stream: stream.write(traceback.format_exc())
        raise

if __name__=='__main__': main()
