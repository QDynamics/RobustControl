"""Versioned correction for nonlocal endpoint restoration; original runs preserved."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import time,json,traceback,argparse
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
import run_v12_search as b
import v12_model as v
import v11_global_model as g
import cnot_model_v1 as m
HERE=Path(__file__).resolve().parent

def residual(p,kind):
    u,du,ks,dks,ds=g.product_responses(p,True)
    projected=kind in ('projected','noise_projected')
    if projected:
        u=u[:4,:4]; du=du[:,:4,:4]; target=m.UTARGET[:4,:4]
        z=np.trace(target.conj().T@u)/4; dz=np.trace(target.conj().T@du,axis1=1,axis2=2)/4
        diff=u-z*target; derivative=du-dz[:,None,None]*target
    else: diff=u-m.UTARGET; derivative=du
    r=np.r_[diff.real.ravel(),diff.imag.ravel()]
    j=np.vstack([derivative.real.reshape(132,-1).T,derivative.imag.reshape(132,-1).T])
    if not projected:
        da=np.zeros(132); da.reshape(3,44)[:,29:]=1; r=np.r_[r,np.asarray(p).reshape(3,44)[:,29:].sum()]; j=np.vstack([j,da])
    if kind=='projected':
        rr,jj=v.constraints(p,kind); r=np.r_[r,rr[15:]]; j=np.vstack([j,jj[15:]])
    elif kind=='whole':
        rr,jj=g.global_constraints(p)[:2]; r=np.r_[r,rr[19:]]; j=np.vstack([j,jj[19:]])
    return r,j

def restore(p,kind,out):
    start=time.perf_counter(); calls=0; cache={}
    def evaluate(x):
        nonlocal calls
        if time.perf_counter()-start>45: raise b.RestoreTimeout()
        if 'x' not in cache or not np.array_equal(cache['x'],x):
            r,j=residual(x,kind); cache.update(x=x.copy(),r=r,j=j); calls+=1
        return cache['r'],cache['j']
    try:
        opt=least_squares(lambda x:evaluate(x)[0],p,jac=lambda x:evaluate(x)[1],bounds=(-g.LIMITS,g.LIMITS),max_nfev=160,ftol=1e-11,xtol=1e-11,gtol=1e-11,method='trf')
        q=opt.x; status=opt.message
    except b.RestoreTimeout: q=cache.get('x',p); status='wall_time_limit'
    np.savez(out/'restoration_final.npz',params=q.reshape(3,44)); audit=v.matrix_audit(q,kind)
    b.write(out/'restoration.json',{'needed':True,'accepted':audit['accepted'],'evaluations':calls,'elapsed_s':time.perf_counter()-start,'status':status,'audit':audit})
    return q if audit['accepted'] else None

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['preflight','search','library']); args=parser.parse_args()
    if args.stage=='preflight':
        out=HERE/'runs/v12_restoration_preflight_v2_attempt2'; out.mkdir(exist_ok=False); records={}; rng=np.random.default_rng(12092741)
        p=np.load(b.NOM)['params'].ravel()+rng.normal(0,.01,132)
        for kind in ('projected','whole','noise_projected','noise_complete'):
            r,j=residual(p,kind); direction=rng.normal(size=132); direction/=np.linalg.norm(direction); h=1e-6
            fd=(residual(p+h*direction,kind)[0]-residual(p-h*direction,kind)[0])/(2*h)
            records[kind]=float(np.max(np.abs(fd-j@direction)))
        b.write(out/'result.json',{'pass':max(records.values())<1e-6,'directional_errors':records}); print(records); return
    if args.stage=='library':
        out=HERE/'runs/v12_library_v1'; out.mkdir(exist_ok=False); b.bindings(out)
        records=[]; selected={}
        for d in sorted((HERE/'runs').glob('v12_search_*')):
            rec=json.loads((d/'result.json').read_text()); rec['directory']=d.name; records.append(rec)
        for kind in (*v.ROBUST,'noise_projected','noise_complete','nominal'):
            good=[r for r in records if r['kind']==kind and r['accepted'] and (r['directory'].endswith('_v2') or '_warm_' in r['directory'] or kind not in ('projected','whole','noise_projected','noise_complete'))]
            if not good: continue
            best=min(good,key=lambda r:r['training_objective'] if kind.startswith('noise_') else r['audit']['gate_time_t0'])
            p=np.load(HERE/'runs'/best['directory']/'final.npz')['params']; np.savez(out/(kind+'.npz'),params=p)
            selected[kind]={'source':best['directory'],'sha256':b.sha(out/(kind+'.npz')),'audit':best['audit'],'training_objective':best['training_objective']}
        b.write(out/'result.json',{'selected':selected,'all_trials':records,'policy':'Frozen v2 addendum: no validation selection.'}); print(json.dumps(selected)); return
    if not json.loads((HERE/'runs/v12_restoration_preflight_v2_attempt2/result.json').read_text())['pass']: raise RuntimeError('No passing preflight')
    starts=[('nominal',np.load(b.NOM)['params'].ravel())]
    for seed in (12092701,12092702,12092703):
        rng=np.random.default_rng(seed); p=np.zeros((3,44)); p[:,:15]=rng.uniform(-np.pi,np.pi,(3,15)); p[:,15:29]=rng.uniform(-np.pi,np.pi,(3,14)); p[:,29:]=rng.uniform(-np.pi/4,np.pi/4,(3,15)); starts.append(('random'+str(seed),p.ravel()))
    for kind in ('projected','whole','noise_projected','noise_complete'):
        for label,p in starts:
            out=HERE/'runs'/('v12_search_'+kind+'_'+label+'_v2'); out.mkdir(exist_ok=False); b.bindings(out)
            b.write(out/'successor_bindings.json',{z.name:b.sha(z) for z in [Path(__file__),HERE/'run_v12_search.py',HERE/'V12_RESTORATION_ADDENDUM_20260927_v2.md']})
            np.savez(out/'initial.npz',params=p.reshape(3,44))
            try:
                q=restore(p,kind,out)
                if q is None:
                    rec={'kind':kind,'initialization':label,'accepted':False,'stopping_reason':'restoration_failed'}; b.write(out/'result.json',rec)
                else: rec=b.optimize(q,kind,out)
                print('SUCCESSOR',out.name,json.dumps(rec),flush=True)
            except Exception:
                (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8'); raise
if __name__=='__main__': main()
