"""Frozen bounded searches. All output paths must be new."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
from pathlib import Path
import time,json,hashlib,sys,traceback,argparse
import numpy as np
import scipy
from scipy.linalg import lstsq
from scipy.optimize import linprog,least_squares
from scipy.integrate import solve_ivp
import v12_model as v
import v11_global_model as g
import v11_full8 as f
import cnot_model_v1 as m
HERE=Path(__file__).resolve().parent
PROTOCOL=HERE/'V12_PROTOCOL_20260927_v1.md'
WARM=HERE/'runs/v11_modular_20260927_v1/final.npz'
NOM=HERE/'runs/cnot_baselines_20260926_v1/nominal_time.npz'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):
    with p.open('x',encoding='utf-8') as h: json.dump(obj,h,indent=2,allow_nan=False)
def bindings(out):
    write(out/'bindings.json',{p.name:sha(p) for p in [Path(__file__),PROTOCOL,WARM,NOM,HERE/'v12_model.py',HERE/'cnot_model_v1.py',HERE/'v11_global_model.py',HERE/'v11_full8.py']})
    write(out/'environment.json',{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'dtype':'float64/complex128','threads':1})

def preflight(out):
    p=np.load(WARM)['params'].ravel(); rng=np.random.default_rng(12092731); checks={}; h=1e-6
    for kind in (*v.ROBUST,'noise_projected','noise_complete','nominal'):
        t=time.perf_counter(); r,j=v.constraints(p,kind); direction=rng.normal(size=132); direction/=np.linalg.norm(direction)
        fd=(v.constraints(p+h*direction,kind,False)[0]-v.constraints(p-h*direction,kind,False)[0])/(2*h)
        checks[kind]={'rows':len(r),'rank':int(np.linalg.matrix_rank(j,tol=1e-8)),'fd_max':float(np.max(np.abs(fd-j@direction))),'elapsed_s':time.perf_counter()-t,'audit':v.matrix_audit(p,kind)}
    for kind in ('noise_projected','noise_complete'):
        direction=rng.normal(size=132); direction/=np.linalg.norm(direction); t=time.perf_counter()
        value,grad=v.loss(p,kind)
        fd=(v.loss(p+h*direction,kind,False)[0]-v.loss(p-h*direction,kind,False)[0])/(2*h)
        checks[kind]['objective_fd_abs']=abs(float(fd-grad@direction)); checks[kind]['objective_eval_s']=time.perf_counter()-t; checks[kind]['objective']=value
    # Verify inverse-filter output with an independent adaptive differential equation.
    ode=[]
    for a,cap in ((.0001,1.),(.15,1.),(-2.5,10.),(np.pi,10.)):
        tau=float(v.duration_and_slope(np.array([a]),cap)[0][0]); tb=v.TB
        def rhs(t,y):
            ideal=2*a/tau*np.sin(np.pi*t/tau)**2
            derivative=2*np.pi*a/tau**2*np.sin(2*np.pi*t/tau)
            cmd=ideal+tb*derivative
            return [(cmd-y[0])/tb,y[0]]
        sol=solve_ivp(rhs,(0,tau),[0.,0.],rtol=2e-11,atol=2e-13,method='DOP853',dense_output=True,max_step=tau/16)
        t=np.linspace(0,tau,1001); ideal=2*a/tau*np.sin(np.pi*t/tau)**2
        cmd=ideal+tb*2*np.pi*a/tau**2*np.sin(2*np.pi*t/tau)
        ode.append({'area':a,'cap':cap,'tau':tau,'area_error':abs(float(sol.y[1,-1]-a)),'output_error':float(np.max(np.abs(sol.sol(t)[0]-ideal))),'peak_command':float(np.max(np.abs(cmd)))})
    record={'constraints':checks,'actuator_ode':ode,'physical_warm':f.audit(p)}
    record['pass']=bool(all(x['fd_max']<1e-6 and x['audit']['accepted'] for x in checks.values()) and all(x.get('objective_fd_abs',0)<1e-7 for x in checks.values()) and all(x['area_error']<1e-9 and x['output_error']<1e-8 and x['peak_command']<=x['cap']+1e-9 for x in ode))
    write(out/'result.json',record); print(json.dumps(record),flush=True)
    if not record['pass']: raise RuntimeError('Preflight failed')

def retract(p,kind):
    p=p.copy()
    for _ in range(12):
        try: r,j=v.constraints(p,kind)
        except (ValueError,np.linalg.LinAlgError): return None
        if np.max(np.abs(r))<2e-12: break
        d=lstsq(j,-r,cond=1e-12,lapack_driver='gelsy')[0]
        if np.linalg.norm(d)>1: return None
        for b in range(10):
            q=p+d*2**(-b)
            if np.max(np.abs(q)-g.LIMITS)>1e-10: continue
            try: rq=v.constraints(q,kind,False)[0]
            except ValueError: continue
            if np.linalg.norm(rq)<np.linalg.norm(r): p=q; break
        else: return None
    return p if v.matrix_audit(p,kind)['accepted'] else None

class RestoreTimeout(Exception): pass
def restore(p,kind,out):
    if v.matrix_audit(p,kind)['accepted']:
        write(out/'restoration.json',{'needed':False,'accepted':True,'evaluations':0,'elapsed_s':0}); return p
    start=time.perf_counter(); calls=0; cache={}
    def evaluate(x):
        nonlocal calls
        if time.perf_counter()-start>45: raise RestoreTimeout()
        if 'x' not in cache or not np.array_equal(cache['x'],x):
            r,j=v.constraints(x,kind); cache.update(x=x.copy(),r=r,j=j); calls+=1
        return cache['r'],cache['j']
    try:
        opt=least_squares(lambda x:evaluate(x)[0],p,jac=lambda x:evaluate(x)[1],bounds=(-g.LIMITS,g.LIMITS),max_nfev=160,ftol=1e-11,xtol=1e-11,gtol=1e-11,method='trf')
        q=opt.x; status=opt.message
    except RestoreTimeout:
        q=cache.get('x',p); status='wall_time_limit'
    np.savez(out/'restoration_final.npz',params=q.reshape(3,44))
    audit=v.matrix_audit(q,kind)
    write(out/'restoration.json',{'needed':True,'accepted':audit['accepted'],'evaluations':calls,'elapsed_s':time.perf_counter()-start,'status':status,'audit':audit})
    return q if audit['accepted'] else None

def optimize(p,kind,out):
    noise=kind.startswith('noise_'); start=time.perf_counter(); trust=.1; reason='iteration_budget'; iterates=[p.copy()]; rows=[]
    aub=np.zeros((180,222))
    for k,idx in enumerate(g.CHARGED):
        aub[k,idx]=1; aub[k,132+k]=-1; aub[90+k,idx]=-1; aub[90+k,132+k]=-1
    current=v.loss(p,kind)[0] if noise else v.gate_time(p)
    with (out/'trace.jsonl').open('x',encoding='utf-8') as log:
        for it in range(120):
            if time.perf_counter()-start>=90: reason='wall_time_limit'; break
            r,j=v.constraints(p,kind)
            if noise:
                current,grad=v.loss(p,kind)
                grad-=j.T@lstsq(j@j.T,j@grad,cond=1e-12,lapack_driver='gelsy')[0]
                norm=np.max(np.abs(grad))
                if norm<1e-12: reason='projected_stationarity'; break
                delta=-trust*grad/norm; predicted=-float(grad@delta)
            else:
                _,weights=v.duration_and_slope(p[g.CHARGED]); cost=np.r_[np.zeros(132),weights]
                lp=linprog(cost,A_ub=aub,b_ub=np.r_[-p[g.CHARGED],p[g.CHARGED]],A_eq=np.c_[j,np.zeros((len(r),90))],b_eq=-r,bounds=list(zip(np.maximum(-trust,-g.LIMITS-p),np.minimum(trust,g.LIMITS-p)))+[(0,None)]*90,method='highs')
                if not lp.success: reason='linear_subproblem_failed'; write(out/'linear_failure.json',{'iteration':it,'message':lp.message}); break
                delta=lp.x[:132]; predicted=float(weights@np.abs(p[g.CHARGED])-lp.fun)
                if predicted<1e-8: reason='local_linear_stationarity'; break
            accepted=False
            for back in range(10):
                candidate=p+delta*2**(-back)
                if np.max(np.abs(candidate)-g.LIMITS)>1e-10: continue
                q=retract(candidate,kind)
                if q is None: continue
                if noise and v.gate_time(q)>120: continue
                nxt=v.loss(q,kind,False)[0] if noise else v.gate_time(q)
                if nxt<current-1e-4*predicted*2**(-back): accepted=True; break
            row={'iteration':it,'accepted':accepted,'objective_before':current,'objective_after':nxt if accepted else current,'gate_time_t0':v.gate_time(q if accepted else p),'backtracks':back,'trust':trust,'elapsed_s':time.perf_counter()-start}
            rows.append(row); log.write(json.dumps(row)+'\n'); log.flush()
            if accepted: p=q; current=nxt; iterates.append(p.copy())
            else:
                trust/=2
                if trust<1e-5: reason='trust_limit'; break
            if it%40==0: print(out.name,json.dumps(row),flush=True)
    np.savez(out/'final.npz',params=p.reshape(3,44),accepted_iterates=np.array(iterates))
    audit=v.matrix_audit(p,kind)
    rec={'kind':kind,'accepted':audit['accepted'],'audit':audit,'iterations':len(rows),'accepted_steps':len(iterates)-1,'elapsed_s':time.perf_counter()-start,'stopping_reason':reason,'global_optimum':False,'training_objective':v.loss(p,'noise_projected' if kind in ('projected','noise_projected') else 'noise_complete',False)[0],'final_sha256':sha(out/'final.npz')}
    write(out/'result.json',rec); return rec

def run_trial(kind,label,p):
    out=HERE/'runs'/('v12_search_'+kind+'_'+label+'_v1'); out.mkdir(exist_ok=False)
    bindings(out); np.savez(out/'initial.npz',params=p.reshape(3,44))
    try:
        q=restore(p,kind,out)
        if q is None:
            rec={'kind':kind,'initialization':label,'accepted':False,'stopping_reason':'restoration_failed'}; write(out/'result.json',rec)
        else: rec=optimize(q,kind,out)
        print('RESULT',out.name,json.dumps(rec),flush=True)
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8'); raise

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['preflight','matched','multistart','library']); args=parser.parse_args()
    warm=np.load(WARM)['params'].ravel(); nominal=np.load(NOM)['params'].ravel()
    if args.stage=='preflight':
        out=HERE/'runs/v12_preflight_v1'; out.mkdir(exist_ok=False); bindings(out); preflight(out); return
    if not json.loads((HERE/'runs/v12_preflight_v1/result.json').read_text())['pass']: raise RuntimeError('No passing preflight')
    if args.stage=='matched':
        for kind in (*v.ROBUST,'noise_projected','noise_complete'): run_trial(kind,'warm',warm)
        run_trial('nominal','nominal',nominal)
    elif args.stage=='multistart':
        starts=[('nominal',nominal)]
        for seed in (12092701,12092702,12092703):
            rng=np.random.default_rng(seed); p=np.zeros((3,44)); p[:,:15]=rng.uniform(-np.pi,np.pi,(3,15)); p[:,15:29]=rng.uniform(-np.pi,np.pi,(3,14)); p[:,29:]=rng.uniform(-np.pi/4,np.pi/4,(3,15)); starts.append(('random'+str(seed),p.ravel()))
        for kind in ('projected','whole','noise_projected','noise_complete'):
            for label,p in starts: run_trial(kind,label,p)
    else:
        out=HERE/'runs/v12_library_v1'; out.mkdir(exist_ok=False); bindings(out); records=[]; selected={}
        for d in sorted((HERE/'runs').glob('v12_search_*_v1')):
            rec=json.loads((d/'result.json').read_text()); rec['directory']=d.name; records.append(rec)
        for kind in (*v.ROBUST,'noise_projected','noise_complete','nominal'):
            good=[r for r in records if r['kind']==kind and r['accepted']]
            if not good: continue
            best=min(good,key=lambda r:r['training_objective'] if kind.startswith('noise_') else r['audit']['gate_time_t0'])
            p=np.load(HERE/'runs'/best['directory']/'final.npz')['params']; np.savez(out/(kind+'.npz'),params=p)
            selected[kind]={'source':best['directory'],'sha256':sha(out/(kind+'.npz')),'audit':best['audit'],'training_objective':best['training_objective']}
        write(out/'result.json',{'selected':selected,'all_trials':records,'policy':'Frozen training selection; validation not used.'}); print(json.dumps(selected),flush=True)
if __name__=='__main__': main()
