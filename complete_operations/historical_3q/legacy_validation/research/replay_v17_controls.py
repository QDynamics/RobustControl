"""Portable independent replay of the 128-point confirmation; NEW output required."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
from pathlib import Path
import argparse,json,time,hashlib,sys
import numpy as np
import scipy
from scipy.integrate import solve_ivp
import v12_channel as c
import v12_model as v
import v11_full8 as f
from run_v12_validation import physical_audit
HERE=Path(__file__).resolve().parent

def independent_state_propagation(p,errors,gamma,states):
    """Adaptive physical density-matrix ODE; no local-channel factorization."""
    raw=np.asarray(p).reshape(3,44); caps=np.r_[np.full(15,10.),np.ones(15)]
    durations=[]
    for j in (2,1,0):
        t=v.duration_and_slope(np.r_[raw[j,:15],raw[j,29:]],caps)[0]
        for k in range(15): durations.extend([t[15+k],t[k]])
    rho=np.asarray(states).copy(); bits=np.array([[i>>a&1 for a in (2,1,0)] for i in range(8)])
    hamming=np.sum(bits[:,None,:]!=bits[None,:,:],axis=2)
    for idx,((j,a,h,_,_),tau) in enumerate(zip(f.schedule(raw),durations)):
        area_h=(1+errors[j,a])*h
        def rhs(t,flat):
            r=flat.reshape(rho.shape); field=2/tau*np.sin(np.pi*t/tau)**2
            comm=np.einsum('ik,kjb->ijb',area_h,r)-np.einsum('ikb,kj->ijb',r,area_h)
            return (-1j*field*comm-gamma*hamming[:,:,None]*r).ravel()
        sol=solve_ivp(rhs,(0,tau),rho.ravel(),method='DOP853',rtol=2e-10,atol=2e-12,max_step=tau/12)
        if not sol.success: raise RuntimeError(sol.message)
        rho=sol.y[:,-1].reshape(rho.shape)
        if idx<89: rho*=np.exp(-gamma*.001*hamming)[:,:,None]
    return rho

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--full-ode',action='store_true'); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False); start=time.perf_counter()
    lib=HERE/'runs/v17_library_v1'; record=json.loads((lib/'result.json').read_text()); saved=np.load(HERE/'runs/v17_confirmation_v1/arrays.npz')
    steps=json.loads((HERE/'runs/v17_channel_preflight_v1/result.json').read_text())['steps']; names=saved['names'].tolist(); results=[]; ode=[]
    for kind in names:
        p=np.load(lib/(kind+'.npz'))['params']; ctrl=c.compile_controls(p); audit=physical_audit(ctrl,kind)
        diffs=[]
        for a,b in ((0,0),(4,3),(6,5)):
            e=v.errors(saved['samples'][0],saved['alphas'][a]); s=c.propagate(ctrl,e,saved['gammas'][b],steps)
            diffs.append(float(np.max(np.abs(c.metrics(s)-saved['metrics'][names.index(kind),a,b,0]))))
        audit['kind']=kind; audit['map_replay_max']=max(diffs); audit['duration_difference']=abs(v.gate_time(p)-record['selected'][kind]['audit']['gate_time_t0']); results.append(audit)
        print('REPLAY',kind,audit['required_max'],max(diffs),flush=True)
        if args.full_ode and kind in ('module_response','modular','noise_projected','noise_complete'):
            rng=np.random.default_rng(12092761); states=[]
            for _ in range(3):
                psi=rng.normal(size=8)+1j*rng.normal(size=8); psi/=np.linalg.norm(psi); states.append(np.outer(psi,psi.conj()))
            states.append(f.P/4); bank=np.stack(states,axis=2); e=v.errors(saved['samples'][0],.3); gamma=.001
            exact=independent_state_propagation(p,e,gamma,bank)
            packed=bank.reshape(64,4)[c.PERM]; evolved=c.propagate(ctrl,e,gamma,steps,packed)[c.INV].reshape(8,8,4)
            diff=float(np.max(np.abs(evolved-exact))); ode.append({'kind':kind,'max_element_difference':diff,'fro_differences':np.linalg.norm((evolved-exact).reshape(64,4),axis=0).tolist()}); print('FULL_ODE',ode[-1],flush=True)
    passed=all(r['passed'] and r['map_replay_max']<1e-12 and r['duration_difference']<1e-10 for r in results) and all(r['max_element_difference']<2e-7 for r in ode)
    payload={'pass':passed,'results':results,'independent_full8_ode':ode,'elapsed_s':time.perf_counter()-start,'versions':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__},'bound_sources':{q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in [Path(__file__),HERE/'v12_channel.py',HERE/'v11_full8.py',lib/'result.json']}}
    with (out/'result.json').open('x',encoding='utf-8') as h: json.dump(payload,h,indent=2,allow_nan=False)
    if not passed: raise RuntimeError('Replay acceptance failed')
if __name__=='__main__': main()
