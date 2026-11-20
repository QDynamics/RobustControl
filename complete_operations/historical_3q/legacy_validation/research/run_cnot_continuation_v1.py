from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import sys,json,time,hashlib,platform,traceback
from pathlib import Path
import numpy as np
import scipy
from scipy.linalg import schur,lstsq
import cnot_model_v1 as model

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=HERE/'runs'/'cnot_continuation_20260926_v1'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(path,obj):
    with path.open('x',encoding='utf-8') as f: json.dump(obj,f,indent=2,allow_nan=False)

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    source=ROOT/'runs'/'manifold_finite_fiber_20260913_v1'/'arrays.npz'
    a=np.load(source); p=a['source_params'].ravel().copy(); source_gate=a['source_target_gate4']
    np.savez(OUT/'inputs.npz',source_params=p,source_gate=source_gate,target=model.GTARGET)
    protocol=HERE/'V10_PROTOCOL_20260926_v1.md'
    (OUT/'protocol.md').write_bytes(protocol.read_bytes())
    write(OUT/'bindings.json',{str(x.relative_to(ROOT)):sha(x) for x in [source,protocol,Path(__file__),HERE/'cnot_model_v1.py']})
    write(OUT/'environment.json',{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__,'threads':1})
    sys.path.insert(0,str(ROOT/'src'))
    import manifold_full8_v1 as full
    import manifold_finite_fiber_v1 as ff
    r,j,g,ds=model.combined(p,source_gate)
    rr,jj=ff.block_module_constraints(p,True)
    tr,tj,gg,_=ff.target_local_chart(p,source_gate,True)
    residual_diff=float(np.max(np.abs(r-np.r_[rr,tr])))
    jac_diff=float(np.max(np.abs(j-np.vstack((jj,tj)))))
    u6,_=model.total(p)
    ev=[full.module_propagator(x) for x in p.reshape(3,44)]
    u8,_,_=full.compose_full8(ev)
    from scipy.linalg import block_diag
    u6_ind=block_diag(full.block(u8,full.VISIBLE_ISOMETRY),full.block(u8,full.HIDDEN_ISOMETRY))
    fderr=[]; rng=np.random.default_rng(20260926)
    for _ in range(3):
        v=rng.normal(size=132); v/=np.linalg.norm(v); h=2e-6
        fd=(model.combined(p+h*v,source_gate,False)[0]-model.combined(p-h*v,source_gate,False)[0])/(2*h)
        fderr.append(float(np.max(np.abs(fd-j@v))))
    preflight={'source_residual':float(np.max(np.abs(r))),'residual_diff_full8':residual_diff,'jacobian_diff_full8':jac_diff,'propagator_diff_full8':float(np.linalg.norm(u6-u6_ind)),'directional_fd_max_errors':fderr,'CNOT_det_error':float(abs(np.linalg.det(model.GTARGET)-1))}
    preflight['passed']=residual_diff<1e-10 and jac_diff<1e-10 and np.linalg.norm(u6-u6_ind)<1e-10 and max(fderr)<1e-7
    write(OUT/'preflight.json',preflight)
    print('PREFLIGHT',json.dumps(preflight),flush=True)
    if not preflight['passed']: raise RuntimeError('preflight failed')
    # The eigenphase branch is chosen solely to minimize the traceless generator norm.
    t,z=schur(source_gate.conj().T@model.GTARGET,output='complex')
    angles=np.angle(np.diag(t)); winding=int(np.rint(angles.sum()/(2*np.pi)))
    for _ in range(abs(winding)):
        idx=np.argmax(angles) if winding>0 else np.argmin(angles)
        angles[idx]-=np.sign(winding)*2*np.pi
    angles-=angles.sum()/4
    def target(s): return source_gate@(z*np.exp(1j*s*angles))@z.conj().T
    assert np.linalg.norm(target(1)-model.GTARGET)<1e-10
    np.savez(OUT/'path.npz',eigenvectors=z,eigenphases=angles)
    theta_chi=np.r_[np.arange(15),np.arange(29,44)]
    def domain(x):
        q=x.reshape(3,44)
        return np.max(np.abs(q[:,theta_chi]))<=4*np.pi and np.max(np.abs(q[:,15:29]))<=8*np.pi and model.resources(x)<=150
    s=0.; step=.02; accepted=0; calls=0; trials=0; history=[]
    with (OUT/'trace.jsonl').open('x',encoding='utf-8') as trace:
        while s<1-1e-14 and accepted<800 and calls<4000 and time.perf_counter()-start<1800:
            trials+=1; sn=min(1.,s+step); goal=target(sn); x=p.copy(); ok=False; error=''
            np.savez(OUT/f'trial_{trials:04d}_input.npz',params=p,s=s,next_s=sn)
            it=0
            try:
                for it in range(15):
                    r,j,actual,_=model.combined(x,goal); calls+=1
                    if np.max(np.abs(r))<2e-11 and np.linalg.norm(actual-goal)<1e-9: ok=True; break
                    delta=lstsq(j,-r,cond=1e-12,lapack_driver='gelsy')[0]
                    norm=np.linalg.norm(delta)
                    if norm>1: delta/=norm
                    found=False
                    for back in range(12):
                        xx=x+delta*2**(-back)
                        if not domain(xx): continue
                        try: rn,_,_,_=model.combined(xx,goal,False)
                        except ValueError: continue
                        if np.linalg.norm(rn)<(1-1e-4*2**(-back))*np.linalg.norm(r):
                            x=xx; found=True; break
                    if not found: error='line_search_failed'; break
            except ValueError as exc: error=str(exc)
            record={'trial':trials,'s_from':s,'s_to':sn,'step':step,'accepted':ok,'iterations':it+1,'max_residual':float(np.max(np.abs(r))),'target_fro':float(np.linalg.norm(actual-goal)),'time_t0':model.resources(x),'elapsed_s':time.perf_counter()-start,'error':error}
            if ok:
                p=x; s=sn; accepted+=1
                sv=np.linalg.svd(j,compute_uv=False); record['sigma_min']=float(sv[-1]); record['condition']=float(sv[0]/sv[-1])
                np.savez(OUT/f'step_{accepted:04d}.npz',params=p,s=s)
                if it<=5: step=min(.05,step*1.3)
            else:
                np.savez(OUT/f'trial_{trials:04d}_failure.npz',params=x,s=sn,residual=r)
                step/=2
            trace.write(json.dumps(record)+'\n'); trace.flush(); history.append(record)
            print('STEP',json.dumps(record),flush=True)
            if step<1e-5: break
    r,j,g,ds=model.combined(p)
    success=s>=1-1e-14 and np.max(np.abs(r))<1e-9 and np.linalg.norm(g-model.GTARGET)<1e-9
    np.savez(OUT/'final.npz',params=p.reshape(3,44),target_gate=model.GTARGET,target_block=model.UTARGET,jacobian=j)
    result={'status':'NUMERICALLY_SUPPORTED' if success else 'CONTINUATION_INCOMPLETE','success':success,'path_position':s,'accepted_steps':accepted,'trial_count':trials,'jacobian_calls':calls,'elapsed_s':time.perf_counter()-start,'target_fro':float(np.linalg.norm(g-model.GTARGET)),'max_residual':float(np.max(np.abs(r))),'time_t0_eta10':model.resources(p),'eta_diagnostics':{str(e):model.resources(p,e) for e in [1,10,100]},'source_target_distance_fro':float(np.linalg.norm(source_gate-model.GTARGET)),'path_generator_fro':float(np.linalg.norm(angles)),'rank_126':int(np.linalg.matrix_rank(j,tol=1e-8)),'interval_certified':False}
    write(OUT/'result.json',result); print('RESULT',json.dumps(result),flush=True)

if __name__=='__main__':
    try: main()
    except Exception:
        if OUT.exists():
            with (OUT/'failure.txt').open('x',encoding='utf-8') as f: f.write(traceback.format_exc())
        raise
