"""Independent acceptance, finite-ensemble selection maps, and phase readout."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
from pathlib import Path
import time,json,hashlib,sys,traceback,argparse
import numpy as np
import scipy
from scipy.stats import t as student
import v12_channel as c
import v12_model as v
import v11_full8 as f
import v11_global_model as g
HERE=Path(__file__).resolve().parent
LIB=HERE/'runs/v12_library_v1'
ALPHAS=np.array([0,.01,.03,.1,.3,.6,1.])
GAMMAS=np.array([0,1e-7,1e-6,1e-5,1e-4,1e-3])
SAMPLES=v.ensemble(12092721)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):
    with p.open('x',encoding='utf-8') as h: json.dump(obj,h,indent=2,allow_nan=False)
def setup(stage):
    out=HERE/'runs'/('v12_'+stage+'_v1'); out.mkdir(exist_ok=False)
    paths=[Path(__file__),HERE/'v12_model.py',HERE/'v12_channel.py',HERE/'v11_full8.py',HERE/'V12_PROTOCOL_20260927_v1.md',LIB/'result.json',*LIB.glob('*.npz')]
    write(out/'bindings.json',{str(p.relative_to(HERE)):sha(p) for p in paths})
    write(out/'environment.json',{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'precision':'float64/complex128'})
    library=json.loads((LIB/'result.json').read_text())['selected']
    controls={kind:c.compile_controls(np.load(LIB/(kind+'.npz'))['params']) for kind in library}
    return out,library,controls

def physical_audit(ctrl,kind):
    p=ctrl['params']; u,k=f.propagate(f.schedule(p),responses=True); shared=k.sum(axis=0)
    visible=f.VD.conj().T@u@f.VD; target=f.VD.conj().T@f.TARGET@f.VD
    phase=np.trace(target.conj().T@visible); phase/=abs(phase)
    pv=[np.linalg.norm(f.VD.conj().T@kk@f.VD-np.trace(f.VD.conj().T@kk@f.VD)*np.eye(4)/4) for kk in shared]
    full=[np.linalg.norm(kk) for kk in shared]
    # Transport is unitary, hence the norm of each independent-module response is invariant.
    mods=np.linalg.norm(k,axis=(2,3)); mismatch=np.linalg.norm(u-f.TARGET); projected=np.linalg.norm(visible-phase*target)
    required=[projected] if kind in ('projected','noise_projected') else [mismatch]
    if kind in ('projected','nominal_projected'): required+=pv
    if kind=='whole': required+=full
    if kind in ('module_response','modular'): required+=mods.ravel().tolist()
    hidden=[]
    for j in range(3):
        uj,_=f.propagate([entry for entry in f.schedule(p) if entry[0]==j])
        block=f.Q@uj@f.Q; hidden.append(float(np.linalg.norm(block-np.trace(block)*f.Q/4)))
    if kind=='modular': required+=hidden
    out={'target_full_fro':float(mismatch),'target_projected_fro':float(projected),'responses_full_fro':np.array(full).tolist(),'responses_projected_fro':np.array(pv).tolist(),'module_responses_fro':mods.tolist(),'required_max':float(max(required)),'leakage_unitary_fro':float(np.linalg.norm(f.Q@u@f.P)),'passed':bool(max(required)<1e-8)}
    out['module_hidden_nonscalar_fro']=hidden
    return out

def preflight(out,library,controls):
    start=time.perf_counter(); audits={}; selected_steps=32; checks=[]; localchecks=[]; passed=True
    for kind,ctrl in controls.items():
        audit=physical_audit(ctrl,kind); audits[kind]=audit; passed&=audit['passed']
        e=v.errors(SAMPLES[0],.3); s=c.propagate(ctrl,e,0); u,_=f.propagate(f.schedule(ctrl['params']),errors=e)
        audit['unitary_super_fro']=float(np.linalg.norm(s-c.superop(u)))
        audit['unitary_witness_abs']=float(np.max(np.abs(c.witness(ctrl,e,0)-np.array([f.probability(u,phi) for phi in (-np.pi/4,3*np.pi/4)]))))
        passed&=audit['unitary_super_fro']<1e-9 and audit['unitary_witness_abs']<1e-9
        for alpha,row in ((0,np.ones(8)),(1,np.ones(8)),(1,np.array([1,-1,1,-1,-1,1,1,-1]))):
            e=v.errors(row,alpha); prev_s=c.propagate(ctrl,e,.001,32); prev_w=c.witness(ctrl,e,.001,32)
            for steps in (64,128):
                s=c.propagate(ctrl,e,.001,steps); w=c.witness(ctrl,e,.001,steps)
                diff=max(float(np.max(np.abs(c.metrics(s)-c.metrics(prev_s)))),float(np.max(np.abs(w-prev_w))))
                checks.append({'kind':kind,'alpha':alpha,'row':row.tolist(),'steps':steps,'metric_witness_change':diff,'cptp':c.channel_checks(s)})
                selected_steps=max(selected_steps,steps)
                if diff<=2e-7: break
                prev_s=s; prev_w=w
            else:
                if diff>2e-7: passed=False
        index=int(np.argmax(np.abs(ctrl['angles'])))
        angle=ctrl['angles'][index]*1.02; phi=ctrl['phis'][index]; tau=ctrl['durations'][index]
        exact=c.adaptive_local_channel(angle,phi,tau,.001)
        approx=c.local_channels(np.array([angle]),phi,np.array([tau]),.001,128)[0]
        localchecks.append({'kind':kind,'angle':float(angle),'tau':float(tau),'adaptive_local_fro':float(np.linalg.norm(exact-approx))})
        print('CHANNEL_PREFLIGHT',kind,'steps',selected_steps,flush=True)
    passed&=all(x['cptp']['trace_preservation_fro']<1e-9 and x['cptp']['choi_min_eigenvalue']>-1e-9 for x in checks)
    passed&=all(x['adaptive_local_fro']<2e-7 for x in localchecks)
    record={'pass':bool(passed),'steps':selected_steps,'physical':audits,'convergence':checks,'local_adaptive':localchecks,'elapsed_s':time.perf_counter()-start,'tolerance':2e-7}
    write(out/'result.json',record)
    if not passed: raise RuntimeError('Channel acceptance failed; preserve output and report unresolved')

def rank(data,names,metric,eligible):
    indices=[i for i,k in enumerate(names) if k in eligible]
    winner=np.zeros((len(ALPHAS),len(GAMMAS)),int); resolved=np.zeros_like(winner,bool); gap=np.zeros_like(winner,float); interval=np.zeros((*winner.shape,2))
    means=data[:,:,:,:,metric].mean(axis=3)
    for a in range(len(ALPHAS)):
        for b in range(len(GAMMAS)):
            order=sorted(indices,key=lambda i:means[i,a,b]); first,second=order[:2]; winner[a,b]=first
            differences=data[second,a,b,:,metric]-data[first,a,b,:,metric]
            pairs=(differences[:8]+differences[8:])/2
            margin=float(pairs.mean()); se=float(pairs.std(ddof=1)/np.sqrt(8)); half=student.ppf(.975,7)*se
            gap[a,b]=margin; interval[a,b]=[margin-half,margin+half]; resolved[a,b]=margin>4e-7 and margin-half>0
    return {'winner':winner.tolist(),'resolved':resolved.tolist(),'margin':gap.tolist(),'paired_95_interval':interval.tolist(),'eligible':eligible}

def mapping(out,library,controls):
    start=time.perf_counter(); pre=json.loads((HERE/'runs/v12_channel_preflight_v1/result.json').read_text())
    if not pre['pass']: raise RuntimeError('Missing passing channel preflight')
    steps=pre['steps']; names=list(controls); data=np.empty((len(names),len(ALPHAS),len(GAMMAS),16,3)); witness={}
    with (out/'progress.jsonl').open('x',encoding='utf-8') as log:
        for i,kind in enumerate(names):
            ctrl=controls[kind]
            for a,alpha in enumerate(ALPHAS):
                for b,gamma in enumerate(GAMMAS):
                    for n,row in enumerate(SAMPLES):
                        e=v.errors(row,alpha); channel=c.propagate(ctrl,e,gamma,steps); data[i,a,b,n]=c.metrics(channel)
                    if a in (0,4,6) and b in (0,3,5):
                        values=np.array([c.witness(ctrl,v.errors(row,alpha),gamma,steps) for row in SAMPLES])
                        witness[kind+'_'+str(a)+'_'+str(b)]={'probabilities':values.tolist(),'mean_probabilities':values.mean(axis=0).tolist(),'max_abs_bias':float(np.max(np.abs(values-7/18))),'mean_abs_bias':float(np.mean(np.abs(values-7/18))),'mean_contrast':float(np.mean(values[:,1]-values[:,0]))}
                row={'kind':kind,'alpha':float(alpha),'elapsed_s':time.perf_counter()-start}; log.write(json.dumps(row)+'\n'); log.flush(); print('MAP',json.dumps(row),flush=True)
            np.savez(out/(kind+'_samples.npz'),metrics=data[i],alphas=ALPHAS,gammas=GAMMAS,samples=SAMPLES)
    np.savez(out/'arrays.npz',metrics=data,alphas=ALPHAS,gammas=GAMMAS,samples=SAMPLES,names=np.array(names))
    complete=[k for k in names if k not in ('projected','noise_projected')]
    q1p=[k for k in names if k in v.ROBUST]; q1f=[k for k in names if k in ('whole','module_response','modular')]
    rankings={'projected_finite':rank(data,names,0,names),'complete_finite':rank(data,names,1,complete),'projected_q1_shared':rank(data,names,0,q1p),'complete_q1_shared':rank(data,names,1,q1f),'complete_q1_module':rank(data,names,1,[k for k in q1f if k!='whole'])}
    write(out/'result.json',{'names':names,'alphas':ALPHAS.tolist(),'gammas':GAMMAS.tolist(),'means':data.mean(axis=3).tolist(),'rankings':rankings,'witness':witness,'steps':steps,'elapsed_s':time.perf_counter()-start,'cells':int(len(names)*len(ALPHAS)*len(GAMMAS)),'channel_evaluations':int(np.prod(data.shape[:4])),'metric_order':['projected_entanglement_infidelity','complete_entanglement_infidelity','projected_survival'],'interpretation':'Finite ensemble, eight independent antithetic pairs; intervals are exploratory, not simultaneous coverage.'})
    print('MAPPING_COMPLETE',time.perf_counter()-start,flush=True)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['channel_preflight','map']); args=parser.parse_args()
    out,library,controls=setup(args.stage)
    try:
        if args.stage=='channel_preflight': preflight(out,library,controls)
        else: mapping(out,library,controls)
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8'); raise
if __name__=='__main__': main()
