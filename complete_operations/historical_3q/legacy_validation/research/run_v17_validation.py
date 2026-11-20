"""Fresh v17 validation after incumbent-preserving selection; no validation tuning."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
from pathlib import Path
import json,time,traceback
import numpy as np
from scipy.stats import t as student
import run_v12_validation as b
import v12_model as v
import v12_channel as c
HERE=Path(__file__).resolve().parent

def rank(data,names,metric,eligible):
    ids=[i for i,k in enumerate(names) if k in eligible]; n=data.shape[3]//2; pairs=len(ids)*(len(ids)-1)//2; critical=float(student.ppf(1-.05/(2*pairs),n-1))
    winner=np.zeros((7,6),int); resolved=np.zeros((7,6),bool); gap=np.zeros((7,6)); intervals=np.zeros((7,6,2)); means=data[:,:,:,:,metric].mean(axis=3)
    minimum_lower=np.zeros((7,6)); limiting=np.zeros((7,6),int)
    for a in range(7):
        for z in range(6):
            order=sorted(ids,key=lambda i:means[i,a,z]); first,second=order[:2]; winner[a,z]=first
            delta=data[second,a,z,:,metric]-data[first,a,z,:,metric]; independent=(delta[:n]+delta[n:])/2
            margin=float(independent.mean()); half=critical*float(independent.std(ddof=1)/np.sqrt(n)); gap[a,z]=margin; intervals[a,z]=[margin-half,margin+half]
            checks=[]
            for challenger in order[1:]:
                d=data[challenger,a,z,:,metric]-data[first,a,z,:,metric]
                units=(d[:n]+d[n:])/2
                lower=float(units.mean()-critical*units.std(ddof=1)/np.sqrt(n))
                checks.append((lower,challenger))
            minimum_lower[a,z],limiting[a,z]=min(checks)
            resolved[a,z]=margin>4e-7 and minimum_lower[a,z]>0
    return {'winner':winner.tolist(),'resolved':resolved.tolist(),'margin':gap.tolist(),'paired_95_interval':intervals.tolist(),'minimum_adjusted_lower':minimum_lower.tolist(),'limiting_challenger':limiting.tolist(),'eligible':eligible,'independent_pairs':n,'within_cell_pair_comparisons':pairs,'critical_value':critical,'interval_scope':'Approximate Bonferroni within cell; every challenger checked; not simultaneous across parameter cells.'}

def main():
    library_dir=HERE/'runs/v17_library_v1'
    library=json.loads((library_dir/'result.json').read_text())['selected']
    controls={kind:c.compile_controls(np.load(library_dir/(kind+'.npz'))['params']) for kind in library}
    preout=HERE/'runs/v17_channel_preflight_v1'; preout.mkdir(exist_ok=False)
    paths=[Path(__file__),HERE/'V17_REVISION_PROTOCOL_20260929_v1.md',HERE/'V17_VALIDATION_ADDENDUM_20260929_v2.md',HERE/'run_v12_validation.py',HERE/'run_v12_confirmation.py',HERE/'v12_channel.py',HERE/'v12_model.py',HERE/'v11_full8.py',library_dir/'result.json',*library_dir.glob('*.npz')]
    b.write(preout/'bindings.json',{str(p.relative_to(HERE)):b.sha(p) for p in paths})
    b.preflight(preout,library,controls)
    out=HERE/'runs/v17_confirmation_v1'; out.mkdir(exist_ok=False)
    b.write(out/'bindings.json',{str(p.relative_to(HERE)):b.sha(p) for p in paths})
    import sys, scipy
    b.write(out/'environment.json',{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'precision':'float64/complex128','threads':1})
    start=time.perf_counter()
    b.write(out/'confirmation_bindings.json',{p.name:b.sha(p) for p in [Path(__file__),HERE/'V17_REVISION_PROTOCOL_20260929_v1.md']})
    try:
        samples=v.ensemble(17092822,128); names=list(controls); pre=json.loads((HERE/'runs/v17_channel_preflight_v1/result.json').read_text())
        if not pre['pass']: raise RuntimeError('Missing channel acceptance')
        data=np.empty((len(names),7,6,128,3)); witness={}; steps=pre['steps']
        with (out/'progress.jsonl').open('x',encoding='utf-8') as log:
            for i,kind in enumerate(names):
                ctrl=controls[kind]
                for a,alpha in enumerate(b.ALPHAS):
                    for z,gamma in enumerate(b.GAMMAS):
                        for n,row in enumerate(samples): data[i,a,z,n]=c.metrics(c.propagate(ctrl,v.errors(row,alpha),gamma,steps))
                        if a in (0,4,6) and z in (0,3,5):
                            values=np.array([c.witness(ctrl,v.errors(row,alpha),gamma,steps) for row in samples])
                            witness[kind+'_'+str(a)+'_'+str(z)]={'probabilities':values.tolist(),'mean_probabilities':values.mean(axis=0).tolist(),'max_abs_bias':float(np.max(np.abs(values-7/18))),'mean_abs_bias':float(np.mean(np.abs(values-7/18))),'mean_contrast':float(np.mean(values[:,1]-values[:,0]))}
                    rec={'kind':kind,'alpha':float(alpha),'elapsed_s':time.perf_counter()-start}; log.write(json.dumps(rec)+'\n'); log.flush(); print('CONFIRM',json.dumps(rec),flush=True)
                np.savez(out/(kind+'_samples.npz'),metrics=data[i],alphas=b.ALPHAS,gammas=b.GAMMAS,samples=samples)
        np.savez(out/'arrays.npz',metrics=data,alphas=b.ALPHAS,gammas=b.GAMMAS,samples=samples,names=np.array(names))
        full=[k for k in names if k not in ('projected','noise_projected')]; q1p=[k for k in names if k in v.ROBUST]; q1f=[k for k in names if k in ('whole','module_response','modular')]
        rankings={'projected_finite':rank(data,names,0,names),'complete_finite':rank(data,names,1,full),'projected_q1_shared':rank(data,names,0,q1p),'complete_q1_shared':rank(data,names,1,q1f),'complete_q1_module':rank(data,names,1,[k for k in q1f if k!='whole'])}
        b.write(out/'result.json',{'names':names,'alphas':b.ALPHAS.tolist(),'gammas':b.GAMMAS.tolist(),'means':data.mean(axis=3).tolist(),'rankings':rankings,'witness':witness,'steps':steps,'elapsed_s':time.perf_counter()-start,'cells':336,'channel_evaluations':43008,'sample_count':128,'independent_pairs':64,'seed':17092822,'metric_order':['projected_entanglement_infidelity','complete_entanglement_infidelity','projected_survival']})
        print('CONFIRMATION_COMPLETE',time.perf_counter()-start,flush=True)
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8'); raise
if __name__=='__main__': main()
