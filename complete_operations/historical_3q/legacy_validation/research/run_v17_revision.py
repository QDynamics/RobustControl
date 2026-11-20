"""Bounded, incumbent-preserving v17 searches; see the frozen protocol."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import sys,time,json,traceback
import numpy as np
import scipy
import run_v12_search as b
import run_v12_projected_chart_v3  # installs the already verified regular phase chart
import run_v12_validation as val
import v12_model as v
import v12_channel as channel
HERE=Path(__file__).resolve().parent
OLD=HERE/'runs/v12_library_v1'
ROOT=HERE/'runs/v17_revision_v1'
PROTOCOL=HERE/'V17_REVISION_PROTOCOL_20260929_v1.md'
KINDS=(*v.ROBUST,'noise_projected','noise_complete','nominal')

def bindings(out):
    paths=[PROTOCOL,Path(__file__),HERE/'run_v12_search.py',HERE/'run_v12_projected_chart_v3.py',HERE/'v12_model.py',HERE/'v11_global_model.py',HERE/'cnot_model_v1.py',HERE/'v11_full8.py',HERE/'v12_channel.py',HERE/'run_v12_validation.py',OLD/'result.json',b.WARM,*OLD.glob('*.npz')]
    b.write(out/'bindings.json',{str(p.relative_to(HERE)):b.sha(p) for p in paths})
    b.write(out/'environment.json',{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'precision':'float64/complex128','blas_threads':1})

def perturb(p,kind,seed,out):
    start=time.perf_counter(); _,j=v.constraints(p,kind)
    _,s,vh=np.linalg.svd(j,full_matrices=True); rank=int(np.sum(s>1e-8)); null=vh[rank:].T
    z=np.random.default_rng(seed).normal(size=132); direction=null@(null.T@z); direction/=np.max(np.abs(direction))
    attempts=[]; q=None
    for h in range(7):
        if time.perf_counter()-start>45: break
        radius=.20*2**(-h); trial=p+radius*direction
        q=b.retract(trial,kind)
        attempts.append({'radius':radius,'accepted':q is not None})
        if q is not None: break
    b.write(out/'initialization.json',{'seed':seed,'nullity':132-rank,'attempts':attempts,'elapsed_s':time.perf_counter()-start,'accepted':q is not None})
    return q

def trial(kind,label,p,seed=None):
    out=ROOT/(kind+'_'+label); out.mkdir(exist_ok=False)
    try:
        q=perturb(p,kind,seed,out) if seed is not None else p.copy()
        if q is None:
            rec={'kind':kind,'accepted':False,'stopping_reason':'initialization_failed'}
            b.write(out/'result.json',rec)
        else:
            np.savez(out/'initial.npz',params=q.reshape(3,44))
            audit=v.matrix_audit(q,kind)
            if not audit['accepted']: raise RuntimeError('Declared feasible start failed matrix audit')
            b.write(out/'initial_audit.json',{'audit':audit,'training_objective':v.loss(q,'noise_projected' if kind in ('projected','noise_projected') else 'noise_complete',False)[0]})
            rec=b.optimize(q,kind,out)
        print('TRIAL_DONE',kind,label,json.dumps(rec),flush=True)
        return {**rec,'source':str((out/'final.npz').relative_to(HERE)),'label':label}
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8'); raise

def load_source(source):
    return np.load(HERE/source)['params'].ravel()

def main():
    ROOT.mkdir(exist_ok=False); bindings(ROOT); start=time.perf_counter()
    pre=ROOT/'preflight'; pre.mkdir(); b.preflight(pre)
    old={k:np.load(OLD/(k+'.npz'))['params'].ravel() for k in KINDS}
    selected={k:{'source':str((OLD/(k+'.npz')).relative_to(HERE))} for k in KINDS}
    records=[]; robust_summary={}
    for kind in ('module_response','modular'):
        trials=[trial(kind,'incumbent',old[kind]),trial(kind,'common_warm',np.load(b.WARM)['params'].ravel())]
        for seed in (17092801,17092802,17092803):
            trials.append(trial(kind,'local'+str(seed),old[kind],seed))
        candidates=[selected[kind]]+[r for r in trials if r['accepted']]
        best=min(candidates,key=lambda r:v.gate_time(load_source(r['source'])))
        selected[kind]={'source':best['source']}; records+=trials
        robust_summary[kind]={'old_time':v.gate_time(old[kind]),'selected_time':v.gate_time(load_source(best['source'])),'accepted':sum(r['accepted'] for r in trials),'trials':trials}
    b.write(ROOT/'robust_summary.json',robust_summary)
    newr=load_source(selected['module_response']['source']); newm=load_source(selected['modular']['source'])
    pool=[{'name':'old_'+k,'source':str((OLD/(k+'.npz')).relative_to(HERE))} for k in KINDS]+[{'name':'new_'+k,**selected[k]} for k in ('module_response','modular')]
    starts={'noise_projected':[('P',old['projected']),('C',old['whole']),('old_FP',old['noise_projected']),('N',old['nominal']),('new_R',newr)],
            'noise_complete':[('C',old['whole']),('new_R',newr),('new_M',newm),('old_FC',old['noise_complete']),('N',old['nominal'])]}
    comparator_summary={}
    for kind in ('noise_projected','noise_complete'):
        feasible=[]
        for candidate in pool:
            p=load_source(candidate['source']); audit=v.matrix_audit(p,kind)
            if audit['accepted'] and v.gate_time(p)<=120:
                feasible.append({**candidate,'training_objective':v.loss(p,kind,False)[0]})
        incumbent=min(feasible,key=lambda r:r['training_objective'])
        trials=[trial(kind,label,p) for label,p in starts[kind]]; records+=trials
        candidates=feasible+[r for r in trials if r['accepted']]
        best=min(candidates,key=lambda r:r['training_objective'])
        if best['training_objective']>incumbent['training_objective']+1e-12: raise RuntimeError('Incumbent regression')
        selected[kind]={'source':best['source']}
        comparator_summary[kind]={'old_objective':v.loss(old[kind],kind,False)[0],'incumbent':incumbent,'selected':best,'feasible_pool':feasible,'trials':trials}
    b.write(ROOT/'comparator_summary.json',comparator_summary)
    library=HERE/'runs/v17_library_v1'; library.mkdir(exist_ok=False)
    for kind in KINDS:
        p=load_source(selected[kind]['source']); audit=v.matrix_audit(p,kind)
        physical=val.physical_audit(channel.compile_controls(p),kind)
        if not audit['accepted'] or not physical['passed']: raise RuntimeError('Selected library failed independent acceptance')
        np.savez(library/(kind+'.npz'),params=p.reshape(3,44))
        selected[kind].update(sha256=b.sha(library/(kind+'.npz')),audit=audit,physical=physical,training_objective=v.loss(p,'noise_projected' if kind in ('projected','noise_projected') else 'noise_complete',False)[0])
    b.write(library/'result.json',{'selected':selected,'all_trials':records,'policy':'Fixed training/duration selection before fresh validation; known feasible incumbents retained.','protocol_sha256':b.sha(PROTOCOL),'elapsed_s':time.perf_counter()-start})
    b.write(ROOT/'result.json',{'status':'SEARCH_AND_LIBRARY_ACCEPTED','library_sha256':b.sha(library/'result.json'),'elapsed_s':time.perf_counter()-start})
    print('LIBRARY_FROZEN',json.dumps(selected),flush=True)

if __name__=='__main__':
    try: main()
    except Exception:
        if ROOT.exists() and not (ROOT/'failure.txt').exists():
            (ROOT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
