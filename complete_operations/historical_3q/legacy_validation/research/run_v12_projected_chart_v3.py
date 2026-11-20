"""Regular projected phase chart, versioned finite-noise baseline successor."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
from pathlib import Path
import numpy as np
import json,argparse,traceback
import run_v12_search as b
import run_v12_restoration_v2 as r2
import v12_model as v
import v11_global_model as g
import cnot_model_v1 as m
HERE=Path(__file__).resolve().parent
original=v.constraints
def constraints(p,kind,jac=True):
    if kind!='noise_projected': return original(p,kind,jac)
    u,du,_,_,_=g.product_responses(p,jac); w=m.UTARGET[:4,:4].conj().T@u[:4,:4]; dw=m.UTARGET[:4,:4].conj().T@du[:,:4,:4]
    z=np.trace(w)/4
    if abs(z)<.25: raise ValueError('projected overlap chart singular')
    dz=np.trace(dw,axis1=1,axis2=2)/4; phase=z/abs(z); w=w/phase
    dw=dw/phase-1j*np.imag(dz/z)[:,None,None]*w
    h=.5j*(w-w.conj().T); dh=.5j*(dw-dw.conj().transpose(0,2,1))
    return g.project(m.B4,h),g.project(m.B4,dh).T
v.constraints=constraints
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['preflight','search','library']); args=parser.parse_args()
    if args.stage=='preflight':
        out=HERE/'runs/v12_phase_chart_preflight_v3'; out.mkdir(exist_ok=False); rows=[]; rng=np.random.default_rng(12092751)
        for label in ('nominal','random12092701','random12092702','random12092703'):
            p=np.load(HERE/('runs/v12_search_noise_projected_'+label+'_v2/final.npz'))['params'].ravel(); rr,j=constraints(p,'noise_projected'); direction=rng.normal(size=132); direction/=np.linalg.norm(direction); h=1e-6
            fd=(constraints(p+h*direction,'noise_projected')[0]-constraints(p-h*direction,'noise_projected')[0])/(2*h)
            rows.append({'start':label,'rank':int(np.linalg.matrix_rank(j,tol=1e-8)),'fd_max':float(np.max(np.abs(fd-j@direction)))})
        b.write(out/'result.json',{'pass':all(x['rank']==15 and x['fd_max']<1e-6 for x in rows),'checks':rows}); print(rows); return
    if args.stage=='library':
        out=HERE/'runs/v12_library_v1'; out.mkdir(exist_ok=False); b.bindings(out); records=[]; selected={}
        for d in sorted((HERE/'runs').glob('v12_search_*')):
            rec=json.loads((d/'result.json').read_text()); rec['directory']=d.name; records.append(rec)
        for kind in (*v.ROBUST,'noise_projected','noise_complete','nominal'):
            good=[r for r in records if r['kind']==kind and r['accepted']]
            if kind=='noise_projected': good=[r for r in good if r['directory'].endswith('_v3')]
            elif kind in ('projected','whole','noise_complete'): good=[r for r in good if r['directory'].endswith('_v2') or '_warm_' in r['directory']]
            if not good: continue
            best=min(good,key=lambda r:r['training_objective'] if kind.startswith('noise_') else r['audit']['gate_time_t0'])
            p=np.load(HERE/'runs'/best['directory']/'final.npz')['params']; np.savez(out/(kind+'.npz'),params=p)
            selected[kind]={'source':best['directory'],'sha256':b.sha(out/(kind+'.npz')),'audit':best['audit'],'training_objective':best['training_objective']}
        b.write(out/'result.json',{'selected':selected,'all_trials':records,'policy':'Registered v2/v3 corrections; no validation selection.'})
        b.write(out/'addendum_bindings.json',{p.name:b.sha(p) for p in [Path(__file__),HERE/'V12_RESTORATION_ADDENDUM_20260927_v2.md',HERE/'V12_PROJECTED_CHART_ADDENDUM_20260927_v3.md']}); print(json.dumps(selected)); return
    if not json.loads((HERE/'runs/v12_phase_chart_preflight_v3/result.json').read_text())['pass']: raise RuntimeError('No phase-chart preflight')
    starts=[('warm',np.load(b.WARM)['params'].ravel()),('nominal',np.load(b.NOM)['params'].ravel())]
    for seed in (12092701,12092702,12092703):
        rng=np.random.default_rng(seed); p=np.zeros((3,44)); p[:,:15]=rng.uniform(-np.pi,np.pi,(3,15)); p[:,15:29]=rng.uniform(-np.pi,np.pi,(3,14)); p[:,29:]=rng.uniform(-np.pi/4,np.pi/4,(3,15)); starts.append(('random'+str(seed),p.ravel()))
    for label,p in starts:
        out=HERE/('runs/v12_search_noise_projected_'+label+'_v3'); out.mkdir(exist_ok=False); b.bindings(out)
        b.write(out/'successor_bindings.json',{q.name:b.sha(q) for q in [Path(__file__),HERE/'run_v12_restoration_v2.py',HERE/'V12_PROJECTED_CHART_ADDENDUM_20260927_v3.md']}); np.savez(out/'initial.npz',params=p.reshape(3,44))
        try:
            q=r2.restore(p,'noise_projected',out)
            if q is None:
                rec={'kind':'noise_projected','initialization':label,'accepted':False,'stopping_reason':'restoration_failed'}; b.write(out/'result.json',rec)
            else: rec=b.optimize(q,'noise_projected',out)
            print('PHASE_SUCCESSOR',out.name,json.dumps(rec),flush=True)
        except Exception:
            (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8'); raise
if __name__=='__main__': main()
