"""Fixed-slot compact smoothing and correction of the prescribed encoded CNOT."""
from __future__ import annotations
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import sys,time,json
from pathlib import Path
import numpy as np
from scipy.linalg import lstsq
import cnot_model_v1 as m
import run_cnot_optimization_v1 as io
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(io.ROOT/'src'))
import waveform_framework_v1 as wf
import manifold_full8_v1 as f
OUT=HERE/'runs'/'cnot_smoothing_20260926_v1'

def schedules(p,width):
    out=[]
    for k,q in enumerate(p.reshape(3,44)):
        dur=np.empty(30); dur[::2]=np.maximum(1.1*np.abs(q[29:]),.01); dur[1::2]=np.maximum(1.1*np.abs(q[:15])/10,.01)
        out.append(wf.ModuleSchedule(k,width,dur,width+np.r_[0,np.cumsum(dur)],tuple(['twist','rotation']*15),np.repeat(np.arange(15),2)))
    return out

def evaluate(p,ss,tight=False,full=False):
    settings=wf.TIGHT_INTEGRATION if tight else wf.IntegrationSettings(2e-11,2e-13,4.)
    analyses=[wf.analyze_module(wf.evaluate_module(s,q,settings,'full8' if full else 'schur')) for s,q in zip(ss,p.reshape(3,44))]
    rmod=np.concatenate([a.chart.independent37 for a in analyses]); rt,_,gate,_=wf.source_target_chart(analyses,m.GTARGET)
    return np.r_[rmod,rt],gate,analyses

def full_metrics(analyses):
    ev=[]
    for a in analyses:
        b=a.propagation.full8
        ev.append(f.ModuleEvaluation(b.unitary,b.d_error_rotation,b.d_error_twist,b.k_error_rotation,b.k_error_twist))
    u,kr,kt=f.compose_full8(ev)
    target=-(f.VISIBLE_ISOMETRY@m.GTARGET@f.VISIBLE_ISOMETRY.conj().T+f.COMPLEMENT_PROJECTOR)
    return {'full8_target_fro':float(np.linalg.norm(u-target)),'full8_KR_fro':float(np.linalg.norm(kr)),'full8_KT_fro':float(np.linalg.norm(kt)),'module_max_K_fro':float(max(max(np.linalg.norm(e.k_error_rotation),np.linalg.norm(e.k_error_twist)) for e in ev))}

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    source=HERE/'runs'/'cnot_optimization_20260926_v2'/'robust_time.npz'
    p0=np.load(source)['params'].ravel(); start=time.perf_counter()
    io.write(OUT/'protocol_addendum.json',{'fixed_slot_floor_t0':.01,'reason':'Time descent leaves a small nonzero twisting area. A positive slot floor avoids making the bandwidth scan depend on this accidental small amplitude; it only reduces that slot amplitude.','fixed_slot_formula':'max(1.1*abs(area)/cap,0.01 t0)','width_fractions_of_min_slot':[.001,.005,.01,.025,.05,.1,.25,.45],'max_correction_iterations':8,'search_stop_max_residual':2e-11,'full8_acceptance_max':1e-8,'budget_s':1200})
    io.write(OUT/'bindings.json',{str(x.relative_to(io.ROOT)):io.sha(x) for x in [Path(__file__),source,HERE/'cnot_model_v1.py',io.ROOT/'src'/'waveform_framework_v1.py',io.ROOT/'src'/'manifold_full8_v1.py']})
    rows=[]
    for fraction in [.001,.005,.01,.025,.05,.1,.25,.45]:
        if time.perf_counter()-start>1200: break
        width=fraction*.01; ss=schedules(p0,width)
        r0,g0,_=evaluate(p0,ss); p=p0.copy(); r=r0; g=g0; iterations=0
        for iterations in range(8):
            if np.max(np.abs(r))<2e-11: break
            _,j,_,_=m.combined(p)
            p+=lstsq(j,-r,cond=1e-12,lapack_driver='gelsy')[0]
            r,g,_=evaluate(p,ss)
        # Independent full-space evaluation with tighter ODE tolerances.
        rf,gf,af=evaluate(p,ss,tight=True,full=True)
        metrics=full_metrics(af)
        raw=[s.raw_fields(q) for s,q in zip(ss,p.reshape(3,44))]
        rotation_peak=max(np.max(np.linalg.norm(v[:,:2],axis=1)) for v in raw)
        twist_peak=max(np.max(np.abs(v[:,2])) for v in raw)
        row={'fraction_min_slot':fraction,'width_t0':width,'uncorrected_residual_max':float(np.max(np.abs(r0))),'uncorrected_target_fro':float(np.linalg.norm(g0-m.GTARGET)),'correction_norm':float(np.linalg.norm(p-p0)),'correction_iterations':iterations,'corrected_residual_max':float(np.max(np.abs(rf))),'total_duration_t0':float(sum(s.total_duration for s in ss)),'rotation_peak':float(rotation_peak),'twist_peak':float(twist_peak),'elapsed_s':time.perf_counter()-start,**metrics}
        row['accepted']=bool(max(metrics.values())<1e-8 and rotation_peak<=10 and twist_peak<=1)
        rows.append(row)
        np.savez(OUT/f'width_{fraction:g}.npz',params=p.reshape(3,44),source_params=p0.reshape(3,44),durations=np.array([s.durations for s in ss]),width=width)
        io.write(OUT/f'width_{fraction:g}.json',row)
        print('SMOOTH',json.dumps(row),flush=True)
    io.write(OUT/'result.json',{'rows':rows,'all_accepted':all(x['accepted'] for x in rows),'elapsed_s':time.perf_counter()-start,'interval_certified':False,'rank_status':'Not inherited from the sharp source; checked separately if used as a smooth-manifold claim.'})

if __name__=='__main__': main()
