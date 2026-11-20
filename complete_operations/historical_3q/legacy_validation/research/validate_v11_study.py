"""Frozen-domain physical-register checks for matched CNOT designs and witness."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
from pathlib import Path
from itertools import product
import time,json,hashlib,sys
import numpy as np
import scipy
from scipy.linalg import block_diag
import v11_full8 as f
import cnot_model_v1 as m

HERE=Path(__file__).resolve().parent
OUT=HERE/'runs/v11_validation_20260927_v1'
FILES={
 'v10_modular':HERE/'runs/cnot_optimization_20260926_v2/robust_time.npz',
 'matched_modular':HERE/'runs/v11_modular_20260927_v1/final.npz',
 'whole_gate':HERE/'runs/v11_whole_gate_20260927_v1/final.npz',
 'nominal_time':HERE/'runs/cnot_baselines_20260926_v1/nominal_time.npz',
 'noise_aware':HERE/'runs/cnot_baselines_20260926_v1/noise_aware.npz',
}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):
    with p.open('x',encoding='utf-8') as stream: json.dump(obj,stream,indent=2,allow_nan=False)

def main():
    OUT.mkdir(parents=True,exist_ok=False); start=time.perf_counter()
    bindings={str(p.relative_to(HERE)):sha(p) for p in [Path(__file__),HERE/'v11_full8.py',HERE/'cnot_model_v1.py',HERE/'V11_PROTOCOL_20260927_v1.md',*FILES.values()]}
    write(OUT/'bindings.json',bindings)
    for kind in ('modular','whole_gate'):
        record=json.loads((HERE/('runs/v11_'+kind+'_20260927_v1/result.json')).read_text())
        if not record['accepted']: raise RuntimeError('Optimization output failed acceptance: '+kind)
    rng=np.random.default_rng(20260927)
    common=np.r_[np.array(list(product(np.linspace(-1,1,21),repeat=2))),rng.uniform(-1,1,(128,2))]
    separate=np.r_[np.array(list(product((-1.,1.),repeat=6))),rng.uniform(-1,1,(128,6))].reshape(-1,3,2)
    radii=np.array([1e-4,1e-3,1e-2]); phases=np.array([-np.pi/4,3*np.pi/4])
    arrays={'common_normalized_points':common,'module_normalized_points':separate,'radii':radii,'analysis_phases':phases}
    results={}
    for label,path in FILES.items():
        p=np.load(path,allow_pickle=False)['params'].ravel(); seq=f.schedule(p)
        actual,k=f.propagate(seq,responses=True)
        state=f.local_rotation(np.pi,0.)[:,0]
        derivatives=np.empty((2,3,2))
        for z,phase in enumerate(phases):
            row=f.local_rotation(np.pi/2,phase)[0]
            amp=row@actual@state
            for j in range(3):
                for a in range(2): derivatives[z,j,a]=2*np.real(np.conj(amp)*(row@(-1j*actual@k[j,a])@state))
        dc=np.empty((3,len(common))); dm=np.empty((3,len(separate)))
        pc=np.empty((3,len(common),2)); pm=np.empty((3,len(separate),2))
        for ri,radius in enumerate(radii):
            for points,dist,prob in ((common,dc,pc),(separate,dm,pm)):
                for ii,err in enumerate(points):
                    u,_=f.propagate(seq,radius*err)
                    dist[ri,ii]=f.half_diamond(u)
                    for z,phase in enumerate(phases): prob[ri,ii,z]=f.probability(u,phase)
            print('VALIDATE',label,float(radius),float(dc[ri].max()),float(dm[ri].max()),flush=True)
        cross=[]
        for err in np.r_[np.zeros((1,6)),(.01*separate[-8:]).reshape(8,6)].reshape(-1,3,2):
            ds=[m.module(q,False,errors=e)[0] for q,e in zip(p.reshape(3,44),err)]
            compact=ds[0]@ds[1]@ds[2]
            physical,_=f.propagate(seq,err)
            projected=block_diag(f.VD.conj().T@physical@f.VD,f.VH.conj().T@physical@f.VH)
            cross.append(float(np.linalg.norm(compact-projected)))
        data={'full8_audit':f.audit(p),'time_t0':m.resources(p),
              'common_grid_max':dc[:,:441].max(axis=1).tolist(),'common_random_max':dc[:,441:].max(axis=1).tolist(),
              'module_vertex_max':dm[:,:64].max(axis=1).tolist(),'module_random_max':dm[:,64:].max(axis=1).tolist(),
              'witness_common_grid_max_abs_deviation':np.abs(pc[:,:441]-7/18).max(axis=(1,2)).tolist(),
              'witness_module_vertex_max_abs_deviation':np.abs(pm[:,:64]-7/18).max(axis=(1,2)).tolist(),
              'witness_module_derivatives':derivatives.tolist(),
              'witness_shared_derivatives':derivatives.sum(axis=1).tolist(),
              'block_cross_checks_fro':cross,'max_parameter_bound_excess':float(max(np.abs(p.reshape(3,44)[:,:15]).max()-4*np.pi,np.abs(p.reshape(3,44)[:,29:]).max()-4*np.pi,np.abs(p.reshape(3,44)[:,15:29]).max()-8*np.pi))}
        arrays[label+'_common_D']=dc; arrays[label+'_module_D']=dm
        arrays[label+'_common_prob']=pc; arrays[label+'_module_prob']=pm
        results[label]=data
        write(OUT/(label+'.json'),data)
    fringe_phases=np.linspace(0,2*np.pi,65); deltas=np.array([0.,.1])
    fringe=np.empty((2,65)); closed=np.empty_like(fringe)
    for k,delta in enumerate(deltas):
        u=(f.P+np.exp(-1j*delta)*f.Q)@f.TARGET
        for i,phase in enumerate(fringe_phases):
            fringe[k,i]=f.probability(u,phase)
            closed[k,i]=7/18+2/(3*np.sqrt(3))*np.cos(phase+delta+3*np.pi/4)
    hs=1e-6
    spam={'prep_probability_derivative':(f.probability(f.TARGET,prep_error=hs)-f.probability(f.TARGET,prep_error=-hs))/(2*hs),
          'analysis_probability_derivative':(f.probability(f.TARGET,analysis_error=hs)-f.probability(f.TARGET,analysis_error=-hs))/(2*hs),
          'two_settings_overhead_active_t0_at_eta_addr10':3*np.pi/(2*10),
          'two_setting_probability_difference_delta_point_one':f.probability((f.P+np.exp(-.1j)*f.Q)@f.TARGET,3*np.pi/4)-f.probability((f.P+np.exp(-.1j)*f.Q)@f.TARGET,-np.pi/4)}
    arrays.update(fringe_phases=fringe_phases,fringe_deltas=deltas,fringe_probabilities=fringe)
    np.savez(OUT/'arrays.npz',**arrays)
    passed=all(v['full8_audit']['target_fro']<1e-8 and max(v['block_cross_checks_fro'])<1e-9 and v['max_parameter_bound_excess']<1e-9 for v in results.values())
    passed=passed and all(max(results[k]['full8_audit']['shared_response_fro'])<1e-8 for k in ('matched_modular','whole_gate'))
    passed=passed and np.max(results['matched_modular']['full8_audit']['module_response_fro'])<1e-8
    passed=passed and np.max(np.abs(fringe-closed))<1e-12
    same={p:sha(HERE/p)==h for p,h in bindings.items()}
    report={'status':'INDEPENDENT_NUMERICAL_VALIDATION_PASS' if passed and all(same.values()) else 'FAIL',
            'candidates':results,'witness_closed_form_max_error':float(np.max(np.abs(fringe-closed))),
            'witness_SPAM':spam,'source_hashes_unchanged':all(same.values()),'elapsed_s':time.perf_counter()-start,
            'seed':20260927,'common_grid_points':441,'common_random_points':128,'module_vertices':64,'module_random_points':128,
            'interval_certified':False,'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__}
    write(OUT/'result.json',report); print('VALIDATION_RESULT',report['status'],report['elapsed_s'],report['witness_closed_form_max_error'],flush=True)

if __name__=='__main__': main()
