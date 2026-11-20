from __future__ import annotations
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import sys,time,json
from pathlib import Path
import numpy as np
from scipy.linalg import block_diag
import cnot_model_v1 as m
import run_cnot_optimization_v1 as io
HERE=Path(__file__).resolve().parent
OUT=HERE/'runs'/'cnot_benchmark_20260926_v1'

def main():
    OUT.mkdir(parents=True,exist_ok=False); start=time.perf_counter()
    files={'first_cnot':HERE/'runs/cnot_continuation_20260926_v1/final.npz','robust_time':HERE/'runs/cnot_optimization_20260926_v2/robust_time.npz','nominal_time':HERE/'runs/cnot_baselines_20260926_v1/nominal_time.npz','noise_aware':HERE/'runs/cnot_baselines_20260926_v1/noise_aware.npz'}
    ps={k:np.load(v)['params'].ravel() for k,v in files.items()}
    io.write(OUT/'bindings.json',{str(f.relative_to(io.ROOT)):io.sha(f) for f in [Path(__file__),HERE/'cnot_model_v1.py',*files.values()]})
    grid=np.array([(x,y) for x in np.linspace(-1,1,21) for y in np.linspace(-1,1,21)])
    rng=np.random.default_rng(20260926); random=rng.uniform(-1,1,(128,2)); points=np.r_[grid,random]
    radii=np.array([1e-4,1e-3,1e-2]); depths=np.array([1,2,4,8,16]); arrays={'normalized_points':points,'radii':radii,'depths':depths}; results={}
    sys.path.insert(0,str(io.ROOT/'src'))
    import manifold_full8_v1 as full
    import manifold_spectral_v1 as spectral
    full_target=-(full.VISIBLE_ISOMETRY@m.GTARGET@full.VISIBLE_ISOMETRY.conj().T+full.COMPLEMENT_PROJECTOR)
    for label,p in ps.items():
        curve=np.empty((3,len(points),len(depths)))
        for ri,r in enumerate(radii):
            for i,v in enumerate(points):
                u,_=m.total(p,False,errors=r*v)
                for k,l in enumerate(depths): curve[ri,i,k]=m.half_diamond(np.linalg.matrix_power(u,int(l)),np.linalg.matrix_power(m.UTARGET,int(l)))
            print('GRID',label,float(r),'single max',float(curve[ri,:441,0].max()),flush=True)
        arrays[label+'_half_diamond']=curve
        cross=[]
        for err in np.r_[np.zeros((1,2)),.01*random[:8]]:
            u6,_=m.total(p,False,errors=err)
            es=[full.module_propagator(q,*err,with_error_derivatives=False) for q in p.reshape(3,44)]
            u8=es[0].unitary@es[1].unitary@es[2].unitary
            small=block_diag(full.block(u8,full.VISIBLE_ISOMETRY),full.block(u8,full.HIDDEN_ISOMETRY))
            # Independently compute the spectral arc on the physical register.
            a=np.sort(np.angle(np.linalg.eigvals(full_target.conj().T@u8))); arc=2*np.pi-np.max(np.diff(np.r_[a,a[0]+2*np.pi])); d=1. if arc>=np.pi else np.sin(max(0.,arc)/2)
            cross.append({'propagator_fro_difference':float(np.linalg.norm(u6-small)),'distance_difference':float(abs(m.half_diamond(u6)-d))})
        module_budgets=[]
        for q in p.reshape(3,44):
            jet=spectral.module_jet(q,(1.,1.)); a=1.5*np.abs(q[:15]).sum()+3*np.abs(q[29:]).sum()
            module_budgets.append({'A':float(a),'I1_op':float(np.linalg.norm(jet.u0.conj().T@jet.u1,2)),'I2_op':float(np.linalg.norm(jet.u0.conj().T@jet.u2,2))})
        q=p.reshape(3,44); limits={'theta_chi_abs_max':float(max(np.max(np.abs(q[:,:15])),np.max(np.abs(q[:,29:])))),'phi_abs_max':float(np.max(np.abs(q[:,15:29])))}
        results[label]={'time_t0':m.resources(p),'grid_max_by_radius_depth':curve[:,:441,:].max(axis=1).tolist(),'random_max_by_radius_depth':curve[:,441:,:].max(axis=1).tolist(),'independent_full8_checks':cross,'module_direction_11_budgets':module_budgets,'parameter_limits':limits,'limits_pass':bool(limits['theta_chi_abs_max']<=4*np.pi and limits['phi_abs_max']<=8*np.pi),'grid_is_continuous_certificate':False}
        io.write(OUT/(label+'_result.json'),results[label])
    np.savez(OUT/'arrays.npz',**arrays)
    io.write(OUT/'result.json',{'candidates':results,'radii':radii.tolist(),'depths':depths.tolist(),'grid_points':441,'random_points':128,'random_seed':20260926,'elapsed_s':time.perf_counter()-start,'all_cross_checks_pass':bool(max(max(c['propagator_fro_difference'],c['distance_difference']) for r in results.values() for c in r['independent_full8_checks'])<1e-10)})
    print('BENCHMARK_COMPLETE',time.perf_counter()-start,flush=True)

if __name__=='__main__': main()
