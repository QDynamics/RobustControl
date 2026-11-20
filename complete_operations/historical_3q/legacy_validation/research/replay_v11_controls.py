"""Replay archived v11 controls without an optimization or archive mutation.

Usage: python replay_v11_controls.py --out PATH_TO_A_NEW_DIRECTORY
Requires NumPy; all propagation is independent physical-basis 8x8 linear algebra.
"""
from pathlib import Path
import argparse, hashlib, json, sys, time
import numpy as np
import v11_full8 as f

HERE=Path(__file__).resolve().parent
FILES={
 'v10_modular':'runs/cnot_optimization_20260926_v2/robust_time.npz',
 'matched_modular':'runs/v11_modular_20260927_v1/final.npz',
 'whole_gate':'runs/v11_whole_gate_20260927_v1/final.npz',
 'nominal_time':'runs/cnot_baselines_20260926_v1/nominal_time.npz',
 'noise_aware':'runs/cnot_baselines_20260926_v1/noise_aware.npz',
}

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--out',required=True)
    args=parser.parse_args(); out=Path(args.out).resolve()
    out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    validation=HERE/'runs/v11_validation_20260927_v1'
    bindings=json.loads((validation/'bindings.json').read_text())
    sources_ok={key:sha(HERE/key)==value for key,value in bindings.items()}
    if not all(sources_ok.values()): raise RuntimeError('Archived input hash mismatch')
    data=np.load(validation/'arrays.npz',allow_pickle=False)
    results={}; max_replay_error=0.
    for label,rel in FILES.items():
        params=np.load(HERE/rel,allow_pickle=False)['params']
        seq=f.schedule(params); audit=f.audit(params)
        if audit['target_fro']>1e-8: raise RuntimeError('Target mismatch: '+label)
        if label in ('v10_modular','matched_modular','whole_gate') and max(audit['shared_response_fro'])>1e-8:
            raise RuntimeError('Shared-response mismatch: '+label)
        if label in ('v10_modular','matched_modular') and np.max(audit['module_response_fro'])>1e-8:
            raise RuntimeError('Module-response mismatch: '+label)
        error=0.; samples=0
        for mode in ('common','module'):
            points=data[mode+'_normalized_points']
            indexes=np.unique(np.linspace(0,len(points)-1,7,dtype=int))
            for ir,radius in enumerate(data['radii']):
                for index in indexes:
                    u,_=f.propagate(seq,radius*points[index])
                    error=max(error,abs(f.half_diamond(u)-data[label+'_'+mode+'_D'][ir,index]))
                    for setting,phase in enumerate(data['analysis_phases']):
                        error=max(error,abs(f.probability(u,phase)-data[label+'_'+mode+'_prob'][ir,index,setting]))
                    samples+=1
        if error>1e-10: raise RuntimeError('Finite-error replay mismatch: '+label)
        max_replay_error=max(max_replay_error,error)
        results[label]={'control_sha256':sha(HERE/rel),'full8_audit':audit,
                        'finite_error_points_replayed':samples,'max_array_difference':error}
    fringe_error=0.
    for delta in data['fringe_deltas']:
        u=(f.P+np.exp(-1j*delta)*f.Q)@f.TARGET
        for phase in data['fringe_phases']:
            exact=7/18+2/(3*np.sqrt(3))*np.cos(phase+delta+3*np.pi/4)
            fringe_error=max(fringe_error,abs(f.probability(u,phase)-exact))
    if fringe_error>1e-12: raise RuntimeError('Closed-form readout mismatch')
    record={'status':'V11_CONTROL_REPLAY_PASS','source_hashes_match':sources_ok,
            'candidates':results,'max_array_difference':max_replay_error,
            'witness_closed_form_max_error':fringe_error,
            'interpretation':{'per_shot_addressed_time_t0_at_eta_addr10':3*np.pi/20,
             'legacy_field_name':'two_settings_overhead_active_t0_at_eta_addr10',
             'legacy_field_scope_correction':'Stored value is per shot; two settings with n shots require 2n cycles.'},
            'python':sys.version,'numpy':np.__version__,'elapsed_s':time.perf_counter()-start,
            'replay_script_sha256':sha(Path(__file__)), 'interval_certified':False}
    with (out/'result.json').open('x',encoding='utf-8') as stream:
        json.dump(record,stream,indent=2,allow_nan=False)
    print(record['status'], 'max_array_difference=',max_replay_error,
          'witness_error=',fringe_error,'elapsed_s=',record['elapsed_s'])

if __name__=='__main__': main()
