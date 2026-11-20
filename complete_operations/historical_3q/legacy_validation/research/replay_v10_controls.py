"""Replay immutable CNOT controls into a fresh audit directory. No optimization."""
from pathlib import Path
import argparse,json,hashlib,sys,time
import numpy as np
from check_v10_science_v1 import propagate,TARGET
HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True,type=Path,help='New output directory; existing paths are refused.')
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    files={'first_cnot':HERE/'runs/cnot_continuation_20260926_v1/final.npz','robust_time':HERE/'runs/cnot_optimization_20260926_v2/robust_time.npz','nominal_time':HERE/'runs/cnot_baselines_20260926_v1/nominal_time.npz','noise_aware':HERE/'runs/cnot_baselines_20260926_v1/noise_aware.npz'}
    rows={}
    for k,path in files.items():
        p=np.load(path)['params']; u,kr,kt,_,_=propagate(p)
        r={'target_fro':float(np.linalg.norm(u-TARGET)),'KR_fro':float(np.linalg.norm(kr)),'KT_fro':float(np.linalg.norm(kt)),'time_t0_eta10':float(np.abs(p[:,:15]).sum()/10+np.abs(p[:,29:]).sum()),'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        r['passed']=bool(r['target_fro']<1e-9 and (k not in ['first_cnot','robust_time'] or max(r['KR_fro'],r['KT_fro'])<1e-9))
        rows[k]=r
    record={'status':'PASS' if all(r['passed'] for r in rows.values()) else 'FAIL','controls':rows,'elapsed_s':time.perf_counter()-start,'python':sys.version,'replay_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'Independent 8x8 reconstruction of fixed targets, derivatives, and duration; not an interval certificate or an optimizer replay.'}
    with (args.out/'result.json').open('x',encoding='utf-8') as f: json.dump(record,f,indent=2)
    print(json.dumps(record,indent=2))
    if record['status']!='PASS': raise SystemExit(1)

if __name__=='__main__': main()
