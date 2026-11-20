"""Replay continuous-domain bounds into a NEW directory (no archive writes).

Run from the original workspace or the extracted reproducibility package:
python tools/replay_cnot_tolerance_20261012_v1.py --out runs/my_new_replay
Optionally --names whole limits the replay to the primary C implementation.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import cnot_tolerance_v1 as p
import cnot_interval_v1 as b
import cnot_interval_ball_v2 as ball


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);parser.add_argument('--names',nargs='+')
    args=parser.parse_args();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();run=ROOT/'runs/cnot_tolerance_20261012_v1'
    controls=np.load(run/'numerical/CONTROLS.npz')
    saved=json.loads((run/'interval_v2/RESULTS.json').read_text())['results']
    names=args.names or list(saved)
    rows={};target=b.exact_target()
    for name in names:
        if name not in saved:raise ValueError(name)
        params=controls[name]
        ij,center,radius,trace=ball.taylor_balls(p.pulses(params),8,p.JX,p.JY,p.ZZ)
        coeff=b.CM.point(center);coeff[0]=coeff[0]-target
        norm_bounds=[b.posadd(b.spectral_upper(coeff[k])[0],float(radius[k])) for k in range(len(ij))]
        degree=[b.possum(norm_bounds[k] for k,(a,c) in enumerate(ij) if a+c==d) for d in range(9)]
        q=params.reshape(3,44)
        action=b.posadd(b.posmul(1.5,b.possum(np.abs(q[:,:15]).ravel())),b.posmul(3.,b.possum(np.abs(q[:,29:]).ravel())))
        r=saved[name]['noise_radii']['0.01']['delivered_radius']
        ub=b.bound_upper(degree,action,r,8)
        assert ub<float(b.down(.01))
        b1=b.bound_upper(degree,action,.01,8)
        rows[name]={'delivered_radius':r,'fresh_upper_bound_at_radius':ub,'fresh_upper_bound_at_1percent':b1,'absolute_difference_from_original_1percent_bound':abs(b1-saved[name]['bounds']['8']['0.01'])}
        print(name,rows[name],flush=True)
    result={'status':'REPLAY_PASS_FOR_LISTED_CONTROLS','controls':rows,'elapsed_s':time.perf_counter()-start,'versions':{'python':sys.version,'numpy':np.__version__},'sha256':{str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in [Path(__file__),ROOT/'src/cnot_tolerance_v1.py',ROOT/'src/cnot_interval_v1.py',ROOT/'src/cnot_interval_ball_v2.py',run/'numerical/CONTROLS.npz']},'scope':'Recomputes the continuous shared-error bounds for listed controls. The analytic interpolation and common-schedule proof are in the primary acceptance record.','failures':[]}
    with (out/'RESULTS.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)


if __name__=='__main__':main()
