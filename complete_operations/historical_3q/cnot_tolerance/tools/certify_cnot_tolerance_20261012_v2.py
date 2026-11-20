"""Successor certificate with unitary coefficient norm balls; same registered task."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
from pathlib import Path
import hashlib
import json
import sys
import time
import platform
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/cnot_tolerance_20261012_v1'
sys.path.insert(0,str(ROOT/'src'))
import cnot_tolerance_v1 as p
import cnot_interval_v1 as b
import cnot_interval_ball_v2 as ball
from cnot_interval_preflight_20261012_v1 import run as preflight


def save(path,value):
    with path.open('x',encoding='utf-8') as h:json.dump(value,h,indent=2,allow_nan=False)


def radius(degrees,action,delta):
    lo,hi=0.,.05
    limit=float(b.down(delta))
    for _ in range(60):
        mid=(lo+hi)/2
        if b.bound_upper(degrees,action,mid,8)<=limit:lo=mid
        else:hi=mid
    # Decimal reporting grid of 1e-6; this deliberately rounds the radius down.
    delivered=float(np.floor(lo*1e6)/1e6)
    assert b.bound_upper(degrees,action,delivered,8)<=limit
    return {'bisection_lower':lo,'delivered_radius':delivered,'bound_at_delivered':b.bound_upper(degrees,action,delivered,8),'tolerance_lower':limit}


def control_radius(bound,r,delta,time_upper,gamma=0.):
    # Half-diamond bound for known local Markov dephasing is 3 gamma T / 2.
    gam=float(b.up(gamma)) if gamma else 0.
    addition=b.posmul(b.posmul(1.5,gam),time_upper) if gamma else 0.
    numerator=float(b.down(float(b.down(delta))-b.posadd(bound,addition)))
    denominator=b.posadd(b.posadd(1.,float(b.up(r))),b.posmul(1.5,gam))
    eta=max(0.,float(b.down(numerator/denominator)))
    # Leave a visible safety margin, keeping decimal table values easy to audit.
    eta=float(np.floor(eta*1e6)/1e6)
    total=b.posadd(bound,b.posmul(b.posadd(1.,float(b.up(r))),eta))
    if gamma:total=b.posadd(total,b.posmul(b.posmul(1.5,gam),b.posadd(time_upper,eta)))
    if eta>0:assert total<=float(b.down(delta))
    return {'weighted_l1_radius':eta,'equal_per_area_radius_rad':float(b.down(eta/202.5)) if eta else 0.,
            'bound_including_control_and_dephasing':total,'dephasing_rate_t0':gamma}


def main():
    start=time.perf_counter();out=RUN/'interval_v2';out.mkdir(exist_ok=False)
    sources=[Path(__file__),ROOT/'src/cnot_interval_v1.py',ROOT/'src/cnot_interval_ball_v2.py',ROOT/'src/cnot_tolerance_v1.py',ROOT/'tools/cnot_interval_preflight_20261012_v1.py',RUN/'CONFIG.json',RUN/'numerical/CONTROLS.npz']
    save(out/'INPUTS.json',{'sha256':{str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in sources},'versions':{'python':sys.version,'numpy':np.__version__,'platform':platform.platform()},'arithmetic_contract':'IEEE-754 binary64 pointwise basic operations and sqrt are rounded to nearest; nextafter enlarges outward. Explicit matrix sums; Decimal directed 90-digit Taylor intervals. No BLAS or libm transcendental result is trusted as a bound.'})
    preflight_result=preflight();save(out/'PREFLIGHT.json',preflight_result)
    arrays=np.load(RUN/'numerical/CONTROLS.npz')
    num=json.loads((RUN/'numerical/RESULTS.json').read_text())['results']
    results={};cert_arrays={};target=b.exact_target()
    for name in arrays.files:
        if name=='tangent_z':continue
        params=arrays[name]
        stamp=time.perf_counter()
        ij,centers,errors,trace=ball.taylor_balls(p.pulses(params),8,p.JX,p.JY,p.ZZ,
            progress=lambda n:print(name,'interval pulses',n,flush=True))
        cert_arrays[name+'_centers']=centers
        cert_arrays[name+'_radii']=errors
        coeff=b.CM.point(centers)
        coeff[0]=coeff[0]-target
        coefficient_bounds=[];norm_audits=[]
        for j,pair in enumerate(ij):
            bound,detail=b.spectral_upper(coeff[j]);bound=b.posadd(bound,float(errors[j]));coefficient_bounds.append(bound)
            detail['coefficient_ball_radius']=float(errors[j])
            detail['degree']=list(pair);norm_audits.append(detail)
        degrees=[b.possum(coefficient_bounds[j] for j,(a,c) in enumerate(ij) if a+c==k) for k in range(9)]
        q=params.reshape(3,44)
        sr=b.posmul(1.5,b.possum(np.abs(q[:,:15]).ravel()))
        st=b.posmul(3.,b.possum(np.abs(q[:,29:]).ravel()))
        action=b.posadd(sr,st)
        bounds={str(k):{str(r):b.bound_upper(degrees,action,r,k) for r in [.001,.005,.01,.02]} for k in [2,4,6,8]}
        radii={str(d):radius(degrees,action,d) for d in [.001,.01,.05]}
        # Independently rounded-up whole-microsecond units in dimensionless t0.
        # Time is diagnostic until a separate analytic actuator check binds it.
        time_reported=float(np.ceil(num[name]['gate_time_t0']*1e6)/1e6)
        ctrl={str(r):control_radius(bounds['8'][str(r)],r,.01,time_reported) for r in [.001,.005,.01,.02]}
        ctrl_markov=control_radius(bounds['8']['0.01'],.01,.01,time_reported,1e-5)
        results[name]={'coefficient_spectral_upper_bounds':coefficient_bounds,'degree_sum_bounds':degrees,
            'action_rotation_upper':sr,'action_twisting_upper':st,'action_sum_upper':action,
            'bounds':bounds,'noise_radii':radii,'control_radii_delta_0p01':ctrl,
            'secondary_dephasing_control_radius_pending_duration_check':ctrl_markov,
            'gate_time_rounded_up_t0':time_reported,
            'largest_coefficient_ball_radius':float(np.max(errors)),'nominal_coefficient_ball_radius':float(errors[0]),'coefficient_propagation_trace':trace,'norm_factorization_audits':norm_audits,
            'elapsed_s':time.perf_counter()-stamp}
        print('CERT',name,'B(.01)',bounds['8']['0.01'],'r(delta=.01)',radii['0.01']['delivered_radius'],'eta',ctrl['0.01']['weighted_l1_radius'],flush=True)
        if time.perf_counter()-start>1800:raise TimeoutError('Registered wall budget exceeded')
    np.savez_compressed(out/'COEFFICIENT_BALLS.npz',**cert_arrays)
    save(out/'RESULTS.json',{'status':'OUTWARD_ROUNDED_CONTINUOUS_SHARED_ERROR_BOUNDS_COMPUTED_PENDING_PRIMARY_AUDIT','order':8,'arithmetic_preflight':preflight_result,'results':results,'elapsed_s':time.perf_counter()-start,'failures':[]})


if __name__=='__main__':main()
