"""Primary-agent audit and continuous-family extension of the CNOT bound.

No optimization or candidate reselection. The same five registered C points are
joined in order with affine area segments, retaining their common fixed axes.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
from pathlib import Path
from fractions import Fraction as F
import hashlib
import json
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/cnot_tolerance_20261012_v1'
sys.path.insert(0,str(ROOT/'src'))
import cnot_tolerance_v1 as p
import cnot_interval_v1 as b
from cnot_interval_preflight_20261012_v1 import run as arithmetic_preflight


def save(path,value):
    with path.open('x',encoding='utf-8') as h:json.dump(value,h,indent=2,allow_nan=False)


def pi_bounds():
    def atan(n):
        x=F(1,n)
        total=sum((-1)**k*x**(2*k+1)/F(2*k+1) for k in range(80))
        next_term=x**161/F(161)
        return total,total+next_term
    a,c=atan(5),atan(239)
    lower,upper=16*a[0]-4*c[1],16*a[1]-4*c[0]
    lo,hi=float(b.down(float(lower))),float(b.up(float(upper)))
    assert F.from_float(lo)<=lower<=upper<=F.from_float(hi)
    return lo,hi


def duration_certificate(area_bounds,pi_upper):
    caps=np.tile(np.r_[np.full(15,10.),np.ones(15)],3)
    approximate=p.duration(area_bounds,caps)
    durations=[];peaks=[]
    tb=b.posdiv(1.,50.)
    pi_tb=b.posmul(b.posmul(2.,pi_upper),tb)
    for area,cap,tau in zip(area_bounds,caps,approximate):
        # Actual delivered durations are binary64 real numbers with a 1e-6
        # nominal grid and one extra grid step for strict command headroom.
        candidate=float((np.ceil(tau*1e6)+1)/1e6)
        ratio=b.posdiv(pi_tb,candidate)
        root=float(b.up(np.sqrt(b.posadd(1.,b.posmul(ratio,ratio)))))
        peak=b.posmul(b.posdiv(float(area),candidate),b.posadd(1.,root))
        assert peak<=float(cap) and candidate>=.04,(area,candidate,peak,cap)
        durations.append(candidate);peaks.append(peak)
    gate_time=b.posadd(b.possum(durations),b.posmul(89.,b.posdiv(1.,1000.)))
    return {'durations_t0':durations,'peak_command_upper':peaks,'gate_time_upper_t0':gate_time,'max_rotation_command_upper':max(peaks[:15]+peaks[30:45]+peaks[60:75]),'max_twisting_command_upper':max(peaks[15:30]+peaks[45:60]+peaks[75:90])}


def weighted_difference(a,c):
    # Difference of stored floats can itself round; enclose each subtraction.
    x=b.CM.point(a[p.AREA_INDICES])-b.CM.point(c[p.AREA_INDICES])
    absolute=np.maximum(np.abs(x.rl),np.abs(x.rh))
    return b.possum(b.posmul(w,t) for w,t in zip(p.AREA_WEIGHTS,absolute))


def main():
    start=time.perf_counter();out=RUN/'primary_audit';out.mkdir(exist_ok=False)
    sources=[Path(__file__),ROOT/'src/cnot_interval_v1.py',ROOT/'src/cnot_interval_ball_v2.py',ROOT/'src/cnot_tolerance_v1.py',RUN/'interval_v2/RESULTS.json',RUN/'interval_v2/COEFFICIENT_BALLS.npz',RUN/'numerical/CONTROLS.npz']
    save(out/'INPUTS.json',{'role':'Primary agent, not delegated','sha256':{str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in sources}})
    original=json.loads((RUN/'INPUTS_SHA256.json').read_text())
    for entry in original['entries'].values():
        assert hashlib.sha256((RUN/entry['destination']).read_bytes()).hexdigest()==entry['sha256']
    assert hashlib.sha256(Path(original['archive']).read_bytes()).hexdigest()==original['archive_sha256']
    for stage in ['numerical','interval_v2']:
        inputs=json.loads((RUN/stage/'INPUTS.json').read_text())
        for relative,sha in inputs['sha256'].items():assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==sha,relative
    preflight=arithmetic_preflight()
    cert=json.loads((RUN/'interval_v2/RESULTS.json').read_text())['results']
    numeric=json.loads((RUN/'numerical/RESULTS.json').read_text())['results']
    params=np.load(RUN/'numerical/CONTROLS.npz')
    coefficients=np.load(RUN/'interval_v2/COEFFICIENT_BALLS.npz')
    floating=np.load(RUN/'numerical/ARRAYS.npz')
    diffs={}
    for name in cert:
        c=coefficients[name+'_centers'];radii=coefficients[name+'_radii']
        other=floating[name+'_coefficients']
        normdiff=np.array([np.linalg.norm(x-y,2) for x,y in zip(c,other)])
        # Floating cross-check only; rigor of the ball follows from its
        # interval local products and analytic propagation, audited below.
        assert np.all(normdiff<=radii+1e-10),name
        diffs[name]={'largest_coefficient_difference':float(np.max(normdiff)),'nominal_difference':float(normdiff[0]),'largest_difference_to_ball_radius_ratio':float(np.max(normdiff/np.maximum(radii,1e-300)))}
        for rstr,points in numeric[name]['sampled_corner_errors'].items():
            assert max(x[2] for x in points)<=cert[name]['bounds']['8'][rstr]
        if numeric[name]['grid_max_at_r_0p01'] is not None:
            assert numeric[name]['grid_max_at_r_0p01']<=cert[name]['bounds']['8']['0.01']
    names=['C_s_-0.04','C_s_-0.02','whole','C_s_+0.02','C_s_+0.04']
    phase_indices=np.setdiff1d(np.arange(132),p.AREA_INDICES)
    assert all(np.array_equal(params[n][phase_indices],params['whole'][phase_indices]) for n in names)
    segments=[];r=.01;r_up=float(b.up(r))
    for left,right in zip(names[:-1],names[1:]):
        w=weighted_difference(params[left],params[right])
        extra=b.posdiv(b.pospow(b.posmul(b.posadd(1.,r_up),w),2),8.)
        base=max(cert[left]['bounds']['8']['0.01'],cert[right]['bounds']['8']['0.01'])
        bound=b.posadd(base,extra)
        diagnostics=[]
        for t in [.25,.5,.75]:
            control=(1-t)*params[left]+t*params[right]
            seq=p.prepare(control)
            errors=[p.halfdiamond(p.propagate(seq,(x,y))) for x in [-r,0.,r] for y in [-r,0.,r]]
            actual=max(errors)
            assert actual<=bound
            diagnostics.append({'t':t,'nine_point_max_halfdiamond':actual,'nominal_target_residual':float(np.linalg.norm(p.propagate(seq)-p.TARGET,2))})
        segments.append({'left':left,'right':right,'weighted_l1_edge_upper':w,'interpolation_remainder_upper':extra,'uniform_error_upper':bound,'sampled_diagnostics':diagnostics})
    family_bound=max(s['uniform_error_upper'] for s in segments)
    pi_lo,pi_hi=pi_bounds()
    # Report an easy-to-use conservative radius, chosen by rounding DOWN the
    # analytically available margin, not by testing favorable controls.
    available=float(b.down(float(b.down(.01))-family_bound))/b.posadd(1.,r_up)
    eta_coherent=float(np.floor(available*1e6)/1e6)
    area_max=np.max(np.abs(np.stack([params[n][p.AREA_INDICES] for n in names])),axis=0)
    schedules={}
    for eta in [0.,eta_coherent,.0037]:
        area_bounds=np.array([b.posadd(float(a),b.posdiv(eta,float(w))) for a,w in zip(area_max,p.AREA_WEIGHTS)])
        schedules[str(eta)]=duration_certificate(area_bounds,pi_hi)
    assert all(s['gate_time_upper_t0']<20. for s in schedules.values())
    coherent=b.posadd(family_bound,b.posmul(b.posadd(1.,r_up),eta_coherent))
    assert coherent<float(b.down(.01))
    gamma=b.posdiv(1.,100000.)
    markov_eta=.0037
    noise_term=b.posmul(b.posmul(1.5,gamma),schedules[str(markov_eta)]['gate_time_upper_t0'])
    with_markov=b.posadd(b.posadd(family_bound,b.posmul(b.posadd(1.,r_up),markov_eta)),noise_term)
    assert with_markov<float(b.down(.01))
    per_area=1.8e-5
    assert b.posmul(202.5,float(b.up(per_area)))<=float(b.down(markov_eta))
    primary=cert['whole']['control_radii_delta_0p01']['0.01']['weighted_l1_radius']>=.001
    assert primary
    audit={
        'status':'PRIMARY_ACCEPTED_CONTINUOUS_ERROR_BOUNDS_AND_TOLERANCE_FAMILY',
        'primary_registered_criterion_passed':primary,
        'original_archive_and_all_staged_inputs_unchanged':True,
        'independent_arithmetic_preflight':preflight,
        'coefficient_crosschecks':diffs,
        'pi_enclosure_from_exact_rational_Machin_series':[pi_lo,pi_hi],
        'continuous_family':{'vertices':names,'construction':'Affine interpolation of areas between adjacent registered C controls, with fixed rotation axes; a finite-tolerance family, not a claim that every interpolant lies on an exact robustness manifold.','segments':segments,'uniform_unitary_error_bound_r_0p01':family_bound,'weighted_l1_radius_coherent':eta_coherent,'uniform_error_with_coherent_radius':coherent,'common_actuator_schedules':schedules,'weighted_l1_radius_with_gamma_t0_1e_5':markov_eta,'uniform_halfdiamond_upper_with_dephasing':with_markov,'dephasing_contribution_upper':noise_term,'equal_independent_area_error_box_rad':per_area,'equal_area_box_requires_fixed_axes':True},
        'analytic_audit':[
            'A primitive Taylor coefficient P_k has norm <= ||H||^k/k!, with exactly unitary P_0. The coefficient-ball recurrence follows from P_true C_true - P_mid C_mid = P_true E_old + E_P C_mid; all local products and norm-factorization residuals are enclosed.',
            'The multivariate Taylor polynomial is the ordered product convolution. Along any real direction in the calibration square, the derivative of order K+1 is bounded by (S_R+S_T)^(K+1), from Leibniz and real-parameter unitarity. Integral Taylor remainder gives the factorial bound without an exponential factor.',
            'All nominal and first-order coefficient residuals remain in the delivered upper bound. No stored numerical point is replaced by an exact root.',
            'Half diamond distance of unitary channels is <= ||U-V||_op, including arbitrary reference systems. The target enclosure includes the exact 1/sqrt(2), 1/sqrt(3) factors and both hidden multiplicity copies.',
            'For affine area interpolation, the second derivative along the segment has norm <= ((1+r) W_edge)^2. The Banach-valued linear-interpolation error is <= t(1-t) M/2 <= M/8, which gives a uniform continuum bound from the two endpoint bounds.',
            'Duhamel and telescoping products give ||U(a+h,e)-U(a,e)|| <= (1+r) W(h;a). The rotation-axis estimate uses 2|sin(Delta phi/2)|<=|Delta phi|; every uncertainty direction is charged.',
            'Common waveform durations are verified by upper bounds on the analytic command peak using an exact-rational enclosure of pi. Every slot and all 89 switches are included. Each tube member fits the same reserved schedule.',
            'The local-dephasing generator is (gamma/2) sum_j (Z_j rho Z_j-rho). Channel contractivity and Duhamel give half-diamond addition <= 3 gamma T/2, valid during the control and all idle switches.'
        ],
        'evidence_limits':[
            'Outward-rounded finite-error upper bounds under the stated arithmetic contract and physical model; no interval root or uniform rank certificate for the CNOT manifold is claimed.',
            'The continuous interpolating family is guaranteed within tolerance, but its interiors can have nonzero nominal and first-order residuals.',
            'Guaranteed radii are sufficient lower bounds on the true maximal acceptable radius. A failed sufficient bound does not prove an implementation fails the task.',
            'No arbitrary-depth threshold, unknown actuator robustness, experimental device validation, or global optimality is established.',
            'The optional dephasing guarantee concerns the gate operation; addressed preparation, readout, initialization and reset belong to a different full-cycle task.'
        ],
        'source_for_channel_norm_convention':'John Watrous, The Theory of Quantum Information, Chapter 3: https://cs.uwaterloo.ca/~watrous/TQI/TQI.double.3.pdf',
        'elapsed_s':time.perf_counter()-start,'failures':[]
    }
    save(out/'ACCEPTANCE.json',audit)
    print(json.dumps({'status':audit['status'],'family_bound':family_bound,'eta_coherent':eta_coherent,'eta_with_dephasing':markov_eta,'total_with_dephasing':with_markov,'common_time_t0':schedules[str(markov_eta)]['gate_time_upper_t0'],'per_area_box':per_area}))


if __name__=='__main__':main()
