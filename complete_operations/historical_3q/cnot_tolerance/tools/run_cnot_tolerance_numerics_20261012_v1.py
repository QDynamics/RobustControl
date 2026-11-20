"""Registered numerical diagnostics and local CNOT-family continuation."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/cnot_tolerance_20261012_v1'
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(RUN / 'legacy_model'))
import cnot_tolerance_v1 as p
import v11_global_model as legacy


def save(path, data):
    with path.open('x', encoding='utf-8') as h:
        json.dump(data, h, indent=2, allow_nan=False)


def main():
    start = time.perf_counter()
    out = RUN / 'numerical'
    out.mkdir(exist_ok=False)
    save(out/'INPUTS.json', {'sha256': {str(q.relative_to(ROOT)): hashlib.sha256(q.read_bytes()).hexdigest() for q in [Path(__file__), ROOT/'src/cnot_tolerance_v1.py', RUN/'CONFIG.json']}, 'versions': {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__}})
    kinds = ['whole', 'module_response', 'modular', 'noise_complete', 'nominal']
    centers = {kind: np.load(RUN/'inputs'/f'{kind}.npz')['params'].ravel() for kind in kinds}
    center = centers['whole']
    r0, j0, *_ = legacy.global_constraints(center)
    jz = j0[:, p.AREA_INDICES] / p.AREA_WEIGHTS
    u, sv, vh = np.linalg.svd(jz, full_matrices=True)
    rank = int(np.sum(sv > 1e-8))
    assert rank == 52, ('area-only rank', rank)
    seed = np.random.default_rng(26101219).normal(size=90)
    tangent = vh[rank:].T @ (vh[rank:] @ seed)
    tangent /= np.linalg.norm(tangent)
    target_j = jz[:19]
    response_j = np.vstack((jz[0], jz[19:]))
    geometry = {'joint_singular_values': sv.tolist(), 'joint_rank': rank,
                'target_singular_values': np.linalg.svd(target_j, compute_uv=False).tolist(),
                'response_singular_values': np.linalg.svd(response_j, compute_uv=False).tolist(),
                'area_dimension': 90, 'numerical_fiber_dimension': 90-rank,
                'coordinate_weights': '1.5 for rotation area, 3 for twisting area',
                'tangent_linear_residual': float(np.linalg.norm(jz@tangent))}
    fd_checks = []
    for z in [tangent, vh[0], vh[rank-1]]:
        v = np.zeros(132); v[p.AREA_INDICES] = z/p.AREA_WEIGHTS
        h = 2e-6
        rp = legacy.global_constraints(center+h*v, False)[0]
        rm = legacy.global_constraints(center-h*v, False)[0]
        fd_checks.append(float(np.max(np.abs((rp-rm)/(2*h)-j0@v))))
    geometry['directional_Jacobian_FD_max_errors'] = fd_checks
    assert max(fd_checks) < 1e-7
    traces = []
    for s in [-.04, -.02, .02, .04]:
        z = s*tangent
        history = []
        for step in range(15):
            q = center.copy(); q[p.AREA_INDICES] += z/p.AREA_WEIGHTS
            residual, jac, *_ = legacy.global_constraints(q)
            augr = np.r_[residual, tangent@z-s]
            history.append({'iteration': step, 'max_residual': float(np.max(np.abs(augr))), 'norm_z': float(np.linalg.norm(z))})
            if np.max(np.abs(augr)) <= 1e-11:
                break
            augj = np.vstack((jac[:, p.AREA_INDICES]/p.AREA_WEIGHTS, tangent))
            dz = np.linalg.lstsq(augj, -augr, rcond=1e-12)[0]
            accepted = False
            for back in range(8):
                nz = z + 2.**(-back)*dz
                nq = center.copy(); nq[p.AREA_INDICES] += nz/p.AREA_WEIGHTS
                nr = np.r_[legacy.global_constraints(nq, False)[0], tangent@nz-s]
                if np.linalg.norm(nr) < np.linalg.norm(augr):
                    z = nz; accepted = True; break
            if not accepted:
                break
        q = center.copy(); q[p.AREA_INDICES] += z/p.AREA_WEIGHTS
        final_residual = float(np.max(np.abs(legacy.global_constraints(q, False)[0])))
        passed = final_residual <= 1e-10 and p.gate_time(q) <= 20 and np.all(np.abs(q) <= p.LIMITS)
        traces.append({'s': s, 'accepted': bool(passed), 'residual': final_residual, 'z_norm': float(np.linalg.norm(z)), 'trace': history})
        if passed:
            centers[f'C_s_{s:+.2f}'] = q
    save(out/'GEOMETRY.json', {'geometry': geometry, 'continuation': traces})
    np.savez(out/'CONTROLS.npz', **centers, tangent_z=tangent)
    results = {}
    data = {}
    for name, params in centers.items():
        seq = p.prepare(params)
        nominal, k = p.propagate(seq, responses=True)
        ij, coeff = p.taylor_float(seq)
        data[name+'_coefficients'] = coeff
        diff0 = float(np.linalg.norm(coeff[0]-nominal))
        diffr = float(np.linalg.norm(coeff[ij.index((1,0))]+1j*nominal@k[0]))
        difft = float(np.linalg.norm(coeff[ij.index((0,1))]+1j*nominal@k[1]))
        check_points = [(.01, -.01), (-.005, .003), (0., 0.)]
        independent = [float(np.linalg.norm(p.propagate(seq,e)-p.propagate(seq,e,direct=True),2)) for e in check_points]
        assert max(independent) < 1e-11
        assert max(diff0, diffr, difft) < 1e-10
        residual_coeff = coeff.copy(); residual_coeff[0] -= p.TARGET
        sample = {}
        for radius in [.001, .005, .01, .02]:
            vals = [(x, y, p.halfdiamond(p.propagate(seq,(x,y)))) for x,y in [(0,0),(-radius,-radius),(-radius,radius),(radius,-radius),(radius,radius)]]
            sample[str(radius)] = vals
        grid_max, grid_at = 0., None
        grid = []
        if name in kinds:
            for x in np.linspace(-.01,.01,41):
                for y in np.linspace(-.01,.01,41):
                    e = p.halfdiamond(p.propagate(seq,(x,y)))
                    grid.append([x,y,e])
                    if e>grid_max: grid_max,grid_at=e,[float(x),float(y)]
            data[name+'_grid'] = np.array(grid)
        bounds = {str(order): {str(r): p.reference_bound(residual_coeff,order,r,sum(p.action(params))) for r in [.001,.005,.01,.02]} for order in [2,4,6,8]}
        result = {'target_operator_residual': float(np.linalg.norm(nominal-p.TARGET,2)),
                  'response_operator_residuals': [float(np.linalg.norm(a,2)) for a in k],
                  'leakage_operator_residual': float(np.linalg.norm((p.I8-p.P)@nominal@p.D,2)),
                  'unitarity_operator_residual': float(np.linalg.norm(nominal.conj().T@nominal-p.I8,2)),
                  'gate_time_t0': p.gate_time(params), 'actions': p.action(params),
                  'direct_expm_operator_differences': independent,
                  'Taylor_response_identity_errors': [diff0,diffr,difft],
                  'sampled_corner_errors': sample, 'grid_max_at_r_0p01': grid_max if grid else None,
                  'grid_argmax': grid_at, 'floating_bounds_diagnostic_only': bounds}
        results[name] = result
        print(name, 'T', result['gate_time_t0'], 'B8(.01)',bounds['8']['0.01'], 'grid',result['grid_max_at_r_0p01'],flush=True)
        if time.perf_counter()-start > 1800:
            raise TimeoutError('Registered wall budget exceeded')
    np.savez_compressed(out/'ARRAYS.npz', **data)
    save(out/'RESULTS.json', {'status':'NUMERICAL_DIAGNOSTICS_COMPLETE_NOT_INTERVAL_CERTIFICATION', 'results':results, 'elapsed_s':time.perf_counter()-start, 'failures':[]})


if __name__ == '__main__':
    main()
