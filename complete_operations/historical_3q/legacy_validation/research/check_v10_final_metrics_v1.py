"""Independent coefficient check and consistent before/after waveform metrics.

No control search or replacement of existing run evidence. Budget: 300 seconds.
"""
from pathlib import Path
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import json, time, hashlib, sys
import numpy as np
import run_cnot_smoothing_v1 as smooth
from check_v10_science_v1 import propagate

HERE = Path(__file__).resolve().parent
OUT = HERE / 'runs' / 'v10_final_metrics_20260926_v1'


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    save(OUT / 'protocol.json', {
        'purpose': 'Reconstruct stored results only; no search.',
        'budget_s': 300,
        'coefficient_check': 'Independent physical 8x8 Pauli-tensor second Taylor coefficient for all three robust modules; operator norm tolerance 1e-10.',
        'waveform_metric': 'Maximum Frobenius norm over complete target mismatch, full product KR/KT, and all module KR/KT; same definition before and after compensation.',
        'widths': 'Exactly the eight previously stored widths; tighter full8 integration settings.',
    })
    run = HERE / 'runs'
    source = run / 'cnot_optimization_20260926_v2/robust_time.npz'
    p0 = np.load(source)['params']
    stored = json.loads((run / 'cnot_benchmark_20260926_v1/result.json').read_text())['candidates']['robust_time']['module_direction_11_budgets']
    coefficients = []
    for k, q in enumerate(p0):
        one = np.zeros((3, 44)); one[0] = q
        _, _, _, i1, i2 = propagate(one)
        a = float(1.5 * np.abs(q[:15]).sum() + 3 * np.abs(q[29:]).sum())
        norm = float(np.linalg.norm(i2, 2))
        coefficients.append({'module': k + 1, 'A': a, 'I1_op': float(np.linalg.norm(i1, 2)), 'I2_op': norm, 'stored_I2_op': stored[k]['I2_op'], 'absolute_difference': abs(norm - stored[k]['I2_op'])})
    previous = json.loads((run / 'cnot_smoothing_20260926_v1/result.json').read_text())['rows']
    widths = []
    for row in previous:
        if time.perf_counter() - start > 300:
            raise TimeoutError('Diagnostic replay budget exhausted')
        ss = smooth.schedules(p0, row['width_t0'])
        _, _, analysis = smooth.evaluate(p0.ravel(), ss, tight=True, full=True)
        metrics = smooth.full_metrics(analysis)
        after = {key: row[key] for key in metrics}
        item = {'fraction_min_slot': row['fraction_min_slot'], 'width_t0': row['width_t0'], 'before': metrics, 'after': after, 'before_max': max(metrics.values()), 'after_max': max(after.values())}
        widths.append(item)
        print('WIDTH', row['fraction_min_slot'], item['before_max'], item['after_max'], flush=True)
    u = propagate(p0)[0]
    from check_v10_science_v1 import TARGET
    d0 = 16 * float(np.linalg.norm(u - TARGET, 2))
    r = .001
    sharp_budget = d0 + 16 * sum(r * v['I1_op'] + r*r*v['I2_op'] + (r*v['A'])**3/6 for v in coefficients)
    coarse_budget = d0 + 16 * sum(r * v['I1_op'] + (r*v['A'])**2/2 for v in coefficients)
    passed = max(v['absolute_difference'] for v in coefficients) < 1e-10 and all(v['after_max'] < 3e-11 and v['after_max'] < v['before_max'] for v in widths)
    inputs = [Path(__file__), HERE/'check_v10_science_v1.py', HERE/'run_cnot_smoothing_v1.py', source, run/'cnot_benchmark_20260926_v1/result.json', run/'cnot_smoothing_20260926_v1/result.json']
    save(OUT/'bindings.json', {str(path.relative_to(smooth.io.ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs})
    result = {'status': 'PASS' if passed else 'FAIL', 'module_coefficients': coefficients, 'r001_depth16_taylor_budget': sharp_budget, 'r001_depth16_coarse_budget': coarse_budget, 'waveform_rows': widths, 'elapsed_s': time.perf_counter()-start, 'python': sys.version, 'interval_certified': False}
    save(OUT/'result.json', result)
    print('FINAL_METRICS', result['status'], sharp_budget, coarse_budget, flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
