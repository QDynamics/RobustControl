"""Refit paired Dicke-code gates on 17 offsets and replay their held-out channels.

With --scores-from, replay saved choices including their training rankings and
all held-out/leakage records. The destination must be new in either mode.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

# The archived pilot requests single-thread BLAS before importing NumPy.
import npj_qi_register_depth_20261118_v1 as pilot
import npj_qi_independent_audit_20261118_v1 as lab
import npj_qi_register_depth_projection_20261118_v1 as projection
import numpy as np
import scipy
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'docs/NPJ_QI_FULL_RANGE_REVIEW_PROTOCOL_20261118_v1.md'
PROTOCOL_HASH = '6b5f4b154c6f95e41d3f0526fd46576c7b56a46c2e3cd6844ce8748905ed041c'
RUN_ID = 'npj_qi_full_range_review_20261118_v1'
BASES = (6112300, 6112400, 6112500, 6112600)
PAIRS = [(n, base + n) for base in BASES for n in (4, 5)]
ARMS = ('direct', 'seeded')
POSITIONS = ('nominal', 'midpoint', 'finite', 'restored')
DEPTHS = (1, 2, 4, 8, 16, 32)
LEAK_DEPTHS = (1, 4, 16, 32)
QUALIFY = 1e-6
AUDIT_TOL = 1e-8
TRAIN = np.array([(0., 0.), *product((-.01, .01), repeat=2),
                  (-.01, 0.), (.01, 0.), (0., -.01), (0., .01),
                  *((a * s, b * t) for a, b in ((.003, .007), (.007, .003))
                    for s, t in product((-1., 1.), repeat=2))])


def test_offsets():
    rng = np.random.Generator(np.random.PCG64(6112700))
    interior = rng.uniform(-.01, .01, (64, 2))
    edge = []
    for axis in (0, 1):
        for end in (-.01, .01):
            for cross in (-.00875, -.00375, .00375, .00875):
                edge.append((end, cross) if axis == 0 else (cross, end))
    heldout = np.vstack((interior, edge))
    keys = lambda a: set(map(tuple, np.round(a, 14)))
    excluded = np.vstack((TRAIN, pilot.JET, lab.grid(5), lab.grid(9)))
    if (len(keys(TRAIN)) != 17 or len(keys(heldout)) != 80 or
            keys(heldout) & keys(excluded) or not np.all(np.abs(interior) < .01)):
        raise ValueError('duplicate or overlapping calibration sets')
    return heldout


def write_csv(path, rows, columns=None):
    with path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def spin_blocks(n):
    return [pilot.Block(rep, n) for rep in pilot.schur_representations(n)]


def label(arm, position):
    return 'nominal' if position == 'nominal' else f'{arm}_{position}'


def development_files(n):
    return {'nominal': lab.DEVELOPMENT / f'n{n}_nominal.npy',
            'corrected': lab.DEVELOPMENT / f'n{n}_corrected.npy',
            'direct': lab.RESTORED / f'n{n}_nominal_restored.npy',
            'refined': lab.RESTORED / f'n{n}_corrected_restored.npy'}


def input_bindings():
    paths = [Path(__file__).resolve(), PROTOCOL, ROOT / 'src/manifold_schur_v1.py',
             ROOT / 'src/cnot_tolerance_v1.py', Path(pilot.__file__), Path(lab.__file__),
             Path(projection.__file__), Path(projection.matched.__file__)]
    for n in (4, 5):
        paths.extend(development_files(n).values())
    return {p.relative_to(ROOT).as_posix(): lab.sha(p) for p in paths}


def physical_clock(n):
    pairs = pilot.LENGTHS[n]
    rot, twist = float(pilot.duration(np.pi, 10.)), float(pilot.duration(1.2, 1.))
    single = pairs * (rot + twist) + (2 * pairs - 1) * .001
    return {'pulse_pairs_per_gate': pairs, 'rotation_slot_t0': rot, 'twist_slot_t0': twist,
            'inter_slot_switch_t0': .001, 'single_gate_t0': single,
            'depth32_t0': 32 * single + 31 * .001, 'depth32_slots': 64 * pairs}


def finite_residual_jacobian(blocks, n, params):
    residuals, derivatives = [], []
    for block in blocks:
        u, jac = block.word(params, TRAIN, jac=True)
        weight = np.sqrt(block.mult / (2 ** n))
        residual = (weight * (u - block.target)).ravel()
        derivative = (weight * jac).reshape(-1, len(params))
        residuals.extend((residual.real, residual.imag))
        derivatives.extend((derivative.real, derivative.imag))
    return np.concatenate(residuals), np.concatenate(derivatives)


def fit_finite(blocks, n, start, cap, checkpoint=None):
    """Cache each distinct residual/Jacobian pair and track best seen at a fixed call."""
    t0 = time.perf_counter()
    best, middle = start.copy(), None
    best_cost = np.inf
    prior, values = None, None
    history = []

    def compute(params):
        nonlocal prior, values, best, best_cost, middle
        if prior is None or not np.array_equal(params, prior):
            values = finite_residual_jacobian(blocks, n, params)
            prior = params.copy()
            score = float(values[0] @ values[0])
            history.append(score)
            if score < best_cost:
                best, best_cost = params.copy(), score
            if len(history) == checkpoint:
                middle = best.copy()
        return values

    bound = np.repeat([np.pi, 3 * np.pi, .6], len(start) // 3)
    sol = least_squares(lambda p: compute(p)[0], start, jac=lambda p: compute(p)[1],
                        bounds=(-bound, bound), max_nfev=cap, ftol=1e-13, xtol=1e-13,
                        gtol=1e-13)
    if len(history) > cap:
        raise RuntimeError('offset budget exceeded')
    info = {'evaluations': len(history), 'reason': sol.message,
            'wall_s': time.perf_counter() - t0, 'objective_squared_history': history,
            'best_squared_objective': best_cost}
    return best, best.copy() if middle is None else middle, info


def score_training(blocks, controls):
    matrices = [b.word(controls, TRAIN) for b in blocks]
    targets = [b.target for b in blocks]
    mults = [b.mult for b in blocks]
    return [pilot.metrics(targets, [g[i] for g in matrices], mults)[0] for i in range(len(TRAIN))]


def rank_controls(n, blocks, controls):
    candidates, selection = {}, {}
    for arm in ARMS:
        candidates[arm] = {}
        for pos in POSITIONS:
            name = label(arm, pos)
            scores = score_training(blocks, controls[name])
            candidates[arm][name] = {'nominal_D': scores[0], 'max_train_D': max(scores),
                                     'eligible': scores[0] <= QUALIFY, 'all_train_D': scores}
        valid = [key for key, v in candidates[arm].items() if v['eligible']]
        selection[arm] = min(valid, key=lambda key: candidates[arm][key]['max_train_D'],
                             default=None)
    jet = pilot.propagate(pilot.full_tensor_representation(n), np.array(pilot.unpack(controls['seeded_midpoint'])).T)
    norms = [float(np.linalg.norm(z, ord=2)) for z in (jet.generator_R, jet.generator_T)]
    raw = candidates['seeded']['seeded_midpoint']
    return {'candidates': candidates, 'selected': selection, 'seeded_midpoint_generator_R_T': norms,
            'seeded_midpoint_first_order_qualified': raw['eligible'] and max(norms) <= 1e-5}


def fit_one_pair(args):
    n, seed, parent = args
    folder = parent / f'n{n}_seed{seed}'
    folder.mkdir(exist_ok=False)
    t0 = time.perf_counter()
    try:
        blocks, length = spin_blocks(n), pilot.LENGTHS[n]
        rng = np.random.Generator(np.random.PCG64(seed))
        controls = {'initial': np.concatenate((rng.uniform(-1.6, 1.6, length),
                                               rng.uniform(-np.pi, np.pi, length),
                                               rng.uniform(-.45, .45, length)))}
        fits = {}
        controls['nominal'], fits['nominal'] = pilot.fit(blocks, n, controls['initial'], 'nominal', 100, np.inf)
        controls['seeded_midpoint'], fits['correction'] = pilot.fit(blocks, n, controls['nominal'],
                                                                   'corrected', 85, np.inf)
        controls['direct_finite'], controls['direct_midpoint'], fits['direct_finite'] = fit_finite(
            blocks, n, controls['nominal'], 80, checkpoint=25)
        controls['seeded_finite'], _, fits['seeded_finite'] = fit_finite(blocks, n, controls['seeded_midpoint'], 55)
        for arm in ARMS:
            controls[f'{arm}_restored'], fits[f'{arm}_restore'] = projection.restore(n, blocks, controls[f'{arm}_finite'])
        np.savez(folder / 'controls.npz', **controls)
        lab.write_json(folder / 'fit_histories.json', fits)
        info = rank_controls(n, blocks, controls)
        info.update(N=n, seed=seed, wall_before_validation_s=time.perf_counter() - t0)
        lab.write_json(folder / 'selection.json', info)
        print(f'N={n} seed={seed} training selection sealed {info["selected"]}, {time.perf_counter()-t0:.1f} s', flush=True)
        return info
    except Exception:
        lab.write_json(folder / 'failure.json', {'traceback': traceback.format_exc(), 'wall_s': time.perf_counter()-t0})
        raise


def schur_channels(blocks, controls, xy):
    words = [b.word(controls, xy) for b in blocks]
    wanted = [np.eye(b.d, dtype=complex) for b in blocks]
    actual = [np.broadcast_to(i, word.shape).copy() for i, word in zip(wanted, words)]
    mult = [b.mult for b in blocks]
    result = {}
    for depth, phi in enumerate(pilot.PHASES, 1):
        for j, block in enumerate(blocks):
            phase = np.exp(-1j * phi * block.difference)
            wanted[j] = (phase * block.target) @ wanted[j]
            actual[j] = (phase * words[j]) @ actual[j]
        if depth in DEPTHS:
            result[depth] = np.array([pilot.metrics(wanted, [v[i] for v in actual], mult)[0]
                                      for i in range(len(xy))])
    return result


def validate_one_pair(args):
    n, seed, folder, xy = args
    fixed = lab.read_json(folder / 'selection.json')
    blocks = spin_blocks(n)
    phases = lab.sequences()['original']
    points, maxima = [], []
    with np.load(folder / 'controls.npz', allow_pickle=False) as controls:
        for arm, selected in fixed['selected'].items():
            if selected is None:
                continue
            p = controls[selected]
            block_scores = schur_channels(blocks, p, xy)
            physical = lab.follow(n, lab.words(n, p, xy), phases, DEPTHS)
            for depth in DEPTHS:
                D = physical[depth][0]
                diff = np.abs(D - block_scores[depth])
                if not np.all(np.isfinite(D)) or np.max(diff) > AUDIT_TOL:
                    raise ValueError(f'channel discrepancy N={n} seed={seed} {arm} M={depth}: {np.max(diff):g}')
                context = {'N': n, 'seed': seed, 'arm': arm, 'selected': selected, 'depth': depth}
                for k, ((x, y), d) in enumerate(zip(xy, D)):
                    points.append({**context, 'sample': k, 'location': 'interior' if k < 64 else 'edge',
                                   'x': float(x), 'y': float(y), 'D_tensor': float(d),
                                   'D_schur': float(block_scores[depth][k]), 'abs_difference': float(diff[k])})
                k = int(np.argmax(D))
                maxima.append({**context, 'max_D': float(D[k]), 'worst_sample': k,
                               'x': float(xy[k, 0]), 'y': float(xy[k, 1]),
                               'max_interior_D': float(np.max(D[:64])), 'max_edge_D': float(np.max(D[64:])),
                               'below_1pct': bool(D[k] < .01), 'max_schur_tensor_discrepancy': float(np.max(diff))})
    return points, maxima


def squared_opnorm(a):
    return np.linalg.svd(a, compute_uv=False)[..., 0] ** 2


def development_leakage(n, method, file):
    """Follow four input columns in the lab tensor basis; separately follow them in the one spin block."""
    xy = np.array(lab.grid(5))
    p = np.load(file, allow_pickle=False)
    words = lab.words(n, p, xy)
    magnetic, _ = lab.z_diagonals(n)
    spin_basis = np.column_stack([lab.dicke(n, w) for w in range(n + 1)])
    weights = np.arange(n, n - 4, -1)
    code = spin_basis[:, weights]
    spectator = spin_basis[:, :n - 3]
    block = max(spin_blocks(n), key=lambda b: b.d)
    block_word = block.word(p, xy)
    columns = np.broadcast_to(code, (len(xy), *code.shape)).copy()
    block_columns = np.eye(n + 1, dtype=complex)[:, weights]
    rows, maxima = [], []
    for depth, phi in enumerate(lab.sequences()['original'], 1):
        z = np.exp(-1j * phi * magnetic)
        columns = ((words * z[:, None]) * z.conj()[None, :]) @ columns
        block_columns = (block_word * np.exp(-1j * pilot.PHASES[depth - 1] * block.difference)) @ block_columns
        if depth not in LEAK_DEPTHS:
            continue
        total = squared_opnorm(columns - code @ (code.conj().T @ columns))
        within = squared_opnorm(spectator.conj().T @ columns)
        outside = squared_opnorm(columns - spin_basis @ (spin_basis.conj().T @ columns))
        schur = squared_opnorm(block_columns[:, :n - 3, :])
        diff = np.abs(total - schur)
        if (not np.all(np.isfinite(total)) or np.max(total) > 1 + AUDIT_TOL or
                np.max(np.abs(within - total)) > AUDIT_TOL or np.max(diff) > AUDIT_TOL or np.max(outside) > 1e-20):
            raise ValueError(f'code leakage discrepancy at N={n} {method} depth={depth}')
        context = {'N': n, 'development_seed': 6111800 + n, 'method': method, 'depth': depth}
        for i, (x, y) in enumerate(xy):
            rows.append({**context, 'sample': i, 'x': x, 'y': y, 'leakage': float(total[i]),
                         'within_sym_outside_code': float(within[i]), 'outside_symmetric': float(outside[i]),
                         'schur_within_sym': float(schur[i]), 'abs_tensor_schur_difference': float(diff[i])})
        k = int(np.argmax(total))
        maxima.append({**context, 'max_leakage': float(total[k]), 'worst_sample': k,
                       'x': xy[k, 0], 'y': xy[k, 1], 'max_within_sym_outside_code': float(np.max(within)),
                       'max_outside_symmetric': float(np.max(outside)), 'max_tensor_schur_difference': float(np.max(diff))})
    return rows, maxima


def run_many(fn, inputs, workers):
    if workers == 1:
        return [fn(item) for item in inputs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, inputs))


def budgets(folder, selected):
    n, seed = selected['N'], selected['seed']
    fits = lab.read_json(folder / 'fit_histories.json')
    nblocks = len(pilot.schur_representations(n))
    result = []
    for arm in ARMS:
        nominal, finite, restore = fits['nominal'], fits[f'{arm}_finite'], fits[f'{arm}_restore']
        corr = fits['correction'] if arm == 'seeded' else None
        correction_units = 5 * corr['evaluations'] if corr else 0
        finite_units = 17 * finite['evaluations']
        after_shared = correction_units + finite_units + restore['evaluations']
        if (nominal['evaluations'] > 100 or (corr and corr['evaluations'] > 85) or
                finite['evaluations'] > (55 if corr else 80) or restore['evaluations'] > 8 or after_shared > 1368):
            raise ValueError('stage exceeded its registered compute ceiling')
        restored = selected['candidates'][arm][f'{arm}_restored']
        result.append({'N': n, 'seed': seed, 'arm': arm, 'shared_nominal_actual': nominal['evaluations'],
                       'shared_nominal_cap': 100, 'shared_nominal_stop': nominal['reason'],
                       'correction_5point_evals': corr['evaluations'] if corr else 0,
                       'correction_5point_cap': 85 if corr else 0,
                       'correction_stop': corr['reason'] if corr else '',
                       'finite_17point_evals': finite['evaluations'], 'finite_17point_cap': 55 if corr else 80,
                       'finite_stop': finite['reason'], 'actual_search_offset_jac': correction_units + finite_units,
                       'search_offset_jac_cap': 1360, 'restore_1point_evals': restore['evaluations'],
                       'restore_1point_cap': 8, 'restore_stop': restore['reason'], 'restore_eligible': restored['eligible'],
                       'restore_nominal_D': restored['nominal_D'], 'after_shared_actual_offset_jac': after_shared,
                       'after_shared_offset_jac_cap': 1368, 'distinct_spin_blocks': nblocks,
                       'actual_after_shared_block_offset_jac': nblocks * after_shared,
                       'training_score_offsets_no_jacobian': 68,
                       'shared_nominal_solver_s': nominal['wall_s'], 'correction_solver_s': corr['wall_s'] if corr else 0.,
                       'finite_solver_s': finite['wall_s'], 'restore_solver_s': restore['wall_s'],
                       'total_post_shared_solver_s': (corr['wall_s'] if corr else 0.) + finite['wall_s'] + restore['wall_s']})
    return result


def candidate_tables(frozen):
    summary, points = [], []
    for s in frozen:
        for arm in ARMS:
            for position in POSITIONS:
                name = label(arm, position)
                info = s['candidates'][arm][name]
                ctx = {'N': s['N'], 'seed': s['seed'], 'arm': arm, 'position': position, 'control': name}
                summary.append({**ctx, 'selected': s['selected'][arm] == name, 'eligible': info['eligible'],
                                'nominal_D': info['nominal_D'], 'max_train_D': info['max_train_D']})
                for i, (d, (x, y)) in enumerate(zip(info['all_train_D'], TRAIN)):
                    points.append({**ctx, 'sample': i, 'x': x, 'y': y, 'D': d})
    return summary, points


def comparisons(frozen, maxima):
    m32 = {(r['N'], r['seed'], r['arm']): r['max_D'] for r in maxima if r['depth'] == 32}
    rows = []
    for s in frozen:
        n, seed = s['N'], s['seed']
        direct = m32.get((n, seed, 'direct'))
        seeded = m32.get((n, seed, 'seeded'))
        both = direct is not None and seeded is not None
        delta = direct - seeded if both else None
        if not both:
            outcome = 'unqualified'
        elif abs(delta) <= AUDIT_TOL:
            outcome = 'tie'
        else:
            outcome = 'seeded_lower' if delta > 0 else 'direct_lower'
        rows.append({'N': n, 'seed': seed, 'direct_control': s['selected']['direct'],
                     'seeded_control': s['selected']['seeded'], 'direct_max_D32': direct,
                     'seeded_max_D32': seeded, 'direct_minus_seeded': delta, 'outcome': outcome,
                     'direct_below_1pct': direct < .01 if direct is not None else None,
                     'seeded_below_1pct': seeded < .01 if seeded is not None else None,
                     'direct_restoration_qualified': s['candidates']['direct']['direct_restored']['eligible'],
                     'seeded_restoration_qualified': s['candidates']['seeded']['seeded_restored']['eligible'],
                     'seeded_midpoint_first_order_qualified': s['seeded_midpoint_first_order_qualified'],
                     'seeded_midpoint_generator_R': s['seeded_midpoint_generator_R_T'][0],
                     'seeded_midpoint_generator_T': s['seeded_midpoint_generator_R_T'][1]})
    return rows


def archived_bindings(source):
    manifest = lab.read_json(source / 'sha256.json')
    actually_used = [f'n{n}_seed{seed}/{file}' for n, seed in PAIRS
                     for file in ('controls.npz', 'selection.json', 'fit_histories.json')]
    actually_used.extend(('heldout_points.csv', 'development_leakage_points.csv', 'inputs.json'))
    hashes = {name: lab.sha(source / name) for name in actually_used}
    for name, binding in hashes.items():
        if manifest.get(name) != binding:
            raise ValueError(f'archive input binding mismatch: {name}')
    hashes['sha256.json'] = lab.sha(source / 'sha256.json')
    return hashes


def replay_selection(n, seed, source):
    folder = source / f'n{n}_seed{seed}'
    registered = lab.read_json(folder / 'selection.json')
    if (registered['N'], registered['seed']) != (n, seed):
        raise ValueError('wrong archived start')
    with np.load(folder / 'controls.npz', allow_pickle=False) as p:
        computed = rank_controls(n, spin_blocks(n), p)
    if (computed['selected'] != registered['selected'] or
            computed['seeded_midpoint_first_order_qualified'] != registered['seeded_midpoint_first_order_qualified']):
        raise ValueError(f'archive train-only selection / derivative qualification mismatch N={n} seed={seed}')
    difference = max(float(np.max(np.abs(np.array(c['all_train_D']) - registered['candidates'][arm][name]['all_train_D'])))
                     for arm, items in computed['candidates'].items() for name, c in items.items())
    if difference > AUDIT_TOL:
        raise ValueError(f'archive training distance mismatch {difference:g}')
    computed.update(N=n, seed=seed, max_archived_training_discrepancy=difference)
    return computed


def verify_table(path, rows, ids, score_fields):
    prior = lab.records(path)
    if len(prior) != len(rows):
        raise ValueError(f'archive table count differs for {path.name}')
    biggest = 0.
    for a, b in zip(prior, rows):
        if any(a[k] != str(b[k]) for k in ids) or any(abs(float(a[k]) - float(b[k])) > 1e-15 for k in ('x', 'y')):
            raise ValueError(f'archive table labels or positions differ for {path.name}')
        biggest = max(biggest, *(abs(float(a[k]) - float(b[k])) for k in score_fields))
    if biggest > AUDIT_TOL:
        raise ValueError(f'archive table numerical difference for {path.name}: {biggest:g}')
    return {'rows': len(rows), 'max_absolute_difference': biggest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path, help='new directory with an existing parent')
    parser.add_argument('--scores-from', type=Path, help='saved full-range run to replay using its frozen controls')
    parser.add_argument('--workers', type=int, default=4, help='concurrent fitting / channel replay processes (default 4)')
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error('--workers must be between 1 and 8')
    out = args.out.resolve()
    source = args.scores_from.resolve() if args.scores_from else None
    out.mkdir(exist_ok=False)
    t0 = time.perf_counter()
    try:
        if lab.sha(PROTOCOL) != PROTOCOL_HASH:
            raise ValueError('frozen protocol SHA-256 changed')
        xy = test_offsets()
        hashes = input_bindings()
        archive = archived_bindings(source) if source else {}
        if source and lab.read_json(source / 'inputs.json')['hashes'] != hashes:
            raise ValueError('replay source differs from the archived code, protocol, or controls')
        metadata = {'utc_start': datetime.now(timezone.utc).isoformat(), 'run_id': RUN_ID,
                    'mode': 'scores_from_saved_controls' if source else 'fit_and_validate',
                    'python': sys.version, 'interpreter': sys.executable, 'platform': platform.platform(),
                    'numpy': np.__version__, 'scipy': scipy.__version__, 'precision': 'float64 / complex128',
                    'blas_thread_settings': {v: os.environ[v] for v in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                                                         'OMP_NUM_THREADS', 'BLIS_NUM_THREADS')},
                    'workers': args.workers, 'hashes': hashes, 'score_replay_archive_hashes': archive,
                    'new_pairs': PAIRS, 'heldout_interior_seed': 6112700, 'train': TRAIN.tolist(),
                    'test': xy.tolist(), 'near_zero_correction_stencil': pilot.JET.tolist(),
                    'azimuths': pilot.PHASES.tolist(), 'heldout_depths': DEPTHS,
                    'development_grid': lab.grid(5), 'leakage_depths': LEAK_DEPTHS,
                    'physical_clock_by_width': {n: physical_clock(n) for n in (4, 5)},
                    'signed_controls': True, 'gain_constant_per_circuit': True,
                    'ranking_offsets_per_candidate': 17, 'nominal_D_threshold': QUALIFY,
                    'fit_tolerance': 1e-13, 'post_nominal_search_offset_cap_per_arm': 1360,
                    'nominal_restoration_cap': 8, 'tie_and_agreement_tolerance': AUDIT_TOL,
                    'outside_symmetric_leakage_tolerance': 1e-20}
        lab.write_json(out / 'inputs.json', metadata)
        control_root = source or out
        if source:
            selections = [replay_selection(n, seed, source) for n, seed in PAIRS]
        else:
            selections = run_many(fit_one_pair, [(n, seed, out) for n, seed in PAIRS], args.workers)
        # All choices are sealed before generating a single held-out channel score.
        frozen = {f'n{s["N"]}_seed{s["seed"]}': s['selected'] for s in selections}
        lab.write_json(out / 'training_selections.json', frozen)
        candidate_rows, training_rows = candidate_tables(selections)
        write_csv(out / 'candidate_scores.csv', candidate_rows)
        write_csv(out / 'training_points.csv', training_rows)
        budget_rows = [row for s in selections for row in budgets(control_root / f'n{s["N"]}_seed{s["seed"]}', s)]
        write_csv(out / 'solver_budgets.csv', budget_rows)
        print('All training choices sealed. Replaying held-out lab and Schur channels.', flush=True)
        checks = run_many(validate_one_pair,
                          [(n, seed, control_root / f'n{n}_seed{seed}', xy) for n, seed in PAIRS], args.workers)
        heldout = [row for pair_rows, _ in checks for row in pair_rows]
        maxima = [row for _, pair_maxima in checks for row in pair_maxima]
        write_csv(out / 'heldout_points.csv', heldout,
                  ('N', 'seed', 'arm', 'selected', 'depth', 'sample', 'location', 'x', 'y', 'D_tensor', 'D_schur', 'abs_difference'))
        write_csv(out / 'heldout_maxima.csv', maxima,
                  ('N', 'seed', 'arm', 'selected', 'depth', 'max_D', 'worst_sample', 'x', 'y', 'max_interior_D',
                   'max_edge_D', 'below_1pct', 'max_schur_tensor_discrepancy'))
        paired = comparisons(selections, maxima)
        write_csv(out / 'paired_results.csv', paired)
        leak_points, leak_maxima = [], []
        for n in (4, 5):
            for method, path in development_files(n).items():
                points, worst = development_leakage(n, method, path)
                leak_points.extend(points)
                leak_maxima.extend(worst)
        write_csv(out / 'development_leakage_points.csv', leak_points)
        write_csv(out / 'development_leakage_maxima.csv', leak_maxima)
        eligible_arms = sum(v is not None for s in selections for v in s['selected'].values())
        if (len(heldout) != eligible_arms * 80 * 6 or len(leak_points) != 800 or len(training_rows) != 1088 or
                len(paired) != 8 or len(budget_rows) != 16 or len(leak_maxima) != 32):
            raise ValueError('incomplete output coverage')
        if source:
            replay = {'training_rankings': len(selections), 'max_training_score_difference': max(
                s['max_archived_training_discrepancy'] for s in selections),
                'heldout': verify_table(source / 'heldout_points.csv', heldout, ('N', 'seed', 'arm', 'selected', 'depth', 'sample'),
                                        ('D_tensor', 'D_schur')),
                'leakage': verify_table(source / 'development_leakage_points.csv', leak_points,
                                        ('N', 'development_seed', 'method', 'depth', 'sample'),
                                        ('leakage', 'within_sym_outside_code', 'outside_symmetric', 'schur_within_sym'))}
            lab.write_json(out / 'replay_check.json', replay)
        counters = {n: {name: sum(r['N'] == n and r['outcome'] == name for r in paired)
                        for name in ('seeded_lower', 'direct_lower', 'tie', 'unqualified')} for n in (4, 5)}
        report = {'status': 'passed_numerical_checks', 'mode': metadata['mode'], 'paired_outcomes': counters,
                  'counts': {'pairs': len(paired), 'qualified_arms': eligible_arms, 'heldout_rows': len(heldout),
                             'training_rows': len(training_rows), 'development_leakage_rows': len(leak_points),
                             'failed_restorations': sum(not row['restore_eligible'] for row in budget_rows),
                             'selected_nominal_fallbacks': sum(v == 'nominal' for s in selections for v in s['selected'].values()),
                             'seeded_midpoints_first_order_qualified': sum(s['seeded_midpoint_first_order_qualified'] for s in selections)},
                  'maximum_heldout_tensor_schur_difference': max((r['max_schur_tensor_discrepancy'] for r in maxima), default=None),
                  'maximum_leakage_tensor_schur_difference': max(r['max_tensor_schur_difference'] for r in leak_maxima),
                  'maximum_outside_symmetric_probability': max(r['max_outside_symmetric'] for r in leak_maxima),
                  'paired_results': paired, 'development_depth32_leakage': [r for r in leak_maxima if r['depth'] == 32],
                  'scope': 'Local finite-budget search; sample maxima on 80 held-out points, 4 new starts per width, one circuit. Development leakage uses 25 archived-design positions and one original seed per width.',
                  'utc_finish': datetime.now(timezone.utc).isoformat(), 'wall_s': time.perf_counter() - t0}
        lab.write_json(out / 'summary.json', report)
        print(f'Passed: {len(heldout)} held-out channel scores, {len(leak_points)} development leakage scores; {counters}', flush=True)
    except BaseException:
        lab.write_json(out / 'failure.json', {'traceback': traceback.format_exc(), 'wall_s': time.perf_counter()-t0})
        raise
    finally:
        lab.write_json(out / 'sha256.json', {f.relative_to(out).as_posix(): lab.sha(f) for f in sorted(out.rglob('*')) if f.is_file()})


if __name__ == '__main__':
    main()
