"""Independent physics replay for the Dicke-code gate comparison. No project imports."""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import scipy
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
DEVELOPMENT = RUNS / "npj_qi_register_depth_20261118_v1"
RESTORED = RUNS / "npj_qi_register_depth_projection_20261118_v1"
REPORT = RUNS / "npj_qi_register_depth_report_20261118_v1"
CONFIRM = RUNS / "npj_qi_confirm_starts_20261118_v1"
OPEN = RUNS / "npj_qi_dephasing_20261118_v1"
INTERVAL = RUNS / "cnot_tolerance_20261012_v1"
DEPTHS = (1, 2, 4, 8, 16, 32)
RATE = 1e-5


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def kron_power(matrix, n):
    result = matrix
    for _ in range(1, n):
        result = np.kron(result, matrix)
    return result


def dicke(n, k):
    return np.array([float(i.bit_count() == k) / math.sqrt(math.comb(n, k))
                     for i in range(1 << n)], complex)


def target(n, phase=math.pi / 4):
    code = np.stack([dicke(n, n - j) for j in range(4)], axis=1)
    cnot = np.eye(4)[:, [0, 1, 3, 2]]
    return (np.eye(1 << n)
            + code @ (np.exp(1j * phase) * cnot - np.eye(4)) @ code.conj().T)


def z_diagonals(n):
    signs = np.array([[1 - 2 * ((s >> j) & 1) for j in range(n)] for s in range(1 << n)])
    zz = sum(signs[:, i] * signs[:, j] for i in range(n) for j in range(i))
    return signs.sum(axis=1) / 2, zz


def single_rotation(angle, azimuth):
    axis = np.array([[0, np.exp(-1j * azimuth)], [np.exp(1j * azimuth), 0]])
    return np.eye(2) * np.cos(angle / 2) - 1j * np.sin(angle / 2) * axis


def words(n, controls, offsets):
    """Apply physical ZZ phases and tensor products of one-qubit rotations."""
    _, zz = z_diagonals(n)
    angles, axes, twists = np.split(controls, 3)
    x, y = np.asarray(offsets).T
    gains, indices = np.unique(x, return_inverse=True)
    result = np.broadcast_to(np.eye(1 << n, dtype=complex), (len(x), 1 << n, 1 << n)).copy()
    for angle, azimuth, twist in zip(angles, axes, twists - np.mean(twists)):
        result *= np.exp(-1j * (1 + y[:, None]) * twist * zz)[:, :, None]
        rotations = np.stack([kron_power(single_rotation((1 + gain) * angle, azimuth), n)
                              for gain in gains])
        result = rotations[indices] @ result
    return result


def channels(relative):
    """Use longest chord if the spectrum fits inside a semicircle, else one."""
    eig = np.linalg.eigvals(relative)
    eig /= np.abs(eig)
    orientation = (eig[..., :, None].conj() * eig[..., None, :]).imag
    semicircle = np.any(np.all(orientation >= -8e-14, axis=-1), axis=-1)
    chord = np.max(np.abs(eig[..., :, None] - eig[..., None, :]), axis=(-1, -2)) / 2
    d = relative.shape[-1]
    trace = np.trace(relative, axis1=-2, axis2=-1)
    avg = np.maximum(0., (d * d - np.abs(trace) ** 2) / (d * (d + 1)))
    return np.where(semicircle, chord, 1.), avg


def sequences():
    j = np.arange(32)
    seq = {"original": np.pi * (np.sqrt(2.) * np.square(j) + np.sqrt(3.) * j) % (2 * np.pi)}
    for seed in (99181, 99182, 99183):
        seq[str(seed)] = np.r_[0., np.random.Generator(np.random.PCG64(seed)).uniform(-np.pi, np.pi, 31)]
    return seq


def follow(n, gates, phases, depths):
    mag, _ = z_diagonals(n)
    ideal, actual = np.eye(1 << n), np.broadcast_to(np.eye(1 << n), gates.shape)
    wanted = target(n)
    values = {}
    for k, phi in enumerate(phases, 1):
        diag = np.exp(-1j * phi * mag)
        mask = diag[:, None] * diag.conj()
        ideal = (wanted * mask) @ ideal
        actual = (gates * mask) @ actual
        if k in depths:
            values[k] = channels(ideal.conj().T @ actual)
    return values


def grid(side):
    return [(x, y) for x in np.linspace(-.01, .01, side) for y in np.linspace(-.01, .01, side)]


def offset_key(x, y):
    return round(float(x) * 1e6), round(float(y) * 1e6)


def first_order(n, p):
    h = 1e-4
    norms = []
    for axis in range(2):
        points = np.zeros((4, 2))
        points[:, axis] = np.array([-2, -1, 1, 2]) * h
        um2, um1, up1, up2 = words(n, p, points)
        derivative = (um2 - 8 * um1 + 8 * up1 - up2) / (12 * h)
        norms.append(float(np.linalg.norm(derivative, ord=2)))
    return norms


def coherent_replay():
    sources = {}
    for n in (4, 5):
        start = read_json(DEVELOPMENT / f"n{n}_summary.json")
        for method, path in (("nominal", DEVELOPMENT / f"n{n}_nominal.npy"),
                             ("corrected", DEVELOPMENT / f"n{n}_corrected.npy"),
                             ("direct", RESTORED / f"n{n}_nominal_restored.npy"),
                             ("refined", RESTORED / f"n{n}_corrected_restored.npy"),
                             ("relaxed", DEVELOPMENT / f"n{n}_{start['noise_aware_source']}.npy")):
            sources[n, 0, method] = np.load(path)
    paired = read_json(CONFIRM / "summary.json")["pairs"]
    for pair in paired:
        n, seed = pair["N"], pair["seed"]
        with np.load(CONFIRM / f"n{n}_seed{seed}/controls.npz") as controls:
            for method in ("direct", "seeded"):
                sources[n, seed, method] = controls[pair["validation"][method]["selected"]]

    rows = records(REPORT / "all_validation_points.csv") + records(CONFIRM / "validation_25.csv")
    reference = {(int(r["N"]), int(r.get("seed", 0)), r.get("method", r.get("arm")), r["circuit"],
                  int(r["depth"]), *offset_key(r["x"], r["y"])): (float(r["D"]), float(r["avg_infidelity"]))
                 for r in rows}
    dense_rows = records(CONFIRM / "validation_81.csv")
    dense = {(int(r["N"]), int(r["seed"]), r["arm"], *offset_key(r["x"], r["y"])): float(r["D32"])
             for r in dense_rows}
    if len(reference) != 15600 or len(dense) != 1296:
        raise ValueError("incomplete grids or duplicate records")
    times = {}
    maxima, development, qualification = {}, {}, {}
    grids_checked = dense_checked = 0
    diff_D = diff_avg = diff_dense = 0.
    stable_sequences = sequences()
    for (n, seed, method), p in sources.items():
        offsets = grid(9 if seed else 5)
        gates = words(n, p, offsets)
        five = [i for i, xy in enumerate(offsets) if offset_key(*xy) in {offset_key(*q) for q in grid(5)}]
        for name, phase_sequence in stable_sequences.items():
            result = follow(n, gates[five], phase_sequence, DEPTHS)
            for k, (dist, avg) in result.items():
                expected = np.array([reference[n, seed, method, name, k, *offset_key(*offsets[i])]
                                     for i in five])
                diff_D = max(diff_D, float(np.max(np.abs(dist - expected[:, 0]))))
                diff_avg = max(diff_avg, float(np.max(np.abs(avg - expected[:, 1]))))
                grids_checked += len(five)
                if name == "original" and k == 32:
                    maxima[n, seed, method] = float(dist.max())
                if name == "original" and seed == 0:
                    development.setdefault(str(n), {}).setdefault(method, {})[k] = float(dist.max())
        if seed:
            dist, _ = follow(n, gates, stable_sequences["original"], (32,))[32]
            expected = [dense[n, seed, method, *offset_key(*xy)] for xy in offsets]
            diff_dense = max(diff_dense, float(np.max(np.abs(dist - expected))))
            dense_checked += len(offsets)
        elif method == "corrected":
            qualification[str(n)] = first_order(n, p)
    for pair in paired:
        n, seed, calls = pair["N"], pair["seed"], pair["actual_calls"]
        times[seed] = dict(N=n, direct=maxima[n, seed, "direct"], seeded=maxima[n, seed, "seeded"],
                           ratio=maxima[n, seed, "direct"] / maxima[n, seed, "seeded"],
                           five_offset_calls=[calls["direct"], calls["raw"] + calls["seeded"]],
                           restoration_calls=[calls["direct_restore"], calls["seeded_restore"]])
    if max(diff_D, diff_avg, diff_dense) > 2e-8 or grids_checked != len(rows) or dense_checked != len(dense_rows):
        raise ValueError(f"coherent physical replay mismatch {diff_D}, {diff_avg}, {diff_dense}")
    return dict(five_by_five_rows=grids_checked, nine_by_nine_rows=dense_checked,
                max_abs_D_difference=diff_D, max_abs_avg_infidelity_difference=diff_avg,
                max_abs_D_difference_dense=diff_dense, development_original=development,
                development_raw_first_derivative_opnorms=qualification, confirmation_depth32_original=times)


def phase_witness():
    p0 = np.eye(8, dtype=complex)[:, 0]
    prepared = np.kron(single_rotation(np.pi, 0), np.eye(4)) @ p0
    error = 0.
    examples = {}
    for phi in np.r_[0., np.pi / 4, np.linspace(-np.pi, np.pi, 25)]:
        psi = target(3, phi) @ prepared
        readings = []
        for beta in (-np.pi / 4, 3 * np.pi / 4):
            r = np.kron(single_rotation(np.pi / 2, beta), np.eye(4))
            p = float(abs((r @ psi)[0]) ** 2)
            predicted = 7 / 18 - 2 * np.sin(phi + beta) / (3 * np.sqrt(3))
            error = max(error, float(abs(p - predicted)))
            readings.append(p)
        if phi in (0., np.pi / 4):
            examples[str(float(phi))] = {"p_plus": readings[0], "p_minus": readings[1]}
    at_min, _ = channels(target(3, 0).conj().T @ target(3))
    if error > 2e-14 or abs(at_min - math.sin(math.pi / 8)) > 2e-14:
        raise ValueError("three-qubit witness or equality representative incorrect")
    return dict(maximum_probability_difference=error, examples=examples,
                complete_distance_at_equality=float(at_min))


def slot(area, cap):
    s = abs(area) / cap
    def constraint(t):
        return t - s * (1 + math.hypot(1, 2 * math.pi * .02 / t))
    return max(.04, brentq(constraint, s, 3 * s + .1, xtol=1e-14))


def interaction_pulse(angle, axis, tau, gamma, degree):
    """Quadrature of the first two time-ordered dissipators in the rotating frame."""
    z = np.diag([1., -1.])
    identity = np.eye(4)
    nodes, weights = np.polynomial.legendre.leggauss(degree)
    nodes, weights = (nodes + 1) / 2, weights / 2
    def dissipator(t):
        integrated = t - math.sin(2 * math.pi * t) / (2 * math.pi)
        u = single_rotation(angle * integrated, axis)
        zi = u.conj().T @ z @ u
        return (gamma * tau / 2) * (np.kron(zi, zi.conj()) - identity)
    interaction = identity.astype(complex)
    if gamma:
        for t, w in zip(nodes, weights):
            outer = dissipator(t)
            inner = sum(w1 * dissipator(t * t1) for t1, w1 in zip(nodes, weights))
            interaction += w * (outer + t * outer @ inner)
    u = single_rotation(angle, axis)
    return np.kron(u, u.conj()) @ interaction


def density_check(method, p, x, y, gamma, expected):
    """Propagate the full 16x16 density. Every operation acts in the computational basis."""
    n, d = 4, 16
    mag, zz = z_diagonals(n)
    angles, axes, twists = np.split(p, 3)
    twists = twists - twists.mean()
    rot_time, zz_time, switch = slot(np.pi, 10), slot(1.2, 1), .001
    hamming = np.array([[(i ^ j).bit_count() for j in range(d)] for i in range(d)])
    decay_switch = np.exp(-gamma * switch * hamming)
    zz_maps = [np.exp(-gamma * (zz_time + switch) * hamming
                      - 1j * (1 + y) * c * (zz[:, None] - zz)) for c in twists]
    # Convert between global rho[i,j] order and interleaved site indices (i1,j1,...).
    order = np.arange(d ** 2).reshape([2] * 8).transpose(0, 4, 1, 5, 2, 6, 3, 7).ravel()
    inv = np.argsort(order)
    channels4 = []
    for a, b in zip(angles, axes):
        m = interaction_pulse((1 + x) * a, b, rot_time, gamma, degree=12)
        channels4.append(kron_power(m, n)[inv[:, None], inv])
    doubled = interaction_pulse((1 + x) * angles[0], axes[0], rot_time, gamma, degree=24)
    one = interaction_pulse((1 + x) * angles[0], axes[0], rot_time, gamma, degree=12)
    quadrature_check = float(np.linalg.norm(doubled - one, ord=2))
    singlet = (np.eye(d)[:, 4] - np.eye(d)[:, 8]) / np.sqrt(2)
    ideal = (dicke(n, 2) + singlet) / np.sqrt(2)
    rho = np.outer(ideal, ideal.conj())
    goal = target(4)
    observed = {}
    for depth, phase in enumerate(sequences()["original"], 1):
        z = np.exp(-1j * phase * mag)
        mask = z[:, None] * z.conj()
        ideal = z * (goal @ (z.conj() * ideal))
        rho *= mask.conj()
        for j, (twist, rot) in enumerate(zip(zz_maps, channels4)):
            rho = (rot @ (twist * rho).ravel()).reshape(d, d)
            if j + 1 < len(angles):
                rho *= decay_switch
        rho *= mask
        if depth in DEPTHS:
            loss = float(1 - np.vdot(ideal, rho @ ideal).real)
            observed[depth] = loss
        rho *= decay_switch
    maxdiff = max(abs(value - expected[method, gamma, depth, *offset_key(x, y)])
                  for depth, value in observed.items())
    if maxdiff > 2e-8 or quadrature_check > 1e-10:
        raise ValueError(f"GKSL replay mismatch {method}, {gamma}, {maxdiff}, {quadrature_check}")
    return dict(x=x, y=y, gamma_t0=gamma, max_difference=maxdiff, projector_loss=observed,
                one_site_quadrature_12_to_24_opnorm=quadrature_check,
                one_site_dyson_tail_upper=math.exp(gamma * rot_time) * (gamma * rot_time) ** 3 / 6)


def dephasing_replay():
    rates = (0., RATE)
    expected = {(r["method"], float(r["gamma_t0"]), int(r["depth"]), *offset_key(r["x"], r["y"])):
                float(r["probability_loss"]) for r in records(OPEN / "readout.csv")}
    replays = {}
    for name, source, point in (("direct", "nominal", (.01, -.01)), ("refined", "corrected", (.01, .01))):
        controls = np.load(RESTORED / f"n4_{source}_restored.npy")
        replays[name] = [density_check(name, controls, *point, gamma, expected) for gamma in rates]
    raw = records(OPEN / "channel_brackets.csv")
    gt = {4: 48, 5: 72}
    difference = 0.
    for row in raw:
        n, depth = int(row["N"]), int(row["depth"])
        c = float(row["coherent_grid_D"])
        t = depth * gt[n] * (slot(np.pi, 10) + slot(1.2, 1) + .002) - .001
        p = -math.expm1(-n * RATE * t / 2)
        vals = [abs(a - float(row[k])) for a, k in [(t, "time_t0"), (p, "p_any_jump"),
                (max(0., c - p), "open_grid_D_lower"), (min(1., c + p), "open_grid_D_upper")]]
        difference = max(difference, *vals)
    peak = max(value for (method, g, depth, _, _), value in expected.items()
               if method == "refined" and g == RATE and depth == 32)
    upper = next(float(r["open_grid_D_upper"]) for r in raw
                 if r["N"] == "4" and r["depth"] == "32" and r["method"] == "refined")
    if difference > 2e-8 or len(raw) != 48 or len(expected) != 600:
        raise ValueError(f"open channel bracket mismatch {difference}")
    return dict(independent_readouts=replays, bracket_rows=len(raw),
                greatest_absolute_clock_or_bracket_difference=difference,
                grid_refined_N4_M32=[peak, upper])


def certificate_arithmetic():
    archive = read_json(INTERVAL / "interval_v2/RESULTS.json")["results"]
    family = read_json(INTERVAL / "primary_audit/ACCEPTANCE.json")["continuous_family"]
    worst = 0.
    endpoint = {}
    for name, data in archive.items():
        approx = sum(c * .01 ** k for k, c in enumerate(data["degree_sum_bounds"]))
        approx += (.01 * data["action_sum_upper"]) ** 9 / math.factorial(9)
        endpoint[name] = approx
        worst = max(worst, abs(approx - data["bounds"]["8"]["0.01"]))
    edges = []
    for e in family["segments"]:
        correction = (1.01 * e["weighted_l1_edge_upper"]) ** 2 / 8
        edges.append(max(endpoint[e["left"]], endpoint[e["right"]]) + correction)
    common = family["common_actuator_schedules"]["0.0037"]
    duration = sum(common["durations_t0"]) + .089
    total = max(edges) + 1.01 * .0037 + 3 * RATE * duration / 2
    if worst > 1e-12 or abs(total - family["uniform_halfdiamond_upper_with_dephasing"]) > 1e-12:
        raise ValueError(f"interval-envelope aggregation mismatch {worst}, {total}")
    return dict(archived_coefficient_envelopes_checked=len(archive),
                max_disagreement_at_one_percent=worst, schedule_sum_t0=duration,
                path_upper_reaggregated=max(edges), combined_upper_reaggregated=total,
                scope="Arithmetic on archived upper bounds; no regeneration of interval coefficient balls.")


def inputs():
    files = [Path(__file__).resolve(), ROOT / "docs/NPJ_QI_CONFIRMATION_PROTOCOL_20261118_v1.md",
             ROOT / "docs/NPJ_QI_REGISTER_DEPTH_PROTOCOL_20261118_v1.md",
             ROOT / "docs/NPJ_QI_REGISTER_DEPTH_PROTOCOL_20261118_v2.md",
             REPORT / "all_validation_points.csv", CONFIRM / "summary.json",
             CONFIRM / "validation_25.csv", CONFIRM / "validation_81.csv",
             OPEN / "readout.csv", OPEN / "channel_brackets.csv", INTERVAL / "interval_v2/RESULTS.json",
             INTERVAL / "primary_audit/ACCEPTANCE.json"]
    for n in (4, 5):
        files.extend([DEVELOPMENT / f"n{n}_summary.json", DEVELOPMENT / f"n{n}_nominal.npy",
                      DEVELOPMENT / f"n{n}_corrected.npy", DEVELOPMENT / f"n{n}_noise_from_corrected.npy",
                      RESTORED / f"n{n}_nominal_restored.npy", RESTORED / f"n{n}_corrected_restored.npy"])
    files.extend(sorted(CONFIRM.glob("n*_seed*/controls.npz")))
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    out = parser.parse_args().out.resolve()
    out.mkdir(exist_ok=False)
    start = time.perf_counter()
    try:
        write_json(out / "inputs.json", dict(utc=datetime.now(timezone.utc).isoformat(),
                   python=sys.version, interpreter=sys.executable, platform=platform.platform(),
                   numpy=np.__version__, scipy=scipy.__version__, precision="float64/complex128",
                   hashes=inputs()))
        result = dict(phase_witness=phase_witness(), coherent=coherent_replay(),
                      dephasing=dephasing_replay(), certificate=certificate_arithmetic())
        result.update(status="passed", elapsed_s=time.perf_counter() - start)
        write_json(out / "results.json", result)
    except BaseException:
        write_json(out / "failure.json", dict(traceback=traceback.format_exc(), elapsed_s=time.perf_counter()-start))
        raise
    finally:
        write_json(out / "sha256.json", {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()})


if __name__ == "__main__":
    main()
