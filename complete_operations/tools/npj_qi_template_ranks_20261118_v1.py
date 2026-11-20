"""Evaluate full and fixed-complement control Jacobians at two archived pulse words."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

for thread_setting in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[thread_setting] = "1"

import numpy as np
import scipy
from scipy.linalg import expm, helmert, svd


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/NPJ_QI_TEMPLATE_RANKS_20261118_v1.md"
SOURCE = ROOT / "runs/npj_qi_register_depth_20261118_v1"
LENGTHS = {4: 48, 5: 72}
INPUT_HASHES = {
    4: "7ff5a2720b3a8fdfc1350f5822213c78fe21ee03367b7e5a4bd7064d9d156446",
    5: "3d51043361221b87e659a67b3028531157ee339ee42ed25f49d5db11025c9657",
}
STEPS = (2e-4, 1e-4)
RANK_TOL = 1e-9
SENSITIVITY = (1e-6, 1e-8, RANK_TOL, 1e-10, 1e-12)
LOG_DEGREE = 8


def adjoint(matrix):
    return matrix.conj().swapaxes(-1, -2)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")


def control_chart(length):
    """Columns give orthonormal physical increments in (theta, phi, chi)."""
    chart = np.zeros((3 * length, 3 * length - 1))
    chart[: 2 * length, : 2 * length] = np.eye(2 * length)
    chart[2 * length :, 2 * length :] = helmert(length, full=False).T
    return chart


def hermitian_basis(dimension, index_order, code_size):
    basis, in_code = [], []
    for a in range(dimension):
        for b in range(a + 1, dimension):
            i, j = index_order[a], index_order[b]
            for entry in (1, 1j):
                matrix = np.zeros((dimension, dimension), complex)
                matrix[i, j] = entry / np.sqrt(2)
                matrix[j, i] = matrix[i, j].conjugate()
                basis.append(matrix)
                in_code.append(b < code_size)
    for k in range(1, dimension):
        diagonal = np.zeros(dimension)
        diagonal[index_order[:k]] = 1
        diagonal[index_order[k]] = -k
        basis.append(np.diag(diagonal / np.sqrt(k * (k + 1))))
        in_code.append(k < code_size)
    return np.asarray(basis, complex).reshape(-1, dimension, dimension), np.asarray(in_code, bool)


class Sector:
    def __init__(self, n, two_j):
        self.two_j = two_j
        self.d = d = two_j + 1
        self.magnetic = two_j / 2 - np.arange(d)
        self.mdiff = self.magnetic[:, None] - self.magnetic
        coupling = np.sqrt(np.arange(1, d) * np.arange(d - 1, 0, -1)) / 2
        self.jx = np.diag(coupling, 1) + np.diag(coupling, -1)
        self.jy = -1j * (np.diag(coupling, 1) - np.diag(coupling, -1))
        self.charges = 2 * self.magnetic**2 - n / 2
        self.eig, self.eigvec = np.linalg.eigh(self.jx)
        self.target = np.eye(d, dtype=complex)
        order = np.arange(d)
        if two_j == n:
            code = np.arange(n, n - 4, -1)
            order = np.concatenate((code, np.arange(d - 4)))
            cnot = np.exp(1j * np.pi / 4) * np.eye(4)[[0, 1, 3, 2]]
            self.target[np.ix_(code, code)] = cnot
        self.basis, self.logical = hermitian_basis(d, order, 4 if two_j == n else 0)

    def coordinates(self, hermitian):
        # Taking the real part is the orthogonal projection on Hermitian matrices.
        return np.einsum("aij,...ij->...a", self.basis.conj(), hermitian, optimize=True).real


def target_log(sector, endpoints, differential=None):
    """Principal i*log(V^dag U) and its derivative, via a bounded local series."""
    z = adjoint(sector.target) @ endpoints - np.eye(sector.d)
    size = float(np.max(np.linalg.norm(z, axis=(-2, -1))))
    if size >= 0.1:
        raise ValueError(f"endpoint outside the registered local log series, ||Z||_F={size}")
    tail = size ** (LOG_DEGREE + 1) / ((LOG_DEGREE + 1) * (1 - size))
    dz = None if differential is None else adjoint(sector.target) @ differential
    power = z.copy()
    total = power.copy()
    if dz is not None:
        dpower, dtotal = dz.copy(), dz.copy()
    for degree in range(2, LOG_DEGREE + 1):
        coeff = (-1) ** (degree + 1) / degree
        if dz is not None:
            dpower = dpower @ z + power @ dz
            dtotal += coeff * dpower
        power = power @ z
        total += coeff * power
    return 1j * total, (None if dz is None else 1j * dtotal), tail


def analytic(sector, physical, chart):
    """Product-rule derivatives of U, its gain derivatives, and its gain generators."""
    theta, phi, chi = physical
    length, d = len(theta), sector.d
    word = np.eye(d, dtype=complex)
    gain = np.zeros((2, d, d), complex)
    dword = np.zeros((3 * length, d, d), complex)
    dgain = np.zeros((2, 3 * length, d, d), complex)
    for slot, (angle, phase, twist) in enumerate(zip(theta, phi, chi)):
        axis = np.cos(phase) * sector.jx + np.sin(phase) * sector.jy
        rotation = expm(-1j * angle * axis)
        layer = rotation * np.exp(-1j * twist * sector.charges)[None, :]
        dt = -1j * axis @ layer
        dp = -1j * sector.mdiff * layer
        dc = -1j * layer * sector.charges
        local = np.stack((dt, dp, dc))
        gain_layer = np.stack((angle * dt, twist * dc))
        dt_dc = -1j * dt * sector.charges
        mixed = np.stack((
            np.stack((dt + angle * (-1j * axis @ dt),
                      angle * (-1j * sector.mdiff * dt), angle * dt_dc)),
            np.stack((twist * dt_dc, twist * (-1j * dp * sector.charges),
                      dc + twist * (-1j * dc * sector.charges))),
        ))
        slots = slot + length * np.arange(3)
        dgain = layer @ dgain + gain_layer[:, None] @ dword[None]
        dgain[:, slots] += local[None] @ gain[:, None] + mixed @ word
        dword = layer @ dword
        dword[slots] += local @ word
        gain = layer @ gain + gain_layer @ word
        word = layer @ word

    dword = np.einsum("rm,rij->mij", chart, dword, optimize=True)
    dgain = np.einsum("rm,arij->amij", chart, dgain, optimize=True)
    generators = 1j * adjoint(word) @ gain
    dgenerators = 1j * (adjoint(dword)[None] @ gain[:, None] + adjoint(word) @ dgain)
    nominal, dnominal, log_tail = target_log(sector, word, dword)
    matrices = [sector.coordinates(x).T for x in (dnominal, dgenerators[0], dgenerators[1])]
    center = [sector.coordinates(x) for x in (nominal, generators[0], generators[1])]
    azimuth = slice(length, 2 * length)
    common_word = dword[azimuth].sum(axis=0)
    common_gen = dgenerators[:, azimuth].sum(axis=1)
    trace_dg = np.trace(dgenerators, axis1=-2, axis2=-1) / d
    checks = {
        "sector_dimension": d,
        "endpoint_minus_target_opnorm": float(np.linalg.norm(word - sector.target, 2)),
        "rotation_and_twist_generator_opnorms": np.linalg.norm(generators, 2, axis=(-2, -1)).tolist(),
        "max_generator_trace_per_dimension": float(np.max(abs(np.trace(generators, axis1=-2, axis2=-1))) / d),
        "max_generator_derivative_trace_per_dimension": float(np.max(abs(trace_dg))),
        "max_twist_generator_derivative_column_frobenius": float(np.linalg.norm(dgenerators[1], axis=(-2, -1)).max()),
        "common_azimuth_endpoint_opnorm": float(np.linalg.norm(common_word, 2)),
        "common_azimuth_conjugation_defect_endpoint_frobenius": float(np.linalg.norm(common_word + 1j * sector.mdiff * word)),
        "common_azimuth_conjugation_defect_generator_frobenius": float(np.linalg.norm(common_gen + 1j * sector.mdiff * generators)),
        "nominal_log_tail_bound": log_tail,
    }
    return matrices, center, checks


def toggling_coordinates(sector, batch):
    """Independent evaluation by prefix conjugation of the applied gain generators."""
    count, _, length = batch.shape
    prefix = np.broadcast_to(np.eye(sector.d, dtype=complex), (count, sector.d, sector.d)).copy()
    rotation_response, twist_response = np.zeros_like(prefix), np.zeros_like(prefix)
    for slot in range(length):
        theta, phi, chi = batch[:, :, slot].T
        twist = np.exp(-1j * chi[:, None] * sector.charges)
        after_twist = twist[:, :, None] * prefix
        phase = np.exp(-1j * phi[:, None, None] * sector.mdiff)
        axis = sector.jx * phase
        rotation_response += theta[:, None, None] * adjoint(after_twist) @ axis @ after_twist
        twist_response += chi[:, None, None] * adjoint(prefix) @ (sector.charges[:, None] * prefix)
        phases = np.exp(-1j * theta[:, None] * sector.eig)
        rotation = ((sector.eigvec * phases[:, None, :]) @ sector.eigvec.T) * phase
        prefix = rotation @ after_twist
    nominal, _, tail = target_log(sector, prefix)
    return [sector.coordinates(v) for v in (nominal, rotation_response, twist_response)], tail


def centered(sector, physical, chart, step):
    m = chart.shape[1]
    increments = chart.T.reshape(m, *physical.shape)
    batch = np.concatenate((physical + step * increments, physical - step * increments))
    values, tail = toggling_coordinates(sector, batch)
    return [(value[:m] - value[m:]).T / (2 * step) for value in values], tail


def matrix_rank(singular, tolerance=RANK_TOL):
    return int(np.count_nonzero(singular > tolerance * singular[0])) if len(singular) else 0


def matrix_diagnostics(analytic_matrix, broad, fine, extrapolated):
    rows, columns = analytic_matrix.shape
    if rows == 0:
        return {"shape": [rows, columns], "rank": 0, "nullity": columns,
                "status": "structurally_absent", "singular_values": [], "richardson_singular_values": []}
    _, singular, vt = svd(analytic_matrix, full_matrices=False, lapack_driver="gesvd")
    independent = svd(extrapolated, compute_uv=False, lapack_driver="gesdd")
    discrepancies = [float(np.linalg.norm(estimate - analytic_matrix, 2)) for estimate in (broad, fine, extrapolated)]
    rank = matrix_rank(singular)
    weak_error = float(np.linalg.norm((analytic_matrix - extrapolated) @ vt[-1]))
    resolved = rank == rows == matrix_rank(independent) and singular[-1] > 10 * discrepancies[-1]
    return {
        "shape": [rows, columns], "rank": rank, "nullity": columns - rank,
        "richardson_rank": matrix_rank(independent),
        "status": "full_row_rank_resolved_numerically" if resolved else "rank_or_resolution_threshold_sensitive",
        "sigma_max": float(singular[0]), "sigma_min": float(singular[-1]),
        "sigma_min_over_max": float(singular[-1] / singular[0]),
        "analytic_vs_central_and_richardson_opnorms": discrepancies,
        "weak_direction_richardson_error_norm": weak_error,
        "ranks_by_relative_threshold": {f"{t:.0e}": [matrix_rank(singular, t), matrix_rank(independent, t)] for t in SENSITIVITY},
        "singular_values": singular.tolist(), "richardson_singular_values": independent.tolist(),
    }


def nullspace(matrix):
    _, singular, vt = svd(matrix, full_matrices=True, lapack_driver="gesvd")
    return vt[matrix_rank(singular):].T


def restricted_diagnostics(constraint_matrices, target_matrices):
    analytic_kernel = nullspace(constraint_matrices[0])
    independent_kernel = nullspace(constraint_matrices[-1])
    result = matrix_diagnostics(*(matrix @ analytic_kernel for matrix in target_matrices))
    independent = svd(target_matrices[-1] @ independent_kernel, compute_uv=False)
    result["richardson_rank_on_own_kernel"] = matrix_rank(independent)
    result["richardson_sigma_min_on_own_kernel"] = float(independent[-1])
    result["kernel_projector_discrepancy_opnorm"] = float(np.linalg.norm(
        analytic_kernel @ analytic_kernel.T - independent_kernel @ independent_kernel.T, 2))
    result["analytic_constraint_times_own_kernel_opnorm"] = float(np.linalg.norm(constraint_matrices[0] @ analytic_kernel, 2))
    return result


def compute_register(n, stored, out):
    length = LENGTHS[n]
    if stored.shape != (3 * length,) or stored.dtype != np.float64 or not np.isfinite(stored).all():
        raise ValueError(f"unexpected archived controls for N={n}")
    theta, phi, raw = np.split(stored, 3)
    physical = np.stack((theta, phi, raw - raw.mean()))
    chart = control_chart(length)
    m = chart.shape[1]
    groups = {key: [[], [], [], []] for key in ("nominal", "rotation_response", "twist_response")}
    logical_masks, sectors = [], {}
    for two_j in range(n, -1, -2):
        sector = Sector(n, two_j)
        exact, center, checks = analytic(sector, physical, chart)
        independently_evaluated, _ = toggling_coordinates(sector, physical[None])
        checks["center_coordinate_discrepancy_frobenius"] = float(np.sqrt(sum(
            np.linalg.norm(a - b[0])**2 for a, b in zip(center, independently_evaluated))))
        broad, tail_broad = centered(sector, physical, chart, STEPS[0])
        fine, tail_fine = centered(sector, physical, chart, STEPS[1])
        checks["log_tail_bounds_max_over_perturbations"] = [tail_broad, tail_fine]
        extrap = [(4 * small - large) / 3 for large, small in zip(broad, fine)]
        block_diagnostics = {}
        response_parts = []
        for component, key in enumerate(groups):
            parts = [version[component] for version in (exact, broad, fine, extrap)]
            if key == "twist_response" and two_j < 2:
                parts = [matrix[:0] for matrix in parts]
            block_diagnostics[key] = matrix_diagnostics(*parts)
            for group, part in zip(groups[key], parts):
                group.append(part)
            response_parts.append(parts)
        block_response = [np.vstack(v) for v in zip(*response_parts[1:])]
        block_joint = [np.vstack((r, t)) for r, t in zip(block_response, response_parts[0])]
        block_diagnostics["response"] = matrix_diagnostics(*block_response)
        block_diagnostics["joint"] = matrix_diagnostics(*block_joint)
        sectors[f"two_j_{two_j}"] = {"checks": checks, "diagnostics": block_diagnostics}
        logical_masks.append(sector.logical)
    groups = {key: [np.vstack(rows) for rows in versions] for key, versions in groups.items()}
    logical = np.concatenate(logical_masks)
    if logical.sum() != 15:
        raise ValueError("logical basis has incorrect dimension")
    nominal = groups["nominal"]
    response = [np.vstack((a, b)) for a, b in zip(groups["rotation_response"], groups["twist_response"])]
    full = [np.vstack((r, t)) for r, t in zip(response, nominal)]
    closed = [np.vstack((r, t[~logical])) for r, t in zip(response, nominal)]
    diagnostics = {key: matrix_diagnostics(*matrices) for key, matrices in {
        **groups, "response": response, "full_joint": full, "response_and_fixed_complement": closed,
    }.items()}
    diagnostics["target_on_response_kernel"] = restricted_diagnostics(response, nominal)
    diagnostics["logical_on_response_and_fixed_complement_kernel"] = restricted_diagnostics(
        closed, [t[logical] for t in nominal])
    extension = "n%d_jacobians.npz" % n
    with (out / extension).open("xb") as handle:
        np.savez_compressed(handle, analytic=full[0], central_2e_4=full[1], central_1e_4=full[2],
                            richardson=full[3], nominal_logical_row_mask=logical)
    return {
        "N": n, "pairs": length, "physical_parameters": m,
        "physical_chart_orthonormality_frobenius_defect": float(np.linalg.norm(chart.T @ chart - np.eye(m))),
        "raw_bounds_min_slack_theta_phi_v": [float(limit - abs(a).max()) for limit, a in zip((np.pi, 3 * np.pi, .6), (theta, phi, raw))],
        "min_abs_rotation": float(abs(theta).min()), "min_abs_centered_twist": float(abs(physical[2]).min()),
        "signed_twist_sum": float(physical[2].sum()),
        "jacobian_archive_row_order": ["rotation_response", "twist_response", "nominal"],
        "jacobian_archive": extension, "response_rows": len(response[0]), "full_target_rows": len(nominal[0]),
        "fixed_complement_rows": int((~logical).sum()), "logical_rows": int(logical.sum()),
        "diagnostics": diagnostics, "sectors": sectors,
    }


def render_report(registers, path):
    lines = [
        "# Local response and target derivatives at the archived development controls", "",
        "Real ranks below use the registered relative threshold of 1e-9. The two evaluations use analytic mixed control/gain derivatives and Richardson-extrapolated control differences of independently propagated input-frame gain generators.", "",
        "| N | Pairs | Parameters | Response rank | Full joint rank | Full target rank on response kernel | Response and fixed complement rank | Logical rank on their kernel | Full joint nullity |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in registers:
        d = r["diagnostics"]
        lines.append("| " + " | ".join(str(v) for v in (r["N"], r["pairs"], r["physical_parameters"],
            d["response"]["rank"], d["full_joint"]["rank"], d["target_on_response_kernel"]["rank"],
            d["response_and_fixed_complement"]["rank"], d["logical_on_response_and_fixed_complement_kernel"]["rank"],
            d["full_joint"]["nullity"])) + " |")
    lines += ["", "The full endpoint includes every distinct spin sector. The embedded SU(4) target varies only the coded action while the spectator and lower-spin actions stay fixed. The physical fit has one redundant raw mean-twist coordinate. A common shift of drive azimuth changes the laboratory-frame gate.", "",
              "| N | Matrix | Smallest singular value | Largest singular value | Analytic versus Richardson operator norm | Numerical resolution |",
              "|---:|---|---:|---:|---:|---|"]
    for r in registers:
        for key in ("response", "full_joint", "target_on_response_kernel",
                    "response_and_fixed_complement", "logical_on_response_and_fixed_complement_kernel"):
            d = r["diagnostics"][key]
            lines.append(f"| {r['N']} | {key.replace('_', ' ')} | {d['sigma_min']:.8g} | {d['sigma_max']:.8g} | {d['analytic_vs_central_and_richardson_opnorms'][-1]:.3g} | {d['status'].replace('_', ' ')} |")
    lines += ["", "Both words have a finite nominal and gain-response residual. These ranks and nullities describe the linear maps at the stored controls. An exact robust solution, a corresponding fixed-target manifold and a certified radius would require further evidence.",
              "", "The register JSON files include every singular value, five rank thresholds, each block, weakest-direction cross-checks, and the remaining basis and model checks. The NPZ files contain the underlying matrices in their common coordinates.", ""]
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="fresh output directory with an existing parent")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=False, exist_ok=False)
    begin = time.perf_counter()
    try:
        paths = {n: SOURCE / f"n{n}_corrected.npy" for n in LENGTHS}
        inputs = [Path(__file__).resolve(), PROTOCOL, *paths.values()]
        write_json(out / "inputs.json", {
            "input_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs},
            "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "numpy": np.__version__, "scipy": scipy.__version__, "blas_threads_requested": 1,
            "precision": "float64/complex128", "physical_pairs_by_N": LENGTHS, "steps": STEPS,
            "relative_rank_thresholds": SENSITIVITY, "reported_relative_threshold": RANK_TOL,
            "log_series_degree": LOG_DEGREE,
        })
        registers, times = [], {}
        for n in LENGTHS:
            if sha(paths[n]) != INPUT_HASHES[n]:
                raise ValueError(f"candidate digest differs from registration for N={n}")
            start = time.perf_counter()
            result = compute_register(n, np.load(paths[n], allow_pickle=False), out)
            write_json(out / f"n{n}.json", result)
            registers.append(result)
            times[n] = time.perf_counter() - start
            print(f"N={n} response={result['diagnostics']['response']['rank']} "
                  f"joint={result['diagnostics']['full_joint']['rank']} wall={times[n]:.2f}s", flush=True)
        render_report(registers, out / "report.md")
        unresolved = [f"N={r['N']} {key}" for r in registers for key, value in r["diagnostics"].items()
                      if value["status"] != "full_row_rank_resolved_numerically"]
        write_json(out / "summary.json", {"status": "completed_fixed_control_diagnostic",
            "unresolved_aggregate_maps": unresolved, "wall_s_by_register": times,
            "wall_s": time.perf_counter() - begin, "failures": []})
    except BaseException:
        write_json(out / "failure.json", {"traceback": traceback.format_exc(), "wall_s": time.perf_counter() - begin})
        raise
    finally:
        write_json(out / "sha256.json", {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()})


if __name__ == "__main__":
    main()
