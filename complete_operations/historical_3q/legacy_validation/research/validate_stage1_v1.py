#!/usr/bin/env python3
"""Bounded Stage-1 transcription checks.

This script performs only the fixed checks registered in
STAGE1_PROTOCOL_20260923_v1.md.  It refuses to overwrite its run directory,
does not optimize any control, and treats floating-point output as a
cross-check rather than proof.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import expm


SCRIPT = Path(__file__).resolve()
V9_ROOT = SCRIPT.parents[1]
PROTOCOL = SCRIPT.with_name("STAGE1_PROTOCOL_20260923_v1.md")
RUN_DIR = V9_ROOT / "runs" / "stage1_validation_20260923_v1"
TOL = 1.0e-10
BUDGET_SECONDS = 120.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_json(path: Path, payload: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")


def fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def max_abs(matrix: np.ndarray) -> float:
    return float(np.max(np.abs(matrix)))


def unitary_half_diamond(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    """Return half diamond distance and the minimal spectral arc width.

    For unitary channels the distance is sin(L/2) when the relative spectrum
    lies in an arc L < pi, and is one otherwise.
    """

    relative = left.conj().T @ right
    phases = np.sort(np.mod(np.angle(np.linalg.eigvals(relative)), 2.0 * math.pi))
    gaps = np.diff(np.concatenate((phases, phases[:1] + 2.0 * math.pi)))
    arc = float(max(0.0, 2.0 * math.pi - np.max(gaps)))
    if arc >= math.pi:
        return 1.0, arc
    return float(math.sin(arc / 2.0)), arc


def graph_edges(kind: str, n: int) -> list[tuple[int, int, int]]:
    if kind == "path":
        return [(i, i + 1, i + 1) for i in range(n - 1)]
    if kind == "star":
        return [(0, leaf, leaf) for leaf in range(1, n)]
    if kind == "complete":
        edges: list[tuple[int, int, int]] = []
        weight = 1
        for a in range(n):
            for b in range(a + 1, n):
                edges.append((a, b, weight))
                weight += 1
        return edges
    raise ValueError(f"unknown graph kind: {kind}")


def enumerate_graph(kind: str, n: int) -> dict[str, object]:
    edges = graph_edges(kind, n)
    total_weight = sum(weight for _, _, weight in edges)
    values_by_k: list[list[int]] = [[] for _ in range(n + 1)]
    cut_values: list[int] = []
    eigenvalues: list[int] = []

    for bits in range(1 << n):
        k = bits.bit_count()
        cut = 0
        eigenvalue = 0
        for a, b, weight in edges:
            different = ((bits >> a) & 1) != ((bits >> b) & 1)
            if different:
                cut += weight
                eigenvalue -= weight
            else:
                eigenvalue += weight
        values_by_k[k].append(eigenvalue)
        cut_values.append(cut)
        eigenvalues.append(eigenvalue)

    observed_b = [Fraction(sum(values), len(values)) for values in values_by_k]
    formula_b = [
        Fraction(total_weight * ((n - 2 * k) ** 2 - n), n * (n - 1))
        for k in range(n + 1)
    ]
    max_cut = max(cut_values)
    delta_b = max(eigenvalues) - min(eigenvalues)
    delta_sec = max(observed_b) - min(observed_b)
    delta_sec_formula = (
        Fraction(total_weight * n, n - 1)
        if n % 2 == 0
        else Fraction(total_weight * (n + 1), n)
    )
    center_b = Fraction(max(eigenvalues) + min(eigenvalues), 2)

    return {
        "kind": kind,
        "n": n,
        "edge_count": len(edges),
        "total_weight": total_weight,
        "b_observed": [fraction_text(x) for x in observed_b],
        "b_formula": [fraction_text(x) for x in formula_b],
        "b_exact_match": observed_b == formula_b,
        "max_cut": max_cut,
        "delta_B": delta_b,
        "delta_B_equals_2MaxCut": delta_b == 2 * max_cut,
        "delta_sec": fraction_text(delta_sec),
        "delta_sec_formula": fraction_text(delta_sec_formula),
        "delta_sec_exact_match": delta_sec == delta_sec_formula,
        "delta_sec_le_delta_B": delta_sec <= delta_b,
        "spectral_center_c_B": fraction_text(center_b),
        "lambda_max_equals_total_weight": max(eigenvalues) == total_weight,
        "lambda_min_equals_W_minus_2MaxCut": min(eigenvalues) == total_weight - 2 * max_cut,
    }


I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def kron_all(*operators: np.ndarray) -> np.ndarray:
    result = np.array([[1.0 + 0.0j]])
    for operator in operators:
        result = np.kron(result, operator)
    return result


def embedded(operator: np.ndarray, site: int, n: int) -> np.ndarray:
    factors = [I2] * n
    factors[site] = operator
    return kron_all(*factors)


def two_site(left: np.ndarray, a: int, right: np.ndarray, b: int, n: int) -> np.ndarray:
    factors = [I2] * n
    factors[a] = left
    factors[b] = right
    return kron_all(*factors)


def axis(phi: float, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return math.cos(phi) * p + math.sin(phi) * q


def rotation(phi: float, area: float, error: float, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return expm(-0.5j * (1.0 + error) * area * axis(phi, p, q))


def rotation_and_derivative(
    phi: float, area: float, p: np.ndarray, q: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    generator = axis(phi, p, q)
    nominal = expm(-0.5j * area * generator)
    derivative = (-0.5j * area * generator) @ nominal
    return nominal, derivative


def n2_composite_checks() -> dict[str, object]:
    p = np.kron(Z, Z)
    q = np.kron(Z, X)
    r = np.kron(I2, Y)
    theta = math.pi / 2.0
    cos_phi = -1.0 / 8.0
    sin_phi = 3.0 * math.sqrt(7.0) / 8.0
    phi = math.acos(cos_phi)

    algebra_residuals = {
        "P2_minus_I": max_abs(p @ p - np.eye(4)),
        "Q2_minus_I": max_abs(q @ q - np.eye(4)),
        "R2_minus_I": max_abs(r @ r - np.eye(4)),
        "comm_PQ_minus_2iR": max_abs(p @ q - q @ p - 2j * r),
        "comm_QR_minus_2iP": max_abs(q @ r - r @ q - 2j * p),
        "comm_RP_minus_2iQ": max_abs(r @ p - p @ r - 2j * q),
        "normalized_gram_minus_identity": max_abs(
            np.array(
                [[np.trace(a.conj().T @ b) / 4.0 for b in (p, q, r)] for a in (p, q, r)]
            )
            - np.eye(3)
        ),
    }

    local_y = expm(-0.5j * phi * r)
    local_conjugation_residual = max_abs(
        local_y @ p @ local_y.conj().T - axis(phi, p, q)
    )

    u1, du1 = rotation_and_derivative(0.0, theta, p, q)
    u2, du2 = rotation_and_derivative(phi, 2.0 * math.pi, p, q)
    u3, du3 = rotation_and_derivative(-phi, 2.0 * math.pi, p, q)
    composite_zero = u3 @ u2 @ u1
    target = expm(-0.5j * theta * p)
    d_total_1 = u3 @ u2 @ du1
    d_total_2 = u3 @ du2 @ u1
    d_total_3 = du3 @ u2 @ u1
    d_total = d_total_1 + d_total_2 + d_total_3
    right_generators = [1j * target.conj().T @ term for term in (d_total_1, d_total_2, d_total_3)]
    shared_generator = 1j * target.conj().T @ d_total

    nominal_distance, nominal_arc = unitary_half_diamond(target, composite_zero)
    identity = np.eye(4, dtype=np.complex128)
    target_to_identity, target_identity_arc = unitary_half_diamond(target, identity)

    error_values = [
        0.1,
        -0.1,
        0.03,
        -0.03,
        0.01,
        -0.01,
        0.003,
        -0.003,
        0.001,
        -0.001,
        0.0003,
        -0.0003,
    ]
    samples: list[dict[str, float]] = []
    for error in error_values:
        composite = (
            rotation(-phi, 2.0 * math.pi, error, p, q)
            @ rotation(phi, 2.0 * math.pi, error, p, q)
            @ rotation(0.0, theta, error, p, q)
        )
        distance, arc = unitary_half_diamond(target, composite)
        native = rotation(0.0, theta, error, p, q)
        native_distance, _ = unitary_half_diamond(target, native)
        samples.append(
            {
                "e": error,
                "full_channel_half_diamond": distance,
                "minimal_spectral_arc": arc,
                "D_over_abs_e": distance / abs(error),
                "D_over_e_squared": distance / (error * error),
                "native_half_diamond": native_distance,
                "native_exact_formula": abs(math.sin(error * theta / 2.0)),
            }
        )

    excitation = np.diag([0.0, 1.0, 1.0, 2.0]).astype(np.complex128)
    y_breaks_excitation = max_abs(r @ excitation - excitation @ r)

    return {
        "theta": theta,
        "phi": phi,
        "cos_phi": cos_phi,
        "sin_phi": sin_phi,
        "exact_cos_phi": "-1/8",
        "exact_sin_phi": "3*sqrt(7)/8",
        "execution_order_early_to_late": ["R_0(theta)", "R_phi(2*pi)", "R_-phi(2*pi)"],
        "matrix_product_late_to_early": "R_-phi(2*pi) R_phi(2*pi) R_0(theta)",
        "algebra_residuals": algebra_residuals,
        "local_Y_conjugation_residual": local_conjugation_residual,
        "nominal_target_half_diamond": nominal_distance,
        "nominal_target_arc": nominal_arc,
        "shared_first_jet_max_abs": max_abs(shared_generator),
        "individual_first_jet_frobenius_norms": [float(np.linalg.norm(x, "fro")) for x in right_generators],
        "sum_individual_first_jets_max_abs": max_abs(sum(right_generators)),
        "exact_shared_coefficient_in_units_of_pi": fraction_text(Fraction(1, 4) - Fraction(1, 4)),
        "native_interaction_angle": theta,
        "composite_interaction_angle": theta + 4.0 * math.pi,
        "interaction_action_ratio": (theta + 4.0 * math.pi) / theta,
        "exact_action_ratio": "9/1",
        "target_to_Gc_attainer_identity_half_diamond": target_to_identity,
        "target_to_Gc_exact_claim": 1.0 / math.sqrt(2.0),
        "target_identity_minimal_arc": target_identity_arc,
        "native_error_slope_exact": theta / 2.0,
        "native_error_slope_exact_at_theta_pi_over_2": math.pi / 4.0,
        "local_Y_breaks_excitation_commutator_max_abs": y_breaks_excitation,
        "shared_error_samples": samples,
    }


def sector_indices(n: int) -> list[list[int]]:
    return [[basis for basis in range(1 << n) if basis.bit_count() == k] for k in range(n + 1)]


def n3_schedule_checks() -> dict[str, object]:
    n = 3
    z1, z2, z3 = (embedded(Z, site, n) for site in range(n))
    x1, x2, x3 = (embedded(X, site, n) for site in range(n))
    y1, y2, y3 = (embedded(Y, site, n) for site in range(n))
    del x1, x2, x3, y1, y2, y3
    e12 = (two_site(X, 0, X, 1, n) + two_site(Y, 0, Y, 1, n)) / 2.0
    e23 = (two_site(X, 1, X, 2, n) + two_site(Y, 1, Y, 2, n)) / 2.0
    b_operator = two_site(Z, 0, Z, 1, n) + 2.0 * two_site(Z, 1, Z, 2, n)
    number = sum((np.eye(1 << n) - embedded(Z, site, n)) / 2.0 for site in range(n))

    durations_q = [Fraction(2, 5), Fraction(7, 10), Fraction(3, 10), Fraction(3, 5)]
    g_values_q = [Fraction(3, 10), Fraction(4, 5), Fraction(1, 2), Fraction(1, 5)]
    duration = [float(x) for x in durations_q]
    g_values = [float(x) for x in g_values_q]
    h_controls = [
        0.4 * z1 + 0.3 * e23,
        -0.2 * z2 + 0.5 * e12,
        0.3 * z3 - 0.4 * e23,
        0.2 * (z1 - z3) + 0.3 * (e12 + e23),
    ]
    action_q = sum((dt * g for dt, g in zip(durations_q, g_values_q)), Fraction(0, 1))
    action = float(action_q)
    delta_b = 6.0
    delta_sec = 4.0

    def propagate(error: float) -> np.ndarray:
        endpoint = np.eye(1 << n, dtype=np.complex128)
        for dt, g_value, h_control in zip(duration, g_values, h_controls):
            segment = expm(-1j * dt * (h_control + (1.0 + error) * g_value * b_operator))
            endpoint = segment @ endpoint
        return endpoint

    u0 = propagate(0.0)
    sectors = sector_indices(n)
    total_weight = 3
    b_values = [
        Fraction(total_weight * ((n - 2 * k) ** 2 - n), n * (n - 1))
        for k in range(n + 1)
    ]
    eig_b = np.linalg.eigvalsh(b_operator)
    lambda_min = float(np.min(eig_b))
    lambda_max = float(np.max(eig_b))
    spectral_center = (lambda_max + lambda_min) / 2.0
    shifted_norm = float(np.linalg.norm(b_operator - spectral_center * np.eye(1 << n), 2))

    error_values = [
        0.5,
        -0.5,
        0.3,
        -0.3,
        0.1,
        -0.1,
        0.03,
        -0.03,
        0.01,
        -0.01,
        0.003,
        -0.003,
        0.001,
        -0.001,
        0.0003,
        -0.0003,
    ]
    samples: list[dict[str, object]] = []
    for error in error_values:
        endpoint = propagate(error)
        relative = u0.conj().T @ endpoint
        distance, arc = unitary_half_diamond(u0, endpoint)
        lower_angle = abs(error) * action * delta_sec / 2.0
        upper_angle = abs(error) * action * delta_b / 2.0
        lower_bound = math.sin(lower_angle)
        upper_bound = math.sin(upper_angle)

        determinant_checks = []
        for k, indices in enumerate(sectors):
            block = relative[np.ix_(indices, indices)]
            observed = np.linalg.det(block)
            expected = np.exp(-1j * error * action * len(indices) * float(b_values[k]))
            determinant_checks.append(
                {
                    "k": k,
                    "dimension": len(indices),
                    "b_k": fraction_text(b_values[k]),
                    "observed_real": float(observed.real),
                    "observed_imag": float(observed.imag),
                    "expected_real": float(expected.real),
                    "expected_imag": float(expected.imag),
                    "abs_residual": float(abs(observed - expected)),
                }
            )

        mask = np.zeros_like(relative, dtype=bool)
        for indices in sectors:
            mask[np.ix_(indices, indices)] = True
        off_sector_max = max_abs(np.where(mask, 0.0, relative))
        samples.append(
            {
                "e": error,
                "half_diamond": distance,
                "minimal_spectral_arc": arc,
                "lower_angle": lower_angle,
                "upper_angle": upper_angle,
                "lower_bound_sin_abs_e_S_Delta_sec_over_2": lower_bound,
                "upper_bound_sin_abs_e_S_Delta_B_over_2": upper_bound,
                "lower_margin": distance - lower_bound,
                "upper_margin": upper_bound - distance,
                "domain_upper_angle_lt_pi_over_2": upper_angle < math.pi / 2.0,
                "relative_off_sector_max_abs": off_sector_max,
                "sector_determinants": determinant_checks,
            }
        )

    commutators = [max_abs(h @ b_operator - b_operator @ h) for h in h_controls]
    conservation_residuals = [max_abs(h @ number - number @ h) for h in h_controls]
    conservation_residuals.append(max_abs(b_operator @ number - number @ b_operator))

    return {
        "durations": duration,
        "g_values": g_values,
        "segment_actions": [float(dt * g) for dt, g in zip(durations_q, g_values_q)],
        "exact_action_S": fraction_text(action_q),
        "action_S": action,
        "Delta_B": delta_b,
        "Delta_sec": delta_sec,
        "lambda_min_B": lambda_min,
        "lambda_max_B": lambda_max,
        "spectral_center_c_B": spectral_center,
        "shifted_operator_norm": shifted_norm,
        "Delta_B_over_2": delta_b / 2.0,
        "excitation_conservation_residuals": conservation_residuals,
        "Hc_B_commutator_max_abs_by_segment": commutators,
        "has_noncommuting_segment": max(commutators) > TOL,
        "samples": samples,
    }


def main() -> int:
    started_utc = utc_now()
    started = time.perf_counter()
    if RUN_DIR.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {RUN_DIR}")
    RUN_DIR.mkdir(parents=True, exist_ok=False)

    config = {
        "protocol": str(PROTOCOL.relative_to(V9_ROOT)).replace(os.sep, "/"),
        "script": str(SCRIPT.relative_to(V9_ROOT)).replace(os.sep, "/"),
        "run_directory": str(RUN_DIR.relative_to(V9_ROOT)).replace(os.sep, "/"),
        "purpose": "bounded formula transcription and counterexample checks; no optimization",
        "matrix_tolerance": TOL,
        "runtime_budget_seconds": BUDGET_SECONDS,
        "graph_families": ["path", "star", "complete"],
        "n_values": list(range(2, 9)),
        "theta": "pi/2",
        "shared_error_values": [
            0.1,
            -0.1,
            0.03,
            -0.03,
            0.01,
            -0.01,
            0.003,
            -0.003,
            0.001,
            -0.001,
            0.0003,
            -0.0003,
        ],
        "n3_additional_error_values": [0.3, -0.3, 0.5, -0.5],
        "evidence_level": "NUMERICAL_CROSSCHECK plus exact discrete enumeration",
    }
    dump_json(RUN_DIR / "config.json", config)

    versions = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    dump_json(RUN_DIR / "versions.json", versions)

    failures: list[dict[str, object]] = []
    checks: list[dict[str, object]] = []

    def check(name: str, condition: bool, detail: object) -> None:
        record = {"name": name, "pass": bool(condition), "detail": detail}
        checks.append(record)
        if not condition:
            failures.append(record)

    results: dict[str, object] = {}
    try:
        graph_results = [
            enumerate_graph(kind, n)
            for n in range(2, 9)
            for kind in ("path", "star", "complete")
        ]
        results["graph_enumeration"] = graph_results
        for row in graph_results:
            label = f"graph_{row['kind']}_N{row['n']}"
            check(label + "_b_k", bool(row["b_exact_match"]), row["b_observed"])
            check(label + "_Delta_B", bool(row["delta_B_equals_2MaxCut"]), row["delta_B"])
            check(label + "_Delta_sec", bool(row["delta_sec_exact_match"]), row["delta_sec"])
            check(label + "_Delta_sec_le_Delta_B", bool(row["delta_sec_le_delta_B"]), row)
            check(label + "_lambda_max", bool(row["lambda_max_equals_total_weight"]), row)
            check(label + "_lambda_min", bool(row["lambda_min_equals_W_minus_2MaxCut"]), row)

        n2 = n2_composite_checks()
        results["n2_target_and_composite"] = n2
        for name, residual in n2["algebra_residuals"].items():
            check("n2_algebra_" + name, float(residual) <= TOL, residual)
        check("n2_local_Y_conjugation", n2["local_Y_conjugation_residual"] <= TOL, n2["local_Y_conjugation_residual"])
        check("n2_nominal_target", n2["nominal_target_half_diamond"] <= TOL, n2["nominal_target_half_diamond"])
        check("n2_shared_first_jet_zero", n2["shared_first_jet_max_abs"] <= TOL, n2["shared_first_jet_max_abs"])
        check(
            "n2_each_primitive_nonrobust",
            min(n2["individual_first_jet_frobenius_norms"]) > TOL,
            n2["individual_first_jet_frobenius_norms"],
        )
        check("n2_individual_jets_cancel", n2["sum_individual_first_jets_max_abs"] <= TOL, n2["sum_individual_first_jets_max_abs"])
        check("n2_action_ratio_exact", abs(n2["interaction_action_ratio"] - 9.0) <= TOL, n2["interaction_action_ratio"])
        check(
            "n2_target_identity_attains_exact_Gc_distance",
            abs(n2["target_to_Gc_attainer_identity_half_diamond"] - n2["target_to_Gc_exact_claim"]) <= TOL,
            n2["target_to_Gc_attainer_identity_half_diamond"],
        )
        check("n2_local_Y_breaks_excitation", n2["local_Y_breaks_excitation_commutator_max_abs"] > TOL, n2["local_Y_breaks_excitation_commutator_max_abs"])
        for sample in n2["shared_error_samples"]:
            check(
                f"n2_native_formula_e_{sample['e']:+.4g}",
                abs(sample["native_half_diamond"] - sample["native_exact_formula"]) <= TOL,
                sample,
            )

        n3 = n3_schedule_checks()
        results["n3_fixed_noncommuting_schedule"] = n3
        check("n3_action_exact_19_over_20", n3["exact_action_S"] == "19/20", n3["exact_action_S"])
        check("n3_shifted_norm_Delta_B_over_2", abs(n3["shifted_operator_norm"] - n3["Delta_B_over_2"]) <= TOL, n3)
        check("n3_excitation_conservation", max(n3["excitation_conservation_residuals"]) <= TOL, n3["excitation_conservation_residuals"])
        check("n3_noncommuting_schedule", bool(n3["has_noncommuting_segment"]), n3["Hc_B_commutator_max_abs_by_segment"])
        for sample in n3["samples"]:
            tag = f"n3_e_{sample['e']:+.4g}"
            check(tag + "_finite_bound_domain", bool(sample["domain_upper_angle_lt_pi_over_2"]), sample["upper_angle"])
            check(tag + "_lower_bound", sample["lower_margin"] >= -TOL, sample)
            check(tag + "_upper_bound", sample["upper_margin"] >= -TOL, sample)
            check(tag + "_sector_block", sample["relative_off_sector_max_abs"] <= TOL, sample["relative_off_sector_max_abs"])
            for determinant in sample["sector_determinants"]:
                check(
                    tag + f"_det_sector_{determinant['k']}",
                    determinant["abs_residual"] <= TOL,
                    determinant,
                )

        results["assumption_checks"] = {
            "single_sector": {
                "N2_k1_B_restriction": "-J I_2",
                "consequence": "the cross-sector trace obstruction is absent on this restricted input set",
            },
            "independent_primitive_errors": {
                "individual_first_jet_frobenius_norms": n2["individual_first_jet_frobenius_norms"],
                "consequence": "independent pulse errors do not generically cancel",
            },
            "finite_error_without_action_bound": {
                "consequence": "no schedule-uniform nonwrapping interval follows without S <= S_max",
            },
            "added_local_Y": {
                "excitation_commutator_max_abs": n2["local_Y_breaks_excitation_commutator_max_abs"],
                "consequence": "the added resource invalidates the original excitation-sector invariant during those pulses",
            },
        }
    except Exception as exc:  # preserve the failed run rather than overwriting it
        failures.append(
            {
                "name": "unhandled_exception",
                "pass": False,
                "detail": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )

    duration_seconds = time.perf_counter() - started
    check("runtime_budget", duration_seconds <= BUDGET_SECONDS, duration_seconds)
    status = "PASS" if not failures else "FAIL"
    results["status"] = status
    results["check_count"] = len(checks)
    results["failure_count"] = len(failures)
    results["checks"] = checks
    dump_json(RUN_DIR / "results.json", results)
    dump_json(RUN_DIR / "failures.json", failures)

    runtime = {
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "duration_seconds": duration_seconds,
        "budget_seconds": BUDGET_SECONDS,
        "within_budget": duration_seconds <= BUDGET_SECONDS,
        "status": status,
    }
    dump_json(RUN_DIR / "runtime.json", runtime)

    files_to_hash = [
        SCRIPT,
        PROTOCOL,
        RUN_DIR / "config.json",
        RUN_DIR / "versions.json",
        RUN_DIR / "results.json",
        RUN_DIR / "failures.json",
        RUN_DIR / "runtime.json",
    ]
    hashes = {
        str(path.relative_to(V9_ROOT)).replace(os.sep, "/"): {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in files_to_hash
    }
    dump_json(RUN_DIR / "hashes.json", hashes)

    print(json.dumps({"status": status, "checks": len(checks), "failures": len(failures), "runtime_seconds": duration_seconds}, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
