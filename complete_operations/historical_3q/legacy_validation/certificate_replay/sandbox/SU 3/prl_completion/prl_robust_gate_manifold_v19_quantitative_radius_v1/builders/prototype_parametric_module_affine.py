"""Affine-predictor parametric Krawczyk prototype for one affected module."""
from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

import mpmath as mp

ROOT = Path(__file__).resolve().parents[1]
PRL = ROOT.parent
UPGRADE = PRL / "prl_robust_gate_manifold_v18_compiler_upgrade_v1"
TC2 = (PRL / "three_qubit_symmetry_extension" / "sq3b_r3_tc1_b3v_r2_cont_r6_r1a" /
       "repair_a1r" / "semantic_r1" / "tc2_int_r1" / "runtime")
sys.path.insert(0, str(TC2))
sys.path.insert(0, str(UPGRADE / "runtime"))
sys.path.insert(0, str(UPGRADE))

from dyadic_interval import DIContext, RI  # noqa: E402
from interval_active_runtime import evaluate_active  # noqa: E402
from certify_composite_endpoint_rank import (  # noqa: E402
    infinity_norm_upper,
    midpoint_mp,
    mp_fraction,
    rational_preconditioner,
    radd,
    rdefect,
    rmatmul,
)

PIVOTS = (181, 6, 27, 111, 97, 20, 112, 42, 154, 168, 129, 13, 0, 185, 156)
MODULE_DIRECTIONS: dict[int, list[int]] = {}
for column in PIVOTS:
    MODULE_DIRECTIONS.setdefault(column // 7, []).append(column % 7)


def matvec(a: list[list[RI]], x: list[RI]) -> list[RI]:
    return [sum((a[i][j] * x[j] for j in range(len(x))), a[i][0].ctx.zero)
            for i in range(len(a))]


def vadd(*vectors: list[RI]) -> list[RI]:
    return [sum((vector[i] for vector in vectors), vectors[0][i].ctx.zero)
            for i in range(len(vectors[0]))]


def rational_predictor(a: list[list[RI]], b: list[list[RI]], digits: int):
    ctx, n, m = a[0][0].ctx, len(a), len(b[0])
    am = mp.matrix([[midpoint_mp(a[i][j]) for j in range(n)] for i in range(n)])
    bm = mp.matrix([[midpoint_mp(b[i][j]) for j in range(m)] for i in range(n)])
    pm = -(am ** -1) * bm
    return [[ctx.from_fraction(mp_fraction(pm[i, j], digits)) for j in range(m)]
            for i in range(n)]


def certify(module: int, bits: int, rho: Fraction, zeta: Fraction) -> dict:
    root_path = UPGRADE / "candidates" / f"generic_product_module_{module:02d}_{bits}.json"
    root = json.loads(root_path.read_text(encoding="utf-8"))
    selected = tuple(root["selected_variable_indices"])
    free = tuple(root["fixed_variable_indices"])
    directions = MODULE_DIRECTIONS[module]
    varying = tuple(free[d] for d in directions)
    center = [Fraction(text) for text in root["refined_parameter_fractions"]]
    ctx = DIContext(bits)
    digits = int(bits * 0.30103) + 45
    exact = [ctx.from_fraction(value) for value in center]
    active = selected + varying
    f0, jac0, _, _ = evaluate_active(exact, active)
    a0 = [[jac0[i][j] for j in range(37)] for i in range(37)]
    b0 = [[jac0[i][37 + j] for j in range(len(varying))] for i in range(37)]
    y = rational_preconditioner(a0, digits)
    predictor = rational_predictor(a0, b0, digits)
    delta = [ctx.from_bounds(-rho, rho) for _ in varying]
    pdelta = matvec(predictor, delta)
    z = [ctx.from_bounds(-zeta, zeta) for _ in range(37)]

    box = [ctx.from_fraction(value) for value in center]
    for local, index in enumerate(selected):
        box[index] = box[index] + pdelta[local] + z[local]
    for index in varying:
        box[index] = ctx.from_bounds(center[index] - rho, center[index] + rho)
    _, jac_box, _, hidden_box = evaluate_active(box, active)
    a_box = [[jac_box[i][j] for j in range(37)] for i in range(37)]

    coefficient = radd(rmatmul(a0, predictor), b0)
    linear = matvec(coefficient, delta)
    h = pdelta + delta
    jdiff = [[jac_box[i][j] - jac0[i][j] for j in range(len(active))]
             for i in range(37)]
    remainder = matvec(jdiff, h)
    f_affine = vadd(f0, linear, remainder)
    defect = rdefect(y, a_box)
    q = infinity_norm_upper(defect)
    yf = matvec(y, f_affine)
    dz = matvec(defect, z)
    k = [-yf[i] + dz[i] for i in range(37)]
    ratios = [k[i].abs().upper_fraction() / zeta for i in range(37)]
    worst = max(range(37), key=lambda i: ratios[i])
    tau = (hidden_box[0][0] + hidden_box[1][1]).v
    return {
        "schema": "qr1-affine-parametric-module-prototype-v1",
        "module": module,
        "bits": bits,
        "directions": directions,
        "varying_global_indices": list(varying),
        "rho": str(rho),
        "zeta": str(zeta),
        "predictor_abs_max": str(max(value.abs().upper_fraction() for value in pdelta)),
        "neumann_q_upper": str(q),
        "affine_residual_after_y_abs_max": str(max(value.abs().upper_fraction() for value in yf)),
        "krawczyk_ratio_max": str(max(ratios)),
        "worst_coordinate": worst,
        "strict_interior": max(ratios) < 1,
        "trace_abs_lower": str(tau.abs().lower_fraction()),
        "trace_chart": tau.abs().lo > 0,
        "pass": max(ratios) < 1 and q < 1 and tau.abs().lo > 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=int, required=True, choices=sorted(MODULE_DIRECTIONS))
    parser.add_argument("--bits", type=int, default=320, choices=(320, 512))
    parser.add_argument("--rho", default="1e-12")
    parser.add_argument("--zeta", default="1e-14")
    args = parser.parse_args()
    result = certify(args.module, args.bits, Fraction(args.rho), Fraction(args.zeta))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
