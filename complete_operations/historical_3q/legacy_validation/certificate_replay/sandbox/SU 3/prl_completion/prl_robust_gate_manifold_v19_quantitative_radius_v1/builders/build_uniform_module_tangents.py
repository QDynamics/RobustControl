"""Build outward-rounded uniform gate/tangent enclosures on affine module boxes."""
from __future__ import annotations

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
from interval_active_runtime import evaluate_active, visible_interface_with_derivatives  # noqa: E402
from certify_composite_endpoint_rank import (  # noqa: E402
    cdagger,
    ci_record,
    cmatmul,
    czeros,
    hermitian_traceless,
    implicit_branch_box,
    infinity_norm_upper,
    midpoint_mp,
    mp_fraction,
    rational_preconditioner,
    radd,
    rdefect,
    ri_record,
    rmatmul,
)
from prototype_parametric_module_affine import MODULE_DIRECTIONS  # noqa: E402


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


def build_record(module: int, bits: int, rho_text: str, zeta_text: str) -> dict:
    rho, zeta = Fraction(rho_text), Fraction(zeta_text)
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
    _, jac_box, visible, hidden = evaluate_active(box, active)
    a_box = [[jac_box[i][j] for j in range(37)] for i in range(37)]
    b_box = [[jac_box[i][37 + j] for j in range(len(varying))] for i in range(37)]

    coefficient = radd(rmatmul(a0, predictor), b0)
    linear = matvec(coefficient, delta)
    jdiff = [[jac_box[i][j] - jac0[i][j] for j in range(len(active))]
             for i in range(37)]
    remainder = matvec(jdiff, pdelta + delta)
    f_affine = vadd(f0, linear, remainder)
    defect = rdefect(y, a_box)
    q_cont = infinity_norm_upper(defect)
    k = vadd([-value for value in matvec(y, f_affine)], matvec(defect, z))
    ratios = [value.abs().upper_fraction() / zeta for value in k]
    if max(ratios) >= 1:
        raise ValueError(f"module continuation failed {module}: {max(ratios)}")

    implicit, q_implicit, correction = implicit_branch_box(a_box, b_box, digits)
    gate, raw, chart = visible_interface_with_derivatives(visible, hidden)
    tangents = []
    imag = ctx.complex(0, 1)
    for direction in range(len(varying)):
        dgate = czeros(ctx, 4, 4)
        for i in range(4):
            for j in range(4):
                value = raw[i][j][37 + direction]
                for dependent in range(37):
                    value = value + raw[i][j][dependent] * implicit[dependent][direction]
                dgate[i][j] = value
        tangent = cmatmul(cdagger(gate), dgate)
        tangent = [[imag * tangent[i][j] for j in range(4)] for i in range(4)]
        tangents.append(hermitian_traceless(tangent))
    return {
        "module": module,
        "bits": bits,
        "directions": directions,
        "varying_global_indices": list(varying),
        "rho": rho_text,
        "zeta": zeta_text,
        "continuation_q_upper": str(q_cont),
        "continuation_ratio_max": str(max(ratios)),
        "implicit_q_upper": str(q_implicit),
        "implicit_correction_upper": str(correction),
        "hidden_chart_abs": ri_record(chart),
        "gate": [[ci_record(value) for value in row] for row in gate],
        "tangents": [[[ci_record(value) for value in row] for row in tangent]
                     for tangent in tangents],
        "pass": q_cont < 1 and max(ratios) < 1 and q_implicit < 1 and chart.lo > 0,
    }
