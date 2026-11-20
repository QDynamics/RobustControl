"""Build strict 44-dimensional augmented-rank certificates at the I2 root."""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

import mpmath as mp


ROOT = Path(__file__).resolve().parent.parent
I2 = ROOT.parent / "impact_robust_i2_v1"
sys.path.insert(0, str(ROOT / "runtime"))
sys.path.insert(0, str(I2 / "runtime"))
from interval_b1_augmented import DIContext, VISIBLE_ROWS, evaluate_augmented  # noqa: E402


def midpoint(value, dps):
    with mp.workdps(dps):
        return mp.mpf(value.lo + value.hi) / (mp.mpf(2) * mp.mpf(2) ** value.ctx.bits)


def matmul(a, b):
    ctx = a[0][0].ctx
    return [[sum((a[i][k] * b[k][j] for k in range(len(b))), ctx.zero)
             for j in range(len(b[0]))] for i in range(len(a))]


def q_infinity(matrix):
    ctx = matrix[0][0].ctx
    rows = [sum((item.abs() for item in row), ctx.zero) for row in matrix]
    return max(item.hi for item in rows), ctx.scale


def record(value):
    return {"lo_scaled": str(value.lo), "hi_scaled": str(value.hi),
            "scale_bits": value.ctx.bits}


def build(bits, source):
    ctx = DIContext(bits)
    exact = [Fraction(text) for text in source["parameter_exact_decimals"]]
    selected_root = tuple(source["selected_variable_indices"])
    center = [ctx.from_fraction(value) for value in exact]
    box = list(center)
    radius = Fraction(source["results"][0]["radius_exact"])
    for index in selected_root:
        box[index] = ctx.from_bounds(exact[index] - radius, exact[index] + radius)
    all_controls = tuple(range(44))
    _, _, _, center_augmented, _ = evaluate_augmented(center, all_controls)
    _, constraint_box, visible_box, augmented_box, tau = evaluate_augmented(box, all_controls)
    dps = max(150, bits // 3 + 60)
    with mp.workdps(dps):
        matrix = mp.matrix([[midpoint(center_augmented[i][j], dps) for j in range(44)]
                            for i in range(44)])
        inverse = matrix ** -1
        digits = bits // 3 + 25
        fractions = [[Fraction(mp.nstr(inverse[i, j], digits)) for j in range(44)]
                     for i in range(44)]
    y = [[ctx.point_nearest(fractions[i][j]) for j in range(44)] for i in range(44)]
    product = matmul(y, augmented_box)
    defect = [[(ctx.one if i == j else ctx.zero) - product[i][j]
               for j in range(44)] for i in range(44)]
    q_hi, scale = q_infinity(defect)
    trace_chart = tau.v.re.abs().lo > 0
    passed = q_hi < scale and trace_chart
    return {
        "bits": bits,
        "root_box_radius": str(radius),
        "visible_coordinate_rows": list(VISIBLE_ROWS),
        "augmented_dimension": 44,
        "q_infinity": {"hi_scaled": str(q_hi), "scale_bits": bits},
        "q_lt_1": q_hi < scale,
        "trace_chart_separated": trace_chart,
        "preconditioner": [[record(value) for value in row] for row in y],
        "augmented_jacobian_box": [[record(value) for value in row]
                                   for row in augmented_box],
        "constraint_jacobian_box": [[record(value) for value in row]
                                    for row in constraint_box],
        "selected_visible_jacobian_box": [[record(value) for value in visible_box[index]]
                                          for index in VISIBLE_ROWS],
        "pass": passed
    }


def main():
    source = json.loads((I2 / "certificates/i2_strict_interval_root.json").read_text(
        encoding="utf-8"))
    results = [build(bits, source) for bits in (320, 512)]
    output = {
        "schema": "prl-upgrade-b1-augmented-rank-certificate-v1",
        "classification": "STRICT_AUGMENTED_RANK_PASS" if all(x["pass"] for x in results)
                          else "AUGMENTED_RANK_INCOMPLETE",
        "meaning": "rank dR=37 and rank(dV restricted to ker dR)=7 at the exact I2 root",
        "results": results,
        "pass": all(x["pass"] for x in results)
    }
    path = ROOT / "certificates" / "b1_interval_augmented_rank.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, ensure_ascii=True) + "\n",
                    encoding="utf-8", newline="\n")
    if not output["pass"]:
        raise SystemExit("B1 strict augmented rank incomplete")
    print("PRL_UPGRADE_B1_STRICT_AUGMENTED_RANK_PASS", *(x["bits"] for x in results))


if __name__ == "__main__":
    main()
