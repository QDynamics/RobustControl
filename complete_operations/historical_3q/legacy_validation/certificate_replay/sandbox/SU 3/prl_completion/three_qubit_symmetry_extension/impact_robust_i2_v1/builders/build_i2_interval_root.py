"""Build 37-dimensional strict interval Krawczyk certificates for I2."""
from __future__ import annotations

import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import mpmath as mp


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORKSPACE = next(path for path in ROOT.parents if path.name == "SU 3")
sys.path.insert(0, str(ROOT / "runtime"))
from interval_robust_reduced import DIContext, INDEPENDENT_ROWS, RI, evaluate  # noqa: E402


CANDIDATE = ROOT / "candidates" / "i2_search_attempt3.json"
RADIUS = Fraction("1e-50")


def identity(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": path.relative_to(WORKSPACE).as_posix(), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest().upper()}


def midpoint(value: RI, dps: int) -> mp.mpf:
    with mp.workdps(dps):
        return mp.mpf(value.lo + value.hi) / (mp.mpf(2) * mp.mpf(2) ** value.ctx.bits)


def matmul(a: list[list[RI]], b: list[list[RI]]) -> list[list[RI]]:
    ctx = a[0][0].ctx
    out = [[ctx.zero for _ in range(len(b[0]))] for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(b[0])):
            value = ctx.zero
            for k in range(len(b)):
                value = value + a[i][k] * b[k][j]
            out[i][j] = value
    return out


def matvec(a: list[list[RI]], b: list[RI]) -> list[RI]:
    ctx = a[0][0].ctx
    return [sum((a[i][j] * b[j] for j in range(len(b))), ctx.zero)
            for i in range(len(a))]


def q_infinity(matrix: list[list[RI]]) -> RI:
    ctx = matrix[0][0].ctx
    rows = []
    for row in matrix:
        rows.append(sum((item.abs() for item in row), ctx.zero))
    return RI(ctx, min(item.lo for item in rows), max(item.hi for item in rows))


def record(value: RI) -> dict:
    return {"lo_scaled": str(value.lo), "hi_scaled": str(value.hi),
            "scale_bits": value.ctx.bits}


def load_candidate() -> tuple[list[str], tuple[int, ...], tuple[int, ...], dict]:
    document = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    selected = tuple(int(i) for i in document["selected_variable_indices"])
    rows = tuple(int(i) for i in document["independent_rows"])
    if len(document["parameters"]) != 44 or len(selected) != 37:
        raise RuntimeError("candidate dimension mismatch")
    if rows != INDEPENDENT_ROWS or len(set(selected)) != 37:
        raise RuntimeError("candidate constraint/coordinate convention mismatch")
    return document["parameters"], selected, rows, document


def build(bits: int, parameters: list[str], selected: tuple[int, ...]) -> dict:
    ctx = DIContext(bits)
    exact = [Fraction(value) for value in parameters]
    center = [ctx.from_fraction(value) for value in exact]
    box = list(center)
    for global_index in selected:
        box[global_index] = ctx.from_bounds(exact[global_index] - RADIUS,
                                            exact[global_index] + RADIUS)

    f0, j0, tau0, witness0 = evaluate(center, selected)
    _, jbox, taubox, witness_box = evaluate(box, selected)
    dps = max(140, bits // 3 + 50)
    with mp.workdps(dps):
        midpoint_j = mp.matrix([[midpoint(j0[i][j], dps) for j in range(37)]
                                for i in range(37)])
        inverse = midpoint_j ** -1
        digits = bits // 3 + 20
        fractions = [[Fraction(mp.nstr(inverse[i, j], digits)) for j in range(37)]
                     for i in range(37)]
    y = [[ctx.point_nearest(fractions[i][j]) for j in range(37)]
         for i in range(37)]
    identity_matrix = [[ctx.one if i == j else ctx.zero for j in range(37)]
                       for i in range(37)]
    yj = matmul(y, jbox)
    defect = [[identity_matrix[i][j] - yj[i][j] for j in range(37)]
              for i in range(37)]
    displacement = [box[global_index] - center[global_index]
                    for global_index in selected]
    yf0 = matvec(y, f0)
    remainder = matvec(defect, displacement)
    image = [center[global_index] - yf0[i] + remainder[i]
             for i, global_index in enumerate(selected)]
    coordinate_box = [box[index] for index in selected]
    margins = []
    strict = True
    for i in range(37):
        lower = image[i].lo - coordinate_box[i].lo
        upper = coordinate_box[i].hi - image[i].hi
        strict = strict and lower > 0 and upper > 0
        margins.append({"lower_scaled": str(lower), "upper_scaled": str(upper)})
    qnorm = q_infinity(defect)
    contraction = qnorm.hi < ctx.scale
    denominator_safe = taubox.lo > 0
    entangling = witness_box.lo > 0
    passed = strict and contraction and denominator_safe and entangling
    return {
        "bits": bits,
        "radius_exact": str(RADIUS),
        "coordinate_box": [record(value) for value in coordinate_box],
        "f_center": [record(value) for value in f0],
        "jacobian_box": [[record(value) for value in row] for row in jbox],
        "preconditioner": [[record(value) for value in row] for row in y],
        "krawczyk_image": [record(value) for value in image],
        "strict_margins": margins,
        "q_infinity": record(qnorm),
        "trace_abs_center": record(tau0),
        "trace_abs_box": record(taubox),
        "entangling_minor_abs_center": record(witness0),
        "entangling_minor_abs_box": record(witness_box),
        "strict_interior": strict,
        "contraction_q_lt_1": contraction,
        "denominator_safe": denominator_safe,
        "entangling_minor_separated": entangling,
        "pass": passed,
    }


def main() -> None:
    parameters, selected, rows, source = load_candidate()
    results = [build(bits, parameters, selected) for bits in (320, 512)]
    output = {
        "schema": "impact-i2-robust-root-krawczyk-v1",
        "classification": ("STRICT_INTERVAL_ROOT_PASS" if all(x["pass"] for x in results)
                           else "INTERVAL_ROOT_INCOMPLETE"),
        "bound_candidate": identity(CANDIDATE),
        "selected_variable_indices": list(selected),
        "fixed_variable_indices": source["fixed_variable_indices"],
        "independent_rows": list(rows),
        "dropped_rows_restored_by_exact_identities": source["dropped_rows_restored_by_exact_identities"],
        "parameter_exact_decimals": parameters,
        "backend": "fixed-scale dyadic outward intervals plus mixed control/error jets",
        "results": results,
        "pass": all(x["pass"] for x in results),
    }
    path = ROOT / "certificates" / "i2_strict_interval_root.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, ensure_ascii=True) + "\n",
                    encoding="utf-8", newline="\n")
    if not output["pass"]:
        raise SystemExit("I2 strict interval root incomplete")
    print("IMPACT_I2_STRICT_INTERVAL_ROOT_PASS", *(x["bits"] for x in results))


if __name__ == "__main__":
    main()
