"""Refine and strictly certify one module of the parity-phase word.

Numerical finite differences are used only to obtain a better center.  The
load-bearing root claim is made solely by the outward-rounded interval
Krawczyk calculation at the end of the worker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

import mpmath as mp


ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT.parent
I2 = EXT / "impact_robust_i2_v1"
COMPILER = EXT / "compiler_depth_reduction_v1"
SOURCE = COMPILER / "candidates" / "polished_symmetric_parity_phase_v1.json"
ROOT_CERT = I2 / "certificates" / "i2_strict_interval_root.json"
sys.path.insert(0, str(I2 / "runtime"))
from interval_robust_reduced import DIContext, evaluate  # noqa: E402
from robust_reduced_runtime import evaluate_reduced  # noqa: E402


SETTINGS = {
    320: {"dps": 135, "digits": 120, "fd_step": "1e-50",
          "radius": Fraction("1e-55"), "target": mp.mpf("1e-85")},
    512: {"dps": 195, "digits": 178, "fd_step": "1e-80",
          "radius": Fraction("1e-90"), "target": mp.mpf("1e-130")},
}


def identity(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    workspace = next(parent for parent in ROOT.parents if parent.name == "SU 3")
    return {
        "path": path.relative_to(workspace).as_posix(),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest().upper(),
    }


def midpoint(value, dps: int) -> mp.mpf:
    with mp.workdps(dps):
        return (mp.mpf(value.lo) + mp.mpf(value.hi)) / (mp.mpf(2) ** (value.ctx.bits + 1))


def decimal_fraction(value: mp.mpf, digits: int) -> Fraction:
    return Fraction(mp.nstr(value, digits, strip_zeros=False))


def residual(point: list[mp.mpf], rows: tuple[int, ...], dps: int) -> mp.matrix:
    values = evaluate_reduced(point, dps=dps, length=15)[0]
    return mp.matrix([values[index] for index in rows])


def residual_norm(vector: mp.matrix) -> mp.mpf:
    return max(abs(vector[index]) for index in range(vector.rows))


def finite_difference_jacobian(point: list[mp.mpf], selected: tuple[int, ...],
                               rows: tuple[int, ...], dps: int,
                               step: mp.mpf) -> mp.matrix:
    jacobian = mp.matrix(37, 37)
    for local, global_index in enumerate(selected):
        plus, minus = list(point), list(point)
        plus[global_index] += step
        minus[global_index] -= step
        fp, fm = residual(plus, rows, dps), residual(minus, rows, dps)
        for row in range(37):
            jacobian[row, local] = (fp[row] - fm[row]) / (2 * step)
    return jacobian


def refine(initial: list[str], selected: tuple[int, ...], free: tuple[int, ...],
           rows: tuple[int, ...], bits: int):
    settings = SETTINGS[bits]
    dps, digits = int(settings["dps"]), int(settings["digits"])
    fixed = {index: Fraction(initial[index]) for index in free}
    with mp.workdps(dps):
        current = [mp.mpf(text) for text in initial]
        step = mp.mpf(str(settings["fd_step"]))
        target = mp.mpf(str(settings["target"]))
        vector = residual(current, rows, dps)
        history = [mp.nstr(residual_norm(vector), 35)]
        jacobian = finite_difference_jacobian(current, selected, rows, dps, step)
        inverse = jacobian ** -1
        rebuilds = 1
        for _ in range(14):
            norm = residual_norm(vector)
            if norm < target:
                fractions = [fixed.get(index, decimal_fraction(current[index], digits))
                             for index in range(len(current))]
                return fractions, history, rebuilds
            correction = inverse * vector
            accepted = None
            for exponent in range(8):
                scale = mp.mpf(2) ** (-exponent)
                trial = list(current)
                for local, global_index in enumerate(selected):
                    trial[global_index] -= scale * correction[local]
                for global_index, exact in fixed.items():
                    trial[global_index] = mp.mpf(exact.numerator) / exact.denominator
                next_vector = residual(trial, rows, dps)
                if residual_norm(next_vector) < norm:
                    accepted = trial, next_vector
                    break
            if accepted is None:
                jacobian = finite_difference_jacobian(current, selected, rows, dps,
                                                      step)
                inverse = jacobian ** -1
                rebuilds += 1
                correction = inverse * vector
                trial = list(current)
                for local, global_index in enumerate(selected):
                    trial[global_index] -= correction[local]
                for global_index, exact in fixed.items():
                    trial[global_index] = mp.mpf(exact.numerator) / exact.denominator
                next_vector = residual(trial, rows, dps)
                if residual_norm(next_vector) >= norm:
                    raise RuntimeError(f"refinement stalled at {mp.nstr(norm, 20)}")
                accepted = trial, next_vector
            current, vector = accepted
            history.append(mp.nstr(residual_norm(vector), 35))
        raise RuntimeError(f"refinement did not converge: {history[-1]}")


def matmul(a, b):
    ctx = a[0][0].ctx
    return [[sum((a[i][k] * b[k][j] for k in range(len(b))), ctx.zero)
             for j in range(len(b[0]))] for i in range(len(a))]


def matvec(a, b):
    ctx = a[0][0].ctx
    return [sum((a[i][j] * b[j] for j in range(len(b))), ctx.zero)
            for i in range(len(a))]


def record(value) -> dict[str, object]:
    return {"lo_scaled": str(value.lo), "hi_scaled": str(value.hi),
            "scale_bits": value.ctx.bits}


def certify(fractions: list[Fraction], selected: tuple[int, ...], bits: int):
    settings = SETTINGS[bits]
    radius = settings["radius"]
    dps, digits = int(settings["dps"]), int(settings["digits"])
    ctx = DIContext(bits)
    center = [ctx.from_fraction(value) for value in fractions]
    box = list(center)
    for index in selected:
        box[index] = ctx.from_bounds(fractions[index] - radius,
                                     fractions[index] + radius)
    f0, j0, trace0, _ = evaluate(center, selected)
    _, jbox, trace_box, _ = evaluate(box, selected)
    with mp.workdps(dps):
        matrix = mp.matrix([[midpoint(j0[i][j], dps) for j in range(37)]
                            for i in range(37)])
        inverse = matrix ** -1
        inv_fractions = [[decimal_fraction(inverse[i, j], digits) for j in range(37)]
                         for i in range(37)]
    y = [[ctx.point_nearest(inv_fractions[i][j]) for j in range(37)]
         for i in range(37)]
    yj = matmul(y, jbox)
    defect = [[(ctx.one if i == j else ctx.zero) - yj[i][j]
               for j in range(37)] for i in range(37)]
    displacement = [box[index] - center[index] for index in selected]
    yf0, remainder = matvec(y, f0), matvec(defect, displacement)
    image = [center[index] - yf0[i] + remainder[i]
             for i, index in enumerate(selected)]
    coordinate_box = [box[index] for index in selected]
    margins, strict = [], True
    for i in range(37):
        lower = image[i].lo - coordinate_box[i].lo
        upper = coordinate_box[i].hi - image[i].hi
        strict = strict and lower > 0 and upper > 0
        margins.append({"lower_scaled": str(lower), "upper_scaled": str(upper)})
    row_sums = [sum((item.abs() for item in row), ctx.zero) for row in defect]
    q_hi = max(item.hi for item in row_sums)
    passed = strict and q_hi < ctx.scale and trace_box.lo > 0
    return {
        "bits": bits,
        "radius": str(radius),
        "f_center": [record(value) for value in f0],
        "coordinate_box": [record(value) for value in coordinate_box],
        "krawczyk_image": [record(value) for value in image],
        "strict_margins": margins,
        "q_infinity_hi_scaled": str(q_hi),
        "q_infinity_scale_bits": bits,
        "strict_interior": strict,
        "q_lt_1": q_hi < ctx.scale,
        "trace_center": record(trace0),
        "trace_chart": trace_box.lo > 0,
        "pass": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=int, required=True)
    parser.add_argument("--bits", type=int, choices=sorted(SETTINGS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing overwrite: {output}")
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    root = json.loads(ROOT_CERT.read_text(encoding="utf-8"))
    if not 0 <= args.module < int(source["module_count"]):
        raise SystemExit("module index out of range")
    selected = tuple(root["selected_variable_indices"])
    free = tuple(root["fixed_variable_indices"])
    rows = tuple(root["independent_rows"])
    module = source["modules"][args.module]
    started = time.perf_counter()
    fractions, history, rebuilds = refine(module["parameters"], selected, free,
                                          rows, args.bits)
    refined_seconds = time.perf_counter() - started
    certificate = certify(fractions, selected, args.bits)
    total_seconds = time.perf_counter() - started
    payload = {
        "schema": "named-parity-module-strict-root-v1",
        "classification": "NAMED_PARITY_MODULE_STRICT_ROOT_PASS"
        if certificate["pass"] else "NAMED_PARITY_MODULE_STRICT_ROOT_FAIL",
        "source_candidate": identity(SOURCE),
        "source_root_certificate": identity(ROOT_CERT),
        "module": args.module,
        "bits": args.bits,
        "selected_variable_indices": list(selected),
        "fixed_variable_indices": list(free),
        "free_coordinates_preserved": all(
            fractions[index] == Fraction(module["parameters"][index]) for index in free
        ),
        "refinement_history": history,
        "finite_difference_jacobian_rebuilds": rebuilds,
        "refined_parameter_fractions": [str(value) for value in fractions],
        "refinement_seconds": refined_seconds,
        "total_seconds": total_seconds,
        "certificate": certificate,
        "pass": certificate["pass"],
        "claim_boundary": [
            "Finite differences refine the center but do not support the proof.",
            "The strict root claim is supplied only by the outward-rounded Krawczyk inclusion.",
            "This file certifies one fixed-free-coordinate module, not the composed target word.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
                      encoding="utf-8", newline="\n")
    if not payload["pass"]:
        raise SystemExit(f"module {args.module} at {args.bits} bits failed")
    print("NAMED_PARITY_MODULE_STRICT_ROOT_PASS", args.module, args.bits,
          f"refine={refined_seconds:.3f}s", f"total={total_seconds:.3f}s")


if __name__ == "__main__":
    main()
