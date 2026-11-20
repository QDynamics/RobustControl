"""Strict interval rank certificate for the 28-module robust product word.

For each module used by the deterministic CU0 minor, this script encloses the
implicit seven-coordinate tangent of the exact robust-root branch.  It then
transports those tangents through the full visible projective-U(4) word and
proves that the fixed 15-by-15 traceless Maurer--Cartan coordinate minor is
nonsingular.  Equivalently, this is an SU(4) statement after choosing a local
determinant-normalized lift; no global phase branch is asserted.

All proof arithmetic after the midpoint preconditioners is exact fixed-scale
dyadic interval arithmetic.  mpmath is used only to choose rational
preconditioners and approximate centers; their quality is verified by the
outward-rounded Neumann defects.
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

ROOT = Path(__file__).resolve().parent
PRL = ROOT.parent
EXT = PRL / "three_qubit_symmetry_extension"
TC2_RUNTIME = (EXT / "sq3b_r3_tc1_b3v_r2_cont_r6_r1a" / "repair_a1r" /
               "semantic_r1" / "tc2_int_r1" / "runtime")
sys.path.insert(0, str(TC2_RUNTIME))
sys.path.insert(0, str(ROOT / "runtime"))

from dyadic_interval import CI, DIContext, RI  # noqa: E402
from interval_active_runtime import (  # noqa: E402
    evaluate_active,
    visible_interface_with_derivatives,
)


PIVOT_COLUMNS = (181, 6, 27, 111, 97, 20, 112, 42, 154, 168, 129, 13, 0, 185, 156)
EXPECTED_MODULES = tuple((column // 7, column % 7) for column in PIVOT_COLUMNS)
SOURCE = EXT / "compiler_depth_reduction_v1" / "candidates" / "polished_generic_product_v1.json"
EXPECTED_SOURCE = (108315, "785EF3930E278588175F2DF119AF33624B2B029F75044EB0D45D5D730090EA4B")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def identity(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {"path": path.relative_to(PRL.parent).as_posix(), "bytes": len(data),
            "sha256": sha(data)}


def ri_record(value: RI) -> dict[str, object]:
    return {"lo_scaled": str(value.lo), "hi_scaled": str(value.hi),
            "scale_bits": value.ctx.bits}


def ci_record(value: CI) -> dict[str, object]:
    return {"re": ri_record(value.re), "im": ri_record(value.im)}


def parse_ri(ctx: DIContext, record: dict[str, object]) -> RI:
    if record.get("scale_bits") != ctx.bits:
        raise ValueError("interval scale mismatch")
    return RI(ctx, int(record["lo_scaled"]), int(record["hi_scaled"]))


def parse_ci(ctx: DIContext, record: dict[str, object]) -> CI:
    return CI(parse_ri(ctx, record["re"]), parse_ri(ctx, record["im"]))


def midpoint_mp(value: RI) -> mp.mpf:
    return mp.mpf(value.lo + value.hi) / mp.mpf(2 * value.ctx.scale)


def mp_fraction(value: mp.mpf, digits: int) -> Fraction:
    return Fraction(mp.nstr(value, digits, strip_zeros=False))


def rmatmul(a: list[list[RI]], b: list[list[RI]]) -> list[list[RI]]:
    if not a or len(a[0]) != len(b):
        raise ValueError("real matrix shape mismatch")
    ctx = a[0][0].ctx
    out = [[ctx.zero for _ in range(len(b[0]))] for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(b[0])):
            total = ctx.zero
            for k in range(len(b)):
                total = total + a[i][k] * b[k][j]
            out[i][j] = total
    return out


def radd(a: list[list[RI]], b: list[list[RI]]) -> list[list[RI]]:
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))]
            for i in range(len(a))]


def ridentity(ctx: DIContext, size: int) -> list[list[RI]]:
    return [[ctx.one if i == j else ctx.zero for j in range(size)]
            for i in range(size)]


def rdefect(y: list[list[RI]], a: list[list[RI]]) -> list[list[RI]]:
    ya = rmatmul(y, a)
    ident = ridentity(a[0][0].ctx, len(a))
    return [[ident[i][j] - ya[i][j] for j in range(len(a))]
            for i in range(len(a))]


def infinity_norm_upper(a: list[list[RI]]) -> Fraction:
    return max(sum((entry.abs().upper_fraction() for entry in row), Fraction(0))
               for row in a)


def rational_preconditioner(a: list[list[RI]], digits: int) -> list[list[RI]]:
    ctx, n = a[0][0].ctx, len(a)
    midpoint = mp.matrix([[midpoint_mp(a[i][j]) for j in range(n)] for i in range(n)])
    inverse = midpoint ** -1
    return [[ctx.from_fraction(mp_fraction(inverse[i, j], digits)) for j in range(n)]
            for i in range(n)]


def implicit_branch_box(a: list[list[RI]], b: list[list[RI]], digits: int):
    """Enclose X solving A X + B = 0 by a verified Neumann correction."""
    ctx, n, nrhs = a[0][0].ctx, len(a), len(b[0])
    amid = mp.matrix([[midpoint_mp(a[i][j]) for j in range(n)] for i in range(n)])
    bmid = mp.matrix([[midpoint_mp(b[i][j]) for j in range(nrhs)] for i in range(n)])
    ymp = amid ** -1
    xmp = -(ymp * bmid)
    y = [[ctx.from_fraction(mp_fraction(ymp[i, j], digits)) for j in range(n)]
         for i in range(n)]
    x0_fraction = [[mp_fraction(xmp[i, j], digits) for j in range(nrhs)]
                   for i in range(n)]
    x0 = [[ctx.from_fraction(x0_fraction[i][j]) for j in range(nrhs)]
          for i in range(n)]
    defect = rdefect(y, a)
    q = infinity_norm_upper(defect)
    if q >= 1:
        raise ValueError(f"dependent Neumann defect is not contractive: {q}")
    residual = radd(rmatmul(a, x0), b)
    yr = rmatmul(y, residual)
    correction = infinity_norm_upper(yr) / (1 - q)
    enclosure = [[ctx.from_bounds(x0_fraction[i][j] - correction,
                                  x0_fraction[i][j] + correction)
                  for j in range(nrhs)] for i in range(n)]
    return enclosure, q, correction


def czeros(ctx: DIContext, rows: int, cols: int) -> list[list[CI]]:
    return [[ctx.complex() for _ in range(cols)] for _ in range(rows)]


def ceye(ctx: DIContext, size: int) -> list[list[CI]]:
    out = czeros(ctx, size, size)
    for i in range(size):
        out[i][i] = ctx.complex(1)
    return out


def cmatmul(a: list[list[CI]], b: list[list[CI]]) -> list[list[CI]]:
    if not a or len(a[0]) != len(b):
        raise ValueError("complex matrix shape mismatch")
    ctx = a[0][0].ctx
    out = czeros(ctx, len(a), len(b[0]))
    for i in range(len(a)):
        for j in range(len(b[0])):
            total = ctx.complex()
            for k in range(len(b)):
                total = total + a[i][k] * b[k][j]
            out[i][j] = total
    return out


def cdagger(a: list[list[CI]]) -> list[list[CI]]:
    return [[a[j][i].conjugate() for j in range(len(a))]
            for i in range(len(a[0]))]


def ctrace(a: list[list[CI]]) -> CI:
    total = a[0][0].ctx.complex()
    for i in range(len(a)):
        total = total + a[i][i]
    return total


def hermitian_traceless(a: list[list[CI]]) -> list[list[CI]]:
    ctx, n = a[0][0].ctx, len(a)
    half = ctx.from_fraction(Fraction(1, 2))
    quarter = ctx.from_fraction(Fraction(1, n))
    adj = cdagger(a)
    out = [[(a[i][j] + adj[i][j]) * half for j in range(n)] for i in range(n)]
    tr = ctrace(out) * quarter
    for i in range(n):
        out[i][i] = out[i][i] - tr
    return out


def kron(a: list[list[CI]], b: list[list[CI]]) -> list[list[CI]]:
    ctx = a[0][0].ctx
    out = czeros(ctx, len(a) * len(b), len(a[0]) * len(b[0]))
    for i in range(len(a)):
        for j in range(len(a[0])):
            for k in range(len(b)):
                for ell in range(len(b[0])):
                    out[i * len(b) + k][j * len(b[0]) + ell] = a[i][j] * b[k][ell]
    return out


def pauli_basis(ctx: DIContext) -> list[list[list[CI]]]:
    z, o, mi, pi = ctx.complex(), ctx.complex(1), ctx.complex(0, -1), ctx.complex(0, 1)
    one = [[o, z], [z, o]]
    x = [[z, o], [o, z]]
    y = [[z, mi], [pi, z]]
    zz = [[o, z], [z, ctx.complex(-1)]]
    half = ctx.from_fraction(Fraction(1, 2))
    return [[[entry * half for entry in row] for row in kron(a, b)]
            for ia, a in enumerate((one, x, y, zz))
            for ib, b in enumerate((one, x, y, zz)) if (ia, ib) != (0, 0)]


def coordinates(h: list[list[CI]], basis: list[list[list[CI]]]) -> list[RI]:
    return [ctrace(cmatmul(item, h)).re for item in basis]


def module_branch_tangents(root_record: dict[str, object], ctx: DIContext, digits: int):
    selected = tuple(root_record["selected_variable_indices"])
    free = tuple(root_record["fixed_variable_indices"])
    fractions = [Fraction(text) for text in root_record["refined_parameter_fractions"]]
    radius = Fraction(root_record["certificate"]["radius"])
    params = [ctx.from_fraction(value) for value in fractions]
    for index in selected:
        params[index] = ctx.from_bounds(fractions[index] - radius, fractions[index] + radius)
    active = tuple(range(44))
    _, jac, visible, hidden = evaluate_active(params, active)
    a = [[jac[i][index] for index in selected] for i in range(37)]
    b = [[jac[i][index] for index in free] for i in range(37)]
    implicit, q, correction = implicit_branch_box(a, b, digits)
    gate, raw, chart = visible_interface_with_derivatives(visible, hidden)
    tangents = []
    imag = ctx.complex(0, 1)
    for direction, free_index in enumerate(free):
        dgate = czeros(ctx, 4, 4)
        for i in range(4):
            for j in range(4):
                value = raw[i][j][free_index]
                for local, dependent_index in enumerate(selected):
                    value = value + raw[i][j][dependent_index] * implicit[local][direction]
                dgate[i][j] = value
        tangent = cmatmul(cdagger(gate), dgate)
        tangent = [[imag * tangent[i][j] for j in range(4)] for i in range(4)]
        tangents.append(hermitian_traceless(tangent))
    return gate, tangents, q, correction, chart


def load_endpoint(ctx: DIContext, module: int, bits: int):
    path = ROOT / "candidates" / f"generic_product_endpoint_{module:02d}_{bits}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not (payload.get("pass") is True and payload.get("module") == module
            and payload.get("bits") == bits):
        raise ValueError(f"endpoint admission {module}")
    return ([[parse_ci(ctx, entry) for entry in row]
             for row in payload["visible_interface"]], identity(path))


def load_root(module: int, bits: int):
    path = ROOT / "candidates" / f"generic_product_module_{module:02d}_{bits}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not (payload.get("pass") is True and payload.get("module") == module
            and payload.get("bits") == bits
            and payload.get("certificate", {}).get("strict_interior") is True):
        raise ValueError(f"root admission {module}")
    return payload, identity(path)


def certify_minor(minor: list[list[RI]], digits: int):
    y = rational_preconditioner(minor, digits)
    defect = rdefect(y, minor)
    q = infinity_norm_upper(defect)
    if q >= 1:
        raise ValueError(f"composite minor is not certified nonsingular: {q}")
    return q, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, choices=(320, 512), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing overwrite: {output}")
    source_bytes = SOURCE.read_bytes()
    if (len(source_bytes), sha(source_bytes)) != EXPECTED_SOURCE:
        raise ValueError("generic-product source binding")
    word = json.loads(source_bytes)
    if word.get("module_count") != 28 or word.get("target") != "generic_product":
        raise ValueError("generic-product source semantics")

    started = time.perf_counter()
    ctx = DIContext(args.bits)
    digits = int(args.bits * 0.30103) + 45
    gates, endpoint_sources = [], []
    for module in range(28):
        gate, source_identity = load_endpoint(ctx, module, args.bits)
        gates.append(gate)
        endpoint_sources.append(source_identity)
    suffix = [ceye(ctx, 4) for _ in range(29)]
    for module in range(27, -1, -1):
        suffix[module] = cmatmul(gates[module], suffix[module + 1])

    basis = pauli_basis(ctx)
    requested = sorted(set(module for module, _ in EXPECTED_MODULES))
    module_tangents = {}
    root_sources = []
    branch_records = []
    with mp.workdps(digits + 25):
        for count, module in enumerate(requested, 1):
            root_record, source_identity = load_root(module, args.bits)
            _, tangents, q, correction, chart = module_branch_tangents(root_record, ctx, digits)
            module_tangents[module] = tangents
            root_sources.append(source_identity)
            branch_records.append({
                "module": module,
                "dependent_neumann_q_upper": str(q),
                "implicit_correction_upper": str(correction),
                "hidden_chart_abs": ri_record(chart),
            })
            print(f"COMPOSITE_RANK_BRANCH {args.bits} {count}/{len(requested)} module={module}",
                  flush=True)

        minor = [[ctx.zero for _ in range(15)] for _ in range(15)]
        for column, (module, direction) in enumerate(EXPECTED_MODULES):
            transported = cmatmul(cmatmul(cdagger(suffix[module + 1]),
                                          module_tangents[module][direction]),
                                  suffix[module + 1])
            coordinate = coordinates(hermitian_traceless(transported), basis)
            for row in range(15):
                minor[row][column] = coordinate[row]
        q, preconditioner = certify_minor(minor, digits)

    payload = {
        "schema": "prl-v18-generic-product-composite-rank-interval-v1",
        "classification": "GENERIC_PRODUCT_COMPOSITE_ENDPOINT_RANK_PASS",
        "bits": args.bits,
        "source_candidate": identity(SOURCE),
        "module_count": 28,
        "primitive_count": 420,
        "pivot_columns": list(PIVOT_COLUMNS),
        "pivot_module_directions": [list(item) for item in EXPECTED_MODULES],
        "unique_differentiated_modules": requested,
        "root_sources": root_sources,
        "endpoint_sources": endpoint_sources,
        "branch_certificates": branch_records,
        "minor": [[ri_record(value) for value in row] for row in minor],
        "minor_neumann_q_upper": str(q),
        "minor_preconditioner": [[ri_record(value) for value in row]
                                  for row in preconditioner],
        "minor_nonsingular": q < 1,
        "composite_differential_rank": 15 if q < 1 else None,
        "elapsed_seconds": time.perf_counter() - started,
        "pass": q < 1,
        "claim_boundary": [
            "This certifies local surjectivity in PU(4), equivalently in a locally chosen SU(4) lift, at the exact 28-module robust-root word.",
            "It does not yet provide a quantitative target-neighborhood radius.",
            "The result is a fixed-depth local theorem, not global optimality or hardware robustness.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
                      encoding="utf-8", newline="\n")
    print("PRL_V18_GENERIC_PRODUCT_COMPOSITE_RANK_INTERVAL_PASS",
          args.bits, "RANK=15", f"q={q}", f"elapsed={payload['elapsed_seconds']:.3f}s")


if __name__ == "__main__":
    main()
