"""Independent computational-Pauli reconstruction of the registered N=2 chart.

Uses SymPy exact arithmetic, not the constructor's Fraction field or routines.
No search, numerical propagation, interval calculation, or old-file mutation.
The certificate adapter is finalized against the actual schema before execution.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import sympy as sp
from sympy.polys.matrices import DomainMatrix

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs/n2_q1_exact_chart_20260917_v1"
OUTPUT = ROOT / "runs/n2_q1_root_audit_20260917_v1"
P = sp.Symbol("p")
I = sp.I
S2 = sp.sqrt(2)
ZERO = sp.zeros(4)
IDENTITY = sp.eye(4)


def canon(a):
    return a.applyfunc(sp.expand)


def comm(a, b):
    return canon(a * b - b * a)


def rational(value):
    if isinstance(value, int):
        return sp.Rational(value)
    if isinstance(value, list) and len(value) == 2:
        return sp.Rational(*value)
    if isinstance(value, dict):
        for n, d in (("numerator", "denominator"), ("num", "den"), ("n", "d")):
            if n in value and d in value:
                return sp.Rational(value[n], value[d])
    raise TypeError(("unrecognized exact rational", value))


def field(value):
    return rational(value["a"]) + S2 * rational(value["b"])


def polynomial(value):
    return sp.expand(sum(field(c) * P**k for k, c in enumerate(value)))


def binding(path):
    b = path.read_bytes()
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": len(b),
            "sha256": hashlib.sha256(b).hexdigest()}


def write(name, value):
    with (OUTPUT / name).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def physical_generators():
    sx = sp.Matrix([[0, 1], [1, 0]])
    sy = sp.Matrix([[0, -I], [I, 0]])
    sz = sp.diag(1, -1)
    one = sp.eye(2)
    return {
        "X": -I * (sp.kronecker_product(sx, one) + sp.kronecker_product(one, sx)) / 2,
        "Y": -I * (sp.kronecker_product(sy, one) + sp.kronecker_product(one, sy)) / 2,
        "Z": -I * sp.kronecker_product(sz, sz),
    }


# Columns are Cartesian triplet |x>, |y>, |z>, and singlet in |00>,...,|11>.
SCHUR = sp.Matrix([[-1, I, 0, 0], [0, 0, 1, 1],
                   [0, 0, 1, -1], [1, I, 0, 0]]) / S2
CENTER = sp.diag(-I / 3, -I / 3, -I / 3, I)


def nu(a):
    return [sp.expand(v) for v in (
        (a[0, 1] - a[1, 0]) / 2, (a[0, 1] + a[1, 0]) / (2 * I),
        (a[0, 2] - a[2, 0]) / 2, (a[0, 2] + a[2, 0]) / (2 * I),
        (a[1, 2] - a[2, 1]) / 2, (a[1, 2] + a[2, 1]) / (2 * I),
        -I * a[0, 0], -I * a[1, 1])]


def coordinates(a, r, t):
    a, r, t = [canon(SCHUR.H * z * SCHUR) for z in (a, r, t)]
    c = sp.expand(-I * a[3, 3])
    assert sp.expand(-I * t[3, 3] - c) == 0
    assert r[3, 3] == 0
    a, t = canon(a - c * CENTER), canon(t - c * CENTER)
    for z in (a, r, t):
        assert canon(z + z.H) == ZERO
        assert sp.expand(sp.trace(z)) == 0
        assert all(z[3, k] == 0 and z[k, 3] == 0 for k in range(4))
    return [c] + nu(a) + nu(r) + nu(t)


def reconstruct(column, generators):
    u, er, et = IDENTITY, ZERO, ZERO
    for token in column["word"]:
        axis, sign = token["axis"], token["sign"]
        assert axis in generators and sign in (-1, 1)
        a = generators[axis]
        sine, cosine = sign * S2 / 2, S2 / 2
        pulse = (cosine * IDENTITY + sine * a if axis == "Z" else
                 IDENTITY + sine * a + (1 - cosine) * a * a)
        pulse = canon(pulse)
        assert canon(pulse * pulse.H) == IDENTITY
        # er, et are the exact coefficients multiplying physical pi.
        er = canon(pulse * er * pulse.H + (sp.Rational(sign, 4) * a if axis != "Z" else ZERO))
        et = canon(pulse * et * pulse.H + (sp.Rational(sign, 4) * a if axis == "Z" else ZERO))
        u = canon(pulse * u)
    axis = column["primitive"]
    a = canon(u * generators[axis] * u.H)
    constant = coordinates(a, a if axis != "Z" else ZERO, a if axis == "Z" else ZERO)
    linear = coordinates(ZERO, comm(er, a), comm(et, a))
    return [sp.expand(c + P * d) for c, d in zip(constant, linear)]


def main():
    started = time.monotonic()
    OUTPUT.mkdir(exist_ok=False)
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Independent full 4x4 computational Pauli construction; exact SymPy arithmetic.",
        "python": sys.version, "sympy": sp.__version__,
        "budget_seconds": 180, "search": False, "floating_arithmetic": False,
        "inverse_bound_method": "Exact symbolic inverse; Laurent coefficient triangle bounds using 3<pi<4 and sqrt(2)<3/2. No interval solver or numerical enclosure library.",
        "inputs": [binding(SOURCE / name) for name in
                   ("config.json", "result.json", "selected_columns.json", "minor_certificate.json", "resource_certificate.json")],
        "tool": binding(Path(__file__)),
    }
    write("config.json", config)
    try:
        load = lambda name: json.loads((SOURCE / name).read_text(encoding="utf-8"))
        source_result = load("result.json")
        selected = load("selected_columns.json")
        minor = load("minor_certificate.json")
        resources = load("resource_certificate.json")
        assert source_result["status"] == "EXACT_CHART_PASS"
        columns = selected["selected"]
        assert len(columns) == 25
        generators = physical_generators()
        assert canon(SCHUR.H * SCHUR) == IDENTITY
        for axis, actual in generators.items():
            encoded = selected["primitive_matrices"][axis]
            expected = sp.Matrix([[field(v["re"]) + I * field(v["im"]) for v in row]
                                  for row in encoded])
            assert canon(SCHUR.H * actual * SCHUR - expected) == ZERO

        rebuilt = []
        for j, column in enumerate(columns):
            assert time.monotonic() - started < 180
            actual = reconstruct(column, generators)
            expected = [polynomial(v) for v in column["raw_column"]]
            assert all(sp.expand(a - b) == 0 for a, b in zip(actual, expected)), j
            rebuilt.append(sp.Matrix(actual))
        bmatrix = sp.Matrix.hstack(*rebuilt)
        normalizations = [c["normalization_integer"] for c in columns]
        normalized = bmatrix * sp.diag(*normalizations)
        recorded_normalized = sp.Matrix([[polynomial(v) for v in row]
                                         for row in minor["normalized_matrix"]])
        assert canon(normalized - recorded_normalized) == sp.zeros(25)

        domain = sp.QQ.algebraic_field(S2).poly_ring(P)
        dm = DomainMatrix.from_Matrix(bmatrix).convert_to(domain)
        exact_determinant = sp.expand(domain.to_sympy(dm.det()))
        reported = polynomial(minor["determinant_coefficients_ascending"]) / sp.prod(normalizations)
        assert sp.expand(exact_determinant - reported) == 0
        assert sp.expand(exact_determinant - P**11 / 2048) == 0
        assert sp.prod(normalizations) == minor["raw_determinant_denominator"]
        assert exact_determinant.subs(P, 1) == field(selected["raw_determinant_at_p1_from_pivots"])
        depths = [len(c["word"]) for c in columns]
        assert sum(depths) == resources["sum_conjugator_depths"] == 34
        assert 2 * sum(depths) + 25 == resources["raw_primitive_count"] == 93
        assert 2 * sum(depths) + 50 == resources["nonzero_pair_primitive_count"] == 118
        assert rational(resources["total_fixed_center_absolute_area_pi_coefficient"]) == sp.Rational(93, 4)
        assert rational(resources["center_duration_pi_coefficient"]) == sp.Rational(59, 2)
        assert time.monotonic() - started < 180

        field_matrix = dm.to_field()
        inverse = field_matrix.inv()
        identity_product = (field_matrix * inverse).to_Matrix()
        assert identity_product == sp.eye(25)
        inv_expressions = inverse.to_Matrix()
        bounds = sp.zeros(25)
        serialized_inverse = []
        maximum_pole = 0
        for row_index in range(25):
            row_expressions = []
            for col_index in range(25):
                expr = sp.cancel(inv_expressions[row_index, col_index], extension=S2)
                row_expressions.append(str(expr))
                num, den = sp.fraction(expr)
                denominator = sp.Poly(den, P, extension=S2)
                assert len(denominator.terms()) == 1, (row_index, col_index, expr)
                pole = denominator.degree()
                maximum_pole = max(maximum_pole, pole)
                laurent = sp.Poly(sp.expand(expr * P**pole), P, extension=S2)
                bound = sp.Rational(0)
                for (degree,), coefficient in laurent.terms():
                    coefficient = sp.expand(coefficient)
                    sqrt_coefficient = coefficient.coeff(S2)
                    rational_coefficient = sp.expand(coefficient - sqrt_coefficient * S2)
                    assert rational_coefficient.is_Rational and sqrt_coefficient.is_Rational
                    coefficient_bound = abs(rational_coefficient) + sp.Rational(3, 2) * abs(sqrt_coefficient)
                    exponent = degree - pole
                    bound += coefficient_bound * (sp.Rational(4)**exponent if exponent >= 0 else sp.Rational(3)**exponent)
                bounds[row_index, col_index] = bound
            serialized_inverse.append(row_expressions)
            assert time.monotonic() - started < 180
        row_norm = max(sum(bounds[i, j] for j in range(25)) for i in range(25))
        column_norm = max(sum(bounds[i, j] for i in range(25)) for j in range(25))
        beta_integer = int(sp.ceiling((row_norm + column_norm) / 2))
        inverse_result = {
            "status": "EXACT_SYMBOLIC_INVERSE_AND_ANALYTIC_NORM_MAJORANT_PASS",
            "variable": "p; physical value pi", "basis": selected["basis_labels"],
            "inverse_entries": serialized_inverse,
            "inverse_product_exact_identity": True,
            "entry_upper_bounds": [[str(bounds[i, j]) for j in range(25)] for i in range(25)],
            "row_sum_norm_bound": str(row_norm), "column_sum_norm_bound": str(column_norm),
            "beta_euclidean_integer_upper_bound": beta_integer,
            "max_laurent_pole_order": maximum_pole,
            "analytic_inequalities": ["3<pi<4", "sqrt(2)<3/2", "||B^-1||_2 <= sqrt(||B^-1||_1*||B^-1||_infinity) <= (row_bound+column_bound)/2"],
            "rounding": "Exact rational arithmetic followed by integer ceiling; no binary floating-point bound.",
        }
        write("inverse_bound.json", inverse_result)
        result = {
            "status": "ROOT_EXACT_CHART_AUDIT_PASS", "created_utc": datetime.now(timezone.utc).isoformat(),
            "reconstruction": "Full computational Pauli generators and independent Bell/Cartesian transform.",
            "constructor_code_imported": False, "exact_polynomial_column_count": 25,
            "exact_polynomial_entries_matched": 625,
            "determinant_method": "Direct polynomial-domain exact determinant, independent of interpolation.",
            "raw_determinant": str(exact_determinant), "physical_determinant": "pi^11/2048 > 0",
            "resource_counts": {"sum_depths": 34, "raw_slots": 93, "nonzero_pair_slots": 118},
            "beta_euclidean_integer_upper_bound": beta_integer,
            "elapsed_seconds": time.monotonic() - started,
        }
        write("result.json", result)
        write("manifest.json", {"files": [binding(OUTPUT / name) for name in ("config.json", "inverse_bound.json", "result.json")]})
        print(json.dumps(result))
    except Exception as exc:
        write("result.json", {"status": "ROOT_EXACT_CHART_AUDIT_FAILED",
                              "error": repr(exc), "elapsed_seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    main()
