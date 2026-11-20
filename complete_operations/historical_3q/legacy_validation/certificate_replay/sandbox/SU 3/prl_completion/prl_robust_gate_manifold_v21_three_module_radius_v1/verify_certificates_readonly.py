"""Exact-rational, cross-precision, and mutant replay of the stored certificates."""
from __future__ import annotations

import copy
import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODULES = [0, 15, 25]
PIVOTS = [20, 6, 13, 14, 0, 1, 3, 7, 8, 17, 10, 12, 5, 2, 16]
DIRECTIONS = {0: [6, 0, 1, 3, 5, 2], 15: [6, 0, 1, 3, 5], 25: [6, 0, 3, 2]}


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def interval(record: dict) -> tuple[Fraction, Fraction]:
    scale = 1 << int(record["scale_bits"])
    lo, hi = Fraction(int(record["lo_scaled"]), scale), Fraction(int(record["hi_scaled"]), scale)
    require(lo <= hi, "interval order")
    return lo, hi


def iadd(a, b):
    return a[0] + b[0], a[1] + b[1]


def imul(a, b):
    products = (a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1])
    return min(products), max(products)


def abs_upper(a):
    return max(abs(a[0]), abs(a[1]))


def exact_q(cert: dict) -> Fraction:
    matrix = [[interval(value) for value in row] for row in cert["minor"]]
    y = [[interval(value) for value in row] for row in cert["minor_preconditioner"]]
    worst = Fraction(0)
    for i in range(15):
        row_sum = Fraction(0)
        for j in range(15):
            total = (Fraction(0), Fraction(0))
            for k in range(15):
                total = iadd(total, imul(y[i][k], matrix[k][j]))
            target = (Fraction(1), Fraction(1)) if i == j else (Fraction(0), Fraction(0))
            residual = (target[0] - total[1], target[1] - total[0])
            row_sum += abs_upper(residual)
        worst = max(worst, row_sum)
    return worst


def y_norm(cert: dict) -> Fraction:
    return max(sum((abs_upper(interval(value)) for value in row), Fraction(0))
               for row in cert["minor_preconditioner"])


def qualify(cert: dict, bits: int) -> None:
    require(cert.get("schema") == "prl-v21-three-module-quantitative-radius-v1", "schema")
    require(cert.get("classification") == "THREE_MODULE_QUANTITATIVE_PATH_LIFTING_PASS", "class")
    require(cert.get("bits") == bits and cert.get("modules") == MODULES, "identity")
    require(cert.get("module_count") == 3 and cert.get("primitive_count") == 45, "resources")
    require(cert.get("pivot_columns") == PIVOTS, "pivots")
    require(Fraction(cert["input_control_radius"]) == Fraction(4, 10**16), "rho")
    require(Fraction(cert["dependent_remainder_radius"]) == Fraction(1, 10**20), "zeta")
    records = cert.get("module_records", [])
    require([row.get("module") for row in records] == MODULES, "module records")
    for row in records:
        require(row.get("directions") == DIRECTIONS[row["module"]], "directions")
        require(Fraction(row["continuation_q_upper"]) < 1, "module q")
        require(Fraction(row["continuation_ratio_max"]) < 1, "module image")
        require(Fraction(row["implicit_q_upper"]) < 1 and row.get("pass") is True, "implicit")
        require(interval(row["hidden_chart_abs"])[0] > 0, "chart")
    replay_q = exact_q(cert)
    stored_q = Fraction(cert["uniform_neumann_q_upper"])
    require(replay_q <= stored_q < 1, "Neumann replay")
    yn = y_norm(cert)
    require(yn == Fraction(cert["preconditioner_infinity_norm_upper"]), "Y norm")
    delta = Fraction(4, 10**16) * (1 - stored_q) / yn
    require(delta == Fraction(cert["principal_log_frobenius_radius_lower"]), "delta formula")
    require(delta > Fraction("3.4e-17") and cert.get("strictly_above_declared_bound") is True,
            "strict radius")
    require(cert.get("pass") is True, "pass")


def qualify_pair(low: dict, high: dict) -> None:
    qualify(low, 320)
    qualify(high, 512)
    for field in ("minor",):
        for i in range(15):
            for j in range(15):
                outer, inner = interval(low[field][i][j]), interval(high[field][i][j])
                require(outer[0] <= inner[0] <= inner[1] <= outer[1], "minor nesting")
    for i in range(4):
        for j in range(4):
            for part in ("re", "im"):
                outer = interval(low["composite_endpoint"][i][j][part])
                inner = interval(high["composite_endpoint"][i][j][part])
                require(outer[0] <= inner[0] <= inner[1] <= outer[1], "endpoint nesting")


def main() -> None:
    low = json.loads((ROOT / "certificates" / "three_module_radius_320.json").read_text(encoding="utf-8"))
    high = json.loads((ROOT / "certificates" / "three_module_radius_512.json").read_text(encoding="utf-8"))
    qualify_pair(low, high)
    mutants = []
    for mutate in (
        lambda a, b: a.update(modules=[0, 15, 24]),
        lambda a, b: a.update(pivot_columns=list(reversed(PIVOTS))),
        lambda a, b: a["module_records"][0].update(directions=[0, 1, 2, 3, 5, 6]),
        lambda a, b: a["module_records"][0].update(continuation_ratio_max="1"),
        lambda a, b: a.update(uniform_neumann_q_upper="1"),
        lambda a, b: a.update(principal_log_frobenius_radius_lower="1/10"),
        lambda a, b: a.update(strictly_above_declared_bound=False),
        lambda a, b: b["minor"][0][0].update(lo_scaled="0", hi_scaled="0"),
        lambda a, b: b["composite_endpoint"][0][0]["re"].update(lo_scaled="0", hi_scaled="0"),
        lambda a, b: a.update(primitive_count=44),
        lambda a, b: a.update(dependent_remainder_radius="1e-19"),
        lambda a, b: a.update({"pass": False}),
    ):
        a, b = copy.deepcopy(low), copy.deepcopy(high)
        mutate(a, b)
        mutants.append((a, b))
    rejected = 0
    for pair in mutants:
        try:
            qualify_pair(*pair)
        except (ValueError, ZeroDivisionError):
            rejected += 1
    require(rejected == len(mutants), f"mutants {rejected}/{len(mutants)}")
    print("PRL_V21_THREE_MODULE_RADIUS_CERTIFICATES_READONLY_PASS "
          f"MINOR_NESTING=225/225 ENDPOINT_NESTING=16/16 MUTANTS={rejected}/{len(mutants)}")


if __name__ == "__main__":
    main()
