"""Build the final quantitative three-module path-lifting certificate."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRL = ROOT.parent
QR1 = PRL / "prl_robust_gate_manifold_v19_quantitative_radius_v1"
UPGRADE = PRL / "prl_robust_gate_manifold_v18_compiler_upgrade_v1"
V21 = PRL / "prl_robust_gate_manifold_v21"
DEPTH = PRL / "prl_robust_gate_manifold_v20_depth_reduction_v1"
sys.path.insert(0, str(QR1 / "builders"))
sys.path.insert(0, str(UPGRADE))

import build_uniform_module_tangents as uniform  # noqa: E402
from certify_composite_endpoint_rank import (  # noqa: E402
    DIContext, cdagger, ceye, ci_record, cmatmul, coordinates, hermitian_traceless,
    pauli_basis, parse_ci, ri_record, certify_minor,
)

MODULES = (0, 15, 25)
PIVOTS = (20, 6, 13, 14, 0, 1, 3, 7, 8, 17, 10, 12, 5, 2, 16)
MODULE_DIRECTIONS = {0: [6, 0, 1, 3, 5, 2], 15: [6, 0, 1, 3, 5], 25: [6, 0, 3, 2]}
RHO = Fraction(4, 10**16)
ZETA = Fraction(1, 10**20)
EXPECTED_BINDINGS = {
    V21 / "package_manifest_v21.json": "672A9F9D7A67EF0378925A5C671A78133FFC831D8A746D87FF485213059F299D",
    DEPTH / "output_manifest_depth_reduction_v1.json": "37FA83389EDFA853D7C3B8D1FE789EA46ADE550452A2008F2987A8588CFF6B09",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def identity(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": path.relative_to(PRL.parent).as_posix(), "bytes": len(data), "sha256": sha(data)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, choices=(320, 512), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing overwrite: {output}")
    for path, expected in EXPECTED_BINDINGS.items():
        if sha(path.read_bytes()) != expected:
            raise ValueError(f"binding mismatch: {path}")

    uniform.MODULE_DIRECTIONS = MODULE_DIRECTIONS
    records = [uniform.build_record(module, args.bits, str(RHO), str(ZETA)) for module in MODULES]
    if not all(record["pass"] for record in records):
        raise ValueError("module continuation admission")

    ctx = DIContext(args.bits)
    digits = int(args.bits * 0.30103) + 45
    gates = [
        [[parse_ci(ctx, entry) for entry in row] for row in record["gate"]]
        for record in records
    ]
    tangents = {
        record["module"]: [
            [[parse_ci(ctx, entry) for entry in row] for row in tangent]
            for tangent in record["tangents"]
        ] for record in records
    }
    position = {
        (record["module"], direction): record["directions"].index(direction)
        for record in records for direction in record["directions"]
    }
    suffix = [ceye(ctx, 4) for _ in range(4)]
    for local in range(2, -1, -1):
        suffix[local] = cmatmul(gates[local], suffix[local + 1])
    basis = pauli_basis(ctx)
    minor = [[ctx.zero for _ in range(15)] for _ in range(15)]
    for column, global_column in enumerate(PIVOTS):
        local, direction = divmod(global_column, 7)
        module = MODULES[local]
        tangent = tangents[module][position[(module, direction)]]
        transported = cmatmul(cmatmul(cdagger(suffix[local + 1]), tangent), suffix[local + 1])
        coordinate = coordinates(hermitian_traceless(transported), basis)
        for row in range(15):
            minor[row][column] = coordinate[row]
    q, y = certify_minor(minor, digits)
    y_norm = max(sum((entry.abs().upper_fraction() for entry in row), Fraction(0)) for row in y)
    delta = RHO * (1 - q) / y_norm
    declared = Fraction(34, 10**18)
    if not delta > declared:
        raise ValueError("declared target radius is not strict")

    payload = {
        "schema": "prl-v21-three-module-quantitative-radius-v1",
        "classification": "THREE_MODULE_QUANTITATIVE_PATH_LIFTING_PASS",
        "bits": args.bits,
        "modules": list(MODULES),
        "module_count": 3,
        "primitive_count": 45,
        "pivot_columns": list(PIVOTS),
        "pivot_module_directions": [[MODULES[col // 7], col % 7] for col in PIVOTS],
        "input_control_radius": str(RHO),
        "dependent_remainder_radius": str(ZETA),
        "module_records": records,
        "composite_endpoint": [[ci_record(value) for value in row] for row in suffix[0]],
        "minor": [[ri_record(value) for value in row] for row in minor],
        "minor_preconditioner": [[ri_record(value) for value in row] for row in y],
        "uniform_neumann_q_upper": str(q),
        "preconditioner_infinity_norm_upper": str(y_norm),
        "principal_log_frobenius_radius_lower": str(delta),
        "declared_decimal_lower_bound": "3.4e-17",
        "strictly_above_declared_bound": delta > declared,
        "bound_sources": [identity(path) for path in EXPECTED_BINDINGS],
        "pass": q < 1 and delta > declared,
        "claim_boundary": [
            "The ball is centered at the certified three-module entangling word and lies in a fixed 15-dimensional branch slice.",
            "The three modules retain exact hidden closure and the two certified first-order error cancellations along the continued branches.",
            "The bound is rigorous but limited by interval wrapping; it is not a global compiler, shortest primitive word, named-target, hardware, or higher-order claim.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")
    print("PRL_V21_THREE_MODULE_QUANTITATIVE_RADIUS_PASS", args.bits,
          f"q={float(q):.12g}", f"delta={float(delta):.12g}")


if __name__ == "__main__":
    main()
