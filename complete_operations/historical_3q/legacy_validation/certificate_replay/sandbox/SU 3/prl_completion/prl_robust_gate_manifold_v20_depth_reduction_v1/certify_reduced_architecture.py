"""Outward-rounded rank and entanglement certificate for the selected 3-module word."""
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
UPGRADE = PRL / "prl_robust_gate_manifold_v18_compiler_upgrade_v1"
sys.path.insert(0, str(UPGRADE))

from certify_composite_endpoint_rank import (  # noqa: E402
    DIContext,
    cdagger,
    ceye,
    ci_record,
    cmatmul,
    coordinates,
    certify_minor,
    hermitian_traceless,
    identity,
    module_branch_tangents,
    pauli_basis,
    ri_record,
)

SEARCH = ROOT / "results" / "depth_reduction_search_128.json"
EXPECTED_SEARCH = (32899, "8CFE02C455421FB9FD3878435148AD5A58F9AFFE35A9C3A96D7C9630DF0B78B6")
MODULES = (0, 15, 25)
PIVOTS = (20, 6, 13, 14, 0, 1, 3, 7, 8, 17, 10, 12, 5, 2, 16)
ENT_ROWS = (0, 3)
ENT_COLS = (1, 3)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def realigned(gate):
    ctx = gate[0][0].ctx
    out = [[ctx.complex() for _ in range(4)] for _ in range(4)]
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for d in range(2):
                    out[2*a+c][2*b+d] = gate[2*a+b][2*c+d]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, choices=(320, 512), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing overwrite: {output}")
    search_data = SEARCH.read_bytes()
    if (len(search_data), sha(search_data)) != EXPECTED_SEARCH:
        raise ValueError("search binding")
    search = json.loads(search_data)
    selected = search.get("selected_candidate", {})
    if tuple(selected.get("modules", ())) != MODULES or tuple(selected.get("pivot_columns", ())) != PIVOTS:
        raise ValueError("candidate contract")

    started = time.perf_counter()
    ctx = DIContext(args.bits)
    digits = int(args.bits * 0.30103) + 45
    gates, tangent_sets, roots, branches = [], [], [], []
    with mp.workdps(digits + 25):
        for local, number in enumerate(MODULES):
            root_path = UPGRADE / "candidates" / f"generic_product_module_{number:02d}_{args.bits}.json"
            root = json.loads(root_path.read_text(encoding="utf-8"))
            if not (root.get("pass") is True and root.get("bits") == args.bits and
                    root.get("module") == number and
                    root.get("certificate", {}).get("strict_interior") is True):
                raise ValueError(f"root admission {number}")
            gate, tangents, q, correction, chart = module_branch_tangents(root, ctx, digits)
            gates.append(gate)
            tangent_sets.append(tangents)
            roots.append(identity(root_path))
            branches.append({"module": number, "dependent_neumann_q_upper": str(q),
                             "implicit_correction_upper": str(correction),
                             "hidden_chart_abs": ri_record(chart)})
            print(f"DEPTH_CERT_BRANCH {args.bits} {local+1}/{len(MODULES)} module={number}", flush=True)

        suffix = [ceye(ctx, 4) for _ in range(len(MODULES) + 1)]
        for local in range(len(MODULES) - 1, -1, -1):
            suffix[local] = cmatmul(gates[local], suffix[local + 1])
        basis = pauli_basis(ctx)
        full = [[ctx.zero for _ in range(7 * len(MODULES))] for _ in range(15)]
        for local in range(len(MODULES)):
            for direction in range(7):
                transported = cmatmul(cmatmul(cdagger(suffix[local + 1]),
                                              tangent_sets[local][direction]),
                                      suffix[local + 1])
                column = coordinates(hermitian_traceless(transported), basis)
                for row in range(15):
                    full[row][7 * local + direction] = column[row]
        minor = [[full[row][column] for column in PIVOTS] for row in range(15)]
        q_minor, preconditioner = certify_minor(minor, digits)

        re = realigned(suffix[0])
        r0, r1 = ENT_ROWS
        c0, c1 = ENT_COLS
        ent = re[r0][c0] * re[r1][c1] - re[r0][c1] * re[r1][c0]
        ent_abs = ent.abs()
        if ent_abs.lower_fraction() <= Fraction(1, 2):
            raise ValueError(f"entanglement witness too weak: {ent_abs.lower_fraction()}")

    payload = {
        "schema": "prl-v20-three-module-open-entangling-family-interval-v1",
        "classification": "THREE_MODULE_MINIMAL_OPEN_ENTANGLING_FAMILY_PASS",
        "bits": args.bits,
        "modules": list(MODULES),
        "module_count": 3,
        "primitive_count": 45,
        "module_dimension": 7,
        "target_group_dimension": 15,
        "dimension_lower_bound_modules": 3,
        "pivot_columns": list(PIVOTS),
        "pivot_module_directions": [list(item) for item in selected["pivot_module_directions"]],
        "root_sources": roots,
        "branch_certificates": branches,
        "minor": [[ri_record(value) for value in row] for row in minor],
        "minor_preconditioner": [[ri_record(value) for value in row] for row in preconditioner],
        "minor_neumann_q_upper": str(q_minor),
        "minor_nonsingular": q_minor < 1,
        "composite_differential_rank": 15 if q_minor < 1 else None,
        "endpoint": [[ci_record(value) for value in row] for row in suffix[0]],
        "entangling_minor_rows": list(ENT_ROWS),
        "entangling_minor_cols": list(ENT_COLS),
        "entangling_minor": ci_record(ent),
        "entangling_minor_abs": ri_record(ent_abs),
        "entangling_minor_abs_gt_one_half": ent_abs.lower_fraction() > Fraction(1, 2),
        "operator_schmidt_rank_gt_one": ent_abs.lower_fraction() > 0,
        "search_source": {"path": SEARCH.relative_to(ROOT).as_posix(),
                          "bytes": len(search_data), "sha256": sha(search_data)},
        "elapsed_seconds": time.perf_counter() - started,
        "pass": q_minor < 1 and ent_abs.lower_fraction() > Fraction(1, 2),
        "claim_boundary": [
            "The three-module count is minimal only within compositions of seven-dimensional certified module branches, by 2*7<15.",
            "The certificate establishes a relative-open entangling SU(4) family around this center, not a named target or shortest primitive word.",
            "No hardware, noise beyond the certified first-order model, quantitative target radius, or publication claim follows.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
                      encoding="utf-8", newline="\n")
    print("PRL_V20_THREE_MODULE_INTERVAL_PASS", args.bits,
          f"q={q_minor}", f"ent_lo={ent_abs.lower_fraction()}",
          f"elapsed={payload['elapsed_seconds']:.3f}s")


if __name__ == "__main__":
    main()
