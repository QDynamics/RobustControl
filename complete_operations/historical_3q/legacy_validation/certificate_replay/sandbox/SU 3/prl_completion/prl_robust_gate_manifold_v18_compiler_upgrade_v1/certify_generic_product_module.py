"""Create-new strict Krawczyk certificate for one generic-product module."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRL = ROOT.parent
EXT = PRL / "three_qubit_symmetry_extension"
SOURCE = EXT / "compiler_depth_reduction_v1" / "candidates" / "polished_generic_product_v1.json"
ROOT_CERT = EXT / "impact_robust_i2_v1" / "certificates" / "i2_strict_interval_root.json"
BASE_WORKER = EXT / "named_parity_interval_v1" / "builders" / "certify_parity_module.py"
EXPECTED_SOURCE = (108315, "785EF3930E278588175F2DF119AF33624B2B029F75044EB0D45D5D730090EA4B")


def load_base():
    spec = importlib.util.spec_from_file_location("cu1_base_module_certifier", BASE_WORKER)
    if spec is None or spec.loader is None:
        raise ImportError(BASE_WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identity(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {"path": path.relative_to(PRL.parent).as_posix(), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest().upper()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=int, required=True)
    parser.add_argument("--bits", type=int, choices=(320, 512), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing overwrite: {output}")
    source_bytes = SOURCE.read_bytes()
    if (len(source_bytes), hashlib.sha256(source_bytes).hexdigest().upper()) != EXPECTED_SOURCE:
        raise ValueError("generic-product source binding")
    source = json.loads(source_bytes)
    root = json.loads(ROOT_CERT.read_text(encoding="utf-8"))
    if source.get("target") != "generic_product" or source.get("module_count") != 28:
        raise ValueError("generic-product source semantics")
    if not 0 <= args.module < 28:
        raise ValueError("module index")
    selected = tuple(root["selected_variable_indices"])
    free = tuple(root["fixed_variable_indices"])
    rows = tuple(root["independent_rows"])
    if not (len(selected) == len(rows) == 37 and len(free) == 7):
        raise ValueError("root coordinate contract")
    record = source["modules"][args.module]
    base = load_base()
    started = time.perf_counter()
    fractions, history, rebuilds = base.refine(record["parameters"], selected, free,
                                               rows, args.bits)
    refined_seconds = time.perf_counter() - started
    certificate = base.certify(fractions, selected, args.bits)
    total_seconds = time.perf_counter() - started
    preserved = all(fractions[index] == Fraction(record["parameters"][index]) for index in free)
    passed = certificate["pass"] and preserved
    payload = {
        "schema": "prl-v18-generic-product-module-strict-root-v1",
        "classification": "GENERIC_PRODUCT_MODULE_STRICT_ROOT_PASS" if passed
                          else "GENERIC_PRODUCT_MODULE_STRICT_ROOT_FAIL",
        "source_candidate": identity(SOURCE),
        "source_root_certificate": identity(ROOT_CERT),
        "source_certifier": identity(BASE_WORKER),
        "module": args.module,
        "bits": args.bits,
        "selected_variable_indices": list(selected),
        "fixed_variable_indices": list(free),
        "free_coordinates_preserved": preserved,
        "refinement_history": history,
        "finite_difference_jacobian_rebuilds": rebuilds,
        "refined_parameter_fractions": [str(value) for value in fractions],
        "refinement_seconds": refined_seconds,
        "total_seconds": total_seconds,
        "certificate": certificate,
        "pass": passed,
        "claim_boundary": [
            "Finite differences refine the center but do not support the proof.",
            "The strict root claim is supplied only by the outward-rounded Krawczyk inclusion.",
            "This file certifies one fixed-free-coordinate module, not the composed generic target word.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
                      encoding="utf-8", newline="\n")
    if not passed:
        raise SystemExit(f"module {args.module} at {args.bits} bits failed")
    print("GENERIC_PRODUCT_MODULE_STRICT_ROOT_PASS", args.module, args.bits,
          f"refine={refined_seconds:.3f}s", f"total={total_seconds:.3f}s")


if __name__ == "__main__":
    main()
