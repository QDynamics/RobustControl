"""Portable, fixed-candidate N=3 interval replay.

This script never searches or optimizes.  It copies the bundled isolated source
tree to a new path, refuses an existing output root, reserves every output
before spawning its worker, and runs the selected 320/512-bit chain serially.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "sandbox" / "SU 3"
PRODUCED_IN_SOURCE = {
    "prl_completion/three_qubit_symmetry_extension/impact_robust_i2_v1/certificates/i2_strict_interval_root.json",
    "prl_completion/three_qubit_symmetry_extension/prl_upgrade_robust_family_b_v1/certificates/b1_interval_augmented_rank.json",
    *{f"prl_completion/prl_robust_gate_manifold_v18_compiler_upgrade_v1/candidates/generic_product_module_{m:02d}_{b}.json"
      for m in (0, 15, 25) for b in (320, 512)},
}


def normal(path: str | Path) -> Path:
    """Normalize an ordinary or extended Windows path to an ordinary Path."""
    text = str(path)
    if os.name == "nt" and text.startswith("\\\\?\\UNC\\"):
        text = "\\\\" + text[8:]
    elif os.name == "nt" and text.startswith("\\\\?\\"):
        text = text[4:]
    return Path(text).resolve()


def long(path: str | Path) -> str:
    """Return an idempotent extended Windows path, including UNC paths."""
    text = str(path)
    if os.name != "nt" or text.startswith("\\\\?\\"):
        return text
    text = str(normal(text))
    return "\\\\?\\UNC\\" + text[2:] if text.startswith("\\\\") else "\\\\?\\" + text


def exists(path: str | Path) -> bool:
    return os.path.exists(long(path))


def make_dir(path: str | Path) -> None:
    os.makedirs(long(path), exist_ok=False)


def write_utf8(path: str | Path, text: str, exclusive: bool = False) -> None:
    mode = "x" if exclusive else "w"
    with open(long(path), mode, encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(long(path), "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def reject_existing(path: Path) -> None:
    if exists(path):
        raise SystemExit(f"refusing existing output: {path}")


def copy_source(destination: Path) -> None:
    if not exists(SOURCE):
        raise SystemExit(f"bundled source missing: {SOURCE}")
    reject_existing(destination)
    def ignore(directory: str, names: list[str]) -> set[str]:
        base = normal(directory)
        ignored: set[str] = set()
        for name in names:
            if name == "__pycache__" or name.endswith(".pyc"):
                ignored.add(name)
                continue
            candidate = base / name
            try:
                relative = normal(candidate).relative_to(normal(SOURCE)).as_posix()
            except ValueError:
                continue
            if relative in PRODUCED_IN_SOURCE:
                ignored.add(name)
        return ignored
    shutil.copytree(long(SOURCE), long(destination), ignore=ignore, copy_function=shutil.copy2)
    closure = destination / "prl_completion/three_qubit_symmetry_extension/impact_robust_i2_v1/runtime/dyadic_interval.py"
    if not exists(closure):
        raise SystemExit("path-closure dependency dyadic_interval.py missing after copy")


def reserve(lock_dir: Path, label: str, output: Path) -> None:
    reject_existing(output)
    os.makedirs(long(lock_dir), exist_ok=True)
    lock = lock_dir / f"{label}.lock"
    write_utf8(lock, json.dumps({"stage": label, "output": str(output)}, indent=2) + "\n", exclusive=True)


def run(label: str, command: list[str], output: Path, lock_dir: Path, log_dir: Path, marker: str) -> float:
    reserve(lock_dir, label, output)
    started = time.perf_counter()
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    elapsed = time.perf_counter() - started
    write_utf8(log_dir / f"{label}.log", result.stdout)
    if result.returncode:
        raise SystemExit(f"{label} failed with exit {result.returncode}; retained {log_dir / (label + '.log')}")
    if not any(line.split() and line.split()[0] == marker for line in result.stdout.splitlines()) or not exists(output):
        raise SystemExit(f"{label} did not report required PASS marker or create its reserved output")
    expected = {
        "IMPACT_I2_STRICT_INTERVAL_ROOT_PASS": "STRICT_INTERVAL_ROOT_PASS",
        "PRL_UPGRADE_B1_STRICT_AUGMENTED_RANK_PASS": "STRICT_AUGMENTED_RANK_PASS",
        "GENERIC_PRODUCT_MODULE_STRICT_ROOT_PASS": "GENERIC_PRODUCT_MODULE_STRICT_ROOT_PASS",
        "PRL_V20_THREE_MODULE_INTERVAL_PASS": "THREE_MODULE_MINIMAL_OPEN_ENTANGLING_FAMILY_PASS",
        "PRL_V21_THREE_MODULE_QUANTITATIVE_RADIUS_PASS": "THREE_MODULE_QUANTITATIVE_PATH_LIFTING_PASS",
    }
    with open(long(output), encoding="utf-8") as handle:
        certificate = json.load(handle)
    if certificate.get("pass") is not True or certificate.get("classification") != expected[marker]:
        raise SystemExit(f"{label} certificate has no exact matching PASS classification")
    return elapsed


def import_preflight(work: Path) -> None:
    targets = [
        "prl_completion/three_qubit_symmetry_extension/impact_robust_i2_v1/builders/build_i2_interval_root.py",
        "prl_completion/three_qubit_symmetry_extension/prl_upgrade_robust_family_b_v1/builders/build_b1_interval_rank_certificate.py",
        "prl_completion/prl_robust_gate_manifold_v18_compiler_upgrade_v1/certify_generic_product_module.py",
        "prl_completion/prl_robust_gate_manifold_v20_depth_reduction_v1/certify_reduced_architecture.py",
        "prl_completion/prl_robust_gate_manifold_v21_three_module_radius_v1/build_three_module_radius_certificate.py",
    ]
    for relative in targets:
        script = work / relative
        command = [sys.executable, "-B", "-c",
                   "import importlib.util,sys; s=importlib.util.spec_from_file_location('x',sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)",
                   long(script)]
        if subprocess.run(command).returncode:
            raise SystemExit(f"import preflight failed: {relative}")


def assert_produced_outputs_absent(work: Path) -> None:
    copied = [relative for relative in PRODUCED_IN_SOURCE if exists(work / relative)]
    if copied:
        raise SystemExit("prepare copy retained produced outputs: " + ", ".join(sorted(copied)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True, help="new, nonexistent replay root")
    parser.add_argument("--prepare-only", action="store_true", help="copy and import-check only")
    args = parser.parse_args()
    try:
        import mpmath  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"mpmath preflight failed under {sys.executable}: {exc}") from exc
    out = args.out.resolve()
    if normal(out) == normal(SOURCE) or normal(SOURCE) in normal(out).parents:
        raise SystemExit("output root must be outside the bundled source tree")
    reject_existing(out)
    make_dir(out)
    work = out / "SU 3"
    copy_source(work)
    assert_produced_outputs_absent(work)
    import_preflight(work)
    if args.prepare_only:
        print("PREPARE_ONLY_PASS", out)
        return
    logs, locks, fresh = out / "logs", out / "locks", out / "fresh_certificates"
    make_dir(logs); make_dir(fresh)
    py = sys.executable
    i2 = work / "prl_completion/three_qubit_symmetry_extension/impact_robust_i2_v1"
    b1 = work / "prl_completion/three_qubit_symmetry_extension/prl_upgrade_robust_family_b_v1"
    v18 = work / "prl_completion/prl_robust_gate_manifold_v18_compiler_upgrade_v1"
    v20 = work / "prl_completion/prl_robust_gate_manifold_v20_depth_reduction_v1"
    v21 = work / "prl_completion/prl_robust_gate_manifold_v21_three_module_radius_v1"
    timings = {}
    timings["I2"] = run("I2", [py, "-B", long(i2 / "builders/build_i2_interval_root.py")],
        i2 / "certificates/i2_strict_interval_root.json", locks, logs, "IMPACT_I2_STRICT_INTERVAL_ROOT_PASS")
    timings["B1"] = run("B1", [py, "-B", long(b1 / "builders/build_b1_interval_rank_certificate.py")],
        b1 / "certificates/b1_interval_augmented_rank.json", locks, logs, "PRL_UPGRADE_B1_STRICT_AUGMENTED_RANK_PASS")
    for module in (0, 15, 25):
        for bits in (320, 512):
            target = v18 / f"candidates/generic_product_module_{module:02d}_{bits}.json"
            label = f"M{module:02d}_{bits}"
            timings[label] = run(label, [py, "-B", long(v18 / "certify_generic_product_module.py"),
                "--module", str(module), "--bits", str(bits), "--output", long(target)], target, locks, logs, "GENERIC_PRODUCT_MODULE_STRICT_ROOT_PASS")
    for bits in (320, 512):
        target = fresh / f"three_module_open_family_{bits}.json"
        label = f"R15_{bits}"
        timings[label] = run(label, [py, "-B", long(v20 / "certify_reduced_architecture.py"),
            "--bits", str(bits), "--output", long(target)], target, locks, logs, "PRL_V20_THREE_MODULE_INTERVAL_PASS")
    for bits in (320, 512):
        target = fresh / f"three_module_radius_{bits}.json"
        label = f"QRAD_{bits}"
        timings[label] = run(label, [py, "-B", long(v21 / "build_three_module_radius_certificate.py"),
            "--bits", str(bits), "--output", long(target)], target, locks, logs, "PRL_V21_THREE_MODULE_QUANTITATIVE_RADIUS_PASS")
    canonical = [i2 / "certificates/i2_strict_interval_root.json", b1 / "certificates/b1_interval_augmented_rank.json"]
    canonical += [v18 / f"candidates/generic_product_module_{m:02d}_{b}.json" for m in (0, 15, 25) for b in (320, 512)]
    canonical += [fresh / f"three_module_open_family_{b}.json" for b in (320, 512)]
    canonical += [fresh / f"three_module_radius_{b}.json" for b in (320, 512)]
    bindings = []
    for source in canonical:
        target = fresh / source.name
        if normal(source) != normal(target):
            shutil.copy2(long(source), long(target))
        if sha256(source) != sha256(target):
            raise SystemExit(f"canonical hash mismatch: {source.name}")
        bindings.append({"name": source.name, "sha256": sha256(target), "bytes": os.path.getsize(long(target))})
    write_utf8(out / "fresh_certificate_bindings.json", json.dumps(bindings, indent=2) + "\n")
    write_utf8(out / "execution_receipt.json", json.dumps({"python": sys.version, "mpmath": mpmath.__version__, "timings_seconds": timings, "canonical": bindings}, indent=2) + "\n")
    print("FIXED_SELECTED_CHAIN_PASS", out)


if __name__ == "__main__":
    main()
