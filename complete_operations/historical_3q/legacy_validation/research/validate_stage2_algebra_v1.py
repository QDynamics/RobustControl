"""Bounded exact-modular Lie-rank witnesses; no optimization or legacy reruns.

The bracket is represented on anti-Hermitian Pauli strings, with integer
coefficients and truncated polynomial errors. A rank modulo a prime witnesses
a rational rank lower bound. Equality uses the separately proved analytic
upper bound, not an extrapolation of these finite cases.
"""
import hashlib
import json
import math
import platform
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
RUN = BASE / "runs" / "stage2_algebra_20260924_v1"
PRIME = 1000003
BUDGET = 120.0


def write_json(name, value):
    (RUN / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def monomials(q):
    return [(a, total - a) for total in range(q + 1) for a in range(total + 1)]


def primitive_terms(n, q, ideal=False, uniform=False):
    mons = monomials(q)
    mid = {m: i for i, m in enumerate(mons)}
    size = 4 ** n

    def lift(paulis, profile):
        terms = {p: c for p, c in paulis.items()}
        if q and profile != (0, 0):
            for mu, value in enumerate(profile):
                if value:
                    degree = (1, 0) if mu == 0 else (0, 1)
                    for p, c in paulis.items():
                        terms[mid[degree] * size + p] = c * value
        return terms

    control_profile = (0, 0) if ideal else (1, 0)
    generators = [lift({1 << (n + a): 1}, control_profile) for a in range(n)]
    # Multiplying exchange by two does not change the real Lie algebra.
    for a in range(n - 1):
        xx = (1 << a) | (1 << (a + 1))
        yy = xx | (xx << n)
        generators.append(lift({xx: 1, yy: 1}, control_profile))
    zz = {(1 << (n + a)) | (1 << (n + b)): 1
          for a in range(n) for b in range(a + 1, n)
          if uniform or (a == 0 and b == 1)}
    generators.append(lift(zz, (0, 1)))
    return mons, generators


def adjoint_maps(n, mons, terms):
    size = 4 ** n
    mask = (1 << n) - 1
    mid = {m: i for i, m in enumerate(mons)}
    maps = []
    for index, coefficient in terms.items():
        gen_mon, pauli = divmod(index, size)
        gx, gz = pauli & mask, pauli >> n
        gphase = (gx & gz).bit_count()
        source, dest, signs = [], [], []
        for mi, (a, b) in enumerate(mons):
            total_mon = (a + mons[gen_mon][0], b + mons[gen_mon][1])
            if total_mon not in mid:
                continue
            for p in range(1, size):
                px, pz = p & mask, p >> n
                rx, rz = gx ^ px, gz ^ pz
                phase = (gphase + (px & pz).bit_count()
                         - (rx & rz).bit_count()
                         + 2 * (gz & px).bit_count()) % 4
                if phase % 2:
                    source.append(mi * size + p)
                    dest.append(mid[total_mon] * size + (rx | (rz << n)))
                    signs.append(coefficient * (2 if phase == 1 else -2))
        maps.append((np.array(source, dtype=np.int64),
                     np.array(dest, dtype=np.int64),
                     np.array(signs, dtype=np.int64)))
    return maps


def expected_rank(n, q, ideal, uniform):
    if uniform:
        return 2 + (n * n - 1) * (1 if ideal else q + 1)
    semisimple = sum(math.comb(n, k) ** 2 - 1 for k in range(1, n))
    if n == 2:
        return 2 + semisimple * (1 if ideal else q + 1)
    return 2 + semisimple * (q + 1 if ideal else math.comb(q + 2, 2))


def closure(n, q, ideal, uniform, started):
    mons, generators = primitive_terms(n, q, ideal, uniform)
    maps = [adjoint_maps(n, mons, g) for g in generators]
    size = len(mons) * 4 ** n
    echelon, queue, pivots = {}, [], []
    bracket_count = 0

    def insert(v):
        v %= PRIME
        for pivot in sorted(echelon):
            if v[pivot]:
                v = (v - v[pivot] * echelon[pivot]) % PRIME
        nonzero = np.flatnonzero(v)
        if not len(nonzero):
            return
        pivot = int(nonzero[0])
        v = v * pow(int(v[pivot]), -1, PRIME) % PRIME
        echelon[pivot] = v
        pivots.append(pivot)
        queue.append(v)

    for g in generators:
        row = np.zeros(size, dtype=np.int64)
        for index, value in g.items():
            row[index] = value
        insert(row)
    cursor = 0
    while cursor < len(queue):
        if time.perf_counter() - started > BUDGET:
            raise TimeoutError(f"Budget reached at N={n}, q={q}, rank={len(queue)}")
        row = queue[cursor]
        cursor += 1
        for mapping in maps:
            out = np.zeros(size, dtype=np.int64)
            for src, dst, signs in mapping:
                # Each map has distinct destinations, so advanced += is exact.
                out[dst] += signs * row[src]
            insert(out)
            bracket_count += 1
    wanted = expected_rank(n, q, ideal, uniform)
    return {"N": n, "q": q, "ideal_controls": ideal,
            "uniform_all_to_all_B": uniform,
            "rank_mod_prime": len(queue), "analytic_upper_bound": wanted,
            "status": "EXACT_RANK_WITNESS_PASS" if len(queue) == wanted else "FAIL",
            "brackets_evaluated": bracket_count,
            "echelon_sha256": hashlib.sha256(
                np.array([echelon[k] for k in sorted(echelon)], dtype="<i8").tobytes()
            ).hexdigest(), "pivot_columns": sorted(pivots)}


def check_pauli_brackets():
    # Independent matrix check of all one- and two-qubit basis brackets.
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.diag([1, -1]).astype(complex)
    local = {(0, 0): I, (1, 0): X, (1, 1): Y, (0, 1): Z}
    max_error = 0.0
    for n in (1, 2):
        matrices = []
        for code in range(4 ** n):
            mat = np.ones((1, 1), dtype=complex)
            for a in reversed(range(n)):
                mat = np.kron(mat, local[((code >> a) & 1, (code >> (n + a)) & 1)])
            matrices.append(-1j * mat)
        for g in range(4 ** n):
            maps = adjoint_maps(n, [(0, 0)], {g: 1})
            mapping = {int(src): (int(dst), int(sign))
                       for src, dst, sign in zip(*maps[0])}
            for p in range(4 ** n):
                expected = matrices[g] @ matrices[p] - matrices[p] @ matrices[g]
                dst, sign = mapping.get(p, (0, 0))
                max_error = max(max_error, float(np.max(np.abs(expected - sign * matrices[dst]))))
    if max_error != 0:
        raise AssertionError(f"Pauli sign convention failed: {max_error}")
    return {"max_matrix_residual": max_error, "status": "EXACT_INTEGER_CHECK_PASS"}


def main():
    RUN.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    cases = [(n, q, False, False) for n, q in
             [(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1)]]
    cases += [(3, 1, True, False), (4, 1, True, False),
              (3, 1, False, True), (4, 1, False, True)]
    config = {"prime": PRIME, "budget_seconds": BUDGET, "cases": cases,
              "arithmetic": "integer Pauli brackets and exact prime-field elimination",
              "meaning": "rank lower witnesses matched to analytic upper bounds",
              "optimization": False, "legacy_rerun": False,
              "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json("config.json", config)
    write_json("versions.json", {"python": platform.python_version(), "numpy": np.__version__,
                                "platform": platform.platform()})
    results, failures = [], []
    convention = None
    try:
        convention = check_pauli_brackets()
        for n, q, ideal, uniform in cases:
            result = closure(n, q, ideal, uniform, started)
            results.append(result)
            if result["status"] != "EXACT_RANK_WITNESS_PASS":
                failures.append(result)
            print(json.dumps({k: v for k, v in result.items()
                              if k not in ("pivot_columns", "echelon_sha256")}), flush=True)
            write_json("checkpoint.json", results)
    except Exception as exc:
        failures.append({"error": str(exc), "traceback": traceback.format_exc()})
    complete = len(results) == len(cases)
    write_json("results.json", {"status": "PASS" if complete and not failures else "FAIL",
                                "pauli_convention": convention, "cases": results})
    write_json("failures.json", failures)
    write_json("runtime.json", {"wall_seconds": time.perf_counter() - started,
                                "complete": complete})
    files = [Path(__file__).resolve(), BASE / "STAGE2_PROTOCOL_20260924_v1.md"]
    files += sorted(p for p in RUN.iterdir() if p.is_file())
    write_json("hashes.json", {str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in files})
    if failures or not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
