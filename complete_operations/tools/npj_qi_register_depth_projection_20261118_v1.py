"""Alternate endpoint restoration registered before running this script.

Reuse the byte-identical 82-evaluation finite-noise endpoints from the matched
protocol. Its generic eight-evaluation trust-region restoration missed the fixed
accuracy requirement at both widths. Keep the start set, target, error domain,
five training points, ranking, eight endpoint evaluations, seven-second budget,
four validation circuits and acceptance thresholds fixed. Replace only the
endpoint restoration by a minimum-norm Newton step on the analytic nominal
Jacobian, discarding singular values <= 1e-8 of its largest. Freeze any coordinate
whose proposed full step would violate the original box and solve again. Halve
a step until the nominal residual norm drops. Stop at residual norm 1e-11 or
the original budget. Preserve all previous results. No further fitting is part
of this diagnostic. Compare all four methods at matched physical times.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import npj_qi_register_depth_matched_20261118_v1 as matched
from npj_qi_register_depth_matched_20261118_v1 import np, pilot

PRIOR = pilot.ROOT / "runs/npj_qi_register_depth_matched_20261118_v1"


def restore(n, blocks, initial):
    start = time.perf_counter()
    p = initial.copy()
    bound = np.repeat([np.pi, 3*np.pi, .6], len(p)//3)
    residual, jac = pilot.evaluate(blocks, n, p, "nominal")
    scores = [float(np.linalg.norm(residual))]
    stop = "evaluation_limit"
    while len(scores) < 8 and time.perf_counter()-start < 7:
        if np.linalg.norm(residual) <= 1e-11:
            stop = "nominal_residual"
            break
        free = np.ones(len(p), dtype=bool)
        step = np.zeros_like(p)
        while free.any():
            u, s, vh = np.linalg.svd(jac[:, free], full_matrices=False)
            keep = s > 1e-8*s[0]
            step[:] = 0
            step[free] = -(vh[keep].T @ ((u[:, keep].T @ residual)/s[keep]))
            blocked = free & (np.abs(p+step) > bound)
            if not blocked.any():
                break
            free[blocked] = False
        if not free.any():
            stop = "box_blocked"
            break
        scale = 1.
        improved = False
        while len(scores) < 8 and time.perf_counter()-start < 7:
            trial = p + scale*step
            r_new, j_new = pilot.evaluate(blocks, n, trial, "nominal")
            value = float(np.linalg.norm(r_new))
            scores.append(value)
            if value < np.linalg.norm(residual):
                p, residual, jac = trial, r_new, j_new
                improved = True
                break
            scale /= 2
        if not improved:
            break
    if time.perf_counter()-start >= 7:
        stop = "time_limit"
    return p, {"evaluations": len(scores), "residual_norms": scores, "reason": stop,
               "wall_s": time.perf_counter()-start}


def run(n, out, seqs):
    earlier = matched.read(PRIOR/f"n{n}_summary.json")
    blocks = [pilot.Block(r, n) for r in pilot.schur_representations(n)]
    starts = {key: np.load(matched.BASE/f"n{n}_{key}.npy") for key in ("nominal", "corrected")}
    candidates = dict(starts)
    for name in starts:
        p, status = restore(n, blocks, np.load(PRIOR/f"n{n}_{name}_finite.npy"))
        candidates[name+"_restored"] = p
        np.save(out/f"n{n}_{name}_restored.npy", p, allow_pickle=False)
        pilot.dump(out/f"n{n}_{name}_restoration.json", status)
    nominal, training = {}, {}
    for name, p in candidates.items():
        gates = [b.word(p, pilot.JET[:1])[0] for b in blocks]
        nominal[name] = pilot.metrics([b.target for b in blocks], gates, [b.mult for b in blocks])[0]
        training[name] = pilot.training_score(blocks, p)
    selected = min((k for k in candidates if nominal[k]<=1e-6), key=training.get)
    p = candidates[selected]
    t, mag, charges = pilot.tensor_setup(n)
    diag, points = pilot.diagnostics(n, blocks, p, t, mag, charges, pilot.full_tensor_representation(n))
    pilot.dump(out/f"n{n}_original_validation.json", points)
    physical = matched.direct_shifted_audit(n, p, t, mag, charges, diag["by_depth"], blocks)
    by_circuit = {method: {label: {int(k): x for k,x in line.items()} for label,line in series.items()}
                  for method,series in earlier["circuit_max_D"].items()}
    grid = {label: matched.held_out(blocks, p, seq) for label,seq in seqs.items()}
    pilot.dump(out/f"n{n}_matched_all_circuits.json", grid)
    by_circuit["matched"] = {label: {k:max(r["D"] for r in rows if r["depth"]==k)
                                     for k in pilot.DEPTHS} for label, rows in grid.items()}
    info = {"N": n, "nominal_D": nominal, "training_max_D": training, "selected": selected,
            "matched": diag, "direct_physical_crosscheck": physical, "circuit_max_D": by_circuit}
    pilot.dump(out/f"n{n}_summary.json", info)
    print(f"N={n} selected={selected} D0={diag['nominal_D']:.4g} D32={diag['by_depth'][32]['max_D']:.5g}",
          flush=True)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    out = parser.parse_args().out.resolve()
    out.mkdir(exist_ok=False, parents=False)
    start = time.perf_counter()
    try:
        for directory in (matched.BASE, PRIOR):
            if any(pilot.sha(directory/f)!=h for f,h in matched.read(directory/"sha256.json").items()):
                raise ValueError(f"input run changed {directory}")
        deps = [Path(__file__).resolve(), Path(matched.__file__).resolve(), Path(pilot.__file__).resolve(),
                pilot.PROTOCOL, matched.PROTOCOL, matched.BASE/"sha256.json", PRIOR/"sha256.json",
                pilot.ROOT/"src/manifold_schur_v1.py", pilot.ROOT/"src/cnot_tolerance_v1.py"]
        pilot.dump(out/"inputs.json", {"utc": datetime.now(timezone.utc).isoformat(), "protocol": __doc__,
                   "hashes": {str(f.relative_to(pilot.ROOT)): pilot.sha(f) for f in deps},
                   "python": sys.version, "interpreter": sys.executable, "numpy": np.__version__,
                   "scipy": matched.scipy.__version__, "precision": "float64/complex128"})
        seqs = matched.sequences()
        results = {n: run(n, out, seqs) for n in pilot.LENGTHS}
        matched.figure(out, results)
        pilot.dump(out/"summary.json", {"status": "completed_alternate_restoration",
                   "wall_s": time.perf_counter()-start, "registers": results})
    except BaseException:
        pilot.dump(out/"failure.json", {"traceback": traceback.format_exc(), "wall_s": time.perf_counter()-start})
        raise
    finally:
        pilot.dump(out/"sha256.json", {f.name: pilot.sha(f) for f in sorted(out.iterdir()) if f.is_file()})


if __name__ == "__main__":
    main()
