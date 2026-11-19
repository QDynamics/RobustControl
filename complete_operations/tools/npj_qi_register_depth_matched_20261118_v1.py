"""Registered nominal-accuracy-matched comparison and extra held-out circuits."""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import npj_qi_register_depth_20261118_v1 as pilot
import numpy as np
import scipy
from scipy.spatial import ConvexHull

BASE = pilot.ROOT / "runs/npj_qi_register_depth_20261118_v1"
PROTOCOL = pilot.ROOT / "docs/NPJ_QI_REGISTER_DEPTH_PROTOCOL_20261118_v2.md"
SEEDS = (99181, 99182, 99183)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sequences():
    result = {"original": pilot.PHASES}
    for seed in SEEDS:
        gen = np.random.default_rng(seed)
        result[str(seed)] = np.r_[0., gen.uniform(-np.pi, np.pi, 31)]
    return result


def hull_distance(target, actual):
    """Independently evaluate unitary-channel distance from the spectral convex hull."""
    z = np.linalg.eigvals(target.conj().T @ actual)
    z /= np.abs(z)
    xy = np.column_stack([z.real, z.imag])
    hull = ConvexHull(xy)
    if np.max(hull.equations[:, 2]) <= 0:
        return 1.
    ordered = xy[hull.vertices]
    nxt = np.roll(ordered, -1, axis=0)
    direction = nxt - ordered
    t = np.clip(-np.sum(ordered*direction, axis=1)/np.sum(direction*direction, axis=1), 0., 1.)
    dist2 = np.min(np.sum((ordered+t[:, None]*direction)**2, axis=1))
    return math.sqrt(max(0., 1.-dist2))


def direct_shifted_audit(n, params, target, m, charges, by_depth, blocks):
    """Shift each physical rotation azimuth afresh; do not conjugate a stored noisy gate."""
    actual = np.eye(2**n, dtype=complex)
    ideal = actual.copy()
    vals = {}
    gates = [b.word(params, np.array([[.01, -.01]]))[0] for b in blocks]
    schur = pilot.trace_circuit([b.target for b in blocks], gates, [b.m for b in blocks],
                                [b.mult for b in blocks])
    for k, phi in enumerate(pilot.PHASES, 1):
        physical = params.copy()
        length = len(params)//3
        physical[length:2*length] += phi
        actual = pilot.tensor_word(n, charges, physical, .01, -.01) @ actual
        z = np.exp(-1j*phi*m)
        ideal = ((z[:, None]*target)*z.conj()[None, :]) @ ideal
        if k in (2, 32):
            val = hull_distance(ideal, actual)
            diff = abs(val-schur[k][0])
            if diff > 1e-8 or val > by_depth[k]["max_D"]+1e-8:
                raise ValueError(f"direct phased pulse check failed {n}, {k}, {diff}")
            vals[k] = {"tensor_and_hull_D": val, "absolute_difference": diff}
    return vals


def held_out(blocks, params, seq):
    """All error points propagated in parallel, through the same fixed gate sequence."""
    ref = [b.target for b in blocks]
    gates = [b.word(params, pilot.TEST) for b in blocks]
    ref_prefix = [np.eye(b.d, dtype=complex) for b in blocks]
    prefix = [np.broadcast_to(r, (len(pilot.TEST), *r.shape)).copy() for r in ref_prefix]
    rows = []
    mult = [b.mult for b in blocks]
    for k, phi in enumerate(seq, 1):
        for j, b in enumerate(blocks):
            z = np.exp(-1j*phi*b.difference)
            ref_prefix[j] = (z*ref[j]) @ ref_prefix[j]
            prefix[j] = (z*gates[j]) @ prefix[j]
        if k in pilot.DEPTHS:
            for i, (x, y) in enumerate(pilot.TEST):
                d, fav = pilot.metrics(ref_prefix, [g[i] for g in prefix], mult)
                rows.append({"depth": k, "x": float(x), "y": float(y), "D": d, "avg_infidelity": fav})
    return rows


def run_n(n, out, seqs):
    base = read(BASE/f"n{n}_summary.json")
    blocks = [pilot.Block(r, n) for r in pilot.schur_representations(n)]
    starts = {key: np.load(BASE/f"n{n}_{key}.npy") for key in ("nominal", "corrected")}
    candidates = dict(starts)
    for key, initial in starts.items():
        finite, info = pilot.fit(blocks, n, initial, "noise_aware", 82, 68)
        restored, projection = pilot.fit(blocks, n, finite, "nominal", 8, 7)
        for label, p, report in [("finite", finite, info), ("restored", restored, projection)]:
            prefix = f"n{n}_{key}_{label}"
            np.save(out/f"{prefix}.npy", p, allow_pickle=False)
            pilot.dump(out/f"{prefix}_fit.json", report)
        candidates[f"{key}_restored"] = restored
    nominal = {}
    train = {}
    for key, p in candidates.items():
        gates = [b.word(p, pilot.JET[:1])[0] for b in blocks]
        nominal[key] = pilot.metrics([b.target for b in blocks], gates, [b.mult for b in blocks])[0]
        train[key] = pilot.training_score(blocks, p)
    chosen = min((key for key in candidates if nominal[key]<=1e-6), key=train.get)
    p = candidates[chosen]
    target, mag, charges = pilot.tensor_setup(n)
    summary, points = pilot.diagnostics(n, blocks, p, target, mag, charges,
                                        pilot.full_tensor_representation(n))
    pilot.dump(out/f"n{n}_matched_validation.json", points)
    independent = direct_shifted_audit(n, p, target, mag, charges, summary["by_depth"], blocks)
    methods = {"nominal": starts["nominal"], "corrected": starts["corrected"],
               "matched": p, "relaxed": np.load(BASE/f"n{n}_{base['noise_aware_source']}.npy")}
    by_circuit = {}
    for name, params in methods.items():
        scores = {}
        by_circuit[name] = {}
        for label, seq in seqs.items():
            rows = held_out(blocks, params, seq)
            scores[label] = rows
            by_circuit[name][label] = {k: max(r["D"] for r in rows if r["depth"]==k)
                                        for k in pilot.DEPTHS}
        pilot.dump(out/f"n{n}_{name}_all_circuits.json", scores)
    for k in pilot.DEPTHS:
        for name, data in [("nominal", base["methods"]["nominal"]["by_depth"]),
                           ("corrected", base["methods"]["corrected"]["by_depth"]),
                           ("relaxed", base["methods"]["noise_aware"]["by_depth"])]:
            if abs(by_circuit[name]["original"][k] - data[str(k)]["max_D"]) > 1e-10:
                raise ValueError(f"base replay mismatch: {n}, {name}, {k}")
    info = {"N": n, "nominal_D": nominal, "training_max_D": train, "selected": chosen,
            "matched": summary, "direct_physical_crosscheck": independent, "circuit_max_D": by_circuit}
    pilot.dump(out/f"n{n}_summary.json", info)
    print(f"N={n} selected={chosen} D0={summary['nominal_D']:.3g} "
          f"D32={summary['by_depth'][32]['max_D']:.6g}", flush=True)
    return info


def figure(out, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.85), sharey=True)
    styles = [("nominal", "Nominal fit", "#47586E", "o", "--"),
              ("corrected", "First-order correction", "#007E82", "s", "-"),
              ("matched", "Finite-noise fit, target restored", "#BA5129", "^", "-."),
              ("relaxed", "Finite-noise fit, target relaxed", "#8858A4", "x", ":")]
    for ax, (n, report) in zip(axes, results.items()):
        for key, label, color, marker, linestyle in styles:
            series = report["circuit_max_D"][key]["original"]
            ax.plot(pilot.DEPTHS, [series[d] for d in pilot.DEPTHS],
                    color=color, marker=marker, linestyle=linestyle, ms=5, lw=2, label=label)
        ax.axhline(.01, color="0.4", lw=1, linestyle=(0, (2, 3)), label="1% reference")
        ax.set(xscale="log", yscale="log", xticks=pilot.DEPTHS, xticklabels=pilot.DEPTHS,
               xlabel="Completed encoded gates", title=f"{n} physical qubits")
        ax.minorticks_off()
        ax.grid(axis="y", alpha=.22)
    axes[0].set_ylabel("Maximum half-diamond distance on 25 test points")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=8.6,
               frameon=False, bbox_to_anchor=(.5, -.001))
    fig.tight_layout(rect=(0, .12, 1, 1))
    fig.savefig(out/"depth_comparison.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    out = parser.parse_args().out.resolve()
    out.mkdir(exist_ok=False, parents=False)
    start = time.perf_counter()
    try:
        source_manifest = read(BASE/"sha256.json")
        if any(pilot.sha(BASE/f)!=value for f,value in source_manifest.items()):
            raise ValueError("upstream run changed")
        seqs = sequences()
        files = [Path(__file__).resolve(), Path(pilot.__file__).resolve(), BASE/"sha256.json",
                 PROTOCOL, pilot.PROTOCOL, pilot.ROOT/"src/manifold_schur_v1.py", pilot.ROOT/"src/cnot_tolerance_v1.py"]
        pilot.dump(out/"inputs.json", {"hashes": {str(f.relative_to(pilot.ROOT)): pilot.sha(f) for f in files},
                   "utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
                   "interpreter": sys.executable, "numpy": np.__version__, "scipy": scipy.__version__,
                   "platform": platform.platform(), "precision": "float64/complex128",
                   "phase_sequences": {k: v.tolist() for k,v in seqs.items()}})
        results = {n: run_n(n, out, seqs) for n in pilot.LENGTHS}
        figure(out, results)
        pilot.dump(out/"summary.json", {"status": "completed_nominal_matched_pilot",
                   "wall_s": time.perf_counter()-start, "registers": results})
    except BaseException:
        pilot.dump(out/"failure.json", {"traceback": traceback.format_exc(), "wall_s": time.perf_counter()-start})
        raise
    finally:
        pilot.dump(out/"sha256.json", {f.name: pilot.sha(f) for f in sorted(out.iterdir()) if f.is_file()})


if __name__ == "__main__":
    main()
