"""Collect all equal-nominal-accuracy depth arms, including both independent fit starts.

No optimization. Lines show the predeclared sequence and bands the three other frozen
sequences. Keep the relaxed-target data separately identified in the machine-readable table.
"""
import argparse
import csv
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import npj_qi_register_depth_matched_20261118_v1 as matched
from npj_qi_register_depth_matched_20261118_v1 import np, pilot

FIRST = matched.BASE
SECOND = pilot.ROOT / "runs/npj_qi_register_depth_matched_20261118_v1"
THIRD = pilot.ROOT / "runs/npj_qi_register_depth_projection_20261118_v1"
SERIES = [("nominal", "Nominal fit", "#47586E", "o", "--"),
          ("direct", "Finite-noise fit from nominal", "#B7601F", "v", "-."),
          ("corrected", "First-order correction", "#08756B", "s", "-"),
          ("refined", "Corrected seed + finite-noise fit", "#7353A6", "D", "-")]


def report(n, sequences):
    blocks = [pilot.Block(r, n) for r in pilot.schur_representations(n)]
    p = np.load(THIRD/f"n{n}_nominal_restored.npy")
    grid = {name: matched.read(SECOND/f"n{n}_{name}_all_circuits.json")
            for name in ("nominal", "corrected", "relaxed")}
    grid["refined"] = matched.read(THIRD/f"n{n}_matched_all_circuits.json")
    grid["direct"] = {key: matched.held_out(blocks, p, seq) for key,seq in sequences.items()}
    target, m, zz = pilot.tensor_setup(n)
    direct, _ = pilot.diagnostics(n, blocks, p, target, m, zz, pilot.full_tensor_representation(n))
    audit = matched.direct_shifted_audit(n, p, target, m, zz, direct["by_depth"], blocks)
    if not direct["nominal_qualified"]:
        raise ValueError("direct baseline failed nominal tolerance")
    rows = []
    maxima = {}
    for name, series in grid.items():
        maxima[name] = {}
        for label, data in series.items():
            if len(data) != len(pilot.TEST)*len(pilot.DEPTHS):
                raise ValueError("incomplete validation panel")
            maxima[name][label] = {}
            for depth in pilot.DEPTHS:
                sub = [v for v in data if v["depth"]==depth]
                maxima[name][label][depth] = max(v["D"] for v in sub)
                rows.extend({"N": n, "method": name, "circuit": label, **v} for v in sub)
    return rows, {"grid_max_D": maxima, "direct": direct, "dense_direct_pulse_audit": audit}


def plot(path, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.25), sharey=True)
    for ax, (n, data) in zip(axes, results.items()):
        for name, label, color, marker, style in SERIES:
            vals = data["grid_max_D"][name]
            original = np.array([vals["original"][k] for k in pilot.DEPTHS])*100
            extra = np.array([[vals[str(seed)][k] for k in pilot.DEPTHS] for seed in matched.SEEDS])*100
            ax.fill_between(pilot.DEPTHS, extra.min(axis=0), extra.max(axis=0), color=color, alpha=.14)
            ax.plot(pilot.DEPTHS, original, label=label, color=color, marker=marker, lw=1.9, ms=5,
                    linestyle=style, markerfacecolor="white" if name in ("nominal", "direct") else color)
        ax.axhline(1, color=".45", linewidth=1.1, linestyle=(0, (2, 3)), label="1% reference")
        ax.set(xscale="log", yscale="log", title=f"{n} physical qubits", xlabel="Completed encoded gates",
               xticks=pilot.DEPTHS, xticklabels=pilot.DEPTHS, ylim=(.28, 135))
        ax.minorticks_off()
        ax.yaxis.set_major_locator(FixedLocator([.5, 1, 2, 5, 10, 20, 50, 100]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:g}"))
        ax.grid(axis="y", color=".88", lw=.65)
    axes[0].set_ylabel("Grid-maximum half-diamond distance (%)")
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=3, fontsize=9.0, frameon=False,
               loc="lower center", bbox_to_anchor=(.5, .035))
    fig.text(.5, .015, "Lines: fixed circuit.  Bands: range over three additional circuits.  "
             "Each circuit: 25 shared calibration offsets.", ha="center", fontsize=8.5, color=".25")
    fig.tight_layout(rect=(0, .18, 1, 1))
    fig.savefig(path, dpi=210)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    out = parser.parse_args().out.resolve()
    out.mkdir(exist_ok=False, parents=False)
    start = time.perf_counter()
    try:
        for directory in (FIRST, SECOND, THIRD):
            for f, h in matched.read(directory/"sha256.json").items():
                if pilot.sha(directory/f)!=h:
                    raise ValueError(f"changed archived input {directory/f}")
        files = [Path(__file__).resolve(), Path(pilot.__file__).resolve(), Path(matched.__file__).resolve(),
                 FIRST/"sha256.json", SECOND/"sha256.json", THIRD/"sha256.json"]
        pilot.dump(out/"inputs.json", {"utc": datetime.now(timezone.utc).isoformat(),
                   "precision":"float64/complex128", "python":sys.version,
                   "numpy":np.__version__,"scipy":matched.scipy.__version__,
                   "hashes": {str(f.relative_to(pilot.ROOT)): pilot.sha(f) for f in files}})
        results = {}
        with (out/"all_validation_points.csv").open("x", newline="", encoding="utf-8") as h:
            csv_out = csv.DictWriter(h, fieldnames=["N","method","circuit","depth","x","y","D","avg_infidelity"])
            csv_out.writeheader()
            for n in pilot.LENGTHS:
                rows, results[n] = report(n, matched.sequences())
                csv_out.writerows(rows)
        plot(out/"register_depth.png", results)
        pilot.dump(out/"summary.json", {"status":"all_returned_arms_reported","wall_s":time.perf_counter()-start,
                                       "registers":results})
    except BaseException:
        pilot.dump(out/"failure.json", {"traceback":traceback.format_exc(),"wall_s":time.perf_counter()-start})
        raise
    finally:
        pilot.dump(out/"sha256.json", {f.name:pilot.sha(f) for f in sorted(out.iterdir()) if f.is_file()})


if __name__ == "__main__":
    main()
