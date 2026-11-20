"""Reproduce the revision's theory, phase and dephasing figures in a fresh directory."""

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FixedLocator, FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/NPJ_QI_REVIEW_FIGURES_PROTOCOL_20261118_v1.md"
NOISE = ROOT / "runs/npj_qi_dephasing_20261118_v1/summary.json"
CERT = ROOT / "runs/cnot_tolerance_20261012_v1/interval_v2/RESULTS.json"
ORIGINAL = ROOT / "runs/npj_qi_register_depth_20261118_v1"
RESTORED = ROOT / "runs/npj_qi_register_depth_projection_20261118_v1"
CONTROLS = [ORIGINAL / "n4_nominal.npy", ORIGINAL / "n4_corrected.npy",
            RESTORED / "n4_nominal_restored.npy", RESTORED / "n4_corrected_restored.npy"]
TEAL, BLUE, ORANGE, INK, LIGHT = "#006F66", "#334F82", "#AA541D", "#29323D", "#DCE2E6"
DEPTHS = [1, 2, 4, 8, 16, 32]
STYLE = {"font.family": "DejaVu Sans", "font.size": 8.2, "axes.titlesize": 9,
         "axes.labelsize": 8, "axes.linewidth": 0.65, "xtick.labelsize": 7.4,
         "ytick.labelsize": 7.4, "xtick.major.size": 2.7, "ytick.major.size": 2.7,
         "axes.spines.right": False, "axes.spines.top": False, "legend.frameon": False,
         "legend.fontsize": 7.2, "pdf.fonttype": 42, "mathtext.fontset": "dejavusans"}
plt.rcParams.update(STYLE)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_out(path, contents):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(contents, stream, indent=2, allow_nan=False)
        stream.write("\n")


def csv_out(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def save(fig, out, name):
    fig.savefig(out / (name + ".pdf"), metadata={"Creator": "Matplotlib", "Title": name,
                                                "CreationDate": None, "ModDate": None})
    fig.savefig(out / (name + ".png"), dpi=165)
    plt.close(fig)


def panel(ax, letter, title):
    ax.set_title(r"$\bf{" + letter + "}$   " + title, loc="left", pad=7)


def theory_figure(out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.05, 2.85),
                              gridspec_kw={"width_ratios": [1.06, 1]})
    fig.subplots_adjust(left=.028, right=.985, bottom=.13, top=.87, wspace=.16)
    panel(a, "a", "Which targets survive higher orders?")
    panel(b, "b", "What can a regular control change?")
    for ax in (a, b):
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
    a.text(.018, .91, "Cancellation order", color=INK)
    a.text(.74, .91, "Target image", ha="center", color=INK)
    for y, order in ((.70, "1"), (.49, "2"), (.24, "q")):
        a.text(.02, y, r"$\mathcal{M}_{" + order + "}$", fontsize=13, color=BLUE, va="center")
        arrow = FancyArrowPatch((.21, y), (.54, .50 + .37*(y-.49)),
                                arrowstyle="-|>", mutation_scale=9, color="#6A7C8B", lw=1.0)
        a.add_patch(arrow)
    a.text(.07, .365, r"$\vdots$", ha="center", va="center", color=BLUE)
    a.add_patch(FancyBboxPatch((.55, .355), .40, .28, boxstyle="round,pad=.012,rounding_size=.03",
                               ec=TEAL, fc="#ECF5F3", lw=1.1))
    a.text(.75, .495, r"$\mathcal{K}_1=\mathcal{K}_q$", fontsize=12.5, color=TEAL,
           ha="center", va="center")
    a.text(.47, .065, r"Endpoint map   $\pi(a)=U_0(a)$", ha="center", color=INK)
    def xy(t, s):
        return (.095 + .72*t + .08*np.sin(np.pi*s)*(.4+t),
                .29 + .49*s + .11*np.sin(np.pi*t)*(.65+s))
    grid = np.linspace(0, 1, 110)
    for s in [0, .25, .5, .75, 1]:
        b.plot(*xy(grid, s), color="#CBDBDD", lw=.75)
    for t in [0, .25, .5, .75, 1]:
        is_fiber = t == .5
        b.plot(*xy(t, grid), color=TEAL if is_fiber else "#A8C2C4", lw=2 if is_fiber else .85)
    tvals = np.linspace(.15, .81, 40)
    b.plot(*xy(tvals, .39), color=BLUE, lw=1.8)
    b.annotate("", xy=xy(.82,.39), xytext=xy(.77,.39),
               arrowprops=dict(arrowstyle="-|>", color=BLUE, mutation_scale=10, lw=1.2))
    for s in [.20, .78]:
        b.plot(*xy(.5, s), "o", color=TEAL, ms=4.4, markeredgecolor="white", markeredgewidth=.55)
    b.text(.025, .075, r"$\theta$  changes the target", color=BLUE, fontsize=7.9)
    b.text(.49, .075, r"$\xi$  keeps it fixed", color=TEAL, fontsize=7.9)
    save(fig, out, "order_reachability")


def arc_distance(eigenvalues):
    phases = np.sort(np.mod(np.angle(eigenvalues), 2*np.pi))
    width = 2*np.pi - np.max(np.diff(np.r_[phases, phases[0]+2*np.pi]))
    return float(np.sin(min(width, np.pi)/2))


def phase_figure(out, rng):
    x = np.linspace(0, 2, 401)
    phi = np.pi*x
    dist = np.sin(np.abs((phi % (np.pi/2))-np.pi/4)/2)
    plus = 7/18-2/(3*np.sqrt(3))*np.sin(phi-np.pi/4)
    minus = 7/18+2/(3*np.sqrt(3))*np.sin(phi-np.pi/4)
    csv_out(out/"phase_distance_and_readout.csv", [
        dict(phi_rad=p, distance=d, p_plus=a, p_minus=b) for p,d,a,b in zip(phi,dist,plus,minus)])
    g = np.eye(4, dtype=complex)[[0, 1, 3, 2]]
    exact_error = 0.
    random_shortfall = 0.
    for p, d in zip(phi, dist):
        nearest = np.pi/4 + round((p-np.pi/4)/(np.pi/2))*(np.pi/2)
        eigen = np.linalg.eigvals((np.exp(1j*p)*g).conj().T @ (np.exp(1j*nearest)*g))
        exact_error = max(exact_error, abs(arc_distance(np.r_[eigen, np.ones(4)])-d))
        blocks = []
        for dim in [4, 2]:
            z = rng.normal(size=(dim, dim)) + 1j*rng.normal(size=(dim, dim))
            u, _ = np.linalg.qr(z)
            u[:, -1] /= np.linalg.det(u)
            blocks.append(u)
        rel = np.linalg.eigvals((np.exp(1j*p)*g).conj().T @ blocks[0])
        random_shortfall = max(random_shortfall, d-arc_distance(np.r_[rel, np.linalg.eigvals(blocks[1])]))
    assert exact_error < 2e-14 and random_shortfall < 2e-14
    fig, (a,b) = plt.subplots(1, 2, figsize=(7.05, 2.76))
    fig.subplots_adjust(left=.09, right=.986, bottom=.21, top=.86, wspace=.32)
    panel(a, "a", "Distance to the correctable gates")
    panel(b, "b", "Interference across spin sectors")
    a.plot(x, dist, color=BLUE, lw=1.6)
    a.scatter([.25,.75,1.25,1.75], [0]*4, color=TEAL, s=23, zorder=4, clip_on=False)
    a.scatter([0],[dist[0]], color=ORANGE, marker="D", s=22, zorder=3)
    a.annotate(r"$\sin(\pi/8)$", (0,dist[0]), (.17,.449), color=ORANGE, fontsize=7.5,
               arrowprops={"arrowstyle":"-", "lw":.7, "color":ORANGE})
    a.set(ylabel="Minimum half-diamond distance", ylim=(-.017,.5), yticks=[0,.1,.2,.3,.4,.5])
    b.plot(x, plus, color=BLUE, lw=1.6, label=r"$p_+$")
    b.plot(x, minus, color=ORANGE, ls="--", lw=1.6, label=r"$p_-$")
    b.scatter([0,0,.25], [plus[0],minus[0],7/18], color=[BLUE,ORANGE,TEAL], s=20, zorder=4)
    b.annotate(r"$7/18$", (.25,7/18), (.61,.41), color=TEAL, fontsize=7.5,
               arrowprops={"arrowstyle":"-", "lw":.7, "color":TEAL})
    b.set(ylabel=r"Probability of $000$", ylim=(-.025,.83), yticks=[0,.2,.4,.6,.8])
    b.legend(loc="upper center", bbox_to_anchor=(.53,1.02), ncol=2, handlelength=1.6, columnspacing=.9)
    for ax in (a,b):
        ax.set(xlim=(-.035,2.03), xlabel=r"Relative phase $\phi$")
        ax.set_xticks([0,.5,1,1.5,2], ["0",r"$\pi/2$",r"$\pi$",r"$3\pi/2$",r"$2\pi$"])
        ax.grid(axis="y", lw=.48, color=LIGHT)
    save(fig,out,"sector_phase_distance")
    return dict(equality_distance_max_difference=exact_error, sampled_lower_bound_violation=random_shortfall)


def peak(area, duration):
    return abs(area)/duration * (1+math.hypot(1, 2*math.pi*.02/duration))


def common_schedule(out):
    decoded = []
    for path in CONTROLS:
        theta, _, raw = np.split(np.load(path), 3)
        decoded.append(np.column_stack((raw-raw.mean(),theta)).ravel())
    areas = np.asarray(decoded)
    durations = []
    rows = []
    for index, column in enumerate(areas.T):
        cap = 1 if index % 2 == 0 else 10
        dose = np.max(np.abs(column))
        low, high = .04, max(.04, 3*dose/cap+.1)
        if peak(dose, low) > cap:
            for _ in range(75):
                mid = (low+high)/2
                if peak(dose, mid) > cap:
                    low = mid
                else:
                    high = mid
            duration = math.ceil(high*1e6)*1e-6
        else:
            duration = .04
        assert max(peak(value, duration) for value in column) <= cap*(1+2e-14)
        durations.append(duration)
        rows.append(dict(slot=index+1, cap=cap, maximum_area=dose, duration_t0=duration))
    csv_out(out/"common_four_control_clock.csv",rows)
    return math.fsum(durations)+(len(durations)-1)*.001


def bounds(distance, probability):
    radical = math.hypot(probability, 2*math.sqrt(1-probability)*distance)
    lower = .5*(radical-probability)
    return lower, probability+(1-probability)*distance


def mixture_check(rng):
    excess = 0.
    for _ in range(400):
        d, p = rng.random(2)
        u = np.array([math.sqrt(1-d*d), d], dtype=complex)
        v = np.array([1.,0.])
        j = rng.normal(size=2) + 1j*rng.normal(size=2)
        j /= np.linalg.norm(j)
        contrast = (1-p)*np.outer(u,u.conj())+p*np.outer(j,j.conj())-np.outer(v,v)
        actual = sum(abs(np.linalg.eigvalsh(contrast)))/2
        lower, upper = bounds(d,p)
        excess = max(excess,lower-actual,actual-upper)
    assert excess < 2e-14
    return excess


def dephasing_figure(out):
    record = json.loads(NOISE.read_text(encoding="utf-8"))
    short_gate = common_schedule(out)
    rows = []
    for value in record["brackets"]:
        arm, m = value["method"], value["depth"]
        if value["N"] != 4 or arm not in ("direct","refined"):
            continue
        for clock, duration in (("original",value["time_t0"]),("shorter",m*short_gate+(m-1)*.001)):
            p = -math.expm1(-2*1e-5*duration)
            coherent = value["coherent_grid_D"]
            lower, upper = bounds(coherent,p)
            witness = record["readout"][arm]["1e-05"]["max_loss"][str(m)] if clock=="original" else None
            rows.append(dict(clock=clock, method=arm, depth=m, duration_t0=duration,
                             probability_any_jump=p, coherent_D=coherent, mixture_lower=lower,
                             readout_lower=witness, lower=max(lower,witness or 0), upper=upper))
    csv_out(out/"dephasing_bounds.csv", rows)
    fig,(a,b)=plt.subplots(1,2,figsize=(7.05,2.95))
    fig.subplots_adjust(left=.105, right=.982,bottom=.21,top=.86,wspace=.44)
    panel(a,"a","Loss in a specified readout")
    panel(b,"b","Complete channel at depth 32")
    for arm,color,mark in (("direct",ORANGE,"v"),("refined",BLUE,"o")):
        for rate,ls in (("0.0","--"),("1e-05","-")):
            loss=record["readout"][arm][rate]["max_loss"]
            a.plot(DEPTHS,[100*loss[str(m)] for m in DEPTHS],ls=ls,marker=mark,color=color,ms=3.7,lw=1.3)
    a.set(xscale="log",yscale="log",xticks=DEPTHS,xticklabels=DEPTHS,ylim=(.0003,52),
          xlabel="Completed encoded gates",ylabel="Maximum probability loss (%)")
    a.yaxis.set_major_locator(FixedLocator([.001,.01,.1,1,10]))
    a.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f"{v:g}"))
    a.minorticks_off()
    a.legend(handles=[Line2D([],[],color=ORANGE,marker="v",label="Direct"),
                      Line2D([],[],color=BLUE,marker="o",label="Refined"),
                      Line2D([],[],color=INK,ls="--",label=r"$\gamma=0$"),
                      Line2D([],[],color=INK,label=r"$\gamma t_0=10^{-5}$")],
             ncol=2,loc="upper left",fontsize=6.4,handlelength=1.1,columnspacing=.8)
    selected=[r for r in rows if r["depth"]==32]
    for r in selected:
        y=(2.45 if r["clock"]=="original" else .65) + (.28 if r["method"]=="direct" else -.28)
        color=ORANGE if r["method"]=="direct" else BLUE
        b.plot([100*r["lower"],100*r["upper"]],[y,y],lw=2.7,color=color,solid_capstyle="butt")
        b.vlines([100*r["lower"],100*r["upper"]],y-.085,y+.085,lw=1.1,color=color)
    b.set(xlim=(0,30),ylim=(0,3.25),xticks=[0,5,10,15,20,25,30],
          yticks=[.65,2.45],yticklabels=["Shorter\nclock","Original\nclock"],
          xlabel="Grid maximum half-diamond distance (%)")
    b.tick_params(axis="y",length=0)
    b.spines["left"].set_visible(False)
    b.legend(handles=[Line2D([],[],color=ORANGE,lw=2,label="Direct"),
                      Line2D([],[],color=BLUE,lw=2,label="Refined")],
             loc="upper right",bbox_to_anchor=(1,1.03),ncol=2,fontsize=6.9,handlelength=1.5)
    a.grid(axis="y",color=LIGHT,lw=.48)
    b.grid(axis="x",color=LIGHT,lw=.48)
    save(fig,out,"dephasing_clocks")
    return dict(shorter_gate_t0=short_gate, depth32=selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    out=parser.parse_args().out.resolve()
    out.mkdir(exist_ok=False)
    start=time.perf_counter()
    try:
        files=[Path(__file__).resolve(),PROTOCOL,NOISE,CERT,*CONTROLS]
        json_out(out/"inputs.json",dict(hashes={str(f.relative_to(ROOT).as_posix()):digest(f) for f in files},
                 utc=datetime.now(timezone.utc).isoformat(),python=sys.version,platform=platform.platform(),
                 numpy=np.__version__,matplotlib=matplotlib.__version__,precision="binary64",seed=6111819))
        rng=np.random.default_rng(6111819)
        theory_figure(out)
        phase=phase_figure(out,rng)
        clocks=dephasing_figure(out)
        whole=json.loads(CERT.read_text(encoding="utf-8"))["results"]["whole"]
        term=whole["degree_sum_bounds"][2]*.01**2
        at_one_percent=whole["bounds"]["8"]["0.01"]
        json_out(out/"summary.json",dict(phase=phase,clocks=clocks,mixture_check_violation=mixture_check(rng),
                 quadratic_term=term,upper_envelope=at_one_percent,quadratic_fraction=term/at_one_percent,
                 elapsed_s=time.perf_counter()-start))
    except BaseException:
        json_out(out/"failure.json",dict(traceback=traceback.format_exc(),elapsed_s=time.perf_counter()-start))
        raise
    finally:
        json_out(out/"sha256.json",{f.name:digest(f) for f in sorted(out.iterdir()) if f.is_file()})


if __name__=="__main__":
    main()
