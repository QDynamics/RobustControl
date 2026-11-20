"""Draw all full-range compiler pairs and code-input leakage from the frozen run."""

import argparse
import csv
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import npj_qi_review_figures_20261118_v1 as style
from npj_qi_review_figures_20261118_v1 import (
    ROOT, BLUE, TEAL, ORANGE, LIGHT, INK, np, plt, matplotlib, csv_out, json_out, digest, panel, save,
)


def make_figure(out, source):
    with (source/"paired_results.csv").open(encoding="utf-8") as stream:
        pairs=list(csv.DictReader(stream))
    with (source/"development_leakage_maxima.csv").open(encoding="utf-8") as stream:
        leakage=[r for r in csv.DictReader(stream) if r["depth"]=="32"]
    assert len(pairs)==8 and len(leakage)==8
    fig,(a,b)=plt.subplots(1,2,figsize=(7.05,2.95))
    fig.subplots_adjust(left=.11,right=.98,top=.85,bottom=.23,wspace=.42)
    panel(a,"a","Full-range training, fresh starts")
    panel(b,"b","Leakage from the logical subspace")
    a.plot([0,31],[0,31],lw=.9,ls="--",color="#84919A")
    points=[]
    for n,color,marker in (("4",TEAL,"o"),("5",BLUE,"D")):
        selected=[r for r in pairs if r["N"]==n]
        x=[100*float(r["direct_max_D32"]) for r in selected]
        y=[100*float(r["seeded_max_D32"]) for r in selected]
        a.scatter(x,y,color=color,marker=marker,s=26,edgecolor="white",linewidth=.5,
                  label=n+" qubits",zorder=3)
        for r,d,s in zip(selected,x,y):
            points.append(dict(panel="paired",N=n,category=r["seed"],x=d,y=s))
            if s>d:
                a.annotate("Direct fit lower",(d,s),(2.6,20.7),color=INK,fontsize=6.8,
                           arrowprops={"arrowstyle":"-", "lw":.7, "color":"#68747C"})
    a.set(xlim=(0,31),ylim=(0,31),xticks=[0,10,20,30],yticks=[0,10,20,30],
          xlabel="Direct fit, maximum distance (%)",ylabel="Seeded fit, maximum distance (%)")
    a.legend(loc="upper left",fontsize=6.9)
    names=["nominal","raw","direct","refined"]
    actual={r["method"] for r in leakage}
    names[1]=next(n for n in actual if n not in ("nominal","direct","refined"))
    for n,color,marker,shift in (("4",TEAL,"o",-.08),("5",BLUE,"D",.08)):
        ordered=[next(r for r in leakage if r["N"]==n and r["method"]==name) for name in names]
        vals=[100*float(r["max_leakage"]) for r in ordered]
        xs=np.arange(4)+shift
        b.scatter(xs,vals,marker=marker,color=color,s=31,label=n+" qubits",zorder=3)
        for name,x,y in zip(names,xs,vals):
            points.append(dict(panel="leakage",N=n,category=name,x=x,y=y))
    b.set(yscale="log",ylim=(.008,100),xlim=(-.45,3.45),xticks=range(4),
          xticklabels=["Nominal","First-\norder","Direct","Refined"],yticks=[.01,.1,1,10,100],
          ylabel="Maximum leakage probability (%)",xlabel="Development controls at depth 32")
    b.set_yticklabels(["0.01","0.1","1","10","100"])
    b.minorticks_off()
    b.legend(loc="upper right",fontsize=6.9)
    for ax in (a,b):
        ax.grid(axis="y",lw=.45,color=LIGHT)
    csv_out(out/"plotted_values.csv",points)
    save(fig,out,"full_range_and_leakage")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    out=parser.parse_args().out.resolve()
    out.mkdir(exist_ok=False)
    start=time.perf_counter()
    source=ROOT/"runs/npj_qi_full_range_review_20261118_v2"
    try:
        files=[Path(__file__).resolve(),Path(style.__file__).resolve(),
               ROOT/"docs/NPJ_QI_COMPARISON_FIGURE_20261118_v1.md",
               source/"paired_results.csv",source/"development_leakage_maxima.csv"]
        json_out(out/"inputs.json",dict(hashes={p.relative_to(ROOT).as_posix():digest(p) for p in files},
                 python=sys.version,numpy=np.__version__,matplotlib=matplotlib.__version__,
                 platform=platform.platform(),utc=datetime.now(timezone.utc).isoformat(),precision="binary64"))
        make_figure(out,source)
        json_out(out/"summary.json",dict(pairs=8,leakage_points=8,elapsed_s=time.perf_counter()-start))
    except BaseException:
        json_out(out/"failure.json",dict(traceback=traceback.format_exc(),elapsed_s=time.perf_counter()-start))
        raise
    finally:
        json_out(out/"sha256.json",{p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file()})


if __name__=="__main__":
    main()
