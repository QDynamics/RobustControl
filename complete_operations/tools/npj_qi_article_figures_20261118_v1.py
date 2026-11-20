"""Build the phase, finite-depth and dephasing scientific figures from the frozen data."""
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
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/"runs/npj_qi_register_depth_report_20261118_v1"
STARTS=ROOT/"runs/npj_qi_confirm_starts_20261118_v1"
NOISE=ROOT/"runs/npj_qi_dephasing_20261118_v1"
OLD=ROOT/"submission/NatureCommunications_20261013_v1/04_source"
DEPTHS=[1,2,4,8,16,32]
TEAL="#00786E"; ORANGE="#AD581C"; VIOLET="#6F4C94"; SLATE="#58697E"; BLUE="#27598D"
STYLES=[("nominal","Nominal fit",SLATE,"o","--"),("direct","Direct finite-noise fit",ORANGE,"v","-."),
        ("corrected","First-order correction",TEAL,"s","-"),("refined","Correction then finite-noise fit",VIOLET,"D","-")]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.titlesize":9,"axes.labelsize":8,
                     "axes.linewidth":.65,"xtick.labelsize":7.3,"ytick.labelsize":7.3,
                     "xtick.major.width":.65,"ytick.major.width":.65,"xtick.major.size":2.6,"ytick.major.size":2.6,
                     "legend.fontsize":7.1,"legend.frameon":False,"pdf.fonttype":42,"ps.fonttype":42,
                     "mathtext.fontset":"dejavusans","axes.spines.top":False,"axes.spines.right":False})


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path,data):
    with path.open("x",encoding="utf-8") as f:
        json.dump(data,f,indent=2,allow_nan=False)


def write_csv(path,rows):
    with path.open("x",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)


def title(ax,letter,name):
    ax.set_title(r"$\bf{"+letter+r"}$   "+name,loc="left",pad=6)


def save(fig,out,name):
    fig.savefig(out/(name+".pdf"),metadata={"Creator":"Matplotlib","Title":name})
    fig.savefig(out/(name+".png"),dpi=185)
    plt.close(fig)


def phase_figure(out):
    phi_over_pi=np.linspace(0,2,401)
    delta=np.pi*(phi_over_pi-.25)
    pplus=7/18-2/(3*np.sqrt(3))*np.sin(delta)
    pminus=7/18+2/(3*np.sqrt(3))*np.sin(delta)
    write_csv(out/"phase_probabilities.csv",[{"phi_radians":float(t*np.pi),"p_plus":float(a),"p_minus":float(b)}
                                             for t,a,b in zip(phi_over_pi,pplus,pminus)])
    fig=plt.figure(figsize=(6.8,4.05))
    grid=fig.add_gridspec(2,2,width_ratios=[1,1],height_ratios=[.87,1.12],
                          left=.092,right=.982,bottom=.12,top=.91,wspace=.27,hspace=.49)
    a=fig.add_subplot(grid[0,0]);a.axis("off");title(a,"a","Two complete gates on three qubits")
    a.set(xlim=(0,1),ylim=(0,1))
    for xx,name in [(.04,"Logical block"),(.55,"Rest")]:
        a.text(xx,.81,name,color="#484E55",fontsize=8,va="center")
    a.text(.97,.81,"Reachable?",color="#484E55",fontsize=8,ha="right",va="center")
    a.plot([.03,.97],[.69,.69],c="#A5AAB0",lw=.8)
    for yy,lhs,col,result in [(.52,r"$\mathrm{CNOT}_{L}$",ORANGE,"Excluded"),
                              (.22,r"$e^{i\pi/4}\mathrm{CNOT}_{L}$",TEAL,"Admitted")]:
        a.text(.04,yy,lhs,va="center",fontsize=9.3);a.text(.61,yy,r"$I_Q$",va="center",fontsize=9)
        a.text(.97,yy,result,ha="right",va="center",color=col,fontsize=7.8,fontweight="bold")
    b=fig.add_subplot(grid[0,1]);title(b,"b","Which relative phases are reachable?")
    b.set_xlim(-.11,2.08);b.set_ylim(-.45,.65)
    b.axhline(0,lw=.8,c="#A5AAB0")
    b.scatter([.25,.75,1.25,1.75],[0]*4,marker="o",color=TEAL,s=29,zorder=4)
    b.scatter([0],[0],color=ORANGE,s=30,marker="x",linewidths=1.5,zorder=4)
    b.text(.03,.84,r"Allowed:  $\phi=\pi/4 \ (\mathrm{mod}\ \pi/2)$",transform=b.transAxes,color=TEAL,fontsize=8.1)
    b.text(.03,.04,r"At $\phi=0$: distance to set $=\sin(\pi/8)$",transform=b.transAxes,fontsize=7.6)
    b.set_xticks([0,.5,1,1.5,2],["0",r"$\pi/2$",r"$\pi$",r"$3\pi/2$",r"$2\pi$"])
    b.set_yticks([]);b.spines["left"].set_visible(False)
    c=fig.add_subplot(grid[1,:]);title(c,"c","A two-setting readout of the sector phase")
    c.plot(phi_over_pi,pplus,c=BLUE,lw=1.65,label=r"$p_+$  ($\beta=-\pi/4$)")
    c.plot(phi_over_pi,pminus,c=ORANGE,lw=1.65,ls="--",label=r"$p_-$  ($\beta=3\pi/4$)")
    c.vlines([0,.25],0,[pplus[0],7/18],linestyles=":",colors="#7D868D",lw=.8)
    c.scatter([0,0,.25],[pplus[0],pminus[0],7/18],color=[BLUE,ORANGE,TEAL],s=19,zorder=4)
    c.set(xlim=(-.04,2.04),ylim=(-.02,.83),ylabel=r"Probability of $000$",xlabel=r"Relative phase $\phi$")
    c.set_xticks([0,.25,.5,1,1.5,2],["0",r"$\pi/4$",r"$\pi/2$",r"$\pi$",r"$3\pi/2$",r"$2\pi$"])
    c.set_yticks([0,.2,.4,.6,.8]);c.grid(axis="y",color="#E1E3E5",lw=.55)
    c.annotate(r"Both $7/18$",(.25,7/18),xytext=(.52,.47),color="#223A34",fontsize=7.3,
               arrowprops={"arrowstyle":"-","color":"#65736C","lw":.7})
    c.legend(loc="upper right",bbox_to_anchor=(1.0,1.30),ncol=2,columnspacing=1.0,handlelength=2.0)
    save(fig,out,"phase_decision")


def circuit_figure(out):
    report=read(REPORT/"summary.json")["registers"]
    pairs=read(STARTS/"summary.json")["pairs"]
    cells=[]
    with (REPORT/"all_validation_points.csv").open() as h:
        for row in csv.DictReader(h):
            if (row["N"],row["method"],row["circuit"],row["depth"])==("4","refined","original","32"):
                cells.append({k:float(row[k]) for k in ("x","y","D")})
    if len(cells)!=25:raise ValueError("incomplete grid")
    write_csv(out/"depth32_gain_map.csv",cells)
    fig=plt.figure(figsize=(6.8,5.85))
    grid=fig.add_gridspec(3,2,left=.105,right=.982,bottom=.09,top=.95,hspace=.41,wspace=.38,
                          height_ratios=[1.04,.12,1])
    axes=[]
    for col,n in enumerate(("4","5")):
        ax=fig.add_subplot(grid[0,col]);axes.append(ax)
        title(ax,"ab"[col],f"{n} physical qubits")
        for name,label,color,mark,line in STYLES:
            d=report[n]["grid_max_D"][name]
            vals=[100*d["original"][str(k)] for k in DEPTHS]
            extra=np.array([[100*v[str(k)] for k in DEPTHS] for key,v in d.items() if key!="original"])
            ax.fill_between(DEPTHS,extra.min(axis=0),extra.max(axis=0),color=color,alpha=.16,lw=0)
            ax.plot(DEPTHS,vals,color=color,marker=mark,ms=3.4,lw=1.25,linestyle=line,label=label,
                    markerfacecolor="white" if name in ("nominal","direct") else color)
        ax.axhline(1,color=".4",lw=.8,ls=(0,(2,3)),label="1% budget")
        ax.set(xscale="log",yscale="log",xticks=DEPTHS,xticklabels=DEPTHS,ylim=(.32,135),
               xlabel="Completed encoded gates")
        if col==0:ax.set_ylabel("Maximum half-diamond distance (%)")
        ax.minorticks_off();ax.yaxis.set_major_locator(FixedLocator([.5,1,2,5,10,25,50,100]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f"{v:g}"))
        ax.grid(axis="y",lw=.48,color=".86")
    legend=fig.add_subplot(grid[1,:]);legend.axis("off")
    legend.legend(*axes[0].get_legend_handles_labels(),loc="center",ncol=3,columnspacing=1.5,
                  handlelength=2.3,fontsize=7.3)
    c=fig.add_subplot(grid[2,0]);title(c,"c",r"Four qubits at depth 32, $D$ (%)")
    xy=sorted(set(r["x"] for r in cells))
    arr=np.zeros((5,5))
    for r in cells:arr[xy.index(r["y"]),xy.index(r["x"])]=100*r["D"]
    c.imshow(arr,origin="lower",vmin=0,vmax=5.0,cmap="viridis",interpolation="nearest",aspect="auto")
    for j,row in enumerate(arr):
        for i,v in enumerate(row):
            c.text(i,j,f"{v:.2f}",ha="center",va="center",fontsize=7.1,
                   color="#121619" if v>2.6 else "white")
    c.set_xticks(range(5),["−1","−0.5","0","0.5","1"]);c.set_yticks(range(5),["−1","−0.5","0","0.5","1"])
    c.set(xlabel=r"Drive gain error $x$ (%)",ylabel=r"Interaction gain error $y$ (%)")
    d=fig.add_subplot(grid[2,1]);title(d,"d","Eight matched compiler starts")
    lim=(0,43);d.plot(lim,lim,color=".52",ls=(0,(3,2)),lw=.85)
    for n,color,mark in [(4,TEAL,"o"),(5,VIOLET,"D")]:
        subset=[v["validation"] for v in pairs if v["N"]==n]
        xx=[100*v["direct"]["depth_scores"]["original"]["32"] for v in subset]
        yy=[100*v["seeded"]["depth_scores"]["original"]["32"] for v in subset]
        d.scatter(xx,yy,s=28,c=color,marker=mark,edgecolors="white",linewidths=.45,label=f"{n} qubits",zorder=4)
    d.set(xlim=lim,ylim=(0,43),xticks=[0,10,20,30,40],yticks=[0,10,20,30,40],
          xlabel=r"Direct fit $D_{32}$ (%)",ylabel=r"Correction-seeded $D_{32}$ (%)")
    d.text(26,30,"Equal error",rotation=41,color=".38",fontsize=7)
    d.legend(loc="upper left",fontsize=7,handlelength=1);d.grid(lw=.42,color=".9")
    save(fig,out,"register_depth")


def dephasing_figure(out):
    dep=read(NOISE/"summary.json");fig,axs=plt.subplots(1,2,figsize=(6.8,2.65))
    fig.subplots_adjust(left=.099,right=.982,bottom=.20,top=.87,wspace=.38)
    a,b=axs;title(a,"a","Loss of the ideal output probability");title(b,"b","Complete-channel bracket")
    for name,color,mark in [("direct",ORANGE,"v"),("refined",VIOLET,"D")]:
        for gamma,key,ls in [(0.,"0.0","--"),(1e-5,"1e-05","-")]:
            curve=[100*dep["readout"][name][key]["max_loss"][str(m)] for m in DEPTHS]
            a.plot(DEPTHS,curve,ls=ls,lw=1.35,ms=3,marker=mark,color=color)
    a.set(xscale="log",yscale="log",ylabel="Maximum probability loss (%)",xticks=DEPTHS,
          xticklabels=DEPTHS,xlabel="Completed encoded gates",ylim=(.00035,25))
    a.legend(handles=[Line2D([0],[0],color=ORANGE,marker="v",lw=1,label="Direct"),
                      Line2D([0],[0],color=VIOLET,marker="D",lw=1,label="Refined"),
                      Line2D([0],[0],color=".25",ls="--",lw=1,label=r"$\gamma=0$"),
                      Line2D([0],[0],color=".25",lw=1,label=r"$\gamma t_0=10^{-5}$")],
             loc="upper left",fontsize=6.6,ncol=2,columnspacing=.65,handlelength=1.25,
             borderaxespad=.15,handletextpad=.4)
    a.yaxis.set_major_locator(FixedLocator([.001,.01,.1,1,10]))
    a.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f"{v:g}"))
    band=[r for r in dep["brackets"] if r["N"]==4 and r["method"]=="refined"]
    loss=dep["readout"]["refined"]["1e-05"]["max_loss"]
    low=[100*max(r["open_grid_D_lower"],loss[str(r["depth"])]) for r in band]
    high=[100*r["open_grid_D_upper"] for r in band]
    b.fill_between(DEPTHS,low,high,color=VIOLET,alpha=.24,label="Dephasing bracket")
    b.plot(DEPTHS,low,color=VIOLET,lw=.8);b.plot(DEPTHS,high,color=VIOLET,lw=.8)
    b.plot(DEPTHS,[100*r["coherent_grid_D"] for r in band],linestyle="--",marker="o",ms=3,
           color="#34495E",lw=1.3,label="Coherent distance")
    b.axhline(1,color=".5",linestyle=":",lw=.8);b.set(xscale="log",yscale="log",xticks=DEPTHS,
           xticklabels=DEPTHS,ylabel=r"Grid maximum $D$ (%)",xlabel="Completed encoded gates",ylim=(.19,23))
    b.yaxis.set_major_locator(FixedLocator([.25,.5,1,2,5,10,20]))
    b.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f"{v:g}"))
    for ax in axs:ax.minorticks_off();ax.grid(axis="y",lw=.44,color=".86")
    b.legend(loc="upper left",fontsize=6.6,borderaxespad=.16);save(fig,out,"finite_time_dephasing")


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--out",required=True,type=Path)
    out=parser.parse_args().out.resolve();out.mkdir(parents=False,exist_ok=False);start=time.perf_counter()
    try:
        files=[Path(__file__).resolve(),OLD/"main.tex",OLD/"supplement.tex",REPORT/"all_validation_points.csv",
               REPORT/"summary.json",STARTS/"summary.json",NOISE/"summary.json"]
        dump(out/"inputs.json",{"hashes":{str(f.relative_to(ROOT)):digest(f) for f in files},
                                "utc":datetime.now(timezone.utc).isoformat(),"python":sys.version,
                                "platform":platform.platform(),"matplotlib":matplotlib.__version__,"numpy":np.__version__,
                                "precision":"float64 plotted data; vector PDF"})
        phase_figure(out);circuit_figure(out);dephasing_figure(out)
        dump(out/"summary.json",{"status":"three_vector_figures_and_source_data","wall_s":time.perf_counter()-start})
    except BaseException:
        dump(out/"failure.json",{"traceback":traceback.format_exc(),"wall_s":time.perf_counter()-start})
        raise
    finally:
        dump(out/"sha256.json",{f.name:digest(f) for f in sorted(out.iterdir()) if f.is_file()})


if __name__=="__main__":main()
