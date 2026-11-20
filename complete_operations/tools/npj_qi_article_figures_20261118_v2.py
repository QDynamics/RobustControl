"""Article figures with exact finite-order reachability identified in the phase panel.

The depth and dephasing panels use the original registered numerical results and
generators. Only the phase-panel text changes from the first manuscript render.
"""
import argparse
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime, timezone
import platform

import npj_qi_article_figures_20261118_v1 as previous
from npj_qi_article_figures_20261118_v1 import (
    ROOT, REPORT, STARTS, NOISE, OLD, TEAL, ORANGE, BLUE,
    np, plt, matplotlib, write_csv, title, save, digest, dump,
    circuit_figure, dephasing_figure,
)


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
    a.text(.97,.81,"Correctable?",color="#484E55",fontsize=8,ha="right",va="center")
    a.plot([.03,.97],[.69,.69],c="#A5AAB0",lw=.8)
    for yy,lhs,col,result in [(.52,r"$\mathrm{CNOT}_{L}$",ORANGE,"Excluded"),
                              (.22,r"$e^{i\pi/4}\mathrm{CNOT}_{L}$",TEAL,"Admitted")]:
        a.text(.04,yy,lhs,va="center",fontsize=9.3);a.text(.61,yy,r"$I_Q$",va="center",fontsize=9)
        a.text(.97,yy,result,ha="right",va="center",color=col,fontsize=7.8,fontweight="bold")
    b=fig.add_subplot(grid[0,1]);title(b,"b","Which phases admit exact correction?")
    b.set_xlim(-.11,2.08);b.set_ylim(-.45,.65)
    b.axhline(0,lw=.8,c="#A5AAB0")
    b.scatter([.25,.75,1.25,1.75],[0]*4,marker="o",color=TEAL,s=29,zorder=4)
    b.scatter([0],[0],color=ORANGE,s=30,marker="x",linewidths=1.5,zorder=4)
    b.text(.03,.84,r"Allowed:  $\phi=\pi/4 \ (\mathrm{mod}\ \pi/2)$",transform=b.transAxes,color=TEAL,fontsize=8.1)
    b.text(.03,.04,r"At $\phi=0$: robust-set gap $D=\sin(\pi/8)$",transform=b.transAxes,fontsize=7.6)
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


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--out",required=True,type=Path)
    out=parser.parse_args().out.resolve();out.mkdir(parents=False,exist_ok=False);start=time.perf_counter()
    try:
        files=[Path(__file__).resolve(),Path(previous.__file__).resolve(),OLD/"main.tex",OLD/"supplement.tex",REPORT/"all_validation_points.csv",
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
