"""Render the frozen v17 library and fresh validation; preserve the original matched ablation."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'research'))
import v12_model as v
RUNS=BASE/'research/runs'
CODES={'projected':'P','nominal_projected':'P0','whole':'C','module_response':'R','modular':'M','noise_projected':r'$F_P$','noise_complete':r'$F_C$','nominal':'N'}
COLORS={'projected':'#009E73','nominal_projected':'#56B4E9','whole':'#0072B2','module_response':'#E69F00','modular':'#D55E00','noise_projected':'#CC79A7','noise_complete':'#7B61A8','nominal':'#666666'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':10,'axes.titlesize':10,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
from matplotlib import font_manager
from matplotlib.text import Text
for name in ('times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf'):
    font_manager.fontManager.addfont(str(Path(r'C:\Windows\Fonts')/name))
plt.rcParams.update({'font.family':'Times New Roman','font.serif':['Times New Roman'],
    'mathtext.fontset':'custom','mathtext.rm':'Times New Roman','mathtext.it':'Times New Roman:italic',
    'mathtext.bf':'Times New Roman:bold','mathtext.bfit':'Times New Roman:bold:italic',
    'mathtext.sf':'Times New Roman','mathtext.tt':'Times New Roman','mathtext.fallback':'stix',
    'axes.unicode_minus':False,'font.size':10,'axes.labelsize':11,'axes.titlesize':11,'legend.fontsize':10})
OUT=BASE/'figures/v17_results_v2';OUT.mkdir(exist_ok=False)
inputs=[]
def read(p):
    inputs.append(p);return json.loads(p.read_text(encoding='utf-8'))
lib=read(RUNS/'v17_library_v1/result.json');selected=lib['selected']
mapped=read(RUNS/'v17_confirmation_v1/result.json');names=mapped['names']
def save(fig,name):
    for artist in fig.findobj(match=Text):
        artist.set_fontfamily('Times New Roman'); artist.set_math_fontfamily('custom')
        artist.set_fontsize(max(artist.get_fontsize(),10))
    for ext in ('pdf','png'):
        p=OUT/f'{name}.{ext}'
        if p.exists():raise FileExistsError(p)
        fig.savefig(p,bbox_inches='tight',dpi=180)
    plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(8.9,3.0),gridspec_kw={'wspace':.34})
ax=axs[0]
for kind in v.ROBUST:
    p=RUNS/('v12_search_'+kind+'_warm_v1')/'trace.jsonl';inputs.append(p)
    rows=[json.loads(s) for s in p.read_text().splitlines()]
    ax.plot([0]+[r['iteration']+1 for r in rows],[76.80886723498]+[r['gate_time_t0'] for r in rows],color=COLORS[kind],label=CODES[kind])
ax.set_xlabel('Outer search step');ax.set_ylabel(r'Shaped gate time $T_g/t_0$')
ax.legend(ncol=3,frameon=False);ax.set_title('(a) Same-start constraint ablation',loc='left',fontweight='bold',pad=10);ax.grid(alpha=.2)
ax=axs[1];x=np.arange(len(names))
rect=[selected[k]['audit']['rectangular_time_t0'] for k in names]
shaped=[selected[k]['audit']['gate_time_t0'] for k in names]
ax.bar(x-.18,rect,.35,color='#bbc4cc',label='Rectangular areas')
ax.bar(x+.18,shaped,.35,color=[COLORS[k] for k in names],label='Declared actuator')
ax.set_xticks(x,[CODES[k] for k in names]);ax.set_ylabel(r'Gate time / $t_0$')
ax.set_title('(b) Fixed selection library',loc='left',fontweight='bold',pad=10)
ax.legend(frameon=False,loc='upper left',fontsize=10)
save(fig,'cnot_manifold_cost_v17')

fig,axs=plt.subplots(1,2,figsize=(8.9,3.0),gridspec_kw={'wspace':.34})
p=RUNS/'v17_library_v1/whole.npz';inputs.append(p);params=np.load(p)['params']
angle=float(params[:,:15].ravel()[np.argmax(np.abs(params[:,:15]))])
tau=float(v.duration_and_slope(np.array([angle]),10.)[0][0]);t=np.linspace(0,tau,500)
out=2*angle/tau*np.sin(np.pi*t/tau)**2
cmd=out+v.TB*2*np.pi*angle/tau**2*np.sin(2*np.pi*t/tau)
ax=axs[0];ax.plot(t/tau,cmd/10,'--',color='#D55E00',label='Command');ax.plot(t/tau,out/10,color='#0072B2',label='Physical output')
ax.axhline(np.sign(angle),color='gray',ls=':',lw=1);ax.set_xlabel(r'Normalized slot time $t/\tau$');ax.set_ylabel(r'Field / $\Omega_{\max}$')
ax.set_title('(a) Compensated rotation of C',loc='left',fontweight='bold',pad=10);ax.legend(frameon=False);ax.grid(alpha=.2)
ax=axs[1]
for kind in ('whole','module_response','modular','noise_complete','nominal'):
    vals=[mapped['witness'][kind+'_'+str(a)+'_3']['max_abs_bias'] for a in (0,4,6)]
    ax.plot([0,.3,1],vals,'-o',ms=4,color=COLORS[kind],label=CODES[kind])
ax.set_yscale('log');ax.set_xlabel(r'Module-drift amplitude $\alpha$');ax.set_ylabel(r'Max sampled $|p_\pm-7/18|$')
ax.set_title('(b) Noisy addressed phase readout',loc='left',fontweight='bold',pad=10);ax.legend(ncol=3,frameon=False);ax.grid(alpha=.2)
save(fig,'actuator_readout_v17')


from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch
means=np.asarray(mapped['means'])
fig,axs=plt.subplots(2,2,figsize=(8.9,6.3),gridspec_kw={'hspace':.5,'wspace':.33})
for ax,key,title,letter in zip(axs.ravel()[:3],('projected_finite','complete_finite','complete_q1_shared'),('Projected task: finite-error score','Complete task: finite-error score','Complete task: shared first-order required'),('a','b','c')):
    ranking=mapped['rankings'][key]; winner=np.array(ranking['winner']); resolved=np.array(ranking['resolved']); palette=[COLORS[k] for k in names]
    ax.imshow(winner.T,origin='lower',aspect='auto',cmap=ListedColormap(palette),norm=BoundaryNorm(np.arange(len(names)+1)-.5,len(names)))
    for a in range(winner.shape[0]):
        for b in range(winner.shape[1]):
            name=names[winner[a,b]]; code=CODES[name]; ax.text(a,b,code,color='white' if name in ('whole','noise_complete','nominal') else '#17212b',ha='center',va='center',fontsize=8,fontweight='bold')
            if not resolved[a,b]: ax.plot(a+.32,b+.27,'x',color='black',ms=4,mew=1)
    ax.set_xticks(range(7),['0','.01','.03','.1','.3','.6','1']); ax.set_yticks(range(6),['0',r'$10^{-7}$',r'$10^{-6}$',r'$10^{-5}$',r'$10^{-4}$',r'$10^{-3}$']); ax.set_xlabel(r'Module-drift amplitude $\alpha$'); ax.set_ylabel(r'Local dephasing $\gamma t_0$'); ax.set_title('('+letter+') '+title,loc='left',fontweight='bold',pad=9)
ax=axs[1,1]
for kind in ('whole','module_response','modular','noise_complete','nominal'):
    i=names.index(kind); ax.plot(mapped['alphas'],means[i,:,3,1],'-o',color=COLORS[kind],ms=3,label=CODES[kind])
ax.set_yscale('log'); ax.set_xscale('symlog',linthresh=.01); ax.set_xlim(-.0007,1.06); ax.set_xlabel(r'Module-drift amplitude $\alpha$'); ax.set_ylabel(r'Mean complete error $1-F_e$'); ax.set_xticks([0,.01,.03,.1,.3,.6,1],['0','.01','.03','.1','.3','.6','1']); ax.set_title(r'(d) Error scale at $\gamma t_0=10^{-5}$',loc='left',fontweight='bold',pad=9); ax.grid(alpha=.2); ax.legend(ncol=3,loc='best',frameon=False)
legend=[Patch(facecolor=COLORS[k],label=CODES[k]) for k in names]
fig.legend(handles=legend,ncol=8,loc='lower center',bbox_to_anchor=(.5,-.035),frameon=False,handlelength=1.2,columnspacing=1.6)
save(fig,'interface_selection_v17')

(OUT/'bindings.json').write_text(json.dumps({str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs+[Path(__file__)]},indent=2),encoding='utf-8')
print('Rendered three v17 figures')
