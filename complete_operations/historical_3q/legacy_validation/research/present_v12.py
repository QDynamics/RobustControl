"""Generate figures/tables directly from the frozen validation records."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch
import v12_model as v
ROOT=Path(__file__).resolve().parents[1]; RUNS=ROOT/'research/runs'
CODES={'projected':'P','nominal_projected':'P0','whole':'C','module_response':'R','modular':'M','noise_projected':r'$F_P$','noise_complete':r'$F_C$','nominal':'N'}
COLORS={'projected':'#009E73','nominal_projected':'#56B4E9','whole':'#0072B2','module_response':'#E69F00','modular':'#D55E00','noise_projected':'#CC79A7','noise_complete':'#7B61A8','nominal':'#666666'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':10,'axes.titlesize':10,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
lib=json.loads((RUNS/'v12_library_v1/result.json').read_text()); selected=lib['selected']; mapped=json.loads((RUNS/'v12_map_v1/result.json').read_text()); pre=json.loads((RUNS/'v12_channel_preflight_v1/result.json').read_text()); names=mapped['names']; arrays=np.load(RUNS/'v12_map_v1/arrays.npz'); means=arrays['metrics'].mean(axis=3)
def save(fig,name):
    fig.savefig(ROOT/'figures'/(name+'.pdf'),bbox_inches='tight'); fig.savefig(ROOT/'figures'/(name+'.png'),dpi=180,bbox_inches='tight'); plt.close(fig)
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
ax.set_yscale('log'); ax.set_xscale('symlog',linthresh=.01); ax.set_xlabel(r'Module-drift amplitude $\alpha$'); ax.set_ylabel(r'Mean complete error $1-F_e$'); ax.set_xticks([0,.01,.1,1],['0','.01','.1','1']); ax.set_title(r'(d) Error scale at $\gamma t_0=10^{-5}$',loc='left',fontweight='bold',pad=9); ax.grid(alpha=.2); ax.legend(ncol=3,loc='best',frameon=False)
legend=[Patch(facecolor=COLORS[k],label=CODES[k]) for k in names]
fig.legend(handles=legend,ncol=8,loc='lower center',bbox_to_anchor=(.5,-.035),frameon=False,handlelength=1.2,columnspacing=1.6)
save(fig,'interface_selection_v12')

fig,axs=plt.subplots(2,2,figsize=(8.9,6.2),gridspec_kw={'hspace':.52,'wspace':.33})
ax=axs[0,0]
for kind in v.ROBUST:
    trial=RUNS/('v12_search_'+kind+'_warm_v1'); rows=[json.loads(s) for s in (trial/'trace.jsonl').read_text().splitlines()]
    ax.plot([0]+[r['iteration']+1 for r in rows],[76.80886723498]+[r['gate_time_t0'] for r in rows],color=COLORS[kind],label=CODES[kind])
ax.set_xlabel('Outer search step'); ax.set_ylabel(r'Shaped gate time $T_g/t_0$'); ax.legend(ncol=3,frameon=False); ax.set_title('(a) Same-start constraint ablation',loc='left',fontweight='bold',pad=9); ax.grid(alpha=.2)
ax=axs[0,1]; x=np.arange(len(names)); rect=[selected[k]['audit']['rectangular_time_t0'] for k in names]; shaped=[selected[k]['audit']['gate_time_t0'] for k in names]
ax.bar(x-.18,rect,.35,color='#bbc4cc',label='Rectangular areas'); ax.bar(x+.18,shaped,.35,color=[COLORS[k] for k in names],label='Declared actuator'); ax.set_xticks(x,[CODES[k] for k in names]); ax.set_ylabel(r'Gate time / $t_0$'); ax.set_title('(b) Fixed selection library',loc='left',fontweight='bold',pad=9); ax.legend(frameon=False,loc='upper right',fontsize=7)
ax=axs[1,0]; p=np.load(RUNS/'v12_library_v1/whole.npz')['params']; angle=float(p[:,:15].ravel()[np.argmax(np.abs(p[:,:15]))]); tau=float(v.duration_and_slope(np.array([angle]),10.)[0][0]); t=np.linspace(0,tau,500); out=2*angle/tau*np.sin(np.pi*t/tau)**2; cmd=out+v.TB*2*np.pi*angle/tau**2*np.sin(2*np.pi*t/tau)
ax.plot(t/tau,cmd/10,'--',color='#D55E00',label='Command'); ax.plot(t/tau,out/10,color='#0072B2',label='Physical output'); ax.axhline(np.sign(angle),color='gray',ls=':',lw=1); ax.set_xlabel(r'Normalized slot time $t/\tau$'); ax.set_ylabel(r'Field / $\Omega_{\max}$'); ax.set_title('(c) Compensated rotation of C',loc='left',fontweight='bold',pad=9); ax.legend(frameon=False); ax.grid(alpha=.2)
ax=axs[1,1]
for kind in ('whole','module_response','modular','noise_complete','nominal'):
    vals=[mapped['witness'][kind+'_'+str(a)+'_3']['max_abs_bias'] for a in (0,4,6)]; ax.plot([0,.3,1],vals,'-o',ms=4,color=COLORS[kind],label=CODES[kind])
ax.set_yscale('log'); ax.set_xlabel(r'Module-drift amplitude $\alpha$'); ax.set_ylabel(r'Max sampled $|p_\pm-7/18|$'); ax.set_title('(d) Noisy addressed phase readout',loc='left',fontweight='bold',pad=9); ax.legend(ncol=3,frameon=False); ax.grid(alpha=.2)
save(fig,'interface_cost_readout_v12')

overhead=selected['whole']['audit']['cycle_time_t0']-selected['whole']['audit']['gate_time_t0']
(ROOT/'v12_numbers.tex').write_text('\\newcommand{\\VtwelveOverhead}{%.5f}\n'%overhead,encoding='utf-8')
lines=[r'\begin{table}[tb]',r'\caption{Library fixed before validation. Times use the declared actuator; cycle times include addressed preparation/analysis, all dead intervals, initialization/reset, and readout. ``Full'' refers to the nominal target; exact coherent cancellation requirements are distinguished in the text.}',r'\label{tab:v12-library}',r'\centering\small',r'\begin{ruledtabular}',r'\begin{tabular}{lcrr}',r'Code & Full nominal & $T_g/t_0$ & $T_{\rm cycle}/t_0$\\',r'\hline']
for k in names:
    code=CODES[k]; a=selected[k]['audit']; lines.append('%s & %s & %.3f & %.3f \\\\'%(code,'No' if k in ('projected','noise_projected') else 'Yes',a['gate_time_t0'],a['cycle_time_t0']))
lines.extend([r'\end{tabular}',r'\end{ruledtabular}',r'\end{table}']); (ROOT/'v12_library_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')

rows=lib['all_trials']; lines=[r'\begin{table}[htb]',r'\caption{Multistart acceptance and achieved gate times. The final five-start group uses v2 non-warm starts plus the v1 warm start for P and C, and for $F_C$; $F_P$ uses all five v3 starts. Rejected v1 trials remain separately archived. Success means the matrix constraints met the registered tolerance, not optimizer convergence.}',r'\centering\begin{ruledtabular}',r'\begin{tabular}{lrrrrr}',r'Family & v1 accepted/5 & Final accepted/5 & Warm $T_g/t_0$ & Nominal $T_g/t_0$ & Random range $T_g/t_0$\\',r'\hline']
for kind in ('projected','whole','noise_projected','noise_complete'):
    old=[r for r in rows if r['kind']==kind and r['directory'].endswith('_v1')]
    final=[r for r in rows if r['kind']==kind and (r['directory'].endswith('_v3') if kind=='noise_projected' else (r['directory'].endswith('_v2') or '_warm_v1' in r['directory']))]
    def val(label):
        rr=[r for r in final if '_'+label+'_' in r['directory']]; return ('%.3f'%rr[0]['audit']['gate_time_t0']) if rr and rr[0]['accepted'] else '--'
    random=[r['audit']['gate_time_t0'] for r in final if 'random' in r['directory'] and r['accepted']]
    ran='%.3f--%.3f'%(min(random),max(random)) if random else '--'
    lines.append('%s & %s & %s & %s & %s & %s \\\\'%(CODES[kind],str(sum(r['accepted'] for r in old))+'/5',str(sum(r['accepted'] for r in final))+'/5',val('warm'),val('nominal'),ran))
lines.extend([r'\end{tabular}\end{ruledtabular}',r'\end{table}'])
lines.append('The final comparison includes %d recorded search trials across the original and successor procedures. Complete robust and complete nominal random starts that fail restoration are retained; no replacement random seed is drawn. Recorded iteration and wall-clock limits, rather than convergence or a global optimum, define the budget.'%len(rows))
(ROOT/'v12_search_summary.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')

maxmatrix=max(r['required_max'] for r in pre['physical'].values()); maxunit=max(r['unitary_super_fro'] for r in pre['physical'].values()); maxchange=max(r['metric_witness_change'] for r in pre['convergence']); mintp=max(r['cptp']['trace_preservation_fro'] for r in pre['convergence']); local=max(r['adaptive_local_fro'] for r in pre['local_adaptive'])
summary=('All eight selected controls pass independent physical-basis target/response acceptance; the largest required Frobenius residual is $%.2g$. Zero-dephasing superoperators agree with independent eight-dimensional unitary propagation within $%.2g$ in Frobenius norm. The registered pulse-doubling checks select %d substeps for all map calculations; the largest tested change in a score or witness probability is $%.2g$, below $2\\times10^{-7}$. Independent adaptive single-qubit channels agree within $%.2g$ in Frobenius norm. Registered full-channel examples have trace-preservation residual below $%.2g$ and nonnegative Choi eigenvalues up to rounding. These checks bound observed discretization differences, not all-point rigorous integration errors. The map contains %d channel evaluations on %d candidate/parameter cells.\n'%(maxmatrix,maxunit,pre['steps'],maxchange,local,mintp,mapped['channel_evaluations'],mapped['cells']))
(ROOT/'v12_validation_summary.tex').write_text(summary,encoding='utf-8')
print(json.dumps({'figures':2,'library':{k:round(r['audit']['gate_time_t0'],4) for k,r in selected.items()},'max_matrix_residual':maxmatrix,'max_convergence_change':maxchange,'steps':pre['steps']}))
