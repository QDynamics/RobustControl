"""Plot the accepted CNOT bound and continuous-family certificate."""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(ROOT/'tmp/cnot_tolerance_mpl_20261012_v1')
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import font_manager

RUN=ROOT/'runs/cnot_tolerance_20261012_v1'
OUT=ROOT/'output/pdf/cnot_tolerance_20261012_v1'
OUT.mkdir(parents=True,exist_ok=True)
assert not (OUT/'cnot_tolerance_summary.pdf').exists() and not (OUT/'cnot_tolerance_summary.png').exists()
font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
plt.rcParams.update({'font.family':'Times New Roman','font.size':10,'mathtext.fontset':'custom',
 'mathtext.rm':'Times New Roman','mathtext.it':'Times New Roman:italic','mathtext.bf':'Times New Roman:bold',
 'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
cert=json.loads((RUN/'interval_v2/RESULTS.json').read_text())['results']
num=json.loads((RUN/'numerical/RESULTS.json').read_text())['results']
audit=json.loads((RUN/'primary_audit/ACCEPTANCE.json').read_text())['continuous_family']
fig,axes=plt.subplots(1,2,figsize=(9.4,3.8),constrained_layout=True)
colors={'whole':'#1668a9','module_response':'#318b68','modular':'#978141','noise_complete':'#b45a8d','nominal':'#686868'}
labels={'whole':'C: joint correction','module_response':'R: module correction','modular':'M: closed modules','noise_complete':r'$F_C$: finite-noise search','nominal':'N: nominal search'}
rr=np.geomspace(1e-4,.03,280)
for name in colors:
    row=cert[name];degree=row['degree_sum_bounds'];s=row['action_sum_upper']
    yy=sum(degree[k]*rr**k for k in range(9))+(s*rr)**9/362880
    axes[0].loglog(100*rr,yy,color=colors[name],lw=1.8 if name=='whole' else 1.2,label=labels[name])
    r=row['noise_radii']['0.01']['delivered_radius']
    axes[0].plot(100*r,.01,'o',color=colors[name],ms=3.5)
axes[0].axhline(.01,ls='--',lw=1,color='black')
axes[0].text(.012,.012,r'Tolerance $\delta=0.01$',fontsize=9)
axes[0].set(xlim=(.01,3),ylim=(1e-6,.7),xlabel=r'Shared calibration half-width $r$ (%)',ylabel='Continuous-domain upper bound')
axes[0].legend(loc='lower right',fontsize=8.4,frameon=False)
axes[0].set_title('(a) Complete-channel tolerance',loc='left',fontsize=11)
ss=[-.04,-.02,0.,.02,.04]
first=True
for i,edge in enumerate(audit['segments']):
    t=np.linspace(0,1,81);s=ss[i]+t*(ss[i+1]-ss[i])
    left=cert[edge['left']]['bounds']['8']['0.01'];right=cert[edge['right']]['bounds']['8']['0.01']
    ub=np.maximum(left,right)+.5*t*(1-t)*(1.01*edge['weighted_l1_edge_upper'])**2
    addition=1.01*.0037+audit['dephasing_contribution_upper']
    axes[1].plot(s,ub,color='#1668a9',lw=1.7,label='Control path: verified bound' if first else None)
    axes[1].plot(s,ub+addition,color='#8e4aa8',lw=1.7,label='With control budget and dephasing' if first else None)
    axes[1].fill_between(s,ub,ub+addition,color='#8e4aa8',alpha=.10)
    sample_x=[ss[i]+x['t']*(ss[i+1]-ss[i]) for x in edge['sampled_diagnostics']]
    sample_y=[x['nine_point_max_halfdiamond'] for x in edge['sampled_diagnostics']]
    axes[1].plot(sample_x,sample_y,'o',ms=3,color='#318b68',label='Sampled error (diagnostic)' if first else None)
    first=False
axes[1].axhline(.01,ls='--',lw=1,color='black')
axes[1].set(xlim=(-.042,.042),ylim=(0,.013),xlabel=r'Area-family coordinate $s$',ylabel='Half-diamond error / upper bound')
axes[1].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
axes[1].set_title('(b) Entire interpolating control family',loc='left',fontsize=11)
axes[1].legend(loc='upper center',bbox_to_anchor=(.5,.995),fontsize=8.2,frameon=False)
axes[1].text(.02,.01035,r'$r=1\%$',fontsize=9)
fig.savefig(OUT/'cnot_tolerance_summary.pdf')
fig.savefig(OUT/'cnot_tolerance_summary.png',dpi=180)
plt.close(fig)
print(OUT)
