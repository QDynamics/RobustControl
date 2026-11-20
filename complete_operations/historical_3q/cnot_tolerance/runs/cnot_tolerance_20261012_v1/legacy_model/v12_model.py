"""Declared actuator, task-specific constraints, and finite-noise objectives."""
import numpy as np
import cnot_model_v1 as m
import v11_global_model as g

TB=.02; MIN_SLOT=.04; SWITCH=.001
CAP=np.tile(np.r_[np.full(15,10.),np.ones(15)],3)
ROBUST=('projected','nominal_projected','whole','module_response','modular')

def duration_and_slope(areas,caps=CAP):
    s=np.abs(np.asarray(areas))/caps; b=2*np.pi*TB
    lo=2*s; hi=2*s+np.sqrt(b*s)+MIN_SLOT
    for _ in range(48):
        tau=(lo+hi)/2
        above=tau>=s*(1+np.sqrt(1+(b/tau)**2))
        hi=np.where(above,tau,hi); lo=np.where(above,lo,tau)
    raw=(lo+hi)/2; tau=np.maximum(raw,MIN_SLOT)
    w=np.sqrt(1+(b/tau)**2)
    slope=np.where(raw>MIN_SLOT,(1+w)**2/(1+2*w-1/w)/caps,0.)
    return tau,slope

def gate_time(p): return float(duration_and_slope(np.asarray(p).ravel()[g.CHARGED])[0].sum()+89*SWITCH)
def cycle_time(p): return gate_time(p)+float(duration_and_slope(np.array([np.pi,np.pi/2]),10.)[0].sum())+2*SWITCH+.2

def proj(b,a): return g.project(b,a)

def constraints(p,kind,jac=True):
    p=np.asarray(p).ravel()
    if kind=='modular': return m.combined(p,jac=jac)[:2]
    u,du,ks,dks,ds=g.product_responses(p,jac)
    da=np.zeros(132); da.reshape(3,44)[:,29:]=1
    area=float(p.reshape(3,44)[:,29:].sum())
    w=m.UTARGET.conj().T@u; dw=m.UTARGET.conj().T@du
    if kind in ('projected','noise_projected'):
        w=np.exp(1j*area)*w
        dw=np.exp(1j*area)*dw+1j*da[:,None,None]*w
    h=.5j*(w-w.conj().T); dh=.5j*(dw-dw.conj().transpose(0,2,1))
    rv=proj(m.B4,h[:4,:4]); jv=proj(m.B4,dh[:,:4,:4]).T
    rh=proj(m.B2,h[4:,4:]); jh=proj(m.B2,dh[:,4:,4:]).T
    if kind=='noise_projected': return rv,jv
    rn=np.r_[area,rv,rh]; jn=np.vstack([da,jv,jh])
    if kind in ('noise_complete','nominal'): return rn,jn
    kr=proj(m.B4,ks[0,:4,:4]); jr=proj(m.B4,dks[0,:,:4,:4]).T
    kt=proj(m.B4,ks[1,:4,:4]); jt=proj(m.B4,dks[1,:,:4,:4]).T
    if kind=='projected': return np.r_[rv,kr,kt],np.vstack([jv,jr,jt])
    if kind=='nominal_projected': return np.r_[rn,kr,kt],np.vstack([jn,jr,jt])
    if kind=='whole': return g.global_constraints(p,jac)[:2]
    if kind!='module_response': raise ValueError(kind)
    rr=[]; jj=[]
    for k,d in enumerate(ds):
        _,_,km,dk=d
        dam=np.zeros(44); dam[29:]=1
        rr.append(np.r_[p.reshape(3,44)[k,29:].sum(),proj(m.B4,km[0,:4,:4]),proj(m.B2,km[0,4:,4:]),proj(m.B4,km[1,:4,:4])])
        block=np.vstack([dam,proj(m.B4,dk[0,:,:4,:4]).T,proj(m.B2,dk[0,:,4:,4:]).T,proj(m.B4,dk[1,:,:4,:4]).T])
        full=np.zeros((34,132)); full[:,44*k:44*(k+1)]=block; jj.append(full)
    return np.r_[*rr,rv,rh],np.vstack([*jj,jv,jh])

def matrix_audit(p,kind):
    u,du,ks,dks,ds=g.product_responses(p,False)
    phase=np.trace(m.UTARGET[:4,:4].conj().T@u[:4,:4])/4
    phase=phase/abs(phase) if abs(phase)>1e-15 else 1
    pn=float(np.linalg.norm(u[:4,:4]-phase*m.UTARGET[:4,:4]))
    full=float(np.linalg.norm(u-m.UTARGET))
    pv=[float(np.linalg.norm(k[:4,:4]-np.trace(k[:4,:4])*np.eye(4)/4)) for k in ks]
    whole=[float(np.linalg.norm(k)) for k in ks]
    modules=[[float(np.linalg.norm(k)) for k in d[2]] for d in ds]
    hidden=[float(np.linalg.norm(d[0][4:,4:]-np.trace(d[0][4:,4:])*np.eye(2)/2)) for d in ds]
    required=[pn] if kind in ('projected','noise_projected') else [full]
    if kind in ('projected','nominal_projected'): required+=pv
    if kind=='whole': required+=whole
    if kind in ('module_response','modular'): required+=np.ravel(modules).tolist()
    if kind=='modular': required+=hidden
    return {'projected_target_fro':pn,'complete_target_fro':full,'projected_response_fro':pv,'complete_response_fro':whole,'module_response_fro':modules,'module_hidden_nonscalar_fro':hidden,'required_max':max(required),'accepted':bool(max(required)<1e-8 and np.max(np.abs(p)-g.LIMITS)<1e-10),'gate_time_t0':gate_time(p),'cycle_time_t0':cycle_time(p),'rectangular_time_t0':m.resources(p)}

def ensemble(seed,count=16):
    x=np.random.default_rng(seed).uniform(-1,1,(count//2,8))
    return np.concatenate([x,-x])
TRAIN=ensemble(12092711)

def errors(row,alpha=.3): return .01*(row[:2]+alpha*row[2:].reshape(3,2))

def loss(p,kind,jac=True):
    projected=kind in ('noise_projected','projected')
    val=0.; grad=np.zeros(132)
    for row in TRAIN:
        es=errors(row)
        ds=[m.module(q,jac,errors=tuple(es[k])) for k,q in enumerate(np.asarray(p).reshape(3,44))]
        u=np.eye(6,dtype=complex); du=np.zeros((132,6,6),complex)
        for k in (2,1,0):
            v,dv,_,_=ds[k]
            if jac:
                du=v@du; du[44*k:44*(k+1)]+=dv@u
            u=v@u
        w=m.UTARGET.conj().T@u; dw=m.UTARGET.conj().T@du
        tr=np.trace(w[:4,:4]); dtr=np.trace(dw[:,:4,:4],axis1=1,axis2=2); dim=4
        if not projected:
            tr+=2*np.trace(w[4:,4:]); dtr+=2*np.trace(dw[:,4:,4:],axis1=1,axis2=2); dim=8
        val+=max(0.,1-abs(tr/dim)**2)/len(TRAIN)
        grad+=-2*np.real(tr.conjugate()*dtr)/dim**2/len(TRAIN)
    val+=1.5e-5*gate_time(p)
    _,slopes=duration_and_slope(np.asarray(p).ravel()[g.CHARGED])
    grad[g.CHARGED]+=1.5e-5*slopes*np.sign(np.asarray(p).ravel()[g.CHARGED])
    return float(val),grad
