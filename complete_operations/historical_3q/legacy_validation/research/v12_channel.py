"""Full eight-dimensional CPTP propagation; no symmetry-sector truncation."""
import numpy as np
from scipy.integrate import solve_ivp
import v12_model as v
import v11_full8 as f
import v11_global_model as g

# Interleaved Liouville index (i0,j0,i1,j1,i2,j2).
BITS=np.array([[k>>2&1,k>>1&1,k&1] for k in range(8)])
PAIRS=np.array([[(k//16)%4,(k//4)%4,k%4] for k in range(64)])
RI=(PAIRS//2)@np.array([4,2,1]); RJ=(PAIRS%2)@np.array([4,2,1])
PERM=8*RI+RJ; INV=np.argsort(PERM)
HAMMING=np.sum(BITS[RI]!=BITS[RJ],axis=1)
E=2*(1.5-BITS.sum(axis=1))**2-1.5
DELTA_E=E[RI]-E[RJ]
TRACE=(RI==RJ).astype(float)
# The common minus sign of the target cancels in its channel.
def superop(u): return np.kron(u,u.conj())[PERM][:,PERM]
TARGET_FULL=superop(f.TARGET)
TARGET_P=superop(f.P@f.TARGET@f.P)
P_VECTOR=f.P.reshape(64)[PERM]
P_BRA=f.P.T.reshape(64)[PERM]

def local_unitaries(angles,phis):
    angles=np.asarray(angles); phis=np.asarray(phis)
    u=np.zeros((angles.size,2,2),complex)
    u[:,0,0]=u[:,1,1]=np.cos(angles/2)
    u[:,0,1]=-1j*np.sin(angles/2)*np.exp(-1j*phis)
    u[:,1,0]=-1j*np.sin(angles/2)*np.exp(1j*phis)
    return u

def local_channels(angles,phis,durations,gamma,steps):
    angles=np.atleast_1d(angles); phis=np.broadcast_to(phis,angles.shape); durations=np.broadcast_to(durations,angles.shape)
    if gamma==0:
        u=local_unitaries(angles,phis)
        return np.einsum('aik,ajl->aijkl',u,u.conj()).reshape(-1,4,4)
    dt=durations/steps; decay=np.exp(-gamma*dt/2)
    d=np.stack([np.ones_like(decay),decay,decay,np.ones_like(decay)],axis=1)
    result=np.broadcast_to(np.eye(4,dtype=complex),(len(angles),4,4)).copy()
    for k in range(steps):
        frac=1/steps-(np.sin(2*np.pi*(k+1)/steps)-np.sin(2*np.pi*k/steps))/(2*np.pi)
        u=local_unitaries(angles*frac,phis)
        su=np.einsum('aik,ajl->aijkl',u,u.conj()).reshape(-1,4,4)
        result=(d[:,:,None]*su*d[:,None,:])@result
    return result

def apply_local(bank,local,qubit):
    shape=bank.shape; x=bank.reshape(4,4,4,-1)
    y=np.tensordot(local,x,axes=(1,qubit))
    return np.moveaxis(y,0,qubit).reshape(shape)

def compile_controls(p):
    p=np.asarray(p).reshape(3,44)
    tau=v.duration_and_slope(p.ravel()[g.CHARGED])[0].reshape(3,30)
    slots=[]; angles=[]; phis=[]; durations=[]; modules=[]
    for j in (2,1,0):
        for k in range(15):
            slots.append((j,p[j,29+k],tau[j,15+k],len(angles)))
            angles.append(p[j,k]); phis.append(0. if k==0 else p[j,14+k]); durations.append(tau[j,k]); modules.append(j)
    return {'slots':slots,'angles':np.array(angles),'phis':np.array(phis),'durations':np.array(durations),'modules':np.array(modules),'params':p}

def propagate(control,errors,gamma,steps=32,bank=None):
    """Propagate arbitrary operator columns, or a complete channel if bank omitted."""
    errors=np.broadcast_to(errors,(3,2))
    rots=local_channels(control['angles']*(1+errors[control['modules'],0]),control['phis'],control['durations'],gamma,steps)
    s=np.eye(64,dtype=complex) if bank is None else np.array(bank,dtype=complex,copy=True)
    idle=np.exp(-gamma*v.SWITCH*HAMMING)
    for n,(j,chi,tau,k) in enumerate(control['slots']):
        factor=np.exp(-1j*(1+errors[j,1])*chi*DELTA_E-gamma*tau*HAMMING)
        s*=factor[:,None] if s.ndim==2 else factor
        s*=idle[:,None] if s.ndim==2 else idle
        for qubit in range(3): s=apply_local(s,rots[k],qubit)
        if n!=44: s*=idle[:,None] if s.ndim==2 else idle
    return s

def metrics(channel):
    full=float(1-np.vdot(TARGET_FULL,channel).real/64)
    projected=float(1-np.vdot(TARGET_P,channel).real/16)
    survival=float(np.real(P_BRA@channel@P_VECTOR)/4)
    return np.array([max(0,projected),max(0,full),survival])

def witness(control,errors,gamma,steps=32,prep_error=0.,analysis_error=0.):
    times=v.duration_and_slope(np.array([np.pi,np.pi/2]),10.)[0]
    rho=np.zeros(64,complex); rho[np.where(PERM==0)[0][0]]=1
    pre=local_channels(np.array([np.pi*(1+prep_error)]),0.,times[0],gamma,steps)[0]
    rho=apply_local(rho,pre,0)
    # The other two physical qubits dephase during addressed operations too.
    other=(BITS[RI,1:]!=BITS[RJ,1:]).sum(axis=1)
    rho*=np.exp(-gamma*(times[0]*other+v.SWITCH*HAMMING))
    rho=propagate(control,errors,gamma,steps,rho)
    rho*=np.exp(-gamma*v.SWITCH*HAMMING)
    probs=[]
    for phi in (-np.pi/4,3*np.pi/4):
        ana=local_channels(np.array([np.pi/2*(1+analysis_error)]),phi,times[1],gamma,steps)[0]
        final=apply_local(rho,ana,0)*np.exp(-gamma*times[1]*other)
        probs.append(float(final[np.where(PERM==0)[0][0]].real))
    return np.array(probs)

def channel_checks(s):
    ordinary=s[INV][:,INV].reshape(8,8,8,8)
    choi=ordinary.transpose(0,2,1,3).reshape(64,64)
    return {'trace_preservation_fro':float(np.linalg.norm(TRACE@s-TRACE)),'choi_hermitian_fro':float(np.linalg.norm(choi-choi.conj().T)),'choi_min_eigenvalue':float(np.linalg.eigvalsh((choi+choi.conj().T)/2)[0]),'choi_trace':float(np.trace(choi).real)}

def adaptive_local_channel(angle,phi,tau,gamma):
    h=np.array([[0,np.exp(-1j*phi)],[np.exp(1j*phi),0]],complex)/2
    z=np.diag([1.,-1.]); ident=np.eye(2)
    lc=-1j*(np.kron(h,ident)-np.kron(ident,h.T))
    ld=gamma/2*(np.kron(z,z)-np.eye(4))
    def rhs(t,s): return ((2*angle/tau*np.sin(np.pi*t/tau)**2*lc+ld)@s.reshape(4,4)).ravel()
    sol=solve_ivp(rhs,(0,tau),np.eye(4,dtype=complex).ravel(),method='DOP853',rtol=2e-11,atol=2e-13,max_step=tau/32)
    return sol.y[:,-1].reshape(4,4)
