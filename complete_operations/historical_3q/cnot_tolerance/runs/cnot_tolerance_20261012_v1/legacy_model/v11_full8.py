"""Independent physical-basis propagation, module-error responses and readout."""
from functools import reduce
import numpy as np

I=np.eye(2,dtype=complex)
X=np.array([[0,1],[1,0]],complex)
Y=np.array([[0,-1j],[1j,0]],complex)
Z=np.diag([1.,-1.]).astype(complex)
def tensor(parts): return reduce(np.kron,parts)
JX=sum(tensor([X if k==a else I for k in range(3)]) for a in range(3))/2
JY=sum(tensor([Y if k==a else I for k in range(3)]) for a in range(3))/2
ZZ=sum(tensor([Z if k in (a,b) else I for k in range(3)]) for a in range(3) for b in range(a+1,3))
VD=np.zeros((8,4),complex); VD[7,0]=VD[0,3]=1
VD[[3,5,6],1]=1/np.sqrt(3); VD[[1,2,4],2]=1/np.sqrt(3)
VH=np.zeros((8,2),complex); VH[3,0]=VH[2,1]=1/np.sqrt(2); VH[5,0]=VH[4,1]=-1/np.sqrt(2)
P=VD@VD.conj().T; Q=np.eye(8)-P
CNOT=np.eye(4,dtype=complex); CNOT[2:,2:]=X
TARGET=-(np.exp(1j*np.pi/4)*VD@CNOT@VD.conj().T+Q)

def schedule(params):
    out=[]
    for j in (2,1,0):
        q=np.asarray(params).reshape(3,44)[j]
        phi=np.r_[0.,q[15:29]]
        for theta,angle,chi in zip(q[:15],phi,q[29:]):
            for a,h in [(1,chi*ZZ),(0,theta*(np.cos(angle)*JX+np.sin(angle)*JY))]:
                e,v=np.linalg.eigh(h); out.append((j,a,h,e,v))
    return out

def propagate(seq,errors=None,responses=False):
    err=np.zeros((3,2)) if errors is None else np.broadcast_to(np.asarray(errors),(3,2))
    u=np.eye(8,dtype=complex); ks=np.zeros((3,2,8,8),complex)
    for j,a,h,e,v in seq:
        if responses: ks[j,a]+=u.conj().T@h@u
        op=(v*np.exp(-1j*(1+err[j,a])*e))@v.conj().T
        u=op@u
    return u,ks

def half_diamond(u,target=TARGET):
    phases=np.sort(np.angle(np.linalg.eigvals(target.conj().T@u)))
    width=max(0.,2*np.pi-float(np.max(np.diff(np.r_[phases,phases[0]+2*np.pi]))))
    return 1. if width>=np.pi else float(np.sin(width/2))

def local_rotation(angle,phi):
    return tensor([np.cos(angle/2)*I-1j*np.sin(angle/2)*(np.cos(phi)*X+np.sin(phi)*Y),I,I])

def probability(u,phi=-np.pi/4,prep_error=0.,analysis_error=0.):
    psi=local_rotation(np.pi*(1+prep_error),0.)[:,0]
    psi=local_rotation(np.pi*(1+analysis_error)/2,phi)@u@psi
    return float(abs(psi[0])**2)

def audit(params):
    u,k=propagate(schedule(params),responses=True)
    shared=k.sum(axis=0)
    return {'target_fro':float(np.linalg.norm(u-TARGET)),
            'shared_response_fro':[float(np.linalg.norm(a)) for a in shared],
            'module_response_fro':[[float(np.linalg.norm(a)) for a in b] for b in k],
            'unitarity_fro':float(np.linalg.norm(u.conj().T@u-np.eye(8))),
            'leakage_fro':float(np.linalg.norm(Q@u@VD)),
            'witness_probability':probability(u),
            'witness_nominal_residual':abs(probability(u)-7/18)}
