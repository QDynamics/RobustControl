"""N=3 spin-block search model; full8 acceptance is implemented separately."""
from __future__ import annotations
import numpy as np
from scipy.linalg import block_diag

ROWS = np.array((0,1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,26,27,28,29,30,31,32,33,34,35,36,37,38))
I2=np.eye(2,dtype=complex)
PAULI=[I2,np.array([[0,1],[1,0]],complex),np.array([[0,-1j],[1j,0]]),np.diag([1.,-1.])]
B4=np.array([np.kron(a,b)/2 for i,a in enumerate(PAULI) for j,b in enumerate(PAULI) if (i,j)!=(0,0)])
B16=np.concatenate((np.eye(4,dtype=complex)[None]/2,B4))
B2=np.array(PAULI[1:])/np.sqrt(2)
X4=np.diag([np.sqrt(3)/2,1,np.sqrt(3)/2],1)+np.diag([np.sqrt(3)/2,1,np.sqrt(3)/2],-1)
JX=block_diag(X4,PAULI[1]/2)
JZ=np.diag([-1.5,-.5,.5,1.5,-.5,.5])
JY=-1j*(JZ@JX-JX@JZ)
HZ=np.diag([3.,-1,-1,3,-1,-1])
E,V=np.linalg.eigh(JX)
EYE=np.eye(6,dtype=complex)
CNOT=np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],complex)
GTARGET=np.exp(1j*np.pi/4)*CNOT
UTARGET=block_diag(-GTARGET,-I2)

def coords(h):
    d=h.shape[-1]
    out=[h[...,i,i].real for i in range(d)]
    for i in range(d):
        for j in range(i+1,d): out.extend((h[...,i,j].real,h[...,i,j].imag))
    return np.stack(out,axis=-1)

def rotation(theta,phi):
    z=np.exp(-1j*phi*np.diag(JZ))
    r=(V*np.exp(-1j*theta*E))@V.conj().T
    return z[:,None]*r*z.conj()[None,:]

def module(p,jac=True,errors=(0.,0.)):
    """U, dU, input-frame K_R,K_T and control derivatives."""
    theta=p[:15]; phi=np.r_[0.,p[15:29]]; chi=p[29:]
    u=EYE.copy(); du=np.zeros((44,6,6),complex)
    ks=np.zeros((2,6,6),complex); dks=np.zeros((2,44,6,6),complex)
    for k in range(15):
        for typ in range(2):
            if typ==0:
                h=chi[k]*HZ; op=np.diag(np.exp(-1j*(1+errors[1])*np.diag(h)))
                dh={29+k:HZ}; dop={29+k:-1j*(1+errors[1])*HZ@op}; a=1
            else:
                axis=np.cos(phi[k])*JX+np.sin(phi[k])*JY
                daxis=-np.sin(phi[k])*JX+np.cos(phi[k])*JY
                h=theta[k]*axis; op=rotation((1+errors[0])*theta[k],phi[k]); a=0
                dh={k:axis}; dop={k:-1j*(1+errors[0])*axis@op}
                if k:
                    dh[14+k]=theta[k]*daxis
                    dop[14+k]=-1j*(JZ@op-op@JZ)
            adj=u.conj().T
            ks[a]+=adj@h@u
            if jac:
                dks[a]+=du.conj().transpose(0,2,1)@h@u+adj@h@du
                for c,v in dh.items(): dks[a,c]+=adj@v@u
                du=op@du
                for c,v in dop.items(): du[c]+=v@u
            u=op@u
    ks=(ks+ks.conj().transpose(0,2,1))/2
    dks=(dks+dks.conj().transpose(0,1,3,2))/2
    return u,du,ks,dks

def closure(u,du):
    b=u[4:,4:]; db=du[:,4:,4:]; tau=np.trace(b); dtau=np.trace(db,axis1=1,axis2=2)
    if abs(tau)<.25: raise ValueError('hidden chart singular')
    num=np.array([b[1,0]+b[0,1],b[1,0]-b[0,1],b[0,0]-b[1,1]])
    dn=np.array([db[:,1,0]+db[:,0,1],db[:,1,0]-db[:,0,1],db[:,0,0]-db[:,1,1]])
    q=num/tau; dq=dn/tau-num[:,None]*dtau/tau**2
    return np.array([-q[0].imag,q[1].real,-q[2].imag]),np.array([-dq[0].imag,dq[1].real,-dq[2].imag])

def module_constraints(data):
    u,du,ks,dks=data
    c,dc=closure(u,du)
    vals=[c]; jacs=[dc]
    for a in range(2):
        vals.extend([coords(ks[a,:4,:4]),coords(ks[a,4:,4:])[[0,2,3]]])
        jacs.extend([coords(dks[a,:,:4,:4]).T,coords(dks[a,:,4:,4:])[:,[0,2,3]].T])
    return np.concatenate(vals)[ROWS],np.vstack(jacs)[ROWS]

def combined(p,target=GTARGET,jac=True):
    p=np.asarray(p).reshape(3,44)
    ds=[module(x,jac) for x in p]
    rs=[]; j=np.zeros((126,132))
    gs=[]; dgs=[]
    for k,data in enumerate(ds):
        r,d=module_constraints(data); rs.append(r); j[k*37:(k+1)*37,k*44:(k+1)*44]=d
        u,du,_,_=data; lam=np.trace(u[4:,4:])/2; dl=np.trace(du[:,4:,4:],axis1=1,axis2=2)/2
        gs.append(u[:4,:4]/lam); dgs.append(du[:,:4,:4]/lam-u[None,:4,:4]*dl[:,None,None]/lam**2)
    g=gs[0]@gs[1]@gs[2]
    dg=np.concatenate((dgs[0]@gs[1]@gs[2],gs[0]@dgs[1]@gs[2],gs[0]@gs[1]@dgs[2]))
    w=target.conj().T@g; dw=target.conj().T@dg
    h=.5j*(w-w.conj().T); dh=.5j*(dw-dw.conj().transpose(0,2,1))
    t=np.einsum('aij,ji->a',B4,h).real
    j[111:]=np.einsum('aij,kji->ak',B4,dh).real
    return np.r_[np.concatenate(rs),t],j,g,ds

def total(p,jac=True,errors=(0.,0.)):
    ds=[module(x,jac,errors) for x in np.asarray(p).reshape(3,44)]
    us=[d[0] for d in ds]; u=us[0]@us[1]@us[2]
    du=np.concatenate((ds[0][1]@us[1]@us[2],us[0]@ds[1][1]@us[2],us[0]@us[1]@ds[2][1]))
    return u,du

def nominal(p,target=UTARGET,jac=True):
    u,du=total(p,jac); w=target.conj().T@u; dw=target.conj().T@du
    h=.5j*(w-w.conj().T); dh=.5j*(dw-dw.conj().transpose(0,2,1))
    r=np.r_[np.einsum('aij,ji->a',B16,h[:4,:4]).real,np.einsum('aij,ji->a',B2,h[4:,4:]).real]
    j=np.vstack((np.einsum('aij,kji->ak',B16,dh[:,:4,:4]).real,np.einsum('aij,kji->ak',B2,dh[:,4:,4:]).real))
    return r,j,u

def resources(p,eta=10.):
    p=np.asarray(p).reshape(3,44)
    return float(np.abs(p[:,:15]).sum()/eta+np.abs(p[:,29:]).sum())

def time_gradient(p,eta=10.):
    p=np.asarray(p).reshape(3,44); g=np.zeros_like(p)
    g[:,:15]=np.sign(p[:,:15])/eta; g[:,29:]=np.sign(p[:,29:]); return g.ravel()

def half_diamond(u,target=UTARGET):
    phases=np.sort(np.angle(np.linalg.eigvals(target.conj().T@u)))
    arc=2*np.pi-np.max(np.diff(np.r_[phases,phases[0]+2*np.pi]))
    return float(1. if arc>=np.pi else np.sin(max(0.,arc)/2))
