"""Whole-word CNOT response constraints; inherited primitive derivatives are explicit."""
from __future__ import annotations
import numpy as np
import cnot_model_v1 as m

CHARGED = np.array([44*k+i for k in range(3) for i in list(range(15))+list(range(29,44))])
WEIGHTS = np.tile(np.r_[np.ones(15)/10, np.ones(15)], 3)
LIMITS = np.tile(np.r_[np.full(15,4*np.pi),np.full(14,8*np.pi),np.full(15,4*np.pi)],3)

def product_responses(p, jac=True):
    ds = [m.module(q,jac) for q in np.asarray(p).reshape(3,44)]
    u = np.eye(6,dtype=complex)
    du = np.zeros((132,6,6),complex)
    ks = np.zeros((2,6,6),complex)
    dks = np.zeros((2,132,6,6),complex)
    for k in (2,1,0):
        v,dv,kv,dkv = ds[k]
        sl = slice(44*k,44*(k+1))
        adj = u.conj().T
        for a in range(2):
            ks[a] += adj @ kv[a] @ u
            if jac:
                dks[a] += du.conj().transpose(0,2,1)@kv[a]@u + adj@kv[a]@du
                dks[a,sl] += adj@dkv[a]@u
        if jac:
            du = v @ du
            du[sl] += dv @ u
        u = v @ u
    return u,du,ks,dks,ds

def project(basis, a):
    return np.einsum('aij,...ji->...a',basis,a).real

def global_constraints(p, jac=True):
    p=np.asarray(p).ravel()
    u,du,ks,dks,ds=product_responses(p,jac)
    w=m.UTARGET.conj().T@u
    dw=m.UTARGET.conj().T@du
    h=.5j*(w-w.conj().T)
    dh=.5j*(dw-dw.conj().transpose(0,2,1))
    area=float(p.reshape(3,44)[:,29:].sum())
    da=np.zeros(132); da.reshape(3,44)[:,29:]=1
    r=np.r_[area,project(m.B4,h[:4,:4]),project(m.B2,h[4:,4:]),
            project(m.B4,ks[0,:4,:4]),project(m.B2,ks[0,4:,4:]),
            project(m.B4,ks[1,:4,:4])]
    j=np.vstack((da,project(m.B4,dh[:,:4,:4]).T,project(m.B2,dh[:,4:,4:]).T,
                 project(m.B4,dks[0,:,:4,:4]).T,project(m.B2,dks[0,:,4:,4:]).T,
                 project(m.B4,dks[1,:,:4,:4]).T))
    assert r.shape==(52,) and j.shape==(52,132)
    return r,j,u,ks,ds

def evaluation(p,kind,jac=True):
    if kind=='whole_gate':
        r,j,u,ks,ds=global_constraints(p,jac)
    elif kind=='modular':
        r,j,_,ds=m.combined(p,jac=jac)
        u=ds[0][0]@ds[1][0]@ds[2][0]
        ks=None
    else:
        raise ValueError(kind)
    return r,j,u,ks,ds

def audit(p,kind):
    r,j,u,ks,ds=global_constraints(p)
    sv=np.linalg.svd(j,compute_uv=False)
    try:
        modular_max=float(np.max(np.abs(m.combined(p)[0])))
    except ValueError:
        modular_max=None
    return {'target_block_fro':float(np.linalg.norm(u-m.UTARGET)),
            'shared_response_block_fro':[float(np.linalg.norm(k)) for k in ks],
            'largest_module_response_block_fro':float(max(np.linalg.norm(d[2][a]) for d in ds for a in range(2))),
            'modular_residual_max':modular_max,
            'whole_gate_residual_max':float(np.max(np.abs(r))),
            'whole_gate_singular_values':sv.tolist(),'whole_gate_rank':int(np.sum(sv>1e-8)),
            'time_t0':m.resources(p),'max_bound_excess':float(np.max(np.abs(p)-LIMITS))}
