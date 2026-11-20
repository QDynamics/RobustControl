"""Successor enclosure: coefficient centers plus operator-norm error balls.

Entrywise rectangles are retained for local arithmetic. Between pulses, use the
known exact primitive norm ||P_k|| <= ||H||^k/k!, including ||P_0||=1.
This removes artificial exponential wrapping in the nominal unitary product.
"""
import numpy as np
import cnot_interval_v1 as b


def fro_batch(a):
    real=np.maximum(np.abs(a.rl),np.abs(a.rh))
    imag=np.maximum(np.abs(a.il),np.abs(a.ih))
    total=np.zeros(a.shape[:-2])
    for i in range(a.shape[-2]):
        for j in range(a.shape[-1]):
            total=b.up(total+b.up(real[...,i,j]*real[...,i,j]))
            total=b.up(total+b.up(imag[...,i,j]*imag[...,i,j]))
    return b.up(np.sqrt(total))


def taylor_balls(pulse_list,order,jx,jy,zz,progress=None):
    ij=[(a,n-a) for n in range(order+1) for a in range(n+1)]
    lookup={pair:k for k,pair in enumerate(ij)}
    maps={}
    for kind in (0,1):
        for k in range(order+1):
            pairs=[(j,lookup[(a-k,c)] if kind==0 else lookup[(a,c-k)]) for j,(a,c) in enumerate(ij) if (a if kind==0 else c)>=k]
            maps[kind,k]=np.array(pairs).T
    centers=np.zeros((len(ij),8,8),complex);centers[0]=np.eye(8)
    errors=np.zeros(len(ij));trace=[]
    for pulse_no,(kind,area,phi) in enumerate(pulse_list):
        pc=b.pulse_coefficients(kind,area,phi,order,b.CM.point(jx),b.CM.point(jy),b.CM.point(zz))
        pmid=[x.midpoint() for x in pc]
        perr=[float(fro_batch(x-b.CM.point(y))) for x,y in zip(pc,pmid)]
        hnorm=b.posmul(3. if kind else 1.5,abs(area))
        beta=[1.]
        for k in range(1,order+1):beta.append(b.posdiv(b.posmul(beta[-1],hnorm),float(k)))
        oldnorm=fro_batch(b.CM.point(centers))
        new=b.CM.zeros(centers.shape);newerror=np.zeros(len(ij))
        for k in range(order+1):
            dst,src=maps[kind,k]
            if kind:
                contribution=b.CM.point(np.diag(pmid[k]))[None,:,None]*b.CM.point(centers[src])
            else:
                contribution=b.matmul(b.CM.point(pmid[k]),b.CM.point(centers[src]))
            new[dst]=new[dst]+contribution
            # P_true C_true - P_mid C_mid = P_true E_old + E_P C_mid.
            term=b.up(b.up(beta[k]*errors[src])+b.up(perr[k]*oldnorm[src]))
            newerror[dst]=b.up(newerror[dst]+term)
        centers=new.midpoint()
        fresh=fro_batch(new-b.CM.point(centers))
        errors=b.up(newerror+fresh)
        assert np.all(np.isfinite(errors))
        trace.append({'pulse':pulse_no+1,'nominal_radius':float(errors[0]),'largest_coefficient_radius':float(np.max(errors))})
        if progress and (pulse_no+1)%30==0:progress(pulse_no+1)
    return ij,centers,errors,trace
