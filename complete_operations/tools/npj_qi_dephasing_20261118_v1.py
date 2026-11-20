"""Full-register finite-time readout under local Markov dephasing for the npj article."""
import argparse
import csv
import json
import math
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import npj_qi_register_depth_matched_20261118_v1 as matched
from npj_qi_register_depth_matched_20261118_v1 import np, pilot
from scipy.integrate import solve_ivp

ROOT=pilot.ROOT
PROTOCOL=ROOT/"docs/NPJ_QI_CONFIRMATION_PROTOCOL_20261118_v1.md"
PROJECTION=ROOT/"runs/npj_qi_register_depth_projection_20261118_v1"
REPORT=ROOT/"runs/npj_qi_register_depth_report_20261118_v1"
PAULI=np.array([np.eye(2),[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]],complex)
ZDIFF=np.array([0.,1.,-1.,0.])
RATE=1e-5
XVAL=np.linspace(-.01,.01,5)


def rotation_map(area,tau,gamma,rtol=2e-11,atol=2e-13):
    """Return computational-basis channel matrices for five distinct drive gains."""
    angles=(1+XVAL)*area
    if gamma==0:
        c=np.cos(angles);s=np.sin(angles)
        yz=np.stack([c,-s,s,c],axis=-1).reshape(-1,2,2)
    else:
        def ode(t,flat):
            blocks=flat.reshape(-1,2,2)
            freq=angles*(1-np.cos(2*np.pi*t))
            d=np.empty_like(blocks)
            d[:,0,:]=-gamma*tau*blocks[:,0,:]-freq[:,None]*blocks[:,1,:]
            d[:,1,:]=freq[:,None]*blocks[:,0,:]
            return d.ravel()
        initial=np.tile(np.eye(2),(len(XVAL),1,1))
        sol=solve_ivp(ode,(0.,1.),initial.ravel(),method="DOP853",rtol=rtol,atol=atol)
        if not sol.success:
            raise ValueError(sol.message)
        yz=sol.y[:,-1].reshape(-1,2,2)
    ptm=np.zeros((len(XVAL),4,4))
    ptm[:,0,0]=1
    ptm[:,1,1]=np.exp(-gamma*tau)
    ptm[:,2:,2:]=yz
    return (np.einsum("aij,sab,blk->sijkl",PAULI,ptm,PAULI)/2).reshape(-1,4,4)


def independent_pulse(area,phase,tau,gamma,x):
    xaxis=np.cos(phase)*PAULI[1]+np.sin(phase)*PAULI[2]
    z=PAULI[3]
    ident=np.eye(2)
    comm=-.5j*(np.kron(xaxis,ident)-np.kron(ident,xaxis.T))
    diss=.5*gamma*tau*(np.kron(z,z.T)-np.eye(4))
    def rhs(t,vec):
        return ((diss+(1+x)*area*(1-math.cos(2*math.pi*t))*comm)@vec.reshape(4,4)).ravel()
    sol=solve_ivp(rhs,(0.,1.),np.eye(4,dtype=complex).ravel(),method="DOP853",rtol=2e-13,atol=2e-15)
    if not sol.success:
        raise ValueError(sol.message)
    return sol.y[:,-1].reshape(4,4)


def local_channels(rho, channel, n):
    count=len(rho)
    tensor=rho.reshape((count,)+(2,)*(2*n))
    for site in range(n):
        positions=(1+site,1+n+site)
        reordered=np.moveaxis(tensor,positions,(1,2))
        product=channel@reordered.reshape(count,4,-1)
        tensor=np.moveaxis(product.reshape(reordered.shape),(1,2),positions)
    return tensor.reshape(count,2**n,2**n)


def state_vector(n):
    code=np.array([float(int(i).bit_count()==2)/math.sqrt(6) for i in range(2**n)],complex)
    singlet=np.zeros(2**n,complex)
    singlet[0b0100],singlet[0b1000]=1/math.sqrt(2),-1/math.sqrt(2)
    return (code+singlet)/math.sqrt(2)


def simulate(p,gamma,n=4):
    theta,alpha,chi=pilot.unpack(p)
    tgt,mag,charges=pilot.tensor_setup(n)
    d=2**n
    initial=state_vector(n)
    goal=initial.copy()
    rho=np.broadcast_to(np.outer(initial,initial.conj()),(len(pilot.TEST),d,d)).copy()
    hamming=np.array([[(i^j).bit_count() for j in range(d)] for i in range(d)])
    dcharge=charges[:,None]-charges[None,:]
    tr=float(pilot.duration(np.pi,10.));tt=float(pilot.duration(1.2,1.));sw=.001
    tw_dephase=np.exp(-gamma*(tt+sw)*hamming)
    switch_dephase=np.exp(-gamma*sw*hamming)
    twist=[np.exp(-1j*(1+pilot.TEST[:,1,None,None])*a*dcharge)*tw_dephase for a in chi]
    maps=np.array([rotation_map(a,tr,gamma) for a in theta])
    first_axis=.37
    z=np.exp(-1j*first_axis*ZDIFF)
    at=maps[0,-1]*z[:,None]*z.conj()[None,:]
    independently=independent_pulse(theta[0],first_axis,tr,gamma,XVAL[-1])
    tightened=rotation_map(theta[0],tr,gamma,rtol=2e-13,atol=2e-15)
    audit={"pulse_liouville_maxdiff":float(np.max(np.abs(at-independently))),
           "pulse_tightening_maxdiff":float(np.max(np.abs(maps[0]-tightened))),
           "trace_maxdiff":0.,"hermiticity_maxdiff":0.,"negative_eigenvalue_magnitude":0.}
    for key in ("pulse_liouville_maxdiff","pulse_tightening_maxdiff"):
        if audit[key]>2e-8:
            raise ValueError(f"{key} {audit[key]}")
    selected=np.repeat(np.arange(5),5)
    states=[]
    for depth,gate_phi in enumerate(pilot.PHASES,1):
        z=np.exp(-1j*gate_phi*mag)
        goal=z*(tgt@(z.conj()*goal))
        for j in range(len(theta)):
            rho*=twist[j]
            rz=np.exp(-1j*(alpha[j]+gate_phi)*ZDIFF)
            channel=maps[j][selected]*rz[None,:,None]*rz.conj()[None,None,:]
            rho=local_channels(rho,channel,n)
            if j!=len(theta)-1:
                rho*=switch_dephase
        if depth in pilot.DEPTHS:
            probabilities=np.einsum("i,sij,j->s",goal.conj(),rho,goal).real
            states.extend({"depth":depth,"x":float(x),"y":float(y),"p_ideal_projector":float(val),
                           "probability_loss":float(1-val)} for (x,y),val in zip(pilot.TEST,probabilities))
            audit["trace_maxdiff"]=max(audit["trace_maxdiff"],float(np.max(np.abs(np.trace(rho,axis1=1,axis2=2)-1))))
            audit["hermiticity_maxdiff"]=max(audit["hermiticity_maxdiff"],
                float(np.max(np.abs(rho-rho.conj().swapaxes(1,2)))))
            audit["negative_eigenvalue_magnitude"]=max(audit["negative_eigenvalue_magnitude"],
                max(0.,-float(np.linalg.eigvalsh(rho).min())))
        rho*=switch_dephase
    if max(audit[k] for k in ("trace_maxdiff","hermiticity_maxdiff","negative_eigenvalue_magnitude"))>2e-9:
        raise ValueError(f"density matrix checks failed {audit}")
    return states,audit


def coherent_reference(p):
    n=4
    target,mag,charges=pilot.tensor_setup(n)
    u=np.array([pilot.tensor_word(n,charges,p,x,y) for x,y in pilot.TEST])
    ideal=state_vector(n);psi=np.broadcast_to(ideal,(len(pilot.TEST),len(ideal))).copy()
    levels={}
    for depth,phi in enumerate(pilot.PHASES,1):
        z=np.exp(-1j*phi*mag)
        ideal=z*(target@(z.conj()*ideal))
        psi=z*np.einsum("sij,sj->si",u,z.conj()*psi)
        if depth in pilot.DEPTHS:
            levels[depth]=np.abs(psi@ideal.conj())**2
    return levels


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True,type=Path)
    out=parser.parse_args().out.resolve()
    out.mkdir(parents=False,exist_ok=False)
    start=time.perf_counter()
    try:
        names={"direct":"nominal","refined":"corrected"}
        params={k:PROJECTION/f"n4_{v}_restored.npy" for k,v in names.items()}
        files=[PROTOCOL,Path(__file__).resolve(),Path(pilot.__file__).resolve(),Path(matched.__file__).resolve(),
               ROOT/"src/manifold_schur_v1.py",ROOT/"src/cnot_tolerance_v1.py",REPORT/"summary.json",
               *[matched.BASE/f"n{n}_summary.json" for n in (4,5)],*params.values()]
        pilot.dump(out/"inputs.json",{"utc":datetime.now(timezone.utc).isoformat(),"numpy":np.__version__,
            "scipy":matched.scipy.__version__,"python":sys.version,"interpreter":sys.executable,
            "precision":"float64/complex128","gamma_t0":[0.,RATE],"hashes":{str(f.relative_to(ROOT)):pilot.sha(f) for f in files}})
        source=matched.read(REPORT/"summary.json")["registers"]
        bracket=[]
        for n in (4,5):
            gt=matched.read(matched.BASE/f"n{n}_summary.json")["fixed_gate_time_t0"]
            for method in ("nominal","direct","corrected","refined"):
                for depth in pilot.DEPTHS:
                    duration=gt*depth+(depth-1)*.001
                    jump=-math.expm1(-n*RATE*duration/2)
                    d=source[str(n)]["grid_max_D"][method]["original"][str(depth)]
                    bracket.append({"N":n,"method":method,"depth":depth,"time_t0":duration,
                                    "p_any_jump":jump,"coherent_grid_D":d,"open_grid_D_lower":max(0.,d-jump),
                                    "open_grid_D_upper":min(1.,d+jump)})
        payload=[];stats={}
        for method,path in params.items():
            p=np.load(path);ref=coherent_reference(p);stats[method]={}
            for g in (0.,RATE):
                rows,audit=simulate(p,g);key=str(g)
                if g==0.:
                    difference=max(abs(r["p_ideal_projector"]-ref[r["depth"]][idx%25]) for idx,r in enumerate(rows))
                    audit["independent_unitary_probability_maxdiff"]=float(difference)
                    if difference>2e-8:raise ValueError(f"unitary witness difference {difference}")
                stats[method][key]={"audit":audit,"max_loss":{m:max(r["probability_loss"] for r in rows if r["depth"]==m) for m in pilot.DEPTHS}}
                payload.extend({"method":method,"gamma_t0":g,**r} for r in rows)
                print(method,g,stats[method][key]["max_loss"],flush=True)
        for row0,row1 in zip(payload[:150]+payload[300:450],payload[150:300]+payload[450:]):
            n=4;d=row0["depth"];method=row0["method"]
            b=next(v for v in bracket if v["N"]==n and v["depth"]==d and v["method"]==method)
            if abs(row1["p_ideal_projector"]-row0["p_ideal_projector"])>b["p_any_jump"]+2e-8:
                raise ValueError("readout violates jump mixture bound")
            if row1["probability_loss"]>b["open_grid_D_upper"]+2e-8:
                raise ValueError("readout exceeds full channel grid bound")
        for file,rows in [("readout.csv",payload),("channel_brackets.csv",bracket)]:
            with (out/file).open("x",newline="") as handle:
                writer=csv.DictWriter(handle,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
        pilot.dump(out/"summary.json",{"status":"fixed_register_fixed_time_open_system_readout",
            "wall_s":time.perf_counter()-start,"gamma_t0":RATE,"readout":stats,"brackets":bracket})
    except BaseException:
        pilot.dump(out/"failure.json",{"traceback":traceback.format_exc(),"wall_s":time.perf_counter()-start})
        raise
    finally:
        pilot.dump(out/"sha256.json",{f.name:pilot.sha(f) for f in sorted(out.iterdir()) if f.is_file()})


if __name__=="__main__":
    main()
