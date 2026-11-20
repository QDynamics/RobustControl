"""Prospective paired compiler confirmation for the npj article; no overwriting."""
import argparse
import csv
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import npj_qi_register_depth_projection_20261118_v1 as projection
from npj_qi_register_depth_projection_20261118_v1 import matched, np, pilot

PROTOCOL = pilot.ROOT / "docs/NPJ_QI_CONFIRMATION_PROTOCOL_20261118_v1.md"
SEED_BASES = (6111900, 6112000, 6112100, 6112200)
DENSE = np.array(list(product(np.linspace(-.01, .01, 9), repeat=2)))


def endpoint(blocks, p):
    gates = [b.word(p, pilot.JET[:1])[0] for b in blocks]
    return pilot.metrics([b.target for b in blocks], gates, [b.mult for b in blocks])[0]


def depth32(blocks, p, offsets):
    gates = [b.word(p, offsets) for b in blocks]
    ideals = [np.eye(b.d,dtype=complex) for b in blocks]
    actual = [np.broadcast_to(v,(len(offsets),*v.shape)).copy() for v in ideals]
    for phi in pilot.PHASES:
        for j, b in enumerate(blocks):
            z = np.exp(-1j*phi*b.difference)
            ideals[j] = (z*b.target) @ ideals[j]
            actual[j] = (z*gates[j]) @ actual[j]
    return np.array([pilot.metrics(ideals,[g[i] for g in actual],[b.mult for b in blocks])[0]
                     for i in range(len(offsets))])


def pair(n, seed, folder, writer, dense_writer):
    folder.mkdir(exist_ok=False)
    blocks=[pilot.Block(r,n) for r in pilot.schur_representations(n)]
    length=pilot.LENGTHS[n]
    rng=np.random.default_rng(seed)
    initial=np.concatenate([rng.uniform(-1.6,1.6,length),rng.uniform(-np.pi,np.pi,length),
                            rng.uniform(-.45,.45,length)])
    p={"initial":initial}
    fit={}
    p["nominal"],fit["nominal"]=pilot.fit(blocks,n,initial,"nominal",100,150)
    p["raw"],fit["raw"]=pilot.fit(blocks,n,p["nominal"],"corrected",80,150)
    used=fit["raw"]["evaluations"]
    for arm,start,cap in [("direct","nominal",128),("seeded","raw",128-used)]:
        p[arm+"_finite"],fit[arm]=pilot.fit(blocks,n,p[start],"noise_aware",cap,150)
        p[arm+"_restored"],fit[arm+"_restore"]=projection.restore(n,blocks,p[arm+"_finite"])
    np.savez(folder/"controls.npz",**p)
    pilot.dump(folder/"fit_histories.json",fit)
    candidates=("nominal","raw","direct_restored","seeded_restored")
    d0={name:endpoint(blocks,p[name]) for name in candidates}
    train={name:pilot.training_score(blocks,p[name]) for name in candidates}
    rep=pilot.full_tensor_representation(n)
    raw_deriv=pilot.propagate(rep,np.array(pilot.unpack(p["raw"])).T)
    normen=[float(np.linalg.norm(g,2)) for g in (raw_deriv.generator_R,raw_deriv.generator_T)]
    raw_pass=d0["raw"]<=1e-6 and max(normen)<=1e-5
    validation={}
    for arm, options in [("direct",("nominal","direct_restored")),
                         ("seeded",("nominal","raw","seeded_restored"))]:
        eligible=[k for k in options if d0[k]<=1e-6 and (k!="raw" or raw_pass)]
        if not eligible:
            validation[arm]={"qualified":False}
            continue
        best=min(eligible,key=train.get)
        controls=p[best]
        target,mag,charges=pilot.tensor_setup(n)
        diag,_=pilot.diagnostics(n,blocks,controls,target,mag,charges,rep)
        if not diag["nominal_qualified"]:
            raise ValueError("tensor and Schur qualification disagree")
        all_scores={}
        for circ,phases in matched.sequences().items():
            rows=matched.held_out(blocks,controls,phases)
            all_scores[circ]={m:max(x["D"] for x in rows if x["depth"]==m) for m in pilot.DEPTHS}
            writer.writerows({"N":n,"seed":seed,"arm":arm,"circuit":circ,**v} for v in rows)
        d9=depth32(blocks,controls,DENSE)
        dense_writer.writerows([n,seed,arm,float(x),float(y),float(v)] for (x,y),v in zip(DENSE,d9))
        validation[arm]={"qualified":True,"selected":best,"numerics":diag,"depth_scores":all_scores,
                         "dense_D32":float(d9.max()),"dense_argmax_xy":DENSE[d9.argmax()].tolist()}
    result={"N":n,"seed":seed,"nominal_distances":d0,"training_distances":train,
            "raw_generator_norms":normen,"raw_pass":raw_pass,"validation":validation,
            "actual_calls":{k:v["evaluations"] for k,v in fit.items()},
            "solver_s":{k:v["wall_s"] for k,v in fit.items()}}
    pilot.dump(folder/"summary.json",result)
    print(n,seed,[(arm, v["depth_scores"]["original"][32] if v["qualified"] else None)
                  for arm,v in validation.items()],flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True,type=Path)
    out=parser.parse_args().out.resolve()
    out.mkdir(exist_ok=False,parents=False)
    start=time.perf_counter()
    try:
        files=[PROTOCOL,Path(__file__).resolve(),Path(pilot.__file__).resolve(),
               Path(matched.__file__).resolve(),Path(projection.__file__).resolve(),
               pilot.ROOT/"src/manifold_schur_v1.py",pilot.ROOT/"src/cnot_tolerance_v1.py"]
        pilot.dump(out/"inputs.json",{"utc":datetime.now(timezone.utc).isoformat(),"python":sys.version,
             "interpreter":sys.executable,"numpy":np.__version__,"scipy":matched.scipy.__version__,
             "precision":"float64/complex128","seed_bases":SEED_BASES,"threads":1,
             "hashes":{str(f.relative_to(pilot.ROOT)):pilot.sha(f) for f in files}})
        results=[]
        with (out/"validation_25.csv").open("x",newline="") as a,(out/"validation_81.csv").open("x",newline="") as b:
            writer=csv.DictWriter(a,fieldnames=["N","seed","arm","circuit","depth","x","y","D","avg_infidelity"])
            writer.writeheader()
            dense_writer=csv.writer(b); dense_writer.writerow(["N","seed","arm","x","y","D32"])
            for n in (4,5):
                for base in SEED_BASES:
                    results.append(pair(n,base+n,out/f"n{n}_seed{base+n}",writer,dense_writer))
                    a.flush();b.flush()
        pilot.dump(out/"summary.json",{"status":"all_registered_seeds_evaluated","wall_s":time.perf_counter()-start,
                                      "pairs":results})
    except BaseException:
        pilot.dump(out/"failure.json",{"traceback":traceback.format_exc(),"wall_s":time.perf_counter()-start})
        raise
    finally:
        pilot.dump(out/"sha256.json",{x.relative_to(out).as_posix():pilot.sha(x) for x in sorted(out.rglob("*")) if x.is_file()})


if __name__=="__main__":
    main()
