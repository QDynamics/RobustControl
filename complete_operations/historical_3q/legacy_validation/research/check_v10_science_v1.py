"""Independent physical reconstruction and exact arithmetic checks for new claims."""
from pathlib import Path
from fractions import Fraction
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import numpy as np
from scipy.linalg import expm
import json,math,hashlib,time,sys
HERE=Path(__file__).resolve().parent
OUT=HERE/'runs'/'v10_independent_science_20260926_v1'
I=np.eye(2); X=np.array([[0,1],[1,0]]); Y=np.array([[0,-1j],[1j,0]]); Z=np.diag([1,-1])
def kron(parts):
    v=np.array([[1.]])
    for a in parts: v=np.kron(v,a)
    return v
JX=sum(kron([X if s==q else I for s in range(3)]) for q in range(3))/2
JY=sum(kron([Y if s==q else I for s in range(3)]) for q in range(3))/2
HZZ=sum(kron([Z if s in (q,r) else I for s in range(3)]) for q in range(3) for r in range(q+1,3))
V=np.zeros((8,4)); V[7,0]=1; V[[3,5,6],1]=1/np.sqrt(3); V[[1,2,4],2]=1/np.sqrt(3); V[0,3]=1
Q=np.eye(8)-V@V.T
C=np.eye(4,dtype=complex); C[2:,2:]=X
TARGET=-(np.exp(1j*np.pi/4)*V@C@V.T+Q)

def propagate(p,error=(0.,0.),direction=(1.,1.)):
    u=np.eye(8,dtype=complex); dr=np.zeros((8,8),complex); dt=dr.copy(); d1=dr.copy(); d2=dr.copy()
    # Product M1 M2 M3 executes module 3 first.
    for q in np.asarray(p).reshape(3,44)[::-1]:
        for l in range(15):
            phi=0 if l==0 else q[14+l]; axis=np.cos(phi)*JX+np.sin(phi)*JY
            for kind,a,h in [(1,q[29+l],HZZ),(0,q[l],axis)]:
                b=-1j*a*h; e=expm((1+error[kind])*b); db=b@e
                dr=e@dr+(db@u if kind==0 else 0); dt=e@dt+(db@u if kind==1 else 0)
                v=direction[kind]; e1=v*db; e2=.5*v*v*b@db
                d2=e@d2+e1@d1+e2@u; d1=e@d1+e1@u; u=e@u
    return u,1j*u.conj().T@dr,1j*u.conj().T@dt,u.conj().T@d1,u.conj().T@d2

def main():
    OUT.mkdir(parents=True,exist_ok=False); start=time.perf_counter()
    exact=[]
    for s in range(1,1001):
        j=Fraction(2*s-1,2)
        lhs=Fraction(2*s,4)-Fraction(2*s)*j*(j+1)/3
        exact.append(lhs==-4*math.comb(s+1,3))
    magic=np.column_stack(([1,0,0,1],1j*np.array([1,0,0,-1]),1j*np.array([0,1,1,0]),[0,1,-1,0]))/np.sqrt(2)
    cb=magic.conj().T@(np.exp(1j*np.pi/4)*C)@magic
    cm=cb.T@cb; ev=np.linalg.eigvals(cm)
    swap=np.eye(4,dtype=complex); swap[1:3,1:3]=1j*X; iw=1j*swap
    singlet=np.array([0,1,-1,0])/np.sqrt(2)
    iw_b=magic.conj().T@iw@magic
    b1={'odd_determinant_identity_exact_1000_checks':all(exact),'CNOT_invariant_spectrum':[[float(z.real),float(z.imag)] for z in ev],'CNOT_invariant_square_plus_I':float(np.linalg.norm(cm@cm+np.eye(4))),'CNOT_invariant_trace_abs':float(abs(np.trace(cm))),'CNOT_invariant_distance_from_eigenvalue_one':float(min(abs(ev-1))),'i_iSWAP_singlet_error':float(np.linalg.norm(iw@singlet-singlet)),'i_iSWAP_triplet_det_error':float(abs(np.linalg.det(iw_b[:3,:3])-1)),'encoded_CNOT_full_target_unitarity':float(np.linalg.norm(TARGET.conj().T@TARGET-np.eye(8))),'encoded_CNOT_visible_det_error':float(abs(np.linalg.det(np.exp(1j*np.pi/4)*C)-1)),'analytic_proof_file':'B1_STANDARD_GATES_20260926_v1.md'}
    files={'first_cnot':HERE/'runs/cnot_continuation_20260926_v1/final.npz','robust_time':HERE/'runs/cnot_optimization_20260926_v2/robust_time.npz','nominal_time':HERE/'runs/cnot_baselines_20260926_v1/nominal_time.npz','noise_aware':HERE/'runs/cnot_baselines_20260926_v1/noise_aware.npz'}
    results={}
    for label,path in files.items():
        p=np.load(path)['params']; u,kr,kt,i1,i2=propagate(p)
        h=2e-4; plus=propagate(p,(h,h))[0]; minus=propagate(p,(-h,-h))[0]
        second_fd=u.conj().T@(plus+minus-2*u)/(2*h*h)
        result={'target_fro':float(np.linalg.norm(u-TARGET)),'KR_fro':float(np.linalg.norm(kr)),'KT_fro':float(np.linalg.norm(kt)),'unitarity_fro':float(np.linalg.norm(u.conj().T@u-np.eye(8))),'symmetric_hidden_leakage_fro':float(np.linalg.norm(Q@u@V)),'second_coefficient_central_difference_error':float(np.linalg.norm(second_fd-i2)),'second_coefficient_norm':float(np.linalg.norm(i2)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        results[label]=result
    passed=all(exact) and b1['CNOT_invariant_square_plus_I']<1e-12 and b1['i_iSWAP_singlet_error']<1e-12 and all(v['target_fro']<1e-9 for v in results.values()) and all(max(results[k]['KR_fro'],results[k]['KT_fro'])<1e-9 for k in ['first_cnot','robust_time'])
    output={'status':'INDEPENDENT_NUMERICAL_CHECK_PASS' if passed else 'FAIL','B1':b1,'controls':results,'elapsed_s':time.perf_counter()-start,'python':sys.version,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'interval_certified':False}
    text=json.dumps(output,indent=2)
    with (OUT/'result.json').open('x',encoding='utf-8') as f: f.write(text)
    print(text)

if __name__=='__main__': main()
