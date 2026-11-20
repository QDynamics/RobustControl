"""Independent exact-rational checks of the interval arithmetic used by the run."""
from pathlib import Path
from fractions import Fraction as F
import sys
import math
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import cnot_interval_v1 as b


def run():
    checks={}
    trig_tests=0
    for value in [-25.5,-3.5,-.001,0.,.1,3.5,25.]:
        x=F.from_float(value)
        for cosine in (False,True):
            term=F(1) if cosine else x
            total=term; degree=0 if cosine else 1
            for _ in range(180):
                term *= -x*x/F((degree+1)*(degree+2))
                degree+=2;total+=term
            remainder=abs(x)**(degree+1)/math.factorial(degree+1)
            lo,hi=b.trig_interval(value,cosine)
            assert F.from_float(lo)<=total-remainder<=total+remainder<=F.from_float(hi),(value,cosine)
            trig_tests+=1
    checks['trig_exact_rational_Taylor_comparisons']=trig_tests
    for n in (2,3):
        a=b.sqrt_reciprocal(n)
        assert F.from_float(float(a.rl))**2<=F(1,n)<=F.from_float(float(a.rh))**2
    checks['radicals_exact_rational_squares']=2
    rng=np.random.default_rng(26101220)
    for _ in range(10):
        a=rng.normal(size=(3,3))+1j*rng.normal(size=(3,3))
        c=rng.normal(size=(3,3))+1j*rng.normal(size=(3,3))
        z=b.matmul(b.CM.point(a),b.CM.point(c))
        for i in range(3):
            for j in range(3):
                real=sum(F.from_float(float(a[i,k].real))*F.from_float(float(c[k,j].real))-F.from_float(float(a[i,k].imag))*F.from_float(float(c[k,j].imag)) for k in range(3))
                imag=sum(F.from_float(float(a[i,k].real))*F.from_float(float(c[k,j].imag))+F.from_float(float(a[i,k].imag))*F.from_float(float(c[k,j].real)) for k in range(3))
                assert F.from_float(float(z.rl[i,j]))<=real<=F.from_float(float(z.rh[i,j]))
                assert F.from_float(float(z.il[i,j]))<=imag<=F.from_float(float(z.ih[i,j]))
        n,details=b.spectral_upper(b.CM.point(a))
        assert n>=np.linalg.norm(a,2)
    checks['complex_matrix_products_exact_rational_entries']=90
    checks['verified_SVD_factorization_proposals']=10
    a=b.CM.point(np.diag([3.,4.,0.]))
    n,_=b.spectral_upper(a)
    assert 4<=n<4.000000000001
    checks['known_spectral_norm']=True
    return checks


if __name__=='__main__':
    print(run())
