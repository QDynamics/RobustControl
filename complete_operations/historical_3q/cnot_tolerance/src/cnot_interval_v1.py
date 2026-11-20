"""Outward-rounded complex rectangles for a bounded CNOT Taylor certificate.

Certificate arithmetic uses individual binary64 add/multiply/divide/sqrt ufuncs,
with nextafter outward after each operation. Matrix products are explicit sums;
BLAS/SVD is used only to propose norm factorizations whose residuals are enclosed.
Elementary trigonometry uses directed Decimal Taylor intervals, not libm.
"""
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
from functools import lru_cache
import math
import numpy as np

NEG, POS = -np.inf, np.inf


def down(x):
    return np.nextafter(x, NEG)


def up(x):
    return np.nextafter(x, POS)


def posadd(a, b):
    return float(up(float(a)+float(b)))


def posmul(a, b):
    return float(up(float(a)*float(b)))


def posdiv(a, b):
    assert b > 0
    return float(up(float(a)/float(b)))


def possum(values):
    out = 0.
    for x in values:
        assert x >= 0
        out = posadd(out, x)
    return out


def pospow(x, n):
    out = 1.
    for _ in range(n):
        out = posmul(out, x)
    return out


def radd(a, b):
    return down(a[0]+b[0]), up(a[1]+b[1])


def rneg(a):
    return -a[1], -a[0]


def rmul(a, b):
    values = [a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1]]
    return down(np.minimum.reduce(values)), up(np.maximum.reduce(values))


class CM:
    def __init__(self, rl, rh, il, ih):
        self.rl, self.rh, self.il, self.ih = [np.asarray(x, dtype=float) for x in (rl,rh,il,ih)]

    @staticmethod
    def point(z):
        a = np.asarray(z, dtype=complex)
        return CM(a.real.copy(),a.real.copy(),a.imag.copy(),a.imag.copy())

    @staticmethod
    def zeros(shape):
        return CM.point(np.zeros(shape,complex))

    @property
    def shape(self):
        return self.rl.shape

    def copy(self):
        return CM(*(x.copy() for x in (self.rl,self.rh,self.il,self.ih)))

    def __getitem__(self, key):
        return CM(*(x[key] for x in (self.rl,self.rh,self.il,self.ih)))

    def __setitem__(self, key, value):
        for x,y in zip((self.rl,self.rh,self.il,self.ih),(value.rl,value.rh,value.il,value.ih)):
            x[key]=y

    def __add__(self, other):
        if not isinstance(other,CM): other=CM.point(other)
        return CM(*radd((self.rl,self.rh),(other.rl,other.rh)),*radd((self.il,self.ih),(other.il,other.ih)))

    def __neg__(self):
        return CM(-self.rh,-self.rl,-self.ih,-self.il)

    def __sub__(self, other):
        if not isinstance(other,CM): other=CM.point(other)
        return self+(-other)

    def __mul__(self, other):
        if not isinstance(other,CM): other=CM.point(other)
        ar,ai=(self.rl,self.rh),(self.il,self.ih)
        br,bi=(other.rl,other.rh),(other.il,other.ih)
        real=radd(rmul(ar,br),rneg(rmul(ai,bi)))
        imag=radd(rmul(ar,bi),rmul(ai,br))
        return CM(*real,*imag)

    def div(self, positive):
        assert positive > 0
        return CM(down(self.rl/positive),up(self.rh/positive),down(self.il/positive),up(self.ih/positive))

    def minus_i(self):
        return CM(self.il,self.ih,-self.rh,-self.rl)

    def adjoint(self):
        swap=lambda a:np.swapaxes(a,-1,-2)
        return CM(swap(self.rl),swap(self.rh),-swap(self.ih),-swap(self.il))

    def reshape(self, shape):
        return CM(*(x.reshape(shape) for x in (self.rl,self.rh,self.il,self.ih)))

    def midpoint(self):
        return (self.rl/2+self.rh/2)+1j*(self.il/2+self.ih/2)

    def max_width(self):
        return float(max(np.max(self.rh-self.rl),np.max(self.ih-self.il)))


def matmul(a,b):
    assert a.shape[-1]==b.shape[-2]
    shape=np.broadcast_shapes(a.shape[:-2],b.shape[:-2])+(a.shape[-2],b.shape[-1])
    out=CM.zeros(shape)
    for k in range(a.shape[-1]):
        out=out+a[..., :,k:k+1]*b[...,k:k+1,:]
    return out


def kron(a,b):
    outer=a[:,None,:,None]*b[None,:,None,:]
    return outer.reshape((a.shape[0]*b.shape[0],a.shape[1]*b.shape[1]))


def diag(a):
    n=a.shape[0]
    out=CM.zeros((n,n))
    idx=np.arange(n)
    out[idx,idx]=a
    return out


def fro_upper(a):
    r=np.maximum(np.abs(a.rl),np.abs(a.rh)).ravel()
    i=np.maximum(np.abs(a.il),np.abs(a.ih)).ravel()
    total=possum([posmul(x,x) for x in np.r_[r,i]])
    return float(up(np.sqrt(total)))


def spectral_upper(a):
    u,s,vh=np.linalg.svd(a.midpoint())
    ui,vi=CM.point(u),CM.point(vh)
    gram_u=fro_upper(matmul(ui.adjoint(),ui)-CM.point(np.eye(u.shape[1])))
    gram_v=fro_upper(matmul(vi,vi.adjoint())-CM.point(np.eye(vh.shape[0])))
    residual=fro_upper(a-matmul(ui*CM.point(s[None,:]),vi))
    un=float(up(np.sqrt(posadd(1.,gram_u))))
    vn=float(up(np.sqrt(posadd(1.,gram_v))))
    bound=posadd(posmul(posmul(un,float(np.max(np.abs(s)))),vn),residual)
    return bound, {'proposal_sigma_max':float(np.max(s)), 'factorization_residual_fro_upper':residual,
                   'left_Gram_residual_fro_upper':gram_u,'right_Gram_residual_fro_upper':gram_v}


def dadd(a,b):
    with localcontext() as ctx:
        ctx.prec=90; ctx.rounding=ROUND_FLOOR; lo=a[0]+b[0]
        ctx.rounding=ROUND_CEILING; hi=a[1]+b[1]
    return lo,hi


def dmul(a,b):
    with localcontext() as ctx:
        ctx.prec=90; ctx.rounding=ROUND_FLOOR
        lo=min(x*y for x in a for y in b)
        ctx.rounding=ROUND_CEILING
        hi=max(x*y for x in a for y in b)
    return lo,hi


def ddiv(a, positive):
    with localcontext() as ctx:
        ctx.prec=90; ctx.rounding=ROUND_FLOOR; lo=a[0]/positive
        ctx.rounding=ROUND_CEILING; hi=a[1]/positive
    return lo,hi


@lru_cache(maxsize=8192)
def trig_interval(value, cosine=False):
    """The input float is an exact binary rational, not an uncertain measurement."""
    x=Decimal.from_float(float(value))
    assert abs(x)<128
    xx=dmul((x,x),(x,x))
    negative=(xx[1].copy_negate(),xx[0].copy_negate())
    term=(Decimal(1),Decimal(1)) if cosine else (x,x)
    result=term
    degree=0 if cosine else 1
    # A real Taylor remainder is <= |x|^(degree+1)/(degree+1)! since
    # all real derivatives of sin/cos have absolute value <= 1.
    for k in range(1,513):
        term=ddiv(dmul(term,negative),Decimal((degree+1)*(degree+2)))
        degree+=2
        result=dadd(result,term)
        with localcontext() as ctx:
            ctx.prec=90;ctx.rounding=ROUND_CEILING
            remainder=(abs(x)**(degree+1))/Decimal(math.factorial(degree+1))
        if degree>abs(x)+10 and remainder<Decimal('1e-75'):
            break
    else:
        raise RuntimeError('Trigonometric enclosure did not converge')
    result=dadd(result,(remainder.copy_negate(),remainder))
    lo=float(down(float(result[0])));hi=float(up(float(result[1])))
    return lo,hi


def cosi(x):
    lo,hi=trig_interval(float(x),True)
    return CM(lo,hi,0.,0.)


def sini(x):
    lo,hi=trig_interval(float(x),False)
    return CM(lo,hi,0.,0.)


def sqrt_reciprocal(n):
    with localcontext() as ctx:
        ctx.prec=90
        # Decimal.sqrt is correctly rounded to nearest, independent of the
        # context mode. Adjacent Decimal values provide directed bounds.
        root=Decimal(n).sqrt()
        lo,hi=root.next_minus(),root.next_plus()
        ctx.rounding=ROUND_FLOOR; a=Decimal(1)/hi
        ctx.rounding=ROUND_CEILING; b=Decimal(1)/lo
    return CM(float(down(float(a))),float(up(float(b))),0.,0.)


def exact_target():
    d=CM.zeros((8,4));d[7,0]=CM.point(1);d[0,3]=CM.point(1)
    root3=sqrt_reciprocal(3)
    for i in (3,5,6):d[i,1]=root3
    for i in (1,2,4):d[i,2]=root3
    c=np.eye(4);c[2:,2:]=[[0,1],[1,0]]
    phase=sqrt_reciprocal(2)*CM.point(1+1j)
    return -(phase*matmul(matmul(d,CM.point(c)),d.adjoint())+CM.point(np.eye(8))-matmul(d,d.adjoint()))


def pulse_coefficients(kind,area,phi,order,jx,jy,zz):
    if kind:
        # area*3 must also be interval-enclosed; use the identity exp(-i 3a)
        # as (exp(-i a))^3 to avoid treating a rounded product as exact.
        e=cosi(area)+sini(area).minus_i()
        inv=cosi(area)-sini(area).minus_i()
        three=e*e*e
        entries=CM.zeros((8,))
        for k in range(8):entries[k]=three if k in (0,7) else inv
        op=diag(entries);h=CM.point(area)*zz
    else:
        # Binary division by two is exact for the normal input angles here.
        axis=cosi(phi)*jx+sini(phi)*jy
        h=CM.point(area)*axis
        x=CM.point([[0,1],[1,0]])
        y=CM.point([[0,-1j],[1j,0]])
        one=cosi(area/2)*CM.point(np.eye(2))+ (sini(area/2)*(cosi(phi)*x+sini(phi)*y)).minus_i()
        op=kron(kron(one,one),one)
    coeff=[op]
    for k in range(1,order+1):coeff.append(matmul(h.minus_i(),coeff[-1]).div(k))
    return coeff


def taylor_interval(pulse_list,order,jx,jy,zz,progress=None):
    ij=[(a,n-a) for n in range(order+1) for a in range(n+1)]
    lookup={pair:k for k,pair in enumerate(ij)}
    maps={}
    for kind in (0,1):
        for k in range(order+1):
            pairs=[(j,lookup[(a-k,b)] if kind==0 else lookup[(a,b-k)]) for j,(a,b) in enumerate(ij) if (a if kind==0 else b)>=k]
            maps[kind,k]=np.array(pairs).T
    c=CM.zeros((len(ij),8,8));c[0]=CM.point(np.eye(8))
    for pulse_no,(kind,area,phi) in enumerate(pulse_list):
        pc=pulse_coefficients(kind,area,phi,order,CM.point(jx),CM.point(jy),CM.point(zz))
        new=CM.zeros(c.shape)
        for k in range(order+1):
            dst,src=maps[kind,k]
            if kind:
                diagonal=pc[k][np.arange(8),np.arange(8)]
                contribution=diagonal[None,:,None]*c[src]
            else:
                contribution=matmul(pc[k],c[src])
            new[dst]=new[dst]+contribution
        c=new
        if progress and (pulse_no+1)%30==0:progress(pulse_no+1)
    return ij,c


def bound_upper(degree_bounds,action_sum,r,order):
    # Inflate registered decimal radii as well as intermediate products.
    r=float(up(r))
    terms=[posmul(degree_bounds[k],pospow(r,k)) for k in range(order+1)]
    rem=posdiv(pospow(posmul(action_sum,r),order+1),float(math.factorial(order+1)))
    return posadd(possum(terms),rem)
