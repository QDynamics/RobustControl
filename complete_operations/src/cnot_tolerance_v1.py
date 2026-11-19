"""Full-register encoded-CNOT propagation and finite-error bounds (v1).

The physical propagation below is implemented from tensor rotations and diagonal
ZZ pulses, independently of the archived six-dimensional search code.
"""
import math
import numpy as np
from scipy.linalg import expm

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.diag([1., -1.]).astype(complex)
I8 = np.eye(8, dtype=complex)


def tensor3(a, b, c):
    return np.kron(np.kron(a, b), c)


JX = sum(tensor3(*(X if j == k else I2 for j in range(3))) for k in range(3)) / 2
JY = sum(tensor3(*(Y if j == k else I2 for j in range(3))) for k in range(3)) / 2
ZZ = sum(tensor3(*(Z if j in pair else I2 for j in range(3))) for pair in [(0, 1), (0, 2), (1, 2)])
ZZ_DIAG = np.diag(ZZ).real
D = np.zeros((8, 4), complex)
D[7, 0] = D[0, 3] = 1
D[[3, 5, 6], 1] = 1 / np.sqrt(3)
D[[1, 2, 4], 2] = 1 / np.sqrt(3)
P = D @ D.conj().T
CNOT = np.eye(4, dtype=complex)
CNOT[2:, 2:] = X
TARGET = -(np.exp(1j * np.pi / 4) * D @ CNOT @ D.conj().T + I8 - P)
AREA_INDICES = np.array([44*j + k for j in range(3) for k in list(range(15)) + list(range(29, 44))])
AREA_WEIGHTS = np.tile(np.r_[np.full(15, 1.5), np.full(15, 3.)], 3)
LIMITS = np.tile(np.r_[np.full(15, 4*np.pi), np.full(14, 8*np.pi), np.full(15, 4*np.pi)], 3)


def pulses(params):
    p = np.asarray(params).reshape(3, 44)
    out = []
    for j in (2, 1, 0):
        for k in range(15):
            phi = 0. if k == 0 else float(p[j, 14+k])
            out.extend([(1, float(p[j, 29+k]), 0.), (0, float(p[j, k]), phi)])
    return out


def rotation(theta, phi):
    r = np.cos(theta/2) * I2 - 1j * np.sin(theta/2) * (np.cos(phi)*X + np.sin(phi)*Y)
    return tensor3(r, r, r)


def prepare(params):
    out = []
    for kind, a, phi in pulses(params):
        h = a * (ZZ if kind else np.cos(phi)*JX + np.sin(phi)*JY)
        op = np.diag(np.exp(-1j*a*ZZ_DIAG)) if kind else rotation(a, phi)
        out.append((kind, a, phi, h, op))
    return out


def propagate(seq, error=(0., 0.), direct=False, responses=False):
    u = I8.copy()
    ks = np.zeros((2, 8, 8), complex)
    for kind, a, phi, h, op0 in seq:
        if responses:
            ks[kind] += u.conj().T @ h @ u
        if direct:
            op = expm(-1j * (1+error[kind]) * h)
        elif error[kind] == 0:
            op = op0
        elif kind:
            op = np.diag(np.exp(-1j*(1+error[kind])*a*ZZ_DIAG))
        else:
            op = rotation((1+error[kind])*a, phi)
        u = op @ u
    return (u, ks) if responses else u


def halfdiamond(u):
    phases = np.sort(np.angle(np.linalg.eigvals(TARGET.conj().T @ u)))
    width = max(0., 2*np.pi - float(np.max(np.diff(np.r_[phases, phases[0]+2*np.pi]))))
    return 1. if width >= np.pi else float(np.sin(width/2))


def indices(order):
    return [(a, n-a) for n in range(order+1) for a in range(n+1)]


def taylor_float(seq, order=8):
    ij = indices(order)
    lookup = {pair: k for k, pair in enumerate(ij)}
    c = np.zeros((len(ij), 8, 8), complex)
    c[0] = I8
    for kind, _, _, h, op in seq:
        pc = [op]
        for k in range(1, order+1):
            pc.append((-1j*h) @ pc[-1] / k)
        new = np.zeros_like(c)
        for k in range(order+1):
            pairs = [(j, lookup[(a-k, b)] if kind == 0 else lookup[(a, b-k)])
                     for j, (a, b) in enumerate(ij) if (a if kind == 0 else b) >= k]
            dst, src = np.array(pairs).T
            new[dst] += pc[k] @ c[src]
        c = new
    return ij, c


def duration(areas, caps):
    s = np.abs(np.asarray(areas)) / np.asarray(caps)
    b = 2*np.pi*.02
    lo, hi = 2*s, 2*s + np.sqrt(b*s) + .04
    for _ in range(60):
        tau = (lo+hi)/2
        ok = tau >= s*(1+np.sqrt(1+(b/tau)**2))
        hi, lo = np.where(ok, tau, hi), np.where(ok, lo, tau)
    return np.maximum((lo+hi)/2, .04)


def gate_time(params):
    return float(duration(np.asarray(params).ravel()[AREA_INDICES], np.tile(np.r_[np.full(15, 10.), np.ones(15)], 3)).sum() + .089)


def action(params):
    p = np.asarray(params).reshape(3, 44)
    return [float(1.5*np.abs(p[:, :15]).sum()), float(3*np.abs(p[:, 29:]).sum())]


def reference_bound(coeff, order, r, action_sum):
    return sum(float(np.linalg.norm(coeff[j], 2))*r**(a+b) for j, (a, b) in enumerate(indices(8)) if a+b <= order) + (action_sum*r)**(order+1)/math.factorial(order+1)
