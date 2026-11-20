"""R2-1+R2-2: high-precision endpoint runtime using precision context.
Closed-form R, T, analytic Jacobian.  No float/complex in scientific path.
"""
import mpmath as mp
from runtime.high_precision_context import ctx as get_ctx
I_c = mp.mpc(0, 1)

sx2 = mp.matrix([[0, 1], [1, 0]])
sy2 = mp.matrix([[0, -I_c], [I_c, 0]])
s0_2 = mp.eye(2)
ZW = [3, -1, -1, -1, -1, -1, -1, 3]

def r_phi(theta, phi):
    c = mp.cos(theta / 2); s = mp.sin(theta / 2)
    return c * s0_2 - I_c * s * (mp.cos(phi) * sx2 + mp.sin(phi) * sy2)

def dr_dtheta(theta, phi):
    c = mp.cos(theta / 2); s = mp.sin(theta / 2)
    return mp.mpf('-0.5') * s * s0_2 - I_c * mp.mpf('0.5') * c * (mp.cos(phi) * sx2 + mp.sin(phi) * sy2)

def dr_dphi(theta, phi):
    s = mp.sin(theta / 2)
    return -I_c * s * (-mp.sin(phi) * sx2 + mp.cos(phi) * sy2)

def kron22(A, B):
    C = mp.matrix(4, 4)
    for i in range(2):
        for j in range(2):
            a = A[i, j]
            for p in range(2):
                for q in range(2): C[i * 2 + p, j * 2 + q] = a * B[p, q]
    return C

def kron24(A, B4):
    C = mp.matrix(8, 8)
    for i in range(2):
        for j in range(2):
            a = A[i, j]
            for p in range(4):
                for q in range(4): C[i * 4 + p, j * 4 + q] = a * B4[p, q]
    return C

def R_block(theta, phi):
    r = r_phi(theta, phi)
    r4 = kron22(r, r)
    return kron24(r, r4)

def T_block(chi):
    T = mp.matrix(8, 8)
    for i in range(8): T[i, i] = mp.e ** (-I_c * chi * ZW[i])
    return T

def dR_dtheta(theta, phi):
    r = r_phi(theta, phi); dr = dr_dtheta(theta, phi)
    r4 = kron22(r, r)
    return kron24(dr, r4) + kron24(r, kron22(dr, r)) + kron24(r, kron22(r, dr))

def dR_dphi(theta, phi):
    r = r_phi(theta, phi); dph = dr_dphi(theta, phi)
    r4 = kron22(r, r)
    return kron24(dph, r4) + kron24(r, kron22(dph, r)) + kron24(r, kron22(r, dph))

def dT_dchi(chi):
    dT = mp.matrix(8, 8)
    for i in range(8): dT[i, i] = -I_c * ZW[i] * mp.e ** (-I_c * chi * ZW[i])
    return dT

def U_word(params, c, L=7):
    th = [params[i] for i in range(L)]
    ph = [mp.mpf('0')] + [params[L + i] for i in range(L - 1)]
    ch = [params[2 * L - 1 + i] for i in range(L)]
    U = c.I_mp
    for el in range(L): U = R_block(th[el], ph[el]) * T_block(ch[el]) * U
    return U

def dU_dparam(params, param_idx, L=7):
    n = L
    th = [params[i] for i in range(L)]
    ph = [mp.mpf('0')] + [params[L + i] for i in range(L - 1)]
    ch = [params[2 * L - 1 + i] for i in range(L)]
    if param_idx < L:
        seg = param_idx + 1; der = dR_dtheta(th[seg - 1], ph[seg - 1]) * T_block(ch[seg - 1])
    elif param_idx < 2 * L - 1:
        seg = param_idx - L + 2; der = dR_dphi(th[seg - 1], ph[seg - 1]) * T_block(ch[seg - 1])
    else:
        seg = param_idx - (2 * L - 1) + 1; der = R_block(th[seg - 1], ph[seg - 1]) * dT_dchi(ch[seg - 1])
    above = mp.eye(8)
    for ell in range(seg + 1, n + 1): above = R_block(th[ell - 1], ph[ell - 1]) * T_block(ch[ell - 1]) * above
    below = mp.eye(8)
    for ell in range(1, seg): below = R_block(th[ell - 1], ph[ell - 1]) * T_block(ch[ell - 1]) * below
    return above * der * below

def residual(U, c):
    V = c.Gtarg.H * U
    C = -I_c * (V - c.I_mp) * mp.inverse(V + c.I_mp)
    res = mp.zeros(19, 1)
    for k in range(19):
        s = mp.mpc('0')
        for i in range(8):
            for j in range(8): s += C[i, j] * c.basis_mp[k][j, i]
        res[k] = mp.re(s)
    return res

def analytic_jacobian(params, c, L=7):
    nparam = 3 * L - 1
    U = U_word(params, c, L)
    V = c.Gtarg.H * U
    Winv = mp.inverse(V + c.I_mp)
    J = mp.zeros(19, nparam)
    for a in range(nparam):
        dU = dU_dparam(params, a, L)
        dV = c.Gtarg.H * dU
        dC = -2 * I_c * Winv * dV * Winv
        for k in range(19):
            s = mp.mpc('0')
            for i in range(8):
                for j in range(8): s += dC[i, j] * c.basis_mp[k][j, i]
            J[k, a] = mp.re(s)
    return J
