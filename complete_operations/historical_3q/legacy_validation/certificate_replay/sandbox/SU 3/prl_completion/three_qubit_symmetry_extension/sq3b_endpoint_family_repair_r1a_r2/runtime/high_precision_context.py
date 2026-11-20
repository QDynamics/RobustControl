"""R2-1: arbitrary-precision context.  Rebuilds all precision-dependent
objects after mp.mp.dps is set.  No numeric cache files; all reconstructions
use sympy-srepr basis or exact construction.  In-memory caches keyed by dps.
"""
import json, os, mpmath as mp
I_c = mp.mpc(0, 1)

_HERE = os.path.dirname(os.path.abspath(__file__))
_UP = os.path.dirname(_HERE)
_GRAND = os.path.dirname(_UP)
_R1A = os.path.join(_GRAND, "sq3b_endpoint_family_repair_r1a")

# Load srepr exact basis (once, as strings)
with open(os.path.join(_R1A, "exact_data", "commutant_basis_exact.json"), "r") as f:
    _SREPR_DATA = json.load(f)

_cache = {}  # keyed by dps

class PrecisionContext:
    def __init__(self, dps):
        mp.mp.dps = dps
        if dps in _cache:
            self.__dict__ = _cache[dps].__dict__.copy()
            return
        # Build generators at this dps
        # Pauli matrices
        s0 = mp.eye(2)
        sx = mp.matrix([[0, 1], [1, 0]])
        sy = mp.matrix([[0, -I_c], [I_c, 0]])
        sz = mp.matrix([[1, 0], [0, -1]])

        def kron2(A, B):
            C = mp.matrix(4, 4)
            for i in range(2):
                for j in range(2):
                    a = A[i, j]
                    for p in range(2):
                        for q in range(2):
                            C[i * 2 + p, j * 2 + q] = a * B[p, q]
            return C

        def kron3(A, B, Cmat):
            return mp.kronecker(mp.kronecker(A, B), Cmat)

        # Jx = 0.5*(sx⊗s0⊗s0 + s0⊗sx⊗s0 + s0⊗s0⊗sx)
        # Build via kron22 and kron24
        s8 = mp.eye(8)
        def pauli3_8(idx, mat):
            """Build 8x8 with mat at position idx."""
            # kron3(s0, s0, s0) but with mat at idx
            mats = [s0, s0, s0]
            mats[idx] = mat
            result = mp.matrix(8, 8)
            for i in range(2):
                for j in range(2):
                    for p in range(2):
                        for q in range(2):
                            for r in range(2):
                                for s_idx in range(2):
                                    val = mats[0][i, j] * mats[1][p, q] * mats[2][r, s_idx]
                                    row = i * 4 + p * 2 + r
                                    col = j * 4 + q * 2 + s_idx
                                    result[row, col] = val
            return result

        half = mp.mpf('0.5')
        self.Mx = half * (pauli3_8(0, sx) + pauli3_8(1, sx) + pauli3_8(2, sx))
        self.My = half * (pauli3_8(0, sy) + pauli3_8(1, sy) + pauli3_8(2, sy))
        self.Mz = half * (pauli3_8(0, sz) + pauli3_8(1, sz) + pauli3_8(2, sz))
        # H_ZZ = Z1Z2 + Z1Z3 + Z2Z3
        self.Mzz = (pauli3_8(0, sz) * pauli3_8(1, sz) +
                    pauli3_8(0, sz) * pauli3_8(2, sz) +
                    pauli3_8(1, sz) * pauli3_8(2, sz))

        # Gtarg = e^{i pi/8} * CCZ
        CCZ = mp.diag([1, 1, 1, 1, 1, 1, 1, -1])
        self.Gtarg = mp.exp(I_c * mp.pi / 8) * CCZ

        # Commutant basis from srepr, evaluated at this dps
        import sympy as sp
        basis = []
        for entry in _SREPR_DATA["basis"]:
            M = mp.matrix(8, 8)
            for i in range(8):
                for j in range(8):
                    re_str = entry["rows"][i]["real"][j]
                    im_str = entry["rows"][i]["imag"][j]
                    re_expr = sp.sympify(re_str) if re_str != "0" else sp.Integer(0)
                    im_expr = sp.sympify(im_str) if im_str != "0" else sp.Integer(0)
                    re_val = mp.mpf(sp.N(re_expr, dps + 5).evalf(dps + 5))
                    im_val = mp.mpf(sp.N(im_expr, dps + 5).evalf(dps + 5))
                    M[i, j] = mp.mpc(re_val, im_val)
            basis.append(M)
        self.basis_mp = basis
        self.I_mp = mp.eye(8)
        _cache[dps] = self


def ctx(dps=50):
    """Get or build a PrecisionContext at the given dps."""
    return PrecisionContext(dps)
