"""Independent bounded checks; closed Pauli exponentials and block derivatives.

No optimization, archived-code imports, or package installations.
Numerics are cross-checks, not an interval certificate or theorem proof.
"""
import hashlib
import json
import math
import platform
import time
from datetime import datetime, timezone
from fractions import Fraction
from functools import reduce
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import expm

root = Path(__file__).resolve().parent
run = root / "runs" / "stage1_independent_20260923_v1"
run.mkdir(parents=True, exist_ok=False)
started = time.perf_counter()
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.diag([1., -1.]).astype(complex)
failures = []


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(name, residual, tolerance=1e-10):
    value = float(residual)
    if not np.isfinite(value) or abs(value) > tolerance:
        failures.append({"name": name, "residual": value, "tolerance": tolerance})
    return value


def channel_distance(U, V):
    """For unitaries, shortest eigenphase arc; avoid 1-|overlap|^2."""
    phases = np.sort(np.angle(np.linalg.eigvals(V.conj().T @ U)))
    gaps = np.diff(np.r_[phases, phases[0] + 2 * np.pi])
    width = 2 * np.pi - float(np.max(gaps))
    return float(np.sin(max(0., width) / 2)) if width < np.pi else 1.


def pauli_word(symbols):
    return reduce(np.kron, ({"I": I, "X": X, "Y": Y, "Z": Z}[s] for s in symbols))


P, Q, R = [pauli_word(s) for s in ["ZZ", "ZX", "IY"]]
theta = np.pi / 2
c = -1 / 8
s = math.sqrt(63) / 8
phi = math.acos(c)
Id = np.eye(4, dtype=complex)


def rotation(a, axis, epsilon=0.):
    value = (1 + epsilon) * a / 2
    return math.cos(value) * Id - 1j * math.sin(value) * axis


target = rotation(theta, P)
Aplus, Aminus = c * P + s * Q, c * P - s * Q
L = rotation(phi, R)
residuals = {
    "embedded_su2": check("embedded_su2", np.linalg.norm(P @ Q - Q @ P - 2j * R)),
    "physical_axis_conjugation": check("axis", np.linalg.norm(L @ P @ L.conj().T - Aplus)),
    "nominal_target": check("target", np.linalg.norm(rotation(2*np.pi, Aminus) @ rotation(2*np.pi, Aplus) @ target-target)),
}
K = theta/2*P + np.pi*target.conj().T@(Aplus+Aminus)@target
residuals["common_first_jet"] = check("common_first_jet", np.linalg.norm(K))
exact_first_jet = Fraction(1, 4) + 2*Fraction(-1, 8)
assert exact_first_jet == 0
scaling = []
for epsilon in [sign*v for v in [.1, .03, .01, .003, .001, .0003] for sign in [-1, 1]]:
    composite = rotation(2*np.pi, Aminus, epsilon) @ rotation(2*np.pi, Aplus, epsilon) @ rotation(theta, P, epsilon)
    d = channel_distance(composite, target)
    native = channel_distance(rotation(theta, P, epsilon), target)
    check("native_exact_formula", native-abs(math.sin(epsilon*theta/2)))
    scaling.append({"epsilon": epsilon, "composite_D": d,
                    "D_over_epsilon_squared": d/epsilon**2, "native_D": native})
coefficient = np.pi**2 * math.sqrt(63) / 32
limiting_discrepancy = max(abs(r["D_over_epsilon_squared"]-coefficient)
                          for r in scaling if abs(r["epsilon"]) == .0003)
check("second_order_coefficient", limiting_discrepancy, 1e-5)

# The added control genuinely leaves the old excitation sectors.
Nexc2 = np.diag([0, 1, 1, 2])
mixing_norm = float(np.linalg.norm(L @ Nexc2 - Nexc2 @ L))
if mixing_norm <= 1e-10:
    failures.append({"name": "resource_addition_did_not_break_symmetry"})

# Noncommuting N=3 schedule, independently differentiate the block exponential.
z1, z2, z3 = [pauli_word(w) for w in ["ZII", "IZI", "IIZ"]]
e12 = (pauli_word("XXI") + pauli_word("YYI"))/2
e23 = (pauli_word("IXX") + pauli_word("IYY"))/2
B = pauli_word("ZZI") + 2*pauli_word("IZZ")
durations = [.4, .7, .3, .6]
strengths = [.3, .8, .5, .2]
hc = [.4*z1+.3*e23, -.2*z2+.5*e12,
      .3*z3-.4*e23, .2*(z1-z3)+.3*(e12+e23)]
S = sum(t*g for t, g in zip(durations, strengths))
U = np.eye(8, dtype=complex)
dU = np.zeros((8, 8), dtype=complex)
for t, g, h in zip(durations, strengths, hc):
    A = -1j*t*(h+g*B)
    E = -1j*t*g*B
    augmented = np.block([[A, E], [np.zeros_like(A), A]])
    answer = expm(augmented)
    J, dJ = answer[:8, :8], answer[:8, 8:]
    dU = dJ@U + J@dU
    U = J@U
K3raw = 1j*U.conj().T@dU
check("derivative_hermiticity", np.linalg.norm(K3raw-K3raw.conj().T))
K3 = (K3raw+K3raw.conj().T)/2
mean_records = []
for k in range(4):
    indices = [i for i in range(8) if i.bit_count() == k]
    actual = np.trace(K3[np.ix_(indices, indices)]).real/len(indices)
    expected = S*3*((3-2*k)**2-3)/6
    check("sector_trace", actual-expected)
    mean_records.append({"k": k, "actual": float(actual), "expected": expected})
eigen = np.linalg.eigvalsh(K3)
slope = float((eigen[-1]-eigen[0])/2)
if slope < S*4/2 - 1e-10:
    failures.append({"name": "slope_lower_bound", "slope": slope})
finite_records = []
for epsilon in [sign*v for v in [.5, .3, .1, .03, .01, .003, .001, .0003] for sign in [-1, 1]]:
    Ue = np.eye(8, dtype=complex)
    for t, g, h in zip(durations, strengths, hc):
        Ue = expm(-1j*t*(h+(1+epsilon)*g*B))@Ue
    D = channel_distance(Ue, U)
    lower = math.sin(abs(epsilon)*S*4/2)
    if D < lower-1e-10:
        failures.append({"name": "finite_lower_bound", "epsilon": epsilon, "gap": D-lower})
    finite_records.append({"epsilon": epsilon, "L": abs(epsilon)*S*6/2,
                           "D": D, "lower": lower, "gap": D-lower})

# Check the collective coefficient by exact fractions, not a copied expression.
collective = []
for n in range(2, 31):
    js = [Fraction(j2, 2) for j2 in range(n % 2, n+1, 2)]
    means = [Fraction(2, 3)*j*(j+1)-Fraction(n, 2) for j in js]
    spectrum = [Fraction((n-2*k)**2-n, 2) for k in range(n+1)]
    rho = (max(means)-min(means))/(max(spectrum)-min(spectrum))
    predicted = Fraction(n+2, 3*n) if n % 2 == 0 else Fraction(n+3, 3*(n+1))
    assert rho == predicted
    collective.append({"N": n, "rho": str(rho)})

# Explicitly retain the rejected midpoint argument and resonance example.
testB = np.array([-3., 1., 0.])
wrong_midpoint = -.5
correct_midpoint = -1.
midpoint_example = {
    "wrong_center_norm": float(max(abs(testB-wrong_midpoint))),
    "correct_center_norm": float(max(abs(testB-correct_midpoint))),
    "half_spectral_width": 2.,
}
resonances = []
for n in [0, 1, 10, 100]:
    action = np.pi/4+n*np.pi
    epsilon = np.pi/action
    native = rotation(2*action, P, epsilon)
    distance = channel_distance(native, target)
    check("unbounded_action_resonance", distance)
    resonances.append({"n": n, "S": action, "epsilon": epsilon, "D": distance})

result = {
    "status": "PASS" if not failures else "FAIL",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "Independent matrix cross-checks and exact finite identities, not a proof or interval certificate.",
    "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
    "precision": "IEEE 754 binary64 complex arithmetic; Fraction for stated exact identities",
    "script_sha256": digest(__file__),
    "protocol_sha256": digest(root/"STAGE1_PROTOCOL_20260923_v1.md"),
    "residuals": residuals, "exact_first_jet_coefficient_over_pi": str(exact_first_jet),
    "predicted_second_order_coefficient": coefficient,
    "finite_e_second_order_coefficient_discrepancy": limiting_discrepancy,
    "resource_breaks_sector_commutator_norm": mixing_norm,
    "composite_checks": scaling,
    "n3_schedule": {"durations": durations, "g": strengths, "S": S,
                    "Delta_B": 6, "Delta_sec": 4,
                    "response_slope": slope, "slope_lower_bound": S*2,
                    "sector_means": mean_records, "finite_errors": finite_records},
    "collective_exact_checks": collective,
    "rejected_midpoint_example": midpoint_example,
    "unbounded_action_resonances": resonances,
    "failures": failures,
    "runtime_seconds": time.perf_counter()-started,
}
with (run/"results.json").open("x", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2)
print(json.dumps({"status": result["status"], "runtime": result["runtime_seconds"],
                  "failures": failures, "n3_slope": slope, "quadratic_coefficient": coefficient}))
raise SystemExit(bool(failures))
