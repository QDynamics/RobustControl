"""Independent FULL8 propagation for the N=3 manifold re-entry audit.

The implementation deliberately uses Pauli tensor products on the complete
eight-dimensional Hilbert space.  The Schur isometries below are used only to
read out the symmetric target block and one representative multiplicity copy;
they are never used as a substitute for FULL8 propagation or residuals.

Legacy parameter convention (15 layers, 44 real controls)::

    theta = p[0:15]
    phi   = [0] + p[15:29]
    chi   = p[29:44]

For increasing ``layer`` the code left-multiplies
``R_phi(theta) @ T(chi)``.  Thus one layer acts chronologically as twist then
rotation, and a module is ``U_14 ... U_0``.  Here ``chi`` is the old H_ZZ
angle in ``T=exp(-i chi H_ZZ)``; a Jz^2 coefficient would be ``kappa=2 chi``
up to a global phase.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence

import numpy as np
from scipy.linalg import expm, expm_frechet


N_QUBITS = 3
DIMENSION = 8
LENGTH = 15
PARAMETER_COUNT = 44
INDEPENDENT_ROWS = (
    0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
    36, 37, 38,
)
RANK_THRESHOLDS = (1.0e-6, 1.0e-8, 1.0e-10, 1.0e-12)


@dataclass(frozen=True)
class ModuleEvaluation:
    unitary: np.ndarray
    d_error_rotation: np.ndarray
    d_error_twist: np.ndarray
    k_error_rotation: np.ndarray
    k_error_twist: np.ndarray


def _kron_all(items: Iterable[np.ndarray]) -> np.ndarray:
    out = np.array([[1.0 + 0.0j]])
    for item in items:
        out = np.kron(out, item)
    return out


I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


def _one_body(operator: np.ndarray, site: int) -> np.ndarray:
    return _kron_all(operator if q == site else I2 for q in range(N_QUBITS))


JX = sum((_one_body(X, q) for q in range(N_QUBITS)), np.zeros((8, 8), complex)) / 2.0
JY = sum((_one_body(Y, q) for q in range(N_QUBITS)), np.zeros((8, 8), complex)) / 2.0
JZ = sum((_one_body(Z, q) for q in range(N_QUBITS)), np.zeros((8, 8), complex)) / 2.0
HZZ = sum(
    (
        _kron_all(Z if q in (i, j) else I2 for q in range(N_QUBITS))
        for i in range(N_QUBITS)
        for j in range(i + 1, N_QUBITS)
    ),
    np.zeros((8, 8), complex),
)


def schur_isometries() -> tuple[np.ndarray, np.ndarray]:
    """Return old-order spin-3/2 and one spin-1/2 isometry.

    The visible columns are ``|111>, |D_2>, |D_1>, |000>``.  The hidden
    columns are the first-two-qubit singlet tensored with ``|1>, |0>``;
    this ordering realizes the old hidden block ``[[d,c],[b,a]]``.
    """
    visible = np.zeros((8, 4), dtype=np.complex128)
    visible[7, 0] = 1.0
    visible[[3, 5, 6], 1] = 1.0 / np.sqrt(3.0)
    visible[[1, 2, 4], 2] = 1.0 / np.sqrt(3.0)
    visible[0, 3] = 1.0

    hidden = np.zeros((8, 2), dtype=np.complex128)
    hidden[3, 0] = 1.0 / np.sqrt(2.0)   # (|011>-|101>)/sqrt(2)
    hidden[5, 0] = -1.0 / np.sqrt(2.0)
    hidden[2, 1] = 1.0 / np.sqrt(2.0)   # (|010>-|100>)/sqrt(2)
    hidden[4, 1] = -1.0 / np.sqrt(2.0)
    return visible, hidden


VISIBLE_ISOMETRY, HIDDEN_ISOMETRY = schur_isometries()
VISIBLE_PROJECTOR = VISIBLE_ISOMETRY @ VISIBLE_ISOMETRY.conj().T
COMPLEMENT_PROJECTOR = np.eye(8, dtype=np.complex128) - VISIBLE_PROJECTOR


def parse_fraction_array(values: Sequence[str]) -> np.ndarray:
    if len(values) != PARAMETER_COUNT:
        raise ValueError(f"expected {PARAMETER_COUNT} parameters, got {len(values)}")
    return np.asarray([float(Fraction(value)) for value in values], dtype=np.float64)


def split_parameters(params: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(params, dtype=np.float64)
    if p.shape != (PARAMETER_COUNT,):
        raise ValueError(f"one module requires shape ({PARAMETER_COUNT},)")
    theta = p[:LENGTH]
    phi = np.concatenate((np.zeros(1), p[LENGTH:2 * LENGTH - 1]))
    chi = p[2 * LENGTH - 1:]
    return theta, phi, chi


def layer_unitaries(theta: float, phi: float, chi_old: float,
                    epsilon_rotation: float = 0.0,
                    epsilon_twist: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    jphi = np.cos(phi) * JX + np.sin(phi) * JY
    rotation = expm(-1.0j * (1.0 + epsilon_rotation) * theta * jphi)
    twist = expm(-1.0j * (1.0 + epsilon_twist) * chi_old * HZZ)
    return rotation, twist


def module_propagator(params: Sequence[float], epsilon_rotation: float = 0.0,
                      epsilon_twist: float = 0.0,
                      with_error_derivatives: bool = True) -> ModuleEvaluation:
    """Propagate one module and analytically differentiate common errors."""
    theta, phi, chi = split_parameters(params)
    unitary = np.eye(DIMENSION, dtype=np.complex128)
    d_rotation = np.zeros_like(unitary)
    d_twist = np.zeros_like(unitary)
    for layer in range(LENGTH):
        jphi = np.cos(phi[layer]) * JX + np.sin(phi[layer]) * JY
        rotation, twist = layer_unitaries(
            theta[layer], phi[layer], chi[layer],
            epsilon_rotation, epsilon_twist,
        )
        primitive = rotation @ twist
        if with_error_derivatives:
            d_primitive_rotation = (-1.0j * theta[layer] * jphi @ rotation) @ twist
            d_primitive_twist = rotation @ (-1.0j * chi[layer] * HZZ @ twist)
            d_rotation = d_primitive_rotation @ unitary + primitive @ d_rotation
            d_twist = d_primitive_twist @ unitary + primitive @ d_twist
        unitary = primitive @ unitary
    if with_error_derivatives:
        k_rotation = 1.0j * unitary.conj().T @ d_rotation
        k_twist = 1.0j * unitary.conj().T @ d_twist
        k_rotation = (k_rotation + k_rotation.conj().T) / 2.0
        k_twist = (k_twist + k_twist.conj().T) / 2.0
    else:
        k_rotation = np.zeros_like(unitary)
        k_twist = np.zeros_like(unitary)
    return ModuleEvaluation(unitary, d_rotation, d_twist, k_rotation, k_twist)


def module_control_derivatives(params: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return nominal U and analytic derivatives dU/dp_j for all 44 controls."""
    theta, phi, chi = split_parameters(params)
    layers: list[np.ndarray] = []
    local_derivatives: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for layer in range(LENGTH):
        jphi = np.cos(phi[layer]) * JX + np.sin(phi[layer]) * JY
        djphi = -np.sin(phi[layer]) * JX + np.cos(phi[layer]) * JY
        exponent = -1.0j * theta[layer] * jphi
        rotation = expm(exponent)
        twist = expm(-1.0j * chi[layer] * HZZ)
        dtheta = (-1.0j * jphi @ rotation) @ twist
        dphi = expm_frechet(exponent, -1.0j * theta[layer] * djphi,
                            compute_expm=False) @ twist
        dchi = rotation @ (-1.0j * HZZ @ twist)
        layers.append(rotation @ twist)
        local_derivatives.append((dtheta, dphi, dchi))

    prefix = [np.eye(DIMENSION, dtype=np.complex128)]
    for primitive in layers:
        prefix.append(primitive @ prefix[-1])
    left_after: list[np.ndarray] = [np.empty((8, 8), complex) for _ in range(LENGTH)]
    carry = np.eye(DIMENSION, dtype=np.complex128)
    for layer in range(LENGTH - 1, -1, -1):
        left_after[layer] = carry
        carry = carry @ layers[layer]

    derivatives = np.zeros((PARAMETER_COUNT, DIMENSION, DIMENSION), complex)
    for layer in range(LENGTH):
        derivatives[layer] = left_after[layer] @ local_derivatives[layer][0] @ prefix[layer]
        if layer > 0:
            derivatives[LENGTH + layer - 1] = (
                left_after[layer] @ local_derivatives[layer][1] @ prefix[layer]
            )
        derivatives[2 * LENGTH - 1 + layer] = (
            left_after[layer] @ local_derivatives[layer][2] @ prefix[layer]
        )
    return prefix[-1], derivatives


def transported_generators_with_control_derivatives(
    params: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Differentiate U and both prefix-transported error generators analytically.

    For a chronological primitive with ideal prefix ``P`` and error generator
    ``h``, its input-frame contribution is ``P^dagger h P``.  Its control
    derivative is evaluated before updating ``P``.  The chronological primitive
    list is ``T_0,R_0,T_1,R_1,...,T_14,R_14``.
    """
    theta, phi, chi = split_parameters(params)
    p = np.eye(DIMENSION, dtype=np.complex128)
    dp = np.zeros((PARAMETER_COUNT, DIMENSION, DIMENSION), dtype=np.complex128)
    k_rotation = np.zeros_like(p)
    k_twist = np.zeros_like(p)
    dk_rotation = np.zeros_like(dp)
    dk_twist = np.zeros_like(dp)

    def add_contribution(k: np.ndarray, dk: np.ndarray, h: np.ndarray,
                         dh: dict[int, np.ndarray]) -> None:
        k += p.conj().T @ h @ p
        for column in range(PARAMETER_COUNT):
            local = dh.get(column)
            term = dp[column].conj().T @ h @ p + p.conj().T @ h @ dp[column]
            if local is not None:
                term = term + p.conj().T @ local @ p
            dk[column] += term

    def update_prefix(primitive: np.ndarray, dprimitive: dict[int, np.ndarray]) -> None:
        nonlocal p, dp
        old_p = p
        old_dp = dp
        p = primitive @ old_p
        dp = np.einsum("ab,jbc->jac", primitive, old_dp)
        for column, local in dprimitive.items():
            dp[column] += local @ old_p

    for layer in range(LENGTH):
        # Twist acts first in physical time.
        twist = expm(-1.0j * chi[layer] * HZZ)
        chi_index = 2 * LENGTH - 1 + layer
        add_contribution(k_twist, dk_twist, chi[layer] * HZZ,
                         {chi_index: HZZ})
        update_prefix(twist, {chi_index: -1.0j * HZZ @ twist})

        # Rotation acts second.  phi_0 is fixed and is not a coordinate.
        jphi = np.cos(phi[layer]) * JX + np.sin(phi[layer]) * JY
        djphi = -np.sin(phi[layer]) * JX + np.cos(phi[layer]) * JY
        rotation = expm(-1.0j * theta[layer] * jphi)
        theta_index = layer
        dprimitive = {theta_index: -1.0j * jphi @ rotation}
        dh = {theta_index: jphi}
        if layer > 0:
            phi_index = LENGTH + layer - 1
            # R_phi = exp(-i phi Jz) R_0 exp(+i phi Jz).
            dprimitive[phi_index] = -1.0j * JZ @ rotation + 1.0j * rotation @ JZ
            dh[phi_index] = theta[layer] * djphi
        add_contribution(k_rotation, dk_rotation, theta[layer] * jphi, dh)
        update_prefix(rotation, dprimitive)

    # Remove roundoff anti-Hermitian parts without changing the analytic map.
    k_rotation = (k_rotation + k_rotation.conj().T) / 2.0
    k_twist = (k_twist + k_twist.conj().T) / 2.0
    dk_rotation = (dk_rotation + np.swapaxes(dk_rotation.conj(), 1, 2)) / 2.0
    dk_twist = (dk_twist + np.swapaxes(dk_twist.conj(), 1, 2)) / 2.0
    return p, dp, (k_rotation, k_twist), (dk_rotation, dk_twist)


def block(matrix: np.ndarray, isometry: np.ndarray) -> np.ndarray:
    return isometry.conj().T @ matrix @ isometry


def generator_coordinates(generator: np.ndarray) -> np.ndarray:
    coordinates: list[float] = [float(np.real(generator[i, i])) for i in range(generator.shape[0])]
    for i in range(generator.shape[0]):
        for j in range(i + 1, generator.shape[0]):
            coordinates.extend((float(np.real(generator[i, j])), float(np.imag(generator[i, j]))))
    return np.asarray(coordinates)


def module_constraint_vector(params: Sequence[float], independent: bool = True) -> np.ndarray:
    """Evaluate the old 41-row chart from an independent FULL8 propagation."""
    evaluation = module_propagator(params)
    visible = block(evaluation.unitary, VISIBLE_ISOMETRY)
    hidden = block(evaluation.unitary, HIDDEN_ISOMETRY)
    tau = np.trace(hidden)
    if abs(tau) < 0.5:
        raise ValueError("hidden trace chart is singular")
    quotient = (
        (hidden[1, 0] + hidden[0, 1]) / tau,
        (hidden[1, 0] - hidden[0, 1]) / tau,
        (hidden[0, 0] - hidden[1, 1]) / tau,
    )
    rows: list[float] = [-quotient[0].imag, quotient[1].real, -quotient[2].imag]
    for derivative in (evaluation.d_error_rotation, evaluation.d_error_twist):
        d_visible = block(derivative, VISIBLE_ISOMETRY)
        d_hidden = block(derivative, HIDDEN_ISOMETRY)
        k_visible = 1.0j * visible.conj().T @ d_visible
        k_hidden = 1.0j * hidden.conj().T @ d_hidden
        k_visible = (k_visible + k_visible.conj().T) / 2.0
        k_hidden = (k_hidden + k_hidden.conj().T) / 2.0
        vc = generator_coordinates(k_visible)
        hc = generator_coordinates(k_hidden)
        rows.extend(vc.tolist())
        rows.extend(hc[list((0, 2, 3))].tolist())
    full = np.asarray(rows, dtype=np.float64)
    if full.shape != (41,):
        raise AssertionError(f"wrong constraint shape {full.shape}")
    return full[list(INDEPENDENT_ROWS)] if independent else full


def analytic_constraint_value_and_jacobian(
    params: Sequence[float], independent: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the old constraint chart and its analytic control Jacobian.

    Both error generators and every derivative are built on the complete 8x8
    space.  Block compression happens only after propagation.
    """
    unitary, dunitary, generators, dgenerators = (
        transported_generators_with_control_derivatives(params)
    )
    hidden = block(unitary, HIDDEN_ISOMETRY)
    dhidden = np.asarray([block(item, HIDDEN_ISOMETRY) for item in dunitary])
    tau = np.trace(hidden)
    dtau = np.trace(dhidden, axis1=1, axis2=2)
    if abs(tau) < 0.5:
        raise ValueError("hidden trace chart is singular")
    numerators = np.asarray((
        hidden[1, 0] + hidden[0, 1],
        hidden[1, 0] - hidden[0, 1],
        hidden[0, 0] - hidden[1, 1],
    ))
    dnumerators = np.asarray((
        dhidden[:, 1, 0] + dhidden[:, 0, 1],
        dhidden[:, 1, 0] - dhidden[:, 0, 1],
        dhidden[:, 0, 0] - dhidden[:, 1, 1],
    ))
    quotients = numerators / tau
    dquotients = dnumerators / tau - numerators[:, None] * dtau[None, :] / (tau * tau)
    values: list[float] = [-quotients[0].imag, quotients[1].real, -quotients[2].imag]
    rows: list[np.ndarray] = [-dquotients[0].imag, dquotients[1].real, -dquotients[2].imag]

    for generator, dgenerator in zip(generators, dgenerators):
        visible_generator = block(generator, VISIBLE_ISOMETRY)
        hidden_generator = block(generator, HIDDEN_ISOMETRY)
        visible_derivatives = np.asarray([block(item, VISIBLE_ISOMETRY) for item in dgenerator])
        hidden_derivatives = np.asarray([block(item, HIDDEN_ISOMETRY) for item in dgenerator])
        vc = generator_coordinates(visible_generator)
        hc = generator_coordinates(hidden_generator)
        values.extend(vc.tolist())
        values.extend(hc[list((0, 2, 3))].tolist())
        for coordinate in range(16):
            rows.append(np.asarray([
                generator_coordinates(visible_derivatives[column])[coordinate]
                for column in range(PARAMETER_COUNT)
            ]))
        for coordinate in (0, 2, 3):
            rows.append(np.asarray([
                generator_coordinates(hidden_derivatives[column])[coordinate]
                for column in range(PARAMETER_COUNT)
            ]))
    value_array = np.asarray(values, dtype=np.float64)
    jacobian = np.asarray(rows, dtype=np.float64)
    if value_array.shape != (41,) or jacobian.shape != (41, PARAMETER_COUNT):
        raise AssertionError(f"wrong analytic constraint shapes {value_array.shape}, {jacobian.shape}")
    if independent:
        indices = list(INDEPENDENT_ROWS)
        return value_array[indices], jacobian[indices, :]
    return value_array, jacobian


def finite_difference_constraint_jacobian(params: Sequence[float], step: float = 1.0e-6) -> np.ndarray:
    """Central numerical 37x44 Jacobian; the error derivatives inside are analytic."""
    p = np.asarray(params, dtype=np.float64)
    jacobian = np.empty((len(INDEPENDENT_ROWS), PARAMETER_COUNT), dtype=np.float64)
    for column in range(PARAMETER_COUNT):
        plus, minus = p.copy(), p.copy()
        plus[column] += step
        minus[column] -= step
        jacobian[:, column] = (
            module_constraint_vector(plus) - module_constraint_vector(minus)
        ) / (2.0 * step)
    return jacobian


def interface_gate_and_derivatives(params: Sequence[float]) -> tuple[np.ndarray, np.ndarray, complex]:
    """Return V/lambda and analytic control derivatives in the hidden-trace chart."""
    unitary, derivatives = module_control_derivatives(params)
    visible = block(unitary, VISIBLE_ISOMETRY)
    hidden = block(unitary, HIDDEN_ISOMETRY)
    lam = np.trace(hidden) / 2.0
    if abs(lam) < 0.25:
        raise ValueError("hidden scalar chart is singular")
    d_visible = np.asarray([block(item, VISIBLE_ISOMETRY) for item in derivatives])
    d_hidden = np.asarray([block(item, HIDDEN_ISOMETRY) for item in derivatives])
    d_lam = np.trace(d_hidden, axis1=1, axis2=2) / 2.0
    gate = visible / lam
    d_gate = d_visible / lam - visible[None, :, :] * d_lam[:, None, None] / (lam * lam)
    return gate, d_gate, lam


def normalized_pauli_basis_4() -> list[np.ndarray]:
    return [
        np.kron(left, right) / 2.0
        for i, left in enumerate((I2, X, Y, Z))
        for j, right in enumerate((I2, X, Y, Z))
        if (i, j) != (0, 0)
    ]


def su4_coordinates(hermitian: np.ndarray) -> np.ndarray:
    h = (hermitian + hermitian.conj().T) / 2.0
    h = h - np.trace(h) * np.eye(4) / 4.0
    return np.asarray([np.trace(basis @ h).real for basis in normalized_pauli_basis_4()])


def rank_report(matrix: np.ndarray, thresholds: Sequence[float] = RANK_THRESHOLDS) -> dict[str, object]:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return {
        "shape": list(matrix.shape),
        "singular_values": singular_values.tolist(),
        "ranks": {f"{threshold:.0e}": int(np.count_nonzero(singular_values > threshold))
                  for threshold in thresholds},
    }


def null_basis(matrix: np.ndarray, rank_threshold: float = 1.0e-8) -> np.ndarray:
    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    rank = int(np.count_nonzero(singular_values > rank_threshold))
    return vh[rank:].conj().T


def target_map_on_module_nullspaces(
    params_by_module: Sequence[Sequence[float]],
    nullspaces: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Return 15x(sum nullities) fixed-target differential and composite gate."""
    if len(params_by_module) != 3 or len(nullspaces) != 3:
        raise ValueError("the registered target uses exactly three modules")
    gates: list[np.ndarray] = []
    gate_derivatives: list[np.ndarray] = []
    for params in params_by_module:
        gate, derivatives, _ = interface_gate_and_derivatives(params)
        gates.append(gate)
        gate_derivatives.append(derivatives)
    target = gates[0] @ gates[1] @ gates[2]
    columns: list[np.ndarray] = []
    for module in range(3):
        left = np.eye(4, dtype=complex)
        for index in range(module):
            left = left @ gates[index]
        right = np.eye(4, dtype=complex)
        for index in range(module + 1, 3):
            right = right @ gates[index]
        for direction in range(nullspaces[module].shape[1]):
            dp = nullspaces[module][:, direction]
            dg = np.tensordot(dp, gate_derivatives[module], axes=(0, 0))
            dtarget = left @ dg @ right
            hermitian = 1.0j * target.conj().T @ dtarget
            columns.append(su4_coordinates(hermitian))
    return np.column_stack(columns), target, gates, gate_derivatives


def compose_full8(evaluations: Sequence[ModuleEvaluation]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compose the registered mathematical product U00 U15 U25 and derivatives."""
    if len(evaluations) != 3:
        raise ValueError("expected three module evaluations")
    u0, u1, u2 = (item.unitary for item in evaluations)
    total = u0 @ u1 @ u2
    derivatives = []
    for field in ("d_error_rotation", "d_error_twist"):
        d0, d1, d2 = (getattr(item, field) for item in evaluations)
        derivatives.append(d0 @ u1 @ u2 + u0 @ d1 @ u2 + u0 @ u1 @ d2)
    return total, derivatives[0], derivatives[1]


def unitary_channel_metrics(target: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    """Stable half-diamond metric from the minimal eigenphase covering arc."""
    relative = target.conj().T @ actual
    eigenvalues = np.linalg.eigvals(relative)
    phases = np.sort(np.mod(np.angle(eigenvalues), 2.0 * np.pi))
    wrapped = np.concatenate((phases, phases[:1] + 2.0 * np.pi))
    gaps = np.diff(wrapped)
    cut = int(np.argmax(gaps))
    cover = float(2.0 * np.pi - gaps[cut])
    if cover >= np.pi:
        half_diamond = 1.0
    else:
        half_diamond = float(np.sin(cover / 2.0))
    start = phases[(cut + 1) % len(phases)]
    unwrapped = np.mod(phases - start, 2.0 * np.pi)
    centered = unwrapped - np.mean(unwrapped)
    d = target.shape[0]
    trace_overlap = np.trace(relative)
    average_infidelity = float(1.0 - (abs(trace_overlap) ** 2 + d) / (d * (d + 1.0)))
    return {
        "half_diamond": half_diamond,
        "one_minus_fmin": half_diamond * half_diamond,
        "minimal_covering_arc": cover,
        "average_gate_infidelity_aux": max(0.0, average_infidelity),
        "eigenphase_variance_aux": float(np.mean(centered * centered)),
        "relative_unitarity_fro": float(np.linalg.norm(relative.conj().T @ relative - np.eye(d))),
    }


def complex_array_record(array: np.ndarray) -> dict[str, object]:
    return {"shape": list(array.shape), "real": array.real.tolist(), "imag": array.imag.tolist()}


def full8_residuals(evaluation: ModuleEvaluation) -> dict[str, float]:
    u = evaluation.unitary
    q = COMPLEMENT_PROJECTOR
    scalar = np.trace(q @ u @ q) / 4.0
    return {
        "unitarity_fro": float(np.linalg.norm(u.conj().T @ u - np.eye(8))),
        "symmetric_to_complement_leakage_fro": float(np.linalg.norm(q @ u @ VISIBLE_PROJECTOR)),
        "complement_to_symmetric_leakage_fro": float(np.linalg.norm(VISIBLE_PROJECTOR @ u @ q)),
        "complement_scalar_closure_fro": float(np.linalg.norm(q @ u @ q - scalar * q)),
        "complement_scalar_real": float(np.real(scalar)),
        "complement_scalar_imag": float(np.imag(scalar)),
        "k_rotation_fro": float(np.linalg.norm(evaluation.k_error_rotation)),
        "k_twist_fro": float(np.linalg.norm(evaluation.k_error_twist)),
        "k_rotation_trace_abs": float(abs(np.trace(evaluation.k_error_rotation))),
        "k_twist_trace_abs": float(abs(np.trace(evaluation.k_error_twist))),
    }


__all__ = [
    "COMPLEMENT_PROJECTOR", "DIMENSION", "HIDDEN_ISOMETRY", "HZZ",
    "INDEPENDENT_ROWS", "JX", "JY", "JZ", "LENGTH", "ModuleEvaluation",
    "PARAMETER_COUNT", "RANK_THRESHOLDS", "VISIBLE_ISOMETRY",
    "VISIBLE_PROJECTOR", "block", "complex_array_record", "compose_full8",
    "finite_difference_constraint_jacobian", "full8_residuals",
    "interface_gate_and_derivatives", "module_constraint_vector",
    "module_control_derivatives", "module_propagator", "null_basis",
    "parse_fraction_array", "rank_report", "split_parameters", "su4_coordinates",
    "target_map_on_module_nullspaces", "transported_generators_with_control_derivatives",
    "unitary_channel_metrics", "analytic_constraint_value_and_jacobian",
]
