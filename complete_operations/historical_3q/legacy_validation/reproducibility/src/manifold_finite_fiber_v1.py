"""Finite-step retraction on the registered N=3 fixed-target fiber.

This module supplies the analytic 132-equation chart used by M1b:

* 111 independent per-module FULL8 robustness/closure coordinates;
* 15 source-target coordinates in the visible SU(4) chart; and
* 6 affine graph coordinates fixed by the accepted source fiber basis.

The mathematical module product is ``U00 @ U15 @ U25`` (chronological
execution 25, 15, 00).  No Schur block is used in place of FULL8 acceptance
or finite-error propagation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

import manifold_full8_v1 as m8


MODULE_COUNT = 3
MODULE_LABELS = (0, 15, 25)
TOTAL_PARAMETER_COUNT = MODULE_COUNT * m8.PARAMETER_COUNT
Z_DIRECTION = np.asarray((-1.0, -1.0, -1.0, -1.0, 1.0, 1.0))


@dataclass(frozen=True)
class AugmentedEvaluation:
    value: np.ndarray
    jacobian: np.ndarray | None
    module_values: np.ndarray
    target_value: np.ndarray
    graph_value: np.ndarray
    target_gate4: np.ndarray
    lambdas: np.ndarray


def as_module_parameters(params: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(params, dtype=np.float64)
    if array.shape == (TOTAL_PARAMETER_COUNT,):
        array = array.reshape(MODULE_COUNT, m8.PARAMETER_COUNT)
    if array.shape != (MODULE_COUNT, m8.PARAMETER_COUNT):
        raise ValueError("parameters must have shape (3,44) or (132,)")
    return array


def block_module_constraints(
    params: Sequence[float] | np.ndarray, with_jacobian: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    modules = as_module_parameters(params)
    values: list[np.ndarray] = []
    jacobian = np.zeros((111, 132), dtype=np.float64) if with_jacobian else None
    for module in range(MODULE_COUNT):
        if with_jacobian:
            value, local_jacobian = m8.analytic_constraint_value_and_jacobian(modules[module])
            assert jacobian is not None
            jacobian[37 * module:37 * (module + 1),
                     44 * module:44 * (module + 1)] = local_jacobian
        else:
            value = m8.module_constraint_vector(modules[module], independent=True)
        values.append(value)
    return np.concatenate(values), jacobian


def _gate_product_data(
    params: Sequence[float] | np.ndarray, with_jacobian: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    modules = as_module_parameters(params)
    gates: list[np.ndarray] = []
    derivatives: list[np.ndarray] = []
    lambdas: list[complex] = []
    for module in range(MODULE_COUNT):
        if with_jacobian:
            gate, derivative, lam = m8.interface_gate_and_derivatives(modules[module])
            derivatives.append(derivative)
        else:
            evaluation = m8.module_propagator(modules[module], with_error_derivatives=False)
            visible = m8.block(evaluation.unitary, m8.VISIBLE_ISOMETRY)
            hidden = m8.block(evaluation.unitary, m8.HIDDEN_ISOMETRY)
            lam = np.trace(hidden) / 2.0
            if abs(lam) < 0.25:
                raise ValueError("hidden scalar chart is singular")
            gate = visible / lam
        gates.append(gate)
        lambdas.append(complex(lam))
    target = gates[0] @ gates[1] @ gates[2]
    if not with_jacobian:
        return target, None, np.asarray(lambdas, dtype=np.complex128)

    dtarget = np.zeros((TOTAL_PARAMETER_COUNT, 4, 4), dtype=np.complex128)
    dtarget[0:44] = np.einsum("jab,bc,cd->jad", derivatives[0], gates[1], gates[2])
    dtarget[44:88] = np.einsum("ab,jbc,cd->jad", gates[0], derivatives[1], gates[2])
    dtarget[88:132] = np.einsum("ab,bc,jcd->jad", gates[0], gates[1], derivatives[2])
    return target, dtarget, np.asarray(lambdas, dtype=np.complex128)


def target_local_chart(
    params: Sequence[float] | np.ndarray,
    source_target_gate4: np.ndarray,
    with_jacobian: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """Evaluate a smooth 15-coordinate source-target chart.

    For ``W=G_source^dagger G(p)``, the chart is the normalized-Pauli
    coordinate vector of ``i(W-W^dagger)/2`` after removing its scalar part.
    Its derivative at the source is exactly the accepted target-map chart.
    """
    source = np.asarray(source_target_gate4, dtype=np.complex128)
    if source.shape != (4, 4):
        raise ValueError("source target must have shape (4,4)")
    target, dtarget, lambdas = _gate_product_data(params, with_jacobian)
    relative = source.conj().T @ target
    hermitian = 0.5j * (relative - relative.conj().T)
    value = m8.su4_coordinates(hermitian)
    if not with_jacobian:
        return value, None, target, lambdas
    assert dtarget is not None
    drelative = np.einsum("ab,jbc->jac", source.conj().T, dtarget)
    dhermitian = 0.5j * (drelative - np.swapaxes(drelative.conj(), 1, 2))
    jacobian = np.column_stack([m8.su4_coordinates(item) for item in dhermitian])
    return value, jacobian, target, lambdas


def augmented_value_and_jacobian(
    params: Sequence[float] | np.ndarray,
    source_params: Sequence[float] | np.ndarray,
    source_target_gate4: np.ndarray,
    fiber_basis132x6: np.ndarray,
    h: float,
    z: Sequence[float] = Z_DIRECTION,
    with_jacobian: bool = True,
) -> AugmentedEvaluation:
    modules = as_module_parameters(params)
    source_modules = as_module_parameters(source_params)
    fiber = np.asarray(fiber_basis132x6, dtype=np.float64)
    direction = np.asarray(z, dtype=np.float64)
    if fiber.shape != (132, 6) or direction.shape != (6,):
        raise ValueError("fiber must be 132x6 and z must have length 6")

    module_value, module_jacobian = block_module_constraints(modules, with_jacobian)
    target_value, target_jacobian, target_gate, lambdas = target_local_chart(
        modules, source_target_gate4, with_jacobian
    )
    graph_value = fiber.T @ (modules.reshape(-1) - source_modules.reshape(-1)) - float(h) * direction
    value = np.concatenate((module_value, target_value, graph_value))
    jacobian: np.ndarray | None = None
    if with_jacobian:
        assert module_jacobian is not None and target_jacobian is not None
        jacobian = np.vstack((module_jacobian, target_jacobian, fiber.T))
    if value.shape != (132,) or (jacobian is not None and jacobian.shape != (132, 132)):
        raise AssertionError("wrong augmented chart shape")
    return AugmentedEvaluation(
        value=value,
        jacobian=jacobian,
        module_values=module_value,
        target_value=target_value,
        graph_value=graph_value,
        target_gate4=target_gate,
        lambdas=lambdas,
    )


def phase_aligned_frobenius(target: np.ndarray, actual: np.ndarray) -> tuple[float, complex]:
    overlap = np.trace(target.conj().T @ actual)
    phase = complex(overlap / abs(overlap)) if abs(overlap) else 1.0 + 0.0j
    return float(np.linalg.norm(actual - phase * target)), phase


def charged_signs(params: Sequence[float] | np.ndarray) -> np.ndarray:
    modules = as_module_parameters(params)
    return np.sign(np.concatenate((modules[:, :15], modules[:, 29:44]), axis=1))


def candidate_audit(
    params: Sequence[float] | np.ndarray,
    source_params: Sequence[float] | np.ndarray,
    source_target_full8: np.ndarray,
    source_target_gate4: np.ndarray,
    fiber_basis132x6: np.ndarray,
    h: float,
    z: Sequence[float] = Z_DIRECTION,
) -> dict[str, object]:
    modules = as_module_parameters(params)
    source_modules = as_module_parameters(source_params)
    augmented = augmented_value_and_jacobian(
        modules, source_modules, source_target_gate4, fiber_basis132x6, h, z,
        with_jacobian=False,
    )
    evaluations = [m8.module_propagator(item) for item in modules]
    total, _, _ = m8.compose_full8(evaluations)
    target_fro, target_phase = phase_aligned_frobenius(source_target_full8, total)
    module_records = []
    for label, evaluation, lam in zip(MODULE_LABELS, evaluations, augmented.lambdas):
        residual = m8.full8_residuals(evaluation)
        residual.update({
            "module": label,
            "lambda_real": float(lam.real),
            "lambda_imag": float(lam.imag),
            "lambda_distance_to_minus_one": float(abs(lam + 1.0)),
            "lambda_source_branch_minus_one": bool(lam.real < 0.0 and abs(lam + 1.0) < 1.0e-8),
        })
        module_records.append(residual)
    trust_distance = float(np.linalg.norm(modules - source_modules))
    sign_match = bool(np.array_equal(charged_signs(modules), charged_signs(source_modules)))
    max_k_closure = max(
        max(item["k_rotation_fro"], item["k_twist_fro"],
            item["complement_scalar_closure_fro"],
            item["symmetric_to_complement_leakage_fro"],
            item["complement_to_symmetric_leakage_fro"])
        for item in module_records
    )
    checks = {
        "target_phase_aligned_fro_lt_1e-10": target_fro < 1.0e-10,
        "module_full8_k_closure_lt_1e-9": max_k_closure < 1.0e-9,
        "target_chart_lt_1e-9": float(np.max(np.abs(augmented.target_value))) < 1.0e-9,
        "module_chart_lt_1e-9": float(np.max(np.abs(augmented.module_values))) < 1.0e-9,
        "graph_chart_lt_1e-9": float(np.max(np.abs(augmented.graph_value))) < 1.0e-9,
        "trust_distance_le_0p1": trust_distance <= 0.1,
        "charged_signs_unchanged": sign_match,
        "all_lambda_minus_one_branch": all(item["lambda_source_branch_minus_one"] for item in module_records),
    }
    return {
        "accepted": bool(all(checks.values())),
        "checks": checks,
        "target_phase_aligned_fro": target_fro,
        "target_aligning_phase_real": float(target_phase.real),
        "target_aligning_phase_imag": float(target_phase.imag),
        "module_chart_max_abs": float(np.max(np.abs(augmented.module_values))),
        "target_chart_max_abs": float(np.max(np.abs(augmented.target_value))),
        "graph_chart_max_abs": float(np.max(np.abs(augmented.graph_value))),
        "augmented_chart_max_abs": float(np.max(np.abs(augmented.value))),
        "trust_distance": trust_distance,
        "max_module_full8_k_closure": float(max_k_closure),
        "modules": module_records,
    }


def validation_point_sets(seed: int = 33013) -> dict[str, np.ndarray]:
    grid_axis = np.linspace(-0.01, 0.01, 21)
    grid = np.asarray([(x, y) for x in grid_axis for y in grid_axis], dtype=np.float64)
    midpoint_axis = (grid_axis[:-1] + grid_axis[1:]) / 2.0
    midpoint = np.asarray([(x, y) for x in midpoint_axis for y in midpoint_axis], dtype=np.float64)
    random = np.random.Generator(np.random.PCG64(seed)).uniform(-0.01, 0.01, size=(128, 2))
    old25: list[tuple[float, float]] = [(0.0, 0.0)]
    for radius in (1.0e-4, 1.0e-3, 1.0e-2):
        old25.extend(((radius, 0.0), (-radius, 0.0), (0.0, radius), (0.0, -radius)))
        old25.extend(((radius, radius), (radius, -radius), (-radius, radius), (-radius, -radius)))
    return {"grid21": grid, "midpoint20": midpoint, "pcg64_128": random,
            "legacy25_diagnostic": np.asarray(old25, dtype=np.float64)}


def channel_metric_and_witness(target: np.ndarray, actual: np.ndarray) -> dict[str, object]:
    metrics = m8.unitary_channel_metrics(target, actual)
    relative = target.conj().T @ actual
    eigenvalues, eigenvectors = np.linalg.eig(relative)
    phases = np.mod(np.angle(eigenvalues), 2.0 * np.pi)
    order = np.argsort(phases)
    phases = phases[order]
    eigenvectors = eigenvectors[:, order]
    wrapped = np.concatenate((phases, phases[:1] + 2.0 * np.pi))
    gaps = np.diff(wrapped)
    cut = int(np.argmax(gaps))
    start_index = (cut + 1) % len(phases)
    unwrapped = np.mod(phases - phases[start_index], 2.0 * np.pi)
    low = int(np.argmin(unwrapped))
    high = int(np.argmax(unwrapped))
    psi = eigenvectors[:, low] + eigenvectors[:, high]
    psi = psi / np.linalg.norm(psi)
    ideal_output = target @ psi
    actual_output = actual @ psi
    overlap = np.vdot(ideal_output, actual_output)
    orthogonal = actual_output - overlap * ideal_output
    witness_distance = float(np.linalg.norm(orthogonal))
    return {
        **metrics,
        "witness_trace_distance": witness_distance,
        "witness_gap_to_half_diamond": float(abs(witness_distance - metrics["half_diamond"])),
        "witness_state_real": psi.real.tolist(),
        "witness_state_imag": psi.imag.tolist(),
        "extreme_eigenphase_indices_in_sorted_order": [low, high],
    }


def evaluate_points_full8(
    params: Sequence[float] | np.ndarray,
    source_target_full8: np.ndarray,
    points: np.ndarray,
) -> tuple[dict[str, object], np.ndarray]:
    """Evaluate every point by direct complete 8x8 Pauli propagation."""
    modules = as_module_parameters(params)
    point_array = np.asarray(points, dtype=np.float64)
    distances = np.empty(len(point_array), dtype=np.float64)
    worst_actual: np.ndarray | None = None
    worst_index = -1
    for index, (epsilon_rotation, epsilon_twist) in enumerate(point_array):
        evaluations = [
            m8.module_propagator(module, float(epsilon_rotation), float(epsilon_twist),
                                 with_error_derivatives=False)
            for module in modules
        ]
        actual, _, _ = m8.compose_full8(evaluations)
        distance = m8.unitary_channel_metrics(source_target_full8, actual)["half_diamond"]
        distances[index] = distance
        if worst_index < 0 or distance > distances[worst_index]:
            worst_index = index
            worst_actual = actual
    assert worst_actual is not None
    witness = channel_metric_and_witness(source_target_full8, worst_actual)
    return {
        "point_count": int(len(point_array)),
        "max_half_diamond": float(distances[worst_index]),
        "worst_index": int(worst_index),
        "worst_point": point_array[worst_index].tolist(),
        "worst_metrics_and_input_state_witness": witness,
        "propagation": "direct FULL8 Pauli-tensor 8x8; one common global phase only",
    }, distances


def resource_record(params: Sequence[float] | np.ndarray) -> dict[str, object]:
    modules = as_module_parameters(params)
    a_theta = float(np.sum(np.abs(modules[:, :15])))
    a_chi = float(np.sum(np.abs(modules[:, 29:44])))
    return {
        "a_theta": a_theta,
        "a_chi_old_hzz": a_chi,
        "dimensionless_gmax_time": {
            str(eta): a_theta / eta + a_chi for eta in (1, 10, 100)
        },
        "layer_count": 45,
        "nonzero_segment_count": 90,
        "internal_transition_count": 89,
        "conditional_model": "T=A_theta/Omega_max+A_chi/g_max; eta=Omega_max/g_max; zero extra overhead",
    }


__all__ = [
    "AugmentedEvaluation", "MODULE_COUNT", "MODULE_LABELS", "TOTAL_PARAMETER_COUNT",
    "Z_DIRECTION", "as_module_parameters", "augmented_value_and_jacobian",
    "block_module_constraints", "candidate_audit", "channel_metric_and_witness",
    "charged_signs", "evaluate_points_full8", "phase_aligned_frobenius",
    "resource_record", "target_local_chart", "validation_point_sets",
]
