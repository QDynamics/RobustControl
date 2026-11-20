"""Finite target programming on the registered N=3 FULL8 robust manifold.

The target family is fixed before solving.  For the normalized two-qubit
Pauli basis ``B_k = P_k / 2`` (Hilbert--Schmidt norm one), the two targets in
direction ``k`` are

    G_(k,s) = G_source @ exp(-i s r B_k),  s in {-1,+1}.

The complete-space target keeps the accepted hidden branch fixed:

    U_(k,s) = -(V G_(k,s) V^dagger + Q).

Here ``V`` is the registered Dicke-order visible isometry and ``Q`` is the
complete-space complement projector.  The solve uses 111 per-module robust
constraints, 15 desired-target chart coordinates, and the same six source
graph coordinates ``fiber.T @ (p-p_source)=0``.  No finite-error performance
quantity enters the residual or its analytic Jacobian.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

import manifold_finite_fiber_v1 as ff
import manifold_full8_v1 as m8


TARGET_RADIUS = 1.0e-3
TARGET_SIGNS = (-1, 1)
PAULI_LABELS = (
    "IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI",
    "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ",
)


@dataclass(frozen=True)
class TargetSpecification:
    target_index: int
    target_id: str
    direction_index: int
    pauli_label: str
    sign: int
    radius: float
    direction_basis4: np.ndarray
    target_gate4: np.ndarray
    target_full8: np.ndarray
    actual_gate4_frobenius_displacement: float


@dataclass(frozen=True)
class TargetAugmentedEvaluation:
    value: np.ndarray
    jacobian: np.ndarray | None
    module_values: np.ndarray
    target_value: np.ndarray
    graph_value: np.ndarray
    actual_gate4: np.ndarray
    lambdas: np.ndarray


def _relative_target_step(direction: np.ndarray, sign: int, radius: float) -> np.ndarray:
    """Return exp(-i sign radius direction) for a normalized Pauli/2 basis."""
    basis = np.asarray(direction, dtype=np.complex128)
    if basis.shape != (4, 4):
        raise ValueError("target direction must have shape (4,4)")
    if sign not in TARGET_SIGNS:
        raise ValueError("target sign must be -1 or +1")
    # Every registered B=P/2 obeys B^2=I/4, so this is the exact exponential.
    return (
        np.cos(float(radius) / 2.0) * np.eye(4, dtype=np.complex128)
        - 2.0j * float(sign) * np.sin(float(radius) / 2.0) * basis
    )


def lift_gate4_fixed_hidden_minus(target_gate4: np.ndarray) -> np.ndarray:
    """Lift a visible target with the registered composite hidden phase -I."""
    gate = np.asarray(target_gate4, dtype=np.complex128)
    if gate.shape != (4, 4):
        raise ValueError("target gate must have shape (4,4)")
    visible = m8.VISIBLE_ISOMETRY
    return -(visible @ gate @ visible.conj().T + m8.COMPLEMENT_PROJECTOR)


def build_target_registry(
    source_target_gate4: np.ndarray,
    radius: float = TARGET_RADIUS,
) -> list[TargetSpecification]:
    """Build the immutable 15-direction, two-sign target registry."""
    source = np.asarray(source_target_gate4, dtype=np.complex128)
    if source.shape != (4, 4):
        raise ValueError("source target gate must have shape (4,4)")
    basis = m8.normalized_pauli_basis_4()
    if len(basis) != 15 or len(PAULI_LABELS) != 15:
        raise AssertionError("registered SU(4) Pauli basis must have 15 elements")
    targets: list[TargetSpecification] = []
    for direction_index, (label, direction) in enumerate(zip(PAULI_LABELS, basis)):
        for sign in TARGET_SIGNS:
            step = _relative_target_step(direction, sign, radius)
            gate = source @ step
            sign_label = "minus" if sign < 0 else "plus"
            targets.append(TargetSpecification(
                target_index=len(targets),
                target_id=f"{direction_index:02d}_{label}_{sign_label}",
                direction_index=direction_index,
                pauli_label=label,
                sign=sign,
                radius=float(radius),
                direction_basis4=np.asarray(direction, dtype=np.complex128),
                target_gate4=gate,
                target_full8=lift_gate4_fixed_hidden_minus(gate),
                actual_gate4_frobenius_displacement=float(np.linalg.norm(gate - source)),
            ))
    if len(targets) != 30:
        raise AssertionError("target registry must contain exactly 30 targets")
    return targets


def augmented_value_and_jacobian(
    params: Sequence[float] | np.ndarray,
    source_params: Sequence[float] | np.ndarray,
    desired_target_gate4: np.ndarray,
    fiber_basis132x6: np.ndarray,
    with_jacobian: bool = True,
) -> TargetAugmentedEvaluation:
    """Evaluate the fixed-target square chart with source graph coordinate h=0."""
    modules = ff.as_module_parameters(params)
    source_modules = ff.as_module_parameters(source_params)
    fiber = np.asarray(fiber_basis132x6, dtype=np.float64)
    if fiber.shape != (132, 6):
        raise ValueError("fiber basis must have shape (132,6)")

    module_value, module_jacobian = ff.block_module_constraints(modules, with_jacobian)
    target_value, target_jacobian, actual_gate, lambdas = ff.target_local_chart(
        modules, desired_target_gate4, with_jacobian
    )
    graph_value = fiber.T @ (modules.reshape(-1) - source_modules.reshape(-1))
    value = np.concatenate((module_value, target_value, graph_value))
    jacobian: np.ndarray | None = None
    if with_jacobian:
        assert module_jacobian is not None and target_jacobian is not None
        jacobian = np.vstack((module_jacobian, target_jacobian, fiber.T))
    if value.shape != (132,) or (jacobian is not None and jacobian.shape != (132, 132)):
        raise AssertionError("wrong target-programming augmented chart shape")
    return TargetAugmentedEvaluation(
        value=value,
        jacobian=jacobian,
        module_values=module_value,
        target_value=target_value,
        graph_value=graph_value,
        actual_gate4=actual_gate,
        lambdas=lambdas,
    )


def resource_caps(source_params: Sequence[float] | np.ndarray, factor: float = 1.01) -> dict[str, float]:
    source_record = ff.resource_record(source_params)
    return {
        eta: float(factor) * float(value)
        for eta, value in source_record["dimensionless_gmax_time"].items()
    }


def candidate_audit(
    params: Sequence[float] | np.ndarray,
    source_params: Sequence[float] | np.ndarray,
    target: TargetSpecification,
    fiber_basis132x6: np.ndarray,
    time_cap_factor: float = 1.01,
) -> dict[str, object]:
    """Audit one candidate without scanning or optimizing a finite error domain."""
    modules = ff.as_module_parameters(params)
    source = ff.as_module_parameters(source_params)
    augmented = augmented_value_and_jacobian(
        modules, source, target.target_gate4, fiber_basis132x6, with_jacobian=False
    )
    evaluations = [m8.module_propagator(item) for item in modules]
    actual_full8, _, _ = m8.compose_full8(evaluations)
    target_fro, target_phase = ff.phase_aligned_frobenius(target.target_full8, actual_full8)
    nominal_metrics = m8.unitary_channel_metrics(target.target_full8, actual_full8)

    module_records: list[dict[str, object]] = []
    for label, evaluation, lam in zip(ff.MODULE_LABELS, evaluations, augmented.lambdas):
        residual = m8.full8_residuals(evaluation)
        residual.update({
            "module": label,
            "lambda_real": float(lam.real),
            "lambda_imag": float(lam.imag),
            "lambda_distance_to_minus_one": float(abs(lam + 1.0)),
            "lambda_source_branch_minus_one": bool(
                lam.real < 0.0 and abs(lam + 1.0) < 1.0e-8
            ),
        })
        module_records.append(residual)

    max_k_closure = max(
        max(
            float(item["k_rotation_fro"]),
            float(item["k_twist_fro"]),
            float(item["complement_scalar_closure_fro"]),
            float(item["symmetric_to_complement_leakage_fro"]),
            float(item["complement_to_symmetric_leakage_fro"]),
        )
        for item in module_records
    )
    trust_distance = float(np.linalg.norm(modules - source))
    sign_match = bool(np.array_equal(ff.charged_signs(modules), ff.charged_signs(source)))
    resources = ff.resource_record(modules)
    source_resources = ff.resource_record(source)
    caps = resource_caps(source, time_cap_factor)
    time_checks = {
        eta: float(resources["dimensionless_gmax_time"][eta]) <= caps[eta]
        for eta in caps
    }
    checks = {
        "target_full8_phase_aligned_fro_lt_1e-9": target_fro < 1.0e-9,
        "nominal_half_diamond_lt_1e-9": float(nominal_metrics["half_diamond"]) < 1.0e-9,
        "module_full8_k_and_closure_lt_1e-9": max_k_closure < 1.0e-9,
        "module_chart_max_abs_lt_1e-9": float(np.max(np.abs(augmented.module_values))) < 1.0e-9,
        "target_chart_max_abs_lt_1e-9": float(np.max(np.abs(augmented.target_value))) < 1.0e-9,
        "graph_chart_max_abs_lt_5e-11": float(np.max(np.abs(augmented.graph_value))) < 5.0e-11,
        "source_euclidean_distance_le_0p05": trust_distance <= 0.05,
        "charged_theta_chi_signs_unchanged": sign_match,
        "all_module_lambdas_on_source_minus_one_branch": all(
            bool(item["lambda_source_branch_minus_one"]) for item in module_records
        ),
        "all_registered_time_diagnostics_le_1p01_source": all(time_checks.values()),
    }
    return {
        "accepted": bool(all(checks.values())),
        "checks": checks,
        "target_id": target.target_id,
        "target_full8_phase_aligned_fro": target_fro,
        "target_aligning_phase_real": float(target_phase.real),
        "target_aligning_phase_imag": float(target_phase.imag),
        "nominal_target_half_diamond": float(nominal_metrics["half_diamond"]),
        "nominal_target_channel_metrics": nominal_metrics,
        "module_chart_max_abs": float(np.max(np.abs(augmented.module_values))),
        "target_chart_max_abs": float(np.max(np.abs(augmented.target_value))),
        "graph_chart_max_abs": float(np.max(np.abs(augmented.graph_value))),
        "augmented_chart_max_abs": float(np.max(np.abs(augmented.value))),
        "trust_distance": trust_distance,
        "max_module_full8_k_and_closure": float(max_k_closure),
        "resources": resources,
        "source_resources": source_resources,
        "time_cap_factor": float(time_cap_factor),
        "time_caps": caps,
        "time_checks": time_checks,
        "modules": module_records,
    }


__all__ = [
    "PAULI_LABELS", "TARGET_RADIUS", "TARGET_SIGNS", "TargetAugmentedEvaluation",
    "TargetSpecification", "augmented_value_and_jacobian", "build_target_registry",
    "candidate_audit", "lift_gate4_fixed_hidden_minus", "resource_caps",
]
