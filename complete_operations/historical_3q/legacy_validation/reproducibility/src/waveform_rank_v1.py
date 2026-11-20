"""Analytic positive-width waveform Jacobian for the registered N=3 task.

The frozen waveform framework propagates ``U``, its two calibration-error
variations, and nominal control variations.  This successor adds the mixed
control/error variations required to differentiate the complete error
generators ``K_R`` and ``K_T``.  The primary path is the Schur 4+2
representation; a direct FULL8 path is available for bounded independent
audits, but is not used by the production rank runner.

All derivatives are local, binary64 diagnostics at one positive waveform
width.  This module does not provide an interval certificate or an exact
manifold proof.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm, expm_frechet

import manifold_finite_fiber_v1 as ff
import manifold_full8_v1 as m8
import waveform_framework_v1 as wf


PARAMETER_COUNT = wf.PARAMETER_COUNT
TOTAL_PARAMETER_COUNT = wf.TOTAL_PARAMETER_COUNT
MODULE_COUNT = wf.MODULE_COUNT
FULL41_COUNT = 41
INDEPENDENT_COUNT = 37
TARGET_COUNT = 15
GRAPH_COUNT = 6
RANK_THRESHOLDS = (1.0e-6, 1.0e-8, 1.0e-10, 1.0e-12)


@dataclass(frozen=True)
class MixedBlockPropagation:
    """Nominal, error, control, and mixed variations in one invariant block."""

    unitary: np.ndarray
    d_error_rotation: np.ndarray
    d_error_twist: np.ndarray
    k_error_rotation: np.ndarray
    k_error_twist: np.ndarray
    d_control: np.ndarray
    d_control_error_rotation: np.ndarray
    d_control_error_twist: np.ndarray
    d_k_error_rotation: np.ndarray
    d_k_error_twist: np.ndarray
    transition_count: int
    plateau_count: int
    nfev: int


@dataclass(frozen=True)
class MixedModulePropagation:
    """One waveform module in either Schur or direct FULL8 representation."""

    representation: str
    visible: MixedBlockPropagation | None
    hidden: MixedBlockPropagation | None
    full8: MixedBlockPropagation | None


@dataclass(frozen=True)
class DifferentiatedModuleChart:
    """Complete 41-row chart and the justified 37-row subchart."""

    full41: np.ndarray
    jacobian41x44: np.ndarray
    independent37: np.ndarray
    jacobian37x44: np.ndarray
    omitted41: np.ndarray
    omitted_jacobian4x44: np.ndarray
    lambda_hidden: complex
    hidden_trace_abs: float


@dataclass(frozen=True)
class DifferentiatedModuleAnalysis:
    """Block propagation, chart, and phase-quotiented interface gate."""

    propagation: MixedModulePropagation
    visible: MixedBlockPropagation
    hidden: MixedBlockPropagation
    chart: DifferentiatedModuleChart
    interface_gate4: np.ndarray
    interface_gate_derivatives44x4x4: np.ndarray


@dataclass(frozen=True)
class WaveformRankEvaluation:
    """All local matrices needed for the registered numerical rank diagnosis."""

    analyses: tuple[DifferentiatedModuleAnalysis, ...]
    r111: np.ndarray
    target_value15: np.ndarray
    graph_value6: np.ndarray
    r126: np.ndarray
    value132: np.ndarray
    jacobian111x132: np.ndarray
    target_jacobian15x132: np.ndarray
    jacobian126x132: np.ndarray
    graph_jacobian6x132: np.ndarray
    jacobian132x132: np.ndarray
    target_gate4: np.ndarray
    target_lambdas: np.ndarray
    target_phase_aligned_fro: float
    target_aligning_phase: complex


def _hamiltonian_parts(
    field: np.ndarray, operators: wf.OperatorBlock
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    omega_x, omega_y, twist = (float(value) for value in field)
    rotation = omega_x * operators.jx + omega_y * operators.jy
    twist_part = twist * operators.hzz
    return rotation + twist_part, rotation, twist_part


def _control_hamiltonian_parts(
    field_derivatives: np.ndarray, operators: wf.OperatorBlock
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if field_derivatives.shape != (PARAMETER_COUNT, 3):
        raise ValueError("field derivatives must have shape (44,3)")
    rotation = (
        field_derivatives[:, 0, None, None] * operators.jx[None, :, :]
        + field_derivatives[:, 1, None, None] * operators.jy[None, :, :]
    )
    twist = field_derivatives[:, 2, None, None] * operators.hzz[None, :, :]
    return rotation + twist, rotation, twist


def _augmented_generator(
    total: np.ndarray, rotation: np.ndarray, twist: np.ndarray
) -> np.ndarray:
    """Generator for vertically stacked ``(U,U_R,U_T)``."""

    dimension = total.shape[0]
    a = -1.0j * total
    a_rotation = -1.0j * rotation
    a_twist = -1.0j * twist
    augmented = np.zeros((3 * dimension, 3 * dimension), dtype=np.complex128)
    for index in range(3):
        start = index * dimension
        augmented[start : start + dimension, start : start + dimension] = a
    augmented[dimension : 2 * dimension, :dimension] = a_rotation
    augmented[2 * dimension : 3 * dimension, :dimension] = a_twist
    return augmented


def _augmented_control_generator(
    total: np.ndarray, rotation: np.ndarray, twist: np.ndarray
) -> np.ndarray:
    """Control derivative of the augmented ``(U,U_R,U_T)`` generator."""

    return _augmented_generator(total, rotation, twist)


def _constant_update(
    state: np.ndarray,
    field: np.ndarray,
    field_derivatives: np.ndarray,
    operators: wf.OperatorBlock,
    duration: float,
) -> np.ndarray:
    """Exactly update all base and mixed variations on one plateau.

    If ``X=(U,U_R,U_T)``, then ``X'=M X`` and every control derivative obeys
    ``X_p'=M X_p+M_p X``.  The Frechet derivative of ``exp(M dt)`` therefore
    gives the exact constant-Hamiltonian update without control finite
    differences.
    """

    dimension = operators.dimension
    expected = 3 + 3 * PARAMETER_COUNT
    if state.shape != (expected, dimension, dimension):
        raise ValueError("mixed state has the wrong plateau shape")
    total, rotation, twist = _hamiltonian_parts(field, operators)
    hp, hp_rotation, hp_twist = _control_hamiltonian_parts(
        field_derivatives, operators
    )
    generator = _augmented_generator(total, rotation, twist)
    scaled = generator * duration
    propagator = expm(scaled)
    old = state.copy()
    old_base = old[:3].reshape(3 * dimension, dimension)
    updated = old.copy()
    updated[:3] = (propagator @ old_base).reshape(3, dimension, dimension)

    old_control_triplets = np.stack(
        (
            old[3 : 3 + PARAMETER_COUNT],
            old[3 + PARAMETER_COUNT : 3 + 2 * PARAMETER_COUNT],
            old[3 + 2 * PARAMETER_COUNT : 3 + 3 * PARAMETER_COUNT],
        ),
        axis=1,
    ).reshape(PARAMETER_COUNT, 3 * dimension, dimension)
    new_control_triplets = np.einsum(
        "ab,qbc->qac", propagator, old_control_triplets
    )
    for coordinate in range(PARAMETER_COUNT):
        control_generator = _augmented_control_generator(
            hp[coordinate], hp_rotation[coordinate], hp_twist[coordinate]
        )
        frechet = expm_frechet(
            scaled, control_generator * duration, compute_expm=False
        )
        new_control_triplets[coordinate] += frechet @ old_base
    triplets = new_control_triplets.reshape(
        PARAMETER_COUNT, 3, dimension, dimension
    )
    updated[3 : 3 + PARAMETER_COUNT] = triplets[:, 0]
    updated[3 + PARAMETER_COUNT : 3 + 2 * PARAMETER_COUNT] = triplets[:, 1]
    updated[3 + 2 * PARAMETER_COUNT : 3 + 3 * PARAMETER_COUNT] = triplets[:, 2]
    return updated


def _transition_rhs(
    time: float,
    flat_state: np.ndarray,
    schedule: wf.ModuleSchedule,
    raw_fields: np.ndarray,
    raw_field_derivatives: np.ndarray,
    operators: wf.OperatorBlock,
) -> np.ndarray:
    """Analytic transition RHS including both mixed calibration variations."""

    dimension = operators.dimension
    state = flat_state.reshape(3 + 3 * PARAMETER_COUNT, dimension, dimension)
    field = schedule.convolved_field_from_raw(time, raw_fields)
    field_derivatives = schedule.convolved_derivatives_from_raw(
        time, raw_field_derivatives
    )
    total, rotation, twist = _hamiltonian_parts(field, operators)
    hp, hp_rotation, hp_twist = _control_hamiltonian_parts(
        field_derivatives, operators
    )
    a = -1.0j * total
    ar = -1.0j * rotation
    at = -1.0j * twist
    ap = -1.0j * hp
    apr = -1.0j * hp_rotation
    apt = -1.0j * hp_twist

    unitary, error_rotation, error_twist = state[:3]
    control = state[3 : 3 + PARAMETER_COUNT]
    mixed_rotation = state[3 + PARAMETER_COUNT : 3 + 2 * PARAMETER_COUNT]
    mixed_twist = state[3 + 2 * PARAMETER_COUNT :]
    derivative = np.empty_like(state)
    derivative[0] = a @ unitary
    derivative[1] = a @ error_rotation + ar @ unitary
    derivative[2] = a @ error_twist + at @ unitary
    derivative[3 : 3 + PARAMETER_COUNT] = (
        np.einsum("ab,qbc->qac", a, control)
        + np.einsum("qab,bc->qac", ap, unitary)
    )
    derivative[3 + PARAMETER_COUNT : 3 + 2 * PARAMETER_COUNT] = (
        np.einsum("ab,qbc->qac", a, mixed_rotation)
        + np.einsum("qab,bc->qac", ap, error_rotation)
        + np.einsum("ab,qbc->qac", ar, control)
        + np.einsum("qab,bc->qac", apr, unitary)
    )
    derivative[3 + 2 * PARAMETER_COUNT :] = (
        np.einsum("ab,qbc->qac", a, mixed_twist)
        + np.einsum("qab,bc->qac", ap, error_twist)
        + np.einsum("ab,qbc->qac", at, control)
        + np.einsum("qab,bc->qac", apt, unitary)
    )
    return derivative.reshape(-1)


def _finish_block(
    state: np.ndarray, transition_count: int, plateau_count: int, nfev: int
) -> MixedBlockPropagation:
    unitary, error_rotation, error_twist = state[:3]
    control = state[3 : 3 + PARAMETER_COUNT]
    mixed_rotation = state[3 + PARAMETER_COUNT : 3 + 2 * PARAMETER_COUNT]
    mixed_twist = state[3 + 2 * PARAMETER_COUNT :]
    k_rotation = 1.0j * unitary.conj().T @ error_rotation
    k_twist = 1.0j * unitary.conj().T @ error_twist
    d_k_rotation = 1.0j * (
        np.einsum("qba,bc->qac", control.conj(), error_rotation)
        + np.einsum("ab,qbc->qac", unitary.conj().T, mixed_rotation)
    )
    d_k_twist = 1.0j * (
        np.einsum("qba,bc->qac", control.conj(), error_twist)
        + np.einsum("ab,qbc->qac", unitary.conj().T, mixed_twist)
    )
    k_rotation = 0.5 * (k_rotation + k_rotation.conj().T)
    k_twist = 0.5 * (k_twist + k_twist.conj().T)
    d_k_rotation = 0.5 * (
        d_k_rotation + np.swapaxes(d_k_rotation.conj(), 1, 2)
    )
    d_k_twist = 0.5 * (
        d_k_twist + np.swapaxes(d_k_twist.conj(), 1, 2)
    )
    return MixedBlockPropagation(
        unitary=unitary,
        d_error_rotation=error_rotation,
        d_error_twist=error_twist,
        k_error_rotation=k_rotation,
        k_error_twist=k_twist,
        d_control=control,
        d_control_error_rotation=mixed_rotation,
        d_control_error_twist=mixed_twist,
        d_k_error_rotation=d_k_rotation,
        d_k_error_twist=d_k_twist,
        transition_count=transition_count,
        plateau_count=plateau_count,
        nfev=nfev,
    )


def integrate_mixed_block(
    schedule: wf.ModuleSchedule,
    module_params: Sequence[float],
    operators: wf.OperatorBlock,
    settings: wf.IntegrationSettings,
) -> MixedBlockPropagation:
    """Propagate one block and all 44 control/error mixed variations."""

    parameters = np.asarray(module_params, dtype=np.float64)
    if parameters.shape != (PARAMETER_COUNT,):
        raise ValueError("one module requires 44 parameters")
    raw_fields = schedule.raw_fields(parameters)
    raw_derivatives = schedule.raw_field_derivatives(parameters)
    dimension = operators.dimension
    state = np.zeros(
        (3 + 3 * PARAMETER_COUNT, dimension, dimension), dtype=np.complex128
    )
    state[0] = np.eye(dimension, dtype=np.complex128)
    transitions = 0
    plateaus = 0
    nfev = 0
    for kind, slot_or_boundary, left, right in schedule.chronological_intervals():
        if kind == "plateau":
            state = _constant_update(
                state,
                raw_fields[slot_or_boundary],
                raw_derivatives[:, slot_or_boundary, :],
                operators,
                right - left,
            )
            plateaus += 1
            continue
        solution = solve_ivp(
            _transition_rhs,
            (left, right),
            state.reshape(-1),
            args=(schedule, raw_fields, raw_derivatives, operators),
            method=settings.method,
            rtol=settings.rtol,
            atol=settings.atol,
            max_step=settings.max_step(schedule.lambda_width),
        )
        if not solution.success:
            raise RuntimeError(
                f"{operators.name} mixed transition {slot_or_boundary} failed: "
                f"{solution.message}"
            )
        state = solution.y[:, -1].reshape(
            3 + 3 * PARAMETER_COUNT, dimension, dimension
        )
        transitions += 1
        nfev += int(solution.nfev)
    return _finish_block(state, transitions, plateaus, nfev)


def evaluate_mixed_module(
    schedule: wf.ModuleSchedule,
    module_params: Sequence[float],
    settings: wf.IntegrationSettings,
    representation: Literal["schur", "full8"] = "schur",
) -> MixedModulePropagation:
    """Evaluate one module in a declared representation."""

    if representation == "full8":
        full = integrate_mixed_block(
            schedule, module_params, wf.operator_block("full8"), settings
        )
        return MixedModulePropagation("full8", None, None, full)
    if representation != "schur":
        raise ValueError("representation must be schur or full8")
    visible = integrate_mixed_block(
        schedule, module_params, wf.operator_block("visible"), settings
    )
    hidden = integrate_mixed_block(
        schedule, module_params, wf.operator_block("hidden"), settings
    )
    return MixedModulePropagation("schur", visible, hidden, None)


def _project_block(
    full: MixedBlockPropagation, isometry: np.ndarray
) -> MixedBlockPropagation:
    def project(array: np.ndarray) -> np.ndarray:
        if array.ndim == 2:
            return m8.block(array, isometry)
        return np.asarray([m8.block(item, isometry) for item in array])

    unitary = project(full.unitary)
    error_rotation = project(full.d_error_rotation)
    error_twist = project(full.d_error_twist)
    control = project(full.d_control)
    mixed_rotation = project(full.d_control_error_rotation)
    mixed_twist = project(full.d_control_error_twist)
    state = np.concatenate(
        (
            np.asarray((unitary, error_rotation, error_twist)),
            control,
            mixed_rotation,
            mixed_twist,
        ),
        axis=0,
    )
    return _finish_block(
        state, full.transition_count, full.plateau_count, full.nfev
    )


def module_blocks(
    propagation: MixedModulePropagation,
) -> tuple[MixedBlockPropagation, MixedBlockPropagation]:
    if propagation.representation == "schur":
        if propagation.visible is None or propagation.hidden is None:
            raise ValueError("Schur propagation is missing a block")
        return propagation.visible, propagation.hidden
    if propagation.full8 is None:
        raise ValueError("FULL8 propagation is missing its block")
    return (
        _project_block(propagation.full8, m8.VISIBLE_ISOMETRY),
        _project_block(propagation.full8, m8.HIDDEN_ISOMETRY),
    )


def _coordinate_jacobian(derivatives: np.ndarray) -> np.ndarray:
    return np.asarray(
        [m8.generator_coordinates(item) for item in derivatives],
        dtype=np.float64,
    ).T


def differentiated_module_chart(
    visible: MixedBlockPropagation, hidden: MixedBlockPropagation
) -> DifferentiatedModuleChart:
    """Return the actual 41x44 and independent 37x44 module Jacobians."""

    hidden_unitary = hidden.unitary
    d_hidden = hidden.d_control
    trace_hidden = np.trace(hidden_unitary)
    trace_abs = float(abs(trace_hidden))
    if trace_abs < wf.MIN_HIDDEN_TRACE_ABS:
        raise ValueError("hidden trace chart is singular")
    d_trace = np.trace(d_hidden, axis1=1, axis2=2)
    numerators = np.asarray(
        (
            hidden_unitary[1, 0] + hidden_unitary[0, 1],
            hidden_unitary[1, 0] - hidden_unitary[0, 1],
            hidden_unitary[0, 0] - hidden_unitary[1, 1],
        )
    )
    d_numerators = np.asarray(
        (
            d_hidden[:, 1, 0] + d_hidden[:, 0, 1],
            d_hidden[:, 1, 0] - d_hidden[:, 0, 1],
            d_hidden[:, 0, 0] - d_hidden[:, 1, 1],
        )
    )
    quotients = numerators / trace_hidden
    d_quotients = d_numerators / trace_hidden - (
        numerators[:, None] * d_trace[None, :] / (trace_hidden * trace_hidden)
    )
    values: list[float] = [
        float(-quotients[0].imag),
        float(quotients[1].real),
        float(-quotients[2].imag),
    ]
    rows: list[np.ndarray] = [
        -d_quotients[0].imag,
        d_quotients[1].real,
        -d_quotients[2].imag,
    ]
    for visible_k, hidden_k, visible_dk, hidden_dk in (
        (
            visible.k_error_rotation,
            hidden.k_error_rotation,
            visible.d_k_error_rotation,
            hidden.d_k_error_rotation,
        ),
        (
            visible.k_error_twist,
            hidden.k_error_twist,
            visible.d_k_error_twist,
            hidden.d_k_error_twist,
        ),
    ):
        visible_coordinates = m8.generator_coordinates(visible_k)
        hidden_coordinates = m8.generator_coordinates(hidden_k)
        visible_jacobian = _coordinate_jacobian(visible_dk)
        hidden_jacobian = _coordinate_jacobian(hidden_dk)
        values.extend(visible_coordinates.tolist())
        values.extend(hidden_coordinates[[0, 2, 3]].tolist())
        rows.extend(visible_jacobian)
        rows.extend(hidden_jacobian[[0, 2, 3]])
    full = np.asarray(values, dtype=np.float64)
    jacobian = np.asarray(rows, dtype=np.float64)
    if full.shape != (FULL41_COUNT,) or jacobian.shape != (
        FULL41_COUNT,
        PARAMETER_COUNT,
    ):
        raise AssertionError(
            f"wrong differentiated chart shapes {full.shape}, {jacobian.shape}"
        )
    independent = wf.INDEPENDENT_37_INDICES
    omitted = wf.OMITTED_41_INDICES
    return DifferentiatedModuleChart(
        full41=full,
        jacobian41x44=jacobian,
        independent37=full[independent],
        jacobian37x44=jacobian[independent],
        omitted41=full[omitted],
        omitted_jacobian4x44=jacobian[omitted],
        lambda_hidden=complex(trace_hidden / 2.0),
        hidden_trace_abs=trace_abs,
    )


def interface_gate_and_derivatives(
    visible: MixedBlockPropagation, hidden: MixedBlockPropagation
) -> tuple[np.ndarray, np.ndarray, complex]:
    """Return ``V/lambda`` and its phase-quotiented control derivatives."""

    scalar = complex(np.trace(hidden.unitary) / 2.0)
    if abs(2.0 * scalar) < wf.MIN_HIDDEN_TRACE_ABS:
        raise ValueError("hidden trace chart is singular")
    d_scalar = np.trace(hidden.d_control, axis1=1, axis2=2) / 2.0
    gate = visible.unitary / scalar
    derivatives = visible.d_control / scalar - (
        visible.unitary[None, :, :]
        * d_scalar[:, None, None]
        / (scalar * scalar)
    )
    return gate, derivatives, scalar


def analyze_module(
    propagation: MixedModulePropagation,
) -> DifferentiatedModuleAnalysis:
    visible, hidden = module_blocks(propagation)
    chart = differentiated_module_chart(visible, hidden)
    gate, derivatives, _ = interface_gate_and_derivatives(visible, hidden)
    return DifferentiatedModuleAnalysis(
        propagation=propagation,
        visible=visible,
        hidden=hidden,
        chart=chart,
        interface_gate4=gate,
        interface_gate_derivatives44x4x4=derivatives,
    )


def target_chart_and_jacobian(
    analyses: Sequence[DifferentiatedModuleAnalysis],
    source_target_gate4: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the 15-coordinate target chart and its 15x132 Jacobian."""

    if len(analyses) != MODULE_COUNT:
        raise ValueError("expected three module analyses")
    source = np.asarray(source_target_gate4, dtype=np.complex128)
    if source.shape != (4, 4):
        raise ValueError("source target gate must have shape (4,4)")
    gates = [item.interface_gate4 for item in analyses]
    local = [item.interface_gate_derivatives44x4x4 for item in analyses]
    target = gates[0] @ gates[1] @ gates[2]
    relative = source.conj().T @ target
    hermitian = 0.5j * (relative - relative.conj().T)
    value = m8.su4_coordinates(hermitian)
    d_target = np.zeros(
        (TOTAL_PARAMETER_COUNT, 4, 4), dtype=np.complex128
    )
    d_target[0:44] = np.einsum(
        "qab,bc,cd->qad", local[0], gates[1], gates[2]
    )
    d_target[44:88] = np.einsum(
        "ab,qbc,cd->qad", gates[0], local[1], gates[2]
    )
    d_target[88:132] = np.einsum(
        "ab,bc,qcd->qad", gates[0], gates[1], local[2]
    )
    d_relative = np.einsum("ab,qbc->qac", source.conj().T, d_target)
    d_hermitian = 0.5j * (
        d_relative - np.swapaxes(d_relative.conj(), 1, 2)
    )
    jacobian = np.asarray(
        [m8.su4_coordinates(item) for item in d_hermitian], dtype=np.float64
    ).T
    lambdas = np.asarray(
        [item.chart.lambda_hidden for item in analyses], dtype=np.complex128
    )
    if value.shape != (TARGET_COUNT,) or jacobian.shape != (
        TARGET_COUNT,
        TOTAL_PARAMETER_COUNT,
    ):
        raise AssertionError("wrong target chart/Jacobian shape")
    return value, jacobian, target, lambdas


def evaluate_rank_matrices(
    model: wf.FrozenWaveformModel,
    params: Sequence[float] | np.ndarray,
    settings: wf.IntegrationSettings,
    representation: Literal["schur", "full8"] = "schur",
    h: float = wf.H_VALUE,
) -> WaveformRankEvaluation:
    """Evaluate the registered local positive-width matrices once."""

    if model.fiber_basis is None:
        raise ValueError("fiber basis is required for graph augmentation")
    modules = ff.as_module_parameters(params)
    propagations = tuple(
        evaluate_mixed_module(
            model.schedules[index], modules[index], settings, representation
        )
        for index in range(MODULE_COUNT)
    )
    analyses = tuple(analyze_module(item) for item in propagations)
    r111 = np.concatenate([item.chart.independent37 for item in analyses])
    jacobian111 = np.zeros(
        (MODULE_COUNT * INDEPENDENT_COUNT, TOTAL_PARAMETER_COUNT),
        dtype=np.float64,
    )
    for module, analysis in enumerate(analyses):
        row = module * INDEPENDENT_COUNT
        column = module * PARAMETER_COUNT
        jacobian111[
            row : row + INDEPENDENT_COUNT,
            column : column + PARAMETER_COUNT,
        ] = analysis.chart.jacobian37x44
    target_value, target_jacobian, target_gate, lambdas = (
        target_chart_and_jacobian(analyses, model.source_target_gate4)
    )
    graph_value = model.fiber_basis.T @ (
        modules.reshape(-1) - model.source_params.reshape(-1)
    ) - float(h) * model.z
    graph_jacobian = model.fiber_basis.T.copy()
    r126 = np.concatenate((r111, target_value))
    value132 = np.concatenate((r126, graph_value))
    jacobian126 = np.vstack((jacobian111, target_jacobian))
    jacobian132 = np.vstack((jacobian126, graph_jacobian))
    target_fro, target_phase = wf.phase_aligned_frobenius(
        model.source_target_gate4, target_gate
    )
    expected_shapes = (
        (r111.shape, (111,)),
        (r126.shape, (126,)),
        (value132.shape, (132,)),
        (jacobian111.shape, (111, 132)),
        (target_jacobian.shape, (15, 132)),
        (jacobian126.shape, (126, 132)),
        (graph_jacobian.shape, (6, 132)),
        (jacobian132.shape, (132, 132)),
    )
    for actual, expected in expected_shapes:
        if actual != expected:
            raise AssertionError(f"wrong rank-evaluation shape {actual}, expected {expected}")
    return WaveformRankEvaluation(
        analyses=analyses,
        r111=r111,
        target_value15=target_value,
        graph_value6=graph_value,
        r126=r126,
        value132=value132,
        jacobian111x132=jacobian111,
        target_jacobian15x132=target_jacobian,
        jacobian126x132=jacobian126,
        graph_jacobian6x132=graph_jacobian,
        jacobian132x132=jacobian132,
        target_gate4=target_gate,
        target_lambdas=lambdas,
        target_phase_aligned_fro=target_fro,
        target_aligning_phase=target_phase,
    )


def singular_value_report(
    matrix: np.ndarray, thresholds: Sequence[float] = RANK_THRESHOLDS
) -> dict[str, object]:
    """Report every singular value and threshold-dependent numerical nullity."""

    array = np.asarray(matrix, dtype=np.float64)
    singular_values = np.linalg.svd(array, compute_uv=False)
    minimum_dimension = min(array.shape)
    ranks = {
        f"{threshold:.0e}": int(np.count_nonzero(singular_values > threshold))
        for threshold in thresholds
    }
    return {
        "shape": list(array.shape),
        "singular_values": singular_values.tolist(),
        "ranks": ranks,
        "nullities": {
            key: int(array.shape[1] - rank) for key, rank in ranks.items()
        },
        "exact_zero_entry_count": int(np.count_nonzero(array == 0.0)),
        "nonfinite_entry_count": int(array.size - np.count_nonzero(np.isfinite(array))),
        "minimum_dimension": minimum_dimension,
        "spectral_norm": float(singular_values[0]) if singular_values.size else 0.0,
        "minimum_reported_singular": (
            float(singular_values[-1]) if singular_values.size else 0.0
        ),
    }


def module_null_basis(
    evaluation: WaveformRankEvaluation, threshold: float = 1.0e-8
) -> np.ndarray:
    """Return the block-local numerical null basis in the 132 coordinates."""

    bases: list[np.ndarray] = []
    for analysis in evaluation.analyses:
        _, singular_values, vh = np.linalg.svd(
            analysis.chart.jacobian37x44, full_matrices=True
        )
        rank = int(np.count_nonzero(singular_values > threshold))
        bases.append(vh[rank:].conj().T)
    width = sum(item.shape[1] for item in bases)
    block_basis = np.zeros((TOTAL_PARAMETER_COUNT, width), dtype=np.float64)
    offset = 0
    for module, basis in enumerate(bases):
        block_basis[
            module * PARAMETER_COUNT : (module + 1) * PARAMETER_COUNT,
            offset : offset + basis.shape[1],
        ] = basis
        offset += basis.shape[1]
    return block_basis


def target_on_module_nullspace(
    evaluation: WaveformRankEvaluation, threshold: float = 1.0e-8
) -> tuple[np.ndarray, np.ndarray]:
    basis = module_null_basis(evaluation, threshold)
    return evaluation.target_jacobian15x132 @ basis, basis


def central_directional_check(
    model: wf.FrozenWaveformModel,
    params: Sequence[float] | np.ndarray,
    directions: np.ndarray,
    step: float,
    settings: wf.IntegrationSettings,
) -> dict[str, np.ndarray]:
    """Optional secondary check using fixed directions, never the primary matrix."""

    modules = ff.as_module_parameters(params)
    flat = modules.reshape(-1)
    direction_array = np.asarray(directions, dtype=np.float64)
    if direction_array.ndim != 2 or direction_array.shape[0] != TOTAL_PARAMETER_COUNT:
        raise ValueError("directions must have shape (132,k)")
    analytic = evaluate_rank_matrices(model, modules, settings).jacobian126x132
    numerical = np.empty((126, direction_array.shape[1]), dtype=np.float64)
    for index in range(direction_array.shape[1]):
        direction = direction_array[:, index]
        plus = ff.as_module_parameters(flat + step * direction)
        minus = ff.as_module_parameters(flat - step * direction)
        plus_value = model.augmented_residual(plus, settings=settings).r126
        minus_value = model.augmented_residual(minus, settings=settings).r126
        numerical[:, index] = (plus_value - minus_value) / (2.0 * step)
    analytic_directional = analytic @ direction_array
    return {
        "analytic": analytic_directional,
        "central": numerical,
        "difference": analytic_directional - numerical,
    }


__all__ = [
    "DifferentiatedModuleAnalysis",
    "DifferentiatedModuleChart",
    "FULL41_COUNT",
    "GRAPH_COUNT",
    "INDEPENDENT_COUNT",
    "MixedBlockPropagation",
    "MixedModulePropagation",
    "RANK_THRESHOLDS",
    "TARGET_COUNT",
    "WaveformRankEvaluation",
    "analyze_module",
    "central_directional_check",
    "differentiated_module_chart",
    "evaluate_mixed_module",
    "evaluate_rank_matrices",
    "integrate_mixed_block",
    "interface_gate_and_derivatives",
    "module_blocks",
    "module_null_basis",
    "singular_value_report",
    "target_chart_and_jacobian",
    "target_on_module_nullspace",
]
