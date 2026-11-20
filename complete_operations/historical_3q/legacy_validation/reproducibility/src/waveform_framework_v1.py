"""Bounded smooth-waveform framework for the registered N=3 FULL8 task.

This module implements the frozen first-validation protocol in
docs/WAVEFORM_FIRST_VALIDATION_DESIGN_TERRA_20260914_v1.md.  It deliberately
keeps the old rectangle-pulse APIs read-only and uses them only where the
protocol explicitly permits a frozen quasi-Newton preconditioner.

The production representation integrates the spin-3/2 block and one
spin-1/2 block.  A direct tensor-product FULL8 path remains available for
candidate audit; no Schur block is substituted for that audit.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence
import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm, expm_frechet

import manifold_finite_fiber_v1 as ff
import manifold_full8_v1 as m8


MODULE_COUNT = 3
MODULE_LABELS = ff.MODULE_LABELS
PARAMETER_COUNT = m8.PARAMETER_COUNT
TOTAL_PARAMETER_COUNT = MODULE_COUNT * PARAMETER_COUNT
LENGTH = m8.LENGTH

OMEGA_MAX = 10.0
G_MAX = 1.0
TIME_STRETCH = 1.1
LAMBDA_WIDTH = 1.0e-3
KERNEL_ORDER = 6
KERNEL_BETA = math.gamma(0.5) * math.gamma(KERNEL_ORDER + 1) / math.gamma(
    KERNEL_ORDER + 1.5
)
KERNEL_NORMALIZATION = 1.0 / KERNEL_BETA
MIN_HIDDEN_TRACE_ABS = 0.5
LAMBDA_BRANCH_REAL_MAX = -0.5
CORRECTION_RATIO_MIN = 0.9
CORRECTION_RATIO_MAX = 1.1
PHASE_OFFSET_MAX = 0.1
GLOBAL_SOURCE_DISTANCE_MAX = 0.05
H_VALUE = 0.0
ARMIJO_COEFFICIENT = 1.0e-4

# These are written explicitly because their structural justification is part
# of the waveform protocol, rather than an implicit inheritance of m8 rows.
INDEPENDENT_37_INDICES = np.asarray(
    (
        0, 1, 2,
        3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24,
        26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
    ),
    dtype=np.int64,
)
OMITTED_41_INDICES = np.asarray((6, 25, 39, 40), dtype=np.int64)
if tuple(INDEPENDENT_37_INDICES.tolist()) != tuple(m8.INDEPENDENT_ROWS):
    raise RuntimeError("explicit waveform 37-row ordering no longer matches frozen API ordering")


@dataclass(frozen=True)
class IntegrationSettings:
    """Settings for one deterministic waveform propagation."""

    rtol: float
    atol: float
    max_step_divisor: float
    method: str = "DOP853"

    def max_step(self, width: float) -> float:
        if self.max_step_divisor <= 0.0:
            raise ValueError("max_step_divisor must be positive")
        return width / self.max_step_divisor


WORKING_INTEGRATION = IntegrationSettings(
    rtol=2.0e-11, atol=2.0e-13, max_step_divisor=8.0
)
TIGHT_INTEGRATION = IntegrationSettings(
    rtol=2.0e-12, atol=2.0e-14, max_step_divisor=12.0
)


@dataclass(frozen=True)
class OperatorBlock:
    """Hamiltonian matrices for one exact symmetry representation block."""

    name: str
    jx: np.ndarray
    jy: np.ndarray
    hzz: np.ndarray

    @property
    def dimension(self) -> int:
        return int(self.jx.shape[0])


@dataclass(frozen=True)
class ModuleSchedule:
    """Fixed raw-slot schedule for one module."""

    module_index: int
    lambda_width: float
    durations: np.ndarray
    boundaries: np.ndarray
    slot_kinds: tuple[str, ...]
    slot_layers: np.ndarray

    @property
    def raw_start(self) -> float:
        return float(self.boundaries[0])

    @property
    def raw_end(self) -> float:
        return float(self.boundaries[-1])

    @property
    def total_duration(self) -> float:
        return self.raw_end + self.lambda_width

    @property
    def raw_duration(self) -> float:
        return float(np.sum(self.durations))

    def raw_fields(self, module_params: Sequence[float]) -> np.ndarray:
        """Return the 30 raw (Omega_x, Omega_y, g) fields."""

        theta, phi, chi = m8.split_parameters(module_params)
        fields = np.zeros((2 * LENGTH, 3), dtype=np.float64)
        for layer in range(LENGTH):
            twist_index = 2 * layer
            rotation_index = twist_index + 1
            fields[twist_index, 2] = chi[layer] / self.durations[twist_index]
            fields[rotation_index, 0] = (
                theta[layer] * math.cos(phi[layer]) / self.durations[rotation_index]
            )
            fields[rotation_index, 1] = (
                theta[layer] * math.sin(phi[layer]) / self.durations[rotation_index]
            )
        return fields

    def raw_field_derivatives(self, module_params: Sequence[float]) -> np.ndarray:
        """Return d raw_field / d p_q with shape (44, 30, 3)."""

        theta, phi, _ = m8.split_parameters(module_params)
        derivatives = np.zeros((PARAMETER_COUNT, 2 * LENGTH, 3), dtype=np.float64)
        for layer in range(LENGTH):
            twist_index = 2 * layer
            rotation_index = twist_index + 1
            rotation_duration = self.durations[rotation_index]
            twist_duration = self.durations[twist_index]

            derivatives[layer, rotation_index, 0] = math.cos(phi[layer]) / rotation_duration
            derivatives[layer, rotation_index, 1] = math.sin(phi[layer]) / rotation_duration
            if layer > 0:
                phase_index = LENGTH + layer - 1
                derivatives[phase_index, rotation_index, 0] = (
                    -theta[layer] * math.sin(phi[layer]) / rotation_duration
                )
                derivatives[phase_index, rotation_index, 1] = (
                    theta[layer] * math.cos(phi[layer]) / rotation_duration
                )
            chi_index = 2 * LENGTH - 1 + layer
            derivatives[chi_index, twist_index, 2] = 1.0 / twist_duration
        return derivatives

    def _boundary_transition_index(self, time: float) -> int | None:
        width = self.lambda_width
        if time < -1.0e-15 or time > self.total_duration + 1.0e-15:
            return None
        candidate = int(np.searchsorted(self.boundaries, time, side="left"))
        for index in (candidate - 1, candidate):
            if 0 <= index < len(self.boundaries):
                boundary = self.boundaries[index]
                if boundary - width - 1.0e-15 <= time <= boundary + width + 1.0e-15:
                    return index
        return None

    def _slot_index(self, time: float) -> int | None:
        if time < self.raw_start or time >= self.raw_end:
            return None
        index = int(np.searchsorted(self.boundaries, time, side="right") - 1)
        return index if 0 <= index < len(self.durations) else None

    def convolved_field_from_raw(self, time: float, fields: np.ndarray) -> np.ndarray:
        """Evaluate the exact compact-convolution field at one time."""

        if fields.shape != (2 * LENGTH, 3):
            raise ValueError("raw fields must have shape (30,3)")
        transition = self._boundary_transition_index(float(time))
        if transition is not None:
            left = np.zeros(3, dtype=np.float64) if transition == 0 else fields[transition - 1]
            right = (
                np.zeros(3, dtype=np.float64)
                if transition == len(self.boundaries) - 1
                else fields[transition]
            )
            fraction = compact_cdf((time - self.boundaries[transition]) / self.lambda_width)
            return left + (right - left) * fraction
        slot = self._slot_index(float(time))
        return np.zeros(3, dtype=np.float64) if slot is None else fields[slot].copy()

    def convolved_derivatives_from_raw(
        self, time: float, derivatives: np.ndarray
    ) -> np.ndarray:
        """Evaluate d u_lambda(t) / d p, shape (44,3)."""

        if derivatives.shape != (PARAMETER_COUNT, 2 * LENGTH, 3):
            raise ValueError("raw field derivatives must have shape (44,30,3)")
        transition = self._boundary_transition_index(float(time))
        if transition is not None:
            left = (
                np.zeros((PARAMETER_COUNT, 3), dtype=np.float64)
                if transition == 0
                else derivatives[:, transition - 1, :]
            )
            right = (
                np.zeros((PARAMETER_COUNT, 3), dtype=np.float64)
                if transition == len(self.boundaries) - 1
                else derivatives[:, transition, :]
            )
            fraction = compact_cdf((time - self.boundaries[transition]) / self.lambda_width)
            return left + (right - left) * fraction
        slot = self._slot_index(float(time))
        return (
            np.zeros((PARAMETER_COUNT, 3), dtype=np.float64)
            if slot is None
            else derivatives[:, slot, :].copy()
        )

    def chronological_intervals(self) -> Iterable[tuple[str, int, float, float]]:
        """Yield transition and plateau intervals in physical chronological order."""

        width = self.lambda_width
        for boundary_index, boundary in enumerate(self.boundaries):
            left = boundary - width
            right = boundary + width
            yield ("transition", boundary_index, left, right)
            if boundary_index < len(self.durations):
                plateau_left = right
                plateau_right = self.boundaries[boundary_index + 1] - width
                if plateau_right <= plateau_left:
                    raise ValueError("slot is too short for the fixed convolution width")
                yield ("plateau", boundary_index, plateau_left, plateau_right)


@dataclass(frozen=True)
class BlockPropagation:
    """Nominal and first-error derivatives in one invariant block."""

    unitary: np.ndarray
    d_error_rotation: np.ndarray
    d_error_twist: np.ndarray
    k_error_rotation: np.ndarray
    k_error_twist: np.ndarray
    control_derivatives: np.ndarray | None
    transition_count: int
    plateau_count: int
    nfev: int


@dataclass(frozen=True)
class WaveformModulePropagation:
    """A waveform module in either Schur or direct FULL8 representation."""

    representation: str
    visible: BlockPropagation | None
    hidden: BlockPropagation | None
    full8: BlockPropagation | None


@dataclass(frozen=True)
class ModuleConstraintChart:
    """The complete 41-row diagnostic chart and justified 37-row residual."""

    full41: np.ndarray
    independent37: np.ndarray
    closure3: np.ndarray
    lambda_hidden: complex
    hidden_trace_abs: float
    omitted41: np.ndarray
    diagnostics: dict[str, float]


@dataclass(frozen=True)
class ModuleAnalysis:
    """Constraint and target data derived from one propagated module."""

    propagation: WaveformModulePropagation
    visible: BlockPropagation
    hidden: BlockPropagation
    chart: ModuleConstraintChart
    interface_gate4: np.ndarray
    interface_gate_derivatives: np.ndarray | None


@dataclass(frozen=True)
class WaveformAugmentedEvaluation:
    """New waveform residual in the old coordinate ordering."""

    r126: np.ndarray
    graph6: np.ndarray
    value132: np.ndarray
    module_analyses: tuple[ModuleAnalysis, ...]
    target_value15: np.ndarray
    target_gate4: np.ndarray
    target_jacobian15x132: np.ndarray | None
    lambdas: np.ndarray


def compact_kernel(value: float | np.ndarray, width: float = LAMBDA_WIDTH) -> float | np.ndarray:
    """Normalized compact C5 kernel specified by the frozen protocol."""

    array = np.asarray(value, dtype=np.float64)
    ratio = array / width
    out = np.zeros_like(ratio)
    inside = np.abs(ratio) < 1.0
    out[inside] = KERNEL_NORMALIZATION / width * (1.0 - ratio[inside] ** 2) ** KERNEL_ORDER
    return float(out) if np.ndim(value) == 0 else out


def compact_cdf(value: float | np.ndarray) -> float | np.ndarray:
    """Exact polynomial CDF of compact_kernel after rescaling by width."""

    ratio = np.asarray(value, dtype=np.float64)
    out = np.zeros_like(ratio)
    inside = (ratio > -1.0) & (ratio < 1.0)
    if np.any(inside):
        inner = ratio[inside]
        polynomial = np.zeros_like(inner)
        for index in range(KERNEL_ORDER + 1):
            polynomial += (
                (-1.0) ** index
                * math.comb(KERNEL_ORDER, index)
                * inner ** (2 * index + 1)
                / float(2 * index + 1)
            )
        out[inside] = 0.5 + KERNEL_NORMALIZATION * polynomial
    out[ratio >= 1.0] = 1.0
    return float(out) if np.ndim(value) == 0 else out


def operator_block(representation: Literal["full8", "visible", "hidden"]) -> OperatorBlock:
    """Return operators in one declared representation."""

    if representation == "full8":
        return OperatorBlock("full8", m8.JX, m8.JY, m8.HZZ)
    isometry = m8.VISIBLE_ISOMETRY if representation == "visible" else m8.HIDDEN_ISOMETRY
    return OperatorBlock(
        representation,
        m8.block(m8.JX, isometry),
        m8.block(m8.JY, isometry),
        m8.block(m8.HZZ, isometry),
    )


def build_module_schedule(
    source_module_params: Sequence[float], module_index: int, width: float = LAMBDA_WIDTH
) -> ModuleSchedule:
    """Build the source-fixed 30-slot schedule for one module."""

    if width <= 0.0:
        raise ValueError("waveform width must be positive")
    theta, _, chi = m8.split_parameters(source_module_params)
    durations: list[float] = []
    kinds: list[str] = []
    layers: list[int] = []
    for layer in range(LENGTH):
        twist_duration = TIME_STRETCH * abs(chi[layer]) / G_MAX
        rotation_duration = TIME_STRETCH * abs(theta[layer]) / OMEGA_MAX
        if twist_duration <= 0.0 or rotation_duration <= 0.0:
            raise ValueError("source contains a zero area, so a fixed slot cannot be formed")
        durations.extend((twist_duration, rotation_duration))
        kinds.extend(("twist", "rotation"))
        layers.extend((layer, layer))
    duration_array = np.asarray(durations, dtype=np.float64)
    if np.min(duration_array) <= 2.0 * width:
        raise ValueError("fixed width overlaps more than two raw controls")
    boundaries = width + np.concatenate((np.zeros(1), np.cumsum(duration_array)))
    return ModuleSchedule(
        module_index=module_index,
        lambda_width=float(width),
        durations=duration_array,
        boundaries=boundaries,
        slot_kinds=tuple(kinds),
        slot_layers=np.asarray(layers, dtype=np.int64),
    )


def _hamiltonian_parts(field: np.ndarray, operators: OperatorBlock) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    omega_x, omega_y, twist = (float(value) for value in field)
    rotation = omega_x * operators.jx + omega_y * operators.jy
    twist_part = twist * operators.hzz
    return rotation + twist_part, rotation, twist_part


def _field_derivative_hamiltonians(
    field_derivatives: np.ndarray, operators: OperatorBlock
) -> np.ndarray:
    return (
        field_derivatives[:, 0, None, None] * operators.jx[None, :, :]
        + field_derivatives[:, 1, None, None] * operators.jy[None, :, :]
        + field_derivatives[:, 2, None, None] * operators.hzz[None, :, :]
    )


def _core_constant_update(
    state: np.ndarray,
    total_hamiltonian: np.ndarray,
    rotation_hamiltonian: np.ndarray,
    twist_hamiltonian: np.ndarray,
    duration: float,
) -> np.ndarray:
    """Exactly update U, U_R, and U_T during one constant plateau."""

    dimension = total_hamiltonian.shape[0]
    a = -1.0j * total_hamiltonian
    a_rotation = -1.0j * rotation_hamiltonian
    a_twist = -1.0j * twist_hamiltonian
    augmented = np.zeros((3 * dimension, 3 * dimension), dtype=np.complex128)
    augmented[:dimension, :dimension] = a
    augmented[dimension : 2 * dimension, :dimension] = a_rotation
    augmented[dimension : 2 * dimension, dimension : 2 * dimension] = a
    augmented[2 * dimension :, :dimension] = a_twist
    augmented[2 * dimension :, 2 * dimension :] = a
    return (expm(augmented * duration) @ state[:3].reshape(3 * dimension, dimension)).reshape(
        3, dimension, dimension
    )


def _constant_update(
    state: np.ndarray,
    field: np.ndarray,
    field_derivatives: np.ndarray | None,
    operators: OperatorBlock,
    duration: float,
) -> np.ndarray:
    """Exactly update all requested first derivatives on a constant plateau."""

    total, rotation, twist = _hamiltonian_parts(field, operators)
    old = state.copy()
    updated = old.copy()
    updated[:3] = _core_constant_update(old, total, rotation, twist, duration)
    if field_derivatives is None:
        return updated

    a = -1.0j * total
    propagator = expm(a * duration)
    control_hamiltonians = _field_derivative_hamiltonians(field_derivatives, operators)
    for coordinate, hamiltonian in enumerate(control_hamiltonians):
        frechet = expm_frechet(
            a * duration, -1.0j * hamiltonian * duration, compute_expm=False
        )
        updated[3 + coordinate] = (
            propagator @ old[3 + coordinate] + frechet @ old[0]
        )
    return updated


def _transition_rhs(
    time: float,
    flat_state: np.ndarray,
    schedule: ModuleSchedule,
    raw_fields: np.ndarray,
    raw_field_derivatives: np.ndarray | None,
    operators: OperatorBlock,
) -> np.ndarray:
    """Analytic RHS for a compact-convolution transition interval."""

    dimension = operators.dimension
    derivative_count = 0 if raw_field_derivatives is None else PARAMETER_COUNT
    state = flat_state.reshape(3 + derivative_count, dimension, dimension)
    field = schedule.convolved_field_from_raw(time, raw_fields)
    total, rotation, twist = _hamiltonian_parts(field, operators)
    a = -1.0j * total
    a_rotation = -1.0j * rotation
    a_twist = -1.0j * twist

    derivative = np.empty_like(state)
    derivative[0] = a @ state[0]
    derivative[1] = a @ state[1] + a_rotation @ state[0]
    derivative[2] = a @ state[2] + a_twist @ state[0]
    if raw_field_derivatives is not None:
        control_fields = schedule.convolved_derivatives_from_raw(time, raw_field_derivatives)
        control_hamiltonians = _field_derivative_hamiltonians(control_fields, operators)
        derivative[3:] = (
            np.einsum("ab,qbc->qac", a, state[3:])
            + np.einsum("qab,bc->qac", -1.0j * control_hamiltonians, state[0])
        )
    return derivative.reshape(-1)


def integrate_block(
    schedule: ModuleSchedule,
    module_params: Sequence[float],
    operators: OperatorBlock,
    settings: IntegrationSettings,
    with_control_derivatives: bool = False,
) -> BlockPropagation:
    """Integrate one invariant block with analytic error/control derivatives."""

    parameter_array = np.asarray(module_params, dtype=np.float64)
    if parameter_array.shape != (PARAMETER_COUNT,):
        raise ValueError("one module requires 44 parameters")
    raw_fields = schedule.raw_fields(parameter_array)
    raw_derivatives = (
        schedule.raw_field_derivatives(parameter_array) if with_control_derivatives else None
    )
    dimension = operators.dimension
    count = 3 + (PARAMETER_COUNT if with_control_derivatives else 0)
    state = np.zeros((count, dimension, dimension), dtype=np.complex128)
    state[0] = np.eye(dimension, dtype=np.complex128)

    transition_count = 0
    plateau_count = 0
    nfev = 0
    for kind, slot_or_boundary, left, right in schedule.chronological_intervals():
        if kind == "plateau":
            state = _constant_update(
                state,
                raw_fields[slot_or_boundary],
                None if raw_derivatives is None else raw_derivatives[:, slot_or_boundary, :],
                operators,
                right - left,
            )
            plateau_count += 1
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
                f"{operators.name} transition {slot_or_boundary} failed: {solution.message}"
            )
        state = solution.y[:, -1].reshape(count, dimension, dimension)
        transition_count += 1
        nfev += int(solution.nfev)

    unitary, d_rotation, d_twist = state[:3]
    k_rotation = 1.0j * unitary.conj().T @ d_rotation
    k_twist = 1.0j * unitary.conj().T @ d_twist
    k_rotation = (k_rotation + k_rotation.conj().T) / 2.0
    k_twist = (k_twist + k_twist.conj().T) / 2.0
    return BlockPropagation(
        unitary=unitary,
        d_error_rotation=d_rotation,
        d_error_twist=d_twist,
        k_error_rotation=k_rotation,
        k_error_twist=k_twist,
        control_derivatives=None if not with_control_derivatives else state[3:].copy(),
        transition_count=transition_count,
        plateau_count=plateau_count,
        nfev=nfev,
    )


def evaluate_module(
    schedule: ModuleSchedule,
    module_params: Sequence[float],
    settings: IntegrationSettings = WORKING_INTEGRATION,
    representation: Literal["schur", "full8"] = "schur",
    with_control_derivatives: bool = False,
) -> WaveformModulePropagation:
    """Evaluate one smooth waveform module in a declared representation."""

    if representation == "full8":
        full = integrate_block(
            schedule,
            module_params,
            operator_block("full8"),
            settings,
            with_control_derivatives,
        )
        return WaveformModulePropagation("full8", None, None, full)
    if representation != "schur":
        raise ValueError("representation must be schur or full8")
    visible = integrate_block(
        schedule,
        module_params,
        operator_block("visible"),
        settings,
        with_control_derivatives,
    )
    hidden = integrate_block(
        schedule,
        module_params,
        operator_block("hidden"),
        settings,
        with_control_derivatives,
    )
    return WaveformModulePropagation("schur", visible, hidden, None)


def _project_full8_block(
    full: BlockPropagation, isometry: np.ndarray, name: str
) -> BlockPropagation:
    """Project a FULL8 propagation to a symmetry block for comparison only."""

    unitary = m8.block(full.unitary, isometry)
    d_rotation = m8.block(full.d_error_rotation, isometry)
    d_twist = m8.block(full.d_error_twist, isometry)
    k_rotation = 1.0j * unitary.conj().T @ d_rotation
    k_twist = 1.0j * unitary.conj().T @ d_twist
    k_rotation = (k_rotation + k_rotation.conj().T) / 2.0
    k_twist = (k_twist + k_twist.conj().T) / 2.0
    controls = None
    if full.control_derivatives is not None:
        controls = np.asarray([m8.block(item, isometry) for item in full.control_derivatives])
    return BlockPropagation(
        unitary=unitary,
        d_error_rotation=d_rotation,
        d_error_twist=d_twist,
        k_error_rotation=k_rotation,
        k_error_twist=k_twist,
        control_derivatives=controls,
        transition_count=full.transition_count,
        plateau_count=full.plateau_count,
        nfev=full.nfev,
    )


def module_blocks(
    propagation: WaveformModulePropagation,
) -> tuple[BlockPropagation, BlockPropagation]:
    """Return visible and hidden blocks, deriving them from FULL8 if needed."""

    if propagation.representation == "schur":
        assert propagation.visible is not None and propagation.hidden is not None
        return propagation.visible, propagation.hidden
    assert propagation.full8 is not None
    return (
        _project_full8_block(propagation.full8, m8.VISIBLE_ISOMETRY, "visible"),
        _project_full8_block(propagation.full8, m8.HIDDEN_ISOMETRY, "hidden"),
    )


def _block_generator_coordinates(generator: np.ndarray) -> np.ndarray:
    return m8.generator_coordinates(generator)


def module_constraint_chart(
    visible: BlockPropagation, hidden: BlockPropagation
) -> ModuleConstraintChart:
    """Evaluate the fully reported 41 rows and structurally justified 37 rows."""

    hidden_unitary = hidden.unitary
    trace_hidden = np.trace(hidden_unitary)
    trace_abs = float(abs(trace_hidden))
    if trace_abs < MIN_HIDDEN_TRACE_ABS:
        raise ValueError("hidden trace chart is singular")
    closure_quotients = np.asarray(
        (
            (hidden_unitary[1, 0] + hidden_unitary[0, 1]) / trace_hidden,
            (hidden_unitary[1, 0] - hidden_unitary[0, 1]) / trace_hidden,
            (hidden_unitary[0, 0] - hidden_unitary[1, 1]) / trace_hidden,
        )
    )
    closure = np.asarray(
        (
            -closure_quotients[0].imag,
            closure_quotients[1].real,
            -closure_quotients[2].imag,
        ),
        dtype=np.float64,
    )
    values: list[float] = closure.tolist()
    for visible_generator, hidden_generator in (
        (visible.k_error_rotation, hidden.k_error_rotation),
        (visible.k_error_twist, hidden.k_error_twist),
    ):
        values.extend(_block_generator_coordinates(visible_generator).tolist())
        values.extend(_block_generator_coordinates(hidden_generator)[[0, 2, 3]].tolist())
    full41 = np.asarray(values, dtype=np.float64)
    if full41.shape != (41,):
        raise AssertionError("wrong waveform constraint chart shape")
    independent = full41[INDEPENDENT_37_INDICES]
    hidden_twist_scalar = hidden.k_error_twist - (
        np.trace(hidden.k_error_twist) / 2.0
    ) * np.eye(2, dtype=np.complex128)
    diagnostics = {
        "visible_rotation_trace_abs": float(abs(np.trace(visible.k_error_rotation))),
        "hidden_rotation_trace_abs": float(abs(np.trace(hidden.k_error_rotation))),
        "full_rotation_trace_abs": float(
            abs(np.trace(visible.k_error_rotation) + 2.0 * np.trace(hidden.k_error_rotation))
        ),
        "visible_twist_trace_abs": float(abs(np.trace(visible.k_error_twist))),
        "hidden_twist_scalar_defect_fro": float(np.linalg.norm(hidden_twist_scalar)),
        "omitted41_max_abs": float(np.max(np.abs(full41[OMITTED_41_INDICES]))),
    }
    return ModuleConstraintChart(
        full41=full41,
        independent37=independent,
        closure3=closure,
        lambda_hidden=complex(trace_hidden / 2.0),
        hidden_trace_abs=trace_abs,
        omitted41=full41[OMITTED_41_INDICES].copy(),
        diagnostics=diagnostics,
    )


def interface_gate(
    visible: BlockPropagation, hidden: BlockPropagation
) -> tuple[np.ndarray, np.ndarray | None, complex]:
    """Return visible gate divided by hidden scalar, plus analytic dG/dp."""

    scalar = complex(np.trace(hidden.unitary) / 2.0)
    if abs(2.0 * scalar) < MIN_HIDDEN_TRACE_ABS:
        raise ValueError("hidden trace chart is singular")
    gate = visible.unitary / scalar
    if visible.control_derivatives is None or hidden.control_derivatives is None:
        return gate, None, scalar
    d_visible = visible.control_derivatives
    d_hidden = hidden.control_derivatives
    d_scalar = np.trace(d_hidden, axis1=1, axis2=2) / 2.0
    derivatives = d_visible / scalar - (
        visible.unitary[None, :, :] * d_scalar[:, None, None] / (scalar * scalar)
    )
    return gate, derivatives, scalar


def analyze_module(propagation: WaveformModulePropagation) -> ModuleAnalysis:
    """Turn a propagation result into the residual/target interface data."""

    visible, hidden = module_blocks(propagation)
    chart = module_constraint_chart(visible, hidden)
    gate, derivatives, _ = interface_gate(visible, hidden)
    return ModuleAnalysis(
        propagation=propagation,
        visible=visible,
        hidden=hidden,
        chart=chart,
        interface_gate4=gate,
        interface_gate_derivatives=derivatives,
    )


def source_target_chart(
    analyses: Sequence[ModuleAnalysis],
    source_target_gate4: np.ndarray,
    with_jacobian: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """Evaluate the 15-coordinate source-referenced target chart."""

    if len(analyses) != MODULE_COUNT:
        raise ValueError("expected three module analyses")
    source = np.asarray(source_target_gate4, dtype=np.complex128)
    if source.shape != (4, 4):
        raise ValueError("source target gate must have shape (4,4)")
    gates = [analysis.interface_gate4 for analysis in analyses]
    target = gates[0] @ gates[1] @ gates[2]
    relative = source.conj().T @ target
    hermitian = 0.5j * (relative - relative.conj().T)
    value = m8.su4_coordinates(hermitian)
    lambdas = np.asarray([analysis.chart.lambda_hidden for analysis in analyses], dtype=np.complex128)
    if not with_jacobian:
        return value, None, target, lambdas

    local = [analysis.interface_gate_derivatives for analysis in analyses]
    if any(item is None for item in local):
        raise ValueError("control derivatives were not propagated")
    derivatives = [np.asarray(item) for item in local]
    dtarget = np.zeros((TOTAL_PARAMETER_COUNT, 4, 4), dtype=np.complex128)
    dtarget[0:44] = np.einsum("jab,bc,cd->jad", derivatives[0], gates[1], gates[2])
    dtarget[44:88] = np.einsum("ab,jbc,cd->jad", gates[0], derivatives[1], gates[2])
    dtarget[88:132] = np.einsum("ab,bc,jcd->jad", gates[0], gates[1], derivatives[2])
    drelative = np.einsum("ab,jbc->jac", source.conj().T, dtarget)
    dhermitian = 0.5j * (drelative - np.swapaxes(drelative.conj(), 1, 2))
    jacobian = np.column_stack([m8.su4_coordinates(item) for item in dhermitian])
    return value, jacobian, target, lambdas


def direct_full8_residuals(propagation: WaveformModulePropagation) -> dict[str, float]:
    """Evaluate closure/leakage and generator norms directly in the FULL8 space."""

    if propagation.representation != "full8" or propagation.full8 is None:
        raise ValueError("direct FULL8 residuals require a full8 propagation")
    full = propagation.full8
    evaluation = m8.ModuleEvaluation(
        full.unitary,
        full.d_error_rotation,
        full.d_error_twist,
        full.k_error_rotation,
        full.k_error_twist,
    )
    return m8.full8_residuals(evaluation)


def phase_aligned_frobenius(target: np.ndarray, actual: np.ndarray) -> tuple[float, complex]:
    """Use the frozen finite-fiber global-phase convention."""

    return ff.phase_aligned_frobenius(target, actual)


def candidate_domain_record(
    params: Sequence[float] | np.ndarray, source_params: Sequence[float] | np.ndarray
) -> dict[str, Any]:
    """Check the registered component box and global source ball."""

    candidate = ff.as_module_parameters(params)
    source = ff.as_module_parameters(source_params)
    theta_ratio = candidate[:, :LENGTH] / source[:, :LENGTH]
    chi_ratio = candidate[:, 2 * LENGTH - 1 :] / source[:, 2 * LENGTH - 1 :]
    phase_delta = candidate[:, LENGTH : 2 * LENGTH - 1] - source[:, LENGTH : 2 * LENGTH - 1]
    distance = float(np.linalg.norm(candidate - source))
    theta_ok = bool(np.all((theta_ratio >= CORRECTION_RATIO_MIN) & (theta_ratio <= CORRECTION_RATIO_MAX)))
    chi_ok = bool(np.all((chi_ratio >= CORRECTION_RATIO_MIN) & (chi_ratio <= CORRECTION_RATIO_MAX)))
    phase_ok = bool(np.max(np.abs(phase_delta)) <= PHASE_OFFSET_MAX)
    ball_ok = bool(distance <= GLOBAL_SOURCE_DISTANCE_MAX)
    return {
        "passed": bool(theta_ok and chi_ok and phase_ok and ball_ok),
        "theta_ratio_min": float(np.min(theta_ratio)),
        "theta_ratio_max": float(np.max(theta_ratio)),
        "chi_ratio_min": float(np.min(chi_ratio)),
        "chi_ratio_max": float(np.max(chi_ratio)),
        "phase_offset_max_abs": float(np.max(np.abs(phase_delta))),
        "source_distance": distance,
        "theta_box_passed": theta_ok,
        "chi_box_passed": chi_ok,
        "phase_box_passed": phase_ok,
        "source_ball_passed": ball_ok,
    }


class FrozenWaveformModel:
    """The source-bound waveform map and its direct/Schur evaluation paths."""

    def __init__(
        self,
        source_params: Sequence[float] | np.ndarray,
        source_target_gate4: np.ndarray,
        source_target_u_full8: np.ndarray | None = None,
        fiber_basis132x6: np.ndarray | None = None,
        z: Sequence[float] = ff.Z_DIRECTION,
        width: float = LAMBDA_WIDTH,
    ) -> None:
        self.source_params = ff.as_module_parameters(source_params).copy()
        self.source_target_gate4 = np.asarray(source_target_gate4, dtype=np.complex128).copy()
        if self.source_target_gate4.shape != (4, 4):
            raise ValueError("source target gate must have shape (4,4)")
        self.source_target_u_full8 = (
            None
            if source_target_u_full8 is None
            else np.asarray(source_target_u_full8, dtype=np.complex128).copy()
        )
        if self.source_target_u_full8 is not None and self.source_target_u_full8.shape != (8, 8):
            raise ValueError("source full target must have shape (8,8)")
        self.fiber_basis = (
            None if fiber_basis132x6 is None else np.asarray(fiber_basis132x6, dtype=np.float64).copy()
        )
        if self.fiber_basis is not None and self.fiber_basis.shape != (132, 6):
            raise ValueError("fiber basis must have shape (132,6)")
        self.z = np.asarray(z, dtype=np.float64).copy()
        if self.z.shape != (6,):
            raise ValueError("z must have shape (6,)")
        self.width = float(width)
        self.schedules = tuple(
            build_module_schedule(self.source_params[module], module, self.width)
            for module in range(MODULE_COUNT)
        )

    @classmethod
    def from_frozen_arrays(cls, arrays: dict[str, np.ndarray], width: float = LAMBDA_WIDTH) -> "FrozenWaveformModel":
        required = ("source_params", "source_target_gate4", "source_target_u_full8", "fiber_basis132x6")
        missing = [key for key in required if key not in arrays]
        if missing:
            raise KeyError(f"missing waveform source arrays: {missing}")
        return cls(
            arrays["source_params"],
            arrays["source_target_gate4"],
            arrays["source_target_u_full8"],
            arrays["fiber_basis132x6"],
            arrays.get("z", ff.Z_DIRECTION),
            width,
        )

    def resource_record(self, params: Sequence[float] | np.ndarray | None = None) -> dict[str, Any]:
        """Report fixed physical time and separate bang-bang action diagnostics."""

        candidate = self.source_params if params is None else ff.as_module_parameters(params)
        action_theta = float(np.sum(np.abs(candidate[:, :LENGTH])))
        action_chi = float(np.sum(np.abs(candidate[:, 2 * LENGTH - 1 :])))
        raw_by_module = [schedule.raw_duration for schedule in self.schedules]
        physical_by_module = [schedule.total_duration for schedule in self.schedules]
        return {
            "omega_max": OMEGA_MAX,
            "g_max_old_hzz_rate": G_MAX,
            "time_stretch": TIME_STRETCH,
            "lambda_width": self.width,
            "module_padding_count": 6,
            "module_padding_total_time": 6.0 * self.width,
            "physical_time_fixed_by_schedule": float(sum(physical_by_module)),
            "raw_stretched_slot_time": float(sum(raw_by_module)),
            "physical_time_by_module": physical_by_module,
            "area_theta_candidate_diagnostic": action_theta,
            "area_chi_candidate_diagnostic": action_chi,
            "same_area_bang_bang_proxy_eta10_only": action_theta / OMEGA_MAX + action_chi / G_MAX,
            "time_interpretation": (
                "physical_time_fixed_by_schedule is the implementation time; "
                "same_area_bang_bang_proxy_eta10_only is diagnostic and not a replacement"
            ),
        }

    def overlap_diagnostics(
        self, module_index: int, module_params: Sequence[float], samples_per_transition: int = 17
    ) -> dict[str, Any]:
        """Static control diagnostics proving nonzero noncommuting overlap."""

        if samples_per_transition < 3:
            raise ValueError("at least three transition samples are required")
        schedule = self.schedules[module_index]
        fields = schedule.raw_fields(module_params)
        sample_ratios = np.linspace(-0.98, 0.98, samples_per_transition)
        max_rotation = 0.0
        max_twist = 0.0
        max_product = 0.0
        max_commutator = 0.0
        overlap_count = 0
        records: list[dict[str, float]] = []
        for boundary_index in range(1, len(schedule.boundaries) - 1):
            boundary = schedule.boundaries[boundary_index]
            boundary_max_product = 0.0
            boundary_max_commutator = 0.0
            for ratio in sample_ratios:
                field = schedule.convolved_field_from_raw(
                    boundary + schedule.lambda_width * ratio, fields
                )
                rotation_norm = float(np.linalg.norm(field[:2]))
                twist_abs = float(abs(field[2]))
                product = rotation_norm * twist_abs
                rotation_hamiltonian = field[0] * m8.JX + field[1] * m8.JY
                twist_hamiltonian = field[2] * m8.HZZ
                commutator = float(
                    np.linalg.norm(rotation_hamiltonian @ twist_hamiltonian - twist_hamiltonian @ rotation_hamiltonian)
                )
                max_rotation = max(max_rotation, rotation_norm)
                max_twist = max(max_twist, twist_abs)
                max_product = max(max_product, product)
                max_commutator = max(max_commutator, commutator)
                boundary_max_product = max(boundary_max_product, product)
                boundary_max_commutator = max(boundary_max_commutator, commutator)
            if boundary_max_product > 1.0e-12 and boundary_max_commutator > 1.0e-12:
                overlap_count += 1
            records.append(
                {
                    "boundary_index": float(boundary_index),
                    "max_overlap_product": boundary_max_product,
                    "max_commutator_fro": boundary_max_commutator,
                }
            )
        slope_rotation_bound = KERNEL_NORMALIZATION * OMEGA_MAX / self.width
        slope_twist_bound = KERNEL_NORMALIZATION * G_MAX / self.width
        return {
            "module_index": module_index,
            "internal_boundary_count": len(schedule.boundaries) - 2,
            "noncommuting_overlap_interval_count": overlap_count,
            "max_rotation_amplitude_sampled": max_rotation,
            "max_twist_amplitude_sampled": max_twist,
            "max_overlap_product_sampled": max_product,
            "max_commutator_fro_sampled": max_commutator,
            "rise_fall_width": 2.0 * self.width,
            "rotation_slope_bound": slope_rotation_bound,
            "twist_slope_bound": slope_twist_bound,
            "strict_bandlimit_claimed": False,
            "transition_records": records,
        }

    def static_preflight(self) -> dict[str, Any]:
        """Run no propagation: only source shape, schedule, and control checks."""

        source_domain = candidate_domain_record(self.source_params, self.source_params)
        overlaps = [
            self.overlap_diagnostics(index, self.source_params[index])
            for index in range(MODULE_COUNT)
        ]
        raw_amplitudes = []
        for index, schedule in enumerate(self.schedules):
            raw = schedule.raw_fields(self.source_params[index])
            raw_amplitudes.append(
                {
                    "module": index,
                    "max_rotation_raw": float(np.max(np.linalg.norm(raw[:, :2], axis=1))),
                    "max_twist_raw": float(np.max(np.abs(raw[:, 2]))),
                    "min_slot_duration": float(np.min(schedule.durations)),
                    "raw_duration": schedule.raw_duration,
                    "padded_duration": schedule.total_duration,
                }
            )
        return {
            "status": "STATIC_PREFLIGHT_PASS",
            "source_shape": list(self.source_params.shape),
            "width": self.width,
            "kernel_order": KERNEL_ORDER,
            "kernel_normalization": KERNEL_NORMALIZATION,
            "shortest_slot_guard_passed": all(
                float(np.min(schedule.durations)) > 2.0 * self.width
                for schedule in self.schedules
            ),
            "source_domain": source_domain,
            "source_raw_amplitudes": raw_amplitudes,
            "overlap": overlaps,
            "resource": self.resource_record(self.source_params),
            "no_propagation_performed": True,
        }

    def evaluate_modules(
        self,
        params: Sequence[float] | np.ndarray,
        settings: IntegrationSettings = WORKING_INTEGRATION,
        representation: Literal["schur", "full8"] = "schur",
        with_control_derivatives: bool = False,
    ) -> tuple[WaveformModulePropagation, ...]:
        """Evaluate the three modules in mathematical product ordering."""

        modules = ff.as_module_parameters(params)
        return tuple(
            evaluate_module(
                self.schedules[index],
                modules[index],
                settings,
                representation,
                with_control_derivatives,
            )
            for index in range(MODULE_COUNT)
        )

    def augmented_residual(
        self,
        params: Sequence[float] | np.ndarray,
        h: float = H_VALUE,
        settings: IntegrationSettings = WORKING_INTEGRATION,
        with_control_derivatives: bool = False,
        propagations: Sequence[WaveformModulePropagation] | None = None,
    ) -> WaveformAugmentedEvaluation:
        """Evaluate r126 + graph6 in the declared waveform coordinate order."""

        if self.fiber_basis is None:
            raise ValueError("fiber basis is required for augmented waveform residual")
        modules = ff.as_module_parameters(params)
        if propagations is None:
            propagations = self.evaluate_modules(
                modules, settings, "schur", with_control_derivatives
            )
        if len(propagations) != MODULE_COUNT:
            raise ValueError("expected three module propagations")
        analyses = tuple(analyze_module(item) for item in propagations)
        target_value, target_jacobian, target_gate, lambdas = source_target_chart(
            analyses, self.source_target_gate4, with_control_derivatives
        )
        module_value = np.concatenate([item.chart.independent37 for item in analyses])
        graph = self.fiber_basis.T @ (modules.reshape(-1) - self.source_params.reshape(-1)) - float(h) * self.z
        r126 = np.concatenate((module_value, target_value))
        value = np.concatenate((r126, graph))
        if r126.shape != (126,) or graph.shape != (6,) or value.shape != (132,):
            raise AssertionError("wrong waveform augmented residual dimensions")
        return WaveformAugmentedEvaluation(
            r126=r126,
            graph6=graph,
            value132=value,
            module_analyses=analyses,
            target_value15=target_value,
            target_gate4=target_gate,
            target_jacobian15x132=target_jacobian,
            lambdas=lambdas,
        )

    def full8_cross_audit(
        self,
        params: Sequence[float] | np.ndarray,
        h: float = H_VALUE,
        settings: IntegrationSettings = TIGHT_INTEGRATION,
        schur_evaluation: WaveformAugmentedEvaluation | None = None,
        full_propagations: Sequence[WaveformModulePropagation] | None = None,
    ) -> dict[str, Any]:
        """Direct FULL8 candidate audit and comparison against Schur production."""

        if self.source_target_u_full8 is None:
            raise ValueError("source full8 target is required for direct audit")
        modules = ff.as_module_parameters(params)
        if schur_evaluation is None:
            schur_evaluation = self.augmented_residual(modules, h, settings)
        if full_propagations is None:
            full_propagations = self.evaluate_modules(modules, settings, "full8", False)
        if len(full_propagations) != MODULE_COUNT:
            raise ValueError("expected three direct FULL8 propagations")
        if any(item.representation != "full8" for item in full_propagations):
            raise ValueError("full8 cross audit requires direct FULL8 propagations")
        full_analyses = tuple(analyze_module(item) for item in full_propagations)
        full_target_value, _, full_target_gate, full_lambdas = source_target_chart(
            full_analyses, self.source_target_gate4, False
        )
        full_module_values = np.concatenate(
            [analysis.chart.independent37 for analysis in full_analyses]
        )
        graph = self.fiber_basis.T @ (modules.reshape(-1) - self.source_params.reshape(-1)) - float(h) * self.z
        full_r126 = np.concatenate((full_module_values, full_target_value))
        full_value = np.concatenate((full_r126, graph))

        comparison_modules: list[dict[str, Any]] = []
        full8_records: list[dict[str, float]] = []
        for module_index, (schur_analysis, full_propagation) in enumerate(
            zip(schur_evaluation.module_analyses, full_propagations)
        ):
            full_visible, full_hidden = module_blocks(full_propagation)
            comparison_modules.append(
                {
                    "module": module_index,
                    "visible_unitary_fro": float(
                        np.linalg.norm(full_visible.unitary - schur_analysis.visible.unitary)
                    ),
                    "hidden_unitary_fro": float(
                        np.linalg.norm(full_hidden.unitary - schur_analysis.hidden.unitary)
                    ),
                    "visible_k_rotation_fro": float(
                        np.linalg.norm(
                            full_visible.k_error_rotation
                            - schur_analysis.visible.k_error_rotation
                        )
                    ),
                    "hidden_k_rotation_fro": float(
                        np.linalg.norm(
                            full_hidden.k_error_rotation
                            - schur_analysis.hidden.k_error_rotation
                        )
                    ),
                    "visible_k_twist_fro": float(
                        np.linalg.norm(
                            full_visible.k_error_twist
                            - schur_analysis.visible.k_error_twist
                        )
                    ),
                    "hidden_k_twist_fro": float(
                        np.linalg.norm(
                            full_hidden.k_error_twist
                            - schur_analysis.hidden.k_error_twist
                        )
                    ),
                }
            )
            full8_records.append(direct_full8_residuals(full_propagation))

        assert all(item.full8 is not None for item in full_propagations)
        full_total = (
            full_propagations[0].full8.unitary
            @ full_propagations[1].full8.unitary
            @ full_propagations[2].full8.unitary
        )
        target_fro, target_phase = phase_aligned_frobenius(
            self.source_target_u_full8, full_total
        )
        return {
            "full8_r126_max_abs": float(np.max(np.abs(full_r126))),
            "full8_graph6_max_abs": float(np.max(np.abs(graph))),
            "full8_value132_max_abs": float(np.max(np.abs(full_value))),
            "schur_full8_r126_max_abs_difference": float(
                np.max(np.abs(full_r126 - schur_evaluation.r126))
            ),
            "schur_full8_value132_max_abs_difference": float(
                np.max(np.abs(full_value - schur_evaluation.value132))
            ),
            "full8_target_chart_max_abs": float(np.max(np.abs(full_target_value))),
            "full8_target_phase_aligned_fro": target_fro,
            "full8_target_aligning_phase_real": float(target_phase.real),
            "full8_target_aligning_phase_imag": float(target_phase.imag),
            "full8_lambdas_real": full_lambdas.real.tolist(),
            "full8_lambdas_imag": full_lambdas.imag.tolist(),
            "module_full8_residuals": full8_records,
            "module_full8_chart_diagnostics": [
                {
                    "independent37_max_abs": float(np.max(np.abs(analysis.chart.independent37))),
                    "full41_max_abs": float(np.max(np.abs(analysis.chart.full41))),
                    "omitted41_max_abs": float(np.max(np.abs(analysis.chart.omitted41))),
                    "hidden_trace_abs": analysis.chart.hidden_trace_abs,
                    "hidden_twist_scalar_defect_fro": analysis.chart.diagnostics[
                        "hidden_twist_scalar_defect_fro"
                    ],
                }
                for analysis in full_analyses
            ],
            "schur_full8_block_comparison": comparison_modules,
            "full8_target_gate4_real": full_target_gate.real.tolist(),
            "full8_target_gate4_imag": full_target_gate.imag.tolist(),
        }


def augmented_summary(evaluation: WaveformAugmentedEvaluation) -> dict[str, Any]:
    """JSON-safe compact record for a waveform residual evaluation."""

    return {
        "r126_l2": float(np.linalg.norm(evaluation.r126)),
        "r126_max_abs": float(np.max(np.abs(evaluation.r126))),
        "graph6_l2": float(np.linalg.norm(evaluation.graph6)),
        "graph6_max_abs": float(np.max(np.abs(evaluation.graph6))),
        "value132_l2": float(np.linalg.norm(evaluation.value132)),
        "value132_max_abs": float(np.max(np.abs(evaluation.value132))),
        "target_value15_max_abs": float(np.max(np.abs(evaluation.target_value15))),
        "lambdas_real": evaluation.lambdas.real.tolist(),
        "lambdas_imag": evaluation.lambdas.imag.tolist(),
        "module_charts": [
            {
                "module": index,
                "independent37_max_abs": float(np.max(np.abs(analysis.chart.independent37))),
                "full41_max_abs": float(np.max(np.abs(analysis.chart.full41))),
                "omitted41_max_abs": float(np.max(np.abs(analysis.chart.omitted41))),
                "hidden_trace_abs": analysis.chart.hidden_trace_abs,
                "lambda_real": float(analysis.chart.lambda_hidden.real),
                "lambda_imag": float(analysis.chart.lambda_hidden.imag),
                "diagnostics": analysis.chart.diagnostics,
            }
            for index, analysis in enumerate(evaluation.module_analyses)
        ],
    }


def source_arrays_record(path: Path) -> dict[str, np.ndarray]:
    """Load a read-only .npz source into ordinary arrays."""

    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key].copy() for key in loaded.files}


__all__ = [
    "ARMIJO_COEFFICIENT",
    "CORRECTION_RATIO_MAX",
    "CORRECTION_RATIO_MIN",
    "FrozenWaveformModel",
    "G_MAX",
    "GLOBAL_SOURCE_DISTANCE_MAX",
    "H_VALUE",
    "INDEPENDENT_37_INDICES",
    "IntegrationSettings",
    "KERNEL_NORMALIZATION",
    "LAMBDA_BRANCH_REAL_MAX",
    "LAMBDA_WIDTH",
    "MIN_HIDDEN_TRACE_ABS",
    "MODULE_COUNT",
    "MODULE_LABELS",
    "OMEGA_MAX",
    "PHASE_OFFSET_MAX",
    "TIGHT_INTEGRATION",
    "TIME_STRETCH",
    "WORKING_INTEGRATION",
    "WaveformAugmentedEvaluation",
    "analyze_module",
    "augmented_summary",
    "candidate_domain_record",
    "compact_cdf",
    "compact_kernel",
    "direct_full8_residuals",
    "evaluate_module",
    "module_constraint_chart",
    "phase_aligned_frobenius",
    "source_arrays_record",
]
