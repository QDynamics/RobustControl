"""Registered N=3 FULL8 circuit framework with a target/time ablation.

This module deliberately contains no optimizer.  It works only with the
three frozen 15-layer source modules and evaluates complete 8x8 propagation.
The caller owns all file creation and authorization gates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable, Mapping, Sequence
import time

import numpy as np

import manifold_full8_v1 as m8


MODULE_LABELS = ("00", "15", "25")
SOURCE_ROWS = {"00": 0, "15": 1, "25": 2}
DEPTHS = (1, 2, 4, 8, 16)
ERROR_MODELS = ("common", "profile")
PROFILE_COEFFICIENTS = {
    "00": (1.0, -0.5),
    "15": (-2.0 / 3.0, 1.0),
    "25": (1.0 / 3.0, 2.0 / 3.0),
}
GRID_AXIS = np.linspace(-0.01, 0.01, 21, dtype=np.float64)
SMALL_ERROR_RAYS = (
    (1.0, 0.0),
    (0.0, 1.0),
    (1.0, 1.0),
    (1.0, -1.0),
)
SMALL_ERROR_SIGNS = (-1.0, 1.0)
SMALL_ERROR_RADII = (1.0e-5, 1.0e-4, 1.0e-3)


class BudgetViolation(RuntimeError):
    """Raised before an unregistered scientific-call budget is exceeded."""


def _as_vector(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (m8.PARAMETER_COUNT,) or not np.all(np.isfinite(array)):
        raise ValueError("one module must be a finite 44-vector")
    return array


def source_rows_to_params(source_params: Sequence[Sequence[float]] | np.ndarray) -> dict[str, np.ndarray]:
    """Bind the frozen array rows to their pre-registered labels."""
    array = np.asarray(source_params, dtype=np.float64)
    if array.shape != (3, m8.PARAMETER_COUNT) or not np.all(np.isfinite(array)):
        raise ValueError("source_params must have shape (3,44) and be finite")
    return {label: array[row].copy() for label, row in SOURCE_ROWS.items()}


def validate_params_by_label(params_by_label: Mapping[str, Sequence[float] | np.ndarray]) -> dict[str, np.ndarray]:
    if set(params_by_label) != set(MODULE_LABELS):
        raise ValueError("parameters must contain exactly labels 00, 15, 25")
    return {label: _as_vector(params_by_label[label]).copy() for label in MODULE_LABELS}


def word_labels(depth: int) -> tuple[str, ...]:
    if depth not in DEPTHS:
        raise ValueError(f"unregistered depth {depth!r}")
    return tuple(MODULE_LABELS[index % len(MODULE_LABELS)] for index in range(depth))


def grid_points() -> np.ndarray:
    return np.asarray([(er, et) for er in GRID_AXIS for et in GRID_AXIS], dtype=np.float64)


def error_coefficients(error_model: str, label: str) -> tuple[float, float]:
    if label not in MODULE_LABELS:
        raise ValueError(f"unknown module label {label!r}")
    if error_model == "common":
        return 1.0, 1.0
    if error_model == "profile":
        return PROFILE_COEFFICIENTS[label]
    raise ValueError(f"unregistered error model {error_model!r}")


@dataclass
class CallLedger:
    """Hard bounded accounting for direct scientific API calls."""

    max_module_propagations: int = 12000
    max_single_call_seconds: float = 30.0
    module_propagations: int = 0
    module_cache_hits: int = 0
    control_derivative_calls: int = 0
    maximum_call_seconds: float = 0.0
    timings: dict[str, float] = field(default_factory=dict)

    def _timed(self, kind: str, function: Callable[[], Any]) -> Any:
        started = time.perf_counter()
        try:
            return function()
        finally:
            elapsed = time.perf_counter() - started
            self.maximum_call_seconds = max(self.maximum_call_seconds, elapsed)
            self.timings[kind] = self.timings.get(kind, 0.0) + elapsed
            if elapsed > self.max_single_call_seconds:
                raise BudgetViolation(
                    f"{kind} used {elapsed:.6f}s, above {self.max_single_call_seconds:.6f}s"
                )

    def module(self, function: Callable[[], m8.ModuleEvaluation]) -> m8.ModuleEvaluation:
        if self.module_propagations >= self.max_module_propagations:
            raise BudgetViolation(
                f"module propagation budget {self.max_module_propagations} exhausted"
            )
        self.module_propagations += 1
        return self._timed("module_propagator", function)

    def control_derivative(
        self, function: Callable[[], tuple[np.ndarray, np.ndarray]]
    ) -> tuple[np.ndarray, np.ndarray]:
        self.control_derivative_calls += 1
        return self._timed("module_control_derivatives", function)

    def summary(self) -> dict[str, Any]:
        return {
            "module_propagations": int(self.module_propagations),
            "module_cache_hits": int(self.module_cache_hits),
            "control_derivative_calls": int(self.control_derivative_calls),
            "scientific_api_calls": int(self.module_propagations + self.control_derivative_calls),
            "maximum_single_call_seconds": float(self.maximum_call_seconds),
            "time_by_api_seconds": {key: float(value) for key, value in self.timings.items()},
            "caps": {
                "module_propagations": int(self.max_module_propagations),
                "single_call_seconds": float(self.max_single_call_seconds),
            },
        }


class CachedModuleOracle:
    """Cache one fixed three-label library without changing its propagation."""

    def __init__(
        self,
        params_by_label: Mapping[str, Sequence[float] | np.ndarray],
        ledger: CallLedger | None = None,
    ) -> None:
        self.params = validate_params_by_label(params_by_label)
        self.ledger = ledger
        self._cache: dict[tuple[str, str, str, bool], m8.ModuleEvaluation] = {}

    def evaluate(
        self,
        label: str,
        epsilon_rotation: float,
        epsilon_twist: float,
        with_error_derivatives: bool,
    ) -> m8.ModuleEvaluation:
        if label not in MODULE_LABELS:
            raise ValueError(f"unknown module label {label!r}")
        key = (
            label,
            float(epsilon_rotation).hex(),
            float(epsilon_twist).hex(),
            bool(with_error_derivatives),
        )
        cached = self._cache.get(key)
        if cached is not None:
            if self.ledger is not None:
                self.ledger.module_cache_hits += 1
            return cached

        def invoke() -> m8.ModuleEvaluation:
            return m8.module_propagator(
                self.params[label],
                epsilon_rotation=float(epsilon_rotation),
                epsilon_twist=float(epsilon_twist),
                with_error_derivatives=with_error_derivatives,
            )

        result = self.ledger.module(invoke) if self.ledger is not None else invoke()
        self._cache[key] = result
        return result

    def control_derivatives(self, params: Sequence[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vector = _as_vector(params)

        def invoke() -> tuple[np.ndarray, np.ndarray]:
            return m8.module_control_derivatives(vector)

        return self.ledger.control_derivative(invoke) if self.ledger is not None else invoke()


def _traceless_generator(ideal: np.ndarray, derivative: np.ndarray) -> np.ndarray:
    generator = 1.0j * ideal.conj().T @ derivative
    generator = (generator + generator.conj().T) / 2.0
    return generator - np.trace(generator) * np.eye(ideal.shape[0], dtype=complex) / ideal.shape[0]


def generator_record(ideal: np.ndarray, derivative: np.ndarray) -> dict[str, float]:
    raw = 1.0j * ideal.conj().T @ derivative
    raw = (raw + raw.conj().T) / 2.0
    physical = _traceless_generator(ideal, derivative)
    return {
        "raw_fro": float(np.linalg.norm(raw)),
        "trace_real": float(np.trace(raw).real),
        "trace_imag": float(np.trace(raw).imag),
        "traceless_physical_fro": float(np.linalg.norm(physical)),
    }


def compose_word(
    oracle: CachedModuleOracle,
    labels: Sequence[str],
    epsilon: Sequence[float] | np.ndarray,
    error_model: str,
    with_error_derivatives: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Compose a chronological word as U_last ... U_1 U_0 on the FULL8 space."""
    eps = np.asarray(epsilon, dtype=np.float64)
    if eps.shape != (2,) or not np.all(np.isfinite(eps)):
        raise ValueError("epsilon must be a finite pair")
    total = np.eye(m8.DIMENSION, dtype=np.complex128)
    derivative_rotation = np.zeros_like(total) if with_error_derivatives else None
    derivative_twist = np.zeros_like(total) if with_error_derivatives else None
    for label in labels:
        a_value, b_value = error_coefficients(error_model, label)
        evaluation = oracle.evaluate(
            label,
            a_value * float(eps[0]),
            b_value * float(eps[1]),
            with_error_derivatives=with_error_derivatives,
        )
        old_total = total
        if with_error_derivatives:
            assert derivative_rotation is not None and derivative_twist is not None
            derivative_rotation = (
                a_value * evaluation.d_error_rotation @ old_total
                + evaluation.unitary @ derivative_rotation
            )
            derivative_twist = (
                b_value * evaluation.d_error_twist @ old_total
                + evaluation.unitary @ derivative_twist
            )
        total = evaluation.unitary @ old_total
    return total, derivative_rotation, derivative_twist


def word_first_order_record(
    oracle: CachedModuleOracle, labels: Sequence[str], error_model: str
) -> dict[str, Any]:
    ideal, derivative_rotation, derivative_twist = compose_word(
        oracle, labels, (0.0, 0.0), error_model, with_error_derivatives=True
    )
    assert derivative_rotation is not None and derivative_twist is not None
    rotation = generator_record(ideal, derivative_rotation)
    twist = generator_record(ideal, derivative_twist)
    return {
        "rotation": rotation,
        "twist": twist,
        "maximum_traceless_physical_fro": max(
            rotation["traceless_physical_fro"], twist["traceless_physical_fro"]
        ),
        "first_order_cancelled_at_1e-8": bool(
            max(rotation["traceless_physical_fro"], twist["traceless_physical_fro"]) <= 1.0e-8
        ),
    }


def registered_word_ideals(
    oracle: CachedModuleOracle, words: Mapping[int, Sequence[str]]
) -> dict[int, np.ndarray]:
    """Return the fixed source-channel target for every registered word."""
    return {
        depth: compose_word(
            oracle, tuple(words[depth]), (0.0, 0.0), "common",
            with_error_derivatives=False,
        )[0]
        for depth in DEPTHS
    }


def phase_aligned_frobenius(target: np.ndarray, actual: np.ndarray) -> tuple[float, complex]:
    overlap = np.trace(target.conj().T @ actual)
    phase = complex(overlap / abs(overlap)) if abs(overlap) else 1.0 + 0.0j
    return float(np.linalg.norm(actual - phase * target)), phase


def pairwise_noncommutation(oracle: CachedModuleOracle) -> list[dict[str, Any]]:
    unitaries = {
        label: oracle.evaluate(label, 0.0, 0.0, with_error_derivatives=False).unitary
        for label in MODULE_LABELS
    }
    rows: list[dict[str, Any]] = []
    for left, right in combinations(MODULE_LABELS, 2):
        commutator = unitaries[left] @ unitaries[right] - unitaries[right] @ unitaries[left]
        normalized = float(np.linalg.norm(commutator) / np.sqrt(m8.DIMENSION))
        rows.append({
            "pair": [left, right],
            "full8_commutator_fro_over_sqrt8": normalized,
            "pairwise_noncommuting_at_1e-6": bool(normalized > 1.0e-6),
        })
    return rows


def word_resource_record(
    params_by_label: Mapping[str, Sequence[float] | np.ndarray],
    labels: Sequence[str],
    eta: float = 10.0,
) -> dict[str, Any]:
    params = validate_params_by_label(params_by_label)
    if not (np.isfinite(eta) and eta > 0.0):
        raise ValueError("eta must be finite and positive")
    a_theta = 0.0
    a_chi = 0.0
    segment_types: list[str] = []
    for label in labels:
        theta, _, chi = m8.split_parameters(params[label])
        a_theta += float(np.sum(np.abs(theta)))
        a_chi += float(np.sum(np.abs(chi)))
        for layer in range(m8.LENGTH):
            if chi[layer] != 0.0:
                segment_types.append("T")
            if theta[layer] != 0.0:
                segment_types.append("R")
    return {
        "eta": float(eta),
        "a_theta": float(a_theta),
        "a_chi_old_hzz": float(a_chi),
        "dimensionless_gmax_time": float(a_theta / eta + a_chi),
        "module_count": len(labels),
        "layer_slots": int(m8.LENGTH * len(labels)),
        "segment_slots": int(2 * m8.LENGTH * len(labels)),
        "nonzero_segment_count_exact": int(len(segment_types)),
        "internal_segment_boundary_count": int(max(0, len(segment_types) - 1)),
        "control_type_switch_count": int(
            sum(left != right for left, right in zip(segment_types[:-1], segment_types[1:]))
        ),
        "segment_types_chronological": segment_types,
        "time_model": "gmax*T=sum(abs(theta))/eta+sum(abs(chi)); zero extra overhead",
    }


def full8_target_basis19() -> np.ndarray:
    """Construct the fixed target chart: visible 15, hidden 3, relative phase 1."""
    visible = [
        m8.VISIBLE_ISOMETRY @ basis4 @ m8.VISIBLE_ISOMETRY.conj().T
        for basis4 in m8.normalized_pauli_basis_4()
    ]
    complement = m8.COMPLEMENT_PROJECTOR
    hidden = [complement @ generator @ complement for generator in (m8.JX, m8.JY, m8.JZ)]
    relative_phase = (m8.VISIBLE_PROJECTOR - complement) / np.sqrt(8.0)
    basis = np.asarray(visible + hidden + [relative_phase], dtype=np.complex128)
    if basis.shape != (19, 8, 8):
        raise AssertionError("FULL8 target basis must have shape (19,8,8)")
    return (basis + np.swapaxes(basis.conj(), 1, 2)) / 2.0


def single_module_target19_value_jacobian(
    params: Sequence[float] | np.ndarray,
    target: np.ndarray,
    basis19: np.ndarray,
    derivative_provider: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the single-module FULL8 19-coordinate target chart."""
    vector = _as_vector(params)
    target_array = np.asarray(target, dtype=np.complex128)
    basis = np.asarray(basis19, dtype=np.complex128)
    if target_array.shape != (8, 8) or basis.shape != (19, 8, 8):
        raise ValueError("wrong target or target19 basis shape")
    unitary, derivatives = derivative_provider(vector)
    relative = target_array.conj().T @ unitary
    drelative = np.einsum("ab,jbc->jac", target_array.conj().T, derivatives)
    hermitian = 0.5j * (relative - relative.conj().T)
    dhermitian = 0.5j * (drelative - np.swapaxes(drelative.conj(), 1, 2))
    value = np.einsum("aij,ji->a", basis, hermitian).real
    jacobian = np.einsum("aij,pji->ap", basis, dhermitian).real
    if value.shape != (19,) or jacobian.shape != (19, m8.PARAMETER_COUNT):
        raise AssertionError("wrong single-module target chart shape")
    return np.asarray(value, dtype=np.float64), np.asarray(jacobian, dtype=np.float64)


def source_sign_time_gradient(params: Sequence[float] | np.ndarray, eta: float = 10.0) -> np.ndarray:
    vector = _as_vector(params)
    if not (np.isfinite(eta) and eta > 0.0):
        raise ValueError("eta must be finite and positive")
    gradient = np.zeros(m8.PARAMETER_COUNT, dtype=np.float64)
    gradient[:m8.LENGTH] = np.sign(vector[:m8.LENGTH]) / eta
    gradient[2 * m8.LENGTH - 1:] = np.sign(vector[2 * m8.LENGTH - 1:])
    return gradient


def single_module_time(params: Sequence[float] | np.ndarray, eta: float = 10.0) -> float:
    theta, _, chi = m8.split_parameters(_as_vector(params))
    return float(np.sum(np.abs(theta)) / eta + np.sum(np.abs(chi)))


def _charged_signs(params: Sequence[float] | np.ndarray) -> np.ndarray:
    vector = _as_vector(params)
    return np.sign(np.concatenate((vector[:m8.LENGTH], vector[2 * m8.LENGTH - 1:])))


def canonical_null_basis(jacobian: np.ndarray, rcond: float = 1.0e-12) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.asarray(jacobian, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("jacobian must be a matrix")
    _, singular, vh = np.linalg.svd(matrix, full_matrices=True)
    cutoff = float(rcond * singular[0]) if len(singular) else 0.0
    rank = int(np.count_nonzero(singular > cutoff))
    basis = vh[rank:].T.copy()
    for column in range(basis.shape[1]):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0.0:
            basis[:, column] *= -1.0
    return basis, {
        "rank": rank,
        "nullity": int(basis.shape[1]),
        "rcond": float(rcond),
        "cutoff": cutoff,
        "singular_values": singular.tolist(),
        "orthonormality_fro": float(np.linalg.norm(basis.T @ basis - np.eye(basis.shape[1]))),
        "jacobian_null_fro": float(np.linalg.norm(matrix @ basis)),
    }


def deterministic_seed(label_rank: int) -> np.ndarray:
    if label_rank not in (1, 2, 3):
        raise ValueError("label_rank must be 1, 2, or 3")
    values = [((17 * (index + 1) + 13 * label_rank) % 31 - 15) / 15.0
              for index in range(m8.PARAMETER_COUNT)]
    return np.asarray(values, dtype=np.float64)


def construct_target_time_ablation(
    label: str,
    source_params: Sequence[float] | np.ndarray,
    source_oracle: CachedModuleOracle,
    eta: float = 10.0,
    h: float = 0.01,
    tolerance: float = 5.0e-12,
    maximum_updates: int = 12,
    maximum_halvings: int = 8,
    rcond: float = 1.0e-12,
    maximum_seconds: float = 30.0,
) -> dict[str, Any]:
    """Make one deterministic target/time-matched, non-robust module.

    The final 24 source-kernel affine rows make a 44-equation finite Newton
    correction.  No objective function or replacement seed is used.
    """
    if label not in MODULE_LABELS:
        raise ValueError(f"unknown module label {label!r}")
    source = _as_vector(source_params)
    started = time.perf_counter()
    basis = full8_target_basis19()
    target, source_derivatives = source_oracle.control_derivatives(source)
    source_value, source_jacobian = single_module_target19_value_jacobian(
        source, target, basis, source_oracle.control_derivatives
    )
    gradient = source_sign_time_gradient(source, eta=eta)
    combined = np.vstack((source_jacobian, gradient[None, :]))
    null_basis, null_report = canonical_null_basis(combined, rcond=rcond)
    record: dict[str, Any] = {
        "label": label,
        "source_params": source,
        "target_full8": target,
        "source_target19_max_abs": float(np.max(np.abs(source_value))),
        "source_control_derivative_fro": float(np.linalg.norm(source_derivatives)),
        "combined20_basis_report": null_report,
        "eta": float(eta),
        "h": float(h),
        "tolerance": float(tolerance),
        "maximum_updates": int(maximum_updates),
        "maximum_halvings": int(maximum_halvings),
    }
    if null_report["rank"] != 20 or null_basis.shape != (m8.PARAMETER_COUNT, 24):
        record.update({
            "status": "ABLATION_UNAVAILABLE_UNEXPECTED_20X44_RANK",
            "accepted": False,
            "elapsed_seconds": float(time.perf_counter() - started),
        })
        return record

    label_rank = MODULE_LABELS.index(label) + 1
    seed = deterministic_seed(label_rank)
    projected = null_basis @ (null_basis.T @ seed)
    projected_norm = float(np.linalg.norm(projected))
    record.update({
        "label_rank": int(label_rank),
        "seed": seed,
        "projected_seed_norm": projected_norm,
        "kernel_basis": null_basis,
    })
    if projected_norm <= 1.0e-8:
        record.update({
            "status": "ABLATION_UNAVAILABLE_SEED_PROJECTION_TOO_SMALL",
            "accepted": False,
            "elapsed_seconds": float(time.perf_counter() - started),
        })
        return record

    direction = projected / projected_norm
    trial = source + h * direction
    source_time = single_module_time(source, eta=eta)

    def value_and_jacobian(point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        target_value, target_jacobian = single_module_target19_value_jacobian(
            point, target, basis, source_oracle.control_derivatives
        )
        value = np.concatenate((
            target_value,
            np.asarray([gradient @ (point - source)], dtype=np.float64),
            null_basis.T @ (point - trial),
        ))
        jacobian = np.vstack((target_jacobian, gradient[None, :], null_basis.T))
        if value.shape != (44,) or jacobian.shape != (44, 44):
            raise AssertionError("target/time correction must remain square")
        return value, jacobian

    point = trial.copy()
    history: list[dict[str, Any]] = []
    success = False
    terminal_reason = "update_budget_exhausted"
    for update in range(maximum_updates + 1):
        if time.perf_counter() - started > maximum_seconds:
            terminal_reason = "per_module_time_budget_exhausted"
            break
        value, jacobian = value_and_jacobian(point)
        residual = float(np.max(np.abs(value)))
        history.append({"update": int(update), "residual_max_abs": residual})
        if residual < tolerance:
            success = True
            terminal_reason = "residual_tolerance_reached"
            break
        if update == maximum_updates:
            break
        try:
            step = np.linalg.solve(jacobian, -value)
        except np.linalg.LinAlgError:
            terminal_reason = "square_newton_jacobian_singular"
            break
        accepted_line = False
        for halvings in range(maximum_halvings + 1):
            factor = 0.5 ** halvings
            candidate = point + factor * step
            candidate_value, _ = value_and_jacobian(candidate)
            candidate_residual = float(np.max(np.abs(candidate_value)))
            if candidate_residual < residual:
                point = candidate
                history[-1].update({
                    "line_factor": float(factor),
                    "line_halvings": int(halvings),
                    "next_residual_max_abs": candidate_residual,
                })
                accepted_line = True
                break
        if not accepted_line:
            terminal_reason = "no_residual_decreasing_newton_line"
            break

    record.update({
        "trial_params": trial,
        "direction": direction,
        "history": history,
        "terminal_reason": terminal_reason,
        "candidate_params": point,
    })
    final_value, _ = value_and_jacobian(point)
    candidate_evaluation = source_oracle.evaluate(
        label, 0.0, 0.0, with_error_derivatives=True
    ) if np.array_equal(point, source) else None
    if candidate_evaluation is None:
        # The candidate is not part of the source oracle's registered parameter
        # library, so use a short-lived oracle while preserving the shared ledger.
        candidate_oracle = CachedModuleOracle(
            {"00": point if label == "00" else source_oracle.params["00"],
             "15": point if label == "15" else source_oracle.params["15"],
             "25": point if label == "25" else source_oracle.params["25"]},
            ledger=source_oracle.ledger,
        )
        candidate_evaluation = candidate_oracle.evaluate(
            label, 0.0, 0.0, with_error_derivatives=True
        )
    target_fro, target_phase = phase_aligned_frobenius(target, candidate_evaluation.unitary)
    target_metrics = m8.unitary_channel_metrics(target, candidate_evaluation.unitary)
    candidate_time = single_module_time(point, eta=eta)
    signs_match = bool(np.array_equal(_charged_signs(point), _charged_signs(source)))
    generators = {
        "rotation": generator_record(candidate_evaluation.unitary, candidate_evaluation.d_error_rotation),
        "twist": generator_record(candidate_evaluation.unitary, candidate_evaluation.d_error_twist),
    }
    maximum_generator = max(
        generators["rotation"]["traceless_physical_fro"],
        generators["twist"]["traceless_physical_fro"],
    )
    checks = {
        "newton_residual_lt_tolerance": bool(float(np.max(np.abs(final_value))) < tolerance),
        "target19_max_abs_lt_tolerance": bool(float(np.max(np.abs(final_value[:19])) < tolerance)),
        "source_sign_time_residual_lt_tolerance": bool(abs(float(final_value[19])) < tolerance),
        "phase_aligned_full8_target_fro_lt_1e-10": bool(target_fro < 1.0e-10),
        "charged_signs_match_source": signs_match,
        "exact_sumabs_time_difference_lt_1e-10": bool(abs(candidate_time - source_time) < 1.0e-10),
        "source_distance_le_0p1": bool(float(np.linalg.norm(point - source)) <= 0.1),
        "within_per_module_seconds": bool(time.perf_counter() - started <= maximum_seconds),
    }
    construction_accepted = bool(success and all(checks.values()))
    status = (
        "TARGET_TIME_MATCHED_UNCORRECTED_ABLATION"
        if construction_accepted and maximum_generator > 1.0e-6
        else "TARGET_TIME_MATCHED_DEGENERATE_ABLATION"
        if construction_accepted
        else "TARGET_TIME_ABLATION_REJECTED"
    )
    record.update({
        "status": status,
        "accepted": construction_accepted,
        "checks": checks,
        "final_system_max_abs": float(np.max(np.abs(final_value))),
        "target_phase_aligned_fro": target_fro,
        "target_aligning_phase_real": float(target_phase.real),
        "target_aligning_phase_imag": float(target_phase.imag),
        "target_half_diamond": float(target_metrics["half_diamond"]),
        "source_time_eta10": source_time,
        "candidate_time_eta10": candidate_time,
        "generators": generators,
        "maximum_traceless_physical_generator_fro": float(maximum_generator),
        "uncorrected_generator_distinguishable_at_1e-6": bool(maximum_generator > 1.0e-6),
        "elapsed_seconds": float(time.perf_counter() - started),
    })
    return record


def finite_grid_evaluation(
    oracle: CachedModuleOracle,
    words: Mapping[int, Sequence[str]],
    error_model: str,
    points: np.ndarray,
    ideal_words: Mapping[int, np.ndarray] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Evaluate every word separately at every finite error-grid point."""
    point_array = np.asarray(points, dtype=np.float64)
    if point_array.ndim != 2 or point_array.shape[1] != 2:
        raise ValueError("points must have shape (count,2)")
    table = np.empty((len(DEPTHS), len(point_array)), dtype=np.float64)
    records: list[dict[str, Any]] = []
    for depth_index, depth in enumerate(DEPTHS):
        labels = tuple(words[depth])
        ideal = (
            np.asarray(ideal_words[depth], dtype=np.complex128)
            if ideal_words is not None
            else compose_word(
                oracle, labels, (0.0, 0.0), error_model, with_error_derivatives=False
            )[0]
        )
        if ideal.shape != (m8.DIMENSION, m8.DIMENSION):
            raise ValueError("ideal word must be an 8x8 matrix")
        for point_index, eps in enumerate(point_array):
            actual, _, _ = compose_word(
                oracle, labels, eps, error_model, with_error_derivatives=False
            )
            table[depth_index, point_index] = m8.unitary_channel_metrics(
                ideal, actual
            )["half_diamond"]
        worst_index = int(np.argmax(table[depth_index]))
        worst_actual, _, _ = compose_word(
            oracle, labels, point_array[worst_index], error_model, with_error_derivatives=False
        )
        metrics = m8.unitary_channel_metrics(ideal, worst_actual)
        target_fro, _ = phase_aligned_frobenius(ideal, worst_actual)
        records.append({
            "depth": int(depth),
            "chronological_labels": list(labels),
            "point_count": int(len(point_array)),
            "max_grid_half_diamond": float(table[depth_index, worst_index]),
            "worst_grid_index": worst_index,
            "worst_epsilon": point_array[worst_index].tolist(),
            "worst_metrics": {key: float(value) for key, value in metrics.items()},
            "worst_phase_aligned_fro": target_fro,
        })
    return table, records


def small_error_scaling(
    oracle: CachedModuleOracle,
    words: Mapping[int, Sequence[str]],
    error_model: str,
    ideal_words: Mapping[int, np.ndarray] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Return fixed-ray small-error data without selecting rays after evaluation."""
    values = np.empty(
        (len(DEPTHS), len(SMALL_ERROR_RAYS), len(SMALL_ERROR_SIGNS), len(SMALL_ERROR_RADII)),
        dtype=np.float64,
    )
    records: list[dict[str, Any]] = []
    for depth_index, depth in enumerate(DEPTHS):
        labels = tuple(words[depth])
        ideal = (
            np.asarray(ideal_words[depth], dtype=np.complex128)
            if ideal_words is not None
            else compose_word(
                oracle, labels, (0.0, 0.0), error_model, with_error_derivatives=False
            )[0]
        )
        if ideal.shape != (m8.DIMENSION, m8.DIMENSION):
            raise ValueError("ideal word must be an 8x8 matrix")
        nominal, _, _ = compose_word(
            oracle, labels, (0.0, 0.0), error_model, with_error_derivatives=False
        )
        nominal_distance = m8.unitary_channel_metrics(ideal, nominal)["half_diamond"]
        floor = max(1.0e-14, float(nominal_distance))
        for ray_index, ray in enumerate(SMALL_ERROR_RAYS):
            direction = np.asarray(ray, dtype=np.float64)
            for sign_index, sign in enumerate(SMALL_ERROR_SIGNS):
                for radius_index, radius in enumerate(SMALL_ERROR_RADII):
                    actual, _, _ = compose_word(
                        oracle, labels, sign * radius * direction, error_model,
                        with_error_derivatives=False,
                    )
                    values[depth_index, ray_index, sign_index, radius_index] = (
                        m8.unitary_channel_metrics(ideal, actual)["half_diamond"]
                    )
                ray_values = values[depth_index, ray_index, sign_index]
                resolved = bool(np.all(ray_values > 10.0 * floor))
                exponent = None
                if resolved:
                    exponent = float(np.polyfit(np.log(SMALL_ERROR_RADII), np.log(ray_values), 1)[0])
                records.append({
                    "depth": int(depth),
                    "chronological_labels": list(labels),
                    "ray": direction.tolist(),
                    "sign": float(sign),
                    "radii": list(SMALL_ERROR_RADII),
                    "half_diamond_values": ray_values.tolist(),
                    "nominal_half_diamond": float(nominal_distance),
                    "resolution_floor": float(floor),
                    "resolved": resolved,
                    "loglog_exponent": exponent,
                    "quadratic_at_1p75_if_resolved": None if exponent is None else bool(exponent >= 1.75),
                })
    return values, records


def words_for_registered_depths() -> dict[int, tuple[str, ...]]:
    return {depth: word_labels(depth) for depth in DEPTHS}


def jsonable(value: Any) -> Any:
    """Convert NumPy data held in records into ordinary JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


__all__ = [
    "BudgetViolation", "CallLedger", "CachedModuleOracle", "DEPTHS", "ERROR_MODELS",
    "GRID_AXIS", "MODULE_LABELS", "PROFILE_COEFFICIENTS", "SMALL_ERROR_RADII",
    "SMALL_ERROR_RAYS", "SMALL_ERROR_SIGNS", "canonical_null_basis",
    "compose_word", "construct_target_time_ablation", "error_coefficients",
    "finite_grid_evaluation", "full8_target_basis19", "generator_record",
    "grid_points", "jsonable", "pairwise_noncommutation", "phase_aligned_frobenius",
    "registered_word_ideals", "small_error_scaling", "source_rows_to_params", "word_first_order_record",
    "word_labels", "word_resource_record", "words_for_registered_depths",
]
