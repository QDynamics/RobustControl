"""Schur-sector propagation for collective rotations and full-space H_ZZ twists.

The physical layer convention is the legacy convention

    P_l = R_{phi_l}(theta_l) T(chi_l),

so ``T`` acts first and a word is ``U = P_L ... P_1``.  Unlike a fixed-j
``Jz**2`` implementation, this module retains the sector-dependent constant
in ``H_ZZ = 2 Jz**2 - (N/2) I``.  Consequently relative phases between
different Schur sectors remain in every returned block.

This is a representation and diagnostic module.  It contains no optimizer
and makes no reachability claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb, sqrt
from typing import Iterable, Mapping

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class Representation:
    """One matrix representation of the collective generators."""

    N: int
    label: str
    dimension: int
    multiplicity: int
    two_j: int | None
    Jx: Array
    Jy: Array
    Jz: Array
    Hzz: Array

    @property
    def j(self) -> float | None:
        return None if self.two_j is None else self.two_j / 2.0


@dataclass(frozen=True)
class Propagation:
    """Nominal propagator, scale derivatives, and input-frame generators."""

    U: Array
    derivative_R: Array
    derivative_T: Array
    generator_R: Array
    generator_T: Array


def allowed_two_j(N: int) -> tuple[int, ...]:
    """Return the spin labels ``2j`` in descending order."""
    if not isinstance(N, int) or isinstance(N, bool) or N < 1:
        raise ValueError("N must be a positive integer")
    return tuple(range(N, -1, -2))


def schur_multiplicity(N: int, two_j: int) -> int:
    """Multiplicity of spin j in N spin-1/2 particles."""
    if two_j not in allowed_two_j(N):
        raise ValueError(f"2j={two_j} is not an allowed sector for N={N}")
    k = (N - two_j) // 2
    return comb(N, k) - (comb(N, k - 1) if k else 0)


def mean_hzz_charge_exact(N: int, two_j: int) -> Fraction:
    """Return Tr(H_ZZ|j)/(2j+1) as an exact rational number."""
    if two_j not in allowed_two_j(N):
        raise ValueError(f"2j={two_j} is not an allowed sector for N={N}")
    # 2 j(j+1)/3 - N/2 = [two_j(two_j+2)-3N]/6.
    return Fraction(two_j * (two_j + 2) - 3 * N, 6)


def trace_hzz_charge_exact(N: int, two_j: int) -> Fraction:
    """Return Tr(H_ZZ|j) as an exact rational number."""
    return (two_j + 1) * mean_hzz_charge_exact(N, two_j)


def sector_table(N: int) -> list[dict[str, int | float | str]]:
    """Return a JSON-ready analytic Schur table for N qubits."""
    table: list[dict[str, int | float | str]] = []
    for two_j in allowed_two_j(N):
        multiplicity = schur_multiplicity(N, two_j)
        dimension = two_j + 1
        mean_charge = mean_hzz_charge_exact(N, two_j)
        trace_charge = dimension * mean_charge
        table.append(
            {
                "two_j": two_j,
                "j": two_j / 2.0,
                "irrep_dimension": dimension,
                "multiplicity": multiplicity,
                "full_sector_dimension": dimension * multiplicity,
                "mean_hzz_charge_fraction": str(mean_charge),
                "mean_hzz_charge": float(mean_charge),
                "trace_hzz_charge_fraction": str(trace_charge),
                "trace_hzz_charge": float(trace_charge),
                "multiplicity_weighted_trace_hzz_fraction": str(
                    multiplicity * trace_charge
                ),
            }
        )
    return table


def spin_irrep(N: int, two_j: int) -> Representation:
    """Construct one spin-j irrep in descending-m order."""
    if two_j not in allowed_two_j(N):
        raise ValueError(f"2j={two_j} is not an allowed sector for N={N}")
    dimension = two_j + 1
    j = two_j / 2.0
    m = j - np.arange(dimension, dtype=np.float64)
    raising = np.zeros((dimension, dimension), dtype=np.complex128)
    for column in range(1, dimension):
        m_column = m[column]
        raising[column - 1, column] = sqrt(
            (j - m_column) * (j + m_column + 1.0)
        )
    lowering = raising.conj().T
    Jx = (raising + lowering) / 2.0
    Jy = (raising - lowering) / (2.0j)
    Jz = np.diag(m).astype(np.complex128)
    identity = np.eye(dimension, dtype=np.complex128)
    Hzz = 2.0 * (Jz @ Jz) - (N / 2.0) * identity
    return Representation(
        N=N,
        label=f"spin_{two_j}/2",
        dimension=dimension,
        multiplicity=schur_multiplicity(N, two_j),
        two_j=two_j,
        Jx=Jx,
        Jy=Jy,
        Jz=Jz,
        Hzz=Hzz,
    )


def schur_representations(N: int) -> tuple[Representation, ...]:
    """Return one copy of every distinct spin irrep."""
    return tuple(spin_irrep(N, two_j) for two_j in allowed_two_j(N))


def full_tensor_representation(N: int, *, maximum_N: int = 6) -> Representation:
    """Construct collective generators in the full 2**N tensor space.

    The explicit guard prevents accidental use as a large-N implementation.
    """
    if not isinstance(N, int) or isinstance(N, bool) or N < 1:
        raise ValueError("N must be a positive integer")
    if N > maximum_N:
        raise ValueError(f"explicit tensor construction is limited to N<={maximum_N}")
    identity_2 = np.eye(2, dtype=np.complex128)
    paulis = (
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
        np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
        np.diag([1.0, -1.0]).astype(np.complex128),
    )

    def collective(pauli: Array) -> Array:
        result = np.zeros((2**N, 2**N), dtype=np.complex128)
        for site in range(N):
            factors = [identity_2] * N
            factors[site] = pauli
            term = factors[0]
            for factor in factors[1:]:
                term = np.kron(term, factor)
            result += term / 2.0
        return result

    Jx, Jy, Jz = (collective(pauli) for pauli in paulis)
    identity = np.eye(2**N, dtype=np.complex128)
    # Construct H_ZZ directly from Pauli pairs.  The runner separately checks
    # its equality to 2 Jz^2-NI/2, avoiding use of that identity on both sides
    # of the tensor/Schur comparison.
    Hzz = np.zeros_like(identity)
    pauli_z = paulis[2]
    for first in range(N):
        for second in range(first + 1, N):
            factors = [identity_2] * N
            factors[first] = pauli_z
            factors[second] = pauli_z
            term = factors[0]
            for factor in factors[1:]:
                term = np.kron(term, factor)
            Hzz += term
    return Representation(
        N=N,
        label="full_tensor_product",
        dimension=2**N,
        multiplicity=1,
        two_j=None,
        Jx=Jx,
        Jy=Jy,
        Jz=Jz,
        Hzz=Hzz,
    )


def _unitary_from_hermitian(generator: Array, angle: float) -> Array:
    eigenvalues, eigenvectors = np.linalg.eigh(generator)
    return (eigenvectors * np.exp(-1.0j * angle * eigenvalues)) @ eigenvectors.conj().T


def rotation(rep: Representation, theta: float, phase: float) -> Array:
    """Return exp[-i theta(cos(phi)Jx+sin(phi)Jy)]."""
    axis = np.cos(phase) * rep.Jx + np.sin(phase) * rep.Jy
    return _unitary_from_hermitian(axis, theta)


def twist(rep: Representation, chi: float) -> Array:
    """Return exp(-i chi H_ZZ), retaining the sector-dependent scalar."""
    return _unitary_from_hermitian(rep.Hzz, chi)


def normalize_word(rows: Iterable[Iterable[float]]) -> Array:
    """Validate rows in the order ``(theta, phase, chi)``."""
    word = np.asarray(tuple(tuple(row) for row in rows), dtype=np.float64)
    if word.ndim != 2 or word.shape[1:] != (3,) or word.shape[0] < 1:
        raise ValueError("word must have shape (L,3) with rows (theta,phase,chi)")
    if not np.all(np.isfinite(word)):
        raise ValueError("word contains a non-finite control")
    return word


def propagate(rep: Representation, rows: Iterable[Iterable[float]]) -> Propagation:
    """Propagate a word and exact derivatives for both common scale errors."""
    word = normalize_word(rows)
    identity = np.eye(rep.dimension, dtype=np.complex128)
    U = identity.copy()
    derivative_R = np.zeros_like(identity)
    derivative_T = np.zeros_like(identity)
    for theta_value, phase_value, chi_value in word:
        axis = np.cos(phase_value) * rep.Jx + np.sin(phase_value) * rep.Jy
        R = _unitary_from_hermitian(axis, float(theta_value))
        T = _unitary_from_hermitian(rep.Hzz, float(chi_value))
        layer = R @ T
        derivative_layer_R = (-1.0j * theta_value) * (axis @ R) @ T
        derivative_layer_T = R @ ((-1.0j * chi_value) * (rep.Hzz @ T))
        derivative_R = layer @ derivative_R + derivative_layer_R @ U
        derivative_T = layer @ derivative_T + derivative_layer_T @ U
        U = layer @ U
    generator_R = 1.0j * U.conj().T @ derivative_R
    generator_T = 1.0j * U.conj().T @ derivative_T
    return Propagation(
        U=U,
        derivative_R=derivative_R,
        derivative_T=derivative_T,
        generator_R=generator_R,
        generator_T=generator_T,
    )


def propagate_schur_blocks(
    N: int, rows: Iterable[Iterable[float]]
) -> dict[int, Propagation]:
    """Propagate every distinct irrep, keyed by the integer label ``2j``."""
    word = normalize_word(rows)
    return {rep.two_j: propagate(rep, word) for rep in schur_representations(N)}  # type: ignore[misc]


def multiplicity_weighted_trace(
    N: int, blocks: Mapping[int, Array]
) -> complex:
    """Trace of the represented full-space operator from one-copy blocks."""
    expected = set(allowed_two_j(N))
    if set(blocks) != expected:
        raise ValueError(f"block labels {sorted(blocks)} do not equal {sorted(expected)}")
    return sum(
        schur_multiplicity(N, two_j) * np.trace(blocks[two_j])
        for two_j in allowed_two_j(N)
    )


def expanded_schur_eigenvalues(N: int, blocks: Mapping[int, Array]) -> Array:
    """Return block eigenvalues repeated according to Schur multiplicity."""
    expected = set(allowed_two_j(N))
    if set(blocks) != expected:
        raise ValueError(f"block labels {sorted(blocks)} do not equal {sorted(expected)}")
    values: list[complex] = []
    for two_j in allowed_two_j(N):
        eigenvalues = np.linalg.eigvals(blocks[two_j])
        for _ in range(schur_multiplicity(N, two_j)):
            values.extend(eigenvalues.tolist())
    return np.asarray(values, dtype=np.complex128)


def twist_trace_invariant(N: int, two_j: int, chi_sum: float) -> float:
    """Analytic Tr(K_T|j) for a common multiplicative twist error."""
    return float(trace_hzz_charge_exact(N, two_j)) * float(chi_sum)


def same_sign_scalar_residual_lower_bound(N: int, chi_sum: float) -> float:
    """Normalized Frobenius lower bound after removing one full-space scalar.

    Since H_ZZ is traceless on the full tensor space, K_T is also traceless.
    The sector means alone therefore give the stated lower bound.  It is a
    necessary-condition diagnostic; reaching zero does not establish a root.
    """
    total = 0.0
    for two_j in allowed_two_j(N):
        dimension = two_j + 1
        multiplicity = schur_multiplicity(N, two_j)
        mean = float(mean_hzz_charge_exact(N, two_j)) * float(chi_sum)
        total += multiplicity * dimension * mean * mean
    return sqrt(total / (2**N))
