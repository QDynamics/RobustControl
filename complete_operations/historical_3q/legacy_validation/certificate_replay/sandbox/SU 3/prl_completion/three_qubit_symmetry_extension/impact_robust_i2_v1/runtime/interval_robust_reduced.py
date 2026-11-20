"""Rigorous reduced 4+2 Library-B robust residual and interval Jacobian."""
from __future__ import annotations

from fractions import Fraction
from math import comb
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
TC2_RUNTIME = (EXT / "sq3b_r3_tc1_b3v_r2_cont_r6_r1a" / "repair_a1r" /
               "semantic_r1" / "tc2_int_r1" / "runtime")
sys.path.insert(0, str(TC2_RUNTIME))
from dyadic_interval import CI, DIContext, RI
from dyadic_bijet import (CJet, RJet, complex_constant, complex_from_real,
                          exp_minus_i_jet, real_constant, real_control,
                          sin_cos_jet)


LENGTH = 15
PARAMETER_COUNT = 44
FULL_ROWS = 41
DROPPED_ROWS = (6, 25, 39, 40)
INDEPENDENT_ROWS = tuple(i for i in range(FULL_ROWS) if i not in DROPPED_ROWS)


def zjet(ctx: DIContext, n: int) -> CJet:
    return complex_constant(ctx.complex(), n)


def eye(ctx: DIContext, size: int, n: int) -> list[list[CJet]]:
    out = [[zjet(ctx, n) for _ in range(size)] for _ in range(size)]
    for i in range(size):
        out[i][i] = complex_constant(ctx.complex(1), n)
    return out


def matmul(a: list[list[CJet]], b: list[list[CJet]]) -> list[list[CJet]]:
    rows, inner, cols = len(a), len(b), len(b[0])
    if len(a[0]) != inner:
        raise ValueError("matrix shape mismatch")
    ctx, n = a[0][0].ctx, a[0][0].n
    out = [[zjet(ctx, n) for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            value = zjet(ctx, n)
            for k in range(inner):
                value = value + a[i][k] * b[k][j]
            out[i][j] = value
    return out


def dagger(a: list[list[CJet]]) -> list[list[CJet]]:
    return [[a[j][i].conjugate() for j in range(len(a))]
            for i in range(len(a[0]))]


def single_rotation(theta: RJet, phi: RJet) -> list[list[CJet]]:
    st, ct = sin_cos_jet(theta / 2)
    sp, cp = sin_cos_jet(phi)
    e_minus = complex_from_real(cp, -sp)
    e_plus = complex_from_real(cp, sp)
    minus_i_st = complex_from_real(real_constant(theta.ctx.zero, theta.n), -st)
    cct = complex_from_real(ct)
    return [[cct, minus_i_st * e_minus],
            [minus_i_st * e_plus, cct]]


def sym3(single: list[list[CJet]]) -> list[list[CJet]]:
    ctx, n = single[0][0].ctx, single[0][0].n
    a, b, c, d = single[0][0], single[0][1], single[1][0], single[1][1]
    output = [[zjet(ctx, n) for _ in range(4)] for _ in range(4)]
    for column, k in enumerate((3, 2, 1, 0)):
        polynomial = [complex_constant(ctx.complex(1), n)]
        for u, v in [(a, c)] * (3 - k) + [(b, d)] * k:
            expanded = [zjet(ctx, n) for _ in range(len(polynomial) + 1)]
            for power, value in enumerate(polynomial):
                expanded[power] = expanded[power] + value * u
                expanded[power + 1] = expanded[power + 1] + value * v
            polynomial = expanded
        for row, ell in enumerate((3, 2, 1, 0)):
            ratio = Fraction(comb(3, k), comb(3, ell))
            scale = ctx.sqrt(ctx.from_fraction(ratio))
            output[row][column] = polynomial[ell] * scale
    return output


def hidden_rotation(single: list[list[CJet]]) -> list[list[CJet]]:
    return [[single[1][1], single[1][0]],
            [single[0][1], single[0][0]]]


def diagonal(entries: list[CJet]) -> list[list[CJet]]:
    ctx, n = entries[0].ctx, entries[0].n
    out = [[zjet(ctx, n) for _ in entries] for _ in entries]
    for i, value in enumerate(entries):
        out[i][i] = value
    return out


def endpoint_blocks(params: list[RI], selected: tuple[int, ...]) -> tuple[list[list[CJet]], list[list[CJet]]]:
    if len(params) != PARAMETER_COUNT or len(selected) != 37:
        raise ValueError("wrong parameter or selected-coordinate count")
    ctx, n = params[0].ctx, len(selected)
    local = {global_index: j for j, global_index in enumerate(selected)}
    theta = [real_control(params[i], n, local.get(i), 0) for i in range(LENGTH)]
    phi = [real_constant(ctx.zero, n)]
    phi.extend(real_control(params[i], n, local.get(i), None)
               for i in range(LENGTH, 2 * LENGTH - 1))
    chi = [real_control(params[i], n, local.get(i), 1)
           for i in range(2 * LENGTH - 1, PARAMETER_COUNT)]

    visible = eye(ctx, 4, n)
    hidden = eye(ctx, 2, n)
    for layer in range(LENGTH):
        single = single_rotation(theta[layer], phi[layer])
        rv = sym3(single)
        rh = hidden_rotation(single)
        tv = diagonal([exp_minus_i_jet(3 * chi[layer]),
                       exp_minus_i_jet(-chi[layer]),
                       exp_minus_i_jet(-chi[layer]),
                       exp_minus_i_jet(3 * chi[layer])])
        th = diagonal([exp_minus_i_jet(-chi[layer]),
                       exp_minus_i_jet(-chi[layer])])
        visible = matmul(matmul(rv, tv), visible)
        hidden = matmul(matmul(rh, th), hidden)
    return visible, hidden


def quotient_value_dp(numerator: CJet, denominator: CJet) -> tuple[CI, tuple[CI, ...]]:
    value = numerator.v / denominator.v
    denominator2 = denominator.v * denominator.v
    dp = tuple((numerator.dp[j] * denominator.v - numerator.v * denominator.dp[j]) /
               denominator2 for j in range(numerator.n))
    return value, dp


def error_generator(endpoint: list[list[CJet]], channel: int) -> tuple[list[list[CI]], list[list[tuple[CI, ...]]]]:
    size, n, ctx = len(endpoint), endpoint[0][0].n, endpoint[0][0].ctx
    values = [[ctx.complex() for _ in range(size)] for _ in range(size)]
    derivatives = [[tuple(ctx.complex() for _ in range(n)) for _ in range(size)]
                   for _ in range(size)]
    imag = ctx.complex(0, 1)
    for i in range(size):
        for j in range(size):
            value = ctx.complex()
            dp = [ctx.complex() for _ in range(n)]
            for k in range(size):
                left = endpoint[k][i]
                right = endpoint[k][j]
                value = value + left.v.conjugate() * right.de[channel]
                for q in range(n):
                    dp[q] = (dp[q] + left.dp[q].conjugate() * right.de[channel]
                             + left.v.conjugate() * right.cross[q][channel])
            values[i][j] = imag * value
            derivatives[i][j] = tuple(imag * item for item in dp)
    return values, derivatives


def generator_coordinates(values: list[list[CI]], derivatives: list[list[tuple[CI, ...]]]) -> tuple[list[RI], list[tuple[RI, ...]]]:
    coordinates: list[RI] = []
    jacobian_rows: list[tuple[RI, ...]] = []
    size = len(values)
    for i in range(size):
        coordinates.append(values[i][i].re)
        jacobian_rows.append(tuple(x.re for x in derivatives[i][i]))
    for i in range(size):
        for j in range(i + 1, size):
            coordinates.extend((values[i][j].re, values[i][j].im))
            jacobian_rows.extend((tuple(x.re for x in derivatives[i][j]),
                                  tuple(x.im for x in derivatives[i][j])))
    return coordinates, jacobian_rows


def evaluate(params: list[RI], selected: tuple[int, ...]) -> tuple[list[RI], list[list[RI]], RI, RI]:
    visible, hidden = endpoint_blocks(params, selected)
    tau = hidden[0][0] + hidden[1][1]
    numerators = [hidden[1][0] + hidden[0][1],
                  hidden[1][0] - hidden[0][1],
                  hidden[0][0] - hidden[1][1]]
    quotients = [quotient_value_dp(value, tau) for value in numerators]
    values = [-quotients[0][0].im, quotients[1][0].re, -quotients[2][0].im]
    rows = [tuple(-x.im for x in quotients[0][1]),
            tuple(x.re for x in quotients[1][1]),
            tuple(-x.im for x in quotients[2][1])]
    for channel in range(2):
        vv, vd = error_generator(visible, channel)
        hv, hd = error_generator(hidden, channel)
        vvalues, vrows = generator_coordinates(vv, vd)
        hvalues, hrows = generator_coordinates(hv, hd)
        # The qualified reduced runtime uses hidden coordinates
        # (K_00, Re K_01, Im K_01); the second diagonal is omitted because
        # the full generator is traceless only after visible and hidden
        # multiplicities are combined.
        hidden_indices = (0, 2, 3)
        values.extend(vvalues + [hvalues[i] for i in hidden_indices])
        rows.extend(vrows + [hrows[i] for i in hidden_indices])
    if len(values) != FULL_ROWS or len(rows) != FULL_ROWS:
        raise AssertionError("wrong robust residual dimension")
    independent_values = [values[i] for i in INDEPENDENT_ROWS]
    independent_jacobian = [[rows[i][j] for j in range(37)] for i in INDEPENDENT_ROWS]
    # Input |000> is visible column 3 in the fixed Dicke ordering.  Across
    # qubit 1 | qubits 2,3, one 2x2 flattening minor is
    # U_000,000 U_111,000 - U_011,000 U_100,000.  A nonzero enclosure proves
    # Schmidt rank two.  Permutation symmetry then gives the same conclusion
    # for every one-versus-two cut.
    inv_sqrt3 = params[0].ctx.one / params[0].ctx.sqrt(
        params[0].ctx.from_fraction(Fraction(3)))
    witness = (visible[3][3].v * visible[0][3].v
               - (visible[1][3].v * inv_sqrt3) *
                 (visible[2][3].v * inv_sqrt3))
    return independent_values, independent_jacobian, tau.v.abs(), witness.abs()
