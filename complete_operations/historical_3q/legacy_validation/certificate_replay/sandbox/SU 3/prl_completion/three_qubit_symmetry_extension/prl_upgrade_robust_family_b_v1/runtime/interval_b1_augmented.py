"""Outward interval oracle for the B1 constraint-plus-visible differential."""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
I2_RUNTIME = ROOT.parent / "impact_robust_i2_v1" / "runtime"
sys.path.insert(0, str(I2_RUNTIME))
from dyadic_bijet import real_constant, real_control  # noqa: E402
from interval_robust_reduced import (DIContext, INDEPENDENT_ROWS, LENGTH, PARAMETER_COUNT,  # noqa: E402
                                     diagonal, error_generator, exp_minus_i_jet,
                                     generator_coordinates, hidden_rotation, matmul,
                                     quotient_value_dp, single_rotation, sym3, eye)


VISIBLE_ROWS = (0, 3, 6, 12, 5, 1, 9)


def endpoint_blocks_any(params, selected):
    if len(params) != PARAMETER_COUNT or len(set(selected)) != len(selected):
        raise ValueError("wrong parameter or selected-coordinate convention")
    ctx, n = params[0].ctx, len(selected)
    local = {global_index: j for j, global_index in enumerate(selected)}
    theta = [real_control(params[i], n, local.get(i), 0) for i in range(LENGTH)]
    phi = [real_constant(ctx.zero, n)]
    phi.extend(real_control(params[i], n, local.get(i), None)
               for i in range(LENGTH, 2 * LENGTH - 1))
    chi = [real_control(params[i], n, local.get(i), 1)
           for i in range(2 * LENGTH - 1, PARAMETER_COUNT)]
    visible, hidden = eye(ctx, 4, n), eye(ctx, 2, n)
    for layer in range(LENGTH):
        single = single_rotation(theta[layer], phi[layer])
        rv, rh = sym3(single), hidden_rotation(single)
        tv = diagonal([exp_minus_i_jet(3 * chi[layer]),
                       exp_minus_i_jet(-chi[layer]),
                       exp_minus_i_jet(-chi[layer]),
                       exp_minus_i_jet(3 * chi[layer])])
        th = diagonal([exp_minus_i_jet(-chi[layer]),
                       exp_minus_i_jet(-chi[layer])])
        visible = matmul(matmul(rv, tv), visible)
        hidden = matmul(matmul(rh, th), hidden)
    return visible, hidden


def robust_rows(visible, hidden):
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
        hidden_indices = (0, 2, 3)
        values.extend(vvalues + [hvalues[i] for i in hidden_indices])
        rows.extend(vrows + [hrows[i] for i in hidden_indices])
    return ([values[i] for i in INDEPENDENT_ROWS],
            [list(rows[i]) for i in INDEPENDENT_ROWS], tau)


def visible_tangent_rows(visible, hidden):
    ctx, n = visible[0][0].ctx, visible[0][0].n
    half = ctx.from_fraction(__import__("fractions").Fraction(1, 2))
    lam = (hidden[0][0] + hidden[1][1]) * half
    gate = [[None for _ in range(4)] for _ in range(4)]
    dgate = [[None for _ in range(4)] for _ in range(4)]
    for i in range(4):
        for j in range(4):
            gate[i][j], dgate[i][j] = quotient_value_dp(visible[i][j], lam)
    imag = ctx.complex(0, 1)
    coordinate_rows = []
    for i in range(4):
        row = []
        for q in range(n):
            value = ctx.complex()
            for k in range(4):
                value = value + gate[k][i].conjugate() * dgate[k][i][q]
            row.append((imag * value).re)
        coordinate_rows.append(row)
    for i in range(4):
        for j in range(i + 1, 4):
            real_row, imag_row = [], []
            for q in range(n):
                value = ctx.complex()
                for k in range(4):
                    value = value + gate[k][i].conjugate() * dgate[k][j][q]
                value = imag * value
                real_row.append(value.re); imag_row.append(value.im)
            coordinate_rows.extend((real_row, imag_row))
    return coordinate_rows


def evaluate_augmented(params, selected):
    visible, hidden = endpoint_blocks_any(params, selected)
    values, constraint_rows, tau = robust_rows(visible, hidden)
    visible_rows = visible_tangent_rows(visible, hidden)
    augmented = constraint_rows + [visible_rows[index] for index in VISIBLE_ROWS]
    return values, constraint_rows, visible_rows, augmented, tau

