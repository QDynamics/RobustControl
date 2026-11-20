"""Arbitrary-active-coordinate interval jets for the robust Library-B module.

This create-new adapter reuses the qualified outward-rounded primitive algebra
but removes the historical ``len(selected) == 37`` restriction.  It is used to
differentiate the 37 independent robust constraints and the visible endpoint
with respect to all 44 controls in one interval evaluation.
"""
from __future__ import annotations

from fractions import Fraction
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OVERLAY = HERE.parent
PRL = OVERLAY.parent
EXT = PRL / "three_qubit_symmetry_extension"
I2_RUNTIME = EXT / "impact_robust_i2_v1" / "runtime"
sys.path.insert(0, str(I2_RUNTIME))

from interval_robust_reduced import (  # noqa: E402
    LENGTH,
    PARAMETER_COUNT,
    FULL_ROWS,
    INDEPENDENT_ROWS,
    diagonal,
    error_generator,
    exp_minus_i_jet,
    generator_coordinates,
    hidden_rotation,
    eye,
    matmul,
    quotient_value_dp,
    real_constant,
    real_control,
    single_rotation,
    sym3,
)


def endpoint_blocks_active(params, active: tuple[int, ...]):
    """Return visible/hidden blocks with jets for arbitrary distinct controls."""
    if len(params) != PARAMETER_COUNT:
        raise ValueError("wrong parameter count")
    if len(set(active)) != len(active) or any(not 0 <= i < PARAMETER_COUNT for i in active):
        raise ValueError("invalid active coordinates")
    ctx, n = params[0].ctx, len(active)
    local = {global_index: j for j, global_index in enumerate(active)}
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


def evaluate_active(params, active: tuple[int, ...]):
    """Return the 37 robust residuals/Jacobian plus endpoint block jets."""
    visible, hidden = endpoint_blocks_active(params, active)
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
    if len(values) != FULL_ROWS or len(rows) != FULL_ROWS:
        raise AssertionError("wrong robust residual dimension")
    independent_values = [values[i] for i in INDEPENDENT_ROWS]
    independent_jacobian = [[rows[i][j] for j in range(len(active))]
                            for i in INDEPENDENT_ROWS]
    return independent_values, independent_jacobian, visible, hidden


def visible_interface_with_derivatives(visible, hidden):
    """Divide out the scalar hidden endpoint chart, retaining all jet columns."""
    lam = (hidden[0][0] + hidden[1][1]) * hidden[0][0].ctx.from_fraction(Fraction(1, 2))
    if lam.v.abs().lo <= 0:
        raise ValueError("hidden scalar denominator chart")
    values = [[None for _ in range(4)] for _ in range(4)]
    derivatives = [[None for _ in range(4)] for _ in range(4)]
    for i in range(4):
        for j in range(4):
            values[i][j], derivatives[i][j] = quotient_value_dp(visible[i][j], lam)
    return values, derivatives, lam.v.abs()
