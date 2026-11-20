"""Fast exact-Schur reduced Library-B runtime on the 4+2 endpoint blocks."""
from __future__ import annotations

import sys
from math import comb
from pathlib import Path

import mpmath as mp


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXT = ROOT.parent
R2 = EXT / "sq3b_endpoint_family_repair_r1a_r2"
sys.path.insert(0, str(R2))
from runtime.endpoint_runtime_high_precision import (dT_dchi, r_phi,  # noqa: E402
                                                     dr_dphi, dr_dtheta)
from robust_word_runtime import (I, embed_i1_seed, hidden_residual,  # noqa: E402
                                 parameter_count, trace2, unpack)


SIGMA_X = mp.matrix([[0, 1], [1, 0]])


def sym3_and_derivative(single, dsingle):
    """Symmetric cube in normalized Dicke order k=3,2,1,0."""
    output = mp.matrix(4, 4)
    derivative = mp.matrix(4, 4)
    a, b, c, d = single[0, 0], single[0, 1], single[1, 0], single[1, 1]
    da, db, dc, dd = dsingle[0, 0], dsingle[0, 1], dsingle[1, 0], dsingle[1, 1]
    for column, k in enumerate((3, 2, 1, 0)):
        polynomial = [(mp.mpc(1), mp.mpc(0))]
        factors = [((a, c), (da, dc))] * (3 - k) + [((b, d), (db, dd))] * k
        for (u, v), (du, dv) in factors:
            expanded = [(mp.mpc(0), mp.mpc(0)) for _ in range(len(polynomial) + 1)]
            for power, (value, dvalue) in enumerate(polynomial):
                expanded[power] = (expanded[power][0] + value * u,
                                   expanded[power][1] + dvalue * u + value * du)
                expanded[power + 1] = (expanded[power + 1][0] + value * v,
                                       expanded[power + 1][1] + dvalue * v + value * dv)
            polynomial = expanded
        for row, ell in enumerate((3, 2, 1, 0)):
            scale = mp.sqrt(mp.mpf(comb(3, k)) / comb(3, ell))
            output[row, column] = scale * polynomial[ell][0]
            derivative[row, column] = scale * polynomial[ell][1]
    return output, derivative


def layer_blocks(theta, phi, chi):
    single = r_phi(theta, phi)
    dtheta_single = dr_dtheta(theta, phi)
    dphi_single = dr_dphi(theta, phi)
    visible_r, visible_dtheta = sym3_and_derivative(single, dtheta_single)
    _, visible_dphi = sym3_and_derivative(single, dphi_single)
    hidden_r = SIGMA_X * single * SIGMA_X
    hidden_dtheta = SIGMA_X * dtheta_single * SIGMA_X
    hidden_dphi = SIGMA_X * dphi_single * SIGMA_X
    visible_t = mp.diag([mp.exp(-3 * I * chi), mp.exp(I * chi),
                         mp.exp(I * chi), mp.exp(-3 * I * chi)])
    visible_dt = mp.diag([-3 * I * visible_t[0, 0], I * visible_t[1, 1],
                          I * visible_t[2, 2], -3 * I * visible_t[3, 3]])
    hidden_t = mp.exp(I * chi) * mp.eye(2)
    hidden_dt = I * hidden_t
    return {
        "visible": visible_r * visible_t,
        "hidden": hidden_r * hidden_t,
        "visible_theta": visible_dtheta * visible_t,
        "visible_phi": visible_dphi * visible_t,
        "visible_chi": visible_r * visible_dt,
        "hidden_theta": hidden_dtheta * hidden_t,
        "hidden_phi": hidden_dphi * hidden_t,
        "hidden_chi": hidden_r * hidden_dt,
    }


def product_and_derivatives(params, length):
    theta, phi, chi = unpack(params, length)
    layers = [layer_blocks(theta[index], phi[index], chi[index]) for index in range(length)]

    def one_sector(key, size):
        factors = [layer[key] for layer in layers]
        prefix = [mp.eye(size)]
        for factor in factors:
            prefix.append(factor * prefix[-1])
        suffix = [None] * (length + 1)
        suffix[length] = mp.eye(size)
        for layer in range(length - 1, -1, -1):
            suffix[layer] = suffix[layer + 1] * factors[layer]
        derivatives = []
        for layer in range(length):
            derivatives.append(suffix[layer + 1] * layers[layer][key + "_theta"] * prefix[layer])
        for layer in range(1, length):
            derivatives.append(suffix[layer + 1] * layers[layer][key + "_phi"] * prefix[layer])
        for layer in range(length):
            derivatives.append(suffix[layer + 1] * layers[layer][key + "_chi"] * prefix[layer])
        return prefix[-1], derivatives

    visible, dvisible = one_sector("visible", 4)
    hidden, dhidden = one_sector("hidden", 2)
    return visible, hidden, dvisible, dhidden


def error_coordinates(generator_visible, generator_hidden):
    coordinates = [mp.re(generator_visible[i, i]) for i in range(4)]
    for i in range(4):
        for j in range(i + 1, 4):
            coordinates.extend((mp.re(generator_visible[i, j]), mp.im(generator_visible[i, j])))
    coordinates.extend((mp.re(generator_hidden[0, 0]),
                        mp.re(generator_hidden[0, 1]),
                        mp.im(generator_hidden[0, 1])))
    return mp.matrix(coordinates)


def evaluate_reduced(params, dps=45, length=15, anchors=None):
    with mp.workdps(dps):
        point = [mp.mpf(value) for value in params]
        theta, _, chi = unpack(point, length)
        visible, hidden, dvisible, dhidden = product_and_derivatives(point, length)
        closure, trace_abs = hidden_residual(hidden)
        damp_v, damp_h = mp.zeros(4), mp.zeros(2)
        dtwist_v, dtwist_h = mp.zeros(4), mp.zeros(2)
        for layer in range(length):
            damp_v += theta[layer] * dvisible[layer]
            damp_h += theta[layer] * dhidden[layer]
            index = 2 * length - 1 + layer
            dtwist_v += chi[layer] * dvisible[index]
            dtwist_h += chi[layer] * dhidden[index]
        amp_v, amp_h = I * visible.H * damp_v, I * hidden.H * damp_h
        twist_v, twist_h = I * visible.H * dtwist_v, I * hidden.H * dtwist_h
        amp = error_coordinates(amp_v, amp_h)
        twist = error_coordinates(twist_v, twist_h)
        gate = visible / (trace2(hidden) / 2)
        residual = list(closure) + list(amp) + list(twist)
        if anchors is not None:
            values = (mp.re(gate[0, 0]), mp.im(gate[0, 0]), mp.re(gate[0, 1]))
            residual.extend(values[index] - anchors[index] for index in range(3))
        return mp.matrix(residual), {
            "visible": visible, "hidden": hidden, "visible_gate": gate,
            "amp_visible_generator": amp_v, "amp_hidden_generator": amp_h,
            "twist_visible_generator": twist_v, "twist_hidden_generator": twist_h,
            "amp_coordinates": amp, "twist_coordinates": twist,
            "trace_abs": trace_abs,
        }

