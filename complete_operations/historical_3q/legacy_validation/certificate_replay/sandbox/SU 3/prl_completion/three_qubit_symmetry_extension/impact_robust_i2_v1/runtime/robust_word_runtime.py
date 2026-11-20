"""Optimized arbitrary-length Library-B endpoint and two error generators."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mpmath as mp


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXT = ROOT.parent
R2 = EXT / "sq3b_endpoint_family_repair_r1a_r2"
TC2_INPUT = (EXT / "sq3b_r3_tc1_b3v_r2_cont_r6_r1a" / "repair_a1r" /
             "semantic_r1" / "tc2_int_r1" / "inputs" / "refined_center.json")
SEED = EXT / "sq3b_symmetric_target_family_r3" / "results" / "accepted_seed.json"
sys.path.insert(0, str(R2))
from runtime.high_precision_context import ctx as get_ctx  # noqa: E402
from runtime.endpoint_runtime_high_precision import (R_block, T_block,  # noqa: E402
                                                     dR_dphi, dR_dtheta,
                                                     dT_dchi)


I = mp.mpc(0, 1)
SIGMAS = (mp.matrix([[0, 1], [1, 0]]),
          mp.matrix([[0, -I], [I, 0]]),
          mp.matrix([[1, 0], [0, -1]]))


def parameter_count(length: int) -> int:
    return 3 * length - 1


def unpack(params, length):
    if len(params) != parameter_count(length):
        raise ValueError("wrong Library-B parameter count")
    theta = list(params[:length])
    phi = [mp.mpf("0")] + list(params[length:2 * length - 1])
    chi = list(params[2 * length - 1:3 * length - 1])
    return theta, phi, chi


def schur_matrix():
    s2, s3, s6 = mp.sqrt(2), mp.sqrt(3), mp.sqrt(6)
    def cv(bits):
        vector = mp.zeros(8, 1)
        vector[4 * bits[0] + 2 * bits[1] + bits[2]] = 1
        return vector
    columns = [cv((1, 1, 1)),
               (cv((1, 1, 0)) + cv((1, 0, 1)) + cv((0, 1, 1))) / s3,
               (cv((1, 0, 0)) + cv((0, 1, 0)) + cv((0, 0, 1))) / s3,
               cv((0, 0, 0)),
               (cv((0, 1, 1)) - cv((1, 0, 1))) / s2,
               (cv((0, 1, 1)) + cv((1, 0, 1)) - 2 * cv((1, 1, 0))) / s6,
               (cv((0, 1, 0)) - cv((1, 0, 0))) / s2,
               -(cv((1, 0, 0)) + cv((0, 1, 0)) - 2 * cv((0, 0, 1))) / s6]
    transform = mp.matrix(8, 8)
    for j, vector in enumerate(columns):
        for i in range(8):
            transform[i, j] = vector[i]
    return transform


def blocks(matrix, transform):
    schur = transform.H * matrix * transform
    visible = mp.matrix([[schur[i, j] for j in range(4)] for i in range(4)])
    hidden = mp.matrix([[schur[4, 4], schur[4, 6]],
                        [schur[6, 4], schur[6, 6]]])
    return visible, hidden


def word_and_derivatives(params, context, length):
    theta, phi, chi = unpack(params, length)
    factors = []
    derivatives = []
    for layer in range(length):
        rotation = R_block(theta[layer], phi[layer])
        twisting = T_block(chi[layer])
        factor = rotation * twisting
        factors.append(factor)
        derivatives.append({
            "theta": dR_dtheta(theta[layer], phi[layer]) * twisting,
            "phi": dR_dphi(theta[layer], phi[layer]) * twisting,
            "chi": rotation * dT_dchi(chi[layer]),
        })
    prefix = [mp.eye(8)]
    for factor in factors:
        prefix.append(factor * prefix[-1])
    suffix = [None] * (length + 1)
    suffix[length] = mp.eye(8)
    for layer in range(length - 1, -1, -1):
        suffix[layer] = suffix[layer + 1] * factors[layer]
    all_derivatives = []
    for layer in range(length):
        all_derivatives.append(suffix[layer + 1] * derivatives[layer]["theta"] * prefix[layer])
    for layer in range(1, length):
        all_derivatives.append(suffix[layer + 1] * derivatives[layer]["phi"] * prefix[layer])
    for layer in range(length):
        all_derivatives.append(suffix[layer + 1] * derivatives[layer]["chi"] * prefix[layer])
    return prefix[-1], all_derivatives


def trace2(matrix):
    return matrix[0, 0] + matrix[1, 1]


def hidden_residual(hidden):
    tau = trace2(hidden)
    normalized = 2 * hidden / tau
    return mp.matrix([mp.re((I / 2) * trace2(sigma * normalized)) for sigma in SIGMAS]), abs(tau)


def commutant_coordinates(generator, context):
    coordinates = mp.matrix(19, 1)
    for index, basis in enumerate(context.basis_mp):
        coordinates[index] = mp.re(mp.fsum(generator[i, j] * basis[j, i]
                                            for i in range(8) for j in range(8)))
    return coordinates


def evaluate(params, dps=60, length=15, anchors=None):
    with mp.workdps(dps):
        context = get_ctx(dps)
        point = [mp.mpf(value) for value in params]
        theta, _, chi = unpack(point, length)
        endpoint, derivatives = word_and_derivatives(point, context, length)
        transform = schur_matrix()
        visible, hidden = blocks(endpoint, transform)
        closure, trace_abs = hidden_residual(hidden)
        amp_derivative = mp.zeros(8, 8)
        twist_derivative = mp.zeros(8, 8)
        for layer in range(length):
            amp_derivative += theta[layer] * derivatives[layer]
            twist_derivative += chi[layer] * derivatives[2 * length - 1 + layer]
        amp_generator = I * endpoint.H * amp_derivative
        twist_generator = I * endpoint.H * twist_derivative
        amp_coordinates = commutant_coordinates(amp_generator, context)
        twist_coordinates = commutant_coordinates(twist_generator, context)
        residual = list(closure) + list(amp_coordinates) + list(twist_coordinates)
        lam = trace2(hidden) / 2
        gate = visible / lam
        if anchors is not None:
            values = [mp.re(gate[0, 0]), mp.im(gate[0, 0]), mp.re(gate[0, 1])]
            residual.extend(values[index] - anchors[index] for index in range(3))
        return mp.matrix(residual), {
            "endpoint": endpoint, "visible_gate": gate, "hidden": hidden,
            "trace_abs": trace_abs, "amp_generator": amp_generator,
            "twist_generator": twist_generator,
            "amp_coordinates": amp_coordinates, "twist_coordinates": twist_coordinates,
        }


def embed_i1_seed(length=15):
    center_doc = json.loads(TC2_INPUT.read_text(encoding="utf-8"))
    seed_doc = json.loads(SEED.read_text(encoding="utf-8"))
    source = [mp.mpf(text) for text in center_doc["center"] + seed_doc["seed"][3:]]
    target = [mp.mpf("0") for _ in range(parameter_count(length))]
    target[:7] = source[:7]
    target[length:length + 6] = source[7:13]
    target[2 * length - 1:2 * length - 1 + 7] = source[13:20]
    return target


def action(params, length):
    theta, _, chi = unpack(params, length)
    return mp.fsum(abs(value) for value in theta + chi)

