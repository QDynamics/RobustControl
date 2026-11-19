"""Registered four/five-physical-qubit encoded CNOT depth pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from itertools import combinations, product
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import scipy
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cnot_tolerance_v1 import duration
from src.manifold_schur_v1 import full_tensor_representation, propagate, schur_representations

PROTOCOL = ROOT / "docs/NPJ_QI_REGISTER_DEPTH_PROTOCOL_20261118_v1.md"
LENGTHS = {4: 48, 5: 72}
DEPTHS = (1, 2, 4, 8, 16, 32)
TEST = np.array(list(product(np.linspace(-.01, .01, 5), repeat=2)))
TRAIN = np.array([(0., 0.), *product((-.006, .006), repeat=2)])
H = 2e-5
JET = np.array([(0., 0.), (H, 0.), (-H, 0.), (0., H), (0., -H)])
PHASES = np.mod(np.pi * (np.sqrt(2.) * np.arange(32)**2 + np.sqrt(3.) * np.arange(32)), 2*np.pi)
METHODS = ("nominal", "corrected", "noise_aware")
G = np.exp(1j*np.pi/4) * np.eye(4)[[0, 1, 3, 2]]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, data):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)


def unpack(p):
    theta, phase, raw = np.split(p, 3)
    return theta, phase, raw - raw.mean()


class Block:
    def __init__(self, rep, n):
        self.d = rep.dimension
        self.mult = rep.multiplicity
        self.m = rep.Jz.diagonal().real.copy()
        self.difference = self.m[:, None] - self.m[None, :]
        self.hzz = rep.Hzz.diagonal().real.copy()
        self.eig, self.basis = np.linalg.eigh(rep.Jx.real)
        self.target = np.eye(self.d, dtype=complex)
        if rep.two_j == n:
            code = np.arange(n, n-4, -1)
            self.target[np.ix_(code, code)] = G

    def word(self, params, offsets, jac=False):
        theta, phase, chi = unpack(params)
        x = 1 + offsets[:, 0, None, None]
        y = 1 + offsets[:, 1, None, None]
        ep = np.exp(-1j * x * theta[None, :, None] * self.eig)
        rz = np.exp(-1j * phase[:, None, None] * self.difference)
        zz = np.exp(-1j * y * chi[None, :, None] * self.hzz)
        rotation = ((self.basis * ep[..., None, :]) @ self.basis.T) * rz
        layers = rotation * zz[..., None, :]
        s, length, d, _ = layers.shape
        prefix = np.empty((s, length+1, d, d), complex)
        prefix[:, 0] = np.eye(d)
        for j in range(length):
            prefix[:, j+1] = layers[:, j] @ prefix[:, j]
        last = prefix[:, -1]
        if not jac:
            return last
        left = last[:, None] @ prefix[:, 1:].conj().swapaxes(-1, -2)
        right = prefix[:, :-1]
        dtheta = (((self.basis * (ep * (-1j*x*self.eig))[..., None, :]) @ self.basis.T)
                  * rz * zz[..., None, :])
        dphi = layers * (-1j*self.difference)
        dchi = layers * (-1j*y*self.hzz)[..., None, :]
        dtheta = left @ dtheta @ right
        dphi = left @ dphi @ right
        dchi = left @ dchi @ right
        dchi -= dchi.mean(axis=1, keepdims=True)
        jacobian = np.concatenate((dtheta, dphi, dchi), axis=1).transpose(0, 2, 3, 1)
        return last, jacobian


def jet_rows(arr):
    return np.stack((arr[0], .12*(arr[1]-arr[2])/(2*H), .12*(arr[3]-arr[4])/(2*H)))


def evaluate(blocks, n, p, method):
    offsets = {"nominal": JET[:1], "corrected": JET, "noise_aware": TRAIN}[method]
    parts, derivatives = [], []
    for block in blocks:
        u, jac = block.word(p, offsets, jac=True)
        err = u - block.target
        if method == "corrected":
            err = jet_rows(err)
            jac = jet_rows(jac)
        scale = math.sqrt(block.mult / (2**n))
        flat = (scale*err).ravel()
        dflat = (scale*jac).reshape(-1, len(p))
        parts.extend((flat.real, flat.imag))
        derivatives.extend((dflat.real, dflat.imag))
    return np.concatenate(parts), np.concatenate(derivatives)


def distance(relative, multiplicities):
    eigenvalues = np.concatenate([np.linalg.eigvals(v) for v in relative])
    a = np.sort(np.angle(eigenvalues))
    cuts = np.diff(np.r_[a, a[0]+2*np.pi])
    cut = int(np.argmax(cuts))
    width = a[-1]-a[0] if cut == len(a)-1 else 2*np.pi+a[cut]-a[cut+1]
    diamond = 1. if width >= np.pi else np.sin(max(0., width)/2)
    d = sum(m*u.shape[0] for m, u in zip(multiplicities, relative))
    trace = sum(m*np.trace(u) for m, u in zip(multiplicities, relative))
    infidelity = max(0., (d*d-abs(trace)**2)/(d*(d+1)))
    return float(diamond), float(infidelity)


def metrics(targets, actual, multiplicities):
    return distance([v.conj().T @ u for v, u in zip(targets, actual)], multiplicities)


def training_score(blocks, p):
    matrices = [b.word(p, TRAIN) for b in blocks]
    return max(metrics([b.target for b in blocks], [m[k] for m in matrices],
                       [b.mult for b in blocks])[0] for k in range(len(TRAIN)))


def fit(blocks, n, start, method, max_eval, seconds):
    t0 = time.perf_counter()
    best_cost = np.inf
    best = start.copy()
    cached = cached_args = None
    costs = []

    def compute(p):
        nonlocal cached_args, cached, best_cost, best
        if cached_args is None or not np.array_equal(p, cached_args):
            if costs and time.perf_counter()-t0 >= seconds:
                raise TimeoutError
            cached = evaluate(blocks, n, p, method)
            cached_args = p.copy()
            score = float(np.dot(cached[0], cached[0]))
            costs.append(score)
            if score < best_cost:
                best_cost, best = score, p.copy()
        return cached

    length = len(start)//3
    bounds = np.repeat([np.pi, 3*np.pi, .6], length)
    try:
        solution = least_squares(lambda p: compute(p)[0], start, jac=lambda p: compute(p)[1],
                                 bounds=(-bounds, bounds), max_nfev=max_eval,
                                 ftol=1e-13, xtol=1e-13, gtol=1e-13)
        reason = solution.message
    except TimeoutError:
        reason = "time_limit"
    return best, {"evaluations": len(costs), "reason": reason, "wall_s": time.perf_counter()-t0,
                  "objective_squared_history": costs, "best_squared_objective": best_cost}


def trace_circuit(targets, gates, magnetic, mults):
    target_prefix = [np.eye(len(b), dtype=complex) for b in magnetic]
    actual_prefix = [u.copy() for u in target_prefix]
    scores = {}
    for depth, phi in enumerate(PHASES, 1):
        for b, m in enumerate(magnetic):
            z = np.exp(-1j * phi * (m[:, None]-m[None, :]))
            target_prefix[b] = (targets[b] * z) @ target_prefix[b]
            actual_prefix[b] = (gates[b] * z) @ actual_prefix[b]
        if depth in DEPTHS:
            scores[depth] = metrics(target_prefix, actual_prefix, mults)
    return scores


def tensor_setup(n):
    labels = np.arange(2**n)
    weights = np.array([int(i).bit_count() for i in labels])
    v = np.zeros((2**n, 4))
    for k in range(4):
        v[:, k] = (weights == n-k) / math.sqrt(math.comb(n, n-k))
    target = np.eye(2**n) + v @ (G-np.eye(4)) @ v.T
    signs = 1-2*((labels[:, None] >> np.arange(n)) & 1)
    charges = sum(signs[:, a]*signs[:, b] for a, b in combinations(range(n), 2))
    return target, n/2-weights, charges


def tensor_word(n, charges, p, x, y):
    u = np.eye(2**n, dtype=complex)
    for theta, phi, chi in zip(*unpack(p)):
        a = theta*(1+x)/2
        single = np.array([[np.cos(a), -1j*np.sin(a)*np.exp(-1j*phi)],
                           [-1j*np.sin(a)*np.exp(1j*phi), np.cos(a)]])
        collective = single
        for _ in range(n-1):
            collective = np.kron(collective, single)
        twist = np.exp(-1j*(1+y)*chi*charges)
        u = collective @ (twist[:, None] * u)
    return u


def check_gradient(blocks, n, p):
    direction = np.cos(np.arange(len(p))+.125)
    direction /= np.linalg.norm(direction)
    e = 1e-6
    resid, jac = evaluate(blocks, n, p, "nominal")
    plus = evaluate(blocks, n, p+e*direction, "nominal")[0]
    minus = evaluate(blocks, n, p-e*direction, "nominal")[0]
    rel = float(np.linalg.norm(jac @ direction-(plus-minus)/(2*e)) /
                max(1., np.linalg.norm(resid), np.linalg.norm(jac @ direction)))
    if rel > 1e-7:
        raise ValueError(f"endpoint Jacobian check failed {rel}")
    return rel


def diagnostics(n, blocks, p, target, magnetic, charges, tensor_rep):
    rows = np.asarray(unpack(p)).T
    analytic = propagate(tensor_rep, rows)
    actual = tensor_word(n, charges, p, 0., 0.)
    direct_agreement = float(np.linalg.norm(actual-analytic.U, ord=2))
    if direct_agreement > 1e-8:
        raise ValueError(f"tensor word mismatch {direct_agreement}")
    nominal, _ = metrics([target], [actual], [1])
    norms = [float(np.linalg.norm(gen, ord=2)) for gen in (analytic.generator_R, analytic.generator_T)]
    dtheta = [[], []]
    finite_matrices = [b.word(p, JET) for b in blocks]
    for pos, offset in enumerate((1, 3)):
        for value in finite_matrices:
            dtheta[pos].append(float(np.linalg.norm((value[offset]-value[offset+1])/(2*H), 2)))
    short = float(duration(rows[:, 0], 10.).sum()+duration(rows[:, 2], 1.).sum()
                  +(2*len(rows)-1)*.001)
    gates = [b.word(p, TEST) for b in blocks]
    reference = [b.target for b in blocks]
    mult = [b.mult for b in blocks]
    mag = [b.m for b in blocks]
    results = []
    max_check = 0.
    for idx, (x, y) in enumerate(TEST):
        values = trace_circuit(reference, [g[idx] for g in gates], mag, mult)
        if (x, y) in ((0., 0.), (.01, -.01), (-.01, .01)):
            direct_u = tensor_word(n, charges, p, x, y)
            other = trace_circuit([target], [direct_u], [magnetic], [1])
            for m in (1, 32):
                max_check = max(max_check, *(abs(a-b) for a, b in zip(values[m], other[m])))
        for depth, (diamond, avg) in values.items():
            results.append({"x": float(x), "y": float(y), "depth": depth,
                            "half_diamond": diamond, "average_gate_infidelity": avg})
    if max_check > 1e-8:
        raise ValueError(f"Schur/tensor validation mismatch {max_check}")
    by_depth = {}
    for depth in DEPTHS:
        group = [r for r in results if r["depth"] == depth]
        worst = max(group, key=lambda v: v["half_diamond"])
        by_depth[depth] = {"max_D": worst["half_diamond"], "worst_xy": [worst["x"], worst["y"]],
                           "max_avg_infidelity": max(r["average_gate_infidelity"] for r in group)}
    return {"nominal_D": nominal, "analytic_derivative_opnorms": norms,
            "fd_derivative_opnorms": [max(v) for v in dtheta],
            "nominal_qualified": nominal <= 1e-6,
            "first_order_qualified": nominal <= 1e-6 and max(norms) <= 1e-5,
            "compressed_gate_time": short, "dense_unitary_crosscheck": direct_agreement,
            "max_metric_crosscheck": max_check, "by_depth": by_depth}, results


def compute_register(n, out):
    length = LENGTHS[n]
    blocks = [Block(b, n) for b in schur_representations(n)]
    rng = np.random.default_rng(6111800+n)
    start = np.concatenate([rng.uniform(-1.6, 1.6, length), rng.uniform(-np.pi, np.pi, length),
                            rng.uniform(-.45, .45, length)])
    p = {"start": start}
    optimizer = {}
    gradient_check = check_gradient(blocks, n, start)
    for label, initial, kind, budget, secs in [
        ("nominal", "start", "nominal", 100, 150),
        ("corrected", "nominal", "corrected", 180, 150),
        ("noise_from_nominal", "nominal", "noise_aware", 90, 75),
        ("noise_from_corrected", "corrected", "noise_aware", 90, 75),
    ]:
        p[label], optimizer[label] = fit(blocks, n, p[initial], kind, budget, secs)
        np.save(out/f"n{n}_{label}.npy", p[label], allow_pickle=False)
        dump(out/f"n{n}_{label}_fit.json", optimizer[label])
        print(f"N={n} {label} calls={optimizer[label]['evaluations']} "
              f"loss={optimizer[label]['best_squared_objective']:.3g} "
              f"time={optimizer[label]['wall_s']:.1f}s", flush=True)
    train = {key: training_score(blocks, p[key]) for key in
             ("nominal", "corrected", "noise_from_nominal", "noise_from_corrected")}
    selected = min(train, key=train.get)
    p["noise_aware"] = p[selected]
    np.save(out/f"n{n}_initial.npy", start, allow_pickle=False)
    target, magnetic, charges = tensor_setup(n)
    tensor_rep = full_tensor_representation(n)
    z = np.exp(-1j*PHASES[1]*(magnetic[:, None]-magnetic[None, :]))
    b = target*z
    comm = float(np.linalg.norm(target@b-b@target, 2))
    if comm <= 1e-6:
        raise ValueError("the circuit's leading gates accidentally commute")
    summary = {}
    for name in METHODS:
        summary[name], points = diagnostics(n, blocks, p[name], target, magnetic, charges, tensor_rep)
        dump(out/f"n{n}_{name}_validation.json", points)
    fixed = float(length*(duration(np.pi, 10.)+duration(1.2, 1.))+(2*length-1)*.001)
    results = {"N": n, "seed": 6111800+n, "layers_per_gate": length,
               "fixed_gate_time_t0": fixed, "fixed_depth32_time_t0": 32*fixed+31*.001,
               "fixed_depth32_slots": 32*2*length,
               "target_commutator_opnorm": comm, "objective_jacobian_directional_check": gradient_check,
               "training_single_gate_max_D": train, "noise_aware_source": selected, "methods": summary}
    dump(out/f"n{n}_summary.json", results)
    return results


def figure(out, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.6), sharey=True)
    labels = ("Nominal fit", "First-order fit", "Finite-noise fit + incumbents")
    for ax, (n, result) in zip(axes, results.items()):
        for name, display, color, mark, style in zip(METHODS, labels, ("#526478", "#007C80", "#BA5129"),
                                                    ("o", "s", "^"), ("--", "-", ":")):
            values = [result["methods"][name]["by_depth"][depth]["max_D"] for depth in DEPTHS]
            ax.plot(DEPTHS, values, linestyle=style, color=color, marker=mark, ms=6,
                    markerfacecolor="none" if name != "corrected" else color, label=display)
        ax.axhline(.01, color=".45", linewidth=.9, linestyle="-.", label="1% reference")
        ax.set(xscale="log", yscale="log", xticks=DEPTHS, xticklabels=DEPTHS,
               xlabel="Completed encoded gates", title=f"{n} physical qubits")
        ax.minorticks_off()
        ax.grid(axis="y", alpha=.22)
    axes[0].set_ylabel("Maximum half-diamond distance on 25 test points")
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out/"depth_comparison.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=False, exist_ok=False)
    t0 = time.perf_counter()
    try:
        inputs = [Path(__file__).resolve(), PROTOCOL, ROOT/"src/manifold_schur_v1.py",
                  ROOT/"src/cnot_tolerance_v1.py", ROOT/"submission/NatureCommunications_20261013_v1/04_source/main.tex"]
        dump(out/"inputs.json", {"hashes": {str(i.relative_to(ROOT)): sha(i) for i in inputs},
             "utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
             "interpreter": sys.executable, "python": sys.version, "platform": platform.platform(),
             "numpy": np.__version__, "scipy": scipy.__version__, "precision": "float64/complex128",
             "blas_threads_requested": 1, "test": TEST.tolist(), "train": TRAIN.tolist(),
             "azimuths": PHASES.tolist(), "depths": DEPTHS})
        results = {n: compute_register(n, out) for n in LENGTHS}
        figure(out, results)
        dump(out/"summary.json", {"status": "completed_numerical_pilot", "wall_s": time.perf_counter()-t0,
                                  "registers": results})
    except BaseException:
        dump(out/"failure.json", {"traceback": traceback.format_exc(), "wall_s": time.perf_counter()-t0})
        raise
    finally:
        dump(out/"sha256.json", {f.name: sha(f) for f in sorted(out.iterdir()) if f.is_file()})


if __name__ == "__main__":
    main()
