# Complete quantum operations under dynamical error correction

Core numerical code for *Reachability of complete quantum operations under dynamical error correction*, by Xiu-Hao Deng.

The code implements collective rotations and ZZ interactions on the complete physical register, Schur-sector propagation with relative phases, shared quasistatic gain errors, nominal and first-order fitting, finite-noise refinement, and four/five-qubit circuit validation. The numerical comparisons use a prescribed encoded gate family and fixed finite search budgets.

## Environment

Use Python 3.10 or later with NumPy, SciPy and Matplotlib. The recorded study used Python 3.12.10 and the versions in `requirements-recorded.txt`. For a new environment, install `requirements.txt`. Run commands below from this directory.

## Replay the main circuit comparison

```sh
python -B tools/npj_qi_full_range_review_20261118_v1.py --scores-from runs/npj_qi_full_range_review_20261118_v2 --workers 1 --out output_replay
```

The destination must be new. This re-evaluates the training-only selection of the supplied candidate controls and the independent physical-register and Schur-sector channels on 80 held-out gain settings at depths 1, 2, 4, 8, 16 and 32. It also checks the separate development-control leakage records. All eight matched pairs are retained, including the five-qubit pair where direct fitting performed better. The expected counts are 7,680 held-out channel rows and 800 development leakage rows. The script verifies source, control and reference-record hashes before replay.

## Fit new paired controls

```sh
python -B tools/npj_qi_full_range_review_20261118_v1.py --workers 1 --out output_new_fits
```

This runs the registered finite-budget optimization from the eight fixed starts. It can take appreciably longer than replay. Optimizer timing and new fitted controls depend on the numerical environment. The task, controls, baseline, 17 training offsets and stopping rules are in `docs/NPJ_QI_FULL_RANGE_REVIEW_PROTOCOL_20261118_v1.md`.

## Core modules

For a bounded three-qubit propagation example, run

```sh
python -B tools/replay_three_qubit.py --out output_three_qubit.json
```

This evaluates the five stored controls on a 5 by 5 coherent gain grid, together with their nominal errors, first-order generators and actuator-limited times.

| File | Purpose |
|---|---|
| `src/manifold_schur_v1.py` | Collective generators, sector multiplicities, propagation and gain derivatives |
| `src/cnot_tolerance_v1.py` | Three-qubit full-register propagation, Taylor coefficients and actuator-duration helpers |
| `tools/npj_qi_register_depth_20261118_v1.py` | Nominal, first-order and finite-noise fitting with analytic Jacobians |
| `tools/npj_qi_register_depth_projection_20261118_v1.py` | Nominal endpoint restoration |
| `tools/npj_qi_full_range_review_20261118_v1.py` | Matched-budget paired fits and held-out score replay |
| `tools/npj_qi_independent_audit_20261118_v1.py` | Independent tensor-register routines used by the replay |

The small controls and reference scores under `runs/` are inputs to the replay. `data/three_qubit_controls.npz` supplies the three-qubit control path used with the full-register module. `SOURCE_PROVENANCE.json` binds the extracted files to the accompanying manuscript reproducibility archives. Some supporting modules also retain historical standalone entry points whose broader inputs are supplied in those archives. The commands documented above are self-contained within this release.

## Evidence scope

The replay evaluates finite numerical records and compares two implementations of the coherent channel. The continuous three-qubit interval certificate, extensive search histories and complete figure data are provided with the manuscript's separate reproducibility materials. The Article and Supplementary Information specify the assumptions of the reachability theorem and each numerical or interval result.

This directory uses the repository's existing GPL-3.0 license.
