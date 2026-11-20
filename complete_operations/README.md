# Complete quantum operations: calculation code and numerical data

Public calculation code and selected numerical data for *Reachability of complete quantum operations under dynamical error correction*, by Xiu-Hao Deng. [DATA_MAP.md](DATA_MAP.md) maps the final controls, plotted and tabulated values, interval inputs/outputs, and verification entrypoints. No separate manuscript data attachment is required for the documented commands.

The code implements collective rotations and ZZ interactions on the complete physical register, Schur-sector propagation with relative phases, shared quasistatic gain errors, nominal and first-order fitting, finite-noise refinement, and four/five-qubit circuit validation. The numerical comparisons use a prescribed encoded gate family and fixed finite search budgets.

## Environment

Use Python 3.10 or later with NumPy, SciPy and Matplotlib. The recorded study used Python 3.12.10 and the versions in `requirements-recorded.txt`. SymPy 1.14.0 supports the optional exact two-qubit check; mpmath 1.3.0 supports the generic interval workers. For a new environment, install `requirements.txt` explicitly. No script installs dependencies. Keep assertions enabled. Run commands below from this directory, using new output paths.

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

The controls and reference scores under `runs/` are replay inputs. `data/three_qubit_controls.npz` supplies the three-qubit controls for the full-register module. `PUBLICATION_SOURCE_PROVENANCE.json` binds the selected original bytes; `PUBLICATION_SHA256.json` binds the public payload. The older `SOURCE_PROVENANCE.json` records the previously published core's origins. Its absent archive paths describe historical provenance, not runtime dependencies of the documented commands. The original supporting sources retain some historical entrypoints; use the explicit self-contained commands in this README.

## Further bounded checks

```sh
python -B verify_publication.py
python -B tools/npj_qi_independent_audit_20261118_v1.py --out output_independent
python -B tools/npj_qi_template_ranks_20261118_v1.py --out output_template_ranks
python -B historical_3q/cnot_tolerance/tools/replay_cnot_tolerance_20261012_v1.py --out output_cnot_bounds
python -B tools/verify_certificate_arithmetic.py --directory historical_3q/legacy_validation/certificate_replay/fresh_certificates --output output_certificate_arithmetic.json
python -B historical_3q/legacy_validation/certificate_replay/replay_fixed_chain.py --prepare-only --out output_generic_preparation
```

The first command checks byte identity and syntax. The independent audit reconstructs full-register channels and checks archived bound arithmetic. The CNOT command recomputes directed-arithmetic degree-eight enclosures for all nine controls, including the four neighboring controls. The rational certificate verifier checks 12 stored records/14 precision cases by exact arithmetic, rather than regenerating Hamiltonian/interval calculations. The final command checks the isolated generic worker-source closure/imports only; removing `--prepare-only` requests the more expensive fixed-candidate 320/512-bit certificate-generation chain. It is not part of the bounded packaging validation.

The CNOT interpolation, weighted control neighborhood, common chronological schedule and dephasing bound are separate from vertex coefficient generation. Their inputs/outputs are in `historical_3q/cnot_tolerance/runs/cnot_tolerance_20261012_v1/primary_audit/`. The independent audit checks the archived aggregate arithmetic. Neither that arithmetic check nor a file hash is described as a new interval certificate.

## Numerical tables and figure data

`tables/*.csv` contains exact displayed table cells, with mathematical expressions retained in LaTeX notation. `tables/TABLE_PROVENANCE.json` records their source bindings. DATA_MAP.md identifies the full-precision arrays/JSON/CSV inputs and each panel's analytic or numerical scope. Rendered figures, manuscript copies, historical narrative reports and internal reviewer text are omitted.

```sh
python -B tools/npj_qi_review_figures_20261118_v1.py --out output_review_figures
python -B tools/npj_qi_comparison_figure_20261118_v1.py --out output_comparison_figure
python -B tools/publication_figures.py --out output_other_figures
```

These commands reproduce quantitative displays from the public inputs; exact publication font/layout reproduction is optional. All eight held-out compiler pairs, including the five-qubit reversal, and the finite-depth failures to meet 1% remain in the data. The interval inputs retain the unsuccessful initial enclosure as well as the accepted successor.

## Evidence scope

The release supplies the selected final pulse parameters, numerical figure/table values and the declared interval inputs/outputs and worker sources. Extensive search-history duplicates and presentation-only records are omitted. Numerical replay, checking serialized inequalities, fresh interval-enclosure generation and generic exact-root certification are distinct operations. The CNOT acceptance family is separate from the generic exact-root construction. The Article and SI specify their error models, completed interfaces, resource accounting and evidence scopes. No global-optimum proof, continuous extension of a sampled grid, arbitrary-depth fault-tolerance threshold or device validation is claimed.

This directory uses the repository's existing GPL-3.0 license.
