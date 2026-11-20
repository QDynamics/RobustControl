# Article and Supplementary data map

Paths are relative to `complete_operations/`. Figure basenames identify the panel sets independently of numbering. The documented commands are self-contained; original absolute machine paths in numerical records describe provenance.

| Article panel set | Final data / parameters | Entry point and scope |
|---|---|---|
| Conceptual Fig. 1 | Analytic schematic | No empirical dataset |
| `order_reachability` | Target-set identities in `tools/npj_qi_review_figures_20261118_v1.py` | Analytic schematic |
| `sector_phase_distance` | `runs/npj_qi_review_figures_20261118_v1/phase_distance_and_readout.csv` | Same producer; independent audit checks physical-register readout |
| `cnot_tolerance_summary` | `historical_3q/cnot_tolerance/runs/cnot_tolerance_20261012_v1/{numerical,interval_v2,primary_audit}/` | All nine controls, coefficient balls, interpolation/weighted-budget bounds, dephasing and chronological common schedule |
| `register_depth` | `runs/npj_qi_register_depth_report_20261118_v1/{all_validation_points.csv,summary.json}`, development/restored controls and `runs/npj_qi_confirm_starts_20261118_v1/` | Sampled finite-budget comparisons; `tools/publication_figures.py` |

| Supplementary panel/table set | Final data / parameters | Entry point and scope |
|---|---|---|
| `full_range_and_leakage` | `runs/npj_qi_full_range_review_20261118_v2/{paired_results,heldout_points,heldout_maxima,development_leakage_points,development_leakage_maxima,solver_budgets}.csv` and eight saved candidate-control NPZs | 80 held-out settings, eight pairs including reversal, separate 800-row development leakage study |
| `dephasing_clocks` | `runs/npj_qi_dephasing_20261118_v1/{readout.csv,channel_brackets.csv,summary.json}` and `runs/npj_qi_review_figures_20261118_v1/{common_four_control_clock,dephasing_bounds}.csv` | Declared local Markov model, original/shorter common clocks, separate sampled coherent maxima and analytic mixture bounds |
| `unified_results` | `historical_3q/legacy_validation/figures/build_unified_results.py` | Analytic classification/resource schematic, no empirical dataset |
| `cnot_manifold_cost_v17` | `historical_3q/legacy_validation/research/runs/v17_library_v1/{result.json,*.npz}`, five `v12_search_*_warm_v1/trace.jsonl` | Achieved actuator-limited times, not minima; public plotting entrypoint |
| `actuator_readout_v17` | Same library's `whole.npz`, `v17_confirmation_v1/result.json` witness values | Analytic actuator waveform and sampled addressed-readout biases |
| `interface_selection_v17` | `historical_3q/legacy_validation/research/runs/v17_confirmation_v1/{result.json,arrays.npz}` | Winner/unresolved maps and full confirmation means/samples |
| Displayed tables | `tables/*.csv`, `tables/TABLE_PROVENANCE.json` | Exact printed cells, including analytic tables, local ranks, sufficient bounds and reversed/non-improving comparisons |
| Exact N=2 25-column chart | `historical_3q/legacy_validation/reproducibility/runs/n2_q1_exact_chart_20260917_v1/` and `historical_3q/legacy_validation/reproducibility/tools/audit_n2_q1_exact_chart_20260917_v1.py` | Exact selected columns/minor/resource data and symbolic checker |
| Network-algebra small-N checks | `historical_3q/legacy_validation/research/runs/stage2_algebra_20260924_v1/results.json`, `historical_3q/legacy_validation/research/validate_stage2_algebra_v1.py` | Exact integer-Pauli elimination modulo prime 1000003 at the listed sizes; lower-rank witness matched to analytic upper bound |
| Generic exact-root/rank/radius certificates | `historical_3q/legacy_validation/certificate_replay/fresh_certificates/` (12 JSONs), isolated `sandbox/` worker inputs and `tools/verify_certificate_arithmetic.py` | Rational verification of recorded inequalities; optional separate 320/512-bit fixed-candidate generation |
| N=4,5 local Jacobian counts | `runs/npj_qi_template_ranks_20261118_v1/n{4,5}.json` and raw corrected `runs/npj_qi_register_depth_20261118_v1/n{4,5}_corrected.npy` | Standalone rank producer regenerates matrices, spectra and finite-difference checks; numerical diagnostics at finite residuals |

No archived full Supplementary manuscript is needed to access these values. The former full-SI compression index and historical report PDFs are intentionally omitted. Original column/minor/radius values and their small verification inputs are supplied directly instead. The repository does not claim that narrative archives or all exploratory searches have been reproduced.

Additional legacy values cited in the SI are retained under `historical_3q/legacy_validation/reproducibility/runs/`: `waveform_framework_terra_20260914_v1`, `waveform_rank_sol_20260915_v1`, `parent_waveform_rank_acceptance_20260915_v1`, `v8_review_error_diagnostics_20260920_v1`, `circuit_framework_terra_20260914_v2`, `target_programming_sol_20260915_v1`, and `manifold_finite_fiber_20260913_v1`. These contain the selected final parameter arrays, rank/error/finite-circuit records and target registries, without the historical narrative reports or exploratory derivative caches.

## Distinct evidence scopes

The generic exact-root construction, CNOT finite-error acceptance family, sampled circuit comparisons and numerical tangent counts are separate. CNOT exact fixed-target fiber membership is not inferred from its continuous finite-error path. Shared gains remain fixed through the completed operation; complete-channel metrics retain arbitrary physical register/reference inputs and all output sectors. The stated local-dephasing and leakage studies retain their own models and clocks. There is no hardware evidence or global optimization certificate.

Negative results include the five-qubit paired reversal, the original depth-16/32 failures of the 1% reference, CNOT sufficient bounds that exceed the tolerance despite lower sampled errors, zero outside-symmetric coherent leakage within numerical precision, and the failed first interval enclosure. These remain recorded in the selected data rather than being removed from the comparisons.
