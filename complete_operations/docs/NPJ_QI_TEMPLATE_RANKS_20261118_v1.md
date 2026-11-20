# Fixed-template rank diagnostic for four and five physical qubits

Registered on 2026-11-18 before evaluating any Jacobian singular values at these two controls. The diagnostic uses the raw first-order-corrected **development** words from the [registered depth experiment](NPJ_QI_REGISTER_DEPTH_PROTOCOL_20261118_v1.md). It evaluates one archived setting per register, with no optimization, continuation or candidate selection. Both settings were already known to approximate the complete encoded CNOT target.

## Controls and target

Each layer is a signed Ising twist followed by an equatorial collective rotation, with independent common fractional gain offsets $y$ and $x$, respectively. All azimuths refer to the laboratory frame. For $N=4,5$ use $L=48,72$ and the archived `n4_corrected.npy` and `n5_corrected.npy` in `runs/npj_qi_register_depth_20261118_v1`. Their SHA-256 hashes are `7ff5a2720b3a8fdfc1350f5822213c78fe21ee03367b7e5a4bd7064d9d156446` and `3d51043361221b87e659a67b3028531157ee339ee42ed25f49d5db11025c9657`.

The stored vector lists θ, φ and raw twists $v$, each of length $L$. The physical twists are χ = $v-\bar v$. Quotient the one exact redundancy $v\mapsto v+t\mathbf1$ using orthonormal Helmert columns for the zero-sum twist plane. Rotation areas and all $L$ laboratory azimuths use ordinary coordinates. The independent control dimension is $m=3L-1$. Discrete sign/azimuth representations of a rotation give the same local control dimension. Record the archived representative's distance to the original fitting bounds and its smallest absolute rotation angle. Check the differential for shifting all drive azimuths by one radian against conjugation by $e^{-itJ_z}$.

Use the complete physical unitary on $2^N$ dimensions. Schur symmetry permits one copy of each spin block because $U=\bigoplus_j U_j\otimes I_{M_j}$. This retains all lower-spin actions, relative phases and multiplicities. There is no projection onto the encoded space, nor postselection. The target $V$ acts as $e^{i\pi/4}\mathrm{CNOT}$ on the top-spin Dicke states of weights $N,N-1,N-2,N-3$ in that order and as identity on their full orthogonal complement. Arbitrary inputs with a reference and arbitrary continuations are allowed. This is coherent gate control with no signal parameter. Timing, finite gain error, and gate comparisons are outside this fixed-word differential calculation.

## Maps, coordinates and structural limits

At zero gain offsets define $K_s=iU^\dagger\partial_sU$ for $s=x,y$. Let ℛ collect the sector-traceless Hermitian coordinates of these matrices. Use real Hilbert–Schmidt orthonormal bases separately in each distinct sector, with no multiplicity or task weight. In these units, the common rotations and the zero-sum twists imply unit determinant separately in every nominal block and zero trace of $K_s$ in every block. In spin 0 both responses vanish. In spin 1/2 the twist generator $H_{ZZ}=2J_z^2-NI/2$ is scalar and the zero total twist makes $K_y\equiv0$. Report numerical sizes of the omitted scalar and half-spin responses. The dimension $n_j=d_j^2-1$ below counts independent real Hermitian directions.

| $N$ | spin $j$ | nominal rows | rotation-response rows | twist-response rows |
|---|---|---:|---:|---:|
| 4 | 2, 1, 0 | 24, 8, 0 | 24, 8, 0 | 24, 8, 0 |
| 5 | 5/2, 3/2, 1/2 | 35, 15, 3 | 35, 15, 3 | 35, 15, 0 |

The nominal target coordinates Θ are the sector coefficients of $i\log(V_j^\dagger U_j)$ on the principal branch near $V$. At $V$ these linearize to the coefficients of $iV_j^\dagger\delta U_j$. The resulting $k=32,53$ coordinates allow changes to the full reachable block action, including the complement and couplings to spectator states in the top spin. For varying only the logical operation with fixed identity complement, split the chart into the embedded traceless $4\times 4$ logical coordinates Θₗ and their orthogonal complement Θ⊥. The latter includes changes of leakage couplings and relative phase. Then Θ⊥ = 0 selects the 15-dimensional embedded $SU(4)$ target family locally. The response plus fixed-complement map is $(\mathcal R,\Theta_\perp)$. The full joint map $(\mathcal R,\Theta)$ is the same up to rearrangement of rows in both interpretations.

## Numerical checks and reporting rule

1. Propagate the endpoint, both gain derivatives, and their control derivatives analytically using the product rule for each twist–rotation pair. Project onto the independent control chart and the Hermitian coordinates above. Report the two response norms and the nominal target residual at both controls.
2. Independently evaluate $K_x,K_y$ as prefix-conjugated sums of local generators. Use a rotation obtained by conjugating the spectral decomposition of $J_x$ by a $z$-rotation. Take centered differences of this independently evaluated map along **every** physical coordinate at steps $2\times10^{-4}$ and $10^{-4}$, then form the fourth-order Richardson estimate. Near the specified target, evaluate the principal logarithm by eight terms of its Taylor series and report a bound on its tail over the perturbations. Compare the analytic matrices with both central estimates and the extrapolation in operator norm. Compare a weakest right-singular direction of each direct matrix to the independent extrapolation. Compute the rank and weak-mode direction for each spin block as well as the aggregate maps.
3. Use real double and complex double precision and one BLAS thread. Report every analytic singular value of the aggregate and sector matrices and the independent extrapolated spectra, including changes in rank across relative thresholds $10^{-6},10^{-8},10^{-9},10^{-10},10^{-12}$ times each matrix's largest singular value. The registered reporting threshold is $10^{-9}$. For a full-row numerical rank, require agreement with Richardson and the smallest analytic singular value to exceed **ten times** the measured operator-norm discrepancy between the two evaluations. This is an empirical resolution test. A disagreement or unresolved mode remains visible in the report.
4. Use an SVD nullspace of $D\mathcal R$ to report the endpoint differential on response-preserving tangent directions. Separately restrict $D\Theta_{\mathrm L}$ to the nullspace of $D(\mathcal R,\Theta_\perp)$. For those restrictions, compare both methods on the same analytic nullspace. Also recompute their ranks using the independent nullspace to check sensitivity to the choice of tangent basis. Record the full joint numerical nullity $m-\operatorname{rank}D(\mathcal R,\Theta)$.
5. Preserve both results, including negative results. Stop after the fixed four- and five-qubit candidates, even if a rank is unresolved. The calculation characterizes first derivatives at the stored approximate settings. Exact cancellation, an open exact target patch, and exact fixed-target manifold dimensions for these words would require additional existence evidence. An independently evaluated floating-point Jacobian does not provide an interval certificate.

## Replay

From a project or reviewer workspace with these files, Python 3, NumPy and SciPy, run the following command with a fresh output directory. The driver checks the two input hashes, writes its versions, controls, precision, runtime, failures and SHA-256 bindings, and refuses to reuse an existing directory.

```powershell
python -B tools/npj_qi_template_ranks_20261118_v1.py --out runs/npj_qi_template_ranks_replay_v1
```

The original run uses `runs/npj_qi_template_ranks_20261118_v1`. Its [result summary](../runs/npj_qi_template_ranks_20261118_v1/report.md) and the machine-readable [four-qubit](../runs/npj_qi_template_ranks_20261118_v1/n4.json) and [five-qubit](../runs/npj_qi_template_ranks_20261118_v1/n5.json) records contain the measured ranks and sensitivity checks.
