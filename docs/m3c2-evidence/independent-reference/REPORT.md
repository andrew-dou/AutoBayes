# PR10 deterministic Gaussian reference — completed

Execution date: 2026-09-20 UTC (September 19 America/New_York). Exact unchanged head `6a58eb3c9b8ab510aa8bd5d3d04936e64990dace`.

**The deterministic computation and prespecified stability checks passed. This is not PR acceptance: the original PyMC/NumPyro 3/6-divergence failures and external runner TypeError remain unchanged.** No posterior fit, response draw, retry, repository edit, network request, install, GPU job, or merge occurred.

## Execution and independent target

One supervised invocation, 18 fixed Gauss–Legendre rules, **exactly 3,840 marginal-target evaluations**, below the 4,000 hard cap. Exit **0**, no timeout; total wall time **24.809044833 seconds** under a 120-second process-group deadline. The exclusive launch marker prevents another invocation in this directory. Numerical imports, all evaluations, comparisons, and serialization were inside the deadline.

The four coefficients were integrated analytically for the frozen six observations, raw design `[dose, I(control), I(treated), dose*I(treated)]`, independent beta covariance `D=diag(1,4,4,1)`, and HalfNormal(1) sigma prior. At each node, `C=sigma²I+XDXᵀ`; a Cholesky solve gives the normalized Gaussian marginal density and conditional coefficient mean `DXᵀC⁻¹y`. Integration uses `t=log(sigma)` with the +t Jacobian and stable logsumexp weights. AutoBayes, Bambi, PyMC, and NumPyro imports were forbidden and absent.

Full rules used sigma intervals `[0.02,12]` with 128/256 nodes and `[0.01,16]` with 256 nodes; five corresponding numerator rules per configuration used the frozen CDF points. Every target node was evaluated and retained. Relative-weight exponentiation underflowed to zero for 378 of the 3,840 rule nodes; their finite log densities and conditional means are retained in `quadrature-nodes.npz`. No nodes were omitted ad hoc.

## Prespecified stability

| Comparison | Largest absolute change across 5 means, 5 CDFs and log-normalizer | Criterion |
|---|---:|---:|
| primary128_vs_primary256 | 1.243449787580175e-14 | ≤ 1e-8; passed |
| primary256_vs_expanded256 | 1.054711873393899e-15 | ≤ 1e-8; passed |

The frozen reported reference is expanded256. Its log-normalizer is **-9.835717167656208**. These are empirical rule/interval checks, not rigorous tail bounds or proof of convergence.

## Reference means and retained-draw comparison

Four-MCSE columns are descriptive bands around the corresponding retained estimates, using the previously saved named-parameter MCSE. All five means for both backends fall within those bands of the reference. Divergent draws were retained, so this is not a valid uncertainty guarantee.

| Parameter | Reference mean | PyMC mean | PyMC 4×MCSE | NumPyro mean | NumPyro 4×MCSE |
|---|---:|---:|---:|---:|---:|
| dose | 0.262383460552 | 0.264726151478 | 0.0370530654965 | 0.271227713533 | 0.0392005248294 |
| arm[control] | 0.420074991194 | 0.413848347489 | 0.0492877404161 | 0.410678997853 | 0.0533220837383 |
| arm[treated] | 1.63900871338 | 1.65220144925 | 0.0514608995855 | 1.65098473823 | 0.0580013559855 |
| dose:arm[treated] | -0.0660977768462 | -0.0724967986177 | 0.0511724811893 | -0.0776918128358 | 0.0533793931155 |
| sigma | 0.646441326672 | 0.636765495003 | 0.0351017039005 | 0.656199112654 | 0.0393923473436 |

## Sigma CDF comparison

Indicator MCSE was computed with ArviZ from each original 4×1000 chain array, with explicit single-value extraction and all divergent draws retained. No resampling occurred.

| Sigma point | Reference CDF | PyMC empirical CDF | PyMC 4×MCSE | NumPyro empirical CDF | NumPyro 4×MCSE | Descriptive band comparison |
|---|---:|---:|---:|---:|---:|---|
| 0.2 | 0.00141271496819 | 0.001 | 0.00244477024034 | 0.00025 | 0.000997993987958 | NumPyro outside |
| 0.3 | 0.0531610918197 | 0.0465 | 0.0177049034761 | 0.052 | 0.0242328387937 | both within |
| 0.4 | 0.201243123517 | 0.197 | 0.0405522486364 | 0.19175 | 0.0404973398812 | both within |
| 0.6 | 0.54018588765 | 0.5505 | 0.0559506602569 | 0.52925 | 0.0560757641 | both within |
| 1.0 | 0.879836321707 | 0.89425 | 0.029339998961 | 0.863 | 0.0354978406679 | both within |

**One descriptive rare-tail flag:** NumPyro has 1/4000 draws at sigma≤0.2 (0.00025), against reference 0.0014127149681932303. Absolute gap 0.0011627149681932303 exceeds its estimated four-MCSE band 0.0009979939879578431. PyMC has 4/4000. With such sparse indicators and divergent chains, these MCSE bands are unreliable; this observation does not establish bias, its cause, or a library defect. All other CDF comparisons are within their descriptive bands. The original zero-divergence acceptance remains failed regardless.

## Integrity and reproduction

All **39 frozen input hashes** and the clean exact worktree head verified before and after execution. This includes all 19 original frozen inputs, all 17 original evidence files (with two script overlaps), their manifest, the coordinator dispatch and diagnostic review, and the two new scripts. The existing draw-file hashes remain `041046187d8edc38ccbfc85a04e09951dc83d0397c89c67956d262a94082a3ee` (PyMC) and `db33a43459b91da6bb7154a29a1acdfd357b0b644d8b3230dcb966b558a220a7` (NumPyro).

Frozen plan SHA-256: `bdf477832d502b57b5c972546621594b2e3cb3581e599bcd924ad84075f809ab`. Frozen configuration/script/input manifest SHA-256: `935c59a9f9bacce98fe799cbce9f2518c40913b9c4dc969a80e77c8e3c0a3b8d`.

Outer command: `/usr/bin/python3 /private/tmp/autobayes-intercept-reference-1/supervisor.py`. This is a historical invocation; do not rerun it. `launch.json` and `exit-receipt.json` retain exact child argv, environment, timestamps and exit codes. `environment.json` records the installed pinned versions.

- `reference-result.json`: full-precision reference, all 22 stability differences, retained-draw comparisons and integrity checks.
- `quadrature-estimates.json`, `rules.jsonl`, `quadrature-nodes.npz`: all 18 rules, cumulative evaluation counts, finite log densities, log quadrature weights and conditional means.
- `reference.py`, `supervisor.py`, `freeze.json`: exact frozen computation and controls.
- `stdout.txt`, `stderr.txt`, `exit-receipt.json`: unedited run evidence.

## Proposed next sampler diagnostic — not executed

If separately authorized, one controlled two-backend unit could retain this exact target, seeds, chain count, tuning and retained lengths while changing only target_accept from 0.9 to 0.99. Keep the original diagnostic thresholds, hard deadline and zero-retry policy; compare retained means and the frozen sigma CDF points to this reference and report rare-tail counts explicitly. This would test sensitivity to smaller integration steps; it would not prove a causal explanation or automatically resolve rare-tail uncertainty. Any such work needs its own frozen authorization and original failures remain preserved. No further computation was performed in this unit.
