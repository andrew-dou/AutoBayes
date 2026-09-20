# PR10 retained-draw semantics and descriptive divergence review

2026-09-19. Exact unchanged head `6a58eb3c9b8ab510aa8bd5d3d04936e64990dace`.

**All remaining semantic checks passed for both retained fits. No semantic defect was found. The original CPU acceptance remains failed because PyMC has 3 divergences and NumPyro has 6.** This separate read-only review neither repairs the original runner nor changes its exit 2, gates, or evidence.

One supervised numerical invocation completed in **23.818032875 seconds**, exit 0, no timeout, under a 120-second process-group deadline. Zero new posterior fits and zero new response draws. Model construction and expected-mean predictions only; public AutoBayes/Bambi fit, PyMC sample, posterior-predictive sampling, and random-draw entry points were disabled. Exact command, environment and script hashes are in `exit-receipt.json` and `invocation-started.json`.

## Semantic findings

Each backend passed all 14 checks. The formula and six input rows remain exactly the frozen example. No Intercept exists in graph naming, posterior, prior mapping, or fit-record posterior rows. Requested centering is true; effective centering is false; the raw design and zero centering offsets are exact. Full-level factor metadata, label/coordinate order, all explicit priors, and the five named parameters are correct.

The exact coefficient order is `dose`, `arm[control]`, `arm[treated]`, `dose:arm[treated]`; the fifth posterior parameter is `sigma`. Direct prediction arithmetic used named posterior selections and independently constructed columns `[dose, I(control), I(treated), dose*I(treated)]` from the frozen caller inputs, not `model.data` or its extracted design as the oracle.

| Backend | Fitted rows: maximum absolute error | New rows: maximum absolute error | Comparison count |
|---|---:|---:|---:|
| PyMC | 0 | 8.881784197001252e-16 | 40,000 |
| NumPyro | 0 | 8.881784197001252e-16 | 40,000 |

All 80,000 mean values are finite and meet the unchanged absolute/relative tolerances of 1e-10. Checks include all retained draws, including those marked divergent; arithmetic agreement does not certify those draws as unbiased posterior samples. The four new rows are the originally frozen `(dose, arm)` values `(0, control)`, `(-1.5, treated)`, `(3, control)`, and `(0.5, treated)`.

`pymc-fit-record-posthoc.json` and `numpyro-fit-record-posthoc.json` were generated separately from retained draws and their original execution metadata. The library's diagnostic panel retains its own preexisting warning thresholds; it is not the stricter prespecified zero-divergence acceptance rule. No record or panel status supersedes the original failure.

## Divergent retained endpoints: descriptive only

| Backend | Divergences per chain | Sigma median, divergent | Sigma median, nondivergent | Divergent endpoints below overall sigma 25th percentile | Below overall 1st percentile |
|---|---|---:|---:|---:|---:|
| PyMC | 0, 0, 2, 1 | 0.2367531741 | 0.5675393198 | 3 / 3 | 2 / 3 |
| NumPyro | 0, 1, 4, 1 | 0.4029737184 | 0.5781596734 | 3 / 6 | 1 / 6 |

PyMC divergent-endpoint sigma values range 0.1984723046–0.4160204246; NumPyro values range 0.2218121854–0.5891588338. Their empirical percentile ranks among each backend's retained sigma draws are respectively 0.1%, 0.575%, 22.725%, and 0.525%, 6.975%, 13.825%, 27.2%, 46.325%, 51.325%. Exact row/chain labels, all five parameter values, available sampler statistics, and divergent/nondivergent conditional summaries are in the backend JSON files.

These are retained endpoints of transitions carrying divergence flags, not complete divergent trajectories or necessarily the location at which integration became unstable. There are only 3 and 6 flagged endpoints. The lower-sigma association, strongest for PyMC, does not establish a cause, prove a particular posterior geometry, or distinguish library error from integration difficulty. No divergent draws were discarded or used to relabel acceptance.

## Proposed next diagnostic design — not executed

A bounded **independent numerical reference without sampling** would answer more than simply changing a tuning parameter. Keep exactly the frozen six observations, design, and priors. For coefficient order above, let `D = diag(1,4,4,1)`. Analytically integrate the four Normal coefficients: at each positive sigma, `y ~ Normal(0, C)`, where `C = sigma^2 I + X D X'`; multiply that normalized likelihood by the HalfNormal(1) sigma prior. Compute the remaining one-dimensional integral over log(sigma), including its Jacobian.

Before execution, freeze integration bounds, absolute/relative tolerances, and a tail/interval-expansion convergence check under one small CPU deadline. Cholesky solves yield conditional coefficient mean `D X' C^-1 y`; integrate it and sigma to obtain independent posterior means. Report the sigma CDF at a small fixed set of low-scale points (for example 0.2, 0.3, 0.4, 0.6, 1.0) with numerical error estimates. Compare these reference quantities descriptively to the already retained draws, retaining warnings about MCSE from divergent chains. This can distinguish agreement on a common target from shared backend bias or a target mismatch without further posterior fitting. Any subsequent sampler experiment requires its own fixed design and review; the original zero-divergence failure remains historical fact.

## Integrity and artifacts

All **19 original frozen input hashes and 17 original evidence hashes** verified both before and after this computation, including both retained netCDF files. The working tree remained clean at the exact head. Root's independent diagnostic review is referenced and hashed in `retained-semantic-review.json`; its other-gate/mean-comparison findings were not represented as new computations here.

- `retained-semantic-review.json`: complete semantic results, descriptive endpoint analysis, and input preservation evidence.
- `pymc-semantics-and-divergences.json`, `numpyro-semantics-and-divergences.json`: backend-specific results.
- `pymc-fit-record-posthoc.json`, `numpyro-fit-record-posthoc.json`: newly generated records from the saved fits.
- `review.py`, `supervisor.py`, `stdout.txt`, `stderr.txt`, `exit-receipt.json`: exact computation and unedited execution evidence.

No source, test, example, lockfile, branch, original attempt artifact, or PR was modified. No network, install, GPU job, retry, or merge occurred. This review is not calibration, all-family sampling validation, or GPU evidence.
