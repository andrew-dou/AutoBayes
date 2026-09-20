# Multiple-intercept CPU acceptance — 2026-09-20 UTC

The frozen unit passed all prespecified gates at unchanged clean HEAD `40613ed44db44a7ab84af46b38cad0421f49eb25`. Exactly **two** fits ran, PyMC then NumPyro, with four chains, 1000 tune/1000 retained draws, target_accept 0.99, seeds 109/113. No retries, source changes, new response draws or extra numerical execution. This is one crossed Gaussian CPU backend/API case; independent acceptance remains root’s decision.

The sole process-group invocation ran from 2026-09-20T14:21:13.523073+00:00 to 2026-09-20T14:22:39.944185+00:00, **86.42034725 seconds**, exit 0, no timeout, child reaped and process group 3775 absent. No worker CI waiter was started. Exact command/environment/deadline are in launch.json and exit-receipt.json.

All 37 frozen hashes verified before/after execution and at closeout. Data: 60 rows, 6 counties × 5 years × 2 replicates, x±0.75; residuals generated once with default_rng(731). JSON SHA256 `7e4ce583822f8159dc8dbb6bf9c3325fd242abf7aeb9e7bbecb6e0ce1b6121a9`; CSV `297856c30f410c749aff68d1698ecf5e2737c7d6ecf0cdf8247cd90b7091ab57`; config `f8c5f0504c852be823856dfef0dadb9371c6e1dc4139cd63726279a0c40c6e31`; freeze `79c6bee9fa74758dd1b6b79d8cff43e5e1eb1d49f9379bdb4df7ddbfc44bef10`.

The original-input conditional-Gaussian oracle passed: maximum negative-Hessian precision error 1.13686837722e-13, gradient at analytic mean 7.32747196253e-15, and graph arithmetic error 0. Common and both group matrices, row indices, labels and centering all matched. Full A/Q/mean/Hessian/gradient are retained in oracle.json.

| Backend | Divergences | Max R-hat | Min bulk ESS | Min tail ESS | Min BFMI | Max depth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pymc | 0 | 1.006239811 | 1073.237587 | 1478.378413 | 0.736578123 | 7 |
| numpyro | 0 | 1.004088455 | 1247.245830 | 1330.024080 | 0.713572570 | 7 |

All 16 named scalar/effect diagnostics were finite and passed. Offsets were not retained by the existing backend return convention. All active posterior/statistic values were finite; only PyMC’s prespecified structurally all-NaN smallest_eigval/largest_eigval placeholders were excluded. Per-chain BFMI and exact parameter values are in result.json and parameter-diagnostics.csv.

| Parameter | PyMC mean | NumPyro mean | Absolute difference | 4 combined MCSE |
| --- | ---: | ---: | ---: | ---: |
| Intercept | 0.325340651726 | 0.331596462653 | 0.00625581092743 | 0.0431331123414 |
| x | 0.599126688241 | 0.601661759271 | 0.00253507103072 | 0.0120098530453 |
| 1&#124;county_sigma | 0.539710457938 | 0.53049238011 | 0.009218077828 | 0.0360506110179 |
| 1&#124;year_sigma | 0.142510569154 | 0.145842321738 | 0.0033317525847 | 0.0162110897593 |
| sigma | 0.747209796738 | 0.749035379278 | 0.00182558253945 | 0.00829363960893 |

All five mean gates passed. All eleven group-effect comparisons are also reported in backend-comparisons.csv as descriptive results; they do not establish calibration.

Both backends passed every retained-draw semantic gate. Predictions at all 60 training rows and six frozen reordered rows match raw Intercept+x*b+county+year arithmetic with maximum absolute error 4.440892098500626e-16. Group dimensions/labels, distinct priors and scales, noncentering, requested/effective centering, JSON fit records and recorded execution settings are correct. compare_records accepts equal parameter sets and rejects the deliberately missing-sigma set. Since mean(x)=0, no broader recentering claim is made.

Retained draws, saved immediately after each fit:

- `pymc.nc`: 823220 bytes, SHA256 `e46fb305624d0ed1283da14d2b1d0df50858b2f72ab54ff2a3e2353c192d9665`; actual CPU float64; chain method `None`; fit wall 19.68729375s including compilation.
- `numpyro.nc`: 680171 bytes, SHA256 `454311f7a379669d8716820b7681c366e8cbe95e0089b9ac084816e44de14171`; actual CPU float64; chain method `parallel`; fit wall 4.76830779s including compilation.

Pinned PyMC/NumPyro constructor depth defaults 10 were source-hashed before execution and signature-asserted at runtime; no conflicting sampler kwargs or defaults were changed. environment.json contains actual devices/dtypes and all package versions: pymc=5.28.5, pytensor=2.38.3, bambi=0.17.2, formulae=0.6.2, arviz=0.23.4, numpyro=0.21.0, jax=0.10.2, jaxlib=0.10.2, numpy=2.4.6, pandas=3.0.5, scipy=1.17.1, xarray=2026.7.0.

No failed gate, timeout or execution error occurred. The pinned PyMC Hessian-sign FutureWarning and font-cache message remain in stderr.txt; the oracle verifies the current negative-Hessian convention. Original implementation/full-suite failures remain untouched. No calibration, other-family sampling, novel-group accuracy, GPU or speed claim follows from this unit. Root owns final draw review, hosted CI and merge.
