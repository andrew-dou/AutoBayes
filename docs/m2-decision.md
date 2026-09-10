# M2 decision record: grouped binomial counts, logit link

Date: 2026-09-10. Branch `m2/binomial-trials` from `main` at `4eee25c` (foundation merged at
`8015374`). Written and committed before implementation. Source of the defaults: Andrew's
approval (`iqss-research/plans/2026-09-09-autobayes-approval-brief.md`, 2026-09-09 evening) of
section 3 of `docs/coordination/NEXT_MILESTONES_PROPOSAL.md` (revision 3) with the reviewer's
clarifications; the proposal's "awaiting approval" wording is historical.

## Accepted defaults

- **Formula**: `successes | trials(n) ~ [1 +] x1 [+ x2 ...] + (1 | group)`, brms syntax only,
  numeric predictors, one grouping factor. `ab.binomial(formula, data, priors)`. Internally the
  left-hand side is rewritten to Bambi's `p(successes, n)`; Bambi's `p(k, n)` is not accepted
  as user input.
- **Priors**: `Priors(intercept, b, sd)` on the logit scale, mandatory, no `sigma` (giving one
  is an error). No silent defaults.
- **Counts, validated before any cast**: `successes` and `trials` must be numeric columns
  whose values are finite and integral (integer dtypes, or float columns holding integers
  such as `3.0`); bool, complex, non-integral, non-finite, negative values, `successes >
  trials`, and any value above the cap $N_{\max} = 2^{31}-1$ are refused with a message naming
  the column and the first offending value. The cap is the boundary of the proposed
  validation domain (below), not a claim about larger counts.
- **Zero-trial rows**: rows with `trials = 0` and `successes = 0` are accepted, counted, and
  **dropped before the model is built**, so they do not enter the centring means or the
  design; `describe()` and the fit record report the count and the row positions. A row with
  `trials = 0` and `successes > 0` is impossible data and is refused. A data set with no
  trials at all is refused ("prior-only inference must be requested explicitly").
- **Constant outcomes**: all-zero or all-success data are accepted under the proper priors.
  `describe()` and the fit record carry a separation warning: the likelihood is monotone
  along the intercept direction, so the posterior there is set by the prior's tail and the
  data's extent, and the result must be checked for prior sensitivity (refit with a wider and
  a narrower intercept prior; report the movement). M1's Bernoulli policy (a constant 0/1
  response is refused) is retained unchanged; it is a documented difference between the
  families, not a silent change.
- **Predictions**, three named quantities: `kind="probability"` returns `p` (no trials
  needed; brms `posterior_linpred(transform = TRUE)`); `kind="mean"` returns `p` and
  `expected = n p` (brms `posterior_epred`, which is `mu * trials`); `kind="response"`
  additionally returns integer draws `y ~ Binomial(n, p)` (brms `posterior_predict`). For
  `newdata`, the trials column named in the formula is required for `mean` and `response`
  and refused-if-absent with a message; it is ignored for `probability`. Rows with `n = 0` in
  `newdata` give `expected = 0` and `y = 0` deterministically. New-group semantics are M1's:
  `error` (default), `resample_observed` (one observed level per distinct unseen label per
  draw, shared by rows with that label), `population` (a fresh `Normal(0, sd)` effect on the
  logit scale per label per draw). In-sample predictions cover the fitted rows (the retained
  ones); all binomial predictions are computed in this layer from the posterior, not through
  Bambi's `predict`.
- **Centring and intercept**: predictors centred at the means of the fitted rows (after the
  zero-trial drop), unweighted, as brms and Bambi do; the prior on `intercept` is on the
  log-odds at those means; the reported `Intercept` is the raw-scale log-odds intercept.
  Non-centred group effects. `describe()` states all of this.
- **Devices**: unchanged (C1-C3 below). GPU execution of this family is **pending**.
- **Bernoulli boundary form (D8)**: not changed in this slice. If changed later it needs its
  own diff and the sensitive regression described in the proposal (likelihood term for
  `y = 1` at `eta = 40` compared with `-log(1 + e^{-40})` to 1e-3 relative on both paths).

## Mathematical target

Rows $i = 1..n_{\text{rows}}$ with successes $k_i$ of trials $n_i$, predictors $x_{ip}$,
group index $g_i$ into sorted levels $j = 0..J-1$:

$$\eta_i = a_c + \sum_p (x_{ip} - \bar x_p)\, b_p + s_g z_{g_i}, \qquad
k_i \sim \mathrm{Binomial}\big(n_i, \mathrm{expit}(\eta_i)\big),$$

$$\log p(k_i \mid n_i, \eta_i) = \log\binom{n_i}{k_i} - k_i\,\mathrm{softplus}(-\eta_i)
- (n_i - k_i)\,\mathrm{softplus}(\eta_i), \qquad
\frac{\partial \log p}{\partial \eta_i} = k_i\,\mathrm{expit}(-\eta_i) - (n_i - k_i)\,\mathrm{expit}(\eta_i),$$

with $z_j \sim \mathrm{Normal}(0,1)$, the M1 priors on $a_c$, $b_p$, $s_g$, and the Jacobian
$+\log s_g$ in the sampled density. The sampled value space is M1's: `Intercept` ($a_c$), one
variable per slope, `1|g_sigma_log__`, `1|g_offset`. The binomial coefficient is a constant
in $\eta$, kept so the density is normalised.

## Numerical construction and what was measured first

Measured on this stack (pymc 5.28.5, pytensor 2.38.3, jax 0.10.2) on 2026-09-10 before
choosing the construction: the likelihood term (binomial coefficient subtracted) and its
$\eta$-gradient for one observation, $n \in \{1, 10, 10^6, 2^{31}-1\}$, $k \in \{0, \lfloor n/2
\rfloor, n\}$, $\eta \in \{0, \pm 0.5, \pm 12, \pm 40, \pm 300, \pm 1000, \pm 10^6\}$, 130 points,
against a 60-digit `decimal` log-sigmoid reference, tolerance $\max(10^{-8}, 10^{-12}|v|)$:

| Construction | Failing points of 130 | Per check (PyTensor value / gradient, JAX value / gradient) | Smallest failing $\lvert\eta\rvert$ |
| --- | --- | --- | --- |
| Bambi built-in `binomial` (`p = sigmoid(eta)` with eps substitution, `pm.Binomial(p=)`) | 49 | 49 / 43 / 49 / 43 | 12 for every check (large $n$: the $\log(1-p)$ term loses precision), floored beyond ~37 |
| `pm.Binomial(logit_p=eta)` | 60 | 42 / 42 / 44 / 58 | 40 on the PyTensor path (wrong, finite values and gradients on the floored side); 12 for the JAX value; JAX gradients fail at $\lvert\eta\rvert \ge 40$ including non-finite ones at large magnitudes |

Neither is acceptable. The construction is a `pm.CustomDist` inside a Bambi custom family
(`Likelihood(params=["logit_p"], parent="logit_p", dist=...)`, identity link, `n` attached
through `transform_backend_kwargs` exactly as Bambi's own `BinomialBaseFamily` does; a
build-only check on 2026-09-10 confirmed the observed variable `p(k, n)` holds the
successes, `n` is a constant input, and the free variables equal M1's) with the log density
above written with `stable_softplus` (M1's `Softplus` subclass whose JAX lowering is
`jax.nn.softplus`). Each likelihood term is a count times a non-positive stable quantity and
the two are added, so there is no cancellation at $k = 0$ or $k = n$; M1's form
$k\eta - n\,\mathrm{softplus}(\eta)$ is not used (it returns exactly 0 at $k = n$, $\eta = 40$
for every $n$; arithmetic check of 2026-09-09).

**Proposed validation domain.** Counts in $[0, N_{\max}]$, $N_{\max} = 2^{31}-1$. Test grid:
$n \in \{1, 2, 10, 10^3, 10^6, 10^9, N_{\max}-1, N_{\max}\}$ ($N_{\max}+1$ refused),
$k \in \{0, 1, \lfloor n/2 \rfloor, n-1, n\}$ deduplicated, $\eta$ at the A5 points, both paths,
values and full gradients. If any point fails, $N_{\max}$ is lowered to the largest tested $n$
at which every point passes on both paths, the refusal message states it, and the failures
are recorded here; the tolerance is not loosened.

**Independent reference and its error budget** (`tests/reference_binomial.py`, no PyMC,
no Bambi, no `gammaln`):

- Log-sigmoid terms $\mathrm{softplus}(\pm\eta) = \ln(1 + e^{\pm\eta})$ and $\mathrm{expit}(\pm\eta)$
  in 60-digit `decimal` arithmetic; error below $10^{-50}$ relative.
- $\log\binom{n}{k} = \ln\Gamma(n+1) - \ln\Gamma(k+1) - \ln\Gamma(n-k+1)$ with each $\ln\Gamma(z)$
  computed in `decimal`: exactly, as $\sum_{i=2}^{z-1}\ln i$, for $z \le 2000$; by the Stirling
  series $(z-\tfrac12)\ln z - z + \tfrac12\ln 2\pi + \sum_{m=1}^{6} \frac{B_{2m}}{2m(2m-1)z^{2m-1}}$ for
  $z > 2000$, whose remainder is bounded in magnitude by the first omitted term,
  $|B_{14}|/(14\cdot 13\cdot z^{13}) < 10^{-45}$ for $z > 2000$. The three $\ln\Gamma$ values are
  combined in `decimal`, so the cancellation that a float64 `gammaln` difference suffers
  near the mode does not occur in the reference.
- The reference row term is the `decimal` sum of the three pieces; the conversion of the
  total to float64 contributes at most one unit in the last place, $1.1\times10^{-16}$
  relative, far inside the $10^{-12}$ relative tolerance. Prior terms and the Jacobian use
  the M1 SciPy reference (magnitudes of order 10, absolute error below $10^{-13}$).
- Gradient reference: per-row $r_i = k_i\,\mathrm{expit}(-\eta_i) - (n_i-k_i)\,\mathrm{expit}(\eta_i)$
  in `decimal`, converted to float64, then propagated through the design by the same
  chain rule as M1's analytic gradient (sums of at most $n_{\text{rows}}$ products; relative
  rounding below $10^{-14}$).
- Tolerance applied to every comparison: $\max(10^{-8}, 10^{-12}\,|\text{reference}|)$; the
  reference error budget above is at least four orders of magnitude smaller, so no
  allowance for it is added.

## Acceptance criteria

- A1/A2. Log density with and without the Jacobian equals the reference at five random
  moderate points ($|\eta| < 15$) for `y ~ x + (1|g)` centred and uncentred, `y ~ x + w +
  (1|g)`, and `y ~ 1 + (1|g)`; the Jacobian difference is exactly $\log s_g$.
- A3. `compile_dlogp` equals central finite differences of the reference (rtol $10^{-5}$) and
  the analytic reference gradient (tolerance rule) at those points.
- A4. `get_jaxified_logp` equals the PyTensor path at those points.
- A5. Values and full gradients on both paths over the validation grid above, with rows
  holding mixed $k$ ($0$, $n$, interior) and the listed $n$; finite everywhere; the cap and
  its neighbours included; $N_{\max}+1$ refused before any cast.
- A6. Design and levels as M0/M1: common matrix `[1, x - mean]` over the retained rows,
  level order sorted, one-hot group matrix, `logit_p` from the graph equals the hand-computed
  $\eta$ and `p = expit(eta)` to $10^{-12}$; zero-trial rows dropped and counted, their
  positions reported.
- A6a. **`n = 1` equivalence on the shared domain**: for every row $n_i = 1$, $k_i \in \{0,1\}$,
  both outcomes present, `ab.binomial` and `ab.bernoulli` have the same sampled variable
  names and, at the same value points, log densities and gradients equal under the
  tolerance rule (the binomial coefficient is zero); `predict(kind="probability")` equals
  M1's `kind="mean"` from the same posterior. Observed-variable names (`p(k, n)` versus
  `y`) and the likelihood line of `describe()` may differ. Outside the shared domain the
  policies differ by design: a constant binomial outcome builds with a warning, a constant
  Bernoulli response is refused; a test asserts both.
- A6b. **Expansion with aligned centring**: for random aggregated data, the binomial log
  likelihood minus its combinatorial constant equals M1's Bernoulli log likelihood of the
  row-expanded data (each row repeated $n_i$ times with $k_i$ ones) at the same parameters,
  (i) with `center_predictors=False` on both models, and (ii) with centring on, where the
  expanded model's centring means are the trial-weighted means; the test passes the
  aggregated model the expanded data's means by construction (both models built on data
  whose unweighted and trial-weighted means coincide, obtained by choosing equal trials
  per row) and separately asserts that with unequal trials the two centrings differ.
- A7. Invalid inputs, each refused before any cast with a message naming the column:
  bool, complex, non-integral, non-finite, negative counts; `successes > trials`; a value
  above $N_{\max}$; `trials = 0` with `successes > 0`; all-zero trials; the trials column used
  as a predictor; `sigma` given; missing `sd`; every M0/M1 unsupported formula; Bambi's
  `p(k, n)` left-hand side; `newdata` for `mean`/`response` without the trials column.
- A8. Predictions: shapes (chain, draw, rows); `p` in $[0, 1]$ with saturation allowed;
  `expected = n p` exactly; `y` integer in $[0, n]$; observed levels use their own effect;
  new-group identity and sharing as in M1 (shared helper); `n = 0` rows in `newdata` give
  `expected = 0`, `y = 0`; `probability` needs no trials.
- A9. Predictive check conditional on the saved posterior array: with fresh response draws
  $y_s$ given $p_s$ (independent across draws and rows conditional on the array),
  $T = \sum_s y_s$ per row has conditional mean $E = \sum_s n p_s$ and variance
  $V = \sum_s n p_s(1-p_s)$. Rows with $V = 0$ (zero trials, or every $p_s$ saturated) are
  checked exactly (`y` equals `expected` in every draw). Rows with $V > 0$ use Bernstein's
  inequality for a sum of independent variables bounded by $n$: $|T - E| \le t$ with $t$
  solving $2\exp(-t^2/(2(V + n t/3))) = 10^{-6}$; the pooled check over rows uses the total
  conditional variance $\sum V$ and the largest per-row bound $\max n$ (a conservative bound
  for the pooled sum), with the same $10^{-6}$. Recorded false-failure budget: $10^{-6}$ per
  row and $10^{-6}$ for the pooled statistic.
- B1. Simulated recovery: 300 rows, 30 groups all present (indices cycled then shuffled),
  trials uniform on $[5, 50]$, truth $a = -0.5$, $b = 0.8$, $s_g = 0.7$, seed 7, 4 chains x
  (1000 warmup + 2000 draws), both CPU backends: zero divergences, max R-hat $\le 1.01$, bulk
  and tail ESS $> 400$, truth inside the 94 % HDI for `Intercept`, `x`, `1|g_sigma`, backends
  within 4 combined MCSE. One recovery on one simulated data set; not calibration.
- C1-C3. `device="gpu"` without a visible GPU raises `DeviceError`; the CPU NumPyro report
  reads float64 and `parallel`; GPU execution of this family is pending and labelled so.
- Docs: `examples/m2_binomial_varying_intercept.py`, a family section in `SKILL.md`
  (probability vs expected count vs count draws, zero-trial and constant-outcome policy),
  README supported-path row, STATUS, evidence records in run-specific directories.

## Kill criterion

If `n` cannot reach the custom likelihood through Bambi's `transform_backend_kwargs`
without patching Bambi (the build-only check says it can), or if the validation grid fails
at every tested cap so that no defensible $N_{\max}$ remains, the family is built on Bambi's
design matrices with the response handled entirely in this layer, and that larger diff is
checkpointed for review before it is written.

## If a check fails

Repair and re-run; record the failure here and in `docs/coordination/STATUS.md`. If B1
fails on one backend only, that is a sampler-path bug to investigate before any claim about
either backend. Thresholds and tolerances are not loosened.

## Executed results (2026-09-10, clean locked environment, Python 3.11.0, macOS arm64)

- Grid A5 (`tests/test_binomial.py::test_extreme_logits_and_large_counts_both_paths`): all
  eight $n$ values including $N_{\max}-1$ and $N_{\max}$, every $k$ and $\eta$ on the list,
  values and full gradients on the PyTensor and JAX paths within the tolerance rule, finite
  everywhere; $N_{\max}+1$ refused before any cast. No domain reduction was needed.
- A1-A4, A6, A6a, A6b, A7, A8, A9, C1 and the CPU NumPyro report: 26 tests in the module,
  all passing (`test_binomial.py`); the reference's own self-checks (Stirling versus exact
  summation at the branch point, known binomial coefficients) pass.
- Fixes made while writing the tests, none to the mathematics: the zero-trials-with-successes
  check now precedes the `successes > trials` check so its message is reachable; missing
  counts are reported as missing values before the count checks; `sigma` is refused for every
  non-Gaussian family (it was refused for Bernoulli only); a formula without `~` keeps its
  original message.
- B1 (`test_sampling.py::test_binomial_recovery_both_backends`, evidence
  `M2_B1_recovery.json` in the run directory): 300 rows, 30 groups all present, trials 5-50,
  seed 7, 4 x (1000 + 2000). PyMC: 0 divergences, max R-hat 1.0062, bulk ESS 781, tail ESS
  1557, E-BFMI 0.647; NumPyro (4 parallel CPU chains): 0 divergences, max R-hat 1.0073, bulk
  ESS 681, tail ESS 1055, E-BFMI 0.708. Truth inside every 94 % HDI: Intercept -0.673
  [-0.887, -0.463] (truth -0.5), x 0.851 [0.789, 0.909] (0.8), group SD 0.598 [0.442, 0.764]
  (0.7). Backends within 4 combined MCSE (largest ratio 0.33, on Intercept). One recovery on
  one simulated data set; not calibration. Note on the intercept: in a single data set it
  is identified only up to the realised mean of the simulated group effects; the example
  prints that mean and the identifiable value, and the recovery check keeps the prespecified
  comparison with the population value.
- Example (`examples/m2_binomial_varying_intercept.py`, `docs/m2-evidence/`): both backends,
  4 x (1000 + 2000), panel all pass, backend agreement `agree: True`; the two clinics with no
  patients are dropped and reported; a zero-patient prediction row gives expected 0 and
  count 0; probabilities need no trials column; the GPU refusal on this machine is captured.
- Built-in measurements (table above) were taken before the construction was chosen.
- GPU execution: pending, not run.
