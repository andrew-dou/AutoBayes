# M2 decision record: grouped binomial counts, logit link

> **Current contract (2026-09-10, after two review repairs).** Accepted counts are integers in
> $[0, 10^6]$ per row (`autobayes.formula.COUNT_CAP`); the original default of $2^{31}-1$ in the
> sections below is **historical** and was lowered by the domain-reduction rule (repair record).
> The likelihood's binomial coefficient is a function of the evaluated value (correction record
> at the end), computed inside the graph; the cap is a measured engineering boundary of the
> tested evaluation, not a claim about what other formulations could reach.

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

## Repair record (2026-09-10): the normalised density near the mode failed the tolerance

**Finding (independent review, `memory/reviews/2026-09-10-autobayes-m2-review.md`, artifacts
`artifacts/2026-09-10-autobayes-m2-independent/`).** `binomial_logit_logp` formed
$\log\binom{n}{k}$ as a float64 difference of three `gammaln` values, each of magnitude
$\sim n \ln n$, and then subtracted likelihood terms of magnitude $\sim n$. Near a mode the
true log density is $O(-10)$ while the pieces are $O(10^{10})$, so the float64 spacing of
the pieces ($\sim 10^{-6}$ at $n = 10^9$) becomes the absolute error of the result. Two-row
intercept-only models, one row per group, $s_g = 1$, $z = 0$, Normal(0, 1.5) intercept
prior, HalfNormal(1) SD prior, reference = `tests/reference_binomial.py` (60-digit
`decimal`, Stirling `lnGamma` with a stated remainder), tolerance $\max(10^{-8}, 10^{-12}|v|)$:

| $n$ | $k$ | $\eta$ | reference total | both paths | error |
| --- | --- | --- | --- | --- | --- |
| $10^9$ | $1$ | $-20.723$ | $-101.32223803876772$ | $-101.32223650953556$ | $1.53\times10^{-6}$ |
| $10^9$ | $5\times10^8$ | $0$ | $-25.062920603102775$ | $-25.062917279025356$ | $3.32\times10^{-6}$ |
| $10^9$ | $10^9-1$ | $+20.723$ | $-101.32223803876772$ | $-101.32223650953557$ | $1.53\times10^{-6}$ |
| $2^{31}-2$ | $1$ | $-21.488$ | $-108.4914823522026$ | $-108.49147937122844$ | $2.98\times10^{-6}$ |
| $2^{31}-2$ | $2^{30}-1$ | $0$ | $-25.827217362316176$ | $-25.82720618130807$ | $1.12\times10^{-5}$ |
| $2^{31}-2$ | $2^{31}-3$ | $+21.488$ | $-108.4914823522026$ | $-108.49147937122844$ | $2.98\times10^{-6}$ |
| $2^{31}-1$ | $1$ | $-21.488$ | $-108.49148235664967$ | $-108.49147937660685$ | $2.98\times10^{-6}$ |
| $2^{31}-1$ | $2^{30}-1$ | $-9.3\times10^{-10}$ | $-25.827217362781838$ | $-25.82720904233102$ | $8.32\times10^{-6}$ |
| $2^{31}-1$ | $2^{31}-2$ | $+21.488$ | $-108.49148235664967$ | $-108.49147937660685$ | $2.98\times10^{-6}$ |

Nine of fifteen probed cases ($n \in \{10^3, 10^6, 10^9, 2^{31}-2, 2^{31}-1\}$, $k \in \{1, \lfloor n/2
\rfloor, n-1\}$ at their modes) failed on each path; the $10^3$ and $10^6$ cases passed
($1.4\times10^{-9}$ at $n = 10^6$, $k = n/2$, a margin of only seven). The reviewer's two
standalone regressions fail on the reviewed head `49052ac`.

**Why A5 passed.** The mixed-grid model put $k = 0$ and $k = n$ rows at the same $\eta$ as the
interior rows, so at $\eta = 0$ the total density was $O(n)$ and the relative tolerance was
thousands of times looser than the near-mode check the reference was written for. The
mixed grid is kept, and per-row normalised checks and homogeneous near-mode models with
non-trivial parameters are added (below), so that no large unrelated term can mask a
small component.

**Measured before choosing the repair (2026-09-10, scratch script, both paths, near-mode
$\eta = \mathrm{logit}(k/n) + 10^{-3}$, one row).** With the coefficient computed accurately as
data (Loader decomposition $n H(q) + c(n,k)$, `stirlerr` series) and the stable
log-sigmoid likelihood terms:

| $n$ | $k$ | coefficient error, `gammaln` difference | coefficient error, decomposition | value error | gradient error |
| --- | --- | --- | --- | --- | --- |
| $10^6$ | $n/2$ | $7.0\times10^{-10}$ | $3.1\times10^{-13}$ | $7.0\times10^{-11}$ | $2.4\times10^{-11}$ |
| $10^6$ | $n-1$ | $1.7\times10^{-13}$ | $4.0\times10^{-10}$ | $4.0\times10^{-10}$ | $4\times10^{-17}$ |
| $10^7$ | $n/2$ | $1.2\times10^{-8}$ | $6.2\times10^{-10}$ | $1.3\times10^{-9}$ | $2.9\times10^{-10}$ |
| $10^7$ | $n-1$ | $1.5\times10^{-8}$ | $8.5\times10^{-9}$ | $8.5\times10^{-9}$ | $6\times10^{-17}$ |
| $10^8$ | $n/2$ | $2.3\times10^{-7}$ | $8.9\times10^{-9}$ | $1.8\times10^{-8}$ (fail) | $2.0\times10^{-9}$ |
| $10^9$ | $n/2$ | $2.1\times10^{-6}$ | $2.9\times10^{-8}$ | $3.0\times10^{-8}$ (fail) | $1.2\times10^{-8}$ (fail) |
| $2^{31}-1$ | $2^{30}-1$ | $3.5\times10^{-6}$ | $5.6\times10^{-8}$ | $2.0\times10^{-7}$ (fail) | $6.4\times10^{-8}$ (fail) |

Three float64 facts follow. (i) The coefficient must not be a `gammaln` difference; for
$m = \min(k, n-k)$ small it must be the exact sum $\sum_{j=1}^{m}\log\frac{n-m+j}{j}$ (the
entropy decomposition loses accuracy there: $8.5\times10^{-9}$ at $n = 10^7$, $k = n-1$), and
for larger $m$ the decomposition with the `stirlerr` series is accurate to $\sim 10^{-12}$
up to $10^7$. (ii) Even with an exact coefficient, the likelihood terms $k\,\mathrm{softplus}(-\eta)$
and $(n-k)\,\mathrm{softplus}(\eta)$ are of magnitude $\sim n\ln 2$ near the mode, and their
float64 spacing ($1.2\times10^{-10}$ at $n = 10^6$, $7.5\times10^{-9}$ at $10^8$) is an error
floor of the combined value unless the deviance form $-n\,\mathrm{KL}(q\|p)$ is used instead.
(iii) The gradient as evaluated here, $k - n\,\mathrm{expit}(\eta)$, carries an error
$\sim n\,u\,p(1-p)$ from the float64 rounding of $\mathrm{expit}(\eta)$ ($u = 2^{-53}$):
$1.2\times10^{-8}$ at $n = 10^9$ and $6.4\times10^{-8}$ at $2^{31}-1$, above the tolerance
floor at the mode where the true gradient is $\approx 0$. This is a property of that
evaluation, not of the problem: at $k = n/2$, for example, the derivative equals
$-(n/2)\tanh(\eta/2)$, which subtracts nothing. The cap below is a measured boundary of the
tested construction, not a claim that no construction could pass further.

**Chosen repair (bounded).** (a) The coefficient is computed as data, per row, in
`autobayes/likelihoods.py`: exact summation for $m \le 1000$, Loader decomposition with the
`stirlerr` series for $m > 1000$; the likelihood keeps the stable log-sigmoid form. (b) The
accepted count cap is lowered by the approved domain-reduction rule to $N_{\max} = 10^6$,
the largest value of the acceptance grid at which every point passes on both paths, with a
margin of at least 25 on values (worst measured $4\times10^{-10}$) and 400 on gradients.
Exploration shows $10^7$ also passes at the probed points with a margin of about seven; it
is not adopted because a single-digit margin does not cover libm differences across
platforms. (c) The deviance form, which would allow $\sim 10^8$ before the gradient floor
binds, is a larger change to the sampled graph on both paths (its own saturation regime,
JAX gradient safety of the branch structure) and is recorded here as the route if a user
need for counts above $10^6$ per row ever arises; it is not made in this slice.

**Acceptance grid for the repair (both paths, values and full gradients, tolerance
unchanged, reference error accounted for separately as before):**

- $n \in \{1, 2, 10, 10^3, 10^5, N_{\max}-1, N_{\max}\}$, $N_{\max}+1$ refused;
  $k \in \{0, 1, 2, 10, \lfloor n/4 \rfloor, \lfloor n/2 \rfloor, \lfloor 3n/4 \rfloor, n-2, n-1, n\}$ deduplicated;
  $\eta$ at the A5 points and, for $0 < k < n$, at the mode $\mathrm{logit}(k/n)$ and at
  offsets $\pm 10^{-3}, \pm 10^{-1}, \pm 1$ from it.
- **Per-row normalised checks**: one model per $n$ with one row per $k$, each row its own
  group; the per-row likelihood terms (`logp(sum=False)`) are compared row by row with the
  reference row term, and the per-row derivatives are read off the offset gradient
  components ($s_g = 1$, $z = 0$ make $\partial/\partial z_j$ the row-$j$ derivative exactly).
- **Homogeneous near-mode models with non-trivial parameters**: all rows at one $(n, k)$
  near the mode, with a slope on a varying predictor, $s_g \ne 1$ and $z \ne 0$ so rows sit at
  different logits; full model density and full gradient on both paths.
- The mixed-grid A5 check is kept on the new grid.
- The reviewer's two-row near-mode construction at $N_{\max}-1$ and $N_{\max}$ passes on
  both paths; the same construction at $10^9$ is refused by the cap.

Wording fixed with the repair: the binomial centred intercept is the per-trial success
log-odds, not the log-odds that a count equals 1; README and SKILL count three families;
the package description names M2; the cap and precision statements say what was tested,
not "exact for every finite input".

**Executed after the repair (2026-09-10, clean locked environment, Python 3.11.0).**

- The reviewer's two-row near-mode construction (intercept-only, one row per group, $s_g = 1$,
  $z = 0$) on the repaired tree, both paths, at the mode and $10^{-3}$ above it, tolerance
  $10^{-8}$: worst error $1.6\times10^{-10}$ at $n = 999999$ and $1.4\times10^{-10}$ at $n = 10^6$,
  $k = n/2$ (the reviewer measured $1.4\times10^{-9}$ at $n = 10^6$ on the previous head, and
  $3.3\times10^{-6}$ at $10^9$); $\le 3\times10^{-11}$ at $n = 10^5$; $\le 2\times10^{-13}$ at
  $n = 10^3$; $k = 1$ and $k = n-1$ rows at their modes $\le 1.4\times10^{-14}$ everywhere. A model
  with $n = 10^9$ is refused before any cast with the cap message.
- `tests/test_binomial.py`: 50 passed, including `test_per_row_terms_and_derivatives_both_paths`
  (every $(n, k, \eta)$ of the acceptance grid as an individual row term and derivative on
  both paths, plus the total), `test_homogeneous_near_mode_models_with_nontrivial_parameters`
  (16 cases: four $n$ up to the cap, $k \in \{1, n/4, n/2, n-1\}$, slope, $s_g = 0.3$, random $z$,
  seven mode offsets, full density and gradient on both paths), the reviewer's construction
  at $N_{\max}-1$ and $N_{\max}$, the mixed grid on the new $n$ list, and the refusal of
  $N_{\max}+1$ and of the three previously failing $n$.
- Full suites: fast suite **141 passed, 5 deselected** (144 s); slow suite **5 passed** (30.6 s; new run-specific evidence directories, earlier ones untouched).
- The coefficient routine against the 60-digit reference over $n \in \{1, 2, 10, 10^3, 1001, 2001,
  10^5, 10^6-1, 10^6, 10^7\}$ and thirteen $k$ each: worst error $1.2\times10^{-10}$ at $n \le 10^6$
  (the float64 spacing of the coefficient's own magnitude), $9.3\times10^{-10}$ at $10^7$.
- Example evidence regenerated with the repaired code (`docs/m2-evidence/`); the pre-repair
  outputs are kept under `docs/m2-evidence/before-repair/`. Posterior means moved by less than
  $3\times10^{-4}$ (within Monte Carlo noise), as expected for counts of at most 50 per row.
- Unchanged: the sampled variables, priors, centring, zero-trial policy, prediction
  semantics, the Bernoulli family (no D8 change), and GPU status (pending).

## Correction record (2026-09-10): the exposed distribution was normalised only at the observed outcome

**Finding (independent review, `memory/reviews/2026-09-10-autobayes-m2-distribution-review.md`,
artifacts `artifacts/2026-09-10-autobayes-m2-repair-independent/`; reproduced here on
`d98b730` with the public API, both paths).** The previous repair passed $\log\binom{n}{k_{\rm obs}}$
into the `CustomDist` as a fixed data parameter, so the RV's `logp(value)` used the observed
outcome's coefficient for every `value`. For $n = 2$, $\eta = 0$, the masses over $k = 0, 1, 2$
came out as $[0.25, 0.25, 0.25]$ (sum $0.75$) when $k_{\rm obs} \in \{0, 2\}$ and $[0.5, 0.5, 0.5]$
(sum $1.5$) when $k_{\rm obs} = 1$, against the correct $[0.25, 0.5, 0.25]$; the random method
still drew genuine binomial values, so `logp` and `random` described different
distributions. Out-of-range or non-integral values passed to `pm.logp` returned finite
numbers instead of zero mass. The parameter-space likelihood and gradients for the fitted
observed data were correct (the coefficient is a constant in $\eta$ there), so this finding
does not by itself make the reported posteriors or the recovery wrong; it makes the exposed
PyMC distribution mis-specified for any other outcome (log-likelihood of new responses,
reuse of the RV, generic predictive workflows).

**Correction (bounded; decided before coding).** The coefficient becomes a function of the
evaluated value inside the graph, on both paths, with no observed-outcome constant:

$$\log p(v \mid n, \eta) = \begin{cases} \log\binom{n}{v} - v\,\mathrm{softplus}(-\eta) - (n - v)\,\mathrm{softplus}(\eta) & v \in \{0, 1, \dots, n\} \\ -\infty & \text{otherwise (negative, above } n\text{, or non-integral)} \end{cases}$$

with $\log\binom{n}{v}$ evaluated symbolically as $0$ for $v \in \{0, n\}$ and otherwise by the
decomposition $n H(v/n) + \tfrac12\log\frac{n}{2\pi v (n-v)} + \mathrm{stirlerr}(n) - \mathrm{stirlerr}(v)
- \mathrm{stirlerr}(n-v)$, where $\mathrm{stirlerr}(x)$ is the series for $x \ge 16$ and
$\ln\Gamma(x+1) - (x+\tfrac12)\ln x + x - \tfrac12\ln 2\pi$ (with `gammaln`) below. Every
branch is evaluated on clipped arguments ($v$ and $n - v$ clipped to $[1, n]$) so that no
branch produces NaN or infinity; the case selection is a `switch` on the actual value. The
coefficient does not depend on $\eta$, so parameter gradients are unchanged from the
repaired likelihood; the value is never differentiated. The exact-summation branch of the
previous data-side helper cannot be expressed elementwise and is dropped: the measured
error of the decomposition alone is $\le 4\times10^{-10}$ at $n \le 10^6$ (worst at $k = n - 1$;
interior $\le 3\times10^{-13}$ at $10^6$), which is the value the cap decision already rested
on, so the $10^6$ cap and its margins stand. `random` is unchanged. This keeps the
inspectable-model contract (one normalised RV whose `logp` and `random` agree), downstream
log-likelihood semantics, and the stable gradients; it is not a larger architecture change.

**Acceptance for the correction (tests on the public API, `model.pymc_model`, both paths).**

- N1. Support enumeration: for $n \in \{1, 2, 3, 5, 10, 25\}$ and $\eta \in \{0, \pm 0.7, \pm 3, -12\}$,
  the pmf from a compiled function of a symbolic value equals the exact probabilities
  $\binom{n}{k} p^k (1-p)^{n-k}$ computed independently from `math.comb` and a 60-digit
  `decimal` expit (no SciPy) to $10^{-12}$ absolute, and sums to 1 within $10^{-12}$.
- N2. Observed-outcome independence: for fixed $n$ and $\eta$, models built with each
  possible $k_{\rm obs}$ give identical pmfs (to $10^{-14}$) and the correct one.
- N3. Symbolic value after compilation: one compiled function of a value tensor is evaluated
  repeatedly for every $k$ and again in a second pass; results identical to N1.
- N4. Random against mass: $N = 20000$ draws from the same distribution (`.dist` at fixed
  $n$, $\eta$) have per-outcome frequencies within Hoeffding's bound
  $\sqrt{\ln(2 (n+1)/\delta)/(2N)}$ of the exact probabilities with $\delta = 10^{-6}$, and the
  draw mean within the same bound of $n p$; recorded false-failure budget $10^{-6}$.
- N5. Zero mass: `pm.logp` at $-1$, $n + 1$ and a non-integral value returns $-\infty$ on both
  paths, for integer and float value tensors; valid values are unchanged.
- N6. Everything from the repair record still passes: per-row terms and derivatives over the
  acceptance grid, homogeneous near-mode models, the reviewer's two-row construction,
  $n = 1$ equivalence with the Bernoulli family, expansion with aligned centring,
  predictions, the $10^6$ cap and its refusals, and the CPU recovery.
- N7. For the fitted data of the M2 example (counts $\le 50$), the observed-data log density
  changes from the previous head by less than $10^{-11}$ (the exact-summation and the
  decomposition coefficients agree to rounding at those counts); recorded, not re-sampled.

Wording corrected with this record: the top of this file states the current $10^6$ cap and
labels the $2^{31}-1$ default historical; the refusal message and SKILL no longer suggest
splitting large denominators across rows (duplicating design rows changes the unweighted
centring and therefore what the centred-intercept prior refers to, unless the centring is
preserved or the prior transformed); the coefficient docstring states the measured
$4\times10^{-10}$ / $1.2\times10^{-10}$ figures; and the gradient floor is described as a
property of the tested evaluation $k - n\,\mathrm{expit}(\eta)$, not a universal one (at
$k = n/2$, for example, the derivative equals $-(n/2)\tanh(\eta/2)$, which subtracts nothing).

### Executed results for the correction (2026-09-10, clean locked environment, both paths)

**Reproduction before the change (`d98b730`, `docs/m2-evidence/correction/normalization_before_d98b730.json`,
script `scripts/norm_evidence.py`, public API, $n = 2$, $\eta = 0$):**

| observed $k$ | pmf over $k = 0, 1, 2$ (PyTensor = JAX) | total mass |
|---|---|---|
| 0 | 0.25, 0.25, 0.25 | 0.75 |
| 1 | 0.5, 0.5, 0.5 | 1.5 |
| 2 | 0.25, 0.25, 0.25 | 0.75 |

Off-support values ($-1$, $3$, $5$, $-7$ as integers; $-1.0$, $3.0$, $1.5$, $0.5$ as floats)
all returned $\log p = -0.693$ on both paths. This agrees with the coordinator's independent
probe (`artifacts/2026-09-10-autobayes-m2-repair-independent/normalization-probe.json`).

**After the change (`normalization_after.json`):** pmf $0.25, 0.5, 0.25$ with total mass
$1.000000$ for every observed $k$, integer and float value tensors, both paths; every
off-support value returns $-\infty$ on both paths.

**Coefficient accuracy of the symbolic decomposition (`coefficient_accuracy.json`, script
`scripts/coef_measure.py`; 60-digit reference; acceptance grid `grid_ks(n)` plus $k \in
\{15, 16, 17, n-15, n-16, n-17\}$ across the `stirlerr` branch boundary):**

| $n$ | max abs error, PyTensor | max abs error, JAX | at $k$ | max rel error |
|---|---|---|---|---|
| 1, 2 | 0 | 1.3e-15 | 1 | 1.3e-15 |
| 10 | 1.8e-15 | 1.1e-14 | 2 | 3.5e-15 |
| 1e3 | 1.1e-13 | 1.1e-13 | 250 | 2.7e-16 |
| 1e5 | 1.5e-11 | 1.5e-11 | 50000 | 4.0e-13 |
| 999999 | 1.2e-10 | 1.2e-10 | 249999 | 4.4e-13 |
| 1e6 | 1.2e-10 | 1.2e-10 | 250000 | 2.1e-12 |
| 1e7 (outside the cap) | 9.3e-10 | 9.3e-10 | 2500000 | 3.3e-11 |

The worst absolute error is one float64 ulp of the coefficient's own magnitude, measured at
$k = n/4$ on this grid (the coefficient's mathematical maximum is at $k = n/2$, where the grid
point $n/2$ showed a smaller error; the ulp bound is the same order at both); the $k = n - 1$ and interior
figures quoted in the correction text above ($4\times10^{-10}$, $3\times10^{-13}$) were the
previous data-side helper's measurements and are superseded by this table, which the
coefficient docstring now cites. The cap decision is unchanged: the per-row density and
gradient checks over the grid (N6) pass at the same tolerance as in the repair record.

**N7 (`n7_example_logdensity_*.json`, script `scripts/n7_logdensity.py`):** the M2
example's fitted data (298 rows after the two zero-trial drops, counts $\le 50$) at six fixed
parameter points, previous head `d98b730` against the corrected tree: largest change in the
total model log density $4.5\times10^{-13}$ (both paths), in any per-row likelihood term
$7.1\times10^{-15}$; threshold $10^{-11}$, passed. The committed example outputs
(`docs/m2-evidence/example_*`) were therefore not re-sampled.

**Tests (`tests/test_binomial.py`, new N1-N5: 42 parametrised cases).** N1 support
enumeration: 36 cases pass with the pmf within $10^{-12}$ of `math.comb` $\times$ decimal expit
and total mass within $10^{-12}$ of 1, integer and float value tensors, both paths. N2: pmfs
identical to $10^{-14}$ across observed $k$ and correct. N3: one compiled function per path
evaluated over the support twice and at a second parameter point. N4: 20000 draws from
`BinomialLogit.dist` at $(n, \eta) \in \{(2, 0), (10, -0.7), (25, 3)\}$ within the Hoeffding
bound ($\delta = 10^{-6}$) per outcome and in the mean. N5: $-\infty$ at $-1$, $n+1$, $1.5$,
$5.5$, $-0.5$ on both paths for float and integer value tensors, valid entries of the same
vectors unchanged. Module: **92 passed (190 s)**. Fast suite: **183 passed, 5 deselected (240 s)**. Slow suite: **5 passed (31 s)**
(N6: every repair-record check, the $n = 1$ equivalence, expansion, predictions, the cap
refusals and the CPU recovery on both backends pass unchanged). Evidence directories of
these runs: `evidence-runs/20260910T162905.886985Z-8d76d649a5bc-dirty-p41180-00e060/` (fast) and `evidence-runs/20260910T162959.493761Z-8d76d649a5bc-dirty-p41519-cd3b40/` (slow); earlier run directories untouched.
