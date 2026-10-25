# Response addition terms: `weights()`, `cens()` and `trunc()`

brms-parity features B02 and P6. brms writes information about the response on the left-hand side of the formula, after
a `|`, as addition terms joined with `+`: `k | trials(n)` (the binomial trials, M2), `y | weights(w)`,
`y | cens(c)`, `y | trunc(lb = 0)`, and combinations such as `k | trials(n) + weights(w)`. This library parses all
four into the term model; `weights()` is offered for the nine Tier A families (each verified by a frozen weighted case) and, since D03, for the eight zero-inflated and hurdle families (the frozen cases `zero_inflated_poisson_weighted` and `hurdle_poisson_weighted`; docs/zero-inflated-hurdle.md), and `cens()` and `trunc()` for
the Gaussian family; since E04 (docs/cens-trunc-families.md) interval censoring `cens(c, y2)`, `cens()` for the Weibull and
lognormal families and `trunc(ub = b)` for the Poisson and negative binomial families too. These two were withheld (B02) because brms's own numbers for their frozen fixtures are not exact
everywhere; since P6 they are accepted against two references, brms where brms is accurate and a committed 50-digit
reference where it is not (below, and divergences D24 and D25).

```python
import autobayes as ab
priors = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1))
model = ab.gaussian("y | weights(w) ~ x + (1 | g)", data, priors)
counts = ab.binomial("k | trials(n) + weights(w) ~ x", data, ab.Priors(intercept=ab.Normal(0, 2.5), b=ab.Normal(0, 1.5)))
censored = ab.gaussian("y | cens(cens_ind) ~ x + (1 | g)", data, priors)        # cens_ind: -1 left, 0 none, 1 right
# F08: brms's labels "left", "none", "right" ("interval" with cens(c, y2)) and TRUE / FALSE are read as those codes too
truncated = ab.gaussian("ypos | trunc(lb = 0) ~ x + (1 | g)", data, priors)
```

## The term model

An addition term is data of the formula, not a population-level term: no coefficient, no prior, no design column.
`ModelSpec.additions` holds one `AdditionTerm(kind, column, lb, ub)` per term other than `trials()` (which stays
`ModelSpec.trials`), in the order written; `spec.addition("weights")` finds one; `term` is its canonical spelling
(`weights(w)`, `cens(c)`, `trunc(lb = 0, ub = 5)`), and the canonical formula writes `trials()` first and the others
after it. Bambi's formula never sees them (`y ~ ...`, `p(k, n) ~ ...`): they act on the likelihood.
`formula.addition_data(spec, frame)` turns them into brms's Stan data on a frame, under brms's names, `weights`,
`cens` (-1, 0, 1), `lb` and `ub` (per row, `-inf` / `inf` on an open side, as brms writes both whenever `trunc()` is
present), validating each; `Design.weights`, `Design.cens`, `Design.lb` and `Design.ub` hold them on the fitted rows,
and gate G1 compares them with brms's `standata`. `FamilySpec.addition_terms` says which terms a family's likelihood
takes and `FamilySpec.tails` gives the log-CDF, log-CCDF, log mass between bounds, truncated mean and truncated draws
that `cens()` and `trunc()` need (the Gaussian's: `likelihoods.NORMAL_TAILS`).

## `weights(col)`

brms adds `weights[n] * family_lpdf(Y[n] | ...)` for each row (with `trunc()`, the row's truncated log density is
weighted as a whole). The likelihood here is a `pm.CustomDist` around the family's own distribution
(`likelihoods.addition_family`): its log density is the family's elementwise log density times the row's weight, and
its draws are the family's own. So `model.logp()`, the JAX density and pointwise log likelihoods all carry the
weights, for each of those families: the hand-written Bernoulli, binomial and Beta likelihoods, the probit link, PyMC's
`Normal`, `StudentT`, `Poisson`, `NegativeBinomial`, `Gamma` and `LogNormal`; correlated slopes, offsets and derived
grouping factors compose with it unchanged.

- The weights are a numeric column of positive finite reals, one per row (brms accepts a zero weight; D26); a
  missing, non-numeric, complex, non-finite, zero or negative weight is refused naming the row. Any numeric column can
  be the weights, a predictor included.
- The priors are not weighted. The weights do not enter the centring means of the design (brms centres unweighted too).
- `predict()` ignores the weights, as brms's `posterior_epred` and `posterior_predict` do, and new rows need no weights
  column. `prior_predictive()` draws unweighted responses.
- `describe()` says "with response addition term(s) weights(w)", lists the weights' range and sum, and writes the
  weighted log likelihood; the fit record carries `"addition_terms": ["weights(w)"]`; the target identity includes the
  weights, so the same formula with other weights is another target.
- Refused: `weights(col, scale = TRUE)` (brms rescales the weights to sum to the number of rows; rescale the column),
  any expression inside `weights()`, a second `weights()` term.

## `cens(col)` and `trunc(lb = a, ub = b)` (Gaussian; accepted against two references, P6)

**What brms does.** `cens` per row: 0 observed (`normal_lpdf(Y | mu, sigma)`), 1 right-censored
(`normal_lccdf(Y | mu, sigma)`, `Y` a lower bound), -1 left-censored (`normal_lcdf(Y | mu, sigma)`, `Y` an upper
bound). `trunc`: every row's log density minus the log mass between the bounds, `normal_lccdf(lb | ...)`,
`normal_lcdf(ub | ...)` or `log_diff_exp` of the two; brms refuses responses outside the bounds. `posterior_epred` of a
truncated model is the truncated mean `mu + sigma (phi(a) - phi(b)) / (Phi(b) - Phi(a))`, `posterior_predict` draws from
the truncated distribution, and censoring changes no prediction (the frozen fixtures confirm: `epred` equals `linpred`
for `gauss_censored` and `gauss_weighted`, and is the truncated mean for `gauss_truncated`).

**What is implemented.** The same model for the Gaussian family, exact on both the PyTensor and the JAX path:
`likelihoods.normal_log_cdf`, `normal_log_ccdf` (both through the stable `log_normal_cdf` of A12) and
`normal_log_mass` (from the upper tail, the lower tail, or as a sum of two `erf`s across zero, every branch evaluated on
arguments valid for it), finite with a finite gradient for every finite argument, and `standard_truncated_mean`
(exact in both tails) and `scipy.stats.truncnorm` for predictions. The models build, describe themselves
(`censoring :` and `truncation:` lines), draw and predict; `tests/test_addition_terms.py` checks the values and the
gradients of all three tail functions against a 200-digit reference on `|z| <= 15`.

**Where brms is inexact.** Both frozen fixtures agree with AutoBayes on everything that is exact on the brms side, and
differ where brms's own arithmetic is not exact (the 50-digit gradient below is the exact gradient of the log density
at the fixture's own double-precision parameter values):

| fixture, point | quantity | brms | exact | AutoBayes | tolerance |
| --- | --- | --- | --- | --- | --- |
| `gauss_censored`, 4 | gradient, `sd_1` | 3.4398114909533386 (2.5e-6 off) | 3.4398200705418716 | 9e-15 off | 1e-7 |
| `gauss_censored`, 0 / 5 / 6 / 7 | gradient, worst coordinate | 2.3e-7 / 2.1e-6 / 2.1e-7 / 3.4e-7 off | | 4e-16 or better | 1e-7 |
| `gauss_censored`, 1 / 2 / 3 | gradient, worst coordinate | 1.2e-8 / 7.7e-8 / 9.3e-8 off | | | 1e-7 (pass) |
| `gauss_censored`, all | log density | 2.3e-16 off | | 2e-16 off | 1e-8 |
| `gauss_truncated`, 4 | gradient, `b[1]` (slope of `x`) | -1.5193463027823753 (4.2e-7 off) | -1.5193469340177965 | 4e-14 off | 1e-7 |
| `gauss_truncated`, 4 | `posterior_epred`, fitted row 28 without group effects | 0.04558976966721673 (1.05e-3 off) | 0.04663882589133012 | 8e-15 off | 1e-9 |
| `gauss_truncated`, all | log density | 6.8e-11 off (point 4) | | 4e-16 off | 1e-8 |

(The exact values here centre `x` in double precision, as brms does; the committed reference of P6 centres it exactly,
which moves the truncated one to -1.5193469340177927, 2.5e-15 relative, and leaves the censored one unchanged.)

The causes are in Stan Math (StanHeaders 2.32) and brms 2.23. For a left-censored row, `normal_lcdf` returns an exact
value but differentiates through piecewise approximations in `z = (y - mu) / sigma` (Abramowitz and Stegun 7.1.26 below
`z = 0.14`, with cubic residual corrections below `z = -2.97`, and above `z = 4.1`; degree-six Taylor polynomials in
between); measured against scipy (which agrees with the 50-digit reference) over `|z| <= 12`, its derivative in `z` is
off by up to 4e-7 absolute for `-1 <= z < 0`, up to 3e-6 for `0 <= z < 2.97` and up to 4e-4 below `z = -1`, and within
3e-8 above `z = 2.97`, where `Phi(z) > 0.998`. `normal_lccdf` forms `1 - erf(z / sqrt 2)` by subtraction: exact to 6e-12 below
`z = 4.1`, it loses all its digits by `z = 8.25` (and returns `-inf` beyond). `posterior_epred_trunc_gaussian` forms
`pnorm(ub) - pnorm(lb)` by subtraction, so the truncated mean loses digits once the location lies about four SD or more
beyond the bound, on the excluded side (1e-11 relative at 4.4 SD, 6e-9 at 5.7, 1e-3 at 7.3). At point 4 of
`gauss_truncated` a row's location sits 6.17 SD below the bound `lb = 0` (7.34 without its group effect). So against
brms alone `gauss_censored` fails the gradient gate at five of its eight points and `gauss_truncated` at one, and also
G4, while every other gate passes on both (`test_fixtures_fail_against_brms_alone_only_where_brms_is_inexact`). The
fixtures are not edited; B02 parked both terms rather than loosen a gate, as A12 had parked the cloglog link (D18).

**How they are accepted (P6).** Andrew chose two-reference acceptance (docs/brms-parity/LOOP.md, "Two references"):
both cases opt in through their gate field, and the entries of the listed kinds are compared with brms where brms is
accurate and with a committed 50-digit reference (`tests/oracle/references/<case>.json`, written by
`oracle/reference.py`, an independent implementation) where it is not; log densities never fall back, and every other
gate compares with brms alone and passes.

| case | falls back | entries on the reference | largest \|AutoBayes - reference\| | largest \|brms - reference\| |
| --- | --- | --- | --- | --- |
| `gauss_censored` | gradient | 32 of 160: `z_1[1]` at point 0; `b`, `Intercept`, `sigma`, `sd_1`, `z_1[1]`, `z_1[2]`, `z_1[4]` at point 4; `b`, `Intercept`, `sigma`, `sd_1`, `z_1[1]`, `z_1[2]` at point 5; `z_1[4]` at point 6; `Intercept` at point 7 (each with and without the Jacobian) | 8.7e-15 relative (1.1e-13 absolute) | 2.5e-6 relative (`sd_1` at point 4; 4.1e-4 absolute, `sigma` at point 4, whose gradient is 698) |
| `gauss_truncated` | gradient | 2 of 160: `b[1]` at point 4, with and without the Jacobian | 3.7e-14 relative (2.3e-13 absolute) | 4.2e-7 relative (`b[1]` at point 4) |
| `gauss_truncated` | expected values | 6 of 656: point 4, fitted row 28, and without group effects fitted rows 4, 13, 28 and 32; point 3, new row 3 without group effects | 8.3e-15 | 1.05e-3 (point 4, fitted row 28 without group effects) |

Relative is the gates' measure, `|a - b| / max(1, |a|, |b|)`; brms's log densities are within 2.0e-16 (censored) and
6.7e-11 (truncated) of the reference's, AutoBayes's within 4.2e-16.

**Refused, naming the construct:** `cens()` together with `trunc()` (brms writes a censored
row of a truncated model as the untruncated tail minus the truncation mass, which is not the censored probability of
the truncated distribution); censoring codes other than -1, 0, 1 (and 2 under `cens(col, y2)`; brms
also reads `"left"`, `"none"`, `"right"` and `TRUE` / `FALSE`); per-row truncation bounds read from a column (numbers
only; `Inf` / `-Inf` mark an open side); `trunc()` without a finite bound; `lb >= ub`; responses outside the bounds (as
brms); `cens()` and `trunc()` for the families E04 does not offer them for (docs/cens-trunc-families.md, D122). E04 lifted the
refusal of interval censoring, `cens(col, y2)` with code 2, for every family that takes `cens()`.

The addition term `mi()` (C06, `x | mi() ~ ...`: the response's missing values are parameters) is parsed here too and taken
in a multivariate list of formulas only, alone on the left-hand side (docs/missing-and-measurement-error.md, divergence D72).

F06a adds `se(col)` and `se(col, sigma = TRUE)` (known standard errors, Gaussian and Student-t) and `rate(col)` (a rate's
denominator, Poisson and negative binomial): `AdditionTerm("se", column, sigma=...)` and `AdditionTerm("rate", column)`, brms's
Stan data `se` and `denom` (`Design.se`, `Design.denom`), each alone on the left-hand side (docs/se-rate.md, divergences D156,
D157).

## Evidence

Oracle cases (gates G1-G6 against frozen brms fixtures): `gauss_weighted`, and one weighted case per family frozen for
this feature, `student_weighted`, `poisson_weighted`, `negbinomial_weighted`, `gamma_weighted`, `lognormal_weighted`,
`beta_weighted`, `bernoulli_weighted` and `binomial_weighted` (`k | trials(size) + weights(w)`), each `... ~ x + (1 | g)`
on the `tier_b` data with its positive weights `w`: G1 compares `standata$weights` with `Design.weights`, G2 the
weighted log density and gradient, G4 `posterior_epred` (which ignores the weights). `gauss_censored` and
`gauss_truncated` pass every gate against brms and, where brms is inexact, their committed references (above).
`tests/test_addition_terms.py` covers the grammar and its refusals, the term model, validation, the weighted density
equal to the unweighted one plus `sum((w - 1) log p(y_i))` for all nine families and with correlated slopes, an offset
and the probit link, the JAX density, predictions, identity, record and `describe()`, gate G1/G2 failing when the
weights are dropped or misread, and `cens()` and `trunc()` as above. Nothing samples.
