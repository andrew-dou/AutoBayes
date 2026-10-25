# The Cox proportional hazards model

brms-parity feature F05 (brms's `cox()` family with `time | cens(c) ~ ...`).

```python
import autobayes as ab
priors = ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd=ab.HalfNormal(1), sbhaz=ab.Dirichlet(1))
ab.cox("time | cens(c) ~ x + (1 | g)", data, priors)                          # 0 an event, 1 right-censored
ab.cox("time | cens(c) + bhaz(df = 7) ~ x + (1 | g)", data, priors)           # seven bases
ab.cox("time | bhaz(knots = c(1.5, 3), Boundary.knots = c(0, 6)) ~ x", data, priors)
ab.default_priors("time | cens(c) ~ x + (1 | g)", data, family="cox")         # brms's defaults
```

## The model

Row `i` has the hazard `h0(t) exp(eta_i)` at time `t`: `eta` is the linear predictor (brms's log link; brms's `mu` is
`exp(eta)`, the relative hazard) and the baseline hazard is `h0(t) = sum_k sbhaz[k] M_k(t)`, whose integral from the lower
boundary knot is `H0(t) = sum_k sbhaz[k] I_k(t)`. `M_k` are the cubic M-splines of the time with their intercept (every basis
function kept; brms's Stan data `Zbhaz`) and `I_k` their integrals, the I-splines (`Zcbhaz`), each rising from 0 to 1 (Ramsay
1988). `sbhaz` is brms's `simplex[Kbhaz] sbhaz` under a Dirichlet prior, so `H0` runs from 0 to 1 over the boundary knots and the
intercept sets the hazard's scale. A row with an event contributes `log h0(t_i) + eta_i - H0(t_i) exp(eta_i)`, a right-censored
row the log survival `-H0(t_i) exp(eta_i)`; the time enters only through the bases.

| parameter | brms | prior | sampled as | brms's default |
| --- | --- | --- | --- | --- |
| `sbhaz`, the weights of the bases | `simplex[Kbhaz] sbhaz` | `Dirichlet`, one concentration for all or one per basis (`con_sbhaz`) | `sbhaz_stickbreaking__` (Stan's stick-breaking coordinates) | `dirichlet(1)` |

The bases are built in `autobayes/baseline_hazard.py` from their definitions, not from splines2's code: the B-splines of order 4
on the knots with each boundary knot repeated four times follow from Cox and de Boor's recursion, `M_i = 4 B_i / (t_{i+4} - t_i)`
(each integrating to 1), and since the derivative of an order-5 B-spline on the same knots (each boundary knot once more) is the
difference of two consecutive M-splines, `I_i = sum_{j > i} B'_{j,5}`. They agree with the fixtures' `Zbhaz` and `Zcbhaz` to
7e-16 and with SciPy's B-splines and their antiderivatives (`tests/test_cox.py`).

brms's knots (`bhaz()`): the boundary knots are the range of the times of every row, censored rows too, widened by 2% of its
length at each end; `df - 4` interior knots at the times' quantiles (R's type 7) at the `df - 2` equally spaced probabilities in
[0, 1] less the two ends (`df = 5` by default: the median). `bhaz(df = K)` (K >= 4; brms refuses 3), `bhaz(knots = c(...))` (the
interior knots, then `len + 4` bases) and `bhaz(Boundary.knots = c(lo, hi))` (alone it keeps the quantile knots) set them;
`Design.bhaz` holds the knots, the bases of the fitted rows and which rows are events. `df` together with `knots` is refused
(brms ignores `df` then), as are tied quantiles, interior knots outside the boundary and times outside given boundary knots
(D152).

## Predictions and post-processing

brms refuses `posterior_epred` and `posterior_predict` for cox ("Cannot compute expected values of the posterior predictive
distribution for family 'cox'."). `predict(kind="linear")` returns `eta` (brms's `posterior_linpred`), on fitted rows or new rows
(no time column needed, where brms asks for one), and `predict(kind="relative_hazard")` returns `mu = exp(eta)` (brms's
`posterior_linpred(transform = TRUE)`); `kind="mean"`, `kind="response"`, `prior_predictive()`, `bayes_R2()` and
`conditional_effects()` are refused (D154). `ab.log_lik` is brms's `log_lik()` (each row's event or censored term, checked on
`cox_varint_post`) and `ab.loo` follows from it.

## Priors

`Priors(sbhaz=...)` is required for the cox family and refused for every other. `ab.default_priors(..., family="cox")` gives
brms's: `dirichlet(1)` on `sbhaz`, and the Intercept's `student_t(3, location, scale)` read through the log link, the median of
`log(time)` rounded to one decimal: it moves with the unit of time (0.7 in these units, 4.1 in units thirty times smaller, probed
with brms 2.23) though the likelihood does not, since the I-spline bases are unit-free and the M-spline bases scale by a constant
that the intercept absorbs. G6 checks the table as brms writes it (D152).

## What is refused

Interval censoring `cens(c, y2)` and `trunc()`, which brms 2.23 computes wrongly (D153: an interval row's log likelihood is -inf;
a truncated row is normalised by its survival at its own time, cancelling its survival term); left-censored rows (`cens = -1`),
`weights()`, a stratified `bhaz(gr = )`, `bhaz(degree = )`, `bhaz(intercept = )` and splines2's other arguments, other links,
`mo()`, `me()`, `mi()`, `cs()`, autocorrelation and residual correlation terms, `sar()`, `car()`, `fcor()`, `mm()` and
`gr(by = )`, distributional formulas, multivariate and non-linear models, mixtures and shrinkage priors (D155), each naming the
construct. Varying slopes (correlated or not), `g:h`, categorical predictors and interactions, transforms, offsets, `gp()` and
`center_predictors=False` reach the likelihood through the linear predictor and are tested there.

Verified against the frozen brms cases `cox_varint`, `cox_uncensored_default_priors` (brms's default priors), `cox_bhaz_df`
(`bhaz(df = 7)` with `dirichlet(2)` on `sbhaz`), `cox_bhaz_knots` (explicit knots) and `cox_varint_post` (`log_lik`, `loo`).
