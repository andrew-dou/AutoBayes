# Grouped thresholds: `thres(gr = z)` and `thres(K, gr = z)`

brms-parity feature F06b. brms's addition term `thres()` on the left-hand side of a cumulative model gives each level of a
column `z` its own ordered thresholds:

```python
import autobayes as ab
pri = ab.Priors(intercept=ab.Normal(0, 3), b=ab.Normal(0, 1.5), sd=ab.HalfNormal(1))
m = ab.cumulative("ord | thres(gr = z) ~ x + (1 | g)", data, pri)     # each level: its largest category less one threshold
m3 = ab.cumulative("ord | thres(3, gr = z) ~ x", data, pri)           # three thresholds for every level
```

## The model

For a row of level `l` of `z`, with `mu` the linear predictor (the population-level terms on the raw design, the offset and
the group effects, no intercept) and `tau_l[1] < ... < tau_l[n_l]` the level's thresholds,

    P(y <= k) = expit(tau_l[k] - mu),  k = 1..n_l,      P(y <= n_l + 1) = 1,

so the row's categories are `1..n_l + 1` and every category above them has probability exactly 0 (brms's `posterior_epred`,
the frozen cases' notes). The number of thresholds `n_l` (brms's `nthres`, one entry per level) is the level's largest
category less one under `thres(gr = z)` (a level that never reaches the response's top category has fewer thresholds;
`cumulative_thres_gr`'s level `r` has two of the three) and `K` for every level under `thres(K, gr = z)` (a level's
thresholds above its largest category are then identified by the ordering and the prior alone, `cumulative_thres_k_gr`'s
level `r`). The log density of a row is brms's `ordered_logistic_lpmf` with its level's thresholds, written in the stable form
of the ungrouped family ([ordinal and categorical](ordinal-categorical.md)) with the row's own number of thresholds bounding
its categories (`grouped_thresholds.grouped_ordered_logistic_logp`).

brms declares one `ordered[nthres[k]] Intercept_k` per level `k` (its `merged_Intercept` is their concatenation, and each
row's `Jthres` points to its level's block), puts the class-`Intercept` prior on every threshold, and does not centre the
design when the thresholds are grouped: `Intercept_k` is its `b_Intercept_k`, and the linear predictor is `X b + Z u`. So does
this library: the model's design is never centred with `thres()` (whatever `center_predictors` says; `model.design.centered`
is False), and its thresholds are one vector `Intercept` (brms's `merged_Intercept`: the levels in sorted order, each level's
block in threshold order; dims `Intercept_dim`, coordinates `"<level>[<k>]"`, `"p[1]"`, ..., `"r[2]"`) sampled on Stan's ordered
coordinates within each level's block (`grouped_thresholds.BlockOrdered`, the value variable `Intercept_ordered__`: `(tau_l[1],
log(tau_l[2] - tau_l[1]), ...)` per level, the log Jacobian the sum of the log gaps), under the intercept prior on every entry.
`b_Intercept` is the same vector, the deterministic the likelihood reads. `model.design.thresholds`
(`grouped_thresholds.ThresholdGroups`) holds the levels, their numbers of thresholds and each fitted row's level, and gives
brms's `Jthres` (`jthres()`).

## Using it

- `thres(gr = z)`, `thres(K, gr = z)`, and the spellings R's argument matching gives brms's `thres(x, gr)`: `thres(3, z)`,
  `thres(gr = z, 3)`, `thres(x = 3, gr = z)`. `z` is a text column or a pandas Categorical (its labels; the levels are sorted
  as text, as the grouping factors' are), read by `thres()` alone; `K` is one positive whole number.
- Priors: `Priors(intercept=...)` is the prior of every threshold of every level (brms's class `Intercept`).
  `ab.default_priors(...)` gives brms's `student_t(3, 0, 2.5)`, and `model.prior_table()` writes brms's rows: the class row, then
  for each level a row with `group` the level and one row per threshold (`coef` `1`, ..., sorted as brms sorts them, `1`, `10`,
  `11`, `2`, ...).
- `Model.value_point(intercept={"p": [...], "q": [...], "r": [...]})`: each level's thresholds, strictly increasing, by level.
- `predict()`: `kind="linear"` is `mu` (brms's `posterior_linpred`), `kind="mean"` the probabilities of all the response's
  categories with exact zeros above each row's level's (brms's `posterior_epred`), `kind="response"` 1-based category draws.
  New rows need `z` (for every kind) and a level the fitted rows have (divergence D159). `log_lik()`, `loo()` and `bayes_R2()`
  (the expected category, as for the ungrouped family) read each row's level.
- What combines: the population-level terms with design columns (numeric and categorical predictors, interactions,
  transforms), the offset, group-level terms with independent or correlated slopes, and both unseen-group policies of
  `predict()`; checked by hand in `tests/test_grouped_thresholds.py` (the oracle cases have `x` and `(1 | g)`).

## Refused, with the construct named

As brms refuses them: a numeric `gr` ("Variable 'gr' in 'thres' needs to be factor-like."), a level whose rows all take the
first category ("Could not extract the number of thresholds."), `K` below the largest category less one ("Number of
thresholds is smaller than required by the response."), a number per level ("Number of thresholds should be unique for each
group."). Not verified here (divergence D158): `thres()` without `gr`, other ordinal families (`sratio`, `cratio`, `acat`,
`hurdle_cumulative`) and the cumulative probit and cloglog links, a boolean `gr`, `K + 1` above the response's categories and
an ordered response whose top categories no level reaches, `z` read by another term, and `mo()`, `gp()` and the terms the
cumulative family refuses anyway (`cs()`, `me()`, `mi()`, `mm()`, `gr(by = )`, autocorrelation, other addition terms,
distributional formulas). `hypothesis()`, `conditional_effects()` and `prior_draws()` refuse the model as they refuse the
cumulative family.

## Evidence

Oracle cases `cumulative_thres_gr` (`ord | thres(gr = z) ~ x + (1 | g)`, nthres 3, 3, 2) and `cumulative_thres_k_gr`
(`ord | thres(3, gr = z) ~ x`) pass gates G1-G6 against the frozen brms fixtures: brms's `ngrthres`, `nthres` and `Jthres`
(each brms level matched to the model's by the rows its block holds), `X` uncentred; the log density with and without the
Jacobian and every gradient coordinate, each `Intercept_k`'s ordered coordinates recomputed at every point and checked
against the fixture's before they are mapped to its level's block of `Intercept_ordered__`; the graph's `b_Intercept` against
brms's `b_Intercept_k` and `merged_Intercept`; `predict()` against `posterior_linpred` / `posterior_epred` (8 x 36 x 4, the
zeros included) on fitted and new rows with and without group effects; the JAX density; `ab.default_priors()` against brms's
`default_prior()` table. `tests/test_grouped_thresholds.py` checks the row against the 50-digit reference of
`tests/test_ordinal_categorical.py` on both paths (linear predictors from -1000 to 1000, a gap of 1e-9, a padding threshold
no row reads), the block-ordered transform against Stan's ordered transform, the model's density by hand and its gradient
against a central difference and the JAX gradient, `value_point`, the predictions, `log_lik()` and `bayes_R2()` by hand, sorted
prior-predictive thresholds within each level, brms's printed prior table of a level with eleven thresholds, every refusal,
`describe()` and the target identity. The syntax corpus entry `add_thres_gr` (`yord | thres(3, gr = f2) ~ x`) matches brms's
`make_standata()`. Nothing samples.
