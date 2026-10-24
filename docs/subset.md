# `subset()` and a formula per response in a multivariate model

brms-parity feature F06c (brms's addition term `subset()` in a multivariate model with `set_rescor(FALSE)`; divergence
rows D160 and D161).

```python
import autobayes as ab
priors = ab.Priors(resp={"y": {"intercept": ab.Normal(0, 5), "b": ab.Normal(0, 2), "sd": ab.HalfNormal(1), "sigma": ab.HalfNormal(1)},
                         "y2": {"intercept": ab.Normal(0, 5), "b": ab.Normal(0, 2), "sigma": ab.HalfNormal(1)}})
model = ab.gaussian(["y | subset(sub1) ~ x + (1 | g)", "y2 | subset(sub2) ~ x"], data, priors, rescor=False)
print(model.describe())
```

This is brms's `bf(y | subset(sub1) ~ x + (1 | g)) + bf(y2 | subset(sub2) ~ x) + set_rescor(FALSE)`: two responses, each
with a formula of its own, each observed on its own rows. The rows of `subset(sub1)` are those whose indicator `sub1` is 1
(or True); a row in neither subset is in no response's likelihood, and a response's values outside its subset are never
read (they may be missing, as brms leaves them).

## The model

With `R_y` the rows of `y`'s subset and `R_y2` those of `y2`'s:

    y_i  ~ Normal(mu_y,i,  sigma_y),   i in R_y,    mu_y,i  = a_y  + (x_i - mean_{R_y}(x)) b_y + u_y[g_i]
    y2_i ~ Normal(mu_y2,i, sigma_y2),  i in R_y2,   mu_y2,i = a_y2 + (x_i - mean_{R_y2}(x)) b_y2

independently given the parameters. Each response's model is the univariate model of its own formula on the rows of its
subset, exactly as brms's Stan data have it (`N_y`, `Y_y`, `X_y`, `J_1_y`, `Z_1_y_1` on `y`'s rows, in data order, with `N`
the number of rows of the data): its design is centred on its rows' means, its default priors are read from its rows (the
Intercept's location is the median of `y` on `R_y`), and its group-level term is its own (brms's `sd_1`, `z_1` of `y`).

## How it is built

`parse_multivariate` reads a list of formulas without `mi()` (`multivariate._parse_separate`): each `<response> ~ <terms>` or
`<response> | subset(<indicator>) ~ <terms>` is parsed as the univariate formula it is, without its `subset()` term, and
`MultivariateSpec.subsets` maps each response written with `subset()` to its indicator (`spec.response_formula(r)` and
`spec.canonical_formula` write it back, `bf(y | subset(sub1) ~ x + (1 | g)) + bf(y2 | subset(sub2) ~ x)`).
`MultivariateModel` builds each response's `Model` on its rows (`subset_rows`: 0-based positions in the data, in data
order) and assembles their graphs as C01 does (docs/multivariate.md), each response's observation coordinate renamed after it
(`model.observation_dims`: `__obs_y__`, `__obs_y2__`, whose coordinates are the response's rows of the data), the grouping
coordinates shared. `model.design.rows` holds the rows, `model.design.Y` each response on its rows and NaN elsewhere, and
`model.design.n` the rows of the data (brms's `N`).

## Using it

- Formulas: a list of two or more, one per response, each response once; `subset(col)` alone on a response's left-hand side,
  `col` a column of the data that is logical or numeric 0/1 without missing values, with at least one row in the subset. A
  response without `subset()` is on every row. The right-hand sides take what the univariate Gaussian model takes (numeric
  and declared categorical predictors, interactions, transforms, an offset, `(1 | g)`, `(1 + x || g)`, `(1 + x | g)`, `g:h`,
  `g/h`), each its own. `rescor=False` must be given. The same list without any `subset()` is a formula per response on every
  row (brms's `bf(y ~ x) + bf(y2 ~ z) + set_rescor(FALSE)`, or `mvbf(..., rescor = FALSE)`).
- Priors by response, as for any multivariate model (`Priors(resp={...})`, a correlated term's `cor` at the top level);
  `ab.default_priors([...], data, rescor=False)` returns brms's defaults, each response's read from its subset's rows.
- `predict(fit, resp="y")` (and `ab.predict(..., resp=)`) returns that response's predictions alone, `(chain, draw, __obs__)`,
  on its rows: the fitted rows of its subset, or the rows of `newdata` whose own indicator is 1 (`newdata` must hold the
  indicator column), the `__obs__` coordinates the rows' positions; `kind` as always (`"linear"`, brms's
  `posterior_linpred(resp = "y")`; `"mean"`, `posterior_epred`; `"response"`, draws from the response's Normal), with the same
  unseen-level policies. `predict()` without `resp` is refused, as brms refuses the joint predictions of such a model
  ("Argument 'resp' must be a single variable name for models using addition argument 'subset'."). `resp=` works for every
  multivariate model (C01's too, its predictions the joint ones' slice; with `rescor=True` the draws are the response's
  marginal).
- `describe()` writes each response's subset (its indicator, its number of rows, brms's `N_<response>`), then each
  response's own model; `fit()` samples the assembled graph (C01's `sample_posterior`), each response's raw-scale intercept
  computed with its own rows' means.

## Refused, with the construct named

`rescor=True`: with `subset()` brms refuses it too ("Only 'se', 'weights', 'mi' are supported addition arguments when 'rescor'
is estimated."), and a formula per response without `subset()` is not verified with a residual correlation (D56). `subset()`
with `mi()` (brms requires `mi(x, idx = )`, not part of this slice), with another addition term, twice, or with an expression
(`subset(s == 1)`: write the indicator as a column); an indicator with missing values, text or numbers other than 0 and 1
(brms takes any non-zero number as TRUE), or no row in the subset; missing values on a response's subset rows (brms drops the
row from every response); a grouping level of the data that a response's subset does not reach (D161: brms keeps it, an
effect only its prior informs); the univariate `y | subset(s) ~ x` (D160: brms fits the subset's rows; pass them as the data);
`me()`, `mo()`, `gp()`, autocorrelation and SAR terms, `mm()` and `gr(by = )`, distributional formulas, a response read by
another formula, an indicator that is a response, families other than the Gaussian; `log_lik()`, `loo()`, `waic()`,
`bayes_R2()`, `prior_predictive()` and `fit_record()` of a multivariate model, as before (D95).

## Evidence

Oracle case `gauss_mv_subset` (`bf(y | subset(sub1) ~ x + (1 | g)) + bf(y2 | subset(sub2) ~ x) + set_rescor(FALSE)`, 26 and 18
of 36 rows, 4 rows in neither) passes G1-G6 against brms 2.23: each response's `N_<resp>`, `Y_<resp>`, design and group coding
on its rows (`gates._check_design_subset`), the joint log density and every gradient coordinate with and without the Jacobian
(within 3.6e-16 and 1.2e-15 relative), each response's linear predictor, group effects and raw-scale intercept against brms's
`posterior_linpred(resp = )`, the joint predictions refused as brms refuses them and each response's predictions on its rows
of the fitted data and of `newdata` with and without group effects (`gates.check_response_predictions`, exact), the JAX
density, and brms's default-prior table. `tests/test_subset.py` covers the grammar and every refusal, the indicator's rows,
each response's model on its rows, the assembled graph and its density against the responses' own and scipy's Normal written
out, `predict(resp=)` by hand on fitted and new rows, unseen levels, the default priors read from each subset, `describe()`,
the target identity, the fit path's cleaning with `pm.sample` replaced, and a formula per response without `subset()`. The
syntax corpus's `mv_subset` and `mv_mvbf` entries now read as brms's `make_standata()`. brms's reading of what is refused
was probed with `make_standata()` on the frozen data (rows dropped for a missing value on a subset row, `N_1` counting a level
outside the subset, a population-level factor's subset levels).
