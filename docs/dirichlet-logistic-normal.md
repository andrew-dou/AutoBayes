# Responses on the simplex: dirichlet and logistic_normal

brms-parity feature F01b (brms's `dirichlet()` and `logistic_normal()`; `autobayes/families/simplex.py`).

```python
import autobayes as ab
pri = ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd=ab.HalfNormal(1), phi=ab.Gamma(0.01, 0.01))
dir_model = ab.dirichlet("cbind(s1, s2, s3) ~ x + (1 | g)", data, pri)
pri = ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1), lncor=ab.LKJ(2))
ln_model = ab.logistic_normal("cbind(s1, s2, s3) ~ x + (1 | g)", data, pri, refcat="s2")
```

## The models

The response is written `cbind(y1, ..., yK)`, brms's matrix response: K numeric columns whose rows are K proportions strictly
inside (0, 1) summing to 1 (within 1e-8, Stan's simplex tolerance); anything else is refused naming the column and row. The
categories are the columns. One is the reference: the first column, or `refcat=` (brms's `refcat`; the logistic-normal family
only, D138). `model.categories` lists the reference first and the others in the columns' order; `model.design.y` is brms's `Y`,
`(n, K)` in the columns' order.

As for the categorical family (B05), every non-reference category `k` has its own linear predictor `mu_k` (brms's `mu<k>`,
`mus2` for the column `s2`): its own centred intercept, coefficients and varying intercepts with their own SD (brms gives each
`mu<k>` its own group term, `sd_m`, `z_m`, `r_m_mus2_1`).

| | dirichlet | logistic_normal |
| --- | --- | --- |
| density | `y_i ~ Dirichlet(phi * softmax(0, mu_i))` (brms's `dirichlet_logit_lpdf`) | `log(y_i[k] / y_i[ref]) ~ MultiNormal(mu_i, diag(sigma) Lncor diag(sigma))`, and on the simplex the Jacobian `-sum_k log y_i[k]` over all K parts (brms's `logistic_normal_cholesky_cor_lpdf`) |
| own parameters | the precision `phi` (`real<lower=0>`, sampled as `log(phi)`) | one SD per non-reference category `sigma<k>` (`sigmas2`, ...; on the log scale) and the Cholesky factor `Llncor` of their correlation (Stan's `cholesky_corr` coordinates) |
| priors | `Priors(phi=...)`; brms's default `gamma(0.01, 0.01)` | `Priors(sigma=...)`, one prior for every `sigma<k>` (default `student_t(3, 0, 2.5)`), and `Priors(lncor=ab.LKJ(eta))` (brms class `lncor`, default `lkj(1)`) |
| `predict(kind="mean")` | `softmax(0, mu_i)`, brms's `posterior_epred` | refused, as brms refuses `posterior_epred` (the mean of a logistic-normal vector has no closed form) |

`Priors(intercept=, b=, sd=)` hold for every non-reference category (one prior per class, as for the categorical family,
D37). `ab.default_priors(formula, data, family="dirichlet")` (and `"logistic_normal"`, with `refcat=`) returns brms's
defaults: `student_t(3, 0, 2.5)` on every category's intercept and group SDs (brms does not read such a response for them),
flat coefficients, and the family's own above; `model.prior_table()` writes brms's rows, a block per `mu<k>`.

**Numerics.** The Dirichlet's log mean proportions are `mu_k - log_sum_exp(0, mu)` (the maximum subtracted inside) and a
concentration below 1e-10 takes its `lgamma` from its logarithm (`likelihoods._lgamma_of_shape`), so the density and its
gradient are finite for every finite predictor on both paths. The logistic-normal density whitens the log-ratios by forward
substitution through `diag(sigma) Llncor`. On the PyTensor path the gradient of `lgamma` is PyTensor's digamma, 1e-10
relative off (2.5e-10 on `dirichlet_varint`'s `phi`); JAX's, which NumPyro differentiates, is exact to rounding (4.5e-15).

## Using it

- `value_point(intercept=[...], b=..., sd={"g": [...]}, z={"g": ...}, phi=...)`: one intercept and one column of `b` per
  non-reference category, one SD per category and `z` of shape `(levels, K - 1)`; the logistic-normal family takes
  `sigma=[...]` (K - 1, in category order) and `Llncor=` (the `(K - 1) x (K - 1)` Cholesky factor) instead of `phi`.
- `model.names`: `category_dim` (the non-reference categories, `<response>_reduced_dim`), `response_dim` (the K categories of
  the observed response); `model.simplex_names` the logistic-normal family's `sigma<k>` and `Llncor`. The response's
  synthetic name is `cbind_s1_s2_s3`, the column Bambi is handed (the reference's values; the likelihood reads the K columns).
- `predict()`: `kind="linear"` returns `mu` with a `category` dimension over the non-reference categories (brms's
  `posterior_linpred`), on fitted and new rows (new rows need no response), with or without group effects; `kind="response"`
  draws on the simplex with a `category` dimension over all K (Dirichlet draws from Gamma variates taken on the log scale;
  logistic-normal draws `softmax(0, mu + diag(sigma) Lncor e)`). `log_lik()` (and so `loo()`, `waic()`) is each row's density.
- What combines: population-level terms of every kind the term model has (numeric, declared categorical, interactions,
  transforms, `0 +`) and any number of varying-intercept terms `(1 | g)`, `(1 | g:h)`.
- Refused, naming the construct: another reference category than the first column for the dirichlet family, two-column
  responses for the logistic-normal family, varying slopes, correlation IDs, `mm()`, `gr(by = )`, offsets, addition terms
  after `cbind()`, distributional formulas, `gp()`, `mo()`, `cs()`, `me()`, `mi()`, autocorrelation terms, multivariate and
  mixture models, a prior per `sigma<k>`, new levels in `predict()` under any policy but `"error"`, `bayes_R2()`,
  `prior_predictive()`, `hypothesis()`, `conditional_effects()` and `prior_draws()` (D138, D139).

## Evidence

Oracle cases `dirichlet_varint`, `logistic_normal_varint` and `logistic_normal_refcat` (`cbind(s1, s2, s3) ~ x + (1 | g)`,
the last with `refcat = "s2"`) pass gates G1-G6: brms's `Y`, `ncat`, each category's `X_mu<k>` and group coding (`J_m`,
`Z_m_mu<k>_1`, one group per category); the log density with and without the Jacobian within 1.1e-15 relative and its prior
part within 2.8e-16, every gradient coordinate within 2.5e-10 (3.8e-15 for the logistic-normal cases), `Llncor`'s
`cholesky_corr` coordinate recomputed and checked at every point; the graph's `mu` and each category's group effects against
brms's `posterior_linpred` and `r_m_mu<k>_1`, the raw-scale intercepts against `b_mu<k>_Intercept`; `predict()` against
`posterior_linpred` and (dirichlet) `posterior_epred` on fitted and new rows with and without group effects, brms's
`posterior_epred` refusal for the logistic-normal cases; the JAX density; and `ab.default_priors()` against brms's
`default_prior()` table. `tests/test_dirichlet_logistic_normal.py` checks the formula and its refusals, the response's checks,
`refcat`, the priors, the graph's density against SciPy's Dirichlet and multivariate normal densities with the Jacobian written
out (with and without the Jacobian, the JAX path, two factors and a categorical predictor), the extreme-predictor branch,
`value_point`, by-hand predictions, the draws' stream and moments, `log_lik()`, the refusals, `describe()`, the target identity,
the default prior tables and the fit record. Nothing samples.
