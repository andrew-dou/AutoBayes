# Negative binomial count responses

brms-parity feature A08 (`brmsfamily("negbinomial")`).

```python
import autobayes as ab
model = ab.negbinomial("count ~ x + (1 | group)", data,
    ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd=ab.HalfNormal(1), shape=ab.Gamma(0.01, 0.01)))
print(model.describe())
```

## The model

Log link on the mean, `mu_i = exp(eta_i)`, and brms's `neg_binomial_2` parameterisation: `E[y_i] = mu_i`,
`Var[y_i] = mu_i + mu_i^2 / shape`, Poisson as `shape` grows. `shape` is brms's `real<lower=0> shape` with its own
positive-support prior (brms class `shape`; `ab.Gamma(alpha, beta)` is shape–rate, as Stan's `gamma` and PyMC's
`Gamma`). Bambi's `negativebinomial` family is PyMC's `NegativeBinomial(mu, alpha)` with `alpha = shape`, so the
sampled variable is `alpha_log__` and `model.names["shape"]` is `"alpha"`. Coefficients are on the log-mean scale,
as for Poisson; the centred intercept is the log expected count at the predictor means.

## Using it

- Counts as for Poisson: finite non-negative integers from 0 through 1e6; zero-count rows are retained, and
  all-zero data trigger a prior-sensitivity warning naming the shape prior.
- Priors: `shape` is required exactly for the negative binomial and Gamma families; `sigma` is refused here, and
  `shape` is refused for every other family, with the message naming the family's own auxiliary parameter.
- `Model.value_point(..., shape=0.7)`; `sigma=` is refused with the same rule.
- `predict()`: `kind="linear"` is `eta`, `kind="mean"` is `mu = exp(eta)` (overflow refused, not clipped),
  `kind="response"` adds integer draws `NegativeBinomial(mu, shape)` (numpy's `negative_binomial(shape, shape /
  (shape + mu))`); means beyond the integer generator's range are refused. Both unseen-group policies work.
- `prior_predictive()` works; `fit_record` records the shape prior and the `alpha` posterior row.

## Evidence

Oracle case `negbinomial_varint` (`ynb ~ x + (1 | g)`, `gamma(0.01, 0.01)` on `shape`) passes gates G1–G6 against
the frozen brms fixture: design, log density and gradient with and without the Jacobian (brms's `shape` has the
`log` transform, as PyMC's `alpha_log__`), the shape prior inside `lprior`, group effects, `posterior_linpred` /
`posterior_epred` on fitted and new rows with and without group effects, and the JAX density.
`tests/test_negbinomial.py` checks the density and gradient by hand on both paths at counts 0, 1 and 1e6 and
log means from −30 to 30 for shapes 0.05 to 1e4, the refusals, `value_point`, seeded predictions under both
new-group policies, `describe()`, prior-predictive shapes and the fit record. Nothing samples.

No exposure/offset, zero inflation or distributional (`shape ~ ...`) formula is modeled.
