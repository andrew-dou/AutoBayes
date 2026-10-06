# Lognormal responses

brms-parity feature A10 (`brmsfamily("lognormal")`).

```python
import autobayes as ab
model = ab.lognormal("duration ~ x + (1 | group)", data,
    ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1)))
print(model.describe())
```

## The model

Strictly positive responses with `log(y_i) ~ Normal(mu_i, sigma)`, brms's `lognormal_lpdf(y | mu, sigma)`. The link
is the identity on the log-scale location `mu`, which is the linear predictor: coefficients act on `log y`, and the
centred intercept is the expected log response at the predictor means (so the location priors are "on the log
scale", as for the Poisson and Gamma families, although the link is not `log`). `sigma` is the SD of `log y`, brms's
`real<lower=0> sigma` with its own positive-support prior (brms class `sigma`, `Priors(sigma=...)`, as for the
Gaussian family); the sampled variable is `sigma_log__` and `model.names["sigma"]` is `"sigma"`.

The expected response is `E[y_i] = exp(mu_i + sigma^2 / 2)`, not `exp(mu_i)`, which is the median;
`Var[y_i] = (exp(sigma^2) - 1) exp(2 mu_i + sigma^2)`. Bambi has no plain lognormal family (its `hurdle_lognormal` is a
different model), so `autobayes/likelihoods.py` wraps PyMC's `LogNormal(mu, sigma)` as a Bambi family exactly the way
Bambi wraps `Normal` for `gaussian`; the density is the same expression as Stan's.

## Using it

- Response: finite reals strictly greater than zero; a zero, a negative value, an infinity, a boolean or a
  non-numeric column is refused with the value named. Missing values are refused by the shared check.
- Priors: `sigma` is required; `shape` is refused; any positive-support class (`HalfNormal`, `Gamma`, `Exponential`,
  `HalfCauchy`, `HalfStudentT`) is accepted.
- `Model.value_point(..., sigma=0.7)`; `model.names`: `linpred` and `mu` are `"mu"` (the location deterministic),
  `mean` is `"mean"`.
- `predict()`: `kind="linear"` is `mu` (brms `posterior_linpred`); `kind="mean"` is `mean = exp(mu + sigma^2 / 2)`
  per posterior draw (brms `posterior_epred`; overflow is refused, not clipped) under the variable name `mean`, since
  `mu` already names the location; `kind="response"` adds continuous draws `LogNormal(mu, sigma)`. Both unseen-group
  policies work. `prior_predictive()` works; `fit_record` records the sigma prior and the `sigma` posterior row.
- In the family registry the mean is the first that needs an auxiliary parameter: `FamilySpec.expected_response`
  (`(model, posterior, eta) -> E[y]`) takes over from `inverse_link(eta)` when set, and `draw_response` receives the
  linear predictor next to the mean.

## Evidence

Oracle case `lognormal_varint` (`ypos ~ x + (1 | g)`, `normal(0, 1)` truncated at zero on `sigma`) passes gates G1–G6
against the frozen brms fixture: design, log density and gradient with and without the Jacobian (brms's `sigma` has
the `log` transform, as `sigma_log__`), the sigma prior inside `lprior`, group effects, `posterior_linpred` (`mu`) and
`posterior_epred` (`exp(mu + sigma^2 / 2)`, which differs from `exp(mu)` by up to a factor 32 on the fixture's points)
on fitted and new rows with and without group effects, and the JAX density. `tests/test_lognormal.py` checks the
density and gradient by hand on both paths at responses 1e−6, 1 and 1e6 and locations −30, 0 and 30 for `sigma` from
0.05 to 1e4 (the gradients in `log(sigma)` and the intercept against their closed forms), the refusals, `value_point`,
seeded predictions under both new-group policies with the mean distinguished from the median, `describe()`,
prior-predictive shapes and the fit record. Nothing samples.

No `log` or `inverse` link on `mu`, offset, censoring or distributional (`sigma ~ ...`) formula is modeled.
