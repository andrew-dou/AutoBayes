# Gamma responses with a log link

brms-parity feature A09 (`brmsfamily("Gamma", link = "log")`).

```python
import autobayes as ab
model = ab.gamma("duration ~ x + (1 | group)", data,
    ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd=ab.HalfNormal(1), shape=ab.Gamma(0.01, 0.01)))
print(model.describe())
```

## The model

Strictly positive responses, log link on the mean, `mu_i = exp(eta_i)`, and brms's parameterisation
`gamma_lpdf(y | shape, shape / mu)`: `E[y_i] = mu_i`, `Var[y_i] = mu_i^2 / shape` (a constant coefficient of variation
`1 / sqrt(shape)`). `shape` is brms's `real<lower=0> shape` with its own positive-support prior (brms class `shape`,
`Priors(shape=...)`, as for the negative binomial). Bambi's `gamma` family takes the prior for `alpha = shape` and
hands PyMC `Gamma(mu, sigma = mu / sqrt(alpha))`, which PyMC turns back into `alpha` and `beta = alpha / mu`; the
oracle shows that round trip is inside the tolerance. The sampled variable is `alpha_log__`; `model.names["shape"]`
is `"alpha"`. Coefficients are on the log-mean scale; the centred intercept is the log expected response at the
predictor means.

**Link (divergence D9).** brms's default link for `Gamma` is `inverse` (`mu = 1 / eta`); this library offers the
`log` link only, which keeps `mu > 0` for every `eta` and puts the coefficients on the same scale as the Poisson
and negative binomial families. The fixture `gamma_varint` was frozen with `link = "log"`, so the oracle compares
like with like. The default stays `log`; since E06 brms's `inverse` and `identity` links are offered on request,
`ab.gamma(..., link="inverse")` and `link="identity"`, each needing a positive linear predictor ([links](links.md)).

## Using it

- Response: finite reals strictly greater than zero; a zero, a negative value, an infinity, a boolean or a
  non-numeric column is refused with the value named. Missing values are refused by the shared check.
- Priors: `shape` is required; `sigma` is refused; any positive-support class (`Gamma`, `HalfNormal`,
  `Exponential`, `HalfCauchy`, `HalfStudentT`) is accepted.
- `Model.value_point(..., shape=0.7)`; `predict()`: `kind="linear"` is `eta`, `kind="mean"` is `mu = exp(eta)`
  (overflow refused), `kind="response"` adds continuous draws `Gamma(shape, rate = shape / mu)`; both unseen-group
  policies work. `prior_predictive()` works; `fit_record` records the shape prior and the `alpha` posterior row.

## Evidence

Oracle case `gamma_varint` (`ypos ~ x + (1 | g)`, `gamma(0.01, 0.01)` on `shape`) passes gates G1–G6 against the
frozen brms fixture: design, log density and gradient with and without the Jacobian, the shape prior inside
`lprior`, group effects, `posterior_linpred` / `posterior_epred` on fitted and new rows with and without group
effects, and the JAX density. `tests/test_gamma.py` checks the density and gradient by hand on both paths at
responses 1e−6, 1 and 1e6 and log means −30, 0 and 30 for shapes 0.05 to 1e4 (the gradient in `log(shape)`
against its closed form), the refusals, `value_point`, seeded predictions under both new-group policies,
`describe()`, prior-predictive shapes and the fit record. Nothing samples.

No inverse or identity link, offset, censoring or distributional (`shape ~ ...`) formula is modeled.
