# Beta-binomial, geometric and negbinomial2 responses

brms-parity feature D02 (`beta_binomial()`, `geometric()`, `negbinomial2()`).

```python
import autobayes as ab
base = dict(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd=ab.HalfNormal(1))
ab.beta_binomial("k | trials(size) ~ x + (1 | g)", data, ab.Priors(**base, phi=ab.Gamma(0.01, 0.01)))
ab.geometric("ynb ~ x + (1 | g)", data, ab.Priors(**base))
ab.negbinomial2("ynb ~ x + (1 | g)", data, ab.Priors(**base, sigma=ab.Gamma(0.4, 0.3)))
```

## The models

Each is brms's parameterisation (brms's vignette "Parameterization of Response Distributions in brms"), checked against
brms's `log_prob` on the frozen cases, not assumed. `eta` is the linear predictor.

| family | link | mass | auxiliary parameter (brms class, PyMC coordinate) | `E[y]` (brms `posterior_epred`) |
| --- | --- | --- | --- | --- |
| `beta_binomial` | logit on `mu` | `BetaBinomial(n, mu phi, (1 - mu) phi)` with `n` the trials: `log C(n, k) + log B(k + mu phi, n - k + (1 - mu) phi) - log B(mu phi, (1 - mu) phi)` | `phi`, the precision (`phi_log__`) | `n mu`, `mu = expit(eta)` |
| `geometric` | log on `mu` | the negative binomial of mean `mu` and shape 1: `P(y) = mu^y / (1 + mu)^(y + 1)` | none | `mu = exp(eta)` |
| `negbinomial2` | log on `mu` | the negative binomial of mean `mu` and shape `1 / sigma` | `sigma`, the dispersion (`sigma_log__`) | `mu = exp(eta)` |

`Var[y]` is `n mu (1 - mu) (n + phi) / (1 + phi)` for the beta-binomial (the binomial as `phi` grows), `mu + mu^2` for the
geometric and `mu + sigma mu^2` for `negbinomial2`. `negbinomial2` is `negbinomial` with `shape = 1 / sigma` (brms declares
`real<lower=0> sigma`); its default prior `gamma(0.4, 0.3)` is the distribution of `1 / shape` when `shape` has
`negbinomial`'s default `inv_gamma(0.4, 0.3)`. The geometric family counts failures from 0 (PyMC's own `Geometric` counts
trials from 1 and is not used).

**Graph.** `likelihoods.beta_binomial_logit_family` is PyMC's `BetaBinomial` with `alpha = expit(eta) phi` and
`beta = expit(-eta) phi` (`1 - mu` as `expit(-eta)`, as the Beta family computes it, D13), the linear predictor kept as the
parent `logit_mu` under the identity link and the trials handed to the likelihood as the binomial's are (a two-column
`p(k, size)` response). `likelihoods.geometric_family` and `negbinomial2_family` are PyMC's `NegativeBinomial(mu, alpha)` with
`alpha` 1 and `1 / sigma`, the parent `mu` under the log link, as the negbinomial family's. Constants are float64.

**Trials.** The beta-binomial's successes and trials are validated, cast and dropped where the trials are zero exactly as the
binomial family's (`binomial_counts`; zero-trial rows dropped and reported, D4; counts up to 1e6); every fitted row with zero
or with all successes gets the binomial's separation warning.

**Predictions.** `predict(kind="linear")` returns `eta` (brms's `posterior_linpred`). `kind="mean"` returns the mean `mu`,
and for the beta-binomial also `expected = n mu`, brms's `posterior_epred` (as for the binomial, D5; new rows then need the
trials column); `kind="probability"` returns the beta-binomial's `mu` alone and needs no trials. `kind="response"` draws
`Binomial(n, p)` with `p ~ Beta(mu phi, (1 - mu) phi)` for the beta-binomial and negative binomial draws for the other two.
`ab.log_lik`, `ab.loo` and `ab.bayes_R2` (C07) take each family's per-row mass (`families.pointwise`, equal to the model's
own row masses at the frozen points); `bayes_R2` reads `expected` for the beta-binomial.

## What combines with them

The term model is shared, so population-level terms of every kind, declared categorical predictors and interactions,
transforms, offsets, independent and correlated varying slopes, `g:h` and `g/h`, and Gaussian processes `gp(x)` reach these
likelihoods through the linear predictor (`tests/test_discrete_families.py` checks the observed rows' mass at the graph's
linear predictor with all of them in one formula). Refused, naming the construct, because no frozen case verifies them (D80,
D81): a link other than brms's default, a distributional formula on `phi` or `sigma`, `weights()`, `cens()`, `trunc()`,
`mi()` (as a term or on the response), `mo()`, `me()`, autocorrelation terms, `mm()` and `gr(by = )`, a multivariate formula,
a mixture component (D52) and `nl=True` (D62). E03 lifted the beta-binomial's `phi ~ ...` and `weights()`
([contexts of the newer families](contexts-newer-families.md)).

## Verification

The frozen brms fixtures `beta_binomial_varint` (`k | trials(size) ~ x + (1 | g)`, `phi ~ gamma(0.01, 0.01)`),
`geometric_varint` and `negbinomial2_varint` (`ynb ~ x + (1 | g)`, `sigma ~ gamma(0.4, 0.3)`) pass G1-G6. The largest
relative errors against brms at the eight frozen points:

| case | log density (with / without Jacobian, prior part) | gradient | JAX vs PyTensor | predictions |
| --- | --- | --- | --- | --- |
| `beta_binomial_varint` | 3.1e-16 / 4.7e-16 / 3.7e-16 | 1.3e-9 | 1.9e-15 | 4.4e-16 |
| `geometric_varint` | 3.6e-16 / 5.4e-16 / 1.4e-16 | 4.1e-15 | 3.6e-16 | 9.7e-16 |
| `negbinomial2_varint` | 8.8e-16 / 8.7e-16 / 2.2e-16 | 9.5e-11 | 9.4e-16 | 4.4e-16 |

The beta-binomial's gradient (every coordinate, through the log-gamma terms of `mu phi` and `(1 - mu) phi`) and
`negbinomial2`'s `sigma` coordinate carry PyTensor's digamma, off by about 1e-10 absolute; brms's gradient tolerance is 1e-7.
`ab.default_priors()` reproduces brms's `default_prior()` table on every fixture (G6) and on two test frames of
`tests/test_discrete_families.py` (brms's rows probed in R): the beta-binomial's defaults are constants (logit link), the
geometric's and `negbinomial2`'s Intercept and `sd` read `log(y)` with zeros as 0.1, as the negative binomial's do.

PyMC's beta-binomial log mass is a difference of log-gamma values, which loses about `eps * phi * log(phi)` as the precision
grows: against a 50-digit evaluation it is within 1.4e-13 at `phi = 100`, 1.3e-11 at 1e4, 1.5e-9 at 1e6 and 1.4e-7 at 1e8, where
Stan's `beta_binomial_lpmf` (brms's) is within 8e-15, 4e-13, 1e-11 and 9e-9 (probed with `rstan::expose_stan_functions`; D82). At
counts up to 1e6 the mass is within 2e-9 absolute (3e-10 relative) of the 50-digit value.
