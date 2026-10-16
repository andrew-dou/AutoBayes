# Weibull, exponential, skew-normal, shifted lognormal and ex-Gaussian responses

brms-parity feature D01 (`weibull()`, `exponential()`, `skew_normal()`, `shifted_lognormal()`, `exgaussian()`).

```python
import autobayes as ab
base = dict(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd=ab.HalfNormal(1))
ab.weibull("ypos ~ x + (1 | g)", data, ab.Priors(**base, shape=ab.Gamma(0.01, 0.01)))
ab.exponential("ypos ~ x + (1 | g)", data, ab.Priors(**base))
ab.skew_normal("y ~ x + (1 | g)", data, ab.Priors(**base, sigma=ab.HalfNormal(1), alpha=ab.Normal(0, 4)))
ab.shifted_lognormal("ypos ~ x + (1 | g)", data, ab.Priors(**base, sigma=ab.HalfNormal(1), ndt=ab.Normal(0, 0.3)))
ab.exgaussian("y ~ x + (1 | g)", data, ab.Priors(**base, sigma=ab.HalfNormal(1), beta=ab.Gamma(1, 0.1)))
```

## The models

Each is brms's parameterisation (brms's vignette "Parameterization of Response Distributions in brms"; Azzalini's skew
normal), checked against brms's `log_prob` on the frozen cases, not assumed. `mu` is the linear predictor under the link.

| family | link | density | parameters (brms class, PyMC coordinate) | `E[y]` (brms `posterior_epred`) |
| --- | --- | --- | --- | --- |
| `weibull` | log | Weibull of shape `shape` and scale `mu / Gamma(1 + 1 / shape)` | `shape` (`alpha_log__`) | `mu = exp(eta)` |
| `exponential` | log | exponential of rate `1 / mu` | none | `mu = exp(eta)` |
| `skew_normal` | identity | Azzalini's `SN(xi, omega, alpha)` with mean `mu` and SD `sigma`: `delta = alpha / sqrt(1 + alpha^2)`, `omega = sigma / sqrt(1 - 2 delta^2 / pi)`, `xi = mu - omega delta sqrt(2 / pi)` | `sigma` (`sigma_log__`), `alpha` (`alpha`, unbounded) | `mu` |
| `shifted_lognormal` | identity | `log(y - ndt) ~ Normal(mu, sigma)` | `sigma` (`sigma_log__`), `ndt` in (0, min y) (`ndt_interval__`) | `exp(mu + sigma^2 / 2) + ndt` |
| `exgaussian` | identity | `Normal(mu - beta, sigma)` plus an exponential of mean `beta` | `sigma` (`sigma_log__`), `beta` (`beta_log__`) | `mu` |

The skew normal's `sigma` is the SD of `y` whatever `alpha` (`alpha = 0` is the Normal); the ex-Gaussian's `sigma` is the
SD of its Normal component (`Var[y] = sigma^2 + beta^2`). Weibull, exponential and shifted lognormal responses must be
strictly positive (brms: "requires response greater than 0"); skew-normal and ex-Gaussian responses are any reals.

**Graph.** The Weibull and exponential families are Bambi's own (`beta = mu / gamma(1 + 1 / alpha)` handed to PyMC's
`Weibull`, `lam = 1 / mu` to its `Exponential`), with the shape named `alpha` there as for the Gamma family. The other three
are this library's Bambi families (`likelihoods.skew_normal_family`, `exgaussian_family`, `shifted_lognormal_family`), each a
`pm.CustomDist` of the parent `mu` and the auxiliary parameters: the skew normal's `log 2 - log omega + log phi(z) + log
Phi(alpha z)` and the ex-Gaussian's `-log beta + (mu - beta - y) / beta + sigma^2 / (2 beta^2) + log Phi((y - mu + beta) /
sigma - sigma / beta)` evaluate `log Phi` with `likelihoods.log_normal_cdf`, exact with a finite gradient for every finite
argument on both paths, where brms's `log(erfc(.))` is `-inf` (beyond about -37.5; D77); PyMC's own `SkewNormal` (whose
`log(1 + erf(.))` cancels in the lower tail) and `ExGaussian` (which switches to a Normal approximation when `nu < 0.05
sigma`) are not used. Every constant is a float64.

**The extra parameters.** `FamilySpec.extra_parameters` holds each family's parameter beside its scale parameter, an
`ExtraParameter` with its brms name (the `Priors` field, the `value_point` keyword, the `model.names` key and the PyMC
variable), its meaning, its support and brms's default:

| parameter | brms | prior | sampled as | brms's default |
| --- | --- | --- | --- | --- |
| `alpha` (skew normal) | `real alpha` | a location prior, `Normal` or `StudentT` | itself | `normal(0, 4)` |
| `beta` (ex-Gaussian) | `real<lower=0> beta` | a positive-support prior | `beta_log__` | `gamma(1, 0.1)` |
| `ndt` (shifted lognormal) | `real<lower=0,upper=min_Y> ndt`, `min_Y = min(Y)` | `Normal`, `StudentT`, `HalfNormal`, `HalfCauchy`, `Exponential`, `Gamma` or `InverseGamma`, truncated to (0, min y) and renormalised there; or `Uniform(0, "min_Y")` | `ndt_interval__` = `log(ndt / (min_Y - ndt))`, Stan's lower-upper coordinate | `uniform(0, min_Y)`, `ab.Uniform(0, "min_Y")` |

`ndt`'s prior is brms's `lprior` term: the prior's density minus the log of its mass between the bounds
(`Prior.to_bambi_bounded`; the mass is computed with SciPy from whichever tail keeps it exact, `priors.bounded_log_mass`,
because PyMC's own `Truncated` reaches it through PyMC's normal log-CDF, whose float32 constant `sqrt(2.0)` costs 1e-8, measured
on the frozen case). `min_Y` is the smallest response of the rows the model is built on. Each extra parameter is required
for its family and refused for every other, as is any prior its support does not take; `Uniform` is accepted only as brms's
default (D78).

**Predictions.** `predict(kind="linear")` returns `mu` (brms's `posterior_linpred`), `kind="mean"` brms's `posterior_epred`
(the table above; `mean` for the shifted lognormal, `mu` otherwise) and `kind="response"` draws of the family's distribution
(the skew normal's from `delta |u0| + sqrt(1 - delta^2) u1`, the ex-Gaussian's as a Normal plus an exponential, the shifted
lognormal's above `ndt`). `ab.log_lik` and `ab.loo` (C07) take each family's per-row density (`families.pointwise`, equal to
the model's own row densities at the frozen points), and `ab.bayes_R2` its expected response (equal to brms's definition on
the fixtures' `posterior_epred`).

## What combines with them

The term model is shared, so population-level terms of every kind, declared categorical predictors and interactions,
transforms, offsets, independent and correlated varying slopes, `g:h` and `g/h`, and Gaussian processes `gp(x)` (which C03
offers for every family but the categorical one) reach these likelihoods through the linear predictor
(`tests/test_continuous_families.py` checks the observed rows' density at the graph's linear predictor with all of them in one
formula). Refused, naming the construct, because no frozen case verifies them (D79): a distributional formula on any auxiliary
parameter (`shape`, `sigma`, `alpha`, `beta`, `ndt`), `weights()`, `cens()`, `trunc()`, `mi()`, `mo()`, `me()`, autocorrelation
terms, a multivariate formula, a mixture component (D52), and a link other than brms's default (D76).

## Verification

The frozen brms fixtures `weibull_varint`, `exponential_varint`, `skew_normal_varint`, `shifted_lognormal_varint` and
`exgaussian_varint` (each `y ~ x + (1 | g)` with explicit priors, the family's own parameter included) pass G1-G6. The largest
relative errors against brms at the eight frozen points:

| case | log density (with / without Jacobian, prior part) | gradient | JAX vs PyTensor | predictions |
| --- | --- | --- | --- | --- |
| `weibull_varint` | 2.2e-16 / 2.2e-16 / 2.2e-16 | 9.2e-10 | 1.9e-15 | 4.1e-16 |
| `exponential_varint` | 4.4e-16 / 4.4e-16 / 1.6e-16 | 7.8e-16 | 3.0e-16 | 5.3e-16 |
| `skew_normal_varint` | 3.5e-16 / 3.5e-16 / 3.9e-16 | 4.4e-15 | 4.5e-16 | 2.2e-16 |
| `shifted_lognormal_varint` | 2.2e-16 / 2.3e-16 / 4.6e-16 | 1.1e-15 | 2.2e-16 | 1.0e-15 |
| `exgaussian_varint` | 1.4e-15 / 1.3e-15 / 5.9e-16 | 3.5e-13 | 3.9e-16 | 4.2e-16 |

The Weibull's gradient with respect to `log(shape)` carries PyTensor's digamma (the derivative of `gamma(1 + 1 / shape)`), which
is off by about 1e-10 absolute; brms's gradient tolerance is 1e-7. `ab.default_priors()` reproduces brms's `default_prior()`
table on every fixture (G6) and on two test frames of `tests/test_continuous_families.py` (brms's rows probed in R). Probes of
`default_prior()` on rescaled responses showed that `alpha`, `beta` and `ndt`'s defaults do not depend on the data, and that the
Intercept's location and the scale of the Intercept, `sd` and `sigma` priors are read from `log(y)` for the Weibull,
exponential and shifted lognormal and from `y` for the skew normal and ex-Gaussian (the second test frame's scales are 4.1 and
18.2). The skew-normal and ex-Gaussian tails were probed with Stan's own functions (D77).
