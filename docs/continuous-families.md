# Continuous families: Weibull, exponential, skew-normal, shifted lognormal, ex-Gaussian (D01) and inverse Gaussian, von Mises, asymmetric Laplace, Frechet, generalized extreme value (E02)

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
terms, a multivariate formula, a mixture component (D52), and a link other than brms's default (D76). E03 lifted three of them:
the Weibull's `shape ~ ...` and `weights()` and the skew normal's `alpha ~ ...` ([contexts of the newer families](contexts-newer-families.md)).

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

# Inverse Gaussian, von Mises, asymmetric Laplace, Frechet and generalized extreme value responses

brms-parity feature E02 (`inverse.gaussian()`, `von_mises()`, `asym_laplace()`, `frechet()`, `gen_extreme_value()`).

```python
import autobayes as ab
base = dict(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd=ab.HalfNormal(1))
ab.inverse_gaussian("ypos ~ x + offset(eoff) + (1 | g)", data, ab.Priors(**base, shape=ab.Gamma(0.01, 0.01)))   # 1/mu^2
ab.inverse_gaussian("ypos ~ x + (1 | g)", data, ab.Priors(**base, shape=ab.Gamma(0.01, 0.01)), link="log")
ab.von_mises("yang ~ x + (1 | g)", data, ab.Priors(**base, kappa=ab.Gamma(2, 0.01)))
ab.asym_laplace("y ~ x + (1 | g)", data, ab.Priors(**base, sigma=ab.HalfNormal(1), quantile=ab.Beta(2, 2)))
ab.asym_laplace("y ~ x + (1 | g)", data, ab.Priors(**base, sigma=ab.HalfNormal(1)), quantile=0.25)   # bf(quantile = 0.25)
ab.frechet("ypos ~ x + (1 | g)", data, ab.Priors(**base, nu=ab.Gamma(2, 0.1)))                         # nu learned; nu=3 fixes it
ab.gen_extreme_value("y ~ x + (1 | g)", data, ab.Priors(**base, sigma=ab.HalfNormal(1), xi=ab.Normal(0, 0.5)))
```

## The models

Each is brms's parameterisation (brms's vignette "Parameterization of Response Distributions in brms"), checked against brms's
`log_prob` on the frozen cases, not assumed. `eta` is the linear predictor; every likelihood takes it as Bambi's parent under
Bambi's identity link and applies brms's link itself, so the graph's parent deterministic is brms's `posterior_linpred`.

| family | link | density | parameters (brms class, PyMC coordinate) | `E[y]` (brms `posterior_epred`) |
| --- | --- | --- | --- | --- |
| `inverse_gaussian` | `1/mu^2` (default): `mu = 1 / sqrt(eta)`; or `log` | `0.5 log(shape / (2 pi y^3)) - shape (y - mu)^2 / (2 mu^2 y)` | `shape` (`alpha_log__`, Bambi's name as for the Gamma) | `mu` |
| `von_mises` | `tan_half`: `mu = 2 atan(eta)`, in (-pi, pi) | `kappa cos(y - mu) - log(2 pi) - log I0(kappa)` | `kappa` (`kappa_log__`) | `mu` |
| `asym_laplace` | identity | `log p + log(1 - p) - log sigma - rho_p((y - mu) / sigma)`, `rho_p(u) = u (p - I(u < 0))` | `sigma` (`sigma_log__`), `quantile` p (`quantile_logodds__`), or fixed | `mu + sigma (1 - 2 p) / (p (1 - p))` |
| `frechet` | log: `mu = exp(eta)` | `log nu - log s - (1 + nu) log(y / s) - (y / s)^-nu`, `s = mu / Gamma(1 - 1 / nu)` | `nu > 1` (`nu_interval__`, learned as the Student-t nu, A13), or fixed | `mu` |
| `gen_extreme_value` | identity | `-log sigma - (1 + xi) w - exp(-w)`, `w = log1p(xi z) / xi`, `z = (y - mu) / sigma` | `sigma` (`sigma_log__`), `xi` derived from `tmp_xi` (`tmp_xi`) | `mu + sigma (Gamma(1 - xi) - 1) / xi` |

`mu` of the asymmetric Laplace is the `p`-quantile of `y` (`P(y < mu) = p`): with the quantile fixed, `ab.asym_laplace(...,
quantile=0.25)` is quantile regression. The responses: strictly positive finite reals (inverse Gaussian, Frechet), angles in
`[-pi, pi]` (von Mises, brms's own check), finite reals (asymmetric Laplace, generalized extreme value).

**The inverse Gaussian's 1/mu^2 link.** `mu = 1 / sqrt(eta)` exists for `eta > 0` only. The density is `-inf` with a zero
gradient where `eta <= 0` (brms's `inv_sqrt` of a negative number is NaN and its density an error, D115), and `predict()` refuses
draws with `eta <= 0` rather than return NaN; the frozen case carries `offset(eoff)` so that `eta` lies in (5.8, 12.6) at its
points. Under the log link `mu = exp(eta)`. `link=` is the link machinery of docs/links.md: `FamilySpec.links` of the registered
`1/mu^2` spec reaches the `log` one and back, `get_family("inverse_gaussian", "log")`; brms's `inverse`, `identity` and
`softplus` links are refused (D112).

**log I0.** `likelihoods.log_bessel_i0e(kappa)`, `log I0(kappa) - kappa`, is `log(i0(kappa)) - kappa` below 500 (PyTensor's `i0`,
lowered to `jax.scipy.special.i0`; `i0` overflows past 709.8) and the asymptotic series `-log(2 pi kappa) / 2 + log1p(sum_j a_j /
kappa^j)` (Abramowitz and Stegun 9.7.1, eight terms) at and above, each branch switched into its own range so that the branch not
taken has a zero gradient on both paths; the density is written `-2 kappa sin^2((y - mu) / 2) - log(2 pi) - log_bessel_i0e(kappa)`,
which forms no difference of numbers of kappa's size. Stan's `von_mises_lpdf`, brms's density, raises an overflow error for
kappa above 713.9 (D113).

**The generalized extreme value shape xi.** brms does not sample `xi`: it declares the unbounded `tmp_xi`, puts the prior of
class `xi` on it, and derives `xi = lb + (ub - lb) inv_logit(tmp_xi)` in its model block, with `lb` and `ub` the smaller and the
larger of `-1 / max(x)` and `-1 / min(x)` for `x = (y - mu) / sigma` over the fitted rows, so that `1 + xi x > 0` on every fitted
row. The map was found from the frozen case's points, where it reproduces the fixture's record of brms's `posterior_epred(dpar =
"xi")` (`dpar_epred`) to the last bit, and brms's predictions on new rows and without group effects use that same fitted-row xi
(their `epred` agree with it to 7.8e-14). Here `ExtraParameter(name="xi", parameter="tmp_xi")`: `Priors(xi=...)` is the location
prior of the free variable `tmp_xi` (`value_point(tmp_xi=...)`), and the likelihood registers the derived value as the graph's
deterministic `xi` (`model.names["xi"]`), which a fit's posterior holds and `predict()`, `log_lik()` and the response draws read;
the oracle adapter takes it from `dpar_epred` (`adapter.derived_draws`) and gate G3 compares the graph's `xi` with that record
(`gates.check_derived_parameters`). The density's `w = z g(xi z)`, `g(u) = log1p(u) / u` by its series below `|u| = 1e-3`, is exact
through `xi = 0` (the Gumbel); the mean's `(Gamma(1 - xi) - 1) / xi` is summed from the zeta series of `lgamma(1 - xi)` below `|xi|
= 0.01` (Euler's constant at 0) and is `inf` for `xi >= 1`, where the mean is infinite (brms's formula gives a finite number there;
D114). brms 2.23 warns that the family is deprecated.

**Fixed parameters.** brms's `bf(y ~ x, quantile = 0.25)` writes the quantile into its Stan data and has no quantile parameter
or prior. `ab.asym_laplace(..., quantile=0.25)` (`Model(..., quantile=0.25)`) builds the model with the family's spec of that
parameter fixed, `FamilySpec.with_fixed("quantile", 0.25)` (the registered spec's `fixable`): no extra parameter, `quantile` a
fixed parameter checked strictly inside (0, 1) (brms accepts 0 and 1, D115), held as `model.quantile` and handed to Bambi as a
float64 constant; `families.family_of(model)` returns the model's own spec, so `predict()`, the target identity
(`family_parameters`) and the fit record see it; `Priors(quantile=...)` is then refused, and `ab.default_priors(..., quantile=0.25)`
returns no quantile prior, as brms's table has no row. Gate G1 checks brms's Stan-data `quantile` against the model's
(`gates.check_fixed_parameters`). The Frechet `nu` is learned or fixed (`nu=3`) through the Student-t family's path (A13): its
prior truncated at 1 and renormalised, brms's `real<lower=1> nu`.

**Predictions** are assembled by hand: `kind="linear"` is `eta`, `kind="mean"` the table's `E[y]` (`mu`, or `mean` for the
asymmetric Laplace and generalized extreme value families), `kind="response"` draws (NumPy's `wald` and `vonmises`, the asymmetric
Laplace's two exponential halves, `s E^(-1/nu)` for the Frechet and `mu + sigma (E^-xi - 1) / xi` for the GEV, with `E` a standard
exponential). `ab.log_lik` and `ab.loo` take each family's per-row density (`families.pointwise`), `ab.bayes_R2` its expected
response.

## Default priors

| class | brms 2.23 default | here |
| --- | --- | --- |
| `Intercept` | `student_t(3, location, scale)` with the response read through the link: `1 / y^2` under the inverse Gaussian's `1/mu^2` (the offset's mean subtracted from the location), `log y` under its log link and for the Frechet, `y` for the asymmetric Laplace and GEV; `student_t(1, 0, 1)` for the von Mises under `tan_half`, whatever the data | the same (`default_priors._on_link` reads `"1/mu^2"`; `FamilySpec.default_intercept_prior` for the von Mises) |
| `sd`, `sigma` | `student_t(3, 0, scale)`, `lb` 0, with that scale (2.5 for the von Mises) | `HalfStudentT(3, scale)` |
| `shape` (inverse Gaussian) | `gamma(0.01, 0.01)`, `lb` 0 | `Gamma(0.01, 0.01)` |
| `kappa` | `gamma(2, 0.01)`, `lb` 0 | `Gamma(2, 0.01)` |
| `quantile` | `beta(1, 1)`, `lb` 0, `ub` 1; no row when fixed | `Beta(1, 1)` |
| `nu` (Frechet) | `gamma(2, 0.1)`, `lb` 1; no row when fixed | `Gamma(2, 0.1)`, truncated at 1 |
| `xi` | `normal(0, 2.5)`, unbounded (on `tmp_xi`) | `Normal(0, 2.5)` |

Probed with brms's `default_prior()` on rescaled responses and on the two frames of `tests/test_continuous_families_e02.py`; gate
G6 checks the seven fixtures' tables row by row.

## What combines with them

The term model is shared, so population-level terms of every kind, declared categorical predictors and interactions, transforms,
offsets, independent and correlated varying slopes, `g:h` and `g/h`, and Gaussian processes reach these likelihoods through the
linear predictor (`tests/test_continuous_families_e02.py` checks the observed rows' density at the graph's linear predictor with
all of them in one formula; the GEV's xi is derived from that full predictor). Refused, naming the construct, because no frozen
case verifies them (D112): other links, a distributional formula on any parameter, `weights()`, `cens()`, `trunc()`, `mi()`,
`mo()`, `me()`, `cs()`, autocorrelation terms, `mm()` and `gr(by = )`, multivariate and non-linear formulas, mixture components and
shrinkage priors on `b`.

## Verification

The frozen brms fixtures (each with `(1 | g)` and explicit priors) pass G1-G6. The largest relative errors against brms at the
eight frozen points:

| case | log density (with / without Jacobian, prior part) | gradient | JAX vs PyTensor | predictions |
| --- | --- | --- | --- | --- |
| `inverse_gaussian_varint` | 4.1e-16 / 4.1e-16 / 2.4e-16 | 6.9e-16 | 2.0e-16 | 2.2e-16 |
| `inverse_gaussian_log` | 4.2e-16 / 6.3e-16 / 2.2e-16 | 1.5e-15 | 5.2e-16 | 9.9e-16 |
| `von_mises_varint` | 3.5e-16 / 1.8e-16 / 0 | 4.7e-15 | 2.8e-16 | 4.4e-16 |
| `asym_laplace_varint` | 2.2e-16 / 2.2e-16 / 5.3e-16 | 2.7e-15 | 4.5e-16 | 4.4e-16 |
| `asym_laplace_quantile` | 3.1e-16 / 3.1e-16 / 5.4e-16 | 4.0e-15 | 2.4e-16 | 4.4e-16 |
| `frechet_varint` | 7.8e-16 / 7.8e-16 / 1.3e-16 | 1.1e-10 | 1.8e-15 | 1.1e-15 |
| `gen_extreme_value_varint` | 1.9e-15 / 2.0e-15 / 2.7e-16 | 3.7e-12 | 2.6e-15 | 1.1e-14 (xi: 5.6e-17) |

The Frechet gradient with respect to `log(nu - 1)` carries PyTensor's digamma (the derivative of `lgamma(1 - 1 / nu)`), off by
about 1e-10 absolute, as for the Weibull; brms's gradient tolerance is 1e-7. `ab.default_priors()` reproduces brms's
`default_prior()` table on every fixture (G6) and on the two test frames.
