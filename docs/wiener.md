# The Wiener diffusion model: response times and decisions

brms-parity feature F04 (brms's `wiener()` family with `rt | dec(dec) ~ ...`).

```python
import autobayes as ab
priors = ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), sd=ab.HalfNormal(1),
                   bs=ab.Gamma(1, 1), ndt=ab.Normal(0, 0.3), bias=ab.Beta(2, 2))
ab.wiener("rt | dec(dec) ~ x + (1 | g)", data, priors)
ab.wiener(["rt | dec(dec) ~ x + (1 | g)", "bs ~ z"], data,
          ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), sd=ab.HalfNormal(1), ndt=ab.Normal(0, 0.3), bias=ab.Beta(2, 2),
                    bs={"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 0.5)}), categorical={"z": "p"})
ab.default_priors("rt | dec(dec) ~ x + (1 | g)", data, family="wiener")   # brms's defaults
```

## The model

Each row is a response time `rt` and a decision `dec` (1: the upper boundary, 0: the lower one). A Wiener process with drift
`mu` (the linear predictor, identity link) and unit diffusion starts at `bs * bias` between the boundaries 0 and `bs`; `rt - ndt`
is the time it first reaches one of them, `dec` which one. brms's parameters:

| parameter | brms | prior | sampled as | brms's default |
| --- | --- | --- | --- | --- |
| `bs`, the boundary separation | `real<lower=0> bs` | a positive-support prior, or the priors of its own formula `bs ~ ...` (log link) | `bs_log__` | `gamma(1, 1)`; its formula's intercept `normal(-0.6, 1.3)` |
| `ndt`, the non-decision time | `real<lower=0,upper=min_Y> ndt` | truncated to (0, min rt) and renormalised there, as the shifted lognormal's (D01) | `ndt_interval__` | `uniform(0, min_Y)`, `ab.Uniform(0, "min_Y")` |
| `bias`, the relative starting point | `real<lower=0,upper=1> bias` | a `Beta` prior | `bias_logodds__` | `beta(1, 1)` |

The density is Stan's `wiener_lpdf(rt | bs, ndt, bias, mu)` for `dec = 1` and `wiener_lpdf(rt | bs, ndt, 1 - bias, -mu)` for
`dec = 0`, brms's: the first-passage time density of Navarro and Fuss (2009), `f(t | v, a, w) = a^-2 exp(-v a w - v^2 t / 2)
f(t / a^2 | 0, 1, w)` at the lower boundary, the upper boundary's with `(-v, 1 - w)`. `autobayes/wiener_diffusion.py` evaluates
the standard density `f(u | 0, 1, w)` by the small-time series for `u <= 1/2` and the large-time series above, with a fixed
number of terms (the omitted ones below 1e-20 of the sum) grouped so that no two terms of opposite sign and similar size are
added, and with `1 - w` passed beside `w`. It is one implementation for the model (PyTensor, and so JAX) and `log_lik()`
(NumPy); against a 60-digit evaluation of both series (`tests/test_wiener.py`, Python's `decimal`) the log density is exact to
4e-15 for `1e-12 <= w <= 1 - 1e-12` and `1e-4 <= u <= 200`, and its gradient to 1e-11. Stan 2.32's `wiener_lpdf` truncates the
series: on the frozen points it is within 1.5e-10 of the exact log density (relative to it) and its gradient within 1.2e-8
(D150).

`dec(col)` is a response addition term (`ModelSpec.additions`, `Design.dec`, brms's Stan data `dec`): numeric 0 / 1 or boolean,
every row; the family requires it (brms: "Addition argument 'dec' is required for family 'wiener'"). Response times must be
strictly positive finite reals. `min_Y` is the smallest response time of the fitted rows.

## Predictions

- `predict(kind="linear")` is the drift `mu`, brms's `posterior_linpred`.
- `predict(kind="mean")` is the mean response time `E[rt] = ndt + E[T]`, named `mean`, with `E[T] = (bs P(upper) - bs bias) /
  mu`, `P(upper) = (1 - exp(-2 mu bs bias)) / (1 - exp(-2 mu bs))` and `bs^2 bias (1 - bias)` at `mu = 0` (Wald's identity; the
  mean over both boundaries, which needs no decisions). `wiener_diffusion.mean_decision_time` evaluates it without cancellation
  at any drift (within 3e-15 of a 60-digit evaluation; numerical integration of the density gives the same mean). brms 2.23's
  `posterior_epred` is `ndt - bias / mu + bs / mu (exp(-2 mu bias) - 1) / (exp(-2 mu bs) - 1)`, the relative bias where the
  absolute starting point `bs * bias` belongs: the mean only at `bs = 1` (D148). The two frozen cases' G4 expected values
  therefore fail against brms and are recorded as an expected failure of G4 (`"gates": ["g4"]`), until a 50-digit reference
  (P6) stands in for them; their linear predictor is brms's (`tests/test_wiener.py` checks it, and that brms's expected values
  are its formula on the same draws).
- `kind="response"` (brms's `posterior_predict`) and `prior_predictive()` are refused: brms needs the R package RWiener, and no
  sampler of the first-passage time is implemented here (D149).
- `ab.log_lik()`, `ab.loo()` and `ab.waic()` take each row's density from the model's own; brms needs RWiener for them (D150).
  `ab.bayes_R2()` is its definition on `predict(kind="mean")`, so brms's would differ by D148.

## What combines with it

The term model is shared: population-level terms of every kind, declared categorical predictors and interactions, transforms,
offsets, independent and correlated varying slopes, `g:h` and `g/h`, and Gaussian processes reach the drift through the linear
predictor (`tests/test_wiener.py` checks the rows' density at the graph's linear predictor with all of them in one formula).
`bs ~ ...` takes population-level terms. Refused, naming the construct (D151): formulas on `ndt` and `bias` (which brms takes),
group-level terms in `bs`'s formula, links other than brms's defaults, `weights()` (which brms takes), `cens()`, `trunc()`,
`mi()`, `mo()`, `me()`, `cs()`, `mm()` and `gr(by = )`, autocorrelation terms, multivariate and non-linear formulas, mixture
components, shrinkage priors on `b`, `conditional_effects()` and `prior_draws()`; decisions coded other than 0 / 1 (brms reads
'upper' / 'lower' labels and codes every nonzero number as the upper boundary).

## Verification

The frozen brms fixtures `wiener_varint` (`rt | dec(dec) ~ x + (1 | g)`) and `wiener_distributional_bs` (with `bs ~ z`) pass G1
(with brms's `dec`), G2, G3, G5 and G6, and G4's linear predictor; G4's expected values fail against brms by D148. The largest
relative errors against brms at the eight frozen points:

| case | log density (with / without Jacobian, prior part) | gradient | JAX vs PyTensor | linear predictor | brms's formula on these draws vs its epred |
| --- | --- | --- | --- | --- | --- |
| `wiener_varint` | 3.8e-11 / 3.9e-11 / 4.1e-16 | 8.7e-9 | 5.7e-16 | 2.6e-16 | 5.2e-12 |
| `wiener_distributional_bs` | 1.4e-10 / 1.5e-10 / 2.2e-16 | 1.2e-8 | 2.5e-16 | 2.3e-16 | 8.3e-14 |

The log density and gradient differences are Stan's (D150): this library's density is within 4e-15 of the 60-digit value at
the same points. `ab.default_priors()` reproduces brms's `default_prior()` table on both fixtures (G6) and on two test frames:
brms reads `rt` as given (the Intercept's location its median, the scale `max(2.5, MAD)`), and `bs`, `ndt`, `bias` and the
Intercept of `bs`'s formula have constant defaults (probed on rescaled responses).
