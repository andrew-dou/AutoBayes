# Zero-inflated and hurdle families

brms-parity features B06 (`zero_inflated_poisson()`, `hurdle_poisson()`) and D03 (`zero_inflated_negbinomial()`,
`hurdle_negbinomial()`, `zero_inflated_binomial()`, `zero_inflated_beta()`, `hurdle_gamma()`, `hurdle_lognormal()`, and
`weights()` for all eight; [below](#more-families-d03)).

```python
import autobayes as ab
pri = ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd=ab.HalfNormal(1), zi=ab.Beta(2, 2))
model = ab.zero_inflated_poisson("y ~ x + (1 | group)", data, pri)
hurdle = ab.hurdle_poisson("y ~ x + (1 | group)", data, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1),
                                                                  sd=ab.HalfNormal(1), hu=ab.Beta(2, 2)))
```

## The models

Both have a log link on `mu`, the mean of the Poisson part (`eta = log(mu)` is the linear predictor), and one
probability for the zero part, brms's `real<lower=0,upper=1>` parameter of the same name:

| | zero-inflated (`zi`) | hurdle (`hu`) |
| --- | --- | --- |
| `P(y = 0)` | `zi + (1 - zi) exp(-mu)`: a structural zero, or a Poisson zero | `hu` |
| `P(y = k)`, `k > 0` | `(1 - zi) Poisson(k \| mu)` | `(1 - hu) Poisson(k \| mu) / (1 - exp(-mu))`, a zero-truncated Poisson |
| `E[y]` (brms `posterior_epred`) | `(1 - zi) mu` | `(1 - hu) mu / (1 - exp(-mu))` |

The coefficients act on `log(mu)`, not on `log E[y]`; the centred intercept is the log mean of the Poisson part at the
predictor means. `zi` and `hu` are brms's (the probability of the zero part), not PyMC's `psi`, which is `1 - zi`.

**Graph.** The likelihoods are this library's own Bambi families (`autobayes/likelihoods.py`), as the Beta's is: the
linear predictor is kept as the parent `log_mu` under the identity link, so the graph holds `eta` and the likelihood
never sees `log(exp(eta))`; `zi` / `hu` keeps brms's name, has its own prior and is sampled as `zi_logodds__`
(Stan's logit transform for `real<lower=0,upper=1>`). `FamilySpec.zero_parameter` names it, beside `scale_parameter`,
which the zero-inflated or hurdle form of a family with its own auxiliary parameter would also keep. The forms are
brms's `zero_inflated_poisson_log_lpmf` and `hurdle_poisson_log_lpmf` written to stay finite: the zero-inflated `y = 0`
row is a two-term `log_sum_exp` built on the stable softplus, with `-exp(eta)` evaluated at `eta` capped at 700 (beyond
it `exp(-mu)` is exactly 0 and the row is `log(zi)` exactly); the hurdle's zero truncation `log(1 - exp(-mu))` is
`log(-expm1(-mu))`, and `eta + log1p(-mu / 2)` below `eta = -20` (the same function as the cloglog link's
`log P(y = 1)`); the `y > 0` terms are evaluated on `y > 0` rows only, so that no branch a switch does not take
produces `0 * inf`. A row's value and gradient are finite for every finite `eta`, except a `y > 0` row beyond
`eta = 709.78`, whose true log density (about `-exp(eta)`) is below the most negative double and is `-inf`, as brms's
is; never NaN, on the PyTensor path or under JAX's autodiff. brms's own forms, evaluated with `rstan::log_prob`: a
zero-inflated `y = 0` row keeps its value `log(zi)` beyond `eta = 709.78` but its gradient is NaN there (`-exp(eta)`
overflows to `-inf`, and its zero weight times `-inf` is NaN); a hurdle `y > 0` row has a `-Inf` gradient once
`exp(eta)` is subnormal (at `eta = -740`) and a `+Inf` log density with a NaN gradient once it underflows to 0 (below
`eta` of about -745) (divergence D28).

## Using it

- Response: counts as for the Poisson family (integers from 0 through 1e6, zero rows retained); an all-zero response
  is accepted with a warning (the zero-inflated model then identifies only `P(y = 0)`, the hurdle model's Poisson part
  has no data).
- Priors: `zi` (zero-inflated) or `hu` (hurdle) is required, a `Beta(alpha, beta)` prior (`ab.Beta`, support (0, 1);
  brms's `beta(alpha, beta)`); the other one, `sigma`, `shape` and `phi` are refused with the message naming the
  family's own; `zi` and `hu` are refused for every other family, and `ab.Beta` for every other class. brms accepts any
  prior there (`normal(0.3, 0.1)`, `uniform(0, 1)`, ...) and adds its density as written, not renormalised on (0, 1)
  (divergence D29). `ab.default_priors(..., family="zero_inflated_poisson")` returns brms's
  defaults: `beta(1, 1)` (bounds 0 and 1) and the Poisson family's log-link Intercept and `sd` arithmetic (zeros read
  as 0.1).
- `Model.value_point(..., zi=0.3)` (or `hu=`), stored as `logit(zi)`; `model.names["zi"]` is `"zi"`,
  `model.names["linpred"]` is `"log_mu"` and `model.names["mean"]` is `"mean"`.
- `predict()`: `kind="linear"` is `eta` (brms `posterior_linpred`), `kind="mean"` is brms's `posterior_epred` above
  under the name `mean` (the hurdle's ratio as `mu / -expm1(-mu)`, whose limit 1 it takes where `mu` underflows to 0),
  `kind="response"` adds draws as brms's `posterior_predict` makes them: zero where a uniform draw falls below
  `zi` / `hu`, otherwise a Poisson draw, or for the hurdle a zero-truncated Poisson draw (the first point of a unit-rate
  Poisson process on `[0, mu]` given that there is one, then the points after it). Both unseen-group policies work.
  `prior_predictive()` works; `describe()` writes the mixture out; `fit_record` records the `zi` / `hu` prior and its
  posterior row; the target identity includes the prior.
- The term model is the one every family shares (categorical predictors, interactions, transforms, the offset, which
  enters `log_mu`, any number of grouping factors, independent and correlated slopes); the tests build the categorical,
  intercept-free, slope and offset forms.
- `weights(w)` (D03, `zcnt | weights(w) ~ ...`): each fitted row's log likelihood times its weight, as brms documents
  `weights()` and as the frozen weighted cases' log densities confirm; `log_lik()` is weighted the same way (brms's is: C07 verified it on `binomial_weighted_post`),
  and so are `loo()` and `waic()`, which read it; the expected value and the response draws ignore the weights, as brms's
  `posterior_epred` and `posterior_predict` do. Weights must be positive (D26).

Not modeled: `zi ~ x` / `hu ~ x` (distributional formulas, B01), and brms's `zero_inflated_asym_laplace` (divergence D30); D03
adds six more families, and E01 `zero_inflated_beta_binomial`, `zero_one_inflated_beta` and `hurdle_cumulative`
([bounded and hurdle families](bounded-families.md)).

## More families (D03)

```python
nb = ab.zero_inflated_negbinomial("y ~ x + (1 | g)", data, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1),
                                  sd=ab.HalfNormal(1), shape=ab.InverseGamma(0.4, 0.3), zi=ab.Beta(2, 2)))
zib = ab.zero_inflated_binomial("k | trials(n) + weights(w) ~ x", data, ab.Priors(intercept=ab.Normal(0, 2.5),
                                b=ab.Normal(0, 1.5), zi=ab.Beta(2, 2)))
```

| family | link, parent | parameters | at zero | above zero | `E[y]` (`posterior_epred`) |
| --- | --- | --- | --- | --- | --- |
| `zero_inflated_negbinomial` | log, `log_mu` | `shape`, `zi` | `zi + (1 - zi) NB(0 \| mu, shape)` | `(1 - zi) NB(y \| mu, shape)` | `(1 - zi) mu` |
| `hurdle_negbinomial` | log, `log_mu` | `shape`, `hu` | `hu` | `(1 - hu) NB(y \| mu, shape) / (1 - NB(0 \| mu, shape))` | `(1 - hu) mu / (1 - NB(0 \| mu, shape))` |
| `zero_inflated_binomial` | logit, `logit_p` | `zi` | `zi + (1 - zi) (1 - p)^n` | `(1 - zi) Binomial(y \| n, p)` | `(1 - zi) n p`, named `expected` |
| `zero_inflated_beta` | logit, `logit_mu` | `phi`, `zi` | `zi` (a point mass) | `(1 - zi) Beta(y \| mu phi, (1 - mu) phi)` | `(1 - zi) mu` |
| `hurdle_gamma` | log, `log_mu` | `shape`, `hu` | `hu` (a point mass) | `(1 - hu) Gamma(y \| shape, shape / mu)` | `(1 - hu) mu` |
| `hurdle_lognormal` | identity, `mu` | `sigma`, `hu` | `hu` (a point mass) | `(1 - hu) LogNormal(y \| mu, sigma)` | `(1 - hu) exp(mu + sigma^2 / 2)` |

`NB` is brms's `neg_binomial_2` (mean `mu`, variance `mu + mu^2 / shape`, `NB(0 | mu, shape) = (shape / (mu + shape))^shape`),
`mu = exp(eta)` under the log link and `p`, `mu = expit(eta)` under the logit; these are brms's densities as its vignette
"Parameterization of Response Distributions in brms" writes them. Each family is a `FamilySpec` with both a
`scale_parameter` (`shape`, Bambi's `alpha`, sampled as `alpha_log__`; `phi`; `sigma`) and the `zero_parameter` (`zi` or `hu`,
`Priors(zi=ab.Beta(...))`, sampled as `zi_logodds__`), and keeps its linear predictor as Bambi's parent under the identity link,
as B06's do (`likelihoods._zero_family`).

**Numerics.** The negative binomial in `eta`: `log C(y + shape - 1, y) - shape softplus(eta - log shape) - y softplus(log shape
- eta)`, two non-positive terms on the stable softplus, the coefficient through Stirling's remainder (`likelihoods._log_rising`),
whose terms are of the size of the result for every shape, where `lgamma(y + shape) - lgamma(shape)` loses every digit as the
shape grows (at shape 1e12 the two are 2.6e13 each); its zero mass `-shape softplus(eta - log shape)`; the hurdle's
`log(1 - NB(0))` as `cloglog_log_p1(t)` with `t = log shape + log softplus(eta - log shape)` (B06's hurdle truncation, the
Poisson's `t = eta` in the limit), exact where `1 - NB(0)` by subtraction is 0 (at `mu = 1e-30`). The zero-inflated binomial's
zero row is `log(zi + (1 - zi) exp(-n softplus(eta)))` as B06's two-term log-sum-exp, its other rows M2's binomial; the Beta part
is the Beta family's (`1 - mu` as `expit(-eta)`, D13), the Gamma part written in `eta`, the lognormal part PyMC's. The terms of a
zero or positive part are evaluated only on their own rows (the response at an interior value elsewhere, never at 0, where the
Beta, Gamma and lognormal densities are infinite), so a branch a switch does not take is finite with a finite gradient. The
negative binomial rows are exact to 1e-12 relative (value, and gradient in `eta`, `log shape` and `zi` / `hu`) against a
50-digit reference from `eta = -1000` to `1000` and shapes from 1e-6 to 1e9 on both paths, never NaN.

**Using it.** Responses: counts as for the Poisson family; `successes | trials(n)` as for the binomial (zero-trial rows
dropped, D4); proportions in [0, 1) for `zero_inflated_beta` (an exact 1 is refused, as brms refuses it: brms's
`zero_one_inflated_beta` and `xbeta` model it, offered since E01); non-negative finite reals for the hurdle Gamma and lognormal. An all-zero
response is accepted with a warning. Priors: the family's auxiliary parameter and `zi` / `hu` are required; brms's defaults on
request (`ab.default_priors`): `inv_gamma(0.4, 0.3)` for the negative binomial shape, `gamma(0.01, 0.01)` for `phi` and the
hurdle Gamma's shape, `student_t(3, 0, scale)` for the hurdle lognormal's `sigma` (brms's scale of `log y`), `beta(1, 1)` for
`zi` / `hu`. `Model.value_point(..., shape=1.3, zi=0.3)`. `predict()`: `kind="linear"` is `eta` (`mu` for the lognormal),
`kind="mean"` the table's `E[y]` under the name `mean` (`expected` for the zero-inflated binomial, which needs the trials of new
rows), `kind="response"` a zero where a uniform draw falls below `zi` / `hu`, as brms decides it, otherwise a draw of the other
part. The hurdle negative binomial's positive part is drawn exactly from the zero-truncated negative binomial
(`likelihoods.zero_truncated_negbinomial_draws`: the Gamma-Poisson mixture's rate drawn from its distribution given a positive
count, in closed form, then a zero-truncated Poisson count); brms's `posterior_predict` draws are not that distribution
(divergence D86). `log_lik()`, `loo()`, `waic()`, `hypothesis()` and `conditional_effects()` (not the zero-inflated binomial,
whose grid needs trials, as for the binomial) take these models; `describe()` writes the mixture out; prior-predictive draws and
the fit record work as for B06's.

**Contexts.** The term model (categorical predictors, interactions, transforms, no intercept, any number of grouping factors,
independent and correlated slopes, the offset) and Gaussian processes `gp()` enter the linear predictor as for every family
(C03), and `weights()` is taken. Refused naming the construct (divergence D84): links other than the default; formulas on
`zi` / `hu` or on `shape`, `phi`, `sigma`; `cens()` and `trunc()`; `mo()`; `me()` and `mi()`; `mm()` and `gr(by = )`;
autocorrelation terms; multivariate, non-linear (`nl=True`) and mixture models. E03 lifted `zi ~ ...` for
`zero_inflated_negbinomial` and `hu ~ ...` for `hurdle_lognormal` ([contexts of the newer families](contexts-newer-families.md)).

## Evidence

Oracle cases `zero_inflated_poisson_varint` and `hurdle_poisson_varint` (`zcnt ~ x + (1 | g)`, `beta(2, 2)` on `zi` /
`hu`) pass gates G1–G6 against the frozen brms fixtures: design and group coding; log density and gradient with and
without the Jacobian at every point, within 3e-16 and 2e-15 relative (brms's `zi` / `hu` coordinate is checked to be
`logit` both by its probe step and at every point); the `zi` / `hu` prior inside `lprior`; the graph's `log_mu` against
`posterior_linpred`; `predict()` against `posterior_linpred` / `posterior_epred` on fitted and new rows with and without
group effects; the JAX density; and `ab.default_priors()` against brms's `default_prior()` table (gate G3 compares a
log-link family's parent with `posterior_epred` only when the parent is the expected response, as for the Poisson; here
it is `eta`). `tests/test_zero_inflated_hurdle.py` checks both row densities and their gradients in `eta` and in
`zi` / `hu` against a 50-digit reference on `eta` from -1000 to 1000 on both paths (PyTensor's gradient and JAX's
autodiff of the jaxified graph), the model's density and gradient by hand, the zero-truncated sampler's distribution,
the refusals, `value_point`, seeded predictions under both new-group policies, `describe()`, prior-predictive draws,
three `default_prior()` tables brms printed and the fit record. Nothing samples.

D03: the oracle cases `zero_inflated_negbinomial_varint`, `hurdle_negbinomial_varint`, `zero_inflated_binomial_varint`,
`zero_inflated_beta_varint`, `hurdle_gamma_varint` and `hurdle_lognormal_varint` (each `~ x + (1 | g)` with `beta(2, 2)` on
`zi` / `hu`) and `zero_inflated_poisson_weighted` and `hurdle_poisson_weighted` (`zcnt | weights(w) ~ x + (1 | g)`) pass gates
G1-G6: the weights in the standata; log density within 4.4e-16 relative and gradient within 9.4e-10 (1.5e-14 for the
negative binomial families) with and without the Jacobian at every point; `posterior_epred` within 5.3e-15, on fitted and new
rows with and without group effects; the JAX density; brms's `default_prior()` tables. `tests/test_zero_inflated_hurdle_more.py`
checks the rows against the 50-digit and SciPy references on both paths, the samplers (the zero-truncated negative binomial
against its pmf at shapes from 1e-12 to 1e8, brms's recorded draws against the same pmf, `tests/zero_hurdle_probes`), each
model's density and gradient by hand, priors, responses and `value_point`, predictions by hand, the weighted density and
`log_lik()`, `loo()`, `waic()` of the weighted cases, every refusal, `gp()`, `conditional_effects()`, `describe()`,
prior-predictive draws and the fit record; `tests/test_post_processing.py` checks `log_lik()` against each model's own pointwise
density at the frozen points of all eight cases.
