# Zero-inflated and hurdle Poisson counts

brms-parity feature B06 (`zero_inflated_poisson()`, `hurdle_poisson()`).

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

Not modeled: `zi ~ x` / `hu ~ x` (distributional formulas, B01), and brms's ten other zero-inflated and hurdle families,
the negative binomial ones among them (divergence D30).

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
