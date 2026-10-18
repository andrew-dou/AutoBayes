# Beta responses with a logit link

brms-parity feature A11 (`brmsfamily("Beta")`).

```python
import autobayes as ab
model = ab.beta("share ~ x + (1 | group)", data,
    ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd=ab.HalfNormal(1), phi=ab.Gamma(0.01, 0.01)))
print(model.describe())
```

## The model

Proportions strictly inside `(0, 1)`, logit link on the mean, `mu_i = expit(eta_i)`, and brms's parameterisation
`beta_lpdf(y | mu * phi, (1 - mu) * phi)`: `E[y_i] = mu_i`, `Var[y_i] = mu_i (1 - mu_i) / (1 + phi)`. `phi` is brms's
`real<lower=0> phi`, the precision, with its own positive-support prior (brms class `phi`, `Priors(phi=...)`, the way
`shape` serves the negative binomial and Gamma families); the sampled variable is `phi_log__` and `model.names["phi"]`
is `"phi"`. Coefficients are on the logit (log-odds) scale of the mean; the centred intercept is the logit of the
expected proportion at the predictor means, and with categorical predictors `expit(a_c)` is the expected proportion
at the proportion-weighted predictor, not the average over levels (the link is nonlinear).

**Graph.** The likelihood is PyMC's `Beta(alpha, beta)` with `alpha = expit(eta) phi` and `beta = expit(-eta) phi`,
built by `autobayes/likelihoods.py` as a Bambi family that keeps the linear predictor as the parent `logit_mu` under the
identity link, exactly as the Bernoulli and binomial families do, so `model.names["linpred"]` is `"logit_mu"` and the
graph holds `eta`, not `expit(eta)`. Bambi's own `beta` family (mean `mu` and sample size `kappa`, where `kappa` is
brms's `phi`) is not used: its deterministic `mu` would hold `expit(eta)`, and the precision would be named `kappa`
rather than brms's `phi`. `1 - mu` is computed as `expit(-eta)` rather than by subtraction (divergence D10): the same
number up to rounding, and it keeps `beta > 0` for `|eta|` up to about 745 where brms's subtraction returns exactly zero
beyond `eta` of about 37.

## Using it

- Response: finite reals strictly between 0 and 1; an exact 0 or 1, a value outside the interval, an infinity, a boolean
  or a non-numeric column is refused with the value named. Missing values are refused by the shared check.
- Priors: `phi` is required; `sigma` and `shape` are refused with the message naming `phi`; `phi` is refused for every
  other family. Any positive-support class (`Gamma`, `HalfNormal`, `Exponential`, `HalfCauchy`, `HalfStudentT`) is accepted.
- `Model.value_point(..., phi=0.7)`; `sigma=` and `shape=` are refused with the same rule.
- `predict()`: `kind="linear"` is `eta` (brms `posterior_linpred`), `kind="mean"` is `mu = expit(eta)` (brms
  `posterior_epred`), `kind="response"` adds continuous draws `Beta(mu phi, (1 - mu) phi)`; a draw whose shape parameter
  underflows to zero is refused. Both unseen-group policies work. `prior_predictive()` works; `fit_record` records the
  phi prior and the `phi` posterior row.

## Evidence

Oracle case `beta_varint` (`prop ~ x + (1 | g)`, `gamma(0.01, 0.01)` on `phi`) passes gates G1–G6 against the frozen
brms fixture: design, log density and gradient with and without the Jacobian (brms's `phi` has the `log` transform, as
`phi_log__`), the phi prior inside `lprior`, group effects, `posterior_linpred` / `posterior_epred` on fitted and new
rows with and without group effects, and the JAX density. `tests/test_beta.py` checks the density and gradient by hand
on both paths at responses 1e−6, 0.5 and 1 − 1e−6 and logits −30, 0 and 30 for `phi` from 0.05 to 1e4 (the gradients in
`log(phi)` and the intercept against their closed forms), the refusals, `value_point`, seeded predictions under both
new-group policies, `describe()`, prior-predictive shapes and the fit record. Nothing samples.

No `link_phi` or zero/one inflation is modeled. brms's probit and cloglog links are offered since E06,
`ab.beta(..., link="probit")` and `link="cloglog"`, without a formula on `phi` ([links](links.md)).
