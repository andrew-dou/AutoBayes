# Student-t with learned degrees of freedom

brms-parity feature A13 (`student()` with a prior on `nu`).

```python
import autobayes as ab
model = ab.student("y ~ x + (1 | g)", data,
    ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1),
              nu=ab.Gamma(2, 0.1)))
print(model.describe())
```

## The model

`y_i ~ StudentT(nu, mu_i, sigma)` with the degrees of freedom `nu` a parameter. brms declares `real<lower=1> nu`, adds
the prior's log density minus its log probability of `nu > 1` to `lprior` (for `gamma(2, 0.1)`:
`gamma_lpdf(nu | 2, 0.1) - gamma_lccdf(1 | 2, 0.1)`, the constant being `log(exp(-0.1) * 1.1)`) and samples
`log(nu - 1)`, whose Jacobian adds `log(nu - 1)`. Here the prior of `nu` is `pm.Truncated(<prior>, lower=1)`: the same
truncated and renormalised density, sampled through PyMC's interval transform as the value variable
`nu_interval__ = log(nu - 1)` with the same Jacobian. Bambi takes it through its `Prior(dist=...)` hook for the
Student-t family's auxiliary parameter `nu` (`Prior.to_bambi_truncated`). The family declares which of its fixed
parameters a model may learn instead, with the lower bound (`FamilySpec.learnable_parameters = {"nu": 1.0}` for the
Student-t family, docs/families.md); a model built with `nu=None` lists it in `model.learned_parameters`.

## Using it

- `ab.student(formula, data, priors)` (`nu=None`, the default) with `Priors(nu=...)` learns nu; `ab.student(..., nu=4)`
  fixes it as before: fixed-nu models, their `describe()`, names and target identities are unchanged.
- The prior is one whose truncation PyMC can renormalise: `Gamma` (brms's default for `nu` is `gamma(2, 0.1)`, which
  this library does not fill in: D1; `ab.default_priors(..., family="student")` returns it on request, A07), `Exponential`, `HalfNormal(s)` (brms `normal(0, s)` on `nu`: the truncation makes
  the two equal) or `HalfCauchy(s)` (brms `cauchy(0, s)`).
- Refused, naming what is accepted: `nu=4` together with `Priors(nu=...)`; `nu=None` without a nu prior; a location
  prior (`Normal`, `StudentT`; brms accepts e.g. `normal(5, 2)` on `nu`) and `HalfStudentT`, which PyMC cannot truncate
  (it has no CDF for it), both divergence D22; `Priors(nu=...)` for any other family; `value_point(nu=...)` for a model
  whose nu is fixed.
- `model.nu is None`, `model.names["nu"] == "nu"`, `model.learned_parameters == {"nu": 1.0}`;
  `value_point(..., nu=3.5)` puts `log(2.5)` in `nu_interval__`.
- `predict()`: `kind="mean"` and `kind="linear"` are `mu`, as before; `kind="response"` draws
  `mu + sigma * t(nu)` with each posterior draw's own nu, by hand or (fitted rows) through Bambi.
- `describe()` gives the prior, the truncation and the sampled coordinate `log(nu - 1)`; the target identity records the
  nu prior only when there is one; `fit_record` carries it under `priors` (with no `family_parameters`) and the nu
  posterior row; `prior_predictive()` draws nu from the truncated prior.
- The order of the sampled variables. Bambi creates `sigma` and `nu` in the iteration order of a set of their names
  (`Family.auxiliary_parameters`), which depends on `PYTHONHASHSEED`, so PyMC's order of `sigma_log__` and `nu_interval__`
  (`model.pymc_model.value_vars`) varies from one Python process to the next; the model, its density and its target identity
  do not. `describe()` lists them in one fixed order whatever PyMC's is: `sigma_log__`, then `nu_interval__` (the order of
  brms's `real<lower=0> sigma` and `real<lower=1> nu`), in the places PyMC gives the two (`Model._described_value_names`:
  generally, a family's scale parameter, then its zero parameter, then its learned parameters).
  `tests/test_student_learned_nu.py` builds the model under two hash seeds that order the set differently and compares.

## Evidence

Oracle case `student_learned_nu` (`y ~ x + (1 | g)`, brms `student()` with `gamma(2, 0.1)` on `nu`) passes gates G1-G6
against its frozen brms fixture: design, log density and gradient with and without the Jacobian, the prior part (the
truncated Gamma included), the linear predictor and group effects, `posterior_linpred` / `posterior_epred` on fitted
and new rows, the JAX density and the default-prior table. The adapter maps brms's `nu` (transform `other`) to
`nu_interval__` and checks the probe's step `du` against `log(nu - 1)`. `tests/test_student_learned_nu.py` checks the
refusals, the density against a closed-form reference with the truncation constant, every accepted prior's truncation
against its own closed form, the gradient of the `log(nu - 1)` coordinate on both paths against the analytic
derivative, `value_point`, predictions with per-draw nu, `describe()`, the target identity, the fit record and the
prior predictive. `tests/test_student.py` (fixed nu) passes unchanged. Nothing samples.
