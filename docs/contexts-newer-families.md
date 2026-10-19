# Distributional formulas and `weights()` for the Tier D families

brms-parity feature E03: brms's `bf(y ~ ..., <dpar> ~ ...)` on five parameters of the D01-D03 families, and `weights()` for two of
those families, where a frozen brms case verifies each.

```python
import autobayes as ab
base = dict(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd=ab.HalfNormal(1))
f = {"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 0.5)}           # the priors of the parameter's own formula
ab.weibull(["ypos ~ x + (1 | g)", "shape ~ z"], data, ab.Priors(**base, shape=f), categorical={"z": "p"})
ab.beta_binomial(["k | trials(size) ~ x + (1 | g)", "phi ~ z"], data, ab.Priors(**base, phi=f), categorical={"z": "p"})
ab.zero_inflated_negbinomial(["zcnt ~ x + (1 | g)", "zi ~ z"], data, ab.Priors(**base, shape=ab.InverseGamma(0.4, 0.3), zi=f),
                             categorical={"z": "p"})
ab.hurdle_lognormal(["zpos ~ x + (1 | g)", "hu ~ z"], data, ab.Priors(**base, sigma=ab.HalfNormal(1), hu=f), categorical={"z": "p"})
ab.skew_normal(["y ~ x + (1 | g)", "alpha ~ z"], data, ab.Priors(**base, sigma=ab.HalfNormal(1), alpha=f), categorical={"z": "p"})
ab.weibull("ypos | weights(w) ~ x + (1 | g)", data, ab.Priors(**base, shape=ab.Gamma(0.01, 0.01)))
ab.beta_binomial("k | trials(size) + weights(w) ~ x + (1 | g)", data, ab.Priors(**base, phi=ab.Gamma(0.01, 0.01)))
```

## The formulas

Each is B01's distributional formula (docs/distributional.md): a formula of its own with population-level terms of every kind
(numeric and declared categorical predictors, interactions, transforms; `0 + ...` and `~ 1`), centred on its own means when it has
an intercept (brms's `Xc_<dpar>`), its intercept and coefficients brms's `Intercept_<dpar>` and `b_<dpar>`, and the parameter one
value per row, the inverse of the formula's link at its linear predictor `eta_<dpar>`. Bambi builds it as a distributional
component named after the parameter (`alpha` for the Weibull's shape, as for B01's shapes).

| family | parameter | link (brms's default) | row i's value | brms's default prior of the intercept | frozen case |
| --- | --- | --- | --- | --- | --- |
| `weibull` | `shape` | log | `exp(eta_shape,i)` | `student_t(3, 0, 2.5)` | `weibull_distributional_shape` |
| `beta_binomial` | `phi` | log | `exp(eta_phi,i)` | `student_t(3, 0, 2.5)` | `beta_binomial_distributional_phi` |
| `zero_inflated_negbinomial` | `zi` | logit | `inv_logit(eta_zi,i)` | `logistic(0, 1)` | `zero_inflated_negbinomial_distributional_zi` |
| `hurdle_lognormal` | `hu` | logit | `inv_logit(eta_hu,i)` | `logistic(0, 1)` | `hurdle_lognormal_distributional_hu` |
| `skew_normal` | `alpha` | identity | `eta_alpha,i` | `normal(0, 4)` | `skew_normal_distributional_alpha` |

The family's density is unchanged with each row's parameter in place of the one parameter: the Weibull's scale
`mu_i / Gamma(1 + 1 / shape_i)`, the beta-binomial's shapes `mu_i phi_i` and `(1 - mu_i) phi_i`, `zi_i + (1 - zi_i) NB(0 | mu_i, shape)`
at a zero count, `hu_i` at a zero and `(1 - hu_i) LogNormal(y_i | mu_i, sigma)` above, Azzalini's skew normal with `alpha_i`; and so
are brms's `posterior_epred` (`(1 - zi_i) mu_i`, `(1 - hu_i) exp(mu_i + sigma^2 / 2)`; the others do not read the parameter) and
response draws.

`FamilySpec.distributional_parameters` lists what takes a formula: the scale parameter under `distributional_link` (B01's
mechanism, now set for the Weibull and the beta-binomial too) and the `DistributionalFormula` entries of
`FamilySpec.distributional_formulas` (E03: the zero parameter or an extra parameter, with its link and brms's default intercept
prior); `FamilySpec.distributional_parameter_link(name)` is the formula's link, which `Design.dpars[name].link` records and the
model checks against Bambi's.

## API

- Priors: the parameter's entry is its formula's mapping, `Priors(zi={"intercept": ..., "b": ...})`, read as B01's
  (`Priors.dpar_priors`): location priors, `"intercept"` exactly when the formula has one. `ab.Logistic(mu, s)` (brms's
  `logistic(mu, s)`, Stan's and PyMC's density) is offered for the intercept of a `zi` or `hu` formula, brms's default there, and
  refused naming the class anywhere else (D118). Without a formula the parameter keeps its one prior (`zi=ab.Beta(...)`,
  `alpha=ab.Normal(...)`), and a mapping is refused naming the form.
- Names: `model.names[<parameter>]` and `model.dpar_names[<parameter>]["value"]` are the per-row deterministic (`zi`, `hu`,
  `alpha`; Bambi's `alpha` for the Weibull's `shape`); the formula's variables are `zi_Intercept`, `zi_<term>`.
- `value_point(..., zi={"intercept": a, "b": [...]})`, as `sigma=` for a B01 formula.
- `predict()`: `kind="mean"` and `kind="response"` read each row's parameter; `dpar="zi"` returns its linear predictor
  (`kind="linear"`, attribute `link` `"logit"`) or its values (`kind="mean"`); on new rows the formula's columns are needed.
- `ab.log_lik`, `ab.loo`, `ab.waic`, `ab.bayes_R2`, `prior_predictive()`, `describe()` (the formula's paragraph with its link:
  `logit(zi_i) = eta_zi,i = ...`, `alpha_i = eta_alpha,i = ...`), `prior_table()`, the target identity, fit records
  (`zi:intercept`, `zi:b:z[q]`) and `ab.default_priors()` (the intercept's default above, flat coefficients, no class row of the
  parameter; read from the frozen fixtures and from `tests/contexts_probes`, where a response a thousand times wider leaves the
  intercept's default unchanged) take them.
- `weights()`: `ypos | weights(w) ~ ...` for the Weibull family and `k | trials(size) + weights(w) ~ ...` for the beta-binomial
  family (after `trials()`, as for the binomial): each row's log likelihood is multiplied by its weight (B02's wrapper), `log_lik()`
  is the weighted pointwise density, as brms's, and `loo()` / `waic()` read it; the priors and the predictions ignore the weights.
  A formula and `weights()` combine (each row's own parameter and its weight; the densities are checked by hand).

## Refused, naming the construct

- Group-level terms in these formulas: brms's `zi ~ z + (1 | g)` (D32, as for every distributional formula) and a correlation ID
  shared with the response formula, `zi ~ 1 + (1 |p| g)`, which D08 offers for B01's formulas only (D116); `offset()`, `mo()`,
  `gp()`, `me()` and autocorrelation terms in them, as for every distributional formula.
- Formulas on every other parameter of the D01-D03 families (the skew normal's, ex-Gaussian's, shifted lognormal's and hurdle
  lognormal's `sigma`, the ex-Gaussian's `beta`, the shifted lognormal's `ndt`, the zero-inflated and hurdle negative binomial's and
  hurdle Gamma's `shape`, the other families' `zi` and `hu`, ...), `weights()` for the exponential, skew-normal, shifted lognormal,
  ex-Gaussian, geometric and `negbinomial2` families, and every other construct D79, D81 and D84 list.
- `conditional_effects()` keeps refusing a distributional formula and `weights()` (D99), `prior_draws()` a distributional formula
  (D124), and `hypothesis()` the formulas' coefficients (D97); none is verified for these models.

## Numerics

The formulas' parameter is Bambi's inverse link of the linear predictor, so the row densities read `zi_i` and `hu_i` as
probabilities: `log(1 - zi_i)` is formed from `zi_i` rounded to double (a relative error of about `1e-16 / (1 - zi_i)`: 3.6e-8
at `eta_zi = 20`), and Bambi clips the inverse logit to `[eps, 1 - eps]` where it rounds to 0 or 1 (`|eta| > 36.7` above, below about
-745), where the density saturates and its gradient in the formula's coefficients vanishes; brms evaluates these rows on the logit
scale (Stan's `bernoulli_logit_lpmf` of the linear predictor, in its `_logit` forms of these densities) and stays exact (D117). On the frozen points (`|eta_zi|` and `|eta_hu|` below 1.7) the log densities equal brms's to 4.3e-16.

## Evidence

Seven frozen cases pass gates G1-G6 against brms 2.23.0 (`tier_e` data, 36 rows, a varying intercept of `g`): the five above and
`weibull_weighted` and `beta_binomial_weighted`. G1 matches `X_<dpar>` and `weights` by content; G2's log density with and without
the Jacobian is within 7.9e-16 relative (the prior part within 2.8e-16) and every gradient coordinate within 2.5e-9 (GRAD_REL is
1e-7); G3 checks the graph's per-row parameter against the inverse of brms's default link of that parameter (`exp`, `inv_logit`,
the identity; `gates._BRMS_DPAR_LINK`) at brms's own linear predictor, and the raw-scale intercept `b_<dpar>_Intercept`; G4
`posterior_linpred` and `posterior_epred` on fitted and new rows with and without group effects (within 9.3e-16); G5 the JAX
density; G6 the default-prior table with the `dpar` rows (`logistic(0, 1)` on `Intercept_zi` and `Intercept_hu`). The adapter and
gates were extended, not loosened: a `zi`, `hu` or `alpha` formula is handed to `value_point` as B01's are, brms's `X_<p>` of any
family parameter without a formula in the model is a mismatch, and G2's prior part and G6's expected classes leave out a parameter
with a formula (as B01's do for `sigma`). `tests/test_contexts_newer_families.py` covers the registry, what stays refused, the
priors and `Logistic`'s density against SciPy's, each family's density with one parameter per row written from SciPy's
distributions on both paths (and weighted), `value_point`, predictions (means, seeded draws, `dpar=`, new rows), `log_lik` with and
without weights, `loo` / `waic` on the weighted cases, `describe()`, the identity, fit records, prior-predictive draws, the nine
`default_prior()` tables of `tests/contexts_probes/brms-2.23.0-contexts.json` (`LC_ALL=C Rscript
tests/contexts_probes/probe_contexts.R <out>`), the gates on corrupted fixtures and the refusals of D99. Nothing samples.
