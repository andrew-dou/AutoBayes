# Distributional formulas: `sigma ~ ...`, `shape ~ ...`, `phi ~ ...`

brms-parity feature B01 (`bf(y ~ x, sigma ~ z)`).

```python
import autobayes as ab
priors = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=ab.HalfNormal(1),
                   sigma={"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 0.5)})
model = ab.gaussian(["y ~ x + (1 | g)", "sigma ~ z"], data, priors, categorical={"z": "p"})
print(model.describe())
```

## The model

The family's auxiliary parameter gets its own linear predictor under a log link, one value per row:

    mu_i         = a_c + (x_i - mean(x)) b + u[g_i]                       (the response formula, as before)
    log(sigma_i) = a_sigma_c + (Xs_i - mean(Xs)) b_sigma                  (sigma's formula)
    y_i ~ Normal(mu_i, sigma_i)

`Xs` is the design of `sigma`'s formula, centred on its own fitted-row means whenever that formula has an intercept
(brms's `Xc_sigma`; brms centres a distributional formula even under `bf(..., center = FALSE)`). Its parameters are
brms's `Intercept_sigma` (the centred intercept, which the prior describes; brms reports the raw-scale one as
`b_sigma_Intercept`) and `b_sigma`, each with its own prior (brms's `class = Intercept` and `class = b` with
`dpar = sigma`). The formula takes every population-level term the response formula takes: numeric and declared
categorical predictors, two-way interactions, in-formula transforms; `0 + ...` removes its intercept, `sigma ~ 1` is a
log-scale intercept alone.

Six families offer it, for their one auxiliary parameter (`FamilySpec.distributional_link = "log"`):

| family | parameter | Bambi's component (PyMC names) | brms's default prior of its intercept |
| --- | --- | --- | --- |
| Gaussian, Student-t (fixed or learned `nu`), lognormal | `sigma` | `sigma` (`sigma_Intercept`, `sigma_z`) | `student_t(3, 0, scale)`, `scale` = max(2.5, MAD of `log y` rounded to one decimal; zeros read as 0.1; 2.5 when a response is negative) |
| negative binomial, Gamma (log link) | `shape` | `alpha` (`alpha_Intercept`, `alpha_z`) | `student_t(3, 0, 2.5)` |
| Beta | `phi` | `phi` (`phi_Intercept`, `phi_z`) | `student_t(3, 0, 2.5)` |

E03 (docs/contexts-newer-families.md) adds the Weibull's `shape` (Bambi's `alpha`) and the beta-binomial's `phi` under the same
log link, each with `student_t(3, 0, 2.5)` on its intercept, and formulas on three parameters besides the families' auxiliary one:
the zero-inflated negative binomial's `zi` and the hurdle lognormal's `hu` under the logit link (`logistic(0, 1)` on the
intercept) and the skew normal's `alpha` under the identity link (`normal(0, 4)`), with population-level terms only.

## API: a list of formulas

`formula` is a list (or tuple), the response formula first and then the parameter's formula with the parameter's brms name
on its left, `["y ~ x + (1 | g)", "sigma ~ z"]`; every constructor and `ab.default_priors` take it. This is the form the
frozen case proposed, it is Bambi's own (`bmb.Formula(main, *additional)`), it needs no new object, and a list is what
a case file, a fit record and the target identity can store as it is; an `ab.bf()` helper would have added a type that
only wraps the list. A one-element list is the string. `categorical={...}` is shared: a declared column is categorical
in every formula that names it as a main effect.

- Priors: `Priors(sigma={"intercept": ..., "b": ...})` (`shape=` and `phi=` alike). `"intercept"` is required exactly
  when the formula has one; `"b"` is one prior or a mapping by term (and level), read as the response formula's `b` is,
  and needs no entry for an intercept-only formula. Location priors only (`Normal`, `StudentT`, `Flat`). Without a
  formula the parameter keeps its one positive-support prior, and a mapping is refused naming the form.
  `Priors.dpar_priors(spec, "sigma")` gives the formula's priors as a `Priors` of its own.
- Names: `model.names["sigma"]` is now the per-row deterministic `sigma` (Bambi's `alpha` for `shape`);
  `model.dpar_names["sigma"]` has `"value"`, `"intercept"` (`sigma_Intercept`) and `"b:<term>"` (`sigma_z`, a vector over
  `sigma_z_dim` for a vector term). `model.spec.dpars` holds the formula as a `ModelSpec` of its own (`model.spec.dpar("sigma")`,
  `spec.dpar_names`, `spec.formulas`), `model.design.dpars["sigma"]` its design (`DparDesign`: `X_raw`, `X_model`,
  `x_means`, `columns`, `centered`, `link`).
- `value_point(..., sigma={"intercept": a, "b": [...]})`: the intercept as sampled and the coefficients in design-column
  order, all unconstrained.
- `predict()`: `kind="linear"` and `kind="mean"` keep their meaning (the linear predictor of `mu` and the expected
  response, brms's `posterior_linpred` and `posterior_epred`); each row's own `sigma` enters wherever the family needs
  it: the lognormal's mean `exp(mu + sigma_i^2 / 2)` and every `kind="response"` draw. `dpar="sigma"` returns the
  parameter itself: its linear predictor under `kind="linear"` (as `eta`, attrs `dpar` and `link`) and its values under
  `kind="mean"` (as `sigma`), brms's `posterior_linpred(dpar = "sigma")` and `posterior_epred(dpar = "sigma")`. A
  distributional model is always predicted in this layer, from the raw-scale intercept that Bambi reports after sampling.
- `describe()` writes the formula, its linear predictor and centring, its priors and variables out; `prior_table()`
  writes brms's `dpar` rows (and no class row for the parameter); the target identity includes the formula, its design,
  link and priors (models without one keep theirs); `fit_record` records `"distributional"` and the priors as
  `sigma:intercept`, `sigma:b:z[q]`; `compare_records` refuses records whose formulas differ; `prior_predictive()` draws
  the per-row parameter (and refuses a flat prior in the formula, as everywhere).
- `ab.default_priors(["y ~ x", "sigma ~ z"], data)` returns brms's defaults: the intercept's `student_t(3, 0, scale)`
  above and flat coefficients.

## Refused, naming the construct

- Group-level terms (`sigma ~ z + (1 | g)`) and `offset()` in a distributional formula (divergence D32). Bambi builds such
  group terms (`sigma_1|g`), but their priors, names, prediction policies and oracle mapping are not wired here.
- A formula on any other parameter or with any other link: Student-t `nu` (brms's `logm1` link), `zi` / `hu` (logit; E03 offers
  them for the zero-inflated negative binomial and the hurdle lognormal), a parameter the family does not have (`shape` for the
  Gaussian, `sigma` for the Poisson), `mu` (D33).
- `center_predictors=False` with a distributional formula that has an intercept and terms: brms centres it anyway and
  Bambi centres all formulas or none (D34).
- A distributional formula that reads the response, the trials column or a grouping factor of the response formula (D35);
  two formulas for one parameter; an addition term on its left-hand side; a transform of a declared categorical column.

## Evidence

Six frozen cases pass gates G1-G6 against brms 2.23.0 (`y ~ x + (1 | g)` on the `tier_b` data, `normal(0, 1)` on the
formula's intercept and `normal(0, 0.5)` on its coefficients): `gauss_distributional_sigma` (`sigma ~ z`, the case the
loop froze) and five frozen with this feature, `student_distributional_sigma` (`sigma ~ x`, learned `nu`),
`lognormal_distributional_sigma` (`sigma ~ x + z`), `negbinomial_distributional_shape` (`shape ~ z`),
`gamma_distributional_shape` (`shape ~ log(xpos)`) and `beta_distributional_phi` (`phi ~ x * z`). G1 matches `X_<dpar>`
by content to `Design.dpars` and checks `Kc_<dpar>` and the centring; G2 the log density with and without the Jacobian
(within 8.3e-16 relative), the prior part (`Intercept_<dpar>` and `b_<dpar>` included) and every gradient coordinate,
`Intercept_<dpar>` and `b_<dpar>` mapped to the formula's variables (within 7.2e-9 relative; GRAD_REL is 1e-7); G3 the
graph's per-row parameter against `exp` of brms's own linear predictor and the raw-scale intercept against
`b_<dpar>_Intercept`; G4 `posterior_linpred` / `posterior_epred` on fitted and new rows with and without group effects
(the lognormal's `posterior_epred` uses each row's `sigma`, within 1.1e-14); G5 the JAX density; G6 the default-prior
table with its `dpar` rows. The five new fixtures regenerate identically (`oracle/regen.py check`).
`tests/test_distributional.py` covers the term model and its refusals, the priors, a density written by hand on both
paths, `value_point`, predictions with each row's parameter for every family and `dpar=`, `describe()`, the identity,
fit records, prior-predictive draws, nine `default_prior()` tables printed by brms for a small frame (log-scale scales
of 4.4 and 2.5, no intercepts, `sigma ~ 1`, an interaction, a transform, each family), and that the gates fail on a
corrupted `gauss_distributional_sigma` fixture (`X_sigma`, `b_sigma`, the gradient in `Intercept_sigma`,
`b_sigma_Intercept`, the `dpar` rows of the default-prior table) and on a model that lacks brms's formula. Nothing
samples.
