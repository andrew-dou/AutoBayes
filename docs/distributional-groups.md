# Group-level terms, offsets and more parameters in distributional formulas

brms-parity feature G06 (brms's `bf(y ~ x + (1 | g), sigma ~ z + (1 | g))`, `(1 |p| g)` across formulas, `sigma ~ z +
offset(o)`, `nu ~ z`, `zi ~ z + (1 | g)`, `disc ~ z`). It extends B01's distributional formulas (docs/distributional.md) and
D08's correlation ID shared with sigma's formula (docs/multi-membership-and-by.md).

```python
import autobayes as ab
priors = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=ab.HalfNormal(1),
                   sigma={"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 0.5), "sd": ab.HalfNormal(0.5)})
model = ab.gaussian(["y ~ x + (1 | g)", "sigma ~ z + (1 | g)"], data, priors, categorical={"z": "p"})
```

## What is offered

| construct | brms | the model | verified on |
| --- | --- | --- | --- |
| a group of the parameter's own, `sigma ~ z + (1 \| g)` (Gaussian) | an ID of its own (`sd_2`, `z_2`, `r_2_sigma_1`, prior class `sd` with `dpar = sigma`) | `log(sigma_i) = a_sigma_c + Xc_sigma b_sigma + u_sigma,g[g_i]`, `u_sigma,g = sd_sigma,g * z_sigma,g`, uncorrelated with every other group | `gauss_sigma_varint`, `gauss_sigma_varint_post` |
| varying slopes in a correlation ID shared with the response formula, `(1 + x \|p\| g)` in `y ~ x + ...` and in `sigma ~ z + x + ...` (Gaussian) | one group, `cholesky_factor_corr[4] L_1` over `Intercept`, `x`, `sigma_Intercept`, `sigma_x` | one correlation matrix over the four coefficients (`model.correlation_coefs("g")`), sigma's slope's block the level indicator times the raw `x` | `gauss_sigma_shared_id_slopes` |
| `offset()` in sigma's formula (Gaussian) | `offsets_sigma` added to sigma's linear predictor | `DparDesign.offset`, added with no coefficient on every row, new rows included | `gauss_sigma_offset` |
| `nu ~ z` (Student-t) | `nu = 1 + exp(eta)` (link `logm1`), `Intercept_nu` with default `normal(2.7, 0.8)` | the same, `nu_i` per row; `ab.student([...], data, priors, nu=None)` with `Priors(nu={"intercept": ..., "b": ...})`; `nu=4` next to `nu ~ ...` is refused | `student_nu_formula` |
| `zi ~ z + (1 \| g)` (zero-inflated Poisson) | logit link, `Intercept_zi` with default `logistic(0, 1)`, zi's group an ID of its own (`sd \| dpar zi` default `student_t(3, 0, 2.5)`) | the same; `Priors(zi={"intercept": ..., "b": ..., "sd": ...})`; predictions without group effects drop zi's too | `zip_zi_formula` |
| `disc ~ z` (cumulative, logit link) | `disc = exp(Intercept_disc + Xc_disc b_disc)`, `P(y <= k) = inv_logit(disc (theta_k - mu))`, `Intercept_disc` with default `normal(0, 1)` | the same likelihood; `Priors(disc={"intercept": ..., "b": ...})`, a warning that the scale is identified by the priors alone (D188) | `cumulative_disc` |

The priors of a formula's group-level terms are its `"sd"` entry, read as the response formula's `sd` is: one prior for a
single intercept-only term, `{"g": prior}`, or `{"g": {"Intercept": prior, "x": prior}}` for a term with varying slopes (brms's
`class = sd, dpar = sigma`). `value_point` takes them as `sigma={"intercept": ..., "b": [...], "sd": {...}, "z": {...}}`, the
same shapes; the Student-t's `nu`, the zero-inflated Poisson's `zi` and the cumulative `disc` (a new `value_point` keyword) take
their formula's `{"intercept": ..., "b": ...}` (and `"sd"`, `"z"` for zi's groups). `model.dpar_group_names` names each
coefficient's PyMC variables (`sigma_1|g`, `sigma_1|g_sigma`, `sigma_1|g_offset`; `sigma_x|g` for a slope), and `model.names`
holds each parameter's per-row values under its brms name (`names["nu"]`, `names["disc"]`).

`predict()` adds each row's effects and offset to the parameter's linear predictor, `include_group_effects=False` drops them
(brms's `re_formula = NA` drops every parameter's), and `dpar=` returns the parameter itself (`predict(..., dpar="disc")`). New
levels of a distributional formula's own term are refused, naming it (a shared ID's are refused as D08 refused them). `log_lik`
and the summaries take sigma's own varying intercept: `fixef()` interleaves `sigma_Intercept` and sigma's coefficients,
`ranef()`, `coef()` and `VarCorr()` put sigma's terms in the factor's tables (`sigma_Intercept`), and `posterior_summary()`
names its effects `r_g__sigma[level,Intercept]` and its SD `sd_g__sigma_Intercept`, as brms does (`gauss_sigma_varint_post`;
the summaries take it where the response formula's only group-level term is `(1 | g)` of the same factor).

`ab.default_priors()` returns brms's defaults: `student_t(3, 0, scale)` on a sigma group's SD (brms's `def_scale_prior` of
`log y`), `student_t(3, 0, 2.5)` on zi's (the logit link reads no response), `normal(2.7, 0.8)` on nu's intercept and
`normal(0, 1)` on disc's; `prior_table()` writes the `sd | dpar` rows. `describe()` writes each formula's group-level terms, slopes
and offset into its linear predictor, the SD priors, and the parameter's likelihood role.

## Implementation notes

- Bambi builds a distributional formula's group-level terms in the parameter's component (`sigma_1|g`, `sigma_x|g`) and its
  `offset()`; `Model._dpar_design` checks each block against the data (the level indicator, times the slope's raw column) and
  the offset against its values, and a shared ID's coefficients are correlated after the build as D08's are.
- brms's `logm1` link is not Bambi's: the model sets the Student-t family's link of `nu` to `likelihoods.logm1_link()`.
- The cumulative family's mean is built on its raw design (B05), and Bambi centres every formula or none, while brms centres
  `disc`'s (`Xc_disc`): the family hands Bambi the disc formula uncentred under `likelihoods.centred_log_link`, whose inverse
  takes the column means times the coefficients off the linear predictor, so that the sampled `disc_Intercept` is brms's
  centred `Intercept_disc` and the graph's `disc` is brms's (`DistributionalFormula.centred_by_likelihood`). A fit's posterior
  then holds that centred intercept (`DparDesign.centred_draws`), which `predict()` reads with the centred design.

## Refused, naming the construct

- A group of its own, varying slopes in a shared ID and `offset()` in the formula of any parameter other than the Gaussian's
  sigma, and a group of its own in any parameter's but that and the zero-inflated Poisson's zi (`FamilySpec.distributional_constructs`,
  `DistributionalFormula.group_terms`; D32): no frozen case verifies them.
- Several `offset()` terms, or an expression inside one (G04's summed offsets, docs/expressions.md), in a distributional formula
  (D32): `offset(col)` and `offset(log(col))` are offered in the Gaussian's sigma formula.
- Varying slopes in a term of the formula's own, a term without its varying intercept, a factor of two columns (`g:h`),
  `mm()` and `gr()` arguments in a distributional formula (D32); a correlation ID in zi's formula (D116); formulas on the other
  parameters (D33), `disc ~ ...` under the cumulative probit and cloglog links, with `thres(gr = )` and for the other ordinal
  families (D38); new levels of a distributional formula's own term.
- The summaries of any other distributional formula with group-level terms (D162).

## Divergences

- D187: with `(1 |p| g)` in both formulas and the priors `sd, group = g` and `sd, dpar = sigma`, brms 2.23.0's
  `prior_summary()` gives sigma's SD the dpar row's prior while its Stan code puts the group row's on the whole `sd_1` vector,
  sigma's SD included. AutoBayes applies what the table says. `gauss_sigma_shared_id` keeps per-gate expected failures on
  `g2_logp` and `g2_gradient` (point 0: the prior part -19.5032 here, -11.5087 in brms; the likelihood equal; every other gate
  passes); written with its group (`sd, group = g, dpar = sigma`) brms applies sigma's prior, and the feature is accepted on
  `gauss_sigma_shared_id_slopes`.
- D188: the cumulative likelihood is unchanged by `(disc, thresholds, b, r) -> (c disc, thresholds / c, b / c, r / c)`, so a disc
  formula with an intercept (or a factor coded by all its levels, `disc ~ 0 + z`) leaves the model's scale to the priors. brms
  estimates it so and warns of nothing (probed with `make_standata()`, `make_stancode()` and `default_prior()` for `disc ~ z`,
  `disc ~ 0 + z` and `disc ~ 0 + x`); the model is brms's, and it warns. `disc ~ 0 + x` with numeric terms only (disc = 1 where
  they are 0) does not warn.

## Evidence

Eight frozen cases (fixtures #109): `gauss_sigma_varint`, `gauss_sigma_shared_id_slopes`, `gauss_sigma_offset`,
`student_nu_formula`, `zip_zi_formula`, `cumulative_disc` and `gauss_sigma_varint_post` pass gates G1-G6 and the post gate
against brms 2.23.0; `gauss_sigma_shared_id` passes every gate but the two D187 names. G1 matches each formula's own groups
(`Z_m_<dpar>_c`, `J_m`, `M_m`, no correlation) and checks `offsets_<dpar>` (`gates.check_dpar_offsets`); G2 the log density
with and without the Jacobian and every gradient coordinate (`sd_m`, `z_m` of a formula's own group mapped to the parameter's
variables; the shared ID's four-coefficient `cholesky_corr` coordinates); G3 each parameter's per-row values against brms's own
linear predictor under its link (`logm1` for nu, `log` for disc) and every group's effects; G4 `posterior_linpred` /
`posterior_epred` with and without group effects; G5 the JAX density; G6 the default-prior tables. The syntax corpus entries
`dist_sigma_group`, `dist_sigma_offset`, `dist_student_nu`, `dist_zip_zi` and `dist_cumulative_disc` now build and match brms's
reading (the audit compares `offsets_<dpar>`). `tests/test_distributional_groups.py` covers the term model and every refusal,
the priors and defaults, the blocks, value points, predictions, describe() and identity, and the log density of each construct
against an independent computation.
