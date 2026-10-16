# Shrinkage priors on b: `horseshoe()` and `R2D2()`

brms-parity feature D07 (brms's `set_prior(horseshoe(...), class = "b")` and `set_prior(R2D2(...), class = "b")`).

```python
import autobayes as ab
priors = ab.Priors(intercept=ab.Normal(0, 5), sigma=ab.HalfNormal(1),
                   b=ab.Horseshoe(df=3, scale_global=0.5, df_global=1, scale_slab=2, df_slab=4))
model = ab.gaussian("y ~ x + w + z + xpos", data, priors, categorical={"z": "p"})
print(model.describe())
r2d2 = ab.gaussian("y ~ x + w + z + xpos", data, ab.Priors(intercept=ab.Normal(0, 5), sigma=ab.HalfNormal(1),
                                                           b=ab.R2D2(mean_R2=0.4, prec_R2=3, cons_D2=0.7)),
                   categorical={"z": "p"})
```

`ab.Horseshoe(df=1, scale_global=1, df_global=1, scale_slab=2, df_slab=4)` and `ab.R2D2(mean_R2=0.5, prec_R2=2,
cons_D2=0.5)` have brms's defaults and brms's argument order. Each is one joint prior on every coefficient of class `b`
(every design column but the intercept), given as the bare `b` of `Priors`.

## The model

Both are built non-centred, as brms builds them: `b[k] = zb[k] * sdb[k]` with `zb[k] ~ Normal(0, 1)`, so `b` is a
transformed parameter and the sampled coordinates are `zb` and the prior's own scales. The scale of each coefficient is:

- **Regularised horseshoe** (Piironen & Vehtari 2017, *Electronic Journal of Statistics* 11):
  `sdb[k] = hs_global * sqrt(c2 * hs_local[k]^2 / (c2 + hs_global^2 * hs_local[k]^2))` with
  `hs_local[k] ~ half-Student-t(df, 0, 1)`, `hs_global ~ half-Student-t(df_global, 0, scale_global * sigma)`,
  `hs_slab ~ inverse-gamma(df_slab / 2, df_slab / 2)` and `c2 = scale_slab^2 * hs_slab`.
- **R2D2** (Zhang, Naughton, Bondell & Reich 2022, *JASA* 117): `sdb[k] = sqrt(R2D2_phi[k] * R2D2_tau2)` with
  `R2D2_tau2 = sigma^2 * R2D2_R2 / (1 - R2D2_R2)`, `R2D2_R2 ~ Beta(mean_R2 * prec_R2, (1 - mean_R2) * prec_R2)` and the
  simplex `R2D2_phi ~ Dirichlet(cons_D2)` over the coefficients (`cons_D2` one number for all, or one per coefficient in
  the design-column order `describe()` lists).

The residual SD `sigma` scales both (brms's `autoscale = TRUE`). brms sums the densities of `hs_global` and `hs_slab`
(or `R2D2_R2`) into its `lprior` and adds those of `zb`, `hs_local` (or `R2D2_phi`) in its model block; the graph's
density is the same sum, normalising constants included (the half-Student-t's `log 2`).

## How it is built

`Priors.to_bambi` hands Bambi, for each population-level term, a prior whose `dist` returns the term's slice of `b` as a
deterministic named after the term (so `predict()`, the posterior and the fit record read the coefficients as for any
other prior). The first term Bambi builds creates the shared variables in the graph: `zb` and the scales on the
coordinate `zb_dim` (the design columns), under brms's Stan names, `hs_global`, `hs_slab`, `hs_local` on their logs, or
`R2D2_R2` on its log-odds (Stan's logit) and `R2D2_phi` on Stan's stick-breaking coordinates (`monotonic.StickBreakingTransform`,
B03), and the deterministics `sdb` and `R2D2_tau2`. Bambi builds `sigma` before the linear predictor's terms, so the
prior can be scaled by it. `Model.shrinkage_names` lists the names; `check_graph` verifies them and `b = zb * sdb`
column by column when the model is built. `Model.value_point` takes `shrinkage={"zb": ..., "hs_global": ...,
"hs_slab": ..., "hs_local": ...}` (or `{"zb": ..., "R2D2_R2": ..., "R2D2_phi": ...}`) and no `b`.

`describe()` writes the prior out, `prior_table()` puts brms's spelling on the class row of `b` with the coefficient rows
empty, as brms's prior summary does, the target identity records the prior and its arguments, and `prior_predictive()`
draws from it. `ab.default_priors()` is unchanged: brms's default for `b` is flat.

## Verification

Against the frozen cases `gauss_horseshoe` (`horseshoe(df = 3, scale_global = 0.5, df_global = 1, scale_slab = 2,
df_slab = 4)`) and `gauss_r2d2` (`R2D2(mean_R2 = 0.4, prec_R2 = 3, cons_D2 = 0.7)`), Gaussian, `y ~ x + w + z + xpos`:
G1 also compares brms's `Kscales` and the prior's data (`hs_df`, ..., `R2D2_cons_D2`); G2's coordinates are `zb`,
`hs_global`, `hs_slab`, `hs_local`, `R2D2_R2` and the four stick-breaking coordinates of `R2D2_phi`, which the probe does
not reach one by one and the adapter recomputes from `R2D2_phi` and checks against brms's unconstrained values at every
point; G3 compares the graph's `b`, `sdb` and `R2D2_tau2` with brms's. The log density is brms's within 1.3e-10 relative
on `gauss_horseshoe` (PyMC's half-Student-t computes its `log 2` in float32, 1.9e-9 per variable, divergence D103) and
within 7e-16 on `gauss_r2d2`; the gradient within 6.3e-15. `tests/test_shrinkage_priors.py` checks the density against
one written out with SciPy from the definitions above.

## What is refused

Only what the frozen cases verify is offered (divergences D100-D102); everything else is refused naming the construct:

- the prior on anything but the whole class `b` of the response formula: a single term or coefficient, `sd`, `sigma`, the
  intercept, a distributional formula's `b`, a mixture's component, a response of a multivariate model, a non-linear
  parameter; brms's `main = TRUE` (D100);
- brms's `par_ratio` and `autoscale = FALSE` (D101);
- every family but the Gaussian; a formula without population-level terms or without an intercept;
  `center_predictors=False`; group-level terms, an offset, addition terms, distributional formulas, `mo()`, `me()` and
  `mi()` (brms's class `bsp`, which brms's shrinkage would include), `gp()` and autocorrelation terms; `R2D2()` with a
  single coefficient and a `cons_D2` vector of another length (D102).
