# Missing values `mi()` and measurement error `me()`

brms-parity feature C06 (brms's `mi()` and `me()`; `?mi`, `?me` and the missing-data vignette). Built on the multivariate
models of C01 (docs/multivariate.md).

```python
import autobayes as ab
# brms: bf(y ~ mi(xm) + w) + bf(xm | mi() ~ w) + set_rescor(FALSE)
priors = ab.Priors(resp={"y": {"intercept": ab.Normal(0, 5), "b": ab.Normal(0, 2), "sigma": ab.HalfNormal(1)},
                         "xm": {"intercept": ab.Normal(0, 5), "b": ab.Normal(0, 2), "sigma": ab.HalfNormal(1)}})
model = ab.gaussian(["y ~ mi(xm) + w", "xm | mi() ~ w"], data, priors, rescor=False)

# brms: y ~ me(xobs, sdx)
priors = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1),
                   meanme=ab.Normal(0, 2), sdme=ab.HalfNormal(1))
model = ab.gaussian("y ~ me(xobs, sdx)", data, priors)
print(model.describe())
```

## The models

**`mi(col)`.** A column with missing values (NaN) is the response of a formula of its own, `col | mi() ~ <terms>`, and a
predictor of another formula of the same multivariate model through `mi(col)`. Its missing values are parameters, brms's
`Ymi_col` under a flat prior; `Yl_col` is the column with them in the missing rows. For the case above, row `i`:

    xm's formula:  Yl_xm,i ~ Normal(a_xm + (w_i - mean(w)) b_xm, sigma_xm)       (every row, observed or imputed)
    y's formula:   y_i     ~ Normal(a_y + (w_i - mean(w)) b_w + b_mi(xm) Yl_xm,i, sigma_y)

The coefficient of `mi(xm)` is brms's `bsp_y` (class `b`, coefficient `mixm`); the term is not centred. The missing values
are informed by both formulas.

**`me(x, sdx)`.** A predictor `x` observed with known standard deviations `sdx` has latent values

    Xme_i = meanme + sdme * zme_i,   zme_i ~ Normal(0, 1),   x_i ~ Normal(Xme_i, sdx_i),   mu_i = ... + b_me Xme_i

with `meanme` and `sdme` under their own priors (brms's classes `meanme` and `sdme`) and the coefficient brms's `bsp` (class
`b`, coefficient `mexsdx`); the term is not centred.

## How it is built

Both terms are special terms like `mo()` (B03): a `PopTerm` of kind `"missing"` or `"measurement_error"` with no design
column, which Bambi never sees; the model adds them to the parent's linear predictor while Bambi builds the graph (the
family's `transform_backend_eta` hook, `likelihoods.monotonic_family`). For `me(x, sdx)` it creates `meanme_x`, `sdme_x`
(sampled as `sdme_x_log__`), `zme_x` (one per row), the deterministic `Xme_x`, the observed `Xn_x ~ Normal(Xme_x, sdx)` and the
coefficient `me(x, sdx)`. For `mi(col)` it creates the flat `Ymi_col` on the coordinate `Ymi_col_dim` (the missing rows,
0-based; brms's `Jmi` is 1-based) and the coefficient `mi(col)`, and adds `mi(col) * Yl_col`. A response with `| mi()` is handed
to Bambi with a placeholder at its missing rows, and its likelihood is a `CustomDist` that evaluates the family's density at
the completed response (`likelihoods.missing_response_family`; the placeholder is never read).

Each response's model holds the missing values it reads (`Model.missing_names`); `MultivariateModel` assembles the responses'
graphs as C01 does and keeps one variable `Ymi_col` per column, by brms's name, shared by every response that holds it
(`MultivariateModel.missing_names`, `value_name`). After the build the model checks the graph (`Model._measurement_design`,
`_missing_design`, `_check_measurement_effects`): the variables free, scalar or on their coordinates, the scale on the log
scale, the missing values flat, the noisy values observed, and the parent's linear predictor moving by exactly `c` times each
term's values when its coefficient goes from 0 to `c`.

## Using it

- Formulas. `mi()`: a list of formulas, one per response, `["y ~ mi(xm) + w", "xm | mi() ~ w"]`, with `rescor=False` and the
  Gaussian family; each `mi(col)` needs `col | mi() ~ ...` in the list; several `mi()` terms in a formula and imputations that
  read other imputed columns (`x2 | mi() ~ mi(x1) + w`) are taken, a cycle is refused. Responses may have their own
  population- and group-level terms. A list of formulas with `mi()` is recognised as a multivariate model
  (`multivariate.is_multivariate_formula`); `MultivariateSpec.separate`, `imputed` and `canonical_formula` (brms's
  `bf(...) + bf(...)`) describe it. `me()`: one `me(x, sdx)` population-level term of a univariate Gaussian formula, next to any
  other term kind but `mo()` and `mi()`.
- Data. A missing value is NaN (or pandas's NA) in a numeric column; the other rows must be finite and at least one observed.
  `me()`'s noisy values must be finite and its standard deviations positive.
- Priors. `mi()`: the coefficient is class `b` of its response (`resp={"y": {"b": ...}}`, a single prior or `{"mi(xm)": ...}`);
  the missing values have no prior of their own. `me()`: the coefficient is class `b` (`{"me(xobs, sdx)": ...}`), `meanme` a location
  prior and `sdme` a positive-support prior or `ab.Flat()`, each bare or keyed by the noisy column (`meanme={"xobs": ...}`).
  `ab.default_priors()` returns brms's defaults: a response with missing values gets brms's location 0 and scale 2.5 (its median
  and MAD are NA in R), the coefficients are flat, `meanme` and `sdme` flat (`sdme` on the positive half line).
- Points. `MultivariateModel.value_point(resp={...}, Ymi={"xm": array(Nmi)})`, each response's arguments with `bsp={"xm": ...}`
  for its `mi()` terms; `Model.value_point(..., bsp={"xobs": ...}, meanme=..., sdme=..., zme=array(n))`.
- Names. `model.missing_names["xm"]` (`values`, `coordinate`, `response`); `model.me_names["xobs"]` (`b`, `mean`, `sd`, `z`,
  `latent`, `noisy`), also in `model.names` as `meanme:xobs`, `sdme:xobs`, `zme:xobs`, `Xme:xobs`, `Xn:xobs`, `Ymi:xm`.
- `predict()`. On the fitted rows each term adds its coefficient times the draws' completed column (brms's `Yl`) or latent
  values (`Xme`), as brms's `posterior_linpred` does; on new rows an `mi()` column must be observed, and new rows of a model with
  `me()` are refused (brms draws their latent values; D74).
- `describe()` writes the terms, the latent values and the missing rows out; the target identity holds the noisy values, the
  noise, the priors of the latent values and each column's observed values and missing rows; `prior_table()` writes brms's
  `meanme` and `sdme` rows and the `mixm`, `mexsdx` coefficients; `prior_predictive()` draws a model with `me()` under proper
  priors (and refuses flat ones, as for every model); fit records list the `meanme` and `sdme` priors.

## Refused, naming the construct

`mi()`: without its column's `| mi()` formula (brms refuses it too), with `rescor=True`, in interactions, group-level and
distributional formulas, with brms's `idx` argument, `mi(sdy)` on a response, other families, a cycle of imputations, a
univariate `y | mi() ~ ...`, `mi()` with `weights()`, `cens()` or `trunc()`, with `mo()` or `me()` in one formula, inside an
`mvbind()` formula, and a response read by another formula other than through `mi()` (D72). `me()`: with brms's `gr` argument,
several `me()` terms (brms correlates their latent values by default), in interactions, group-level and distributional formulas,
in multivariate models, other families, expressions inside `me()`, missing noisy values and non-positive noise (D73).
`predict(newdata=...)` of a model with `me()`, missing values of an `mi()` column on new rows, and `log_lik()`, `loo()` and
`bayes_R2()` of models with `me()` or `mi()` (D74). Names and layout differ from brms's (D75).

## With the other features

No frozen case verifies `me()` or `mi()` next to another Tier C feature, so each pair is refused, naming the term:

- Non-linear models (C02, `nl=True`): `me()` and `mi()` in the expression (brms's terms, not functions of it) or in a
  parameter's formula, `mi()` on the response or a parameter's left-hand side, and the `meanme` and `sdme` priors.
- Gaussian processes (C03) and autocorrelation terms (C04): `me()`, `mi()` and the response's `| mi()` in a formula with
  `gp()` or with `ar()`, `ma()`, `arma()` (C04 refuses an autocorrelation term next to any addition term, `| mi()` among them,
  first); in a list of formulas with `mi()`, `gp()` and autocorrelation terms are refused in every formula, as they are in an
  `mvbind()` formula.
- Mixtures (C05): `me()`, `mi()` and `| mi()` are not available for the mixture family, a mixture takes one formula (so no
  list with `mi()`), and `meanme` and `sdme` are refused at the top level of a mixture's priors, in a component's, and by its
  `value_point` together with `zme` and `Ymi`.
- Priors. Each model refuses the fields of the features it does not have: an `me()` model `components`, `theta`, `resp`,
  `rescor`, `sdgp`, `lscale`, `nlpars`, `ar` and `ma`; the multivariate `mi()` model the same at the top level (`sdgp` and
  `lscale` there included, which a multivariate model used to take without a word) and in each response's priors; a model
  without `me()` (a mixture, a non-linear model, a model with a Gaussian process or an autocorrelation term, a plain one)
  `meanme` and `sdme`.
- Sampling and post-processing. `ab.fit()` draws the multivariate `mi()` model through its own `sample_posterior` and the
  `me()` model through Bambi (`inference._posterior_sampler`); `log_lik()`, `loo()` and `bayes_R2()` (C07) refuse both (D74).

## Evidence

The frozen cases `gauss_mi` (`bf(y ~ mi(xm) + w) + bf(xm | mi() ~ w) + set_rescor(FALSE)`, 7 of 36 `xm` values missing) and
`gauss_me` (`y ~ me(xobs, sdx)`) pass G1-G6 against brms 2.23.0: G1 brms's `Y_xm` with Inf at the missing rows, `Jmi_xm`,
`Nmi_xm` and the open bounds `lbmi_xm`, `ubmi_xm`, `Ksp`, `Xn_1`, `noise_1`, `Mme_1`; G2 the log density with and without the
Jacobian, its prior part, and every gradient coordinate, `Ymi_xm`, `bsp`, `meanme_1`, `sdme_1` and `zme_1` included; G3 each
response's linear predictor (y's with the imputed values) and the latent values `Xme_1`; G4 the draws x rows x responses
predictions on fitted and new rows (`gauss_mi`) and on fitted rows (`gauss_me`); G5 the JAX density; G6 brms's default-prior
tables. `tests/oracle/test_missing_measurement_harness.py` shows each of those paths failing on a corrupted fixture;
`tests/test_missing_measurement.py` covers the grammar and its refusals, the priors, the densities written out with scipy
(including several `mi()` terms and a chained imputation), points, predictions, defaults, `describe()` and the target
identity. Nothing samples.
