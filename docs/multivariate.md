# Multivariate Gaussian models, `mvbind(y, y2) ~ ...` with or without residual correlation

brms-parity feature C01. brms's `bf(mvbind(y, y2) ~ x + (1 | g)) + set_rescor(TRUE | FALSE)` models several Gaussian
responses of the same rows with one right-hand side: each response has its own population-level coefficients, residual
SD and group-level effects, and with `set_rescor(TRUE)` each row's residuals are correlated across the responses.

## The model

For row `i` and responses `y` and `y2` (any number `K >= 2` of them):

    mu_y,i  = a_y  + X_i b_y  + u_y[g_i]         (y's own univariate linear predictor)
    mu_y2,i = a_y2 + X_i b_y2 + u_y2[g_i]        (y2's, the same design X and grouping)
    rescor = TRUE:   (y_i, y2_i) ~ MultiNormal((mu_y,i, mu_y2,i), diag(sigma) Rescor diag(sigma)),  sigma = (sigma_y, sigma_y2)
                     Rescor = Lrescor Lrescor^T,  Lrescor ~ lkj_corr_cholesky(eta)
    rescor = FALSE:  y_i ~ Normal(mu_y,i, sigma_y),  y2_i ~ Normal(mu_y2,i, sigma_y2), independently

brms's names: one sub-model per response (`Intercept_y`, `b_y`, `sigma_y`, `X_y`, `Y_y`; the same for `y2`), the group
terms separate per response (`(1 | g)` is group 1 for `y`, `sd_1`/`z_1` with `J_1_y`, `Z_1_y_1`, and group 2 for `y2`;
uncorrelated across responses because there is no shared ID), and with rescor `Lrescor`, a `cholesky_factor_corr[nresp]`
under `lkj_corr_cholesky(eta)`, in `multi_normal_cholesky(Y_n | Mu_n, diag_pre_multiply(sigma, Lrescor))`, with the
generated `Rescor` and the scalars `rescor`. `Rescor[y, y2]` is the correlation of the two residuals at the same predictors
and group effects, not of the responses themselves.

## How it is built

`ab.gaussian("mvbind(y, y2) ~ x + (1 | g)", data, priors, rescor=True)` returns an `ab.MultivariateModel`
(`autobayes/multivariate.py`). It builds one univariate Gaussian `Model` per response on the same data,
`model.submodels["y"]` with the formula `y ~ x + (1 | g)` (its own design, group terms, priors, Bambi graph and checks,
exactly the univariate model), and assembles their PyMC graphs into one through `pymc.model.fgraph`: every variable is
renamed after its response (`variable(response, name)`: `y_Intercept`, `y_x`, `y_sigma`, `y_1|g`, `y_1|g_sigma`,
`y_1|g_offset`, `y_mu`; value variables alike, `y_sigma_log__`), the coordinates (`__obs__`, `g__factor_dim`, ...) are
checked equal across the responses and shared, and the model is rebuilt. With `rescor=True` the responses' Normal
likelihoods are dropped and replaced by the joint observed variable `mvbind(y, y2)` (dims `__obs__` x `__resp__`), whose
density is Stan's `multi_normal_cholesky_lpdf` written out (`multi_normal_cholesky_logp`: the residual solved forward
through `diag(sigma) Lrescor`); `Lrescor` is A02's correlation Cholesky factor (`autobayes.correlation.cholesky_corr_rv`:
Stan's `lkj_corr_cholesky_lpdf` with its constant, sampled on Stan's `cholesky_corr` coordinates, `Lrescor_cholesky_corr__`)
and `rescor = Lrescor Lrescor^T` is a deterministic (brms's `Rescor`), both on `__resp__` x `__resp_T__`. With
`rescor=False` each response keeps its own likelihood, the observed variables `y` and `y2`. `MultivariateModel` then checks
the graph: every response's variables under their new names with their transforms and shapes, the observed variables, and
`Lrescor`'s coordinates and dims.

`MultivariateSpec` (`model.spec`) is the parsed formula: the responses in order, the shared right-hand side and each
response's `ModelSpec` (`specs`), the per-response sub-models over one data frame being the extension point for a formula
per response (C06's `bf(y ~ mi(x)) + bf(x | mi() ~ w)`).

## Using it

- Formula: `mvbind(y1, y2, ...) ~ <terms>`, two or more distinct response columns; the right-hand side takes what the
  univariate Gaussian model takes (numeric and declared categorical predictors, interactions, transforms, an offset,
  `(1 | g)`, `(1 + x || g)`, `(1 + x | g)`, `g:h`, `g/h`), the same for every response. `rescor=True` or `rescor=False`
  must be given (D58). `center_predictors=` and `categorical=` apply to every response.
- Priors by response, as brms's `resp =`: `ab.Priors(resp={"y": {"intercept": ab.Normal(0, 5), "b": ab.Normal(0, 2),
  "sd": ab.HalfNormal(1), "sigma": ab.HalfNormal(1)}, "y2": {...}}, rescor=ab.LKJ(2))`. Each response's entry is a
  mapping of the univariate classes or an `ab.Priors`, validated against that response's formula; `rescor` (brms class
  `rescor`, `lkj(eta)`) is required exactly with `rescor=True`; a correlated term's `cor={"g": ab.LKJ(eta)}` sits at the top
  level and holds for every response (brms's `cor` prior has no `resp`); nothing else may be given at the top level.
  `ab.default_priors("mvbind(y, y2) ~ ...", data, rescor=True)` returns brms's defaults in that shape: each response's
  own (its Intercept location and scale read from that response), `lkj(1)` on `cor` and on `rescor`.
- `model.value_point(resp={"y": {...}, "y2": {...}}, Lrescor=array(K, K))`: each response's arguments of its univariate
  `Model.value_point`, and the Cholesky factor of the residual correlation matrix (`rescor=True` only).
- Names: `model.response_names["y"]` maps the univariate roles to the graph's names (`"intercept": "y_Intercept"`,
  `"linpred": "y_mu"`, `"sd": "y_1|g_sigma"`, ...), `model.names` has them as `"y:intercept"` beside `"Lrescor"`, `"rescor"`
  and the joint observed `"y"`; `model.observed_names` lists the observed variables.
- `fit()`: no one Bambi model holds the assembled graph, so `ab.fit` samples it with PyMC's `pm.sample` (`nuts_sampler`
  `"pymc"` or `"numpyro"`, `MultivariateModel.sample_posterior`) and cleans the posterior as Bambi does
  (`clean_posterior`: the standardised effects `*_offset` dropped, each centred intercept moved to the raw scale, brms's
  `b_y_Intercept`).
- `predict()` (and `ab.predict`): each response's linear predictor is computed by its univariate model's prediction code
  from its own draws, with the same unseen-level policies, and the responses are stacked on a last `response` dimension in
  the order written, brms's prediction arrays: `kind="linear"` returns `eta`, `kind="mean"` `mu`
  (`posterior_linpred`, `posterior_epred`), `kind="response"` also `y`, each row's responses drawn jointly as
  `mu + diag(sigma) Lrescor z` (independent normals without rescor). One seeded random stream serves every response, so
  each response's unseen levels draw their own effects.
- `describe()` writes the joint likelihood or the independent ones, the residual correlation and its prior, then each
  response's own description and the assembled graph's value variables; `prior_table()` writes brms's table, each
  response's rows under its `resp`, the shared `cor` rows and `rescor`.

## Refused, with the construct named

A formula per response (`["y ~ x", "y2 ~ z"]`, brms's `bf(y ~ ...) + bf(y2 ~ ...)`), a distributional formula in a
multivariate model, addition terms after `mvbind()`, `mi()` (feature C06) and `mo()` (D56); correlated group effects across
responses `(1 |ID| g)` (D57); families other than the Gaussian and a missing `rescor` (D58); a response that is also a
predictor, grouping factor or offset; fewer than two or repeated responses; a top-level class other than `resp`, `cor` and
`rescor`, a response-specific `cor`, a `rescor` prior without `rescor=True`; `resp` or `rescor` for a model with one
response. Names and layout differ from brms's (D59). Not part of this slice: `prior_predictive()` and `fit_record()` of a
multivariate model (each refuses with the alternative).

## Evidence

Oracle cases `gauss_mv_rescor` and `gauss_mv_norescor` (the brief's `mvbind(y, y2) ~ x + (1 | g)` with and without
`set_rescor`) and `gauss_mv_slopes` (frozen with this feature: a categorical predictor and its interaction, an offset,
correlated slopes `(1 + x | g)` per response under one shared `lkj(2)`, a second factor `(1 | h)`, different priors per
response, `Lrescor` under `lkj(3)`) pass G1-G6 against brms 2.23: each response's design, response, offsets and group
coding (`adapter.ResponseView` shows the fixture as each response's univariate model), the joint log density and every
gradient coordinate with and without the Jacobian (`Lrescor`'s on Stan's `cholesky_corr` coordinates, recomputed and checked
at every point), each response's linear predictor, group effects, correlation matrices and raw-scale intercept and
`Rescor`, the draws x rows x responses predictions on fitted and new rows with and without group effects, the JAX
density, and brms's default-prior table; `tests/oracle/test_multivariate_harness.py` shows each of those multivariate paths
failing on a corrupted fixture. `tests/test_multivariate.py` covers the grammar and its refusals, the priors' coverage, the
assembled graph against the responses' own graphs and against scipy's multivariate normal with the closed-form LKJ density
and Stan's Jacobian, value points, predictions (joint draws, seeds, unseen levels), the fit path's sampling call and
cleaning with `pm.sample` replaced, brms's default priors, `describe()` and the target identity. Nothing samples.
