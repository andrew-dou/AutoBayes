# Correlated varying intercept and slopes, `(1 + x | g)` with an LKJ prior

brms-parity feature A02. A group term `(1 + x | g)` lets the slope of the numeric population-level term `x` vary by
level of `g` next to the varying intercept, with the two deviations of a level correlated (brms's `|`, its default;
the independent form `(1 + x || g)` is A01, docs/varying-slopes.md). Several slopes are allowed in one term,
`(1 + x + w | g)`: all its coefficients share one correlation matrix. The prior on that matrix is brms's `lkj(eta)`.

## The model

For row `i` in level `j = g_i`, with `x` raw (never centred, exactly as brms's `Z_m_c`) and `M` coefficients
(`Intercept` first, then the slopes in formula order):

    eta_i = a + X_i b + u_g[j] + x_i * v_x|g[j]
    (u_g[j], v_x|g[j]) = diag(sd_g, sd_x|g) L_g z[:, j],   z[c, j] ~ Normal(0, 1)
    Cor_g = L_g L_g^T ~ LKJ(eta)

brms's names: `sd_1` (one entry per coefficient), `z_1` a `matrix[M_1, N_1]` of standard normals, `L_1` a
`cholesky_factor_corr[M_1]` with `lkj_corr_cholesky_lpdf(L_1 | eta)`, `r_1 = scale_r_cor(z_1, sd_1, L_1) =
(diag_pre_multiply(sd_1, L_1) * z_1)'`, `r_1_1 = r_1[, 1]`, `r_1_2 = r_1[, 2]`, and the generated `Cor_1`. The first
row of `L_1` is `(1, 0, ...)`, so the intercept's effect is `sd_1[1] z_1[1, ]` as in the independent model; the slope's
is `sd_1[2] (L_1[2, 1] z_1[1, ] + L_1[2, 2] z_1[2, ])`.

The library's: `GroupTerm("g", coefs=("Intercept", "x"), correlated=True)` (`term.has_correlation`),
`ModelSpec.correlated_terms`, the same `Design.group_blocks` as the independent model, and the PyMC variables
`1|g`, `1|g_sigma`, `1|g_offset`, `x|g`, `x|g_sigma`, `x|g_offset` (as A01) plus `L|g` (brms `L_1`) and the
deterministic `cor|g = L|g L|g^T` (brms `Cor_1`), both with dims `g__cor_dim` x `g__cor_dim_T` whose coordinates are
the coefficient labels. `model.correlation_names["g"]` maps `L`, `cor`, `coordinate`, `coordinate_T` to those names;
`model.group_names["g"]` lists `L` and `cor` next to the coefficient roles. brms reports the correlation as the
scalar `cor_g__Intercept__x`; here it is `cor|g` at (`Intercept`, `x`) (divergence D21, names and layout only).

### How it is built

Bambi has no LKJ path for group-specific terms, so `Model` builds the independent model through Bambi as before and
then edits the PyMC graph (`Model._correlate_group_effects`, through `pymc.model.fgraph`): a free variable `L|g` and
the deterministic `cor|g` are added, and each slope's deterministic `x|g = x|g_sigma * x|g_offset` is replaced by
`x|g_sigma * (L|g[1, 0] * 1|g_offset + L|g[1, 1] * x|g_offset)` everywhere it is used, the linear predictor included.
`Model._extract_design` then verifies the blocks, names, dims and coordinates of the rebuilt graph, and
`model.bambi_model.backend.model` is the rebuilt graph, so `fit()`, `prior_predictive()` and Bambi's `predict` see it.

`L|g` is `autobayes.correlation.cholesky_corr_rv`: a `pm.CustomDist` whose density is Stan's
`lkj_corr_cholesky_lpdf` with its normalising constant (`lkj_cholesky_logp`;
`-log c_M(eta) + sum_{d>=2} (M - d + 2 eta - 2) log L[d, d]`, with `c_M(eta)` the LKJ constant written with Beta
functions) and whose value transform is Stan's
`cholesky_corr` transform (`CholeskyCorrTransform`: `M (M - 1) / 2` canonical partial correlations through `tanh`,
read row by row below the diagonal, with Stan's Jacobian). The sampled coordinates are therefore brms's own, so the
Jacobian-adjusted density, the density without the Jacobian and every gradient coordinate are comparable with
`rstan::log_prob`; the value variable is `L|g_cholesky_corr__`. Forward draws (`prior_predictive`) use the C-vine
construction of Lewandowski, Kurowicka and Joe (Stan's `lkj_corr_cholesky_rng`).

## Using it

- Formula: `y ~ x + (1 + x | g)`; every slope must also be a numeric population-level term (divergence D8, as for
  `||`); `(1 + x | g:h)` and `(1 + x | g/h)` (= `(1 + x | g) + (1 + x | g:h)`) work with the derived factors of A03.
- Priors: `sd` as for A01, per coefficient; `cor={"g": ab.LKJ(2)}` (brms `prior(lkj(2), class = cor)`), one entry per
  correlated factor exactly (a single correlated factor also accepts the bare `ab.LKJ(2)`). `cor` is refused for an
  independent (`||`) or intercept-only term, for a factor the formula lacks, and when it is not an `LKJ` prior; a
  correlated term without `cor` is refused naming the prior to pass. `ab.LKJ(eta)` needs `eta > 0`; `eta = 1` is
  uniform over correlation matrices, larger `eta` concentrates on the identity (for two coefficients the correlation
  is `2 Beta(eta, eta) - 1`).
- `Model.value_point`: `sd` and `z` as for A01 and `L={"g": array(M, M)}`, the lower-triangular Cholesky factor of
  the correlation matrix (unit-norm rows); it is stored on Stan's coordinates as `L|g_cholesky_corr__`.
- `predict()`: the fitted levels' effects are the posterior's `1|g` and `x|g`, exactly as for the independent model.
  For unseen levels, `new_groups="population"` draws the level's whole effect vector from the implied multivariate
  normal (`diag(sd) L z` with the draw's `L|g`, the same random numbers as the independent case where `L` is the
  identity); `new_groups="resample_observed"` reuses one observed level's whole vector.
- `describe()` says "with correlated varying slope(s)", writes the construction, the correlation matrix and its
  meaning (a statement at `x = 0`: shifting a predictor by `c` turns `u_g` into `u_g + c v` and changes the
  correlation), the `Cor_g ~ LKJ(eta=2)` prior and the value-space role of `L|g_cholesky_corr__`. The fit record
  carries `"correlated": true`, `"cor_prior"` and `"correlation_names"` in the factor's entry and `"cor:g"` among the
  priors; the target identity includes the LKJ prior, so a correlated model is a different target from the
  independent one.

## Refused, with the construct named

Correlation IDs (`(1 + x |p| g)`, brms's sharing of a correlation matrix across terms) and brms's `gr()` options
(`(1 + x | gr(g, cor = FALSE))`; `||` is the independent form), divergence D20; `cor` for a term without a
correlation matrix (`||`, or one coefficient); a non-LKJ `cor` prior; everything A01 refuses about slopes (no varying
intercept, `(x | g)` being refused with the form `(1 + x | g)` to write; categorical or interaction slopes; a slope
that is not a population-level term).

## Evidence

Oracle case `gauss_slope_correlated` (`y ~ x + (1 + x | g)`, `prior(lkj(2), class = cor)`) passes gates G1-G6 against
its frozen brms fixture: G1 the design and group coding; G2 the log density with and without the Jacobian, `lprior`
(which includes the LKJ term) and every gradient coordinate, the `L_1` coordinate included after `coordinate_map`
recomputes Stan's `cholesky_corr` coordinate from `L_1` at every point and finds the fixture's `u`; G3 `r_1_1`,
`r_1_2`, `Cor_1` and the linear predictor; G4 `posterior_linpred`/`posterior_epred` on fitted and new rows with and
without group effects; G5 the JAX density; G6 brms's default-prior table, which has the `cor` class.
`tests/oracle/test_harness.py` shows that the new mappings are checked rather than passed through: a fixture whose
`L_1` coordinate, `L_1` gradient or `Cor_1` is perturbed fails, and an independent model is not matched against the
correlated fixture (nor a correlated model against the independent one). `tests/test_correlated_slopes.py` covers
the grammar, the prior coverage rules, the transform's round trip and Jacobian against finite differences, the LKJ
constant against closed forms, the density against a closed form at a point (two and three coefficients, and two
correlated factors `(1 + x | g/h)` with different `eta`), `value_point`, predictions under both policies, the fit
record, the target identity and `describe()`. Nothing samples.
