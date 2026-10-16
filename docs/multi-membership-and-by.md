# Multi-membership, `gr(by = )` and correlation IDs shared with `sigma`

brms-parity feature D08 (`D08_cs_mm_gr`). Three of brms's group-level constructions, each accepted against a frozen brms
fixture; brms's category-specific `cs()`, the fourth construction of the feature, is parked (below).

## `(1 | mm(g, g2, weights = cbind(w1, w2)))`: multi-membership

Each row belongs to one level of each member column, with a weight per member:

    mu_i = ... + W_1,i * u[g_i] + W_2,i * u[g2_i],      u[j] = sd * z[j],   z[j] ~ Normal(0, 1)

The levels are the union of the members' labels, sorted (one varying intercept per level, whichever column it appears in).
The weights are the weight columns divided by their sum on each row, brms's default `scale = TRUE`; without `weights` every
member has weight `1 / members` (both checked against brms 2.23's `make_standata`). The term is brms's group `mmgg2` ("mm"
and the member columns): its factor here is `"mmgg2"`, so the priors are `sd={"mmgg2": ...}` (a single factor also takes the
bare prior), the PyMC variables `1|mmgg2_sigma`, `1|mmgg2_offset` and the effects `1|mmgg2` on `mmgg2__factor_dim`.
`Design.group_blocks[("mmgg2", "Intercept")]` holds the weighted indicator matrix `Z` and, as `memberships` and `weights`, each
member's level index and weight per row (brms's `J_m_k - 1` and `W_m_k`).

`predict()` adds every member's effect times its weight; `newdata` needs the member and weight columns. An unseen label
follows `new_groups` as an unseen level of any factor (`"population"`: a fresh `Normal(0, sd)` effect per label and draw;
`"resample_observed"`: an observed level's effect), and rows sharing the label share it, whichever member it appears in.

## `(1 | gr(g, by = gby))`: one SD per level of `gby`

    u[j] = sd[gby of level j] * z[j]

Each level of `g` must lie within one level of `gby` (brms refuses the term otherwise, and so does this library). The by
levels are sorted; the SD `1|g_sigma` is a vector on `g__by_dim` under one `sd` prior for every by level (brms has no prior per
by level); `value_point(sd={"g": [sd_u, sd_v]})`. brms's prior density of `sd_m` (`M_m` by `Nby_m`, `M_m = 1` here) subtracts the
lower bound's log CCDF once for the term rather than once per by level (checked on the fixture's `lprior`), so for a prior symmetric about 0 (`HalfNormal`, `HalfStudentT`,
`HalfCauchy`) brms's log density is `log 2` below the product of half densities for every by level after the first; the
model reproduces that constant as the potential `1|g_sigma_by_normalisation` (nothing moves, the log density is brms's).
`gr(g)` alone is `g`. On new rows a fitted level keeps its fitted by level (a row that contradicts it is refused) and an
unseen level is refused.

## `["y ~ x + (1 + x |p| g)", "sigma ~ 1 + (1 |p| g)"]`: an ID shared with sigma's formula

brms's `bf(y ~ x + (1 + x |p| g), sigma ~ (1 |p| g))` puts the response formula's intercept and slope and log sigma's
varying intercept in one group with one correlation matrix: `(u, v_x, u_sigma) = diag(sd) L z` per level. Bambi builds the
three coefficients (`1|g`, `x|g` and `sigma_1|g` with their SDs and offsets) and the model correlates them after the build, as
for `(1 + x | g)` (docs/correlated-slopes.md), with `L|g` and `cor|g` on the coordinates `Intercept`, `x`, `sigma_Intercept`
(brms's names; `model.correlation_coefs("g")`). The priors are `sd={"g": {"Intercept": ..., "x": ...}}`, `cor={"g":
ab.LKJ(eta)}` and the distributional formula's `sigma={"intercept": ..., "sd": ...}` (brms's class `sd` with `dpar = sigma`,
whose default is the formula's own `student_t(3, 0, scale)`). `value_point` takes the formula's `sd` and `z` in `sigma=` too;
`model.dpar_group_names` names its variables. `predict(dpar="sigma")`, `kind="response"` and `ab.log_lik` add the level's
effect to log sigma; `include_group_effects=False` drops it with the others. An ID needs both formulas: the distributional
formula's term is the varying intercept of the same factor, written with `1 +` (`sigma ~ 1 + (1 |p| g)`; brms's
`sigma ~ (1 |p| g)` reads the same). This lifts divergence D32 for this construction only.

## Refused, with the construct named

Varying slopes in `mm()` and with `gr(by = )`, `mm()`'s `scale = FALSE` and `mmc()`, negative weights or rows whose weights sum
to 0, `gr()`'s `cor`, `id`, `cov` and `dist` (D104, D105); `mm()` and `gr(by = )` next to `mo()`, `gp()`, `me()`, `mi()`, an
autocorrelation term, an addition term or a distributional formula, in multivariate and non-linear models, and for the families
whose link is not identity or log (D104, D105); new levels of a `gr(by = )` term or of a factor whose ID is shared; an ID in a
formula on its own, used by two factors, written `||p|` (not brms syntax), with `mm()` or `gr()`, with varying slopes in the
distributional formula, and group-level terms of a distributional formula without one (D32, D106); `cs()` came with D04 (D107).

## `cs()`: parked here, built by D04

The frozen case `acat_cs` (`ord ~ w + cs(x) + (1 | g)`, `acat()`) needed brms's adjacent-category family, which is D04's:
unordered thresholds, one linear predictor per threshold (`posterior_linpred` is draws x rows x thresholds), its likelihood
and its default-prior rows, on top of which `cs()` adds a `Kcs x nthres` coefficient matrix `bcs`. This slice parked the case
and refused `cs()` naming it (D107); D04 built both, and the case passes (docs/ordinal-categorical.md, D89).

## Evidence

The oracle cases `gauss_mm`, `gauss_gr_by` and `gauss_shared_id` pass gates G1-G6 against frozen brms 2.23 fixtures: the
memberships, weights, by levels and the shared group's coefficients by content; the log density with and without the Jacobian
and every gradient coordinate (the shared ID's three-coefficient `cholesky_corr` coordinates recomputed and checked against
brms's); the effects per level and the correlation matrix; `posterior_linpred` / `posterior_epred` on fitted and new rows with
and without group effects; the JAX density; the default-prior table, rows with `dpar = sigma` included.
`tests/test_mm_by_shared_id.py` covers the term model, the refusals, the blocks, the log density against a hand computation
(a Poisson model with an `mm()` term next to `(1 | h)` too), brms's normalisation constant, predictions by hand under both
`new_groups` policies, the shared ID's sigma, the default priors, `describe()` and the target identity. Nothing samples.
