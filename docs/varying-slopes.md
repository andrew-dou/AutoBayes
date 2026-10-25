# Independent varying slopes, `(1 + x || g)`

brms-parity feature A01. A group term `(1 + x || g)` lets the slope of the numeric population-level term `x`
vary by level of `g` next to the varying intercept, with no correlation between the two (brms's `||`). Several
slopes are allowed in one term, `(1 + x + w || g)`; each is independent of the others and of the intercept.
brms's correlated form `(1 + x | g)` is feature A02 (docs/correlated-slopes.md).

## The model

For row `i` in level `j = g_i`, with `x` raw (never centred, exactly as brms's `Z_m_c`):

    eta_i = a + X_i b + u_g[j] + x_i * v_x|g[j]
    u_g[j]   = sd_g   * z_g[j],     z_g[j]   ~ Normal(0, 1)
    v_x|g[j] = sd_x|g * z_x|g[j],   z_x|g[j] ~ Normal(0, 1)

brms's names: one group id `m` for `g` with `M_m = 2` coefficients (`Intercept`, `x`), `sd_m` with one entry per
coefficient, `z_m` an `(M_m, N_m)` array, `r_m_c = sd_m[c] * z_m[c]`, `Z_m_1` a column of ones and `Z_m_2` the
`x` column. The library's: `GroupTerm("g", coefs=("Intercept", "x"), correlated=False)`,
`Design.group_blocks[("g", "x")]` whose `Z` is the level indicator times `x`, Bambi/PyMC variables `x|g`,
`x|g_sigma`, `x|g_offset` on the coordinate `g__factor_dim` shared with `1|g` (docs/term-model.md).

Independence of `u_g` and `v_x|g` is a statement at `x = 0`: shifting `x` by a constant `c` turns the intercept
deviation into `u_g + c v_x|g`, which is correlated with `v_x|g`. `describe()` says so.

## Using it

- Formula: `y ~ x + (1 + x || g)`. Every slope must also be a population-level term of the formula (brms does
  not require this; divergence D8). `(1 || g)` is accepted as `(1 | g)`: one coefficient has nothing to
  correlate with. Bambi has no `||`, so it is handed `(1 + x | g)`, which it builds as the independent terms
  `1|g` and `x|g`; `Model._extract_design` checks each block against the data and raises on a mismatch.
- Priors: `sd` stays keyed by factor; a factor with slopes takes a mapping over exactly its coefficients,
  `sd={"g": {"Intercept": HalfNormal(1), "x": HalfNormal(0.5)}}` (brms `class = sd, group = g, coef = x`). An
  intercept-only factor keeps accepting a bare prior or `{"Intercept": prior}`. A single-factor model still names
  the factor; the bare `{"Intercept": ..., "x": ...}` is refused with a hint. `Priors.sd_priors(spec)` returns one
  prior per `(factor, coefficient)`; `group_priors` keeps the per-factor varying-intercept view.
- `Model.value_point`: `sd={"g": {"Intercept": 0.8, "x": 0.3}}`, `z={"g": {"Intercept": array(J), "x": array(J)}}`.
- Names: `model.group_coefficient_names[("g", "x")]` gives the slope's `sd`, `offset`, `group_effect` and
  `coordinate`; `model.group_names["g"]` lists them as `sd:x`, `offset:x`, `group_effect:x` next to the
  intercept's roles, and so does `model.names` for a single factor.
- `predict()`: each slope adds its effect at the row's level times the row's `x`. `include_group_effects=False`
  drops every group effect. For unseen levels, `new_groups="population"` draws every coefficient of the level from
  its own `Normal(0, sd)`; `new_groups="resample_observed"` reuses one observed level's whole effect vector
  (intercept and slopes from the same level, as brms's `sample_new_levels = "uncertainty"`).

As in brms, the intercept is implied (F08, docs/syntax-fixes.md): `(x || g)` is `(1 + x || g)` and `(x | g)` is `(1 + x | g)`,
and `gr(g, cor = FALSE)` is `||`.

## Refused, with the construct named

`(0 + x || g)` and `(x - 1 || g)` (a slope without the varying intercept; next to `(1 | g)` the refusal gives brms's reading,
`(1 + x || g)`, F08); a slope that is not a numeric population-level term; a
categorical or interaction slope; a slope equal to the factor, the response or the trials column; a factor in two
group terms; function calls inside the group term.

## Evidence

Oracle cases `gauss_slope_independent`, `gauss_slope_two_groups` (a slope factor next to an intercept-only
factor), `bernoulli_slope_categorical`, `poisson_slope_independent` and `gauss_slope_two_slopes` pass gates G1-G6
against frozen brms fixtures: design, log density and gradient with and without the Jacobian, `r_m_c` per
coefficient, `posterior_linpred`/`posterior_epred` on fitted and new rows with and without group effects, the
JAX density. `tests/test_varying_slope.py` covers the refusals, the term model and blocks, prior coverage,
`value_point`, by-hand predictions under both `new_groups` policies and `describe()`. Nothing samples.
