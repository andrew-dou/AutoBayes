# Summaries of a fit: `fixef`, `ranef`, `coef`, `VarCorr`, `posterior_summary`, `summary`, `residuals`

brms-parity feature F07 (brms's summaries of draws; divergence rows D162 to D165).

```python
import autobayes as ab
fit = ab.fit(model, chains=4, draws=1000, seed=1)
ab.fixef(model, fit)                 # DataFrame: one row per coefficient (Intercept, x, ...), Estimate, Est.Error, Q2.5, Q97.5
ab.ranef(model, fit)["g"]["x"]       # brms's ranef(fit)$g[, , "x"]: one row per level of g
ab.coef(model, fit)["g"]             # each level's coefficients, columns (coef, statistic)
ab.VarCorr(model, fit)["g"]["cor"]   # {"g": {"sd", "cor", "cov"}, "residual__": {"sd"}}
ab.posterior_summary(model, fit)     # every variable by brms's name, lprior and lp__ included
s = ab.summary(model, fit)           # s.fixed, s.spec_pars, s.cor_pars, s.random["g"]: with Rhat, Bulk_ESS, Tail_ESS
print(s)
ab.residuals(model, fit)             # residuals(type = "ordinary", method = "posterior_epred")
```

Each takes the model and its fit (a `FitResult` or its `InferenceData`, checked against the model as `predict()` checks it)
and summarises every posterior draw, the chains together. No function here samples.

## The draws under brms's names

`autobayes.summaries.brms_draws(model, fit)` names the draws as brms names the variables of its fit, in brms's order
(`as_draws_array()`): `b_Intercept` (the raw-scale intercept) and each distributional formula's `b_sigma_Intercept`, then
the coefficients `b_x`, `b_zq`, `b_x:zq`, ..., `b_sigma_x`; every group-level SD `sd_g__Intercept`, `sd_g__x`, factor by
factor in formula order; every correlation `cor_g__Intercept__x`, the upper triangle column by column (`Intercept__x`,
`Intercept__w`, `x__w`, `Intercept__cnt`, ...); `sigma`; the intercepts of the centred designs `Intercept` and
`Intercept_sigma` (the raw-scale intercept plus `mean(X) b`, brms's `Intercept`); the effects `r_g[a,Intercept]`, ...,
`r_g[f,x]`, coefficient by coefficient; `lprior` and `lp__`. A fit's posterior holds neither of the last two (a sampler
records `lp`, PyMC under `sample_stats`, and no sampler records the prior part), so both are the model's own log densities
at each draw (`summaries.log_densities`): the draw is mapped back to the sampled coordinates (the centred intercepts, the
standardised effects `z = L^-1 (r / sd)`, `r / sd` without a correlation; `Model.value_point`) and the graph's log density
with the Jacobian is `lp__`, what a sampler records; its log density of the priors (the intercept of the centred design, the
coefficients, the SDs, the correlation's Cholesky factor, sigma or its formula's intercept and coefficients; not the
standardised effects, which brms puts in its model block) without the Jacobian is `lprior`.

## The summaries

Every summary is the mean, the SD (divisor `S - 1`) and the 2.5 % and 97.5 % quantiles (R's type 7) of the draws, brms's
`posterior_summary()` (`summaries.summarise`).

- `posterior_summary`: one row per variable of `brms_draws`, the columns `Estimate`, `Est.Error`, `Q2.5`, `Q97.5`.
- `fixef`: the `b_` variables without the prefix (`Intercept`, `sigma_Intercept`, `x`, `sigma_x`).
- `ranef`: per grouping factor, a DataFrame with one row per level and the columns `(coef, statistic)`, brms's `[level,
  statistic, coef]` array; `ranef(model, fit)["g"]["x"]` is brms's `ranef(fit)$g[, , "x"]`.
- `coef`: laid out as `ranef`: the coefficient plus the level's effect for the factor's own coefficients, first and in the
  factor's order, then every other row of `fixef` alone, in its order (brms's order, probed: `Intercept, x, w, zq, zr, x:zq`
  for `y ~ x * z + w + (1 + x + w || g)`).
- `VarCorr`: per grouping factor `{"sd": DataFrame}` (one row per coefficient) and, for a correlated term, `"cor"` and `"cov"`
  laid out as `ranef` with the coefficients as rows (1 and `sd^2` on the diagonal, `sd_a sd_b cor_ab` off it); then
  `"residual__"`, sigma's `{"sd": ...}` with the row name `""`, for a Gaussian model without a formula of sigma.
- `summary`: a `Summary` with brms's tables `fixed` (`fixef`'s rows), `spec_pars` (`sigma`), `cor_pars` (empty: no
  autocorrelation term is taken) and `random` (per factor `sd(Intercept)`, `sd(x)`, then `cor(Intercept,x)`, ...), each with
  the columns `Estimate`, `Est.Error`, `l-95% CI`, `u-95% CI`, `Rhat`, `Bulk_ESS`, `Tail_ESS`; and `nobs`, `ngrps`, `chains`,
  `iter`, `warmup`, `thin` and `total_ndraws`. A `FitResult`'s tuning steps are the warmup (`iter` is tuning plus draws, as
  brms counts it), as are an `InferenceData`'s `tuning_steps` (PyMC records them); draws with neither had none. Printing it
  prints brms's layout.
- `residuals(model, fit, type="ordinary", method="posterior_epred")`: per fitted row, in data order, the summary of `y -
  posterior_epred` (`predict(kind="mean")`), unnamed rows as brms's.

ranef(), coef() and VarCorr() of a model brms refuses them for are refused with brms's message: no group-level terms ("The
model does not contain group-level effects.", "No group-level effects detected. Call method 'fixef' to access population-level
effects."), and for VarCorr() no residual SD either ("The model does not contain covariance matrices.").

## Rhat, Bulk_ESS and Tail_ESS

brms takes them from the posterior package (1.6.1 for the frozen cases): the rank-normalised split R-hat and the bulk and
tail effective sample sizes of Vehtari, Gelman, Simpson, Carpenter and Buerkner (2021), "Rank-normalization, folding, and
localization: an improved R-hat for assessing convergence of MCMC", Bayesian Analysis 16(2), 667-718. `autobayes.convergence`
implements them from the paper; posterior's code is not used, and what the paper leaves open was settled on posterior's outputs
on 136 recorded probes (`tests/summary_probes/probe_posterior.R`, the draws exact hexadecimal doubles):

| part | definition | source |
| --- | --- | --- |
| split chains | each chain's first and last `floor(n / 2)` draws; an odd chain's middle draw left out | the paper; the middle draw by probing (dropping the first or last misses) |
| rank normalisation | `z = Phi^-1((r - 3/8) / (S + 1/4))`, `r` the average rank over the split draws | the paper (Blom's offsets) |
| R-hat | `sqrt(var+ / W)` of the split chains, the larger of the bulk's (of `z`) and the tail's (of `z` of `\|theta - median(theta)\|`) | the paper; the median of all draws, an odd chain's middle draw included, by probing |
| ESS | `S / tau`, `tau = -1 + 2 sum_k P_k + rho_2K`, Geyer's initial monotone sequence of the pair sums `P_k = rho_2k + rho_2k+1` of `rho_t = 1 - (W - mean_m c_m(t)) / var+` (`c_m` the autocovariance, divisor `N`); pairs formed from `k = 1` while the previous is positive and `k <= floor((N - 4) / 2)`, the stopping pair left out and its even lag added (`max(rho_2K, 0)` after a negative pair); `tau >= 1 / log10(S)` | the paper; the limit and the even lag by probing |
| short chains | fewer than six draws per split chain: `S / 2`; fewer than three, or constant draws: none | probing |
| bulk ESS | the ESS of `z` | the paper |
| tail ESS | the smaller ESS of the indicators `theta <= q` at the 5 % and 95 % quantiles (type 7, of all draws), split; none when a draw is not finite or the draws' range is below the double's epsilon | the paper; the tolerance and the non-finite rule by probing |

ArviZ 0.23 (Apache-2.0) computes the same R-hat (to 1e-12 on every probe it takes) and the same ESS but for the lag limit,
`N - 3` where posterior's is `N - 4`: on the frozen F07 cases its bulk ESS is up to 8 % below brms's, so ArviZ's is not used.

## What is taken

A univariate, linear Gaussian (`sigma`, or a formula of it) or Bernoulli (any of its links) model (`FamilySpec.summaries`) with an
intercept and centred predictors; population-level terms with design columns (numeric, categorical, interactions,
transforms); group-level terms `(1 | g)`, `(1 + x || g)` and `(1 + x | g)` of grouping factors read from one column, any
number of them; a formula of sigma with an intercept and population-level terms. Everything else is refused naming the
construct (D162): other families, multivariate and non-linear models, mixtures, formulas without an intercept,
`center_predictors=False`, shrinkage priors, offsets, addition terms, `mo()`, `me()`, `mi()`, `cs()`, `gp()`, autocorrelation,
SAR and CAR terms, `fcor()`, `unstr()`, `cosy()`, `g:h` and `g/h`, `mm()`, `gr(by = )`, shared correlation IDs, and a formula of sigma without an intercept or with
group-level terms (a construct that adds a field to the parsed formula is refused by that field's name until it is verified
here).

## Verification

- The oracle gate `check_post` (`tests/oracle/gates.py`, `check_summary`) compares each function with brms's record on the
  frozen cases `gauss_varint_summaries` (`y ~ x + (1 | g)`), `bernoulli_slope_correlated_summaries` (`yb ~ x + (1 + x | g)`,
  brms's `lkj(2)`) and `gauss_distributional_sigma_summaries` (`y ~ x + (1 | g)`, `sigma ~ x`), on the fixture's summary draws:
  four chains of fifty seeded points, AR(1) around chain-specific centres (R-hat 1.01 to 1.24, bulk ESS 15 to 87, tail ESS 27
  to 146), held as a fit's posterior holds them (`summary_draws_fit`). Every label exactly, every number to 1e-9; they agree to
  2e-15. Harness tests (`tests/oracle/test_summaries_harness.py`) show a corrupted entry, label, part, count or argument fails.
- `tests/test_summaries.py`: `brms_draws` equals brms's draws array of the three cases, every variable in order at every draw
  (`lprior` and `lp__` to 1e-10); every summary equals brms's; `rhat`, `ess_bulk` and `ess_tail` equal posterior 1.6.1's on the
  probes (to 1e-12; NA and Inf as posterior gives them); on the eight model structures of the names probe
  (`tests/summary_probes/probe_brms_names.R`: two grouping factors, independent slopes, three and four correlated coefficients,
  categorical predictors and interactions, a formula of sigma with several terms or none, no group-level terms; brms 2.23 on the
  same data under `fixed_param`) every label and order equals brms's and brms's refusals are refused; `log_densities` returns
  the graph's log densities at the points the draws came from (to 1e-10) on each of them; the refusals and options.

## Divergences (docs/brms-parity/divergences.md)

- D162: the structures taken (above); brms summarises any model.
- D163: `residuals()` takes `type="ordinary"` with `method="posterior_epred"` only; brms's default method `"posterior_predict"`
  draws random numbers and its `"pearson"` type is deprecated. brms's further options of the summaries (`robust`, `probs`,
  `summary = FALSE`, `pars`, `groups`, `newdata`, ...) are not offered.
- D164: R-hat and ESS of chains of two or three draws (one draw per split chain) are NaN; posterior 1.6.1 returns numbers there
  (an R-hat from one-draw chains, an ESS of `S / 2`) and NA for four or five draws.
- D165: the representation: DataFrames for brms's matrices, a DataFrame with `(coef, statistic)` columns for each of its
  three-way arrays, a `Summary` for its summary object; `lp__` and `lprior` recomputed from the model at each draw; levels in
  AutoBayes's order (D3).
