# Post-processing: `log_lik`, `bayes_R2`, `loo`

brms-parity feature C07 (brms's `log_lik()`, `bayes_R2()` and `loo()`, loo 2.9.0).

```python
import autobayes as ab
fit = ab.fit(model, chains=4, draws=1000, seed=1)
ll = ab.log_lik(model, fit)          # (chain, draw, __obs__): each fitted row's log density at each draw
r2 = ab.bayes_R2(model, fit)         # (chain, draw): Bayesian R-squared per draw
res = ab.loo(model, fit)             # PSIS-LOO: res.pointwise, res.estimates, res.pareto_k, res.n_eff
print(res)
```

Each takes the model and its fit (a `FitResult` or its `InferenceData`, checked against the model as `predict()` checks
it) and works on the fitted rows with every posterior draw. No function here samples.

## `log_lik(model, fit)`

brms's `log_lik()`: the log density of each fitted row's response at each draw, an `xarray.DataArray` named `log_lik`
with dims `(chain, draw, __obs__)`. It is the per-row term of the model's own likelihood: the family's
`FamilySpec.log_likelihood` (NumPy, `autobayes/families/pointwise.py`) at `predict(kind="linear")`'s linear predictor
(group effects, offsets and monotonic terms included) and the draw's auxiliary parameters (each row's own `sigma`,
`shape` or `phi` under a distributional formula), in the stable forms the likelihoods use (the softplus of the logit
link, `log Phi` of the probit, the zero-inflated and hurdle forms of D28, the ordinal `log_inv_logit_diff` form).
Under `weights(w)` each row's log density is multiplied by its weight, as brms multiplies it. Offered for every family:
Gaussian, Student-t (fixed or learned `nu`), Poisson, negative binomial, Gamma, lognormal, Beta, Bernoulli and binomial
(logit and probit), zero-inflated and hurdle Poisson, cumulative (the log probability of the 1-based category) and
categorical. A non-linear model (C02, `nl=True`, docs/nonlinear.md) is taken the same way, its linear predictor the
non-linear predictor `mu` (brms's `posterior_linpred`); `tests/test_nonlinear.py` checks `log_lik`, `bayes_R2` and `loo` on
it against the graph's own density and `mu` at the frozen brms points. A model with an autocorrelation term (C04,
docs/autocorrelation.md) is taken the same way: its linear predictor carries the ARMA term from the observed residuals, the
mean brms's `log_lik()` reads with `cov = FALSE` (all rows at once, then in data order); `tests/test_autocorrelation.py`
checks `log_lik`, `bayes_R2` and `loo` against the graph's own density and `mu` at the frozen brms points of its three cases.

## `bayes_R2(model, fit)`

brms's `bayes_R2(summary = FALSE)` (Gelman, Goodrich, Gabry and Vehtari 2019, "R-squared for Bayesian regression
models"): per draw, `var(yhat) / (var(yhat) + var(y - yhat))` over the fitted rows, `yhat` the expected response
(`predict(kind="mean")`, brms's `posterior_epred`) and `y - yhat` the residuals, for every family (the Gaussian's
included: brms uses the residuals, not `sigma`). A binomial model's `yhat` is the expected count `n p` against the
successes; weights play no part. For the cumulative family brms's `yhat` is the expected category `sum_k k P(y = k)`
against the 1-based category (settled by probing brms); brms refuses the categorical family ("'bayes_R2' is not defined
for unordered categorical models") and so does `ab.bayes_R2`. Returns an `xarray.DataArray` named `R2`, dims
`(chain, draw)`; summarise it as you like (brms's `summary = TRUE` is the mean, SD and quantiles).

## `loo(model, fit, r_eff=None)`

brms's `loo()`: approximate leave-one-out cross-validation by Pareto-smoothed importance sampling of `log_lik()`. The
result, an `ab.Loo`, holds brms's tables:

- `pointwise`: one row per fitted row, brms's columns `elpd_loo`, `mcse_elpd_loo`, `p_loo`, `looic`,
  `influence_pareto_k`;
- `estimates`: rows `elpd_loo`, `p_loo`, `looic`, columns `Estimate` and `SE`; also as `res.elpd_loo`,
  `res.se_elpd_loo`, `res.p_loo`, `res.looic`;
- `pareto_k` and `n_eff`, loo's `diagnostics`; `r_eff`, the relative efficiencies used; `n_draws`;
  `pareto_k_threshold`, loo's `min(1 - 1 / log10(S), 0.7)`, above which an estimate is not reliable.

`r_eff=None` is brms's default: the relative efficiency of each observation's likelihood `exp(log_lik)` across the
chains (brms hands loo `relative_eff(exp(log_lik), chain_id)`; probed with a multi-chain fixed-parameter fit); a number,
or one per observation, is used as given (`r_eff=1` is loo's own default for a log-likelihood matrix, NaN entries
become 1 as loo makes them). `r_eff` sets the PSIS effective sample sizes and the MCSE, and the tail length through
`3 sqrt(S / r_eff)`; it does not change the expected value of `elpd_loo`.

`ab.post_processing.loo_from_log_lik(log_lik, r_eff)` runs the same computation on any `(chain, draw, __obs__)`
log-likelihood; `psis`, `gpd_fit` and `relative_eff` are the steps.

### What is computed, and where each part comes from

loo is GPL-3 and AutoBayes MIT: no loo source is used. The steps come from ArviZ (Apache-2.0) and the published
algorithm, and loo's documented behaviour and its outputs on probe inputs (run in R) settle what neither fixes.

| part | definition | source |
| --- | --- | --- |
| tail length | `M = ceil(min(S / 5, 3 sqrt(S / r_eff)))` of the largest ratios `r_s = 1 / p(y_i \| theta_s)` | Vehtari, Simpson, Gelman, Yao and Gabry (2024), "Pareto smoothed importance sampling", JMLR 25(72); as ArviZ's `psislw` |
| tail fit | generalized Pareto by Zhang and Stephens (2009, Technometrics 51(3)) with `30 + floor(sqrt(M))` grid points and the PSIS paper's prior `k = (M k + 10 * 0.5) / (M + 10)` | the papers; as ArviZ's `_gpdfit` |
| smoothing | the tail replaced by the fit's quantiles at `(z - 1/2) / M`, `z = 1..M`, none above the largest raw ratio | the PSIS paper; as ArviZ |
| no fit | fewer than five draws in the tail: raw ratios, `k = Inf` | ArviZ (`tail_len <= 4`), and loo's outputs (8 draws: `k = Inf`; 24 draws, a 5-draw tail: fitted) |
| ties and degenerate tails | the tail is the `M` largest draws in order, ties included, cut at the `(M + 1)`-th largest; a tail whose lower quartile ties with the cutoff (the fit does not exist) leaves raw ratios and `k = Inf`; no floor at the smallest double | settled by probing loo: ArviZ keeps only draws strictly above the cutoff and floors it at `log(tiny)`, which differs only in these cases |
| `elpd_loo`, `p_loo`, `looic` | `log sum_s w_s p_s` (normalised smoothed weights), `lpd_i - elpd_loo_i` with `lpd_i = log mean_s p_s`, `-2 elpd_loo_i` | Vehtari, Gelman and Gabry (2017), "Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC", Statistics and Computing 27 |
| estimates | sums; `SE = sqrt(N var)` of the pointwise values (their eq. 23); `2 SE(elpd_loo)` for looic | the same paper |
| `n_eff` | `r_eff / sum_s w_s^2` | the PSIS paper; the `r_eff` factor settled by probing |
| `mcse_elpd_loo` | `sqrt(log(1 + V / E^2))`, `V = sum_s w_s^2 (p_s - E)^2 / r_eff` the self-normalised variance of `E = exp(elpd_loo_i)` carried to the log scale by a log-normal approximation | loo 2.7.0's NEWS ("simplify mcse_elpd using log-normal approximation"); the formula and the `r_eff` divisor settled by probing |
| `r_eff` | ArviZ's `ess(method="mean")` of `exp(log_lik)` over `S`; 1 for chains of fewer than four draws or a constant likelihood (loo's NA, replaced by 1) | ArviZ; brms's use of it and the NA rule settled by probing |

## Verification

- The oracle gate `check_post` (`tests/oracle/gates.py`, test `test_c07_post_processing_equals_log_lik_bayes_R2_and_loo`)
  compares the three functions with the `post` block of every case that lists them, to `PRED_REL` (1e-9 relative), every
  non-finite entry exactly (Inf with Inf, NaN with NaN), on the fixture's draws held as brms's fixed-parameter fit holds
  them, one chain per point (so `r_eff` is loo's NA, used as 1): `gauss_varint_post`, `poisson_varint_post`
  (8 draws: a 2-draw tail, `k = Inf`, raw importance weights), `gauss_varint_post_psis` (64 draws: a 13-draw tail fitted
  and smoothed, `k` from -0.49 to 1.74), `cumulative_varint_post` (the ordinal `bayes_R2`), `binomial_weighted_post`
  (trials and weights) and `categorical_poponly_post` (`log_lik` and `loo`; brms refuses `bayes_R2`). Harness tests show a
  corrupted log-likelihood entry, R-squared draw, pointwise elpd, k-hat, `n_eff` or SE fails, as does a changed
  non-finite entry.
- `tests/test_post_processing.py`: at every frozen point, `log_lik` row by row equals the PyMC model's elementwise log
  density (whose sum gate G2 checks against brms's `log_prob`) on one case of every family and feature, to 1e-12;
  `loo` and `relative_eff` equal loo 2.9.0's outputs recorded in `tests/loo_probes/loo-2.9.0.json` on deterministic
  inputs (S from 8 to 4000, `r_eff` given, NA and computed, heavy tails with `k` above 1, ties at and inside the tail,
  constant columns, log-likelihoods near -380 and -800), within 3e-13 except D50; `psis` equals ArviZ's `psislw` where
  the two take the same tail.

## Divergences (docs/brms-parity/divergences.md)

- D48: brms's further options are not offered: `newdata`, `re_formula`, `resp`, `ndraws`, `draw_ids` (log_lik);
  `summary = TRUE` (bayes_R2); moment matching, `reloo`, `save_psis`, several models and `loo_compare`, `waic`,
  `kfold`, `loo_R2`.
- D49: at extreme linear predictors brms's R-side `log_lik` works from probabilities that round to 0 or 1 and returns
  `-Inf` or `NaN` (or loses digits) where `ab.log_lik` returns the finite value of the model's own density.
- D50: `mcse_elpd_loo` of an observation with `elpd_loo_i` below about -372, where `exp(elpd_loo_i)^2` underflows: loo's
  is NaN, `ab.loo`'s the value of the same formula, computed as `sum_s (w_s p_s / E - w_s)^2`.
- D51: `r_eff` for chains of fewer than about sixteen draws, and for some strongly autocorrelated chains of about fifty:
  loo's ESS (the posterior package's) and ArviZ's differ there.
