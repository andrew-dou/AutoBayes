# brms/Stan to PyMC

Porting a fitted `brms` model into explicit PyMC. The formula interface is doing more
work than it looks like, and most of the porting cost is discovering what it did
implicitly.

The target is an explicit model: every prior written out, every design matrix built by
hand, every distributional parameter given its own regression line. That is more code
than `brm(...)`, and it is the point — the explicit version is the one you can put a
custom likelihood into, run on a GPU, and diff against a previous version.

## Orientation

| R | Python | Note |
| --- | --- | --- |
| `brms` | `pymc` + `numpyro` | Formula interface has no equivalent; you write the model out |
| `cmdstanr` / `rstan` | not needed | PyMC compiles its own graph |
| `posterior`, `bayesplot`, `loo` | `arviz` | One package covers all three |
| `lme4` | `statsmodels` | Only for quick frequentist checks |
| `pyreadr` | — | Reads `.rds` during validation; invaluable while both versions coexist |

Pin the accelerator wheel to your driver's CUDA version — see `pymc-to-jax.md`.

## Model fitting

```r
fit <- brms::brm(
  formula = bf(y_l | cens(censored, y_u) ~ 0 + predictors + (1 | group),
               sigma ~ 1 + x, nu ~ 1 + z),
  family = student(),
  data = df, prior = priors,
  chains = 40, iter = 4000, warmup = 2500,
  control = list(adapt_delta = 0.993, max_treedepth = 12),
  backend = "cmdstanr", seed = 1234
)
```

becomes a `pm.Model()` block in which you build `mu`, `sigma` and `nu` as three
separate linear predictors. `adapt_delta` is `target_accept`; `max_treedepth` carries
its name; `chains`/`iter`/`warmup` map directly. There is no `threads` equivalent —
on GPU the parallelism is across chains, not within a gradient.

## Formulas

`y ~ x1 + x2 + (1 + x1 | group)` has no Python counterpart. You build the design
matrix yourself and index the varying effects with an integer array:

```python
mu = pt.dot(X, beta) + alpha_group[group_idx] + beta_group[group_idx] * x1
```

Deriving `group_idx` with a stable, deterministic factor ordering is the single most
common source of silent divergence between an R and a Python port. Sort the levels
explicitly and store the mapping — do not rely on the order pandas or R happened to
produce.

`bf(y ~ ..., sigma ~ x3, nu ~ x4)` becomes three separate regression lines inside the
same model block. Each distributional parameter gets its own intercept, its own
coefficients, and its own link — `sigma` and `nu` are modelled on the log scale and
exponentiated, and a `nu` constrained above a floor is built as `1 + exp(.)` so the
variance stays defined.

## Priors

```r
prior(student_t(5, 0.13, 0.25), class = "b", coef = "x1")
prior(normal(-0.3, 0.025),      class = "b", coef = "x2")
prior(student_t(4, 0, 0.12),    class = "sd", group = "group")
prior(exponential(1/0.12),      class = "sigma")
```

```python
beta_x1 = pm.StudentT("beta_x1", nu=5, mu=0.13, sigma=0.25)
beta_x2 = pm.Normal("beta_x2", mu=-0.3, sigma=0.025)

# class = "sd" is half-Student-t: brms truncates at zero implicitly
sigma_group = pm.Truncated(
    "sigma_group", pm.StudentT.dist(nu=4, mu=0, sigma=0.12), lower=0
)

sigma_y = pm.Exponential("sigma_y", lam=1 / 0.12)
```

Three traps, in decreasing order of how much time they cost:

**`class = "sd"` is half-Student-t.** brms truncates scale priors at zero silently.
Port it as `pm.Truncated(..., lower=0)`, not `pm.StudentT`. A scale prior with mass
below zero is a different model, and it will sample — badly, and without complaint.

**brms parameterises `exponential` by rate.** `exponential(1/0.12)` is a rate of
`1/0.12`, mean `0.12`. PyMC's `lam` is also a rate, so this one transfers, but check
it rather than assuming: getting it inverted moves the prior mean by a factor of
seventy.

**A prior on a transformed parameter is not the same prior.** `sigma ~ 1 + x` in brms
puts the prior on the log-scale intercept. Writing it on `sigma` directly changes the
model.

## Censoring

`y_l | cens(censored, y_u)` covers four cases at once — uncensored, left-, right-, and
interval-censored — via a per-observation code. Stan dispatches these to `lpdf`,
`lcdf`, `lccdf` and a CDF difference respectively.

In PyMC you build the same four branches explicitly, with masks, and add each
contribution with `pm.Potential`. Preserve the exact convention for the censoring code
(commonly `0` uncensored, `-1` left, `1` right, `2` interval) and assert that the four
masks partition the data — an observation silently in no branch contributes nothing to
the likelihood and is very hard to see afterwards.

The interval-censored branch is where GPU ports usually break, because the natural
implementation is a CDF difference and the Student-t CDF is not differentiable with
respect to `nu` under JAX. `pymc-to-jax.md` covers the fix.

## Missing data and `mi()`

```r
bf(z | mi() ~ 1 + w) + bf(y | cens(...) ~ 0 + mi(z) + ...) + set_rescor(FALSE)
```

brms turns `mi()` into a joint model: the missing entries of `z` become parameters,
sampled alongside everything else and propagating their uncertainty into `y`.

```python
beta_z = pm.Normal("beta_z", 0, 1, shape=n_pred_z)
mu_z = intercept_z + pt.dot(X_z, beta_z)
sigma_z = pm.Exponential("sigma_z", lam=1 / 0.12)

z_imputed = pm.Normal("z_imputed", mu=mu_z[missing_mask],
                      sigma=sigma_z, shape=n_missing)
z_complete = pt.set_subtensor(z_observed[missing_idx], z_imputed)
# z_complete is then a predictor in the main model
```

This is a joint fit, not an imputation step. Imputing `z` beforehand with a point
estimate and treating it as data gives a different — and overconfident — posterior.
If you do pre-impute for speed, that is a modelling decision and belongs in
`DECISIONS.md`, not a preprocessing detail.

`set_rescor(FALSE)` means the sub-models share parameters but not a residual
correlation; separate PyMC likelihood terms give you that by default.

## Post-processing

| brms / bayesplot | ArviZ |
| --- | --- |
| `summary(fit)` | `az.summary(trace)` |
| `rhat(fit)` | `az.rhat(trace)` |
| `neff_ratio(fit)` | `az.ess(trace)` |
| `nuts_params(fit)` | `trace.sample_stats.diverging` |
| `loo(fit)` / `waic(fit)` | `az.loo(trace)` / `az.waic(trace)` |
| `pp_check(fit)` | `az.plot_ppc(trace)` |
| `mcmc_trace(fit)` | `az.plot_trace(trace)` |
| `pairs(fit)` | `az.plot_pair(trace)` |

`posterior_predict(fit, newdata = ..., allow_new_levels = TRUE)` has no one-line
equivalent, and this is the part of a port most likely to be wrong while looking
right. Read `brms`'s behaviour for unseen grouping levels before reimplementing it:
under `sample_new_levels = "gaussian"` it draws a fresh effect from the fitted
population distribution, and a hand-written predictor that instead reuses a known
level's effect, or substitutes zero, produces predictions that are plausible,
systematically overconfident, and pass every convergence check.

Validate prediction separately from fitting. Matching posteriors does not imply
matching predictions.

## Verifying a port

Posterior means agreeing to two decimals is not verification — it is the weakest test
that can pass. In rough order of strength:

1. **Log-density parity at a fixed point.** Evaluate both implementations' log
   posterior at the *same* parameter vector. This is the only check that isolates the
   likelihood from the sampler. If they disagree, nothing downstream matters.
2. **Gradient parity at the same point.** Catches a correct log-density with a wrong
   derivative — which shows up as bad geometry rather than a wrong answer.
3. **Prior-only fits.** Drop the likelihood and confirm both recover the prior. Isolates
   prior-specification errors from likelihood errors.
4. **Simulation-based calibration.** Fit to data simulated from the model and check the
   parameters land in their credible intervals at the nominal rate.
5. **Posterior comparison on the real data.** Last, and interpret it with care: if one
   of the two implementations has not converged, "the posteriors differ" tells you
   nothing about which is correct.

That last point is worth stating plainly, because it is the trap. A disagreement
between an R and a Python fit is evidence that *at least one is wrong*, and the prior
that it must be the new one is unfounded. Check convergence on both before attributing
the difference. A long-established fit that never converged will happily disagree with
a correct port for months.
