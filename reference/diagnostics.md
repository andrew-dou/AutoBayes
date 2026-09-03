# Diagnostics

What to compute after a fit, what to hold it to, and what each failure tells you about
the model.

The reason to fix thresholds in advance is that a threshold chosen after seeing the
output is not a threshold. Put them in the `DECISIONS.md` entry before submitting.

## The panel

Compute all of these, every fit. Any one alone is defeatable.

| Diagnostic | Pass | Investigate | Fail |
| --- | --- | --- | --- |
| Divergences | 0 | < 0.1% of post-warmup draws | > 0.1%, or clustered in one region |
| Max split R-hat | ≤ 1.01 | 1.01–1.05 | > 1.05 |
| Bulk ESS | > 400/chain | 100–400 | < 100 |
| Tail ESS | > 400/chain | 100–400 | < 100 |
| E-BFMI | > 0.3 | 0.2–0.3 | < 0.2 |
| Tree-depth saturation | < 1% | 1–20% | > 20% |

These are conventional values, not laws. Tighten them when a downstream quantity is
sensitive — a tail probability needs far more tail ESS than a posterior mean.

```python
import arviz as az
az.summary(trace)                          # R-hat, ESS, per parameter
trace.sample_stats.diverging.sum()         # divergence count
az.bfmi(trace)                             # one value per chain
(trace.sample_stats.tree_depth >= max_treedepth).mean()
```

Read the *maximum* across parameters, never the mean: one non-mixing parameter
contaminates everything derived from it, and averaging hides exactly the thing you are
looking for.

### What ArviZ actually computes

`az.rhat` defaults to `method="rank"`, which is **the maximum of a bulk and a tail
R-hat**, both rank-normalized: chains are split in half and z-transformed, and the tail
component is the same statistic applied to the absolute deviation from the median. This
is the Vehtari et al. (2021) diagnostic, and it is what `az.summary`'s `r_hat` column
reports.

`method="split"` is the *un*-rank-normalized classic split-R̂. It is a different
statistic and is **not comparable** to the default — do not mix the two across a
comparison table.

For ESS, `method="bulk"` is the rank-normalized default and `method="tail"` is the
minimum of the 5% and 95% quantile ESS.

Two behaviours that look like bugs and are not:

- **ESS can exceed the number of draws.** The estimator floors $\hat\tau$, which
  ceilings ESS at $n\log_{10}n$ — 14,408 for 4 chains × 1,000 draws. Values above N
  indicate antithetic behaviour, which NUTS produces routinely. Because the ceiling is a
  heuristic, treat very large ESS as *saturated* rather than as a meaningful magnitude.
- **`rhat` returns `NaN` rather than raising** for fewer than 2 chains, fewer than 4
  draws, or non-finite input. A `NaN` here means "not computed", not "converged".

**Version warning.** ArviZ 1.0 is a breaking rewrite: `InferenceData` becomes
`xarray.DataTree`, the default credible-interval probability changes from **0.94 to
0.89**, the default interval kind from **HDI to ETI**, and WAIC is removed. Any script
or figure caption that hard-codes `hdi_3%` / `hdi_97%` or says "94% interval" changes
meaning across that boundary without erroring. Pin the version and state it in the
write-up.

## Reading the failures

**Divergences.** The integrator hit curvature it could not resolve. Where they land is
the whole diagnostic: plot divergent transitions against the parameters in a pair plot.

- *Concentrated at small values of a group-level scale* — a funnel. Non-centre that
  parameter. This is the single most common failure in hierarchical models and has a
  standard fix.
- *Spread across the posterior* — not geometry. Usually misspecification, or a
  likelihood that is wrong somewhere.
- *A handful, no pattern* — raise `target_accept` to 0.95–0.99 and re-run. If they
  vanish, they were step-size noise. If the count is stable, they are structural.

Raising `target_accept` to silence divergences without asking where they were is how
you get a biased posterior with clean diagnostics.

**High R-hat.** Chains disagree. Distinguish two cases, because the fixes are opposite:

- *Chains still moving, no separation* — not converged. Run longer warmup.
- *Chains in visibly separate places, each stable* — multimodality. More iterations will
  not help; each chain has found a different mode and is sampling it correctly. Look at
  the trace plot before deciding.

**Low ESS with acceptable R-hat.** Sampling, but inefficiently. High autocorrelation.
Reparameterizing usually beats running longer. Low *tail* ESS with fine bulk ESS is
specifically a warning about quantiles and tail probabilities — the posterior mean is
fine and your 95% interval is not.

**Low E-BFMI.** The momentum resampling is not exploring the energy distribution — the
signature of a heavy-tailed or funnel-shaped posterior. Almost always accompanies
divergences; when it appears alone, suspect a scale parameter with a prior that lets it
approach zero.

**Tree-depth saturation.** Trajectories are being truncated before the U-turn
criterion fires, so NUTS is not doing its job. Raising `max_treedepth` doubles cost per
increment and is a diagnostic step, not a fix — if saturation persists at a higher
limit, the geometry is bad and needs reparameterizing. Under vectorized chains this is
also a throughput problem: one saturating chain paces the whole batch.

## Convergence is not correctness

Every diagnostic above tests whether the sampler explored the posterior you wrote. None
tests whether that posterior is any good. A clean fit to a badly specified model gives
you R-hat 1.00 and a confidently wrong answer.

For that you need, roughly in order of cost:

- **Posterior predictive checks.** Simulate from the fitted model and compare to
  observed data. Look for the features you care about, not just the mean.
- **Calibration on held-out data.** Do 50/80/95% intervals contain the truth 50/80/95%
  of the time? Coverage is the diagnostic that catches overconfidence, and
  overconfidence is what a well-sampled wrong model produces.
- **Simulation-based calibration.** Fit to data simulated from the model itself. Isolates
  implementation errors from specification errors — if SBC fails, the code is wrong, not
  the model.

## Recording

Record every number, every fit, next to the thresholds you set beforehand. Not "it
converged" — the values.

A run that fails its threshold is a result. It goes in the wiki and, if it kills an
approach, in `wiki/hypotheses/ruled-out.md` with the number that killed it. The
diagnostic panel from a failed fit is frequently more informative than a clean one, and
it is the thing an agent six weeks from now will need in order not to propose it again.
