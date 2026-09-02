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

Always use **split** R-hat (ArviZ's default) — it splits each chain in half and
compares, which catches a chain drifting within itself. And read the *maximum* across
parameters, never the mean: one non-mixing parameter contaminates everything derived
from it, and averaging hides exactly the thing you are looking for.

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
