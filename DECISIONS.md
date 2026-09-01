# Decision log

**Rule: before submitting any job that costs real compute, add an entry below.
No entry, no job.**

Each entry answers three questions:

1. **What** are we changing and **why** do we believe it will help?
2. **How will we know if it worked?** A specific diagnostic or metric, with a
   threshold, chosen before the run.
3. **What do we do if it fails?** A concrete next action, not "try something else".

Append with `python3 scripts/new_decision.py "<short title>"`. Newest entries go at
the top of Active. Move an entry to Resolved once its Result section is filled in.

---

## Active

<!-- new entries are inserted directly below this line -->


---

## Resolved

### Example — non-centred parameterization for the group-level scale (2026-01-14)

> Kept as a worked example of the intended altitude. Delete it once you have real
> entries. It is real, from the project this harness came out of, with names
> generalized.

**What.** Replace the centred parameterization of the group-level scale with a
non-centred one: sample a standard normal `z` and form the group effect as
`mu + sigma * z`, rather than sampling the group effect directly conditional on
`sigma`. Model file gets a new version suffix; the existing one is untouched.

**Why.** The previous fit showed the classic funnel signature — divergences
concentrated where the posterior for `sigma` is small, tree depth saturating, and
step size collapsing to keep the acceptance rate up. That geometry is what
non-centring is for. The alternative reading, that the model is simply
misspecified, predicts divergences spread across the posterior rather than
concentrated at small `sigma`, and that is not what the pairs plot shows.

**How we will know it worked.** Three thresholds, all set before the run:

- divergences below 0.1% of post-warmup draws (previous run: 1.6%);
- max split R-hat at or below 1.01 across all parameters (previous: 1.79);
- tree-depth saturation below 20% of transitions (previous: 98.4%).

Sampling wall clock is allowed to rise; this is a geometry fix, not a speed fix.

**What we do if it fails.** If divergences stay above 1% *and* remain concentrated
at small `sigma`, the funnel is not the whole story and the next step is to put a
boundary-avoiding prior on `sigma` rather than to reparameterize further. If
divergences drop but R-hat stays above 1.05 with chains visibly in separate modes,
the problem is multimodality, not geometry, and the next step is a
multi-initialization study — not another parameterization.

**Files.** `models/model_v2.py` (new, cloned from `model_v1.py`),
`cluster/fit_v2.sbatch` (new). No data changes.

**Result (filled in after the run).** 0 divergences in 120,000 draws. Max R-hat
1.01, no parameter above 1.05. Tree-depth saturation 0%. Wall clock rose from
2.4 h to 3.7 h, as expected. Non-centring confirmed as the fix; recorded on
`wiki/concepts/funnel-geometry.md` and the centred version added to
`wiki/hypotheses/ruled-out.md`.
