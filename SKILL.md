---
name: autobayes
description: >
  Fit and check a Bayesian hierarchical regression in Python the way a careful
  researcher would: brms-style formula `y ~ x + (1 | group)`, or `y ~ x` without a group
  term, with a Gaussian outcome or a binary (Bernoulli, logit) outcome, explicit priors, an inspectable PyMC model built
  through Bambi, NUTS on the CPU or through JAX/NumPyro on an explicitly chosen GPU, a
  diagnostic panel with fixed thresholds, predictive draws with named new-group
  semantics (probabilities versus 0/1 outcomes), and a fit record that a later session
  can resume from. Use when someone asks for a multilevel / mixed-effects /
  varying-intercept / random-intercept model, a logistic regression with a group
  intercept, a brms or lme4 formula in Python, PyMC or NumPyro sampling on a GPU, help
  reading divergences / R-hat / ESS, or to set up a Bayesian modelling project that must
  survive many sessions. Three supported families; the skill says what is verified and
  routes everything else to the reference pack.
---

# AutoBayes

One library (`autobayes/`), one reference pack (`reference/`), one scaffold
(`scaffold/`) for long programs. This file is the researcher workflow for the verified
model family, then the routing for everything else.

**What is verified.** One formula subset with numeric predictors, treatment-coded
categorical predictors declared with `categorical={column: reference}`, one
numeric-by-categorical interaction `x:z` with both main effects present (`x*z` is the
shorthand for `x + z + x:z`), and optional independent varying intercepts for distinct factors (M3c-3)
(milestone M3c-1: a formula without it is a population-only model, as in brms),

    y ~ [1 +] x1 [+ x2 ...] [+ z ...] [+ x:z ...] [+ (1 | factor) ...]

Multiple grouping factors have deterministic coverage across these three families and one
independently accepted crossed Gaussian CPU example on both backends; see
`docs/m3c3-evidence/cpu-validation/README.md`. This does not extend the M0 GPU evidence
or establish calibration or other-family sampling validation.

for the outcome families:

- **Gaussian, identity link (M0)**: `ab.gaussian(...)`, priors on the intercept, every
  slope, the group SD and the residual SD `sigma`. Log density equals an independent
  reference at fixed points (with and without the Jacobian, and through the JAX
  compilation NumPyro samples), gradient equals finite differences, Hessian equals the
  closed-form precision with the scales fixed, NUTS reproduces an exact conditional
  Gaussian posterior, a simulated recovery passes the panel on both backends, and one
  GPU run on an H100 matched the CPU posterior. Evidence `docs/m0-evidence/`, checks in
  `docs/m0-decision.md`.
- **Bernoulli, logit link (M1)**: `ab.bernoulli(...)` for a strictly 0/1 response, priors
  on the logit-scale intercept, slopes and the group SD, no `sigma`. Log density and
  gradient against an independent reference at moderate and extreme logits (to 1e6,
  both signs, both outcomes) on both the PyTensor and the JAX path, prediction meaning
  (probability versus 0/1 draw), and a simulated recovery on both backends. GPU
  execution for this family is **pending** (the same code path, not yet run on a GPU).
  Evidence `docs/m1-evidence/`, checks and the repair record in `docs/m1-decision.md`.

- **Binomial counts, logit link (M2)**: `ab.binomial("successes | trials(n) ~ x + (1 | group)", df, priors)`,
  brms syntax, counts validated before any cast, zero-trial rows dropped and reported
  before centring, a data set without trials refused, constant outcomes accepted with a
  separation warning. Log density and full gradients against a 60-digit high-precision
  reference at logits to 1e6 and counts to 1e6 per row on both paths, row by row and near
  the mode, `n = 1` equivalence with the Bernoulli family, expansion to Bernoulli rows with
  aligned centring, three prediction quantities. Counts above 1e6 per row are refused: the
  float64 density near a binomial mode loses the prespecified accuracy there
  (`docs/m2-decision.md`, repair record). GPU execution **pending**. Evidence
  `docs/m2-evidence/`, checks in `docs/m2-decision.md`.

**What is not verified.** Every other formula and family (other links,
varying slopes, correlation IDs, grouping `/` or `:` syntax, multimembership,
beta-binomial). The library refuses them with an error naming the construct. Population-only
models (M3c-1) are verified deterministically (density, gradients, closed form, predictions)
and one Gaussian CPU example has sampling evidence accepted in `docs/m3c1-decision.md`; this is not calibration or all-family sampling evidence. Do not loosen the parser to get past that
error; go to "Outside the verified family" below.

**Population intercept control (M3c-2).** `0 + terms` and `terms - 1`
remove it. Omit `Priors.intercept` and `value_point(intercept=...)`; supplied values
are refused. The first categorical main effect becomes full-level coded, including
its declared reference; later categorical main effects and all interactions
retain non-reference differences. Inspect `model.spec.columns` and cover the actual
coefficient set with explicit priors. Reordering factors can change independent
prior meaning. No-intercept models use the raw design even when centering is
requested, with requested/effective flags recorded separately. At least one
population predictor is required; conflicting inclusion/removal tokens are refused.
See `docs/m3c2-decision.md` and `examples/m3c2_no_intercept.py`. Three-family deterministic checks and one Gaussian CPU example at target_accept .99 were independently accepted; earlier .9 divergence failures remain preserved. This is not calibration or GPU evidence.

**Multiple independent intercepts (M3c-3, merged PR #11).** Use explicit additive terms
`(1 | county) + (1 | year)` and exact-key `sd={"county": HalfNormal(1), "year": Exponential(2)}`.
Scalar SD remains a one-factor shorthand. Inspect `spec.groups`, `spec.levels_by_group`,
`design.group_designs`, `design.group_slices` and `model.group_names`; ambiguous singular
fields raise with multiple factors. Effects have independent priors, not necessarily
independent posteriors. Common centering and no-intercept coding do not change.
`value_point` uses factor-keyed SD/offset mappings. `new_groups` accepts either the existing
policy string for all factors or an exact-key policy mapping. Repeated unseen labels share
one effect within their factor, including across rows with different other memberships.
Equal labels in separate factors are different identities; factor order determines RNG
traversal. Fit records retain all factor scales and coordinates. Never silently expand
nesting or reinterpret reused within-parent labels. Three-family deterministic feature checks and
one crossed-Gaussian CPU backend/API unit passed independent review; broader sampling,
calibration and GPU validation remain future work. Follow
`docs/m3c3-decision.md` and `examples/m3c3_multiple_intercepts.py`.

## Session start, every time

1. If the project has `docs/coordination/STATUS.md` or a `wiki/`, read it before
   touching code. It records what was fit, with which seed, and what is pending.
2. Install the pinned stack from the lock file and print the environment; keep the
   output for the handoff:

```bash
uv sync --locked --extra jax --extra dev
uv run --no-sync python -c "import autobayes as ab; print(ab.format_environment(ab.environment()))"
```

   Tested stack (`pyproject.toml` pins it, `uv.lock` fixes the rest): Python 3.11,
   pymc 5.28.5, pytensor 2.38.3, bambi 0.17.2, formulae 0.6.2, arviz 0.23.4, numpyro
   0.21.0, jax/jaxlib 0.10.2, numpy 2.4.6. Those are the versions the fit records in
   `docs/` name; the lock's transitive versions were frozen on 2026-09-10 and were not
   recorded for earlier runs. Other versions are untested, not "probably fine". Do not
   upgrade to get past an error; record the error and stop.

3. Three environment traps seen on this stack, and the fix for each:
   - `~/.pytensor` not writable (sandboxes, shared accounts): every model build fails
     at compile time. Set `PYTENSOR_FLAGS=base_compiledir=<writable dir>`.
   - arviz 0.23 fails to import with matplotlib 3.11.1. Pin `matplotlib<3.11`.
   - JAX touched before `import autobayes` (any `jax.devices()` call, any JAX import
     that initialises the backend): CPU chains on the NumPyro path then run
     sequentially. `fit()` says so in its notes and reports `chain method : sequential`.
     Import autobayes first, or accept the wall time. The posterior is unaffected.
     Both behaviours are checked in fresh interpreters by `tests/test_fresh_process.py`,
     as is `AUTOBAYES_HOST_DEVICES=0` (turns the import-time setting off) and a process
     without JAX (`environment()` reports it; `backend="numpyro"` raises `DeviceError`
     naming the `[jax]` extra; `backend="pymc"` still works).

## The workflow

Run the steps in order. Each has a check; a step whose check fails is not done.

### 1. Data

A pandas data frame with a response, predictors, and, for a hierarchical model, a grouping
column (any dtype; it is coded as a factor with levels in **sorted** order; a population-only
formula needs none). Predictors are numeric, or
categorical when you **declare** them with `categorical={"column": reference_level}`
(string, pandas Categorical, or integer codes; an undeclared integer column is numeric,
an undeclared label column is refused with the declaration to use). Gaussian: a numeric
response. Bernoulli: a numeric column with values exactly 0 and 1, both present, where
**1 is the outcome being modelled**; booleans and labels are refused with a message
telling you to recode them, because a label column would silently make the sorted
first level the "success". The library refuses missing values, a constant predictor,
a categorical predictor with one level, a reference level that does not occur, and a
grouping factor with one level, each with a message saying so. Transform columns in
the data frame, not in the formula.

### 2. Formula and priors

```python
import autobayes as ab

priors = ab.Priors(
    intercept=ab.Normal(0, 5),   # applies to the CENTRED intercept: E[y] at the predictor means
    b=ab.Normal(0, 2),           # one prior for every slope, or {"x": ..., "w": ...}
    sd=ab.HalfNormal(1),         # SD of the varying intercepts (brms class "sd"); omit it for a model without (1 | group)
    sigma=ab.HalfNormal(1),      # residual SD
)
model = ab.gaussian("y ~ x + (1 | school)", df, priors)
```

For a model without a group-level term (milestone M3c-1; brms's base case, where
group-level terms are optional):

```python
priors = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1))   # no sd
model = ab.gaussian("y ~ x + arm", df, priors, categorical={"arm": "control"})           # y ~ 1 is intercept-only
```

The same predictor vocabulary applies (numeric terms, declared categorical terms, one
interaction); the data need no grouping column. Passing `sd=` here is refused ("has no
group-level term"), and a grouped formula without `sd=` is refused at build ("needs a prior
on the SD of its varying intercept"): nothing is filled in or ignored. `describe()` says
"no group-level term (population-level model)" and lists `Intercept`, one variable per term
and `sigma_log__` as the sampled variables. Predictions take `newdata` without a grouping
column; `new_groups` does not apply and any value other than the default is refused.

For a binary outcome:

```python
priors = ab.Priors(
    intercept=ab.Normal(0, 1.5),  # log-odds of y = 1 at the predictor means
    b=ab.Normal(0, 1),            # log-odds ratio per unit of the predictor
    sd=ab.HalfNormal(1),          # SD of the group intercepts, logit scale
)                                 # no sigma: passing one is an error for this family
model = ab.bernoulli("y ~ x + (1 | clinic)", df, priors)
```

For counts with known denominators (k successes of n trials per row):

```python
priors = ab.Priors(intercept=ab.Normal(0, 1.5), b=ab.Normal(0, 1), sd=ab.HalfNormal(1))
model = ab.binomial("deaths | trials(patients) ~ x + (1 | clinic)", df, priors)
```

Counts must be real integers (int columns, or floats holding integers); bool, complex,
non-integral, negative, non-finite, `successes > trials`, and values above `1e6` per row
are refused before any cast, naming the column. The cap is the tested numerical domain of
this implementation, not a modelling limit; do not work around it by splitting a row into
several rows, which changes the unweighted centring and therefore what the centred-intercept
prior refers to unless the centring is preserved or the prior transformed. A row with `trials = 0` and no successes
carries no information and is **dropped before centring** (`describe()` reports how many
and where); successes with zero trials is impossible data and refused; a data set with
no trials at all is refused. If every fitted row has zero successes or all successes the
model still builds, with a warning: the data are separated along the intercept direction
and the posterior there is set by the intercept prior's tail, so refit with a wider and a
narrower intercept prior and report the movement. Bambi's `p(k, n)` left-hand side is
refused; write the brms form.

For a categorical predictor (treatment coding, milestone M3a-1):

```python
priors = ab.Priors(
    intercept=ab.Normal(0, 5),
    b={"x": ab.Normal(0, 2),                                     # one prior per term ...
       "arm": {"low": ab.Normal(0, 2), "high": ab.Normal(0, 2)}},  # ... or per non-reference level
    sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1),
)
model = ab.gaussian("y ~ x + arm + (1 | site)", df, priors, categorical={"arm": "control"})
```

`categorical` names the column and its **reference level**; both are required. The
levels are the distinct values of the fitted rows as string labels in sorted order (a
pandas Categorical's own order is not used, and its declared categories that never
occur are reported and get no coefficient). Labels are formed value by value with no
float intermediate: integers of any width keep their exact decimal form (`int64`,
`uint64`, nullable and object columns alike), floats are accepted only when integral and
at most 2**53 in magnitude, strings are themselves; an integer and its decimal string are
the same label (`categorical={"z": 1}` and `{"z": "1"}` name the same level). Booleans,
complex values, missing values, larger or fractional floats, and a column that mixes
text labels with numeric codes are refused before any model is built. Each non-reference level gets one dummy
column `arm[level]` and one coefficient: the difference between that level and the
reference on the link scale (response scale for Gaussian, log-odds for Bernoulli and
binomial) at the same predictors and group, a constant contrast unless the term also
interacts with a numeric predictor (see the interaction paragraph). The posterior variable is `arm` with the
levels as coordinates, so `az.summary(fit.idata, var_names=["arm"])` shows rows
`arm[low]`, `arm[high]`; `model.coefficients` is the exact map from design columns to
variables. Per-level priors must cover exactly the non-reference levels (the reference
has none), must share one distribution family, and a key such as `"arm[low]"` at the
term level is refused: nothing is filled in by default. Changing the reference level
relabels the coefficients by a known linear map and leaves the likelihood unchanged,
but independent priors in the new coding are a **different model** (their induced joint
prior is correlated); `docs/m3a1-decision.md` derives the map and the test checks it.

For an interaction between a numeric and a categorical predictor (milestone M3a-2):

```python
priors = ab.Priors(
    intercept=ab.Normal(0, 5),
    b={"dose": ab.Normal(0, 2),                                   # the reference arm's dose slope
       "arm": ab.Normal(0, 2),                                    # arm minus control at dose = 0
       "dose:arm": {"low": ab.Normal(0, 1), "high": ab.Normal(0, 1)}},  # slope differences (or one prior)
    sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1),
)
model = ab.gaussian("y ~ dose + arm + dose:arm + (1 | site)", df, priors, categorical={"arm": "control"})
```

`x*z` means `x + z + x:z` (measured identical in Bambi); the canonical spelling handed to
Bambi is `numeric:categorical`, which `describe()` prints as `expanded :` when it differs
from what you wrote, and it is the prior key (`"dose:arm"`, never `"arm:dose"`). Both main
effects are required: `x + x:z` or `z + x:z` are refused because they are different
models (a level intercept or the reference slope forced to zero, and formulae recodes the
interaction). The reference level has the raw intercept and the slope `dose`; every other
level has an intercept difference `arm[l]` and a slope difference `dose:arm[l]`, so its
slope is `dose + dose:arm[l]`. Once the interaction is present, `arm[l]` is the contrast
with the reference **at dose = 0** only; at dose `d` the contrast is `arm[l] + d *
dose:arm[l]` (on the log-odds scale for Bernoulli and binomial), and `describe()` says
so. Multiple distinct pairs are now admitted (M3a-3): use explicit mains once, such as
`x + w + z + v + x:z + w:z + x:v`. A factor contrast sums all its numeric partners;
a numeric slope sums all its categorical partners, with reference contributions zero.
The statements above describe the single-pair example with other numeric predictors
zero, other categorical predictors at their references and group effects zero.
Overlapping shorthand still repeats mains (`x*z + w*z` is refused); use explicit colons.
Each pair retains its own exact term/per-level priors, labels and coordinates.
With no population intercept, only the first categorical main is full-level; every
interaction remains a non-reference difference and effective common centering is off.
Independent priors are not automatically transported when references change.
Use `docs/m3a3-decision.md` and `examples/m3a3_multiple_interactions.py`. Focused deterministic
checks passed; the local full suite was interrupted after a wall-deadline supervision failure,
with no retry, so hosted checks remain required. Feature-specific sampling/calibration/GPU
evidence remains pending. The posterior variable is `dose:arm` with the levels as
coordinates (`az.summary(fit.idata, var_names=["dose:arm"])`). Check
`model.spec.columns` and `model.design.X_raw[:5]`: the product columns are
`dose * 1[arm = l]`, built and verified against an independent one-hot construction.

Every class the family has is required. There are no defaults, because a default
prior is the thing you cannot see in the formula and cannot reproduce later. Location
classes take `Normal` or `StudentT(nu, mu, sigma)`; scale classes take `HalfNormal`,
`HalfStudentT(nu, sigma)`, `Exponential(lam)` or `HalfCauchy(beta)`. Parameters are
PyMC's: `sigma` is a standard deviation, `lam` a rate, `beta` a scale. For the
Bernoulli family the location priors are on the **logit scale**: `Normal(0, 1.5)` on
the intercept puts most mass on probabilities between 0.05 and 0.95; `Normal(0, 5)`
would be a strong prior for extreme probabilities, not a weak one.

**The centring trap.** Bambi, like brms, centres the predictors and samples the
intercept at the predictor means. Your prior on `intercept` is a prior on that
quantity. The `Intercept` column in the posterior is the raw-scale intercept, back-
transformed after sampling. Choose the prior for the centred quantity (the typical
response), not for "y at x = 0". `center_predictors=False` turns this off if the raw
intercept is what the prior is about. With a categorical predictor the dummy columns
are centred too, so the sampled intercept is the linear predictor at the numeric means
**and at the sample proportions of the levels** (group effect zero): for the Gaussian
family the expected response there; for the logit families a log-odds whose `expit` is
the probability at that proportion-weighted predictor, **not** the average probability
across the levels (the link is nonlinear). The posterior `Intercept` is the linear
predictor at x = 0 and the reference level. With an interaction every expanded column is
centred at its **own** fitted-row mean: the product column `x * 1[z = l]` at
`p_l * mean(x within level l)`, which is not `mean(x) * p_l` when `x` differs by level
(`describe()` prints both). The sampled intercept is then the linear predictor at a point of
the expanded design that no covariate row can occupy (a row has `x*1[z=l]` equal to `x` or
0); the reported `Intercept` is `a_c` minus the sum of column mean times coefficient, the
linear predictor at x = 0 in the reference level.

Check: `print(model.describe())`. Read the `levels` line (index to label map), the
categorical line (reference, sorted levels, proportions, unused categories), the
`mu_i =` (or `eta_i =`) line, the centring paragraph, and the "PyMC variables sampled"
block. Check the design itself: `model.spec.columns` and `model.design.X_raw[:5]` are
the columns that multiply the coefficients (the library has already verified them
against an independent one-hot construction; you can too). That block is the model that will be sampled: `Intercept` (centred), one
variable per slope, `sigma_log__` (Gaussian only), `1|school_sigma_log__` (the group
SD, log scale) and `1|school_offset` (the J standard-normal offsets of the non-centred
parameterisation). `model.pymc_model` is the graph itself;
`pytensor.dprint(model.pymc_model["mu"])` (or `["p"]`) shows the linear predictor if
`describe()` is not enough. For the Bernoulli family the likelihood is written by hand
as `y * eta - softplus(eta)` with a softplus whose value and gradient are exact for
every finite logit on both the PyTensor and the JAX path (`autobayes/likelihoods.py`).
Bambi's built-in `bernoulli` family is not used for the likelihood because it floors
`p` at machine epsilon: at `eta = 40` it returns -72.1 for two zero and two one
observations where the density is -80, and the gradient vanishes. The design
matrices, priors, centring, group coding and prediction machinery are still Bambi's.
Separation in the data (a predictor or group that perfectly predicts the outcome)
is still a modelling problem: the likelihood is now right there, and the posterior is
then governed by the logit-scale priors, which should be chosen and stated deliberately.

### 3. Fit

```python
fit = model.fit(backend="pymc",    device="cpu", chains=4, draws=1000, tune=1000, seed=7)
fit = model.fit(backend="numpyro", device="cpu", chains=4, draws=1000, tune=1000, seed=7)
fit = model.fit(backend="numpyro", device="gpu", chains=4, draws=1000, tune=1000, seed=7)
print(fit.report())
```

- `backend="pymc"` is PyMC's own NUTS, CPU only; `device="gpu"` there is an error.
- `backend="numpyro"` compiles the graph to JAX and samples with NumPyro's NUTS.
  `device` is resolved against `jax.devices()`. If a GPU is requested and none is
  visible the call raises `DeviceError` listing the visible devices. **It never falls
  back to the CPU silently**, so a report that says `device actual : cuda:...` means a
  GPU ran, and nothing else does.
- Always pass `seed`. The report and the fit record carry it.

Check the report: `device actual`, `dtype` (must be float64; float32 divergences look
like geometry and are not), `chain method` (`parallel` on CPU, `vectorized` on GPU,
`sequential` only with the import-order note above), and any `note` line. Wall time
includes XLA compilation on the NumPyro path; the first fit of a graph is slow for
that reason and that reason only.

### 4. Diagnostics

```python
from autobayes.diagnostics import panel, format_panel
p = panel(fit.idata)          # max_treedepth=10 unless you changed it
print(format_panel(p))
```

Thresholds (from `reference/diagnostics.md`; the panel applies them to the *worst*
parameter, never an average):

| Diagnostic | pass | investigate | fail |
| --- | --- | --- | --- |
| divergences | 0 | < 0.1 % of draws | more, or clustered |
| max split R-hat | <= 1.01 | 1.01-1.05 | > 1.05 |
| min bulk / tail ESS | > 400 total | 100-400 | < 100 |
| min E-BFMI | > 0.3 | 0.2-0.3 | < 0.2 |
| tree-depth saturation | < 1 % | 1-20 % | > 20 % |

The ESS row is 400 *total* over chains, the smoke-run threshold; for a fit you will
report, hold it to 400 per chain as the reference says.

What to do, in this family specifically:

- **R-hat 1.01-1.03 or bulk ESS of a few hundred on `Intercept` and on
  `1|group_sigma`, with everything else clean.** This is the known weak direction of
  the model: the intercept trades off against the mean of the J offsets. At 4 x 1000
  draws on the M0 example it gave R-hat 1.02 and bulk ESS 350 on both backends; at
  4 x 2500 it passed. Lengthen the run. Do not loosen the threshold, and do not
  change the parameterisation first; it is already non-centred.
- **A handful of divergences with no pattern.** Refit with `target_accept=0.95` (the
  default is 0.9). If they vanish they were step-size noise. If the count is stable,
  look at where they are before doing anything else (`reference/diagnostics.md`,
  "Reading the failures").
- **Divergences concentrated at small `1|group_sigma`.** The funnel. The model is
  already non-centred, so this points at a group SD prior that lets the scale reach
  zero with too few groups. Look at J and at the `sd` prior.
- **Anything at `fail`.** Stop. The fit is not evidence. Record the panel anyway; a
  failed panel is a result.

`fit.idata` is an ArviZ `InferenceData`; `az.summary(fit.idata)` gives the per-
parameter table. In it, `Intercept` is the raw-scale intercept (log-odds for the
Bernoulli family), `1|school_sigma` is the group SD, `1|school` are the J group
effects `u`, and `1|school_offset` the standard-normal offsets (ignore them in
reporting). The interval columns are the 94 % HDI under arviz 0.23; arviz 1.0 changes
the default to an 89 % ETI, so state the version next to any interval you quote.
Binary outcomes carry less information than Gaussian ones: expect wider posteriors
and, with few groups, a group SD posterior that leans on its prior.

### 5. Predictions

```python
ppc = model.predict(fit, kind="response", seed=1)                       # in-sample y draws
mu  = model.predict(fit, newdata=new_df, kind="mean")                    # E[y | x, group]
new = model.predict(fit, newdata=new_df, kind="response", new_groups="population", seed=1)
```

`kind="mean"` returns draws of the expected response: `mu` for Gaussian, `p = P(y = 1)`
for Bernoulli, and for binomial `p` together with `expected = n p` (brms
`posterior_epred`, which is `mu * trials` for binomial models). `kind="response"` also
returns draws `y` of a new observation, `Normal(mu, sigma)`, a 0/1 `Bernoulli(p)` draw,
or an integer `Binomial(n, p)` draw (brms `posterior_predict`). For binomial models
`kind="probability"` returns `p` alone and needs no trials column (brms
`posterior_linpred(transform = TRUE)`); `mean` and `response` need the trials column
named in the formula in `newdata`, and a row with zero trials gives `expected = 0` and
`y = 0`. Report a probability from `p`, not from the mean of `y`; report an interval
for an outcome from `y`, not from `p`. For a level of the grouping factor that was not
in the training data:

- `new_groups="error"` (default): refuse, so an unseen level is never silently treated
  as an observed one.
- `"population"`: draw the level's effect from `Normal(0, sd_group)` for each posterior
  draw, on the linear-predictor scale (the logit scale for Bernoulli). This is the
  distribution the model fit (brms `sample_new_levels = "gaussian"`) and is the right
  choice for "a new school".
- `"resample_observed"`: for each posterior draw and each distinct unseen label,
  reuse the fitted effect of a uniformly chosen observed level (brms's default
  `"uncertainty"`). Rows sharing a label share the choice; distinct labels are drawn
  independently and coincide only by chance. It understates between-level spread
  because it resamples shrunk effects from a finite set. Computed in this layer:
  Bambi's `sample_new_groups=True` gives every unseen label the same effect in a draw.

The M0 example prints the two predictive SDs side by side (observed school 0.32, new
school 0.74 on the simulated data): that gap is the between-group variance, and it is
what a prediction for a new unit has to carry.

### 6. Record

```python
import json
rec = ab.fit_record(model, fit)                 # spec, execution, environment, panel, posterior
json.dump(rec, open("fits/<name>_record.json", "w"), indent=1)
```

A fit record is the unit of evidence. It is what goes into the status file or wiki
page, and what a later session compares against. Never write over an earlier record;
name a new one. To check that two fits target the same posterior (CPU vs GPU, PyMC vs
NumPyro, two seeds):

```python
c = ab.compare_records(rec_a, rec_b)            # population means within 4 combined MCSE
print(ab.format_comparison(c))
```

Agreement says the two runs sampled the same posterior. It says nothing about speed.
A speed claim needs the same model, priors, dtype, chains, warmup and draws, the
diagnostic panels side by side, and the hardware named; `reference/acceleration.md`
has the audit of how the published numbers fail that test.

The worked examples do all six steps and write a record:

```bash
.venv/bin/python examples/m0_varying_intercept.py --backend numpyro --device cpu \
    --record fits/example_cpu.json
.venv/bin/python examples/m1_bernoulli_varying_intercept.py --backend numpyro --device cpu \
    --record fits/example_bernoulli_cpu.json
.venv/bin/python examples/m2_binomial_varying_intercept.py --backend numpyro --device cpu \
    --record fits/example_binomial_cpu.json
```

## Recovering from failures

| Symptom | Cause | Do |
| --- | --- | --- |
| `FormulaError: ... not supported in M0` | Formula outside the verified subset | Not a bug. Use the subset, or write the model by hand: "Outside the verified family" |
| `FormulaError: response ... outside {0, 1}` / `boolean` / `constant` | Bernoulli response not coded 0/1 | Recode so 1 is the modelled outcome; for counts with denominators use `ab.binomial` |
| `FormulaError: ... needs 'successes \| trials(n)'` or `Bambi's 'p(...)'` | Binomial left-hand side not in brms form | Write `k \| trials(n) ~ ...` |
| `FormulaError: ... exceed trials` / `zero trials is impossible` / `no trials in the data` | Count data inconsistent | Fix the data; zero-trial rows without successes are fine and are dropped and reported |
| `FormulaError: ... above the tested limit` | A count above 1e6 in one row | Outside the validated float64 domain (`docs/m2-decision.md`); not silently accepted |
| Binomial `describe()` shows a separation WARNING | All rows zero successes or all successes | Refit with a wider and a narrower intercept prior; report the movement; the result is prior-sensitive by construction |
| `FormulaError: ... is complex-valued` | A complex column (pandas counts it as numeric; a cast would drop the imaginary part) | Take the real part explicitly if that is what you mean |
| `PriorError: the Bernoulli family has no residual SD` | `sigma` given to `ab.bernoulli` | Drop `sigma`; the priors are on the logit scale |
| `PriorError: prior for 'sd' must be a positive-support prior` | Wrong class of prior | Location classes for intercept and slopes, scale classes for `sd` and `sigma` |
| `PriorError: the formula has no group-level term ... do not pass sd` | `sd=` given to a population-only formula (M3c-1) | Drop `sd`, or add `(1 \| group)` if a varying intercept was intended |
| `PriorError: ... needs a prior on the SD of its varying intercept` | `sd=` missing for a grouped formula | Pass `sd=`; it is required whenever the formula has `(1 \| group)` |
| `ValueError: this model has no group-level term; new_groups does not apply` | `new_groups=` set on a population-only model | Leave `new_groups` at its default; every row is a population-level prediction |
| Bernoulli fit: divergences or a group SD posterior stuck at its prior | Separation, or very few groups | Check for a group or predictor that perfectly predicts the outcome; tighten the logit-scale priors deliberately, and say so |
| A hand-written PyMC logistic likelihood gives NaN gradients at large logits | `pm.Bernoulli(logit_p=...)` (PyTensor path) or `pt.softplus` (JAX path) branch on the untaken side | Use `autobayes.likelihoods.stable_softplus`, or verify values and gradients on both paths at logits up to 1e6 before trusting any form |
| `DeviceError: ... JAX sees no GPU` | No CUDA jaxlib, or no GPU on this machine | Check `jax.devices()`. On a GPU host install `"jax[cuda12]"`; else `device="cpu"`. Do not "fix" this by catching it |
| `DeviceError: backend='pymc' runs ... CPU only` | GPU asked of the PyMC sampler | `backend="numpyro"` |
| report note: chains sequential | JAX initialised before autobayes import | Reorder imports; posterior unaffected |
| report `dtype : float32` | x64 disabled after import | The library enables x64 before sampling; if you see float32, something reset `jax_enable_x64`. Refit; do not trust the float32 run |
| `PermissionError` under `~/.pytensor` | Read-only home | `PYTENSOR_FLAGS=base_compiledir=<dir>` |
| `DeviceError: backend='numpyro' needs jax` | The `[jax]` extra is not installed | `uv sync --locked --extra jax --extra dev`, or `backend="pymc"` |
| `uv sync --locked` refuses | `pyproject.toml` and `uv.lock` disagree | Do not edit pins to get past it; a version change is its own reviewed slice |
| `ImportError` from arviz mentioning matplotlib | matplotlib 3.11.1 | `matplotlib<3.11` |
| `az.summary` MCSE column reads 0.000 | Rounding to 3 decimals | `az.summary(..., round_to="none")` before dividing by it |
| Minutes of nothing, then sampling | XLA compilation | Expected once per graph shape; the report's wall time includes it |
| GPU run slower than CPU on a small model | 200 rows x 4 chains does not fill an H100 | Expected. Agreement is the evidence here, not time |
| `get_jaxified_logp` sign looks wrong | pymc 5.28.5 names `negative_logp` backwards: the default returns `+logp` | Use the default; test A4 pins this |
| `compile_d2logp` gives the negative Hessian | PyMC's `hessian(negate_output=True)` | It is the precision matrix; test A5 pins this |
| R-hat just over 1.01 on `Intercept` | Intercept vs mean-offset direction | More draws (see step 4) |

Anything else: match the symptom in the routing table at the end of this file.

## Handing off a session

Write before context runs out, not at the end. Before any long operation, and any
time you have learned something and not written it down, update the status file
(`docs/coordination/STATUS.md` in this repository; `wiki/` in a scaffolded project).
A handoff a fresh session can use states:

- **Environment**: the `format_environment` output, verbatim.
- **What was fit**: formula, priors, n, J, `describe()` output or the record path.
- **How**: backend, device actual, dtype, chain method, chains/tune/draws, seed.
- **Panel**: every value and its status, not "converged".
- **Records**: the path of each fit record; which pair was compared and the result.
- **Executed vs implemented**: name what actually ran on which hardware. "GPU
  supported" and "GPU validated" are different sentences; write the one that is true.
- **Failures**: what failed, the message, what you changed. A failed run is a result.
- **Next action**: one concrete step, and the command to resume:

```bash
git checkout <branch> && uv sync --locked --extra jax --extra dev
uv run --no-sync pytest tests -q -m "not slow"                       # fast checks, a few minutes
uv run --no-sync pytest tests -q -m slow                             # three sampling checks, ~1 min
```

Where evidence goes: a test run allocates its own directory
`evidence-runs/<UTC time>-<git head>-p<pid>-<token>/` (gitignored) and writes its JSON
records and a `run.json` (environment, git head, pid) there, each file once. Two runs
never share a directory; `AUTOBAYES_EVIDENCE_DIR` names a parent for new run directories
and is refused if it resolves into `docs/` or into an existing run. The committed
`docs/m*-evidence/` files are history; a rerun never rewrites them, and
`tests/test_evidence_provenance.py` checks their bytes against HEAD. To promote a run's
records into the history, copy them under a new name with the decision record that
explains them. Acceptance for the sampling checks is the criteria the tests assert
(analytic z-scores, panel thresholds, truth inside the 94 % HDI, backends within
4 combined MCSE), never agreement with an earlier JSON: Monte Carlo variation and
platform differences change valid summaries.

CI: `.github/workflows/ci.yml` runs the fast suite on every push and pull request on the
pinned stack (Ubuntu, CPU); `slow.yml` is started by hand and runs the three sampling
checks. Both upload `evidence-runs/` as an artifact. A green run is evidence for the
pinned environment on that runner, nothing more; there is no GPU workflow.

## GPU and remote execution

The reproducible command, and what counts as evidence, is in
`docs/m0-gpu-validation.md` (executed once for the Gaussian family; **pending for the
Bernoulli and binomial families**, same code path, not yet run). In short: run the example with
`--backend numpyro --device gpu --record <path> --compare docs/m0-evidence/B2_numpyro_cpu_record.json`,
keep the full output, and require `device actual : cuda:...`, `dtype : float64`,
`chain method : vectorized`, a panel at `pass`, and `agree: True` against the CPU
record. Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` so JAX does not take the whole card
at import. On a rented node, install exactly the verified versions with
`"jax[cuda12]==0.10.2"` on a CUDA 12.x image, put a `timeout` on the run, and stop the
node the moment the output is copied. The SLURM form is `cluster/m0_gpu_smoke.sbatch`.

## Outside the verified family

The library will not build it, and you should not extend the parser to make it. Two
routes:

- **A brms/Stan model exists.** `reference/brms-to-pymc.md`: port from the generated
  Stan code and `standata()`, not from the formula; reproduce the centring; write every
  default prior explicitly; sort factor levels yourself. Then verify the port the way
  M0 was verified (log density at fixed points against an independent reference,
  before any sampling).
- **A new model.** Write it in PyMC directly, keep the M0 conventions (non-centred
  group effects, explicit priors, sorted levels, a `describe`-style statement of what
  is sampled), and take diagnostics through the same `panel`. If the likelihood needs
  a custom operation, the companion skill `autobayes-custom-op` decides whether it
  really does.

Bambi accepts far more formulas than M0 verifies. If you use Bambi directly for a
richer formula, the M0 checks are the template for what to verify before believing
its posterior: design matrix and level order, log density against a reference,
prediction semantics for new levels.

## A program longer than one session

For a modelling program that will run for weeks with an agent doing most of the
typing, lay down the scaffold once (`bin/autobayes-init <project>`) and follow its
`AGENTS.md`. The parts that do the work:

1. **Research before code.** `reference/bibliography.md` and `reference/diagnostics.md`
   first; [USED] entries are provenance, [STANDARD] entries are recommendations.
   Findings go to `research/`, concepts to `wiki/concepts/`.
2. **Scope.** Data provenance and column dictionary in `wiki/data/`; the quantity of
   interest and the success thresholds written down before the first fit.
3. **Decide.** No `DECISIONS.md` entry, no job. Each entry: what and why, the named
   diagnostic and threshold, the kill criterion, the next action on failure.
   `python3 scripts/new_decision.py "<title>"` refuses an empty field. Read
   `wiki/hypotheses/ruled-out.md` before proposing anything.
4. **Build.** Clone scripts, never edit a working one in place. `cluster/preflight.py`
   before any queued job. Record the job ID the moment it exists. On completion, fill
   in the entry's Result, write the `wiki/models/` page, append to `wiki/log.md`.

Do not write findings into `reference/`; it is a fixed input.

## Symptom routing

| Symptom | Read |
| --- | --- |
| A library is behaving in a way that makes no sense | `reference/libraries.md` |
| Results changed after a version upgrade | `reference/libraries.md` (version boundaries) |
| Divergences concentrated where a scale is small | `reference/diagnostics.md` (funnel), `failure-atlas.md` |
| High R-hat, chains stable in separate places | `reference/failure-atlas.md` (multimodality) |
| Tree depth saturating | `reference/diagnostics.md` |
| Job dies seconds in on a missing shared library | `reference/failure-atlas.md` (environment) |
| Imports fine, fails on first GPU operation | `reference/failure-atlas.md` (CUDA/driver mismatch) |
| Gradient returns `nan` or zero | `reference/failure-atlas.md` (implementation) |
| JAX will not differentiate a primitive | `reference/pymc-to-jax.md`, then `autobayes-custom-op` |
| Fast on CPU, impossibly slow on GPU | `reference/pymc-to-jax.md` (unjaxified node) |
| Elementwise op fails on a long loop | `reference/failure-atlas.md` (vectorize with `cumsum`) |
| A port disagrees with the R/Stan original | `reference/failure-atlas.md` (comparison) |
| Predictions wrong while posteriors match | `reference/brms-to-pymc.md` (unseen levels); step 5 above |
| GPU backend no faster than CPU | `reference/acceleration.md`, `failure-atlas.md` |
| Choosing a sampler, or quoting a speedup | `reference/acceleration.md` (claims audit) |
| R-hat stuck above 1.01 with many short chains | `reference/acceleration.md` (nested R-hat) |
| Divergences that ignore reparameterization | `reference/failure-atlas.md` (float32) |
