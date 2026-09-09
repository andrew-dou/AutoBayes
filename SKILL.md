---
name: autobayes
description: >
  Fit and check a Bayesian hierarchical regression in Python the way a careful
  researcher would: brms-style formula `y ~ x + (1 | group)`, explicit priors, an
  inspectable PyMC model built through Bambi, NUTS on the CPU or through JAX/NumPyro on
  an explicitly chosen GPU, a diagnostic panel with fixed thresholds, predictive draws
  with named new-group semantics, and a fit record that a later session can resume
  from. Use when someone asks for a multilevel / mixed-effects / varying-intercept /
  random-intercept model, a brms or lme4 formula in Python, PyMC or NumPyro sampling on
  a GPU, help reading divergences / R-hat / ESS, or to set up a Bayesian modelling
  project that must survive many sessions. Milestone M0 covers one model family; the
  skill says what is verified and routes everything else to the reference pack.
---

# AutoBayes

One library (`autobayes/`), one reference pack (`reference/`), one scaffold
(`scaffold/`) for long programs. This file is the researcher workflow for the verified
model family, then the routing for everything else.

**What is verified (milestone M0).** A Gaussian regression with numeric predictors and
one varying intercept:

    y ~ [1 +] x1 [+ x2 ...] + (1 | group)

with explicit priors on the intercept, every slope, the group SD and the residual SD.
The log density of the Bambi-built PyMC model equals an independent reference at fixed
points (with and without the Jacobian, and through the JAX compilation NumPyro
samples), the gradient equals finite differences, the Hessian equals the closed-form
precision when the scales are fixed, NUTS reproduces an exact conditional Gaussian
posterior, and a simulated recovery passes the panel on both backends and both
devices with the two posteriors agreeing within Monte Carlo error. Evidence:
`docs/m0-evidence/`, checks A1-A8, B1-B2, C1-C3 in `docs/m0-decision.md`.

**What is not verified.** Every other formula. The library refuses them with an error
naming the construct. Do not loosen the parser to get past that error; go to
"Outside the verified family" below.

## Session start, every time

1. If the project has `docs/coordination/STATUS.md` or a `wiki/`, read it before
   touching code. It records what was fit, with which seed, and what is pending.
2. Print the environment and keep the output for the handoff:

```bash
.venv/bin/python -c "import autobayes as ab; print(ab.format_environment(ab.environment()))"
```

   Verified stack: Python 3.11, pymc 5.28.5, pytensor 2.38.3, bambi 0.17.2, formulae
   0.6.2, arviz 0.23.4, numpyro 0.21.0, jax 0.10.2. Install:

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -e ".[jax,dev]" "matplotlib<3.11"
```

3. Three environment traps seen on this stack, and the fix for each:
   - `~/.pytensor` not writable (sandboxes, shared accounts): every model build fails
     at compile time. Set `PYTENSOR_FLAGS=base_compiledir=<writable dir>`.
   - arviz 0.23 fails to import with matplotlib 3.11.1. Pin `matplotlib<3.11`.
   - JAX touched before `import autobayes` (any `jax.devices()` call, any JAX import
     that initialises the backend): CPU chains on the NumPyro path then run
     sequentially. `fit()` says so in its notes and reports `chain method : sequential`.
     Import autobayes first, or accept the wall time. The posterior is unaffected.

## The workflow

Run the steps in order. Each has a check; a step whose check fails is not done.

### 1. Data

A pandas data frame with a numeric response, numeric predictors, and a grouping
column (any dtype; it is coded as a factor with levels in **sorted** order). The
library refuses missing values, categorical predictors, a constant predictor, and a
grouping factor with one level, each with a message saying so. Transform columns in
the data frame, not in the formula.

### 2. Formula and priors

```python
import autobayes as ab

priors = ab.Priors(
    intercept=ab.Normal(0, 5),   # applies to the CENTRED intercept: E[y] at the predictor means
    b=ab.Normal(0, 2),           # one prior for every slope, or {"x": ..., "w": ...}
    sd=ab.HalfNormal(1),         # SD of the varying intercepts (brms class "sd")
    sigma=ab.HalfNormal(1),      # residual SD
)
model = ab.gaussian("y ~ x + (1 | school)", df, priors)
```

All four classes are required. There are no defaults, because a default prior is the
thing you cannot see in the formula and cannot reproduce later. Location classes take
`Normal` or `StudentT(nu, mu, sigma)`; scale classes take `HalfNormal`,
`HalfStudentT(nu, sigma)`, `Exponential(lam)` or `HalfCauchy(beta)`. Parameters are
PyMC's: `sigma` is a standard deviation, `lam` a rate, `beta` a scale.

**The centring trap.** Bambi, like brms, centres the predictors and samples the
intercept at the predictor means. Your prior on `intercept` is a prior on that
quantity. The `Intercept` column in the posterior is the raw-scale intercept, back-
transformed after sampling. Choose the prior for the centred quantity (the typical
response), not for "y at x = 0". `center_predictors=False` turns this off if the raw
intercept is what the prior is about.

Check: `print(model.describe())`. Read the `levels` line (index to label map), the
`mu_i =` line, the centring paragraph, and the "PyMC variables sampled" block. That
block is the model that will be sampled: `Intercept` (centred), one variable per
slope, `sigma_log__`, `1|school_sigma_log__` (the group SD, log scale) and
`1|school_offset` (the J standard-normal offsets of the non-centred parameterisation).
`model.pymc_model` is the graph itself; `pytensor.dprint(model.pymc_model["mu"])`
shows the linear predictor if `describe()` is not enough.

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
parameter table. In it, `Intercept` is the raw-scale intercept, `1|school_sigma` is
the group SD, `1|school` are the J group effects `u`, and `1|school_offset` the
standard-normal offsets (ignore them in reporting). The interval columns are the
94 % HDI under arviz 0.23; arviz 1.0 changes the default to an 89 % ETI, so state the
version next to any interval you quote.

### 5. Predictions

```python
ppc = model.predict(fit, kind="response", seed=1)                       # in-sample y draws
mu  = model.predict(fit, newdata=new_df, kind="mean")                    # E[y | x, group]
new = model.predict(fit, newdata=new_df, kind="response", new_groups="population", seed=1)
```

`kind="mean"` returns draws of `mu`; `kind="response"` also returns draws of a new
observation. For a level of the grouping factor that was not in the training data:

- `new_groups="error"` (default): refuse, so an unseen level is never silently treated
  as an observed one.
- `"population"`: draw the level's effect from `Normal(0, sd_group)` for each posterior
  draw. This is the distribution the model fit (brms `sample_new_levels = "gaussian"`)
  and is the right choice for "a new school".
- `"resample_observed"`: reuse a random observed level's fitted effect per draw
  (brms's default `"uncertainty"`, Bambi's `sample_new_groups=True`). It understates
  between-level spread because it resamples shrunk effects from a finite set.

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

The worked example does all six steps and writes a record:

```bash
.venv/bin/python examples/m0_varying_intercept.py --backend numpyro --device cpu \
    --record fits/example_cpu.json
```

## Recovering from failures

| Symptom | Cause | Do |
| --- | --- | --- |
| `FormulaError: ... not supported in M0` | Formula outside the verified family | Not a bug. Use the family, or write the model by hand: "Outside the verified family" |
| `PriorError: prior for 'sd' must be a positive-support prior` | Wrong class of prior | Location classes for intercept and slopes, scale classes for `sd` and `sigma` |
| `DeviceError: ... JAX sees no GPU` | No CUDA jaxlib, or no GPU on this machine | Check `jax.devices()`. On a GPU host install `"jax[cuda12]"`; else `device="cpu"`. Do not "fix" this by catching it |
| `DeviceError: backend='pymc' runs ... CPU only` | GPU asked of the PyMC sampler | `backend="numpyro"` |
| report note: chains sequential | JAX initialised before autobayes import | Reorder imports; posterior unaffected |
| report `dtype : float32` | x64 disabled after import | The library enables x64 before sampling; if you see float32, something reset `jax_enable_x64`. Refit; do not trust the float32 run |
| `PermissionError` under `~/.pytensor` | Read-only home | `PYTENSOR_FLAGS=base_compiledir=<dir>` |
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
git checkout <branch> && .venv/bin/python -m pytest tests -q        # fast checks
.venv/bin/python -m pytest tests -q -m slow                          # two sampling checks, ~1 min
```

## GPU and remote execution

The reproducible command, and what counts as evidence, is in
`docs/m0-gpu-validation.md`. In short: run the example with
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
