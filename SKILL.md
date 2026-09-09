---
name: autobayes
description: >
  Set up and run a long Bayesian model-iteration program with a coding agent in the
  loop — hierarchical models, MCMC/HMC/NUTS, PyMC or Stan or NumPyro, fitted on a GPU
  or cluster you pay for. Use when starting a new Bayesian modelling project, porting a
  brms/Stan model to PyMC, scaffolding a research repo that needs to survive many
  sessions, or when an existing project keeps re-proposing approaches it already ruled
  out. Also covers pre-run decision gates, MCMC diagnostic thresholds, and cluster
  submission for sampling jobs.
---

# AutoBayes

Scaffolding and pre-research for a Bayesian model-iteration program that runs longer
than any context window.

Two halves. **`reference/`** is background that ships populated — read it instead of
rediscovering it. **`scaffold/`** is what gets copied into the user's project, where
that project's own knowledge accumulates.

## Before anything else

If the project already has a `wiki/` and a `DECISIONS.md`, **it is initialized** — do
not re-run init. Go to `AGENTS.md` in the project root and follow it. It supersedes
this file for anything project-specific.

If not, and the user is starting Bayesian modelling work, run the four phases below.

## Phase 1 — Research

Do not write model code yet. The cost of this phase is small and the cost of skipping
it is a month of work in a direction the literature already closed.

1. Read `reference/bibliography.md`. Note the tier labels: **[USED]** entries are
   provenance, **[STANDARD]** entries are recommendations. Do not report a
   **[STANDARD]** entry as something this project has verified.
2. Read `reference/diagnostics.md`. You will need the thresholds in Phase 3.
3. If porting from R: read `reference/brms-to-pymc.md`.
4. If sampling on GPU: read `reference/pymc-to-jax.md`.
5. Search for work specific to *this* problem — the pack covers the fixed cost of the
   field, not the user's domain.
6. Write findings to `research/01_<topic>.md` and seed `wiki/concepts/` with one page
   per core technical concept.

End state: the user can name the modelling approach, the main risk, and the prior art.

## Phase 2 — Scope

1. **Data provenance** → `wiki/data/provenance.md`. Where every input came from, which
   file is canonical, what has been derived from what. Do this before modelling; it is
   painful to reconstruct and it is the page most often needed later.
2. **Column dictionary** → `wiki/data/dictionary.md`. Every column, its meaning, its
   range, whether it is a predictor, an identifier, or vestigial.
3. **Quantity of interest.** What is this model *for*? A model built without a stated
   QoI gets evaluated on whatever is convenient, which is usually in-sample fit.
4. **Success criteria.** The diagnostics and the calibration target, with thresholds,
   written down now.

## Phase 3 — Decide

**No entry in `DECISIONS.md`, no job.**

`python3 scripts/new_decision.py "<title>"`. Every entry answers:

1. **What** are we changing, and **why** should it help?
2. **How will we know it worked?** A named diagnostic with a threshold, chosen before
   the run. Not "we will look at the output."
3. **What is the kill criterion?** The outcome that means this direction is dead.
4. **What do we do if it fails?** A concrete next action, not "try something else."

Writing (2) before the run regularly reveals that the run as specified cannot produce
the diagnostic. That is much cheaper to discover on a login node.

Before proposing anything, read `wiki/hypotheses/ruled-out.md`. Do not revisit what is
on it. If you believe an entry deserves another look, say so explicitly and give the
reason the original evidence no longer applies.

## Phase 4 — Build and prepare

1. Write the model. **Clone, don't edit in place** — a working script gets copied with
   a version suffix, never edited.
2. `python3 cluster/preflight.py` — builds the model, takes one log density and one
   gradient, asserts both finite, prints the free parameter count. Thirty seconds on a
   login node beats an hour in the queue followed by a seven-second failure.
3. Copy `cluster/slurm_gpu_fit.sbatch`, fill every `CHANGEME`, submit, record the job
   ID immediately.
4. On completion: fill in the entry's **Result**, write the `wiki/models/` page, append
   to `wiki/log.md`, register the run.

## The library

If the model fits the M0 family — a Gaussian regression with numeric predictors and one
varying intercept, `y ~ x + (1 | group)` — use `autobayes/` rather than hand-writing the
PyMC model: `ab.gaussian(formula, data, priors)` builds an inspectable PyMC model through
Bambi with explicit priors, `.fit(backend=..., device=...)` samples on the CPU or through
JAX/NumPyro with an explicit device, and `.predict(...)` returns predictive draws with
named new-group semantics. `examples/m0_varying_intercept.py` is the worked example and
`docs/m0-decision.md` states what the layer adds over Bambi. Outside that family the
library refuses the formula; write the model out per `reference/brms-to-pymc.md`.

## Symptom routing

When something goes wrong mid-project, match the symptom and read the file. Do not
re-derive these.

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
| JAX will not differentiate a primitive | `reference/pymc-to-jax.md` — **and see below** |
| Fast on CPU, impossibly slow on GPU | `reference/pymc-to-jax.md` (unjaxified node) |
| Elementwise op fails on a long loop | `reference/failure-atlas.md` (vectorize with `cumsum`) |
| A port disagrees with the R/Stan original | `reference/failure-atlas.md` (comparison) |
| Predictions wrong while posteriors match | `reference/brms-to-pymc.md` (unseen levels) |
| GPU backend no faster than CPU | `reference/acceleration.md`, `failure-atlas.md` |
| Choosing a sampler, or quoting a speedup number | `reference/acceleration.md` (claims audit) |
| R-hat stuck above 1.01 with many short chains | `reference/acceleration.md` (nested R̂) |
| Divergences that ignore reparameterization | `reference/failure-atlas.md` (float32) |

**On custom operations specifically:** if the likelihood cannot be expressed with
built-in distributions, or will not differentiate, the answer is often a custom
PyTensor `Op` with a JAX implementation — but it is the *last* answer, not the first.
The companion skill `autobayes-custom-op` gates that decision and gives the pattern.
Invoke it rather than improvising.

## Writing things down

The rule that matters most in practice: **write to the wiki before context runs out,
not at the end of the session.** An agent that waits until it is done loses everything
to compaction.

After any significant finding, write it down immediately. Before any long-running
operation, flush what you know. If you have been working a while and have not written
to the wiki, stop and do it now.

If you learned it this session and it is not in the wiki, it did not happen.

Do not write findings into `reference/` — that is a fixed input, not a working
surface. Project knowledge goes in `wiki/` and `research/`.
