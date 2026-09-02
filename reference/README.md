# Reference: the pre-research pack

Curated, pre-digested background for a Bayesian model-iteration program run with a
coding agent. It ships populated. You read it instead of rediscovering it.

## Why this exists

An agent starting a GPU-sampling project cold spends its first several sessions doing
literature discovery and API archaeology: which sampler, what the diagnostics mean,
how a `brms` formula becomes a PyMC model, why the JAX backend refuses to
differentiate the likelihood. That work is high-variance. Two agents on the same
question return different papers, different API idioms, and different confidence, and
you pay full token price each time for an answer that was already known.

None of it is your research. It is the fixed cost of entering the field, and it is the
same fixed cost for everybody. This directory pays it once.

The organising principle is that a compact, opinionated digest beats a pile of source
material. A PDF of a paper costs an agent thousands of tokens to read and yields a
summary it will lose at the next compaction. Three sentences saying what the paper
establishes, which number you should take from it, and which decision it should change
costs almost nothing and survives.

## What is here

| File | What it settles |
| --- | --- |
| `bibliography.md` | The papers worth reading, what each one establishes, and which are load-bearing. Pointers and summaries only — no redistributed PDFs. |
| `brms-to-pymc.md` | Translating a `brms`/Stan formula model into explicit PyMC. Priors, censoring, missing data, distributional parameters, post-processing. |
| `pymc-to-jax.md` | Getting that model onto a GPU through NumPyro, and writing a custom differentiable likelihood when the built-in one will not do. |
| `diagnostics.md` | What to compute after a fit, what threshold to hold it to, and what each failure implies about the model. |
| `failure-atlas.md` | Failure modes with their signatures — the funnel, multimodality, and the environment errors that kill a job seconds into an allocation. |

## How to use it

Read `bibliography.md` and `diagnostics.md` before the first model. Read
`brms-to-pymc.md` if you are porting an existing R model. Read `pymc-to-jax.md` before
the first GPU run, and again the first time a gradient comes back `nan`. Treat
`failure-atlas.md` as a lookup: match the signature you are seeing, then read the
entry.

Everything here is background that does not change as your project advances. Your
project's own accumulating knowledge goes in `wiki/`, and your decisions go in
`DECISIONS.md`. Do not write findings back into this directory — it is a fixed input,
not a working surface.

## Provenance and honesty

This pack was written out of one multi-month hierarchical-model program that fit
roughly sixty successive model versions on GPU. That is its strength and its limit: it
is opinionated where that program had evidence, and thin where it did not.

Claims that came from a source are attributed. Claims that came from experience on one
codebase are marked as such. Where a number is quoted from the literature rather than
re-derived, `bibliography.md` says so. Nothing here is a substitute for checking a
threshold against your own model.
