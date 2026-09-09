# AutoBayes

Pre-research and scaffolding for running a long Bayesian model-iteration program with a
coding agent in the loop, on hardware you pay for by the hour.

Two halves:

- **`reference/`** — a pre-research pack that ships populated. The fixed cost of
  entering the field, paid once: sampler theory and its citations, how each library in
  the stack actually works, `brms`→PyMC porting, the JAX/NumPyro GPU path, diagnostic
  thresholds, and a failure atlas keyed by symptom.
- **`scaffold/`** — what gets laid down in your project: an agent contract, a decision
  gate, a compiled wiki, a `research/` tier, and cluster templates with the expensive
  failures already encoded.

- **`autobayes/`** — the library: a brms-like Python interface that builds an
  inspectable PyMC model through Bambi, requires explicit priors, samples on the CPU
  with PyMC's NUTS or through JAX/NumPyro with an explicit device choice, and returns
  posterior and predictive draws. Two families on one formula subset
  `y ~ x + (1 | group)`: Gaussian (M0) and Bernoulli with a logit link for a 0/1
  outcome (M1); anything else is refused with an error naming the construct. See
  [Library](#library-milestones-m0-and-m1) below.

It came out of a multi-month hierarchical-modelling program that fit roughly sixty
successive model versions on GPU, each conditional on the last. The modelling was the
research; this is what made the modelling survivable once an agent was doing most of
the typing.

## The problem

An agent on a long research codebase has three failure modes that compound.

**It forgets.** Context is finite. By week three the agent no longer knows that AR(1)
on the year slopes was tried and abandoned, so it proposes AR(1) on the year slopes.

**It is confidently wrong about the past.** Asked "did we test this?", an agent with no
durable record reconstructs a plausible answer. Plausible is worse than absent, because
you act on it.

**It burns real money doing it.** A wrong batch script fails seven seconds into a
queued GPU allocation, and you find out an hour later.

And before any of that, there is a fourth cost that is not a failure mode at all, just
waste: **every project pays the same entry fee.** An agent starting cold spends its
first sessions on literature discovery and API archaeology — which sampler, what the
diagnostics mean, how a `brms` formula becomes a PyMC model, why JAX will not
differentiate the likelihood. That work is high-variance and it is not your research.
Two agents asked the same question return different papers and different confidence,
and you pay full token price each time for an answer that was already known.

`reference/` is the answer to the fourth. The scaffold is the answer to the first three.

## Install

```bash
git clone https://github.com/andrew-dou/autobayes && cd autobayes
./install.sh
```

Symlinks the skills into `~/.claude/skills/` so `git pull` keeps them current.
`--copy` pins them instead; `--uninstall` reverses it.

Or use it without installing:

```bash
bin/autobayes-init /path/to/project
```

## Use

Ask your agent to set up a Bayesian modelling project, and it runs four phases:

**Research** → read `reference/`, search for prior art specific to your problem, write
`research/01_*.md`, seed `wiki/concepts/`. No model code yet.

**Scope** → data provenance and column dictionary into `wiki/data/`; state the quantity
of interest and the success criteria, with thresholds.

**Decide** → a `DECISIONS.md` entry. No entry, no job.

**Build** → model, preflight, batch script, submit.

Mid-project, the skills are symptom-triggered. When a gradient comes back `nan` or a
likelihood will not differentiate, `autobayes-custom-op` loads on its own and — before
showing you the pattern — tries to talk you out of needing it.

## Library (milestones M0 and M1)

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -e ".[jax,dev]" "matplotlib<3.11"
.venv/bin/python examples/m0_varying_intercept.py --backend numpyro --device cpu
```

```python
import autobayes as ab

priors = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2),
                   sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1))
model = ab.gaussian("y ~ x + (1 | school)", df, priors)
print(model.describe())          # the density that is actually sampled, centring included
model.pymc_model                 # the PyMC model; inspect it directly
fit = model.fit(backend="numpyro", device="cpu", chains=4, seed=7)   # or backend="pymc"; device="gpu"
print(fit.report())              # backend, requested and actual device, dtype, chain method, seed
print(format_panel(panel(fit.idata)))                       # divergences, R-hat, ESS, E-BFMI, tree depth
draws = model.predict(fit, newdata=new_df, new_groups="population")
rec = ab.fit_record(model, fit)                             # one JSON record per fit, for the handoff
ab.compare_records(rec, earlier_rec)                        # same posterior? (4 combined MCSE)

# binary outcome nested in groups: Bernoulli likelihood, logit link, logit-scale priors, no sigma
bpriors = ab.Priors(intercept=ab.Normal(0, 1.5), b=ab.Normal(0, 1), sd=ab.HalfNormal(1))
bmodel = ab.bernoulli("y ~ x + (1 | clinic)", df01, bpriors)   # y must be exactly 0/1
bfit = bmodel.fit(backend="pymc", seed=7)
probs = bmodel.predict(bfit, newdata=new_df, kind="mean")        # p = P(y = 1)
draws = bmodel.predict(bfit, newdata=new_df, kind="response", new_groups="population")  # p and 0/1 y
```

`SKILL.md` is the researcher workflow for these families, step by step with the checks,
failure recovery and the session handoff. What Bambi supplies and what this layer adds,
with the reason for each addition, is in `docs/m0-decision.md` (Gaussian) and
`docs/m1-decision.md` (Bernoulli, including the one measured deviation of Bambi's
likelihood from the textbook density at extreme logits); the acceptance checks and their
evidence are in those records and `docs/m0-evidence/`, `docs/m1-evidence/`.
`docs/coordination/STATUS.md` is the live milestone status. GPU execution has been run
once on an H100 for the Gaussian family and matched to the CPU posterior
(`docs/m0-gpu-validation.md`); for the Bernoulli family it is pending. No speed claim is
made.

## The decision gate

**Before submitting any job that costs real compute, append an entry to
`DECISIONS.md`.**

Four questions: what and why; how we will know it worked, with a threshold chosen
beforehand; the kill criterion that would end this direction; and the concrete next
action if it fails.

It is a forcing function, not bureaucracy. Writing "how will we know it worked" before
the run regularly reveals that the run as specified *cannot produce that diagnostic* —
the output it saves does not contain the quantity, or the comparison needs a baseline
nobody has fit. That costs nothing to discover on a login node and half a day in the
queue. In the original program this changed the run more often than it changed the
record.

The separate kill criterion exists because "try something else" is how a research
program becomes a random walk.

`scripts/new_decision.py` refuses to write an entry with an empty field.

## Two kinds of memory

`wiki/` is compiled state for an agent to retrieve: short pages, cross-referenced,
edited in place, always current. `research/` is long-form and dated: the investigation
with its reasoning intact, never revised — superseded.

Using one for the other's job degrades both. A real investigation compressed into a
wiki page loses the reasoning that made it checkable; an investigation left only in
`research/` is never found.

Raw sources — model code, logs, data, fitted objects — are read-only to the agent.

The highest-value file in either is `wiki/hypotheses/ruled-out.md`. The knowledge that
matters most in a long program is negative: the twelve parameterizations that did not
fix the funnel, the covariate indistinguishable from zero, the sampler setting that was
a measurement artifact. Negative results are exactly what an agent will not
reconstruct, and `AGENTS.md` makes reading that file a precondition for proposing
anything.

The rule that matters most in practice: **write to the wiki before context runs out,
not at the end of the session.**

## What is not here

No workflow engine, no DAG, no scheduler abstraction, no tracker. Every layer of that
sits between the agent and the thing that failed, and the debugging loop is long
enough. The templates are shell scripts you can read in one screen; the scripts are
single files with no dependencies outside the standard library.

This is also not MLflow or W&B. Those record what a run did. This is upstream: it
constrains what gets run, and keeps what was learned somewhere an agent will find it.

## Honest limits

- The `reference/` pack is opinionated where one program had evidence and thin where it
  did not. Thresholds are conventions, not laws; check them against your model.
- `bibliography.md` tags every entry **[USED]** (in that program's written record, with
  the result taken from it) or **[STANDARD]** (the canonical reference for a technique
  used here, recommended but not a provenance claim). Do not report the second kind as
  something this project verified.
- The PyMC/JAX material covers **both PyMC 5 and 6**, because the sampling API differs
  between them and the published docs are not internally consistent about their own
  version. `pymc-to-jax.md` gives a two-second runtime check to determine which you are
  on, and a snippet for each. Verified against an installed 5.28.2. The porting and
  diagnostic material is stack-agnostic.
- The custom-`Op` pattern in `pymc-to-jax.md` is **executed, not just written** —
  `reference/examples/custom_op_verification.py` runs it end to end against `scipy`,
  finite differences, both PyTensor backends, and a live PyMC model. Running it is also
  how the quadrature accuracy limit documented there was measured.
- Where official documentation does not exist, the pack says so rather than guessing.
  `bibliography.md` ends with six such gaps — including that no systematic brms→PyMC
  mapping reference exists anywhere, which is most of the reason this pack does.
- Numbers are traced to their own tables. `acceleration.md` carries a claims audit of
  the published sampler speedups, most of which shrink sharply or turn out to measure
  something else — a GPU-versus-CPU comparison, a lattice-physics benchmark, a toy
  target, or warmup charged to one side only. On realistic hierarchical models the
  honest figure for modern samplers versus NUTS is about 2×.
- No third-party PDFs are redistributed here — citations and summaries only.

## Licence

MIT. The `wiki/` pattern is adapted from
[Karpathy's LLM Wiki gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).
