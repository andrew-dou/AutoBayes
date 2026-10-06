# AutoBayes

Pre-research and scaffolding for running a long Bayesian model-iteration program with a
coding agent in the loop, on hardware you pay for by the hour.

Three parts:

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
  posterior and predictive draws. Seven families on one formula subset
  `y ~ x + z + x:z [+ (1 | group)]` with numeric and declared categorical predictors (M3a-1)
  and distinct numeric-by-categorical interactions with their main effects (M3a-3); varying intercepts are optional
  and may have multiple independent grouping factors (M3c-3), and a formula without it is a population-only model:
  Gaussian (M0), Bernoulli with a logit link for a 0/1
  outcome (M1), binomial counts `successes | trials(n)` with a logit link (M2),
  and [Student-t responses](docs/student-responses.md) with explicit fixed finite `nu > 1`
  and a scale prior, plus [Poisson count responses](docs/poisson-responses.md),
  [negative binomial counts](docs/negbinomial.md) and [Gamma responses](docs/gamma.md) with a log link.
  Student-t, Poisson, negative binomial and Gamma have deterministic CPU/JAX checks; new recovery fits and
  GPU execution are not yet validated;
  anything else is refused with an error naming the construct. See
  [Library](#library-milestones-m0-m1-and-m2) below.

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
git clone https://github.com/andrew-dou/AutoBayes && cd AutoBayes
./install.sh
```

Symlinks the skills into `~/.claude/skills/` so `git pull` keeps them current.
`--copy` pins them instead; `--uninstall` reverses it. The Python library and its tests
are installed separately from the lock file; see [Install and test](#install-and-test-the-pinned-tested-stack).

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

## Library (milestones M0, M1, M2, M3a-1, M3a-2, M3a-3, M3c-1, M3c-2 and M3c-3)

### Install and test (the pinned, tested stack)

`pyproject.toml` pins the versions that the fit records under `docs/` name (Python 3.11;
pymc 5.28.5, pytensor 2.38.3, bambi 0.17.2, formulae 0.6.2, arviz 0.23.4, jax/jaxlib 0.10.2,
numpyro 0.21.0, numpy 2.4.6). `uv.lock` was frozen on 2026-09-10 from the environment that
produced the foundation results; the earlier M0 and M1 records did not record transitive
versions, so the lock is evidence for the new environment, not a reconstruction of the old
one. Anything else is untested.

```bash
git clone https://github.com/andrew-dou/AutoBayes && cd AutoBayes
uv sync --locked --extra jax --extra dev                 # creates .venv from uv.lock; nothing is resolved anew
uv run --no-sync python -c "import autobayes as ab; print(ab.format_environment(ab.environment()))"
uv run --no-sync pytest tests -q -m "not slow"           # fast checks, a few minutes
uv run --no-sync pytest tests -q -m slow                 # stochastic validation; inspect the current bounded run plan first
uv run --no-sync python examples/m0_varying_intercept.py --backend numpyro --device cpu --record fits/m0.json
```

If `~/.pytensor` is not writable (sandboxes), set `PYTENSOR_FLAGS=base_compiledir=<writable dir>`.
Each test run allocates its own directory `evidence-runs/<UTC time>-<git head>-p<pid>-<token>/`
(gitignored; `run.json` records the environment and commit; files are written once, never
replaced). `AUTOBAYES_EVIDENCE_DIR` names another *parent* for such directories (CI passes
one); anything that resolves into `docs/` or into an existing run is refused before a byte is
written, so reruns and retries cannot touch the committed `docs/m*-evidence/` history.

### Supported paths

See the [current capability and test map](docs/current-capabilities.md) for the
September 27 merged baseline, family extensions, provenance checks and remaining gaps.

"CPU-validated" means the checks in `docs/m0-decision.md` / `docs/m1-decision.md` passed on
the pinned stack: the fast suite in hosted CI (`.github/workflows/ci.yml`, every push and
pull request) and the three sampling checks locally in the locked environment (2026-09-10;
`docs/coordination/STATUS.md` records each run). The manual slow workflow (`slow.yml`)
exists; its execution status is recorded in STATUS, not assumed here. "GPU-validated" means
one executed run on a GPU matched the CPU posterior. Nothing here is a claim about other
versions.

| Family, formula | PyMC NUTS, CPU | JAX/NumPyro, CPU | JAX/NumPyro, GPU | Status |
| --- | --- | --- | --- | --- |
| Gaussian, `y ~ [1 +] x… + (1 \| g)` | CPU-validated | CPU-validated | GPU-validated once, H100 (`docs/m0-gpu-validation.md`) | M0, merged |
| Bernoulli (logit), `y ~ [1 +] x… + (1 \| g)`, y strictly 0/1 | CPU-validated | CPU-validated | **pending** (same code path, not run on a GPU) | M1, merged (PR #3) |
| Binomial (logit), `k \| trials(n) ~ [1 +] x… + (1 \| g)`, integer counts up to 1e6 per row | CPU-validated | CPU-validated | **pending** (same code path, not run on a GPU) | M2 |
| Declared categorical predictors `z` (treatment coding, `categorical={"z": reference}`), all three families | CPU-validated (`docs/m3a1-decision.md`) | CPU-validated | **pending** | M3a-1, merged (PR #6) |
| One numeric-by-categorical interaction `x:z` (`x*z`) with both main effects, all three families | CPU-validated (`docs/m3a2-decision.md`) | CPU-validated | **pending** | M3a-2, merged (PR #7) |
| Multiple numeric-by-categorical interactions, shared or disjoint sides; existing families, intercept controls and multiple groups | deterministic checks; one Gaussian CPU example independently accepted | same deterministic coverage and Gaussian CPU example | **pending** | M3a-3, merged (PR #12); [evidence summary](docs/current-capabilities.md#validation-boundaries) |
| Population-only models, no group-level term: `y ~ 1`, `y ~ x`, `y ~ x + z + x:z`, `k \| trials(n) ~ x`, all three families | deterministic density/gradient/design/prediction checks for all three families; Gaussian example CPU sampling accepted after explicit finite-statistic policy correction ([review](docs/m3c1-decision.md#independent-acceptance-review-2026-09-19)) | same deterministic coverage; Gaussian example CPU sampling and seven-mean agreement accepted in the same review | **pending** | M3c-1, PR #9; merged; one Gaussian development example, not calibration |
| One independent varying slope `(1 + x \|\| g)` | under independent review, not merged | same | same | M3b-1, PR #8 open |
| No population intercept: `0 + x`, `x - 1`, categorical and one interaction, optional group | three-family deterministic checks; one Gaussian CPU example accepted at `target_accept=0.99` | same deterministic coverage and Gaussian CPU acceptance | **pending** | M3c-2, PR #10; [review and preserved initial failures](docs/m3c2-decision.md#independent-acceptance-review-2026-09-20-utc) |
| Multiple independent intercepts `(1 \| county) + (1 \| year)`, all three families | three-family deterministic checks; one crossed Gaussian CPU example independently accepted | same deterministic coverage and Gaussian CPU acceptance | **pending** | M3c-3, merged (PR #11); [decision](docs/m3c3-decision.md) |
| Fixed-nu Student-t, identity link | deterministic graph/density/gradient/prediction checks | deterministic CPU JAX checks; recovery sampling pending | **pending** | merged (PR #14); [semantics](docs/student-responses.md) |
| Poisson, log link | deterministic graph/density/gradient/prediction checks | deterministic CPU JAX checks; recovery sampling pending | **pending** | merged (PR #15); [semantics](docs/poisson-responses.md) |
| Correlated effects, grouping `/` or `:`, varying slopes, additional families and links | refused with an error naming the construct | same | same | later slices, `docs/coordination/NEXT_MILESTONES_PROPOSAL.md` |

Device rules on every path: `device="gpu"` without a visible GPU raises `DeviceError`
(no silent CPU fallback); the fit report states the actual device, dtype (float64) and
chain method; if JAX is not installed, `backend="numpyro"` raises `DeviceError` naming
the `[jax]` extra.

### Multiple numeric-by-categorical interactions

Use explicit main effects and distinct pairs, for example
`y ~ x + w + z + v + x:z + w:z + x:v`, with `z` and `v` declared in
`categorical=`. Canonical terms keep first-appearance order, main effects before
interaction pairs, always numeric first. `Priors.b` still accepts one explicit prior
or exact per-term mappings, with optional per-level priors for each vector term.
Variables, coordinates and columns are `x:z`, `x:z_dim` and `x:z[level]`.

The contrast for level `l` of `z` is `b_z[l] + x*b_x:z[l] + w*b_w:z[l]`;
the slope in `x` is `b_x + b_x:z[z_i] + b_x:v[v_i]`, with reference contributions zero.
Other predictors and group effects are held fixed. With centering enabled and a
population intercept, each expanded product column is centered at its own fitted-row mean.
Without one, common centering is disabled, the first categorical main is full-level,
and all interaction terms still encode non-reference slope differences. The full-level
factor contrast uses `b_z[l] - b_z[reference]` as its baseline difference.

Both main effects are required. Strict duplicate rules remain: `x*z + w*z` repeats
`z`; write `x + w + z + x:z + w:z`. Numeric-by-numeric, categorical-by-categorical,
higher-order interactions and transforms remain unsupported. Changing a reference
can change independent-prior meaning; the library does not transport priors.
Unseen categorical levels are always refused, independently of any new-group policy.
See the build-only [example](examples/m3a3_multiple_interactions.py) and
[decision/test matrix](docs/m3a3-decision.md) and [execution evidence](docs/m3a3-evidence/README.md).
The focused checks passed; the one local fast suite was interrupted after a wall-deadline
supervision failure and was not rerun. Hosted checks remain required. New-feature CPU sampling, calibration
and GPU validation remain separate future work.

### Multiple independent varying intercepts

```python
priors = ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1),
                   sd={"county": ab.HalfNormal(1), "year": ab.Exponential(2)},
                   sigma=ab.HalfNormal(1))
model = ab.gaussian("y ~ x + (1 | county) + (1 | year)", df, priors)
```

Every factor needs its own SD prior. A scalar `sd` remains valid for a single factor;
multiple factors require a mapping covering exactly their names. Group effects have
independent Normal priors with separate scales; posterior effects need not be independent.
Use `model.spec.groups`, `levels_by_group`, `model.design.group_designs`, and
`model.group_names`. Factors follow formula order; levels use sorted fitted string labels.
`design.Z` concatenates one-hot blocks, located by `design.group_slices`.
Singular `group`, `levels`, `group_term`, `J` and `group_index` access is ambiguous with
multiple groups and raises. Existing zero/one-group access remains valid.

`value_point(sd={"county": .5, "year": .2}, z={"county": [...], "year": [...]}, ...)`
requires exact keys and one offset per fitted level. Predictions accept a policy string
for every factor or `new_groups={"county": "population", "year": "error"}`.
Unseen effects are shared per factor/label, independent across factors. RNG traversal is
factor formula order then sorted unseen labels; changing factor order can change unseen
draws with the same seed. Per-factor prediction attributes are JSON strings that can be
saved to netCDF. Fit records list every factor and SD; multiple groups omit ambiguous
legacy `group`/`J` entries. Common centering and intercept back-transformation are unchanged.

Distinct additive factors can describe crossed or observationally nested memberships;
the library never expands nesting syntax or combines labels with their parent. Slopes,
duplicate factors, correlation IDs, `||`, grouping `/` or `:` and multimembership remain
refused. See the build-only [example](examples/m3c3_multiple_intercepts.py) and
[decision](docs/m3c3-decision.md). One crossed Gaussian CPU example passed independent
validation on PyMC and NumPyro at `target_accept=0.99`; see the [evidence and limits](docs/m3c3-evidence/cpu-validation/README.md).
Other-family sampling, calibration and GPU validation remain pending.

### Removing the population intercept

Write `y ~ 0 + x` or `y ~ x - 1` and omit the intercept prior:

```python
priors = ab.Priors(b=ab.Normal(0, 2), sigma=ab.HalfNormal(1))
model = ab.gaussian("y ~ 0 + x + arm + x:arm", df, priors,
                    categorical={"arm": "control"})
print(model.describe())
```

The first categorical main effect has one coefficient per level (reference first),
including `arm[control]`. Later categorical main effects use treatment coding. The
interaction still estimates non-reference slope differences. Per-level priors must
cover those actual coefficient labels; reordering factors can change independent
prior meaning. The raw design is used for either `center_predictors` request,
because centering without a free intercept changes the model. Fit records show
requested and effective centering separately. There is no hidden intercept prior,
posterior variable, or back-transformation. Supplying an intercept prior/value is
an error; `value_point(b=..., sigma=...)` omits it.

At least one population predictor is required. Conflicting `1` and intercept-removal
tokens, general subtraction, and group-only/zero-only models remain refused. See
[the decision and test map](docs/m3c2-decision.md) and the
[build-only example](examples/m3c2_no_intercept.py). Its Gaussian model passed a bounded
two-backend CPU check at `target_accept=0.99`; the initial `0.9` run had divergences.
The library default is unchanged. Inspect diagnostics for each fitted model; this one
example does not establish calibration, rare-tail accuracy or GPU coverage.

### Using it

```python
import autobayes as ab
from autobayes.diagnostics import panel, format_panel

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

# a categorical predictor: declare the column and its reference level; one coefficient per other level
cpriors = ab.Priors(intercept=ab.Normal(0, 5), b={"x": ab.Normal(0, 2), "arm": ab.Normal(0, 2)},
                    sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1))     # or "arm": {"low": ..., "high": ...}
amodel = ab.gaussian("y ~ x + arm + (1 | site)", df, cpriors, categorical={"arm": "control"})
amodel.coefficients                                               # (("x","x",None), ("arm[high]","arm",0), ("arm[low]","arm",1))
amodel.predict(afit, newdata=new_df, kind="mean")                 # an arm that was not fitted is refused

# a population-only model (no group-level term): the same priors without sd; predictions need no grouping column
ppriors = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1))
pmodel = ab.gaussian("y ~ x + arm", df, ppriors, categorical={"arm": "control"})    # sd= here would be refused
pmodel.predict(pfit, newdata=new_df_without_site, kind="mean")                   # new_groups does not apply

# a numeric-by-categorical interaction: each arm gets its own dose slope (x*z is x + z + x:z)
ipriors = ab.Priors(intercept=ab.Normal(0, 5), b={"dose": ab.Normal(0, 2), "arm": ab.Normal(0, 2), "dose:arm": ab.Normal(0, 1)},
                    sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1))
imodel = ab.gaussian("y ~ dose + arm + dose:arm + (1 | site)", df, ipriors, categorical={"arm": "control"})
print(imodel.describe())                                          # prints the actual product-column means used for centring

# counts with known denominators: k successes of n trials per row, brms syntax
cmodel = ab.binomial("deaths | trials(patients) ~ x + (1 | clinic)", counts_df, bpriors)
cfit = cmodel.fit(backend="pymc", seed=7)
cmodel.predict(cfit, kind="probability")                          # p only, no trials needed
cmodel.predict(cfit, newdata=new_counts, kind="mean")             # p and expected = n p (needs the trials column)
cmodel.predict(cfit, newdata=new_counts, kind="response")         # ... and integer Binomial(n, p) draws
```

`SKILL.md` is the researcher workflow for these families, step by step with the checks,
failure recovery and the session handoff. What Bambi supplies and what this layer adds,
with the reason for each addition, is in `docs/m0-decision.md` (Gaussian) and
`docs/m1-decision.md` (Bernoulli, including why its likelihood is written by hand: Bambi's
built-in one is floored at machine epsilon for large logits, and the obvious PyMC and
PyTensor alternatives have NaN gradients on one path or the other) and `docs/m2-decision.md`
(binomial: accepted data policies, the log-sigmoid density chosen after measuring the
built-ins, the high-precision reference and its error budget, the validated count and
logit domain) and `docs/m3a1-decision.md` (categorical predictors: declaration, level
order, reference validation, prior forms, the naming map, what the centred intercept is
with dummies, and what a change of reference level does and does not preserve) and
`docs/m3a2-decision.md` (the interaction: canonical expansion, both main effects required,
expanded-column centring and the reported-intercept transformation, the reference change
with slopes) and `docs/m3c1-decision.md` (population-only models: the optional group term,
the `sd` prior refused without one and required with one, `new_groups` not applicable, the
deterministic checks and the independently reviewed two-backend Gaussian CPU example); the acceptance checks and
their evidence are in those records and `docs/m0-evidence/`, `docs/m1-evidence/`,
`docs/m2-evidence/`, `docs/m3a1-evidence/`, `docs/m3a2-evidence/`, `docs/m3c1-evidence/`.
`docs/coordination/STATUS.md` is the live milestone status; the roadmap is
`docs/coordination/NEXT_MILESTONES_PROPOSAL.md`. No speed claim is made anywhere.

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

Fit comparisons now require a [frozen target identity](docs/record-identity.md). Historical records remain readable; metadata-only comparison requires explicit `allow_legacy=True` and is labeled unverified.

New results carry [fit provenance](docs/fit-binding.md) through saved draws. Record creation rejects mismatched or unbound fits unless the documented unverified legacy mode is explicitly selected.

### Prior-predictive checks

Use `model.prior_predictive(draws=100, seed=42)` for graph-native CPU forward draws before fitting. See [semantics and validation](docs/prior-predictive.md).

Use `model.predict(fit, kind="linear")` for draws before inverse-link transformation; see [linear prediction](docs/linear-prediction.md).

Set `include_group_effects=False` in predictions to set all group contributions to zero; see [semantics](docs/group-exclusion.md). This does not marginalize over groups.
