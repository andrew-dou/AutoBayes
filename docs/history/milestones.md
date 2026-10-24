# Library milestones M0 to M3

The library's milestone record before the brms-parity loop (September 2026), moved here verbatim from the README on
17 October 2026: only the link targets were re-pointed for this file's location. The current state of the library is
the [README](../../README.md); where this record calls a construct unsupported, the README supersedes it.

---

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

See the [current capability and test map](../current-capabilities.md) for the
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
| Multiple numeric-by-categorical interactions, shared or disjoint sides; existing families, intercept controls and multiple groups | deterministic checks; one Gaussian CPU example independently accepted | same deterministic coverage and Gaussian CPU example | **pending** | M3a-3, merged (PR #12); [evidence summary](../current-capabilities.md#validation-boundaries) |
| Population-only models, no group-level term: `y ~ 1`, `y ~ x`, `y ~ x + z + x:z`, `k \| trials(n) ~ x`, all three families | deterministic density/gradient/design/prediction checks for all three families; Gaussian example CPU sampling accepted after explicit finite-statistic policy correction ([review](m3c1-decision.md#independent-acceptance-review-2026-09-19)) | same deterministic coverage; Gaussian example CPU sampling and seven-mean agreement accepted in the same review | **pending** | M3c-1, PR #9; merged; one Gaussian development example, not calibration |
| One independent varying slope `(1 + x \|\| g)` | under independent review, not merged | same | same | M3b-1, PR #8 open (superseded: A01, #29, merged; PR #8 closed) |
| No population intercept: `0 + x`, `x - 1`, categorical and one interaction, optional group | three-family deterministic checks; one Gaussian CPU example accepted at `target_accept=0.99` | same deterministic coverage and Gaussian CPU acceptance | **pending** | M3c-2, PR #10; [review and preserved initial failures](m3c2-decision.md#independent-acceptance-review-2026-09-20-utc) |
| Multiple independent intercepts `(1 \| county) + (1 \| year)`, all three families | three-family deterministic checks; one crossed Gaussian CPU example independently accepted | same deterministic coverage and Gaussian CPU acceptance | **pending** | M3c-3, merged (PR #11); [decision](m3c3-decision.md) |
| Fixed-nu Student-t, identity link | deterministic graph/density/gradient/prediction checks | deterministic CPU JAX checks; recovery sampling pending | **pending** | merged (PR #14); [semantics](../student-responses.md) |
| Poisson, log link | deterministic graph/density/gradient/prediction checks | deterministic CPU JAX checks; recovery sampling pending | **pending** | merged (PR #15); [semantics](../poisson-responses.md) |
| Correlated effects, grouping `/` or `:`, varying slopes, additional families and links | refused with an error naming the construct | same | same | later slices, `docs/coordination/NEXT_MILESTONES_PROPOSAL.md` (superseded: supported since Tier A, see Current status) |

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
higher-order interactions and transforms remain unsupported (superseded: two-way interactions of every kind and
the transforms in `docs/transforms.md` are supported since A04/A05; three-way interactions are still refused). Changing a reference
can change independent-prior meaning; the library does not transport priors.
Unseen categorical levels are always refused, independently of any new-group policy.
See the build-only [example](../../examples/m3a3_multiple_interactions.py) and
[decision/test matrix](m3a3-decision.md) and [execution evidence](m3a3-evidence/README.md).
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
refused. (Superseded: `||`, `/` and `:` are supported since A01/A03;
correlation IDs and multimembership are still refused.) See the build-only [example](../../examples/m3c3_multiple_intercepts.py) and
[decision](m3c3-decision.md). One crossed Gaussian CPU example passed independent
validation on PyMC and NumPyro at `target_accept=0.99`; see the [evidence and limits](m3c3-evidence/cpu-validation/README.md).
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
[the decision and test map](m3c2-decision.md) and the
[build-only example](../../examples/m3c2_no_intercept.py). Its Gaussian model passed a bounded
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


Fit comparisons now require a [frozen target identity](../record-identity.md). Historical records remain readable; metadata-only comparison requires explicit `allow_legacy=True` and is labeled unverified.

New results carry [fit provenance](../fit-binding.md) through saved draws. Record creation rejects mismatched or unbound fits unless the documented unverified legacy mode is explicitly selected.

### Prior-predictive checks

Use `model.prior_predictive(draws=100, seed=42)` for graph-native CPU forward draws before fitting. See [semantics and validation](../prior-predictive.md).

Use `model.predict(fit, kind="linear")` for draws before inverse-link transformation; see [linear prediction](../linear-prediction.md).

Set `include_group_effects=False` in predictions to set all group contributions to zero; see [semantics](../group-exclusion.md). This does not marginalize over groups.
