# AutoBayes: next milestones, for Andrew's review

Status: proposal, revision 3, 2026-09-09. Written by the M1 worker session
(`e5628a7d-e572-4bad-9feb-82ee8361c261`) under the planning dispatch
`iqss-research/memory/dispatches/2026-09-09-autobayes-roadmap-planning.txt`, revised under
`2026-09-09-autobayes-roadmap-revision.txt` to address the independent review
`iqss-research/memory/reviews/2026-09-09-autobayes-roadmap-review.md`, and again under
`2026-09-09-autobayes-roadmap-final-clarifications.txt` (acceptance-specification
clarifications only). Section 9 records each review finding, how it was verified, and its
disposition. Nothing in this file is
implemented. Every "existing" claim is traced to code, a test, or a record in this
repository; every "proposed" item is marked as such. Facts about Bambi come from the
installed `bambi 0.17.2` / `formulae 0.6.2` source in `.venv`; facts about brms come from the
brms reference pages fetched on 2026-09-09 (`brmsfamily`, `prepare_predictions`,
`posterior_epred`, `posterior_predict`) and from the upstream `R/posterior_epred.R` file
fetched raw on 2026-09-09 (function `posterior_epred_binomial` at lines 373-375 of the file
as fetched). Statements marked "build-only, worker" are things this session observed by
building a Bambi model without fitting it; they are worker observations, not independent
validation.

Verified starting point: `main` at `925554b` (M0 merged, one H100 agreement run); PR #3
open and clean at `bd63348` (M1 Bernoulli/logit with the stable likelihood, the
unseen-label identity fix and complex-column refusal). No CI. No GPU run of M1.

---

## 1. What the tests actually cover, and what they do not

Collected without running them (`pytest --collect-only`): **62 tests, 59 fast and 3 slow**.
Evidence attribution: the coordinator independently reran the 59 fast tests (59 passed,
`memory/reviews/2026-09-09-autobayes-followup-review.md`); the 3 slow passes are
worker-reported from this session's last run (`docs/coordination/STATUS.md`). The fast
suite is not 59 isolated unit tests: `tests/conftest.py` and `tests/test_bernoulli.py` build
Bambi/PyMC models and run short NUTS fits as fixtures (`small_fit`: 2 chains x 150 draws;
`bfit`: 2 x 200), so about a third of the fast tests are small integration tests that
depend on those fits.

| File (collected) | What it establishes | Kind |
| --- | --- | --- |
| `test_formula.py` (22) | 17 unsupported constructs refused by name; accepted subset; data validation (missing column, categorical predictor, NA, single level); prior class/coverage rules; Gaussian without `sigma` refused; `describe()` content; complex columns refused (Gaussian) | pure unit |
| `test_logp.py` (8) | Gaussian A1-A6: log density with/without Jacobian vs `tests/reference_logp.py` at 5 random points (4 formula/centring variants), gradient vs central differences, JAX-compiled density vs PyTensor, design and level coding, Hessian equals the closed-form precision with scales fixed, maximiser equals the exact posterior mean | unit, independent reference |
| `test_bernoulli.py` (16) | Bernoulli A1-A4 at moderate logits; R1/R2 values and full gradients on PyTensor and JAX at logits to 1e6 (both signs, both outcomes); the two-zeros/two-ones likelihood case; design and `logit_p` from the graph; 13 invalid inputs; prediction meaning (`p` vs 0/1 draws); new-group semantics incl. identity/sharing; fit record; GPU refusal; complex refusal with `ComplexWarning` as error | unit + fixture fit |
| `test_predict.py` (5) | Gaussian A8: in-sample `mu` equals the hand-computed linear predictor, residual SD, observed-group prediction, `population` spread and independence, `resample_observed` membership; synthetic-posterior identity/sharing/coincidence (1/J) and real-fit non-collapse | unit + fixture fit |
| `test_device.py` (4) | C1/C2: GPU on the PyMC backend refused; GPU absent refused (skipped if a GPU is visible); bad device/backend; NumPyro CPU report says float64 and `parallel` | integration |
| `test_session.py` (4) | float64 enabled at import; environment report; fit record complete and JSON-serialisable; `compare_records` agreement/disagreement/different-model refusal | unit + fixture fit |
| `test_sampling.py` (3, slow) | B1 Gaussian exact conditional posterior (z-scores < 4, SDs within 10 %); B2 Gaussian recovery on both backends (panel thresholds, truth in HDI, 4-MCSE agreement); M1 recovery with the same criteria. Each writes a JSON evidence file into `docs/m*-evidence/`; B1 and the M1 test write before their final assertions, B2 after all of them | sampling, ~25 s |

Gaps, in decreasing order of what they could hide:

- **No CI.** No `.github/` directory. The coordinator independently ran the 59 fast checks
  (59 passed); the 3 slow checks are worker-reported. Both are local runs; nothing
  automated ties a result to a PR head. Whether GitHub
  Actions minutes, runners, or any GPU runner are available to this repository is
  unverified.
- **Version coverage is one point.** Everything ran on Python 3.11.0 (laptop) and 3.11.15
  (H100 node) with pymc 5.28.5, pytensor 2.38.3, bambi 0.17.2, formulae 0.6.2, arviz 0.23.4,
  numpyro 0.21.0, jax 0.10.2, numpy 2.4.6 (fit records in `docs/m0-evidence/` and
  `docs/m1-evidence/`). `pyproject.toml` says `pymc>=5.28,<6` and leaves `jax`, `numpyro`,
  `numpy`, `pandas` unpinned, which implies combinations that were never run. CI on one
  pinned environment would be evidence for that environment only.
- **Prior families other than Normal/HalfNormal are untested end to end.** `Priors` accepts
  StudentT, HalfStudentT, Exponential, HalfCauchy and the reference density implements them,
  but no test builds a model with them or checks the density (a wrong parameter name in
  `to_bambi` would only surface at build time).
- **Multi-predictor prediction is untested**: `y ~ x + w + (1 | g)` is covered for the log
  density only; `predict` with two predictors (including the hand-computed new-group path)
  is not exercised.
- **Panel thresholds are only exercised at `pass`**: no test drives `investigate`/`fail`
  statuses or `format_panel`; the panel's `n/a` branches (no `energy`, no `tree_depth`) are
  untested.
- **Testable only in a fresh process, not yet tested**: the sequential-chain fallback when
  JAX is initialised before `import autobayes`; `AUTOBAYES_HOST_DEVICES=0`; the JAX-absent
  branch of `environment()`. These need subprocess tests (`python -c ...` in a fresh
  interpreter, or an environment without JAX); they are not intrinsically untestable.
- **GPU**: `test_gpu_unavailable_is_error_not_fallback` skips when a GPU is visible; no test
  asserts a GPU result. M0 has one executed H100 run recorded outside pytest
  (`docs/m0-gpu-validation.md`); M1 has none.
- **Slow tests overwrite evidence files** (`B1_*.json`, `B2_recovery.json`) on every run. The
  tests do assert their analytic and recovery criteria, so a regression fails; the risk is
  that a failing run has already overwritten the committed artifact (B1 and the M1 test
  write before their last assertions), not that it passes silently.
- **No coverage measurement**; the numbers above are by inspection.

Pinned stack versus live documentation, as encountered (each one cost time once):

| Topic | Installed / measured | Live documentation |
| --- | --- | --- |
| PyMC JAX sampler API | 5.28.5: `pm.sample(nuts_sampler="numpyro", nuts_sampler_kwargs={...})` | "stable" docs describe 6.x with `nuts={...}` and nutpie as default (`reference/pymc-to-jax.md`) |
| PyMC `get_jaxified_logp(negative_logp=...)` | flag named backwards in 5.28.5; default returns +logp (test A4) | undocumented |
| PyMC `compile_d2logp` | returns the negative Hessian (test A5) | docstring silent |
| PyTensor `Softplus` on JAX | lowers to nested `jnp.where`; NaN gradient at magnitude >= ~710 (M1 repair record) | undocumented; C path is fine |
| Bambi Bernoulli success coding | numeric: success = 1; bool accepted; string: first sorted level unless `y['level']` (source `families/univariate.py`) | logistic-regression notebook shows `vote['clinton']` only |
| Bambi `sample_new_groups=True` | one placeholder group for all unseen labels in a draw (`model_components.py`, reproduced 2026-09-09) | docstring: "each posterior sample for the new groups is drawn from the posterior draws of a randomly selected existing group" |
| Bambi inverse links | logit/probit/cloglog replace exact 0/1 by machine epsilon (`backend/links.py`) | undocumented |
| Bambi group-specific terms | separate `1\|g` and `x\|g` terms, each with its own SD; the LKJ helper in `backend/model_components.py` is commented out (source inspection; build-only, worker) | FAQ/docs do not state that `\|` means "uncorrelated" here, unlike brms |
| brms `posterior_epred` for binomial | n/a | returns `mu * trials`, expected counts, not probabilities (`R/posterior_epred.R`, fetched 2026-09-09) |
| ArviZ | 0.23.4: 94 % HDI default | 1.0: DataTree, 89 % ETI default, WAIC removed (`reference/diagnostics.md`) |
| matplotlib | must be `<3.11` for arviz 0.23 import | no note anywhere |
| brms bernoulli links | Bambi: identity, logit, probit, cloglog | brms: logit, probit, probit_approx, cloglog, cauchit, identity, log |
| brms new-level sampling | Bambi has "uncertainty" only; AutoBayes adds "population" (= brms "gaussian") | brms: uncertainty (default), gaussian, old_levels |

---

## 2. Capability matrix

"Existing" means present in the installed package or in this repository with a test or
record naming it. "Proposed" is this document. brms entries are from its documentation, its
`posterior_epred.R` source and the reference pack; I have not run brms. Bambi entries
marked "source" are from reading the installed package; "build-only, worker" means this
session built a model to look at its variables without fitting or validating it.

| Capability | brms | Bambi 0.17.2 (installed) | PyMC 5.28.5 | AutoBayes now (`main` + PR #3) | Proposed |
| --- | --- | --- | --- | --- | --- |
| Formula `y ~ x + (1 \| g)` | yes | yes (formulae) | hand-written | yes, restricted subset, 17 constructs refused by name | unchanged |
| Numeric predictors, several | yes | yes | yes | yes (density tested; prediction with >1 predictor untested) | test gap to close in M2 |
| Categorical predictors, interactions | yes, treatment contrasts by default | source: formulae `Treatment`, reference = first level; `x:z`, `x*z`; common columns incl. dummies are centred when an intercept is present | design by hand | **refused** (`categorical predictors are not supported`) | M3a |
| Varying slopes `(1 + x \| g)` | yes, correlated by default (LKJ), `\|\|` for independent | source: separate `1\|g`, `x\|g` terms, LKJ helper commented out; build-only, worker: free variables `1\|g_sigma, 1\|g_offset, x\|g_sigma, x\|g_offset`; `\|\|` is a formulae parse error | hand-written | **refused** (`varying slopes are not supported`) | M3b, independent first, correlated as a separate slice |
| Several / nested / crossed groups | yes | yes | yes | refused | after M3 |
| Gaussian family | yes | yes | yes | **M0, CPU both backends, one H100 run** | unchanged |
| Bernoulli, logit | yes | yes, but `p` floored at eps (wrong target above \|eta\| ~37) | `Bernoulli(logit_p=)` correct density, NaN gradients at large logits | **M1 on PR #3**: stable custom likelihood; CPU both backends; GPU pending | unchanged; see D8 on its boundary term |
| Binomial with trials | `k \| trials(n)`; `posterior_epred` = `n * mu` | source: `p(k, n) ~ ...` (formulae `proportion`), `n` passed to `pm.Binomial`; same clipped link; build-only, worker: an `n = 0` row builds with a finite log density | `Binomial(n, p)` / `logit_p` | none | **M2** |
| Other families (Poisson, NB, Student-t, beta, ordinal, hurdle, ZI, ...) | yes | 26 families in `defaults/families.py` | yes | none | not in this sequence |
| Explicit priors by class | yes; defaults otherwise | yes; auto-scaled defaults otherwise | yes | **mandatory, no defaults**; location/scale class checks | per-coefficient names for M3a |
| Centred predictors, intercept prior on the centred intercept | yes | yes (`center_predictors=True`, undone after sampling) | manual | yes, stated in `describe()` | extend statement to dummies in M3a |
| Non-centred group effects | always | `noncentered=True` | manual | yes, stated | unchanged |
| Inspectable model | Stan code | `model.backend.model` (undocumented workflow) | native | `pymc_model`, `describe()`, `value_point()`, verified density | unchanged |
| CPU NUTS | Stan | PyMC | PyMC | `backend="pymc"` | unchanged |
| JAX/NumPyro path with explicit device | no | `inference_method="numpyro"`, no device management | `nuts_sampler="numpyro"` | `device="cpu"\|"gpu"`, refusal without GPU, actual device/dtype/chain method reported | unchanged |
| Diagnostics panel with thresholds | via `posterior`/`bayesplot` | ArviZ summary | ArviZ | `panel()` with pass/investigate/fail | unchanged |
| Predictions: expected value vs new observation | `posterior_epred` (expected response: `n * mu` for binomial) / `posterior_predict`; probability via `posterior_linpred(transform = TRUE)` | `kind="response_params"`/`"response"` | manual | `kind="mean"`/`"response"`; `p` vs 0/1 for Bernoulli | `probability`, expected count, count draws for M2 (3.5) |
| New-level prediction semantics | uncertainty / gaussian / old_levels | uncertainty (one placeholder for all new labels) | manual | `error` / `resample_observed` (per-label) / `population` | unchanged |
| Prior predictive | `sample_prior_only`, `posterior_predict` | `model.prior_predictive()` | `pm.sample_prior_predictive` | **none** | M4 (thin wrapper) |
| Fit records and agreement check | no | no | no | `fit_record`, `compare_records` | unchanged |
| Session handoff / skill | no | no | no | `SKILL.md`, STATUS, records | M4 onboarding |
| GPU validation | n/a | not managed | not managed | M0 executed once; M1 pending | bounded run before M5 |

What this says about the contribution: Bambi supplies formulas, priors, centring, families,
the NumPyro path and predictions. What AutoBayes has added so far is (a) a verified subset
with refusals, (b) mandatory explicit priors, (c) a statement of what is actually sampled,
(d) device honesty, (e) named prediction semantics that Bambi and brms disagree on, (f) a
likelihood that is correct where Bambi's is not, and (g) records for handoff. Every proposed
item below should be judged by whether it adds to that list rather than by feature count.

---

## 3. M2: grouped binomial counts

### 3.1 Researcher need and what Bambi already supplies

Aggregated binary data (k successes of n trials per row: clinics, precincts, classrooms)
is the common form of the M1 use case. Mathematically it can be fit by expanding each row
into n Bernoulli rows with the same predictors and group label: the parameter-dependent
part of the likelihood is identical and only the combinatorial constant differs, provided
the centring is aligned (expanded rows give trial-weighted column means, aggregated rows
give unweighted ones). The binomial family therefore adds no new statistical content; it
adds the row as the unit researchers reason about (expected counts per clinic), the
brms/Bambi syntax they know, a likelihood evaluation that is O(rows) rather than
O(trials), and, as a by-product, an independent check (A6b below). Bambi supplies the
`p(k, n)` response syntax, the design, priors, centring, and passes `n` to the likelihood
(`families/univariate.py::BinomialBaseFamily`). Bambi's binomial likelihood uses the same
eps-floored inverse link as its Bernoulli one (`backend/links.py`), so it is expected to
share M1's defect; that is an expectation to be measured under A5, not a fact. The M1
construction does not carry over unchanged either: its boundary term cancels (3.3).

### 3.2 Proposed API (decision D1 for Andrew: brms syntax on the left-hand side)

```python
priors = ab.Priors(intercept=ab.Normal(0, 1.5), b=ab.Normal(0, 1), sd=ab.HalfNormal(1))   # no sigma
model  = ab.binomial("deaths | trials(patients) ~ x + (1 | clinic)", df, priors)
fit    = model.fit(backend="numpyro", device="cpu", seed=7)
model.predict(fit, kind="probability")                            # p          (chain, draw, rows); no trials needed
model.predict(fit, kind="mean")                                   # p, expected = n p              (brms posterior_epred)
model.predict(fit, kind="response")                               # p, expected, y ~ Binomial(n, p) (brms posterior_predict)
model.predict(fit, newdata=new, kind="mean", new_groups="population")   # new must carry `patients` for mean/response
```

- Formula: exactly `successes | trials(n) ~ [1 +] x1 [+ ...] + (1 | group)`, brms's syntax,
  parsed by `autobayes.formula` and translated to Bambi's `p(successes, n)`. Alternative:
  accept Bambi's `p(k, n)` too. I recommend one syntax, brms's, because the skill is
  brms-like and the parser already owns the left-hand side; the translation is a string
  rewrite, not a parser.
- `ab.binomial` is a third `family` value in `Model`; `Priors` as for Bernoulli (logit
  scale, no `sigma`).
- Likelihood: `pm.CustomDist` inside a Bambi custom family whose `transform_backend_kwargs`
  mirrors `BinomialBaseFamily` (two-column response: `observed = k`, `n` from column 2),
  with the log-sigmoid form of 3.3 built from `stable_softplus`. Bambi names the observed
  variable `p(k, n)`, so `Model.names["y"]` must map to that name for this family.
- `describe()` states the equation, the trials column, the number of `n = 0` rows dropped
  before fitting (D2), and whether all outcomes are zero or all successes (D3).

### 3.3 Mathematical target and numerical construction

For rows $i$ with $k_i$ successes of $n_i$ trials, predictors $x_{ip}$, group index $g_i$:

$$\eta_i = a_c + \sum_p (x_{ip} - \bar x_p)\, b_p + s_g z_{g_i}, \qquad
k_i \sim \mathrm{Binomial}(n_i, \mathrm{expit}(\eta_i)),$$

$$\log p(k_i \mid n_i, \eta_i) = \log\binom{n_i}{k_i}
- k_i\,\mathrm{softplus}(-\eta_i) - (n_i - k_i)\,\mathrm{softplus}(\eta_i),$$

with $z_j \sim \mathrm{Normal}(0,1)$, priors on $a_c$, $b_p$, $s_g$ as in M1, the Jacobian
$+\log s_g$ in the sampled density, and

$$\frac{\partial \log p}{\partial \eta_i} = k_i\,\mathrm{expit}(-\eta_i) - (n_i - k_i)\,\mathrm{expit}(\eta_i).$$

The two likelihood terms are each a count times a non-positive stable quantity and are
added, never subtracted, so there is no cancellation at the boundaries $k = 0$ or $k = n$.
The algebraically equal form $k\eta - n\,\mathrm{softplus}(\eta)$ (M1's construction with
$n = 1$) is **not** acceptable for counts: at $k = n$ it subtracts nearly equal numbers, and
in float64 it returns exactly $0$ at $\eta = 40$ for every $n$ where the true value is
$-n \cdot 4.2\times10^{-18}$ (an absolute error of $4.2\times10^{-9}$ at $n = 10^9$), and its
gradient $k - n\,\mathrm{expit}(\eta)$ returns exactly $0$ where the true value is
$n\,\mathrm{expit}(-\eta)$. The log-sigmoid form and the gradient form above reproduce a
60-digit decimal reference to rounding at $n$ up to $10^{15}$ at those points (pure
NumPy/`decimal` arithmetic on 2026-09-09; not a library run). For $n = 1$ the two forms
differ only by that same boundary term, which is why M1's tests at 1e-8 absolute could not
see it (D8 below). The binomial coefficient is constant in $\eta$ and kept so that the
density is normalised; its float64 spacing is about $10^{-7}$ at $n = 10^9$, so comparisons
that include it use a relative tolerance. Centring, the reported raw-scale intercept, and
the non-centred coding are unchanged from M1.

**Proposed validation domain.** This is a domain to be validated by the M2 tests, not a
domain already validated and not a guarantee for any denominator. Proposed accepted counts:
integers in $[0, N_{\max}]$ with $N_{\max} = 2^{31}-1$, refused above with a message. The
test grid must include the cap and its neighbours: $n \in \{1, 2, 10, 10^3, 10^6, 10^9,
N_{\max}-1, N_{\max}\}$, with $N_{\max}+1$ refused; $k \in \{0, 1, \lfloor n/2 \rfloor, n-1, n\}$
(deduplicated); $\eta$ at the A5 points; both backends; values and full gradients;
tolerance $\max(10^{-8}, 10^{-12}\,|\text{value}|)$ per model log density and per gradient
component. Nothing is claimed outside that grid, and the grid is the only basis for the cap.

**Independent references.** All proposed references are independent of the library's
float64 `gammaln` expression. Use high-precision arithmetic with a recorded numerical-error
budget smaller than the acceptance tolerance; this reference code itself must be reviewed
before it is used as an oracle. At k=0 or k=n, use high-precision log-sigmoid values directly.
For m=min(k,n-k) <= 10^4, including k=1 and k=n-1 at large n, sum
$\log\binom{n}{m}=\sum_{j=1}^{m}\log((n-m+j)/j)$ in high-precision `decimal` arithmetic,
then combine it with high-precision log-sigmoid terms. This is a finite-precision computation
with an error budget, not an exact sum. For larger interior m, either use an independently
implemented, high-precision log-gamma reference or a reviewed Stirling expansion for n!,
k!, and (n-k)! with explicit remainder bounds for all three factorials. A KL rearrangement
may help, but its arithmetic and remainder also need error control near q=p and at extreme
logits. Do not apply a large-k Stirling remainder to k=1. The same acceptance tolerance
$\max(10^{-8},10^{-12}|\text{value}|)$ applies to comparisons throughout A1-A5; reference
error must be accounted for separately and kept below that tolerance. D10 is the proposed
engineering choice of reference implementation, not evidence that one has already passed.

**Condition for reducing the domain.** If any grid point fails its tolerance on either
backend, $N_{\max}$ is lowered to the largest tested $n$ at which every $k$ and $\eta$ passes
on both backends, the refusal message states that value, and the failing points are
recorded in the decision record; the tolerance is not loosened to keep the cap.

### 3.4 Data edge cases: decisions to make before implementation

| Case | Proposed decision | Reason / alternative |
| --- | --- | --- |
| Count dtypes | Accept integer dtypes and float columns whose values are all finite and integral (CSV readers produce `3.0`); refuse bool, complex, non-integral, negative, non-finite, and `k > n`, all **before** any cast, naming the column and the first offending value | Same rule as M1's complex refusal; a silent cast is how M1's finding 2 arose |
| Count magnitude | Proposed cap $N_{\max} = 2^{31}-1$, included in the test grid with its neighbours and lowered by the rule in 3.3 if the tolerances fail; refuse larger. Exact float64 representation of larger integers is not the issue; untested likelihood arithmetic is | 3.3 |
| Rows with `n = 0`, `k = 0` | **D2**: accept, count them, and **drop them before the model is built**; report the count in `describe()` and the fit record. They contribute no likelihood, and if kept they would enter the centring means and thereby change what the fixed prior on the centred intercept refers to | Bambi builds and evaluates such rows (build-only, worker) but centring over them is the problem, not the likelihood. Alternative: keep them and centre over `n > 0` rows only, which Bambi's backend does not offer without patching |
| `n = 0` with `k > 0` | refuse | impossible data |
| All rows `n = 0` | refuse: "no trials in the data; a prior-only fit must be requested explicitly" | The plan asks for an explicit policy; prior-only inference is a different task |
| All `k = 0` or all `k = n` across rows | **D3**: accept under the proper priors, with a `describe()` warning that the data are separated along the intercept direction (the likelihood is monotone there), that the posterior in that direction is set by the prior's tail and the data's extent, and that the result should be checked for prior sensitivity; recorded in the fit record | The plan prefers acceptance with warnings. Such data can still update a proper prior substantially; "prior-dominated" is not the right description. This differs from M1's refusal of a constant Bernoulli response, which stays as documented (the plan: do not silently change it). Alternative: align the two in a later slice |
| Group with a single row | accept (as now) | shrinkage handles it |
| Trials column also used as a predictor | refuse (as the group column is) | ambiguity |
| Centring policy | column means over the fitted rows (after dropping `n = 0` rows), unweighted, as brms and Bambi do | keeps the M1 meaning of the intercept prior; documented in `describe()` |

### 3.5 Predictions (three quantities, named)

- `kind="probability"`: `p` only; no trials needed. In brms this is
  `posterior_linpred(transform = TRUE)`, not `posterior_epred`; the AutoBayes name is a
  deliberate difference so that researchers cannot mistake a probability for an expected
  count.
- `kind="mean"`: `p` and `expected` ($n p$), the expected response, which is what brms
  `posterior_epred` returns for binomial models (`mu * trials`). Requires trials: in-sample
  from the training data (after D2 dropping), for `newdata` the trials column named in the
  formula, integral and non-negative, else a `FormulaError` naming it. Expected counts are
  therefore available without generating response draws.
- `kind="response"`: additionally `y ~ \mathrm{Binomial}(n, p)` as `int64` (brms
  `posterior_predict`). `n = 0` rows in `newdata` give `expected = 0`, `y = 0`.
- Numerical range: `p` is `expit(eta)` in float64 and may equal exactly 0 or 1 where the
  sigmoid saturates; the logit magnitudes at which it rounds to 1 and to 0 are different,
  so no symmetric threshold is stated and tests accept `[0, 1]`. The likelihood is never
  evaluated from `p`; it is evaluated from logits (3.3), so saturation in the reported
  probability does not feed back into the fit.
- New-group semantics exactly as M1 (`error` / `resample_observed` per distinct label /
  `population` on the logit scale); the shared test helper from `tests/test_predict.py` is
  reused so identity and sharing are tested for the third family too.

### 3.6 Acceptance criteria (to be copied into `docs/m2-decision.md` before code)

- A1/A2. Log density with and without Jacobian equals an independent NumPy/SciPy reference
  (`gammaln`, `log_expit`) at five random points, centred and uncentred, one and two
  predictors, intercept-only, to $\max(10^{-8}, 10^{-12}|\text{value}|)$; the Jacobian
  difference is exactly $\log s_g$.
- A3. `compile_dlogp` equals central differences of the reference (rtol 1e-5) and the
  analytic gradient $k\,\mathrm{expit}(-\eta) - (n-k)\,\mathrm{expit}(\eta)$ through the design
  (same tolerance rule).
- A4. `get_jaxified_logp` equals the PyTensor path at the same points.
- A5. Extreme logits and large counts, both paths, values and full gradients on the grid of
  3.3 ($\eta \in \{0, \pm 0.5, \pm 12, \pm 40, \pm 300, \pm 1000, \pm 10^6\}$, the listed $n$ and
  boundary/interior $k$): equal to the stable reference under the tolerance rule, finite
  everywhere. Before choosing the construction, measure Bambi's built-in binomial and
  `pm.Binomial(logit_p=)` at the same points and record the table in the decision record,
  as was done for M1.
- A6. **`n = 1` equivalence on the shared domain**: for data with every $n_i = 1$,
  $k_i \in \{0,1\}$, both outcomes present, `ab.binomial` and `ab.bernoulli` have the same
  sampled parameter names (`Intercept`, one per slope, `1|g_sigma_log__`, `1|g_offset`),
  and their log densities and gradients agree at the same value points under the tolerance
  rule after subtracting the binomial coefficient term (zero here); the observed-variable
  names (`y` versus `p(k, n)`) and the likelihood line of `describe()` are allowed to
  differ and the test says so. `predict(kind="probability")` agrees with M1's `kind="mean"`
  from the same posterior. Outside that domain (constant outcomes) the two families differ
  by policy (D3), and a test asserts the documented difference.
- A6b. **Expansion check**: for random aggregated data with `center_predictors=False`, the
  binomial log likelihood minus its combinatorial constant equals the M1 Bernoulli log
  likelihood of the row-expanded data at the same parameters; with centring on, the same
  holds once the aggregated model is given the trial-weighted means (or both models are
  uncentred). This is the independent statistical check that the family adds nothing new.
- A7. Design: the two-column response is split correctly (`k` observed, `n` attached);
  level coding as M0/M1; `n = 0` rows dropped and counted (D2).
- A8. Invalid inputs of 3.4, each refused with a message naming the column, before any
  cast (`ComplexWarning` and `RuntimeWarning` promoted to errors in the test).
- A9. Predictions: shapes (chain, draw, rows); `p` in $[0, 1]$; `expected = n p` exactly;
  `y` integer in $[0, n]$. Predictive check conditional on the saved posterior array: given
  the saved $p_s$ (chain, draw, row), the response draws $y_s$ are fresh RNG draws that are
  independent across draws and rows conditional on that array, so with $r_s = y_s - n\,p_s$
  the total $T = \sum_s y_s$ for a row has conditional mean $E = \sum_s n\,p_s$ and variance
  $V = \sum_s n\,p_s(1-p_s)$. Three regimes, decided per row from the saved array:
  (i) $V = 0$ (zero trials, or every $p_s$ saturated at 0 or 1): deterministic exact check,
  `y` equals `expected` exactly in every draw, and `y = 0` when $n = 0$; (ii) $V > 0$: the
  acceptance interval is $|T - E| \le t$ with $t$ from Bernstein's inequality for a sum of
  independent variables bounded by $n$, $2\exp\!\big(-t^2 / (2(V + n t/3))\big) \le 10^{-6}$,
  solved for $t$; this is an exact bound, so the false-failure rate is at most $10^{-6}$ per
  row whether the conditional distribution is near-normal or discrete with a small
  expected count, and it replaces the normal $|z| < 4$ cutoff; (iii) the same Bernstein
  interval for the sum over all rows with $E$, $V$ and the bound summed (variances add
  under conditional independence). A row-wise Bonferroni budget of $10^{-6} \times$ rows is
  stated in the test. No pooled percentage rule. `newdata` without the trials column
  refused for `kind="mean"` and `"response"` and accepted for `"probability"`; new-group
  identity and sharing as in M1.
- B1. Simulated recovery, $n_{rows} = 300$, $J = 30$ with every group present by
  construction (indices cycled then shuffled), trials uniform in $[5, 50]$, truth
  $a = -0.5$, $b = 0.8$, $s_g = 0.7$, 4 x (1000 + 2000) draws, seed 7, both CPU backends: zero
  divergences, max R-hat <= 1.01, bulk and tail ESS > 400 total, truth inside the 94 % HDI,
  backends within 4 combined MCSE. One recovery; not calibration.
- C1-C3. Device refusal and CPU report as M1; GPU execution **pending** and labelled so.
- Docs/skill: `examples/m2_binomial_varying_intercept.py`, a family section in `SKILL.md`
  (probability vs expected count vs count draw), README, STATUS.

Kill criterion: if the custom family cannot receive `n` without patching Bambi, M2 builds
the likelihood on Bambi's design matrices with the response handled entirely in this layer
(a bigger PR; to be flagged before proceeding).

Estimated size: one PR of the same shape as M1 (library ~150 lines, tests ~300, docs).

---

## 4. Recommended order after M2, with PR boundaries

The plan's order is: repeatable checks/CI, M2, categorical predictors and interactions,
varying slopes, skill/onboarding alongside, then benchmarks. I agree with the order and
add one change: the CI/environment slice should land **before M2 is merged**, not after,
because a PR currently carries no result tied to its head: the fast suite has been run
independently by the coordinator and the slow suite by the worker, both locally. Everything below is proposed; none of it is authorised by this
document.

### 4.0 Repeatable checks (small, first)

- **What**: a GitHub Actions workflow on push/PR running the fast suite on CPU with a lock
  file that pins the tested stack (uv lock; pymc 5.28.5, bambi 0.17.2, jax 0.10.2, numpyro
  0.21.0, arviz 0.23.4, matplotlib <3.11, Python 3.11); fresh-process tests for import
  order and JAX absence; a second, manually triggered job for the slow suite whose
  acceptance is the tests' own prespecified statistical criteria (analytic z-scores,
  panel thresholds, HDI coverage of the truth, 4-MCSE backend agreement), **not** equality
  of regenerated JSON: floating-point paths, platforms and Monte Carlo variation can change
  valid summaries. Candidate evidence is written to a run-specific directory (for example
  `docs/evidence-runs/<date>-<sha>/`) and committed historical records are immutable;
  deterministic checks (schema, provenance fields, version strings) are separate from the
  stochastic criteria. A third, manual, GPU job asserts `device actual` contains
  `gpu`/`cuda` and fails otherwise.
- **Why it helps a researcher**: the README support table (implemented / CPU-validated on
  the pinned stack / GPU-validated / unsupported) is backed by a result tied to a PR head.
  It is evidence for the pinned environment, not for every version the package metadata
  allows.
- **Bambi already supplies**: nothing here.
- **Acceptance**: a fresh clone with the lock file passes the fast suite in CI; the README
  support matrix lists the four states per family and path; `pyproject.toml` minimums
  either match the lock or the README says the lock is the tested set.
- **Assumption, unverified**: GitHub Actions minutes and runners are available to this
  private repository (~10 minutes per run, environment install dominating); no GPU runner
  is assumed.

### 4.1 M3a: one categorical predictor, then one interaction (two PRs)

- **What Bambi supplies** (source): formulae treatment coding with the first sorted level
  as the reference (`formulae/categorical.py::Treatment`), `x:z` and `x*z` expansions,
  coefficient names like `z[T.b]`, and centring of all common columns including the dummy
  columns when an intercept is present (`backend/model_components.py`). brms behaves the
  same on centring (`brms-to-pymc.md`, "The mean-centring trap").
- **What AutoBayes adds** (the evidenced gap): the intercept after centring dummies is the
  expected response at the *sample proportions of the levels*, not at the reference level,
  and neither Bambi's `print(model)` nor brms's summary says so; explicit declared reference
  level; per-coefficient priors by a stated naming rule; the design matrix checked against
  an independently built one (level order, reference, interaction columns); refusal of an
  unseen category level in `newdata`; `describe()` writing the equation with the dummy
  columns and what the intercept means.
- **PR 1 (M3a-1)**: `y ~ x + z + (1 | g)` with `z` categorical (`category`/string/int-coded
  declared as categorical), treatment contrasts, `reference=` argument, priors `b={"x": ...,
  "z": ...}` applying one prior to all dummies of `z` or per level. Acceptance: A1-A4 with the
  reference density extended to the dummy design; **reference-level invariance stated
  precisely**: changing the reference level maps the coefficients by a known linear map
  $\beta' = M\beta$ and leaves the design span, the linear predictor and the likelihood at
  corresponding points unchanged (tested); the *posterior* is not invariant under
  independent coefficient priors, so a same-model posterior comparison is made only with
  the transformed joint prior (multivariate normal with covariance $M \Sigma M^\top$ and mean
  $M\mu$; the map is affine with unit Jacobian, so no density correction), and otherwise
  the change of reference is documented as a change of model; predictions for each level
  and refusal of unseen levels; recovery on both backends.
- **PR 2 (M3a-2)**: one interaction, numeric x categorical first (`x:z`, then `x*z`),
  because it is the interaction whose coefficients a researcher reads as "slope of x within
  each level"; categorical x categorical later.
- **Out of scope**: sum/Helmert contrasts, ordered factors, polynomial terms.

### 4.2 M3b: one varying slope (two PRs)

- **What Bambi supplies** (source, plus a build-only worker observation): `(1 + x | g)`
  builds two group-specific terms `1|g` and `x|g`, each with its own SD hyperprior and
  non-centred offsets; the LKJ helper in `backend/model_components.py` is commented out;
  formulae raises `ParseError` on `||`. So Bambi's `(1 + x | g)` is independent slopes.
  This is **not** brms's default: in brms `(1 + x | g)` is correlated with an LKJ prior and
  `||` requests independence. A researcher porting a brms model through Bambi silently
  loses the correlation. That divergence is the evidenced gap; it is a source-inspection
  finding, not a statistically validated one.
- **PR 1 (M3b-1, independent)**: accept only `(1 + x || g)` (brms syntax for what Bambi
  builds); refuse `(1 + x | g)` with a message saying that correlated slopes are not yet
  supported and how to write the independent form. AutoBayes's parser accepts `||`,
  records the intent, and rewrites it to Bambi's `(1 + x | g)`, the same string-rewrite
  pattern as the binomial left-hand side. Priors: `sd={"Intercept": ..., "x": ...}`.
  `describe()` states the two SDs and "no correlation". Acceptance: reference density with
  two group-effect vectors; design check of the slope's group matrix; new-group semantics
  with two effects drawn together per label (identity rule as M1); recovery.
- **PR 2 (M3b-2, correlated)**: `(1 + x | g)` with a Cholesky-factored LKJ prior built in
  this layer on Bambi's design matrices, non-centred (`u = diag(sd) L z`), `cor=ab.LKJ(eta)`
  in `Priors`, correlation reported in the posterior. This is the first piece of model code
  that Bambi does not supply; it is justified only by the semantic divergence above.
  Acceptance: density against a reference with the LKJ term written from its definition,
  the correlation identified in a recovery with $J \geq 40$, `||` and `|` giving the same
  posterior when `cor` is fixed at identity.
- **Order question from the plan (categorical first or slopes first)**: categorical first.
  Binary indicators can already be entered as 0/1 numeric columns, so M3a's marginal gain
  is multi-level factors and interactions, which appear in nearly every applied model; M3b
  is a modelling capability rather than a coding convenience but its correlated form is the
  heavier PR and depends on the naming and `describe()` machinery M3a introduces. If the
  first onboarding example (4.3) needs a varying slope, swap them; that example should be
  chosen before either starts.

### 4.3 M4: researcher skill and onboarding (one focused PR after M2, then per-PR upkeep)

- **What exists**: `SKILL.md` rewritten on 2026-09-09 around the verified workflow (six
  checked steps, failure table, handoff checklist); fit records; `docs/coordination/STATUS.md`
  conventions; the scaffold's `DECISIONS.md` gate and wiki. **What is missing**: a prior
  predictive step (Bambi has `model.prior_predictive()`; AutoBayes has nothing), and any
  evidence that an agent starting cold can get from a question to a handoff using the skill.
- **PR (M4-1)**: `ab.prior_predict(model, draws, seed)` as a thin wrapper returning the same
  dataset shape as `predict` (so prior and posterior predictive checks look alike), plus a
  "before you fit" step in the skill; and the onboarding exercise: a public or simulated
  grouped-outcome data set with a data dictionary (proposal: simulated clinics for Bernoulli
  and, for Gaussian, the public radon data used in the PyMC examples, which needs no
  confidential data), a research question, and a scripted checklist an independent
  inspector runs on the outputs: `describe()` present and matching the formula; a prior
  predictive summary; panel values with statuses; a fit record path; a handoff paragraph
  with the resume command; one deliberately unsupported request (e.g. a Poisson outcome)
  that the agent must refuse by name and escalate rather than approximate.
- **Acceptance** (set before the exercise): the inspector script passes on the produced
  files; a second fresh session resumes from the handoff and reproduces the fit record's
  posterior means within 4 MCSE; the unsupported request is refused, not "solved".
- **Why**: this is the only way to find out whether the skill works for someone other than
  the session that wrote it.

### 4.4 M5: matched CPU/GPU benchmark protocol (proposal only; runs need approval)

- **Prerequisite**: one capped M1 GPU correctness run with the M0 recipe
  (`docs/m0-gpu-validation.md`: same seed, priors, precision, 4 vectorized chains; agreement
  within 4 combined MCSE). The M0 run cost $0.72 at the rate observed on 2026-09-09
  ($0.0666/min for one H100 on GiveMeANode); that is a dated observation, not a quote.
- **Full grid, for scale only**: rows $n \in \{200, 2000, 20000\}$ x groups $J \in \{20, 200\}$ x
  families {Gaussian, Bernoulli, Binomial if M2 is merged} x chains $\{4, 16, 64\}$ is 54
  configurations per device (36 without binomial); at 10 replicates on 2 devices that is
  1080 fits (720). That is a campaign, not the next milestone, and nothing here approves it.
- **Immediate pilot (the only run proposed)**: Gaussian only, $J = 20$, $n \in \{200, 20000\}$,
  chains $\in \{4, 64\}$: 4 configurations on 2 devices. One **fit** is one call of `ab.fit`
  that produces draws for one configuration on one device, in one process state; every
  such call is counted, including the dry run, and cold and warm calls are separate fits.
  Accounting:

  | Block | Purpose | Calls per configuration per device | Configurations x devices | Fits |
  | --- | --- | --- | --- | --- |
  | Dry run | timing at 64 chains, cost cap, shutdown rules; results recorded but excluded from the statistical summary | 1 cold (fresh process) | 4 x 2 | 8 |
  | Main, cold | compile + sample in a fresh process, one per replicate | 5 | 4 x 2 | 40 |
  | Main, warm | second call of the same graph in the same process as its cold call, compilation excluded | 5 | 4 x 2 | 40 |
  | Total pilot | | | | **88** |
  | Optional, not in the pilot | identical-graph overhead comparison (4.4 baselines (i)): one warm call through `pm.sample` directly per replicate | 5 | 4 x 2 | +40 (D11) |

  Per-fit time at 64 chains is unknown before the dry run (M0's 4-chain runs took 4 s on
  the laptop CPU and 39 s on the H100 including compilation, at 200 rows; those numbers do
  not extrapolate to 64 chains or 20000 rows). The cost is therefore uncertain until the
  capped dry run has been reviewed; the cap for the main blocks is set from the dry-run
  times with a stated safety factor, the node is stopped and deleted at the cap or on
  completion, and no price in this document is a commitment. The 54/36-configuration grid
  above is out of the immediate scope.
- **Design generation**: group indices cycled then shuffled so every one of the $J$ groups
  is present; seeds recorded per replicate; the same simulated data set on both devices.
- **Cache rules**: "cold" = fresh process, compilation time measured separately with
  `block_until_ready`; "warm" = a second fit of the same graph in the same process,
  compilation excluded; both reported, never mixed.
- **Measured per run** (rules from `reference/acceleration.md`): compile, warmup, sampling,
  post-processing, peak device memory, wall time, failure rate (divergences, non-finite),
  and ESS per second for named estimands (intercept, slope, group SD, and for Gaussian
  `sigma`; means **and** variances, since mean-only ESS is the documented recipe for a fake
  speedup). Distributions of per-run ratios across replicates, never a single ratio;
  minimum ESS alongside the estimand ESS, not as the headline.
- **Chain concurrency vs fit batching**: vectorized chains on one GPU are one axis; running
  many independent fits (a simulation study) is another and is not supported by the API;
  the protocol states this and does not conflate them.
- **Baselines, two distinct comparisons**: (i) *implementation overhead*: the identical
  AutoBayes-built PyMC graph and sampler settings run through `pm.sample` / Bambi's `fit`
  directly versus through `ab.fit`; this measures the wrapper, and only this comparison
  can be expected to show no difference; (ii) *statistical baseline*: Bambi's built-in
  family or brms with aligned design matrices, priors and prediction targets, noting that
  Bambi's built-in Bernoulli is a different target at extreme logits and brms is a
  different sampler and precision; (ii) needs an R environment for brms (assumption: not on
  the laptop; unverified on the cluster).
- **Acceptance for the protocol document itself**: a reader can rerun a cell from the
  record alone (versions, seeds, device, settings); every number in the write-up traces to
  a fit record; failed or slower cells are reported. A null or negative speed result is a
  reportable finding; nothing here claims it is publishable.

---

## 5. Where I disagree with or would qualify the roadmap

1. "Finish M1 checks and reconcile stale STATUS sections" is right, but STATUS is already
   long; I propose the next PR moves the M0 and M1 narrative into `docs/m0-decision.md` /
   `docs/m1-decision.md` result sections and keeps STATUS to current state, evidence
   pointers, blockers and next action, which is what a fresh session reads.
2. Accepting `n = 0` rows (D2) is simple in the likelihood; the reason to drop them before
   the fit is centring, not Bambi's response handling.
3. Accepting all-zero/all-success binomial data while Bernoulli refuses a constant response
   (D3) is a documented inconsistency. I would rather revisit the Bernoulli refusal in a
   tiny follow-up than carry two policies, but the plan says not to change it silently, so
   it is Andrew's call.
4. The plan lists "varying slopes with explicit independent-versus-correlated semantics" as
   one increment; I split it in two because the correlated form is the first hand-built
   model component and deserves its own review.
5. Benchmarks before a second GPU correctness run would measure a path that has been
   executed once, for one family. The M1 GPU run should precede any timing.
6. The plan's "JSS is a possible venue": the contribution that the evidence supports today
   is semantic verification and honest device/prediction reporting on top of Bambi, not
   speed. The benchmark protocol is designed so that a null or negative speed result is
   still reported; whether any of this is publishable is not a claim this document makes.

## 6. Assumptions recorded

- The stack stays pinned at the tested versions until a deliberate upgrade slice; PyMC 6
  and ArviZ 1.0 both change APIs that the code and the docs rely on.
- Bambi's custom-family hook (`Likelihood(dist=...)`, `Family(link="identity")`) remains
  the way to attach a hand-written likelihood; it worked for M1 without patching Bambi.
- GiveMeANode rates and images are as observed on 2026-09-09; future runs are quoted at
  creation and capped separately.
- No R/brms is available locally; a brms baseline needs an environment decision.
- Public data for onboarding is acceptable; no confidential election data is required.
- GitHub Actions availability for this private repository is unverified.

## 7. Unresolved choices for Andrew

- D1: brms `k | trials(n)` syntax only, or also Bambi's `p(k, n)`.
- D2: accept `n = 0, k = 0` rows by dropping them before the fit with a reported count
  (recommended), or refuse them.
- D3: accept all-zero/all-success binomial data with a separation warning (recommended)
  and leave the Bernoulli refusal as is, or align the two.
- D4: order of M3a (categorical) and M3b (varying slope); recommended categorical first
  unless the onboarding example needs a slope.
- D5: whether CI (4.0) precedes the M2 merge (recommended) or follows it.
- D6: approve the capped M1 GPU correctness run as a bounded task independent of M5.
- D7: the onboarding data set (simulated clinics + public radon proposed).
- D8: whether M2 should also switch M1's Bernoulli likelihood to the log-sigmoid form
  `-y softplus(-eta) - (1-y) softplus(eta)` for uniformity. Keep this an explicit scope choice.
  If approved, add a targeted likelihood-only regression on both PyTensor/JAX paths in
  addition to existing R1/R2: for y=1, eta=40 the log likelihood is
  $-\log(1+e^{-40})$ and its derivative with respect to eta is $+\mathrm{expit}(-40)$;
  for y=0, eta=-40 the value is the same and the derivative is $-\mathrm{expit}(-40)$.
  Use an independent reference and a relative tolerance capable of detecting zero in
  place of these quantities (proposed 1e-3 relative, about4e-21 absolute). The positive
  boundary exposes cancellation in the current expression; the negative boundary is a
  symmetry/gradient-sign check. Existing absolute-tolerance tests cannot detect the tiny
  positive-boundary loss. These are future acceptance requirements, not newly executed
  tests or authorization to alter M1 now.
- D9: prediction kind names (`probability` / `mean` / `response`) versus brms's
  `posterior_linpred(transform=TRUE)` / `posterior_epred` / `posterior_predict`.
- D10: whether a high-precision arithmetic library (e.g. `mpmath`) may be added as a
  development-only dependency for reference values; not assumed (the `decimal` and
  Stirling/KL references in 3.3 need nothing new).
- D11: whether the pilot benchmark includes the identical-graph overhead block (+40 fits)
  or leaves it to a later table.

## 8. Next prompt outline (for the coordinator, once Andrew has edited this file)

1. Confirm D1-D3, D5, D8, D9; state the PR budget (one PR, review before merge).
2. Instruct: write `docs/m2-decision.md` from section 3 (API, target and numerical
   construction, edge-case table, A1-A9 incl. A6b, B1, C1-C3, kill criterion) and commit it
   before code; measure Bambi's built-in binomial and `pm.Binomial(logit_p=)` at the A5 grid
   and record the table; implement the custom binomial family with the log-sigmoid form on
   `stable_softplus`; tests including the n = 1 equivalence, the expansion check, the
   conditional predictive statistic and the shared new-group helper; run fast and slow
   suites; example, skill section, README, STATUS; open a PR under andrew-dou with no agent
   attribution and leave it for review.
3. Limits to restate: no GPU/cluster/paid services, no dependency changes, no CI in that PR
   (separate slice), no merge or cleanup by the worker, stop at usage limits.

## 9. Revision record (2026-09-09, review findings and dispositions)

Each finding of `memory/reviews/2026-09-09-autobayes-roadmap-review.md` was checked against
the cited source or by arithmetic before the text was changed.

1. **Probability vs expected counts.** Verified: `posterior_epred_binomial` returns
   `prep$dpars$mu * trials` (`R/posterior_epred.R` as fetched, lines 373-375; the review
   cited 365-368, same content). Changed: 3.5 now names three quantities
   (`probability`, `mean` = expected count, `response`), expected counts are available
   without response draws, `p` may be exactly 0 or 1 after saturation, the likelihood is
   evaluated from logits. Added D9. Agreed in full.
2. **Count precision vs likelihood accuracy.** Verified by pure NumPy/`decimal` arithmetic
   (no library run): the M1-style form returns exactly 0 at $k = n$, $\eta = 40$ for every $n$
   and a zero gradient; the log-sigmoid form reproduces a 60-digit reference to rounding
   up to $n = 10^{15}$. Changed: 3.3 adopts the log-sigmoid form and its gradient, a
   validated count domain with a test grid and a relative tolerance rule, and states that
   `gammaln` does not address cancellation and that M1's construction does not carry over.
   Added D8. Agreed in full.
3. **Coding invariance and the prior; zero-trial centring.** Verified by reasoning: an
   affine coefficient map leaves the likelihood invariant but not independent priors;
   `n = 0` rows change the centring means. Changed: 4.1 states likelihood/design invariance
   as the test and requires a transformed joint prior for any posterior comparison; D2 now
   drops `n = 0` rows before the fit and 3.4 has an explicit centring policy. Agreed.
4. **Data semantics and predictive checks.** Verified: expansion to Bernoulli rows gives the
   same parameter-dependent likelihood up to the constant and centring alignment; the M1
   test's standard error $\sqrt{\bar p(1-\bar p)/S}$ is conservative by Jensen's inequality
   rather than wrong, and its pooled 1 % rule has no derivation. Changed: 3.1 corrected,
   A6b added, D3 reworded (separation and prior sensitivity, not prior dominance), A9 uses
   the conditional statistic with correctly aggregated variance and drops the pooled rule,
   A6 states the parameter-name mapping and allows observed-variable names and text to
   differ. Agreed.
5. **Sampling CI.** Verified in `tests/test_sampling.py`: B1 (line 41) and the M1 test
   (line 116) write their JSON before their final assertions; B2 writes after. Changed: 4.0
   uses the tests' statistical criteria, run-specific candidate evidence, immutable history,
   deterministic checks separated from stochastic ones; section 1 corrects the "silently"
   wording, marks the fresh-process cases as testable, and records the unverified GitHub
   dependency. Agreed.
6. **Benchmark grid and cost.** Verified the arithmetic: 54 (36) configurations per device,
   1080 (720) fits at 10 replicates on 2 devices; my "6-cell subset" was 4 configurations at
   one group count. Changed: 4.4 separates the full grid (scale only) from a 40-fit pilot,
   requires a dry run before any cost estimate, dates the rate, states design generation and
   cache rules, and approves nothing. Agreed.
7. **Matched baseline.** Verified against `autobayes/likelihoods.py`: M1's graph is not
   Bambi's built-in graph. Changed: 4.4 distinguishes the identical-graph overhead
   comparison from the statistical baseline and removes the publishability sentence; 5.6
   reworded. Agreed.
8. **Evidence attribution and support language.** Changed: section 1 attributes the fast
   run to the coordinator and the slow run to the worker; "true by construction" removed;
   build-only observations are labelled as worker observations throughout; the varying-
   slope row of the matrix separates the source-inspection fact from validation. Agreed.

Preserved positions, with reasons: the recommendation to accept only the brms left-hand
syntax (D1) stands because the parser already owns that side and one syntax is easier to
verify; categorical-before-slopes (D4) stands for the reasons in 4.2; CI before the M2 merge
(D5) stands because a result not tied to a PR head is the only evidence a PR carries today.
The review did not ask for these to change; they remain Andrew's decisions.

**Revision 3 (final clarifications, `2026-09-09-autobayes-roadmap-final-clarifications.txt`).**
No arithmetic or software was run for this pass; the changes are specification wording.

1. *Numerical-domain boundary.* The cap $2^{31}-1$ is now inside the proposed test grid with
   its neighbours ($N_{\max}-1$, $N_{\max}$, and $N_{\max}+1$ refused); the domain is called a
   proposed validation domain throughout and the "every realistic denominator" wording is
   gone; interior large-$n$ cases get an independent reference that avoids the `gammaln`
   subtraction (exact `decimal` summation for $n \le 10^4$, the Stirling/KL
   cancellation-free form with its remainder bound above that); the rule for lowering the
   cap on failure is stated. Agreed.
2. *Predictive-test edge cases.* A9 now handles zero conditional variance by exact checks,
   states that the conditional independence is that of fresh response RNG draws given the
   saved posterior array, and replaces the normal $|z| < 4$ cutoff with a Bernstein
   interval whose false-failure bound ($10^{-6}$ per row) holds for discrete small-count
   rows as well; the symmetric "$\pm 37$" saturation claim is removed in favour of a
   generic statement that the two saturation magnitudes differ. Agreed.
3. *Sensitive boundary regression for D8.* D8 now requires, if approved, a targeted
   likelihood-term regression at $\eta = 40$ for $y = 1$ with a tolerance relative to the
   $4.2\times10^{-18}$ magnitude on both paths, and says explicitly that the existing R1/R2
   suite cannot detect the loss. Agreed.
4. *Benchmark accounting.* An explicit table defines a fit and counts dry-run, cold and warm
   calls (88 in the pilot; the overhead block is optional, D11); cost is stated as uncertain
   until the capped dry run is reviewed; the larger grid stays out of scope. Agreed.
5. *Attribution.* The sentence implying that only a worker reports the fast suite is
   corrected in sections 1 and 4.0: the coordinator ran the 59 fast checks independently,
   the slow checks are worker-reported, and automated CI is absent. Agreed.

**Coordinator final review note, September9.** Planning queue complete and ready for Andrew's
roadmap edits, not implementation approval. The final read corrected the signed negative-boundary
gradient in D8, made the independent reference cover small k at large n, aligned the numerical
tolerance wording, and distinguished finite high-precision arithmetic from an exact result.
No model or test code was edited and none of these proposed references has been executed here.
Use plans/2026-09-09-autobayes-approval-brief.md in the IQSS hub for the concise decision.
