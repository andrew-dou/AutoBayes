# AutoBayes status

## Current checkpoint — 2026-09-13T05:14:47.897062+00:00

PR8 remains OPEN; original F7 recovery and the marginal-reference protocol remain unaccepted. The reference integration is paused. The prior 1a37c3b hosted checks succeeded.

The restart/accounting correction rejects reuse of an existing attempt identity within the unit ledger, freezes completed-attempt time, checks complete finite and consistent budget fields, counts diagnostic requests and rejects fractional confirmation sizes. Thirty-nine tests in the two allowed fake/stub modules pass (0.65s, exit0); evidence docs/m3b1-evidence/guard_v4_checks.json. No new real posterior evaluation, quadrature or sampling. The correction was integrated after the previous worker completed; its checkout is retained. Await hosted checks on this update, then independent final acceptance review of the controller. Numerical continuation still requires a bounded protocol decision; do not start it automatically.

## M3b-1 slice (2026-09-11): independent varying intercept and one numeric varying slope

- Dispatched by the coordinator after the M3a-2 acceptance
  (`memory/reviews/2026-09-10-autobayes-m3a2-repair-review.md`; dispatch
  `memory/dispatches/2026-09-10-autobayes-m3b1-independent-slopes.txt`). Branch
  `m3b1/independent-varying-slope` from `main` at `95dce8c` in the worktree
  `projects/AutoBayes-worktrees/m3b1-varying-slope`; session `e5628a7d-e572-4bad-9feb-82ee8361c261`.
  Decision and acceptance record `docs/m3b1-decision.md` committed at `8307a90` before any code, after
  a build observation of Bambi's `(1 + x | g)` (two independent group terms; the slope block is the raw
  covariate times the one-hot matrix, uncentred under both centring settings; formulae rejects `||`).
- **M3a-2 closure recorded here (per REPOSITORY_WORKFLOW):** PR https://github.com/andrew-dou/AutoBayes/pull/7
  merged by andrew-dou at `95dce8c3b57bb949566f161a306503bcec2527c2` (2026-09-11T00:05:13Z); the merge tree
  `afe73f3586041ac0dd043a19121036f9c344cc87` equals the reviewed head `aa2e4dc`'s tree (verified with
  `git rev-parse` and `git diff`). No process used the M3a-2 worktree (only the session shell's working
  directory). Its five ignored `evidence-runs/` directories were copied with SHA-256 sums (28 files) to
  `iqss-research/artifacts/2026-09-10-autobayes-m3a2-worker-evidence/` with a checkpoint note; the local
  `main` ref was fast-forwarded to `95dce8c`; the new worktree was created from `origin/main` at the
  absolute path; then the clean merged worktree `AutoBayes-worktrees/m3a2-interaction` was removed and
  `m3a2/numeric-by-categorical-interaction` deleted locally and on the remote. The original M1 planning
  checkout at `projects/AutoBayes` (dirty) and the independent review artifacts were not touched.
- **Delivered (this slice):** `(1 + x || g)` (and `(x + 1 || g)`; `(1 || g)` = `(1 | g)`) on the three
  families: `u_j = sd0 z0_j`, `v_j = sd1 z1_j`, independent standard-normal offsets, `eta = fixed + u[g] +
  x v[g]` with the RAW covariate (Bambi does not centre group columns; verified at build time, kill
  criterion otherwise); `(1 + x | g)` refused as brms's correlated form with the independent spelling
  named; implicit-intercept, slope-only, multi-slope, categorical/interaction-slope and absent-fixed-term
  forms refused; `sd={"Intercept": ..., "x": ...}` required for a varying slope (scalar `sd` kept for
  varying-intercept models; missing/unknown keys refused); names `x|g_sigma`, `x|g_offset`, `x|g`;
  `value_point(sd_slope=, z_slope=)`; `Design.Z_slope`; `describe()` states both SDs, the two offset
  vectors, the raw-x independence statement and the refused correlated form; fit records carry both SD
  priors and the slope; predictions in-layer for every family: known groups `a + Xb + u[g] + x v[g]`,
  `population` draws one independent (u, v) pair per new label per draw, `resample_observed` takes both
  effects from one observed group's joint draw. Version 0.5.0.
- **Tests** (clean locked environment, fresh evidence directories, real pytest exit codes):
  `tests/test_varying_slope.py` 18 passed (grammar and refusals; the group design from the original rows and
  labels for 3 families x 2 centrings with independently selected binomial rows; the separately written
  reference density with both SD priors, two offset vectors and both Jacobians against PyTensor and JAX
  values and full gradients; deterministic synthetic-posterior predictions for known and unseen groups in
  both modes with the pairing verified; SD prior forms; describe/record/value_point). Fast suite
  **365 passed, 8 deselected (312 s; pytest exit 0)**. Slow suite **1 failed, 7 passed (54 s; pytest exit 1)**: the failed test is F7, the prespecified
  recovery, whose `x|g_sigma` 99 % HDI [0.100, 0.27] excludes the generating population SD 0.30 on both backends
  (NumPyro also has `1|g_sigma` R-hat 1.0117 and `Intercept` bulk ESS 395); FAILED at the original thresholds,
  evidence frozen (`docs/m3b1-evidence/F7_*`); see the 2026-09-12 repair bullet for the corrected estimand
  statement, fit accounting and decision proposal. Independent regressions 27 passed (nine label-identity, three M2 normalisation, fifteen M3a-2 original-input). Example on both backends in `docs/m3b1-evidence/`.

- **Review repair (2026-09-12, `memory/reviews/2026-09-12-autobayes-pr8-review.md`; probes
  `artifacts/2026-09-12-autobayes-m3b1-review/`, preserved).** (1) `value_point` tested the conjunction of
  `sd_slope` and `z_slope`, so either alone was silently ignored on a varying-intercept model; now either extra
  is refused by name, and a varying-slope model refuses a missing one by name (four-way presence coverage on
  both model forms, valid and wrong shapes; numerical targets unchanged). (2) New
  `tests/test_varying_slope_combined.py`: the supported combination `dose*category + (1 + dose || site)` for
  the three families and both centrings from the ORIGINAL rows and labels with explicit mathematics and no
  library reference helper (design, fully normalised joint density with both SD priors and both Jacobians,
  full gradients on both paths against central differences, synthetic-posterior predictions with the (u, v)
  pairing for known and unseen sites), and the analytic conditional-Gaussian posterior with BOTH group-effect
  blocks at fixed SDs: prior precision and design built independently from the original inputs, the graph's
  gradient equals `-P (theta - mean)` and its Hessian equals `-P` on the PyTensor and JAX paths; no fit
  involved. (3) Reporting corrected in the record, the PR body and this checkpoint: the gated `x|g_sigma` is the
  population SD of the Normal group slopes; the realised sample SD of the 30 simulated slopes (0.2197) is a
  different random quantity and not a substitute truth; a single 99 % interval at one generating value carries
  no repeated-sample coverage guarantee and no 5 % familywise statement follows; the investigation supports
  rare realised variation as a possible explanation only. **F7 stays FAILED** at the original thresholds, its
  evidence frozen; actual fit accounting: F7 sampled in two slow suites (same seed, same failure), one
  investigation fit (PyMC 2 x 500 + 500) and one separate live demonstration fit on other data; nothing widened,
  reseeded, skipped or removed. A decision proposal (deterministic correctness / sampler convergence /
  finite-data recovery separated; a truth-free marginal-posterior check of the SDs and a fixed-budget
  24-replicate coverage study, both prespecified) is in the record for the coordinator; no new recovery fits
  were run in this repair. Repair record `4a8ea4f`; repair `d572df8`. Checks on the candidate (real pytest exit
  codes, `docs/m3b1-evidence/checks-repair.txt`): combined module 9 passed (85 s; pytest exit 0); coordinator probe exit 0, six composite cases with max |logp error| 1.42e-14, both partial-argument cases now refused, 0 fits (`docs/m3b1-evidence/coordinator_probe_after_repair.json`);
  `tests/test_varying_slope.py` 18 passed (pytest exit 0); fast suite **374 passed, 8 deselected (322 s; pytest exit 0)**; independent regressions 27 passed (pytest exit 0). The slow suite was
  not re-run (it would sample F7 again); its recorded result remains 1 failed (F7), 7 passed.
- **Second review correction, marginal reference and cap overrun (2026-09-12/13,
  `memory/reviews/2026-09-12-autobayes-pr8-repair-acceptance-review.md`;
  `artifacts/2026-09-13-autobayes-marginal-cap-review/`).** The value_point repair and the deterministic checks
  were independently confirmed (17 tests incl. 8 crossed-fixture variants, 0 fits); the crossed category/site
  fixture is adopted in `tests/test_varying_slope_combined.py`. Proposal corrected (history kept): the
  24-replicate fixed-hyperparameter study is a fixed-parameter coverage study, not SBC (true SBC draws parameters
  from the fitting prior, then data, and handles posterior autocorrelation; Stan User's Guide), its uniform-rank
  gate and the "coverage failure implies a defect" claim withdrawn and the study not approved or run; the
  `pm.do` conditional check withdrawn (it changes the target); "convergence established" withdrawn for the F7
  run (NumPyro missed the R-hat/ESS gates); coverage, computation and convergence kept distinct. New independent
  deterministic reference for the FULL Gaussian F7 posterior of the three scales (`tests/reference_marginal.py`,
  Gaussian coefficients and both group blocks integrated analytically from the original rows; low-rank algebra
  equal to dense covariance on a crossed fixture and the F7 data, `tests/test_marginal_reference.py`); protocol
  recorded before evaluation (`f596d7b`). **Cap overrun:** the quadrature driver checked its evaluation counter
  after computing each batch and reset it per process; attempt 1 (2,289,209 evaluations, raised, no output),
  attempt 2 (1,738,745, wrote the saved result) and a tail addendum (97,600, no budget check) total 4,125,554
  evaluations against the 2,000,000-evaluation unit cap (206 %); the shell wrappers also mis-reported exit codes.
  The saved `targets_met=True` is withdrawn: the lower log-sd1 face missed the -25 tail criterion (-20.04) and the
  CDF-stability condition was not checked at every reference point with resolution and domain separated; the
  reference is **incomplete/exploratory** (means 0.3549/0.1758/0.4245, P(sd1 <= 0.30 | y) = 0.9981, saved MCMC
  means within 4 stored MCSE: exploratory observations about computation only). F7 stays FAILED, its data, seed,
  estimand, thresholds, test and evidence untouched; no sampling. Repair, code only and not run: budget
  controller `tests/quadrature_budget.py` (cumulative persisted ledger never reset, pre-batch remaining check,
  completed vs attempted, deadline, checkpoints, stop on exhaustion) with fake-evaluator tests (5 passed), the
  driver refactored onto it with acceptance requiring every condition, `scripts/run_logged.py` for real exit
  codes, the ledger seeded with the 4,125,554 spent evaluations so any rerun is refused. A scoped completion
  proposal (about 1.0 million evaluations, new authorisation required) is in the record. Checks in this unit:
  targeted deterministic modules 39 passed (exit 0); controller tests 5 passed (exit 0); no fast-suite rerun.
- **Second budget-guard review (2026-09-13, `memory/reviews/2026-09-13-autobayes-budget-guard-independent-review.md`),
  repaired in code only, nothing run:** reservations persisted before work and never refunded (a raising or crashing
  batch stays counted; requested / reserved-uncertain / confirmed distinct; explicit ownership with recorded takeovers;
  stale writes refused); one unit deadline in the ledger with cumulative measured time, conservative estimates for
  unconfirmed batches, expired retries refused, post-batch overrun recorded (granularity one batch); fail-closed
  acceptance against an explicit complete inventory (omitted/unchecked/false/NumPy-false/malformed/unexpected all
  block); driver restructured (guarded main, attempt name no longer shadowed, the two diagnostic marginal calls
  charged, full inventory of six faces and 14 CDF conditions always present, stub-tested assembly for complete,
  unmet-face, stopped and exhausted paths); tail reporter reports ledger cumulative counts; historical spend
  restated as 4,125,558 under the new convention (raw outputs keep 4,125,554). Targeted tests 36 passed (exit 0),
  of which only 15 were fake/stub; the other 21 (marginal-reference validation, combined-feature module) performed
  real deterministic density/gradient calculations, reported separately (correction of the earlier "fake/stub"
  wording). Reference still incomplete/exploratory; F7 FAILED; PR open. Commit `c59c8b8`.
- **Third budget-guard review (2026-09-13, `memory/reviews/2026-09-13-autobayes-guard-v2-review.md`), repaired in code
  only:** one central gate for batches and diagnostic charges (charge over cap now refused before work; confirmations
  validated by reservation id, size and finite time); measured failure time recorded and whole-attempt wall time
  counted, unknown crash duration conservatively charged on takeover, expired units refused at construction,
  post-batch overrun recorded; acceptance and exit code require run validity (stopped/over-budget/unconfirmed
  units never accepted; exact-cap valid completion passes); OS-level exclusive lock rejects simultaneous owners.
  Local checks: the two allowed fake/stub modules only, `--noconftest`, 18 passed (exit 0), zero posterior
  evaluations; the coordinator's three cases replayed (`docs/m3b1-evidence/guard_v3_fake_reproduction.txt`).
  Commit `0043517`.
- PR: https://github.com/andrew-dou/AutoBayes/pull/8 (base `main`, author andrew-dou). Record `8307a90`; implementation `9b15865`;
  reviewed head `8907113` (request changes); repair record `4a8ea4f`; repair `d572df8`; second-review checkpoint `24eefec`;
  protocol/reference `f596d7b`; cap-overrun correction `d2e8899` (checkpoint `fe78c3d`); budget-guard repair `c59c8b8`
  (checkpoint `37f4334`); third-review guard repair `0043517`; this STATUS checkpoint follows it (CI runs recorded in the PR
  comment and the final report). Left open for independent
  review with the F7 gate failure reported, not merged by this session; the M3b-1 worktree is kept until
  that review passes. Next approved increment after acceptance: correlated varying slopes (LKJ), separate PR.

## M3a-2 slice (2026-09-10): one numeric-by-categorical fixed interaction

- Dispatched by the coordinator after the M3a-1 acceptance
  (`memory/reviews/2026-09-10-autobayes-m3a1-repair-acceptance.md`; dispatch
  `memory/dispatches/2026-09-10-autobayes-m3a2-interaction.txt`). Branch
  `m3a2/numeric-by-categorical-interaction` from `main` at `df8195a` in the worktree
  `projects/AutoBayes-worktrees/m3a2-interaction`; session `e5628a7d-e572-4bad-9feb-82ee8361c261`.
  Decision and acceptance record `docs/m3a2-decision.md` committed before any code, after
  measuring formulae/Bambi's `x:z` and `x*z` expansion, product-column centring and naming.
- **M3a-1 closure recorded here (per REPOSITORY_WORKFLOW):** PR https://github.com/andrew-dou/AutoBayes/pull/6
  merged by andrew-dou at `df8195a39f5ef877e1c59186d69107e207063aa4` (2026-09-10T21:04:24Z; a squash whose
  tree equals the branch head `15cd913`, verified with `git diff`), confirmed through GitHub before any
  checkout changed. No process depended on the M3a-1 worktree (only the session shell's working
  directory). Its six ignored `evidence-runs/` directories were copied with SHA-256 sums (24 files) to
  `iqss-research/artifacts/2026-09-10-autobayes-m3a1-worker-evidence/` with a checkpoint note; the
  local `main` ref was fast-forwarded to `df8195a`; the new worktree was created from `origin/main`
  at the absolute path; then the clean merged worktree `AutoBayes-worktrees/m3a1-categorical` was
  removed and `m3a1/categorical-predictors` deleted locally and on the remote. The original M1
  planning checkout at `projects/AutoBayes` (dirty) was not touched.

- **Delivered (this slice):** `x:z` (and `x*z` = `x + z + x:z`, measured identical) between a
  numeric term and a declared categorical term, both main effects required (lower-order-omitting
  forms refused with the explanation), deterministic canonical expansion (main effects in first
  appearance order, then interactions spelled `numeric:categorical`), duplicates/overlaps and
  unsupported interactions refused; design columns `x:z[l]` = `x * 1[z = l]` verified against an
  independent construction from the supplied labels; PyMC variable `x:z` with dims `x:z_dim`;
  `Priors.b` keys per canonical term, term-wide or per level; centring of every expanded column
  at its own mean with the product-column means printed by `describe()` (not the product of the
  means) and the reported-intercept transformation verified against the sampler's own log
  density; predictions per level and slope with unseen category levels refused; fit records carry
  the interaction and canonical formula. Version 0.4.0.
- **Tests** (clean locked environment, fresh evidence directories): `tests/test_interaction.py`
  35 passed (37 s) (grammar and refusals; design against an independent construction for 3 families x 2
  references x 2 centrings on shuffled data; the sampled-to-reported intercept check; density,
  Jacobian and gradients on both paths for all three families with term-wide and per-level
  priors; the reference change with slopes; prior refusals; describe/record; predictions by hand
  with per-level slopes, mixed rows, expected vs predictive counts, unseen levels; binomial
  zero-trial drop before the means). Fast suite **332 passed, 7 deselected (288 s)**. Slow suite **7 passed (49 s)** including the
  prespecified interaction recovery: 0 divergences, max R-hat 1.0094889 (PyMC) and 1.0045411 (NumPyro), min bulk ESS 520/662, all eight truths inside their 99 % HDIs, backends within 4 combined MCSE (largest ratio 0.41). Example on both backends in `docs/m3a2-evidence/`.

- **Review repair (2026-09-10, `memory/reviews/2026-09-10-autobayes-m3a2-review.md`).** The
  one-interaction scope was not enforced (`x:z + w:z` built in all families: the coordinator's three
  scope checks failed on `cc3081a`, reproduced before any change); `describe()` kept the constant
  categorical-contrast sentence when the category interacts (the contrast is `b_z[l] + x * b_x:z[l]`,
  `b_z[l]` alone is at x = 0); the density/gradient tests fed `model.data` to the reference. Repair
  record committed before code (`7f48b08`); repair commit `76beb7b`: at most one distinct interaction
  after canonicalisation (shared numeric side, shared categorical side, disjoint and double shorthand
  all refused), conditional link-scale contrast wording in `describe()` (constant contrast kept for
  categories without an interaction) with a nonzero-x numerical contrast test, original-source oracles
  (the reference reads the caller's frame and selects positive-trial rows itself) and the coordinator's
  original-input checks adopted as `tests/test_interaction_original_inputs.py`. After: coordinator's
  file 15 passed (pytest exit 0); new module 16 passed; `tests/test_interaction.py` 35 passed; fast suite **348 passed, 7 deselected (343 s)**; slow suite
  **7 passed (49 s)**; independent regressions 12 passed. Real pytest exit codes recorded; frozen coordinator
  evidence untouched.
- PR: https://github.com/andrew-dou/AutoBayes/pull/7 (base `main`, author andrew-dou). Record `f0d229f`; implementation `9c2ef95`;
  reviewed head `cc3081a` (request changes); repair record `7f48b08`; repair `76beb7b`; this STATUS checkpoint follows it
  (CI runs recorded in the PR comment and the final report). Left open for independent
  review; not merged by this session; the M3a-2 worktree is kept until that review passes. Next approved
  increment after acceptance: independent varying slopes, then correlated slopes, in separate PRs.

## M3a-1 slice (2026-09-10): one treatment-coded categorical predictor

- Dispatched by the coordinator after the M2 acceptance
  (`memory/reviews/2026-09-10-autobayes-m2-final-review.md`; dispatch
  `memory/dispatches/2026-09-10-autobayes-m3a1-categorical.txt`). Branch
  `m3a1/categorical-predictors` from `main` at `f8e116b` in the worktree
  `projects/AutoBayes-worktrees/m3a1-categorical`; session `e5628a7d-e572-4bad-9feb-82ee8361c261`.
  Decision and acceptance record `docs/m3a1-decision.md` committed at `f134f66` before any code,
  after measuring formulae/Bambi's categorical behaviour (sorted levels unless an ordered
  Categorical; dummies centred at their proportions; per-level array priors; unseen levels refused).
- **M2 closure recorded here (per REPOSITORY_WORKFLOW):** PR https://github.com/andrew-dou/AutoBayes/pull/5
  merged by andrew-dou at `f8e116bd76d0becaadebeab4b4ae75ce1aa09f32` (2026-09-10T19:33:43Z), verified
  through GitHub before the transition. The M2 worktree's ignored candidate evidence (seven
  `evidence-runs/` directories) was copied to
  `iqss-research/artifacts/2026-09-10-autobayes-m2-worker-evidence/` with a checkpoint note; the
  clean merged worktree `AutoBayes-worktrees/m2-binomial` was then removed and the branch
  `m2/binomial-trials` deleted locally and on the remote; the local `main` ref was fast-forwarded to
  `f8e116b`; the original M1 planning checkout at `projects/AutoBayes` (dirty) was left untouched.
- **Delivered (this slice):** `categorical={column: reference}` on `ab.gaussian/bernoulli/binomial`
  (string, pandas Categorical or integer-coded columns; an undeclared integer column stays numeric;
  an undeclared label column is refused with the declaration to use); levels = sorted string labels
  of the fitted rows (after the binomial zero-trial drop), reference required and validated,
  unused declared categories reported and given no coefficient; treatment coding through
  formulae with an ordered Categorical built by the library (reference first), the design
  verified column by column against an independent one-hot construction, PyMC variable `z` with
  dims `z_dim` and posterior rows `z[level]`; `Priors.b` per term or per level with every
  missing/unknown/ambiguous/reference/dummy-label key refused; `model.coefficients` naming map;
  `value_point` over design columns; `describe()` with the reference, contrast meaning,
  proportion-centred intercept on the link scale (and the non-average warning for logit
  families); predictions per level with unseen category levels refused for every kind and
  `new_groups`; fit records carry the categorical specification. Version 0.3.0.
- Wording carried: the M2 record's aside about the coefficient being "largest at n/4" corrected
  (maximum at n/2; the measured error was worst at n/4); binomial `describe()` no longer says the
  coefficient is computed as data.
- **Tests** (clean locked environment, unique evidence directories): `tests/test_categorical.py`
  78 passed (33 s): declaration/validation and dtypes, the design against an independent
  construction (3 families x 3 term orders x 3 references x 2 centrings, plus two categorical
  terms), density with and without the Jacobian and gradients (PyMC, JAX, central differences of
  the extended reference) for all three families with term-wide and per-level priors, the
  reference-change map (|det M| = 1; linear predictor and likelihood equal; independent priors a
  different model; equal under the transformed joint prior), eleven prior refusals, `describe()`
  and the fit record, predictions per level with unseen levels refused for every kind and
  `new_groups`, binomial levels after the zero-trial drop, `value_point`. Fast suite **261 passed,
  6 deselected** (272 s). Slow suite **6 passed** (43 s) including the prespecified categorical
  recovery on both CPU samplers: 0 divergences, max R-hat 1.0066 (PyMC) and 1.0099851 (NumPyro, below
  the 1.01 gate), min bulk ESS 669/761, every truth inside its 99 % HDI, backends within 4 combined
  MCSE (largest ratio 0.43). Example on both backends with the comparison and the GPU refusal in
  `docs/m3a1-evidence/`. One pre-code statement of the record was corrected during
  implementation (the sign of the induced prior covariance under a change of reference), noted
  in the record.
- **Review repair (2026-09-10, `memory/reviews/2026-09-10-autobayes-m3a1-review.md`, regressions
  `artifacts/2026-09-10-autobayes-m3a1-independent/test_label_integrity.py`, nine failures on
  `393e4ec`).** Numeric categorical columns were cast through float64 and int64, merging codes above
  2^53 and corrupting unsigned codes at or above 2^63 (a public three-level Gaussian model built two
  levels); complex scalars in object or category containers bypassed the refusal. Repair record
  committed before code (`04a4e2f`, with the before-evidence): one scalar label rule (exact integers
  of any width, floats only if integral and at most 2^53, strings; booleans, complex, missing and
  other types refused; integer/decimal-string equivalence intentional; mixed text/number columns and
  colliding keys refused as ambiguous) applied to data, newdata, references, declared categories and
  prior keys; containers read as Python objects; the model keeps the supplied column's labels and
  checks the design against them; the test reference applies the same rule. Repair commit `3b711f2`.
  Coordinator's regressions after the repair: 12 passed; new `tests/test_label_integrity.py` 36 passed;
  fast suite **297 passed, 6 deselected (254 s)**; slow suite **6 passed (39 s)** (recovery unchanged; R-hat 1.0099851 for NumPyro is
  below the 1.01 gate, not at it). Fresh evidence directories; committed evidence untouched.
- PR: https://github.com/andrew-dou/AutoBayes/pull/6 (base `main`, author andrew-dou). Record
  `f134f66`; implementation `d47ca7f`; reviewed head `393e4ec` (request changes); repair record
  `04a4e2f`; repair `3b711f2`; this STATUS checkpoint follows it (CI runs recorded in the PR comment
  and the final report). Left open for independent review; not merged by this session; the
  M3a-1 worktree is kept until that review passes. Next approved increment after acceptance:
  numeric-by-categorical interaction, in a separate PR.

## M2 slice (2026-09-10): grouped binomial counts, logit link

- Approved by Andrew 2026-09-09 (evening; `iqss-research/plans/2026-09-09-autobayes-approval-brief.md`),
  dispatched after the foundation merge. Branch `m2/binomial-trials` from `main` at `4eee25c`
  in the worktree `/Users/andrewdou/iqss-research/projects/AutoBayes-worktrees/m2-binomial`;
  session `e5628a7d-e572-4bad-9feb-82ee8361c261`. Decision record `docs/m2-decision.md`
  committed at `93ab82d` before any implementation (defaults, target, construction chosen
  after measuring the built-ins, high-precision reference and error budget, A1-A9/B1/C1-C3).
- Planning-file provenance: the original checkout at `projects/AutoBayes` (branch `m1`, dirty)
  still holds the uncommitted planning docs; `main` already contains identical copies (the
  proposal is byte-identical; STATUS on `main` is a superset with the coordinator's corrected
  planning checkpoint, committed in the foundation branch at `552bdfe`). Nothing was copied
  or replaced in this slice. The foundation worktree's ignored `evidence-runs/` candidate
  directories were removed together with that worktree on 2026-09-10 and are not preserved
  locally; the hosted artifacts (`fast-evidence-34430932197-1`, `slow-evidence-34439415443-1`
  and the later run artifacts) and the coordinator's independently preserved run artifacts
  are the surviving copies.
- Cleanup commit `b0b3497` removed a vacuous `... or True` assertion from
  `tests/test_evidence_provenance.py` (explained in its message).
- **Delivered**: `ab.binomial("successes | trials(n) ~ x + (1 | group)", df, priors)` (brms syntax
  only; Bambi's `p(k, n)` refused), counts validated before any cast (bool, complex,
  non-integral, non-finite, negative, `k > n`, above the cap refused; the cap was `2^31-1` at this head and is `1e6` since the repair), zero-trial rows dropped
  and reported before centring, all-zero-trial data refused, constant outcomes accepted with a
  separation warning (M1's Bernoulli refusal retained), likelihood
  `log C(n,k) - k softplus(-eta) - (n-k) softplus(eta)` on `stable_softplus` in a Bambi custom
  family with `n` passed as Bambi's own binomial passes it; predictions computed in this layer:
  `probability` (p), `mean` (p, expected = n p; brms `posterior_epred`), `response` (+ integer
  `Binomial(n, p)` draws), trials column required only for `mean`/`response` newdata, zero-trial
  prediction rows deterministic zero, M1 new-group identity/sharing semantics; fit records carry
  trials, dropped rows and warnings. Version 0.2.0 (`uv.lock` changed only in the project entry).
- **Measured before choosing the construction** (`docs/m2-decision.md`): Bambi's built-in
  binomial fails 49 of 130 grid points (values and gradients, both paths, from |eta| = 12 at
  large n); `pm.Binomial(logit_p=)` fails 60 of 130 including JAX gradient failures.
- **Tests** (clean locked environment): `tests/test_binomial.py` 26 passed, including the full
  A5 grid (n in {1, 2, 10, 1e3, 1e6, 1e9, 2^31-2, 2^31-1}, k at 0/1/mid/n-1/n, 13 logits to
  +-1e6, PyTensor and JAX values and full gradients against the 60-digit `decimal` reference
  with exact / Stirling `lnGamma`), n = 1 equivalence with Bernoulli, expansion to Bernoulli
  rows with aligned centring (and the documented difference with unequal trials), invalid
  inputs, three prediction kinds, Bernstein-bounded conditional predictive check with exact
  zero-variance and saturated cases. Full fast suite: **117 passed, 5 deselected** (69.7 s). Slow suite 5 passed
  (31.3 s), M2 B1: PyMC 0 divergences, R-hat 1.0062, bulk ESS 781; NumPyro 0 divergences,
  R-hat 1.0073, bulk ESS 681; truth inside every 94 % HDI; backends within 4 combined MCSE
  (largest ratio 0.33). Evidence in `evidence-runs/20260910T052144.898878Z-…-p25085-648eaa/`.
- Example evidence `docs/m2-evidence/` (both backends, agreement, GPU refusal). GPU execution
  for this family: **pending**. The Bernoulli boundary-form change (D8) was not made.
- Limitations: one recovery is not calibration; in one simulated data set the intercept is
  identified relative to the realised mean of the group effects (the example prints it); the
  validated count/logit domain is the tested grid; no calibration or benchmark claims.
- **Review repair (2026-09-10, `memory/reviews/2026-09-10-autobayes-m2-review.md`, artifacts
  `artifacts/2026-09-10-autobayes-m2-independent/`).** Finding: the coefficient was a float64
  difference of three `gammaln` values, so the normalised density near a binomial mode carried
  absolute errors of 3.3e-6 at n = 1e9 and 1.1e-5 at the cap against the prespecified 1e-8;
  nine of fifteen probed near-mode cases failed on each path, and the mixed-grid A5 check had
  masked it because k = 0 and k = n rows at the same logit made the total O(n). Before coding,
  the repair record in `docs/m2-decision.md` (commit `8309449`) tabulated the failed cases and
  measured, with an accurate coefficient, where float64 itself gives out: the likelihood terms'
  magnitude (~n ln 2) bounds the near-mode value (1.8e-8 at n = 1e8) and the rounding of
  expit(eta) bounds the gradient (1.2e-8 at n = 1e9), independent of any reformulation short of
  the deviance form. Repair: (a) the coefficient is computed as data per row (exact summation
  for min(k, n-k) <= 1000, Loader decomposition with the `stirlerr` series above) and never
  as a `gammaln` difference; (b) the accepted count cap is lowered by the approved rule to
  **1e6 per row**, the largest acceptance-grid value that passes on both paths with a margin
  of at least 25 (values) and 400 (gradients); 1e7 also passed at probed points with a margin
  of about seven and was not adopted; the deviance form is recorded as the route if counts
  above 1e6 per row are ever needed. Tolerance unchanged.
- Repair tests: per-row normalised terms and per-row derivatives over the whole acceptance
  grid (n in {1, 2, 10, 1e3, 1e5, 999999, 1e6}; k at 0/1/2/10/n/4/n/2/3n/4/n-2/n-1/n; the 13
  extreme logits plus the mode and six offsets), homogeneous near-mode models with a slope,
  sd = 0.3 and random offsets (16 cases), the reviewer's two-row construction at the cap and
  its neighbour, the mixed grid, refusal of 1e6+1 and of the three previously failing n.
  `tests/test_binomial.py` 50 passed (122 s). After the repair the reviewer's construction has
  worst error 1.6e-10 at the cap (was 1.4e-9 at 1e6 and 3.3e-6 at 1e9 before). Full suites on
  the repaired tree: fast suite **141 passed, 5 deselected** (144 s); slow suite **5 passed** (30.6 s; new run-specific evidence directories, earlier ones untouched). Example evidence regenerated with
  the repaired code; the pre-repair outputs are archived under `docs/m2-evidence/before-repair/`.
- Wording fixed with the repair: the binomial centred intercept is the per-trial success
  log-odds; README and SKILL count three families; the package description names M2; the cap
  and precision statements describe the tested domain, not exactness for every input.
- Evidence provenance: the foundation worktree's ignored candidate evidence was removed with
  that worktree on 2026-09-10 and is not preserved locally (hosted artifacts and the
  coordinator's preserved run artifacts remain); the M2 runs below write to new run-specific
  directories under `evidence-runs/`, earlier ones untouched.
- **Second review correction (2026-09-10, `memory/reviews/2026-09-10-autobayes-m2-distribution-review.md`,
  artifacts `artifacts/2026-09-10-autobayes-m2-repair-independent/`).** Finding, reproduced on
  `d98b730` with the public API on both paths: the per-row coefficient was passed into the
  `CustomDist` as a fixed parameter tied to the observed outcome, so the exposed RV's `logp`
  was normalised only at that outcome (n = 2, eta = 0: masses 0.75 or 1.5 instead of 1),
  `logp` and `random` described different distributions, and out-of-range or non-integral
  values had finite mass. The fitted-data likelihood and gradients were unaffected (the
  coefficient is constant in eta there), so this does not by itself change the reported
  posteriors. Correction record written before coding (`docs/m2-decision.md`, commit
  `8d76d64`): the coefficient is now a function of the evaluated value inside the graph on
  both paths (0 at the boundary, Loader/`stirlerr` decomposition on clipped arguments under a
  `switch`), with -inf off the integer support {0, ..., n}; `random` unchanged; the family
  passes only `n`. Measured coefficient error 1.2e-10 at n <= 1e6 on both paths (one ulp of
  its magnitude; table in the record), so the 1e6 cap and its margins stand. Before/after
  evidence in `docs/m2-evidence/correction/` (normalisation probe, N7 example log density
  changed by at most 4.5e-13, coefficient accuracy, scripts). New tests N1-N5 (support
  enumeration against `math.comb` and a decimal expit with mass 1, observed-outcome
  independence, symbolic value reused after compilation, 20000 draws against the mass under
  a Hoeffding bound, zero mass for invalid values on both paths, integer and float tensors).
  Wording: the current 1e6 cap is stated at the top of the decision record with 2^31-1
  labelled historical; the refusal message and SKILL no longer suggest splitting large
  denominators across rows (it changes the unweighted centring); the docstring cites the
  measured table; the gradient-floor sentence describes the tested evaluation only.
  `tests/test_binomial.py` **92 passed (190 s)**; fast suite **183 passed, 5 deselected (240 s)**; slow suite **5 passed (31 s)** (new
  run-specific evidence directories, earlier ones untouched). Example outputs not re-sampled
  (N7). No D8 change, no M3 work, no GPU use.
  Correction commit `cb5e5a1` (record `8d76d64` before it).
- PR: https://github.com/andrew-dou/AutoBayes/pull/5 (base `main`, author andrew-dou). First head `1fcb2cf` (fast checks green: runs 34441129737, 34441162554); reviewed head `49052ac` (green); repair head `5a43caf` (fast checks green for both the push, run 34482262505, and the pull-request, run 34482266661, events); STATUS checkpoint `d98b730`; correction record `8d76d64`; **correction head `cb5e5a1`** followed by this STATUS checkpoint (CI runs recorded in the PR comment). Left open for independent review; not merged by this session; the M2 worktree is kept until that review passes.

## Foundation slice (2026-09-09): reproducible CPU checks (section 4.0 of the proposal)

- Andrew approved the software plan on 2026-09-09 (evening); this is the first implementation
  slice, dispatched by the coordinator. Branch `foundation/reproducible-cpu-checks` from the
  reviewed M1 head `bd63348`, in the worktree
  `/Users/andrewdou/iqss-research/projects/AutoBayes-worktrees/foundation-cpu-checks`
  (the M1 checkout at `projects/AutoBayes` is left as it was, dirty with the planning docs).
  Session `e5628a7d-e572-4bad-9feb-82ee8361c261`. Depends on PR #3 (M1), still open.
  PR: https://github.com/andrew-dou/AutoBayes/pull/4. PR #3 was merged by the coordinator
  (`main` at `52d2b61` contains `bd63348`), so PR #4 was retargeted to `main` on 2026-09-10.
  **Merged by the coordinator: merge commit `8015374` on `main`** (head `626ce24`).
- **Hosted slow workflow, first execution (2026-09-10, after the merge put `slow.yml` on
  the default branch):** run 34439415443 on `main` at `8015374`, `workflow_dispatch`,
  ubuntu-latest, locked install: **4 passed, 91 deselected in 99.7 s**, whole run 2.25
  minutes, artifact `slow-evidence-34439415443-1` (6 KB) uploaded, same versions as the
  local environment report. The README's "CPU-validated" definition now holds for both
  suites on the hosted runner as well as locally.
- **Cleanup (per `REPOSITORY_WORKFLOW.md`):** the foundation worktree
  `projects/AutoBayes-worktrees/foundation-cpu-checks` (clean, merged, no job attached) was
  removed and the branch `foundation/reproducible-cpu-checks` deleted locally and on the
  remote after this checkpoint was pushed. The M1 checkout at `projects/AutoBayes` and the
  `m1/bernoulli-varying-intercept` branch were left to the coordinator, as instructed. The
  locked environment is rebuilt in any checkout with `uv sync --locked --extra jax --extra dev`.
- Next: M2 binomial counts, after the coordinator's review of this merged slice, using the
  approved defaults recorded in the planning checkpoint below.
- **Pinned, locked stack**: `pyproject.toml` now pins the tested versions exactly
  (`requires-python >=3.11,<3.12`; pymc 5.28.5, pytensor 2.38.3, bambi 0.17.2, formulae 0.6.2,
  arviz 0.23.4, numpy 2.4.6, pandas 3.0.5, scipy 1.17.1, xarray 2026.7.0, matplotlib 3.10.9;
  extras `jax` = jax/jaxlib 0.10.2 + numpyro 0.21.0, `dev` = pytest 9.1.1) and `uv.lock`
  (58 packages) fixes the rest. No library was upgraded; the lock reproduces the environment
  every result in `docs/` came from. A clean environment was built from the lock in the
  worktree (`uv sync --locked --extra jax --extra dev`, Python 3.11.0) and used for the runs
  below.
- **Evidence paths**: test runs write candidate evidence to `evidence-runs/<UTC time>-<git
  head>/` (gitignored, `AUTOBAYES_EVIDENCE_DIR` overrides; a `run.json` records environment,
  git head and argv). `tests/test_sampling.py` no longer writes into `docs/m*-evidence/`; the
  committed records are immutable history. Acceptance stays the statistical criteria the
  tests assert.
- **New checks**: `tests/test_fresh_process.py` (fresh interpreters: autobayes-first gives
  parallel CPU chains; JAX-first gives `sequential` with the note; `AUTOBAYES_HOST_DEVICES=0`;
  a process without JAX: `environment()` reports it, `backend="numpyro"` raises `DeviceError`
  naming the `[jax]` extra, `backend="pymc"` works), `tests/test_priors_families.py`
  (StudentT / HalfStudentT / Exponential / HalfCauchy end to end against the reference
  density, both families), `tests/test_multi_predictor_predict.py` (two predictors, in-sample
  and new-group paths, both families), `tests/test_diagnostics_panel.py` (pass / investigate /
  fail / n/a statuses and `format_panel`), `tests/test_evidence_provenance.py` (record schema,
  provenance, history untouched). Library change: `inference.resolve_jax_device` turns a
  missing JAX into a `DeviceError` with the fix named.
- **CI**: `.github/workflows/ci.yml` (fast suite on push and pull request, Ubuntu, CPU, locked
  install, 40-minute timeout, cancel-in-progress per ref, evidence + junit artifacts) and
  `slow.yml` (manual, the three sampling checks, 60-minute timeout, artifacts). No GPU
  workflow. Actions are enabled on the repository (`actions/permissions`: enabled); included
  minutes could not be read through the API (the token lacks the `user` scope), so the first
  run is the availability check.
- **Results (executed).** Clean locked environment in the worktree (Python 3.11.0, macOS
  arm64, fresh PyTensor compiledir): fast suite **81 passed, 3 deselected** (65.6 s on the
  second full run; the first full run found one wrong expectation in the new
  `AUTOBAYES_HOST_DEVICES=0` test, fixed by asserting what the switch does: it only
  matters if JAX is initialised before the first fit, because `fit()` sets the host device
  count itself otherwise; a test for that second fact was added), slow suite **3 passed**
  (27.1 s), evidence in `evidence-runs/20260910T024831Z-552bdfe663f3-dirty/`
  (`B1_conditional_gaussian.json`, `B2_recovery.json`, `M1_B1_recovery.json`, `run.json`).
  Hosted CI (`fast checks`, run 34430932197 on head `72f0afe`, ubuntu-latest, locked
  install): **81 passed, 3 deselected in 152 s**, whole run 3.5 minutes, artifact
  `fast-evidence-34430932197-1` uploaded; the run for the previous head was cancelled by
  the concurrency rule as intended. The environment report in CI shows the same versions
  and 2 CPU devices (2-core runner). The slow workflow has not been dispatched in CI.
- Fix-up commit `72f0afe`: `autobayes.egg-info/` (created by the editable install) was in
  the first commit and is now gitignored and removed.
- **Review repair (2026-09-10, `memory/reviews/2026-09-09-autobayes-foundation-review.md`).**
  Both findings reproduced first on head `1a480ac` in a fake repository: two allocations at
  the same head and second returned one directory and the second replaced the first
  `run.json`; an `AUTOBAYES_EVIDENCE_DIR` pointing into `docs/m0-evidence` was accepted and
  the historical `run.json` overwritten. Repair in `tests/evidence.py`: a run directory is
  claimed atomically with `mkdir(exist_ok=False)` under
  `<UTC time to the microsecond>-<git head>-p<pid>-<random token>` (collisions retried with a
  new token); the override names a *parent* for new run directories, never a run; any parent
  resolving (symlinks followed) into `docs/` or containing a `run.json` is refused with
  `EvidenceDestinationError` before anything is created or written; evidence files are
  written with exclusive create, so a second write to the same name fails and the first
  survives. `tests/test_sampling.py` uses the exclusive writer and gained a slow test that
  compares every committed evidence file's bytes with HEAD after sampling.
  `tests/test_evidence_provenance.py` (13 tests) covers: same-instant same-head allocations
  distinct with the first run's files byte-identical; 20 allocations without collision;
  overrides into `docs/`, `docs/m0-evidence`, a not-yet-existing child of it, and a symlink
  into it refused with the repository's file bytes unchanged; an override naming an existing
  run refused with that run's bytes unchanged; an identical override reused on a retry
  yields a second directory and preserves the first; `write_evidence` refuses overwrites and
  unsafe names; the session's run directory is outside `docs/`; committed evidence bytes
  equal HEAD. Executed before/after: the same assertions applied to the old allocation logic
  fail for distinct directories, history refusal, history bytes and retry reuse, and pass on
  the new one (scratch script, not committed). README and SKILL now attribute the three slow
  passes to the local locked environment, state that the hosted slow workflow's execution
  status is recorded here rather than assumed, and bound the lock claim to the versions the
  fit records name (transitive versions were not recorded before 2026-09-10).
- **Repair results (executed, clean locked environment, Python 3.11.0).** Targeted
  regressions `tests/test_evidence_provenance.py`: 13 passed. Full fast suite **91 passed,
  4 deselected** (64.3 s); slow suite **4 passed** (24.0 s, incl. the new history-bytes check);
  run directories now look like `evidence-runs/20260910T030827.102547Z-1a480acd579f-dirty-p19739-487ea9/`.
  Hosted CI on the repair head `425b3ef`: both `fast checks` runs succeeded (push run
  34432238494, pull-request run 34432241934). The manual slow workflow **could not be
  dispatched**: GitHub only accepts `workflow_dispatch` for a workflow file that exists on
  the default branch (`HTTP 404: workflow slow.yml not found on the default branch`), so
  the hosted slow run stays unexecuted until this PR is merged; the three sampling checks
  plus the history-bytes check ran locally in the locked environment (4 passed).
- Docs: README install/test section and supported-path table; `SKILL.md` session start,
  handoff and failure table updated to the implemented behaviour.
- Not in this slice: M2 and later; GPU runs; the Bernoulli boundary-form change (D8) is a
  separate reviewed correctness change if approved.

## Planning checkpoint (2026-09-09): complete for Andrew's review

- Proposal: `docs/coordination/NEXT_MILESTONES_PROPOSAL.md`, revision3 plus final coordinator
  mathematical/reference clarification. Concise decision brief:
  `/Users/andrewdou/iqss-research/plans/2026-09-09-autobayes-approval-brief.md`.
- All dispatched planning/revision units are complete. Claude's final response was observed
  idle; do not repeat the work. Implementation awaits Andrew's edits/approval.
- Recommended sequence: reproducible environment/tests and CPU CI; M2 binomial counts;
  categorical/interactions and varying slopes in separate slices; researcher skill/onboarding
  throughout; separately scoped GPU correctness and a capped benchmark dry run.
- M2 specifies probability/expected-count/response outputs, explicit zero-trial/constant-outcome
  policy, log-sigmoid likelihood, proposed numerical validation domain and independent references.
  These are acceptance specifications, not implemented or validated behavior.
- Benchmark accounting:8 dry-run fits plus40 cold and40 warm =88 proposed pilot fits;
  optional overhead block adds40. No GPU/benchmark run or budget approved by this document.
- Evidence: coordinator independently reran59 fast tests earlier;3 slow passes are
  worker-reported. Only proposal/STATUS documentation modified in the planning unit.
  Implementation head remainsbd63348 on M1; existing PR3 left open, no commit/push/merge.

## M1 checkpoint (2026-09-09, after review repair): Bernoulli varying intercept, logit link

- Branch `m1/bernoulli-varying-intercept` from `main` at `925554b`; session
  `e5628a7d-e572-4bad-9feb-82ee8361c261`. Decision record `docs/m1-decision.md` (target,
  semantics, acceptance checks A1-A8/B1/C1-C3), plus a **repair record** with acceptance
  criteria R1-R6 written before the fix, the executed failure history, and open review items.
- **Review finding (coordinator, reproduced here): merge-blocking.** The PR as first opened
  used Bambi's built-in `bernoulli` family, whose likelihood floors `p` at machine epsilon:
  with two zero and two one observations the likelihood was -72.087 instead of -80 at
  eta = 40 and still -72.087 instead of -600 at eta = 300 (same floor at eta = -1000), with
  the likelihood's gradient contribution vanishing. The earlier test A5 asserted that
  deviation; the "negligible / watch for separation" framing is withdrawn.
- **Repair.** `autobayes/likelihoods.py`: the likelihood is `y*eta - stable_softplus(eta)` in a
  `pm.CustomDist` inside a Bambi custom family (identity link on `logit_p`); Bambi still
  supplies design matrices, priors, centring, non-centred group coding and the predict
  machinery. `stable_softplus` is a subclass of PyTensor's `Softplus` (same C code and
  gradient) with a JAX lowering to `jax.nn.softplus`. Why not the obvious alternatives,
  all reproduced on this stack: `pm.Bernoulli(logit_p=...)` has NaN PyTensor gradients at
  eta >= 40 and NaN JAX gradients at |eta| >= 1000; `pt.softplus` is right on the PyTensor
  path but PyTensor's JAX lowering is a nested `jnp.where` whose untaken branch overflows,
  giving NaN gradients at |eta| >= 1000; any hand-written stable `log1p(exp(.))` form is
  rewritten back into `Softplus` by PyTensor before lowering. Table in the decision record.
- Tests after the repair (fast suite **54 passed, 3 deselected**; slow suite **3 passed**):
  the old A5 assertion deleted; new `test_extreme_logits_density_and_gradient_both_paths`
  (values and full gradients vs the stable reference and analytic gradient at eta in
  {0, +-0.5, +-12, +-40, +-300, +-1000, +-1e6}, two slope/SD settings, both outcomes
  present, PyTensor `compile_logp`/`compile_dlogp` and `get_jaxified_logp`/`jax.grad`) and
  `test_reviewer_case_two_zeros_two_ones` (likelihood term -80 / -600 / -2000, gradient
  2 - 4 expit(eta)). A1-A4, A6-A8 unchanged and passing; Gaussian suite untouched;
  `describe()` now states the sampled likelihood form (`logit_p` deterministic; `p` is
  computed by `predict`).
- B1 re-run after the repair (`docs/m1-evidence/B1_recovery.json`, same prespecified
  thresholds, n = 600, J = 30, 4 x (1000 + 2000), seed 7): pymc 0 divergences, max R-hat
  1.0014, min bulk ESS 2467, min tail ESS 3693, E-BFMI 0.715; numpyro 0 divergences, R-hat
  1.0016, bulk ESS 3107, tail ESS 4223, E-BFMI 0.729; truth inside every 94 % HDI (x:
  mean 0.635, HDI [0.45, 0.82], truth 0.8 near the upper edge in this one data set);
  backends agree within 4 combined MCSE (largest ratio 0.08). Posterior means moved by less
  than 0.005 relative to the pre-repair run, as expected for data whose logits stay
  moderate; the repair matters where the old likelihood was wrong, not here. Examples
  re-run on both backends (`docs/m1-evidence/example_*_cpu.txt`, records, `agree: True`);
  pre-repair outputs kept under `docs/m1-evidence/before-repair/` for the history.
- **GPU execution for this family: pending** (same code path; the JAX lowering of the new
  op is verified on the CPU JAX path only, not on a GPU). No rented node or cluster job.
- **Open review items, not addressed (for the coordinator):** (1) `new_groups=
  "resample_observed"` inherits Bambi's behaviour of drawing one observed level per
  posterior draw for all unseen labels together, so distinct unseen labels share an effect
  within a draw; `"population"` is unaffected. (2) A complex-valued response such as
  `[0+4j, 1+4j]` passes the 0/1 validation after the imaginary part is discarded by the
  float cast; a dtype check before the cast would refuse it.
- Not done: no PDF write-up for M1; no binomial trials, other links, varying slopes or
  generic family layer; no merge, worktree cleanup, paid compute or Firecrawl.

## M1 follow-up unit (2026-09-09): unseen-label identity, complex responses

- Dispatch: `iqss-research/memory/dispatches/2026-09-09-autobayes-followup.txt`. Scope: the two
  open review items; the stable likelihood at `2aea478` untouched. Semantics and acceptance
  criteria U1-U7 were written into `docs/m1-decision.md` and committed (`0cad179`) before
  editing.
- **Finding 1 (reproduced before the change, both families, real 2 x 300-draw fits):** with
  `newdata` labels `[lvl00, NEW_A, NEW_B, NEW_A]`, `new_groups="resample_observed"` gave
  NEW_A and NEW_B identical effects in 100 % of draws (Bambi's `sample_new_groups=True` maps
  every unseen label to one placeholder group). **Change:** `autobayes/predict.py::
  _predict_by_hand` now handles both `"population"` and `"resample_observed"` when unseen
  labels exist: one observed level is drawn uniformly per distinct unseen label per posterior
  draw, independently across labels and draws; rows sharing a label share the choice. After
  the change with J = 6: NEW_A/NEW_B identical in 17.5 % of draws (1/J = 16.7 %), repeated
  NEW_A rows identical in 100 %. In-sample and observed-only `newdata` still go through Bambi.
- **Finding 2 (reproduced):** complex-valued responses (`y + 4j`) built for both families with
  the imaginary part discarded by the float cast under a NumPy warning (pandas counts complex
  as numeric). **Change:** `autobayes/formula.py::reject_complex` refuses complex response,
  predictor and `newdata` predictor columns before any cast, both families.
- Tests added (fast suite now **59 passed, 3 deselected**; slow suite **3 passed**):
  `test_predict.py::test_resample_observed_keeps_distinct_unseen_labels` (synthetic posterior
  with distinct constant effects: each label's effect is one of the draw's observed effects,
  repeated label shares exactly, observed labels untouched, coincidence fraction near 1/J and
  never 1, seed reproducibility, Gaussian `y` residual SD) and `..._on_a_real_fit_is_not_collapsed`;
  `test_bernoulli.py::test_resample_observed_keeps_distinct_unseen_labels_bernoulli` (same on `p`
  and integer 0/1 draws) and `test_complex_inputs_are_refused_before_casting` (response,
  predictor, `newdata`; `ComplexWarning` turned into an error so a cast cannot be the
  mechanism); `test_formula.py::test_complex_columns_are_refused_gaussian` (float32/int still
  accepted). The old membership-only assertions remain and still pass.
- Limitations: levels are drawn uniformly, as in brms, not weighted by group size; no change
  to `"population"` or in-sample predictions; GPU execution of the M1 likelihood still
  pending; no PDF write-up.
- Not done, by instruction: no merge, cleanup, new PR, paid compute, Firecrawl.

### Next-milestone proposal (not implemented)

**M2: binomial trials for grouped binary data, `successes | trials(n) ~ x + (1 | group)`.**
The smallest step past M1 that changes what researchers can fit: aggregated binary data
(k successes of n per row) is the common form of the M1 use case, and it reuses the logit
link, the logit-scale priors, the stable softplus likelihood (log-binomial coefficient plus
`k*eta - n*softplus(eta)`), centring and the non-centred group coding unchanged.
Acceptance criteria to set before implementation: (a) the log density and full gradient
equal an independent SciPy reference (`log_expit`, `gammaln`) at moderate and extreme
logits on both PyTensor and JAX paths, with `n = 1` reproducing M1 exactly; (b) `trials`
must be a positive integer column with `0 <= successes <= trials` and both non-degenerate
rows present, refused otherwise before any cast; (c) predictions distinguish `p` from
`successes` draws (`Binomial(n, p)`), with the same new-group semantics and identity rules
as M1; (d) simulated recovery on both CPU backends with the M1 thresholds and 4-MCSE
agreement, stated as one recovery, not calibration; (e) formula syntax limited to
`successes | trials(n)` on the left-hand side, everything else refused by name. Out of
scope: other links, overdispersion (beta-binomial), varying slopes, GPU execution beyond
a labelled pending status.

## Repository (M1)

- PR: https://github.com/andrew-dou/AutoBayes/pull/3 (branch `m1/bernoulli-varying-intercept` -> `main`, author andrew-dou). Repair (`2aea478`) and the follow-up unit pushed to the same PR; open for independent review; not merged by this session.

## Milestone

M0: Gaussian varying-intercept model `y ~ x + (1 | group)`, explicit priors, inspectable
PyMC model, posterior and predictive outputs, CPU path plus explicit JAX/NumPyro device
path. Definition: `IQSS-Research/AUTOBAYES_FIRST_MILESTONE.md`. Design decision and
acceptance checks: `docs/m0-decision.md`.

## Session checkpoint (2026-09-09, relocated hub)

- Working clone moved out of Dropbox to `/Users/andrewdou/iqss-research/projects/AutoBayes`.
  Claude session `e5628a7d-e572-4bad-9feb-82ee8361c261` (Fable 5.1) owns this branch; the old
  M0 session `local_79bbf46f-...` is historical.
- `.venv/` repaired in place: activate scripts and console-script wrappers pointed at the
  Dropbox path (`sed` replaced them); `bin/python` is a symlink to the framework Python 3.11
  and needed nothing. `.venv/bin/python -c "import autobayes"` resolves to this checkout.
- Tests on the final code: fast suite 39 passed, 2 deselected; slow suite 2 passed (B1, B2
  regenerated; only the report's new `seed` line changed in `B2_recovery.json`). Use
  `PYTENSOR_FLAGS=base_compiledir=<writable dir>` if `~/.pytensor` is not writable.
- GitHub identity verified: `gh api user` -> andrew-dou, admin on andrew-dou/AutoBayes;
  git user andrew-dou <andrew.dou0805@gmail.com>. No branch protection on `main`.

### Done this session

1. **Skill rewrite** (`SKILL.md`): researcher workflow for the verified family in six checked
   steps (data, formula and priors with the centring trap, fit and report, panel with the
   family-specific responses, prediction semantics, fit record and agreement), a failure
   recovery table of the errors actually seen on this stack, a session-handoff checklist, GPU
   and remote execution, routing outside the family, and the long-program scaffold condensed.
   Every Python snippet in it was executed in order on the example data (scratch script; the
   4x1000 run reproduced the documented `investigate` on `Intercept`, R-hat 1.02, ESS 350).
2. **Library**: `autobayes/session.py` (`environment`, `fit_record`, `compare_records`, and
   their formatters), `seed` on `FitResult` and in the report, `import autobayes` now enables
   float64 and sets the JAX host device count (found while preparing the GPU run: the
   example's environment print initialised JAX first and silently left CPU chains sequential;
   `fit()` now reports `chain method : sequential` with a note if that still happens). Example
   gained `--record` and `--compare`. Tests: `tests/test_session.py` (4).
3. **GPU smoke test executed** on GiveMeANode (node `autobayes-m0-smoke`, h100-1, image
   cuda-12.9, mission `autobayes-m0-gpu-smoke`, https://givemeanode.com/missions/autobayes-m0-gpu-smoke):
   H100 80GB, float64, 4 vectorized chains, 4x(1000+2500), seed 7; panel all pass (0
   divergences, R-hat 1.006, bulk ESS 798, tail ESS 1404, E-BFMI 0.674); posterior means agree
   with the CPU NumPyro record within 4 combined MCSE on every parameter; A4 re-run on the GPU
   to 9e-13. Wall 39.2 s incl. compile (no speed claim). Cost: 10.83 min, $0.72 of the $1118.45
   credit balance; node stopped and deleted. Evidence: `docs/m0-evidence/example_numpyro_gpu.txt`,
   `gpu_record.json`, `B2_numpyro_cpu_record.json`; write-up `docs/m0-gpu-validation.md`;
   PDF `docs/m0-evidence/m0_evidence.pdf` gained a GPU paragraph (6 pages).
   Firecrawl: authenticated, 344/1000 credits this cycle; not used (nothing to acquire for M0).
4. **PR / merge / cleanup**: see "Repository" below (filled in as it happens).

### Repository

- PR: https://github.com/andrew-dou/AutoBayes/pull/2 (branch `m0/gaussian-varying-intercept` -> `main`, author andrew-dou).
- Merged 2026-09-09: merge commit `0f2b4e9` on `main` (merge commit, not squash, to keep the
  branch history). No required checks are configured on the repository; the PR was
  MERGEABLE/CLEAN, and no agent attribution exists in any commit message or author field.
- Cleanup: local branch `m0/gaussian-varying-intercept` deleted, remote branch deleted and
  pruned; this checkout is on `main` at `0f2b4e9`, clean. No worktrees other than the main
  checkout existed. `origin/autobayes-restructure` (PR #1, merged at `f87f639`) predates
  this task and was left alone.

## Checkout
## Checkout

- Clone: `IQSS-Research/projects/AutoBayes` (this file's repository).
- Branch: work was on `m0/gaussian-varying-intercept` (from `main` at `f87f639`); now merged, checkout on `main`.
- Environment: `.venv/` (gitignored), created with `uv venv --python 3.11`. Packages:
  pymc 5.28.5, bambi 0.17.2, formulae 0.6.2, numpyro 0.21.0, jax 0.10.2 (CPU only),
  arviz 0.23.4, pytensor 2.38.3, matplotlib pinned `<3.11` (arviz 0.23 preview import
  breaks on 3.11.1).
- Hardware: Apple M1 Pro, no CUDA device. JAX sees one CpuDevice.

## Completed

- Read direction, milestone, README, SKILL.md, reference pack.
- Bambi assessment (see `docs/m0-decision.md`): supports the model, explicit priors,
  non-centred group effects, predictor centring, accessible PyMC model, NumPyro path via
  `pm.sample(nuts_sampler="numpyro", nuts_sampler_kwargs=...)`, predictions with
  brms-"uncertainty" new-group semantics. Decision: build a thin layer on Bambi.
- Decision record and acceptance checks A1-A8, B1-B2, C1-C3 written.

- Package `autobayes/` (formula, priors, model, inference, predict, diagnostics),
  `tests/`, `examples/m0_varying_intercept.py`, `pyproject.toml` written.
- Passed (`.venv/bin/python -m pytest tests -q`, 34 tests): A1-A7 in
  `tests/test_logp.py` and `tests/test_formula.py`; A8 in `tests/test_predict.py`;
  C1-C2 in `tests/test_device.py`.

- B1 passed (`tests/test_sampling.py::test_conditional_gaussian_posterior`): with
  sigma and the group SD fixed by `pm.do`, 4x2000 NUTS draws vs the exact conditional
  Gaussian posterior: max |z| = 2.80 over 8 parameters (threshold 4), posterior SDs
  within 0.0-3.8% of exact, min ESS 1272. Numbers in
  `docs/m0-evidence/B1_conditional_gaussian.json`.
- B2 passed (`test_simulated_recovery_both_backends`, n=200, J=20, 4 chains, 1000 tune,
  2500 draws, both backends on CPU): 0 divergences, max R-hat 1.0032 (pymc) / 1.0075
  (numpyro), min bulk ESS 755 / 771, min tail ESS 1384 / 1235, E-BFMI 0.70 / 0.67, no
  tree-depth saturation; truth inside every 94% HDI; backend posterior means differ by
  at most 0.27 of the 4-MCSE tolerance. `docs/m0-evidence/B2_recovery.json`.
  Note: at 4x1000 draws the same fits had R-hat 1.02 and bulk ESS 350 on the intercept
  (identically on both backends): the intercept trades off against the mean of the 20
  offsets. Runs were lengthened; thresholds were not loosened.
- Example runs captured: `docs/m0-evidence/example_pymc_cpu.txt`,
  `example_numpyro_cpu.txt` (all panel rows pass), `example_numpyro_gpu_attempt.txt`
  (the clear refusal when no GPU is visible).
- GPU: pending. `docs/m0-gpu-validation.md` and `cluster/m0_gpu_smoke.sbatch` carry
  the reproducible command. Not submitted (outside the handoff's compute boundary).

- B1 seed check (`docs/m0-evidence/B1_seed_check.json`): seeds 11-14, 2000-8000
  draws; all |z| < 1.8 with alternating signs along the weakly identified
  intercept-vs-mean-offset direction, so the committed run's 2.5-sigma pattern was noise.
- Evidence write-up for review: `docs/m0-evidence/m0_evidence.pdf` (source
  `m0_evidence.tex`, pdflatex, 6 pages).

## In progress

- Nothing. Awaiting review before any scope expansion.

## Failures

- A4 first failed with the sign flipped: PyMC 5.28.5's `get_jaxified_logp(model,
  negative_logp=...)` flag is named backwards (default `True` returns the log density;
  `False` negates it). Verified in source; test now uses the default. No library change.
- A5/A6 tests were wrong on first run (they had never executed: the first run stopped at
  A4). A6: a deterministic must have its RVs replaced by value variables before
  compiling (`model.replace_rvs_by_values`), and inputs the output does not use need
  `on_unused_input="ignore"`. A5: `compile_d2logp` returns the *negative* Hessian
  (`pymc.pytensorf.hessian(negate_output=True)`), i.e. the precision matrix; the test
  now checks that against the closed-form precision at two points. Model unchanged.
- B2 at 4x1000 draws: R-hat 1.02 / bulk ESS 350 on the intercept on both backends.
  Lengthened to 2500 draws per chain; passes. Thresholds unchanged.
- B2 agreement check first used `az.summary`'s rounded MCSE (0.000 -> zero tolerance);
  now `round_to="none"`.

## Remaining checks

- None for M0. C3 (GPU execution) done on one rented H100, matched to the CPU record.
  A FAS RC run (`cluster/m0_gpu_smoke.sbatch`) has not been made and is not required by
  the milestone.

## Blockers

- None.

## Next action

- Andrew reviews `docs/m0-evidence/m0_evidence.pdf`, `SKILL.md`, and the merged PR. Then
  choose M1 from an actual researcher need (candidates in `memory/projects/autobayes.md`
  at the hub: one formula feature or one family at a time). No benchmark campaign, no
  package publishing, no FAS RC job without a decision entry.

To resume in a fresh session: `git checkout main && git pull`, then
`.venv/bin/python -m pytest tests -q` (fast) and `-m slow` (two sampling checks, ~30 s).

M0_COMPLETE: yes. Support in code verified on CPU (both backends) and executed once on a
GPU (H100, matched to the CPU posterior). One recovery run is not a calibration claim and
no speedup is claimed.

- **Direct-CDF generic implementation (2026-09-13, code and analytic/stub checks only):** the proposed v3 reference now has an injected-evaluator, batched integration/assembly path in `tests/reference_direct_cdf.py`. Exact independent integrals, nonseparable CDFs, stage/cost checks, partial-stop behavior and final assembly timing are covered. **62 tests passed in 0.84 s** (23 new +39 guard/assembly), exit 0; target imports explicitly prohibited, **zero F7 evaluations**. Evidence: `docs/m3b1-evidence/direct_cdf_v3_checks.json`; protocol: `direct_cdf_v3_protocol.json`. The uniform-node precursor failed an analytic toy resolution check; v3 uses [48,96,48]/[64,128,64], planned 12,038,156 points under a proposed 12.05M/720-second cap. No F7 run is authorized by this implementation, no target-binding CLI was enabled, and the old evidence/ledger stay intact. F7 FAILED, reference inconclusive, PR8 OPEN; next review code and hosted checks, then separately scope any actual target binding/run. Do not repeat the completed guard repair or generic implementation.
- **Anchor reference-consistency repair, v3r1 (2026-09-13, code and stub checks only):** the worker review of
  581123e found (F1) that face deficits used the saved reference without ever evaluating the target at the
  unprojected anchor, so a constant-shifted target passed every gate; reproduced by the coordinator with stubs.
  Repaired in `tests/reference_direct_cdf.py`: one counted `anchor:reference` stage first, fail-closed condition
  `abs(ell(anchor) - fixed_ell_reference) <= 1e-6`, exact one-point count, plan identity, missing/non-finite/
  mismatched data rejected; plans without the tolerance refused (v3 is history). Revised configuration
  `direct_cdf_v3r1_protocol.json` (distinct id, v3 SHA recorded, same grids/points/faces), planned **12,038,157**
  under the unchanged proposed 12,050,000/720 s cap: unexecuted proposal, not an authorisation. Coordinator's
  import-blocking launcher: **72 passed**, exit 0, zero F7 evaluations (`direct_cdf_v3r1_checks.json`). F7 FAILED,
  reference inconclusive, ledger untouched, PR8 OPEN. Next: independent review, then a separately scoped
  target-binding plan; nothing executed now. Commit `4d6ee8d`.
- **F7 binding implementation (2026-09-13, code and stub checks only):** the independently reviewed binding plan
  (`artifacts/2026-09-13-autobayes-f7-binding-plan/`, review with five corrections) is implemented: `run_protocol`
  stops after the single counted anchor preflight when it fails (assembly before release kept, checkpoint survives,
  remaining conditions unchecked, exit 3, no second probe); the import-safe opt-in entrypoint
  `docs/m3b1-evidence/scripts/f7_direct_cdf_binding.py` (`--dry-run | --execute`, refusals on protocol/source/model/
  environment/git identity, atomic exclusive new unit, no override or retry option, no approval shipped); wrapper
  logging outside the unit; `run_logged.py` creates its log parent. Blocked-import checks **109 passed**, exit 0,
  zero target access (`direct_cdf_binding_checks.json`); CLI dry-run on clean head 04f9e08: 45 stages,
  12,038,157 planned under 12,050,000 / 720 s, all identity checks true, nothing created
  (`direct_cdf_binding_dryrun_v3r1.json`); execute without approval refused, exit 2 (`direct_cdf_binding_refusal_execute.json`);
  revised execution identity `direct_cdf_v3r1_execution_identity.json`. Nothing executed; F7 FAILED; exhausted
  ledger untouched; PR8 OPEN. Next: independent review of the binding; a real run needs a separately written approval.
