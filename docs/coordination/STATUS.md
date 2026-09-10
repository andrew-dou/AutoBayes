# AutoBayes status

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
