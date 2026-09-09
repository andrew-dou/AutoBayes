# AutoBayes status

## M1 checkpoint (2026-09-09): Bernoulli varying intercept, logit link

- Branch `m1/bernoulli-varying-intercept` from `main` at `925554b`; session
  `e5628a7d-e572-4bad-9feb-82ee8361c261`. Decision record written and committed before the
  implementation: `docs/m1-decision.md` (target density, Bambi/brms semantics checked against
  the installed 0.17.2 source and the brms/Bambi reference pages, acceptance checks A1-A8,
  B1, C1-C3, kill criterion).
- Scope delivered: `ab.bernoulli("y ~ x + (1 | group)", df, priors)` for a strictly 0/1
  numeric response, logit link, logit-scale priors on intercept and slopes, group SD prior,
  no `sigma` (giving one is an error; omitting it for Gaussian is an error). Same formula
  subset, centring and non-centred effects as M0. `predict(kind="mean")` returns `p`
  (probabilities); `kind="response"` adds 0/1 draws `y`; new-group semantics as in M0 with
  the population draw on the logit scale. Fit records carry family and link;
  `compare_records` refuses different families. Gaussian behaviour unchanged.
- Changed files: `autobayes/{priors,formula,model,predict,session,__init__}.py`,
  `tests/{reference_logp,test_bernoulli,test_sampling,test_formula}.py`,
  `examples/m1_bernoulli_varying_intercept.py`, `SKILL.md`, `README.md`,
  `docs/m1-decision.md`, `docs/m1-evidence/`, `pyproject.toml` (version 0.1.0).
- Tests executed (this machine, Apple M1 Pro, CPU): fast suite **53 passed, 2 deselected**
  (M0's 40 plus 13 for M1); slow suite **3 passed** (M0 B1, B2 and the M1 recovery).
  M1 checks: A1/A2 log density vs an independent NumPy/SciPy reference (`log_expit`
  likelihood, centred and uncentred, one and two predictors, intercept-only) to 1e-8;
  A3 gradient vs central differences; A4 JAX-compiled density vs PyTensor; A5 extreme
  logits; A6 design and `p` from the graph; A7 invalid inputs and unsupported constructs;
  A8 prediction shape, range, meaning and new-group semantics; C1 GPU refusal.
- B1 sampling (`docs/m1-evidence/B1_recovery.json`; n = 600, J = 30, truth a = -0.5,
  b = 0.8, sd = 0.7, 4 x (1000 + 2000), seed 7): both backends 0 divergences, max R-hat
  1.0014 (pymc) / 1.0025 (numpyro), min bulk ESS 2672 / 3344, min tail ESS 4041 / 4688,
  E-BFMI 0.71 / 0.69; truth inside every 94 % HDI (x: posterior mean 0.636, HDI
  [0.45, 0.82], truth 0.8 near the upper edge in this one simulated data set); backend
  means within 4 combined MCSE (largest ratio 0.51, on x). One recovery on one simulated
  data set; not a calibration claim.
- Example outputs: `docs/m1-evidence/example_pymc_cpu.txt`, `example_numpyro_cpu.txt`
  (records `*_record.json`, agreement block `agree: True`), `example_numpyro_gpu_attempt.txt`
  (the refusal on this machine).
- **Limitation, measured and documented** (`docs/m1-decision.md`, `describe()`): Bambi
  computes `p = sigmoid(eta)` and substitutes `1 - eps` when it rounds to 1, so the sampled
  density equals the exact Bernoulli-logit density (to 1e-8) for eta below about 15 and for
  all negative eta, loses precision in the y = 0 term above that, and is floored at
  log(eps) = -36.04 per zero outcome above eta of about 37 (test A5 pins this). Bambi's
  likelihood registry does not accept `logit_p`. Negligible under weakly informative
  logit-scale priors; separation in real data is the case to watch, and the recorded
  fallback is a hand-built `pm.Bernoulli(logit_p=...)` on Bambi's design matrices.
- **GPU execution for this family: pending.** Same code path as M0; not run on a GPU in
  this task (no rented node, no cluster job). The M0 H100 evidence does not transfer.
- Not done: no PDF write-up for M1 (the decision record and STATUS carry the evidence);
  no binomial trials, other links, varying slopes or generic family layer.
- PR: opened for the coordinator's independent review, not merged (recorded below).

## Repository (M1)

- PR: pending.

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
