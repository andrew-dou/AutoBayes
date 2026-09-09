# AutoBayes status

## Milestone

M0: Gaussian varying-intercept model `y ~ x + (1 | group)`, explicit priors, inspectable
PyMC model, posterior and predictive outputs, CPU path plus explicit JAX/NumPyro device
path. Definition: `IQSS-Research/AUTOBAYES_FIRST_MILESTONE.md`. Design decision and
acceptance checks: `docs/m0-decision.md`.

## Checkout

- Clone: `IQSS-Research/projects/AutoBayes` (this file's repository).
- Branch: `m0/gaussian-varying-intercept`, created from `main` at `f87f639`.
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

## In progress

- Evidence write-up (LaTeX -> PDF) for review.

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

- B1, B2, C3 (GPU pending), README/SKILL pointer, evidence write-up.

## Blockers

- No accelerator on this machine: GPU validation (C3) will stay pending with a
  reproducible command.

## Next action

- Implement `autobayes/formula.py`, `priors.py`, `model.py`, `inference.py`,
  `predict.py`, `diagnostics.py`; then `tests/reference_logp.py` and the A-series tests.

M0_COMPLETE: no
