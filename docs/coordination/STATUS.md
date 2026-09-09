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

## In progress

- B1, B2 (`pytest -m slow`) and the example script; evidence to `docs/m0-evidence/`.

## Failures

- A4 first failed with the sign flipped: PyMC 5.28.5's `get_jaxified_logp(model,
  negative_logp=...)` flag is named backwards (default `True` returns the log density;
  `False` negates it). Verified in source; test now uses the default. No library change.

## Remaining checks

- B1, B2, C3 (GPU pending), README/SKILL pointer, evidence write-up.

## Blockers

- No accelerator on this machine: GPU validation (C3) will stay pending with a
  reproducible command.

## Next action

- Implement `autobayes/formula.py`, `priors.py`, `model.py`, `inference.py`,
  `predict.py`, `diagnostics.py`; then `tests/reference_logp.py` and the A-series tests.

M0_COMPLETE: no
