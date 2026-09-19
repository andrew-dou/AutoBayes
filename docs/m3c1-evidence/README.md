# M3c-1 evidence (2026-09-19)

Clean locked environment recreated in `projects/AutoBayes-worktrees/m3c1-population-only` (`uv sync --locked --extra jax --extra dev
--python 3.11`; `environment.txt` is `ab.format_environment(ab.environment())` from that venv). Working tree at the commit that
follows the decision record `55437e6` (the implementation commit records the exact head). Fresh evidence directories under
`evidence-runs/` (gitignored), `PYTENSOR_FLAGS=base_compiledir=<scratch>`.

- `test_population_only.txt`: the M3c-1 module alone, **48 passed in 24 s**, exit 0.
- `fast_suite.txt`: the fast suite in the CI command form (`uv run --no-sync pytest tests -q -m "not slow" -p no:cacheprovider`),
  **395 passed, 7 deselected in 320 s**, exit 0.
- `fast_suite_before_assertion_updates.txt`: the first full run on the implementation, **3 failed, 393 passed**, exit 1: the three
  failures were the stale assertions that `y ~ x` (and `k | trials(n) ~ x`) must be refused with "exactly one
  varying-intercept" (`tests/test_formula.py`, `tests/test_bernoulli.py`, `tests/test_binomial.py`); they assert the
  documented contract change now, and the acceptance run above followed. No library code changed between the two runs.
- Correction (after the independent review): no **population-only** sampling run was executed in this slice and the new
  module contains no `fit` call, but the inherited fast suite's own CPU regression sampling (fixtures and tests that
  predate this slice) ran as part of the required suite. `fit_sites_inventory.txt` is the static call-site inventory
  by file, line and marker; it is not an execution count. The example `examples/m3c1_population_only.py` was
  unexecuted at `6d8e234`; its model was then sampled exactly twice by the bounded two-fit validation recorded in
  `two-fit-validation/` (freeze, panels, labelled comparison, inspection of the retained draws, supervisor record).

## Acceptance update (2026-09-19)

`independent-review.json` contains the independent retained-draw review of the two Gaussian CPU fits:
all required diagnostics and seven mean comparisons pass. The decision record explicitly corrects the
finite-statistic policy after the run to exclude only PyMC's documented unused eigenvalue placeholders.
The original `all_gates_pass: false`, nonzero exit, raw outputs and frozen runner scripts remain unchanged.
Those scripts are historical evidence and must not be rerun as successful validation commands; the
`BaseException` exit guard also catches its own `SystemExit`. No additional fit was performed in review.

The implementation's three-family evidence remains deterministic; only the Gaussian population-only
example has been sampled and independently reviewed on both CPU backends. No new GPU or calibration
claim. Hosted fast checks on `525c8d5` passed; the final documentation head must pass required checks.
