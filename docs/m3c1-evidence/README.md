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
- No sampling run was executed in this slice; the proposed smoke is in `docs/m3c1-decision.md`. The example
  `examples/m3c1_population_only.py` is unexecuted.
