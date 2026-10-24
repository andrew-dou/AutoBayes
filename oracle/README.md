# The brms oracle

brms (R, with rstan) is the reference implementation that AutoBayes is measured against. For any formula,
family and prior that brms accepts, `oracle/R/gen_fixture.R` records what brms computes, deterministically
and without sampling, and the tests in `tests/oracle/` check that the AutoBayes model is the same target.

## Layout

- `tests/oracle/cases/<id>.json` — a case: `brms` (formula, family, factor levels, priors), `data`
  (generator and seeds, see `tests/oracle/datasets.py`), `seed` (for the unconstrained points) and
  `autobayes` (the call expected to build the same model). The `brms` and `data` sections are frozen with
  the fixture; the `autobayes` section may change as the library's API evolves.
- `tests/oracle/fixtures/<id>/` — the frozen brms side: `case_brms.json`, `data.csv`, `newdata.csv`,
  `data2_<name>.csv` (Tier F: the matrices brms receives as `data2`, for `car()`, `sar()` and `fcor()`),
  `fixture.json` (standata, Stan-code hash, parameter declarations, the probed unconstrained-parameter map,
  eight seeded points with `log_prob` and gradient with and without the Jacobian, `posterior_linpred` /
  `posterior_epred` on those points, prior tables). Immutable once committed; a changed case gets a new id.
- `tests/oracle/adapter.py` — maps brms onto the AutoBayes model by name and array content, never position.
- `tests/oracle/gates.py` — gates G1–G6; `tests/oracle/tolerances.py` — the tolerances (a contract).
- `tests/oracle/test_baseline.py` — every frozen case passes every gate; `tests/oracle/test_harness.py` —
  the harness is calibrated against the hand-written reference densities and fails on corrupted fixtures.

## Commands (local; CI needs no R)

```bash
python oracle/regen.py doctor
```

```bash
python oracle/regen.py new <case-id>
```

```bash
python oracle/regen.py check --all
```

Generation compiles one Stan program per case (about 35 s each on this machine) and runs no MCMC: the
"draws" behind the prediction references are the seeded points themselves, via a fixed-parameter run.
brms's Stan code is GPL-3 and is hashed, not stored.
