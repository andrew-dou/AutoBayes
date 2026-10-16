# The brms-parity loop: operating rules

AutoBayes is brms re-implemented in Python on Bambi/PyMC with JAX/NumPyro sampling. Every feature is accepted
by deterministic comparison against brms itself (the oracle in `oracle/` and `tests/oracle/`), never by MCMC.
This file is the contract every agent in the loop works under. State lives in `ledger.json` (one entry per
feature), `divergences.md` (where AutoBayes knowingly differs from brms) and the gate records the tools write.

## Roles

- **Fixture agent.** Writes `tests/oracle/cases/<id>.json` (brms side, data, AutoBayes call), runs
  `python oracle/regen.py new <id>` locally (R, about 35 s per case), commits the fixture, and confirms the
  new oracle tests fail on `main` for the right reason. Fixtures are frozen on commit.
- **Implementation agent.** Works in its own worktree on branch `parity/<id>` off `origin/main`, makes the
  oracle tests pass, keeps every existing test passing, documents the feature, opens the PR as `andrew-dou`.
- **Verifier agent.** Starts from the diff, not from the implementer's summary. Tries to break the feature:
  edits to fixtures, tolerances, gates or adapter mappings count as failure unless the ledger entry says why;
  a gate that passes for the wrong reason (a skipped case, a weakened match) is a failure.
- **Merge queue (one at a time).** Rebase on `main`, run `python tools/parity_gate.py --feature <id>`, wait for
  hosted CI, `gh pr merge --squash` as `andrew-dou`, update the ledger to `merged` with the gate SHA.

## Gates (all deterministic; `tools/parity_gate.py` runs them)

| gate | what passes | where |
|------|-------------|-------|
| G0 | refusals the feature should flip are flipped, every other refusal unchanged | existing suite |
| G1 | design matrices, response, trials and group coding equal `standata()` to 1e-12 | `tests/oracle` |
| G2 | log density and gradient equal `rstan::log_prob` (1e-8 / 1e-7 relative), with and without Jacobian | `tests/oracle` |
| G3 | derived quantities per draw (linear predictor, group effects, raw-scale intercept) | `tests/oracle` |
| G4 | `predict()` equals `posterior_linpred` / `posterior_epred` to 1e-9 relative | `tests/oracle` |
| G5 | the JAX-compiled density equals the PyTensor one | `tests/oracle` |
| G6 | `ab.default_priors()` written as brms's table equals the fixture's `default_prior()` table, every row and column (A07) | `tests/oracle` |
| G7 | the non-sampling suite locally (`-m "not slow and not sampling"`); the full fast suite in hosted CI | `tests` |
| G8 | advisory only: one 120-second sampling smoke, zero retries | `--smoke` |

Immutable under a feature branch: `tests/oracle/fixtures/**` and `tests/oracle/references/**` (additions only),
`tests/oracle/tolerances.py`. Changes to `tests/oracle/gates.py`, `tests/oracle/adapter.py` or
`tests/oracle/two_reference.py` are allowed only to extend them to new parameter kinds and are called out in the PR;
the gate tool flags them.

### Two references (P6; Andrew's decision of 2026-10-11, a change of the gate contract)

Some features are exact in AutoBayes while brms's own arithmetic is off the exact value by more than the tolerance at
some frozen points (the cloglog link, D18; `cens()`, D24; `trunc()`, D25). Such a feature may be accepted against two
references: brms wherever brms is accurate, and a committed high-precision reference where it is not. Nothing looser.

- **Opt-in.** The case's editable `gate` field names the reference, the divergence row that says why brms is inexact,
  and the quantity kinds that may fall back: `{"expect": "pass", "reference": "gauss_censored.json", "divergence":
  "D24", "fallback": ["gradient"]}`. The kinds are `"gradient"` (every G2 gradient coordinate, with and without the
  Jacobian) and `"epred"` (G4's expected values). Log densities (`lp`, `lp_nojac`, `lprior`) never fall back: a gate
  field that lists one is refused.
- **The rule**, for each frozen point and each entry of a listed kind, with the usual tolerance (`tolerances.py`,
  unchanged): (1) the entry passes if AutoBayes agrees with brms within it; (2) otherwise it passes only if brms
  differs from the reference by more than the tolerance **and** AutoBayes agrees with the reference within it; (3)
  wherever AutoBayes and brms agree within the tolerance, the reference must agree with brms within it too, or the
  reference is wrong and the gate fails, whatever AutoBayes does.
- **Everything else is unchanged** and compares with brms only: G1, G3, G5, G6, the log densities of G2, G4's linear
  predictor, and G4's expected values unless `"epred"` is listed. A case without the opt-in runs exactly as before.
- **Report.** The gate returns, and emits as a `ReferenceFallback` warning that pytest's summary shows, how many entries
  passed on the reference and where (`tests/oracle/two_reference.py`; `gates.check_gradient` and
  `gates.check_predictions` hand an opted-in case to it).
- **The reference** is `tests/oracle/references/<case>.json`, written by `python oracle/reference.py <case>`, an
  independent third implementation (the standard library only, nothing from `autobayes`) that reads the fixture's
  `case_brms.json`, `data.csv`, `newdata.csv` and the points' unconstrained values, applies brms's transforms and their
  Jacobians itself, and evaluates the model in Python's `decimal` at 50 significant digits: the log density with and
  without the Jacobian, every gradient coordinate in brms's order (a fourth-order central difference of the 50-digit
  density with step 1e-10, error below 1e-35), and the expected values G4 compares where `"epred"` may fall back, each
  written with 30 significant digits, with the sha256 of every file read. Re-running the script reproduces the file
  byte for byte (`python oracle/reference.py --check --all`, also a harness test), the harness checks the hashes and
  that the reference's log densities equal brms's (exact to `LOGP_REL` on these cases), and a reference is frozen with
  the fixture it certifies: the gate tool refuses a changed one, as it refuses a changed fixture.
- **In use** (P6): `bernoulli_cloglog` (gradient; 2 of 144 entries on the reference), `gauss_censored` (gradient; 32 of
  160), `gauss_truncated` (gradient, 2 of 160; expected values, 6 of 656) and `cumulative_probit_varint` (gradient; 173 of
  176, D88). `binomial_cloglog`, frozen with P6, needs
  no opt-in: brms is within the tolerance at every entry there.

## Park and stop rules

- A feature parks after three failed gate cycles (`parked:<reason>` in the ledger); the lane takes the next
  unblocked feature. Two consecutive parks in a lane stop the lane.
- Where AutoBayes's documented intent differs from brms (explicit priors only, sorted levels, zero-trial rows
  dropped), the fixture, tolerance and default are never edited; the difference goes into `divergences.md`,
  and the brms behaviour is offered as an opt-in or parked for Andrew.
- The campaign stops for Andrew at: a failed merge-queue gate on `main`, 80 % of a wave's budget, the usage
  limit, any dependency install or upgrade, an API-shape choice the oracle cannot decide, anything needing a
  GPU, the cluster or sampling beyond G8, and every tier boundary.

## Conventions

- Commits and PRs are authored by `andrew-dou <andrew.dou0805@gmail.com>` and carry no AI attribution.
- Nothing in a gate samples. Fixture generation samples nothing either (fixed-parameter runs on seeded points).
- brms's Stan code is GPL-3: hash it, never store it.
- Documentation of a merged feature: a short `docs/<feature>.md` and a line in `docs/current-capabilities.md`.

## Commands

```bash
python oracle/regen.py new <case-id>
```

```bash
python tools/parity_gate.py --feature <id> --base origin/main
```

```bash
python tools/parity_status.py
```
