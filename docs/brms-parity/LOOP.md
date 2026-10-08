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

Immutable under a feature branch: `tests/oracle/fixtures/**` (additions only), `tests/oracle/tolerances.py`.
Changes to `tests/oracle/gates.py` or `tests/oracle/adapter.py` are allowed only to extend them to new
parameter kinds and are called out in the PR; the gate tool flags them.

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
