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

### Diff coverage

Every line of library code a feature adds or changes runs under the tests before it merges (Andrew's rule).
`python tools/diff_coverage.py --base origin/main`, or `tools/parity_gate.py --diff-coverage`, takes the lines the branch
adds or modifies under `autobayes/` (committed or not), keeps those the compiler gives code to (docstrings, comments and
blank lines do not count), runs the test files the branch adds or changes and the oracle gates of the cases it adds or
ungates in one traced process, and must end in `DIFF COVERAGE n/n (100.0%)`: 100 % of the changed executable lines, no
lower threshold. The only exemption is a line ending in `# pragma: no cover - <reason>`; a pragma without a reason fails
the gate. The tool prints every exemption with its reason, and the merge queue reads each one and sends the branch back
for any line a deterministic test could reach. Code that runs only in a subprocess, or only as compiled JAX or PyTensor
code, is invisible to the tool (a jitted function's Python counts once, when JAX traces it).

### Gates expected to fail on a landed feature

A case whose feature has landed but on which brms's own arithmetic fails a gate that no reference may stand in for (a log
density) keeps `"gate": {"expect": "fail", "gates": [...], "reason": ...}`: only the listed gates (`g1`, `g2_logp`,
`g2_gradient`, `g3`, `g4`, `g5`, `g6`, `post`) run as strict expected failures, each passing only by raising the oracle's
`OracleMismatch`, and every other gate of the case must pass. The feature is accepted on another frozen case where brms is
exact (E06: `cumulative_cloglog`, with the link accepted on `cumulative_cloglog_narrow`).

### Two references (P6; Andrew's decision of 2026-10-11, a change of the gate contract)

Some features are exact in AutoBayes while brms's own arithmetic is off the exact value by more than the tolerance at
some frozen points (the cloglog link, D18; `cens()`, D24; `trunc()`, D25). Such a feature may be accepted against two
references: brms wherever brms is accurate, and a committed high-precision reference where it is not. Nothing looser.

- **Opt-in.** The case's editable `gate` field names the reference, the divergence row that says why brms is inexact,
  and the quantity kinds that may fall back: `{"expect": "pass", "reference": "gauss_censored.json", "divergence":
  "D24", "fallback": ["gradient"]}`. The kinds are `"gradient"` (every G2 gradient coordinate, with and without the
  Jacobian), `"epred"` (G4's expected values) and `"weights"` (the post gate's stacking weights of `loo_model_weights`,
  as numbers and as logarithms; only a case whose post-processing functions include `loo_model_weights` may list it,
  G01_releases). Log densities (`lp`, `lp_nojac`, `lprior`) never fall back: a gate field that lists one is refused.
- **The rule**, for each frozen point and each entry of a listed kind, with the usual tolerance (`tolerances.py`,
  unchanged): (1) the entry passes if AutoBayes agrees with brms within it; (2) otherwise it passes only if brms
  differs from the reference by more than the tolerance **and** AutoBayes agrees with the reference within it; (3)
  wherever AutoBayes and brms agree within the tolerance, the reference must agree with brms within it too, or the
  reference is wrong and the gate fails, whatever AutoBayes does.
- **Everything else is unchanged** and compares with brms only: G1, G3, G5, G6, the log densities of G2, G4's linear
  predictor, G4's expected values unless `"epred"` is listed, and every quantity of the post gate but the stacking weights
  where `"weights"` is listed. A case without the opt-in runs exactly as before.
- **Report.** The gate returns, and emits as a `ReferenceFallback` warning that pytest's summary shows, how many entries
  passed on the reference and where (`tests/oracle/two_reference.py`; `gates.check_gradient`, `gates.check_predictions` and
  `gates.check_loo_model_weights` hand an opted-in case to it).
- **The reference** is `tests/oracle/references/<case>.json`, written by `python oracle/reference.py <case>`, an
  independent third implementation (the standard library only, nothing from `autobayes`) that reads the fixture's
  `case_brms.json`, `data.csv`, `newdata.csv` and the points' unconstrained values, applies brms's transforms and their
  Jacobians itself, and evaluates the model in Python's `decimal` at 50 significant digits: the log density with and
  without the Jacobian, every gradient coordinate in brms's order (a fourth-order central difference of the 50-digit
  density with step 1e-10, error below 1e-35), the expected values G4 compares where `"epred"` may fall back, and the
  stacking weights of a case that records `loo_model_weights` (the Karush-Kuhn-Tucker condition of the two models' stacking
  objective on the fixture's pointwise elpd_loo, bisected at 65 digits), each written with 30 significant digits, with the
  sha256 of every file read. Re-running the script reproduces the file
  byte for byte (`python oracle/reference.py --check --all`, also a harness test), the harness checks the hashes and
  that the reference's log densities equal brms's (exact to `LOGP_REL` on these cases), and a reference is frozen with
  the fixture it certifies: the gate tool refuses a changed one, as it refuses a changed fixture.
- **In use** (P6): `bernoulli_cloglog` (gradient; 2 of 144 entries on the reference), `gauss_censored` (gradient; 32 of
  160), `gauss_truncated` (gradient, 2 of 160; expected values, 6 of 656) and `gamma_censored_interval` (gradient; 10 of 160, D120) and `cumulative_probit_varint` (gradient; 173 of
  176, D88), and `wiener_varint` and `wiener_distributional_bs` (expected values; 656 of 656 each, D148), and since G01_releases
  `negbinomial_truncated` (expected values; 656 of 656, D121: brms's `posterior_epred` of `trunc(lb = 1)` is the mean from
  2) and `gauss_loo_compare_x_stacking` (stacking weights; 8 of 8, D126: brms's are loo's optimiser's, 1.044e-8 off the
  optimum). `binomial_cloglog`, frozen with P6, needs no opt-in: brms is within the tolerance at every entry there.

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
- A problem a feature finds in brms, Stan Math or loo (a wrong or inexact result, or input accepted that should be refused) goes into the upstream log, `docs/brms-parity/upstream/` (an entry in `log.json` with a standalone R reproduction `repro/<id>.R`), besides its divergence row. The log is kept locally, out of git, until it is reported upstream.

## Commands

```bash
python oracle/regen.py new <case-id>
```

```bash
python tools/parity_gate.py --feature <id> --base origin/main
```

```bash
python tools/diff_coverage.py --base origin/main
```

```bash
python tools/parity_status.py
```
