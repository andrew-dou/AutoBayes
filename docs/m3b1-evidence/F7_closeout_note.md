# F7 closeout note: reference outcome, Q7 interpretation, saved-draw inventory (2026-09-13)

Read-only unit. Inputs and their SHA-256 are listed in `F7_closeout_inputs.json` (records 72bec95cff15f168…,
failed gate db193c18b6e982eb…, unit result 3d2a80294c59e92b…,
unit ledger 5e41bbd2e96b5526…, reviewer verification f1f57ea1b11086bc…).
No fit, evaluation, quadrature, data generation, reseeding or threshold change; the executed unit, its approval and both ledgers are
unchanged; `result["comparison"]` stays empty. Recomputations here are arithmetic on saved files.

## 1. What the executed unit established

One deterministic integration of the analytically integrated Gaussian posterior of (sd0, sd1, σ) on the frozen seed-17 data set:
12,038,157 points in 267.9 s, all 132 prespecified conditions met, largest crossed difference 5.27e-7, all twelve sampled face
deficits ≤ −26.26 (independently verified at 0a22669). It is a computation reference for that posterior on that data set with
sampled, not bounded, tails.

## 2. Q7 assessment (recomputed; identical to the reviewer's table to every printed digit)

| Backend | Parameter | Reference mean (B1, fine rule) | Saved MCMC mean | Stored MCSE | Discrepancy / MCSE | Within 4·MCSE + spread |
|---|---|---:|---:|---:|---:|---|
| pymc | 1|g_sigma | 0.354940520 | 0.354191429 | 0.001899 | 0.394 | True |
| pymc | x|g_sigma | 0.175822518 | 0.175385648 | 0.000900 | 0.485 | True |
| pymc | sigma | 0.424456550 | 0.424134104 | 0.000201 | 1.607 | True |
| numpyro | 1|g_sigma | 0.354940520 | 0.354424757 | 0.002076 | 0.248 | True |
| numpyro | x|g_sigma | 0.175822518 | 0.173907415 | 0.000978 | 1.959 | True |
| numpyro | sigma | 0.424456550 | 0.424736889 | 0.000199 | 1.412 | True |

Reference spreads over the four box/rule means: 1.25e-09, 9.99e-09, 3.11e-09.
Supported reading: the six saved posterior scale means agree with the deterministic reference within the planned allowance
(largest 1.959 MCSE, the NumPyro slope SD). Qualifications: it compares means only, for three of the five reported parameters;
the allowance uses each sampler's own MCSE, which rests on the ESS estimates of the very chains whose diagnostics are in
question; the spread term is an observed sensitivity, not an error bound; discrepancies of 1.4 to 2.0 MCSE for σ and the
NumPyro slope SD are unremarkable for four chains of 1,000 draws and are evidence neither of bias nor against it. Nothing here
establishes convergence, calibration or full-distribution correctness.

## 3. Where the population truth sits

At x|g_sigma = 0.30 the reference CDF is 0.9981845983506333 on B1 with the fine rule (the four box/rule values agree to
5.3e-7), an upper tail of 0.18154 %. The correctly computed posterior therefore excludes 0.30 from any central 99 % interval
on this realisation; the saved 99 % HDIs [0.100, 0.272] (PyMC) and [0.100, 0.275] (NumPyro) are consistent with that. The F7
truth-in-HDI failure is a property of the posterior given this realisation, whose realised slope spread 0.2197 against the
population 0.3 was a ≈2 % event (`F7_investigation.txt`), not a demonstrated sampler or implementation defect. This is one
realisation; it is not a coverage measurement and not a reason to replace the population SD by the realised spread.

## 4. Saved posterior draws: none

Searched every run directory under `evidence-runs/` in this worktree, the sibling worktrees and the original checkout for
InferenceData or array files; only JSON summaries exist. The F7 test writes records, population rows and the cross-backend
agreement and does not persist draws; the committed F7 files are byte-identical to evidence run `…-p70699-5b76d1`. Split-chain,
rank or trace diagnostics of the original NumPyro run are therefore impossible, and draws are not recreated here.

## 5. What the saved summaries show about the NumPyro miss

| Backend | Parameter | R-hat | bulk ESS | 99 % HDI | truth | inside |
|---|---|---:|---:|---|---:|---|
| pymc | Intercept | 1.007392 | 468.1 | [0.8786, 1.2319] | 1.0 | True |
| pymc | x | 1.001301 | 1383.9 | [0.4007, 0.5961] | 0.5 | True |
| pymc | 1|g_sigma | 1.006939 | 800.4 | [0.2403, 0.5130] | 0.5 | True |
| pymc | x|g_sigma | 1.002116 | 1387.3 | [0.1004, 0.2720] | 0.3 | False |
| pymc | sigma | 1.001467 | 4303.4 | [0.3892, 0.4575] | 0.4 | True |
| numpyro | Intercept | 1.002632 | 395.3 | [0.8797, 1.2452] | 1.0 | True |
| numpyro | x | 1.000641 | 1064.7 | [0.4085, 0.6006] | 0.5 | True |
| numpyro | 1|g_sigma | 1.011748 | 677.8 | [0.2283, 0.5134] | 0.5 | True |
| numpyro | x|g_sigma | 1.002046 | 1162.6 | [0.0999, 0.2750] | 0.3 | False |
| numpyro | sigma | 1.002203 | 4332.2 | [0.3928, 0.4583] | 0.4 | True |

Two distinct offenders, as the review states. R-hat 1.011748 sits on 1|g_sigma, the group-intercept SD, whose bulk ESS (678)
is adequate: the miss is between-chain disagreement on a hyperparameter of the non-centred group block, not a lack of draws.
Bulk ESS 395.25 sits on Intercept with R-hat 1.0026: too few effective draws for the fixed intercept, which trades off against
the mean of the group-intercept offsets; PyMC shows the same ordering (Intercept lowest ESS 468, highest R-hat 1.0074) inside
the gates. The mean agreement of section 2 erases neither: R-hat measures chain disagreement and ESS the number of effectively
independent draws, and both can miss while a four-chain pooled mean lands within two MCSE of the reference, as here. The
mechanisms named are hypotheses; only draws could confirm them, and none are saved.

## 6. Acceptance-policy issue and proposal

The F7 test's acceptance assertion is a one-realisation coverage event (truth inside the 99 % HDI for every population
parameter), which misses about 1 % per parameter under perfect calibration and which no correct computation can pass on this
realisation. It is not equivalent to posterior computation accuracy. The exact proposed change, for independent review and not
implemented here, is in `F7_convergence_followup_proposal.json` (`acceptance_policy_proposal`): keep the original test and its
failed observation unchanged; add a separately named computation-and-convergence test gated on the existing diagnostic panel
and on agreement with the frozen deterministic reference (means within 4·MCSE + spread; P̂(x|g_sigma ≤ 0.30) against
0.99818460 within Monte-Carlo error); record truth-in-HDI as an observation; scope calibration separately. No deletion, skip,
xfail, weakening or relabelling of F7, no threshold change, no reseeding, no until-pass loop.

## 7. Single next step (UNAPPROVED)

One frozen-data NumPyro convergence diagnostic, fully predeclared in `F7_convergence_followup_proposal.json`: same model,
priors, data, seed 17 and runtime; 4 chains, 2,000 tune, 4,000 draws, target_accept 0.9; 300 s CPU cap, zero retries;
existing panel gates plus the two reference-agreement checks; draws retained. It can say whether the two diagnostic misses are
chain-length artefacts and whether longer chains agree with the reference; it cannot say anything about calibration or coverage,
which need a separately scoped study. Original F7 stays FAILED; PR #8 stays open; M3b-2 does not start.
