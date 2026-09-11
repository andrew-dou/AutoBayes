# M3b-1 decision record: independent varying intercept and one numeric varying slope

Date: 2026-09-11. Branch `m3b1/independent-varying-slope` from `main` at `95dce8c` (M3a-2 merged),
worktree `projects/AutoBayes-worktrees/m3b1-varying-slope`. Written and committed before any
implementation. Source: Andrew's approval of the CPU roadmap (2026-09-09), section 4.2 PR 1 of
`docs/coordination/NEXT_MILESTONES_PROPOSAL.md`, the coordinator's M3b-1 dispatch
(`memory/dispatches/2026-09-10-autobayes-m3b1-independent-slopes.txt`). Scope: one numeric varying
slope, independent of the varying intercept, on the shared right-hand side of the three families;
correlated effects (LKJ), several varying slopes, categorical varying slopes, additional grouping
factors, and slope-only terms are separate work.

## Measured before deciding (formulae 0.6.2 / Bambi 0.17.2, build observation; scratch probe)

- formulae raises `ParseError` on `||`, so the independent form must be rewritten by this layer.
- Bambi's `(1 + x | g)` builds two group-specific terms `1|g` (kind `intercept`) and `x|g` (kind
  `numeric`), each with its own SD hyperprior and non-centred offsets: value variables
  `1|g_sigma_log__`, `1|g_offset`, `x|g_sigma_log__`, `x|g_offset`; posterior variables `1|g`
  and `x|g` (the effects), both on the shared dims `g__factor_dim`; no correlation parameter
  (the LKJ helper is commented out in Bambi). This is independence by construction.
- The slope block of the group design is $x_i \mathbb 1[g_i = j]$ with **raw** $x$, whether or not
  `center_predictors=True` (Bambi centres only the common columns): the linear predictor equals
  $\text{fixed}_i + u[g_i] + x_i v[g_i]$ exactly (0.0), and the centred alternative differs (1.5).
  The hyperprior log density of the slope term equals $\sum_j \log N(z_j; 0, 1) + \log \text{HalfNormal}(sd_1)$.
- `(x | g)` silently adds the varying intercept; `(0 + x | g)` builds a slope only; `(1 + x + w | g)`
  builds three terms. None of these is admitted here.

## Target

$$\eta_i = \text{fixed}_i + u[g_i] + x_i\, v[g_i], \qquad u_j = sd_0\, z^{(0)}_j,\quad v_j = sd_1\, z^{(1)}_j,\quad
z^{(0)}_j, z^{(1)}_j \overset{iid}{\sim} N(0, 1),$$

with $\text{fixed}_i$ the M0-M3a linear predictor (centred as before when `center_predictors=True`),
$sd_0 \sim$ `sd["Intercept"]`, $sd_1 \sim$ `sd["x"]`, sampled on the log scale with the Jacobian, and
$u_j$, $v_j$ independent a priori by construction. **Coordinates:** $v_j$ multiplies the **raw** $x_i$
(no centring of the group-specific column, verified above and checked at build time), so the
declared independence holds for the intercept at raw $x = 0$. Shifting the covariate by $c$ would
give $u'_j = u_j + c\, v_j$, which is correlated with $v_j$ with covariance $c\, sd_1^2$; this layer
therefore never recentres the slope covariate and does not offer a reparameterisation in this
slice (one would have to carry its transformed, correlated prior). The fixed-effect centring and
the reported raw `Intercept` transformation are unchanged and unaffected by the group slope.

## Accepted defaults (public contract)

1. **Grammar.** `(1 + x || g)` or `(x + 1 || g)` (the brms independent form) with `x` a numeric
   fixed-effect main term of the same formula (not categorical, not an interaction, not the
   grouping factor, not the response or trials column). Canonical spelling `(1 + x || g)`; the
   Bambi formula rewrites it to `(1 + x | g)` (the only construction Bambi has, which is
   independent) and the spec records `varying_slope = "x"`. `(1 || g)` is accepted as `(1 | g)`.
   Existing `(1 | g)` models are unchanged.
2. **Refused with guidance.** `(1 + x | g)` (single bar with a slope): correlated varying slopes
   are brms's default with an LKJ prior and are not supported in this slice; Bambi would build
   independent effects under that spelling, so it is refused and the independent form is named.
   Also refused: `(x || g)`, `(x | g)` (implicit intercept), `(0 + x || g)` (no varying intercept),
   more than one slope, a categorical or interaction slope, a slope covariate absent from the
   fixed part, and the group as a slope.
3. **Priors.** `sd` is either one positive-support prior (existing API, varying-intercept-only
   models) or a mapping covering exactly the varying effects: `{"Intercept": prior}` for `(1 | g)`,
   `{"Intercept": prior, "x": prior}` for `(1 + x || g)`. A varying-slope model with a scalar `sd`,
   a missing key, an unknown key, or a non-positive-support prior is refused; nothing is filled in.
4. **Names** (grouping factor `g`, slope covariate `x`):

   | quantity | name |
   |---|---|
   | intercept SD, offsets, effects | `1|g_sigma`, `1|g_offset`, `1|g` (as before) |
   | slope SD, offsets, effects | `x|g_sigma`, `x|g_offset`, `x|g` |
   | `model.names` | `sd`, `offset`, `group_effect` (intercept) and `sd_slope`, `offset_slope`, `group_slope` |
   | `value_point` | `sd`, `z` (intercept) and `sd_slope`, `z_slope` (required exactly when a slope is present) |
   | `describe()` | `u[j] = sd_g * z0[j]`, `v[j] = sd_x|g * z1[j]`, "independent by construction" |
   | fit record | `priors["sd:Intercept"]`, `priors["sd:x"]`, `varying_slope`, summary rows for both SDs |
   | `Design` | `Z` (n, J) one-hot as before; `Z_slope` (n, J) = raw x times `Z`; `slope` |
5. **Design verification.** Bambi's group design is checked block by block against the original
   labels and raw covariate: the intercept block is one-hot over the sorted levels, the slope block
   equals $x_i \mathbb 1[g_i = j]$ with the supplied (uncentred) $x$; any centring of the group
   column fails the build (kill criterion).
6. **Predictions.** $\eta = a + X_{raw} b + u[g] + x\, v[g]$ per draw. Unseen grouping labels:
   `error` refuses; `population` draws one independent pair $(z^{(0)}, z^{(1)}) \sim N(0,1)^2$ per
   distinct new label per posterior draw, scaled by that draw's SDs, reused for every row with that
   label and independent across labels; `resample_observed` chooses one observed group index per
   new label per draw and takes **both** $u_j$ and $v_j$ from that same group in that same draw.
   Existing varying-intercept behaviour is unchanged. Bernoulli probability, binomial probability,
   expected count and predictive count keep their M1/M2 meanings.
7. **Validation.** Labels, counts, zero-trial rows and the count cap are validated as before; the
   zero-trial drop precedes the group design; the slope covariate must vary within the fitted rows.

## Acceptance (tests on the public API, clean locked environment, fresh evidence directories)

- F1 Grammar: the accepted spellings give the same spec, design and Bambi formula; every refusal
  in default 2 with its message; every earlier formula and refusal unchanged.
- F2 Group design from the original rows and labels: `Z` one-hot over sorted levels and `Z_slope`
  equal to raw $x$ times `Z`, for the three families, fixed centring on and off, and the binomial
  family with zero-trial rows selected independently by the test; PyMC names and dims as in the map.
- F3 A separately written, fully normalised reference density (both SD priors, two offset vectors,
  the two log-scale Jacobians, all fixed priors, likelihood) from the caller's original data,
  compared with the PyTensor and JAX log densities (with and without the Jacobian) and with full
  gradients (analytic and central differences) at several finite points with non-trivial priors,
  for Gaussian, Bernoulli and binomial models.
- F4 Deterministic synthetic-posterior predictions: known groups at several $x$; unseen labels in
  `population` mode (one pair per label per draw, reused for repeated rows, distinct labels
  independent, finite) and `resample_observed` mode (one observed index per label per draw, both
  effects from that group's joint draw); row order and repetition; exact label identity; the
  Bernoulli probability and the binomial probability / expected count / predictive count
  distinctions; unchanged behaviour for varying-intercept models.
- F5 Priors: scalar and mapping forms; refusals of default 3.
- F6 `describe()` states both SDs, the two offset vectors, the raw-$x$ statement and the
  independence-by-construction statement; the fit record carries both SD priors and the slope.
- F7 Recovery (slow, both CPU samplers), prespecified: Gaussian `y ~ x + (1 + x || g)`, 600 rows,
  J = 30 groups of 20 rows, $x \sim N(0, 1)$ within every group (checked to vary within every
  group), seed 17; truth $a = 1.0$, $b = 0.5$, $sd_0 = 0.5$, $sd_1 = 0.3$, sigma $= 0.4$; priors
  Normal(0, 5) intercept, Normal(0, 2) slope, HalfNormal(1) for both SDs and sigma; 4 chains, 1000
  tune, 1000 draws, seed 17, `target_accept` 0.9. Gates fixed now: 0 divergences; max R-hat
  $\le 1.01$; min bulk ESS $\ge 400$ over the five population parameters (`Intercept`, `x`,
  `1|g_sigma`, `x|g_sigma`, `sigma`); each truth inside its 99 % HDI (nominal false-failure
  budget $\le 5$ % under independence); backends within 4 combined MCSE on every population mean.
  The posterior SD of each SD estimand is recorded with the result. A failed gate is investigated
  and reported, not re-run to pass. One recovery is not calibration.
- F8 Every existing test, gate and independent regression passes unchanged.
- F9 Example `examples/m3b1_varying_slope.py` on both backends, outputs in `docs/m3b1-evidence/`;
  SKILL and README explain `||` versus `|`, raw $x = 0$, centring, new-group modes and diagnostics.
- F10 `value_point`, `names`, `Design.Z_slope`, and the fit record as in the map.

## Kill criterion

If Bambi's group design for the slope is not raw $x$ times the one-hot matrix in any tested case,
or the density check fails on either path, the slice stops and reports; no reparameterisation is
substituted silently.

## Executed results (2026-09-11, clean locked environment, Python 3.11.0, fresh evidence directories; real pytest exit codes in `docs/m3b1-evidence/checks.txt`)

- Build observation confirmed at build time in every test: Bambi's slope block equals the raw
  covariate times the one-hot matrix (the kill criterion was never triggered).
- `tests/test_varying_slope.py`: 18 passed. Fast suite: 365 passed, 8 deselected (312 s; pytest exit 0). Slow suite: 1 failed, 7 passed (54 s; pytest exit 1): the failed test is F7, the prespecified recovery (population-SD gate), reported below and not re-run to pass. Independent
  label-identity, M2 normalisation and M3a-2 original-input regressions: 27 passed (nine label-identity, three M2 normalisation, fifteen M3a-2 original-input).
- F7 recovery: **FAILED one prespecified gate on both CPU samplers, recorded and investigated, not re-run.** Divergences 0/0; max R-hat 1.0074 (PyMC) and 1.0117 (NumPyro: `1|g_sigma` 1.0117 exceeds the 1.01 gate on that backend; PyMC passes it); min bulk ESS 468 (PyMC) and 395 (NumPyro: `Intercept` 395 misses the 400 gate); `Intercept`, `x`, `1|g_sigma` and `sigma` inside their 99 % HDIs on both backends; `x|g_sigma` posterior mean 0.175/0.174 with 99 % HDI [0.100, 0.272]/[0.100, 0.275] does NOT contain the population truth 0.30 on either backend; backends agree with each other. Frozen: `docs/m3b1-evidence/F7_varying_slope_recovery_FAILED_GATE.json` and the records. Investigation (`docs/m3b1-evidence/F7_investigation.txt`, script alongside; no library design helper): for seed 17 the realised spread of the 30 simulated slopes is 0.220 (population 0.30; 29 s^2/sigma^2 = 15.6, a ~2 % event) and of the 30 intercepts 0.331 (population 0.50; 12.7, a ~0.4 % event). An independent per-group OLS analysis gives a noise-corrected slope spread of 0.171, matching the model's posterior mean 0.177; the posterior 99 % interval [0.111, 0.276] contains the realised slope SD 0.220; the posterior-mean slopes correlate 0.915 with the true slopes (0.951 for the intercepts). The model recovers the realised group effects; the failed gate compares the SD hyperparameter with a population value that 30 realised groups do not pin down in this draw. The prespecified gate stands as failed; no widening, no new seed. Proposed for the coordinator's decision (not applied): gate SD hyperparameters at finite J on the realised SD of the simulated effects (population value reported), and/or a second prespecified design with more groups.
- Example (`docs/m3b1-evidence/`): both backends recover the population slope and the residual SD, with `1|site_sigma` and `dose|site_sigma` reported with their uncertainty (see the files); PyMC and NumPyro agree within 4 combined MCSE on every population mean; the single-bar form is refused with the guidance; `--device gpu` raises `DeviceError` (captured). Files: `example_pymc_cpu.txt`, `example_numpyro_cpu.txt` (with the comparison), the two fit records, `example_numpyro_gpu_attempt.txt`.
- Evidence directories of these runs: `evidence-runs/20260911T003657.969678Z-8307a90f204f-dirty-p71239-68410b/` (fast, final), `evidence-runs/20260911T003854.676250Z-8307a90f204f-dirty-p71752-7d0f6b/` (slow, final), `evidence-runs/20260911T002903.692270Z-8307a90f204f-dirty-p70699-5b76d1/` (first slow; F7 files frozen in `docs/m3b1-evidence/`) and `evidence-runs/20260911T002709.385688Z-8307a90f204f-dirty-p69234-feca57/` (first fast, six old-expectation failures); earlier run directories untouched.
