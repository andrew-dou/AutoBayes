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
- F7 recovery: **FAILED at the prespecified thresholds; the failure stands.** Original result (first slow suite,
  evidence directory `...-p70699-5b76d1`, frozen as `docs/m3b1-evidence/F7_varying_slope_recovery_FAILED_GATE.json`):
  divergences 0/0; max R-hat 1.0074 (PyMC) and 1.0117 (NumPyro: `1|g_sigma` 1.0117 exceeds the 1.01 gate on that
  backend); min bulk ESS 468 (PyMC) and 395 (NumPyro: `Intercept` 395 misses the 400 gate); `Intercept`, `x`,
  `1|g_sigma` and `sigma` inside their 99 % HDIs on both backends; **`x|g_sigma` (posterior mean 0.175/0.174, 99 % HDI
  [0.100, 0.272] / [0.100, 0.275]) does not contain the generating value 0.30 on either backend**; the two backends
  agree with each other within 4 combined MCSE. **Actual fit accounting** (each a distinct sampling run; none was a
  re-run of F7 "to pass" and nothing was widened or reseeded): (1) the original F7 in the first slow suite above;
  (2) F7 again in the second slow suite `...-p71752-7d0f6b`, run to obtain the suite counts after six old test
  expectations were updated, same deterministic seed, same failure; (3) one additional investigation fit on the F7
  data (PyMC, 2 chains x 500 tune + 500 draws, seed 17, `docs/m3b1-evidence/F7_investigation.txt`); (4) one live
  demonstration fit requested by Andrew on 2026-09-11 (a different model and data: `y ~ dose + arm + (1 + dose ||
  site)`, NumPyro, 4 x 500 + 500), not part of the evidence. **Estimand statement (corrected 2026-09-12):** the gated
  parameter `x|g_sigma` is the *population* SD of the Normal group slopes; the realised sample SD of the 30 simulated
  slopes (0.2197) is a different random quantity and is not a substitute truth for it. The investigation (a ~2 %
  realised-spread event, an independent per-group OLS spread of 0.171 against the posterior mean 0.177, posterior
  containing the realised SD) supports rare realised group variation as a *possible explanation* of the failure; it
  is not proof of sampling correctness and does not fulfil the acceptance gate. A 99 % posterior interval at one
  fixed generating value does not by itself guarantee 99 % repeated-sample coverage, and no familywise false-failure
  rate follows automatically for the five intervals. Deterministic correctness (density, Jacobians, gradients on
  both paths, design) and sampler convergence are established separately from finite-data recovery; see the repair
  record below for the proposal.
- Example (`docs/m3b1-evidence/`): both backends recover the population slope and the residual SD, with `1|site_sigma` and `dose|site_sigma` reported with their uncertainty (see the files); PyMC and NumPyro agree within 4 combined MCSE on every population mean; the single-bar form is refused with the guidance; `--device gpu` raises `DeviceError` (captured). Files: `example_pymc_cpu.txt`, `example_numpyro_cpu.txt` (with the comparison), the two fit records, `example_numpyro_gpu_attempt.txt`.
- Evidence directories of these runs: `evidence-runs/20260911T003657.969678Z-8307a90f204f-dirty-p71239-68410b/` (fast, final), `evidence-runs/20260911T003854.676250Z-8307a90f204f-dirty-p71752-7d0f6b/` (slow, final), `evidence-runs/20260911T002903.692270Z-8307a90f204f-dirty-p70699-5b76d1/` (first slow; F7 files frozen in `docs/m3b1-evidence/`) and `evidence-runs/20260911T002709.385688Z-8307a90f204f-dirty-p69234-feca57/` (first fast, six old-expectation failures); earlier run directories untouched.

## Repair record (2026-09-12): value_point presence validation, combined-feature verification, corrected reporting

**Findings (independent review, `memory/reviews/2026-09-12-autobayes-pr8-review.md`; probes in
`artifacts/2026-09-12-autobayes-m3b1-review/`, preserved).** (1) `value_point` tested the *conjunction*
`sd_slope is not None and z_slope is not None`, so on a varying-intercept-only model `sd_slope` alone or
`z_slope` alone was silently ignored (reproduced by the coordinator: both accepted, no slope variable in the
returned point). (2) The supported combination of a numeric varying slope with fixed categorical main effects
and a numeric-by-categorical interaction had no deterministic repository test of its own (the coordinator's
six composite probes pass with error at most 1.4e-14, without any AutoBayes reference helper). (3) The
executed-results text treated the realised sample SD of the simulated slopes as if it were the truth for the
population SD, claimed a nominal 5 % familywise false-failure budget, and said the failed gate was "not
re-run" although F7 was sampled twice (two slow suites) plus an investigation fit and a separate live demo.

**Repair (decided before coding).**

1. `value_point`: on a model without a varying slope, `sd_slope` or `z_slope` given alone or together is
   refused (`ValueError` naming the extra argument); on a model with a varying slope both are required and
   either missing is refused naming the missing one. Valid numerical targets are unchanged. Regression coverage:
   all four presence combinations on both model forms, valid shapes, wrong shapes, and the returned point's
   exact key set.
2. `tests/test_varying_slope_combined.py`: the coordinator's composite probe adopted and extended, from the
   caller's original rows and labels with no library reference helper: `lhs ~ dose*category + (1 + dose ||
   site)` for the three families, centring on and off, the binomial positive-trial rows selected by the test;
   the fixed and group designs, the fully normalised joint density with both SD priors and both log-scale
   Jacobians (with and without the Jacobian), the full gradient on the PyTensor path (analytic and central
   differences) and the JAX path, and synthetic-posterior predictions with the (u, v) pairing for known and
   unseen sites. Plus the **analytic conditional-Gaussian check with both group-effect blocks**: for the Gaussian
   model at fixed `sigma`, `sd0`, `sd1` the conditional posterior of `(a_c, b, z0, z1)` is Normal with
   precision `P = diag(prior precisions, 1, 1) + W^T W / sigma^2` and `W = [1, X_c, sd0 Z, sd1 x Z]`, built
   independently from the original inputs; the graph's gradient with respect to those variables must equal
   `-P (theta - mean)` at several points and its Hessian must equal `-P`, on both compilation paths. No model
   fit is involved.
3. Reporting: the executed-results text above is corrected (estimand statement, coverage statement, actual fit
   accounting); the same corrections go to `docs/coordination/STATUS.md`, the PR body and the worker memory.
   The frozen F7 evidence and the original investigation file are not edited; a separate note
   `docs/m3b1-evidence/F7_reading_correction.txt` records which sentences of the investigation's "Reading"
   paragraph are superseded. **Correction to the pre-code F7 text of this record:** the sentence "nominal
   false-failure budget <= 5 % under independence" asserted a repeated-sample property that a single 99 %
   posterior interval at a fixed generating value does not provide; it is withdrawn and left visible above.
4. No new F7 sampling, no widened thresholds, no reseeding, no removal, skip or xfail of the failed test; no
   M3b-2 work. The candidate is verified with targeted deterministic tests and the fast suite only; the slow
   suite's existing F7 failure is reported separately.

**Decision proposal for the coordinator (not executed here).** Three questions must be answered separately:

- *Deterministic implementation correctness* is established by the density, Jacobian, gradient and design
  checks (F2, F3, the combined-feature module, the analytic conditional-Gaussian check with both blocks) and does
  not depend on any sampler or truth value.
- *Sampler convergence* on the F7 data is established by the diagnostics (0 divergences; PyMC within every
  R-hat/ESS gate; NumPyro's `1|g_sigma` R-hat 1.0117 and `Intercept` bulk ESS 395 slightly outside) and by the
  backends' mutual agreement; it says nothing about the truth.
- *Finite-data recovery of a population SD* from J = 30 groups is a repeated-sample property. One draw cannot
  establish it, and one failed 99 % interval cannot refute the sampler: the interval's coverage under repeated
  data generation is unknown a priori, and the realised group spread is itself random.

Recommended next bounded check, for review before any sampling: (i) a **sampler-correctness check of the SD
posterior without a truth value**: in the Gaussian F7 model with `sigma`, `a_c`, `b` fixed (via `pm.do`), the
marginal posterior of `sd0` and `sd1` after integrating the offsets is available by 1-D/2-D numerical
integration of a closed-form Gaussian marginal (each group's residual vector is multivariate Normal with
covariance `sigma^2 I + sd0^2 11^T + sd1^2 x x^T`); compare NUTS's marginal against that integration on the
frozen F7 data, one fit per backend, 4 x (1000 + 1000), gates: mean within 4 MCSE and quantiles 0.05/0.5/0.95
within 0.02. (ii) A **fixed-budget repeated-sample study** of the F7 generator for the population-SD estimand:
R = 24 replicate data sets from seeds 1701..1724 (fixed now), the same design (J = 30 x 20), PyMC only, 4 x
(500 + 1000), `target_accept` 0.9, about 25 CPU-minutes; prespecified gates: the empirical coverage of the 90 %
interval for `x|g_sigma` and `1|g_sigma` between 0.70 and 1.00 (two-sided exact binomial p >= 0.01 against
0.90 at R = 24), zero divergences in at least 22 of 24 fits, and the rank of the generating value among the
posterior draws (SBC rank statistic) passing a chi-square uniformity test at 0.01 with 6 bins. (iii) Only if
(i) passes and (ii) fails is a model or sampler defect indicated; if both pass, the original F7 failure is
recorded as a single-draw outcome with the design's coverage now measured. No change to F7's threshold, seed
or estimand in any case.

### Correction of the decision proposal (2026-09-12, second review; the paragraph above is superseded, not erased)

- The proposed 24-replicate study fixes the generating hyperparameters and repeatedly simulates data. That
  is a **fixed-parameter coverage study, not simulation-based calibration**: the posterior ranks of a fixed
  truth carry no uniformity guarantee. True SBC draws the parameters from the *fitting prior*, then the data
  from the likelihood, and must handle posterior autocorrelation when forming ranks
  (https://mc-stan.org/docs/stan-users-guide/simulation-based-calibration.html, sections "Bayes is
  calibrated by construction" and "Simulation-based calibration checking"). The uniform-rank acceptance gate
  is withdrawn, and so is the claim that a failed fixed-truth coverage study with a passing conditional
  check would indicate a model or sampler defect: a failure of nominal coverage at one fixed truth does not
  by itself diagnose either. The 24-fit study is **not approved and is not run**.
- The proposed `pm.do` check fixed `sigma`, the intercept and the slope, which changes the target; its
  result must not be compared with the full F7 summaries. Withdrawn in favour of the full-posterior
  marginal reference below.
- "Sampler convergence is established" is withdrawn for the F7 run: the NumPyro chains missed the original
  R-hat (1.0117 on `1|g_sigma`) and bulk-ESS (395 on `Intercept`) gates. Coverage, computation and
  convergence are kept distinct: the marginal reference below bears on computation (does the sampler's
  posterior for the scales agree with an independent deterministic evaluation of the same posterior on the
  same data), on nothing else.

### Marginal-reference protocol (recorded 2026-09-12 before any posterior comparison was evaluated)

**Target.** The full F7 Gaussian model on the frozen seed-17 data (generator reproduced exactly in
`tests/reference_marginal.py::f7_data`) with exactly the recorded priors. Integrating the Gaussian
coefficients $\beta = (a_c, b) \sim N(m, V)$, $m = 0$, $V = \mathrm{diag}(25, 4)$, and both Gaussian
group-effect blocks analytically leaves the three positive scales:

$$y \sim N\!\big(A m,\; \sigma^2 I + A V A^\top + sd_0^2 Z Z^\top + sd_1^2 Z_x Z_x^\top\big),\qquad
A = [1,\; x - \bar x],\; Z = \text{one-hot groups},\; Z_x = x \odot Z\ (\text{raw } x),$$

and on $\theta = (\log sd_0, \log sd_1, \log \sigma)$ the target is $\ell(\theta) = \log N(y; Am, \Sigma(\theta))
+ \sum_k [\log \mathrm{HalfNormal}(e^{\theta_k}; 1) + \theta_k]$ (the three log-scale Jacobians included). The
evaluation uses group-block low-rank algebra (each block $\sigma^2 I + U_j D U_j^\top$ with $U_j = [1, x_j]$,
$D = \mathrm{diag}(sd_0^2, sd_1^2)$; the fixed-effect term a global rank-2 update; Woodbury and the matrix
determinant lemma), built from the original rows, never from a library design helper. It must equal a
dense $n \times n$ multivariate-normal evaluation on a small crossed fixture over random scale points (four
decades, near-degenerate blocks included) and on the F7 data at fixed points before any quadrature
(`tests/test_marginal_reference.py`).

**Domain selection (from the analytic target only).** Mode $\theta^*$ by numerical optimisation of $\ell$
(Nelder–Mead, refined by BFGS); curvature scales $s_k = ((H^{-1})_{kk})^{1/2}$ from the central-difference
Hessian at $\theta^*$; box $B(c) = \prod_k [\theta^*_k - c s_k, \theta^*_k + c s_k]$ with $c = 8$ (primary) and
$c = 10$ (expansion check). Tail check: on each of the six faces of $B(8)$, sampled by a $48 \times 48$
Gauss–Legendre grid, $\max(\ell - \ell^*) \le -25$ (density ratio $\le 1.4 \times 10^{-11}$); a violated face
expands that axis by $2 s_k$, at most three times.

**Quadrature.** Tensor-product Gauss–Legendre on $B(8)$ with $n \in \{24, 32, 48, 64\}$ nodes per axis;
$\log Z = \mathrm{logsumexp}(\ell + \log w)$; $E[sd_k] = \sum e^{\theta_k} w\, e^{\ell - \log Z}$. Refinement error:
the change from $n = 48$ to $n = 64$. Domain error: the change from $B(8)$ to $B(10)$ at $n = 64$. One-dimensional
marginals: for each $k$, 141 equally spaced $\theta_k$ over $B(8)_k$, the other two axes integrated with 48-node
Gauss–Legendre; each marginal normalised by its own trapezoid integral, its mass compared with the $Z$-based
normalisation and its mean with the three-dimensional mean (consistency); CDF by cumulative trapezoid;
quantiles by inverse linear interpolation.

**Error targets, fixed now.** $|\Delta E[sd_k]| \le 10^{-4}$ under refinement and under domain expansion;
$|\Delta \mathrm{CDF}| \le 10^{-3}$ at the reference points ($sd_1 = 0.30$ and each stored MCMC mean);
$|\Delta \log Z| \le 10^{-4}$; marginal-versus-3-D mean consistency $\le 10^{-4}$. Achieved errors are reported.
If the targets are not met within the caps the check is reported **inconclusive** with the partial evidence;
no criterion is relaxed.

**Caps.** Local CPU only; at most 15 minutes of quadrature runtime and 2,000,000 marginal-density
evaluations (counted); no sampling, no new generated data, no new dependency, no GPU or cluster.

**Comparison rule (applied only after the targets are met).** For each scale and each stored backend record,
$|E_{quad}[sd_k] - \bar{sd}_{k,MCMC}|$ against $4 \times \mathrm{MCSE}_{stored} + \mathrm{err}_{quad}$: a diagnostic
tolerance, not a proof; agreement or discrepancy is labelled with the quadrature uncertainty. The marginal
CDF at the population slope SD, $P(sd_1 \le 0.30 \mid y)$, and central 99 % and 90 % intervals from the
quadrature marginals are reported as a **new deterministic reference**, distinct from the sampled HDIs.
Nothing about F7 (data, seed, estimand, thresholds, test, evidence) changes; no calibration or convergence
claim follows from this check.

### Executed marginal-reference runs and corrected status (2026-09-13; coordinator scope correction)

**Evaluation accounting (unit cap 2,000,000 evaluations, 15 minutes).** Three processes ran
(`docs/m3b1-evidence/F7_marginal_reference_ACCOUNTING.txt`): attempt 1 (02:54:50Z) evaluated 2,289,209
densities before raising, because the driver checked its counter only after computing each whole batch;
attempt 2 (02:56:19Z) reset the counter at process start and evaluated 1,738,745, writing
`F7_marginal_reference.json/.txt`; a tail addendum (02:57Z, `f7_marginal_tail.py`, no budget check) evaluated
97,600. **Cumulative: 4,125,554 evaluations, 206 % of the unit cap.** The cap was exceeded inside attempt 1
and then treated as a per-retry allowance; that was wrong. Both shell wrappers mis-reported exit codes (a grep
status after a pipe; an empty zsh `pipestatus`), so attempt 1's traceback was summarised as "exit 0". The
coordinator's stop record is `artifacts/2026-09-13-autobayes-marginal-cap-review/`. No further evaluation of
any kind has been run since; none is authorised.

**Status of the attempt-2 result: incomplete / exploratory.** The file's `targets_met = True` is withdrawn as an
unconditional claim. Against the recorded protocol: five faces met the $-25$ tail criterion ($-27.0$ to
$-63.6$); the lower $\log sd_1$ face did **not** ($-20.04$ after the three permitted expansions); refinement,
domain, $\log Z$ and marginal-consistency targets were met (means to $4\times10^{-10}$, $\log Z$ to $1.5\times10^{-8}$);
the CDF-stability condition was **not checked as specified** (only $sd_1$ at $0.30$ was recomputed, with
resolution and domain changed together, change $1.3\times10^{-4}$; the six MCMC-mean reference points and the other
two scales were not checked). The addendum's measured mass below the unmet face, $3.3\times10^{-9}$, is an
exploratory observation obtained after the cap was already exceeded.

**Exploratory numbers, preserved and labelled as such (not acceptance, calibration or convergence).** Quadrature
means $0.354941$ ($sd_0$), $0.175823$ ($sd_1$), $0.424457$ ($\sigma$); $P(sd_1 \le 0.30 \mid y) = 0.9981$; central
99 % intervals $[0.247, 0.527]$, $[0.104, 0.282]$, $[0.393, 0.460]$ (distinct from the sampled HDIs); the saved
F7 MCMC means differ from the quadrature means by $7.5\times10^{-4}$ / $4.4\times10^{-4}$ / $3.2\times10^{-4}$ (PyMC)
and $5.2\times10^{-4}$ / $1.9\times10^{-3}$ / $2.8\times10^{-4}$ (NumPyro), each below $4 \times$ its stored MCSE.
Numerical closeness of these means is a diagnostic observation about computation on this data set only; the
original F7 recovery remains **FAILED** and PR #8 remains open.

**Prospective repair (code only; nothing re-run).** `tests/quadrature_budget.py`: a cumulative ledger persisted
across processes and never reset; the remaining budget checked before every batch and batches never larger than
the remainder; completed and attempted counts kept distinct; wall-deadline checks; partial checkpoints; a stop
on an exhausted cap. `tests/test_quadrature_budget.py` (fake evaluator, no density evaluation; 5 tests) covers
each rule, including that a copy of the seeded ledger `F7_marginal_reference_ledger.json` (4,125,554 completed)
refuses the first batch. The driver and the tail script are refactored onto the controller, acceptance requires
every prespecified condition with missing ones reported explicitly, and `scripts/run_logged.py` records the real
subprocess return code. The refactored driver has not been executed.

**Scoped proposal (needs a new written authorisation; not executed).** Completing the reference as specified
needs about 3.0 million evaluations at the protocol's resolutions (the CDF-stability variants alone are
1.65 million), i.e. more than the original unit cap. A reduced completion of roughly 1.0 million evaluations
and under two minutes of wall time would: (a) extend the lower $\log sd_1$ face by two further curvature scales
and re-check all six faces (about 28,000 evaluations); (b) compute the CDF at all seven reference points for all
three scales with 81-point fine grids under a resolution variant (48 nodes) and a domain variant (40 nodes on
$B(10)$) separately (about 950,000 evaluations), reusing the saved attempt-2 marginals as the base. It would run
only through the ledger-based controller with a fresh cap and ledger named in the authorisation, with
`run_logged.py`, and would report every condition. Its outcome would bear on computation only.

### Second budget-guard review and repair (2026-09-13; code and bookkeeping only, nothing run)

The independent fake-only reproduction (`artifacts/2026-09-13-autobayes-budget-guard-independent/`) found that the
first controller did not reserve work before doing it (a batch that raised was refunded and a retry re-spent it),
reset its deadline per constructor, accepted empty, incomplete and NumPy-false condition dictionaries, and that the
driver shadowed the attempt name with an integer loop index and left two diagnostic marginal calls uncounted. All
five are repaired: reservations persisted before work and never refunded (requested / reserved-uncertain /
confirmed kept distinct; a crash leaves its reservation counted; a second live owner needs an explicit, recorded
takeover; stale writes are refused by version); one unit deadline carried in the ledger with measured batch time
accumulated across attempts, unconfirmed batches keeping a conservative estimate, expired retries refused at
construction, and a post-batch check that records an overrun bounded by one batch; fail-closed acceptance against
an explicit complete inventory (omitted, unchecked, false, NumPy false, malformed and unexpected entries all
block); the driver restructured with a guarded `main`, counted diagnostic calls, the full condition inventory
always present, and a stub-tested output assembly; the tail reporter reporting the ledger's cumulative counts.
Count convention: one evaluation per log-posterior point plus one per diagnostic call, so the historical spend is
4,125,558 (the raw outputs keep the old counter's 4,125,554). Status unchanged: the reference is
**incomplete/exploratory**, F7 **FAILED**, PR #8 open; the completion proposal above stands (its budget now includes
the two diagnostic charges) and is not executed.

### Third budget-guard review and repair (2026-09-13; code and bookkeeping only, nothing numerical run)

The fake-only reproduction (`artifacts/2026-09-13-autobayes-guard-v2-independent/`) found that external charges
bypassed the cap and deadline checks, that a batch which raised after 6 s recorded 0 s so a retry ran under a 5 s
unit, and that a stopped, over-budget unit with all scientific conditions met was still accepted. Repaired: one
central gate for every reservation (batches and diagnostic charges) with validated confirmations; measured failure
time recorded, whole-attempt wall time (driver overhead included) counted, unknown crash duration charged as the
allowance remaining at the attempt's last write, expired units refused at construction, post-batch check with a
recorded one-batch overrun; acceptance and the exit code require run validity (not stopped, no unconfirmed
reservations, within cap and allowance) as well as every scientific condition, with a valid exact-cap completion
still passing; an OS-level exclusive lock rejects simultaneous owners (sole-owner controller; no concurrency is
permitted). Reporting correction: of the 36 targeted tests cited on 2026-09-13 only 15 were fake/stub; the other 21
performed real deterministic density and gradient calculations (not posterior quadrature under the cap) and are
reported separately, not repeated. This repair's local checks were the two allowed fake/stub modules only, run with
`--noconftest`: 18 passed. Status unchanged: reference incomplete/exploratory, F7 FAILED, PR #8 open, completion
proposal not executed.


## Restart identity and accounting closure (2026-09-13)

Independent mock checks found that reusing an attempt name could reuse its stored maximum duration instead of adding retry time, and that malformed or missing budget fields could pass output validation. Reject previously used attempt names within the same unit ledger, keep closed-attempt time frozen, require complete finite and consistent run accounting, and count external diagnostic reservations in requested work. Confirmation sizes and constructor limits are validated without numeric coercion of fractional counts. Regression checks use only fake clocks, fake evaluators and stub stage records. No real posterior evaluation, quadrature or sampling is part of this correction. Validation is recorded separately; F7 and the numerical reference remain unaccepted.


## Direct truncated-box CDF implementation (2026-09-13; no F7 execution)

The original reference remains inconclusive and the original F7 recovery remains FAILED. This is a distinct proposed diagnostic protocol, not a rerun or relabeling of either result. The old evidence and exhausted ledger are unchanged.

The earlier roughly 1M completion suggestion confounded marginal-grid resolution and domain changes. A uniform 64/96-node direct-CDF design was then checked against a separable-normal toy before F7 execution; on the wider domain its log-normalizer resolution difference was 0.0005993, exceeding the proposed 0.0001 diagnostic. The revised v3 rules allocate more nodes to the long log-slope-SD axis: [48,96,48] and [64,128,64]. On that analytic toy, maximum log-normalizer/CDF errors were below 2e-8 and mean errors below 3e-9. Toy success is not F7 accuracy evidence.

`docs/m3b1-evidence/direct_cdf_v3_protocol.json` fixes the unchanged full Gaussian posterior/source identities, two nested domains, seven CDF points, both rules, face checks and proposed limits. A full-box integral supplies log normalizer and three natural-scale means; each CDF numerator integrates the same density directly over a separately mapped truncated box. All four crossed domain/resolution comparisons are reported. The planned 12,038,156 evaluations include 110,604 face evaluations. The proposed new cap is 12,050,000 points and 720 seconds, maximum batch 32,768, no retries. This record does not authorize that numerical run or reset the old unit.

`tests/reference_direct_cdf.py` implements the generic injected-evaluator path. It creates bounded tensor batches, uses log-space weights and moments, checks every planned stage and its source/configuration identity, preserves partial results, and uses the existing persistent budget controller. Face grids include the projected saved mode, with every point counted. Missing, nonfinite, wrong-count, mismatched-configuration or unexpected stages block acceptance. Invalid CDF ratios are rejected without clipping. Run validity uses the final wall time including output assembly. There is no F7 import, target-selection logic, CLI, new sampler, retry loop or automatic comparison against stored MCMC output.

Finite face samples and agreement across rules/domains are empirical diagnostics, not certified bounds on omitted posterior mass or total numerical error. Passing them cannot establish sampler convergence, fixed-truth coverage, SBC, or permission to accept the original F7. The generic runner leaves the MCMC comparison empty for a separately reviewed step. A specific F7 binding, reviewed unit cap and output/checkpoint plan remain necessary before a real run.

Validation: 23 new analytic/stub tests plus 39 existing guard/assembly regressions, **62 passed in 0.84 seconds**, exit 0. Target imports were explicitly prohibited by the verification launcher; zero attempts to import the F7 target/model/sampler modules and zero F7 evaluations. Tests cover independent exact integrals, a nonseparable polynomial CDF, shifted-box natural-scale means, large log offsets, face projection, a small tensor run with exact accounting, cap/interruption handling, final assembly time, configuration mismatch and incomplete-result rejection. The small tensor Gaussian fixture intentionally fails its tail diagnostic even while its CDFs are accurate; the result remains inconclusive. The preliminary fixture used too few nodes for its narrower toy, was corrected before integration, and no F7 threshold or data changed. Evidence: `docs/m3b1-evidence/direct_cdf_v3_checks.json`.

PR8 remains open for review. Previous guard-only hosted checks on 6b7ebbe passed; hosted checks on this implementation must be verified separately after push. No F7 quadrature, sampling, GPU, cluster or data work occurred in this implementation.
