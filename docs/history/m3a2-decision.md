# M3a-2 decision record: one numeric-by-categorical fixed interaction

Date: 2026-09-10. Branch `m3a2/numeric-by-categorical-interaction` from `main` at `df8195a`
(M3a-1 merged), worktree `projects/AutoBayes-worktrees/m3a2-interaction`. Written and committed
before any implementation. Source: Andrew's approval of the CPU roadmap (2026-09-09), section
4.1 PR 2 of `docs/coordination/NEXT_MILESTONES_PROPOSAL.md`, the coordinator's M3a-2 dispatch
(`memory/dispatches/2026-09-10-autobayes-m3a2-interaction.txt`). Scope: one interaction between
a numeric term and a declared categorical term, with both main effects, for the three existing
families; no categorical-by-categorical or higher-order interactions, transformed terms, other
contrasts, varying slopes, new families/links or dependency changes.

## Target

$$y \sim x + z + x{:}z + (1 \mid g), \qquad
\eta_i = a + b\,x_i + \sum_{\ell \ne r} \beta_\ell \mathbb 1[z_i = \ell] + \sum_{\ell \ne r} \delta_\ell\, x_i \mathbb 1[z_i = \ell] + u[g_i].$$

The reference level $r$ has intercept $a$ and slope $b$; every other level $\ell$ has the
intercept difference $\beta_\ell$ (`z[l]`) and the slope difference $\delta_\ell$ (`x:z[l]`), so its
slope is $b + \delta_\ell$. Family, link, likelihood, response validation, count domain,
categorical label identity (M3a-1 repair contract) and group structure are unchanged.

## Measured before deciding (formulae 0.6.2 / Bambi 0.17.2, scratch probe)

- `y ~ x + z + x:z + (1|g)` gives common terms `Intercept, x, z, x:z`; `x:z` has kind
  `interaction`, levels = the non-reference levels, labels `x:z[l]`, columns $x \cdot \mathbb 1[z=l]$;
  PyMC variable `x:z` with dims `x:z_dim` and the levels as coordinates.
- `y ~ x*z + (1|g)` expands to exactly the same four terms in the same order with an identical
  design matrix; `z + x + z:x` yields the term `z:x` with labels `z[l]:x` (name follows the
  written order), so a canonical spelling is needed.
- Lower-order-omitting forms change the model: `x + x:z` drops the level intercepts; `z + x:z`
  and `x:z` alone make formulae code the interaction with all levels (`x:z[b], x:z[a], x:z[c]`).
- With `center_predictors=True` Bambi centres every expanded column at its own mean, product
  columns included (hand check 0.0); the product column mean differs from the product of the
  means when $x$ differs by level (measured 0.069 vs 0.194).
- Per-level array priors on the interaction term match SciPy; Bambi's `predict` on reordered
  `newdata` equals the hand computation from the raw posterior.

## Accepted defaults (public contract)

1. **Grammar.** A term `x:z` (or `z:x`) with exactly one numeric side and one side declared in
   `categorical=`; `x*z` (or `z*x`) is shorthand for `x + z + x:z`, admitted because its expansion
   was measured identical. Deterministic **canonical expansion**: every `a*b` is replaced by
   `a`, `b`, `a:b` at its position; then the canonical right-hand side lists the main-effect
   terms in order of first appearance, followed by the interaction terms in order of first
   appearance, each spelled `numeric:categorical`. The canonical spelling is what reaches
   formulae/Bambi, what `describe()` prints and what every name below uses.
2. **Duplicates and overlaps.** A term that appears more than once after expansion (`x + x*z`,
   `x:z + z:x`) is refused with a message naming the expansion; `x:x`, two numeric or two
   categorical sides, the grouping factor, the response, the trials column, or three-way
   interactions are refused as unsupported in this slice.
3. **Both main effects required.** An interaction whose numeric or categorical main effect is
   absent (`x + x:z`, `z + x:z`, `x:z` alone) is refused with the explanation that omitting a
   main effect is a different model (the level intercepts or the reference slope are forced to
   zero, and formulae changes the interaction's coding) and is not part of this slice.
4. **Naming map** (numeric `x`, categorical `z` with sorted levels, reference $r$, non-reference
   levels $m_1..m_{K-1}$):

   | quantity | name |
   |---|---|
   | canonical term | `x:z` |
   | design columns (Bambi/formulae labels, after the main effects) | `x:z[m_1]`, …, `x:z[m_{K-1}]` |
   | PyMC variable (sampled) | `x:z`, shape $(K-1,)$, dims `x:z_dim`, coordinates `[m_1, …]` |
   | posterior / ArviZ / fit-record rows | `x:z[m_1]`, … |
   | `describe()` | `b_x:z[m_j]` (slope difference of level $m_j$ from the reference) |
   | `Priors.b` key | `"x:z"` exactly (the canonical spelling; `"z:x"` is refused with the hint) |
   | `model.coefficients` | `("x:z[m_j]", "x:z", j-1)` after the main-effect entries |
5. **Priors.** `b` is one prior for every coefficient, or a mapping over the canonical terms
   covering them exactly (`{"x": prior, "z": prior_or_per_level, "x:z": prior_or_per_level}`);
   a per-level mapping covers exactly the non-reference levels with the M3a-1 key rules (string
   labels, integer/decimal equivalence, colliding keys refused, one distribution family per
   term, location priors only). Nothing is filled in.
6. **Centring.** Raw columns are built first: $x$, $\mathbb 1[z=\ell]$, $x\,\mathbb 1[z=\ell]$. With
   `center_predictors=True` each is centred at its own fitted-row mean: $\bar x$, $p_\ell$, and
   $q_\ell = \overline{x\,\mathbb 1[z=\ell]} = p_\ell\, \bar x_{(\ell)}$ where $\bar x_{(\ell)}$ is the
   mean of $x$ within level $\ell$; in general $q_\ell \ne \bar x\, p_\ell$. The sampled intercept is
   $$a_c = a + \bar x\, b + \sum_{\ell \ne r} p_\ell \beta_\ell + \sum_{\ell \ne r} q_\ell \delta_\ell,$$
   the linear predictor at the point $(\bar x, p_\ell, q_\ell)$ of the expanded design, which is
   not a realisable covariate row (a row has $x\,\mathbb 1[z=\ell] = x$ or $0$). The reported
   `Intercept` is $a = a_c - \bar x b - \sum p_\ell \beta_\ell - \sum q_\ell \delta_\ell$ (the raw
   intercept: $x = 0$, reference level, group effect zero); verified numerically against Bambi's
   back-transform. For the logit families $\mathrm{expit}(a_c)$ is not an average probability.
   Binomial zero-trial rows are dropped before any of these means and the retained levels are
   determined.
7. **Predictions.** Every level uses its own intercept and slope; `newdata` rows may be mixed
   and reordered; unseen category levels are refused for every kind and `new_groups`; the
   unseen-group policies are M1's; binomial `probability`/`mean`/`response` semantics unchanged.

## Change of reference with the interaction

Level effects $\mu_\ell = a + \beta_\ell$, slopes $s_\ell = b + \delta_\ell$ ($\beta_r = \delta_r = 0$).
New reference $r'$: $a' = a + \beta_{r'}$, $b' = b + \delta_{r'}$, $\beta'_\ell = \beta_\ell - \beta_{r'}$,
$\delta'_\ell = \delta_\ell - \delta_{r'}$ for $\ell \notin \{r, r'\}$, $\beta'_r = -\beta_{r'}$,
$\delta'_r = -\delta_{r'}$: a linear map $M$ on $(a, b, \beta, \delta)$ with $|\det M| = 1$ (computed in
the test). The centred intercept is $a_c = \sum_\ell p_\ell \mu_\ell + \sum_\ell q_\ell s_\ell$ (sum over
all levels, with the actual expanded-column means), a property of the level effects, hence
invariant: $a'_c = a_c$; the sampled slope is not ($b' = b + \delta_{r'}$). Design invariance
(equal linear predictor and likelihood at corresponding points) holds. Independent priors in
the new coding are a different model: the induced joint prior has $\mathrm{Cov}(\beta'_\ell,
\beta'_{\ell'}) = +s^2_{\beta_{r'}}$, $\mathrm{Cov}(\delta'_\ell, \delta'_{\ell'}) = +s^2_{\delta_{r'}}$,
$\mathrm{Cov}(b', \delta'_\ell) = -s^2_{\delta_{r'}}$ and, uncentred, $\mathrm{Cov}(a', \beta'_\ell) =
-s^2_{\beta_{r'}}$. The posterior is invariant only under the transformed joint prior (unit
Jacobian); the test builds it by hand, no public multivariate-prior API.

## Acceptance (tests on the public API, clean locked environment, fresh evidence directories)

- D1 Grammar: `x + z + x:z`, `x*z`, `z*x`, `z:x` and shuffled orders give the same canonical
  terms, columns and design; duplicates/overlaps, lower-order-omitting forms and unsupported
  interactions refused with the stated messages; the M0-M3a-1 refusals unchanged.
- D2 Design against an independent construction from the supplied data (raw and centred
  columns, product-column means $\ne$ product of means with an unequal within-level $x$-mean
  design, nonalphabetical reference, unequal proportions, shuffled data order) for the three
  families; Bambi labels and PyMC dims as in the map; the reported-intercept transformation
  checked against Bambi's back-transform.
- D3 Normalised density with and without the Jacobian and gradients (PyMC, JAX, central
  differences of the independent reference extended to product columns) for Gaussian,
  Bernoulli and binomial models, term-wide and per-level priors, both centrings.
- D4 Reference change with the interaction: map and $|\det M| = 1$; linear predictor and
  likelihood equal; log posteriors differ under independent priors and are equal under the
  transformed joint prior, centred and uncentred, with $a_c$ invariance verified from the
  actual column means.
- D5 Prior forms and refusals of default 5, including the `"z:x"` key.
- D6 `describe()` (reference intercept and slope, differences, expanded-column centring and the
  unrealisable-point statement, the non-average warning for logit families, priors, names);
  fit record with the interaction specification.
- D7 Predictions computed by hand from the posterior draws (raw intercept, slope, level
  differences, group effect; not through any library design helper): in-sample; mixed/reordered
  `newdata`; reference and non-reference slopes; binomial expected versus predictive counts;
  unseen category refused for every kind and `new_groups`; unseen group policies unchanged.
- D8 Binomial: levels, proportions and product-column means computed after the zero-trial
  drop; a level only in dropped rows reported and refused in prediction.
- D9 Every existing test passes, including the label-identity and M2 normalisation regressions.
- D10 Recovery (slow, both CPU samplers), prespecified: Gaussian, `y ~ x + z + x:z + (1 | g)`,
  360 rows, J = 12 groups, K = 3 levels with proportions 0.5/0.3/0.2, $x \sim N(m_\ell, 1)$ with
  $m_a = 0$, $m_b = 1$, $m_c = -0.5$ (different within-level means), seed 13; every level in every
  group and $x$ varying within every level and group checked before fitting; reference `"a"`;
  truth $a = 1.0$, $b = 0.5$, $\beta_b = -0.8$, $\beta_c = 0.6$, $\delta_b = 0.4$, $\delta_c = -0.7$,
  sd $= 0.5$, sigma $= 0.4$; priors Normal(0, 5) intercept, Normal(0, 2) coefficients,
  HalfNormal(1) sd and sigma; 4 chains, 1000 tune, 1000 draws, seed 13, `target_accept` 0.9.
  Gates: 0 divergences; max R-hat $\le 1.01$; min bulk ESS $\ge 400$ over the eight population
  parameters; each truth inside its 99 % HDI (nominal false-failure budget $\le 8$ % for eight
  parameters treated as independent, fixed before sampling); backends within 4 combined MCSE.
  One recovery is not calibration.
- D11 Example `examples/m3a2_interaction.py` on both backends, outputs in `docs/m3a2-evidence/`.
- D12 `value_point` over the design columns, `model.coefficients`, original labels.

## Kill criterion

If formulae's interaction coding differs from the map in any tested case (levels, labels,
order, centring), or a density or prediction check fails on either path, the slice stops and
reports rather than renaming or loosening a tolerance.

## Executed results (2026-09-10, clean locked environment, Python 3.11.0, fresh evidence directories)

- Measured first (scratch probe, recorded above); the kill criterion was not triggered: every
  tested coding, label, order and centring matched the map.
- `tests/test_interaction.py`: 35 passed (37 s). Fast suite: 332 passed, 7 deselected (288 s). Slow suite: 7 passed (49 s).
- D10 recovery: both CPU samplers, 4 x (1000 + 1000), seed 13: PyMC 0 divergences, max R-hat 1.0094889, min bulk ESS 520; NumPyro 0 divergences, max R-hat 1.0045411, min bulk ESS 662; all eight truths (raw intercept 1.0, x 0.5, z[b] -0.8, z[c] 0.6, x:z[b] 0.4, x:z[c] -0.7, sd 0.5, sigma 0.4) inside their 99 % HDIs on both backends; backends within 4 combined MCSE on every population mean (largest ratio 0.41). Raw results in `D10_interaction_recovery.json`, `D10_interaction_recovery_records.json` and `D10_interaction_agreement.json` of the slow run's evidence directory.
- Example (`docs/m3a2-evidence/`): both backends recover the arm-specific slopes (control 0.54, low 0.75, high -0.42 for truths 0.5, 0.8, -0.3) and the arm contrasts; PyMC and NumPyro agree within 4 combined MCSE on all eight population means (largest ratio 0.26); the unfitted arm 'placebo' is refused; `--device gpu` raises `DeviceError` (captured). Files: `example_pymc_cpu.txt`, `example_numpyro_cpu.txt` (with the comparison), the two fit records, `example_numpyro_gpu_attempt.txt`, `independent_regressions.txt`.
- Evidence directories of these runs: `evidence-runs/20260910T213233.367265Z-f0d229f12697-dirty-p57280-fc4e8d/` (fast, final) and `evidence-runs/20260910T212717.645617Z-f0d229f12697-dirty-p56976-4fbce1/` (slow; same library code, run before three refusal-message expectation edits in `tests/test_formula.py` and `tests/test_categorical.py`, whose earlier fast run `...-p56367-77a663/` had those three failures); earlier run directories untouched.

## Repair record (2026-09-10): the one-interaction scope was not enforced; the categorical contrast wording; original-source oracles

**Finding (independent review, `memory/reviews/2026-09-10-autobayes-m3a2-review.md`; extra checks
`artifacts/2026-09-10-autobayes-m3a2-independent/test_iqss_m3a2_independent.py`: 12 original-input
checks pass, 3 scope checks fail on `cc3081a`; reproduced here before any change,
`docs/m3a2-evidence/scope-repair/before-cc3081a.txt`, pytest exit 1, "3 failed, 12 passed").**
(1) `parse_formula` collected every distinct numeric-by-categorical interaction, so
`y ~ x + w + z + x:z + w:z + (1 | g)` built in all three families although this slice
advertises one interaction. (2) `describe()` kept the M3a-1 sentence "b_z[l] is the difference
between level l and the reference at the same predictors and group" even when `z` interacts with
`x`; with the interaction that difference is $\beta_\ell + x\,\delta_\ell$ and $\beta_\ell$ alone
is the contrast at $x = 0$ (with $\beta = -0.2$, $\delta = 0.5$ the contrast at $x = 2$ is $+0.8$).
(3) The density/finite-difference test passed `model.data` (the library's own fitted rows and
recoded labels) to the reference, so a preprocessing mistake could have validated itself.

**Repair (decided before coding).**

1. **Scope guard.** After canonical expansion and duplicate detection, a model with more than one
   distinct interaction is refused (`FormulaError` naming the interactions and this record),
   whether the interactions share the numeric side (`x:z + x:w`), the categorical side
   (`x:z + w:z`) or neither. No-interaction and single-interaction models, `x*z`, and every
   existing refusal are unchanged. The refusal is a scope statement, not a claim that such models
   are wrong.
2. **Conditional contrast in `describe()`.** For a categorical term that interacts with a numeric
   term, the explanation states that `b_z[l]` is the difference between level l and the reference
   **at x = 0** (same group), that at another x the difference is `b_z[l] + x * b_x:z[l]`, on the
   response scale (Gaussian) or the log-odds scale (Bernoulli, binomial); a categorical term without
   an interaction keeps the constant-contrast sentence. Skill and example wording made consistent.
3. **Original-source oracles.** The interaction density/gradient tests read the caller's original
   frame and select the positive-trial rows themselves for the binomial family; the coordinator's
   original-label checks (exact signed/unsigned large integer labels, explicit mathematical
   reference, synthetic-posterior predictions) are incorporated as a repository test module,
   together with the scope tests for shared numeric, shared categorical and disjoint sides in all
   three families.

**Acceptance.** E1 the coordinator's extra file passes unchanged (15 of 15); E2 the new module
passes; E3 every earlier suite, gate and independent regression passes unchanged in fresh evidence
directories, with the actual pytest exit codes and summaries recorded; E4 a nonzero-x numerical
contrast check confirms the stated conditional contrast.

**Executed results of the repair (2026-09-10, clean locked environment, fresh evidence directories;
real pytest exit codes recorded in `docs/m3a2-evidence/scope-repair/after-repair.txt`).** Repair commit
76beb7b. Before, on `cc3081a`: the coordinator's extra file gave pytest exit 1, "3 failed, 12 passed"
(`before-cc3081a.txt`). After: 15 passed (pytest exit 0); `tests/test_interaction_original_inputs.py` (the adapted
original-input, scope and conditional-contrast checks) 16 passed; `tests/test_interaction.py` with the
original-source oracle 35 passed; fast suite 348 passed, 7 deselected (343 s); slow suite 7 passed (49 s) (recovery gates unchanged);
independent label-identity and M2 normalisation regressions 12 passed. Evidence directories:
`evidence-runs/20260910T234927.734308Z-7f48b0864c77-dirty-p63348-86ea40/` (fast) and `evidence-runs/20260910T235131.008154Z-7f48b0864c77-dirty-p64883-5b4505/` (slow); earlier directories untouched. The frozen coordinator evidence under
`artifacts/2026-09-10-autobayes-m3a2-independent/` was not modified.
