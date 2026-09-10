# M3a-1 decision record: one treatment-coded categorical predictor

Date: 2026-09-10. Branch `m3a1/categorical-predictors` from `main` at `f8e116b` (M2 merged),
worktree `projects/AutoBayes-worktrees/m3a1-categorical`. Written and committed before any
implementation. Source: Andrew's approval of the CPU roadmap (2026-09-09,
`iqss-research/plans/2026-09-09-autobayes-approval-brief.md`) of section 4.1 of
`docs/coordination/NEXT_MILESTONES_PROPOSAL.md`; the coordinator's M3a-1 dispatch of
2026-09-10 (`memory/dispatches/2026-09-10-autobayes-m3a1-categorical.txt`). Scope: shared
formula support for the three existing families; no interactions, varying slopes, links,
families, contrasts other than treatment, ordered-factor effects, GPU tests or dependency
changes.

## Target

$$y \sim x + z + (1 \mid g)$$ with $x$ numeric (zero or more numeric terms as before) and $z$
one categorical fixed predictor with levels $\ell_1 < \dots < \ell_K$ and a declared
reference level $r$. Treatment (dummy) coding:

$$\eta_i = a + \sum_p x_{ip} b_p + \sum_{\ell \ne r} \mathbb 1[z_i = \ell]\, \beta_\ell + u[g_i]$$

$\beta_\ell$ is the difference, on the link scale, between level $\ell$ and the reference
$r$ at the same $x$ and group; the reference has no coefficient. The family, link,
likelihood, response validation, count domain and group structure are unchanged from M0-M2.

## What was measured before deciding (formulae 0.6.2, Bambi 0.17.2; `scratch probes`)

- formulae sorts the observed values of a string or **unordered** Categorical column and
  takes the first sorted level as the reference; Bambi converts object columns to
  (unordered) category first. An **ordered** pandas Categorical keeps its declared
  category order, with the first category as the reference, and keeps unused declared
  categories as all-zero dummy columns (measured: `levels ['a','c','unused']`, column sum
  0). `C(z, Treatment('b'))` also works but names the term, the PyMC variable and its
  dimension `C(z, Treatment('b'))`.
- With `center_predictors=True` Bambi centres every common column including the dummies,
  at their sample proportions (measured to 2e-16 against a hand computation; uncentred
  differs). The posterior `Intercept` Bambi reports is the raw-scale intercept.
- One Bambi `Prior` per term; array-valued hyperparameters give per-level priors
  (measured: Normal and StudentT with vector `mu`, `sigma`, `nu` match SciPy per level).
  A term cannot mix distribution families across levels.
- Bambi's `predict` refuses a level absent from the training data (`ValueError`), and
  correctly predicts reordered rows of observed levels (hand check 0.0).

## Accepted defaults (public form)

```python
model = ab.gaussian("y ~ x + z + (1 | g)", df, priors, categorical={"z": "a"})
```

1. **Declaration.** `categorical` maps a column name to its **reference level**; both are
   required. A declared column must be a plain term of the formula (not the response, the
   trials column, or the grouping factor). A non-numeric predictor column (string/object,
   pandas Categorical) that is not declared is refused with a message showing the
   declaration; an integer column is numeric unless declared, and a declared numeric column
   must be integer-valued (labels are the values' decimal strings); floats with fractional
   values, booleans (recode explicitly) and complex columns are refused as categorical.
2. **Levels and order.** The levels are the distinct values present in the fitted rows, as
   string labels, in Python `sorted` order; a pandas Categorical's own category order is
   not used (only its values), and its declared categories that do not occur in the fitted
   rows are **unused**: they get no column, are listed in `describe()`, `model.warnings`
   and the fit record, and predicting for one is refused as an unseen level. The
   coefficient vector is the sorted levels with the reference removed. A categorical
   predictor with a single observed level is refused as constant; missing values are
   refused as before.
3. **Reference validation.** The reference must be one of the observed levels of the
   fitted rows; otherwise the model is refused, listing the observed levels. For the
   binomial family the zero-trial rows are dropped **first** (as in M2), then levels,
   reference validity and dummy proportions are computed over the fitted rows; a level
   present only in dropped rows is reported with the drop and is not a level of the model.
4. **Priors.** `Priors.b` is either one prior (every numeric slope and every dummy
   coefficient) or a mapping over the formula's predictor terms covering them exactly. For
   a categorical term the entry is either one prior (term-wide, the same for every
   non-reference level) or a mapping `{level: prior}` covering exactly the non-reference
   levels; keys are matched by their string label. Refused: a missing term or level, an
   unknown key (including the reference level, which has no coefficient, and dummy-column
   labels such as `"z[b]"` at the term level), keys that collide after string conversion
   (`2` and `"2"`), a per-level mapping whose priors do not share one distribution family
   (one Bambi prior per term), and non-location priors for coefficients. Nothing is filled
   in silently.
5. **Naming map** (exact). User column `z`, sorted labels $\ell_1..\ell_K$, reference $r$,
   non-reference levels $m_1..m_{K-1}$ in sorted order:

   | quantity | name |
   |---|---|
   | design columns (Bambi/formulae labels, after `Intercept` and in formula term order) | `z[m_1]`, …, `z[m_{K-1}]` |
   | PyMC variable (sampled, value space) | `z`, shape $(K-1,)$, dims `z_dim` with coordinates `[m_1, …, m_{K-1}]` |
   | posterior / ArviZ / fit-record rows | `z[m_1]`, …, `z[m_{K-1}]` |
   | `describe()` | `b_z[m_j]`, and the table `reference r; levels …` |
   | `Priors.b` keys | term `"z"`, per-level keys `m_j` (string labels) |
   | `model.coefficients` | `("z[m_j]", "z", j-1)` after the numeric `("x", "x", None)` entries, in design-column order |
   | `value_point(b=...)` | one flat vector in design-column order |

   Implementation: the internal data frame's column is replaced by an **ordered** pandas
   Categorical with categories `[r, m_1, …, m_{K-1}]` built from the observed labels only,
   and the term is handed to formulae as the plain name `z`, so Bambi/formulae build the
   design and naming; the design matrix is then verified column by column against an
   independent one-hot construction, as the numeric design already is.
6. **Centred intercept (statement used by `describe()`, the skill and the README).** With
   `center_predictors=True` (default, as brms and Bambi) the sampled intercept $a_c$ is the
   linear predictor at the numeric means and at the **sample proportions of the levels**
   ($a_c = \sum_\ell p_\ell \mu_\ell + \sum_p \bar x_p b_p$ where $\mu_\ell$ is the level
   effect and $p_\ell$ its proportion among fitted rows), with the group effect zero. For
   the Gaussian family that is the expected response there. For the Bernoulli and binomial
   families it is a log-odds; $\mathrm{expit}(a_c)$ is the probability at that hypothetical
   proportion-weighted predictor and is **not** the average probability across the levels
   (the link is nonlinear). The prior on `intercept` applies to $a_c$; the priors on the
   dummy coefficients are on the link scale. The posterior `Intercept` is the raw intercept
   (numeric predictors at 0, $z$ at the reference, group effect zero).
7. **Predictions.** Every observed level predicts by its own coefficient; `newdata` rows may
   list levels in any order and mixture; a category level not among the fitted levels is
   refused (`FormulaError` naming the level and the fitted levels) for every `kind` and
   regardless of `new_groups`, which continues to govern only unseen levels of the grouping
   factor with M1's semantics. Categorical fixed-effect levels and grouping levels are
   separate namespaces.

## Reference-level change: what is and is not invariant

Level effects $\mu_\ell$ (link scale, numeric predictors and group fixed). Coding with
reference $r$: raw intercept $a = \mu_r$, $\beta_\ell = \mu_\ell - \mu_r$. Coding with reference
$r'$: $a' = a + \beta_{r'}$; $\beta'_\ell = \beta_\ell - \beta_{r'}$ for $\ell \notin \{r, r'\}$;
$\beta'_r = -\beta_{r'}$. The centred intercept is invariant: $a_c = a + \sum_{\ell \ne r} p_\ell
\beta_\ell = \sum_\ell p_\ell \mu_\ell = a'_c$. Hence on the sampled vector $\theta = (a_c, \beta)$ the
change of reference is a linear map $\theta' = M\theta$ with $M = \mathrm{diag}(1, M_\beta)$, and
in the uncentred parameterisation $\theta = (a, \beta)$ it couples the intercept through the
row $a' = a + \beta_{r'}$; $|\det M| = 1$ in both cases (computed in the test, not assumed).

- **Design invariance (holds).** The design spans are equal; at corresponding points
  $\theta' = M\theta$ the linear predictor, hence the likelihood of every family, is identical.
- **Model invariance (does not hold under independent priors).** If $\beta \sim N(m, S)$ with
  $S$ diagonal, the induced prior of $\beta'$ is $N(M_\beta m, M_\beta S M_\beta^\top)$ whose
  off-diagonal entries equal $-s_{r'}^2 \ne 0$ (and in the uncentred case the intercept is
  coupled with $\beta_{r'}$). A model written with independent priors in the new coding is
  therefore a different model with a different posterior, except in measure-zero cases.
- **Posterior invariance (holds only under the transformed joint prior).** With the entire
  joint prior transformed (multivariate normal with mean $M\mu$ and covariance $M \Sigma
  M^\top$, including the intercept coupling when uncentred) the posterior densities at
  corresponding points are equal, since the map is linear with unit Jacobian. No public
  multivariate-prior API is added; the test builds the transformed prior by hand.

## Acceptance (all in `tests/`, clean locked environment, unique evidence directories)

- C1 Declaration and validation as in defaults 1-3, including: undeclared string column
  refused with the declaration hint; undeclared integer column numeric (one numeric
  design column); declared integer column categorical with decimal labels; fractional,
  boolean and complex columns refused as categorical; declaring the response, trials,
  group or an absent column refused; absent reference refused listing levels; unused
  Categorical categories reported and not in the design; single-level categorical refused.
- C2 Design matrix equals an independently built one (one-hot minus the reference,
  sorted levels, centred at the proportions when centring; unequal proportions such as
  0.5/0.3/0.2), for all three families, for `x + z`, `z + x`, `z` alone, each reference
  in turn, and two categorical terms; Bambi labels and PyMC dims/coords as in the map.
- C3 Normalised model density with and without the Jacobian, and gradients (analytic and
  finite differences), against an independent reference extended to the dummy design,
  on both the PyTensor and JAX paths, for Gaussian, Bernoulli and binomial models, with a
  per-level prior in at least one case and both centring settings. Tolerance 1e-8 on
  values as in M0-M2, gradient tolerance as in the existing tests.
- C4 Reference change: the map $M$ is built by the test from the derivation, $|\det M| = 1$
  checked numerically; linear predictors and likelihoods equal at corresponding points
  (centred and uncentred; Gaussian and Bernoulli); log posterior densities differ under
  independent priors in the two codings; are equal under the transformed joint prior
  (hand multivariate normal, intercept coupling included when uncentred).
- C5 Prior forms and refusals of default 4; per-level priors verified through C3.
- C6 `describe()` states the reference, the contrast meaning, the proportion-centred
  intercept on the link scale (with the non-average warning for logit families), the
  priors and the naming table; `fit_record` carries the categorical specification, the
  levels, the reference, unused levels and per-coefficient priors.
- C7 Predictions: in-sample and `newdata` per level equal a hand computation from the
  posterior draws (raw intercept + coefficient + group effect) for all three families and
  kinds; mixed and reordered rows; unseen category level refused for every `kind` and every
  `new_groups`; an unseen grouping level with `new_groups="population"` still works with a
  categorical predictor present; a label that is a group level but not a category level
  is an unseen category.
- C8 Binomial: levels, reference validity and proportions computed after the zero-trial
  drop; a level only in dropped rows is reported and refused in prediction; the M2 count
  and prediction regressions unchanged.
- C9 Every existing test passes unchanged (fast and slow suites).
- C10 Recovery (slow, both CPU samplers), prespecified: Gaussian, `y ~ x + z + (1 | g)`,
  300 rows, J = 10 groups all present, K = 3 levels with proportions 0.5/0.3/0.2 drawn
  with seed 11 and every level present in every group checked before fitting, reference
  `"a"`, truth $a = 1.0$ (raw), $b_x = 0.5$, $\beta_b = -0.8$, $\beta_c = 0.6$, sd = 0.5,
  sigma = 0.4; priors Normal(0, 5) intercept, Normal(0, 2) coefficients, HalfNormal(1)
  sd, HalfNormal(1) sigma; 4 chains, 1000 tune, 1000 draws, seed 11, `target_accept` 0.9.
  Gates: 0 divergences; max R-hat ≤ 1.01; min bulk ESS ≥ 400 for the six population
  parameters; each truth inside its 99 % HDI (nominal false-failure budget ≤ 6 % for six
  parameters treated as independent, recorded, not tuned after seeing the result); the
  two backends agree within 4 combined MCSE on every population mean. One recovery is
  not calibration.
- C11 Example `examples/m3a1_categorical.py` runs on both backends; its outputs and fit
  records are committed under `docs/m3a1-evidence/`.
- C12 `value_point` accepts one flat coefficient vector in design-column order and
  reproduces PyMC's value variables; `model.coefficients` matches the map.

Documentation carried with this slice: `docs/m2-decision.md` said $\log\binom{n}{k}$ is
largest on the grid at $k = n/4$; its maximum is at $k = n/2$, and the measured error was
worst at $n/4$, a different statement. Corrected in prose without touching the evidence
files. `describe()` for the binomial family still said the coefficient is "computed
accurately as data"; it is computed inside the graph since the M2 correction.

## Kill criterion

If Bambi's design or naming for an ordered Categorical differs from the map above in any
tested case, or the density check fails on either path, the slice stops and reports rather
than papering over it with a rename or a looser tolerance.
