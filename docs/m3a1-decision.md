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
  off-diagonal entries equal $+s_{r'}^2 \ne 0$ (every $\beta'_\ell$ contains $-\beta_{r'}$; the
  sign was written as negative in the pre-code version of this record and corrected when
  the test computed the covariance), and in the uncentred case the intercept is coupled with
  $\beta_{r'}$ through $\mathrm{Cov}(a', \beta'_\ell) = -s_{r'}^2$. A model written with independent
  priors in the new coding is therefore a different model with a different posterior, except
  in measure-zero cases.
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

## Executed results (2026-09-10, clean locked environment, Python 3.11.0, macOS arm64)

- Measured first (scratch probes, recorded above): formulae sorts string and unordered
  Categorical levels; an ordered Categorical keeps its order with the first category as the
  reference and keeps unused categories as zero columns; Bambi centres dummies at their
  proportions (2e-16); per-level array priors match SciPy; unseen levels are refused by
  Bambi's `predict`. The kill criterion was not triggered: every tested design and name
  matched the map.
- `tests/test_categorical.py`: 78 passed (33 s). Fast suite: 261 passed, 6 deselected (272 s). Slow suite: 6 passed (43 s).
- C10 recovery: both CPU samplers, 4 x (1000 + 1000), seed 11: PyMC 0 divergences, max R-hat 1.0066, min bulk ESS 669; NumPyro 0 divergences, max R-hat 1.0099851 (below the 1.01 gate), min bulk ESS 761; every truth (raw intercept 1.0, x 0.5, z[b] -0.8, z[c] 0.6, sd 0.5, sigma 0.4) inside its 99 % HDI on both backends (the intercept posterior mean 0.84 sits where the realised mean of the ten group effects puts it); backends within 4 combined MCSE on every population mean (largest ratio 0.43). Raw results in `C10_categorical_recovery.json`, `C10_categorical_recovery_records.json` and `C10_categorical_agreement.json` of the slow run's evidence directory.
- Example (`docs/m3a1-evidence/`): both backends recover the arm contrasts (treatment[high] 0.92, treatment[low] -0.63 for truths 0.9 and -0.6); PyMC and NumPyro agree within 4 combined MCSE on all six population means (largest ratio 0.39); the unfitted arm 'placebo' is refused; `--device gpu` raises `DeviceError` on this machine (captured). Files: `example_pymc_cpu.txt`, `example_numpyro_cpu.txt` (with the comparison), the two fit records, `example_numpyro_gpu_attempt.txt`.
- Corrections to the pre-code text of this record made during implementation: the sign of the
  induced covariance under a change of reference (now $+s_{r'}^2$ between dummies, $-s_{r'}^2$
  intercept coupling when uncentred); nothing else in the accepted defaults changed.
- Evidence directories of these runs: `evidence-runs/20260910T201758.793519Z-f134f6681885-dirty-p46904-eeda91/` (fast, final) and `evidence-runs/20260910T201242.053154Z-f134f6681885-dirty-p46734-a6732f/` (slow; run on the same library code before the one test-expectation edit in `tests/test_formula.py`, whose earlier fast run `...-p46534-dd7ec8/` had that single failure); earlier run directories untouched.

## Repair record (2026-09-10): categorical labels lost their identity in a float round trip

**Finding (independent review, `memory/reviews/2026-09-10-autobayes-m3a1-review.md`; regressions
`artifacts/2026-09-10-autobayes-m3a1-independent/test_label_integrity.py`, nine failures on
`393e4ec`; reproduced here before any change, `docs/m3a1-evidence/label-repair/before-393e4ec.txt`).**
`categorical_labels` converted every numeric categorical column through `float64` and then
`int64`, so integer codes above $2^{53}$ were merged (`[2^{53}, 2^{53}+1, 2^{53}+2]` became two
labels) and unsigned codes at or above $2^{63}$ were corrupted with only a RuntimeWarning; a valid
public `ab.gaussian` call with three exact integer levels built a two-level model. The same
converter served `newdata`, so prediction identity depended on the dtype. Separately, `_label`
fell through to `str(v)` for complex scalars and `reject_complex` looked only at the Series
dtype, so complex values inside `object` or `category` containers bypassed the documented
refusal. A categorical code is an identifier: any conversion that can change it changes the
statistical model before either backend sees it.

**Validation contract (decided before coding).** A categorical value is normalised scalar by
scalar, by one function used for data values, `newdata`, references, a pandas Categorical's
declared categories and per-level prior keys, with no float intermediate and no narrowing cast:

| scalar | label | rule |
|---|---|---|
| Python `int`, any NumPy integer (signed or unsigned, any width), including nullable `Int64`/`UInt64` containers | exact decimal string of the integer | `str(int(v))`, exact for every width; no range limit |
| `float` / NumPy float | decimal string of the integer it equals | accepted only if finite, integral and $\lvert v \rvert \le 2^{53}$ (the range in which float64 represents every integer); larger or fractional floats are refused, not relabelled |
| `str` (including `numpy.str_`) | itself | no normalisation |
| `bool` / `numpy.bool_` | refused | recode explicitly |
| complex, `bytes`, `None`/`NaN`/`pd.NA`, and any other type | refused | missing values are refused as before; complex is refused at the scalar level in every container |

Containers are read with `astype(object)`, which yields exact Python integers for `int64`,
`uint64`, `Int64`, `UInt64` and object columns and the category scalars of a Categorical
(measured). **Intentional equivalence:** an integer and its decimal string denote the same label
(`categorical={"z": 1}` and `{"z": "1"}` name the same level; an integral float `2.0` names
`"2"`). **Refused as ambiguous:** a column that mixes text labels with numeric codes (the
equivalence would silently merge `"1"` and `1`); a prior mapping whose keys collide after
normalisation (`2` and `"2"`); a complex, boolean or other unsupported reference, declared
category or prior key. Nothing outside this domain is accepted and relabelled.

**Original-input checks.** The model keeps the labels of the *supplied* column (before the
internal ordered Categorical is built) and verifies Bambi's dummy columns against indicators of
those original labels; the tests build the expected design directly from the raw values with
exact integer comparison, and check predictions from `newdata` given as `int64`, `uint64`,
nullable, object and string containers of the same codes.

**Acceptance (all in `tests/test_label_integrity.py` plus the coordinator's file run unchanged).**

- R1 The nine independent regressions pass unchanged.
- R2 Exact labels for `[2^{53}, 2^{53}+1, 2^{53}+2]` and `[0, 2^{63}, 2^{64}-1]` in `int64`,
  `Int64`, `uint64`, `UInt64`, object (Python ints) and pandas Categorical containers; a public
  Gaussian model on the three large codes has three levels and its `X_raw` equals the one-hot of
  the raw values; the reference given as an int, as a string and as a NumPy integer selects the
  same level.
- R3 Integral floats: `2.0` and `float(2^{53})` accepted with the integer label; `2^{53}+2` as a
  float, `1e300`, `inf`, `nan` and fractional values refused with the domain in the message.
- R4 Complex scalars refused in float-complex, object and category containers, and as a
  reference, a declared category and a prior key; booleans refused in bool, object and category
  containers and as a reference or key; a column mixing strings and numbers refused as ambiguous;
  missing values (`None`, `NaN`, `pd.NA`) refused in every container.
- R5 Predictions: `newdata` codes in each container map to the same levels and give identical
  predictions; `2^{53}+1` is distinct from `2^{53}` in `newdata`; an unseen large code is refused.
- R6 The reference implementation used by the density tests applies the same scalar rule
  (no float intermediate), and the existing categorical, prediction and numerical regressions
  pass unchanged; suites re-run in fresh evidence directories.

The failed head `393e4ec` stays in the history as the reviewed candidate; this record and the
repair commits follow it.
