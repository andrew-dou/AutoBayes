# M3c-1 decision record: population-only models (no group-level term)

Date: 2026-09-19. Branch `m3c1/population-only-models` from `origin/main` at `95dce8c` (M3a-2 merged, v0.4.0),
worktree `projects/AutoBayes-worktrees/m3c1-population-only` with its own locked environment
(`uv sync --locked --extra jax --extra dev --python 3.11`; imports resolve into this checkout). Written and
committed before any implementation. Source: Andrew's September 19 foundation-first direction
(`plans/2026-09-19-autobayes-foundations.md`, sequence step 3, first increment) and the coordinator's
dispatch of 2026-09-19. Blueprint: brms `brmsformula`
(https://paulbuerkner.com/brms/reference/brmsformula.html), where group-level terms are optional and a
formula with only population-level terms is the base case. Scope: formulas without a `(1 | group)` term
for the three existing families with the existing predictor vocabulary (numeric terms, declared
categorical terms, one numeric-by-categorical interaction). Not in scope: intercept removal (`0 +`,
`- 1`), other families or links, several or correlated group terms, varying slopes (PR #8, unaccepted,
not cherry-picked), dependency changes, sampling runs.

## Target

$$y \sim x_1 + \dots + x_p \;[+\, z + x{:}z], \qquad \eta_i = a + \sum_p b_p x_{ip} \;[+ \text{level and slope differences}],$$

with the same families, links, likelihoods, response validation, count domain, categorical label
contract, canonical expansion and centring as M0 to M3a-2, and no group effect. `y ~ 1` is the
intercept-only model. Centring stays on: the sampled intercept is the linear predictor at the column
means of the expanded design and the reported `Intercept` is the raw-scale intercept, exactly as for
grouped models.

## Measured before deciding (formulae 0.6.2 / Bambi 0.17.2, build-only scratch probe, no sampling)

- `y ~ x`, `y ~ 1`, `y ~ x + w`, `y ~ x + z`, `y ~ x + z + x:z` (Gaussian), `y01 ~ x` (the custom
  Bernoulli-logit family) and `p(k, n) ~ x` (the custom binomial-logit family) build with common terms
  `Intercept, ...` in formula order, no group design (`components[...].design.group is None`), the
  first common column all ones, PyMC value variables `Intercept`, one per term (vector terms with
  `z_dim` / `x:z_dim` coordinates), and `sigma_log__` for the Gaussian family only; deterministic `mu`
  or `logit_p`; no `1|g_sigma`, `1|g_offset`, `1|g`.
- A prior dict entry `1|g` for a formula without that term is **accepted silently** by Bambi (the
  model has no such variable). This layer therefore refuses a group SD prior when there is no
  group-level term.
- `Model.predict(idata, data=newdata)` with a data frame lacking any grouping column returns
  `mu = a + b x` per draw (checked against a synthetic posterior); the `response` kind returns `y`.

## Accepted defaults (public contract)

1. **Grammar.** The varying-intercept term becomes optional: `y ~ 1`, `y ~ x`, `y ~ 1 + x`,
   `y ~ x + w`, `y ~ x + z`, `y ~ x + z + x:z` (and `x*z`), `k | trials(n) ~ x`, with `categorical=`
   as before. Unchanged refusals, same messages: `0 +` and `- 1`, more than one grouping factor,
   varying slopes, nested or crossed factors, `||`, function calls, duplicates, lower-order-omitting
   interactions, `y ~ (1 | g)` (an intercept-only grouped model is written `y ~ 1 + (1 | g)`), an empty
   or malformed right-hand side. The canonical right-hand side of a population-only formula lists the
   main effects then the interactions; an intercept-only formula is canonically `y ~ 1`.
2. **Specification.** `ModelSpec.group` is `None` and `levels == ()` for a population-only model
   (`has_group` says which); the group column is not required in the data and no level check runs.
3. **Priors.** `Priors.sd` becomes optional. A grouped formula without `sd` is refused at model build
   with `PriorError` ("needs a prior on the SD of its varying intercept"), instead of the previous
   `TypeError` at construction; a population-only formula with `sd` is refused with `PriorError`
   ("has no group-level term; do not pass sd"). Nothing is filled in or ignored silently. Every other
   prior rule (location and scale classes, `b` coverage per term and per level, `sigma` exactly for
   the Gaussian family) is unchanged.
4. **Model and graph.** Built through Bambi with the same families, `center_predictors` and the
   same design verification column by column; `Design.Z` is `(n, 0)`, `group_index` is empty,
   `levels == ()`, `J == 0`. The PyMC value variables are `Intercept`, one per term, and
   `sigma_log__` (Gaussian). `model.names` has no `sd`, `offset` or `group_effect` entry. `value_point`
   takes `sd=None, z=None`: both must be omitted for a population-only model and both are required for
   a grouped model, each refusal naming the reason. `describe()` states "no group-level term
   (population-level model)", omits the levels line, the group-effects block and the `sd` prior, and
   prints the linear predictor without `u[g_i]`.
5. **Predictions.** In-sample and `newdata` predictions need no grouping column; `new_groups` must be
   left at its default `"error"` for a population-only model, and `"population"` or
   `"resample_observed"` raise `ValueError` ("has no group-level term; new_groups does not apply").
   Result attributes carry `new_groups = "not_applicable"` and `new_levels = []`. Unseen categorical
   levels are refused as before. The three prediction kinds and the binomial quantities are unchanged.
6. **Records.** The fit record carries `group: null`, `J: 0`, `noncentered: null` (no group block to
   parameterise) and priors without `sd`; the posterior summary lists the intercept, the terms and
   `sigma` (Gaussian). `compare_records` is unchanged.
7. **Execution.** Both routes (`backend="pymc"`, `backend="numpyro"` on CPU) consume the same graph;
   nothing here claims a GPU run or standalone code export. No sampler fit is executed in this slice.
8. **Version.** `0.4.0` -> `0.5.0` (`pyproject.toml`, `autobayes.__version__`, the record default).
   PR #8 (unaccepted) also names `0.5.0` on its own branch; it will need a rebase and renumbering
   whichever lands second. Grouped-model behaviour, names, records and tests are preserved.

## Acceptance (tests on the public API, clean locked environment, fresh evidence directories)

- P1 Grammar: the accepted forms above parse with `group is None`, `levels == ()` and the stated
  canonical spelling; the grouped forms parse as before; every refusal above with its message.
- P2 Data: a data frame without any grouping column is accepted; missing columns, missing values,
  constant predictors, non-numeric undeclared predictors, the Bernoulli 0/1 rules and the binomial
  count rules are enforced as before.
- P3 Priors: the two new refusals; per-term and per-level `b` coverage on a population-only model;
  `sigma` rules per family.
- P4 Design against an independent construction (the `tests/reference_logp.py` helpers extended to
  `group=None`): raw and centred columns, dummy and product columns centred at their own means, labels,
  `Z` empty, `J == 0`, for numeric-only, categorical and interaction formulas, both centrings.
- P5 Density with and without the Jacobian, the PyTensor and JAX log densities and both analytic
  gradients against central differences of the independent reference, for Gaussian, Bernoulli and
  binomial population-only models with numeric, categorical and interaction terms, both centrings
  (the Jacobian difference is `log sigma` for the Gaussian family and zero for the logit families).
- P6 The Gaussian population-only model with `sigma` fixed is the closed-form Gaussian posterior:
  the negative Hessian equals the precision matrix at any point and the maximiser equals the mean.
- P7 The graph: free variables exactly as stated; the deterministic linear predictor from the graph
  equals the hand computation at random points; `value_point` refusals in both directions;
  `model.names` without group entries.
- P8 `describe()` and `fit_record` (with a synthetic posterior and a supplied panel, no sampling):
  the statements and fields of defaults 4 and 6; `compare_records` on two such records.
- P9 Predictions from synthetic fixed draws (no sampling), computed by hand from the draws and the raw
  design: in-sample and `newdata` without a grouping column, mixed and reordered rows, categorical and
  interaction terms, the Bernoulli probability, the binomial `p`/`expected`/`y` with and without the
  trials column, `kind="response"` scatter, unseen category levels refused, `new_groups` refusals.
- P10 Grouped models unchanged: the existing suites pass; an explicit check that a grouped model still
  has its group names, `J > 0`, and requires `sd` and `z` in `value_point`.
- P11 Categorical label contract without a group: integer codes and strings are the same label.
- P12 The fast suite passes in the clean environment; the executed commands, environment and results
  are recorded under `docs/m3c1-evidence/`.

## Bounded sampling smoke (proposed, NOT executed in this slice)

For the coordinator's next review, one CPU smoke on both backends with predeclared gates, seed 29:
Gaussian `y ~ x + z + x:z` without a group term, 240 rows, levels a/b/c with proportions 0.5/0.3/0.2,
within-level x means 0/1/-0.5, truth `a = 1.0`, `b = 0.5`, `beta_b = -0.8`, `beta_c = 0.6`,
`delta_b = 0.4`, `delta_c = -0.7`, `sigma = 0.4`; priors Normal(0, 5) intercept, Normal(0, 2)
coefficients, HalfNormal(1) sigma; 4 chains, 1000 tune, 1000 draws, `target_accept` 0.9. Gates: 0
divergences, max R-hat <= 1.01, min bulk and tail ESS > 400, backends within 4 combined MCSE on the six
population parameters and `sigma`; each truth's position in its 99 % HDI recorded as an observation, not
asserted (one realisation is not calibration). Companion exact check: with `sigma` fixed by `pm.do`,
NUTS means of `(a_c, b)` within 4 MCSE of the closed-form Gaussian posterior mean (as in M0's B2).
Bernoulli and binomial population-only smokes with the same design and logit-scale priors follow the
same gates. Cap: about ten minutes of CPU in total, zero retries.

## Kill criterion

If Bambi's population-only design, coding or prediction differs from the independent construction in
any P4/P5/P9 case, or if the fast suite regresses, the slice stops and reports; no parser loosening,
prior defaulting, threshold change or sampling is used to get past it.

## Compatibility limits (stated, not hidden)

No intercept removal; no `y ~ (1 | g)` shorthand; one interaction per model and one categorical side
per interaction as before; no Student-t or count families; no GPU evidence for this slice; the
smoke above is a proposal. The `TypeError` for a missing `sd` becomes a `PriorError` at build.

## Executed results (2026-09-19, clean locked environment, Python 3.11.0, fresh evidence directories)

Implementation: `autobayes/formula.py` (optional group term, `has_group`, canonical `y ~ 1`), `autobayes/priors.py`
(`sd` optional; refused without a group term, required with one, at build), `autobayes/model.py` (empty group
design, names without group entries, `value_point` refusals, `describe()` statements), `autobayes/predict.py`
(no grouping column needed, `new_groups` refused, `not_applicable` attributes), `autobayes/session.py`
(`group: null`, `J: 0`, `noncentered: null`, priors without `sd`), version `0.5.0` (`pyproject.toml`, `uv.lock`
own-package line only, `autobayes.__version__`). Reference helpers `tests/reference_logp.py` and
`tests/reference_binomial.py` accept `group=None` with the grouped paths unchanged. Two existing assertions that a
`Priors` construction without `sd` raises `TypeError` now assert the build-time `PriorError` (default 3).

- `tests/test_population_only.py`: **48 passed in 24 s** (P1 grammar and refusals; P2 data without a
  grouping column; P3 prior refusals and coverage; P4 design against the independent construction for three families
  x three term structures x two centrings; P5 density, Jacobian and gradients on the PyTensor and JAX paths for nine
  family/prior/structure cases x two centrings including intercept-only models; P6 the closed-form posterior with
  `sigma` fixed; P7 graph names and `value_point` refusals in both directions; P8 `describe()` and the fit record from
  a synthetic posterior with a stub panel; P9 predictions by hand from fixed draws for three families, mixed rows,
  unseen category refused, `new_groups` refused; P10/P11 grouped models unchanged and the label contract).
- Fast suite (CI command form): **395 passed, 7 deselected in 320 s**, exit 0. A first full run
  (3 failed, 393 passed, exit 1, kept as `fast_suite_before_assertion_updates.txt`) found three stale assertions that a formula
  without a group term must be refused; they were updated to the documented contract (default 1) with no library
  change in between. Evidence `docs/m3c1-evidence/`.
- Not executed: any sampler fit, the example, the slow suite, a GPU run. The bounded smoke stays a proposal for the
  coordinator's review. Version bump note: PR #8 names `0.5.0` too; whichever lands second renumbers.
