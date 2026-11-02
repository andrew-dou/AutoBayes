# Group-level term syntax: slopes alone, one factor in several terms, factor, transform, product and `mo()` slopes, three grouping columns

brms-parity feature G05a. Group-level terms brms takes and this library once refused, each read as brms reads it
(docs/term-model.md for the term model; docs/varying-slopes.md, docs/correlated-slopes.md and docs/grouping-syntax.md for the
terms these extend).

## What is taken

| formula | the model | brms |
| --- | --- | --- |
| `(0 + x \| g)`, `(x - 1 \| g)` | the slope of `x` varies by `g`, the intercept does not: `GroupTerm("g", ("x",))` | one group, `sd \| g \| x` and no `sd \| g \| Intercept` |
| `(1 \| g) + (0 + x \| g)` | the varying intercept and slope of `g` without a correlation: one term of `g`, `coefs = ("Intercept", "x")`, `blocks = (("Intercept",), ("x",))` | two group IDs of `g`, each with its own SD and no correlation |
| `z + (1 + z \| g)`, `z` declared categorical | one varying coefficient per non-reference level, `z[q]`, `z[r]` (treatment contrasts, the population design's coding), correlated with the intercept | `Z_1_2 = zq`, `Z_1_3 = zr`, an `L_1` over three coefficients |
| `x * w + I(x^2) + (1 + I(x^2) + x:w \| g)` | the slope of each population-level term varies by `g`, its covariate the term's design column (`x^2`, `x * w`) | the coefficients `IxE2` and `x:w` |
| `mo(ord) + (1 + mo(ord) \| g)` | `mu += (b + v[g]) * mo(simo, ord)`: one simplex shared with the population-level term, no design column | `(bsp + r_1_2[J_1]) * mo(simo_1, Xmo_1)`, no `Z_1_2` |
| `(1 \| g/h/f2)`, `(1 \| g:h:f2)` | `g/h/f2` is `(1 \| g) + (1 \| g:h) + (1 \| g:h:f2)`; a level of `g:h:f2` per observed combination, labelled `a_h1_u`, sorted | three IDs, the same labels |
| `w + (1 + x \| g)` without `x` | the slope of `x` varies by `g` and has no population-level coefficient: `Z` is the raw column (`GroupTerm.slope_columns`) | `X` has no `x`, `Z_1_2 = x` |
| `y ~ (1 + x \| g)` | the population intercept is implied, as R's formula implies it | the same |

A factor written in several terms is one term here, because brms's model of it is a term of uncorrelated coefficients: each of
its terms has one coefficient or is written with `||`. Its canonical form, `describe()` and the fit record keep the terms as
written. The coefficient labels are the population-level design's (`x`, `I(x^2)`, `x:w`, `z[q]`, `mo(ord)`); brms's are their
`rename()` (`IxE2`, `zq`, `moord`), which the prior table, the oracle and the syntax audit compare.

## Using it

- Priors: `sd={"g": {"x": ab.HalfNormal(0.5)}}` for `(0 + x | g)`; `sd={"g": {"Intercept": ..., "x": ...}}` for `(1 | g) + (0 + x
  | g)`, as for `(1 + x || g)`; a factor slope's SD prior is keyed by the factor, `sd={"g": {"Intercept": ..., "z": ab.HalfNormal(1)}}`,
  one prior for the SDs of all its coefficients (brms's `sd | g | zq` and `sd | g | zr` rows, D184); `cor={"g": ab.LKJ(2)}` for every
  correlated term. `ab.default_priors()` returns them in this form.
- `Model.value_point`: `sd` and `z` by coefficient label, `{"z[q]": ..., "z[r]": ...}`, `{"mo(ord)": ...}`.
- `predict()`: each coefficient adds its effect times its covariate on the rows, recomputed from new rows (a product, a factor's
  indicator, a transform, a raw column, `mo()` from the draws' simplex); both `new_groups` policies apply.
- Not in brms's summaries (`ab.fixef` and friends) except a factor in several terms whose first term has the varying intercept
  and slopes that are columns (`gauss_split_slope_post`): `coef(g)` of a slope without its population-level term is its group effect
  alone. Not in multivariate, non-linear or count-matrix models, nor in `prior_draws()`.

## Refused, with the construct named (D183, D184)

A coefficient in two terms of one factor (`(1 | g) + (1 | g)`, which brms reads once; `(1 | g) + (1 | g/h)`); a factor in
several terms of which one correlates its coefficients, has a monotonic slope, a correlation ID, `mm()` or `gr(by = )`; four or more
grouping columns; a transform, a product or `mo()` that is not a population-level term; a product with a factor (`x:z`) and `*`
inside a group term; a factor or `mo()` slope with `||`, without the varying intercept, in a factor's second term or next to a
formula without an intercept; an ordered factor's slope; a slope-only term with a correlation ID; a slope column that is not
numeric or is constant.

## Evidence

Oracle cases `gauss_slope_only`, `gauss_split_same_factor`, `gauss_factor_varying_slope`, `gauss_transform_interaction_slopes`,
`gauss_mo_varying_slope`, `gauss_three_level`, `gauss_slope_without_population_term` and `gauss_split_slope_post` (the summaries)
pass gates G1-G6 against their frozen brms fixtures; the syntax corpus's `grp_slope_only`, `grp_slope_only_minus_one`,
`grp_split_same_factor`, `grp_slope_without_population_term`, `grp_factor_slope`, `grp_transform_slope`, `grp_interaction_slope`,
`grp_mo_slope`, `grp_three_colon` and `grp_three_nested` read as brms reads them. `tests/test_group_term_syntax.py` covers the
grammar and every refusal, the factor slope's expansion, the covariates, SD-prior coverage, the blocks and names, the graph's linear
predictor against a hand computation, predictions by hand on fitted and new rows, `describe()`, the target identity, the fit record
and the refusals elsewhere. Nothing samples.
