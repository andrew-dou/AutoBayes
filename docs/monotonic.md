# Monotonic effects: `mo(x)`

brms-parity feature B03 (brms's `mo()`).

```python
import autobayes as ab
priors = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1),
                   simo={"ord": ab.Dirichlet(2, 2, 2)})
model = ab.gaussian("y ~ x + mo(ord) + (1 | g)", data, priors)
print(model.describe())
```

## The model

An ordinal predictor `x` is coded `0..D` per row (brms's `Xmo`; `D`, brms's `Jmo`, is the number of steps between its
lowest and highest value), and the term adds

    b_mo * mo(simo, Xmo_i),    mo(simo, i) = D * (simo[1] + ... + simo[i])   (0 for i = 0)

to the linear predictor: `simo` is a simplex of `D` shares summing to 1, one per step, under a Dirichlet prior, and
`b_mo` (brms's `bsp`) the average change per step, so `D * b_mo` is the change from the lowest value to the highest and
the effect is monotonic in `x` whatever the sign of `b_mo`. The term is not centred (brms does not centre it): the
intercept, centred or not, refers to rows at the lowest value. brms's coding (`get_mo_values`), which `formula.Monotonic`
keeps once the data are seen:

| column | code of a row | `D` |
| --- | --- | --- |
| whole numbers (any numeric dtype) | the value minus the smallest fitted value | largest minus smallest fitted value: a whole number in between that no row takes is still a value (a warning names it) |
| an ordered pandas Categorical (brms's ordered factor) | its position among the declared categories that fitted rows take | their number minus one: brms drops a factor's unused levels before coding it (a warning names them) |

New rows are coded the same way, and a value outside that coding (a whole number outside the fitted range, a
non-whole number, a category not fitted) is refused, as brms refuses it.

## How it is built

Bambi has no `mo()`: it is handed the formula without the term, and the term is added to the parent's linear predictor
while Bambi builds the model, through the family's `transform_backend_eta` hook (`likelihoods.monotonic_family`), before
the link. For each term the model creates the simplex `simo_x`, a `Dirichlet` with brms's concentrations sampled on
Stan's stick-breaking coordinates (`monotonic.StickBreakingTransform`, Stan 2.32's `simplex_constrain`:
`z_k = inv_logit(y_k - log(D - k))`, `x_k = (1 - x_1 - ... - x_{k-1}) z_k`, with Stan's Jacobian; PyMC's own simplex
transform is another map, whose coordinates and gradient differ), and the coefficient `mo(x)` under its `b` prior, and
adds `mo(x) * D * cumsum(simo_x)[Xmo]` with a leading zero. `Model._extract_design` checks the graph: both variables
free, the simplex on its `D - 1` stick-breaking coordinates with the coordinate `"1"`..`"D"`, and the parent's linear
predictor (on Bambi's link scale) moving by exactly `c * D * sum(simplex[:Xmo_i])` per row, recomputed from the data, when
the coefficient goes from 0 to `c = 1 / D` at a simplex with unequal steps.

## Using it

- Formula: `mo(col)` as a population-level term, next to every other term kind (`y ~ x + mo(ord) + mo(size) + (1 | g)`,
  `y ~ 0 + mo(ord)`), for every family but the categorical one; also with weights, a distributional formula on `sigma`,
  `shape` or `phi`, and correlated slopes. It is a `PopTerm` of kind `"monotonic"` with no design column:
  `spec.terms` lists it, `spec.columns`, `model.coefficients` and `Design.X_raw` do not, `spec.coefficient_labels`
  (brms's class `b`) does, and `Design.monotonic[col]` holds its codes and `D`.
- Priors: its coefficient is class `b`, covered by `b` (one prior for all, or `b={"x": ..., "mo(ord)": ...}` keyed by the
  term; brms `coef = moord`), a location prior. Its simplex takes `simo={"ord": ab.Dirichlet(2, 2, 2)}` keyed by the
  column (brms `class = simo, coef = moord1`), required (D1); a single monotonic term also takes the bare prior.
  `ab.Dirichlet(2, 2, 2)` or `ab.Dirichlet([2, 2, 2])` gives one concentration per step, `ab.Dirichlet(1)` one for every
  step (brms's shorthand, and its default `dirichlet(1)`: `ab.default_priors()` returns it).
- `value_point(..., bsp={"ord": 0.3}, simo={"ord": [0.2, 0.5, 0.3]})`: the coefficient and the simplex (positive, summing
  to 1), stored on Stan's coordinates (`simo_ord_stickbreaking__`); `b` has no entry for the term.
- Names: `model.mo_names["ord"]` gives `"b"` (`mo(ord)`, also `names["b:mo(ord)"]`), `"simplex"` (`simo_ord`, also
  `names["simo:ord"]`) and `"coordinate"` (`simo_ord_dim`).
- `predict()` adds the term on fitted and new rows (new rows need the column), with or without group effects; a model
  with a monotonic term is always predicted in this layer.
- `describe()` writes the coding, `mo()` and its meaning out; the target identity holds each term's coding, codes and
  simplex prior; fit records list `monotonic` and the priors `b:mo(ord)` and `simo:ord`; `prior_table()` writes brms's
  `b` coefficient `moord` and `simo` row `moord1`; `prior_predictive()` draws the simplexes.

## Refused, naming the construct

Interactions with a monotonic term (`mo(x):z`, `mo(x) * z`, `mo(x):mo(w)`; D40), `mo()` in a group-level term
(`(1 + mo(x) | g)`; D41), `mo()`'s `id` argument, an expression inside `mo()`, `mo()` in a distributional formula, the
categorical family, a predictor with two values (one step: write it as a 0/1 numeric term) or more than 10^6 steps (D42),
a column that is not whole numbers or an ordered Categorical (brms refuses those too), with fewer than three values, a
declared categorical column, and the response, the trials column or a grouping factor inside `mo()`.

## Evidence

Thirteen frozen cases pass gates G1-G6 against brms 2.23.0: `gauss_monotonic` (`y ~ mo(ord) + (1 | g)`,
`dirichlet(2, 2, 2)`, the case the loop froze), `gauss_monotonic_two` (`y ~ x + mo(ord) + mo(size) + (1 | g)`: two
simplexes, a coefficient prior by term) and one case per further family frozen with this feature: `student_monotonic`,
`poisson_monotonic` (Bambi's log link), `negbinomial_monotonic` (11 steps, one value without rows, brms's one-number
`dirichlet(2)`), `gamma_monotonic` (a negative minimum), `lognormal_monotonic`, `beta_monotonic`, `bernoulli_monotonic`
(no intercept), `binomial_monotonic`, `zero_inflated_poisson_monotonic`, `hurdle_poisson_monotonic` and
`cumulative_monotonic` (the term in `mu`, outside the thresholds' centring). G1 matches each term to brms's `Xmo_k` and
`Jmo[k]` by content and checks `con_simo_k`, `Ksp` and `Imo`; G2 the log density with and without the Jacobian and the
prior part (`dirichlet_lpdf` included) within 2.4e-10 relative and every gradient coordinate within 5.9e-9, the simplex's
coordinates (which brms's probe cannot reach one at a time) recomputed with Stan's map from `simo_k` at every point and
checked against the fixture's unconstrained values before their gradient is compared (within 1.6e-11); G3 the graph's
linear predictor; G4 `posterior_linpred` / `posterior_epred` on fitted and new rows with and without group effects
(within 2.0e-15); G5 the JAX density; G6 the default-prior table with its `simo` rows. The twelve fixtures frozen with
this feature regenerate identically (`oracle/regen.py check`). `tests/test_monotonic.py` checks
the coding against brms's printed `Xmo` and `Jmo` for whole numbers with a gap, a negative minimum and an ordered factor
with an unused level, the stick-breaking map against Stan's coordinates and a numerical Jacobian, the density by hand on
both paths and every gradient coordinate against central differences, the build-time check, the refusals, priors, `value_point`, predictions,
`describe()`, records, five brms `default_prior()` tables, that a model built from an ordered Categorical or shifted
whole numbers with the frozen case's coding passes every gate against `gauss_monotonic`, and that the gates fail on a
corrupted fixture. Nothing samples.
