# General two-way interactions: `x:w` and `z:h`

G03 (docs/term-algebra.md) reads R's term algebra: interactions of any order, interactions without their main effects (coded as R
codes them) and repeated terms. A two-way interaction of two main effects is still the term this page describes; the refusals
below of a missing main effect, of three-way interactions and of a repeated interaction are lifted there.

brms-parity feature A04. An interaction `a:b` (or the shorthand `a*b` = `a + b + a:b`) may now join any two main
effects of the formula, not only a numeric and a categorical one: `x:w` between two numeric columns and `z:h`
between two declared categorical columns, next to the `x:z` form that existed before (docs/history/m3a2-decision.md).
Both main effects are still required and interactions stay two-way.

## The model

- `x:w` is one design column, the elementwise product `x_i * w_i`, with one coefficient `b_x:w`: the slope in
  `x` is `b_x + w * b_x:w` and the slope in `w` is `b_w + x * b_x:w`. R and brms build the same column (`x:w`).
- `z:h`, with `z` and `h` treatment-coded (reference levels `p` and `h1`, say), has one column per *cell* of
  non-reference levels, the product of the two indicators `1[z = l] * 1[h = m]`, and one coefficient per cell.
  This is R's coding of the interaction of two factors whose main effects are present (`contrasts(z) x
  contrasts(h)`; brms names the columns `zq:hh2`, ..., `zr:hh4`). A row in cell `(l, m)` adds `b_z:h[l, m]` on
  top of `b_z[l] + b_h[m]`; the cell coefficient is zero when either level is the reference, so `b_z[l]` is the
  level difference at the reference of `h` and `b_z:h[l, m]` is how much that difference changes at level `m`.
  A cell with no fitted rows keeps its (zero) column, as brms does, and `Model.warnings` says so: its
  coefficient is informed by the prior alone.
- Centring (`center_predictors=True` with an intercept) treats every expanded column alike: the product column is
  centred at `mean(x * w)`, which is not `mean(x) * mean(w)`, and a cell column at the cell's proportion of the
  fitted rows; `describe()` prints the values used and the raw-scale intercept correction.

## Labels

The term model (docs/term-model.md) keeps the library's own labels, which the oracle matches to brms by content:

| formula | term `name` | `columns` | `levels` | `categorical` |
| --- | --- | --- | --- | --- |
| `x:w` (`x*w`) | `x:w` | `("x:w",)` | `()` | `()` |
| `z:h` (`z*h`), `z` in p, q, r and `h` in h1..h4 | `z:h` | `("z[q]:h[h2]", "z[q]:h[h3]", ..., "z[r]:h[h4]")` | `("q, h2", "q, h3", ..., "r, h4")` | `("z", "h")` |

The cell columns are Bambi/formulae's own labels in their order (the first operand's level changes slowest), as
`x:z[q]` was; the cell levels `"l, m"` are the labels Bambi gives the PyMC coordinate `z:h_dim`, so the posterior
rows read `z:h[q, h2]` and a per-cell prior is keyed by the same string. Two cells whose joined labels coincide
are refused rather than merged. `x:z` keeps its numeric-first spelling; `x:w` and `z:h` are spelled in main-effect
order, as R orders the variables of a term (`y ~ w + x + x:w` becomes `w + x + w:x`), and the reversed spelling
of an existing interaction is a duplicate.

## Using it

- Formula: `y ~ x * w + (1 | g)`, `y ~ z * h`, or any mixture, `y ~ x * w + z * h + x:z + (1 | g)`, with the
  categorical columns declared in `categorical={"z": "p", "h": "h1"}`. `0 + z * h` codes the first categorical
  main effect with all its levels, as before; the cells stay `contrasts x contrasts`.
- Priors: `b` keyed by term takes one prior for `x:w` and one prior or a per-cell mapping for `z:h`,
  `b={..., "z:h": {"q, h2": Normal(0, 1), "q, h3": Normal(0, 1), ...}}` covering exactly the cells (brms `class =
  b, coef = zq:hh2`). The single `b` prior for every coefficient keeps working.
- `Model.value_point` takes `b` in design-column order (`model.coefficients`); `predict()` recomputes products and
  cells from the raw columns on new rows; `describe()` states the slopes and cell meanings and the centring; the fit
  record lists the operand pairs in `interactions` (`[["x", "w"], ["z", "h"]]`).

## Refused, with the construct named

Three-way and higher interactions (`x:w:z`, `x * w * z`); an interaction missing a main effect (both are required,
divergence D14); the same interaction twice in either spelling; a term with itself; the response, the trials
column or a grouping factor as an operand; `offset()` or a function call as an operand (`log(x):z`, divergence
D15); an interaction as a varying slope; a product column that is constant in the fitted rows.

## Evidence

Oracle cases `gauss_interaction_numeric_numeric` (`y ~ x * w + (1 | g)`) and
`gauss_interaction_categorical_categorical` (`y ~ z * h`, twelve brms design columns including a cell with no
rows) pass gates G1-G6 against their frozen brms fixtures: the design columns matched by content, the log density
and gradient with and without the Jacobian, `b_Intercept`, `posterior_linpred`/`posterior_epred` on fitted and new
rows, the JAX density. The adapter and gates needed no change. `tests/test_general_interactions.py` covers the
refusals, the term model, the design against hand-built columns, the graph and density at a point, by-hand and
Bambi-path predictions, per-cell priors, the empty-cell warning, the fit record and `describe()`. Nothing samples.
