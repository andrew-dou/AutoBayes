# The offset term, `offset(col)` and `offset(log(col))`

brms-parity feature A06. An offset is a known quantity added to the linear predictor of every row, with no
coefficient and no prior: brms's `offset()`. Its usual use is an exposure in a count model,
`cnt ~ x + offset(log(exposure)) + (1 | g)`, where `exp(eta)` is then a rate times the exposure.

## The model

For row `i`, with `o_i` the offset value (`log(exposure_i)` or the column as given):

    eta_i = a + X_i b + o_i + Z_i u

brms puts `o` in `standata$offsets` and adds it to `mu` next to the intercept; `posterior_linpred` and
`posterior_epred` include it on fitted and new rows, with and without group effects (`re_formula = NA`). The
library does the same: `ModelSpec.offset` (an `Offset(column, log)`) is the term, `Design.offset` the vector on
the fitted rows, and `predict()` adds the offset of every row it predicts. The offset is never centred and does
not enter the raw-scale intercept correction `a = a_c - mean(x) b`; it has no design column, no entry in
`ModelSpec.columns`, `Model.coefficients` or the prior dictionary, and no PyMC variable.

## Using it

- Formula: one `offset(col)` or `offset(log(col))` term anywhere on the right-hand side; the canonical form
  places it after the population-level terms, `cnt ~ x + offset(log(exposure)) + (1 | g)`. Bambi/formulae
  evaluate the same `offset()` (Bambi's `log` is `numpy.log`); `Model._extract_design` recomputes the vector
  from the data and raises if Bambi's offset differs from it, then removes the column from the coefficient
  design. The column must be numeric and finite; with `log()` it must be positive.
- `predict()`: `newdata` must carry the offset column (refused otherwise, naming the term); the row's own value
  is used. `include_group_effects=False` keeps the offset. Unseen levels follow the `new_groups` policies as
  before; the offset is added on top.
- `describe()` says "with offset", writes `+ offset_i` in the linear predictor and states what `offset_i` is.
  The fit record carries `"offset": "offset(log(exposure))"`, and the target identity includes the offset
  values, so a model with an offset is a different target from the same formula without one.

## Refused, with the construct named

Any other expression inside `offset()` (`offset(exp(x))`, `offset(x + 1)`, `offset(2 * x)`: in-formula
transforms are feature A05; compute the column and write `offset(<column>)`); more than one offset term (add
the columns first); an offset on the response, the trials column, a grouping factor or a column declared
categorical; `offset()` inside a group term; a non-positive column under `log()`; `newdata` without the column.
A formula whose only right-hand-side term is the offset still needs `1` or a predictor (`cnt ~ 1 +
offset(exposure)`), as every formula does. The offset column may also be a predictor, as in brms.

## Evidence

Oracle case `poisson_offset` (`cnt ~ x + offset(log(exposure)) + (1 | g)`, Poisson) passes gates G1-G6 against
its frozen brms fixture: G1 compares `standata$offsets` with `Design.offset`; G2 the log density and gradient;
G3 the graph's `mu = exp(eta)`; G4 `posterior_linpred`/`posterior_epred` on fitted and new rows with and without
group effects; G5 the JAX density. `tests/test_offset.py` covers the refusals, the term model, the design, the
graph's linear predictor at a point, by-hand and Bambi-path predictions (Gaussian and Poisson), the fit record
and `describe()`. Nothing samples.
