# Grouping syntax: `(1 | g:h)` and `(1 | g/h)`

brms-parity feature A03. `g:h` is the interaction of two grouping columns used as one grouping factor: its levels
are the observed `(g, h)` combinations, labelled `g_h` exactly as brms labels them (`paste(g, h, sep = "_")`,
so `a` and `h1` give `a_h1`), one varying intercept (and, with `||`, varying slopes) per combination. `(1 | g/h)`
is brms's nesting shorthand and expands to `(1 | g) + (1 | g:h)`, in that order.

## The model

`g:h` is a grouping factor like any other (docs/term-model.md): `GroupTerm("g:h")`, brms's group `m` with its own
`sd_m`, `z_m` and `r_m_1`, a `GroupBlock` whose `Z` is the one-hot indicator of the row's combination, and
`levels_by_group["g:h"]` the sorted labels. Only observed combinations are levels: a pair of fitted levels that
never occurs together has no effect, and on new rows it is an unseen level of `g:h` (brms refuses such rows by
default too). Priors, `value_point`, `predict()`'s `new_groups` mapping, `describe()`, the fit record and the
target identity all key the factor as written, `"g:h"`.

Bambi/formulae would read `(1 | g:h)` as the full Cartesian product of levels with `:`-joined labels, which is
not brms's model, so the library builds the derived column itself: `GroupTerm.column` is `g_h`, added to the
fitted frame (and to `newdata` at prediction time) from the two operand columns, and Bambi is handed
`(1 | g_h)`. The PyMC variables therefore carry the column's name, `1|g_h`, `1|g_h_sigma`, `1|g_h_offset`, on
the coordinate `g_h__factor_dim`; `model.group_names["g:h"]` maps the roles to them. A formula that also uses
a column named `g_h` is refused, so the derived column cannot shadow a real one.

## Using it

- Formula: `y ~ x + (1 | g) + (1 | g:h)` or, equivalently, `y ~ x + (1 | g/h)`; `(1 | g:h)` alone is allowed;
  `(1 + x || g:h)` and `(1 + x || g/h)` give independent varying slopes per combination (docs/varying-slopes.md).
- Priors: `sd={"g": HalfNormal(1), "g:h": HalfNormal(0.5)}` (brms `class = sd, group = "g:h"`).
- `predict()`: `newdata` needs `g` and `h` (not `g_h`); an unseen combination follows the policy given for
  `"g:h"` in `new_groups` (`"population"`: a fresh `Normal(0, sd_g:h)` effect; `"resample_observed"`: an observed
  combination's effect). A label unseen in `g` is unseen in `g:h` as well, and each factor applies its own policy.
- `describe()` lists the combinations and says how the factor was built and named.

## Refused, with the construct named

Three or more factors (`g:h:k`, `g/h/k`), mixing `/` and `:` (`g/h:k`), a factor combined with itself (`g:g`),
the same partition written twice (`(1 | g:h) + (1 | h:g)`), a duplicate after expansion (`(1 | g) + (1 | g/h)`),
an operand that is the response, the trials column, a predictor or declared categorical, a varying slope equal to
an operand, a derived column name that collides with a formula column (`g_h` next to `g:h`), and two distinct
combinations whose joined labels coincide (`("a_b", "c")` and `("a", "b_c")` both give `a_b_c`): brms merges
those silently, the library refuses them.

## Evidence

Oracle cases `gauss_nested_colon` (`y ~ x + (1 | g) + (1 | g:h)`) and `gauss_nested_slash` (`y ~ x + (1 | g/h)`)
pass gates G1-G6 against their frozen brms fixtures: the row partition and the `a_h1`-style level labels of
`g:h`, the log density and gradient, `r_2_1`, `posterior_linpred`/`posterior_epred` on fitted rows and on new
rows carrying fitted pairs, the JAX density. `tests/test_grouping_syntax.py` covers the expansion, the derived
labels, the refusals, the blocks and names, `value_point`, predictions under both policies for an unseen
combination, the fit record and `describe()`. Nothing samples.
