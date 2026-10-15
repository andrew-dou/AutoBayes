# The term model

A parsed formula (`autobayes/formula.py`, `ModelSpec`) is two tuples of terms. Everything the library once
read from a flat description of the formula (`predictors`, `terms`, `columns`, `categorical_names`,
`interactions`, `groups`, `levels_of`, `term_columns`, `is_vector_term`, `column_terms`) is a view derived
from them, kept so that existing callers keep working. brms's names for the same things are given where they
exist, so the oracle adapter (`tests/oracle/adapter.py`) extends to new term kinds without renaming.

## Population-level terms: `PopTerm(name, kind, columns, levels, categorical)`

One entry per term as written, in design order (main effects, then interactions):

| formula | `kind` | `columns` (design labels, Bambi's own) | `levels` (coefficient levels) | `categorical` |
| --- | --- | --- | --- | --- |
| `x` | `numeric` | `("x",)` | `()` | `()` |
| `z` (declared `categorical={"z": "p"}`) | `categorical` | `("z[q]", "z[r]")`; all levels under `0 +` | the coefficient levels of the `Categorical` | `("z",)` |
| `x:z` (`x*z` expands to `x + z + x:z`) | `interaction` | `("x:z[q]", "x:z[r]")` | the non-reference levels of `z` | `("z",)` |
| `x:w` (A04, docs/interactions.md) | `interaction` | `("x:w",)` | `()` | `()` |
| `z:h` (A04) | `interaction` | `("z[q]:h[h2]", "z[q]:h[h3]", ...)`, first operand slowest | the cells `("q, h2", "q, h3", ...)`, Bambi's labels | `("z", "h")` |
| `log(xpos)`, `I(x^2)` (A05, docs/transforms.md) | `transform` | `("log(xpos)",)`, the evaluated column named as written | `()` | `()` |
| `mo(ord)` (B03, docs/monotonic.md) | `monotonic` | `()`: no column in `X`; its coefficient (brms's `bsp`) and simplex act through its own block | `()` | `()` |

`term.operands` are the data columns the term is built from (`(x,)`, `(z,)`, `(x, z)`, `(x, w)`, `(z, h)`, and
`(xpos,)` for `log(xpos)`); `term.categorical` names the operands declared categorical and `term.is_vector` (one
coefficient per level) is whether there are any, which is known before the data are seen. A transform term's
`term.transform` (a `Transform(fn, column, constant)`) says how its column is evaluated, `term.bambi_name` is the
backticked name under which Bambi reads the evaluated column from the fitted frame, and `ModelSpec.transform_terms`
lists the transform terms. A categorical term's levels are known once the
data are seen: before that it has no levels and its single column is its own name, a placeholder, exactly as
`ModelSpec.columns` has always listed it. The `Categorical` objects (`ModelSpec.categoricals`) keep what a
categorical term needs beyond its name: reference level, fitted levels, unused declared categories, coding.
`pop_term(name, kind, categoricals)` builds a term from those; `pop_term_design(term, frame, categoricals)` is
its `(n, len(columns))` raw design block on any frame (the `Categorical`s are needed only to build the cells of
`z:h`), which both the Bambi verification in `Model._extract_design` and `predict.raw_design` use. The
coefficients of a term are brms's `b` entries for its columns; `Model.coefficients` lists `(column, term, index)`
per column. `ModelSpec.pop_columns` lists the data columns the population-level terms read.
A monotonic term `mo(ord)` (B03) is a population-level term with a coefficient but no design column: `spec.terms` and
`spec.coefficient_labels` (brms's class `b`: the design columns and the monotonic terms, in term order) list it,
`spec.columns` and `Model.coefficients` do not, and `spec.bambi_terms` are the terms without it, which Bambi builds;
`ModelSpec.monotonics` holds each one's coding (`Monotonic`: the column, its fitted range or ordered categories) once
the data are seen, `Design.monotonic[ord]` its codes per fitted row and number of steps, and the model adds the term to
the parent's linear predictor while Bambi builds it (docs/monotonic.md).

## Group-level terms: `GroupTerm(factor, coefs, correlated)`

One entry per `(... | factor)` term, brms's group `m`: `coefs` are the population-level coefficients that
vary by level (brms `M_m` of them; `("Intercept",)` for `(1 | g)`, `("Intercept", "x")` for `(1 + x || g)` and
`(1 + x | g)`) and `correlated` whether they share a correlation matrix (`|`, brms's default: an LKJ prior on
the correlation, A02, docs/correlated-slopes.md) or are independent (`||`; moot for one coefficient, so
`(1 || g)` is `(1 | g)` and `term.has_correlation` is what says a correlation matrix exists). `term.slopes` are
the coefficients other than the intercept and `term.canonical` the term as the canonical formula writes it.
A correlated term also has brms's `L_m` (the Cholesky factor of the correlation matrix, PyMC `L|g`, sampled on
Stan's `cholesky_corr` coordinates `L|g_cholesky_corr__`) and `Cor_m` (`cor|g`), named by
`Model.correlation_names[factor]` and listed in `Model.group_names[factor]` as `L` and `cor`.
Level labels are not part of the term; they are read from the data (`ModelSpec.levels_by_group[factor]`,
sorted strings). The factor is a column, or the derived factor `g:h` (A03, docs/grouping-syntax.md):
`term.operands` are the columns it is read from, `term.derived` whether it is built from two, and `term.column`
the fitted-frame column holding its labels, which is also the name Bambi sees (`g_h`, so the PyMC variables are
`1|g_h`, `1|g_h_sigma`, `1|g_h_offset`); `group_labels(factor, frame)` builds the labels (brms's `g_h` over the
observed combinations) and `(1 | g/h)` parses to the two terms `(1 | g) + (1 | g:h)`.
The parameters of coefficient `c` are brms's `sd_m[c]` (`1|g_sigma`, `x|g_sigma`), `z_m[c]`
(`1|g_offset`, `x|g_offset`, the standardised effects) and `r_m_c` (`1|g`, `x|g`, the effects per level);
`Model.group_coefficient_names[(factor, coef)]` maps those roles to the PyMC names, and
`Model.group_names[factor]` keeps the varying intercept's roles with each slope's as `sd:x`, `offset:x`,
`group_effect:x` (docs/varying-slopes.md).

## The offset term: `ModelSpec.offset`

brms's `offset(col)` / `offset(log(col))` (A06) is not a population-level term: it has no coefficient, no
prior and no design column, so it is a separate field, `Offset(column, log)`, with `term`
(`offset(log(exposure))`), `expression` and `values(frame)`; `Design.offset` holds its vector on the fitted rows
(brms `offsets`) and `predict()` adds it on every row (docs/offset.md). `columns`, `coefficients` and the prior
dictionary never see it.

## Response addition terms: `ModelSpec.additions`

brms's addition terms after `|` other than `trials()` (B02, docs/addition-terms.md) are data of the formula too, with no
coefficient, prior or design column: `AdditionTerm(kind, column, lb, ub)` for `weights(col)`, `cens(col)` and
`trunc(lb = a, ub = b)`, in the order written (`spec.addition(kind)` finds one; `term` is its canonical spelling, which
the canonical formula writes after `trials()`). `formula.addition_data(spec, frame)` gives brms's Stan data (`weights`,
`cens`, `lb`, `ub`) and `Design` holds them on the fitted rows; the likelihood, not Bambi's formula, reads them.

## Distributional formulas: `ModelSpec.dpars`

A model given a list of formulas (B01, docs/distributional.md), `["y ~ x + (1 | g)", "sigma ~ z"]`, holds each formula
after the first as a `ModelSpec` of its own in `ModelSpec.dpars`, whose `response` is the parameter's brms name
(`spec.dpar("sigma")`, `spec.dpar_names`, `spec.formulas`): its population-level terms are `PopTerm`s of every kind, with
its own intercept, columns, coefficients and centring; it has no group-level term, offset or addition term (refused).
`spec.all_categoricals` and `spec.model_columns` gather every formula's categorical columns and data columns;
`Design.dpars[name]` (`DparDesign`) is its design and `Model.dpar_names[name]` its PyMC variables (`sigma_Intercept`,
`sigma_z`), Bambi's distributional component named after the parameter (`alpha` for brms's `shape`).

## Multivariate formulas: `MultivariateSpec`

brms's `mvbind(y, y2) ~ x + (1 | g)` (C01, docs/multivariate.md) is not a `ModelSpec`: `autobayes.multivariate.parse_multivariate`
gives a `MultivariateSpec` holding the responses in order, the shared right-hand side and each response's own formula
`y ~ x + (1 | g)` parsed as the univariate `ModelSpec` it is (`specs`), from which `MultivariateModel` builds one univariate
`Model` per response; every term kind above therefore reaches a multivariate model through its responses' term models.

## Design blocks: `Design.group_blocks[(factor, coef)]`

`Model.design` holds one `GroupBlock(levels, Z, group_index, column_slice)` per (factor, coefficient), in
formula order. `Z` is the `(n, J)` block of that coefficient: row `i` carries the coefficient's covariate
value (one for the intercept, so the block is the one-hot indicator; the raw `x` for the slope of `x`) in the
column of row `i`'s level, which is brms's `Z_m_c[i]` spread over its group index `J_m[i]`; `group_index` is
that index, 0-based; `J` the number of levels (brms `N_m`). `Design.Z` is the blocks side by side, exactly
Bambi's concatenated group design, and `column_slice` says where each sits. `Design.group_designs` (the
varying-intercept block per factor), `levels`, `group_index`, `J` and `group_slices` are derived views.

## Adding a term kind

1. Parse it in `parse_formula` into a `PopTerm` of a new `kind` (add the kind to `POP_TERM_KINDS` and its
   levels/columns rule to `pop_term`) or into a `GroupTerm` with the right `coefs` and `correlated`.
2. Give it its design block: the `kind` branch of `pop_term_design` for a population-level term; for a
   group-level coefficient, the block `Model._extract_design` expects from Bambi (indicator times covariate)
   under the Bambi name `_bambi_group_name(factor, coef)`.
3. Say how it reads: the `kind` branch of the linear predictor in `Model.describe()` and, for a vector term,
   the prior coverage rule in `Priors.coefficient_priors` if it differs from the existing one.
4. Extend the oracle adapter to the new brms parameters and add a frozen case (`oracle/README.md`).

Nothing else keys on the kind: names, coefficients, `value_point`, prediction and the target identity read
the terms and their blocks.
