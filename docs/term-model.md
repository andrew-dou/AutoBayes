# The term model

A parsed formula (`autobayes/formula.py`, `ModelSpec`) is two tuples of terms. Everything the library once
read from a flat description of the formula (`predictors`, `terms`, `columns`, `categorical_names`,
`interactions`, `groups`, `levels_of`, `term_columns`, `is_vector_term`, `column_terms`) is a view derived
from them, kept so that existing callers keep working. brms's names for the same things are given where they
exist, so the oracle adapter (`tests/oracle/adapter.py`) extends to new term kinds without renaming.

## Population-level terms: `PopTerm(name, kind, columns, levels)`

One entry per term as written, in design order (main effects, then interactions):

| formula | `kind` | `columns` (design labels, Bambi's own) | `levels` (coefficient levels) |
| --- | --- | --- | --- |
| `x` | `numeric` | `("x",)` | `()` |
| `z` (declared `categorical={"z": "p"}`) | `categorical` | `("z[q]", "z[r]")`; all levels under `0 +` | the coefficient levels of the `Categorical` |
| `x:z` (`x*z` expands to `x + z + x:z`) | `interaction` | `("x:z[q]", "x:z[r]")` | the non-reference levels of `z` |

`term.operands` are the data columns the term is built from (`(x,)`, `(z,)`, `(x, z)`); `term.is_vector` says
whether it has one coefficient per level. A categorical term's levels are known once the data are seen:
before that it has no levels and its single column is its own name, a placeholder, exactly as
`ModelSpec.columns` has always listed it. The `Categorical` objects (`ModelSpec.categoricals`) keep what a
categorical term needs beyond its name: reference level, fitted levels, unused declared categories, coding.
`pop_term(name, kind, categoricals)` builds a term from those; `pop_term_design(term, frame)` is its
`(n, len(columns))` raw design block on any frame, which both the Bambi verification in
`Model._extract_design` and `predict.raw_design` use. The coefficients of a term are brms's `b` entries for
its columns; `Model.coefficients` lists `(column, term, index)` per column.

## Group-level terms: `GroupTerm(factor, coefs, correlated)`

One entry per `(... | factor)` term, brms's group `m`: `coefs` are the population-level coefficients that
vary by level (brms `M_m` of them; `("Intercept",)` for `(1 | g)`, `("Intercept", "x")` for `(1 + x || g)`)
and `correlated` whether they share a correlation matrix (`|`, brms's default, not supported with slopes
yet) or are independent (`||`; moot for one coefficient, so `(1 || g)` is `(1 | g)`). `term.slopes` are
the coefficients other than the intercept and `term.canonical` the term as the canonical formula writes it.
Level labels are not part of the term; they are read from the data (`ModelSpec.levels_by_group[factor]`,
sorted strings). The parameters of coefficient `c` are brms's `sd_m[c]` (`1|g_sigma`, `x|g_sigma`), `z_m[c]`
(`1|g_offset`, `x|g_offset`, the standardised effects) and `r_m_c` (`1|g`, `x|g`, the effects per level);
`Model.group_coefficient_names[(factor, coef)]` maps those roles to the PyMC names, and
`Model.group_names[factor]` keeps the varying intercept's roles with each slope's as `sd:x`, `offset:x`,
`group_effect:x` (docs/varying-slopes.md).

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
