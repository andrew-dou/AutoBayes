# R's term algebra: interactions of any order and factors whose margin is absent

brms-parity feature G03 (`autobayes/term_algebra.py`). The population-level part of a formula is read as R reads it
(`?formula`, `?terms.formula`), and a factor in an interaction is coded as R's `model.matrix()` codes it, so that the design is
brms's column for column.

## The algebra

| written | terms, in R's order |
| --- | --- |
| `a + b` | `a`, `b` |
| `a:b` | `a:b` (the interaction alone; neither main effect is added) |
| `a * b` | `a`, `b`, `a:b` |
| `a * b * c` | `a`, `b`, `c`, `a:b`, `a:c`, `b:c`, `a:b:c` |
| `(a + b + c)^2` | `a`, `b`, `c`, `a:b`, `a:c`, `b:c` (every interaction up to order 2) |
| `(a + b):c` | `a:c`, `b:c` |
| `a * b - a` | `b`, `a:b` (`-` removes the terms of what follows from those before it) |
| `x + x`, `x:z + z:x` | `x`; `x:z` (a term written twice is one term) |

Terms are ordered by their order (the number of variables) and then by appearance, as R orders them; `- 1` and `0 +` remove the
intercept as before. The variables of a term are written in the order of brms's population-level formula, which brms rebuilds
from R's ordered terms: `y ~ x:z + z` is brms's `z + z:x`, so its columns are `zp:x`, `zq:x`, `zr:x` (probed on brms 2.23).
`tests/test_term_algebra.py` checks the expansion against R 4.5.2's `terms()` on 31 right-hand sides and the column names, order
and content against brms's `make_standata()` on 15 formulas and R's `model.matrix()` on 3 more.

## The coding of a factor in an interaction

A factor `F` in an interaction `T` is coded by treatment contrasts (its non-reference levels) when `T` without `F` is
contained in a term before `T`, and by an indicator of every level otherwise. A main effect, whose margin is the intercept, always
takes contrasts, except that without an intercept the first term with a factor codes it by indicators (`0 + z`, as before).

| formula | columns (brms's names) |
| --- | --- |
| `z + x:z` | `zq`, `zr`, `zp:x`, `zq:x`, `zr:x` (a slope for every level; case `gauss_factor_slopes`) |
| `x + x:z` | `x`, `x:zq`, `x:zr` (`p`'s slope is `x`'s; case `gauss_numeric_colon_factor_one_main`) |
| `x:w + z:h` | `x:w` and the twelve cells `zp:hh1`, ..., `zr:hh4` (case `gauss_colon_only`) |
| `x * z * h` | the main effects, the two-way terms and `x:zq:hh2`, `x:zr:hh2`, ..., `x:zr:hh4` (case `gauss_three_way_factor`) |
| `a:b + a:c` | `aa1:bb1`, ..., `aa3:bb2` and `aa1:cc2`, `aa2:cc2`, `aa3:cc2` (`a` is contained in the earlier `a:b`) |

## In the term model

A two-way interaction of two main effects is A04's `"interaction"` term (docs/interactions.md), unchanged: Bambi builds it and
its labels are Bambi's (`x:z[q]`, `z[q]:h[h2]`). Every other interaction is a `"product"` term (docs/term-model.md): before the
data are seen one placeholder `PopTerm` named as the interaction (`x:z:h`), then one `PopTerm` per design column (`x:zq:hh2`),
each holding a `Product` (its interaction, operands in brms's order, the factors coded by indicators and by contrasts, and the
column's levels). Its column is the product of the numeric operands and one level's indicator of each factor, computed by the
library on the fitted rows and on new rows (`predict(newdata=...)`) and handed to Bambi in backticks, so each column is one
scalar coefficient named as brms names it (`b_x:zq:hh2`). Centring, the raw-scale intercept and the prior
table treat the columns like any other.

- **Priors** are keyed by the interaction as brms orders its variables (`b={"z:x": prior}`; a key in another order is refused
  with the canonical one): one prior for every column, or, for an interaction with a factor, a mapping by each column's levels
  joined by `", "`, `b={"x:z:h": {"q, h2": ..., "r, h2": ..., ...}}`, covering exactly its columns.
- **`describe()`** gives each interaction's coding and columns, and writes a factor that enters interactions only as such.
- **The fit record** lists each interaction's operands, its factors coded by indicators and its columns (`"products"`).
- **Warnings.** When a formula has a product term and its population-level design (with the intercept) is rank deficient,
  `Model.warnings` and `describe()` say how many combinations of the coefficients the data do not identify (`gauss_colon_only`:
  rank 13 of 14, the cells add up to the intercept; `gauss_three_way_factor`: rank 21 of 24). brms builds the same design and
  says nothing (divergence D179).

## Refused, with the construct named

- `x^2` and `(a:b)^2` (R reads the power of one term as the term itself), `^` with anything but a whole number of at least 2, and
  `x:x`, `x*x`, `x:z:x` (R reads `x:x` as `x`): write `I(x^2)` for a square (docs/transforms.md); divergence D180.
- `y ~ 0 + x + x:z`: without an intercept R and brms code `z` in `x:z` by indicators although `x` is a term, and the columns of
  `x:z` add up to `x`; keep the intercept, or drop `x` (`0 + x:z` is a slope per level); divergence D180.
- A column zero on every fitted row (a level combination no fitted row has): brms keeps it, informed by its prior alone; Bambi
  takes no constant column, so it is refused naming the column (divergence D179). A04's `z:h` keeps an empty cell as before.
- As before: interactions with `mo()`, `me()`, `mi()`, `cs()`, `gp()`, `offset()`, a transformed term, an ordered factor
  (D168), the response, the trials column or a grouping factor; `%in%`; an interaction as a varying slope (`(1 + x:z | g)`); a
  function call written twice (`log(x) + log(x)`).
- A product term in a non-linear parameter's formula, a count-matrix or simplex family's category formula and a multivariate
  model, and the summaries (`ab.fixef`, ..., D162) of a model with one: no frozen brms case verifies them (divergence D179).
  Distributional formulas (`sigma ~ z + x:z`) take product terms; the design is the same code.

## Evidence

The oracle cases `gauss_three_way_numeric` (`x * w * v`, a coefficient prior on `x:w:v`), `gauss_three_way_factor`,
`gauss_power_expansion`, `gauss_colon_only`, `gauss_factor_slopes`, `gauss_numeric_colon_factor_one_main` and
`bernoulli_three_way_varint` pass gates G1 to G6 against their frozen brms fixtures; the syntax corpus entries `pop_three_way`,
`pop_power_expansion`, `pop_colon_only`, `pop_factor_colon_only`, `pop_numeric_colon_factor_one_main`, `pop_factor_slopes`,
`pop_duplicate_term`, `pop_minus_term` and `pop_duplicate_interaction` match brms's reading. Every other frozen case's model is
unchanged (target identity, `describe()`, names and coefficients).
