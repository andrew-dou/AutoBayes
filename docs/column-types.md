# Columns whose dtype declares their levels

brms codes a factor predictor with treatment contrasts against its first level, an ordered factor with R's polynomial
contrasts and a logical as one column `<name>TRUE`. A pandas Categorical carries what an R factor carries, its levels and their
order, and a boolean column its two values, so such a predictor needs no `categorical=` entry (divergence D172, changed by
Andrew's decision of 26 October 2026):

```python
data["z"] = pd.Categorical(data["z"], categories=["r", "q", "p"])   # an R factor with levels r, q, p
data["lg"] = data["x"] > 0                                           # an R logical
m = ab.gaussian("y ~ x + z + lg + (1 | g)", data, priors)            # columns x, z[q], z[p], lg[TRUE] (brms: zq, zp, lgTRUE)
```

**What each dtype gives.**

- An unordered pandas Categorical: treatment contrasts against its first category that a fitted row takes (brms drops a level no
  row takes, so the next one is then the reference), one column per other fitted category in the categories' order. A declared
  category no row takes is dropped with a warning, as for a declared column.
- An ordered pandas Categorical: R's polynomial contrasts over its fitted levels in their order (`fo.L`, `fo.Q`, ...), as an
  ordered factor declared with its lowest level or written `ordered(fo)` (docs/ordered-factors.md; at most 16 levels, D168).
- A boolean column: brms's logical, `FALSE` the reference and one column `<name>TRUE`. Declared, it is read the same way against
  the declared reference: `categorical={"lg": False}` is the default coding and `categorical={"lg": True}` makes `TRUE` the
  reference (column `lg[FALSE]`); any other declared value is refused.
- In a formula without an intercept (`0 + z`), the first factor is coded with one indicator per level, as R codes it.

Interactions with such a column are those of a declared factor, coded by R's term algebra (docs/term-algebra.md): `x * lg` gives `lgTRUE` and `x:lgTRUE`, and a typed column in an interaction without its main effect (`x + x:z`) is a factor there too. Text and numeric columns still need the
declaration (`categorical={column: reference}`): their levels would come from sorting (D3) or guessing which numbers are codes. A
declared pandas Categorical keeps the declaration's coding, its levels sorted (D3) as before.

**New rows.** `predict(newdata=...)` codes such a column through the fitted levels: a new frame may hold fewer categories (or only
`TRUE`), and a category the fit did not see is refused, naming it.

**Where.** Every univariate model (`ab.gaussian`, ..., `ab.Model`, a mixture), each response of a multivariate model, and
`ab.default_priors`, which reads the columns as the model does. A non-linear model and the count-matrix families (`ab.multinomial`,
`ab.dirichlet_multinomial`) still need the declaration and refuse such a column naming it. A typed column as a grouping factor,
a varying slope or the response is unchanged (D172).

**Evidence.** The oracle cases `gauss_pandas_categorical` (a Categorical with levels `r, p, q`, coded against `r`) and
`gauss_logical` pass every gate; the syntax corpus entries `pop_pandas_categorical`, `pop_ordered_factor_undeclared` and
`pop_logical` match brms's `standata()`; `tests/test_call_surface.py` checks the coding, the level order, interactions, new rows
and the refusals by hand.
