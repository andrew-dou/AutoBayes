# In-formula expressions, fitted transforms and offsets: `I()`, `log(x + 1)`, `abs()`, `scale()`, `poly()`, `factor()`, `offset()`

brms evaluates any R expression written in a formula into the design. AutoBayes reads a whitelisted grammar of them, wider than
the one-column transforms of [transforms](transforms.md) (A05, whose forms keep their own behaviour), and evaluates every one
itself, never as code:

```python
ab.gaussian("y ~ I(x * w) + I(xpos^0.5) + I(x > 0) + log(xpos + 1) + abs(x) + (1 | g)", data, priors)
ab.gaussian("y ~ scale(x) + poly(w, 2) + poly(v, 3, raw = TRUE) + factor(ord) + (1 | g)", data, priors)
ab.poisson("cnt ~ x + offset(log(exposure)) + offset(2 * off) + (1 | g)", data, priors)
```

## The grammar

`autobayes/expressions.py` reads the R text with Python's `ast` after writing `^` as `**` and every name as a placeholder, and keeps
only these nodes: numbers (`2`, `0.5`, `.5`, `1e-3`), column names (R's names, dots included), `+ - * /`, `^` (also `**`), unary
minus, one comparison (`> < >= <= == !=`, not chained), and the calls `I()`, `log()`, `exp()`, `sqrt()` and `abs()` of one
argument, nested freely. A term is one of these calls; a term may also be `scale(col)`, `poly(col, k)` (degree 1 to 10, positional
or `degree = k`), `poly(col, k, raw = TRUE)` or `factor(col)` of one column, at the top of the term only. Anything else is
refused naming it: another function (`sin()`, `log1p()`, `log(x, base)`), R's operators `%%`, `&`, `|`, `!`, `<-`, an index, a
string, a chained comparison, an integer literal `1L`, a spline basis (`ns()`, `bs()`, `splines::ns()`), `scale()`'s `center =`
and `scale =`, `poly()` of several columns (R's `polym`) and its `coefs =`, and `factor()`'s `levels =`, `labels =` and `ordered =`.

Every column an expression reads must be numeric (not text, bool or complex) and finite, and so must the result on every row,
fitted and new (`log(x + 1)` with `x = -1` is refused, naming the row). The term is spelled as R deparses it, with the fewest
parentheses that keep its meaning and the numbers as written: `I(x*w)` is `I(x * w)`, `I((x)^2)` is `I(x^2)`; that spelling is
its name, the key of its prior and the duplicate check (`I(x * w) + I(x*w)` is a duplicated predictor). A spelling that is one of
A05's forms (`I(x**2)`) is that A05 term.

## The forms and their columns

| written | columns (R's model-matrix names; brms's) | values | new rows |
| --- | --- | --- | --- |
| `I(x * w)`, `I(xpos^0.5)`, `log(xpos + 1)`, `abs(x)`, `exp(-x)` | the term itself (`IxMUw`, `IxposE0.5`, `logxposP1`, `absx`) | the expression | recomputed from the raw columns |
| `I(x > 0)` | `I(x > 0)TRUE` (`Ix>0TRUE`) | 1 where the comparison holds, 0 elsewhere: R's logical, a factor coded against `FALSE` | recomputed (new rows all `TRUE` are coded 1) |
| `scale(x)` | `scale(x)` (`scalex`) | `(x - centre) / scale`, the fitted rows' mean and SD (divisor `n - 1`) | the fitted centre and scale |
| `poly(w, 2)` | `poly(w, 2)1`, `poly(w, 2)2` (`polyw21`, `polyw22`) | the orthonormal polynomials of degree 1 to `k` over the fitted rows | the fitted recurrence coefficients |
| `poly(x, 3, raw = TRUE)` | `poly(x, 3, raw = TRUE)1` .. `3` (`polyx3rawEQTRUE1` ..) | `x`, `x^2`, `x^3` | recomputed |
| `factor(ord)` | `factor(ord)2`, ... per level but the lowest (`factorord2`) | the indicator of each fitted level, treatment-coded against the lowest | the fitted levels (D181) |

brms's coefficient names are its `rename()` of R's model-matrix column, R's deparse of the term (numbers as R writes them, so
`I(x^.5)` is `IxE0.5`) and the column's suffix. `poly()`'s columns come from the three-term recurrence that `?poly` documents (Kennedy
and Gentle 1980): with `p_0 = 1`, `p_{d+1}(x) = (x - alpha_d) p_d(x) - (norm2_d / norm2_{d-1}) p_{d-1}(x)`, `alpha_d` and `norm2_d`
the fitted rows' `sum x p_d^2 / norm2_d` and `sum p_d^2`, column `d` is `p_d / sqrt(norm2_d)`; it equals R's QR construction to
rounding (`gauss_scale_poly`: 1.7e-16) and continues the fitted polynomials on new rows. A degree not below the number of distinct
fitted values is refused, as R refuses it. `factor()` takes whole numbers (an integer column, or floats that are whole); its levels
sort as numbers (`2` before `10`) and its lowest is the reference. A comparison that is `TRUE` (or `FALSE`) on every fitted row, a
`factor()` of one level and a constant `scale()` are refused.

## In the term model

Before the data are seen a term is one placeholder `PopTerm` of kind `transform` named as written; `ModelSpec.expression_terms`
lists them. Once the data are seen (`formula.fit_expressions`, inside `validate_data`) each is replaced by one `PopTerm` per design
column, named by its R column name, with `PopTerm.basis` the term as written and `PopTerm.fitted` the fitted parameters (scale's
centre and scale, poly's `alpha` and `norm2`, factor's levels), plain data that `dataclasses.asdict` round-trips. `PopTerm.transform`
is then an `expressions.Expression(term, fitted, suffix)`, whose `values(df)` evaluates the column on any rows; the fitted frame
handed to Bambi carries each column under its name, in backticks, as A05's are. Priors are keyed by the term as written: one prior
for all its columns, or a mapping by the columns' suffixes, `b={"poly(w, 2)": {"1": ab.Normal(0, 1), "2": ab.Normal(0, 3)},
"factor(ord)": {"2": ..., "3": ..., "4": ...}, "I(x > 0)": ab.Normal(0, 1)}`.

## Offsets

Several `offset()` terms are summed, in formula order, as brms sums them, and the expression inside one is read by the same grammar
(no comparison, no `scale()`, `poly()` or `factor()`): `ModelSpec.offset` is then `Offset(parts=("log(exposure)", "2 * off"))`,
whose `values(df)` is the sum, `columns` every column it reads and `term` the canonical `offset(log(exposure)) + offset(2 * off)`.
The library evaluates the sum and hands it to Bambi as the column `log(exposure) + 2 * off` of the fitted frame (and of new rows),
`` offset(`log(exposure) + 2 * off`) ``; one `offset(col)` or `offset(log(col))` alone is A06's `Offset(column, log)` as before
(docs/offset.md). An offset that reads no column is refused, and so is one offset written twice (R's model frame reads `offset(o) + offset(o)` once; probed with `model.offset()`).

## `conditional_effects()`

brms's tables carry each expression term's model-frame column, named as R deparses the term, between the response and the raw
columns: `scale(x)` (a matrix of one column) and `poly(w, 2)` (two columns), held at the transform of the held raw values (each at
its mean) and unused by the estimates (`gauss_scale_poly_post`). AutoBayes's tables carry them the same way, a row's value an array
for `scale()` and `poly()`. A05's transforms keep D06's tables, without such a column (no frozen case shows brms's for them). A model
with `factor()` or a comparison is refused there (D182).

## Refused, with the construct named (D182)

`factor()` or a comparison in a formula without an intercept (R codes the first factor, a logical included, with one indicator per
level); an expression term as an operand of an interaction (`x:scale(w)`) or inside a group term; an expression term in a
distributional parameter's formula, a non-linear parameter's formula or a category's formula of the multinomial families; a fitted
transform (`scale()`, orthogonal `poly()`, `factor()`) in the formula of a response with `subset()` (fitted on which rows is not
verified); `conditional_effects()` of a model with `factor()` or a comparison; an expression that reads the response, the trials
column, a grouping factor or a declared categorical column.

## Evidence

Oracle cases (frozen with brms 2.23, fixtures #108): `gauss_I_expressions` (`I(x * w) + I(xpos^0.5) + I(x > 0)`, a coef prior on
`IxposE0.5`), `gauss_log_abs`, `gauss_scale_poly` and `gauss_scale_poly_post` (its `conditional_effects()` tables), `gauss_poly_raw`,
`gauss_factor_call`, `gauss_factor_call_newlevels` (new rows lacking a level, D181: `gates.check_newdata_refusal`) and
`poisson_two_offsets` pass gates G1-G6 and `post`; the syntax corpus's `pop_I_product`, `pop_I_fractional_power`, `pop_I_comparison`,
`pop_log_plus_one`, `pop_abs`, `pop_scale`, `pop_poly`, `pop_poly_raw`, `pop_factor_call`, `pop_two_offsets` and `pop_offset_expression`
match brms's `standata()`. `tests/test_g04_transforms.py` covers the grammar and its refusals, the spelling, every operator and
function against NumPy, the fitted transforms against hand-computed values and an independent QR construction, the term model,
priors, `describe()`, the graph, predictions on new rows by hand, `default_priors()` and `conditional_effects()`.
