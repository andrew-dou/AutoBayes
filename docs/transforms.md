# In-formula transforms: `log(x)`, `exp(x)`, `sqrt(x)`, `I(x^k)`, `I(x * k)`, `I(x + k)`, `I(x - k)`

brms-parity feature A05. A population-level term may be a transform of one numeric column written in the formula,
as R and brms allow: `y ~ log(xpos) + I(x^2) + (1 | g)`. The transform is a term of its own (`PopTerm` of kind
`transform`), with one design column, one coefficient and one prior, named as written: `log(xpos)`, `I(x^2)`.

## The model

For a transform `f` of column `c`, the design column is `f(c_i)` on the fitted rows, centred like every other
column when the model is centred (`mean(f(c))` enters the raw-scale intercept correction with it), with the
coefficient `b_f(c)` the slope in `f(c)`, not in `c`. Predictions on new rows recompute `f` from the raw column,
which `newdata` must carry (`xpos`, `x`; refused otherwise). R evaluates the same expressions into the design
matrix; brms names the coefficients with its `rename()` of the term (`logxpos`, `IxE2`, `expx`, `sqrtxpos`,
`IxMU2` for `I(x * 2)`, `IxP1` for `I(x + 1)`, `IxM1` for `I(x - 1)`, `I2MUx` for `I(2 * x)`, `IxEM1` for
`I(x^-1)`), which `describe()` quotes; the oracle matches the columns by content.

## Supported forms

| written | value | notes |
| --- | --- | --- |
| `log(c)` | `log(c_i)` | `c` must be positive on every row (fitted and new) |
| `exp(c)` | `exp(c_i)` | the result must be finite |
| `sqrt(c)` | `sqrt(c_i)` | `c` must be positive on every row |
| `I(c^k)` | `c_i^k` | integer `k` (`I(x^2)`, `I(x^-1)`); the result must be finite |
| `I(c * k)`, `I(k * c)` | `k c_i` | numeric constant `k` |
| `I(c + k)`, `I(k + c)`, `I(c - k)` | `c_i + k`, `c_i - k` | numeric constant `k` |

Whitespace inside the call does not matter; the term is spelled canonically (`log(xpos)`, `I(x^2)`, `I(x * 2)`,
`I(2 * x)`, `I(x + 1)`), and the same transform written twice is a duplicated predictor. The constant is kept as
written, so `I(x * 2)` and `I(x * 2.0)` are different terms here (R deparses both as `I(x * 2)`) with equal
columns, the same collinearity as two copies of one column.

## How it is built

Bambi/formulae know `log` but not `sqrt`, and evaluate `I()` as Python (`^` is not a power), so the library
evaluates every transform itself: the fitted frame handed to Bambi carries the evaluated column under the term's
own name (`log(xpos)`, `I(x^2)`), the formula names it in backticks (`` `log(xpos)` ``), and Bambi's term, PyMC
variable and posterior row are therefore named exactly as the term is written. `Model._extract_design` recomputes
the column from the raw data (`pop_term_design`) and raises if Bambi's column differs; `predict()` adds the
evaluated columns to `newdata` for Bambi's path and recomputes them for its own.

## Using it

- Formula: `y ~ log(xpos) + I(x^2) + (1 | g)`, with or without the raw column as a term too (`x + I(x^2)`); a
  transform may share its column with the offset (`log(exposure) + offset(log(exposure))`).
- Priors: `b` keyed by term, `b={"log(xpos)": Normal(0, 2), "I(x^2)": Normal(0, 1)}` (brms `class = b, coef =
  logxpos`); one prior per transform.
- `describe()` lists each transform with its range over the fitted rows and brms's name, writes it in the linear
  predictor (`(log(xpos_i) - mean(log(xpos))) * b_log(xpos)`), and says the coefficient is the slope in the
  transformed quantity. The fit record lists the transform terms under `transforms`; the target identity includes
  the term and its column values.

## Refused, with the construct named

The grammar of docs/expressions.md reads every other expression of a whitelisted grammar (`I(x * w)`, `I(x^2.5)`, `log(x + 1)`, `exp(-x)`,
`abs(x)`) and the fitted transforms `scale()`, `poly()` and `factor()`; A05's forms above keep their own checks. Still refused:
any other call (`log1p(x)`, `log(x, 2)`, `np.log(x)`, `sin(x)`: the message lists what is read; compute the column in the data
frame and name it instead), a transform of the response, the trials column, a grouping factor or a declared categorical
column, a transform inside a group term or as an operand of an interaction (`log(x):z`), a column that is not
numeric, missing or non-finite values, non-positive values under `log()` or `sqrt()`, a non-finite result
(`exp` overflow, `I(x^-1)` at zero), a transformed column that is constant in the fitted rows, and `newdata`
without the raw column.

## Evidence

Oracle case `gauss_transforms` (`y ~ log(xpos) + I(x^2) + (1 | g)`, brms columns `logxpos` and `IxE2`) passes gates
G1-G6 against its frozen brms fixture: the design columns matched by content, the log density and gradient with and
without the Jacobian, `b_Intercept`, `posterior_linpred`/`posterior_epred` on fitted and new rows (recomputed from
`xpos` and `x` in `newdata.csv`), the JAX density. The adapter and gates needed no change. `tests/test_transforms.py`
covers the refusals, the term model, every supported form against NumPy, the design, the graph and density at a
point, by-hand and Bambi-path predictions, the fit record and `describe()`. Nothing samples.
