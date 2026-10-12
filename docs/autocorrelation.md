# Autocorrelation terms: `ar()`, `ma()`, `arma()`

brms-parity feature C04 (brms's ARMA terms in their regression formulation, `cov = FALSE`).

```python
import autobayes as ab
priors = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1),
                   ar=ab.Normal(0, 0.5), ma=ab.Normal(0, 0.5))
model = ab.gaussian("y ~ x + arma(time = t, gr = g, p = 1, q = 1)", data, priors)
print(model.describe())
```

## The model

The rows form series, the levels of `gr` (one series without it), each ordered by `time` (the data order without it).
With `eta_i` row `i`'s linear predictor without the term (every population-level, group-level, monotonic and offset term)
and `E_i[k]` the residual `e` of the row `k` places before `i` in its own series (0 when the series has no such row):

    mu_i = eta_i + sum_{k <= q} ma[k] E_i[k] + sum_{k <= p} ar[k] E_i[k]
    e_i  = y_i - eta_i - sum_{k <= q} ma[k] E_i[k]
    y_i ~ Normal(mu_i, sigma)

so `e` is the residual of the predictor with its MA part (`y - eta` when `q = 0`) and the AR part reads the same lagged
residuals: brms 2.23's Stan model, with its `Err` matrix filled through `J_lag`. Lags count rows of a series, not units of
`time`: a gap in `time` is one step, as in brms. `ar` (`p` entries, brms's `Kar`) and `ma` (`q`, brms's `Kma`) are
unbounded vectors, as brms 2.23 declares them in this formulation: `|ar| < 1` is not imposed, so the AR part need not be
stationary.

## How it is built

Bambi has no ARMA term: it is handed the formula without it, and the term is added to the parent's linear predictor while
Bambi builds the model, through the family's `transform_backend_eta` hook (`autocor.autocor_family`), after every other
term (a monotonic term's hook runs first). The model creates `ar` and `ma` under their priors and adds the term on the
residuals of the fitted responses (`autocor.arma_effect`). The rows are laid out by position and series, a `(T, G)` table
with a zero where a series is shorter (`AutocorDesign.cells`); without an MA part the lags are that table shifted down, and
with one the residuals solve `e_t = r_t - sum_k ma[k] e_{t-k}` by a `pytensor.scan` over the positions with every series
at once, which PyTensor's JAX backend lowers to `lax.scan` (so a single long series costs one step per row; many short
series are cheap). `autocor.arma_values` is the same model in NumPy written as brms writes it, row by row along brms's
sorted rows with `J_lag`; when the model is built, the graph's `mu` at test coefficients is checked against it, row by row,
on the predictor at `ar = ma = 0` with every other variable moved off its initial point.

`Design.autocor` (`autocor.AutocorDesign`) holds the series of the fitted rows: their sorted labels, each row's series and
0-based position in time order (data order), brms's Stan-data order (`order`: sorted by series, then time) and brms's
`J_lag` on those rows.

## Using it

- Formula: one term per formula, `ar(time = t, gr = g, p = 1)`, `ma(time = t, gr = g, q = 1)` or
  `arma(time = t, gr = g, p = 1, q = 1)`, with R's argument rules (`ar(t, g, 2)`, `arma(t, g)`, named arguments in any
  order) and brms's defaults (`time = NA`: the data order; `gr = NA`: one series; `p = 1`, `q = 1`). Any population-level
  and group-level terms the term model takes stand next to it (`y ~ t + x + mo(ord) + (1 + x || h) + arma(t, g)`), as do
  an offset and `0 +`. The time column may also be a predictor (a trend), the series column a grouping factor or a
  categorical predictor. `ModelSpec.autocor` (`autocor.Autocor`) holds the term; the canonical formula writes it last, and
  Bambi's formula does not have it. The time column is numeric and finite, the series column labels without missing
  values, and two rows of a series at the same time are refused (brms refuses them too).
- Priors: `Priors(ar=..., ma=...)`, one location prior (`Normal`, `StudentT`, `Flat`) for every entry of the vector, as
  brms's classes `ar` and `ma` (no bounds); required exactly for the parts the term has (D1). brms's default is flat:
  `ab.default_priors()` returns `ar=ab.Flat()`, `ma=ab.Flat()`, and `prior_predictive()` then refuses, as for any flat prior.
- `value_point(..., ar=[0.5], ma=0.3)`: `p` and `q` numbers, unconstrained.
- Names: the PyMC vectors `ar` and `ma` (`model.names["ar"]`, `model.autocor_names`), dims `ar_dim` and `ma_dim` with
  coordinates `"1"`..; a posterior lists `ar[1]`, `ma[1]`, as brms does.
- `predict()` on the fitted rows adds the term from the observed residuals, as brms's `posterior_linpred` and
  `posterior_epred` do; without group effects (`include_group_effects=False`) the residuals are those of the predictor
  without them, as under brms's `re_formula = NA`; `kind="response"` draws `Normal(mu_i, sigma)` around the predictor with
  the term. New rows are refused (D71). A model with the term is always predicted in this layer.
- `describe()` writes the series, the term and its residual out; the target identity holds the term, each row's series and
  position and the coefficients' priors; fit records list `autocorrelation` and the priors `ar` and `ma`;
  `prior_table()` writes brms's `ar` and `ma` class rows. A coefficient no row reads (lag `k` needs a series of more than
  `k` rows) is named in a warning.

## Refused, naming the construct

`cov = TRUE`, brms's covariance formulation (D68); `cosy()`, `car()`, `sar()`, `fcor()`, `unstr()` (D69); families other
than the Gaussian, an autocorrelation term next to or inside a distributional formula, with a response addition term
(`weights()`, `cens()`, `trunc()`), a `gr` of two columns (`g:h`), a non-numeric time column (D70); predictions on new rows
(D71); two autocorrelation terms in one formula, `p = q = 0`, a non-integer or negative order, an argument the function
does not have, the response or the trials column as `time` or `gr`, `time` equal to `gr` (brms refuses these too, or they
are not R).

## Evidence

Three frozen cases pass gates G1-G6 against brms 2.23.0 at about 1e-15 relative: `gauss_ar1`
(`y ~ x + ar(time = t, gr = g, p = 1)`), `gauss_arma11` (`arma(..., p = 1, q = 1)`), and `gauss_arma22_varint`
(`y ~ x + (1 | h) + arma(time = t, gr = g, p = 2, q = 2)`, frozen with this feature: lags two deep, a series of one row,
predictions with and without the group effects). The oracle derives brms's row order from the data's `(g, t)` as R's
`order()` sorts them, requires brms's `Y` to be the response under it, puts every per-row Stan array back in data order
(`tests/oracle/adapter.py`, `in_data_order`), and checks `Kar`, `Kma`, `J_lag` and each sorted row's series and position;
G2 maps brms's `ar` and `ma` to the model's. `tests/test_autocorrelation.py` covers the parsing and refusals, the series
against brms's own Stan data, the NumPy and graph recursions against an independent transcription of the model for orders
up to `p = 3`, `q = 3`, the density and its gradient, the JAX density and gradient, priors, defaults, predictions,
`describe()`, identity, records, prior-predictive draws and corrupted fixtures. Nothing samples.
