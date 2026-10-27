# Dropping rows with missing values, on request

brms drops every row with a missing value in a variable the model reads, except the variables modelled with `mi()`, and warns
'Rows containing NAs were excluded from the model.'. AutoBayes refuses such data by default, naming the columns, and drops the
rows when asked (divergence D174, changed by Andrew's decision of 26 October 2026):

```python
m = ab.gaussian("y ~ x + xm + (1 | g)", data, priors, dropna=True)   # warns; the model of the complete rows
m.dropped_na_rows                                                     # their positions in data (0-based), e.g. (0, 7, 8)
priors = ab.default_priors("y ~ x + xm + (1 | g)", data, dropna=True)  # the defaults of the same rows
```

**Why `dropna`.** It is the keyword of pandas's `DataFrame.dropna` and of Bambi's `Model` for the same act, so a reader knows
what it does; it is off by default because which rows a model describes is part of the model, and the drop changes `N`, the
centring of the predictors, the factor levels, the default priors' location and scale, and the rows `predict()` returns.

**What is dropped.** A row with a missing value (NaN, None, `pd.NA`) in a column the model reads: the response, the predictors
(the raw columns of transforms), the grouping factors, the trials, the offset, the addition terms' columns and the columns of
every distributional formula. A column the model does not read drops nothing, and in a multivariate model a column whose missing
values `mi()` models (a response written `col | mi() ~ ...`) keeps its rows, as brms keeps them; one set of rows is dropped for
every response. The rest stay in data order. The model warns (Python's `UserWarning` and `model.warnings`, which `describe()`
prints) with brms's sentence, the number dropped, the columns and the positions, and records the positions as
`model.dropped_na_rows`; a binomial model's zero-trial rows (`model.dropped_rows`) are still counted as positions in the data as
given. Without missing values `dropna=True` changes nothing.

**After the drop.** The model is the model of the rows that remain: its target identity is that of a model built on them.
`predict()` on the fitted rows, `log_lik`, `loo` and `bayes_R2` cover those rows. New rows with missing values are refused, naming
the columns (`dropna` drops fitted rows only).

**Not offered** (D177): a non-linear model (`nl=True`) and the count-matrix families refuse `dropna=True`; a model with a term
over the data's rows as a whole (an `ar()`, `ma()`, `arma()`, `unstr()` or `cosy()` term's series, `sar()`'s and `fcor()`'s `data2`
matrices, `car()`'s locations) or a multivariate model with `subset()` refuses it when a row would be dropped, naming the term:
drop those rows in the data first (with the matching rows of any `data2` matrix).

**Evidence.** The oracle cases `gauss_missing_rows_dropped` (7 of 36 rows dropped) and `gauss_missing_rows_dropped_post` (brms's
`log_lik`, `bayes_R2` and `loo` over the 29 rows that remain) pass every gate; the syntax corpus entries `pop_missing_predictor`
and `resp_missing` (5 of 40 rows) match brms's `standata()`; `tests/test_call_surface.py` checks the drop, the warning, the
positions, the target identity against a model of the complete rows, the default priors, `mi()`'s rows and the refusals.
