# Fixed residual covariance: `fcor(M)`

brms-parity feature F03a (brms's `fcor()`, a known covariance matrix of the residuals).

```python
import autobayes as ab
priors = ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1))
model = ab.gaussian("y ~ x + (1 | g) + fcor(V)", data, priors, data2={"V": V})
print(model.describe())
```

## The model

`V` is data: an `N` by `N` matrix for the `N` fitted rows, handed over by name in `data2` (brms's `data2 = list(V = V)`). The
rows are jointly normal with that covariance scaled by the residual variance,

    y ~ MultiNormal(mu, sigma^2 V),

`mu` the linear predictor with every population- and group-level term and the offset: brms 2.23's
`normal_fcor_hom(Y | mu, sigma, Lfcor)` with `Lfcor = cholesky_decompose(Mfcor)`. The term has no parameter and no prior class
of its own.

## How it is built

Bambi is handed the formula without the term, and the Gaussian likelihood is replaced by a `pymc.CustomDist` over the rows
(`fcor.fcor_family`). With `L` the lower Cholesky factor of `V` (computed once, in NumPy) and `z = L^-1 (y - mu) / sigma` (a
triangular solve), the log density is `sum_i (-z_i^2 / 2 - log sigma - log L_ii - log(2 pi) / 2)`; row `i`'s term is the
density of `y_i` given the rows before it, so the likelihood keeps PyMC's elementwise shape. PyTensor's `solve_triangular` lowers to JAX. When the model is built, the graph's likelihood at a test point is checked
against the dense multivariate normal (`fcor.fcor_values`, the log determinant and `r' V^-1 r` of the full matrix).
`Design.fcor` (`fcor.FcorDesign`) holds `V` and `L`.

## Using it

- Formula: one `fcor(M)` or `fcor(M = V)` per formula, `M` a name in `data2`; any population-level term the term model takes,
  group-level terms (varying intercepts and slopes, correlated or not) and `offset()` stand next to it. The canonical formula writes it
  last; Bambi's formula does not have it.
- `data2`: a mapping of names to matrices (a `pandas.DataFrame`, a NumPy array or nested lists), read by position: row `i` is
  the `i`-th fitted row whatever the labels say, as in brms. A vector is the diagonal matrix of those variances, as in brms.
  Refused, as brms or Stan refuse them: a matrix of another size ("Dimensions of 'M' for FCOR terms must be equal to the number
  of observations"), an asymmetric one (to 100 machine epsilons of its largest entry; "'M' for FCOR terms must be symmetric"),
  one that is not positive definite (Stan's `cholesky_decompose`), missing or non-finite entries and a name not in `data2`;
  and a name in `data2` that no term reads (brms ignores it). `ab.default_priors(formula, data, data2=...)` reads it the same way;
  the term adds no prior class, as in brms (probed).
- `predict()` on the fitted rows: `kind="linear"` and `kind="mean"` are the linear predictor without the term (brms's
  `posterior_linpred` and `posterior_epred`), with or without group effects; `kind="response"` draws the rows jointly,
  `mu + sigma L z` with `z` standard normal from the seeded generator (brms's `posterior_predict`). New rows are refused
  (D145). `prior_predictive()` draws the rows jointly too.
- `describe()` writes the matrix's size, diagonal and log determinant and the joint likelihood; the target identity holds the
  matrix; fit records list `fcor` (the term, the name of its matrix and its size). `hypothesis()` reads the coefficients and is
  offered.

## Refused, naming the construct

Families other than the Gaussian (brms takes `student()` too, D144; it refuses the others: "FCOR terms are not implemented
for this family"); a second `fcor()` term (brms refuses it too: "Can only model one term per autocorrelation class"); the term next to a distributional formula (brms's `normal_fcor_het`, D144), inside
one and in a mixture (brms refuses these too: "Explicit covariance terms can only be specified on mu"); with response addition
terms (brms refuses `weights()` too); next to an ARMA term, a SAR term `sar()` (F02b), `mo()`, `gp()`, `me()`, `mi()`, `mm()` or `gr(by = )` (D144); in a
multivariate formula (brms gives each response its own `Mfcor_y`, D144) and in a non-linear model (D144); a shrinkage prior on
`b` (D102); predictions on new rows, `log_lik()`, `loo()`, `waic()`, `bayes_R2()` (D145), `conditional_effects()` (D99) and
`prior_draws()` (D124).

## Evidence

Two frozen cases pass gates G1-G6 against brms 2.23.0: `gauss_fcor` (`y ~ x + fcor(V)`, `V` the covariance `s_i s_j 0.5^|i - j|`
of the 36 rows) and `gauss_fcor_varint` (`y ~ x + (1 | g) + fcor(V)`, frozen with the feature: predictions with and without the
group effect). G1 compares brms's `Mfcor` with the model's matrix entry by entry and checks `N_tg = 1`
(`tests/oracle/adapter.py`, `check_fcor_design`). `tests/test_fcor.py` covers the parsing, `data2` and every refusal, the
likelihood's rows against the sequential conditionals written by hand and its sum against SciPy's multivariate normal, the
gradient against the analytic one, the JAX density, the build-time check, default priors, predictions and joint draws,
`describe()`, the identity, records, prior-predictive draws, the post-processing refusals and corrupted fixtures. Nothing samples.
