# Simultaneous autoregressive terms: `sar(M, type = "lag")` and `sar(M, type = "error")`

brms-parity feature F02b (brms's spatial SAR terms; divergence rows D142 and D143).

```python
import autobayes as ab
priors = ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1), lagsar=ab.Flat())
model = ab.gaussian('y ~ x + sar(M, type = "lag")', data, priors, data2={"M": M})
print(model.describe())
```

`M` is an `N x N` spatial weight matrix over the rows of `data`, in data order (a NumPy array or a DataFrame, whose labels are
not matched), handed in `data2` under the name the formula gives it, as brms's `data2 = list(M = M)`.

## The model

With `mu` the linear predictor (brms's `posterior_linpred`), `A = I - rho M` and `e ~ Normal(0, sigma^2 I)`:

    lag:    A y = mu + e         y ~ MultiNormal(A^-1 mu, sigma^2 A^-1 A^-T)    rho is brms's lagsar
    error:  A (y - mu) = e       y ~ MultiNormal(mu, sigma^2 A^-1 A^-T)         rho is brms's errorsar

The log density is `log|A| - n log(sigma) - sum_i r_i^2 / (2 sigma^2)` with `r = A y - mu` (lag) or `r = A (y - mu)` (error) and
`log|A| = sum_k log(1 - rho lambda_k)` over the eigenvalues `lambda_k` of `M` (brms's `eigenMsar`). `rho` lies between the
smallest and the largest eigenvalue (brms's `real<lower=min_eigenMsar, upper=max_eigenMsar>`), under brms's flat prior there.
brms 2.23's `normal_lagsar_lpdf` and `normal_errorsar_lpdf` leave out the Gaussian constant `-n/2 log(2 pi)`, and so does the
model's log density, so that it is brms's (G2; the constant changes no posterior, but a marginal likelihood or bridge
sampling on brms's density is off by it; D142, upstream U36). `log_lik()` keeps it, as brms's does.

## How it is built

Bambi builds `mu` and `sigma` as for any Gaussian model (its `mu` deterministic is the linear predictor; the term never
reaches Bambi's formula) and hands them to the likelihood `sar.sar_family` puts in place of the Gaussian one: a
`pm.CustomDist` over the whole response vector (signature `(n),(),()->(n)`) whose log density is the expression above, and
which creates `rho` as a scalar of log density 0 on `(lower, upper)`, sampled on PyMC's interval coordinate
`log((rho - lower) / (upper - rho))` (Stan's lower-upper transform, whose Jacobian is rho's whole contribution, as in brms),
its initial value the interval's midpoint. `M @ y` and `M @ (y - mu)` are dense products and `log|A|` a sum over the
eigenvalues, computed once when the model is built (`sar.sar_design`: LAPACK's symmetric solver for a symmetric `M`, the
general one otherwise). Its forward draws are the joint model's.

## Using it

- Formula: `sar(M)`, `sar(M, type = "lag")`, `sar(M, type = "error")`, with R's argument rules (`sar(W, "error")`, named
  arguments in any order) and `match.arg`'s prefixes (`"err"`), as brms reads them; brms's default type is `"lag"`. One
  term per formula (brms refuses two too). The rest of the formula takes population-level terms: numeric and categorical
  predictors, interactions, transforms, with or without an intercept. `ModelSpec.sar` (`sar.Sar`) holds the term; the
  canonical formula writes it last, `y ~ x + sar(M, type = "lag")`.
- `data2={"M": M}`: real, finite, `N x N` (brms: "Dimensions of 'M' for SAR terms must be equal to the number of
  observations"), with real eigenvalues in `[-1, 1]` (a symmetric matrix, or a non-symmetric one whose eigenvalues are real to
  rounding, such as a row-standardised `D^-1 W`); `D^-1/2 W D^-1/2` and `D^-1 W` of an adjacency `W` with degrees `D` both
  qualify. An entry no term reads is refused. `Model.design.sar` (`sar.SarDesign`) holds the matrix and its eigenvalues.
- Priors: `Priors(lagsar=ab.Flat())` or `Priors(errorsar=ab.Flat())`, brms's classes `lagsar` and `errorsar` with their
  default, required exactly for the term's type. `ab.default_priors(formula, data, data2={"M": M})` returns it, and
  `prior_table()` writes brms's row (class `lagsar`, `lb` `min_eigenMsar`, `ub` `max_eigenMsar`). The flat prior is improper:
  `prior_predictive()` refuses the model.
- `value_point(..., lagsar=0.3)`: strictly between the bounds. Names: the PyMC scalar `lagsar` or `errorsar`
  (`model.sar_names`, `model.names["lagsar"]`), sampled as `lagsar_interval__`.
- `predict()` on the fitted rows: `kind="linear"` is `mu` (brms's `posterior_linpred`); `kind="mean"` is `A^-1 mu` under the
  lag model (brms's `posterior_epred`) and `mu` under the error model, the expected response there (brms's `posterior_epred`
  fails for it, "'a' is 0-diml"; D142); `kind="response"` draws `A^-1 (mu + e)` or `mu + A^-1 e` per posterior draw, seeded.
  New rows are refused (D143): `M` relates the fitted rows to each other.
- `ab.log_lik(model, fit)` is brms's `log_lik()` of this non-factorised model: each row's log density given every other row
  (Buerkner, Gabry and Vehtari 2021), `-log(2 pi)/2 + log(Q_ii)/2 - z_i^2 / (2 Q_ii)` with `Q = A^T A / sigma^2` and
  `z = A^T r / sigma^2`; `ab.loo` and `ab.waic` read it. `describe()`, the target identity (the matrix, its eigenvalues and the
  prior) and fit records hold the term.

## Refused, naming the construct (D143)

Families other than the Gaussian (brms also takes the Student-t), a mixture, the term next to group-level terms, an offset,
response addition terms (`weights()`, `cens()`, `trunc()`, `mi()`), `mo()`, `me()`, `mi()`, `gp()` or an autocorrelation term
(brms takes each; no frozen case verifies the pair), in a distributional formula or next to one (brms refuses it too: "SAR
models are not implemented when predicting 'sigma' or 'nu'"), in a multivariate or non-linear model, a shrinkage prior on `b`,
a prior on `lagsar` / `errorsar` other than `Flat()`, a matrix with complex eigenvalues or with eigenvalues outside
`[-1, 1]`, predictions on new rows, `bayes_R2()`, `conditional_effects()` and `prior_draws()`.

## Evidence

Four frozen cases pass gates G1-G6 and the post-processing gate against brms 2.23.0 on the symmetric normalisation of a
5 x 5 rook grid: `gauss_sar_lag` (`y ~ x + sar(M, type = "lag")`), `gauss_sar_error` (`type = "error"`; brms's
`posterior_epred` refused, recorded as such), `gauss_sar_lag_post` (`log_lik` and `loo`) and `gauss_sar_error_post`, frozen
with this feature (the error model's `log_lik` and `loo`). G1 compares brms's `Msar`, `eigenMsar`, `N_tg` and the parameter's
declared bounds with the model's (`tests/oracle/adapter.py`, `check_sar_design`); G2 maps brms's `lagsar` / `errorsar` to
`<name>_interval__`, checking Stan's lower-upper coordinate at every point and the probe's step. `tests/test_sar.py` covers the
parsing and every refusal, the matrix checks, the density against SciPy's multivariate normal with the covariance written out
(and `slogdet`), the gradient against a central difference of it, the JAX density, the flat prior and the initial point, the
forward draws (their recovered innovations standard normal), priors and defaults, the value point, names, `describe()`, the
identity and records, predictions (the expected values by a dense inverse, the draws' innovations the seeded normals),
`log_lik` against the conditional normal from the covariance's blocks, `loo`, `waic`, and corrupted fixtures. Nothing samples.

## Upstream (brms 2.23.0)

- The lag and error densities omit `-n/2 log(2 pi)` (22.9734633301 for n = 25 on the frozen cases; U36).
- brms bounds `rho` by the smallest and largest eigenvalue of `M`, while `log(1 - rho lambda)` needs `1/lambda_min < rho <
  1/lambda_max`: the two agree only when the eigenvalues lie in `[-1, 1]`. With the binary rook adjacency of the frozen cases'
  grid (eigenvalues in `[-3.4641, 3.4641]`) brms declares `lagsar` on `(-3.4641, 3.4641)` and `log_prob()` fails at
  `lagsar = 0.3` ("log1m: x is 1.03923, but must be less than or equal to 1"), as everywhere outside `(-0.2887, 0.2887)`.
- A row-standardised `D^-1 W` has real eigenvalues, but R's `eigen()` returns the repeated zeros as complex pairs (|Im| 3.6e-17)
  and rstan refuses the complex `eigenMsar`.
- `posterior_epred()` of the error model fails ("'a' is 0-diml"); its expected response is `mu`.
