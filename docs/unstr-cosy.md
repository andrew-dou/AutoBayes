# Residual correlation over time: `unstr()` and `cosy()`

brms-parity feature F03b (brms's `unstr(time, gr)` and `cosy(time, gr)`).

```python
import autobayes as ab
priors = ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1), cortime=ab.LKJ(2))
model = ab.gaussian("y ~ x + unstr(time = tt, gr = subj)", data, priors)
cosy = ab.student("y ~ x + cosy(time = tt, gr = subj)", data,
                  ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1), cosy=ab.Flat(),
                            nu=ab.Gamma(2, 0.1)))
print(model.describe())
```

## The model

The rows form groups, the levels of `gr`, each ordered by `time`. Residuals `e = y - mu` of different groups are
independent; those of group `g` are jointly

    e_g ~ MultiNormal(0, sigma^2 Cor_g)              Gaussian
    e_g ~ MultiStudentT(nu, 0, sigma^2 Cor_g)        Student-t (scale matrix sigma^2 Cor_g, one nu)

so each row on its own still has the family's `Normal(mu_i, sigma)` or `StudentT(nu, mu_i, sigma)`. `mu` is the linear predictor
of every other term.

- `unstr`: `Cor_g` is made of the group's rows and columns of one correlation matrix `Cortime` over every distinct time of
  the fitted rows (sorted; brms's `n_unique_t`). A group may have any subset of the times (brms's `Jtime_tg`). `Cortime` is
  sampled through its Cholesky factor `Lcortime` on Stan's `cholesky_corr` coordinates under `lkj_corr_cholesky(eta)`.
- `cosy`: compound symmetry, every pair of rows in a group correlated `cosy` in (0, 1). Time only orders the rows. A group of
  `k` rows uses the leading `k` by `k` block of the Cholesky factor of the largest group's matrix (brms's `Lcortime`).

## How it is built

Bambi builds the linear predictor from the formula without the term. The family's likelihood is replaced by one
`CustomDist` over all fitted rows (`timecor.timecor_family`). It creates `Lcortime` and the deterministic `Cortime`, or
`cosy`, inside the model Bambi builds. Its log density (`timecor.log_density`) works pattern by pattern: groups that share
their rows of the correlation matrix share a Cholesky factor. That factor is the leading block of `Lcortime` (`Lcortime`
itself for a group with every time), or the Cholesky factor of the group's rows and columns of `Cortime`. Each group's
residuals are solved against the factor, and the family's kernel (`FamilySpec.time_correlation`: `timecor.NORMAL_TIME`,
`timecor.STUDENT_TIME`) turns their quadratic forms and log determinants into the multivariate density. PyTensor's JAX
backend compiles all of it. `Design.timecor` (`timecor.TimeCorDesign`) holds the groups: their sorted labels, each row's group
and row of the correlation matrix, brms's Stan order (sorted by group, then time), `nobs_tg`, `Jtime_tg` and the patterns.

## Using it

- Formula: one term per formula, `unstr(time = tt, gr = subj)` or `cosy(tt, subj)`, with R's argument rules. Both columns
  must be named. Population-level terms, group-level terms (including correlated ones), `mo()` and `offset()` can sit next to
  the term. The time column may also be a predictor, and the grouping column a grouping factor. The time column must be
  numeric and finite, the grouping column labels without missing values, and two rows of one group at the same time are
  refused (brms refuses them too). `ModelSpec.timecor` (`timecor.TimeCor`) holds the term. The canonical formula writes it
  last; Bambi's formula leaves it out.
- Priors: `Priors(cortime=ab.LKJ(eta))` for `unstr` (brms's class `cortime`, default `lkj(1)`) and `Priors(cosy=ab.Flat())`
  or `ab.Beta(a, b)` for `cosy` (brms's class `cosy`, `real<lower=0,upper=1>`, default flat). Each is required exactly for its
  term (D1). The flat prior on `cosy` is uniform on (0, 1), which is proper, so `prior_predictive()` draws from it.
  `ab.default_priors()` returns brms's defaults, and `prior_table()` writes brms's `cortime` row (no bounds) or `cosy` row
  (`lb` 0, `ub` 1).
- `value_point(..., Lcortime=L)` (a correlation Cholesky factor over the sorted times), `value_point(..., cosy=0.4)`.
- Names: `Lcortime`, `Cortime` (dims `cortime_dim`, `cortime_dim_T`, the times as coordinates) or `cosy`
  (`model.timecor_names`; `model.names["Lcortime"]`, ...).
- `predict()` on the fitted rows: the linear predictor and the expected response are `mu`, as brms's `posterior_linpred` and
  `posterior_epred` are. `kind="response"` draws each group's rows jointly (`mu + sigma L z`, for the Student-t divided by
  `sqrt(chi^2_nu / nu)` per group), as brms's `posterior_predict` does. New rows are refused (D146).
- Post-processing: `ab.log_lik` gives each row's log density given its group's other rows. That is brms's `log_lik()` of a
  non-factorised model (Bürkner, Gabry and Vehtari 2021, *Computational Statistics* 36): with `P = (sigma^2 Cor_g)^-1`,
  `g = P e` and `c = diag(P)`, row `i` is `Normal(e_i | e_i - g_i / c_i, 1 / sqrt(c_i))`. For the Student-t it is the
  conditional of the multivariate t, `StudentT(nu + k - 1, e_i - g_i / c_i, sqrt((nu + q_-i) / (nu + k - 1) / c_i))`, with
  `q_-i = e' P e - g_i^2 / c_i`. `ab.loo` and `ab.waic` read it. `ab.bayes_R2` is `var(mu) / (var(mu) + var(y - mu))`, with
  each row's response paired with its own mean (D147).
- `describe()` writes out the groups, the joint density and the prior. The target identity holds the term, each row's
  group and row of the correlation, the times and the prior. Fit records list `time_correlation` and the prior.

## Refused, naming the construct (D146)

Families other than the Gaussian and the Student-t; a distributional formula next to the term (brms's heterogeneous
`sigma`) and the term inside one (brms refuses that too); response addition terms; the term next to `gp()`, `me()`, `mi()`,
an ARMA term, `sar()`, `car()`, `fcor()`, `mm()` or `gr(by = )`; two such terms; `cosy()` without `time` or `gr` (brms's defaults: the data order, one
group); a `gr` of two columns; a non-numeric time; a shrinkage prior on `b`; multivariate formulas (`mvbind()` and lists of
formulas); a non-linear parameter's formula and a mixture (brms refuses both too); `conditional_effects()`, `prior_draws()`
and predictions on new rows. The priors `cortime` and `cosy` are refused in every model without the term.

## Evidence

Six frozen cases pass gates G1-G6 against brms 2.23.0, to about 1e-15 relative in the log density:

- `gauss_unstr`, `gauss_cosy`, `student_cosy`: a balanced panel.
- `gauss_unstr_post`: `log_lik` and `loo`.
- `gauss_unstr_site_varint_post`, frozen with this feature: `unstr(time = tt, gr = site)` next to `(1 | h)` on 25 groups of
  one or two rows. The rows are in neither site nor time order, and the groups have every pair of times and three single
  times, which exercises brms's row order, `Jtime_tg`'s zero padding and the sub-correlations. It also checks `log_lik` and
  `loo`.
- `student_cosy_post`, frozen with this feature: six groups of 3 to 9 rows, not contiguous in the data, and the Student-t's
  conditional `log_lik` and `loo`.

The oracle derives brms's row order from the data, checks each of brms's groups against the model's (`N_tg`, `begin_tg`,
`end_tg`, `nobs_tg`, `n_unique_t`, `n_unique_cortime`, `Jtime_tg`), maps `Lcortime` and `cosy` to their coordinates and
compares the graph's `Cortime` (and `cosy`'s Cholesky factor) with brms's.

`tests/test_unstr_cosy.py` covers:

- the parsing and every refusal;
- the groups and patterns;
- the compound-symmetric Cholesky factor against NumPy's;
- both families' densities against SciPy's multivariate normal and t, the gradient against central differences, and JAX
  against PyTensor;
- `log_lik` against the joint density less the other rows' marginal;
- joint draws against their covariance;
- priors, defaults, `value_point`, predictions, `describe()`, the identity, records and prior-predictive draws;
- the oracle's extension on corrupted fixtures.

Nothing samples.

brms's `bayes_R2()` of `gauss_unstr_site_varint_post` (recorded in its fixture) pairs each row's response with another row's
mean (D147), so the case compares `log_lik` and `loo` only. The test above pins brms's number to the misaligned formula and
`ab.bayes_R2` to the aligned one.
