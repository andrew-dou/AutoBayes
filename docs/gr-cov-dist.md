# A known covariance, Student-t effects and multi-membership slopes

brms's `gr(g, cov = A)` (a known covariance of a grouping factor's levels, as in phylogenetic and kinship models),
`gr(g, dist = "student")` (Student-t group-level effects), varying slopes in `mm()`, `mmc(x1, x2)` and `mm(..., scale = FALSE)`.
The library builds each of these terms itself, next to the terms Bambi builds, as it builds `mm()` and `gr(by = )`
(docs/multi-membership-and-by.md).

```python
import autobayes as ab
priors = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1),
                   sd={"g": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(1)}}, cor={"g": ab.LKJ(2)})
model = ab.gaussian("y ~ x + (1 + x | gr(g, cov = A))", data, priors, data2={"A": A})   # A: a DataFrame labelled by level
student = ab.gaussian('y ~ x + (1 | gr(g, dist = "student"))', data,
                      ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1), sd=ab.HalfNormal(1),
                                df={"g": ab.Gamma(2, 0.1)}))
print(model.describe())
```

## `gr(g, cov = A)`: a known covariance of the levels

`A` comes in `data2`, as `car()`'s adjacency matrix does: a square pandas DataFrame whose index labels the levels of `g` (the
columns the same labels in the same order, or positions), symmetric, holding finite numbers, labelling every fitted level and
positive definite on them. Labels that are no fitted level are dropped with a warning (brms drops them silently). `Lcov` is the
lower Cholesky factor of `A` on the levels, in the sorted level order (brms's `Lcov_m`, `Design.group_cov[g].L`).

    one coefficient     u = sd * Lcov z,                          z ~ Normal(0, I)   so   u ~ Normal(0, sd^2 A)
    several             vec(r) = (Lcov kron diag(sd) L) vec(z),   level by level       so   vec(r) ~ Normal(0, A kron Sigma)

`Sigma = diag(sd) L L^T diag(sd)` is the coefficients' covariance (`L` under `lkj_corr_cholesky`, brms's `L_m`, with `|`; the
identity with `||`). Each coefficient's effects over the levels are `Lcov` times the effects the term would have without `cov`.
brms 2.23.0 builds the case of several correlated coefficients differently: it drops the negative entries of `Lcov`, so that its
effects have covariance `Lcov+ Lcov+^T kron Sigma`, not `A kron Sigma`, whenever chol(A) has a negative entry (a relationship
matrix with small negative entries has some). The library builds the Kronecker product (divergence D185); with a non-negative
Cholesky factor, one coefficient or `||` the two agree.

The model's variables are those of the same term without `cov` (`1|g_sigma`, `1|g_offset`, `x|g_sigma`, `x|g_offset`, `L|g`, the
effects `1|g` and `x|g`, `cor|g`); priors, `value_point` and the default priors are unchanged, and `cov` adds no prior class.

## `gr(g, dist = "student")`: Student-t effects

    u[j] = dfm[j] * sd * z[j],   dfm[j] = sqrt(df * udf[j]),   udf[j] ~ InverseGamma(df / 2, 1 / 2),   z[j] ~ Normal(0, 1)

`InverseGamma(df / 2, 1 / 2)` is Stan's `inv_chi_square(df)`, so each level's effect is `Student-t(df, 0, sd)`; with slopes each
level's vector of effects is multiplied by its `dfm[j]` (a multivariate Student-t, brms's `rep_matrix(dfm, M) .* scale_r_cor`).
The degrees of freedom `df|g` (brms's `real<lower=1> df_m`) take the prior class `df` (brms's class `df` with `group = g`, default
`gamma(2, 0.1)`): `Priors(df={"g": ab.Gamma(2, 0.1)})`, or the bare prior for one such term, one of `Gamma`, `Exponential`,
`HalfNormal` and `HalfCauchy`, truncated to `df > 1` and renormalised there as brms's lprior is (as a learned Student-t `nu`'s,
docs/student-learned-nu.md). The variables are `df|g` (sampled as `log(df - 1)`), `udf|g` (on their logs, `g__factor_dim`) and
the deterministic `dfm|g` (`Model.group_dist_names`); `value_point(df={"g": 4.0}, udf={"g": array(J)})`.

## Varying slopes in `mm()`, `mmc()` and `scale = FALSE`

    mu_i = ... + sum_k W_k,i * (u[J_k,i] + sum_c Z_c,k,i * v_c[J_k,i])

`(1 + x | mm(g, g2))`: the slope's covariate `Z_c,k` is `x` for every member, correlated with the intercept under `|` and
independent under `||`; `x` is a numeric column, a population-level term or not (as in any group-level term; without one, the
population-level slope is 0). `(1 + mmc(x1, x2) | mm(g, g2))`: the
coefficient `mmc(x1, x2)` (brms's label `mmcx1x2`, as in its prior table) multiplies member `k`'s own column, one column per
member. `mm(g, g2, weights = cbind(w1, w2), scale = FALSE)`: the weights as given, not divided by their sum on each row (negative
weights are taken, as brms takes them); without weights the members weigh `1 / members` either way. Priors key the coefficients
as written: `sd={"mmgg2": {"Intercept": ..., "mmc(x1, x2)": ...}}`, `cor={"mmgg2": ab.LKJ(2)}`.
`Design.group_blocks[("mmgg2", c)]` holds each coefficient's block, `covariates` its covariate per member.

## Refused, with the construct named

Two of `by`, `cov` and `dist` in one `gr()`; `cov` or `dist` with a correlation ID; `mm()`'s other arguments (`cov`, `dist`, `id`,
`cor`, `by`); a multi-membership, `cov` or `dist` term without its varying intercept, and a factor, transform, product or
monotonic slope in one (docs/group-term-syntax.md's slopes are verified in the terms Bambi builds); `mmc()` outside `mm()` (brms refuses it too) or with a
number of columns other than the members'; these terms next to `mo()`, `gp()`, `me()`, `mi()`, an autocorrelation or addition
term or a distributional formula, in multivariate and non-linear models and for families whose link is not identity or log (as
for `mm()` and `gr(by = )`); new levels of a `cov` or `dist` term under every `new_groups` policy (a level outside `A` has no
covariance with the fitted ones; brms's draws of new Student-t levels are not verified); `prior_draws()` and the summaries of
these models (divergence D186).

## Evidence

The oracle cases `gauss_gr_cov`, `gauss_gr_cov_slope`, `gauss_gr_student`, `gauss_mm_slope`, `gauss_mmc` and
`gauss_mm_scale_false` pass gates G1-G6 against frozen brms 2.23 fixtures (G1 compares brms's `Lcov_m`,
`gates.check_group_covariance`). `gauss_gr_cov_slope_signed` (chol(A) with five negative entries) passes G1, G4, G5 and G6 and
keeps strict expected failures on G2 and G3, where brms's construction differs (D185). `tests/test_gr_cov_dist.py` checks the
term model and every refusal, the reading of `A`, the Kronecker product on the graph's map from `z` to the effects and the
Gaussian density of `vec(r)` under `A kron Sigma`, the signed case's log density at every frozen point against both
constructions, the Student-t density with df's truncated prior and the scale mixture's density, the multi-membership slopes and
raw weights against hand computations, predictions by hand, the default priors, `describe()` and the target identity. Nothing
samples.
