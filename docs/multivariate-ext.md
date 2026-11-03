# Multivariate extensions: a formula per response with `rescor`, IDs across responses, `weights()`

brms-parity feature G08 (divergence rows D193 and D194). It extends C01's multivariate Gaussian models (docs/multivariate.md)
and F06c's formula per response (docs/subset.md).

```python
import autobayes as ab
resp = {"y": {"intercept": ab.Normal(0, 5), "b": ab.Normal(0, 2), "sd": ab.HalfNormal(1), "sigma": ab.HalfNormal(1)},
        "y2": {"intercept": ab.Normal(0, 5), "b": ab.Normal(0, 2), "sd": ab.HalfNormal(0.5), "sigma": ab.HalfNormal(1)}}
ids = ab.gaussian("mvbind(y, y2) ~ x + (1 |p| g)", data, ab.Priors(resp=resp, cor={"g": ab.LKJ(2)}), rescor=False)
weighted = ab.gaussian("mvbind(y, y2) | weights(w) ~ x + (1 | g)", data, ab.Priors(resp=resp, rescor=ab.LKJ(2)), rescor=True)
separate = ab.gaussian(["y ~ x + (1 | g)", "y2 ~ z"], data, priors, rescor=True, categorical={"z": "p"})
```

## What is offered

| construct | brms | the model | verified on |
| --- | --- | --- | --- |
| a formula per response under a residual correlation, `bf(y ~ x + (1 \| g)) + bf(y2 ~ z) + set_rescor(TRUE)` | a design per response (`X_y`, `X_y2`), `Lrescor` under `lkj(eta)`, one `multi_normal_cholesky(Y[n] \| Mu[n], diag(sigma) Lrescor)` per row with `Mu[n] = (mu_y[n], mu_y2[n])` | each response's univariate model of its own formula, assembled as C01 assembles them: the responses' Normal likelihoods replaced by the joint one | `gauss_mv_separate_rescor` |
| a correlation ID across responses, `(1 \|p\| g)` in `mvbind()` (or in each response's own formula) | one group of `g`: `sd_1` with one entry per response, `z_1` (responses x levels), one `cholesky_factor_corr` `L_1`, `r_1 = (diag(sd_1) L_1 z_1)^T`; each response's `sd \| g \| resp` prior on its own entry of `sd_1` | each response's model has the plain `(1 \| g)` (`y_1\|g_sigma`, `y_1\|g_offset`); the model adds `L\|g` (free, `lkj_corr_cholesky(eta)` from `cor={"g": ...}`, Stan's `cholesky_corr` coordinates) and `cor\|g = L L^T`, and replaces each later response's effects by `sd_k sum_{j<=k} L[k, j] z_j` | `gauss_mv_id_across`, `gauss_mv_id_across_rescor` |
| `weights()` without a residual correlation, `mvbind(y, y2) \| weights(w) ~ ...` | `weights_y`, `weights_y2`; `target += weights_y[n] * normal_lpdf(Y_y[n] \| ...) + weights_y2[n] * normal_lpdf(Y_y2[n] \| ...)` | each response's formula carries `weights(w)` (`y \| weights(w) ~ ...`): its univariate weighted likelihood (docs/addition-terms.md) | `gauss_mv_weights` |
| `weights()` under a residual correlation | `weights = weights_y`; `target += weights[n] * multi_normal_cholesky_lpdf(Y[n] \| Mu[n], LSigma)`, once per row | `model.joint_weights`; the joint density `w_i * multi_normal_cholesky_lpdf(...)` (`weighted_logp`) | `gauss_mv_weights_rescor` |

A formula per response also takes `weights()` on each response (`["y | weights(w) ~ x", "y2 | weights(w2) ~ z"]`): each response's
own without a residual correlation; under one, every response must have the same weights column (or none).

## Using it

- Priors: a correlation ID's LKJ prior is the top-level `cor` of its grouping factor (`cor={"g": ab.LKJ(2)}`, brms's `class = cor,
  group = g`), beside the `cor` of the responses' own correlated terms (`cor={"g": ab.LKJ(2), "h": ab.LKJ(3)}`; a class-wide `"*"`
  covers both, D173); each response's `sd` prior is that response's SD (brms applies each response's `sd | g | resp` prior to its own
  entry of `sd_1`, unlike one ID across mu and sigma, D187). `ab.default_priors()` adds brms's `lkj(1)`; `prior_table()` writes the
  `cor` rows once, as a univariate table writes them.
- Names: `model.id_names["g"]` holds `L|g`, `cor|g`, their coordinates `g__id_dim`, `g__id_dim_T` (labels `y_Intercept`,
  `y2_Intercept`, brms's order of the coefficients), the ID and the graph's correlated effects (`y_1|g`, `y2_1|g`); `model.names` has
  `"L:g"` and `"cor:g"`. `value_point(resp=..., L={"g": array(M, M)})`, each response's `z` its own row of brms's `z_1`.
- `predict()`: the effects are the graph's correlated ones; new levels of an ID's factor are refused in the joint predictions (the
  responses' effects at a new level would be drawn independently); `predict(resp=...)` draws one response's at its marginal.
- `describe()` writes the correlated effects, `L|g`'s prior and coordinates, and under a residual correlation the weights of the
  joint density; the target identity includes each ID and its prior.

## Refused, naming the construct

- An ID across responses with varying slopes or without the varying intercept (`(1 + x |p| g)`, `(0 + x |p| g)`), on `gr()`, `mm()`,
  `g:h` or `g/h`, twice in one formula, joining two grouping factors (brms refuses it too), used by one response alone (as in a
  univariate formula, D106), two IDs on one factor, with `subset()` or with `mi()` (D193).
- Under a residual correlation, weights that differ between the responses: two columns (brms refuses them too) or some responses
  weighted and others not (D194); `weights()` in a model with `subset()`; a response as another's weights; addition terms other
  than `weights()` after `mvbind()` (D56).
- As before: `rescor` must be given (D58); `subset()` and `mi()` with `rescor=True` (D160, D72); `log_lik()`, `loo()`, the summaries,
  `prior_predictive()` and `fit_record()` of a multivariate model (D95, D162); none of them falls out of this feature unverified.

## Divergences

- D193: brms puts any group-level terms that share an ID across responses in one correlation matrix (`(1 + x |p| g)` in
  `mvbind(y, y2)` is `M_1 = 4`, probed); the varying intercepts are taken here, the rest refused (above).
- D194: brms refuses two weights columns under a residual correlation ("All responses should use the same weights if 'rescor' is
  estimated.", compared by expression: a copy of the column is refused too) but takes weights on some responses only: with only the
  first response weighted it weights each row's joint density by them, and with only a later one weighted it writes
  `vector[N] weights = weights_y;` with `weights_y` undeclared, Stan code that does not compile (brms 2.23.0, rstan 2.32.7). Both are
  refused here.

## Evidence

The five frozen cases pass gates G1-G6 against brms 2.23.0 (fixtures #118; `tests/oracle/test_multivariate_gp_special_harness.py`
checks their notes on the frozen points): each response's design, response, weights and group coding (`adapter.ResponseView` shows
an ID's group to each response as a group of its own coefficients, its `sd_1` and `z_1` sliced; `adapter.match_cross_groups` checks
the group's responses, coefficients, `M_1`, `cor` flags and `L_1` against the model's IDs), the joint weights against brms's
`weights_y`, the log density and every gradient coordinate with and without the Jacobian (`L_1`'s `cholesky_corr` coordinates
recomputed and checked at every point), each response's linear predictor and correlated effects, `Cor_1`, the predictions with and
without group effects, the JAX density and the default-prior tables. `tests/oracle/test_multivariate_ext_harness.py` shows those
paths failing on corrupted fixtures and on models without the ID or the weights. `tests/test_multivariate_ext.py` covers the grammar
and every refusal, the split of the `cor` priors, the graph and its density against scipy (the effects written out, the joint
density weighted once, each response weighted), value points, predictions, default priors and the prior table, `describe()` and the
target identity. The syntax corpus's `mv_separate`, `mv_id_across` and `mv_weights` now read as brms's `make_standata()`
(`tests/brms_syntax/audit.py` compares the groups across responses).
