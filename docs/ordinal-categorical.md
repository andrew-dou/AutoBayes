# Ordinal and categorical responses

brms-parity feature B05 (`cumulative("logit")`, `categorical()`).

```python
import autobayes as ab
pri = ab.Priors(intercept=ab.Normal(0, 3), b=ab.Normal(0, 1.5), sd=ab.HalfNormal(1))
ordinal = ab.cumulative("rating ~ x + (1 | group)", data, pri)          # rating: 1..K, or an ordered pd.Categorical
choice = ab.categorical("brand ~ x", data, ab.Priors(intercept=ab.Normal(0, 3), b=ab.Normal(0, 1.5)))
```

## The models

| | cumulative (ordinal) | categorical |
| --- | --- | --- |
| response | K ordered categories: positive integers `1..K` (K is the largest value, brms's rule, so a value no row takes is still a category) or an ordered pandas Categorical (its declared categories, in order: brms's ordered factor) | K labels (text or integer codes), sorted as brms's `as.factor` sorts them (text in C-locale order, Python's `sorted`; integer codes numerically); the first is the reference |
| linear predictor | `mu_i`: the population-level terms on the raw design and the group effects, **no intercept** | one `mu_i[k]` per non-reference category `k`, each with its own centred intercept and coefficients on the same design (brms's `mu<k>`, `muc2`, ...) |
| probabilities | `P(y_i <= k) = expit(b_Intercept[k] - mu_i)` for the K - 1 increasing thresholds `b_Intercept` | `P(y_i = k) = softmax(0, mu_i[2], ..., mu_i[K])_k` |
| brms's likelihood | `ordered_logistic_lpmf(Y | mu, Intercept)` | `categorical_logit_lpmf(Y | (0, mu))` |

The response is fitted as its 1-based category, brms's `Y`; `model.categories` holds the labels in order (the codes of
`model.design.y` and of `predict(kind="response")` index it, and `describe()` lists them).

**Thresholds.** brms samples the thresholds of the centred design,
`Intercept[k] = b_Intercept[k] - sum_p mean(x_p) b_p`, an `ordered[K - 1]` vector on Stan's ordered coordinates
`(Intercept[1], log(Intercept[k] - Intercept[k - 1]))`, with the intercept prior on every threshold, and reports
`b_Intercept`, the thresholds of the raw design. So does this library: `Intercept` is the free variable (PyMC's
`ordered` transform, whose coordinates and Jacobian are Stan's; `likelihoods.ordered_thresholds`),
`b_Intercept = Intercept + sum_p mean(x_p) b_p` is a deterministic of the graph that the likelihood reads with `mu` on
the raw design, and the posterior keeps both under brms's names (unlike the other families, whose posterior `Intercept`
Bambi reports on the raw scale). With `center_predictors=False` the two are equal. The intercept cannot be removed
(`0 +` and `- 1` are refused, as brms refuses them). The thresholds' prior density is the prior's density summed over
them, as brms adds it (not renormalised to the ordered region); their prior-predictive draws are the prior's independent
draws sorted, which is that density's distribution.

**Graph.** Bambi's own families are not used as they stand. Its `cumulative` family has the same sign convention
(`P(y <= k) = F(threshold_k - eta)`) and PyMC's `ordered` transform, whose coordinates and Jacobian are Stan's, but it
drops the intercept without centring, so its sampled thresholds are the raw ones (brms samples those of the centred
design, and its prior is on them: a different target), and it forms the category probabilities by subtraction inside
`pm.Categorical`; its `categorical` family takes the first category as the reference, as brms does, but its parent is
the probabilities, not the linear predictors. So for the cumulative family Bambi is handed the formula with formulae's
intercept, so that a categorical predictor is treatment coded as brms codes it, made to add nothing
(`likelihoods.no_intercept`), the design uncentred and the thresholds as the family's auxiliary parameter
(`likelihoods.cumulative_logit_family`, built per model from its terms and column means through
`FamilySpec.bambi_setup`). The categorical family is Bambi's own categorical family (every coefficient with a trailing
category axis `<response>_reduced_dim`, the design centred, the intercepts uncentred after sampling) with the K - 1
linear predictors kept as the parent `mu` under the identity link (`likelihoods.categorical_logit_family`).

**Numerics.** The cumulative row is Stan's `log_inv_logit_diff` written out,
`-softplus(mu - tau_y) - softplus(tau_{y-1} - mu) + log(1 - exp(-(tau_y - tau_{y-1})))`, never a difference of
probabilities; the categorical row is `mu[y] - log_sum_exp(0, mu)` with the maximum subtracted inside the sum. Both are
finite with a finite gradient for every finite linear predictor on the PyTensor path and under JAX's autodiff (what
NumPyro differentiates). Two traps of this stack are avoided and tested: PyTensor rewrites its own `logsumexp` into a
switch whose untaken branch `exp(max)` overflows above 709.78, and JAX's gradient of that switch is NaN; and PyTensor's
`take_along_axis` builds `arange`s that JAX cannot lower for a category axis Bambi built (or overflows in `int8` beyond
127 rows), so each row's threshold or category is selected by a one-hot switch instead. `predict(kind="mean")`
evaluates the cumulative probabilities in the same stable form; brms's `posterior_epred` subtracts cumulative
probabilities, which loses relative precision for a category between two close thresholds (divergence D39; equal to
rounding on the fixture).

## Using it

- Priors: `Priors(intercept=...)` is the prior of every threshold (cumulative) or of every category's intercept
  (categorical), as brms's class `Intercept` is; `b` is one prior (or the term mapping) shared by every category's
  coefficients (brms takes a separate prior per category, `dpar = mu<k>`; here one per class, divergence D37); `sd` and
  `cor` as for every family (cumulative). `sigma`, `shape`, `phi`, `nu`, `zi` and `hu` are refused.
  `ab.default_priors(..., family="cumulative")` returns brms's `student_t(3, 0, 2.5)` on the thresholds (also uncentred,
  where brms keeps the class `Intercept`) and on `sd`, flat `b`; for `"categorical"` the same per category.
  `model.prior_table()` writes brms's rows: the thresholds' class row and one row per threshold (`coef` `1`, ..., `K -
  1`, sorted as brms sorts them), or the `Intercept` and `b` rows of each category under its `dpar` (`muc2`, ...; brms
  keeps only a category's letters and digits in that name, and categories that would share one are refused).
- `Model.value_point(intercept=[...], b=...)`: the cumulative `intercept` is the K - 1 thresholds (brms's `Intercept`),
  strictly increasing, stored on the ordered coordinates (`Intercept_ordered__`); the categorical `intercept` has one
  entry per non-reference category and `b` is `(columns, K - 1)`. `model.names` adds `thresholds` (`b_Intercept`) and
  `threshold_dim` (`Intercept_dim`, coordinates `"1"`, ..., `"K - 1"`), or `category_dim` (`<response>_reduced_dim`).
- `predict()`: `kind="linear"` returns `eta`, brms's `posterior_linpred`: `mu` without thresholds (cumulative), or
  `(chain, draw, __obs__, category)` over the non-reference categories (categorical). `kind="mean"` returns the
  probabilities `p` with a new `category` dimension over all K categories (brms's `posterior_epred`), and
  `kind="response"` adds 1-based category draws `y` (brms's `posterior_predict`); `attrs["categories_json"]` lists the
  categories. Every other family's output is unchanged. The cumulative family takes both unseen-group policies and
  `include_group_effects=False`.
- The term model is shared: the cumulative family takes categorical predictors, interactions, transforms, the offset
  (which enters `mu`), any number of grouping factors and independent or correlated slopes. The categorical family takes
  population-level terms (and `0 +`); group-level terms and the offset are refused in this slice (divergence D37).
- Not offered: other cumulative links, `disc`, category-specific effects `cs()`, other threshold structures, and brms's
  `sratio`, `cratio` and `acat` families (divergence D38); a categorical reference category other than the first sorted
  label (D36).

## Evidence

Oracle cases `cumulative_varint` (`ord ~ x + (1 | g)`, four categories, `normal(0, 3)` on the thresholds) and
`categorical_poponly` (`cat3 ~ x`, three categories), and two cases frozen with this feature,
`cumulative_poponly_categorical` and `categorical_poponly_categorical` (`+ z`, a treatment-coded predictor: the dummies'
means in the thresholds' centring, and the (level, category) layout of its coefficients against brms's gradient), pass
gates G1–G6 against the frozen brms fixtures: design, response codes and `nthres` / `ncat` (the categorical model's
`X_muc2` and `X_muc3` each matched to the design); log density with and without the Jacobian within 9e-16 relative and
every gradient coordinate within 7e-15, brms's `ordered` coordinates of `Intercept` recomputed at every point and
checked against the fixture's before the gradient is compared; the threshold and category priors inside `lprior`; the
graph's `mu` against `posterior_linpred` (per category for the categorical model), `b_Intercept` against brms's
`b_Intercept` and each category's raw-scale intercept against `b_mu<k>_Intercept`; `predict()` against
`posterior_linpred` / `posterior_epred` (8 x 36 x 4 and 8 x 36 x 3) on fitted and new rows with and without group
effects; the JAX density; and `ab.default_priors()` against brms's `default_prior()` table.
`tests/test_ordinal_categorical.py` checks both rows and their gradients against a 50-digit reference on linear
predictors from -1000 to 1000 (with a threshold gap of 1e-9) on both paths, the response coding and every refusal, the
models' density, graph and gradient by hand on both paths and at 300 rows under JAX, `value_point`, by-hand predictions,
sorted prior-predictive thresholds, `describe()`, the target identity, six `default_prior()` tables brms printed and the
fit record; `tests/oracle/test_harness.py` checks that a corrupted ordered coordinate, threshold gradient,
`b_Intercept`, category probability, category coefficient's gradient, category linear predictor, raw category intercept
or category design fails its gate, and that a model of the other kind is not matched to either fixture. Nothing samples.
