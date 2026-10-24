# Ordinal and categorical responses

brms-parity feature B05 (`cumulative("logit")`, `categorical()`); D04 adds `sratio("logit")`, `cratio("logit")`,
`acat("logit")` and category-specific effects `cs()` (below).

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
- Not offered: other cumulative links than logit and probit (the probit link since P6's reference, D88, below; the cloglog link is
  built since E06 but withheld, because brms's own log density of its frozen case is inexact, D132, [links](links.md)), `disc`, category-specific
  effects `cs()` in the cumulative family and other threshold structures (divergences D38, D89; grouped thresholds `thres(gr = z)`
  are F06b's, [grouped thresholds](grouped-thresholds.md), D158); a categorical reference
  category other than the first sorted label (D36). brms's `sratio`, `cratio` and `acat` families are D04's (below).

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

## More ordinal models: sratio, cratio, acat and cs() (D04)

```python
pri = ab.Priors(intercept=ab.Normal(0, 3), b=ab.Normal(0, 1.5), sd=ab.HalfNormal(1))
ab.sratio("rating ~ x + (1 | group)", data, pri)          # brms's sratio("logit"), the stopping ratio
ab.cratio("rating ~ x + (1 | group)", data, pri)          # cratio("logit"), the continuation ratio
ab.acat("rating ~ w + cs(x) + (1 | group)", data, pri)    # acat("logit"), adjacent categories; x category-specific
```

The response, `mu` (no intercept), the K - 1 thresholds and their centring are the cumulative family's (above). With
`d_k = b_Intercept[k] - mu_i[k]` (`mu_i[k] = mu_i`, plus `x_i bcs_x[k]` for each category-specific term), the published
definitions (Agresti, *Categorical Data Analysis*, ch. 8; Buerkner and Vuorre 2019):

| family | definition | `log P(y = k)` as evaluated | brms's likelihood |
| --- | --- | --- | --- |
| `sratio` | `P(y = k \| y >= k) = expit(d_k)` | `-softplus(-d_k) [k < K] - sum_{j<k} softplus(d_j)` | `sratio_logit_lpmf` |
| `cratio` | `P(y > k \| y >= k) = expit(-d_k)` | the same: `1 - expit(-d) = expit(d)` | `cratio_logit_lpmf` |
| `acat` | `log(P(y = k + 1) / P(y = k)) = -d_k` | `s_k - log_sum_exp(s)`, `s_1 = 0`, `s_k = -sum_{j<k} d_j`, the largest `s` factored out | `acat_logit_lpmf` |

Under the logit link `sratio` and `cratio` give the same probabilities (the logistic is symmetric); they are separate
families, as in brms, and their target identities differ by the family's name. Each row is a sum of stable terms, finite
with a finite gradient for every finite predictor on the PyTensor path and under JAX's autodiff
(`likelihoods.ordinal_logp_rows`; `likelihoods.ordinal_log_probabilities` is its NumPy form for `predict()`, `log_lik()` and
the draws).

**Thresholds.** brms declares them `vector[nthres] Intercept`: unordered, sampled as they are, with the intercept prior on
each. So does this library (`likelihoods.unordered_thresholds`, `FamilySpec.ordered_thresholds = False`): `Intercept` is a
free vector over `Intercept_dim`, `b_Intercept = Intercept + sum_p mean(x_p) b_p` the deterministic the likelihood reads,
`value_point(intercept=...)` takes K - 1 finite numbers in any order, and prior-predictive thresholds are the prior's
independent draws.

**Category-specific effects** `cs(x)` (brms's `bcs`, a `Kcs x nthres` matrix, and `Xcs`): a population-level term of kind
`"category_specific"` with no design column and one coefficient per threshold, the free vector `cs(x)` over
`Intercept_dim` (brms's row of `bcs`), under the class-`b` prior (`Priors(b=...)`, or `b={"x": ..., "cs(x)": ...}`; brms's
table lists it as class `b`, coefficient `x`); its column is never centred and is not part of the thresholds' centring.
Each threshold's predictor `mu_i + sum_c x_c,i bcs_c[k]` is the deterministic `mu_cs` (`names["linpred"]`, dims `__obs__` x
`Intercept_dim`; brms subtracts `Xcs bcs` from the thresholds, which is the same), checked against the data at build;
`value_point(bcs={"x": [...]})` gives the coefficients, and `predict(kind="linear")` returns `mu_cs` with a `threshold`
dimension (brms's `posterior_linpred`, one column per threshold). Offered for the `sratio`, `cratio` and `acat` families,
one numeric column per `cs()`, any number of them; brms's `cs(x + w)`, `cs(1)`, `cs()` of a factor or in an interaction,
category-specific group-level terms `(cs(1) | g)`, `x + cs(x)` (whose `b_x` and `bcs_x` are not separately identified) and
`cs()` in the cumulative family (brms: "experimental") are refused naming the construct (D89).

**What combines.** Categorical predictors, interactions, transforms, the offset, any number of grouping factors, independent
and correlated slopes and Gaussian processes (C03's check at build) enter `mu` as for the cumulative family; `predict()` takes
both unseen-group policies and `include_group_effects=False`; `log_lik()`, `loo()` and `bayes_R2()` (the expected category, as
for the cumulative family) work. Refused, naming the construct: `mo()` (no frozen case, D42), `weights()` and the other
addition terms, a formula for `disc`, `mm()` and `gr(by = )`, autocorrelation, `me()` / `mi()`, multivariate, non-linear and
mixture models, the intercept's removal (brms refuses it too), and every link but logit (`parked_links`, D38).

**The cumulative probit link** (`cumulative("probit")`, brms's `ordered_probit_lpmf`) is built (`cumulative._probit()`: the
same graph with `log Phi(d_1)`, `log Phi(-d_{K-1})` and the log Normal mass between `d_{k-1}` and `d_k` from the tail where it
lies), `ab.cumulative(..., link="probit")` (D88): on the frozen case `cumulative_probit_varint` its log density, linear
predictor, expected values, JAX density and default priors equal brms's (G1, G2's log densities, G3-G6), but brms's own
gradient there is off the 50-digit value by up to 6.5e-5 relative at 86 of its 88 entries, beyond the oracle's 1e-7, where
this library's is within 4e-15. The case is accepted by two-reference acceptance (P6): `oracle/reference.py` implements the
cumulative probit model at 50 digits (Stan's ordered transform and its Jacobian, each row's Normal mass taken in the tail it
lies in), its log densities equal brms's to 3.6e-16 at every frozen point, and G2's gradient falls back on it at the 173
of 176 entries (with and without the Jacobian) where brms is off it.
Two thresholds closer than about 1e-6 in a far tail keep about `|tau - eta| eps / gap` relative accuracy in the probit row
(`likelihoods.normal_log_mass`).

### Evidence (D04)

Oracle cases `sratio_varint`, `cratio_varint`, `acat_varint` (`ord ~ x + (1 | g)`), `acat_cs` (D08's parked case, `ord ~ w +
cs(x) + (1 | g)`) and two frozen with this feature, `sratio_cs` and `cratio_cs` (the same formula), pass gates G1-G6: brms's
`X`, `Xcs`, `Kcs`, `nthres` and group coding; the log density with and without the Jacobian and every gradient coordinate,
the unordered `Intercept` and `bcs` mapped entry by entry; the graph's `mu` or `mu_cs` against `posterior_linpred` (36 x 3 per
point with `cs()`), `b_Intercept`; `predict()` against `posterior_linpred` / `posterior_epred` on fitted and new rows with
and without group effects; the JAX density; and `ab.default_priors()` against brms's `default_prior()` (the `cs()` column a
class-`b` row). `sratio_cs_post`, `cratio_cs_post` and `acat_cs_post`, frozen with this feature too, pass C07's post gate:
`log_lik()`, `bayes_R2()` (the expected category) and `loo()` against brms's. `cumulative_probit_varint` passes with P6's reference for G2's gradient (D88). `tests/test_more_ordinal.py` checks the rows and their
gradients against a 50-digit reference written from the definitions on linear predictors from -1000 to 1000 (with an
unordered threshold vector holding a gap of 1e-9) on both paths, the probit rows likewise, the models' density, graph and
gradient by hand on both paths (unordered thresholds, two `cs()` terms, an offset) and at 300 rows under JAX, `value_point`,
16 formula refusals and the data and link refusals, by-hand predictions with the `threshold` dimension and unseen levels,
`log_lik()` and `bayes_R2()`, unordered prior-predictive draws, `describe()`, the target identity, brms's default prior
tables, and, with the park lifted for the test, the probit model against `cumulative_probit_varint` with its gradient checked
by hand; `tests/test_post_processing.py` checks `log_lik()` against the model's pointwise density on the six cases. Nothing
samples.
