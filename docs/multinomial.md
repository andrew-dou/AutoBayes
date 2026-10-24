# Count-matrix responses: `multinomial()` and `dirichlet_multinomial()`

brms-parity feature F01a (brms's `multinomial()` and `dirichlet_multinomial()` on `cbind(...) | trials(n)`).

```python
import autobayes as ab
pri = ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd=ab.HalfNormal(1))
m = ab.multinomial("cbind(m1, m2, m3) | trials(size) ~ x + (1 | g)", data, pri)

# a category with a formula of its own (brms's bf(..., mum3 ~ z)) and the Dirichlet-multinomial's precision
pri = ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd=ab.HalfNormal(1), phi=ab.Gamma(0.01, 0.01),
                dpars={"mum3": {"intercept": ab.Normal(0, 2), "b": ab.Normal(0, 0.5)}})
dm = ab.dirichlet_multinomial(["cbind(m1, m2, m3) | trials(size) ~ x + (1 | g)", "mum3 ~ z"], data, pri, categorical={"z": "p"})
```

## The model

The response is brms's matrix `cbind(y_1, ..., y_K) | trials(n)`: each row's counts of K categories, the columns in the order
written (the first is the reference category, brms's default `refcat`), and `n` the row's total. Each non-reference category
`k` has its own linear predictor, brms's dpar `mu<k>` (`mu` and the category's letters and digits: `mum2`, `mum3`):

    mu_k,i   = a_k + (X_k,i - mean(X_k)) b_k + sum_g u_k,g[g_i]          k = 2..K;  mu_1,i = 0
    u_k,g[j] = sd_k,g z_k,g[j],   z_k,g[j] ~ Normal(0, 1)
    p_i      = softmax(0, mu_2,i, ..., mu_K,i)
    y_i      ~ Multinomial(n_i, p_i)              log p = log C(y_i) + sum_k y_k,i log p_k,i
    y_i      ~ DirichletMultinomial(n_i, phi p_i) log p = log C(y_i) + lgamma(phi) - lgamma(n_i + phi)
                                                          + sum_k [lgamma(y_k,i + phi p_k,i) - lgamma(phi p_k,i)]

with `C(y) = n! / prod_k y_k!`. By default every category takes the response formula's right-hand side; a list of formulas gives
a category one of its own (`"mum3 ~ z"`), which replaces the response formula's terms for that category entirely, as in brms.
Each category's design is centred on its own column means (brms's `Xc_mu<k>`); its centred intercept is brms's `Intercept_mu<k>`
and the raw-scale one `b_mu<k>_Intercept`. Each group-level term of each category is a group of its own with its own SD (brms
numbers them category by category: `sd_1` for `mum2`'s `(1 | g)`, `sd_2` for `mum3`'s). The densities include the multinomial
coefficient, as brms's `multinomial_logit2_lpmf` and `dirichlet_multinomial_logit2_lpmf` do (checked on the frozen points).

**Numerics** (`autobayes/families/multinomial.py`). The coefficient is `sum_k log C(y_k + ... + y_K, y_k)` through M2's stable
log binomial coefficient, never a difference of lgamma values of size `n log n`; `log p` is the log-softmax with the largest entry
factored out (B05's), finite for every finite linear predictor; each Dirichlet-multinomial difference of lgamma values is a log
rising factorial through Stirling's remainder (D03's `_log_rising`), and below `log(phi p) = -40` it is `lgamma(y) + log(phi p)`,
its value to double precision there, so a category probability that underflows leaves the row finite with a finite gradient on
the PyTensor path and under JAX. A precision as large as 1e8 keeps full accuracy (where the difference of two lgamma values of
size `phi log phi` would not).

**Graph.** Bambi has no such model, so `MultinomialModel` builds the PyMC graph itself (`model.bambi_model` is None), as the
non-linear model does: per category `mum2_Intercept` (centred), `mum2_x` (`mum2_z` over `mum2_z_dim` for a categorical term),
`mum2_1|g_sigma`, `mum2_1|g_offset` and `mum2_1|g` (on `g__factor_dim`), the deterministics `mum2_b_Intercept`, `mum2` (per
row) and `mu` (`(rows, K - 1)` over `mu_dim`), `phi` (sampled as `phi_log__`), and the observed counts `cbind(m1, m2, m3)` over
`(__obs__, category)`. `ab.Model(..., family="multinomial")` returns this class too.

## Using it

- **Formula.** `trials()` is required and each row's counts must sum to it (brms: "Specifying 'trials' is required for this
  model."; "Number of trials does not match the number of events."); counts are whole and non-negative; rows of zero trials are
  kept, as brms keeps them (they contribute 0; the binomial family drops them, D4). The reference category is the first column;
  write the columns in another order for another (brms's `refcat`). `categorical={...}` is shared: a declared
  column is categorical in every category's formula that names it. Each category's formula takes the term model's
  population-level terms (numeric and categorical predictors, interactions, transforms) and varying intercepts `(1 | g)`, `g:h`
  and `g/h`.
- **Priors.** `Priors(intercept=..., b=..., sd=...)` apply to every category whose formula is the response formula's, one prior
  per class shared by those categories (brms takes them per `dpar`: D137); `Priors(dpars={"mum3": {"intercept": ..., "b": ...,
  "sd": ...}})` gives a category its own (brms's `dpar = mum3`), required for a category with a formula of its own and allowed for
  any; `phi` takes one positive-support prior. `ab.default_priors(formula, data, family="multinomial")` returns brms's defaults
  (`student_t(3, 0, 2.5)` on each category's intercept and group SDs, flat coefficients, `gamma(0.01, 0.01)` on `phi`), and
  `model.prior_table()` writes brms's rows, each category's `Intercept`, `b` and `sd` rows under its `dpar`.
- **Names and points.** `model.categories`, `model.names` (`linpred` `mu`, `mean` `p`, `y`, `phi`, `mu:<name>`), `model.mu_names`,
  `model.mu_coefficients`, `model.mu_group_names`; `model.value_point(dpars={"mum2": {"intercept": a, "b": [...], "sd": s, "z":
  [...]}, ...}, phi=...)`.
- **`predict()`**: `kind="linear"` returns `eta` over `(chain, draw, __obs__, category)`, the non-reference categories (brms's
  `posterior_linpred`); `kind="mean"` the probabilities `p` and the expected counts `expected = n p` over every category (brms's
  `posterior_epred`; `newdata` needs the trials then, not the counts); `kind="response"` count draws `y` too (multinomial draws;
  for the Dirichlet-multinomial a probability vector from `Dirichlet(phi p)` first, drawn on the log scale). Both unseen-level
  policies and `include_group_effects=False` apply by grouping factor; a factor in two categories' formulas draws each one's
  effects independently (brms's group ids are separate too).
- `ab.log_lik()` (each row's log mass with its coefficient, brms's `log_lik()`), `ab.loo()`, `ab.waic()`, `describe()`, the
  target identity, `fit()` (PyMC's `sample` on the graph), `fit_record()` and `prior_predictive()`. `bayes_R2()` is refused, as
  brms refuses it ("'bayes_R2' is not defined for unordered categorical models."), and so are `hypothesis()`,
  `conditional_effects()` and `prior_draws()`, as for the categorical family (D99, D124).

## Refused, naming the construct

Divergence D136: another link than the logit (brms has no other), other addition terms (`weights()`, `cens()`,
`trunc()`, `mi()`), a formula of `phi` (brms takes one) or of the reference category (brms refuses it), a response formula whose
terms no category uses (every category with a formula of its own: brms ignores them silently; write `~ 1`), the removal of a
category's intercept, `center_predictors=False`, varying slopes `(1 + x || g)` and `(1 + x | g)`, correlation IDs, `mm()`, `gr(by
= )`, `offset()`, `mo()`, `gp()`, `me()`, `mi()`, `cs()`, autocorrelation and `sar()` terms in a category's formula (and `data2`), and multivariate,
mixture and non-linear models of these families. brms accepts a negative count (its data check passes and its Stan data block
does not bound `Y`): the multinomial's `log_prob` then raises Stan's "multinomial_lpmf: Number of trials variable[1] is -1, but
must be nonnegative!" and the Dirichlet-multinomial's is `-Inf`, so sampling fails with "Initialization failed" (probed with
rstan on `multinomial_varint`'s data, row 1 set to (-1, 2, 2) with trials 3); this library refuses the count when the model is
built.

## Evidence

Six frozen cases pass gates G1-G6 against brms 2.23.0 (and C07's post gate on `multinomial_varint_post`): `multinomial_varint` (`cbind(m1, m2, m3) | trials(size) ~ x + (1
| g)`), `multinomial_dpar_formula` (`mum3 ~ z`: a categorical predictor, no group term), `multinomial_varint_post` (`log_lik()`
and `loo()` of the 8-draw fit) and `dirichlet_multinomial_varint` (frozen with the Tier F fixtures, #83), and two frozen with this
feature: `multinomial_terms` (`cbind(m2, m1, m3)`, so the reference is `m2` and the linear predictors `mum1`, `mum3`; `x * z +
(1 | g) + (1 | h)` for `mum1` with a prior per group, `mum3 ~ log(xpos) + (1 | g)`: three brms groups, two on one factor) and
`dirichlet_multinomial_defaults` (`x + z + (1 | g)` under brms's default priors, the model built from `ab.default_priors()`); both
regenerate identically (`oracle/regen.py check`). G1 matches brms's `Y`, `trials`, `ncat` and each `X_mu<k>` (with `K_mu<k>`,
`Kc_mu<k>`) by content, and each group by `Z_<id>_<mu>_1` and `J_<id>`; G2 the log density with and without the Jacobian within
5.5e-16 relative (3.6e-10 on the defaults case, PyMC's half-Student-t `log 2` in single precision, D103), the prior part and every
gradient coordinate within 6.2e-15 (4.3e-10 where `phi` enters: PyTensor's digamma); G3 the graph's `mu`, each category's values
against brms's own linear predictor, the raw-scale intercepts and the group effects; G4 `posterior_linpred` (draws x rows x
(K - 1)) and `posterior_epred` (draws x rows x K) on fitted and new rows with and without group effects within 8.8e-16; G5 the
JAX density; G6 the default-prior table, every row. `tests/test_multinomial.py` checks both rows and their gradients (and the
Dirichlet-multinomial's in `phi`) against a 50-digit reference on both paths: the multinomial's on linear predictors from -1000 to
1000 and counts up to 1e6, the Dirichlet-multinomial's on linear predictors from -800 to 25 and precisions from 1e-3 to 1e8; the NumPy rows, the draws' distributions, every formula, count and prior refusal,
the names, design and graph, the density by hand on both paths, `value_point`, `describe()`, predictions by hand with both
unseen-level policies, `log_lik()` by hand, the refused summaries, five `default_prior()` tables brms printed, the prior table,
the identity, the fit plumbing with a stubbed sampler and the fit record, prior-predictive draws, and that the gates fail on a
corrupted `multinomial_terms` or `dirichlet_multinomial_varint` fixture (the counts, `ncat`, the trials, a category's design, its
intercept column, a group's `Z`, its levels, a group without a counterpart, a coefficient, a gradient in `Intercept_mum3` or
`phi`, a transform, the linear predictor, an expected count, a group effect, a raw-scale intercept, the default-prior table) and
on a model of another formula or family. `tests/test_post_processing.py` checks `log_lik()` against the graph's own pointwise
density on five of the cases. Nothing samples.
