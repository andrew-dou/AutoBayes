# Bounded and hurdle families

brms-parity feature E01: `zero_one_inflated_beta()`, `zero_inflated_beta_binomial()`, `xbeta()` and `hurdle_cumulative()`, brms
2.23's families of the same names, each with the logit link.

```python
import autobayes as ab
base = dict(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd=ab.HalfNormal(1))
zoib = ab.zero_one_inflated_beta("p ~ x + (1 | g)", data, ab.Priors(**base, phi=ab.Gamma(0.01, 0.01), zoi=ab.Beta(2, 2), coi=ab.Beta(2, 2)))
zibb = ab.zero_inflated_beta_binomial("k | trials(n) ~ x + (1 | g)", data, ab.Priors(**base, phi=ab.Gamma(0.01, 0.01), zi=ab.Beta(2, 2)))
xb = ab.xbeta("p ~ x + (1 | g)", data, ab.Priors(**base, phi=ab.Gamma(0.01, 0.01), kappa=ab.Gamma(0.01, 0.01)))
hc = ab.hurdle_cumulative("o ~ x + (1 | g)", data, ab.Priors(**base, hu=ab.Beta(2, 2)))
```

## The models

`eta` is the linear predictor, `mu = expit(eta)`, and the Beta shapes are `a = mu phi`, `b = (1 - mu) phi` (`1 - mu` computed as
`expit(-eta)`, D13). These are brms's densities as its vignette "Parameterization of Response Distributions in brms" writes them,
and for `xbeta` as Kosmidis and Zeileis (2024, "Extended-support beta regression for [0, 1] responses") define it; each was
checked numerically against brms's `log_prob` and `posterior_epred` on the frozen cases.

| family | response | at the boundary | otherwise | `E[y]` (`posterior_epred`) |
| --- | --- | --- | --- | --- |
| `zero_one_inflated_beta` | proportions in [0, 1] | `zoi (1 - coi)` at 0, `zoi coi` at 1 | `(1 - zoi) Beta(y \| a, b)` | `zoi coi + (1 - zoi) mu` |
| `zero_inflated_beta_binomial` | `k \| trials(n)` | `zi + (1 - zi) BetaBinomial(0 \| n, a, b)` at 0 | `(1 - zi) BetaBinomial(k \| n, a, b)` | `(1 - zi) n mu`, named `expected` |
| `xbeta` | proportions in [0, 1] | `I_c(a, b)` at 0, `I_c(b, a)` at 1, `c = kappa / (1 + 2 kappa)` | `Beta((y + kappa) / (1 + 2 kappa) \| a, b) / (1 + 2 kappa)` | the censored mean below |
| `hurdle_cumulative` | codes `0..K` | `hu` at the hurdle category 0 | `(1 - hu) (F(tau_k - eta) - F(tau_{k-1} - eta))`, `F` the logistic CDF | the K + 1 probabilities |

`zoi` is the probability of an exact 0 or 1 and `coi` the probability of a 1 given that; `zi` and `hu` are brms's zero-inflation
and hurdle probabilities. `xbeta` is one process for the whole of [0, 1]: `B ~ Beta(a, b)` is stretched to `(1 + 2 kappa) B -
kappa` on `(-kappa, 1 + kappa)` and censored to [0, 1], so the exact 0s and 1s are the stretched Beta's mass beyond the ends
(`I` is the regularized incomplete Beta function; the upper mass `1 - I_{1-c}(a, b)` is `I_c(b, a)` by the Beta's symmetry, never a
subtraction). Its expected value is the censored variable's mean, `I_c(b, a) + (1 + 2 kappa) mu (I_d(a + 1, b) - I_c(a + 1, b)) -
kappa (I_d(a, b) - I_c(a, b))` with `d = 1 - c` (from `int u Beta(u | a, b) du = mu I(a + 1, b)`), which equals brms's
`posterior_epred` to 3.1e-15 on the frozen points and a quadrature to 1e-10 in the tests; it is not `mu`.

`hurdle_cumulative`'s ordinal part is the cumulative logit model (docs/ordinal-categorical.md): the linear predictor `mu` without
an intercept, the K - 1 ordered thresholds `tau` (brms's `b_Intercept`), sampled as brms samples them, for the centred design
(`Intercept = tau - sum_p mean(x_p) b_p`, an `ordered[K - 1]` vector on Stan's ordered coordinates, under the intercept prior).

**Responses** (brms 2.23's `standata()`, probed): `hurdle_cumulative` takes whole numbers `0..K`, 0 the hurdle and `1..K` the
ordinal part's categories, every one of them whether observed or not (brms's `1..max(y)`), or an ordered pandas Categorical, whose
first declared category is the hurdle (coded 0, as brms codes an ordered factor's first level) and whose others are the ordinal
part's in order. Negative, fractional and text responses and unordered Categoricals are refused, as brms refuses them, and so is a
response with fewer than two categories above the hurdle (brms: "Could not extract the number of thresholds"). The model is fitted
on brms's codes (`model.design.y`); `model.categories` holds the ordinal part's K categories (so the thresholds are
`len(model.categories) - 1`, as for the cumulative family) and `model.hurdle_category` the hurdle's label. The two proportion
families take [0, 1] with exact 0s and 1s; `zero_inflated_beta_binomial` takes successes and trials as the binomial family does
(zero-trial rows dropped, D4).

## The graph and its numerics

Each likelihood is a `pm.CustomDist` (`autobayes/likelihoods.py`) of the linear predictor, kept as Bambi's parent under the identity
link (`logit_mu`, or `mu` for `hurdle_cumulative`, whose thresholds and `b_Intercept` are built as the cumulative family's), and of
the auxiliary parameters under brms's names: `phi` (`phi_log__`), `zi` / `hu` (`FamilySpec.zero_parameter`, `zi_logodds__`),
and the family's `extra_parameters` (D01's mechanism): `zoi` and `coi` of the new support `"unit"`, brms's `real<lower=0,upper=1>`
(a `Beta` prior, Stan's logit, `zoi_logodds__`, `coi_logodds__`), and `xbeta`'s `kappa`, a `"positive"` one (`kappa_log__`).

- Every term of a branch is evaluated on its own rows only, the other rows at fixed interior arguments (shapes 1, `x = 1/2`), so
  a branch a switch does not take is finite with a finite gradient: an `xbeta` row inside (0, 1) whose `I_c(a, b)` would underflow
  (a precision of thousands) keeps a finite gradient, and the rows at 0 and 1 of `zero_one_inflated_beta` have a zero gradient in
  `eta` and `phi` at every `eta`.
- The zero-inflated beta-binomial's zero row is B06's two-term log-sum-exp; its beta-binomial is PyMC's, as D02's family's.
- `xbeta`'s masses are `likelihoods.shape_betainc`: PyTensor's regularized incomplete Beta function (Cephes' `incbet` in C) and
  PyTensor's gradient (Boik and Robison-Cox's continued fraction for the shape derivatives, 1998, Journal of Statistical Software
  3(1)) on the PyTensor path. JAX's `betainc` has no derivative in its shapes and PyTensor lowers no scalar loop to JAX, so the op
  is distinct, as `stable_softplus` is, and lowers to `jax.scipy.special.betainc` with a custom JVP: the derivative in `x` is the
  Beta density, those in `a` and `b` the same continued fraction differentiated through its recurrences (the coefficients' own
  derivatives by forward-mode autodiff), rescaled every step, to 1e-14 relative, at most 1000 steps. `B(a, b)` is formed from
  `gammaln` there, because `jax.scipy.special.betaln` is 7e-8 off relative at `B(10, 3.3)`. Against a 50-digit reference (the
  hypergeometric series of `B_x(a, b)`, Stirling's series, fourth-order central differences; `tests/test_bounded_families.py`)
  the JAX gradient is within 1e-11 and PyTensor's within 2e-8 on a grid of shapes 0.3 to 80 and `x` from 0.01 to 0.8. Where `I`
  underflows to 0 (a boundary row at an extreme mean and precision, `a log c` below about -745) the row's log density is `-inf` and
  its gradient NaN, on both paths, and PyTensor's gradient is NaN when its 200 steps do not converge (D110).

## Using it

- Priors: `phi`, and `zoi` / `coi`, `zi`, `kappa` or `hu`, are required for the families that have them (`zoi`, `coi`, `zi` and
  `hu` an `ab.Beta`; `phi` and `kappa` a positive-support prior) and refused by name for every other; brms's defaults on request
  (`ab.default_priors`): `gamma(0.01, 0.01)` for `phi` and `kappa`, `beta(1, 1)` for `zoi`, `coi`, `zi` and `hu`, `student_t(3, 0,
  2.5)` for the intercept (the thresholds) and the SDs. brms accepts any prior on these classes (D111).
- `Model.value_point(..., zoi=0.3, coi=0.6)` (stored as their logits), `kappa=0.5` (its log), `zi=` / `hu=` as for D03's families;
  `hurdle_cumulative`'s `intercept` is its K - 1 centred thresholds, strictly increasing.
- `predict()`: `kind="linear"` is `eta` (`mu` for `hurdle_cumulative`); `kind="mean"` is brms's `posterior_epred` under the name
  `mean` (`expected` for the zero-inflated beta-binomial, which needs the trials of new rows; for `hurdle_cumulative` the K + 1
  probabilities `p` with a `category` dimension, the hurdle first, as brms's categories 0..K); `kind="response"` draws the response:
  a boundary value where a uniform draw falls below `zoi` (then 1 below `coi`), a zero below `zi` or `hu`, a censored stretched Beta
  draw for `xbeta`, codes 0..K for `hurdle_cumulative`. Draws whose Beta shapes underflow to 0 are refused.
- `log_lik()`, `loo()` and `waic()` take all four (checked against each model's own pointwise density in
  `tests/test_post_processing.py`); `bayes_R2()` the three that are not ordinal (brms's definition on brms's own `posterior_epred` at
  the frozen points) and is refused for `hurdle_cumulative` (D109). `hypothesis()` takes the three; `conditional_effects()` the two
  proportion families (the zero-inflated beta-binomial's grid needs trials, as the binomial's); both refuse `hurdle_cumulative`, as
  they refuse the cumulative family (D99). `describe()`, prior-predictive draws, the target identity and the fit record cover them.
- Contexts: the term model (categorical predictors, interactions, transforms, no intercept but for `hurdle_cumulative`, any number
  of grouping factors, independent and correlated slopes, the offset) and `gp()` enter the linear predictor as for every family;
  `weights()` is taken by the three that are not ordinal (each row's log likelihood times its weight, as brms writes it and as D03
  takes it, tested row by row and in `log_lik()`). Refused naming the construct (D108): links other than logit, formulas on the
  auxiliary parameters, `cens()`, `trunc()`, `mo()`, `me()`, `mi()`, `mm()`, `gr(by = )`, `cs()`, autocorrelation terms,
  shrinkage priors on `b`, multivariate, non-linear (`nl=True`) and mixture models, and `weights()` for `hurdle_cumulative`.

## Evidence

The oracle cases `zero_one_inflated_beta_varint`, `zero_inflated_beta_binomial_varint`, `xbeta_varint` and
`hurdle_cumulative_varint` (each `~ x + (1 | g)` on `tier_e` data) pass gates G1-G6: design, response, trials and group coding;
log density with and without the Jacobian within 7.1e-16 relative and its prior part within 3.4e-16; every gradient coordinate
within 1.5e-9 (`xbeta`, PyTensor's betainc gradient), 6.3e-10 (the Beta families, PyTensor's digamma) and 4.2e-15
(`hurdle_cumulative`); the linear predictor, group effects and raw-scale thresholds; `predict()` against `posterior_linpred` and
`posterior_epred` within 3.1e-15 on fitted and new rows with and without group effects; the JAX density within 4.4e-15; and brms's
`default_prior()` tables row for row. The JAX gradient (NumPyro's path) agrees with PyTensor's within 1.5e-9 at every frozen point.
`tests/test_bounded_families.py` checks the incomplete Beta function and every family's rows on both paths against the 50-digit
reference, the untaken branches, the likelihoods' draws, each model's density against its rows and priors, priors, responses,
`value_point`, predictions by hand, weights, every refusal, the term model and `gp()`, post-processing, `describe()`, prior
predictive draws, defaults and the fit record. Nothing samples.
