# Finite mixtures: `mixture(gaussian, gaussian)`

brms-parity feature C05 (brms's `mixture()`).

```python
import autobayes as ab
priors = ab.Priors(components=[ab.Priors(intercept=ab.Normal(-1, 2), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1)),
                               ab.Priors(intercept=ab.Normal(1, 2), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1))],
                   theta=ab.Dirichlet(2))
model = ab.mixture("y ~ x", data, priors, families=["gaussian", "gaussian"])
print(model.describe())
```

## The model

K Gaussian components with the same formula, each with its own linear predictor and residual SD, weighed by the mixing
proportions:

    mu_k,i = a_c[k] + (x_i - mean(x)) b[k]                       k = 1..K   (brms's mu1, ..., muK)
    y_i    ~ sum_k theta[k] Normal(mu_k,i, sigma_k)                         (brms's log_mix of normal_lpdf)
    a_c[1] < a_c[2] < ... < a_c[K]                                          (brms's ordered[K] ordered_Intercept)
    theta  ~ Dirichlet(con_theta)                                           (brms's simplex[K] theta)

Each component has its own centred intercept `a_c[k]` (brms's `Intercept_mu<k>`, the component's mean at the predictor
means) and its own coefficients on the same design (brms's `b_mu<k>` on `X_mu<k>`), and its own `sigma_k` (brms's
`sigma<k>`). Under brms's default `order = "mu"` (its default for components of one family) the components are identified
by their centred intercepts, which are ordered: brms declares them `ordered[K] ordered_Intercept` and samples them on Stan's
`ordered` coordinates `(a_c[1], log(a_c[2] - a_c[1]), ...)`, with each component's intercept prior added at its own entry
(not renormalised to the ordered region). Row `i`'s log density is `log_sum_exp_k(log theta[k] + normal_lpdf(y_i | mu_k,i,
sigma_k))`. brms reports the raw-scale intercepts `b_mu<k>_Intercept = a_c[k] - sum_p mean(x_p) b_p[k]`; so does the
posterior here, as for every family whose intercept Bambi uncentres after sampling.

## How it is built

Three earlier pieces are reused rather than copied:

- **B05's per-category design.** Bambi's categorical family gives every coefficient, the intercept included, a trailing
  axis over the response's levels and centres the design; `likelihoods.gaussian_mixture_family` subclasses it with the
  components as the levels (coordinate `mixture_dim`, labels `"1"`..`"K"`), so the graph's `mu` is `(rows, K)`, brms's
  `mu<k>` side by side, and every term's coefficient is one variable over `(levels, K)` or `K`.
- **B05's ordered transform.** The intercept is `likelihoods.ordered_thresholds` (PyMC's `ordered` transform, whose
  coordinates and Jacobian are Stan's), extended to take one prior per entry of one distribution family; its forward draws
  are the entries' independent draws kept where they come out ordered (rejection), which is the product density restricted
  to the ordered region (sorting would be another distribution once the priors differ; with one prior for every entry the
  sorted draws are kept).
- **B03's stick-breaking simplex.** `theta` is `monotonic.simplex_rv`, a Dirichlet under brms's concentrations sampled on
  Stan's stick-breaking coordinates (`theta_stickbreaking__`).

The likelihood creates `sigma1`, ..., `sigmaK` (each under its component's `sigma` prior, sampled on the log scale as Stan's
`real<lower=0>`) and `theta`, in that order, and adds `likelihoods.gaussian_mixture_logp`, the sum over components written
with the largest term factored out (B05's `_log_sum_exp`), finite with a finite gradient for every finite argument on the
PyTensor and JAX paths. `families.mixture.check_graph` verifies the build: the variables, their transforms, dims and
coordinates, and that the graph's `mu` at a point with a distinct value per component is `a_c[k] + (X - mean(X)) b[:, k]`.

## API: the priors

`Priors(components=[...], theta=...)`: one `Priors` per component in the order of `families`, each the component family's
own (a Gaussian component's `intercept`, `b` and `sigma`, brms's `class = Intercept` and `b` with `dpar = mu<k>` and
`class = sigma<k>`), and `theta`, the Dirichlet of the mixing proportions (brms's `class = theta`; `ab.Dirichlet(2)` is
brms's one-number shorthand, `ab.Dirichlet(2, 3)` one concentration per component). The frozen case had proposed brms's flat
names, `Priors(mu1={"intercept": ..., "b": ...}, sigma1=..., ..., theta=...)`; `Priors` is a frozen dataclass whose fields
are fixed, so an open-ended set of `mu<k>` and `sigma<k>` keywords would need either a cap on the number of components or a
hand-written constructor for every field. With one `Priors` per component the number of components is free, each
component's priors are read, validated, tabled and defaulted exactly as a Gaussian model's are (B01's reading of a
parameter's formula priors as a `Priors` of their own, taken one step further), and a later non-Gaussian component would
bring its own family's priors unchanged. The case's `autobayes` section uses this form.

Within a class the components' priors share one distribution family, each with its own parameters: the intercepts are one
ordered vector and each term's coefficients one variable with a component axis (divergence D55). Nothing else is taken at
the top level of a mixture's `Priors`.

## Using it

- `ab.mixture(formula, data, priors, families=[...], order=None, center_predictors=True, categorical=None)`, or
  `ab.Model(..., family="mixture", families=[...])`. `families` names every component (`["gaussian"] * K`, K >= 2); `order`
  is `"mu"` (or `True`, or `None` for brms's default).
- Names: `model.components` (the families), `model.mixture_names` (`dpars` `mu1`.., `sigma` `sigma1`.., `theta`, the
  coordinate `mixture_dim` and its labels); `model.names` adds `sigma<k>`, `theta` and `component_dim`; the expected response
  is `mean`.
- `value_point(intercept=[...K increasing], b=(columns, K), sigma=[...K], theta=[...K simplex])`: the centred intercepts on
  the ordered coordinates (`Intercept_ordered__`), each `sigma<k>_log__`, theta on the stick-breaking coordinates.
- `predict()`: `kind="linear"` returns `eta` with a `component` dimension (brms's `posterior_linpred`, draws x rows x K);
  `kind="mean"` returns `mean = sum_k theta_k mu_k` (brms's `posterior_epred`); `kind="response"` draws, per draw and row, a
  component by theta and then its Normal (brms's `posterior_predict`), seeded. New rows need the formula's columns.
- `ab.default_priors(formula, data, family="mixture", families=[...])` returns brms's defaults: each component the Gaussian
  family's (Intercept `student_t(3, round(median(y), 1), max(2.5, round(mad(y), 1)))`, flat `b`, `sigma<k>`
  `student_t(3, 0, scale)` with lower bound 0) and `dirichlet(1)` on theta; `model.prior_table()` writes brms's rows (class
  `sigma<k>` and `theta`, then each component's `Intercept` and `b` rows under `dpar = mu<k>`, sorted as brms sorts them).
- `describe()` writes the components, their identification, theta, the likelihood and each component's priors out; the
  target identity holds the components, their priors and theta's with its concentrations; fit records label the priors
  `mu<k>:intercept`, `mu<k>:b:<column>`, `sigma<k>`, `theta`; `prior_predictive()` draws ordered intercepts and the response.

## Refused, naming the construct

Components of any family but the Gaussian (D52); `order = "none"` / `False` (D53); a formula per component, or formulas on
`sigma<k>` or `theta` (a list of formulas, brms's `bf(y ~ x, mu2 ~ z, theta1 ~ w)`; D54); group-level terms, `offset()`,
response addition terms and `mo()` (D55); an autocorrelation term `ar()`, `ma()`, `arma()` and the `ar` and `ma` priors (C04,
docs/autocorrelation.md; brms refuses the term in a mixture too); and, as brms refuses them under `order = "mu"`, `0 +` / `- 1` and
`center_predictors=False`, which leave no centred intercept to order. brms's `pp_mixture()` (the components'
posterior membership probabilities) is not offered.

## Evidence

Three frozen cases pass gates G1-G6 against brms 2.23.0: `mixture_gauss_gauss` (`y ~ x`, two components, a different
intercept prior per component, `dirichlet(2)`; the case the loop froze) and two frozen with this feature,
`mixture_gauss3_categorical` (`y ~ x + z` with a treatment-coded `z`, three components: three ordered coordinates and two
stick-breaking coordinates; a different prior per component on every class, a coefficient prior on `zq` of `mu2` only,
`sigma<k>` under `normal`, `student_t` and `exponential`, `dirichlet(2, 3, 4)`) and `mixture_gauss_default_priors`
(`y ~ x + w` under brms's defaults, the one-prior-for-every-entry path of the ordered intercepts). G1 matches each
`X_mu<k>` to the design by content and checks `con_theta`; G2 the log density with and without the Jacobian within
3.1e-16, 2.1e-11 and 5.2e-11 relative, the prior part within 2.3e-16, 9.9e-11 and 5.9e-10 (PyMC's half Student-t adds its
`log 2` in single precision, 1.9e-9 per half prior, docs/default-priors.md), and every gradient coordinate within 1.1e-14,
`ordered_Intercept`'s ordered coordinates and theta's stick-breaking coordinates recomputed from brms's values at every point
and checked against the fixture's unconstrained values before their gradient is compared; G3 the graph's `mu` against
`posterior_linpred` and each component's raw intercept against `b_mu<k>_Intercept`; G4 `posterior_linpred` (draws x rows x
K) and `posterior_epred` on fitted and new rows within 6.8e-16; G5 the JAX density; G6 the default-prior table.
`tests/test_mixture.py` checks the row density and its gradient against a 60-digit reference on both paths (rows 1e5 SDs
from every component, scales of 1e-3 and proportions of 1e-12), the draws, the ordered intercepts under a prior per entry
(their density, Jacobian and the truncated product's mean of the rejection draws), the graph, names and `value_point`, the
density, prior part and JAX gradient of two- and three-component models by hand, every refusal, predictions by hand,
prior-predictive draws, `describe()`, the identity, five `default_prior()` tables printed by brms (ten components sort as
brms sorts them), the fit record, and that the gates fail on a corrupted `mixture_gauss3_categorical` fixture (an ordered
or simplex coordinate, a `sigma3`, theta or `z[r]` gradient, `con_theta`, a raw intercept, the linear predictor, the mean,
`lprior`, `X_mu2`, the default-prior table) and on a model with another number of components or none. Nothing samples.
