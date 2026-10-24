# Known standard errors and rates: `se()` and `rate()`

brms-parity feature F06a (brms's response addition terms `se()` and `rate()`; divergence rows D156 and D157).

```python
import autobayes as ab
meta = ab.gaussian("y | se(sdx) ~ 1 + (1 | g)", data, ab.Priors(intercept=ab.Normal(0, 5), sd=ab.HalfNormal(1)))
both = ab.student("y | se(sdx, sigma = TRUE) ~ x + (1 | g)", data,
                  ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1),
                            nu=ab.Gamma(2, 0.1)))
rate = ab.poisson("cnt | rate(exposure) ~ x + (1 | g)", data, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1),
                                                                       sd=ab.HalfNormal(1)))
```

## The model

| term | families | row i's likelihood | brms's Stan data |
| --- | --- | --- | --- |
| `se(col)` | Gaussian, Student-t | `Normal(mu_i, se_i)`, `StudentT(nu, mu_i, se_i)`: no sigma | `se`, and `sigma = 0` |
| `se(col, sigma = TRUE)` | Gaussian, Student-t | the scale `sqrt(sigma^2 + se_i^2)` | `se` |
| `rate(col)` | Poisson, negative binomial (log link) | the mean `exp(eta_i) d_i`; the negative binomial's shape `shape * d_i` | `denom` |

`se(col)` is a meta-analysis: each row's standard error is known and brms fixes the residual SD at zero (its Stan data hold
`sigma = 0`), so the model has no sigma at all, no sigma prior and no sigma row in brms's `default_prior()`. With `sigma = TRUE`
(also written `se(col, TRUE)`) sigma is learned as usual and adds to each row's standard error in quadrature, brms's
`sqrt(square(sigma) + se2)`. `rate(col)` divides the mean by a known denominator (an exposure): brms writes the log mean as
`eta_i + log(denom_i)` and multiplies the negative binomial's shape by `denom_i` too, so that `Var[y_i] = d_i (mu_i + mu_i^2 /
shape)`. brms's default priors read the response as without either term (the Intercept's location is `median(log y)`, not of
`log(y / d)`; checked on the frozen fixtures).

## How it is built

`AdditionTerm("se", column, sigma=...)` and `AdditionTerm("rate", column)` in `ModelSpec.additions` (docs/addition-terms.md);
`formula.addition_data` gives brms's `se` and `denom` on a frame, validated by `autobayes/se_rate.py`, and `Design.se`,
`Design.denom` hold them on the fitted rows. The likelihood is the family's own PyMC distribution inside the addition-term
`CustomDist` (`likelihoods.addition_family`), with its parameters set per row by `se_rate.likelihood_parameters`: the scale
`FamilySpec.se_parameter` (Bambi's `sigma`) is `se_i`, or `sqrt(sigma^2 + se_i^2)`, and each of `FamilySpec.rate_parameters`
(the negative binomial's shape, Bambi's `alpha`) is multiplied by `d_i`. Under `rate()` the linear predictor gains `log(d_i)`
before the link (the hook `likelihoods.monotonic_family` gives the monotonic terms), so Bambi's parent `mu` is the expected
count `exp(eta_i) d_i`, brms's `posterior_epred`, which gate G3 compares with brms's. Without `sigma = TRUE` the model's family spec
is `se_rate.se_only_spec(fam)`: the family's spec without its scale parameter, whose Bambi likelihood has no `sigma`
(`model.family_spec`, `families.family_of(model)`; `ab.default_priors` resolves the same spec). Draws, `log_lik()` and
`describe()` are the families' own, each reading the row's scale or denominator through `se_rate`.

## Using it

- Formula: `se(col)`, `se(col, sigma = TRUE)` (or `TRUE` / `FALSE` positionally, R's `T` / `F` too), `rate(col)`, alone after the
  `|`. The canonical formula writes `se(col)`, `se(col, sigma = TRUE)` and `rate(col)`.
- Data: the standard errors finite and non-negative (brms: "Standard errors must be non-negative"), and positive without
  `sigma = TRUE`, where a zero is the row's whole scale (D156); the denominators finite and positive (brms refuses a zero or
  negative one too). A non-numeric, boolean, complex or missing value is refused naming the row or column.
- Priors: under `se(col)` no `sigma` (passing one is refused naming the term); otherwise the family's usual classes.
  `value_point()` takes no `sigma` under `se(col)`.
- `predict()`: `kind="linear"` is `eta` (brms's `posterior_linpred`, without the denominator); `kind="mean"` is `mu` under `se()`
  and `exp(eta) * d` under `rate()` (brms's `posterior_epred`), on new rows with their own denominators (the column is required
  then); `kind="response"` draws with each row's scale, or the scaled mean and shape, new rows with their own `se()` or
  `rate()` column. `log_lik()`, `loo()`, `waic()` and `bayes_R2()` read the same per-row parameters.
- `describe()` says what the term does (a data line and a likelihood line); the target identity includes the standard errors
  and denominators, and the fit record lists the term among `addition_terms`.

## Refused, naming the construct

`se()` or `rate()` next to another addition term (`weights()`, `cens()`, `trunc()`, `mi()`; brms accepts
`se(s) + weights(w)`, for one), with a distributional formula (`sigma ~ ...` next to `se(col, sigma = TRUE)`, `shape ~ ...` next to
`rate()`; brms refuses `sigma ~ ...` next to `se(col)` too), an expression inside the call (compute the column), `rate()` under
the Poisson's sqrt and the negative binomial's identity links (D133), and the other families (brms takes `rate()` for the
geometric family as well, for one): D157. Multivariate, non-linear, mixture, autocorrelation, SAR and shrinkage-prior models
refuse every addition term already (D56, D61, D55, D70, D143, D102). The population- and group-level terms, offsets, `mo()` and
the other linear-predictor terms compose with both (`tests/test_se_rate.py` checks the likelihood of the composed model).

## Evidence

Oracle cases (gates G1-G6 against frozen brms fixtures): `gauss_se_meta` (`se(sdx)`, a varying intercept, no sigma),
`gauss_se_sigma` and `student_se_sigma` (`se(sdx, sigma = TRUE)`, the Student-t's nu learned), `poisson_rate` and
`negbinomial_rate` (`rate(exposure)`): G1 compares brms's `se`, `sigma = 0` and `denom` with the model's, G2 the log density and
gradient, G3 the graph's mean with brms's `posterior_epred` under `rate()`, G4 `posterior_linpred` and `posterior_epred` on fitted
and new rows, G6 brms's default-prior table (no sigma row under `se(sdx)`). `tests/test_se_rate.py` covers the grammar and every
refusal, the data's validation, each form's likelihood against SciPy at chosen values, compositions, the spec without sigma,
priors and defaults, predictions and seeded draws, `log_lik()`, prior-predictive draws, `describe()`, the identity and the
record, and G1 failing when a term is dropped or reads another column. The brms syntax corpus records `se(sey)`,
`se(sey, sigma = TRUE)` and `rate(expo)` as `match`. Nothing samples.
