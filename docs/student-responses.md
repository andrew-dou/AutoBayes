# Student-t responses with fixed tails

> brms-parity A13 added learned degrees of freedom: `ab.student(..., nu=None)` with `Priors(nu=...)`; see
> [Student-t learned nu](student-learned-nu.md). This page describes a fixed `nu`, which is unchanged.

```python
import autobayes as ab
model = ab.student("y ~ x + (1 | group)", data,
    ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1),
              sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1)),
    nu=4)
print(model.describe())
```

The response is Student-t with identity-linked location mu, explicit scale prior sigma, and fixed finite degrees of freedom nu>1. The upper-case StudentT helper still denotes a coefficient prior. This API does not learn nu, supply silent priors, or support distributional auxiliary formulas. Existing numeric/categorical terms, interactions and independent varying intercepts compose with this family; unmerged varying-slope work is separate.

Mean predictions return mu. Response predictions draw Student-t noise, including for new groups under either existing policy. Sigma is scale: SD is sigma*sqrt(nu/(nu-2)) only when nu>2; variance is infinite for1<nu<=2. Records retain nu and comparisons refuse unequal tail settings. Predictors/responses named sigma or nu are reserved for this family.

The pinned Bambi/PyMC graph and CPU JAX log density/gradient are checked deterministically. No new Student-t recovery fits or GPU execution are claimed by this change. Use the existing fit interface only with the usual convergence and computational validation; deterministic graph checks are not sampling calibration.
