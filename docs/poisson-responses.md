# Poisson count responses

```python
import autobayes as ab
model = ab.poisson("count ~ x + (1 | group)", data,
    ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd=ab.HalfNormal(1)))
print(model.describe())
```

Counts must be finite nonnegative integers from0 through1e6. Zero-count rows are retained; all-zero outcomes trigger a prior-sensitivity warning. Coefficients describe log expected counts. Mean predictions return mu=exp(eta); response predictions additionally return integer y. Both existing new-group policies work through the original-input prediction path. There is no residual sigma, learned dispersion, nu or zero-inflation in this first count-family implementation; an exposure enters as `offset(log(exposure))` since brms-parity feature A06 (docs/offset.md). Intercept-only formulas use an explicit empty coefficient-prior mapping b={}.

The input-count limit does not truncate the likelihood or generated counts. Nonfinite predicted means and rates beyond the integer random generator's supported range are refused rather than clipped. Finite means beyond that generator range remain available with kind="mean". The ordinary and unseen-group prediction routes both use the same manual design reconstruction to perform these checks before drawing.

Validation uses original-input Poisson log likelihoods and derivatives, including observed counts0,1,1e6 and log means -30,0,log(1e6),30, plus CPU JAX, hierarchical/category composition and fixed-draw predictive checks. This is a bounded numerical domain, not an all-real-log-rate stability guarantee. No new Poisson recovery fits or GPU execution are claimed; model adequacy, overdispersion and sampling diagnostics still require evaluation in applications.
