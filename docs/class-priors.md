# A class-wide prior with overrides

brms gives one prior to a whole class and overrides it for single coefficients or grouping factors:
`prior(normal(0, 2), class = b) + prior(normal(0.5, 1), class = b, coef = x)`, `prior(normal(0, 1), class = sd) +
prior(normal(0, 0.5), class = sd, group = h)`. In AutoBayes the class-wide prior is the entry `"*"` of the class's mapping
(divergence D173, changed by Andrew's decision of 26 October 2026):

```python
priors = ab.Priors(intercept=ab.Normal(0, 5),
                   b={"*": ab.Normal(0, 2), "x": ab.Normal(0.5, 1)},      # every coefficient but x: normal(0, 2)
                   sd={"*": ab.HalfNormal(1), "h": ab.HalfNormal(0.5)},   # every SD but h's: half-normal(1)
                   cor={"g": ab.LKJ(2)}, sigma=ab.HalfNormal(1))
m = ab.gaussian("y ~ x + w + z + (1 + x | g) + (1 | h)", data, priors, categorical={"z": "p"})
```

**Why `"*"`.** The mappings already name what they cover: `b` by term (a categorical term by level), `sd` by grouping factor (a
factor with slopes by coefficient), `cor` by factor (docs/default-priors.md). `"*"` cannot be a term's or a factor's name, so the
class-wide prior is unambiguous inside the same mapping, and a model without it reads exactly as before. A keyword of its own
(`b_default=...`) would have split one class over two arguments.

**Precedence, brms's.** A coefficient's own entry over its factor's over the class's:

- `b`: `{"*": p, "x": q}` gives `x` the prior `q` and every other coefficient `p`; `{"*": p, "z": {"q": r}}` gives level `q`
  of the factor `z` the prior `r` and its other levels `p` (an ordered factor's mapping by contrast, `{".Q": r}`, an
  interaction's by its columns' levels, `{"x:z": {"q": r}}`, and a transform's by its columns, `{"poly(w, 2)": {"2": r}}`,
  likewise).
- `sd`: `{"*": p, "h": q}` gives every SD of `h` (each of its coefficients, brms's group row) `q` and every other SD `p`;
  `{"*": p, "g": {"x": q}}` gives the slope `x` of `g` the prior `q` and `g`'s intercept `p`.
- `cor`: `{"*": ab.LKJ(2), "h": ab.LKJ(4)}` for two correlated factors.

An entry naming no term (factor, level, coefficient) of the model is refused, naming it. Without `"*"` the mappings must name
every term or factor exactly, as before, and a bare `sd=` prior is not broadcast over two factors or a varying slope (D1, D173).
Bambi takes one distribution family per term, so the levels of one categorical term still share a family. `"*"` works wherever
these classes do: a response of a multivariate model, a mixture component, a distributional formula's `b`, a non-linear
parameter's `b` (whose `"Intercept"` is still named) and `sd`.

**The table.** `Model.prior_table()` writes the class-wide prior on the class row and each override on its own row, the rows it
covers empty, as brms's `prior_summary()` shows them; `describe()` lists each parameter's resolved prior and the class-wide ones.
Two columns differ from `prior_summary()` of a brms fit (D173): a covered row's `source` is the covering prior's (`user`), where
brms writes `default`, and the correlation's class is `cor` with `lkj(eta)`, as in `default_prior()`, where a fit's summary says
`L` with `lkj_corr_cholesky(eta)`.

**Not offered** (D178): `"*"` in the other keyed classes (`simo`, `sdgp`, `lscale`, `meanme`, `sdme`, which stay exact mappings by
column) and `ab.prior_draws()` of a model with a class-wide prior, refused because brms names the prior draws of a class row and
its coef rows in a way no probe here verifies.

**Evidence.** The oracle case `gauss_prior_overrides` (the two overrides above on two grouping factors and a correlated slope)
passes every gate: brms's `lprior` is the sum of the resolved priors at every point. `tests/test_call_surface.py` checks the
resolution of each form, the refusals, the table against that fixture's `prior_summary()` and the contexts above.
