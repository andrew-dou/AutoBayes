# brms's default priors, opt-in

brms-parity feature A07 (brms's `default_prior()`).

```python
import autobayes as ab
priors = ab.default_priors("y ~ x + (1 | g)", data)     # brms 2.23's defaults for this formula, data and family
model = ab.gaussian("y ~ x + (1 | g)", data, priors)
print(model.describe())                                   # each default is marked "(brms default ...)"
model.prior_table()                                       # the rows brms's default_prior() prints for this model
```

Nothing is filled in unasked (divergence D1): the constructors still require every prior, and brms's defaults reach a
model only when `ab.default_priors()` is called and its result passed in.

## What brms does and what this returns

| class | brms 2.23 default | `ab.default_priors()` |
| --- | --- | --- |
| `Intercept` (centred) | `student_t(3, location, scale)` | `StudentT(3, location, scale)` |
| `b` | flat (the empty prior string) | `Flat()`: PyMC's `Flat`, improper, adds 0 to the log density |
| intercept with `center_predictors=False` | a `b` coefficient `Intercept`, flat | `Flat()` |
| `sd`, `sigma` | `student_t(3, 0, scale)`, lower bound 0 | `HalfStudentT(3, scale)` |
| `cor` | `lkj(1)` | `LKJ(1)` |
| `shape` (negative binomial) | `inv_gamma(0.4, 0.3)`, lower bound 0 | `InverseGamma(0.4, 0.3)` (shape and scale, as Stan's `inv_gamma`) |
| `shape` (Gamma), `phi` (Beta) | `gamma(0.01, 0.01)`, lower bound 0 | `Gamma(0.01, 0.01)` |
| `nu` (Student-t, learned) | `gamma(2, 0.1)`, lower bound 1 | `Gamma(2, 0.1)`, truncated at 1 as in A13 |
| `zi`, `hu` (zero-inflated, hurdle; B06) | `beta(1, 1)`, lower bound 0, upper bound 1 | `Beta(1, 1)` |
| `shape` (Weibull; D01) | `gamma(0.01, 0.01)`, lower bound 0 | `Gamma(0.01, 0.01)` |
| `alpha` (skew normal; D01) | `normal(0, 4)`, unbounded | `Normal(0, 4)` |
| `beta` (ex-Gaussian; D01) | `gamma(1, 0.1)`, lower bound 0 | `Gamma(1, 0.1)` |
| `ndt` (shifted lognormal; D01) | `uniform(0, min_Y)`, lower bound 0, upper bound `min_Y` (the smallest response) | `Uniform(0, "min_Y")` (docs/continuous-families.md) |
| `phi` (beta-binomial; D02) | `gamma(0.01, 0.01)`, lower bound 0 | `Gamma(0.01, 0.01)` |
| `sigma` (`negbinomial2`; D02) | `gamma(0.4, 0.3)`, lower bound 0 (not data-dependent) | `Gamma(0.4, 0.3)` (docs/discrete-families.md) |
| `Intercept` (von Mises, `tan_half` link; E02) | `student_t(1, 0, 1)` whatever the data; its `sd` scale 2.5 | `StudentT(1, 0, 1)` (`FamilySpec.default_intercept_prior`) |
| `shape` (inverse Gaussian; E02) | `gamma(0.01, 0.01)`, lower bound 0 | `Gamma(0.01, 0.01)` |
| `kappa` (von Mises; E02) | `gamma(2, 0.01)`, lower bound 0 | `Gamma(2, 0.01)` |
| `quantile` (asymmetric Laplace; E02) | `beta(1, 1)`, lower bound 0, upper bound 1; no row when fixed (`bf(quantile = p)`) | `Beta(1, 1)`; none with `quantile=p` |
| `nu` (Frechet; E02) | `gamma(2, 0.1)`, lower bound 1 | `Gamma(2, 0.1)`, truncated at 1 as the Student-t nu |
| `xi` (generalized extreme value; E02) | `normal(0, 2.5)`, unbounded (brms's prior of `tmp_xi`) | `Normal(0, 2.5)` (docs/continuous-families.md) |
| `bs`, `ndt`, `bias` (wiener; F04) | `gamma(1, 1)` (lower bound 0), `uniform(0, min_Y)` (bounds 0 and `min_Y`), `beta(1, 1)` (bounds 0 and 1); the `Intercept` and `sd` read `rt` as given (identity) | `Gamma(1, 1)`, `Uniform(0, "min_Y")`, `Beta(1, 1)` (docs/wiener.md) |
| `Intercept` of `bs ~ ...` (wiener; F04) | `normal(-0.6, 1.3)` whatever the data | `Normal(-0.6, 1.3)` in `Priors(bs={"intercept": ...})` |
| `sbhaz` (cox; F05) | `dirichlet(1)` on the baseline hazard's simplex; the `Intercept` and `sd` read `log(time)` (log link), so the Intercept's location moves with the unit of time (D152) | `Dirichlet(1)` (docs/cox.md) |
| `Intercept` of the cumulative thresholds, and of each category of a categorical model (B05) | `student_t(3, 0, 2.5)` on the class row, one row per threshold (`coef` `1`, ...) or per category (`dpar` `muc2`, ...); the thresholds keep the class `Intercept` when not centred | `StudentT(3, 0, 2.5)`, one prior for all thresholds or categories (`Model.prior_table()` writes the rows) |
| `Intercept` of a distributional formula (`dpar = sigma`, B01) | `student_t(3, 0, scale)` with the scale of `log y` (zeros read as 0.1; 2.5 when a response is negative) | `StudentT(3, 0, scale)` in `Priors(sigma={"intercept": ...})` |
| `Intercept` of `shape ~ ...`, `phi ~ ...` (B01) | `student_t(3, 0, 2.5)` | `StudentT(3, 0, 2.5)` |
| `b` of a distributional formula (B01) | flat | `Flat()` |
| a mixture's `Intercept` and `b` (`dpar = mu<k>`), `sigma<k>`, `theta` (C05) | each Gaussian component the Gaussian family's defaults (its own rows under `dpar = mu<k>` and class `sigma<k>`); `dirichlet(1)` on `theta` | one `Priors` per component from the Gaussian family's defaults, `theta=Dirichlet(1)` (`family="mixture"`, `families=[...]`; docs/mixtures.md) |
| `sdgp` of a Gaussian process (C03) | `student_t(3, 0, scale)` on the class row, lower bound 0 (a group-level SD's default) | `HalfStudentT(3, scale)` in `Priors(sdgp={"x": ...})` |
| `lscale` of a Gaussian process (C03) | on the coefficient row `gpx` (the class row empty, lower bound 0): the inverse gamma with 1% of its mass below the smallest distance between distinct scaled inputs (floored at 0.01) and 1% above the largest, its parameters printed to 6 decimals; `normal(0, 0.5)` with two distinct inputs | `InverseGamma(a, b)` solved exactly (the sixth decimal can differ from brms's own solver's, D67), or `HalfNormal(0.5)`, in `Priors(lscale={"x": ...})` (docs/gaussian-processes.md) |
| `ar`, `ma` of an autocorrelation term (C04, `cov = FALSE`) | flat, no bounds, one class row each | `Flat()` |
| `meanme`, `sdme` of an `me(x, sdx)` term (C06) | flat on the class rows (`sdme` with lower bound 0), one `coef` row `mex` each | `Flat()` keyed by the noisy column (`sdme`'s flat on the positive half line, PyMC's `HalfFlat`; docs/missing-and-measurement-error.md) |
| `Intercept`, `sigma` of a response with missing values (`x \| mi()`, C06) | brms's median and MAD of the column are NA: location 0 and scale 2.5 | `StudentT(3, 0, 2.5)`, `HalfStudentT(3, 2.5)` |

The location and scale are brms's `def_scale_prior` arithmetic on the response read through the family's link
(`FamilySpec.default_prior_link`): the response itself for the Gaussian and Student-t families; `log(y)`, zeros read as
0.1, for the Poisson, negative binomial and Gamma (log link) and the lognormal (brms's log-scale family), and since D01 the
response itself for the skew-normal and ex-Gaussian families and `log(y)` for the Weibull, exponential and shifted lognormal
(checked against brms on rescaled responses; the extra parameters' defaults above are constants), and since D02 `log(y)`,
zeros read as 0.1, for the geometric and `negbinomial2` families (probed on a wide-scale response), and since E02 `1 / y^2`
under the inverse Gaussian's default `1/mu^2` link, `log(y)` under its log link and for the Frechet, and the response itself for
the asymmetric Laplace and generalized extreme value families (probed the same way); and not at
all under a logit or probit link (Beta, Bernoulli, binomial, and since D02 the beta-binomial), where they are 0 and 2.5. Then

- `location = round(median(y_link), 1)`, minus the mean of the offset term when there is one (not rounded again);
- `scale = max(2.5, round(mad(y_link), 1))`, shared by the Intercept, `sd` and `sigma`.

These are R's functions, reproduced on doubles: `median` (the two middle values averaged by R's `mean`; so the location
of counts `1, 2` is the median of their logs, 0.35, rounded to 0.3, not `log(1.5)`), `mad` (constant 1.4826), `mean`
(R's second pass), R 4's `round(x, 1)` (the nearer of the two one-decimal candidates in double arithmetic, so
`round(0.45, 1)` is 0.4 and `round(2.45, 1)` is 2.5), and `as.character` for printing (15 significant digits chosen as
R's `formatReal` does: `student_t(3, -0.978004218445523, 2.5)` for the offset case). The family's auxiliary and learned
parameters take the family's own defaults (`FamilySpec.default_scale_prior`, `default_learned_priors`).

## Using it

- `ab.default_priors(formula, data, *, family="gaussian", categorical=None, nu=None, link=None, center_predictors=True)`
  takes the constructor's own keywords, which say which model the defaults are for. `nu=None` for the Student-t family
  means nu is learned and gets `gamma(2, 0.1)`; a fixed `nu=4` gets no prior.
- Every prior it returns has `source="default"` (a field of `Prior` that equality, `str()`, the target identity and the
  graph ignore). `describe()` marks each with brms's spelling and adds a "Default priors" paragraph,
  `model.default_prior_labels` lists them, and `fit_record()` records them under `"default_priors"`. Replacing a class
  (`dataclasses.replace(priors, b=ab.Normal(0, 1))`) makes that class the user's again.
- `Priors.table(spec, family, center_predictors=...)` and `Model.prior_table()` write any priors in brms's table form:
  the eleven columns of `default_prior()` (`prior`, `class`, `coef`, `group`, `resp`, `dpar`, `nlpar`, `lb`, `ub`, `tag`,
  `source`), brms's coefficient names (`zq`, `x:zq`, `zq:hh2`, `logxpos`, `IxE2`), in brms's order. A prior shared by a
  whole class sits on the class row and the member rows are empty, as brms writes its defaults; otherwise each prior
  sits on the most general row it covers. `brms_prior()` spells one prior (`HalfStudentT(3, 2.5)` on `sd` is
  `student_t(3, 0, 2.5)`).
- A flat prior is improper, so `prior_predictive()` is refused for a model with one (brms likewise refuses to sample
  from improper priors); sampling the posterior is unaffected.
- An intercept-only model has no coefficient and now needs no `b` prior (D2); one given anyway changes nothing.
- A multivariate formula (C01, docs/multivariate.md) takes `rescor=True` or `rescor=False`:
  `ab.default_priors("mvbind(y, y2) ~ x + (1 | g)", data, rescor=True)` gives each response its own defaults under `resp`
  (its Intercept location and scale read from that response), `lkj(1)` on a correlated term's `cor` once for every
  response (brms's `cor` has no `resp`) and `lkj(1)` on `rescor`; the table has each response's rows under its `resp`. A list
  of formulas with `mi()` (C06, `rescor=False`) is one too: each response's own defaults, an `mi()` term's coefficient flat.
- `ab.Flat()` is accepted for the intercept and `b` only; `ab.InverseGamma(alpha, beta)` for `sd`, `sigma`, `shape` and
  `phi`, not for `nu` (its truncation at 1 is not one this library renormalises, D22).
- Refused, with the constructors' messages: a family or link the library lacks (`sratio`, brms's `cauchit` link),
  a formula outside the supported subset, data the constructors refuse, `nu=` for a family without it; and a family
  whose parameter has no recorded brms default.

## Exactness and evidence

The arithmetic is exact for all nine families and both links of the Bernoulli and binomial families: gate G6 compares
`ab.default_priors()` written as brms's table with the `default_prior()` table of every frozen fixture whose model
AutoBayes builds, every row and column (strings after removing whitespace and writing each number as the double it
denotes), and the 38 tables agree; a fixture for a feature not yet implemented stays gated. For a Student-t case that
fixes nu (`student_varint`), G6 compares under `nu=None`, the model brms's `default_prior()` describes (D23). Five
fixtures frozen with no priors on the brms side, so that brms's defaults are the model (`gauss_varint_default_priors`,
`poisson_categorical_default_priors`, `bernoulli_slope_default_priors` with `lkj(1)`, `negbinomial_default_priors` with
`inv_gamma(0.4, 0.3)`, `student_default_priors` with a scale of 4.4 and the learned nu), pass G1-G6: the densities,
gradients and prior parts under the default priors themselves equal brms's. `tests/test_default_priors.py` checks ten
tables brms printed for a small frame (large scale, offset with zeros, uncentred, probit, intercept-only, `g:h`,
transforms, no intercept), R's rounding and printing, the mapping to `Priors`, the graph, the refusals, `describe()` and
fit records.

Two limits of exactness, neither met by the fixtures: R's `mean()` accumulates in extended precision where R is built
with an 80-bit long double (x86-64), and takes a single pass over an integer vector; this transcription follows the
double-precision build the fixtures were frozen on (aarch64), so an offset's mean could differ there in its last bit,
which brms's 15-digit printing can show only when it falls on a rounding boundary. And PyMC's half distributions add
their `log 2` in single precision, 1.9e-9 per half prior in the log density, far inside G2's 1e-8 relative tolerance.
