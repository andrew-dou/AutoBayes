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
| `Intercept` of the cumulative thresholds, and of each category of a categorical model (B05) | `student_t(3, 0, 2.5)` on the class row, one row per threshold (`coef` `1`, ...) or per category (`dpar` `muc2`, ...); the thresholds keep the class `Intercept` when not centred | `StudentT(3, 0, 2.5)`, one prior for all thresholds or categories (`Model.prior_table()` writes the rows) |
| `Intercept` of a distributional formula (`dpar = sigma`, B01) | `student_t(3, 0, scale)` with the scale of `log y` (zeros read as 0.1; 2.5 when a response is negative) | `StudentT(3, 0, scale)` in `Priors(sigma={"intercept": ...})` |
| `Intercept` of `shape ~ ...`, `phi ~ ...` (B01) | `student_t(3, 0, 2.5)` | `StudentT(3, 0, 2.5)` |
| `b` of a distributional formula (B01) | flat | `Flat()` |

The location and scale are brms's `def_scale_prior` arithmetic on the response read through the family's link
(`FamilySpec.default_prior_link`): the response itself for the Gaussian and Student-t families; `log(y)`, zeros read as
0.1, for the Poisson, negative binomial and Gamma (log link) and the lognormal (brms's log-scale family); and not at
all under a logit or probit link (Beta, Bernoulli, binomial), where they are 0 and 2.5. Then

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
