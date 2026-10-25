# Ordered factors: polynomial contrasts (F08)

brms codes an ordered factor predictor with R's polynomial contrasts, `contr.poly`: `y ~ fo` with `fo` an ordered factor of
the levels `lo < mid < hi` has the design columns `fo.L` and `fo.Q` (the coefficients `b_fo.L`, `b_fo.Q`), a linear and a
quadratic trend across the ordered levels, not treatment dummies. AutoBayes reads it the same way. Before F08 it built
treatment dummies against the declared level and said nothing: the syntax audit's one silent misreading
(`pop_ordered_factor`, docs/brms-parity/reports/syntax-audit.pdf).

```python
data["fo"] = pd.Categorical(data["fo"], categories=["lo", "mid", "hi"], ordered=True)
m = ab.gaussian("y ~ x + fo", data, priors, categorical={"fo": "lo"})        # columns x, fo.L, fo.Q
m = ab.gaussian("y ~ x + ordered(z)", data, priors)                           # R's ordered(): ordered(z).L, ordered(z).Q
```

**Which factors.** An ordered pandas Categorical declared with `categorical=` (the declared value must be its lowest
level: polynomial contrasts have no reference level, and any other level is refused; pass the column unordered,
`.cat.as_unordered()`, for treatment coding against another level), and `ordered(col)` written in the formula, as R's
`ordered()`: a pandas Categorical in its own order, whole numbers in numeric order, text sorted (Python's `sorted()`, the
C-locale order brms has under `LC_COLLATE=C`). The levels are the fitted rows' levels in that order; declared categories no
row takes are dropped, as brms drops them. The first factor of a formula without an intercept (`0 + fo`) is coded with one
indicator per level, as R codes it; `ordered(col)` there is refused.

**The contrasts.** Column `d` (degree `d = 1, ..., k - 1`) is the polynomial of degree `d` in the level's rank `1, ..., k`
that is orthogonal to every polynomial of lower degree over the `k` ranks with equal weights, has a positive leading
coefficient and unit Euclidean norm (`formula.poly_contrasts`): the three-term recurrence of monic orthogonal polynomials in
exact rational arithmetic, rounded once. R's names are `.L`, `.Q`, `.C`, `^4`, `^5`, ...; brms renames `^` to `E` (`foE4`).
`tests/test_syntax_fixes.py` checks them against an independent exact Gram-Schmidt construction for 2 to 16 levels (within
4e-16).

**More than 16 levels are refused** (divergence D168). R's `contr.poly`, which brms uses, departs from the polynomial
contrasts as the number of levels grows: probed on 2 to 95 levels, its largest difference from the exact contrasts is 3.1e-15
at 9 levels, 8.7e-13 at 16, 1.6e-12 at 17 (beyond the oracle's 1e-12) and of order one from 23 levels on (0.89 at 23), where
its columns are still orthonormal (to 4e-16) but no longer polynomial in the rank. No AutoBayes model could match brms's there.

**In the term model** (docs/term-model.md) the factor is one population-level term of kind `contrast` per contrast
(`fo.L`, `fo.Q`), each a scalar coefficient with one design column, handed to Bambi in backticks like a transform; the
fitted frame and new rows carry the contrast columns, computed from the labels (a level the factor was not fitted with is
refused). Priors are keyed by the factor's term as written, one prior or a mapping by R's labels:
`b={"x": ..., "fo": ab.Normal(0, 1)}` or `b={"fo": {".L": ..., ".Q": ...}}`. `default_priors()`, `prior_table()`,
`hypothesis()` and `describe()` use brms's names (`fo.L`, `orderedz.L`); `describe()` lists the levels in order and each
level's contrast values.

**Where it is offered.** In the population-level design of every model kind: the univariate families (the design is
the formula's, whatever the likelihood), a distributional formula (`sigma ~ fo`), the multivariate, non-linear (a parameter's
formula) and count-matrix models; `tests/test_syntax_fixes.py` checks the design of the Gaussian, Bernoulli, cumulative,
categorical, mixture, multivariate, non-linear and multinomial models and of a formula of sigma. Refused, naming the construct: an interaction with an ordered factor (brms gives each contrast
its own interaction column, which no frozen case verifies; D168), an ordered factor as a varying slope (as every categorical
slope), a column read by `ordered()` and by another term, and `fixef()` and the other summaries of a model with one (D162).

**Evidence.** The oracle case `gauss_ordered_factor` (`y ~ x + ordered(z) + ordered(h) + (1 | g)`, three and four levels,
per-contrast priors, a varying intercept) passes G1-G6 against brms 2.23; `gen_fixture.R` reads every factor unordered, so the
case writes the ordered factors as `ordered()` in the formula on both sides. The corpus entry `pop_ordered_factor` (an
ordered pandas Categorical) matches brms's `standata()`. brms 2.23's `posterior_linpred(newdata = )` of a formula with
`ordered(h)` fails ("non-conformable arguments") when the new rows lack one of the levels, since `ordered()` of the new rows
then has fewer contrasts; the case's new rows hold every level, and AutoBayes predicts on any fitted levels.
