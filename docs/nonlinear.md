# Non-linear formulas: `nl=True`

brms-parity feature C02 (brms's `bf(y ~ a * exp(b * x), a ~ 1 + (1 | g), b ~ 1, nl = TRUE)`).

```python
import autobayes as ab
priors = ab.Priors(nlpars={"a": {"b": ab.Normal(1, 1), "sd": ab.HalfNormal(1)}, "b": {"b": ab.Normal(0, 1)}},
                   sigma=ab.HalfNormal(1))
model = ab.gaussian(["ypos ~ a * exp(b * x)", "a ~ 1 + (1 | g)", "b ~ 1"], data, priors, nl=True)
print(model.describe())
```

## The model

The first formula's right-hand side is an expression in the non-linear parameters and the data; every other formula
gives one non-linear parameter its own linear predictor:

    mu_i = a_i * exp(b_i * x_i)                 (the expression, row by row: brms's mu)
    a_i  = X_a,i b_a + u_a_g[g_i]               (a's formula: brms's nlp_a)
    b_i  = X_b,i b_b
    y_i  ~ Normal(mu_i, sigma)

- **Names.** A name of the expression is a non-linear parameter when it has a formula in the list, and otherwise a numeric
  data column, a covariate (brms's `C_1, C_2, ...`, numbered by first appearance). Every formula after the first must name a
  parameter the expression uses.
- **The grammar** is R's arithmetic on a subset: `+ - * / ^` (R's `**` is `^`), parentheses, unary minus, numeric constants
  and the functions `exp`, `log`, `sqrt`, `inv_logit` (R's `plogis` is read as `inv_logit`) and `pow(x, y)`, with R's
  precedence: `^` binds tightest and to the right (`-a^2` is `-(a^2)`, `a^-b` is `a^(-b)`, `2^3^2` is 512) and unary minus
  binds tighter than `*` and `/` (`-b * x` is `(-b) * x`). Every constant is a double, as in R (`1 / 2` is 0.5). The
  expression is tokenized and parsed by the library's own recursive-descent parser (`autobayes.nonlinear.parse_expression`)
  into a tree that is turned into PyTensor for the graph and NumPy for `predict()`; nothing the user writes is evaluated as
  code, and anything outside the grammar is refused with the character or function and its position.
- **Each parameter's formula** is a linear formula of the term model (docs/term-model.md) with the parameter's name on its
  left: numeric and declared categorical predictors, two-way interactions, in-formula transforms, `0 +` to remove the
  intercept, and the group-level terms the term model takes (`(1 | g)`, `(1 + x || g)`, `(1 + x | g)`, `g:h`, `g/h`). brms never
  centres a non-linear parameter's formula: the design `X_<nlpar>` keeps its intercept column, and the intercept is the
  coefficient `Intercept` of brms's class `b` (`center_predictors` has no effect, as brms's `center` has none here). Each
  group-level term is a group of its own, also when two parameters' formulas group by the same factor (brms gives them two
  group ids); `|ID|` across parameters is refused, as everywhere.
- `categorical={"z": "p"}` is shared: a declared column is categorical in every parameter's formula that names it as a main
  effect, and a covariate of the expression cannot be declared categorical.

## The graph and its names

Bambi has no non-linear formulas, so `NonlinearModel` builds the PyMC graph itself (`model.bambi_model` is None):

| role | PyMC name | brms |
| --- | --- | --- |
| a parameter's coefficient, one variable per term | `a_Intercept`, `a_x`, `a_z` (a vector over `a_z_dim`) | `b_a[k]` |
| a group-level coefficient's SD, standardised effects and effects | `a_1\|g_sigma`, `a_1\|g_offset`, `a_1\|g`; a slope's `a_x\|g...` | `sd_m`, `z_m`, `r_m_a_k` |
| a correlated term's Cholesky factor and correlation matrix | `a_L\|g` (sampled as `a_L\|g_cholesky_corr__`, Stan's coordinates), `a_cor\|g` | `L_m`, `Cor_m` |
| a parameter's values per row | `a` (deterministic) | `nlp_a` |
| the non-linear predictor | `mu` (deterministic, `model.names["linpred"]`) | `mu` |
| the residual SD | `sigma` | `sigma` |

`model.nlpar_names`, `model.nlpar_coefficients` (where `b_a[k]` lives), `model.nlpar_group_names` and
`model.nlpar_correlation_names` list them; `model.spec` is a `NonlinearSpec` (`nlpars`, `covariates`, `expression`, `rhs`) and
`model.design` a `NonlinearDesign` (`covariates`, and per parameter `nlpars[name].X`, `.columns`, `.group_blocks`). The family's
likelihood on `mu` is `FamilySpec.nonlinear_likelihood`.

## Using it

- **Priors** are per parameter, `Priors(nlpars={"a": {"b": ..., "sd": ...}, ...})` (brms's `class = b` and `class = sd` with
  `nlpar = "a"`): `"b"` is one location prior for every coefficient, the intercept included, or a mapping by term with an
  `"Intercept"` entry exactly when the formula has an intercept, `{"Intercept": ab.Normal(1, 0.5), "z": ab.Normal(0, 1)}` (brms's
  `coef = Intercept`); `"sd"` is read as a linear formula's `sd` and is required exactly when the formula has a group-level
  term. A correlated term's LKJ prior is `Priors(cor={"g": ab.LKJ(2)})`, keyed by factor and shared by every correlated term on
  that factor, because brms's class `cor` has no `nlpar`. The model's `intercept`, `b` and `sd` are refused (its coefficients
  are its parameters'), `sigma` takes its one prior, and every prior is required (D1).
- `ab.default_priors(formulas, data, nl=True)` returns brms's defaults: flat coefficients, `student_t(3, 0, scale)` on every
  group SD and on `sigma` with brms's scale of the response (`max(2.5, round(mad(y), 1))`), and `lkj(1)`.
- `model.value_point(nlpars={"a": {"b": [...], "sd": ..., "z": ..., "L": ...}, ...}, sigma=...)`: `b` in the order of
  `model.design.nlpars["a"].columns`, `sd` / `z` as `Model.value_point` takes them for the formula's group-level terms.
- `predict()`: `kind="linear"` is the non-linear predictor on the link scale (brms's `posterior_linpred`, as `eta`),
  `kind="mean"` and `kind="response"` behave as for the family (brms's `posterior_epred` and `posterior_predict`), and
  `nlpar="a"` returns the parameter's values (`posterior_linpred(nlpar = "a")` under `kind="linear"` as `eta`,
  `posterior_epred(nlpar = "a")` under `kind="mean"` as `a`; the same numbers, a parameter has no link). `newdata` needs the
  covariates and every parameter's columns; `include_group_effects=False` drops every parameter's group effects; `new_groups`
  applies by grouping factor across the parameters, and a factor in two parameters' formulas draws each term's effects for an
  unseen level independently (brms's ids are separate too). A non-finite non-linear predictor (`log` of a negative value) is
  refused rather than returned.
- `describe()` writes out the expression, each parameter's linear predictor, group effects and correlations, the priors under
  brms's coefficient names (`b_a[zq]`), the likelihood and the sampled variables; `prior_table()` writes brms's table with
  the `nlpar` column; the target identity covers every formula, the covariates, each parameter's design and every prior;
  `fit()` samples the graph with PyMC's `sample` (`sample_posterior`, as Bambi's `fit()` samples a model it built, dropping
  the per-row deterministics and the `_offset` variables); `fit_record()` records the formulas, covariates, parameters,
  names and priors; `prior_predictive()` draws the graph (refused under a flat prior).
- **Post-processing** (C07, docs/post-processing.md): `ab.log_lik(model, fit)` is each fitted row's `Normal(y_i | mu_i,
  sigma)` log density per draw, `ab.bayes_R2(model, fit)` the per-draw `var(mu) / (var(mu) + var(y - mu))` (brms's
  `posterior_epred` of a Gaussian non-linear model is `mu`) and `ab.loo(model, fit)` PSIS-LOO of that log-likelihood, through
  the same code as for a linear model. They are checked against the graph's own elementwise log density and `mu` at the
  frozen brms points; no frozen non-linear case holds brms's `log_lik()` / `loo()` tables (the two predate C07's `post`
  block).

## Refused, naming the construct

- brms's `nlf()` and `lf()`, and a non-linear parameter inside another parameter's formula (D60).
- Functions and operators outside the grammar, R's `2L`/`0x10`/`1i` literals and unary plus; the response as a covariate of
  the expression or as a predictor of a parameter; `gp()` and `mo()` written in the expression (brms passes them into its
  Stan program as calls of functions Stan does not have) and in a parameter's formula (brms takes them there: a Gaussian
  process `gp_pred_a_1` or a monotonic term added to `nlp_a`), `offset()` in a parameter's formula and response addition
  terms (`weights()`, `cens()`, `trunc()`) with a non-linear formula (D61). The priors of a Gaussian process (`sdgp`,
  `lscale`) are refused with them. An autocorrelation term `ar()`, `ma()`, `arma()` (C04, docs/autocorrelation.md): in the
  expression it is brms's term, not a function, and brms's way of adding one to a non-linear model, `bf(..., autocor = ~
  ar(...))`, is not offered (D70); in a parameter's formula brms refuses it too; its priors `ar` and `ma` are refused.
  Missing values and measurement error (C06, docs/missing-and-measurement-error.md): `me()` and `mi()` in the expression
  (brms's terms, not functions) and in a parameter's formula, `mi()` on the response's or a parameter's left-hand side, and
  the priors `meanme` and `sdme` (D72, D73).
- A distributional formula with `nl=True` (`sigma ~ x` next to the parameters' formulas); a non-linear model in a
  multivariate formula, `mvbind(y, y2) ~ a * exp(b * x)` (brms gives each response its own parameters, `X_y_a`, `X_y2_a`) or
  a list of response formulas, and `rescor=` with `nl=True`; a non-linear model in a mixture, `ab.mixture(..., nl=True)` and
  `ab.default_priors(..., families=..., nl=True)` (brms puts the expression in every component's mean with shared
  parameters); families other than the Gaussian (D62). A non-linear model's priors refuse the multivariate model's `resp`
  and `rescor` and the mixture's `components` and `theta`; every other model's priors refuse `nlpars` (a multivariate
  model's at the top level and in each response's, a mixture's at the top level and in each component's).
- `loop=False` (D63).
- A covariate that is not a numeric column, or has missing or non-finite values; a formula of a parameter the expression does
  not use; two formulas for one parameter; a name of the expression that is neither a parameter nor a column.
- A parameter's formula with neither an intercept nor a population-level term, `a ~ 0 + (1 | g)`: the term model refuses
  such a formula everywhere ("no intercept requires at least one population-level predictor"), while brms accepts it (an
  empty `X_a` beside `Z_1_a_1`); `a ~ 1 + (1 | g)` is the supported form.

## Evidence

Two frozen cases pass gates G1-G6 against brms 2.23.0, at machine precision: `gauss_nonlinear_exp` (the case the loop
froze: `a * exp(b * x)` with a varying intercept on `a`; log density within 3.8e-16 relative, gradient 5.8e-16, predictions
exact) and `gauss_nonlinear_terms`, frozen with this feature to verify what the first leaves open (`a * exp(-b * x) +
inv_logit(c) * sqrt(xpos) / (1 + pow(w, 2)) - 0.25 * d * log(xpos)^2`: three covariates, one used twice, every function but
`plogis` and every operator; a categorical term and `0 + z` in parameters' formulas; `(1 | g)` in two parameters, two brms
group ids on one factor; a correlated slope `(1 + w | h)` under `lkj(2)`; a column that is both a covariate and a predictor; a
per-coefficient prior; log density within 4.6e-16, gradient 3.8e-14, predictions 5.5e-16). G1 matches the covariates `C_k`,
each `X_<nlpar>` and the group designs by content; G2 compares every `b_<nlpar>` coordinate; G3 the graph's `mu`, each
parameter's values against brms's own `X_<nlpar> b_<nlpar>` plus its group effects, the group effects and `Cor`; G4
`posterior_linpred` / `posterior_epred` on fitted and new rows with and without group effects; G6 brms's default-prior table
with its `nlpar` rows. `tests/test_nonlinear.py` covers the grammar (R's precedence, round trips, every refusal, and that the
module calls no evaluating builtin), the term model and its refusals, the priors, a density written by hand on a model with
every construct, `value_point`, predictions written by hand (fitted and new rows, `nlpar=`, both unseen-level policies),
`describe()`, the identity, the fit plumbing with a stubbed sampler and the fit record, prior-predictive draws, seven
`default_prior()` tables printed by brms for a small frame, that the gates fail on a corrupted non-linear fixture, every
refusal of a combination with the other features (multivariate formulas, mixtures, Gaussian processes and their priors,
`nlpars` in theirs), and `log_lik`, `bayes_R2` and `loo` against the graph's own density and `mu` on both frozen cases
(within 2.1e-15 relative). Nothing samples.
