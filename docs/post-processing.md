# Post-processing: `log_lik`, `bayes_R2`, `loo`, `waic`, `loo_compare`, `model_weights`, `hypothesis`, `conditional_effects`, `prior_draws`

brms-parity feature C07 (brms's `log_lik()`, `bayes_R2()` and `loo()`, loo 2.9.0), D05 (brms's `waic()`,
`loo_compare()` and `model_weights()`), D06 (brms's `hypothesis()` and `conditional_effects()`) and E05 (brms's
`sample_prior = "yes"` and `prior_draws()`, the Savage-Dickey evidence ratios of point hypotheses, interaction effects in
`conditional_effects()`, and the stacking weights, implemented and withheld).

```python
import autobayes as ab
fit = ab.fit(model, chains=4, draws=1000, seed=1)
ll = ab.log_lik(model, fit)          # (chain, draw, __obs__): each fitted row's log density at each draw
r2 = ab.bayes_R2(model, fit)         # (chain, draw): Bayesian R-squared per draw
res = ab.loo(model, fit)             # PSIS-LOO: res.pointwise, res.estimates, res.pareto_k, res.n_eff
print(res)
w = ab.waic(model, fit)              # WAIC: w.pointwise, w.estimates
ab.loo_compare({"m1": ab.loo(m1, fit1), "m2": ab.loo(m2, fit2)})          # loo's table, best model first
ab.model_weights({"m1": ab.loo(m1, fit1), "m2": ab.loo(m2, fit2)}, "loo")  # brms's Akaike weights
h = ab.hypothesis(model, fit, ["x > 0", "x + w = 0", "zr < zq"])          # D06: h.table, h.samples
ce = ab.conditional_effects(model, fit, ["x", "z"])                       # D06: {"x": DataFrame, "z": DataFrame}
pd = ab.prior_draws(model, draws=4000, seed=1)                            # E05: brms's prior_draws() columns
h = ab.hypothesis(model, fit, ["x = 0", "w = 0.5"], prior=pd)             # E05: Savage-Dickey Evid.Ratio, Post.Prob
ce = ab.conditional_effects(model, fit, ["x:z", "x:w"])                   # E05: interaction effects, effect2__
```

Each takes the model and its fit (a `FitResult` or its `InferenceData`, checked against the model as `predict()` checks
it) and works on the fitted rows with every posterior draw. No function here samples.

## `log_lik(model, fit)`

brms's `log_lik()`: the log density of each fitted row's response at each draw, an `xarray.DataArray` named `log_lik`
with dims `(chain, draw, __obs__)`. It is the per-row term of the model's own likelihood: the family's
`FamilySpec.log_likelihood` (NumPy, `autobayes/families/pointwise.py`) at `predict(kind="linear")`'s linear predictor
(group effects, offsets and monotonic terms included) and the draw's auxiliary parameters (each row's own `sigma`,
`shape` or `phi` under a distributional formula), in the stable forms the likelihoods use (the softplus of the logit
link, `log Phi` of the probit, the zero-inflated and hurdle forms of D28, the ordinal `log_inv_logit_diff` form).
Under `weights(w)` each row's log density is multiplied by its weight, as brms multiplies it. Offered for every family:
Gaussian, Student-t (fixed or learned `nu`), Poisson, negative binomial, Gamma, lognormal, Beta, Bernoulli and binomial
(logit and probit), zero-inflated and hurdle Poisson, cumulative, sratio, cratio and acat (the log probability of the
1-based category; D04's with `cs()` terms too) and categorical. A non-linear model (C02, `nl=True`, docs/nonlinear.md) is taken the same way, its linear predictor the
non-linear predictor `mu` (brms's `posterior_linpred`); `tests/test_nonlinear.py` checks `log_lik`, `bayes_R2` and `loo` on
it against the graph's own density and `mu` at the frozen brms points. A model with an autocorrelation term (C04,
docs/autocorrelation.md) is taken the same way: its linear predictor carries the ARMA term from the observed residuals, the
mean brms's `log_lik()` reads with `cov = FALSE` (all rows at once, then in data order); `tests/test_autocorrelation.py`
checks `log_lik`, `bayes_R2` and `loo` against the graph's own density and `mu` at the frozen brms points of its three cases.
A finite mixture (C05, docs/mixtures.md) is taken too: each row's `log_mix(theta, normal_lpdf(y_i | mu_k,i, sigma_k), ...)`
from `predict(kind="linear")`'s components (brms's `log_lik_mixture`), and `bayes_R2` of the mixture's mean `sum_k theta_k
mu_k,i`, brms's `posterior_epred`; verified against the frozen case `mixture_gauss3_post` and, at the points of the three
mixture cases, against the graph's own density. A multivariate model (C01) is refused by `log_lik`, `bayes_R2` and `loo`
naming the construct, as `waic` refuses it (D95; until D06 the first three failed with an `AttributeError`), and so is a
model with `me()` or `mi()` (D74).

## `bayes_R2(model, fit)`

brms's `bayes_R2(summary = FALSE)` (Gelman, Goodrich, Gabry and Vehtari 2019, "R-squared for Bayesian regression
models"): per draw, `var(yhat) / (var(yhat) + var(y - yhat))` over the fitted rows, `yhat` the expected response
(`predict(kind="mean")`, brms's `posterior_epred`) and `y - yhat` the residuals, for every family (the Gaussian's
included: brms uses the residuals, not `sigma`). A binomial model's `yhat` is the expected count `n p` against the
successes; weights play no part. For the ordinal families brms's `yhat` is the expected category `sum_k k P(y = k)`
against the 1-based category (settled by probing brms for the cumulative family; D04's `sratio_cs_post`,
`cratio_cs_post` and `acat_cs_post` record it for the others); brms refuses the categorical family ("'bayes_R2' is not defined
for unordered categorical models") and so does `ab.bayes_R2`. Returns an `xarray.DataArray` named `R2`, dims
`(chain, draw)`; summarise it as you like (brms's `summary = TRUE` is the mean, SD and quantiles).

## `loo(model, fit, r_eff=None)`

brms's `loo()`: approximate leave-one-out cross-validation by Pareto-smoothed importance sampling of `log_lik()`. The
result, an `ab.Loo`, holds brms's tables:

- `pointwise`: one row per fitted row, brms's columns `elpd_loo`, `mcse_elpd_loo`, `p_loo`, `looic`,
  `influence_pareto_k`;
- `estimates`: rows `elpd_loo`, `p_loo`, `looic`, columns `Estimate` and `SE`; also as `res.elpd_loo`,
  `res.se_elpd_loo`, `res.p_loo`, `res.looic`;
- `pareto_k` and `n_eff`, loo's `diagnostics`; `r_eff`, the relative efficiencies used; `n_draws`;
  `pareto_k_threshold`, loo's `min(1 - 1 / log10(S), 0.7)`, above which an estimate is not reliable.

`r_eff=None` is brms's default: the relative efficiency of each observation's likelihood `exp(log_lik)` across the
chains (brms hands loo `relative_eff(exp(log_lik), chain_id)`; probed with a multi-chain fixed-parameter fit); a number,
or one per observation, is used as given (`r_eff=1` is loo's own default for a log-likelihood matrix, NaN entries
become 1 as loo makes them). `r_eff` sets the PSIS effective sample sizes and the MCSE, and the tail length through
`3 sqrt(S / r_eff)`; it does not change the expected value of `elpd_loo`.

`ab.post_processing.loo_from_log_lik(log_lik, r_eff)` runs the same computation on any `(chain, draw, __obs__)`
log-likelihood; `psis`, `gpd_fit` and `relative_eff` are the steps.

### What is computed, and where each part comes from

loo is GPL-3 and AutoBayes MIT: no loo source is used. The steps come from ArviZ (Apache-2.0) and the published
algorithm, and loo's documented behaviour and its outputs on probe inputs (run in R) settle what neither fixes.

| part | definition | source |
| --- | --- | --- |
| tail length | `M = ceil(min(S / 5, 3 sqrt(S / r_eff)))` of the largest ratios `r_s = 1 / p(y_i \| theta_s)` | Vehtari, Simpson, Gelman, Yao and Gabry (2024), "Pareto smoothed importance sampling", JMLR 25(72); as ArviZ's `psislw` |
| tail fit | generalized Pareto by Zhang and Stephens (2009, Technometrics 51(3)) with `30 + floor(sqrt(M))` grid points and the PSIS paper's prior `k = (M k + 10 * 0.5) / (M + 10)` | the papers; as ArviZ's `_gpdfit` |
| smoothing | the tail replaced by the fit's quantiles at `(z - 1/2) / M`, `z = 1..M`, none above the largest raw ratio | the PSIS paper; as ArviZ |
| no fit | fewer than five draws in the tail: raw ratios, `k = Inf` | ArviZ (`tail_len <= 4`), and loo's outputs (8 draws: `k = Inf`; 24 draws, a 5-draw tail: fitted) |
| ties and degenerate tails | the tail is the `M` largest draws in order, ties included, cut at the `(M + 1)`-th largest; a tail whose lower quartile ties with the cutoff (the fit does not exist) leaves raw ratios and `k = Inf`; no floor at the smallest double | settled by probing loo: ArviZ keeps only draws strictly above the cutoff and floors it at `log(tiny)`, which differs only in these cases |
| `elpd_loo`, `p_loo`, `looic` | `log sum_s w_s p_s` (normalised smoothed weights), `lpd_i - elpd_loo_i` with `lpd_i = log mean_s p_s`, `-2 elpd_loo_i` | Vehtari, Gelman and Gabry (2017), "Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC", Statistics and Computing 27 |
| estimates | sums; `SE = sqrt(N var)` of the pointwise values (their eq. 23); `2 SE(elpd_loo)` for looic | the same paper |
| `n_eff` | `r_eff / sum_s w_s^2` | the PSIS paper; the `r_eff` factor settled by probing |
| `mcse_elpd_loo` | `sqrt(log(1 + V / E^2))`, `V = sum_s w_s^2 (p_s - E)^2 / r_eff` the self-normalised variance of `E = exp(elpd_loo_i)` carried to the log scale by a log-normal approximation | loo 2.7.0's NEWS ("simplify mcse_elpd using log-normal approximation"); the formula and the `r_eff` divisor settled by probing |
| `r_eff` | ArviZ's `ess(method="mean")` of `exp(log_lik)` over `S`; 1 for chains of fewer than four draws or a constant likelihood (loo's NA, replaced by 1) | ArviZ; brms's use of it and the NA rule settled by probing |

## `waic(model, fit)` (D05)

brms's `waic()`, the widely applicable information criterion (Watanabe 2010, JMLR 11; the WAIC section of Vehtari,
Gelman and Gabry 2017) of `log_lik()`: `lpd_i = log mean_s p(y_i | theta_s)` (a log-sum-exp, finite at
log-likelihoods near -800, as loo's is), `p_waic_i` the sample variance (divisor `S - 1`) of `log p(y_i | theta_s)`
over the draws, `elpd_waic_i = lpd_i - p_waic_i` and `waic_i = -2 elpd_waic_i`; the estimates are the sums and `SE = sqrt(N var_i)`, NaN for a single
observation (loo's NA). The result, an `ab.Waic`, holds brms's tables: `pointwise` (columns `elpd_waic`, `p_waic`,
`waic`) and `estimates` (rows `elpd_waic`, `p_waic`, `waic`; columns `Estimate`, `SE`), also as `res.elpd_waic`,
`res.se_elpd_waic`, `res.p_waic`, `res.waic`; printing it counts the observations with `p_waic` above 0.4, where loo
warns and recommends PSIS-LOO. `ab.post_processing.waic_from_log_lik(log_lik)` takes any `(chain, draw, __obs__)`
log-likelihood. Fewer than two draws and a non-finite log-likelihood are refused (loo stops on the first and returns NaN
rows for the second), as are a multivariate model (D95) and a model with `me()` or `mi()` (D74).

## `loo_compare(results)` (D05)

brms's `loo_compare()` (loo 2.9.0's) of several models' `ab.loo()` or `ab.waic()` results, given as a mapping
`{name: result}` or a list (named `model1`, `model2`, ... as loo names an unnamed list). Returns loo's table as a
`pandas.DataFrame`, one row per model indexed by name, best first (the largest `elpd`; exact ties keep the order given,
as loo keeps it), with loo's eight columns: `elpd_diff` and `se_diff`, then the model's own estimate and SE of `elpd`,
`p` and the information criterion (`elpd_loo`, `se_elpd_loo`, `p_loo`, `se_p_loo`, `looic`, `se_looic`, or the `waic`
names). For model `k` and the best model `b`, `elpd_diff = sum_i (elpd_k,i - elpd_b,i)` and `se_diff =
sqrt(N var_i(elpd_k,i - elpd_b,i))` (Vehtari, Gelman and Gabry 2017, on model comparison), 0 and 0 for the best model.
brms's `loo_compare(fit1, fit2)` needs each criterion stored first (`add_criterion`); here each result is computed first
(`ab.loo(model, fit)`), which also fixes `r_eff` (D94).

Refused, naming the reason: a single model (loo: at least two), models on different numbers of observations (loo's
error), a single observation (loo stops there too), anything that is not an `ab.loo()` or `ab.waic()` result, and PSIS-LOO
and WAIC results together, which loo 2.9.0 compares silently (D93). Models fit to different responses are compared with a
warning, as brms's `loo_compare` warns when the fits' `yhash` attributes differ (probed): `ab.loo()` and `ab.waic()`
record a fingerprint of the response (its fitted values and, for a binomial model, the trials; `Loo.response`,
`Waic.response`), and results of a bare log-likelihood carry none.

## `model_weights(results, weights, BB=None)` (D05)

brms's `model_weights()` of the same results, a `pandas.Series` in the order given:

- `weights="loo"` (`ab.loo()` results) and `weights="waic"` (`ab.waic()` results): brms's Akaike weights of the
  information criterion, `exp(-dIC_k / 2) / sum_j exp(-dIC_j / 2)` with `dIC` the difference from the smallest, which are
  `exp(elpd_k) / sum_j exp(elpd_j)`;
- `weights="pseudobma", BB=False` (`ab.loo()` results): the pseudo-BMA weights of Yao, Vehtari, Simpson and Gelman (2018,
  "Using stacking to average Bayesian predictive distributions", Bayesian Analysis 13(3)) without the Bayesian bootstrap,
  the same formula on `elpd_loo` (brms hands it to loo's `loo_model_weights`).

`weights` has no default: `"bma"` and `"kfold"` are not offered, and neither is pseudo-BMA with the Bayesian bootstrap (brms's
default `BB = TRUE`), which draws random numbers; `BB` must be `False` for `"pseudobma"` and is refused for the others (D92).
brms's default `"stacking"` is implemented and withheld (E05, D126; below). The other checks are those of `loo_compare`.

### Stacking (E05, withheld)

brms's `model_weights(weights = "stacking")` is loo's `loo_model_weights()` with its default method: the stacking weights of
Yao, Vehtari, Simpson and Gelman (2018, eq. 4), the weights on the simplex maximising the leave-one-out log score of the
combined predictive distribution, `F(w) = sum_i log sum_k w_k exp(elpd_loo_k,i)`. `ab.post_processing.stacking_weights(lpd)`
computes the exact optimum of `F`, which is concave: with `a_ik = exp(lpd_ik - max_k lpd_ik)` and `g = a' (1 / (a w))`, the
optimum is where `g_k = N` for every model with a positive weight and `g_k <= N` for the others (the Karush-Kuhn-Tucker
conditions, `sum_k w_k g_k = N` always), reached by Newton steps on the face of the simplex spanned by the models with positive
weight, a model leaving the face when its weight reaches 0 and joining it when its `g_k` exceeds `N`, to a relative residual
of 1e-14. loo finds the optimum with `stats::constrOptim` (a log-barrier method around BFGS), which stops at its tolerance and
never gives an exact zero: on loo's own probes (`tests/loo_probes/probe_stacking.R`) its weights are off the exact optimum by
up to 3.7e-4, its objective never above the optimum's.

On the frozen case `gauss_loo_compare_x_stacking` brms's weights are 0.5619733196999603 and 0.43802668030003966; the optimum,
solved in 50-digit arithmetic, is 0.56197330926078805, and `stacking_weights` returns the nearest doubles. brms's are 1.044e-8
off it against the oracle's `PRED_REL` of 1e-9. The fixture is not edited and the tolerance not loosened: `"stacking"` (and
`ab.post_processing.loo_model_weights`) are withheld (`PARKED_WEIGHTS`, refused with the reason), the case stays gated with
these numbers, and with the park lifted every other gate passes (`tests/oracle/test_prior_effects_stacking_harness.py`). How
the weights are certified (a two-reference acceptance with the exact optimum as the reference, or otherwise) is Andrew's
decision (D126). Un-parking is removing the `PARKED_WEIGHTS` entry, exporting `loo_model_weights` as `ab.loo_model_weights`
(the name the case proposes) and removing the case's gate field.

## `hypothesis(model, fit, hypotheses, alpha=0.05, robust=False)` (D06)

brms's `hypothesis(fit, hypotheses)` with its default `class = "b"`: tests on the population-level coefficients, named as
brms names them (`x`, `zq` for level `q` of `z`, `x:zq`, `zq:hh2`, `logxpos`, `IxE2`, and `Intercept`, the raw-scale
intercept, brms's `b_Intercept`). A hypothesis is `left (= OR < OR >) right`, a point hypothesis (`=`) or a one-sided one
(`<`, `>`); its draws are those of `left - right` (of `left` when `right` is the text `0`). Returns an `ab.Hypothesis`:

- `table`, brms's columns: `Hypothesis` brms's text (whitespace removed; `(left) op 0` when `right` is `0`, else
  `(left)-(right) op 0`: `"zr < zq"` is `(zr)-(zq) < 0`); `Estimate` the mean and `Est.Error` the SD of the draws (with
  `robust=True` the median and the MAD, `1.4826 median |v - median v|`); `CI.Lower` and `CI.Upper` the `alpha / 2` and
  `1 - alpha / 2` quantiles of a point hypothesis's draws and the `alpha` and `1 - alpha` quantiles of a one-sided one
  (R's type 7, a 90 % interval by default); for a one-sided hypothesis `Post.Prob` the share of draws strictly on its
  side of zero and `Evid.Ratio` the draws on that side over the others (a draw at zero counts against it; `Inf` when every
  draw is on its side, 0 when none is); for a point hypothesis both are NaN without prior draws (D96) and brms's
  Savage-Dickey ratio with them (`prior=`, E05, below); `Star` `"*"` when `Post.Prob > 1 - alpha` (one-sided) or the
  interval excludes zero (point), else `""`;
- `samples`, the draws of each hypothesis, `(draws, hypotheses)` with brms's names `H1`, `H2`, ..., the chains one after
  another; `hypotheses`, `alpha`, `robust`; `fit_binding_status` as `predict()` reports it. Printing it prints the table.

The hypotheses are read by a parser of their own (`autobayes/hypothesis.py`) and never evaluated as code. It takes numbers
in decimal notation, names, parentheses, `+ - * / ^` with R's precedence (`^` above the unary signs and grouping to the
right: `-2^2` is -4, `2^-1` is 0.5, `2^3^2` is 512; the signs above `*` and `/`), and `exp`, `log`, `sqrt` and `abs` of one
argument. Refused, naming the problem: what brms refuses (not exactly one of `=`, `<`, `>`, so `>=`, `==` and `!=` too;
an empty side; no parameter; an unknown name, with the model's names; exponent notation, which brms reads as a name),
draws where a hypothesis is NaN (`log` of a negative value; brms stops there too), and what brms would evaluate in R but
the parser does not take (other functions, `%%`, `&`, backticks; D97). The names of the special terms (`moord`, brms's class
`bsp`) and of a distributional formula's coefficients (`sigma_z`) are refused naming why (D97); so are multivariate and
non-linear models, mixtures and the cumulative and categorical families (D99). `hypothesis_from_draws(draws, hypotheses)`
(`from autobayes.hypothesis import hypothesis_from_draws`) is brms's default method: hypotheses on draws given by name.

## Prior draws and Savage-Dickey ratios (E05)

`ab.prior_draws(model, draws=4000, seed=None)` is brms's `prior_draws()` of a fit run with `sample_prior = "yes"`: a
`pandas.DataFrame`, one row per draw, one column per prior brms samples, named and ordered as brms names them (settled on five
probed brms fits, `tests/hypothesis_probes/brms-2.23.0-prior-draws.json`): `Intercept` (the prior of the centred design's
intercept, brms's class `Intercept`); `b`, one draw per row shared by every coefficient when `b` is one prior, or one column per
coefficient (`b_x`, `b_zq`, `b_x:zq`) when `b` is a mapping by term (brms gives every coefficient its own column as soon as
one has a prior of its own); the family's parameters in brms's order (`sigma`, `shape` or `phi`, then `nu`, then `zi` or
`hu`); for each grouping factor in formula order `sd_g` (an intercept-only factor, or a slope group whose SD priors are all
equal: brms's group-wide prior) or `sd_g__Intercept`, `sd_g__x`, ..., then `cor_g` for a correlated term, one correlation.
The draws come from the model's own prior distributions (its PyMC variables: a Student-t `nu` truncated at 1, half-normal
SDs), independent across columns and rows; a correlation is drawn from the LKJ marginal of one off-diagonal entry, `2 B - 1`
with `B ~ Beta(eta - 1 + d / 2, eta - 1 + d / 2)` (Lewandowski, Kurowicka and Joe 2009, J. Multivariate Analysis 100). A
flat prior is improper and has no column, as brms does not sample it. They are random, as brms's (Stan's RNG) are: what is
compared with brms is computed from given draws, and their distributions. Not offered, each refused naming the construct
(D124): multivariate, non-linear, mixture, ordinal and categorical models, distributional formulas, shrinkage priors, `mo()`,
`gp()`, `me()`, `mi()`, `cs()`, autocorrelation terms, `mm()`, `gr(by = )` and shared IDs, the D01 families' `alpha`, `beta`
and `ndt`, and `center_predictors=False`.

`ab.hypothesis(model, fit, hypotheses, prior=..., seed=None)` takes prior draws named as brms names them (`ab.prior_draws`'s,
or brms's own `prior_draws()`) and gives each point hypothesis brms's Savage-Dickey evidence ratio, the posterior density of
`left - right` at 0 over its prior density there, and `Post.Prob = Evid.Ratio / (1 + Evid.Ratio)` (for `Evid.Ratio = Inf`
NaN, as R's `Inf / Inf`); a one-sided hypothesis is unchanged. The prior draws of `left - right` are the hypothesis evaluated
on each coefficient's prior draws: its own column `b_<name>` if the frame has one, else `b`. Of a hypothesis on several
parameters, each parameter's prior draws are permuted independently (brms pairs them through a random permutation, R's; here
NumPy's, seeded by `seed`, so the ratio is brms's in distribution only, D125); of one parameter, nothing is permuted (brms's
permutation moves only the ratio's last bit). A point hypothesis on `Intercept` (the raw-scale intercept) keeps NaN with a
warning, as brms gives NA and warns ("Sampling from the prior of an overall intercept is not possible by default"; probed),
and so does one on a coefficient without prior draws (a flat prior). Refused: prior draws that are not a frame or mapping of
equal-length finite columns of at least two draws, a hypothesis that is not finite at the prior draws (brms stops there too),
and a model with `center_predictors=False`.

The density at the point is brms's `density_ratio(x, y, point)` (`autobayes.kde.density_ratio`, `density_at`), written from R's
documentation of `density()`, `bw.nrd0()` and `spline()`, the published methods, and brms 2.23's and R 4.5's outputs on probe
inputs (`tests/hypothesis_probes/probe_density_ratio.R`); none of their source is used:

| part | definition | source |
| --- | --- | --- |
| bandwidth | `0.9 min(sd, IQR / 1.34) S^(-1/5)`, the SD with divisor `S - 1`, the quartiles R's type 7 formed as `(1 - h) x_j + h x_(j+1)`; a zero minimum replaced by the SD, else `\|x_1\|`, else 1 | `?bw.nrd0` (Silverman 1986, eq. 3.31); the quantile form and the fallbacks settled by probing |
| estimate | Gaussian kernel; the draws binned linearly on 4096 cells from 4 bandwidths below the range to 4 above, convolved by FFT (on twice as many cells, no wrap-around) with the kernel at the cells' spacing, clipped at 0, interpolated linearly onto 4096 points spanning the range | `?density` (Silverman 1982, Algorithm AS 176; Jones and Lotwick 1984, AS R50); the extension, the kernel's spacing (R 4.4's grid) and the interpolation settled by probing |
| range | the draws' smallest to largest value; below the draws `point - sd / 4` to the largest, above them the smallest to `point + sd / 4` | brms's outputs (the range `[point, max]` misses by 1e-6) |
| at the point | the cubic spline through the 4096 values with Forsythe, Malcolm and Moler end conditions (the third derivative at each end that of the cubic through the four end points); every draw at the point: the estimate's value | `?spline` (method `"fmm"`; Forsythe, Malcolm and Moler 1977); not-a-knot ends and linear interpolation miss by 1e-10 and 1e-8 |

It equals brms's `density_ratio()` on 276 probe points (2 to 4000 draws, ties, constant draws, skewed and heavy-tailed draws,
scales from 1e-5 to 1e6; points inside the draws, at and next to their ends, just and far outside) to 1e-13 relative plus 1e-15
of the estimate's peak, the FFT's round-off in both. Far in the draws' tails, where the estimate falls to that round-off
(below about 1e-12 of its peak), brms's value and this one are both round-off and differ; `density_ratio` warns below 1e-10.
Fewer than two draws and non-finite draws are refused (brms stops on both).

## `conditional_effects(model, fit, effects, resolution=100, prob=0.95)` (D06)

brms's `conditional_effects(fit, effects)` with its defaults: the expected response along one predictor with the others
held fixed. `effects` names predictors of the population-level formula (one name, or a list); for each, a
`pandas.DataFrame` with brms's columns, in a dict keyed by effect in the order given:

- the effect's grid: `resolution` evenly spaced values from the fitted rows' minimum to their maximum of a numeric
  predictor (R's `seq(min, max, length.out = 100)`), the levels of a categorical one, reference first;
- the response and every other variable of the model as held on the grid: numeric predictors at their mean over the
  fitted rows, categorical ones at their reference level (brms's first level), the response at its mean (brms writes it
  into the grid, where it changes nothing), the grouping factors missing (`None`): brms's default `re_formula = NA` leaves
  the group-level effects out; the columns are the effect, the response, the population-level variables in formula order
  and the grouping factors;
- `cond__` (`"1"`), `effect1__` (the grid), and the summaries over the draws of brms's `posterior_epred` at each grid row
  (`predict(kind="mean", include_group_effects=False)`, which gate G4 checks against `posterior_epred(re_formula = NA)`) as
  brms's default `robust = TRUE` summarises them: `estimate__` the median, `se__` the MAD (`1.4826 median |e - median e|`),
  `lower__` and `upper__` the `(1 - prob) / 2` and `1 - (1 - prob) / 2` quantiles (R's type 7).

An interaction effect `"a:b"` (E05) is two distinct population-level predictors with brms's default `int_conditions`: `a` on
its grid as above and `b` at each of its levels (a categorical moderator, reference first) or at its mean minus its SD, its
mean and its mean plus its SD (a numeric one, the SD with divisor `n - 1`); one row per pair with `b` varying fastest (`100 x
3` rows for a numeric `a`). The columns are `a`, `b`, the response and the other variables as held, `cond__`, `effect1__`
(`a`), `effect2__` (`b`'s level, or its value as brms labels it, R's `as.character(round(value, 2))`: `"1.6"`, `"0"` for
-0.004, `"1e+05"`, R's rounding of a tie such as `round(0.005, 2) = 0`; the `b` column keeps the exact value) and the
summaries. Any pair of the model's predictors is taken, numeric or categorical, in either order, whether or not the formula
has their interaction (brms takes them too; probed in `tests/hypothesis_probes/brms-2.23.0-effects-grid.json`). Interactions
of three predictors are refused, and so are effects naming the same pair twice, which brms drops with a warning (D127).

Taken: the Gaussian, Student-t, Bernoulli, Poisson, negative binomial, Gamma, lognormal, Beta, zero-inflated and hurdle
Poisson families, with categorical predictors, interactions, transforms (the grid is on the raw column: `xpos` for
`log(xpos)`) and any group-level terms. Refused, naming the construct (D99): the binomial family (its grid needs trials),
the cumulative and categorical families, mixtures, multivariate and non-linear models, and models with an offset, `mo()`,
`gp()`, `me()`, `mi()`, an autocorrelation term, addition terms or a distributional formula: no frozen case records how brms
holds them on the grid. Grouping factors as effects, brms's default of every effect, and its other arguments
(`int_conditions` among them) are not offered (D98).

## Verification

- The oracle gate `check_post` (`tests/oracle/gates.py`, test `test_c07_post_processing_equals_log_lik_bayes_R2_and_loo`)
  compares the three functions with the `post` block of every case that lists them, to `PRED_REL` (1e-9 relative), every
  non-finite entry exactly (Inf with Inf, NaN with NaN), on the fixture's draws held as brms's fixed-parameter fit holds
  them, one chain per point (so `r_eff` is loo's NA, used as 1): `gauss_varint_post`, `poisson_varint_post`
  (8 draws: a 2-draw tail, `k = Inf`, raw importance weights), `gauss_varint_post_psis` (64 draws: a 13-draw tail fitted
  and smoothed, `k` from -0.49 to 1.74), `cumulative_varint_post` (the ordinal `bayes_R2`), `binomial_weighted_post`
  (trials and weights), `categorical_poponly_post` (`log_lik` and `loo`; brms refuses `bayes_R2`) and `mixture_gauss3_post`
  (three Gaussian components: brms's `log_lik_mixture` and the mixture's `posterior_epred`). Harness tests show a
  corrupted log-likelihood entry, R-squared draw, pointwise elpd, k-hat, `n_eff` or SE fails, as does a changed
  non-finite entry.
- `tests/test_post_processing.py`: at every frozen point, `log_lik` row by row equals the PyMC model's elementwise log
  density (whose sum gate G2 checks against brms's `log_prob`) on one case of every family and feature and on the three
  mixture cases, to 1e-12;
  `loo` and `relative_eff` equal loo 2.9.0's outputs recorded in `tests/loo_probes/loo-2.9.0.json` on deterministic
  inputs (S from 8 to 4000, `r_eff` given, NA and computed, heavy tails with `k` above 1, ties at and inside the tail,
  constant columns, log-likelihoods near -380 and -800), within 3e-13 except D50; `psis` equals ArviZ's `psislw` where
  the two take the same tail.

- D05: the same gate compares `ab.waic` with brms's `waic()` on `gauss_loo_compare_x` and `gauss_loo_compare_xw`
  (`check_waic`), and on `gauss_loo_compare_x`, whose autobayes section names the other case (`"compare_with":
  "gauss_loo_compare_xw"`, the same data, its own frozen points), checks that the other model is the one brms compared (its
  log density at its points equals the fixture's `other_lp`), then both models' pointwise `elpd_loo` and `elpd_waic`,
  `ab.loo_compare` by loo and by waic (rows, columns and every entry) and `ab.model_weights` by `"loo"` and by `"pseudobma"`
  with `BB=False` against brms's `loo_compare()` and `model_weights()` (`check_loo_compare`). The weights are 1 and
  3.7e-115, so they are compared as numbers and as logarithms, the small one to 1e-9 in its exponent. Harness tests
  (`tests/oracle/test_loo_compare_harness.py`) show a corrupted pointwise or estimated WAIC, comparison entry, row order,
  column name, pointwise elpd of either model, weight, other model's log density or other case fails.
- D06: the same gate compares `ab.hypothesis` with brms's `hypothesis()` table and draws, and `ab.conditional_effects` with
  its `conditional_effects()` tables (`check_hypothesis`, `check_conditional_effects`), on `gauss_hypothesis_effects`
  (`y ~ x + w + z + (1 | g)`: `"x > 0"`, `"x + w = 0"`, `"zr < zq"`; effects `x` and `z`, the group term left out) and
  `bernoulli_hypothesis_effects` (`yb ~ x + z`: `"Intercept < 0"`, `"x > 0"`, `"zq - zr = 0"`; the probability's
  summaries), every column, labels, stars and brms's NA included. Harness tests
  (`tests/oracle/test_hypothesis_effects_harness.py`) show a corrupted entry of either table, a label, a star, a draw, a
  grid value, a value held, a level, a grouping factor's NA, `cond__`, the column or effect order, or arguments other than
  brms's fail.
- `tests/test_hypothesis_effects.py`: `hypothesis` equals brms 2.23's `hypothesis()` on recorded draws
  (`tests/hypothesis_probes`, recorded by `probe_hypothesis.R` with brms's default method on a data frame of draws, no
  Stan): 45 hypotheses on 40 draws with `alpha` 0.05, 0.1 and 0.2 and `robust` on and off, and eight edge sets (all draws
  on one side, ties at zero, all zero, one draw, `Post.Prob` exactly and just above `1 - alpha`), every entry to 3e-15 and
  every label and star; the 19 hypotheses brms refuses there are refused; R's precedence; on every frozen case `hypothesis`
  takes (84), each of brms's design-column names and `Intercept` has exactly brms's coefficient at the fixture's points as
  its draws. `conditional_effects` on every frozen case it takes (43, ten families): the grid, the values held and the
  summaries of the predictions at the grid; its options, refusals and the multivariate refusals of `log_lik`, `bayes_R2`,
  `loo` and `waic`.
- `tests/test_loo_compare.py`: WAIC, `loo_compare` by loo and by waic, and pseudo-BMA weights equal loo 2.9.0's outputs
  (`tests/loo_probes/loo-compare-2.9.0.json`, recorded by `probe_compare.R` on the inputs of `compare_inputs.py`): seven sets
  of two to five models, given named and unnamed, with exact ties, 8 to 4000 draws, log-likelihoods near -800, nearly equal
  models and a weight that underflows to zero, all within 1e-9; brms 2.23's `loo_compare` and `model_weights` by `"loo"`,
  `"waic"` and `"pseudobma"` (`BB = FALSE`) of three fixed-parameter fits with weights from 4.6e-11 to 0.99, given worst
  first, equal those recomputed from the log-likelihoods brms computed (`brms-2.23.0-weights.json`, recorded by
  `probe_brms_weights.R`); and the API's names, tie order, response warning and refusals.

- E05: the same gate compares `ab.hypothesis` with the fixture's recorded prior draws (`prior=`) against brms's table, the
  point hypotheses' Savage-Dickey `Evid.Ratio` and `Post.Prob` among them, `ab.prior_draws`'s columns with brms's
  `prior_draws()` (`check_prior_draws`), and `ab.conditional_effects` of `"x:z"` and `"x:w"` (300 rows each, `effect2__`)
  with brms's, on `gauss_hypothesis_prior_effects` (`y ~ x * z + x * w + (1 | g)`, Gaussian; `"x = 0"`, `"w = 0.5"`, `"x:zq =
  0"`, `"x > 0"`) and `bernoulli_hypothesis_prior_effects` (the logit model; `"x = 0"`, `"zq = 0.5"`, `"x:w = 0"`, `"x <
  0"`): the ratios to 2e-15. `check_loo_model_weights` compares `ab.loo_model_weights` and `ab.model_weights(...,
  "stacking")` with brms's `loo_model_weights()` on `gauss_loo_compare_x_stacking`, gated (D126). Harness tests
  (`tests/oracle/test_prior_effects_stacking_harness.py`) show a corrupted prior draw, prior-draw column name or count, ratio
  or posterior probability, moderator value, `effect2__` label or column, the moderator's order, and a corrupted stacking
  weight, model order, method or other model fail; with the park lifted the stacking case fails only on brms's weights.
- `tests/test_prior_draws.py`: `density_at` and `density_ratio` equal brms's `density_ratio()` on the 276 probe points;
  `ab.prior_draws`'s names equal brms's `prior_draws()` on the five probed fits, and brms's 1000 recorded draws of each column
  and 20000 of `ab.prior_draws` follow the prior's distribution (one-sample Kolmogorov-Smirnov, p > 1e-3); brms's
  `hypothesis()` tables of those fits under two seeds recomputed from their recorded draws (every one-parameter ratio to
  1e-12, `Intercept`'s NA, the two-parameter ratio only finite: brms's moves with its seed); interaction grids equal brms's
  on every probed pair order and type and six scales of the moderator, R's labels on 3313 values, and on every frozen case
  `conditional_effects` takes with two predictors or more (20), both orders of its first pair; options and refusals.
- `tests/test_loo_compare.py` (E05): `stacking_weights` meets the optimum's conditions to 1e-14 on loo's probe inputs and 200
  random ones, its objective is never below loo's, loo's weights are within 4e-4 of it, `loo_model_weights()` is
  `stacking_weights()` of the pointwise elpd (loo, to the last digit), and on the frozen stacking case it is within 2e-17 of
  the 50-digit optimum while brms's weight is 1.044e-8 off; the park, and the stacking API with the park lifted.

## Divergences (docs/brms-parity/divergences.md)

- D48: brms's further options are not offered: `newdata`, `re_formula`, `resp`, `ndraws`, `draw_ids` (log_lik);
  `summary = TRUE` (bayes_R2); moment matching, `reloo`, `save_psis`, several models in one `loo()` call, `kfold`,
  `loo_R2` (`waic` and `loo_compare` landed with D05).
- D49: at extreme linear predictors brms's R-side `log_lik` works from probabilities that round to 0 or 1 and returns
  `-Inf` or `NaN` (or loses digits) where `ab.log_lik` returns the finite value of the model's own density.
- D50: `mcse_elpd_loo` of an observation with `elpd_loo_i` below about -372, where `exp(elpd_loo_i)^2` underflows: loo's
  is NaN, `ab.loo`'s the value of the same formula, computed as `sum_s (w_s p_s / E - w_s)^2`.
- D51: `r_eff` for chains of fewer than about sixteen draws, and for some strongly autocorrelated chains of about fifty:
  loo's ESS (the posterior package's) and ArviZ's differ there.
- D92: `model_weights`: `"bma"`, `"kfold"` and pseudo-BMA with the Bayesian bootstrap (brms's default `BB = TRUE`) are not
  offered, and `weights` has no default; brms's default `"stacking"` is implemented and withheld (D126).
- D93: `loo_compare` of PSIS-LOO and WAIC results together: loo 2.9.0 compares them without a warning (the table takes the
  first result's column names); refused here. `model_weights` also refuses models on different numbers of observations and
  warns when the responses differ, where brms's `model_weights` gave the weights without a warning on a probe with a
  shifted response.
- D94: the arguments and the output: results (`ab.loo()`, `ab.waic()`) rather than fits, named by the mapping's keys (or
  `model1`, ... as loo names an unnamed list) rather than brms's argument expressions, and the full table as a DataFrame where
  loo prints `elpd_diff` and `se_diff` only by default.
- D95: `waic()` of a multivariate model is refused; brms computes it from the model's `log_lik()` (not probed here).
- D96: a point hypothesis's `Evid.Ratio` and `Post.Prob` are NaN without prior draws, as brms's are NA; with prior draws
  (`prior=`, E05) they are brms's Savage-Dickey ratio and its posterior probability.
- D97: `hypothesis()` takes class `b` of the response formula, `alpha` and `robust`; not brms's `class`, `group`, `scope`
  or `seed`, a distributional formula's coefficients or the special terms' (class `bsp`), or R expressions beyond the
  parser's grammar.
- D98: `conditional_effects()` takes named effects of one predictor and (E05) two-way interactions with brms's default
  `int_conditions`, `resolution` and `prob`; not brms's default of every effect, `conditions`, `int_conditions`,
  `re_formula`, `method`, `robust = FALSE`, spaghetti and surface plots, `categorical`, `transform` or plots.
- D124: `prior_draws()` covers the intercept, `b`, the family's `sigma`, `shape`, `phi`, `nu`, `zi`, `hu`, `sd` and `cor`;
  the constructs listed above are refused; a slope group's equal SD priors are named as brms's group-wide prior; the draws
  are independent draws from the model's priors, not Stan's.
- D125: a Savage-Dickey ratio of several parameters pairs their prior draws through NumPy's permutation, not R's, so it is
  brms's in distribution only; far-tail densities are round-off in both and differ (warned); `density_ratio`'s `n` and
  density options are not offered.
- D126: stacking weights: the exact optimum, where brms's are loo's `constrOptim` optimum (1.044e-8 off on the frozen case,
  up to 3.7e-4 on loo's probes); implemented and withheld until Andrew decides how they are certified.
- D127: interaction effects naming the same pair twice are refused, where brms drops the second with a warning.
- D99: both refuse multivariate and non-linear models, mixtures and the cumulative and categorical families;
  `conditional_effects()` also the binomial family, offsets, `mo()`, `gp()`, `me()`, `mi()`, autocorrelation, addition terms
  and distributional formulas.
