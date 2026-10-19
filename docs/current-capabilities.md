# Current library capabilities — September 27, 2026

Snapshot of merged main `005ff442e9ff02edcf5bf6769dffa9cda2053caf` (PR17).
This is a capability map, not a percentage of brms implemented.

| Surface | Merged behavior | Tests / documentation |
| --- | --- | --- |
| Families | Gaussian, Bernoulli and grouped binomial with a logit, probit or cloglog link (the cloglog accepted against two references, D18), Student-t with a fixed finite nu > 1 or nu learned under its own prior truncated to nu > 1, Poisson-log, negative binomial-log and Gamma-log with a `shape` prior, lognormal (identity link on the log-scale location, `sigma` the SD of `log y`), Beta-logit proportions with a precision `phi` prior | test_logp.py, test_bernoulli.py, test_binomial.py, test_links.py, test_student.py, test_student_learned_nu.py, test_poisson.py, test_negbinomial.py, test_gamma.py, test_lognormal.py, test_beta.py; [negative binomial](negbinomial.md), [Gamma](gamma.md), [lognormal](lognormal.md), [Beta](beta.md), [links](links.md), [Student-t learned nu](student-learned-nu.md) |
| Formula composition | Numeric and declared categorical predictors; multiple numeric-by-categorical interactions with required main effects; optional intercept; one offset term `offset(col)` or `offset(log(col))` added to the linear predictor without a coefficient | test_intercept_controls.py, test_multiple_interactions.py, test_offset.py; [offset](offset.md) |
| Hierarchy | Population-only models or multiple independent varying-intercept factors; independent varying slopes `(1 + x || g)` on numeric population-level terms, and brms's correlated form `(1 + x \| g)` with an LKJ prior `cor={"g": ab.LKJ(eta)}` on the correlation of the intercept and slope(s) (Stan's `cholesky_corr` coordinates); grouping by the observed combinations of two columns, `(1 \| g:h)`, and brms's nesting shorthand `(1 \| g/h)` = `(1 \| g) + (1 \| g:h)`, with brms's `g_h` level labels | test_population_only.py, test_multiple_groups.py, test_varying_slope.py, test_correlated_slopes.py, test_grouping_syntax.py; [varying slopes](varying-slopes.md), [correlated slopes](correlated-slopes.md), [grouping syntax](grouping-syntax.md) |
| Execution | Inspectable PyMC graph; PyMC NUTS or JAX/NumPyro sampling with explicit device selection | inference.py, test_device.py and test_sampling.py; availability does not establish GPU validation |
| Prediction | Expected response and replicated observations; explicit unseen-group policies | test_predict.py, test_multiple_groups.py, family tests |
| Provenance | Frozen target identity; fit binding captured before sampling; mismatch refusal; explicit unverified legacy route | test_record_identity.py, test_fit_binding.py; [fit binding](fit-binding.md) |

Test filenames are relative to `tests/`; implementation filenames to `autobayes/`.
Family-specific behaviour (parent, link, response validation, scale and fixed parameters, prediction rules,
wording) is dispatched through the registry in `autobayes/families/`, one module per family; see
[families](families.md). `tests/test_family_registry.py` forbids family-name branches elsewhere.
A parsed formula is an explicit term model (`PopTerm`, `GroupTerm`) with one design block per group-level
coefficient; the flat views (`predictors`, `terms`, `columns`, `groups`, `group_designs`) are derived from it;
see [term model](term-model.md).
Student-t scale is not generally its standard deviation. Binomial expected count
is n*p, distinct from probability p. A Poisson exposure enters as `offset(log(exposure))` (A06).
See the family-specific pages for numerical support and refusal boundaries.

## Validation boundaries

Final PR17 hosted run36299542672 passed631 tests, with7 deselected, on reviewed
head c838aec7e4dc26bd8792e28cc7d97643d272fd9f. This count is a dated result, not
proof of arbitrary formula combinations, convergence or scientific calibration.
The fast suite includes existing sampling fixtures; it is not wholly deterministic.

PR12 multiple interactions has a separately reviewed Gaussian CPU example:
zero divergences on both backends, maximum R-hat1.00388470, minimum bulk ESS2267,
and all12 intended posterior-mean comparisons accepted. The initial local full
suite was interrupted and remains incomplete; final hosted CI passed563 tests.
These are distinct evidence sources. Coordinator receipts preserve the original
incorrect double-centering review and corrected raw-intercept review; no new fit
was run for that correction. This evidence is specific to that Gaussian example.

The original Gaussian H100 example is the limited existing GPU evidence.
Student-t and Poisson additions have deterministic CPU/JAX checks, not new
recovery fits or GPU validation. Do not extrapolate one GPU run to all features.

## Next foundations and exclusions

Broader model comparison, distributional parameter
formulas, learned Student-t degrees of freedom, censoring and
truncation remain separate work. Independent varying slopes landed through the brms-parity loop (A01,
verified against frozen brms fixtures; PR8 is superseded), as did the offset term (A06) and correlated varying
slopes (A02, `(1 + x | g)` under an LKJ prior, verified against the frozen fixture `gauss_slope_correlated`; see
[correlated slopes](correlated-slopes.md)).
Standalone generated source-code export is not implemented. The library builds
an inspectable PyMC graph and executes through the selected sampler backend.

The next useful feature design is a prior-predictive interface with explicit
prior/response semantics, preserved row and group labels, and reproducible seeds.
Its API, numerical-domain checks and bounded tests must be decided before code;
this documentation update does not launch sampling or a prior-predictive campaign.

Prior-predictive modeled-row CPU draws are now available; see [prior predictive](prior-predictive.md). The snapshot and historical counts above refer to PR17, not this addition.

General two-way interactions (A04, `x:w` and `z:h` next to `x:z`) landed through the brms-parity loop, verified
against the frozen fixtures `gauss_interaction_numeric_numeric` and `gauss_interaction_categorical_categorical`;
see [interactions](interactions.md). Three-way interactions and interactions with transformed terms stay refused.
In-formula transforms (A05, `log(x)`, `exp(x)`, `sqrt(x)`, `I(x^k)`, `I(x * k)`, `I(x + k)`, `I(x - k)`) followed,
verified against the frozen fixture `gauss_transforms`; see [transforms](transforms.md). Other calls (`scale()`,
`poly()`, `log(x + 1)`) are refused with the supported list.

Negative binomial counts (A08, `ab.negbinomial`, brms's `negbinomial` with its `shape` parameter under an explicit
`Priors(shape=...)`) landed through the brms-parity loop, verified against the frozen fixture `negbinomial_varint`;
see [negative binomial](negbinomial.md). Gamma responses with a log link (A09, `ab.gamma`, the same `shape` prior;
brms's default inverse link is not offered, divergence D9) followed, verified against `gamma_varint`; see
[Gamma](gamma.md). Lognormal responses (A10, `ab.lognormal`, brms's `lognormal()`: the identity link on the log-scale
location `mu`, `sigma` the SD of `log y` under `Priors(sigma=...)`, `predict(kind="mean")` returning brms's
`posterior_epred` `exp(mu + sigma^2 / 2)` as `mean`) followed, verified against `lognormal_varint`; see
[lognormal](lognormal.md). Beta proportions with a logit link (A11, `ab.beta`, brms's `Beta()`: precision `phi` under
`Priors(phi=...)`, the likelihood built as this library's own Bambi family so that the graph keeps the linear predictor
`logit_mu` and brms's name `phi`; `1 - mu` as `expit(-eta)`, divergence D13) followed, verified against `beta_varint`;
see [Beta](beta.md). The probit link for the Bernoulli and binomial families (A12, `link="probit"`: `p = Phi(eta)`
through a stable `log Phi`, the graph keeping `eta` as `probit_p`) followed, verified against `bernoulli_probit` and
`binomial_probit`; brms's cloglog link was withheld because its frozen fixture records a brms gradient off by 3.8e-7
relative where AutoBayes's form is exact (D18), until P6 (below), and its cauchit link and the other families'
non-default links are not offered (D19); see [links](links.md). Student-t with learned degrees of freedom (A13, `ab.student(..., nu=None)` with
`Priors(nu=...)`: the prior truncated to `nu > 1` and renormalised there as brms's `real<lower=1> nu`, sampled as
`log(nu - 1)`; Gamma, Exponential, HalfNormal and HalfCauchy priors, D22) followed, verified against
`student_learned_nu`; its `describe()` lists `sigma_log__` before `nu_interval__` whatever `PYTHONHASHSEED` makes PyMC's
order; see [Student-t learned nu](student-learned-nu.md).
brms's default priors (A07, `ab.default_priors(formula, data, ...)`: brms 2.23's `default_prior()` for the formula, data
and family as a `Priors` to pass to a constructor, with brms's data-dependent Intercept location and scales; flat `b` as
`ab.Flat()`, the negative binomial shape's `inv_gamma(0.4, 0.3)` as `ab.InverseGamma`) are available on request and
marked as defaults in `describe()` and fit records; nothing is filled in unasked (D1). Gate G6 now compares them with
the `default_prior()` table of every fixture, and five fixtures frozen without priors check the densities under them;
an intercept-only model needs no `b` prior (D2); see [default priors](default-priors.md).
Zero-inflated and hurdle Poisson counts (B06, `ab.zero_inflated_poisson` and `ab.hurdle_poisson`, brms's
`zero_inflated_poisson()` and `hurdle_poisson()`: the zero-inflation or hurdle probability `zi` / `hu` under
`Priors(zi=ab.Beta(a, b))`, named and parameterised as brms's, the linear predictor kept in the graph as `log_mu`, the
densities in a stable form, never NaN and finite wherever the true value is a double, where brms's are NaN or
infinite at some extreme linear predictors, D28; `predict(kind="mean")` returning brms's `posterior_epred` as `mean`)
followed, verified against `zero_inflated_poisson_varint` and `hurdle_poisson_varint`; only Beta priors on `zi` / `hu`
(D29), and formulas on `zi` / `hu` are not offered (D30); D03 (at the end of this file) adds six more such families; see
[zero-inflated and hurdle families](zero-inflated-hurdle.md).

brms's response addition terms (B02): `weights(w)` multiplies each row's log likelihood by its weight for every Tier A family (and, since D03, the eight zero-inflated and hurdle families)
(`y | weights(w) ~ ...`, `k | trials(n) + weights(w) ~ ...`), verified against `gauss_weighted` and one frozen weighted
case per family; brms's `cens()` and `trunc()` were parsed and implemented for the Gaussian family but withheld, because
brms's own gradients (Stan's `normal_lcdf` derivative approximations, `normal_lccdf`'s `1 - erf`) and truncated mean
are not exact enough at points of their frozen fixtures to certify them (D24, D25), until P6 (below); see
[addition terms](addition-terms.md).

Ordinal and categorical responses (B05, `ab.cumulative` and `ab.categorical`, brms's `cumulative("logit")` and
`categorical()`: the response fitted as its 1-based category with `model.categories` holding the labels; the cumulative
thresholds sampled as brms samples them, for the centred design on Stan's ordered coordinates, with `b_Intercept` the
thresholds of the raw design and the intercept prior on every threshold; one linear predictor per non-reference category
for the categorical family, under one prior per class; `predict(kind="mean")` returning the category probabilities with a
new `category` dimension) followed, verified against `cumulative_varint`, `categorical_poponly` and two cases with a
categorical predictor frozen with it; a categorical model takes population-level terms only and its categories are
sorted labels (D36, D37), and other ordinal links and `disc` are not offered (D38); see [ordinal and categorical
responses](ordinal-categorical.md).

Distributional formulas (B01, brms's `bf(y ~ x, sigma ~ z)`): a list of formulas, `["y ~ x + (1 | g)", "sigma ~ z"]`,
gives the auxiliary parameter its own population-level linear predictor under a log link, with
`Priors(sigma={"intercept": ..., "b": ...})`, for `sigma` (Gaussian, Student-t, lognormal), `shape` (negative
binomial, Gamma) and `phi` (Beta); `predict()` uses each row's parameter and returns it with `dpar="sigma"`. Verified
against `gauss_distributional_sigma` and one frozen case per further family; group-level terms in such a formula,
formulas on `nu`, `zi` or `hu`, and other links are refused (D32-D35); see
[distributional formulas](distributional.md).

Monotonic effects (B03, brms's `mo(x)`): an ordinal predictor (whole numbers, or an ordered pandas Categorical) coded as
brms codes it adds its coefficient (class `b`) times `D` times the share of a simplex up to each row's value, the simplex
under `Priors(simo={"x": ab.Dirichlet(...)})` and sampled on Stan's stick-breaking coordinates; for every family but the
categorical one, verified against `gauss_monotonic` and one frozen case per further family; interactions with `mo()`,
`mo()` in group-level or distributional formulas, `mo(x, id = ...)` and two-valued predictors are refused (D40-D42; names D43); see
[monotonic effects](monotonic.md).

Two-reference acceptance (P6, Andrew's decision of 2026-10-11; docs/brms-parity/LOOP.md, "Two references"): a case whose
brms side is inexact at some frozen points may opt in to have its gradient and/or expected values compared with brms
where brms is accurate and with a committed 50-digit reference (`oracle/reference.py`, `tests/oracle/references/`)
where it is not; log densities never fall back. It lifted three parks: the cloglog link of the Bernoulli and binomial
families (`link="cloglog"`, verified against `bernoulli_cloglog`, 2 of 144 gradient entries on the reference, and
`binomial_cloglog`, frozen with it, against brms alone; D18), and `cens()` and `trunc()` for the Gaussian family
(verified against `gauss_censored`, 32 of 160 gradient entries on the reference, and `gauss_truncated`, 2 of 160 gradient
entries and 6 of 656 expected values; D24, D25); see [links](links.md) and [addition terms](addition-terms.md).

Finite mixtures (C05, brms's `mixture(gaussian, gaussian)`): `ab.mixture(formula, data, priors, families=["gaussian"] * K)`
fits K Gaussian components with one formula, each with its own centred intercept, coefficients and `sigma<k>`, identified
by ordering the intercepts (brms's default `order = "mu"`, Stan's ordered coordinates) and weighed by `theta` (a Dirichlet
on Stan's stick-breaking simplex), under `Priors(components=[...one Priors per component...], theta=ab.Dirichlet(...))`;
`predict()` returns each component's linear predictor with a `component` dimension, the mixture's mean and draws by
component. Verified against `mixture_gauss_gauss` and two cases frozen with it (three components with a categorical
predictor and a prior per component; brms's default priors); other component families, `order = "none"`, formulas per
component or on `theta`, group-level terms and offsets are refused (D52-D55); see [mixtures](mixtures.md).
`ab.log_lik`, `ab.bayes_R2` and `ab.loo` (C07) take a mixture: each row's `log_mix` of the components' Normal
densities (brms's `log_lik_mixture`) and the R-squared of the mixture's mean, verified against the frozen case
`mixture_gauss3_post`.

Post-processing (C07, brms's `log_lik()`, `bayes_R2()` and `loo()` with loo 2.9.0): `ab.log_lik(model, fit)` is each
fitted row's log density per draw (times its weight under `weights()`) for every family, `ab.bayes_R2(model, fit)` the
per-draw R-squared of the expected response (of the expected category for the cumulative family; refused for the
categorical family, as brms refuses it), and `ab.loo(model, fit)` PSIS-LOO with brms's pointwise table, estimates, Pareto
k and `n_eff`, the relative efficiencies computed from the chains as brms computes them; PSIS is implemented from the
published algorithm with ArviZ's `psislw` as the reference (loo is GPL-3 and none of its source is used), and its tie and
degenerate-tail rules were settled against loo's outputs on recorded probe inputs (`tests/loo_probes`). Verified against
`gauss_varint_post`, `poisson_varint_post`, `gauss_varint_post_psis`, `cumulative_varint_post`, `binomial_weighted_post`,
`categorical_poponly_post` and `mixture_gauss3_post` (gate `check_post`); brms's further options are not offered (D48), and brms's `log_lik` at
extreme linear predictors, loo's MCSE of an observation with `elpd_loo` below about -372 and loo's relative efficiencies
of very short chains differ (D49-D51); see [post-processing](post-processing.md).
`ab.waic(model, fit)` (D05, brms's `waic()`) is WAIC with brms's pointwise table and estimates, `ab.loo_compare(results)`
loo 2.9.0's comparison table of several models' `ab.loo()` or `ab.waic()` results (best first, `elpd_diff`, `se_diff` and each
model's estimates), and `ab.model_weights(results, weights)` brms's Akaike weights by `"loo"` or `"waic"` and pseudo-BMA weights
with `BB=False`; verified against `gauss_loo_compare_x` and `gauss_loo_compare_xw` (gate `check_post`) and against loo's and
brms's outputs on recorded probes; BMA, K-fold, the Bayesian bootstrap, mixed criteria and a multivariate model's WAIC are
refused (D92-D95), and stacking is implemented (the exact optimum, `post_processing.stacking_weights`) and withheld: brms's
weights on `gauss_loo_compare_x_stacking` are loo's optimiser's, 1.04e-8 off the optimum (D126).
`ab.hypothesis(model, fit, hypotheses)` (D06, brms's `hypothesis()`) tests point and one-sided hypotheses on the
population-level coefficients by brms's names (brms's table: Estimate, Est.Error, the interval, Evid.Ratio, Post.Prob, Star,
and the draws), reading the text with a parser of its own, never evaluating it; `ab.conditional_effects(model, fit, effects)`
(brms's `conditional_effects()` with its defaults) gives each predictor's grid with the others at their means and reference
levels, the group-level effects left out, and the median, MAD and 95 % interval of the expected response; verified against
`gauss_hypothesis_effects` and `bernoulli_hypothesis_effects` (gate `check_post`) and against brms's `hypothesis()` on
recorded draws; its other arguments and the models no case verifies are not offered (D97-D99). `ab.log_lik`, `ab.bayes_R2`
and `ab.loo` refuse a multivariate model naming it.
E05: `ab.prior_draws(model, draws, seed)` draws from the model's priors with brms's `prior_draws()` columns (`sample_prior =
"yes"`; D124), and `ab.hypothesis(..., prior=...)` gives each point hypothesis brms's Savage-Dickey `Evid.Ratio` and
`Post.Prob` (brms's `density_ratio()`, `autobayes.kde`; D96, D125); `ab.conditional_effects` takes two-way interaction effects
`"x:z"` with brms's default `int_conditions` (D98, D127); verified against `gauss_hypothesis_prior_effects` and
`bernoulli_hypothesis_prior_effects` (gate `check_post`) and against brms's and R's outputs on recorded probes
(`tests/hypothesis_probes`).

Multivariate Gaussian models (C01, brms's `bf(mvbind(y, y2) ~ ...) + set_rescor()`): `ab.gaussian("mvbind(y, y2) ~ x + (1 | g)",
data, ab.Priors(resp={"y": {...}, "y2": {...}}, rescor=ab.LKJ(eta)), rescor=True)` builds one univariate model per response
on the same rows and right-hand side and assembles their graphs into one, with `rescor=True` the responses' likelihoods
replaced by brms's `multi_normal_cholesky` through the residual correlation's Cholesky factor `Lrescor` (LKJ prior, Stan's
`cholesky_corr` coordinates) and with `rescor=False` independent; `predict()` adds a `response` dimension and draws each row's
responses jointly. Verified against `gauss_mv_rescor`, `gauss_mv_norescor` and `gauss_mv_slopes` (categorical, interaction,
offset, correlated slopes per response under brms's shared `cor` prior); a formula per response (with `mi()`, C06 below), `mi()`, `mo()`, `(1 |ID| g)`
across responses, other families and a missing `rescor` are refused (D56-D58; names D59); see
[multivariate models](multivariate.md).

Gaussian processes (C03, brms's `gp()`): `gp(x)`, the exact process of one numeric column, and `gp(x, k = K, c = C)`, its
Hilbert-space approximation with `K` basis functions, both with the squared-exponential kernel and brms's scaling of the input,
add the process's value at each row's input to the linear predictor, under `Priors(sdgp=..., lscale=...)` keyed by column (brms's
defaults, including the inverse-gamma `lscale` tuned to the inputs, from `ab.default_priors()`); for every family but the
categorical one, verified against `gauss_gp_exact` and `gauss_gp_approx`; several covariates, `by`, other kernels, `iso`,
`gr = FALSE`, `scale = FALSE` and `gp()` in interactions, group-level or distributional formulas are refused (D64), an exact GP
predicts at fitted inputs only (D65) and an approximate one inside its boundary (D66); see [Gaussian processes](gaussian-processes.md).

Non-linear formulas (C02, brms's `bf(y ~ a * exp(b * x), a ~ 1 + (1 | g), b ~ 1, nl = TRUE)`): `ab.gaussian([...], data, priors,
nl=True)` builds an `ab.NonlinearModel` whose mean is an expression in non-linear parameters, numeric covariates and constants
(`+ - * / ^`, unary minus, `exp`, `log`, `sqrt`, `inv_logit`/`plogis`, `pow`; parsed by the library's own parser, never
evaluated), each parameter with its own never-centred linear formula of the term model (population-level terms and every
group-level term the term model takes) under `Priors(nlpars={"a": {"b": ..., "sd": ...}})`; the library builds the PyMC graph
itself, and `predict(nlpar="a")` returns a parameter's values; `ab.log_lik`, `ab.bayes_R2` and `ab.loo` (C07) take it as
they take a linear Gaussian model. Verified against `gauss_nonlinear_exp` and `gauss_nonlinear_terms` (frozen with it) at
machine precision; `nlf()`, `lf()`, nested parameters, functions outside the grammar, `gp()` and `mo()`, distributional,
multivariate and mixture combinations, other families and `loop = FALSE` are refused (D60-D63); see
[non-linear formulas](nonlinear.md).

Autocorrelation terms (C04, brms's `ar(time = t, gr = g, p)`, `ma(time = t, gr = g, q)` and `arma(time = t, gr = g, p, q)`
in the regression formulation, `cov = FALSE`): for the Gaussian family, each row's linear predictor gains `ar` and `ma` times
the lagged residuals of earlier rows of its own series in time order, after every other term, under
`Priors(ar=..., ma=...)` (unbounded; brms's default flat, `ab.Flat()`); the MA recursion is a `pytensor.scan` over the
positions of a series that the JAX backend compiles. Predictions on the fitted rows add the term from the observed
residuals, in data order, as brms's `posterior_linpred` does; new rows are refused. Verified against `gauss_ar1`,
`gauss_arma11` and `gauss_arma22_varint` (lags two deep next to a varying intercept); `ab.log_lik`, `ab.bayes_R2` and
`ab.loo` (C07) read the predictor with the term, as brms's do; `cov = TRUE`, `cosy()`, `car()`, `sar()`, `fcor()`,
`unstr()`, other families, distributional formulas, addition terms, multivariate formulas, non-linear models, mixtures,
`gp()` next to the term and new rows are refused (D68-D71); see [autocorrelation](autocorrelation.md).

Missing values and measurement error (C06, brms's `mi()` and `me()`): a list of formulas, one per response,
`ab.gaussian(["y ~ mi(xm) + w", "xm | mi() ~ w"], data, ab.Priors(resp={...}), rescor=False)`, makes the missing values (NaN) of
`xm` parameters (`Ymi_xm`, flat), one variable shared by `xm`'s likelihood and `y`'s `mi(xm)` term (several `mi()` terms and
acyclic chains of imputations are taken); `ab.gaussian("y ~ me(xobs, sdx)", data, ab.Priors(..., meanme=..., sdme=...))` gives a
predictor observed with known noise latent values `meanme + sdme * zme` that the noisy values measure; `predict()` uses the
draws' imputed and latent values on the fitted rows. Verified against `gauss_mi` and `gauss_me`; `mi()` without its column's
formula, with `rescor=True`, in interactions, group-level or distributional formulas, and `me()` with `gr`, several `me()` terms,
`me()` outside a univariate Gaussian formula, other families, new rows of an `me()` model and `log_lik()` / `loo()` /
`bayes_R2()` of these models are refused (D72-D74; names D75), and so are both terms next to `gp()` or an autocorrelation term,
in non-linear models and in mixtures (C02-C05), with each other feature's priors in these models and `meanme` / `sdme` in
theirs; see [missing values and measurement error](missing-and-measurement-error.md).

Multi-membership, `gr(by = )` and IDs shared with sigma (D08): `(1 | mm(g, g2, weights = cbind(w1, w2)))` (each row a weighted
member of several levels, brms's group `mmgg2`, weights scaled to sum to 1), `(1 | gr(g, by = gby))` (one SD per level of
`gby`, brms's prior constant included) and `["y ~ x + (1 + x |p| g)", "sigma ~ 1 + (1 |p| g)"]` (one correlation matrix across
the response formula's term and log sigma's varying intercept, `Priors(sigma={"intercept": ..., "sd": ...})`); `predict()`,
`ab.log_lik` and the default priors take them. Verified against `gauss_mm`, `gauss_gr_by` and `gauss_shared_id`; slopes with
`mm()` and `by`, the terms next to other special terms, in multivariate and non-linear models, other IDs and new levels of `by`
and shared-ID factors are refused (D104-D106); `cs()` and the case `acat_cs` came with D04's acat family (D107, below); see
[multi-membership and by](multi-membership-and-by.md).

Weibull, exponential, skew-normal, shifted lognormal and ex-Gaussian responses (D01, brms's `weibull()`, `exponential()`,
`skew_normal()`, `shifted_lognormal()`, `exgaussian()`): `ab.weibull` and `ab.exponential` with a log link on the mean,
`ab.skew_normal` and `ab.exgaussian` with the identity link on the mean (brms's mean parameterisations), `ab.shifted_lognormal`
with the identity link on the location of `log(y - ndt)`; the extra parameters `alpha` (skewness, a location prior), `beta` (the
exponential component's mean, a positive-support prior) and `ndt` (the non-decision time below the smallest response, its prior
truncated to (0, min y) as brms does, or brms's default `ab.Uniform(0, "min_Y")`) under `Priors(alpha=..., beta=..., ndt=...)`;
brms's `posterior_epred`, response draws, `log_lik()` and brms's default priors. Verified against `weibull_varint`,
`exponential_varint`, `skew_normal_varint`, `shifted_lognormal_varint` and `exgaussian_varint` (log densities within 1.4e-15
relative); other links, distributional formulas, addition terms, `mo()`, `me()`/`mi()`, autocorrelation terms, multivariate and
mixture use are refused (D76-D79); see [continuous families](continuous-families.md).

Beta-binomial, geometric and `negbinomial2` responses (D02, brms's `beta_binomial()`, `geometric()`, `negbinomial2()`):
`ab.beta_binomial` (`k | trials(n) ~ ...`, the logit link on the mean success probability and the precision `phi`, trials handled
as the binomial's), `ab.geometric` (the negative binomial of shape 1) and `ab.negbinomial2` (the negative binomial of shape
`1 / sigma`, `Priors(sigma=...)`), each with brms's link; brms's `posterior_epred` (`expected = n mu` for the beta-binomial),
response draws, `log_lik()` and brms's default priors. Verified against `beta_binomial_varint`, `geometric_varint` and
`negbinomial2_varint` (log densities within 8.8e-16 relative); other links, distributional formulas, addition terms, `mo()`,
`me()`/`mi()`, `mm()`, autocorrelation terms, multivariate, mixture and non-linear use are refused (D80-D82); see
[discrete families](discrete-families.md).

Shrinkage priors on b (D07, brms's `horseshoe()` and `R2D2()` on class `b`): `ab.Priors(b=ab.Horseshoe(df, scale_global,
df_global, scale_slab, df_slab))` (the regularised horseshoe) and `ab.Priors(b=ab.R2D2(mean_R2, prec_R2, cons_D2))`, brms's
defaults and argument order, one joint prior on every coefficient built non-centred as brms builds it (`b = zb * sdb`, the
scales under brms's names on brms's coordinates) and scaled by the residual SD sigma (brms's `autoscale = TRUE`);
`describe()`, the prior table, the target identity, `prior_predictive()` and `predict()` take them. Verified against
`gauss_horseshoe` and `gauss_r2d2`; the prior on anything but the whole class `b`, `par_ratio`, `autoscale = FALSE`,
`main = TRUE`, other families, formulas without an intercept or population-level terms, `center_predictors=False`, and
group-level, offset, addition, distributional, `mo()`, `me()`, `mi()`, `gp()` and autocorrelation terms are refused
(D100-D103); see [shrinkage priors](shrinkage-priors.md).

More ordinal models (D04, `ab.sratio`, `ab.cratio`, `ab.acat`: brms's `sratio("logit")`, `cratio("logit")` and `acat("logit")`):
the response, `mu` and the centring of the thresholds as for the cumulative family, the K - 1 thresholds unordered as brms's
`vector[nthres] Intercept`, each row's probability in a stable log form; category-specific effects `cs(x)` give a numeric column
one coefficient per threshold (brms's `bcs`, under the class-`b` prior), each threshold's predictor the deterministic `mu_cs`,
returned by `predict(kind="linear")` with a `threshold` dimension; `predict()`, `log_lik()`, `loo()`, `bayes_R2()` and brms's
default priors. Verified against `sratio_varint`, `cratio_varint`, `acat_varint`, `acat_cs`, `sratio_cs` and `cratio_cs` (and
`log_lik()`, `bayes_R2()`, `loo()` against `sratio_cs_post`, `cratio_cs_post`, `acat_cs_post`); other
links, `disc`, `mo()`, addition terms and the other forms of `cs()` are refused (D38, D89). The cumulative probit link,
`ab.cumulative(..., link="probit")`, is offered: brms's own gradient of `cumulative_probit_varint` is off the exact value by
up to 6.5e-5 relative, and the case is accepted against a 50-digit reference of the model where brms is off it (P6, D88); see
[ordinal and categorical responses](ordinal-categorical.md).

Six more zero-inflated and hurdle families (D03, brms's `zero_inflated_negbinomial()`, `hurdle_negbinomial()`,
`zero_inflated_binomial()`, `zero_inflated_beta()`, `hurdle_gamma()`, `hurdle_lognormal()`): `ab.zero_inflated_negbinomial` and
`ab.hurdle_negbinomial` (log link, `shape` and `zi` / `hu`), `ab.zero_inflated_binomial` (`k | trials(n)`, logit link, `zi`),
`ab.zero_inflated_beta` (proportions in [0, 1), logit link, `phi` and `zi`), `ab.hurdle_gamma` (log link, `shape` and `hu`) and
`ab.hurdle_lognormal` (identity link on the log-scale location, `sigma` and `hu`), each with brms's density in a stable form (the
negative binomial coefficient through Stirling's remainder, the hurdle truncation `log(1 - NB(0))` exact at small means),
brms's `posterior_epred`, response draws (the hurdle negative binomial's exactly zero-truncated, where brms's are not, D86),
`log_lik()`, brms's default priors and `describe()`; and `weights()` for these and the zero-inflated and hurdle Poisson.
Verified against `zero_inflated_negbinomial_varint`, `hurdle_negbinomial_varint`, `zero_inflated_binomial_varint`,
`zero_inflated_beta_varint`, `hurdle_gamma_varint`, `hurdle_lognormal_varint`, `zero_inflated_poisson_weighted` and
`hurdle_poisson_weighted` (log densities within 4.4e-16 relative); `gp()` and the term model are taken, while other links,
formulas on `zi` / `hu` and the auxiliary parameters, `cens()`, `trunc()`, `mo()`, `me()`/`mi()`, `mm()`, `gr(by = )`,
autocorrelation terms, multivariate, non-linear and mixture use are refused (D84); see
[zero-inflated and hurdle families](zero-inflated-hurdle.md).

More links for the Tier A families (E06, `link=` on `ab.poisson`, `ab.negbinomial`, `ab.gamma` and `ab.beta`): brms's
`poisson("sqrt")` (`mu = eta^2`), `negbinomial("identity")`, `Gamma("inverse")` and `Gamma("identity")` (each needing a positive
linear predictor), `Beta("probit")` and `Beta("cloglog")`, each with the linear predictor kept as the graph's parent, its
likelihood in a stable form (the Beta's `1 - mu` computed directly, D132), `predict()`, `log_lik()`, brms's default priors under
the link and `describe()`; a non-positive mean under the identity and inverse links is refused in `predict()` and
`prior_predictive()` (D134). Verified against `poisson_sqrt`, `negbinomial_identity`, `gamma_inverse`, `gamma_identity`,
`beta_probit` and `beta_cloglog`; under these links the term model and `offset()` are taken, while `weights()` and the other
addition terms, formulas on `shape` / `phi`, `mo()`, `gp()`, `mm()`, `gr(by = )`, `me()`/`mi()` and autocorrelation terms are
refused naming the link (D133). The cumulative cloglog link is built but withheld: brms's own log density of
`cumulative_cloglog` is 5.9e-7 relative off the exact value at one point, which no gate accepts (D38, D132); see
[links](links.md).

Four bounded and hurdle families (E01, brms's `zero_one_inflated_beta()`, `zero_inflated_beta_binomial()`, `xbeta()` and
`hurdle_cumulative()`, logit link): `ab.zero_one_inflated_beta` (proportions in [0, 1], `phi`, `zoi` and `coi`),
`ab.zero_inflated_beta_binomial` (`k | trials(n)`, `phi` and `zi`), `ab.xbeta` (Kosmidis and Zeileis's extended-support Beta:
proportions in [0, 1] from one censored stretched Beta, `phi` and `kappa`; its masses through an incomplete Beta function
differentiable in its shapes on both paths, a custom JAX derivative where JAX has none) and `ab.hurdle_cumulative` (brms's codes
0..K, the hurdle category 0 with probability `hu` and a cumulative logit model above it), each with brms's density in a stable
form, `posterior_epred` (xbeta's the censored mean), response draws, `log_lik()`, brms's default priors and `describe()`; `weights()`
for the three that are not ordinal. Verified against `zero_one_inflated_beta_varint`, `zero_inflated_beta_binomial_varint`,
`xbeta_varint` and `hurdle_cumulative_varint` (log densities within 7.1e-16 relative, gradients within 1.5e-9); `gp()` and the
term model are taken, while other links, formulas on the auxiliary parameters, `cens()`, `trunc()`, `mo()`, `me()`/`mi()`,
`mm()`, `gr(by = )`, `cs()`, autocorrelation terms, shrinkage priors, multivariate, non-linear and mixture use are refused (D108),
and `hurdle_cumulative`'s other links, `disc`, threshold options and `bayes_R2()` too (D109); see
[bounded and hurdle families](bounded-families.md).

Inverse Gaussian, von Mises, asymmetric Laplace, Frechet and generalized extreme value responses (E02, brms's
`inverse.gaussian()`, `von_mises()`, `asym_laplace()`, `frechet()`, `gen_extreme_value()`): `ab.inverse_gaussian` under brms's
default `1/mu^2` link (`mu = 1 / sqrt(eta)`, `-inf` where `eta <= 0`) or `link="log"`, with `shape`; `ab.von_mises` (angles in
`[-pi, pi]`, `tan_half` link, `kappa`, log I0 exact for every kappa where Stan's overflows above 713.9, D113); `ab.asym_laplace`
(quantile regression, `sigma` and the quantile learned under `Priors(quantile=ab.Beta(...))` or fixed, `quantile=0.25`, brms's
`bf(quantile = 0.25)`); `ab.frechet` (log link, `nu > 1` learned or fixed as the Student-t nu); `ab.gen_extreme_value` (`sigma`, and
brms's xi derived from `tmp_xi` so that every fitted row lies in the support, the graph's deterministic `xi`, `Priors(xi=...)` the
prior of `tmp_xi`); brms's `posterior_epred` (infinite for the GEV where `xi >= 1`, D114), response draws, `log_lik()` and brms's
default priors. Verified against `inverse_gaussian_varint`, `inverse_gaussian_log`, `von_mises_varint`, `asym_laplace_varint`,
`asym_laplace_quantile`, `frechet_varint` and `gen_extreme_value_varint` (log densities within 2e-15 relative; G3 checks the
GEV's xi against brms's `posterior_epred(dpar = "xi")`); the term model and `gp()` are taken, while other links, distributional
formulas, addition terms, `mo()`, `me()`/`mi()`, `cs()`, `mm()`, `gr(by = )`, autocorrelation terms, multivariate, non-linear,
mixture and shrinkage-prior use are refused (D112, D115); see [continuous families](continuous-families.md).

Distributional formulas and `weights()` for the Tier D families (E03): a formula of its own on the Weibull's `shape` and the
beta-binomial's `phi` (log link), the zero-inflated negative binomial's `zi` and the hurdle lognormal's `hu` (logit link; brms's
default `logistic(0, 1)` on the intercept, `ab.Logistic`) and the skew normal's `alpha` (identity link), `["zcnt ~ x", "zi ~ z"]`
with `Priors(zi={"intercept": ..., "b": ...})`, each row's parameter in the density, `predict()` (with `dpar=`), `log_lik()`,
brms's default priors and `describe()`; and `weights()` for the Weibull and beta-binomial families. Verified against
`weibull_distributional_shape`, `beta_binomial_distributional_phi`, `zero_inflated_negbinomial_distributional_zi`,
`hurdle_lognormal_distributional_hu`, `skew_normal_distributional_alpha`, `weibull_weighted` and `beta_binomial_weighted` (log
densities within 7.9e-16 relative, gradients within 2.5e-9); group-level terms in these formulas and formulas on the families'
other parameters stay refused (D116, with D79, D81, D84), the predicted probabilities saturate at extreme linear predictors (D117),
and `conditional_effects()` keeps refusing them (D99); see [contexts of the newer families](contexts-newer-families.md).

`cens()` and `trunc()` beyond the Gaussian family (E04): brms's interval censoring `cens(c, y2)` (code 2, the response in
`(y, y2]`, brms's `rcens`) for every family that takes `cens()`; `cens()` for the Weibull (the cloglog forms of its CDF) and
lognormal (the Normal tails of `log y`) families; `trunc(ub = b)` for the Poisson and negative binomial families (integer,
inclusive bounds, the mass a sum of masses); each exact on both paths, with brms's truncated mean and draws, `log_lik()` (the
Gaussian's censored and truncated rows included, D123), `bayes_R2()`, `loo()` and `describe()`. Verified against
`weibull_censored`, `lognormal_censored`, `poisson_truncated`, `lognormal_censored_interval_weighted`,
`poisson_truncated_weighted_post`, `weibull_censored_interval_post` and `negbinomial_truncated_ub` (log densities within 1.3e-14
relative); the Gamma family's `cens()` is accepted on `gamma_censored_interval` against a 50-digit reference where brms's
gradient in the shape is off it (up to 2.8e-5, P6, D120); a count response's lower bound (brms's expected values are
`E[Y | Y >= lb + 1]` on `negbinomial_truncated`, D121) is implemented and withheld; the other families and contexts are refused naming the construct (D122); see
[censoring and truncation beyond the Gaussian](cens-trunc-families.md).
