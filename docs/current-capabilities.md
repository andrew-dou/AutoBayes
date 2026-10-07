# Current library capabilities — September 27, 2026

Snapshot of merged main `005ff442e9ff02edcf5bf6769dffa9cda2053caf` (PR17).
This is a capability map, not a percentage of brms implemented.

| Surface | Merged behavior | Tests / documentation |
| --- | --- | --- |
| Families | Gaussian, Bernoulli-logit, grouped binomial-logit, fixed finite nu>1 Student-t, Poisson-log, negative binomial-log and Gamma-log with a `shape` prior | test_logp.py, test_bernoulli.py, test_binomial.py, test_student.py, test_poisson.py, test_negbinomial.py, test_gamma.py; [negative binomial](negbinomial.md), [Gamma](gamma.md) |
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
`logit_mu` and brms's name `phi`; `1 - mu` as `expit(-eta)`, divergence D10) followed, verified against `beta_varint`;
see [Beta](beta.md).
