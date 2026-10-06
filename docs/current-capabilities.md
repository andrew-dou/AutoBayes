# Current library capabilities — September 27, 2026

Snapshot of merged main `005ff442e9ff02edcf5bf6769dffa9cda2053caf` (PR17).
This is a capability map, not a percentage of brms implemented.

| Surface | Merged behavior | Tests / documentation |
| --- | --- | --- |
| Families | Gaussian, Bernoulli-logit, grouped binomial-logit, fixed finite nu>1 Student-t, Poisson-log | test_logp.py, test_bernoulli.py, test_binomial.py, test_student.py, test_poisson.py |
| Formula composition | Numeric and declared categorical predictors; multiple numeric-by-categorical interactions with required main effects; optional intercept | test_intercept_controls.py, test_multiple_interactions.py |
| Hierarchy | Population-only models or multiple independent varying-intercept factors | test_population_only.py, test_multiple_groups.py |
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
is n*p, distinct from probability p. Poisson has no exposure/offset interface yet.
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
formulas, learned Student-t degrees of freedom, offsets/exposure, censoring and
truncation remain separate work. Independent varying slopes remain in open PR8
with an unresolved scientific acceptance gate; correlated slopes are not merged.
Standalone generated source-code export is not implemented. The library builds
an inspectable PyMC graph and executes through the selected sampler backend.

The next useful feature design is a prior-predictive interface with explicit
prior/response semantics, preserved row and group labels, and reproducible seeds.
Its API, numerical-domain checks and bounded tests must be decided before code;
this documentation update does not launch sampling or a prior-predictive campaign.

Prior-predictive modeled-row CPU draws are now available; see [prior predictive](prior-predictive.md). The snapshot and historical counts above refer to PR17, not this addition.
