# Response families: the registry

Every family-specific decision in the library is a field of one `FamilySpec` (`autobayes/families/spec.py`).
Each family is one module in `autobayes/families/` that builds its spec; `autobayes/families/__init__.py`
registers them in `FAMILIES`, an immutable mapping from name to spec, and `get_family(name)` returns the spec
or raises `ValueError("family must be one of [...]")`. A built model carries the name as `model.family` and
the spec as `model.family_spec`. No module outside `autobayes/families/` compares a family name against a
string: `tests/test_family_registry.py` parses the package and fails on any such comparison, membership
test or `match`.

## What a spec holds

| Group | Fields | Gaussian | Student-t | Poisson | Bernoulli | binomial |
| --- | --- | --- | --- | --- | --- | --- |
| Names | `name`, `tag`, `parent`, `link`, `link_scale`, `mean_name` | mu / identity / mu | mu / identity / mu | mu / log (log scale) / mu | logit_p / logit (logit scale) / p | logit_p / logit (logit scale) / p |
| Response | `response_kind`, `response_dtype`, `validate_response`, `response_warnings` | continuous, numeric | continuous, numeric | count, int64, counts in [0, 1e6], all-zero warning | binary, int64, strict 0/1 | binomial (trials), int64, `binomial_counts`, separation warning |
| Parameters | `scale_parameter` (prior required), `fixed_parameters` (user scalar with its check) | sigma | sigma; nu > 1 | none | none | none |
| Graph | `bambi_family` | `"gaussian"` | `"t"` | `"poisson"` | `likelihoods.bernoulli_logit_family` | `likelihoods.binomial_logit_family` |
| Prediction | `inverse_link`, `draw_response`, `manual_prediction`, `prediction_kinds`, `response_upper_bound` | identity, Normal | identity, Student-t | exp with overflow check, Poisson, by hand | expit, Bernoulli, bound 1 | expit, Binomial, by hand, `probability` kind, bound = trials |
| Wording | `kind_error`, `sigma_rule`, `sigma_missing_error`, `linpred_symbol`, `scale_phrase`, `intercept_meaning`, `mean_lines`, `likelihood_lines`, `centred_intercept_note`, `deterministic_note` | the sentences `describe()`, `predict()`, `value_point` and `Priors.validate` print for that family | | | | |

The derived properties `continuous`, `discrete` and `has_trials` read `response_kind`. Callable fields take
the built model (or the `ModelSpec` and frame being validated) because that is what the call sites have;
none is called before `Model.__init__` has resolved the spec.

Where the library consults the spec: `Model.__init__` (fixed-parameter check, trials handling, response cast
and warnings, Bambi family, fixed parameters in the prior dict, post-build observed/trials checks),
`Model.names`, `Model.link`, `Model.value_point`, `Model.describe`, `formula.parse_formula` and
`formula.validate_data` (trials syntax, response validation), `Priors.validate` (scale prior),
`predict` (kinds, inverse link, response draws, by-hand routing, trials for `newdata`),
`prior_predictive` (count support), `identity.freeze_target` and `session.fit_record` (fixed parameters,
link scale, trials fields). Outside `Model`, the spec is looked up from `model.family` by name, so an object
that exposes the model's attributes still works.

## Adding a family

1. Write `autobayes/families/<name>.py` building one `FamilySpec`; keep the response validation primitives in
   `autobayes/formula.py` and any hand-written likelihood in `autobayes/likelihoods.py`, and reference them
   from the spec (import the likelihood lazily inside `bambi_family`, as the Bernoulli and binomial modules do).
2. Register it in `FAMILIES` in `autobayes/families/__init__.py`; add its constructor to `autobayes/model.py`
   and `autobayes/__init__.py` with the same signature pattern as the existing five. A new fixed parameter
   needs a keyword on `Model.__init__`; a new auxiliary parameter needs a `Priors` class.
3. Add its row to `EXPECTED` and `EXAMPLES` in `tests/test_family_registry.py`, its family tests, and an
   oracle case under `tests/oracle/cases/` with a frozen brms fixture (`oracle/README.md`).

What a spec does not decide: the formula grammar, the design construction, centring, group coding and the
sampler are shared by all families and live where they did.
