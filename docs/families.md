# Response families: the registry

Every family-specific decision in the library is a field of one `FamilySpec` (`autobayes/families/spec.py`).
Each family is one module in `autobayes/families/` that builds its spec; `autobayes/families/__init__.py`
registers them in `FAMILIES`, an immutable mapping from name to spec, and `get_family(name)` returns the spec
or raises `ValueError("family must be one of [...]")`. A built model carries the name as `model.family` and
the spec as `model.family_spec`. No module outside `autobayes/families/` compares a family name against a
string: `tests/test_family_registry.py` parses the package and fails on any such comparison, membership
test or `match`.

## What a spec holds

| Group | Fields | Gaussian | Student-t | Poisson | negative binomial | Gamma | lognormal | Beta | Bernoulli | binomial |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Names | `name`, `tag`, `parent`, `link`, `link_scale`, `mean_name`, `links`, `parked_links` | mu / identity / mu | mu / identity / mu | mu / log (log scale) / mu | mu / log (log scale) / mu | mu / log (log scale) / mu | mu / identity (log scale: mu is the location of log y) / mean | logit_mu / logit (logit scale) / mu | logit_p / logit (logit scale) / p; `links`: probit (probit_p / probit scale / p), cloglog (cloglog_p / cloglog scale / p); `parked_links`: none | logit_p / logit (logit scale) / p; `links`: probit, cloglog; `parked_links`: none |
| Response | `response_kind`, `response_dtype`, `validate_response`, `response_warnings` | continuous, numeric | continuous, numeric | count, int64, counts in [0, 1e6], all-zero warning | count, int64, counts in [0, 1e6], all-zero warning | continuous, numeric, finite and strictly positive | continuous, numeric, finite and strictly positive | continuous, numeric, strictly inside (0, 1) | binary, int64, strict 0/1 | binomial (trials), int64, `binomial_counts`, separation warning |
| Parameters | `scale_parameter` (brms name; prior required), `bambi_scale_parameter` (Bambi's/PyMC's name), `fixed_parameters` (user scalar with its check), `learnable_parameters` (fixed parameters a model may learn instead, with their lower bound) | sigma / sigma | sigma / sigma; nu > 1 fixed, or learned (`learnable_parameters`: nu > 1) | none | shape / alpha | shape / alpha | sigma / sigma | phi / phi | none | none |
| Graph | `bambi_family` | `"gaussian"` | `"t"` | `"poisson"` | `"negativebinomial"` | Bambi's `gamma` with `link = "log"` | `likelihoods.lognormal_family` (PyMC's `LogNormal`) | `likelihoods.beta_logit_family` (PyMC's `Beta` with alpha = expit(eta) phi, beta = expit(-eta) phi) | `likelihoods.bernoulli_logit_family`; under probit `likelihoods.bernoulli_link_family("probit")` | `likelihoods.binomial_logit_family`; under probit `likelihoods.binomial_link_family("probit")` |
| Prediction | `inverse_link`, `expected_response`, `draw_response`, `manual_prediction`, `prediction_kinds`, `response_upper_bound` | identity, Normal | identity, Student-t | exp with overflow check, Poisson, by hand | exp with overflow check, NegativeBinomial(mu, shape), by hand | exp with overflow check, Gamma(shape, shape / mu), by hand | identity; the mean exp(mu + sigma^2 / 2) from the sigma draws (`expected_response`, overflow check), LogNormal(mu, sigma), by hand | expit, Beta(mu phi, (1 - mu) phi), by hand | expit, Bernoulli, bound 1 | expit, Binomial, by hand, `probability` kind, bound = trials |
| Wording | `kind_error`, `scale_rule`, `scale_missing_error`, `linpred_symbol`, `scale_phrase`, `intercept_meaning`, `mean_lines`, `likelihood_lines`, `centred_intercept_note`, `deterministic_note` | the sentences `describe()`, `predict()`, `value_point` and `Priors.validate` print for that family | | | | | | | | |
| Default priors (A07) | `default_prior_link` (how brms reads the response for its data-dependent defaults), `default_scale_prior` (brms's default of the auxiliary parameter, given brms's scale), `default_learned_priors` | identity; sigma `HalfStudentT(3, scale)` | identity; sigma `HalfStudentT(3, scale)`; nu `Gamma(2, 0.1)` | log | log; shape `InverseGamma(0.4, 0.3)` | log; shape `Gamma(0.01, 0.01)` | log (brms's log-scale family); sigma `HalfStudentT(3, scale)` | none (logit); phi `Gamma(0.01, 0.01)` | none (logit, probit) | none (logit, probit) |
| Addition terms (B02) | `addition_terms` (the response addition terms besides `trials()` the likelihood takes), `tails` (the log-CDF, log-CCDF, log mass, truncated mean and draws that `cens()` and `trunc()` need) | weights, cens and trunc (the last two accepted against two references, P6); `likelihoods.NORMAL_TAILS` | weights | weights | weights | weights | weights | weights | weights | weights |
| Distributional formula (B01) | `distributional_link` (the link of a formula on `scale_parameter`, `["y ~ x", "sigma ~ z"]`; None: no formula), `default_distributional_intercept` (brms's default prior of that formula's intercept, given brms's scale of `log y`) | log; `student_t(3, 0, scale)` | log; `student_t(3, 0, scale)` | none | log (Bambi's `alpha`); `student_t(3, 0, 2.5)` | log (Bambi's `alpha`); `student_t(3, 0, 2.5)` | log; `student_t(3, 0, scale)` | log; `student_t(3, 0, 2.5)` | none | none |
| Monotonic effects (B03) | `monotonic` (the family's linear predictor takes brms's `mo(x)` terms, docs/monotonic.md; set where a frozen case verifies it) | yes | yes | yes | yes | yes | yes | yes | yes | yes |

A family with a `distributional_link` lets its auxiliary parameter have its own formula (docs/distributional.md): Bambi
builds it as a distributional component named `bambi_scale_parameter`, `model.names[scale_parameter]` is then the per-row
deterministic, and the families' mean and response functions read the parameter through `families.spec.auxiliary`, which
broadcasts a per-draw value and a per-row one alike; the zero-inflated and hurdle Poisson families have none (D33).

The derived properties `continuous`, `discrete` and `has_trials` read `response_kind`. Callable fields take
the built model (or the `ModelSpec` and frame being validated) because that is what the call sites have;
none is called before `Model.__init__` has resolved the spec. `expected_response` is `None` for every family whose
mean is `inverse_link(eta)`; the lognormal sets it because its mean `exp(mu + sigma^2 / 2)` needs the `sigma` draws,
and `draw_response` receives the linear predictor next to the mean for the same reason.

Where the library consults the spec: `Model.__init__` (fixed-parameter check, trials handling, response cast
and warnings, Bambi family, fixed parameters in the prior dict, post-build observed/trials checks),
`Model.names`, `Model.link`, `Model.value_point`, `Model.describe`, `formula.parse_formula` and
`formula.validate_data` (trials syntax, response validation), `Priors.validate` (scale prior),
`predict` (kinds, inverse link, response draws, by-hand routing, trials for `newdata`),
`prior_predictive` (count support), `identity.freeze_target` and `session.fit_record` (fixed parameters,
auxiliary prior, link scale, trials fields). Outside `Model`, the spec is looked up from `model.family` by name,
so an object that exposes the model's attributes still works.

A family with several links registers its default link's spec and carries the others in `links`, a mapping from link
name to a builder of the family's spec under that link (one cached spec per link, so `get_family("bernoulli", "probit")`
is always the same object); `FamilySpec.with_link(link)` resolves it, returning the spec itself for `None` or its own
link and refusing anything else by name, `get_family(name, link)` and `family_of(model)` look a model's spec up by
family and link, and `Model(..., link=...)` is the only place a link is chosen. `parked_links` names the links brms
offers for the family that this library withholds, with the reason the refusal carries (none since P6 lifted the
Bernoulli and binomial cloglog link's park, docs/links.md). Everything downstream reads the resolved spec, so a link is
never compared by name outside this package either.

The auxiliary parameter is one of `Priors`' fields `sigma`, `shape` or `phi` (`priors.AUXILIARY`), named by
`scale_parameter`; exactly the family's own must be given to `Priors` and to `Model.value_point`, and
`model.names[scale_parameter]` is its PyMC variable (`bambi_scale_parameter`: Bambi calls brms's `shape` `alpha`; the
Beta's `phi` keeps brms's name because its likelihood is this library's own Bambi family).

A fixed parameter may instead be learned: `learnable_parameters` maps it to its lower bound (the Student-t `nu`, 1,
brms's `real<lower=1> nu`). `Model(..., nu=None)` with `Priors(nu=...)` learns it under that prior truncated at the
bound and renormalised there (`Prior.to_bambi_truncated`, PyMC's `Truncated`, sampled as `log(nu - 1)`);
`model.learned_parameters` lists it, `model.names["nu"]` is its PyMC variable, `Model.value_point(nu=...)` takes it,
and the paths for fixed values (the prior dict, `family_parameters` in the target identity and the fit record) skip it,
while its prior is recorded only when present (docs/student-learned-nu.md).

The zero-inflated and hurdle families (B06, docs/zero-inflated-hurdle.md) have a second kind of auxiliary parameter,
`zero_parameter`: the probability of the zero part, brms's `zi` (zero-inflation) or `hu` (hurdle),
`real<lower=0,upper=1>` with its own prior on (0, 1) (`Priors(zi=...)` / `Priors(hu=...)`, an `ab.Beta`;
`priors.AUXILIARY` lists both). Exactly the family's own must be given to `Priors` and to `Model.value_point`;
`model.names[zero_parameter]` is its PyMC variable, which keeps brms's name and is sampled as `zi_logodds__` (Stan's
logit); `default_zero_prior` is brms's `beta(1, 1)`, written with `lb` 0 and `ub` 1. It sits beside `scale_parameter`,
so that the zero-inflated or hurdle form of a family with its own auxiliary parameter (the negative binomial's `shape`)
would keep both; every other family has `zero_parameter=None`. Their mean needs the `zi` / `hu` draws, so they set
`expected_response`, as the lognormal does for `sigma`.

| Group | zero-inflated Poisson | hurdle Poisson |
| --- | --- | --- |
| Names | `log_mu` / log (log scale) / `mean` | `log_mu` / log (log scale) / `mean` |
| Response | count, int64, counts in [0, 1e6], all-zero warning | count, int64, counts in [0, 1e6], all-zero warning |
| Parameters | `scale_parameter` none; `zero_parameter` zi | `scale_parameter` none; `zero_parameter` hu |
| Graph | `likelihoods.zero_inflated_poisson_family` (the linear predictor as the parent `log_mu` under the identity link; `zi` by brms's name, not PyMC's `psi = 1 - zi`) | `likelihoods.hurdle_poisson_family` (the same, with `hu`) |
| Prediction | `inverse_link` exp with overflow check (the Poisson part's mean); `expected_response` `(1 - zi) mu`; a structural zero below `zi`, else Poisson; by hand | `expected_response` `(1 - hu) mu / (1 - exp(-mu))`; zero below `hu`, else a zero-truncated Poisson; by hand |
| Default priors | log (zeros read as 0.1); zi `Beta(1, 1)` | log (zeros read as 0.1); hu `Beta(1, 1)` |
| Monotonic effects (B03) | yes | yes |

The ordinal and categorical families (B05, docs/ordinal-categorical.md) have a response of K categories, which
`FamilySpec.response_categories` turns into brms's `Y`, each row's 1-based category, keeping the labels as
`model.categories`; their `bambi_setup` builds what Bambi is handed from the model (the family, the prior dict, whether
Bambi centres the design) because their likelihoods need the model's categories, terms and column means, and their
`check_formula` refuses what they do not take. `thresholds` marks the cumulative family, whose intercept is brms's
vector of ordered thresholds `Intercept` (not part of the linear predictor `mu`), and `per_category` the categorical
family, whose every coefficient has a trailing axis over the non-reference categories; `intercept_lines` replaces
`describe()`'s paragraph on the intercept. `predict(kind="mean")` returns the probability of every category with a new
`category` dimension. Every other family has none of these.

| Group | cumulative | categorical |
| --- | --- | --- |
| Names | `mu` / logit (logit scale) / `p` | `mu` (one column per non-reference category) / logit (logit scale) / `p` |
| Response | ordinal, int64: positive integers `1..K` or an ordered pandas Categorical (`formula.ordinal_categories`) | categorical, int64: labels sorted as brms's `as.factor` (`formula.categorical_categories`), the first the reference |
| Parameters | no auxiliary parameter; the thresholds `Intercept` (centred design, Stan's ordered coordinates) and the deterministic `b_Intercept` | no auxiliary parameter; per-category intercepts and coefficients |
| Graph | `likelihoods.cumulative_logit_family` (built per model: `b_Intercept = Intercept + sum_p mean(x_p) b_p`), `likelihoods.ordered_thresholds`, `likelihoods.no_intercept`; Bambi's design uncentred | `likelihoods.categorical_logit_family` (Bambi's categorical family, the parent `mu` under the identity link) |
| Prediction | `expected_response`: the category probabilities from `b_Intercept` (`likelihoods.cumulative_probabilities`); 1-based category draws; by hand | `inverse_link`: the softmax of `(0, mu)`; 1-based category draws; by hand |
| Default priors | none (logit): `student_t(3, 0, 2.5)` on the thresholds, also uncentred | none (logit): `student_t(3, 0, 2.5)` on each category's intercept |
| Monotonic effects (B03) | yes (in `mu`, outside the thresholds' centring) | no: refused (D42) |

## Adding a family

1. Write `autobayes/families/<name>.py` building one `FamilySpec`; keep the response validation primitives in
   `autobayes/formula.py` and any hand-written likelihood in `autobayes/likelihoods.py`, and reference them
   from the spec (import the likelihood lazily inside `bambi_family`, as the Bernoulli and binomial modules do).
2. Register it in `FAMILIES` in `autobayes/families/__init__.py`; add its constructor to `autobayes/model.py`
   and `autobayes/__init__.py` with the same signature pattern as the existing ones. Record brms's default priors for
   it (`default_prior_link`, `default_scale_prior`, `default_learned_priors`, `default_zero_prior`; docs/default-priors.md), read from
   brms's `def_dpar_prior` and the family's own `prior` function; gate G6 checks them against the new fixture. A new fixed parameter
   needs a keyword on `Model.__init__`; a new auxiliary parameter needs a `Priors` field, an entry in
   `priors.AUXILIARY` and a keyword on `Model.value_point`.
3. Add its row to `EXPECTED` and `EXAMPLES` in `tests/test_family_registry.py`, its family tests, and an
   oracle case under `tests/oracle/cases/` with a frozen brms fixture (`oracle/README.md`).

What a spec does not decide: the formula grammar, the design construction, centring, group coding and the
sampler are shared by all families and live where they did.
