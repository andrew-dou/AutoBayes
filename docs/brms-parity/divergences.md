# Known divergences from brms

Where AutoBayes knowingly differs from brms. A fixture is never edited to hide one of these; the entry says what
the oracle shows, why AutoBayes differs, and whether the brms behaviour is offered as an opt-in or parked.

| id | area | brms | AutoBayes | status |
|----|------|------|-----------|--------|
| D1 | priors | fills in default priors (`default_prior()`) for every class | requires every prior explicitly; a class the model lacks is refused | opt-in `ab.default_priors()` planned (A07); the oracle records brms's table for it |
| D2 | priors | an intercept-only model needs no coefficient prior | `y ~ 1 + (1 \| g)` still requires a (vacuous) `b` prior | to decide with A07 |
| D3 | factor levels | levels in R factor order (locale-dependent unless declared) | levels sorted as strings | handled: the oracle matches levels by content; cases declare levels in C-locale order |
| D4 | binomial | keeps rows with zero trials (they contribute no likelihood) | drops zero-trial rows before building the design and reports them | kept; cases avoid zero-trial rows |
| D5 | binomial predictions | `posterior_epred` is `trials * p` | `kind="mean"` returns `p` and `expected = trials * p`; the oracle compares `expected` | documented |
| D6 | new group levels | `sample_new_levels = "uncertainty"` by default | refuses by default (`new_groups="error"`); `resample_observed` and `population` are explicit | documented; stochastic, outside the gates |
| D7 | Bernoulli likelihood | `bernoulli_logit_lpmf` | the same density written with a gradient-stable softplus (not Bambi's floored form) | equal to 1e-8 relative on the fixtures |
| D8 | varying slopes | `y ~ (1 + x \|\| g)` adds a group-level slope on `x` without a population-level `x` | a varying slope must also be a numeric population-level term of the formula; `y ~ (1 + x \|\| g)` is refused with the form to write | kept (A01): the population-level slope is the slope's centre, and omitting it is a different model |
| D9 | Gamma link | `brmsfamily("Gamma")` defaults to the `inverse` link, `mu = 1 / eta` (`log` and `identity` on request) | `ab.gamma` has the `log` link only; the fixture `gamma_varint` was frozen with `link = "log"`, so the oracle compares like with like | kept (A09): `log` keeps `mu > 0` for every `eta` and puts the coefficients on the Poisson/negative binomial scale; the inverse link is not offered |
| D10 | offset | `offset()` takes any expression (`offset(log(x) + 1)`, `offset(2 * x)`) and several offset terms are summed | only `offset(col)` and `offset(log(col))`, one per formula; anything else is refused with the column to compute (in-formula transforms are A05) | kept (A06): every accepted form is checked against the data and against Bambi's evaluation; the fixture `poisson_offset` uses the accepted form |
