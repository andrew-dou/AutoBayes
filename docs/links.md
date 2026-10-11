# The probit and cloglog links for the Bernoulli and binomial families

brms-parity features A12 (`bernoulli(link = "probit")`, `binomial(link = "probit")`) and P6
(`bernoulli(link = "cloglog")`, `binomial(link = "cloglog")`, accepted against two references; below).

```python
import autobayes as ab
model = ab.bernoulli("y ~ x + (1 | group)", data,
    ab.Priors(intercept=ab.Normal(0, 2.5), b=ab.Normal(0, 1.5), sd=ab.HalfNormal(1)), link="probit")
print(model.describe())
```

## The model

`P(y_i = 1) = p_i = Phi(eta_i)`, the standard Normal CDF of the linear predictor, in place of `expit(eta_i)`; for the
binomial family `k_i ~ Binomial(n_i, Phi(eta_i))`. The coefficients act on the probit scale: the centred intercept
`a_c` is `Phi^-1` of the probability at the predictor means, and with categorical predictors `Phi(a_c)` is the
probability at the proportion-weighted predictor, not the average over levels. No parameter is added, so `Priors`,
`value_point` and the oracle adapter are unchanged; the link is part of the target identity.

**Graph.** A family with several links registers its default link's spec and reaches the others through
`FamilySpec.links` (`autobayes/families/links.py`, docs/families.md): `get_family("bernoulli", "probit")` is the
Bernoulli spec under the probit link, `Model(..., link=...)` resolves it with `FamilySpec.with_link`, `model.link`
reports it and `family_of(model)` looks it up again from `model.family` and `model.link`. Under every link the
likelihood is this library's own Bambi family (`likelihoods.bernoulli_link_family("probit")`,
`binomial_link_family("probit")`) with the identity link on the parent `probit_p`, so the deterministic in the graph
is `eta` itself, which is what the oracle compares with brms's `posterior_linpred`. The log probabilities are
`log P(y = 1) = log Phi(eta)` and `log P(y = 0) = log Phi(-eta)` through `likelihoods.log_normal_cdf`: `log1p(-erfc(z /
sqrt 2) / 2)` for `z >= -1`, `log(erfc(-z / sqrt 2) / 2)` for `-20 <= z < -1`, and the Mills-ratio asymptotic series
below `-20`, every branch evaluated on its argument clipped into the branch's own domain so that the branches a switch
does not take are finite too and their zero-weighted gradients never form `0 * inf`. The ops (`erfc`, `log1p`, `exp`,
`switch`, `clip`) have both C code and a JAX lowering; `erfcx`, which PyMC's `normal_lcdf` uses, has neither on this
stack. brms's `bernoulli_lpmf(Y | Phi(eta))` is `-inf` where Stan's `Phi` is exactly 0 (`eta < -37.5`) or exactly 1
(`eta > 8.25`); this form is finite with a finite gradient for every finite `eta` (divergence D17).

## Using it

- `ab.bernoulli(..., link="probit")`, `ab.binomial(..., link="probit")`, or `ab.Model(..., family=..., link="probit")`;
  `link` is `"logit"` by default. `model.link == "probit"`, `model.names["linpred"] == "probit_p"`,
  `model.family_spec is get_family(family, "probit")`.
- Priors: the intercept and slope priors are on the probit scale; `describe()` says so, and `sigma`, `shape` or `phi`
  is refused with "Its intercept and slope priors are on the probit scale".
- `predict()`: `kind="linear"` is `eta` (brms `posterior_linpred`, attribute `link = "probit"`), `kind="mean"` is
  `p = Phi(eta)` (brms `posterior_epred`; for the binomial family `p` with `expected = n p`, and `kind="probability"`),
  `kind="response"` draws `Bernoulli(p)` or `Binomial(n, p)`. Both unseen-group policies draw or reuse effects on the
  probit scale. `prior_predictive()` works; `fit_record` records `"link": "probit"`.
- `link="cloglog"` likewise (below): `model.names["linpred"] == "cloglog_p"`, `p = 1 - exp(-exp(eta))`, the
  coefficients on the complementary log-log scale.
- Refused, naming what is accepted: `link="cauchit"` (not offered), and `link=` for every other family
  (`gaussian(link = "log")`, `poisson(link = "identity")`, `Beta(link = "probit")`, ...; divergence D19).

## The cloglog link (P6)

brms's `bernoulli(link = "cloglog")` is `bernoulli_lpmf(Y | inv_cloglog(eta))` with `inv_cloglog(eta) = 1 - exp(-exp(eta))`
(`binomial_lpmf` likewise). Here `log P(y = 1) = log(1 - exp(-exp(eta)))` and `log P(y = 0) = -exp(eta)`:
`likelihoods.cloglog_log_p1` (`log1mexp(-exp(eta))` for `-20 <= eta <= 40`, `eta + log1p(-exp(eta) / 2)` below) and
`likelihoods.cloglog_log_p0` (`-exp(min(eta, 700))`), finite with a finite gradient for every finite `eta` on both
paths, with `links.inverse_cloglog` (`-expm1(-exp(eta))`) for `predict()`; the graph keeps `eta` as `cloglog_p`. The
coefficients act on the complementary log-log scale, `log(-log(1 - p))`. brms's gradient for a `y = 0` (or `k < n`) row
forms `1 - theta` by subtraction, which loses `eps * exp(exp(eta))` relative: about `1e-7` at `eta = 3`, everything at
`eta` above about 3.6, where brms's density is `-inf` (divergence D18).

**How it is certified.** The link was parked (A12) because the frozen fixture `bernoulli_cloglog` (`yb ~ x + (1 | g)`,
seed 153) records such a point: at seeded point 5 (0-based: the sixth of eight), rows 29 and 34 have `y = 0` at
`eta = 2.71` and `3.01`, and brms's gradient with respect to `b[1]` is `-1.00559172179031` where the exact value is
`-1.00559210649608`, `3.8e-7` relative against `GRAD_REL` of `1e-7`. Since P6 the case opts in to two-reference
acceptance (docs/brms-parity/LOOP.md, "Two references"): its gradient entries are compared with brms where brms is
accurate and with the committed 50-digit reference `tests/oracle/references/bernoulli_cloglog.json` (`oracle/reference.py`)
where it is not. Exactly 2 of the 144 gradient entries pass on the reference (`b[1]` at point 5, with and without the
Jacobian); there brms is `3.8e-7` off the reference and AutoBayes `7e-15`, and at every entry AutoBayes is within
`7.3e-15` of the reference (`5.7e-14` absolute). Every other gate compares with brms only and passes; brms's log
densities are within `1.9e-10` of the reference's.

`binomial_cloglog` (`k | trials(size) ~ x + (1 | g)`, seed 156) is frozen with P6. At the generator's default box of
seeded points, `(-1.2, 1.2)` in every unconstrained coordinate, brms's log density is `-inf` at a seeded point (a
`k < size` row at `eta = 3.95`, where Stan's `1 - inv_cloglog(eta)` is exactly 0; this is why A12 could not freeze it),
and at `(-1.1, 1.1)` its log likelihood is `3.7e-6` relative off the exact value, which no gate accepts (log densities
never fall back); the case draws its points from `(-1, 1)` (`point_halfwidth`, the largest box on a 0.1 grid where
brms's log likelihood is finite and within `LOGP_REL` of the exact value). There brms is within the tolerance at every entry (its gradient within `6.3e-9` of the
reference, its density within `8.0e-11`), so the case passes every gate against brms alone and has no opt-in.

## Evidence

Oracle cases `bernoulli_probit` (`yb ~ x + (1 | g)`), `binomial_probit` (`k | trials(size) ~ x + (1 | g)`) and
`binomial_cloglog` pass gates G1-G6 against frozen brms fixtures, and `bernoulli_cloglog` against brms and, for 2
gradient entries, its 50-digit reference: design, log density and gradient with and without the Jacobian, the prior
part, the graph's linear predictor (brms's `posterior_linpred`) and group effects, `posterior_linpred` /
`posterior_epred` on fitted and new rows with and without group effects, the JAX density and the default-prior table.
`tests/test_links.py` checks the refusals and the registry, the density and gradient on both paths against scipy's
`log_ndtr` and the analytic Mills ratios at linear predictors up to `|eta| = 300` and counts up to the validated cap of
`1e6`, `value_point`, seeded predictions on fitted and new rows under both unseen-group policies, `describe()`,
prior-predictive support, the fit record and the target identity; and for the cloglog link the numerics against a
50-digit Decimal reference on `|eta| <= 40` on both paths, and models, `describe()` and predictions. Nothing samples.
