# Links: the probit and cloglog links for the Bernoulli and binomial families, and more links for the Tier A families

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
- Refused, naming what is accepted: `link="cauchit"` (not offered), and `link=` for every family with one link
  (`gaussian(link = "log")`, `lognormal(link = "log")`, ...; divergence D19); the families E06 gave more links (below) refuse
  the others by name (`poisson(link = "identity")`: "link must be one of ['log', 'sqrt'] for the poisson family").

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

## More links for the Tier A families (E06)

brms-parity E06 adds brms's links of five more families (`brmsfamily()`'s documentation lists them; the densities are those
of brms's vignette "Parameterization of Response Distributions in brms"):

| constructor | brms | mean | parent (the linear predictor) | brms's frozen case |
| --- | --- | --- | --- | --- |
| `ab.poisson(..., link="sqrt")` | `poisson("sqrt")` | `mu = eta^2`, any real `eta` | `sqrt_mu` | `poisson_sqrt` |
| `ab.negbinomial(..., link="identity")` | `negbinomial("identity")` | `mu = eta`, `eta > 0` | `mu` | `negbinomial_identity` |
| `ab.gamma(..., link="inverse")` | `Gamma("inverse")` | `mu = 1 / eta`, `eta > 0` | `inverse_mu` | `gamma_inverse` |
| `ab.gamma(..., link="identity")` | `Gamma("identity")` | `mu = eta`, `eta > 0` | `mu` | `gamma_identity` |
| `ab.beta(..., link="probit")` | `Beta("probit")` | `mu = Phi(eta)` | `probit_mu` | `beta_probit` |
| `ab.beta(..., link="cloglog")` | `Beta("cloglog")` | `mu = 1 - exp(-exp(eta))` | `cloglog_mu` | `beta_cloglog` |
| `ab.cumulative(..., link="cloglog")` | `cumulative("cloglog")` | `P(y <= k) = 1 - exp(-exp(tau_k - mu))` | `mu` | `cumulative_cloglog_narrow` (and `cumulative_cloglog` but for G2) |

Each spec is its family's under the link (`FamilySpec.links`: `get_family("gamma", "inverse")`, `model.link`, `family_of`),
built in the family's module (`poisson._sqrt`, `negbinomial._identity`, `gamma._link`, `beta._link`, `cumulative._cloglog`); the
families' default links are unchanged (the Gamma's stays `log`, D9). The graph keeps `eta` itself as Bambi's parent under the
identity link, so the deterministic the oracle compares with brms's `posterior_linpred` is the linear predictor (with the
offset, which the three identity and inverse cases use to keep `eta` positive). The likelihoods (`autobayes/likelihoods.py`):

- Poisson sqrt: `2 y log|eta| - eta^2 - log(y!)`, the first term 0 on `y = 0` rows; `-inf` at `eta = 0` for `y > 0`, as Stan's
  `poisson_lpmf(y | 0)`.
- Identity and inverse links: the Gamma density with the rate `shape eta` (inverse) or `shape / eta` (identity), and the
  negative binomial as the log link's stable form at `log(eta)`. A row with `eta <= 0` has no valid mean: its log density is
  `-inf` with gradient 0, where brms's Stan rejects the draw (brms's `log_prob` raises "Inverse scale parameter ... must be
  positive finite").
- Beta probit and cloglog: `mu` and `1 - mu` each computed directly (`Phi(eta)` and `Phi(-eta)`; `-expm1(-exp(eta))` and
  `exp(-exp(eta))`), never by subtraction, and the log-gamma of a shape parameter below `1e-10` from its logarithm, so the
  density is finite with a finite gradient for every finite `eta` on both paths. brms forms `1 - mu` by subtraction, which is
  exactly 0 above `eta = 8.25` (probit, as Stan's `Phi` is 1 there, D17) and about `3.6` (cloglog), where its `beta_lpdf`
  rejects, and loses `eps / (1 - mu)` relative before that (divergence D132).
- Cumulative cloglog: with `F(x) = 1 - exp(-exp(x))` and `d_k = tau_k - mu`, `log F(d_1)` for the first category, `-exp(d_{K-1})`
  for the last and, for a middle one, `-exp(d_{y-1}) + log1mexp(-t)` with `t = exp(d_y) - exp(d_{y-1})` formed as `exp(d_y)
  (-expm1(d_{y-1} - d_y))` (`likelihoods.cloglog_log_mass`), never a difference of probabilities.

`predict()`: `kind="linear"` is `eta` (brms's `posterior_linpred`, attribute `link`), `kind="mean"` the mean above as `mu`
(brms's `posterior_epred`), `kind="response"` the family's draws at that mean; both unseen-group policies work on the link's
scale. Under the identity and inverse links a draw and row whose linear predictor is not positive is refused, naming the link
and the rows, where brms's `posterior_epred` returns the non-positive mean unchecked and its `posterior_predict` draws NaN with
a warning (checked on brms 2.23 at a fixed-parameter point; divergence D134). `log_lik()`, `loo()` and `waic()` use the same
rows (`families.pointwise`); `prior_predictive()` draws as `predict(kind="response")` does, and under the identity and inverse
links refuses a prior draw with a non-positive mean, naming the link (the generic "prior predictive generation failed" error,
caused by it). `describe()` writes the link, the mean, the parent and the likelihood; the target identity and the fit record
carry `"link"`.

Default priors (`ab.default_priors(..., link=...)`, gate G6): brms reads the response through the link for the intercept's
location and scale, `sqrt(y)` under sqrt (zeros as they are), `1 / y` under inverse (zeros read as 0.1) and `y` under identity,
minus the mean offset; the probit and cloglog links give `student_t(3, 0, 2.5)` (checked against brms's `default_prior()` on
data with zeros, `tests/test_links_more.py`).

**What the links take.** Population- and group-level terms of the term model (numeric and categorical predictors,
interactions, transforms, varying intercepts and slopes, correlated or not, nested factors) and `offset()`, which enter the
linear predictor as under the family's own link; the frozen cases check `x`, `(1 | g)` and `offset()`. Refused under an E06
link, naming the construct and the link (`FamilySpec.check_formula`, `families.links.check_link_formula`; divergence D133):
`weights()` and the other addition terms, a formula on `shape` or `phi`, `mo()`, `gp()`, `mm()`, `gr(by = )`, `me()`, `mi()` and
autocorrelation terms, none of which a frozen case verifies with these links (each still works under the family's own link).
Multivariate, non-linear and mixture use stay refused for these families as before.

**The cumulative cloglog link** (`ab.cumulative(..., link="cloglog")`, divergences D38 and D132) is accepted on
`cumulative_cloglog_narrow`, the same model and data with its points drawn from `(-0.3, 0.3)`, where brms's arithmetic is exact
to the tolerances and every gate passes. It was withheld at first because on the wider case `cumulative_cloglog` (`ord ~ x + (1 | g)`, seed 832, points from `(-0.6, 0.6)`), gates G1, G3, G4, G5 and G6 pass
and the likelihood equals a 50-digit reference at all eight points (to 2.4e-16 relative; the gradient in `b` checked at points
1 and 7, within 1.4e-16). brms's own log density is off the reference by 5.9e-7 relative at point 7 (0-based; 1.45e-4 absolute)
and 6.4e-9 at point 1, and its gradient by up to 1.3e-4 relative (28 of the 176 entries with and without the Jacobian exceed
the oracle's 1e-7, all at points 1 and 7): brms forms the last category's probability
`1 - inv_cloglog(d)` by subtraction, and rows with `P(y = 4)` near `2e-13` lose their digits. A double-precision evaluation of
that subtraction reproduces brms's log density exactly. The two-reference rule (P6) cannot accept it, because log densities
never fall back to a reference, so the link stays withheld: a narrower box of points or a case whose last category is not
that improbable would need a new frozen case (`tests/test_links_more.py` checks the implementation against the frozen case
with a test-only patch).

**Evidence.** The six offered links pass gates G1-G6 against their frozen brms fixtures (design and offsets, log density and
gradient with and without the Jacobian, the prior part, the graph's linear predictor and group effects, `posterior_linpred` /
`posterior_epred` on fitted and new rows with and without group effects, the JAX density, the default-prior table) with no
reference: brms is within the tolerances at every entry (largest differences: log density 2.4e-12 relative and gradient 6.2e-9 on
`beta_probit`, whose linear predictor reaches 5.35 at the frozen points, where brms's `1 - Phi(eta)` keeps about eight digits;
`beta_cloglog`'s stays in (-1.7, 1.4); every other case within 1e-15 and 7e-10).
`tests/test_links_more.py` checks each likelihood row on both paths against references written from the densities (50-digit
Decimal for the cumulative cloglog mass and the Poisson sqrt row), the Beta rows past the underflow of `1 - mu`, the models'
densities against SciPy on both paths, predictions, the refusals, `log_lik()`, the defaults and the fit record;
`tests/test_post_processing.py` adds the six cases to its `log_lik()` cross-check. Nothing samples.

