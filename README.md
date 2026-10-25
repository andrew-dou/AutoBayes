# AutoBayes

brms in Python. You write a brms formula, family and priors; AutoBayes builds the same model in PyMC (through Bambi
where Bambi covers it), samples it with NUTS, on the CPU or through JAX/NumPyro on a CPU or GPU, and gives back brms's
post-processing: expected values and predictions, `log_lik`, PSIS-LOO, WAIC, `bayes_R2`, `hypothesis()` and
conditional effects.

"The same model" is checked against brms 2.23 itself. A feature is accepted only when it reproduces, on frozen brms
fixtures, `standata()`'s design matrices, the log density and gradient at seeded points, the derived quantities, and
`posterior_linpred` / `posterior_epred`. Every known difference from brms is a numbered row in
[divergences](docs/brms-parity/divergences.md).

## Install

Python 3.11. The tested stack is pinned in `pyproject.toml` and frozen in `uv.lock` (PyMC 5.28.5, PyTensor 2.38.3, Bambi
0.17.2, ArviZ 0.23.4, JAX 0.10.2, NumPyro 0.21.0):

```bash
git clone https://github.com/andrew-dou/AutoBayes && cd AutoBayes
uv sync --locked --extra jax --extra dev
uv run --no-sync pytest tests -q -m "not slow and not sampling"   # the deterministic suite, including the brms oracle
```

If `~/.pytensor` is not writable, set `PYTENSOR_FLAGS=base_compiledir=<writable dir>`. R is needed only to freeze new
oracle fixtures ([oracle](oracle/README.md)).

## Quick start

```python
import autobayes as ab

# brms: brm(y ~ x + (1 + x | g), family = bernoulli(), prior = ...)
priors = ab.Priors(intercept=ab.Normal(0, 2.5), b=ab.Normal(0, 1),
                   sd={"g": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(1)}}, cor={"g": ab.LKJ(2)})
model = ab.bernoulli("y ~ x + (1 + x | g)", df, priors)
print(model.describe())                         # the density that is sampled
fit = ab.fit(model, backend="numpyro", chains=4, seed=1)    # device="gpu" for a GPU

model.predict(fit, kind="mean")                 # posterior_epred
ab.loo(model, fit)
ab.hypothesis(model, fit, "x > 0")
ab.conditional_effects(model, fit, ["x"])

# brms's default priors instead of explicit ones
f = "rating ~ x + (1 | g)"
ordinal = ab.cumulative(f, df, ab.default_priors(f, df, family="cumulative"))
```

Priors are explicit unless you ask for brms's defaults. Anything not supported or not yet verified against brms is
refused with an error naming the construct.

## What it covers

All 45 response families brms 2.23 documents, and most of brms's formula language:

| Area | What you can write | Docs |
|---|---|---|
| Families | Gaussian, Student-t (fixed or learned `nu`), Poisson (log, sqrt), negative binomial (`negbinomial`, `negbinomial2`), geometric, Gamma (log, inverse, identity), lognormal, Beta (logit, probit, cloglog), Bernoulli and binomial (logit, probit, cloglog), beta-binomial, Weibull, exponential, skew-normal, shifted lognormal, ex-Gaussian, inverse Gaussian, von Mises, asymmetric Laplace, Frechet, generalized extreme value, `xbeta`, the Wiener diffusion model (`rt \| dec(dec)`), the Cox proportional hazards model (`time \| cens(c)`, `bhaz()`) | [families](docs/families.md), [continuous](docs/continuous-families.md), [discrete](docs/discrete-families.md), [bounded](docs/bounded-families.md), [links](docs/links.md), [Wiener](docs/wiener.md), [Cox](docs/cox.md) |
| Zero-inflated and hurdle | zero-inflated Poisson, negative binomial, binomial, beta-binomial and Beta; zero-one-inflated Beta; hurdle Poisson, negative binomial, Gamma, lognormal and cumulative | [zero-inflated and hurdle](docs/zero-inflated-hurdle.md), [bounded](docs/bounded-families.md) |
| Ordinal and categorical | `cumulative` (logit, probit, cloglog), grouped thresholds `thres(gr = z)`, `sratio`, `cratio`, `acat`, `hurdle_cumulative`, `cs(x)`, `categorical`; counts of K categories `cbind(...) \| trials(n)` with `multinomial` and `dirichlet_multinomial`; proportions of K categories `cbind(...)` with `dirichlet` and `logistic_normal` | [ordinal and categorical](docs/ordinal-categorical.md), [multinomial](docs/multinomial.md), [grouped thresholds](docs/grouped-thresholds.md) |
| Population-level terms | numeric and categorical predictors, ordered factors, interactions, `log()`, `exp()`, `sqrt()`, `I(...)`, `offset()`, `mo(x)` | [interactions](docs/interactions.md), [transforms](docs/transforms.md), [offset](docs/offset.md), [monotonic](docs/monotonic.md), [ordered factors](docs/ordered-factors.md) |
| Group-level terms | `(1 \| g)`, `(1 + x \|\| g)`, `(1 + x \| g)` with LKJ, `g:h`, `g/h`, `mm(g, g2)`, `gr(g, by = )`, shared correlation IDs | [varying slopes](docs/varying-slopes.md), [correlated slopes](docs/correlated-slopes.md), [grouping syntax](docs/grouping-syntax.md), [multi-membership and by](docs/multi-membership-and-by.md) |
| Other terms | `gp(x)` (exact and Hilbert-space), `ar()`, `ma()`, `arma()`, `fcor(M)`, `unstr()`, `cosy()`, `me(x, sdx)`, `mi(x)`, `car(W, gr, type)`, `sar(M, type)` | [Gaussian processes](docs/gaussian-processes.md), [autocorrelation](docs/autocorrelation.md), [fcor](docs/fcor.md), [unstr and cosy](docs/unstr-cosy.md), [missing values and measurement error](docs/missing-and-measurement-error.md), [spatial CAR](docs/spatial-car.md), [spatial SAR](docs/spatial-sar.md) |
| Model structure | distributional formulas (`sigma ~ z`; `shape`, `phi`, `zi`, `hu`, `alpha`), multivariate Gaussian models with residual correlation, non-linear formulas (`nl = TRUE`), finite mixtures | [distributional](docs/distributional.md), [contexts of the newer families](docs/contexts-newer-families.md), [multivariate](docs/multivariate.md), [non-linear](docs/nonlinear.md), [mixtures](docs/mixtures.md) |
| Addition terms | `weights()`; `cens()`, including `cens(c, y2)`, for the Gaussian, Weibull, lognormal and Gamma families, and right censoring with `bhaz()` for `cox`; `trunc()` for the Gaussian family and `trunc(ub = )` for counts; `se()` (Gaussian, Student-t) and `rate()` (Poisson, negative binomial); `subset()` in a multivariate model with a formula per response | [addition terms](docs/addition-terms.md), [censoring and truncation](docs/cens-trunc-families.md), [se() and rate()](docs/se-rate.md), [subset()](docs/subset.md) |
| Priors | explicit priors on every class; brms's defaults with `ab.default_priors()`; `horseshoe()` and `R2D2()` on `b` | [default priors](docs/default-priors.md), [shrinkage priors](docs/shrinkage-priors.md) |
| Post-processing | `predict()` (`posterior_linpred`, `posterior_epred`, `posterior_predict`), `log_lik`, `loo`, `waic`, `loo_compare`, `model_weights`, `bayes_R2`, `hypothesis` (with Savage-Dickey evidence ratios), `conditional_effects`; `fixef`, `ranef`, `coef`, `VarCorr`, `posterior_summary`, `summary` (Rhat, bulk and tail ESS), `residuals` | [post-processing](docs/post-processing.md), [prediction](docs/linear-prediction.md), [summaries](docs/summaries.md) |

Not yet: `s()` and `t2()`; K-fold cross-validation and stacking weights.
Combinations no frozen brms case
checks yet are refused, each naming its divergence row. The full map is
[current capabilities](docs/current-capabilities.md).

## Speed versus brms

18 models, each fitted by both libraries with the same formula, family, priors, data and sampler settings (four
parallel chains, 1000 warm-up and 1000 kept draws, target acceptance 0.9). brms 2.23.0 used rstan 2.32.7 and AutoBayes
used NumPyro on the CPU, on an Apple M1 Pro laptop, one run each (25 October 2026). Times are seconds; "sampler" is the
samplers' execution alone, compilation excluded; "gradient" is the cost of one log-density gradient in brms over
AutoBayes, measured inside each sampler.

| Model | Rows | brms (s) | AutoBayes (s) | Faster | Sampler | Gradient |
|---|---:|---:|---:|---:|---:|---:|
| Bernoulli, `(1 + x \| g)` | 2,000 | 36.8 | 7.0 | 5.3× | 1.8× | 1.9× |
| Gaussian, `(1 + x \| g)` | 20,000 | 105.9 | 30.5 | 3.5× | 4.3× | 7.2× |
| cumulative, five categories | 1,500 | 58.9 | 6.7 | 8.8× | 6.6× | 7.4× |
| zero-inflated Poisson | 2,000 | 40.6 | 6.0 | 6.8× | 3.6× | 3.2× |
| Weibull | 2,000 | 32.7 | 6.0 | 5.5× | 1.4× | 1.5× |
| zero-inflated negative binomial | 2,000 | 42.7 | 15.2 | 2.8× | 1.3× | 1.3× |
| sratio, five categories | 1,500 | 43.5 | 7.1 | 6.1× | 3.1× | 3.2× |
| Gaussian, horseshoe on 20 predictors | 400 | 27.9 | 5.1 | 5.5× | 0.6× | 0.4× |
| beta-binomial with trials | 2,000 | 49.1 | 22.6 | 2.2× | 1.2× | 1.7× |
| Gaussian, `(1 \| mm(g1, g2))` | 1,000 | 27.6 | 1.9 | 14.4× | 4.9× | 9.7× |
| mixture of two Gaussians | 1,000 | 31.5 | 3.1 | 10.1× | 4.0× | 6.3× |
| exact GP, `gp(x)` | 200 | 497.6 | 619.8 | 0.8× | 0.8× | 0.9× |
| AR(1) panel | 2,000 | 28.3 | 1.6 | 18.1× | 9.8× | 12.1× |
| non-linear, `a * exp(b * x)` | 1,000 | 31.6 | 2.2 | 14.3× | 6.9× | 11.6× |
| `mvbind(y1, y2)` with residual correlation | 1,000 | 43.8 | 3.6 | 12.2× | 13.2× | 17.7× |
| monotonic `mo(edu)` | 1,500 | 35.7 | 3.4 | 10.6× | 6.6× | 8.2× |
| Weibull with `cens()` | 2,000 | 35.6 | 6.8 | 5.2× | 1.1× | 1.1× |
| distributional, `sigma ~ z` | 2,000 | 36.2 | 3.7 | 9.9× | 5.4× | 6.1× |

- **End to end** AutoBayes was faster on 17 of the 18 models: 2.2 to 18.1 times (median 6.5). Most of the gap on
  small models is brms's Stan compile, 25 to 31 seconds per model, against 1 to 2 seconds of JAX compilation.
- **Where the sampling speed comes from:** both samplers take the same number of leapfrog steps (within 5% on 15 models),
  so the difference is the cost of each step. JAX's compiled gradient is 1.1 to 17.7 times cheaper than Stan's on 16
  models. On the horseshoe model a gradient costs microseconds and JAX's per-step overhead dominates; on the exact GP
  (200 rows, a dense Cholesky per step) brms is faster.
- **The posteriors agree:** every compared mean is within 2.71 combined Monte Carlo standard errors.

Caveats: brms can refit new data without recompiling, so for many fits of one model only the sampler column matters.
cmdstanr and GPUs were not tested. These are single runs on a laptop in use. Write-up:
[benchmark-2026-10-25.pdf](docs/brms-parity/reports/benchmark-2026-10-25.pdf) (the 17 October run of the first ten
models: [benchmark-2026-10-17.pdf](docs/brms-parity/reports/benchmark-2026-10-17.pdf)); scripts:
[examples/brms_parity](examples/brms_parity/README.md).

## How it is checked

Each brms feature gets frozen cases, built by an oracle-gated loop ([loop rules](docs/brms-parity/LOOP.md)). A case is a
formula, family, priors and simulated data. brms 2.23.0 with rstan 2.32.7 turns it into a fixture
(`oracle/R/gen_fixture.R`): the Stan data, eight seeded points with the log density and gradient, derived quantities,
and predictions.

A feature merges only when every case passes six gates:

| Gate | Checks | Tolerance |
|---|---|---|
| G1 | design matrices | 1e-12 |
| G2 | log density and gradient | relative 1e-8 and 1e-7 |
| G3 | derived quantities | relative 1e-9 |
| G4 | predictions | relative 1e-9 |
| G5 | JAX density against PyTensor | relative 1e-8 |
| G6 | brms's default-prior table | string for string |

Tolerances are never loosened, fixtures never change once frozen, and `python oracle/regen.py check --all` regenerates
every fixture from R and compares it with the committed one. In a few places brms's own arithmetic is inexact (the
cloglog link, `cens()`, `trunc()`, the cumulative probit gradient). There, a case may instead be accepted against a
committed 50-digit reference, but only at the entries where brms is off.

The gates never sample. Sampler convergence is checked by the sampling tests and the benchmark.

## Documentation

- [Current capabilities](docs/current-capabilities.md): every supported construct and what checks it.
- [Divergences from brms](docs/brms-parity/divergences.md): every known difference, with the evidence.
- Tier reviews: [A](docs/brms-parity/reports/tier-a-review.pdf), [B](docs/brms-parity/reports/tier-b-review.pdf),
  [C](docs/brms-parity/reports/tier-c-review.pdf), [D](docs/brms-parity/reports/tier-d-review.pdf),
  [E](docs/brms-parity/reports/tier-e-review.pdf); the
  [status report](docs/brms-parity/reports/autobayes-status.pdf).
- [Examples](examples/brms_parity/README.md): a tour of model classes and the brms benchmark.
- The feature ledger, [ledger.json](docs/brms-parity/ledger.json) (`python tools/parity_status.py`).
- [Library history](docs/history/milestones.md): the milestones before the brms-parity loop, with their decision
  records and evidence under `docs/history/`.
- The agent scaffold and reference pack: [autobayes-scaffold](https://github.com/andrew-dou/autobayes-scaffold).

## Limits

- Beyond the benchmark models, the features added in October 2026 have not been fitted.
- GPU sampling has been validated for the first milestones only.
- Hosted CI is not running; every merge since 11 October 2026 was verified locally.

## Origins

AutoBayes began as a skill and pre-research pack for running a long Bayesian model-iteration program with a coding
agent in the loop. That material (`SKILL.md`, `skills/`, `reference/`, `scaffold/` and their description) is now its own
repository, [autobayes-scaffold](https://github.com/andrew-dou/autobayes-scaffold), taken from this one at commit
`f5ee3df`; its history stays in this repository's git log. The library's record before the brms-parity loop is
[library milestones](docs/history/milestones.md).

## Licence

MIT.
