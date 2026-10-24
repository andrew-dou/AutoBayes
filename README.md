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

39 of the 45 response families brms 2.23 documents, and most of brms's formula language:

| Area | What you can write | Docs |
|---|---|---|
| Families | Gaussian, Student-t (fixed or learned `nu`), Poisson (log, sqrt), negative binomial (`negbinomial`, `negbinomial2`), geometric, Gamma (log, inverse, identity), lognormal, Beta (logit, probit, cloglog), Bernoulli and binomial (logit, probit, cloglog), beta-binomial, Weibull, exponential, skew-normal, shifted lognormal, ex-Gaussian, inverse Gaussian, von Mises, asymmetric Laplace, Frechet, generalized extreme value, `xbeta` | [families](docs/families.md), [continuous](docs/continuous-families.md), [discrete](docs/discrete-families.md), [bounded](docs/bounded-families.md), [links](docs/links.md) |
| Zero-inflated and hurdle | zero-inflated Poisson, negative binomial, binomial, beta-binomial and Beta; zero-one-inflated Beta; hurdle Poisson, negative binomial, Gamma, lognormal and cumulative | [zero-inflated and hurdle](docs/zero-inflated-hurdle.md), [bounded](docs/bounded-families.md) |
| Ordinal and categorical | `cumulative` (logit, probit, cloglog), `sratio`, `cratio`, `acat`, `hurdle_cumulative`, `cs(x)`, `categorical` | [ordinal and categorical](docs/ordinal-categorical.md) |
| Population-level terms | numeric and categorical predictors, interactions, `log()`, `exp()`, `sqrt()`, `I(...)`, `offset()`, `mo(x)` | [interactions](docs/interactions.md), [transforms](docs/transforms.md), [offset](docs/offset.md), [monotonic](docs/monotonic.md) |
| Group-level terms | `(1 \| g)`, `(1 + x \|\| g)`, `(1 + x \| g)` with LKJ, `g:h`, `g/h`, `mm(g, g2)`, `gr(g, by = )`, shared correlation IDs | [varying slopes](docs/varying-slopes.md), [correlated slopes](docs/correlated-slopes.md), [grouping syntax](docs/grouping-syntax.md), [multi-membership and by](docs/multi-membership-and-by.md) |
| Other terms | `gp(x)` (exact and Hilbert-space), `ar()`, `ma()`, `arma()`, `me(x, sdx)`, `mi(x)` | [Gaussian processes](docs/gaussian-processes.md), [autocorrelation](docs/autocorrelation.md), [missing values and measurement error](docs/missing-and-measurement-error.md) |
| Model structure | distributional formulas (`sigma ~ z`; `shape`, `phi`, `zi`, `hu`, `alpha`), multivariate Gaussian models with residual correlation, non-linear formulas (`nl = TRUE`), finite mixtures | [distributional](docs/distributional.md), [contexts of the newer families](docs/contexts-newer-families.md), [multivariate](docs/multivariate.md), [non-linear](docs/nonlinear.md), [mixtures](docs/mixtures.md) |
| Addition terms | `weights()`; `cens()`, including `cens(c, y2)`, for the Gaussian, Weibull, lognormal and Gamma families; `trunc()` for the Gaussian family and `trunc(ub = )` for counts | [addition terms](docs/addition-terms.md), [censoring and truncation](docs/cens-trunc-families.md) |
| Priors | explicit priors on every class; brms's defaults with `ab.default_priors()`; `horseshoe()` and `R2D2()` on `b` | [default priors](docs/default-priors.md), [shrinkage priors](docs/shrinkage-priors.md) |
| Post-processing | `predict()` (`posterior_linpred`, `posterior_epred`, `posterior_predict`), `log_lik`, `loo`, `waic`, `loo_compare`, `model_weights`, `bayes_R2`, `hypothesis` (with Savage-Dickey evidence ratios), `conditional_effects` | [post-processing](docs/post-processing.md), [prediction](docs/linear-prediction.md) |

Not yet: `s()` and `t2()`; `multinomial`, `dirichlet`, `dirichlet_multinomial`, `logistic_normal`, `cox` and `wiener`;
`car`, `sar`, `fcor`, `unstr` and `cosy`; K-fold cross-validation and stacking weights. Combinations no frozen brms case
checks yet are refused, each naming its divergence row. The full map is
[current capabilities](docs/current-capabilities.md).

## Speed versus brms

Ten models, each fitted by both libraries with the same formula, family, priors, data and sampler settings (four
parallel chains, 1000 warm-up and 1000 kept draws, target acceptance 0.9). brms 2.23.0 used rstan 2.32.7 and AutoBayes
used NumPyro on the CPU, on an Apple M1 Pro laptop, one run each (17 October 2026):

| Model | Rows | brms: compile + sample (s) | AutoBayes (s) | Faster |
|---|---:|---:|---:|---:|
| Bernoulli, `(1 + x \| g)` | 2,000 | 36.2 + 5.8 = 42.0 | 5.3 | 7.9× |
| Gaussian, `(1 + x \| g)` | 20,000 | 35.5 + 87.0 = 122.5 | 29.6 | 4.1× |
| cumulative, five categories | 1,500 | 27.0 + 33.4 = 60.4 | 7.1 | 8.5× |
| zero-inflated Poisson | 2,000 | 26.2 + 15.0 = 41.2 | 5.9 | 7.0× |
| Weibull | 2,000 | 27.1 + 6.4 = 33.5 | 6.0 | 5.6× |
| zero-inflated negative binomial | 2,000 | 26.6 + 16.4 = 43.0 | 14.7 | 2.9× |
| sratio, five categories | 1,500 | 26.1 + 17.6 = 43.6 | 7.1 | 6.2× |
| Gaussian, horseshoe on 20 predictors | 400 | 26.6 + 3.0 = 29.6 | 5.3 | 5.6× |
| beta-binomial with trials | 2,000 | 27.7 + 24.4 = 52.1 | 22.3 | 2.3× |
| Gaussian, `(1 \| mm(g1, g2))` | 1,000 | 26.8 + 3.2 = 30.0 | 2.4 | 12.7× |

- **End to end** AutoBayes was 2.3 to 12.7 times faster (median 5.9). On small models most of the gap is brms's Stan
  compile, 26 to 36 seconds per model, against 2 to 4 seconds of JAX compilation.
- **Sampling alone** was 1.2 to 7.9 times faster (median 3.9). Effective samples per second were higher on nine of the
  ten models; the exception was the beta-binomial (9 against 11).
- **The posteriors agree:** every compared mean is within 2.8 combined Monte Carlo standard errors.
- **Post-processing** in brms took longer, by these median factors:

  | `posterior_epred` | `log_lik` | `loo` | `waic` | `bayes_R2` | `conditional_effects` |
  |---:|---:|---:|---:|---:|---:|
  | 2.8× | 2.3× | 2.0× | 5.1× | 4.0× | 6.9× |

Caveats: brms can refit new data without recompiling, so for many fits of one model only the sampling comparison
matters. cmdstanr and GPUs were not tested. These are single runs on a laptop in use. Write-up:
[benchmark-2026-10-17.pdf](docs/brms-parity/reports/benchmark-2026-10-17.pdf); scripts:
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

## Limits

- Beyond the benchmark models, the features added in October 2026 have not been fitted.
- GPU sampling has been validated for the first milestones only.
- Hosted CI is not running; every merge since 11 October 2026 was verified locally.

## Origins

AutoBayes began as a skill and pre-research pack for running a long Bayesian model-iteration program with a coding
agent in the loop. That material is still in the repository (`reference/`, `scaffold/`, `skills/`, `SKILL.md`) and is
described in [the agent scaffold](docs/agent-scaffold.md). The library's record before the brms-parity loop is
[library milestones](docs/milestones.md).

## Licence

MIT. The `wiki/` pattern is adapted from
[Karpathy's LLM Wiki gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).
