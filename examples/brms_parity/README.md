# AutoBayes next to brms: a tour and a benchmark

These scripts produce the demonstrations and the head-to-head numbers in
[the status report](../../docs/brms-parity/reports/autobayes-status.pdf). Every fit samples with NumPyro on the CPU
(four chains, 1000 warm-up and 1000 kept draws, target acceptance 0.9); nothing here needs a GPU.

- `tour.py <out_dir>`: ten model classes on simulated data with known parameters — correlated slopes, ordinal,
  zero-inflated Poisson, distributional, multivariate, non-linear, Gaussian process, AR(1), mixture, measurement error.
  Writes `tour.json`.
- `benchmark_data.py <data_dir>`: the benchmark data sets: B1 logistic and B2 Gaussian with correlated slopes, B3
  cumulative ordinal, B4 zero-inflated Poisson, and since 17 October 2026 B5 Weibull, B6 zero-inflated negative binomial,
  B7 sequential-ratio ordinal, B8 a horseshoe regression on 20 predictors, B9 beta-binomial with trials and B10
  multi-membership (`mm(g1, g2)`); since 21 October 2026 also B11-B18 (below). B1-B10 are regenerated unchanged (same
  seeds; the new models' seeds continue the scheme, `20261015 + k` with k the model's position counting from 0).
- `benchmark_autobayes.py <data_dir> <out_dir> [models] [--smoke]` and
  `Rscript benchmark_brms.R <data_dir> <out_dir> [models] [--smoke]`: the same models, priors and sampler settings in
  each library, timed by stage (below), with the leapfrog steps, the cost of one gradient, the number of divergent
  transitions and the times of the post-processing functions on each fit (`posterior_epred`, `log_lik`, `loo`, `waic`,
  `bayes_R2`, `hypothesis`, `conditional_effects`: the first call and the median of three more; a function that refuses
  the model is recorded with its message). `--smoke` runs one chain of 50 warm-up and 50 kept draws: it checks the
  scripts, and its numbers mean nothing about speed.
- `compare.py <out_dir> [--date D]`: wall times by stage, where the samplers' time goes (below), bulk and tail ESS per
  second and posterior agreement, through the same ArviZ estimators for both, and the post-processing times; prints the
  decomposition table and writes `<out_dir>/comparison-<date>.json` (the date of the runs).
- `report.py <comparison json> [--previous <older comparison json>] [--pdf]`: the LaTeX report,
  `docs/brms-parity/reports/benchmark-<date>.tex` (and its PDF with `--pdf`), every number and every sentence that
  depends on one generated from the JSON. A report built from smoke runs says so on its first page; do not commit one.

## Running it

```
python examples/brms_parity/benchmark_data.py DATA
Rscript examples/brms_parity/benchmark_brms.R DATA OUT
python examples/brms_parity/benchmark_autobayes.py DATA OUT
python examples/brms_parity/compare.py OUT
cp OUT/comparison-DATE.json examples/brms_parity/results/
python examples/brms_parity/report.py examples/brms_parity/results/comparison-DATE.json \
    --previous examples/brms_parity/results/comparison-2026-10-17.json --pdf
```

Run one library at a time, never both at once, on a machine with as little else running as possible: every stage below
is a wall time or a CPU time, and a busy machine inflates both. Further arguments name the models to run (`B12_gp_exact
B13_ar1_panel`).

## Options, and the scaling benchmark (29 October 2026)

`benchmark_brms.R` takes `--backend rstan|cmdstanr` (default rstan), `--threads t` and `--grainsize g` (brms's
`threading()`, cmdstanr only), `--march-native` (cmdstanr), `--normalize true|false` (default true), `--seed s`
(default 2026) and `--no-post`; `benchmark_autobayes.py` takes `--cores c` (a cap on the process's cores, applied
before JAX starts), `--seed s` and `--no-post`. Without them both behave as before. `benchmark_data.py <dir> --scaling
N` writes the scaling models S1-S5 at N rows; the benchmarks run them only when they are named.
`compare.py <results> --scaling` and `report.py scaling-<date>.json` summarise a sweep. The FASRC scripts, the grid
and how the core cap works on each side are in [scaling/README.md](scaling/README.md).

## The models added on 21 October 2026

Each is a construct AutoBayes on `main` fits the way brms does (an ungated oracle case covers the construct), with the same explicit
priors on both sides; numeric predictors are standardised. A half-normal prior is `ab.HalfNormal(1)` in AutoBayes and
`normal(0, 1)` on a parameter bounded below at zero in brms.

| model | formula | family | priors (brms's classes) | rows |
| --- | --- | --- | --- | --- |
| B11 | `y ~ x` | `mixture(gaussian, gaussian)`, ordered intercepts | `Intercept` normal(-1, 2) for `mu1` and normal(1, 2) for `mu2`; `b` normal(0, 1) for each; `sigma1`, `sigma2` half-normal(1); `theta` dirichlet(2) | 1,000 |
| B12 | `y ~ gp(x)` (exact) | Gaussian | `Intercept` normal(0, 2.5); `sdgp` half-normal(1); `lscale` (coef `gpx`) inv_gamma(3, 1); `sigma` half-normal(1) | 200 |
| B13 | `y ~ x + ar(time = t, gr = g, p = 1)` | Gaussian | `Intercept` normal(0, 2.5); `b` normal(0, 1); `sigma` half-normal(1); `ar` normal(0, 0.5) | 2,000 (40 series of 50) |
| B14 | `bf(y ~ a * exp(b * x), a ~ 1 + (1 \| g), b ~ 1, nl = TRUE)` | Gaussian | `b` of `a` normal(2, 1); `b` of `b` normal(0, 1); `sd` of `a` half-normal(1); `sigma` half-normal(1) | 1,000, 20 groups |
| B15 | `bf(mvbind(y1, y2) ~ x + (1 \| g)) + set_rescor(TRUE)` | Gaussian | per response: `Intercept` normal(0, 2.5), `b` normal(0, 1), `sd` and `sigma` half-normal(1); `rescor` lkj(2) | 1,000, 30 groups |
| B16 | `y ~ x + mo(edu) + (1 \| g)` | Gaussian | `Intercept` normal(0, 2.5); `b` normal(0, 1) (the monotonic coefficient too); `sd`, `sigma` half-normal(1); `simo` (coef `moedu1`) dirichlet(2, 2, 2, 2) | 1,500, 30 groups, 5 levels |
| B17 | `y \| cens(cens_r) ~ x + (1 \| g)` | Weibull, 29% right-censored | `Intercept` normal(0, 2.5); `b` normal(0, 1); `sd` half-normal(1); `shape` gamma(2, 0.5) | 2,000, 40 groups |
| B18 | `bf(y ~ x + (1 \| g), sigma ~ z)` | Gaussian | `Intercept` normal(0, 2.5); `b` normal(0, 1); `sd` half-normal(1); `sigma`'s `Intercept` normal(0, 1) and `b` normal(0, 0.5) | 2,000, 40 groups |

Correlated varying slopes on a larger grouping are not added: B2 already has `(1 + x | g)` on 200 groups.

## What each stage measures

brms (rstan):

- **compile**: `brm(chains = 0)`: Stan code generation, C++ compilation and loading (as on 17 October). **codegen**:
  `stancode()` and `standata()` for the same model run again after the fit, the code-generation part of it (warm).
- **sampling**: `update(recompile = FALSE)` (as on 17 October). Within it, **exec** is the slowest chain's warm-up plus
  sampling time as Stan reports it (`get_elapsed_time`); the rest is brms's and rstan's work around the chains (forking
  them, brms's data preparation, its renaming of the fit). The CPU time of the forked chain processes over `update()`
  comes from `proc.time()`.
- **sampler work**: per chain, `n_leapfrog__` summed over warm-up and over sampling, the mean `treedepth__` and the final
  `stepsize__` (`get_sampler_params(inc_warmup = TRUE)`).
- **gradient**: `rstan::grad_log_prob` (log density with the Jacobian, and its gradient, on the unconstrained scale),
  the median of 2,000 calls after a first one, each timed with `Sys.time()`; also the same call made to the model
  instance directly and a call that does no work (`num_pars_unconstrained()`), the R-to-C++ overhead. The point is the
  draw of a short extra rstan chain on the same compiled model and data (200 warm-up iterations, one kept): brms renames
  its fit's parameters and by default does not keep the standardised group effects, so the fit's own draws cannot be
  mapped back to Stan's unconstrained coordinates. `rstan::log_prob` at the point equals that draw's `lp__`
  (`lp_check` in the record).

AutoBayes (NumPyro): `ab.fit` runs unchanged; the benchmark instruments it from outside (`StageProbe`):

- a `jax.monitoring` listener records every interval JAX spends tracing, lowering to MLIR and compiling with XLA;
- a wrapper around PyMC's `_sample_numpyro_nuts` marks the NumPyro run inside the fit and keeps the potential function it
  was given and the last kept draw of the first chain;
- `numpyro.infer.MCMC.run` is wrapped so that the run also collects the warm-up iterations' leapfrog steps, step sizes
  and divergences; the warm-up part is removed before PyMC reads the run. With the same seed, the posterior draws and
  sample statistics are bitwise identical with and without the probe (checked on B10, B11 and B15 with one chain).

From these: **build** (the model), **pre** (start of the fit to the NumPyro run: PyTensor's graph turned into JAX, the
jittered initial points), **compile** (start of the run to the end of its largest XLA compilation, the NUTS program:
tracing, lowering and compiling it; the part the listener reports as JAX compilation is recorded beside it), **exec**
(the end of that compilation to the end of the run: every chain's warm-up and sampling, with the process's CPU seconds
over it) and **post** (the end of the run to the end of the fit: the draws moved to the constrained scale, the
InferenceData). The one-draw fit before the real one (`compile_estimate_seconds`) is kept so that the 17 October
definition of "sampling" (the fit minus it) can still be computed.

- **gradient**: the potential the sampler used (PyMC's jaxified `-logp` with the Jacobian) as
  `jax.jit(jax.value_and_grad(...))`, at the last kept draw of the first chain on the unconstrained scale: its
  tracing-and-lowering and its XLA compile timed on their own (`jit(...).lower(x)`, `.compile()`), the median of 2,000
  host calls after a first one, a jitted call of the same arguments that does no work, and the cost per evaluation
  inside one compiled loop of 2,000 evaluations (each point moved by 1e-300 times the previous gradient, so that XLA
  cannot hoist the evaluation out of the loop).
- **sampler work**: NumPyro's `num_steps` per iteration summed over warm-up and over sampling per chain, the mean tree
  depth (PyMC's `floor(log2(num_steps)) + 1`) and the final step size per chain.

## Where the samplers' time goes

For each side, T is the samplers' execution time (exec above: every chain's warm-up and sampling, compilation excluded),
S the leapfrog steps summed over chains, warm-up and sampling, g the cost of one log density and gradient inside the
sampler and C the CPU seconds spent over T. With O = C / (S g), the CPU cost of one step counted in gradients (1 when a
step costs exactly one gradient), and U = C / T, the cores kept busy on average,

    T = S g O / U   exactly, so   T_brms / T_AutoBayes = (S_b / S_a) (g_b / g_a) (O_b / O_a) (U_a / U_b),

the product of a steps, a gradient, an overhead and a parallelism factor. `compare.py` prints these for every model,
with the additive form T = S_max g + residual (S_max the steps of the chain that took the most), and names the largest
factor. g is AutoBayes's in-loop cost and rstan's direct call less the call that does no work; both host-call medians
are recorded as well (`g_host_us`).

## Limits

- One run each, on a machine whose load is not controlled: a single number may be off by 10-30%.
- Step counts differ for reasons beyond the gradient. Stan and NumPyro both run NUTS with a diagonal metric and
  dual-averaging step-size adaptation, but their warm-up schedules, windows and step-size searches differ, and they start
  from different points, so they end on different step sizes and tree depths. Neither count includes the gradients spent
  searching for an initial step size. The tree depths are each library's own definition.
- The gradient costs are estimates. A host call of a jitted JAX function on the CPU pays a dispatch overhead the sampler
  does not (tens of microseconds on a busy machine, more than the gradient itself on the small models); rstan's
  `grad_log_prob` pays R's S4 dispatch and the R-to-C++ call. brms's point is the draw of a short extra chain, not of the
  fit; for these models the cost hardly depends on the point.
- CPU time: AutoBayes's is the whole process's (XLA's threads included); brms's is that of the forked chain processes
  over `update()`, which includes each child's small R-side setup. With one chain (the smoke run) rstan samples in the R
  process itself, and brms's CPU time then includes brms's own work in `update()`.
- AutoBayes's compilation is read off the timeline (start of the NumPyro run to the end of its largest XLA compilation);
  the probe's four-chain path (NumPyro's parallel chains) was exercised only by full runs, the smoke runs having one
  chain.
- Repeated fits: brms compiles once per model and can refit new data without recompiling; for many fits of one model
  only the samplers' time matters. rstan only (cmdstanr is not installed), no within-chain threading, the CPU only.

## results/

`results/` holds the summaries: `comparison.json` and `tour.json` from the run in the status report (15 October 2026),
and `comparison-2026-10-17.json` from the ten-model run (17 October 2026, written up in
[benchmark-2026-10-17.pdf](../../docs/brms-parity/reports/benchmark-2026-10-17.pdf)); Apple M1 Pro, one run each. The draws
themselves are not committed. A staged run's `comparison-<date>.json` goes here too.
