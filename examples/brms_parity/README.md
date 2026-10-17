# AutoBayes next to brms: a tour and a benchmark

These scripts produce the demonstrations and the head-to-head numbers in
[the status report](../../docs/brms-parity/reports/autobayes-status.pdf). Every fit samples with NumPyro on the CPU
(four chains, 1000 warm-up and 1000 kept draws, target acceptance 0.9); nothing here needs a GPU.

- `tour.py <out_dir>`: ten model classes on simulated data with known parameters — correlated slopes, ordinal,
  zero-inflated Poisson, distributional, multivariate, non-linear, Gaussian process, AR(1), mixture, measurement error.
  Writes `tour.json`.
- `benchmark_data.py <data_dir>`: the ten benchmark data sets: B1 logistic and B2 Gaussian with correlated slopes, B3
  cumulative ordinal, B4 zero-inflated Poisson, and since 17 October 2026 B5 Weibull, B6 zero-inflated negative binomial,
  B7 sequential-ratio ordinal, B8 a horseshoe regression on 20 predictors, B9 beta-binomial with trials and B10
  multi-membership (`mm(g1, g2)`). B1-B4 are regenerated unchanged (same seeds).
- `benchmark_autobayes.py <data_dir> <out_dir> [models]` and `Rscript benchmark_brms.R <data_dir> <out_dir> [models]`:
  the same models, priors and sampler settings in each library, timed (brms with rstan: compile and sample separately),
  with the number of divergent transitions and the times of the post-processing functions on each fit
  (`posterior_epred`, `log_lik`, `loo`, `waic`, `bayes_R2`, `hypothesis`, `conditional_effects`: the first call and the
  median of three more; a function that refuses the model is recorded with its message).
- `compare.py <out_dir>`: wall times, ESS per second and posterior agreement, through the same ArviZ estimators for
  both, and the post-processing times; writes `comparison.json`.

`results/` holds the summaries: `comparison.json` and `tour.json` from the run in the status report (15 October 2026),
and `comparison-2026-10-17.json` from the ten-model run (17 October 2026, written up in
[benchmark-2026-10-17.pdf](../../docs/brms-parity/reports/benchmark-2026-10-17.pdf)); Apple M1 Pro, one run each. The draws
themselves are not committed.
