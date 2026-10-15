# AutoBayes next to brms: a tour and a benchmark

These scripts produce the demonstrations and the head-to-head numbers in
[the status report](../../docs/brms-parity/reports/autobayes-status.pdf). Every fit samples with NumPyro on the CPU
(four chains, 1000 warm-up and 1000 kept draws, target acceptance 0.9); nothing here needs a GPU.

- `tour.py <out_dir>`: ten model classes on simulated data with known parameters — correlated slopes, ordinal,
  zero-inflated Poisson, distributional, multivariate, non-linear, Gaussian process, AR(1), mixture, measurement error.
  Writes `tour.json`.
- `benchmark_data.py <data_dir>`: the four benchmark data sets (B1 logistic and B2 Gaussian with correlated slopes, B3
  ordinal, B4 zero-inflated Poisson).
- `benchmark_autobayes.py <data_dir> <out_dir>` and `Rscript benchmark_brms.R <data_dir> <out_dir>`: the same models,
  priors and sampler settings in each library, timed (brms with rstan: compile and sample separately).
- `compare.py <out_dir>`: wall times, ESS per second and posterior agreement, through the same ArviZ estimators for
  both; writes `comparison.json`.

`results/` holds the summaries from the run in the report (15 October 2026, Apple M1 Pro, one run each). The draws
themselves are not committed.
