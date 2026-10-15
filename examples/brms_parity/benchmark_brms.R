#!/usr/bin/env Rscript
# The brms side of the benchmark (examples/brms_parity/README.md):
#
#   Rscript examples/brms_parity/benchmark_brms.R <data_dir> <out_dir>
#
# Same models, priors and sampler settings as benchmark_autobayes.py: rstan, four chains in parallel (cores = 4),
# warmup = 1000 and 1000 kept draws each, adapt_delta = 0.9. The model is compiled once (brm with chains = 0) and then
# sampled (update with recompile = FALSE), so compilation and sampling are timed separately. Writes the timings and the
# draws of the compared parameters to <out_dir>/brms_<model>.json for compare.py.
suppressPackageStartupMessages({
  library(brms)
  library(jsonlite)
})
args <- commandArgs(trailingOnly = TRUE)
data_dir <- args[1]
out_dir <- args[2]
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

h1 <- function(cls) set_prior("normal(0, 1)", class = cls)
specs <- list(
  B1_logistic_glmm = list(
    formula = y ~ x + (1 + x | g), family = bernoulli(),
    prior = c(set_prior("normal(0, 2.5)", class = "Intercept"), set_prior("normal(0, 1)", class = "b"),
              h1("sd"), set_prior("lkj(2)", class = "cor"))),
  B2_gaussian_glmm = list(
    formula = y ~ x + z + (1 + x | g), family = gaussian(),
    prior = c(set_prior("normal(0, 2.5)", class = "Intercept"), set_prior("normal(0, 1)", class = "b"),
              h1("sd"), set_prior("lkj(2)", class = "cor"), h1("sigma"))),
  B3_ordinal_glmm = list(
    formula = y ~ x + (1 | g), family = cumulative("logit"),
    prior = c(set_prior("normal(0, 2.5)", class = "Intercept"), set_prior("normal(0, 1)", class = "b"), h1("sd"))),
  B4_zip_glmm = list(
    formula = y ~ x + (1 | g), family = zero_inflated_poisson(),
    prior = c(set_prior("normal(0, 2.5)", class = "Intercept"), set_prior("normal(0, 1)", class = "b"), h1("sd"),
              set_prior("beta(2, 2)", class = "zi")))
)
keep <- c("b_Intercept", "b_Intercept[1]", "b_Intercept[2]", "b_Intercept[3]", "b_Intercept[4]", "b_x", "b_z",
          "sd_g__Intercept", "sd_g__x", "cor_g__Intercept__x", "sigma", "zi")

for (name in names(specs)) {
  s <- specs[[name]]
  d <- read.csv(file.path(data_dir, paste0(name, ".csv")), stringsAsFactors = FALSE)
  d$g <- factor(d$g)
  t0 <- proc.time()
  fit0 <- brm(s$formula, data = d, family = s$family, prior = s$prior, chains = 0, backend = "rstan", silent = 2)
  t_compile <- (proc.time() - t0)[["elapsed"]]
  t1 <- proc.time()
  fit <- update(fit0, recompile = FALSE, chains = 4, cores = 4, iter = 2000, warmup = 1000, seed = 2026,
                control = list(adapt_delta = 0.9), refresh = 0, silent = 2)
  t_sample <- (proc.time() - t1)[["elapsed"]]
  dr <- as_draws_df(fit)
  vars <- intersect(keep, names(dr))
  draws <- lapply(vars, function(v) as.numeric(dr[[v]]))    # chain-major: as_draws_df orders chain by chain
  names(draws) <- vars
  rec <- list(model = name, n = nrow(d), library = "brms", backend = "rstan",
              versions = list(brms = as.character(packageVersion("brms")), rstan = as.character(packageVersion("rstan"))),
              compile_seconds = t_compile, sample_seconds = t_sample,
              chain_seconds = unname(rowSums(rstan::get_elapsed_time(fit$fit))), draws = draws)
  writeLines(toJSON(rec, auto_unbox = TRUE, digits = I(17)), file.path(out_dir, paste0("brms_", name, ".json")))
  cat(sprintf("%s: compile %.1f s, sample %.1f s\n", name, t_compile, t_sample))
}
