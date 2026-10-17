#!/usr/bin/env Rscript
# The brms side of the benchmark (examples/brms_parity/README.md):
#
#   Rscript examples/brms_parity/benchmark_brms.R <data_dir> <out_dir>
#
# Same models, priors and sampler settings as benchmark_autobayes.py: rstan, four chains in parallel (cores = 4),
# warmup = 1000 and 1000 kept draws each, adapt_delta = 0.9. The model is compiled once (brm with chains = 0) and then
# sampled (update with recompile = FALSE), so compilation and sampling are timed separately. Writes the timings, the
# number of divergent transitions, the draws of the compared parameters and the times of the post-processing functions
# on the fit (each the first call and the median of three more) to <out_dir>/brms_<model>.json for compare.py.
# Optional further arguments name the models to run.
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
              set_prior("beta(2, 2)", class = "zi"))),
  B5_weibull_glmm = list(
    formula = y ~ x + (1 | g), family = weibull(),
    prior = c(set_prior("normal(0, 2.5)", class = "Intercept"), set_prior("normal(0, 1)", class = "b"), h1("sd"),
              set_prior("gamma(2, 0.5)", class = "shape"))),
  B6_zinb_glmm = list(
    formula = y ~ x + (1 | g), family = zero_inflated_negbinomial(),
    prior = c(set_prior("normal(0, 2.5)", class = "Intercept"), set_prior("normal(0, 1)", class = "b"), h1("sd"),
              set_prior("gamma(2, 0.5)", class = "shape"), set_prior("beta(2, 2)", class = "zi"))),
  B7_sratio_glmm = list(
    formula = y ~ x + (1 | g), family = sratio("logit"),
    prior = c(set_prior("normal(0, 2.5)", class = "Intercept"), set_prior("normal(0, 1)", class = "b"), h1("sd"))),
  B8_horseshoe_regression = list(
    formula = as.formula(paste("y ~", paste0("x", 1:20, collapse = " + "))), family = gaussian(),
    prior = c(set_prior("normal(0, 2.5)", class = "Intercept"), set_prior("horseshoe()", class = "b"), h1("sigma"))),
  B9_betabinomial_glmm = list(
    formula = k | trials(size) ~ x + (1 | g), family = beta_binomial(),
    prior = c(set_prior("normal(0, 2.5)", class = "Intercept"), set_prior("normal(0, 1)", class = "b"), h1("sd"),
              set_prior("gamma(2, 0.1)", class = "phi"))),
  B10_multimembership = list(
    formula = y ~ x + (1 | mm(g1, g2)), family = gaussian(),
    prior = c(set_prior("normal(0, 2.5)", class = "Intercept"), set_prior("normal(0, 1)", class = "b"), h1("sd"),
              h1("sigma")))
)
if (length(args) > 2) specs <- specs[args[-(1:2)]]
# The post-processing functions timed on each fit, as benchmark_autobayes.py's POST. A function that stops on the model
# is recorded with its message, not timed.
first_coef <- function(name) if (name == "B8_horseshoe_regression") "x1" else "x"
post_fns <- list(
  posterior_epred = function(fit, name) posterior_epred(fit),
  log_lik = function(fit, name) log_lik(fit),
  loo = function(fit, name) loo(fit),
  waic = function(fit, name) waic(fit),
  bayes_R2 = function(fit, name) bayes_R2(fit),
  hypothesis = function(fit, name) hypothesis(fit, paste(first_coef(name), "> 0")),
  conditional_effects = function(fit, name) conditional_effects(fit, first_coef(name)))
time_post <- function(fit, name) {
  lapply(post_fns, function(f) {
    r <- tryCatch({
      first <- system.time(suppressWarnings(f(fit, name)))[["elapsed"]]
      again <- vapply(1:3, function(i) system.time(suppressWarnings(f(fit, name)))[["elapsed"]], 0)
      list(first = first, median = median(again))
    }, error = function(e) list(refused = substr(conditionMessage(e), 1, 160)))
    r
  })
}
keep <- c("b_Intercept", "b_Intercept[1]", "b_Intercept[2]", "b_Intercept[3]", "b_Intercept[4]", "b_x", "b_z",
          paste0("b_x", 1:20), "sd_g__Intercept", "sd_g__x", "sd_mmg1g2__Intercept", "cor_g__Intercept__x", "sigma", "zi",
          "shape", "phi")

for (name in names(specs)) {
  s <- specs[[name]]
  d <- read.csv(file.path(data_dir, paste0(name, ".csv")), stringsAsFactors = FALSE)
  if ("g" %in% names(d)) d$g <- factor(d$g)          # mm()'s g1 and g2 stay character: brms takes the union of their levels
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
  divergent <- sum(subset(nuts_params(fit), Parameter == "divergent__")$Value)
  post <- time_post(fit, name)
  rec <- list(model = name, n = nrow(d), library = "brms", backend = "rstan", divergent = divergent, post = post,
              versions = list(brms = as.character(packageVersion("brms")), rstan = as.character(packageVersion("rstan"))),
              compile_seconds = t_compile, sample_seconds = t_sample,
              chain_seconds = unname(rowSums(rstan::get_elapsed_time(fit$fit))), draws = draws)
  writeLines(toJSON(rec, auto_unbox = TRUE, digits = I(17)), file.path(out_dir, paste0("brms_", name, ".json")))
  cat(sprintf("%s: compile %.1f s, sample %.1f s, %d divergent\n", name, t_compile, t_sample, as.integer(divergent)))
}
