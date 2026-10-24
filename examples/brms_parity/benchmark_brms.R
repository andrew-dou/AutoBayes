#!/usr/bin/env Rscript
# The brms side of the benchmark (examples/brms_parity/README.md):
#
#   Rscript examples/brms_parity/benchmark_brms.R <data_dir> <out_dir> [models ...] [--smoke]
#
# Same models, priors and sampler settings as benchmark_autobayes.py: rstan, four chains in parallel (cores = 4),
# warmup = 1000 and 1000 kept draws each, adapt_delta = 0.9 (--smoke: one chain, 50 and 50, to check the scripts only).
# The model is compiled once (brm with chains = 0) and then sampled (update with recompile = FALSE), so compilation and
# sampling are timed separately. Writes the timings, the stages below, the number of divergent transitions, the draws of
# the compared parameters and the times of the post-processing functions on the fit (each the first call and the median
# of three more) to <out_dir>/brms_<model>.json for compare.py. Optional further arguments name the models to run.
#
# Stages, and how each is measured (README.md has the limits):
# - compile: brm(chains = 0), Stan code generation, C++ compilation and loading, as before; codegen: stancode() and
#   standata() for the same model run again afterwards (warm), the code-generation part of it.
# - sampling: update(recompile = FALSE), as before, with the CPU seconds of this R process and of the forked chain
#   processes over it (proc.time()); exec: the slowest chain's warm-up plus sampling time as Stan reports it
#   (rstan::get_elapsed_time); the rest of the sampling time is brms's and rstan's work around the chains.
# - sampler work: per chain, leapfrog steps (n_leapfrog__) for warm-up and for sampling, the mean tree depth and the final
#   step size (rstan::get_sampler_params(inc_warmup = TRUE)).
# - gradient: rstan::grad_log_prob (log density with the Jacobian and its gradient, unconstrained), the median of 2000
#   calls after a first one, each timed with Sys.time(). The point is the draw of a short extra rstan chain of the same
#   compiled model and data (200 warm-up iterations, 1 kept; --smoke: 50): brms renames its fit's parameters and does not
#   keep the standardised group effects, so the fit's own draws cannot be mapped back to Stan's unconstrained coordinates.
#   Also the median of the same call through the model instance directly (skipping the S4 method) and of a call that does
#   no work (num_pars_unconstrained()), the R-to-C++ call overhead.
suppressPackageStartupMessages({
  library(brms)
  library(jsonlite)
})
args <- commandArgs(trailingOnly = TRUE)
smoke <- "--smoke" %in% args
args <- setdiff(args, "--smoke")
data_dir <- args[1]
out_dir <- args[2]
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
run <- if (smoke) list(chains = 1, cores = 1, warmup = 50, iter = 100, grad_warmup = 50) else
  list(chains = 4, cores = 4, warmup = 1000, iter = 2000, grad_warmup = 200)
grad_calls <- 2000

h1 <- function(cls, ...) set_prior("normal(0, 1)", class = cls, ...)
icpt <- set_prior("normal(0, 2.5)", class = "Intercept")
b01 <- set_prior("normal(0, 1)", class = "b")
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
              h1("sigma"))),
  # since 21 October 2026
  B11_gauss_mixture = list(
    formula = y ~ x, family = mixture(gaussian, gaussian),
    prior = c(set_prior("normal(-1, 2)", class = "Intercept", dpar = "mu1"),
              set_prior("normal(1, 2)", class = "Intercept", dpar = "mu2"),
              set_prior("normal(0, 1)", class = "b", dpar = "mu1"), set_prior("normal(0, 1)", class = "b", dpar = "mu2"),
              h1("sigma1"), h1("sigma2"), set_prior("dirichlet(2)", class = "theta"))),
  B12_gp_exact = list(
    formula = y ~ gp(x), family = gaussian(),
    prior = c(icpt, h1("sdgp"), set_prior("inv_gamma(3, 1)", class = "lscale", coef = "gpx"), h1("sigma"))),
  B13_ar1_panel = list(
    formula = y ~ x + ar(time = t, gr = g, p = 1), family = gaussian(),
    prior = c(icpt, b01, h1("sigma"), set_prior("normal(0, 0.5)", class = "ar"))),
  B14_nonlinear_growth = list(
    formula = bf(y ~ a * exp(b * x), a ~ 1 + (1 | g), b ~ 1, nl = TRUE), family = gaussian(),
    prior = c(set_prior("normal(2, 1)", class = "b", nlpar = "a"), set_prior("normal(0, 1)", class = "b", nlpar = "b"),
              h1("sd", nlpar = "a"), h1("sigma"))),
  B15_mv_rescor = list(
    formula = bf(mvbind(y1, y2) ~ x + (1 | g)) + set_rescor(TRUE), family = gaussian(),
    prior = c(set_prior("normal(0, 2.5)", class = "Intercept", resp = "y1"),
              set_prior("normal(0, 2.5)", class = "Intercept", resp = "y2"),
              set_prior("normal(0, 1)", class = "b", resp = "y1"), set_prior("normal(0, 1)", class = "b", resp = "y2"),
              h1("sd", resp = "y1"), h1("sd", resp = "y2"), h1("sigma", resp = "y1"), h1("sigma", resp = "y2"),
              set_prior("lkj(2)", class = "rescor"))),
  B16_monotonic = list(
    formula = y ~ x + mo(edu) + (1 | g), family = gaussian(),
    prior = c(icpt, b01, h1("sd"), h1("sigma"), set_prior("dirichlet(2, 2, 2, 2)", class = "simo", coef = "moedu1"))),
  B17_weibull_censored = list(
    formula = y | cens(cens_r) ~ x + (1 | g), family = weibull(),
    prior = c(icpt, b01, h1("sd"), set_prior("gamma(2, 0.5)", class = "shape"))),
  B18_distributional_sigma = list(
    formula = bf(y ~ x + (1 | g), sigma ~ z), family = gaussian(),
    prior = c(icpt, b01, h1("sd"), set_prior("normal(0, 1)", class = "Intercept", dpar = "sigma"),
              set_prior("normal(0, 0.5)", class = "b", dpar = "sigma")))
)
if (length(args) > 2) specs <- specs[args[-(1:2)]]
# The post-processing functions timed on each fit, as benchmark_autobayes.py's POST (the same hypothesis and effect where
# a model has no coefficient x). A function that stops on the model is recorded with its message, not timed.
hypotheses <- c(B8_horseshoe_regression = "x1 > 0", B11_gauss_mixture = "mu1_x > 0", B12_gp_exact = "Intercept > 0",
                B14_nonlinear_growth = "a_Intercept > 0", B15_mv_rescor = "y1_x > 0")
hypothesis_of <- function(name) if (name %in% names(hypotheses)) hypotheses[[name]] else "x > 0"
first_coef <- function(name) if (name == "B8_horseshoe_regression") "x1" else "x"
post_fns <- list(
  posterior_epred = function(fit, name) posterior_epred(fit),
  log_lik = function(fit, name) log_lik(fit),
  loo = function(fit, name) loo(fit),
  waic = function(fit, name) waic(fit),
  bayes_R2 = function(fit, name) bayes_R2(fit),
  hypothesis = function(fit, name) hypothesis(fit, hypothesis_of(name)),
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
          "shape", "phi",
          # B11-B18
          "b_mu1_Intercept", "b_mu2_Intercept", "b_mu1_x", "b_mu2_x", "sigma1", "sigma2", "theta1", "theta2",
          "sdgp_gpx", "lscale_gpx", "ar[1]", "b_a_Intercept", "b_b_Intercept", "sd_g__a_Intercept",
          "b_y1_Intercept", "b_y2_Intercept", "b_y1_x", "b_y2_x", "sd_g__y1_Intercept", "sd_g__y2_Intercept",
          "sigma_y1", "sigma_y2", "rescor__y1__y2", "bsp_moedu", paste0("simo_moedu1[", 1:4, "]"),
          "b_sigma_Intercept", "b_sigma_z")

# Seconds per call of f over n calls after a first one, each timed with Sys.time().
time_calls <- function(f, n) {
  f()
  out <- numeric(n)
  for (i in seq_len(n)) {
    s <- unclass(Sys.time())
    f()
    out[i] <- unclass(Sys.time()) - s
  }
  out
}
us <- function(t) list(median_us = median(t) * 1e6, p10_us = unname(quantile(t, 0.1)) * 1e6,
                       p90_us = unname(quantile(t, 0.9)) * 1e6)

# The cost of one rstan::grad_log_prob, at the draw of a short extra chain (see the header).
gradient_cost <- function(fit) {
  sd <- unclass(standata(fit))
  t0 <- proc.time()
  sf <- suppressWarnings(rstan::sampling(fit$fit@stanmodel, data = sd, chains = 1, warmup = run$grad_warmup,
                                         iter = run$grad_warmup + 1, seed = 2026, refresh = 0,
                                         control = list(adapt_delta = 0.9)))
  t_point <- (proc.time() - t0)[["elapsed"]]
  ex <- rstan::extract(sf, permuted = TRUE)
  first <- function(a) {
    d <- dim(a)
    if (is.null(d) || length(d) == 1) return(a[1])
    array(matrix(a, nrow = d[1])[1, ], dim = d[-1])
  }
  point <- lapply(ex[setdiff(names(ex), "lp__")], first)
  upars <- rstan::unconstrain_pars(sf, point)
  lp <- rstan::log_prob(sf, upars, adjust_transform = TRUE)
  inst <- sf@.MISC$stan_fit_instance
  g <- time_calls(function() rstan::grad_log_prob(sf, upars, adjust_transform = TRUE), grad_calls)
  direct <- time_calls(function() inst$grad_log_prob(upars, TRUE), grad_calls)
  null <- time_calls(function() inst$num_pars_unconstrained(), grad_calls)
  list(point = sprintf("draw of an extra rstan chain after %d warm-up iterations, unconstrained", run$grad_warmup),
       calls = grad_calls, log_prob = lp, lp_check = lp - ex$lp__[1], n_unconstrained = length(upars),
       point_seconds = t_point, host = us(g), direct = us(direct), dispatch = us(null))
}

sampler_stages <- function(sf) {
  sp <- rstan::get_sampler_params(sf, inc_warmup = TRUE)
  w <- run$warmup
  kept <- function(m, col) m[(nrow(m) - (run$iter - w) + 1):nrow(m), col]
  saved_warmup <- all(vapply(sp, nrow, 0) == run$iter)
  warm <- function(m, col) if (saved_warmup) m[seq_len(w), col] else NA
  el <- rstan::get_elapsed_time(sf)
  per_chain <- function(f) I(vapply(sp, f, 0))       # I(): a JSON array even for one chain
  list(
    chain_warmup_seconds = I(unname(el[, "warmup"])), chain_sampling_seconds = I(unname(el[, "sample"])),
    exec = max(rowSums(el)),
    steps_warmup = per_chain(function(m) sum(warm(m, "n_leapfrog__"))),
    steps_sampling = per_chain(function(m) sum(kept(m, "n_leapfrog__"))),
    tree_depth_warmup = per_chain(function(m) mean(warm(m, "treedepth__"))),
    tree_depth_sampling = per_chain(function(m) mean(kept(m, "treedepth__"))),
    step_size_final = per_chain(function(m) m[nrow(m), "stepsize__"]),
    divergent_warmup = per_chain(function(m) sum(warm(m, "divergent__"))),
    divergent_sampling = per_chain(function(m) sum(kept(m, "divergent__"))))
}

for (name in names(specs)) {
  tryCatch({                                         # one failing model does not stop the run
    s <- specs[[name]]
    d <- read.csv(file.path(data_dir, paste0(name, ".csv")), stringsAsFactors = FALSE)
    if ("g" %in% names(d)) d$g <- factor(d$g)          # mm()'s g1 and g2 stay character: brms takes the union of their levels
    t0 <- proc.time()
    fit0 <- brm(s$formula, data = d, family = s$family, prior = s$prior, chains = 0, backend = "rstan", silent = 2)
    t_compile <- (proc.time() - t0)[["elapsed"]]
    t1 <- proc.time()
    fit <- update(fit0, recompile = FALSE, chains = run$chains, cores = run$cores, iter = run$iter, warmup = run$warmup,
                  seed = 2026, control = list(adapt_delta = 0.9), refresh = 0, silent = 2)
    dt <- proc.time() - t1
    t_sample <- dt[["elapsed"]]
    dr <- as_draws_df(fit)
    vars <- intersect(keep, names(dr))
    draws <- lapply(vars, function(v) as.numeric(dr[[v]]))    # chain-major: as_draws_df orders chain by chain
    names(draws) <- vars
    divergent <- sum(subset(nuts_params(fit), Parameter == "divergent__")$Value)
    stages <- sampler_stages(fit$fit)
    stages$cpu_self <- dt[["user.self"]] + dt[["sys.self"]]
    stages$cpu_child <- dt[["user.child"]] + dt[["sys.child"]]
    stages$cpu_exec <- if (run$cores > 1 && run$chains > 1) stages$cpu_child else stages$cpu_self
    stages$codegen <- system.time(suppressWarnings({
      stancode(s$formula, data = d, family = s$family, prior = s$prior)
      standata(s$formula, data = d, family = s$family, prior = s$prior)
    }))[["elapsed"]]
    stages$gradient <- gradient_cost(fit)
    post <- time_post(fit, name)
    rec <- list(model = name, n = nrow(d), library = "brms", backend = "rstan", divergent = divergent, post = post,
                date = format(Sys.Date()), smoke = smoke, settings = run, chains = run$chains,
                versions = list(brms = as.character(packageVersion("brms")), rstan = as.character(packageVersion("rstan")),
                                R = paste(R.version$major, R.version$minor, sep = ".")),
                compile_seconds = t_compile, sample_seconds = t_sample,
                chain_seconds = I(unname(rowSums(rstan::get_elapsed_time(fit$fit)))), stages = stages, draws = draws)
    writeLines(toJSON(rec, auto_unbox = TRUE, digits = I(17), na = "null"), file.path(out_dir, paste0("brms_", name, ".json")))
    g <- stages$gradient
    cat(sprintf("%s: compile %.1f s, sample %.1f s (exec %.1f), steps %d+%d, gradient %.1f us (direct %.1f, null %.1f; lp check %.1e), %d divergent\n",
                name, t_compile, t_sample, stages$exec, as.integer(sum(stages$steps_warmup)), as.integer(sum(stages$steps_sampling)),
                g$host$median_us, g$direct$median_us, g$dispatch$median_us, g$lp_check, as.integer(divergent)))
  }, error = function(e) cat(sprintf("%s: FAILED %s\n", name, conditionMessage(e))))
}
