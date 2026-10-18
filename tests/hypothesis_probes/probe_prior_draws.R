#!/usr/bin/env Rscript
# brms 2.23's prior draws (sample_prior = "yes") and hypothesis()'s use of them (E05), on fixed-parameter fits: how
# prior_draws() names the prior draws of each class (a class-wide prior, a coefficient's own prior, a group's SDs, the
# family's parameters), which prior draws hypothesis() pairs with each coefficient, and its evidence ratios.
#
#   Rscript tests/hypothesis_probes/probe_prior_draws.R tests/hypothesis_probes/brms-2.23.0-prior-draws.json
#
# Five models (Gaussian with varying slopes and coefficient-specific priors, Student-t, hurdle negative binomial,
# zero-inflated Beta, Gaussian with two grouping factors) are each compiled once (chains = 0) and run with Stan's
# fixed-parameter algorithm, one chain per seeded point (no MCMC): the posterior draws are the points, and the prior_*
# generated quantities are drawn by Stan's RNG (seed 1); and once more as one chain of 1000 iterations at one point, whose
# 1000 prior draws show their distribution (written with 10 significant digits).
# Recorded: the data and brms's outputs only (prior_draws(), the coefficients' draws, hypothesis() tables and samples
# under two seeds), not its code or its Stan program (both GPL-3). tests/test_prior_draws.py recomputes the evidence ratios
# from the draws and checks ab.prior_draws()'s names and distributions against these.
suppressPackageStartupMessages({ library(brms); library(rstan); library(jsonlite) })
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("usage: Rscript tests/hypothesis_probes/probe_prior_draws.R <output json>")
num <- function(v) ifelse(is.na(v), ifelse(is.nan(v), "NaN", "NA"), ifelse(is.finite(v), sprintf("%.17g", v), ifelse(v > 0, "Inf", "-Inf")))
frame_out <- function(df) list(columns = I(names(df)), values = lapply(as.list(df), function(v) if (is.numeric(v)) I(num(v)) else I(as.character(v))))
set.seed(20261017)
n <- 60
d <- data.frame(x = round(rnorm(n), 4), w = round(rnorm(n, 1, 0.5), 4),
                z = factor(sample(c("p", "q", "r"), n, replace = TRUE), levels = c("p", "q", "r")),
                g = factor(sample(letters[1:6], n, replace = TRUE), levels = letters[1:6]),
                h = factor(sample(c("u", "v", "s", "t"), n, replace = TRUE), levels = c("s", "t", "u", "v")))
d$y <- round(0.5 + 0.8 * d$x - 0.4 * d$w + rnorm(n, 0, 0.7), 4)
d$count <- ifelse(runif(n) < 0.3, 0L, rnbinom(n, mu = exp(0.6 + 0.3 * d$x), size = 2))
d$prop <- ifelse(runif(n) < 0.25, 0, round(plogis(rnorm(n, 0.2 * d$x, 0.8)), 4))
fixed_fit <- function(formula, family, priors, K = 30, seed = 41) {
  fit0 <- brm(formula, data = d, family = family, prior = priors, sample_prior = "yes", chains = 0, backend = "rstan",
              silent = 2, refresh = 0)
  sf <- fit0$fit
  set.seed(seed)
  inits <- lapply(seq_len(K), function(k) constrain_pars(sf, runif(get_num_upars(sf), -1.2, 1.2)))
  fit <- fit0
  fit$fit <- sampling(sf@stanmodel, data = standata(fit0), algorithm = "Fixed_param", chains = K, iter = 1, warmup = 0,
                      init = inits, refresh = 0, seed = 1, cores = 1)
  long <- fit0                                        # one chain at one point: 1000 prior draws, for their distribution
  long$fit <- sampling(sf@stanmodel, data = standata(fit0), algorithm = "Fixed_param", chains = 1, iter = 1000, warmup = 0,
                       init = inits[1], refresh = 0, seed = 2, cores = 1)
  list(fit = rename_pars(fit), long = rename_pars(long))
}
draws_out <- function(fit) {
  dm <- posterior::as_draws_df(fit)
  keep <- grep("^(b_|sd_|cor_|sigma$|nu$|shape$|phi$|zi$|hu$)", names(dm), value = TRUE)
  list(columns = I(keep), values = lapply(setNames(keep, keep), function(nm) I(num(dm[[nm]]))))
}
prior_out <- function(fit, digits = 17) {
  pd <- prior_draws(fit)
  list(columns = I(names(pd)), values = lapply(as.list(pd), function(v) I(sprintf(paste0("%.", digits, "g"), v))))
}
hyp_out <- function(fit, hyps, seed) {
  h <- hypothesis(fit, hyps, seed = seed)
  list(seed = seed, table = frame_out(h$hypothesis), samples = frame_out(as.data.frame(h$samples)))
}
model_out <- function(formula, run, priors, hyps = NULL) {
  out <- list(formula = formula, priors = frame_out(as.data.frame(priors)[c("prior", "class", "coef", "group")]),
              prior_draws = prior_out(run$fit), draws = draws_out(run$fit), long_prior_draws = prior_out(run$long, 10))
  if (length(hyps)) out$hypothesis <- list(hyp_out(run$fit, hyps, 1), hyp_out(run$fit, hyps, 2))
  out
}
pri_a <- c(set_prior("normal(0, 5)", class = "Intercept"), set_prior("normal(0, 2)", class = "b"),
           set_prior("normal(1, 0.5)", class = "b", coef = "w"), set_prior("normal(0, 1)", class = "b", coef = "x:zq"),
           set_prior("normal(0, 1)", class = "sd", group = "g"), set_prior("normal(0, 0.5)", class = "sd", group = "g", coef = "x"),
           set_prior("normal(0, 1)", class = "sigma"), set_prior("lkj(2)", class = "cor", group = "g"))
run_a <- fixed_fit(y ~ x * z + w + (1 + x | g), gaussian(), pri_a)
pri_b <- c(set_prior("normal(0, 5)", class = "Intercept"), set_prior("normal(0, 2)", class = "b"),
           set_prior("normal(0, 1)", class = "sd"), set_prior("normal(0, 1)", class = "sigma"),
           set_prior("gamma(2, 0.1)", class = "nu"))
run_b <- fixed_fit(y ~ x + (1 | g), student(), pri_b)
pri_c <- c(set_prior("normal(0, 2)", class = "Intercept"), set_prior("normal(0, 1)", class = "b"),
           set_prior("normal(0, 1)", class = "sd"), set_prior("gamma(2, 1)", class = "shape"), set_prior("beta(2, 2)", class = "hu"))
run_c <- fixed_fit(count ~ x + (1 | g), hurdle_negbinomial(), pri_c)
pri_d <- c(set_prior("normal(0, 2)", class = "Intercept"), set_prior("normal(0, 1)", class = "b"),
           set_prior("gamma(2, 0.5)", class = "phi"), set_prior("beta(1, 3)", class = "zi"))
run_d <- fixed_fit(prop ~ x + w, zero_inflated_beta(), pri_d)
pri_e <- c(set_prior("normal(0, 5)", class = "Intercept"), set_prior("normal(0, 2)", class = "b"),
           set_prior("normal(0, 1)", class = "sd", group = "g"), set_prior("exponential(2)", class = "sd", group = "h"),
           set_prior("normal(0, 1)", class = "sigma"), set_prior("lkj(2)", class = "cor"))
run_e <- fixed_fit(y ~ x + w + (1 + x + w | g) + (1 | h), gaussian(), pri_e)
out <- list(brms = as.character(packageVersion("brms")), R = paste(R.version$major, R.version$minor, sep = "."),
            data = lapply(as.list(d), function(v) if (is.numeric(v)) I(num(v)) else I(as.character(v))),
            models = list(
              gaussian_slopes = model_out("y ~ x * z + w + (1 + x | g)", run_a, pri_a,
                                          c("Intercept = 0", "x = 0", "w = 1", "x:zq = 0", "zr = 0.2", "x - w = 0", "x > 0",
                                            "Intercept + x = 1")),
              student = model_out("y ~ x + (1 | g)", run_b, pri_b, c("x = 0", "Intercept = 0.5")),
              hurdle_negbinomial = model_out("count ~ x + (1 | g)", run_c, pri_c),
              zero_inflated_beta = model_out("prop ~ x + w", run_d, pri_d, c("x = 0", "w = 0", "Intercept = 0")),
              gaussian_two_groups = model_out("y ~ x + w + (1 + x + w | g) + (1 | h)", run_e, pri_e)))
writeLines(toJSON(out, auto_unbox = TRUE, digits = I(17), pretty = FALSE, null = "null"), args[[1]])
cat(sprintf("brms %s: %d fixed-parameter fits\n", packageVersion("brms"), length(out$models)))
