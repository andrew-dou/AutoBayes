#!/usr/bin/env Rscript
# brms 2.23's outputs for the zero-inflated and hurdle families of D03 (docs/zero-inflated-hurdle.md):
#
#   Rscript tests/zero_hurdle_probes/probe_zero_hurdle.R tests/zero_hurdle_probes/brms-2.23.0-zero-hurdle.json
#
# Recorded: the links brmsfamily() accepts for each family (D84); what make_standata() says to an exact 1 in a
# zero_inflated_beta response, a negative hurdle_gamma response, a fractional zero_inflated_negbinomial count, a zero-trial
# zero_inflated_binomial row and a zero weight; and posterior_predict() of hurdle_negbinomial(), the counts of its draws at
# three fixed points of an intercept-only model (D86). Each point is a fixed-parameter run (Stan's Fixed_param: the draws are
# the point itself, nothing is sampled), and posterior_predict() draws one response per draw and row with set.seed(11).
# brms's outputs only, not its code (brms is GPL-3). tests/test_zero_inflated_hurdle_more.py compares the draws with the
# zero-truncated negative binomial distribution, which brms's own posterior_epred() reports the mean of.
suppressPackageStartupMessages({ library(brms); library(rstan); library(jsonlite) })
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("usage: Rscript tests/zero_hurdle_probes/probe_zero_hurdle.R <output json>")
out <- list(brms = as.character(packageVersion("brms")), R = paste(R.version$major, R.version$minor, sep = "."))
fams <- c("zero_inflated_negbinomial", "hurdle_negbinomial", "zero_inflated_binomial", "zero_inflated_beta", "hurdle_gamma",
          "hurdle_lognormal", "zero_inflated_poisson", "hurdle_poisson")
links <- c("identity", "log", "logit", "probit", "probit_approx", "cloglog", "cauchit", "inverse", "sqrt", "softplus",
           "squareplus", "logm1", "1/mu^2", "tan_half")
out$links <- lapply(setNames(fams, fams), function(f) {
  ok <- vapply(links, function(l) !inherits(try(brmsfamily(f, link = l), silent = TRUE), "try-error"), TRUE)
  list(default = brmsfamily(f)$link, accepted = I(links[ok]))
})
check <- function(expr) tryCatch({ force(expr); "accepted" }, error = function(e) conditionMessage(e))
out$responses <- list(
  zero_inflated_beta_one = check(make_standata(y ~ 1, data.frame(y = c(0, 0.5, 1)), family = zero_inflated_beta())),
  hurdle_gamma_negative = check(make_standata(y ~ 1, data.frame(y = c(0, -0.5, 1)), family = hurdle_gamma())),
  zero_inflated_negbinomial_fraction = check(make_standata(y ~ 1, data.frame(y = c(0, 1.5, 1)), family = zero_inflated_negbinomial())),
  zero_inflated_binomial_zero_trials = check(make_standata(y | trials(n) ~ 1, data.frame(y = c(0, 1, 0), n = c(0, 2, 3)),
                                                           family = zero_inflated_binomial())),
  zero_inflated_poisson_zero_weight = check(make_standata(y | weights(w) ~ 1, data.frame(y = c(0, 1, 2), w = c(0, 1, 2)),
                                                          family = zero_inflated_poisson())))
d <- data.frame(y = c(0, 1, 2, 3, 0, 5, 1, 4))
pri <- c(set_prior("normal(0, 5)", class = "Intercept"), set_prior("gamma(1, 1)", class = "shape"), set_prior("beta(1, 1)", class = "hu"))
fit0 <- brm(y ~ 1, data = d, family = hurdle_negbinomial(), prior = pri, chains = 0, backend = "rstan", silent = 2, refresh = 0)
draws_at <- function(mu, shape, hu, S = 4000, rows = 25) {
  f <- fit0
  f$fit <- sampling(fit0$fit@stanmodel, data = standata(fit0), algorithm = "Fixed_param", chains = 1, iter = S, warmup = 0,
                    init = list(list(Intercept = log(mu), shape = shape, hu = hu)), refresh = 0, seed = 1, cores = 1)
  f <- rename_pars(f)
  set.seed(11)
  y <- as.vector(posterior_predict(f, newdata = data.frame(row = seq_len(rows))))
  epred <- as.vector(posterior_epred(f, newdata = data.frame(row = 1)))[1]
  list(mu = mu, shape = shape, hu = hu, n = length(y), max = max(y), counts = I(as.integer(tabulate(y + 1, nbins = max(y) + 1))),
       epred = sprintf("%.17g", epred))
}
out$hurdle_negbinomial_draws <- list(draws_at(0.5, 0.5, 0.05), draws_at(3, 2, 0.05), draws_at(0.05, 0.2, 0.05))
writeLines(toJSON(out, auto_unbox = TRUE, digits = I(17), pretty = TRUE), args[[1]])
cat(sprintf("brms %s: %d points of hurdle_negbinomial draws\n", packageVersion("brms"), length(out$hurdle_negbinomial_draws)))
