#!/usr/bin/env Rscript
# U05 (D18): bernoulli(link = "cloglog") and binomial(link = "cloglog") evaluate the likelihood of the probability
# theta = inv_cloglog(eta), so a failure's log(1 - theta) is formed from 1 - theta by subtraction: its gradient loses
# up to about eps * exp(exp(eta)) relative (2.8e-8 at eta = 3, 1% at 3.5), and the log density is -Inf once theta rounds to 1 (eta above
# about 3.6), where the exact value is -exp(eta).
#
#   Rscript docs/brms-parity/upstream/repro/U05.R      # exit status 1 while the problem is present
#
# Stock CRAN brms (rstan backend) only. Intercept-only models with one row each (a Bernoulli failure; a binomial 1 of 3),
# evaluated with rstan::log_prob (no Jacobian, the prior's part removed through the model's prior_only flag). Exact:
# log(1 - theta) = -exp(eta), log(theta) = log(-expm1(-exp(eta))).
t_start <- proc.time()[["elapsed"]]
suppressPackageStartupMessages({ library(brms); library(rstan) })
cat(sprintf("brms %s, rstan %s, StanHeaders %s, R %s\n", packageVersion("brms"), packageVersion("rstan"),
            packageVersion("StanHeaders"), getRversion()))
pri <- set_prior("normal(0, 10)", class = "Intercept")
lik_fun <- function(fit0) {
  sf <- fit0$fit
  sf_prior <- sampling(sf@stanmodel, data = modifyList(standata(fit0), list(prior_only = 1L)), chains = 0)
  function(eta) {
    u <- unconstrain_pars(sf, list(Intercept = eta))
    a <- log_prob(sf, u, adjust_transform = FALSE, gradient = TRUE); b <- log_prob(sf_prior, u, adjust_transform = FALSE, gradient = TRUE)
    c(value = as.numeric(a) - as.numeric(b), grad = attr(a, "gradient") - attr(b, "gradient"))
  }
}
cases <- list(
  list(name = "bernoulli, y = 0", fit = function() brm(y ~ 1, data = data.frame(y = 0), family = bernoulli("cloglog"),
         prior = pri, chains = 0, backend = "rstan", silent = 2, refresh = 0),
       exact = function(eta) c(value = -exp(eta), grad = -exp(eta))),
  list(name = "binomial, y = 1 of 3", fit = function() brm(y | trials(n) ~ 1, data = data.frame(y = 1, n = 3),
         family = binomial("cloglog"), prior = pri, chains = 0, backend = "rstan", silent = 2, refresh = 0),
       exact = function(eta) { e <- exp(eta); l1 <- log(-expm1(-e))
         c(value = log(3) + l1 - 2 * e, grad = e * exp(-e) / -expm1(-e) - 2 * e) }))
present <- FALSE; control_ok <- TRUE
for (cs in cases) {
  lik <- lik_fun(cs$fit())
  cat(sprintf("\n%s\n", cs$name))
  for (eta in c(0, 3, 3.5, 4)) {
    b <- lik(eta); x <- cs$exact(eta)
    rv <- abs(b[["value"]] - x[["value"]]) / abs(x[["value"]]); rg <- abs(b[["grad"]] - x[["grad"]]) / abs(x[["grad"]])
    cat(sprintf("  eta = %3g: log lik brms % .16g exact % .16g (rel %.2g); gradient brms % .16g exact % .16g (rel %.2g)\n",
                eta, b[["value"]], x[["value"]], rv, b[["grad"]], x[["grad"]], rg))
    bad <- !is.finite(b[["value"]]) || !is.finite(b[["grad"]]) || rv > 1e-12 || rg > 1e-8
    if (eta == 0) control_ok <- control_ok && !bad else if (bad) present <- TRUE
  }
}
if (!control_ok) { cat("\nthe exact formula does not match brms at a control point: the script is wrong\n"); quit(status = 2) }
cat(if (present) "\nPROBLEM PRESENT: brms's cloglog likelihood loses a failure's log(1 - theta)\n" else "\nproblem absent\n")
cat(sprintf("run time %.1f s\n", proc.time()[["elapsed"]] - t_start))
quit(status = if (present) 1 else 0)
