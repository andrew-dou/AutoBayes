#!/usr/bin/env Rscript
# U06 (D17): bernoulli(link = "probit") and binomial(link = "probit") evaluate the likelihood of theta = Phi(eta). Stan's
# Phi is exactly 1 above eta = 8.25 and exactly 0 below -37.5, so a failure's log(1 - theta) is -Inf above 8.25 (and loses
# digits from about eta = 5) and a success's log(theta) is -Inf below -37.5, with infinite or NaN gradients, where the
# exact values are log Phi(-eta) and log Phi(eta).
#
#   Rscript docs/brms-parity/upstream/repro/U06.R      # exit status 1 while the problem is present
#
# Stock CRAN brms (rstan backend) only. Intercept-only models with one row each, evaluated with rstan::log_prob (no
# Jacobian, the prior's part removed through the model's prior_only flag). Exact: R's pnorm(., log.p = TRUE) and the
# gradient exp(dnorm(eta, log = TRUE) - log Phi(.)).
t_start <- proc.time()[["elapsed"]]
suppressPackageStartupMessages({ library(brms); library(rstan) })
cat(sprintf("brms %s, rstan %s, StanHeaders %s, R %s\n", packageVersion("brms"), packageVersion("rstan"),
            packageVersion("StanHeaders"), getRversion()))
pri <- set_prior("normal(0, 20)", class = "Intercept")
lik_fun <- function(fit0) {
  sf <- fit0$fit
  sf_prior <- sampling(sf@stanmodel, data = modifyList(standata(fit0), list(prior_only = 1L)), chains = 0)
  function(eta) {
    u <- unconstrain_pars(sf, list(Intercept = eta))
    a <- log_prob(sf, u, adjust_transform = FALSE, gradient = TRUE); b <- log_prob(sf_prior, u, adjust_transform = FALSE, gradient = TRUE)
    c(value = as.numeric(a) - as.numeric(b), grad = attr(a, "gradient") - attr(b, "gradient"))
  }
}
lphi <- function(x) pnorm(x, log.p = TRUE); mills <- function(x) exp(dnorm(x, log = TRUE) - pnorm(x, log.p = TRUE))
fb0 <- brm(y ~ 1, data = data.frame(y = 0), family = bernoulli("probit"), prior = pri, chains = 0, backend = "rstan",
           silent = 2, refresh = 0)
fb1 <- update(fb0, newdata = data.frame(y = 1), chains = 0, recompile = FALSE, silent = 2, refresh = 0)
fbin <- brm(y | trials(n) ~ 1, data = data.frame(y = 0, n = 3), family = binomial("probit"), prior = pri, chains = 0,
            backend = "rstan", silent = 2, refresh = 0)
cases <- list(
  list(name = "bernoulli, y = 0", fit = fb0, etas = c(0, 5, 8, 8.5, 9),
       exact = function(e) c(value = lphi(-e), grad = -mills(-e))),
  list(name = "bernoulli, y = 1", fit = fb1, etas = c(0, -37, -38, -40),
       exact = function(e) c(value = lphi(e), grad = mills(e))),
  list(name = "binomial, y = 0 of 3", fit = fbin, etas = c(0, 8, 9),
       exact = function(e) c(value = 3 * lphi(-e), grad = -3 * mills(-e))))
present <- FALSE; control_ok <- TRUE
for (cs in cases) {
  lik <- lik_fun(cs$fit)
  cat(sprintf("\n%s\n", cs$name))
  for (eta in cs$etas) {
    b <- lik(eta); x <- cs$exact(eta)
    rv <- abs(b[["value"]] - x[["value"]]) / abs(x[["value"]]); rg <- abs(b[["grad"]] - x[["grad"]]) / abs(x[["grad"]])
    cat(sprintf("  eta = %3g: log lik brms % .16g exact % .16g (rel %.2g); gradient brms % .16g exact % .16g (rel %.2g)\n",
                eta, b[["value"]], x[["value"]], rv, b[["grad"]], x[["grad"]], rg))
    bad <- !is.finite(b[["value"]]) || !is.finite(b[["grad"]]) || rv > 1e-12 || rg > 1e-8
    if (eta == 0) control_ok <- control_ok && !bad else if (bad) present <- TRUE
  }
}
if (!control_ok) { cat("\nthe exact formula does not match brms at a control point: the script is wrong\n"); quit(status = 2) }
cat(if (present) "\nPROBLEM PRESENT: brms's probit likelihood is -Inf where Phi saturates\n" else "\nproblem absent\n")
cat(sprintf("run time %.1f s\n", proc.time()[["elapsed"]] - t_start))
quit(status = if (present) 1 else 0)
