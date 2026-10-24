#!/usr/bin/env Rscript
# U04 (D13, D132): the Beta family forms its second shape parameter as (1 - mu) * phi with 1 - mu by subtraction, which
# loses eps / (1 - mu) relative and is exactly 0 once mu rounds to 1, where Stan's beta_lpdf rejects the draw: above
# eta = 8.25 under the probit link, about 3.6 under cloglog and about 37 under logit.
#
#   Rscript docs/brms-parity/upstream/repro/U04.R      # exit status 1 while the problem is present
#
# Stock CRAN brms (rstan backend) only. One intercept-only Beta model per link with a single row y = 0.5 and phi = 5,
# evaluated with rstan::log_prob (no Jacobian, the prior's part removed through the model's prior_only flag). Exact: the
# Beta(mu phi, q phi) log density with q = 1 - mu taken from the link's other tail (plogis(-eta), pnorm(-eta),
# exp(-exp(eta))).
t_start <- proc.time()[["elapsed"]]
suppressPackageStartupMessages({ library(brms); library(rstan) })
cat(sprintf("brms %s, rstan %s, StanHeaders %s, R %s\n", packageVersion("brms"), packageVersion("rstan"),
            packageVersion("StanHeaders"), getRversion()))
y <- 0.5; phi <- 5
pri <- c(set_prior("normal(0, 50)", class = "Intercept"), set_prior("gamma(2, 0.5)", class = "phi"))
links <- list(logit = list(mu = function(e) plogis(e), q = function(e) plogis(-e), etas = c(1, 30, 40)),
              probit = list(mu = function(e) pnorm(e), q = function(e) pnorm(-e), etas = c(1, 8, 9)),
              cloglog = list(mu = function(e) -expm1(-exp(e)), q = function(e) exp(-exp(e)), etas = c(0.5, 3.4, 3.7)))
present <- FALSE; control_ok <- TRUE
for (ln in names(links)) {
  L <- links[[ln]]
  fit0 <- brm(y ~ 1, data = data.frame(y = y), family = Beta(link = ln), prior = pri, chains = 0, backend = "rstan",
              silent = 2, refresh = 0)
  sf <- fit0$fit
  sf_prior <- sampling(sf@stanmodel, data = modifyList(standata(fit0), list(prior_only = 1L)), chains = 0)
  cat(sprintf("\nlink %s\n", ln))
  for (eta in L$etas) {
    u <- unconstrain_pars(sf, list(Intercept = eta, phi = phi))
    pr <- as.numeric(log_prob(sf_prior, u, adjust_transform = FALSE))
    b <- tryCatch(as.numeric(log_prob(sf, u, adjust_transform = FALSE)) - pr,
                  error = function(e) structure(NA_real_, msg = strsplit(conditionMessage(e), "\n")[[1]][1]))
    a <- L$mu(eta) * phi; bb <- L$q(eta) * phi
    ex <- lgamma(phi) - lgamma(a) - lgamma(bb) + (a - 1) * log(y) + (bb - 1) * log1p(-y)
    rel <- abs(b - ex) / abs(ex)
    if (is.na(b)) cat(sprintf("  eta = %4g: 1 - mu = %.3g  brms: error (%s)   exact %.16g\n", eta, L$q(eta), attr(b, "msg"), ex))
    else cat(sprintf("  eta = %4g: 1 - mu = %.3g  brms %.16g   exact %.16g   relative error %.2g\n", eta, L$q(eta), b, ex, rel))
    bad <- is.na(b) || !is.finite(b) || rel > 1e-12
    if (eta == L$etas[1]) control_ok <- control_ok && !bad else if (bad) present <- TRUE
  }
}
if (!control_ok) { cat("\nthe exact formula does not match brms at a control point: the script is wrong\n"); quit(status = 2) }
cat(if (present) "\nPROBLEM PRESENT: brms's Beta density loses 1 - mu and rejects draws where it rounds to 0\n" else "\nproblem absent\n")
cat(sprintf("run time %.1f s\n", proc.time()[["elapsed"]] - t_start))
quit(status = if (present) 1 else 0)
