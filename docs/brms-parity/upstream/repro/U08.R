#!/usr/bin/env Rscript
# U08 (D49): log_lik() (and so loo() and waic()) evaluates rows in R from probabilities that round to 0 or 1, so a row with
# an extreme but finite linear predictor gets -Inf, +Inf or NaN, or loses digits, where its log likelihood is an ordinary
# finite number: a Bernoulli or binomial logit failure at large eta, a hurdle Poisson positive count at very negative eta,
# a cumulative logit row whose category is improbable, and a finite mixture row where every component's density underflows.
#
#   Rscript docs/brms-parity/upstream/repro/U08.R      # exit status 1 while the problem is present
#
# Stock CRAN brms (rstan backend) only. Each model is a fixed-parameter run (nothing sampled) and log_lik() is taken on new
# rows whose linear predictor is set through an offset (or, for the mixture, whose response lies far out). Exact values in
# log space: -log1p(exp(eta)) as -(eta + log1p(exp(-eta))); the hurdle row as in U07; plogis(., log.p = TRUE) and
# log(plogis(b) - plogis(a)) = log(plogis(a)) + log(plogis(-b)) + log(expm1(b - a)); a log-sum-exp over the components.
t_start <- proc.time()[["elapsed"]]
suppressPackageStartupMessages({ library(brms); library(rstan) })
cat(sprintf("brms %s, rstan %s, StanHeaders %s, R %s\n", packageVersion("brms"), packageVersion("rstan"),
            packageVersion("StanHeaders"), getRversion()))
fixed <- function(formula, data, family, prior, init) {
  brm(formula, data = data, family = family, prior = prior, backend = "rstan", algorithm = "fixed_param", chains = 1,
      iter = 2, warmup = 0, seed = 1, init = list(init), silent = 2, refresh = 0)
}
present <- FALSE; control_ok <- TRUE
report <- function(title, labels, b, x, ncontrol = 1) {
  rel <- abs(b - x) / abs(x)
  cat(sprintf("\n%s\n", title))
  cat(sprintf("  %-22s brms % .16g   exact % .16g   relative error %.2g\n", labels, b, x, rel), sep = "")
  bad <- !is.finite(b) | rel > 1e-12
  control_ok <<- control_ok && !any(bad[seq_len(ncontrol)])
  if (any(bad[-seq_len(ncontrol)])) present <<- TRUE
}
softplus <- function(e) ifelse(e > 0, e + log1p(exp(-e)), log1p(exp(e)))
pri <- set_prior("normal(0, 5)", class = "Intercept")
# Bernoulli logit, failures
f <- fixed(y ~ 1 + offset(o), data.frame(y = c(0, 1), o = 0), bernoulli(), pri, list(Intercept = 0))
eta <- c(0, 20, 35.07, 40)
report("bernoulli (logit), y = 0", sprintf("eta = %g", eta), log_lik(f, newdata = data.frame(y = 0, o = eta))[1, ], -softplus(eta))
# binomial logit, 0 of 3
f <- fixed(y | trials(n) ~ 1 + offset(o), data.frame(y = c(0, 1), n = 3, o = 0), binomial(), pri, list(Intercept = 0))
eta <- c(0, 27.2, 40)
report("binomial (logit), y = 0 of 3", sprintf("eta = %g", eta),
       log_lik(f, newdata = data.frame(y = 0, n = 3, o = eta))[1, ], -3 * softplus(eta))
# hurdle Poisson, y = 1, hu = 0.3
f <- fixed(y ~ 1 + offset(o), data.frame(y = c(0, 1), o = 0), hurdle_poisson(),
           c(pri, set_prior("beta(2, 2)", class = "hu")), list(Intercept = 0, hu = 0.3))
eta <- c(0, -26.9, -60)
l1m <- ifelse(eta < -30, eta + log1p(-exp(eta) / 2), log(-expm1(-exp(eta))))          # log(1 - exp(-mu))
report("hurdle_poisson, y = 1, hu = 0.3", sprintf("eta = %g", eta), log_lik(f, newdata = data.frame(y = 1, o = eta))[1, ],
       log1p(-0.3) + eta - exp(eta) - l1m)
# cumulative logit, thresholds (-1, 0, 1)
f <- fixed(y ~ 1 + offset(o), data.frame(y = 1:4, o = 0), cumulative(), pri, list(Intercept = c(-1, 0, 1)))
nd <- data.frame(y = c(2, 4, 2, 1), o = c(0, -640, -640, 640))
lpd <- function(a, b) plogis(a, log.p = TRUE) + plogis(-b, log.p = TRUE) + log(expm1(b - a))    # log(plogis(b) - plogis(a))
tau <- c(-Inf, -1, 0, 1, Inf)
x <- mapply(function(y, e) { a <- tau[y] - e; b <- tau[y + 1] - e
  if (is.infinite(a)) plogis(b, log.p = TRUE) else if (is.infinite(b)) plogis(-a, log.p = TRUE) else lpd(a, b) }, nd$y, nd$o)
report("cumulative (logit), thresholds (-1, 0, 1)", sprintf("y = %d, eta = %g", nd$y, nd$o), log_lik(f, newdata = nd)[1, ], x)
# mixture of two Gaussians, means -1 and 1, sigmas 1, theta (0.5, 0.5)
pri_m <- c(set_prior("normal(0, 5)", class = "Intercept", dpar = "mu1"), set_prior("normal(0, 5)", class = "Intercept", dpar = "mu2"),
           set_prior("normal(0, 2)", class = "sigma1"), set_prior("normal(0, 2)", class = "sigma2"))
f <- fixed(y ~ 1, data.frame(y = c(-1, 0, 1)), mixture(gaussian, gaussian), pri_m,
           list(ordered_Intercept = c(-1, 1), sigma1 = 1, sigma2 = 1, theta = c(0.5, 0.5)))
yy <- c(0, 30, 40)
lc <- cbind(log(0.5) + dnorm(yy, -1, 1, log = TRUE), log(0.5) + dnorm(yy, 1, 1, log = TRUE))
x <- apply(lc, 1, function(v) max(v) + log(sum(exp(v - max(v)))))
report("mixture(gaussian, gaussian)", sprintf("y = %g", yy), log_lik(f, newdata = data.frame(y = yy))[1, ], x)
if (!control_ok) { cat("\nthe exact formula does not match brms at a control row: the script is wrong\n"); quit(status = 2) }
cat(if (present) "\nPROBLEM PRESENT: brms's log_lik is infinite, NaN or inexact at extreme but finite rows\n" else "\nproblem absent\n")
cat(sprintf("run time %.1f s\n", proc.time()[["elapsed"]] - t_start))
quit(status = if (present) 1 else 0)
