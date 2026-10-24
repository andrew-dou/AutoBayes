#!/usr/bin/env Rscript
# U11 (D120, E04's tail probes): an interval-censored row, y | cens(c, y2) with c = 2, gets the log mass
# log_diff_exp(lcdf(y2), lcdf(y)). In the upper tail both log CDFs are within a few eps of 0 (Stan's weibull_lcdf and
# gamma_lcdf round there: weibull_lcdf(5 | 3, 1) is 0 where the value is -exp(-125)), so the mass loses every digit: -Inf
# for the Weibull interval (5, 6] at shape 3 where the mass is exp(-125), and log(eps) = -36.04 for Weibull (6, 8] at shape 2
# and Gamma(2, 1) (40, 50], where the values are -36 and -36.29. Taking the interval from the survival functions,
# log_diff_exp(lccdf(y), lccdf(y2)), is exact there; brms's own right-censored rows already use lccdf.
#
#   Rscript docs/brms-parity/upstream/repro/U11.R      # exit status 1 while the problem is present
#
# Stock CRAN brms (rstan backend) only. Intercept-only Weibull and Gamma models with one interval-censored row each,
# evaluated with rstan::log_prob (no Jacobian, the prior's part removed through the model's prior_only flag). Exact:
# the survival functions' difference in log space, S(y) = exp(-(y / scale)^shape) for the Weibull and R's
# pgamma(., lower.tail = FALSE, log.p = TRUE) for the Gamma.
t_start <- proc.time()[["elapsed"]]
suppressPackageStartupMessages({ library(brms); library(rstan) })
cat(sprintf("brms %s, rstan %s, StanHeaders %s, R %s\n", packageVersion("brms"), packageVersion("rstan"),
            packageVersion("StanHeaders"), getRversion()))
evaluator <- function(fit0) {
  sf_prior <- sampling(fit0$fit@stanmodel, data = modifyList(standata(fit0), list(prior_only = 1L)), chains = 0)
  function(row, pars) {
    f <- update(fit0, newdata = row, chains = 0, recompile = FALSE, silent = 2, refresh = 0)
    u <- unconstrain_pars(f$fit, pars)
    as.numeric(log_prob(f$fit, u, adjust_transform = FALSE)) - as.numeric(log_prob(sf_prior, u, adjust_transform = FALSE))
  }
}
log_diff <- function(la, lb) la + log(-expm1(lb - la))                 # log(exp(la) - exp(lb)), la > lb
present <- FALSE; control_ok <- TRUE
report <- function(label, b, x, control = FALSE) {
  rel <- abs(b - x) / abs(x)
  cat(sprintf("  %-40s brms % .16g   exact % .16g   relative error %.2g\n", label, b, x, rel))
  bad <- !is.finite(b) || rel > 1e-12
  if (control) control_ok <<- control_ok && !bad else if (bad) present <<- TRUE
}
cat(sprintf("log(.Machine$double.eps) = %.6g\n", log(.Machine$double.eps)))
pri <- c(set_prior("normal(0, 5)", class = "Intercept"), set_prior("gamma(2, 1)", class = "shape"))
fw <- brm(y | cens(c, y2) ~ 1, data = data.frame(y = 0.5, y2 = 1, c = 2), family = weibull(), prior = pri, chains = 0,
          backend = "rstan", silent = 2, refresh = 0)
ev <- evaluator(fw)
# brms's Weibull: scale = mu / gamma(1 + 1 / shape), so the Intercept for scale 1 is log(gamma(1 + 1 / shape)).
weib <- function(y, y2, k) log_diff(-y^k, -y2^k)
cat("\nWeibull, scale 1\n")
report("(0.5, 1] at shape 2 (control)", ev(data.frame(y = 0.5, y2 = 1, c = 2), list(Intercept = log(gamma(1.5)), shape = 2)),
       weib(0.5, 1, 2), control = TRUE)
report("(5, 6] at shape 3", ev(data.frame(y = 5, y2 = 6, c = 2), list(Intercept = log(gamma(4 / 3)), shape = 3)), weib(5, 6, 3))
report("(6, 8] at shape 2", ev(data.frame(y = 6, y2 = 8, c = 2), list(Intercept = log(gamma(1.5)), shape = 2)), weib(6, 8, 2))
report("right-censored at 5, shape 3 (lccdf)", ev(data.frame(y = 5, y2 = 0, c = 1), list(Intercept = log(gamma(4 / 3)), shape = 3)),
       -5^3, control = TRUE)
fg <- brm(y | cens(c, y2) ~ 1, data = data.frame(y = 1, y2 = 2, c = 2), family = Gamma("log"), prior = pri, chains = 0,
          backend = "rstan", silent = 2, refresh = 0)
ev <- evaluator(fg)
# brms's Gamma: rate = shape / mu, so Gamma(shape 2, rate 1) is mu = 2.
gq <- function(y) pgamma(y, 2, lower.tail = FALSE, log.p = TRUE)
cat("\nGamma(shape 2, rate 1)\n")
report("(1, 2] (control)", ev(data.frame(y = 1, y2 = 2, c = 2), list(Intercept = log(2), shape = 2)), log_diff(gq(1), gq(2)),
       control = TRUE)
report("(40, 50]", ev(data.frame(y = 40, y2 = 50, c = 2), list(Intercept = log(2), shape = 2)), log_diff(gq(40), gq(50)))
if (!control_ok) { cat("\nthe exact formula does not match brms at a control point: the script is wrong\n"); quit(status = 2) }
cat(if (present) "\nPROBLEM PRESENT: brms's interval-censored log mass loses its digits in the upper tail\n" else "\nproblem absent\n")
cat(sprintf("run time %.1f s\n", proc.time()[["elapsed"]] - t_start))
quit(status = if (present) 1 else 0)
