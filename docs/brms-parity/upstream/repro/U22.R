#!/usr/bin/env Rscript
# U22 (D24, D25): Stan's normal_lccdf (Stan Math as shipped with StanHeaders 2.32) loses its value in the upper tail: it is
# off by 2.6e-12 relative at z = 5, 2e-9 at 6.17, 5e-4 at 8 and -Inf from z of about 8.3 on, where log(1 - Phi(z)) is an ordinary
# number (-37.5 at 8.3). brms calls it for a Gaussian right-censored row (cens = 1) and for a truncation's lower bound, so a
# right-censored row 8.3 SD above its mean has log density -Inf and a row of y | trunc(lb = L) with L 9 SD above the mean
# has log density +Inf.
#
#   Rscript docs/brms-parity/upstream/repro/U22.R      # exit status 1 while the problem is present
#
# Stock CRAN brms (rstan backend) only. Intercept-only Gaussian models at mu = 0, sigma = 1, one row each, evaluated with
# rstan::log_prob (no Jacobian, the prior's part removed through the model's prior_only flag). Exact: R's
# pnorm(z, lower.tail = FALSE, log.p = TRUE).
t_start <- proc.time()[["elapsed"]]
suppressPackageStartupMessages({ library(brms); library(rstan) })
cat(sprintf("brms %s, rstan %s, StanHeaders %s, R %s\n", packageVersion("brms"), packageVersion("rstan"),
            packageVersion("StanHeaders"), getRversion()))
pri <- c(set_prior("normal(0, 5)", class = "Intercept"), set_prior("normal(0, 2)", class = "sigma"))
evaluator <- function(fit0) {
  sf_prior <- sampling(fit0$fit@stanmodel, data = modifyList(standata(fit0), list(prior_only = 1L)), chains = 0)
  function(row) {
    f <- update(fit0, newdata = row, chains = 0, recompile = FALSE, silent = 2, refresh = 0)
    u <- unconstrain_pars(f$fit, list(Intercept = 0, sigma = 1))
    as.numeric(log_prob(f$fit, u, adjust_transform = FALSE)) - as.numeric(log_prob(sf_prior, u, adjust_transform = FALSE))
  }
}
lccdf <- function(z) pnorm(z, lower.tail = FALSE, log.p = TRUE)
present <- FALSE; control_ok <- TRUE
report <- function(label, b, x, control) {
  rel <- abs(b - x) / abs(x)
  cat(sprintf("  %-14s brms % .16g   exact % .16g   relative error %.2g\n", label, b, x, rel))
  bad <- !is.finite(b) || rel > 1e-12
  if (control) control_ok <<- control_ok && !bad else if (bad) present <<- TRUE
}
fc <- brm(y | cens(c) ~ 1, data = data.frame(y = 1, c = 1), prior = pri, chains = 0, backend = "rstan", silent = 2, refresh = 0)
cat(sprintf("the generated program calls normal_lccdf: %s\n", grepl("normal_lccdf", stancode(fc), fixed = TRUE)))
ev <- evaluator(fc)
cat("\nright-censored row at y = z: log(1 - Phi(z))\n")
for (z in c(0, 1, 5, 6.17, 8, 8.3, 9, 20)) report(sprintf("z = %g", z), ev(data.frame(y = z, c = 1)), lccdf(z), z <= 1)
# Two identical rows (half their log likelihood is the row's): with a single row brms 2.23.0 passes the per-row bound as a
# scalar and rstan cannot build the model ("mismatch in number dimensions declared and found", variable lb).
ft <- brm(y | trunc(lb = L) ~ 1, data = data.frame(y = c(1.5, 1.5), L = c(1, 1)), prior = pri, chains = 0, backend = "rstan",
          silent = 2, refresh = 0)
ev <- evaluator(ft)
cat("\ntruncated row y = L + 0.5 of y | trunc(lb = L): log phi(y) - log(1 - Phi(L))\n")
for (L in c(0, 1, 6.17, 8, 9)) report(sprintf("L = %g", L), ev(data.frame(y = rep(L + 0.5, 2), L = rep(L, 2))) / 2,
                                      dnorm(L + 0.5, log = TRUE) - lccdf(L), L <= 1)
if (!control_ok) { cat("\nthe exact formula does not match brms at a control row: the script is wrong\n"); quit(status = 2) }
cat(if (present) "\nPROBLEM PRESENT: Stan's normal_lccdf loses its upper tail, so brms's censored and truncated rows do\n" else "\nproblem absent\n")
cat(sprintf("run time %.1f s\n", proc.time()[["elapsed"]] - t_start))
quit(status = if (present) 1 else 0)
