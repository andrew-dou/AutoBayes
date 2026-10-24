#!/usr/bin/env Rscript
# U12 (D114): posterior_epred() of gen_extreme_value() evaluates mu + sigma * (gamma(1 - xi) - 1) / xi for every xi: it
# returns a finite (here negative) number for xi > 1, where the mean is infinite, NaN at xi = 0, where the mean is
# mu + sigma * Euler's gamma, and loses digits as xi passes 0 (1 - xi rounds away xi's low digits).
#
#   Rscript docs/brms-parity/upstream/repro/U12.R      # exit status 1 while the problem is present
#
# Stock CRAN brms (rstan backend) only. xi is fixed in the formula, bf(y ~ 1, xi = X), one model per X, each a
# fixed-parameter run at mu = 0.1, sigma = 1 (nothing sampled). Exact: the mean is
# infinite for xi >= 1; otherwise mu + sigma * expm1(lgamma(1 - xi)) / xi, with lgamma(1 - xi) summed from its series
# euler * xi + sum_k zeta(k) xi^k / k for |xi| < 1e-3 and the limit euler at xi = 0.
t_start <- proc.time()[["elapsed"]]
suppressPackageStartupMessages({ library(brms); library(rstan) })
cat(sprintf("brms %s, rstan %s, StanHeaders %s, R %s\n", packageVersion("brms"), packageVersion("rstan"),
            packageVersion("StanHeaders"), getRversion()))
pri <- c(set_prior("normal(0, 5)", class = "Intercept"), set_prior("normal(0, 2)", class = "sigma"))
d <- data.frame(y = c(0.1, 0.3))
euler <- -digamma(1); zeta <- c(NA, pi^2 / 6, 1.2020569031595942854, pi^4 / 90, 1.0369277551433699263)
exact_mean <- function(mu, sigma, xi) {
  if (xi >= 1) return(Inf)
  if (xi == 0) return(mu + sigma * euler)
  r <- if (abs(xi) < 1e-3) expm1(euler * xi + sum(zeta[2:5] * xi^(2:5) / (2:5))) / xi else (gamma(1 - xi) - 1) / xi
  mu + sigma * r
}
present <- FALSE; control_ok <- TRUE
cat("\nE[y] at mu = 0.1, sigma = 1\n")
for (xi in c(-0.5, 1e-10, 0, 1.5)) {
  fit <- suppressWarnings(brm(bf(y ~ 1, xi = xi), data = d, family = gen_extreme_value(), prior = pri, backend = "rstan",
                              algorithm = "fixed_param", chains = 1, iter = 2, warmup = 0, seed = 1,
                              init = list(list(Intercept = 0.1, sigma = 1)), silent = 2, refresh = 0))
  cat(sprintf("  (the fit's xi: %s)\n", format(standata(fit)$xi)))
  e <- suppressWarnings(posterior_epred(fit, newdata = data.frame(row = 1:2))[1, 1])
  x <- exact_mean(0.1, 1, xi)
  rel <- if (is.infinite(x)) NA else abs(e - x) / abs(x)
  cat(sprintf("  xi = %-6g brms %.16g   exact %.16g%s\n", xi, e, x, if (is.na(rel)) "" else sprintf("   relative error %.2g", rel)))
  bad <- if (is.infinite(x)) is.finite(e) else (!is.finite(e) || rel > 1e-12)
  if (xi == -0.5) control_ok <- control_ok && !bad else if (bad) present <- TRUE
}
if (!control_ok) { cat("\nthe exact formula does not match brms at a control point: the script is wrong\n"); quit(status = 2) }
cat(if (present) "\nPROBLEM PRESENT: brms's generalized extreme value mean is wrong for xi >= 1 and at or near xi = 0\n" else "\nproblem absent\n")
cat(sprintf("run time %.1f s\n", proc.time()[["elapsed"]] - t_start))
quit(status = if (present) 1 else 0)
