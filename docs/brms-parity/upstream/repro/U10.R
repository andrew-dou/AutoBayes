#!/usr/bin/env Rscript
# U10 (D25): posterior_epred() of a truncated Gaussian, y | trunc(lb = ...), is the truncated mean formed with
# pnorm(ub) - pnorm(lb) by subtraction: it loses digits once the bound lies a few standard deviations above the location
# (1e-3 relative at 7.5 SD) and is Inf or NaN beyond about 8.3 SD, where the exact mean is just above the bound.
#
#   Rscript docs/brms-parity/upstream/repro/U10.R      # exit status 1 while the problem is present
#
# Stock CRAN brms (rstan backend) only. A Gaussian model y | trunc(lb = L) ~ 1 at mu = 0, sigma = 1 (a fixed-parameter run,
# nothing sampled), its expected value on new rows with bounds L. Exact: mu + sigma * phi(z) / (1 - Phi(z)), z = (L - mu)
# / sigma, the ratio evaluated as exp(dnorm(z, log = TRUE) - pnorm(z, lower.tail = FALSE, log.p = TRUE)).
t_start <- proc.time()[["elapsed"]]
suppressPackageStartupMessages({ library(brms); library(rstan) })
cat(sprintf("brms %s, rstan %s, StanHeaders %s, R %s\n", packageVersion("brms"), packageVersion("rstan"),
            packageVersion("StanHeaders"), getRversion()))
pri <- c(set_prior("normal(0, 5)", class = "Intercept"), set_prior("normal(0, 2)", class = "sigma"))
fit <- brm(y | trunc(lb = L) ~ 1, data = data.frame(y = c(7, 8), L = c(6.17, 6.17)), prior = pri, backend = "rstan",
           algorithm = "fixed_param", chains = 1, iter = 2, warmup = 0, seed = 1,
           init = list(list(Intercept = 0, sigma = 1)), silent = 2, refresh = 0)
L <- c(0, 3, 6.17, 7.5, 8.2, 9, 40)
e <- posterior_epred(fit, newdata = data.frame(L = L, y = L + 0.5))[1, ]
x <- exp(dnorm(L, log = TRUE) - pnorm(L, lower.tail = FALSE, log.p = TRUE))
rel <- abs(e - x) / x
cat("\nE[y | y > L] at mu = 0, sigma = 1\n")
cat(sprintf("  L = %5g: brms %.16g   exact %.16g   relative error %.2g\n", L, e, x, rel), sep = "")
control_ok <- all(rel[1:2] < 1e-13)
present <- any(!is.finite(e[-(1:2)])) || any(rel[-(1:2)] > 1e-9, na.rm = TRUE)
if (!control_ok) { cat("\nthe exact formula does not match brms at a control point: the script is wrong\n"); quit(status = 2) }
cat(if (present) "\nPROBLEM PRESENT: brms's truncated Gaussian mean loses its digits in the upper tail\n" else "\nproblem absent\n")
cat(sprintf("run time %.1f s\n", proc.time()[["elapsed"]] - t_start))
quit(status = if (present) 1 else 0)
