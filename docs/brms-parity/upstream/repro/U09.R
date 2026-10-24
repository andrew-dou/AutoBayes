#!/usr/bin/env Rscript
# U09 (D39): posterior_epred() of a cumulative logit model forms P(y = k) as plogis(tau_k - eta) - plogis(tau_{k-1} - eta)
# by subtraction, which loses relative precision for a category between two close thresholds (and for any category whose
# probability is small next to its neighbours').
#
#   Rscript docs/brms-parity/upstream/repro/U09.R      # exit status 1 while the problem is present
#
# Stock CRAN brms (rstan backend) only. An intercept-only cumulative logit model at fixed thresholds (a fixed-parameter
# run, nothing sampled). Exact: plogis(b) - plogis(a) = plogis(a) * plogis(-b) * expm1(b - a), which forms no difference.
t_start <- proc.time()[["elapsed"]]
suppressPackageStartupMessages({ library(brms); library(rstan) })
cat(sprintf("brms %s, rstan %s, StanHeaders %s, R %s\n", packageVersion("brms"), packageVersion("rstan"),
            packageVersion("StanHeaders"), getRversion()))
fit0 <- brm(y ~ 1, data = data.frame(y = 1:4), family = cumulative(), prior = set_prior("normal(0, 5)", class = "Intercept"),
            chains = 0, backend = "rstan", silent = 2, refresh = 0)
pdiff <- function(a, b) plogis(a) * plogis(-b) * expm1(b - a)       # plogis(b) - plogis(a), b > a
present <- FALSE; control_ok <- TRUE
for (t in list(c(-1, 0, 1), c(0, 1e-9, 2), c(0, 1e-6, 2), c(-1, 20, 20 + 1e-3))) {
  fit <- update(fit0, algorithm = "fixed_param", chains = 1, iter = 2, warmup = 0, seed = 1, recompile = FALSE,
                init = list(list(Intercept = t)), silent = 2, refresh = 0)
  e <- posterior_epred(fit, newdata = data.frame(row = 1:2))[1, 1, ]
  x <- c(plogis(t[1]), pdiff(t[1], t[2]), pdiff(t[2], t[3]), plogis(-t[3]))
  rel <- abs(e - x) / x
  cat(sprintf("\nthresholds (%s)\n", paste(format(t, digits = 10), collapse = ", ")))
  cat(sprintf("  P(y = %d)  brms %.16g   exact %.16g   relative error %.2g\n", 1:4, e, x, rel), sep = "")
  if (identical(t, c(-1, 0, 1))) control_ok <- max(rel) < 1e-14 else if (max(rel) > 1e-12) present <- TRUE
}
if (!control_ok) { cat("\nthe exact formula does not match brms at the control point: the script is wrong\n"); quit(status = 2) }
cat(if (present) "\nPROBLEM PRESENT: brms's cumulative category probabilities lose digits by subtraction\n" else "\nproblem absent\n")
cat(sprintf("run time %.1f s\n", proc.time()[["elapsed"]] - t_start))
quit(status = if (present) 1 else 0)
