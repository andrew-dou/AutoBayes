#!/usr/bin/env Rscript
# U26 (D113): Stan's von_mises_lpdf (StanHeaders 2.32), which brms's von_mises() family calls, evaluates the Bessel
# function I0(kappa) itself and raises "numeric overflow" for kappa above about 713.9, so brms cannot evaluate (or start a
# fit at) a precision above that, where the density is an ordinary number: log I0(kappa) = kappa + log(i0e(kappa)), and
# R's besselI(kappa, 0, expon.scaled = TRUE) gives it.
#
#   Rscript docs/brms-parity/upstream/repro/U26.R      # exit status 1 while the problem is present
#
# Stock CRAN brms (rstan backend) only. An intercept-only von Mises model with one row y = 0.1, at Intercept 0 (mu = 0 under
# the tan_half link), evaluated with rstan::log_prob (no Jacobian, the prior's part removed through the model's prior_only
# flag). Exact: kappa * (cos(y - mu) - 1) - log(2 pi) - log(besselI(kappa, 0, expon.scaled = TRUE)).
t_start <- proc.time()[["elapsed"]]
suppressPackageStartupMessages({ library(brms); library(rstan) })
cat(sprintf("brms %s, rstan %s, StanHeaders %s, R %s\n", packageVersion("brms"), packageVersion("rstan"),
            packageVersion("StanHeaders"), getRversion()))
pri <- c(set_prior("normal(0, 1)", class = "Intercept"), set_prior("gamma(2, 0.001)", class = "kappa"))
fit0 <- brm(y ~ 1, data = data.frame(y = 0.1), family = von_mises(), prior = pri, chains = 0, backend = "rstan", silent = 2, refresh = 0)
cat(sprintf("the generated program calls von_mises_lpdf: %s\n", grepl("von_mises_lpdf", stancode(fit0), fixed = TRUE)))
sf <- fit0$fit
sf_prior <- sampling(sf@stanmodel, data = modifyList(standata(fit0), list(prior_only = 1L)), chains = 0)
present <- FALSE; control_ok <- TRUE
for (kappa in c(1, 700, 713, 714, 720, 1000)) {
  u <- unconstrain_pars(sf, list(Intercept = 0, kappa = kappa))
  b <- tryCatch(as.numeric(log_prob(sf, u, adjust_transform = FALSE)) - as.numeric(log_prob(sf_prior, u, adjust_transform = FALSE)),
                error = function(e) structure(NA_real_, msg = strsplit(conditionMessage(e), "\n")[[1]][1]))
  x <- kappa * (cos(0.1) - 1) - log(2 * pi) - log(besselI(kappa, 0, expon.scaled = TRUE))
  if (is.na(b)) cat(sprintf("  kappa = %5g: brms error: %s   exact %.16g\n", kappa, attr(b, "msg"), x))
  else cat(sprintf("  kappa = %5g: brms %.16g   exact %.16g   relative error %.2g\n", kappa, b, x, abs(b - x) / abs(x)))
  bad <- is.na(b) || !is.finite(b) || abs(b - x) / abs(x) > 1e-10
  if (kappa <= 700) control_ok <- control_ok && !bad else if (bad) present <- TRUE
}
if (!control_ok) { cat("\nthe exact formula does not match brms at a control point: the script is wrong\n"); quit(status = 2) }
cat(if (present) "\nPROBLEM PRESENT: Stan's von_mises_lpdf overflows at large kappa\n" else "\nproblem absent\n")
cat(sprintf("run time %.1f s\n", proc.time()[["elapsed"]] - t_start))
quit(status = if (present) 1 else 0)
