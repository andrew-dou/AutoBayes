#!/usr/bin/env Rscript
# U21 (D29, checked and not a bug): divergence row D29 says that a prior on a probability parameter (zi, hu) is added as
# written, not renormalised on (0, 1). brms 2.23.0 does renormalise it: the prior density brms adds (lprior) for
# normal(0.5, 1) on zi, and for normal(0.3, 0.1) on zoi and logistic(0, 1) on coi, equals the density divided by its mass
# on (0, 1), as for the lower-bounded classes. Kept in the log so the row is not reported upstream.
#
#   Rscript docs/brms-parity/upstream/repro/U21.R      # exit status 0: the problem is absent
#
# Stock CRAN brms (rstan backend) only: constrain_pars()'s lprior of two compiled models at fixed points, against the
# priors' densities with and without the renormalisation.
t_start <- proc.time()[["elapsed"]]
suppressPackageStartupMessages({ library(brms); library(rstan) })
cat(sprintf("brms %s, rstan %s, StanHeaders %s, R %s\n", packageVersion("brms"), packageVersion("rstan"),
            packageVersion("StanHeaders"), getRversion()))
present <- FALSE
check <- function(label, lprior, as_written, mass) {
  cat(sprintf("  %s\n    lprior brms %.15g   as written %.15g   renormalised on (0, 1) %.15g\n", label, lprior, as_written,
              as_written - mass))
  if (abs(lprior - (as_written - mass)) > 1e-12) present <<- TRUE
}
f1 <- brm(y ~ 1, data = data.frame(y = c(0, 1, 0, 2)), family = zero_inflated_poisson(), chains = 0, backend = "rstan",
          prior = c(set_prior("normal(0, 5)", class = "Intercept"), set_prior("normal(0.5, 1)", class = "zi")),
          silent = 2, refresh = 0)
cp <- constrain_pars(f1$fit, unconstrain_pars(f1$fit, list(Intercept = 0.2, zi = 0.3)))
cat("\nzero_inflated_poisson\n")
check("zi ~ normal(0.5, 1) at zi = 0.3, Intercept ~ normal(0, 5) at 0.2", cp$lprior,
      dnorm(0.2, 0, 5, log = TRUE) + dnorm(0.3, 0.5, 1, log = TRUE), log(pnorm(1, 0.5, 1) - pnorm(0, 0.5, 1)))
f2 <- brm(y ~ 1, data = data.frame(y = c(0, 0.3, 0.6, 1)), family = zero_one_inflated_beta(), chains = 0, backend = "rstan",
          prior = c(set_prior("normal(0, 5)", class = "Intercept"), set_prior("exponential(1)", class = "phi"),
                    set_prior("normal(0.3, 0.1)", class = "zoi"), set_prior("logistic(0, 1)", class = "coi")),
          silent = 2, refresh = 0)
cp <- constrain_pars(f2$fit, unconstrain_pars(f2$fit, list(Intercept = 0.2, phi = 2, zoi = 0.25, coi = 0.6)))
cat("\nzero_one_inflated_beta\n")
check("zoi ~ normal(0.3, 0.1) at 0.25, coi ~ logistic(0, 1) at 0.6, phi ~ exponential(1) at 2, Intercept at 0.2", cp$lprior,
      dnorm(0.2, 0, 5, log = TRUE) + dexp(2, 1, log = TRUE) + dnorm(0.25, 0.3, 0.1, log = TRUE) + dlogis(0.6, 0, 1, log = TRUE),
      log(pnorm(1, 0.3, 0.1) - pnorm(0, 0.3, 0.1)) + log(plogis(1) - plogis(0)))
cat(if (present) "\nPROBLEM PRESENT: a probability parameter's prior is not renormalised on (0, 1)\n"
    else "\nproblem absent: brms renormalises these priors on (0, 1); D29's description is not what brms 2.23.0 does\n")
cat(sprintf("run time %.1f s\n", proc.time()[["elapsed"]] - t_start))
quit(status = if (present) 1 else 0)
