#!/usr/bin/env Rscript
# U13 (D105): with (1 | gr(g, by = gby)) the group-level SD is one parameter per level of gby, each with the half-normal
# (or half-t, half-Cauchy) prior truncated at 0, but brms's prior density (lprior, and so log_prob) adds the truncation's
# normalising constant -log P(sd > 0) = log 2 once per coefficient rather than once per SD: with Nby levels it is
# (Nby - 1) log 2 too small. The constant leaves sampling unchanged but enters anything that needs the normalised density
# (bridge sampling's marginal likelihood and Bayes factors).
#
#   Rscript docs/brms-parity/upstream/repro/U13.R      # exit status 1 while the problem is present
#
# Stock CRAN brms (rstan backend) only. A Gaussian model y ~ 1 + (1 | gr(g, by = gby)) with sd ~ normal(0, 1), compiled
# once and evaluated (constrain_pars()'s lprior) with gby taking three levels and, as a control, one level. Exact: the sum
# of the priors, each SD's density 2 * dnorm(sd, 0, 1), sigma's 2 * dnorm(sigma, 0, 1) and the Intercept's normal(0, 5).
t_start <- proc.time()[["elapsed"]]
suppressPackageStartupMessages({ library(brms); library(rstan) })
cat(sprintf("brms %s, rstan %s, StanHeaders %s, R %s\n", packageVersion("brms"), packageVersion("rstan"),
            packageVersion("StanHeaders"), getRversion()))
set.seed(5)
d3 <- data.frame(g = rep(1:6, each = 3), gby = rep(c("a", "b", "c"), each = 6)); d3$y <- rnorm(18)
d1 <- transform(d3, gby = "a")
pri <- c(set_prior("normal(0, 5)", class = "Intercept"), set_prior("normal(0, 1)", class = "sd"),
         set_prior("normal(0, 1)", class = "sigma"))
fit3 <- brm(y ~ 1 + (1 | gr(g, by = gby)), data = d3, prior = pri, chains = 0, backend = "rstan", silent = 2, refresh = 0)
fit1 <- update(fit3, newdata = d1, chains = 0, recompile = FALSE, silent = 2, refresh = 0)
present <- FALSE; control_ok <- TRUE
for (cs in list(list(name = "gby with 3 levels", fit = fit3, control = FALSE), list(name = "gby with 1 level (control)", fit = fit1, control = TRUE))) {
  sf <- cs$fit$fit
  cp <- constrain_pars(sf, seq(-0.6, 0.6, length.out = get_num_upars(sf)))
  sds <- as.numeric(cp$sd_1)
  ex <- dnorm(cp$Intercept, 0, 5, log = TRUE) + sum(dnorm(sds, 0, 1, log = TRUE) + log(2)) + dnorm(cp$sigma, 0, 1, log = TRUE) + log(2)
  cat(sprintf("\n%s: %d SD parameter(s) %s\n", cs$name, length(sds), paste(sprintf("%.4f", sds), collapse = ", ")))
  cat(sprintf("  lprior brms %.15g   exact %.15g   difference %.15g = %.6g log 2\n", cp$lprior, ex, cp$lprior - ex, (cp$lprior - ex) / log(2)))
  bad <- abs(cp$lprior - ex) > 1e-12
  if (cs$control) control_ok <- !bad else if (bad) present <- TRUE
}
if (!control_ok) { cat("\nthe exact formula does not match brms at the control: the script is wrong\n"); quit(status = 2) }
cat(if (present) "\nPROBLEM PRESENT: brms's gr(by =) SD prior misses (Nby - 1) log 2 of its normalising constant\n" else "\nproblem absent\n")
cat(sprintf("run time %.1f s\n", proc.time()[["elapsed"]] - t_start))
quit(status = if (present) 1 else 0)
