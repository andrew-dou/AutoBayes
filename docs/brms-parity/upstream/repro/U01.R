#!/usr/bin/env Rscript
# U01 (D86): hurdle_negbinomial posterior_predict() does not draw the positive part from the zero-truncated negative
# binomial, so its draws contradict brms's own posterior_epred().
#
#   Rscript docs/brms-parity/upstream/repro/U01.R      # exit status 1 while the problem is present
#
# Stock CRAN brms (rstan backend) only. Each point is a fixed-parameter run (nothing is sampled): an intercept-only
# hurdle_negbinomial model whose every draw is (mu, shape, hu). The exact values are the zero-truncated negative binomial's,
# P(y = 1 | y > 0) = NB(1) / (1 - NB(0)) and E[y | y > 0] = mu / (1 - NB(0)), NB(0) = (shape / (mu + shape))^shape.
t_start <- proc.time()[["elapsed"]]
suppressPackageStartupMessages({ library(brms); library(rstan) })
cat(sprintf("brms %s, rstan %s, StanHeaders %s, R %s\n", packageVersion("brms"), packageVersion("rstan"),
            packageVersion("StanHeaders"), getRversion()))
d <- data.frame(y = c(0, 1, 2, 3))
pri <- c(set_prior("normal(0, 5)", class = "Intercept"), set_prior("gamma(1, 1)", class = "shape"),
         set_prior("beta(1, 1)", class = "hu"))
fit0 <- brm(y ~ 1, data = d, family = hurdle_negbinomial(), prior = pri, chains = 0, backend = "rstan",
            silent = 2, refresh = 0)
points <- list(c(mu = 0.5, shape = 0.5, hu = 0.05), c(mu = 3, shape = 2, hu = 0.05), c(mu = 0.05, shape = 0.2, hu = 0.05))
present <- FALSE
for (p in points) {
  mu <- p[["mu"]]; shape <- p[["shape"]]; hu <- p[["hu"]]
  fit <- update(fit0, algorithm = "fixed_param", chains = 1, iter = 1000, warmup = 0, seed = 1, recompile = FALSE,
                init = list(list(Intercept = log(mu), shape = shape, hu = hu)), silent = 2, refresh = 0)
  set.seed(11)
  yrep <- posterior_predict(fit, newdata = data.frame(row = seq_len(100)))     # 1000 draws x 100 rows
  pos <- yrep[yrep > 0]
  p0 <- dnbinom(0, size = shape, mu = mu)
  p1_exact <- dnbinom(1, size = shape, mu = mu) / (1 - p0)
  mean_exact <- mu / (1 - p0)
  var_exact <- (mu + mu^2 / shape + mu^2) / (1 - p0) - mean_exact^2              # Var[y | y > 0]
  se_p1 <- sqrt(p1_exact * (1 - p1_exact) / length(pos)); se_mean <- sqrt(var_exact / length(pos))
  epred <- posterior_epred(fit, newdata = data.frame(row = 1))[1, 1]
  cat(sprintf("\nmu = %g, shape = %g, hu = %g: %d draws, %d positive\n", mu, shape, hu, length(yrep), length(pos)))
  cat(sprintf("  share of zeros         brms %.4f   exact %.4f\n", mean(yrep == 0), hu))
  cat(sprintf("  P(y = 1 | y > 0)       brms %.4f   exact %.4f   (%.0f standard errors apart)\n",
              mean(pos == 1), p1_exact, abs(mean(pos == 1) - p1_exact) / se_p1))
  cat(sprintf("  E[y | y > 0]           brms %.4f   exact %.4f   (%.0f standard errors apart)\n",
              mean(pos), mean_exact, abs(mean(pos) - mean_exact) / se_mean))
  cat(sprintf("  posterior_epred        brms %.10g   exact (1 - hu) mu / (1 - NB(0)) = %.10g\n", epred, (1 - hu) * mean_exact))
  if (abs(mean(pos == 1) - p1_exact) > 6 * se_p1 || abs(mean(pos) - mean_exact) > 6 * se_mean) present <- TRUE
}
cat(if (present) "\nPROBLEM PRESENT: posterior_predict's positive draws are not zero-truncated negative binomial\n"
    else "\nproblem absent\n")
cat(sprintf("run time %.1f s\n", proc.time()[["elapsed"]] - t_start))
quit(status = if (present) 1 else 0)
