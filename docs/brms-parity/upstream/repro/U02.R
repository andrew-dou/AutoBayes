#!/usr/bin/env Rscript
# U02 (D121): posterior_epred() of a count response with trunc(lb = ...) is not the mean of the truncated distribution the
# likelihood uses: it starts the support at lb + 1 and cuts it off at a finite upper end.
#
#   Rscript docs/brms-parity/upstream/repro/U02.R      # exit status 1 while the problem is present
#
# Stock CRAN brms (rstan backend) only. A negative binomial model y | trunc(lb = 1) ~ 1 is evaluated at fixed points
# (fixed-parameter runs, nothing sampled). The likelihood is normalised by P(Y >= 1) (checked below with log_prob), so the
# expected value is E[Y | Y >= 1] = mu / (1 - NB(0)). The script also prints E[Y | Y >= 2] and the upper end M at which the
# mean over 2..M equals brms's value.
t_start <- proc.time()[["elapsed"]]
suppressPackageStartupMessages({ library(brms); library(rstan) })
cat(sprintf("brms %s, rstan %s, StanHeaders %s, R %s\n", packageVersion("brms"), packageVersion("rstan"),
            packageVersion("StanHeaders"), getRversion()))
pri <- c(set_prior("normal(0, 10)", class = "Intercept"), set_prior("gamma(1, 1)", class = "shape"))
d <- data.frame(y = c(1, 2, 5, 3))
fit0 <- brm(y | trunc(lb = 1) ~ 1, data = d, family = negbinomial(), prior = pri, chains = 0, backend = "rstan",
            silent = 2, refresh = 0)
# The likelihood: log_prob without the prior equals sum log NB(y) - log P(Y >= 1).
u <- unconstrain_pars(fit0$fit, list(Intercept = log(1.5), shape = 1))
ll_brms <- as.numeric(log_prob(fit0$fit, u, adjust_transform = FALSE)) - constrain_pars(fit0$fit, u)$lprior
ll_exact <- sum(dnbinom(d$y, size = 1, mu = 1.5, log = TRUE)) - length(d$y) * log1p(-dnbinom(0, size = 1, mu = 1.5))
cat(sprintf("likelihood at mu = 1.5, shape = 1: brms %.15g, sum log NB(y) - N log P(Y >= 1) = %.15g\n", ll_brms, ll_exact))
mean_from <- function(lb, M, mu, shape) { k <- lb:M; p <- dnbinom(k, size = shape, mu = mu); sum(k * p) / sum(p) }
present <- FALSE
for (cfg in list(list(mu = 1.5, shape = 1, data = d), list(mu = 134, shape = 0.6, data = d),
                 list(mu = 134, shape = 0.6, data = data.frame(y = c(1, 2, 5, 300))))) {
  mu <- cfg$mu; shape <- cfg$shape
  fit <- update(fit0, newdata = cfg$data, algorithm = "fixed_param", chains = 1, iter = 2, warmup = 0, seed = 1,
                recompile = FALSE, init = list(list(Intercept = log(mu), shape = shape)), silent = 2, refresh = 0)
  e <- posterior_epred(fit, newdata = data.frame(row = 1:2))[1, 1]
  p0 <- dnbinom(0, size = shape, mu = mu); p1 <- dnbinom(1, size = shape, mu = mu)
  e1 <- mu / (1 - p0); e2 <- (mu - p1) / (1 - p0 - p1)
  Ms <- 2:20000; pk <- dnbinom(Ms, size = shape, mu = mu); mm <- cumsum(Ms * pk) / cumsum(pk)
  M <- Ms[which.min(abs(mm - e))]
  cat(sprintf("\nmu = %g, shape = %g, max(y) in the data = %g\n", mu, shape, max(cfg$data$y)))
  cat(sprintf("  posterior_epred brms        %.10g\n", e))
  cat(sprintf("  exact E[Y | Y >= 1]         %.10g   (relative difference %.3g)\n", e1, abs(e - e1) / e1))
  cat(sprintf("  E[Y | Y >= 2]               %.10g\n", e2))
  cat(sprintf("  mean over 2..M closest to brms's value at M = %d: %.10g\n", M, mean_from(2, M, mu, shape)))
  if (abs(e - e1) / e1 > 1e-9) present <- TRUE
}
cat(if (present) "\nPROBLEM PRESENT: posterior_epred of trunc(lb = 1) is not E[Y | Y >= 1]\n" else "\nproblem absent\n")
cat(sprintf("run time %.1f s\n", proc.time()[["elapsed"]] - t_start))
quit(status = if (present) 1 else 0)
