#!/usr/bin/env Rscript
# U18 (D115): asym_laplace() with a fixed quantile, bf(y ~ ..., quantile = q), accepts q = 0 and q = 1 (brms refuses only
# values outside [0, 1]), where the asymmetric Laplace density q (1 - q) / sigma * exp(...) is 0 for every response: the
# model has log density -Inf everywhere and cannot be fitted.
#
#   Rscript docs/brms-parity/upstream/repro/U18.R      # exit status 1 while the problem is present
#
# Stock CRAN brms (rstan backend) only: make_standata() for q = 0, 1 and 1.5, brms's own dasym_laplace(), and
# rstan::log_prob of the compiled q = 0 model at an ordinary point. Expected: q must lie strictly between 0 and 1.
t_start <- proc.time()[["elapsed"]]
suppressPackageStartupMessages({ library(brms); library(rstan) })
cat(sprintf("brms %s, rstan %s, StanHeaders %s, R %s\n", packageVersion("brms"), packageVersion("rstan"),
            packageVersion("StanHeaders"), getRversion()))
d <- data.frame(y = c(-0.3, 0.1, 0.5))
accepted <- c()
for (q in c(0, 1, 1.5)) {
  r <- tryCatch({ make_standata(bf(y ~ 1, quantile = q), data = d, family = asym_laplace()); "accepted" },
                error = function(e) paste("refused:", conditionMessage(e)))
  cat(sprintf("quantile = %-4g make_standata: %s\n", q, r))
  accepted[as.character(q)] <- identical(r, "accepted")
}
cat(sprintf("brms's dasym_laplace(y, mu = 0, sigma = 1, quantile = 0) at y = -0.3, 0.1, 0.5: %s\n",
            paste(dasym_laplace(d$y, mu = 0, sigma = 1, quantile = 0), collapse = ", ")))
pri <- c(set_prior("normal(0, 5)", class = "Intercept"), set_prior("normal(0, 2)", class = "sigma"))
fit0 <- brm(bf(y ~ 1, quantile = 0), data = d, family = asym_laplace(), prior = pri, chains = 0, backend = "rstan",
            silent = 2, refresh = 0)
lp <- tryCatch(format(as.numeric(log_prob(fit0$fit, unconstrain_pars(fit0$fit, list(Intercept = 0, sigma = 1)),
                                           adjust_transform = FALSE))),
               error = function(e) paste("error:", strsplit(conditionMessage(e), "\n")[[1]][1]))
cat(sprintf("log_prob of the quantile = 0 model at Intercept = 0, sigma = 1: %s\n", lp))
present <- isTRUE(accepted[["0"]]) || isTRUE(accepted[["1"]])
cat(if (present) "\nPROBLEM PRESENT: brms accepts a fixed quantile of 0 or 1, where the density is 0 everywhere\n" else "\nproblem absent\n")
cat(sprintf("run time %.1f s\n", proc.time()[["elapsed"]] - t_start))
quit(status = if (present) 1 else 0)
