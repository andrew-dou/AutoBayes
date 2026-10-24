#!/usr/bin/env Rscript
# U24 (D88): cumulative("probit"), which brms writes with Stan's ordered_probit_lpmf, has an exact log density but an
# inexact gradient: here off by up to about 1e-5 relative in the thresholds (6.5e-5 on the frozen case
# cumulative_probit_varint). The size and pattern match U23's normal_lcdf derivative, which is a likely common cause (not
# verified: Stan's code was not read).
#
#   Rscript docs/brms-parity/upstream/repro/U24.R      # exit status 1 while the problem is present
#
# Stock CRAN brms (rstan backend) only. An intercept-only cumulative probit model with one row in each of four categories,
# evaluated with rstan::log_prob (no Jacobian, the prior's part removed through the model's prior_only flag) at fixed
# thresholds. Exact: log P(y = k) = log(Phi(t_k) - Phi(t_{k-1})) with its gradient phi(t_k) / P and -phi(t_{k-1}) / P,
# taken through Stan's ordered transform (t_1 = u_1, t_k = t_{k-1} + exp(u_k)); the log density is the control.
t_start <- proc.time()[["elapsed"]]
suppressPackageStartupMessages({ library(brms); library(rstan) })
cat(sprintf("brms %s, rstan %s, StanHeaders %s, R %s\n", packageVersion("brms"), packageVersion("rstan"),
            packageVersion("StanHeaders"), getRversion()))
fit0 <- brm(y ~ 1, data = data.frame(y = 1:4), family = cumulative("probit"), prior = set_prior("normal(0, 5)", class = "Intercept"),
            chains = 0, backend = "rstan", silent = 2, refresh = 0)
cat(sprintf("the generated program calls ordered_probit_lpmf: %s\n", grepl("ordered_probit", stancode(fit0), fixed = TRUE)))
sf <- fit0$fit
sf_prior <- sampling(sf@stanmodel, data = modifyList(standata(fit0), list(prior_only = 1L)), chains = 0)
cdf <- function(lo, hi) {                             # Phi(hi) - Phi(lo) from the tail both lie in
  if (lo > 0) pnorm(lo, lower.tail = FALSE) - pnorm(hi, lower.tail = FALSE) else pnorm(hi) - pnorm(lo)
}
exact <- function(t) {
  tt <- c(-Inf, t, Inf); g <- numeric(length(t)); v <- 0
  for (k in 1:4) {
    P <- cdf(tt[k], tt[k + 1]); v <- v + log(P)
    if (k <= 3) g[k] <- g[k] + dnorm(tt[k + 1]) / P
    if (k >= 2) g[k - 1] <- g[k - 1] - dnorm(tt[k]) / P
  }
  list(value = v, grad = c(sum(g), vapply(2:3, function(j) (t[j] - t[j - 1]) * sum(g[j:3]), 0)))
}
present <- FALSE; control_ok <- TRUE
for (t in list(c(-2, 0.5, 2.5), c(-1, 1, 2), c(-6, -4.5, 0.3))) {
  u <- unconstrain_pars(sf, list(Intercept = t))
  a <- log_prob(sf, u, adjust_transform = FALSE, gradient = TRUE); b <- log_prob(sf_prior, u, adjust_transform = FALSE, gradient = TRUE)
  v <- as.numeric(a) - as.numeric(b); g <- attr(a, "gradient") - attr(b, "gradient"); x <- exact(t)
  rv <- abs(v - x$value) / abs(x$value); rg <- max(abs(g - x$grad) / abs(x$grad))
  cat(sprintf("\nthresholds (%s)\n  log likelihood brms % .16g   exact % .16g   relative error %.2g\n", paste(t, collapse = ", "),
              v, x$value, rv))
  cat(sprintf("  gradient       brms %s\n                 exact %s   largest relative error %.2g\n",
              paste(sprintf("% .14g", g), collapse = " "), paste(sprintf("% .14g", x$grad), collapse = " "), rg))
  control_ok <- control_ok && rv < 1e-13
  if (!all(is.finite(g)) || rg > 1e-7) present <- TRUE
}
if (!control_ok) { cat("\nthe exact log density does not match brms: the script is wrong\n"); quit(status = 2) }
cat(if (present) "\nPROBLEM PRESENT: the cumulative probit gradient (Stan's ordered_probit_lpmf) is inexact\n" else "\nproblem absent\n")
cat(sprintf("run time %.1f s\n", proc.time()[["elapsed"]] - t_start))
quit(status = if (present) 1 else 0)
