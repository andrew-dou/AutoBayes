#!/usr/bin/env Rscript
# U03 (D132): cumulative("cloglog") forms the upper categories' probabilities by subtraction, 1 - inv_cloglog(d) for the
# last and inv_cloglog(d_k) - inv_cloglog(d_{k-1}) for the others, which loses the digits of an improbable category and is
# log(0) = -Inf once inv_cloglog(d) rounds to 1 (d above about 3.6).
#
#   Rscript docs/brms-parity/upstream/repro/U03.R      # exit status 1 while the problem is present
#
# Stock CRAN brms (rstan backend) only. An intercept-only cumulative cloglog model with one row in each of four categories,
# evaluated with rstan::log_prob at fixed thresholds (no Jacobian; the prior's part is removed by evaluating the same compiled
# model with its prior_only flag set). Exact: with F(x) = 1 - exp(-exp(x)), log P(y = 1) = log(-expm1(-exp(t1))),
# log P(y = k) = -exp(t_{k-1}) + log(-expm1(-(exp(t_k) - exp(t_{k-1})))), log P(y = K) = -exp(t_{K-1}).
t_start <- proc.time()[["elapsed"]]
suppressPackageStartupMessages({ library(brms); library(rstan) })
cat(sprintf("brms %s, rstan %s, StanHeaders %s, R %s\n", packageVersion("brms"), packageVersion("rstan"),
            packageVersion("StanHeaders"), getRversion()))
d <- data.frame(y = 1:4)
fit0 <- brm(y ~ 1, data = d, family = cumulative("cloglog"), prior = set_prior("normal(0, 5)", class = "Intercept"),
            chains = 0, backend = "rstan", silent = 2, refresh = 0)
sf <- fit0$fit
sf_prior <- sampling(sf@stanmodel, data = modifyList(standata(fit0), list(prior_only = 1L)), chains = 0)
lik <- function(u) {                                  # brms's log likelihood and its gradient in the unconstrained thresholds
  a <- log_prob(sf, u, adjust_transform = FALSE, gradient = TRUE)
  b <- log_prob(sf_prior, u, adjust_transform = FALSE, gradient = TRUE)
  list(value = as.numeric(a) - as.numeric(b), grad = attr(a, "gradient") - attr(b, "gradient"))
}
exact <- function(t) {                                # exact log likelihood and gradient in t, then in u (ordered transform)
  e <- exp(t); K <- length(t) + 1
  f <- function(x) exp(x - exp(x))                    # F'(x)
  lp <- numeric(K); g <- numeric(length(t))
  lp[1] <- log(-expm1(-e[1])); g[1] <- g[1] + f(t[1]) / -expm1(-e[1])
  for (k in 2:(K - 1)) {
    P <- exp(-e[k - 1]) * -expm1(-(e[k] - e[k - 1]))
    lp[k] <- -e[k - 1] + log(-expm1(-(e[k] - e[k - 1])))
    g[k] <- g[k] + f(t[k]) / P; g[k - 1] <- g[k - 1] - f(t[k - 1]) / P
  }
  lp[K] <- -e[K - 1]; g[K - 1] <- g[K - 1] - e[K - 1]
  gu <- c(sum(g), vapply(2:length(t), function(j) (t[j] - t[j - 1]) * sum(g[j:length(t)]), 0))
  list(rows = lp, value = sum(lp), grad = gu)
}
present <- FALSE; control_ok <- TRUE
for (t in list(c(-1, 0, 1), c(-1, 3, 3.375), c(-1, 3, 4))) {
  u <- unconstrain_pars(sf, list(Intercept = t))
  b <- lik(u); x <- exact(t)
  rel_v <- abs(b$value - x$value) / abs(x$value); rel_g <- max(abs(b$grad - x$grad) / abs(x$grad))
  cat(sprintf("\nthresholds (%s): exact P(y = k) = %s\n", paste(t, collapse = ", "),
              paste(sprintf("%.3g", exp(x$rows)), collapse = ", ")))
  cat(sprintf("  log likelihood  brms % .16g   exact % .16g   relative error %.2g\n", b$value, x$value, rel_v))
  cat(sprintf("  gradient        brms %s\n                  exact %s   largest relative error %.2g\n",
              paste(sprintf("% .12g", b$grad), collapse = " "), paste(sprintf("% .12g", x$grad), collapse = " "), rel_g))
  bad <- !is.finite(b$value) || rel_v > 1e-12 || !all(is.finite(b$grad)) || rel_g > 1e-8
  if (identical(t, c(-1, 0, 1))) control_ok <- !bad else if (bad) present <- TRUE
}
if (!control_ok) { cat("\nthe exact formula does not match brms at the control point: the script is wrong\n"); quit(status = 2) }
cat(if (present) "\nPROBLEM PRESENT: brms's cumulative cloglog log density loses the improbable categories\n" else "\nproblem absent\n")
cat(sprintf("run time %.1f s\n", proc.time()[["elapsed"]] - t_start))
quit(status = if (present) 1 else 0)
