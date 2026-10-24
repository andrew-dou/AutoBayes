#!/usr/bin/env Rscript
# U27 (D77): Stan's skew_normal_lpdf and exp_mod_normal_lpdf (StanHeaders 2.32), which brms's skew_normal() and
# exgaussian() families call, are -Inf once their normal-CDF factor underflows (argument below about -37.5: a skew-normal
# row far on its short side, an ex-Gaussian row far below its mean), where the log density is an ordinary number that
# log Phi(x) = pnorm(x, log.p = TRUE) gives.
#
#   Rscript docs/brms-parity/upstream/repro/U27.R      # exit status 1 while the problem is present
#
# Stock CRAN brms (rstan backend) only. Intercept-only models with one row each, evaluated with rstan::log_prob (no
# Jacobian, the prior's part removed through the model's prior_only flag). brms's parameterisations (mean mu, sd sigma):
# skew normal with delta = alpha / sqrt(1 + alpha^2), omega = sigma / sqrt(1 - 2 delta^2 / pi), xi = mu - omega delta
# sqrt(2 / pi), log density log 2 - log omega + log phi(z) + log Phi(alpha z), z = (y - xi) / omega; ex-Gaussian with
# mean mu, sigma and beta = 1 / lambda, the normal part's mean m = mu - beta, log density
# -log beta + (m - y) / beta + sigma^2 / (2 beta^2) + log Phi((y - m) / sigma - sigma / beta).
t_start <- proc.time()[["elapsed"]]
suppressPackageStartupMessages({ library(brms); library(rstan) })
cat(sprintf("brms %s, rstan %s, StanHeaders %s, R %s\n", packageVersion("brms"), packageVersion("rstan"),
            packageVersion("StanHeaders"), getRversion()))
present <- FALSE; control_ok <- TRUE
run <- function(name, fam, fun, extra_prior, pars, ys, exact) {
  pri <- c(set_prior("normal(0, 5)", class = "Intercept"), set_prior("normal(0, 2)", class = "sigma"), extra_prior)
  fit0 <- brm(y ~ 1, data = data.frame(y = ys[1]), family = fam, prior = pri, chains = 0, backend = "rstan", silent = 2, refresh = 0)
  cat(sprintf("\n%s (the generated program calls %s: %s)\n", name, fun, grepl(fun, stancode(fit0), fixed = TRUE)))
  sf_prior <- sampling(fit0$fit@stanmodel, data = modifyList(standata(fit0), list(prior_only = 1L)), chains = 0)
  for (y in ys) {
    f <- update(fit0, newdata = data.frame(y = y), chains = 0, recompile = FALSE, silent = 2, refresh = 0)
    u <- unconstrain_pars(f$fit, pars)
    b <- as.numeric(log_prob(f$fit, u, adjust_transform = FALSE)) - as.numeric(log_prob(sf_prior, u, adjust_transform = FALSE))
    x <- exact(y); rel <- abs(b - x) / abs(x)
    cat(sprintf("  y = %5g: brms % .16g   exact % .16g   relative error %.2g\n", y, b, x, rel))
    bad <- !is.finite(b) || rel > 1e-12
    if (y == ys[1]) control_ok <<- control_ok && !bad else if (bad) present <<- TRUE
  }
}
# skew normal: mu = 0, sigma = 1, alpha = 10
alpha <- 10; delta <- alpha / sqrt(1 + alpha^2); omega <- 1 / sqrt(1 - 2 * delta^2 / pi); xi <- -omega * delta * sqrt(2 / pi)
run("skew_normal(), mu = 0, sigma = 1, alpha = 10", skew_normal(), "skew_normal_lpdf", set_prior("normal(0, 10)", class = "alpha"),
    list(Intercept = 0, sigma = 1, alpha = alpha), c(0.5, -5, -8, -9),
    function(y) { z <- (y - xi) / omega; log(2) - log(omega) + dnorm(z, log = TRUE) + pnorm(alpha * z, log.p = TRUE) })
# ex-Gaussian: mu = 0, sigma = 1, beta = 1
run("exgaussian(), mu = 0, sigma = 1, beta = 1", exgaussian(), "exp_mod_normal_lpdf", set_prior("gamma(2, 1)", class = "beta"),
    list(Intercept = 0, sigma = 1, beta = 1), c(0.5, -30, -36, -40),
    function(y) { m <- -1; -log(1) + (m - y) / 1 + 1 / 2 + pnorm((y - m) / 1 - 1 / 1, log.p = TRUE) })
if (!control_ok) { cat("\nthe exact formula does not match brms at a control row: the script is wrong\n"); quit(status = 2) }
cat(if (present) "\nPROBLEM PRESENT: Stan's skew_normal_lpdf / exp_mod_normal_lpdf are -Inf where log Phi underflows\n" else "\nproblem absent\n")
cat(sprintf("run time %.1f s\n", proc.time()[["elapsed"]] - t_start))
quit(status = if (present) 1 else 0)
