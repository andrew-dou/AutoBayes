#!/usr/bin/env Rscript
# U25 (D120): Stan's gamma_lccdf (StanHeaders 2.32) has an inexact derivative in the shape: off by up to 1.2e-2 relative
# here, so the gradient of brms's right-censored Gamma rows in log(shape) is off too (up to 2.8e-5 on the frozen case
# gamma_censored_interval). gamma_lcdf's shape derivative is accurate (the left-censored rows are the control; brms's
# interval rows, built from gamma_lcdf, are accurate here too).
#
#   Rscript docs/brms-parity/upstream/repro/U25.R      # exit status 1 while the problem is present
#
# Stock CRAN brms (rstan backend) only. Gamma("log") models y | cens(c, y2) ~ 1 with one censored row each, evaluated with
# rstan::grad_log_prob (no Jacobian, the prior's part removed through the model's prior_only flag). brms's Gamma has
# rate = shape / mu. Exact: a fourth-order central difference (step 1e-4) of R's pgamma(., log.p = TRUE) in
# (log mu, log shape), each tail from pgamma's own lower or upper tail, accurate to about 1e-11.
t_start <- proc.time()[["elapsed"]]
suppressPackageStartupMessages({ library(brms); library(rstan) })
cat(sprintf("brms %s, rstan %s, StanHeaders %s, R %s\n", packageVersion("brms"), packageVersion("rstan"),
            packageVersion("StanHeaders"), getRversion()))
pri <- c(set_prior("normal(0, 5)", class = "Intercept"), set_prior("gamma(2, 1)", class = "shape"))
fit0 <- brm(y | cens(c, y2) ~ 1, data = data.frame(y = 1, y2 = 2, c = 2), family = Gamma("log"), prior = pri, chains = 0,
            backend = "rstan", silent = 2, refresh = 0)
code <- stancode(fit0)
cat(sprintf("the generated program calls gamma_lcdf: %s, gamma_lccdf: %s\n", grepl("gamma_lcdf", code, fixed = TRUE),
            grepl("gamma_lccdf", code, fixed = TRUE)))
sf_prior <- sampling(fit0$fit@stanmodel, data = modifyList(standata(fit0), list(prior_only = 1L)), chains = 0)
row_lik <- function(c, y, y2) function(lmu, lshape) {    # exact log likelihood of one censored row in (log mu, log shape)
  s <- exp(lshape); r <- s / exp(lmu)
  lo <- function(v) pgamma(v, s, rate = r, log.p = TRUE); up <- function(v) pgamma(v, s, rate = r, lower.tail = FALSE, log.p = TRUE)
  if (c == -1) lo(y) else if (c == 1) up(y) else { a <- up(y); b <- up(y2); a + log(-expm1(b - a)) }
}
fd <- function(f, p, i, h = 1e-4) {
  e <- replace(c(0, 0), i, h); g <- function(d) f(p[1] + d[1], p[2] + d[2])
  (-g(2 * e) + 8 * g(e) - 8 * g(-e) + g(-2 * e)) / (12 * h)
}
present <- FALSE; control_ok <- TRUE
cat("\ngradient in (log mu, log shape) of one censored row\n")
for (cs in list(list(c = -1, y = 1, y2 = 0, mu = 2, shape = 2), list(c = -1, y = 5, y2 = 0, mu = 5, shape = 5),
                list(c = 1, y = 3, y2 = 0, mu = 2, shape = 2), list(c = 1, y = 7, y2 = 0, mu = 5, shape = 5),
                list(c = 1, y = 2, y2 = 0, mu = 5, shape = 10), list(c = 2, y = 2, y2 = 4, mu = 3, shape = 3))) {
  f <- update(fit0, newdata = data.frame(y = cs$y, y2 = cs$y2, c = cs$c), chains = 0, recompile = FALSE, silent = 2, refresh = 0)
  u <- unconstrain_pars(f$fit, list(Intercept = log(cs$mu), shape = cs$shape))
  g <- as.numeric(grad_log_prob(f$fit, u, adjust_transform = FALSE) - grad_log_prob(sf_prior, u, adjust_transform = FALSE))
  lf <- row_lik(cs$c, cs$y, cs$y2); p <- c(log(cs$mu), log(cs$shape))
  x <- c(fd(lf, p, 1), fd(lf, p, 2)); rel <- abs(g - x) / abs(x)
  kind <- c("-1" = "left (lcdf)", "1" = "right (lccdf)", "2" = "interval")[[as.character(cs$c)]]
  cat(sprintf("  %-14s y = %g%s, mu = %g, shape = %g: brms %s   exact %s   relative error %s\n", kind, cs$y,
              if (cs$c == 2) sprintf(" to %g", cs$y2) else "", cs$mu, cs$shape, paste(sprintf("% .13g", g), collapse = ", "),
              paste(sprintf("% .13g", x), collapse = ", "), paste(sprintf("%.2g", rel), collapse = ", ")))
  bad <- !all(is.finite(g)) || max(rel) > 1e-7
  if (cs$c == -1) control_ok <- control_ok && max(rel) < 1e-9 else if (bad) present <- TRUE
}
if (!control_ok) { cat("\nthe exact derivative does not match brms at a control row: the script is wrong\n"); quit(status = 2) }
cat(if (present) "\nPROBLEM PRESENT: Stan's gamma_lccdf shape derivative is inexact, and so are brms's censored Gamma gradients\n" else "\nproblem absent\n")
cat(sprintf("run time %.1f s\n", proc.time()[["elapsed"]] - t_start))
quit(status = if (present) 1 else 0)
