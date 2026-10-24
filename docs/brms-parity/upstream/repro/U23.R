#!/usr/bin/env Rscript
# U23 (D24, D25): Stan's normal_lcdf (and std_normal_lcdf, and normal_lccdf in its upper tail) returns accurate values but
# inexact derivatives: its gradient is off by up to 2e-5 relative (z = -10), 1e-5 (z = -38), 6e-6 (z = 2) and 7e-7 (z = 1)
# from phi(z) / Phi(z), and normal_lccdf's by 5e-8 at z = 6.17 and 2e-2 at z = 8. brms's Gaussian cens() and trunc()
# models differentiate through them, so their gradients in mu and sigma carry these errors.
#
#   Rscript docs/brms-parity/upstream/repro/U23.R      # exit status 1 while the problem is present
#
# Stock CRAN brms (rstan backend) only. y | cens(c) ~ 1 at mu = 0, sigma = 1, one censored row each, evaluated with
# rstan::grad_log_prob (no Jacobian, the prior's part removed through the model's prior_only flag). Exact, with the Mills
# ratio m(z) = exp(dnorm(z, log = TRUE) - pnorm(z, log.p = TRUE)): a left-censored row (log Phi(z), c = -1) has gradient
# (-m(z), -z m(z)) in (mu, log sigma), a right-censored row (log(1 - Phi(z)), c = 1) (m(-z), z m(-z)).
t_start <- proc.time()[["elapsed"]]
suppressPackageStartupMessages({ library(brms); library(rstan) })
cat(sprintf("brms %s, rstan %s, StanHeaders %s, R %s\n", packageVersion("brms"), packageVersion("rstan"),
            packageVersion("StanHeaders"), getRversion()))
pri <- c(set_prior("normal(0, 5)", class = "Intercept"), set_prior("normal(0, 2)", class = "sigma"))
fit0 <- brm(y | cens(c) ~ 1, data = data.frame(y = 1, c = 1), prior = pri, chains = 0, backend = "rstan", silent = 2, refresh = 0)
cat(sprintf("the generated program calls normal_lcdf: %s, normal_lccdf: %s\n", grepl("normal_lcdf", stancode(fit0), fixed = TRUE),
            grepl("normal_lccdf", stancode(fit0), fixed = TRUE)))
sf_prior <- sampling(fit0$fit@stanmodel, data = modifyList(standata(fit0), list(prior_only = 1L)), chains = 0)
grad_at <- function(row) {
  f <- update(fit0, newdata = row, chains = 0, recompile = FALSE, silent = 2, refresh = 0)
  u <- unconstrain_pars(f$fit, list(Intercept = 0, sigma = 1))
  grad_log_prob(f$fit, u, adjust_transform = FALSE) - grad_log_prob(sf_prior, u, adjust_transform = FALSE)
}
m <- function(z) exp(dnorm(z, log = TRUE) - pnorm(z, log.p = TRUE))
present <- FALSE; control_ok <- TRUE
cat("\ngradient in (mu, log sigma) of one censored row at mu = 0, sigma = 1\n")
for (cs in list(list(c = 1, z = 0, control = TRUE), list(c = -1, z = 1), list(c = -1, z = 2), list(c = -1, z = -10),
                list(c = -1, z = -38), list(c = 1, z = 6.17), list(c = 1, z = 8))) {
  g <- as.numeric(grad_at(data.frame(y = cs$z, c = cs$c)))
  x <- if (cs$c == -1) c(-m(cs$z), -cs$z * m(cs$z)) else c(m(-cs$z), cs$z * m(-cs$z))
  rel <- max(abs(g - x)[x != 0] / abs(x[x != 0]))
  cat(sprintf("  %-14s z = %5g: brms %s   exact %s   largest relative error %.2g\n",
              if (cs$c == -1) "left (lcdf)" else "right (lccdf)", cs$z, paste(sprintf("% .16g", g), collapse = ", "),
              paste(sprintf("% .16g", x), collapse = ", "), rel))
  bad <- !all(is.finite(g)) || rel > 1e-7
  if (isTRUE(cs$control)) control_ok <- !bad && rel < 1e-14 else if (bad) present <- TRUE
}
if (!control_ok) { cat("\nthe exact formula does not match brms at the control row: the script is wrong\n"); quit(status = 2) }
cat(if (present) "\nPROBLEM PRESENT: Stan's normal_lcdf / normal_lccdf derivatives are inexact, and so are brms's censored gradients\n" else "\nproblem absent\n")
cat(sprintf("run time %.1f s\n", proc.time()[["elapsed"]] - t_start))
quit(status = if (present) 1 else 0)
