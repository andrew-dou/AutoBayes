#!/usr/bin/env Rscript
# U20 (D66): the approximate Gaussian process gp(x, k = ...) weights basis function j by the square root of the spectral
# density, sqrt(S(w_j)) with S(w) = sdgp^2 sqrt(2 pi) lscale exp(-lscale^2 w^2 / 2). Where S underflows to 0
# (lscale * w_j above about 38.6) the log density is finite but its gradient in sdgp and lscale is NaN (the chain rule of
# sqrt at 0, Inf times 0). With k = 30 basis functions and c = 5/4 that is every lscale above about 1.02 on brms's scaled
# inputs, a smooth function's ordinary length-scale, where a sampler gets NaN gradients.
#
#   Rscript docs/brms-parity/upstream/repro/U20.R      # exit status 1 while the problem is present
#
# Stock CRAN brms (rstan backend) only. y ~ gp(x, k = 30, c = 5/4) on 12 rows, evaluated with rstan::log_prob and
# grad_log_prob (no Jacobian) at three length-scales. Expected: the gradient of the (finite, smooth) log density, here a
# central finite difference of brms's own log_prob with step 1e-6 in each unconstrained coordinate.
t_start <- proc.time()[["elapsed"]]
suppressPackageStartupMessages({ library(brms); library(rstan) })
cat(sprintf("brms %s, rstan %s, StanHeaders %s, R %s\n", packageVersion("brms"), packageVersion("rstan"),
            packageVersion("StanHeaders"), getRversion()))
set.seed(1)
d <- data.frame(x = seq(0, 1, length.out = 12)); d$y <- sin(3 * d$x) + rnorm(12, 0, 0.1)
k <- 30
pri <- c(set_prior("normal(0, 5)", class = "Intercept"), set_prior("normal(0, 1)", class = "sigma"),
         set_prior("normal(0, 1)", class = "sdgp"), set_prior("inv_gamma(2, 1)", class = "lscale", coef = "gpx"))
fit0 <- brm(y ~ gp(x, k = k, c = 5/4), data = d, prior = pri, chains = 0, backend = "rstan", silent = 2, refresh = 0)
sf <- fit0$fit
w <- as.vector(standata(fit0)$slambda_1)
cat(sprintf("largest basis frequency w_%d = %.4f: S underflows for lscale above %.3f\n", k, max(w), sqrt(2 * 745.13) / max(w)))
present <- FALSE; control_ok <- TRUE
for (ls in c(0.5, 1.2, 2)) {
  u <- unconstrain_pars(sf, list(Intercept = 0, sdgp_1 = array(0.8), lscale_1 = array(ls, dim = c(1, 1)),
                                 zgp_1 = rep(0.1, k), sigma = 0.3))
  r <- log_prob(sf, u, adjust_transform = FALSE, gradient = TRUE); g <- attr(r, "gradient")
  fd <- vapply(seq_along(u), function(i) { h <- 1e-6; e <- replace(numeric(length(u)), i, h)
    (log_prob(sf, u + e, adjust_transform = FALSE) - log_prob(sf, u - e, adjust_transform = FALSE)) / (2 * h) }, 0)
  cat(sprintf("\nlscale = %g: log density %.12g\n", ls, as.numeric(r)))
  cat(sprintf("  gradient in (log sdgp, log lscale)  brms %s   finite difference %s\n",
              paste(sprintf("% .8g", g[2:3]), collapse = ", "), paste(sprintf("% .8g", fd[2:3]), collapse = ", ")))
  cat(sprintf("  non-finite gradient entries: %d of %d; largest difference from the finite difference elsewhere %.2g\n",
              sum(!is.finite(g)), length(g), max(abs(g - fd)[is.finite(g)])))
  bad <- is.finite(as.numeric(r)) && (any(!is.finite(g)) || max(abs(g - fd)) > 1e-4)
  if (ls == 0.5) control_ok <- !bad else if (bad) present <- TRUE
}
if (!control_ok) { cat("\nbrms's gradient does not match the finite difference at the control: the script is wrong\n"); quit(status = 2) }
cat(if (present) "\nPROBLEM PRESENT: the approximate GP's gradient is NaN where the spectral density underflows\n" else "\nproblem absent\n")
cat(sprintf("run time %.1f s\n", proc.time()[["elapsed"]] - t_start))
quit(status = if (present) 1 else 0)
