#!/usr/bin/env Rscript
# U17 (D134, D115): under links that do not keep the mean in the family's support (negbinomial("identity"),
# Gamma("inverse"), the Gamma family's default link, Gamma("identity"), inverse.gaussian("1/mu^2")), a new row whose linear
# predictor is not positive gets a negative mean or NaN from posterior_epred() without an error or a warning, and NaN draws
# from posterior_predict() with only R's random-number warnings. Stan rejects such a draw in the model, so these values
# describe no distribution of the family.
#
#   Rscript docs/brms-parity/upstream/repro/U17.R      # exit status 1 while the problem is present
#
# Stock CRAN brms (rstan backend) only. Each model y ~ x is a fixed-parameter run (nothing sampled) at b_Intercept = 1,
# b_x = 1, so the fitted rows (x = 0, 1, 2) have eta = 1 + x > 0, and the new row x = -2.33 has eta = -1.33. Expected: an
# error, or NA with a warning naming the row, as for any input outside the family's support.
t_start <- proc.time()[["elapsed"]]
suppressPackageStartupMessages({ library(brms); library(rstan) })
cat(sprintf("brms %s, rstan %s, StanHeaders %s, R %s\n", packageVersion("brms"), packageVersion("rstan"),
            packageVersion("StanHeaders"), getRversion()))
d <- data.frame(y = c(1, 2, 3), x = c(0, 1, 2))
pri <- c(set_prior("normal(0, 5)", class = "Intercept"), set_prior("normal(0, 5)", class = "b"),
         set_prior("gamma(2, 1)", class = "shape"))
init <- list(Intercept = 1 + 1 * mean(d$x), b = array(1), shape = 2)  # Stan's Intercept is the centred one
nd <- data.frame(x = c(1, -2.33))
caught <- function(expr) {                            # value, warnings and error of an expression
  w <- character(0)
  v <- withCallingHandlers(tryCatch(expr, error = function(e) structure(NA, err = conditionMessage(e))),
                           warning = function(c) { w <<- c(w, conditionMessage(c)); invokeRestart("muffleWarning") })
  list(value = v, warnings = unique(w), error = attr(v, "err"))
}
present <- FALSE
for (cs in list(list(name = "negbinomial(\"identity\")", fam = negbinomial("identity"), mean = function(e) e),
                list(name = "Gamma(\"inverse\")", fam = Gamma("inverse"), mean = function(e) 1 / e),
                list(name = "Gamma(\"identity\")", fam = Gamma("identity"), mean = function(e) e),
                list(name = "inverse.gaussian(\"1/mu^2\")", fam = inverse.gaussian("1/mu^2"), mean = function(e) 1 / sqrt(e)))) {
  fit <- brm(y ~ x, data = d, family = cs$fam, prior = pri, backend = "rstan", algorithm = "fixed_param", chains = 1,
             iter = 2, warmup = 0, seed = 1, init = list(init), silent = 2, refresh = 0)
  ep <- caught(posterior_epred(fit, newdata = nd))
  set.seed(1)
  pp <- caught(posterior_predict(fit, newdata = nd))
  eta <- 1 + nd$x
  cat(sprintf("\n%s: eta = %s\n", cs$name, paste(eta, collapse = ", ")))
  cat(sprintf("  posterior_epred:   %s   (the link's inverse of eta: %s)   error: %s   warnings: %s\n",
              paste(sprintf("%.6g", ep$value[1, ]), collapse = ", "), paste(sprintf("%.6g", suppressWarnings(cs$mean(eta))), collapse = ", "),
              if (is.null(ep$error)) "none" else ep$error, if (length(ep$warnings)) paste(ep$warnings, collapse = "; ") else "none"))
  cat(sprintf("  posterior_predict: %s   error: %s   warnings: %s\n", paste(sprintf("%.6g", pp$value[1, ]), collapse = ", "),
              if (is.null(pp$error)) "none" else pp$error, if (length(pp$warnings)) paste(pp$warnings, collapse = "; ") else "none"))
  bad_row <- ep$value[1, 2]
  if (is.null(ep$error) && length(ep$warnings) == 0 && (is.nan(bad_row) || bad_row <= 0)) present <- TRUE
}
cat(if (present) "\nPROBLEM PRESENT: posterior_epred returns means outside the family's support without a word\n" else "\nproblem absent\n")
cat(sprintf("run time %.1f s\n", proc.time()[["elapsed"]] - t_start))
quit(status = if (present) 1 else 0)
