#!/usr/bin/env Rscript
# Go/no-go for the brms oracle on this machine: the packages load, rstan compiles a brms model, and the
# compiled model gives a log density and gradient at an unconstrained point. Prints "DOCTOR OK" on success.
#
#   Rscript oracle/R/doctor.R
suppressPackageStartupMessages({
  library(brms)
  library(rstan)
  library(jsonlite)
})
cat(sprintf("R %s.%s | brms %s | rstan %s | Stan %s | jsonlite %s | %s\n", R.version$major, R.version$minor,
            packageVersion("brms"), packageVersion("rstan"), rstan::stan_version(), packageVersion("jsonlite"),
            R.version$platform))
set.seed(1)
d <- data.frame(y = rnorm(20), x = rnorm(20), g = factor(rep(c("a", "b", "c", "d"), 5)))
pri <- c(prior(normal(0, 5), class = Intercept), prior(normal(0, 2), class = b),
         prior(normal(0, 1), class = sd), prior(normal(0, 1), class = sigma))
t0 <- Sys.time()
fit <- brm(y ~ x + (1 | g), data = d, family = gaussian(), prior = pri, chains = 0,
           backend = "rstan", silent = 2, refresh = 0)
cat(sprintf("compile: %.1f s\n", as.numeric(Sys.time() - t0, units = "secs")))
sf <- fit$fit
n <- rstan::get_num_upars(sf)
up <- seq(-0.3, 0.3, length.out = n)
lp <- rstan::log_prob(sf, up, adjust_transform = TRUE, gradient = TRUE)
cat(sprintf("num_upars %d | log_prob %.12f | gradient[1] %.10f\n", n, as.numeric(lp), attr(lp, "gradient")[1]))
if (!is.finite(as.numeric(lp)) || any(!is.finite(attr(lp, "gradient")))) stop("non-finite log density")
js <- toJSON(list(v = I(c(pi, 1 / 3))), digits = I(17))
if (!identical(fromJSON(js)$v, c(pi, 1 / 3))) stop("jsonlite does not round-trip doubles at 17 significant digits")
cat("DOCTOR OK\n")
