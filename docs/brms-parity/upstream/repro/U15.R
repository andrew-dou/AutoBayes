#!/usr/bin/env Rscript
# U15 (D61): a non-linear formula is written into the Stan program as it stands and evaluated in R for predictions, so a
# division of two integer literals means different things in the two: Stan's integer division (1 / 2 is 0) in the model
# that is fitted, R's real division (0.5) in posterior_epred(). y ~ (1 / 2) * a fits mu = 0 and predicts mu = a / 2.
#
#   Rscript docs/brms-parity/upstream/repro/U15.R      # exit status 1 while the problem is present
#
# Stock CRAN brms (rstan backend) only. bf(y ~ (1 / 2) * a, a ~ 1, nl = TRUE), Gaussian, one row y = 1, at a = 2,
# sigma = 1: the likelihood from rstan::log_prob (no Jacobian, the prior's part removed through the model's prior_only flag)
# and the mean from posterior_epred() of a fixed-parameter run. Exact, as R reads the formula: mu = 0.5 * a = 1, so the
# log likelihood is dnorm(1, 1, 1, log = TRUE) and the mean 1.
t_start <- proc.time()[["elapsed"]]
suppressPackageStartupMessages({ library(brms); library(rstan) })
cat(sprintf("brms %s, rstan %s, StanHeaders %s, R %s\n", packageVersion("brms"), packageVersion("rstan"),
            packageVersion("StanHeaders"), getRversion()))
pri <- c(set_prior("normal(0, 5)", nlpar = "a"), set_prior("normal(0, 2)", class = "sigma"))
fit <- brm(bf(y ~ (1 / 2) * a, a ~ 1, nl = TRUE), data = data.frame(y = 1), prior = pri, backend = "rstan",
           algorithm = "fixed_param", chains = 1, iter = 2, warmup = 0, seed = 1,
           init = list(list(b_a = array(2), sigma = 1)), silent = 2, refresh = 0)
sf <- fit$fit
sf_prior <- sampling(sf@stanmodel, data = modifyList(standata(fit), list(prior_only = 1L)), chains = 0)
u <- unconstrain_pars(sf, list(b_a = array(2), sigma = 1))
ll <- as.numeric(log_prob(sf, u, adjust_transform = FALSE)) - as.numeric(log_prob(sf_prior, u, adjust_transform = FALSE))
e <- posterior_epred(fit, newdata = data.frame(row = 1:2))[1, 1]
cat(sprintf("\nat a = 2, sigma = 1, y = 1\n"))
cat(sprintf("  fitted log likelihood (Stan)  brms %.16g   with mu = 0.5 * a: %.16g   with mu = 0 (integer 1 / 2): %.16g\n",
            ll, dnorm(1, 1, 1, log = TRUE), dnorm(1, 0, 1, log = TRUE)))
cat(sprintf("  posterior_epred (R)           brms %.16g   with mu = 0.5 * a: %.16g\n", e, 1))
present <- abs(ll - dnorm(1, 1, 1, log = TRUE)) > 1e-12 || abs(e - 1) > 1e-12
cat(if (present) "\nPROBLEM PRESENT: the fitted model and brms's predictions read (1 / 2) * a differently\n" else "\nproblem absent\n")
cat(sprintf("run time %.1f s\n", proc.time()[["elapsed"]] - t_start))
quit(status = if (present) 1 else 0)
