#!/usr/bin/env Rscript
# U07 (D28): the zero-inflated and hurdle Poisson densities and the hurdle Poisson mean break at extreme linear predictors:
# a hurdle y > 0 row's log density is +Inf (its true value is finite, about log(1 - hu) for y = 1) once exp(eta) underflows,
# and its gradient is infinite once exp(eta) is subnormal; a zero-inflated y = 0 row's gradient is NaN once exp(eta)
# overflows; posterior_epred of the hurdle model is NaN where mu underflows, where its limit is 1 - hu.
#
#   Rscript docs/brms-parity/upstream/repro/U07.R      # exit status 1 while the problem is present
#
# Stock CRAN brms (rstan backend) only. Models y ~ 1 + offset(o) with one row each, the linear predictor set through the
# offset, evaluated with rstan::log_prob (no Jacobian, the prior's part removed through the model's prior_only flag) and,
# for the mean, a fixed-parameter run. Exact, with mu = exp(eta): the hurdle row log(1 - hu) + y eta - mu - lgamma(y + 1)
# - log(1 - exp(-mu)), log(1 - exp(-mu)) = eta + log1p(-mu / 2) to double precision below eta = -30; the zero-inflated
# zero log(zi + (1 - zi) exp(-mu)).
t_start <- proc.time()[["elapsed"]]
suppressPackageStartupMessages({ library(brms); library(rstan) })
cat(sprintf("brms %s, rstan %s, StanHeaders %s, R %s\n", packageVersion("brms"), packageVersion("rstan"),
            packageVersion("StanHeaders"), getRversion()))
pri_hu <- c(set_prior("normal(0, 10)", class = "Intercept"), set_prior("beta(2, 2)", class = "hu"))
pri_zi <- c(set_prior("normal(0, 10)", class = "Intercept"), set_prior("beta(2, 2)", class = "zi"))
evaluator <- function(fit0, dpar) {
  sf_prior <- sampling(fit0$fit@stanmodel, data = modifyList(standata(fit0), list(prior_only = 1L)), chains = 0)
  function(row, value) {                                # the likelihood of one row at Intercept 0 and the given hu / zi
    f <- update(fit0, newdata = row, chains = 0, recompile = FALSE, silent = 2, refresh = 0)
    pars <- list(Intercept = 0); pars[[dpar]] <- value
    u <- unconstrain_pars(f$fit, pars)
    a <- log_prob(f$fit, u, adjust_transform = FALSE, gradient = TRUE); b <- log_prob(sf_prior, u, adjust_transform = FALSE, gradient = TRUE)
    list(value = as.numeric(a) - as.numeric(b), grad = attr(a, "gradient") - attr(b, "gradient"))
  }
}
log1m_exp_neg_mu <- function(eta) if (eta < -30) eta + log1p(-exp(eta) / 2) else log(-expm1(-exp(eta)))
show <- function(label, b, x) {
  rv <- abs(b$value - x$value) / max(abs(x$value), 1e-300); rg <- max(abs(b$grad - x$grad))
  cat(sprintf("  %s: log lik brms % .16g exact % .16g; gradient (eta, logit) brms %s exact %s\n", label, b$value, x$value,
              paste(sprintf("% .6g", b$grad), collapse = " "), paste(sprintf("% .6g", x$grad), collapse = " ")))
  !is.finite(b$value) || !all(is.finite(b$grad)) || rv > 1e-12 || rg > 1e-10
}
present <- FALSE; control_ok <- TRUE
hu <- 0.3; zi <- 0.3
fh <- brm(y ~ 1 + offset(o), data = data.frame(y = 1, o = 0), family = hurdle_poisson(), prior = pri_hu, chains = 0,
          backend = "rstan", silent = 2, refresh = 0)
ev <- evaluator(fh, "hu")
cat("\nhurdle_poisson, a y = 1 row\n")
for (eta in c(0, -740, -750)) {
  mu <- exp(eta)
  g_eta <- if (eta < -30) -mu / 2 else 1 - mu - mu * exp(-mu) / -expm1(-mu)
  x <- list(value = log1p(-hu) + eta - mu - log1m_exp_neg_mu(eta), grad = c(g_eta, -hu))
  bad <- show(sprintf("eta = %4g", eta), ev(data.frame(y = 1, o = eta), hu), x)
  if (eta == 0) control_ok <- control_ok && !bad else if (bad) present <- TRUE
}
fz <- brm(y ~ 1 + offset(o), data = data.frame(y = 0, o = 0), family = zero_inflated_poisson(), prior = pri_zi, chains = 0,
          backend = "rstan", silent = 2, refresh = 0)
ev <- evaluator(fz, "zi")
cat("\nzero_inflated_poisson, a y = 0 row\n")
for (eta in c(0, 710)) {
  mu <- exp(eta); q <- exp(-mu); den <- zi + (1 - zi) * q
  x <- list(value = log(den), grad = c(if (eta > 700) 0 else -(1 - zi) * q * mu / den, zi * (1 - zi) * (1 - q) / den))
  bad <- show(sprintf("eta = %4g", eta), ev(data.frame(y = 0, o = eta), zi), x)
  if (eta == 0) control_ok <- control_ok && !bad else if (bad) present <- TRUE
}
cat("\nhurdle_poisson posterior_epred, (1 - hu) mu / (1 - exp(-mu)), at hu = 0.3\n")
ff <- update(fh, algorithm = "fixed_param", chains = 1, iter = 2, warmup = 0, seed = 1, recompile = FALSE,
             init = list(list(Intercept = 0, hu = hu)), silent = 2, refresh = 0)
e <- posterior_epred(ff, newdata = data.frame(o = c(0, -750)))[1, ]
x <- c((1 - hu) / -expm1(-1), 1 - hu)
cat(sprintf("  eta = %4g: brms %.16g exact %.16g\n", c(0, -750), e, x), sep = "")
if (!isTRUE(abs(e[1] - x[1]) < 1e-14)) control_ok <- FALSE
if (!isTRUE(abs(e[2] - x[2]) < 1e-14)) present <- TRUE
if (!control_ok) { cat("\nthe exact formula does not match brms at a control point: the script is wrong\n"); quit(status = 2) }
cat(if (present) "\nPROBLEM PRESENT: brms's zero-inflated / hurdle Poisson forms are infinite or NaN at extreme eta\n" else "\nproblem absent\n")
cat(sprintf("run time %.1f s\n", proc.time()[["elapsed"]] - t_start))
quit(status = if (present) 1 else 0)
