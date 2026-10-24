#!/usr/bin/env Rscript
# U31: a truncated model, y | trunc(lb = ...) ~ 1, cannot be built from a single data row, with a per-row bound or a
# constant one: brms passes the bound as a scalar where the program declares an array of length N = 1, and rstan stops at
# data initialisation ("mismatch in number dimensions declared and found in context ... variable name=lb; dims
# declared=(1); dims found=()"), so brm() returns an object without a valid model and log_prob() fails. The same model with
# two identical rows works, and so does a one-row model without truncation, so one row should work too.
#
#   Rscript docs/brms-parity/upstream/repro/U31.R      # exit status 1 while the problem is present
#
# Stock CRAN brms (rstan backend) only, nothing sampled. Found while writing U22. Exact: the row's log likelihood
# log phi(y) - log(1 - Phi(L)) at mu = 0, sigma = 1, y = 1.5, L = 1, twice that for two identical rows, and log phi(y)
# without truncation; the prior's part is removed through the model's prior_only flag.
t_start <- proc.time()[["elapsed"]]
suppressPackageStartupMessages({ library(brms); library(rstan) })
cat(sprintf("brms %s, rstan %s, StanHeaders %s, R %s\n", packageVersion("brms"), packageVersion("rstan"),
            packageVersion("StanHeaders"), getRversion()))
pri <- c(set_prior("normal(0, 5)", class = "Intercept"), set_prior("normal(0, 2)", class = "sigma"))
row_exact <- dnorm(1.5, log = TRUE) - pnorm(1, lower.tail = FALSE, log.p = TRUE)
lik <- function(fit) tryCatch({                      # the log likelihood at mu = 0, sigma = 1, or the error
  sf <- fit$fit
  u <- unconstrain_pars(sf, list(Intercept = 0, sigma = 1))
  sp <- sampling(sf@stanmodel, data = modifyList(standata(fit), list(prior_only = 1L)), chains = 0)
  as.numeric(log_prob(sf, u, adjust_transform = FALSE)) - as.numeric(log_prob(sp, u, adjust_transform = FALSE))
}, error = function(e) structure(NA_real_, msg = strsplit(conditionMessage(e), "\n")[[1]][1]))
show <- function(label, v, x) {
  if (is.na(v)) cat(sprintf("  %-46s error: %s   exact %.16g\n", label, attr(v, "msg"), x))
  else cat(sprintf("  %-46s brms %.16g   exact %.16g\n", label, v, x))
}
cat("\nthe standata entry lb of a one-row model with a per-row bound:\n")
s1 <- make_standata(y | trunc(lb = L) ~ 1, data = data.frame(y = 1.5, L = 1), prior = pri)
cat(sprintf("  N = %d, lb has dim %s and length %d\n", s1$N, if (is.null(dim(s1$lb))) "NULL" else paste(dim(s1$lb), collapse = "x"),
            length(s1$lb)))
cat("\nbuilding each model with brm(..., chains = 0); rstan's own messages, if any, follow each line\n")
f1 <- brm(y | trunc(lb = L) ~ 1, data = data.frame(y = 1.5, L = 1), prior = pri, chains = 0, backend = "rstan",
          silent = 2, refresh = 0)
v1 <- lik(f1)
f2 <- brm(y | trunc(lb = L) ~ 1, data = data.frame(y = c(1.5, 1.5), L = c(1, 1)), prior = pri, chains = 0,
          backend = "rstan", silent = 2, refresh = 0)
v2 <- lik(f2)
fc <- brm(y | trunc(lb = 1) ~ 1, data = data.frame(y = 1.5), prior = pri, chains = 0, backend = "rstan", silent = 2, refresh = 0)
vc <- lik(fc)
fn <- brm(y ~ 1, data = data.frame(y = 1.5), prior = pri, chains = 0, backend = "rstan", silent = 2, refresh = 0)
vn <- lik(fn)
cat("\nlog likelihood at mu = 0, sigma = 1\n")
show("one row, per-row bound trunc(lb = L)", v1, row_exact)
show("one row, constant bound trunc(lb = 1)", vc, row_exact)
show("two identical rows, per-row bound (control)", v2, 2 * row_exact)
show("one row, no truncation (control)", vn, dnorm(1.5, log = TRUE))
control_ok <- !is.na(v2) && abs(v2 - 2 * row_exact) < 1e-12 && !is.na(vn) && abs(vn - dnorm(1.5, log = TRUE)) < 1e-12
if (!control_ok) { cat("\na control model does not give the exact value: the script is wrong\n"); quit(status = 2) }
present <- is.na(v1) || abs(v1 - row_exact) > 1e-12 || is.na(vc) || abs(vc - row_exact) > 1e-12
cat(if (present) "\nPROBLEM PRESENT: a one-row model with trunc() cannot be built\n" else "\nproblem absent\n")
cat(sprintf("run time %.1f s\n", proc.time()[["elapsed"]] - t_start))
quit(status = if (present) 1 else 0)
