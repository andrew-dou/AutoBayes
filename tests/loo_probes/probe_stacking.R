#!/usr/bin/env Rscript
# loo 2.9.0's stacking weights (E05): stacking_weights() of pointwise elpd matrices, the optimum loo_model_weights() and
# brms's model_weights(weights = "stacking") report, found by stats::constrOptim (a log-barrier method around BFGS) and
# so exact only to that optimiser's tolerance.
#
#   Rscript tests/loo_probes/probe_stacking.R tests/loo_probes/loo-2.9.0-stacking.json
#
# Recorded: the inputs (seeded here, written with 17 significant digits and read back, so loo sees exactly those) and
# loo's weights; and loo_model_weights() of three loo() objects of seeded log-likelihood matrices (not recorded) with the
# objects' pointwise elpd_loo and stacking_weights() of it, which must be the same weights. loo's outputs only, not its
# code (loo is GPL-3); nothing is sampled. The Python test (tests/test_loo_compare.py) computes the exact optimum on each
# input and measures loo's distance to it.
suppressPackageStartupMessages({ library(loo); library(jsonlite) })
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("usage: Rscript tests/loo_probes/probe_stacking.R <output json>")
num <- function(v) sprintf("%.17g", as.numeric(v))
exact <- function(m) matrix(as.numeric(sprintf("%.17g", m)), nrow = nrow(m))
set.seed(20261017)
elpd <- function(N, centres, sds) exact(sapply(seq_along(centres), function(k) rnorm(N, centres[k], sds[k])))
shared <- rnorm(80, -1.2, 0.6)
sets <- list(
  two_36 = elpd(36, c(-2, -2.2), c(1.5, 0.8)),
  two_vertex = exact(cbind(rnorm(30, -1, 0.2), rnorm(30, -4, 0.2))),               # one model better everywhere: w = (1, 0)
  two_nearly_equal = exact(cbind(shared, shared + rnorm(80, 0, 1e-3))),
  three_50 = elpd(50, c(-1.5, -1.6, -1.8), c(0.7, 1.1, 0.4)),
  four_one_dominated = exact(cbind(elpd(60, c(-1.4, -1.5, -1.6), c(0.6, 0.9, 1.2)), rnorm(60, -6, 0.3))),
  five_200 = elpd(200, c(-1.3, -1.35, -1.4, -1.5, -1.7), c(0.5, 0.8, 1.0, 0.6, 1.4)),
  three_1000 = elpd(1000, c(-0.9, -1.0, -1.05), c(0.4, 0.7, 0.9)),
  two_far_tails = exact(cbind(c(rnorm(35, -1, 0.3), -110), c(rnorm(35, -1.1, 0.3), -2))))
quiet <- function(expr) suppressWarnings(suppressMessages(expr))
out <- list(loo = as.character(packageVersion("loo")), R = paste(R.version$major, R.version$minor, sep = "."),
            sets = lapply(sets, function(m) list(elpd = lapply(seq_len(ncol(m)), function(k) I(num(m[, k]))),
                                                 weights = I(num(quiet(stacking_weights(m)))))))
ll <- lapply(c(-1, -1.15, -1.3), function(mu) exact(matrix(rnorm(400 * 30, mu, 0.5), nrow = 400)))   # draws x observations
loos <- lapply(ll, function(m) quiet(loo(m, r_eff = rep(1, ncol(m)))))
lpd <- sapply(loos, function(l) l$pointwise[, "elpd_loo"])
out$loo_objects <- list(pointwise_elpd = lapply(seq_len(ncol(lpd)), function(k) I(num(lpd[, k]))),
                        loo_model_weights = I(num(quiet(loo_model_weights(loos)))),
                        stacking_weights = I(num(quiet(stacking_weights(lpd)))))
writeLines(toJSON(out, auto_unbox = TRUE, digits = I(17), pretty = FALSE, null = "null"), args[[1]])
cat(sprintf("loo %s: stacking weights of %d sets\n", packageVersion("loo"), length(sets)))
