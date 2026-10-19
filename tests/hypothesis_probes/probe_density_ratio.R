#!/usr/bin/env Rscript
# brms 2.23's density_ratio() on recorded draws (E05): the kernel density estimate behind hypothesis()'s Savage-Dickey
# evidence ratio of a point hypothesis (the posterior density of the hypothesis at the point over its prior density there).
#
#   Rscript tests/hypothesis_probes/probe_density_ratio.R tests/hypothesis_probes/brms-2.23.0-density-ratio.json
#
# Recorded: the draws (seeded here and written with 17 significant digits, read back so that brms sees exactly those), and
# for each set and point, density_ratio(x, point = p) (the density of x alone), density_ratio(y, point = p) and
# density_ratio(x, y, point = p), each with brms's default n = 4096; stats::bw.nrd0() of each set. brms's and R's outputs
# only, not their code (both are GPL); nothing is sampled and no Stan program runs. The Python test
# (tests/test_prior_draws.py) recomputes every output from the draws.
suppressPackageStartupMessages({ library(brms); library(jsonlite) })
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("usage: Rscript tests/hypothesis_probes/probe_density_ratio.R <output json>")
num <- function(v) ifelse(is.na(v), ifelse(is.nan(v), "NaN", "NA"), ifelse(is.finite(v), sprintf("%.17g", v), ifelse(v > 0, "Inf", "-Inf")))
exact <- function(v) as.numeric(sprintf("%.17g", v))   # the draws as written
set.seed(20261017)
sets <- list(
  gate_like_8 = list(x = rnorm(8, 0.3, 0.7), y = rnorm(8, 0, 2)),
  gate_like_40 = list(x = rnorm(40, 0.2, 0.5), y = rnorm(40, 0, 1.5)),
  moderate_400 = list(x = rnorm(400, -0.4, 0.3), y = rnorm(400, 0, 2)),
  large_4000 = list(x = rnorm(4000, 0.1, 0.2), y = rnorm(4000, 0, 1)),
  skewed_1000 = list(x = exp(rnorm(1000, -1, 0.6)), y = rexp(1000, 0.5)),
  heavy_500 = list(x = rt(500, 2) * 0.4, y = rcauchy(500)),
  two_draws = list(x = c(-0.25, 0.75), y = c(-1.5, 2)),
  three_draws = list(x = c(0.1, 0.4, 0.45), y = c(-2, 0.5, 3)),
  iqr_zero = list(x = c(rep(0.5, 16), -0.3, 0.9, 1.4, 2), y = c(rep(-1, 12), 0, 1, 3, 4)),
  constant_3 = list(x = rep(3, 10), y = rnorm(10, 3, 1)),
  constant_0 = list(x = rep(0, 10), y = rnorm(10)),
  constant_neg = list(x = rep(-2.5, 6), y = rnorm(6, -2, 2)),
  big_scale = list(x = rnorm(300, 1e6, 50), y = rnorm(300, 1e6, 400)),
  small_scale = list(x = rnorm(300, 1e-4, 1e-5), y = rnorm(300, 0, 1e-3)))
sets <- lapply(sets, function(s) lapply(s, exact))
points_for <- function(s) {
  x <- s$x; y <- s$y
  r <- diff(range(c(x, y))); r <- if (r > 0) r else 1
  rx <- diff(range(x))                                  # the grid of x's density spans its draws: 4095 cells
  p <- c(0, 0.5, mean(x), median(y), min(x), max(x), min(x) + 1e-6 * r, max(x) - 1e-6 * r, min(x) + 0.003 * r,
         max(y) - 0.002 * r, min(c(x, y)) - 0.3 * r, max(c(x, y)) + 0.7 * r, min(c(x, y)) - 40 * r, max(c(x, y)) + 1000 * r,
         min(x) + 0.5 * rx / 4095, max(x) - 1.5 * rx / 4095, min(x) + 2.5 * rx / 4095,    # mid-cell next to the ends
         min(x) - 1e-9 * r, max(x) + 1e-9 * r, min(x) - 0.05 * r, max(x) + 0.05 * r)        # just outside the draws
  exact(unique(p))
}
out <- list(brms = as.character(packageVersion("brms")), R = paste(R.version$major, R.version$minor, sep = "."), n = 4096L,
            sets = lapply(sets, function(s) {
              p <- points_for(s)
              list(x = I(num(s$x)), y = I(num(s$y)), points = I(num(p)),
                   bw_x = num(bw.nrd0(s$x)), bw_y = if (sd(s$y) > 0) num(bw.nrd0(s$y)) else NULL,
                   density_x = I(num(vapply(p, function(q) density_ratio(s$x, point = q), 0.0))),
                   density_y = I(num(vapply(p, function(q) density_ratio(s$y, point = q), 0.0))),
                   ratio = I(num(vapply(p, function(q) density_ratio(s$x, s$y, point = q), 0.0))))
            }),
            refused = lapply(list(one_draw = 2, with_na = c(1, NA, 2), with_inf = c(1, Inf, 2)), function(x) {
              msg <- tryCatch({ density_ratio(x, point = 0); NA_character_ }, error = function(e) conditionMessage(e))
              list(x = I(num(x)), error = if (is.na(msg)) NULL else msg)
            }))
writeLines(toJSON(out, auto_unbox = TRUE, digits = I(17), pretty = FALSE, null = "null"), args[[1]])
cat(sprintf("brms %s: %d sets of draws, %d points in all\n", packageVersion("brms"), length(sets),
            sum(vapply(out$sets, function(s) length(s$points), 0L))))
