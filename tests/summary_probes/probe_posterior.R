#!/usr/bin/env Rscript
# The posterior package's rhat(), ess_bulk(), ess_tail() and quantile2() on seeded draws (F07): the convergence diagnostics
# brms's summary() reports (Rhat, Bulk_ESS, Tail_ESS) and its interval, on inputs the frozen brms fixtures do not reach.
#
#   Rscript tests/summary_probes/probe_posterior.R tests/summary_probes/posterior-1.6.1.json
#
# Recorded: each probe's draws (iterations x chains, seeded here, most on a grid of 2^-14 so that some tie, written as exact
# hexadecimal doubles, which R's decimal reader would not give back to the last bit) and posterior's outputs on them.
# posterior's outputs only, not its code; nothing is sampled and no Stan program runs. The Python test (tests/test_summaries.py) recomputes every output from the recorded draws.
suppressPackageStartupMessages({ library(posterior); library(jsonlite) })
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("usage: Rscript tests/summary_probes/probe_posterior.R <output json>")
num <- function(v) ifelse(is.na(v), ifelse(is.nan(v), "NaN", "NA"), ifelse(is.finite(v), sprintf("%.17g", v), ifelse(v > 0, "Inf", "-Inf")))
enc <- function(x) list(dims = I(as.integer(if (is.null(dim(x))) length(x) else dim(x))), values = I(num(as.numeric(as.vector(x)))))
hex <- function(v) ifelse(is.na(v), ifelse(is.nan(v), "NaN", "NA"), ifelse(is.finite(v), sprintf("%a", v), ifelse(v > 0, "Inf", "-Inf")))
written <- function(x) {                                  # the draws as exact hexadecimal doubles (C99's %a), checked
  text <- hex(as.vector(x))
  back <- suppressWarnings(as.numeric(ifelse(grepl("^-?0x", text), text, NA)))
  ok <- is.finite(x) & !is.na(x)
  if (!identical(back[ok], as.vector(x)[ok])) stop("the hexadecimal draws do not read back exactly")
  list(dims = I(dim(x)), values = I(text))
}
grid <- function(x, bits = 14) round(x * 2^bits) / 2^bits   # draws on a grid of 2^-14: short exact hexadecimals, some ties
ar_chains <- function(iter, chains, rho, sd = 1, centre_sd = 0, scale = 1, shift = 0) {
  x <- matrix(NA_real_, iter, chains)
  for (ch in seq_len(chains)) {
    centre <- rnorm(1, 0, centre_sd)
    u <- rnorm(1, 0, sd)
    for (t in seq_len(iter)) {
      if (t > 1) u <- rho * u + sqrt(1 - rho^2) * rnorm(1, 0, sd)
      x[t, ch] <- centre + u
    }
  }
  shift + scale * x
}
outputs <- function(x) {
  one <- function(f) tryCatch(suppressWarnings(f(x)), error = function(e) NA_real_)
  list(rhat = one(rhat), ess_bulk = one(ess_bulk), ess_tail = one(ess_tail),
       quantile2 = enc(tryCatch(unname(quantile2(x, probs = c(0.025, 0.05, 0.5, 0.95, 0.975))), error = function(e) rep(NA_real_, 5))))
}
probes <- list()
add <- function(label, x) {
  probes[[length(probes) + 1]] <<- c(list(label = label, draws = written(x)), outputs(x))
}
set.seed(20261024)
sizes <- list(c(4, 1), c(4, 2), c(4, 3), c(2, 3), c(4, 50), c(4, 51), c(4, 52), c(4, 53), c(2, 9), c(2, 10), c(2, 11), c(2, 12), c(2, 13), c(3, 8),
              c(4, 4), c(4, 5), c(4, 6), c(4, 7), c(1, 100), c(1, 101), c(8, 60))
for (s in sizes) for (rho in c(-0.6, 0.9)) for (centre_sd in c(0, 0.4))
  add(sprintf("chains=%d iter=%d rho=%g centre_sd=%g", s[1], s[2], rho, centre_sd), grid(ar_chains(s[2], s[1], rho, 1, centre_sd)))
for (k in 1:6) add(sprintf("chains=4 iter=50 rho=0.6 centre_sd=0.3 (%d)", k), grid(ar_chains(50, 4, 0.6, 1, 0.3)))   # like the F07 cases
add("antithetic rho=-0.95", grid(ar_chains(50, 4, -0.95)))
add("trending rho=0.995", grid(ar_chains(50, 4, 0.995)))
add("far apart chains", grid(ar_chains(50, 4, 0.3, 1, 5)))
add("large scale and shift", grid(ar_chains(50, 4, 0.5, 1, 0.3, scale = 1e6, shift = 1e8)))
add("small scale", grid(ar_chains(50, 4, 0.5, 1, 0.3), 60) * 2^-30)
add("ties at 0.5", round(2 * ar_chains(50, 4, 0.6, 1, 0.3)) / 2)
add("binary", (ar_chains(50, 4, 0.6, 1, 0.3) > 0) * 1)
add("binary, half and half", matrix(rep(c(0, 1, 1, 0), each = 50), 50, 4))
add("three values", pmin(pmax(round(ar_chains(50, 4, 0.6, 1, 0.3)), -1), 1))
add("constant", matrix(1.5, 50, 4))
add("constant chains, different levels", matrix(rep(c(1, 2, 3, 4), each = 50), 50, 4))
add("one draw differs", { x <- matrix(0.25, 50, 4); x[17, 3] <- 1; x })
add("one chain constant", { x <- grid(ar_chains(50, 4, 0.3)); x[, 2] <- 0.5; x })
add("with NA", { x <- grid(ar_chains(50, 4, 0.3)); x[5, 1] <- NA; x })
add("with Inf", { x <- grid(ar_chains(50, 4, 0.3)); x[5, 1] <- Inf; x })
add("heavy tails", grid(matrix(rt(200, 1.2), 50, 4)))
for (spread in c(1e-12, 1e-15, 1e-17, 1e-20)) for (centre in c(0, 1, 1e6))   # how near to constant is constant
  add(sprintf("spread %g around %g", spread, centre), centre + spread * ar_chains(50, 4, 0.3, 1, 0.3))
for (centre in c(0, 1, 1024)) for (k in c(0.5, 1, 2, 4, 8, 16)) {   # one draw above the others by k units of the double spacing at 1
  x <- matrix(centre, 50, 4); x[17, 3] <- centre + k * .Machine$double.eps * max(1, centre)
  add(sprintf("one draw %g eps above %g", k, centre), x)
}
writeLines(toJSON(list(posterior = as.character(packageVersion("posterior")), R = paste(R.version$major, R.version$minor, sep = "."),
                       probes = probes), auto_unbox = TRUE, digits = I(17), pretty = FALSE), args[[1]])
cat(sprintf("posterior %s: %d probes\n", packageVersion("posterior"), length(probes)))
