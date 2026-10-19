#!/usr/bin/env Rscript
# brms 2.23's conditional_effects() grids for interaction effects (E05): which rows an effect "a:b" has with brms's default
# int_conditions (the levels of a factor; the mean and the mean -/+ the SD of a numeric moderator), in which order, which
# columns, and how effect2__ labels a numeric moderator's values, at several scales of the moderator.
#
#   Rscript tests/hypothesis_probes/probe_effects_grid.R tests/hypothesis_probes/brms-2.23.0-effects-grid.json
#
# One Gaussian model is compiled once (chains = 0); each data set is attached with update(newdata, chains = 0), which
# reuses the compiled program, and run with Stan's fixed-parameter algorithm from four seeded points (no MCMC), so that
# conditional_effects() has draws. Recorded: the data and each table's grid columns (every column but the summaries,
# which depend on the draws), and R's as.character(round(v, 2)) of probe values; brms's and R's outputs only, not their
# code (GPL). tests/test_prior_draws.py rebuilds every grid and label.
suppressPackageStartupMessages({ library(brms); library(rstan); library(jsonlite) })
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("usage: Rscript tests/hypothesis_probes/probe_effects_grid.R <output json>")
num <- function(v) ifelse(is.finite(v), sprintf("%.17g", v), "NA")
column_out <- function(v) if (is.numeric(v)) list(type = "number", values = I(num(v))) else list(type = "text", values = I(as.character(v)))
set.seed(20261018)
n <- 40
base <- data.frame(x = round(rnorm(n), 4), w = round(rnorm(n, 1, 0.5), 4),
                   z = factor(sample(c("p", "q", "r"), n, replace = TRUE), levels = c("p", "q", "r")),
                   h = factor(sample(c("s", "t"), n, replace = TRUE), levels = c("s", "t")))
base$y <- round(0.3 + 0.5 * base$x - 0.2 * base$w + rnorm(n, 0, 0.6), 4)
scaled <- function(a, b) { d <- base; d$w <- as.numeric(sprintf("%.17g", a + b * base$w)); d }
datasets <- list(base = base, large = scaled(1234.5678, 100), near_power_of_ten = scaled(1e5, 0.001),
                 near_zero = scaled(-0.004, 0.001), negative = scaled(-3, 2), tiny = scaled(0, 1e-4))
pri <- c(set_prior("normal(0, 5)", class = "Intercept"), set_prior("normal(0, 2)", class = "b"),
         set_prior("normal(0, 1)", class = "sigma"))
fit0 <- brm(y ~ x * w + z + h, data = base, prior = pri, chains = 0, backend = "rstan", silent = 2, refresh = 0)
sf <- fit0$fit
set.seed(7)
inits <- lapply(1:4, function(k) constrain_pars(sf, runif(get_num_upars(sf), -1, 1)))
calls <- list(base = list(c("x:w", "x:z", "z:h", "h:w"), c("w:x", "z:x"), c("x:w", "w:x")))   # one table per effect and call
out <- list(brms = as.character(packageVersion("brms")), R = paste(R.version$major, R.version$minor, sep = "."), sets = list())
for (nm in names(datasets)) {
  d <- datasets[[nm]]
  fk <- suppressMessages(update(fit0, newdata = d, chains = 0, recompile = FALSE, silent = 2, refresh = 0))
  fk$fit <- sampling(sf@stanmodel, data = standata(fk), algorithm = "Fixed_param", chains = 4, iter = 1, warmup = 0,
                     init = inits, refresh = 0, seed = 1, cores = 1)
  fk <- rename_pars(fk)
  runs <- lapply(if (is.null(calls[[nm]])) list("x:w") else calls[[nm]], function(eff) {
    warned <- character(0)
    ce <- withCallingHandlers(conditional_effects(fk, effects = eff),
                              warning = function(w) { warned <<- c(warned, conditionMessage(w)); invokeRestart("muffleWarning") })
    list(asked = I(eff), effects = I(names(ce)), warnings = I(warned),
         tables = lapply(ce, function(t) {
           t <- as.data.frame(t)
           keep <- setdiff(names(t), c("estimate__", "se__", "lower__", "upper__"))
           list(columns = I(names(t)), grid = lapply(t[keep], column_out))
         }))
  })
  out$sets[[nm]] <- list(data = lapply(as.list(d), column_out), runs = runs)
}
# effect2__ labels a numeric moderator's value as as.character(round(value, 2)) does (each set above): that text of 2000
# values, half of them within 1e-12 of a tie at the third decimal, others large, small, negative or near zero
set.seed(11)
ties <- (round(runif(1000, -50, 50) * 1000) + 0.5) / 1000
probe <- c(ties, ties + 1e-12, ties - 1e-12, rnorm(300) * 10^sample(-4:13, 300, replace = TRUE), c(0, -0.004, 0.005, -0.005,
           1e5, -1e5, 2e15, 123456.789, 0.125, 0.375, 2.675, 1.005, 99999.995))
probe <- as.numeric(sprintf("%.17g", probe))
out$labels <- list(values = I(num(probe)), text = I(as.character(round(probe, 2))))
writeLines(toJSON(out, auto_unbox = TRUE, digits = I(17), pretty = FALSE, null = "null"), args[[1]])
cat(sprintf("brms %s: %d data sets\n", packageVersion("brms"), length(out$sets)))
