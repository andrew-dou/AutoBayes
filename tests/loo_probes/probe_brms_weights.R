#!/usr/bin/env Rscript
# brms 2.23's loo_compare() and model_weights() of three fixed-parameter fits whose log-likelihoods are close (D05).
#
#   Rscript tests/loo_probes/probe_brms_weights.R tests/oracle/fixtures/gauss_loo_compare_x/data.csv \
#       tests/loo_probes/brms-2.23.0-weights.json
#
# The frozen case gauss_loo_compare_x records brms's model_weights() of two models whose elpd differ by 263, so that one
# weight is 1 and the other 3.7e-115. Here three fits of y ~ x on the same data, each a fixed-parameter run (one chain per
# draw, as the oracle's fixtures hold their points; nothing is sampled) at 16 seeded draws near the least-squares fit, give
# weights between 1e-11 and 1. Recorded: each fit's log_lik() (draws x rows), its loo and waic estimates, loo_compare() by
# "loo" and by "waic", and model_weights() by "loo", "waic" and "pseudobma" (BB = FALSE): brms's outputs, not its code
# (brms and loo are GPL-3). The Python test recomputes every output from the recorded log-likelihoods.
suppressPackageStartupMessages({ library(brms); library(rstan); library(jsonlite) })
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("usage: Rscript tests/loo_probes/probe_brms_weights.R <data.csv> <output json>")
data_path <- args[[1]]; out_path <- args[[2]]
enc <- function(x) {
  v <- as.numeric(as.vector(x))
  list(dims = I(as.integer(if (is.null(dim(x))) length(x) else dim(x))),
       values = I(ifelse(is.na(v), ifelse(is.nan(v), "NaN", "NA"), ifelse(is.finite(v), sprintf("%.17g", v), ifelse(v > 0, "Inf", "-Inf")))))
}
d <- read.csv(data_path)[, c("x", "y")]
pri <- c(set_prior("normal(0, 5)", class = "Intercept"), set_prior("normal(0, 2)", class = "b"), set_prior("normal(0, 1)", class = "sigma"))
fit0 <- brm(y ~ x, data = d, family = gaussian(), prior = pri, chains = 0, backend = "rstan", silent = 2, refresh = 0)
ls_fit <- lm(y ~ x, data = d); cf <- coef(ls_fit); s <- summary(ls_fit)$sigma
S <- 16
fixed_fit <- function(seed, spread) {                  # draw k is the k-th seeded point: Intercept is brms's centred one
  set.seed(seed)
  inits <- lapply(seq_len(S), function(k) list(b = array(cf[[2]] + spread * rnorm(1, 0, 0.1)),
                                               Intercept = cf[[1]] + cf[[2]] * mean(d$x) + spread * rnorm(1, 0, 0.1),
                                               sigma = s * exp(spread * rnorm(1, 0, 0.1))))
  f <- fit0
  f$fit <- sampling(fit0$fit@stanmodel, data = standata(fit0), algorithm = "Fixed_param", chains = S, iter = 1, warmup = 0,
                    init = inits, refresh = 0, seed = 1, cores = 1)
  rename_pars(f)
}
ids <- c("m1", "m2", "m3")
quiet <- function(expr) suppressWarnings(suppressMessages(expr))
fits <- Map(function(seed, spread, id) quiet(add_criterion(fixed_fit(seed, spread), c("loo", "waic"), model_name = id)),
            c(13, 11, 12), c(3, 1, 2), ids)                 # given worst first: brms lists them best first
cmp <- function(criterion) {
  m <- quiet(loo_compare(fits[[1]], fits[[2]], fits[[3]], criterion = criterion, model_names = ids))
  list(table = enc(unclass(m)), rows = I(rownames(m)), columns = I(colnames(m)))
}
weights <- function(w, ...) {
  v <- quiet(model_weights(fits[[1]], fits[[2]], fits[[3]], weights = w, model_names = ids, ...))
  list(weights = enc(as.numeric(v)), models = I(names(v)))
}
out <- list(brms = as.character(packageVersion("brms")), loo = as.character(packageVersion("loo")),
            R = paste(R.version$major, R.version$minor, sep = "."), data_md5 = unname(tools::md5sum(data_path)), models = I(ids),
            log_lik = unname(lapply(fits, function(f) enc(log_lik(f)))),
            loo_estimates = unname(lapply(fits, function(f) enc(f$criteria$loo$estimates))),
            waic_estimates = unname(lapply(fits, function(f) enc(f$criteria$waic$estimates))),
            compare_loo = cmp("loo"), compare_waic = cmp("waic"),
            weights_loo = weights("loo"), weights_waic = weights("waic"), weights_pseudobma = weights("pseudobma", BB = FALSE))
writeLines(toJSON(out, auto_unbox = TRUE, digits = I(17), pretty = FALSE), out_path)
cat(sprintf("brms %s, loo %s: three fits of %d draws on %d rows\n", packageVersion("brms"), packageVersion("loo"), S, nrow(d)))
