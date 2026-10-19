#!/usr/bin/env Rscript
# brms 2.23's default_prior() for the distributional formulas of E03 (docs/contexts-newer-families.md):
#
#   LC_ALL=C Rscript tests/contexts_probes/probe_contexts.R tests/contexts_probes/brms-2.23.0-contexts.json
#
# Recorded: the default-prior table of a formula on the Weibull's shape, the beta-binomial's phi, the zero-inflated negative
# binomial's zi, the hurdle lognormal's hu and the skew normal's alpha, on the small frame written below (and into the output,
# which tests/test_contexts_newer_families.py rebuilds it from), including responses of a wide range, which show that the
# formulas' intercept defaults do not read the response. brms's outputs only, not its code (brms is GPL-3).
suppressPackageStartupMessages({ library(brms); library(jsonlite) })
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("usage: Rscript tests/contexts_probes/probe_contexts.R <output json>")
d <- data.frame(
  y = c(-1.2, 0.4, 2.2, -0.3, 1.9, 0.8, -2.5, 1.1, 0.05, -0.7, 3.3, 0.6),
  ypos = c(0.82, 3.5, 1.27, 0.06, 12.4, 2.2, 0.95, 5.75, 0.31, 1.6, 8.9, 0.47),
  ybig = c(820, 3500, 1270, 60, 12400, 2200, 950, 5750, 310, 1600, 8900, 470),
  cnt = c(0, 3, 0, 1, 7, 2, 0, 0, 12, 4, 1, 0),
  k = c(0, 2, 1, 3, 4, 0, 2, 5, 1, 3, 6, 2),
  size = c(4, 5, 3, 6, 4, 2, 7, 8, 3, 5, 9, 2),
  zpos = c(0, 3.1, 0, 0.4, 12, 0, 1.3, 0, 0.2, 2.6, 0, 0.9),
  x = c(0.12, -1.3, 0.85, 1.7, -0.44, 0.03, -2.1, 0.9, 1.15, -0.6, 0.31, -0.05),
  z = c("p", "q", "r", "p", "q", "r", "p", "q", "r", "p", "q", "r"),
  stringsAsFactors = FALSE)
frame <- d
d$z <- factor(d$z, levels = c("p", "q", "r"))
cases <- list(
  "weibull, shape ~ x" = list(bf(ypos ~ x, shape ~ x), weibull()),
  "weibull, shape ~ 1, a wide response" = list(bf(ybig ~ x, shape ~ 1), weibull()),
  "beta_binomial, phi ~ z" = list(bf(k | trials(size) ~ x, phi ~ z), beta_binomial()),
  "zero_inflated_negbinomial, zi ~ x" = list(bf(cnt ~ x, zi ~ x), zero_inflated_negbinomial()),
  "zero_inflated_negbinomial, zi ~ 0 + z" = list(bf(cnt ~ x, zi ~ 0 + z), zero_inflated_negbinomial()),
  "zero_inflated_negbinomial, zi ~ 1" = list(bf(cnt ~ x, zi ~ 1), zero_inflated_negbinomial()),
  "hurdle_lognormal, hu ~ x + z" = list(bf(zpos ~ x, hu ~ x + z), hurdle_lognormal()),
  "skew_normal, alpha ~ x" = list(bf(y ~ x, alpha ~ x), skew_normal()),
  "skew_normal, alpha ~ 1, a wide response" = list(bf(ybig ~ x, alpha ~ 1), skew_normal()))
columns <- c("prior", "class", "coef", "group", "resp", "dpar", "nlpar", "lb", "ub", "source")
out <- list(brms = as.character(packageVersion("brms")), R = paste(R.version$major, R.version$minor, sep = "."),
            frame = frame, tables = list())
for (name in names(cases)) {
  dp <- default_prior(cases[[name]][[1]], data = d, family = cases[[name]][[2]])
  out$tables[[name]] <- lapply(seq_len(nrow(dp)), function(i) as.list(vapply(columns, function(k) as.character(dp[[k]][i]), "")))
}
writeLines(toJSON(out, auto_unbox = TRUE, pretty = TRUE, digits = NA), args[1])
