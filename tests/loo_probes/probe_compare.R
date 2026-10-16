#!/usr/bin/env Rscript
# loo 2.9.0's waic(), loo_compare() and loo_model_weights() on the deterministic inputs of tests/loo_probes/compare_inputs.py
# (D05).
#
#   python -m tests.loo_probes.compare_inputs <dir>
#   Rscript tests/loo_probes/probe_compare.R <dir> tests/loo_probes/loo-compare-2.9.0.json
#
# Records loo's outputs (not its code; loo is GPL-3) as the oracle for autobayes.post_processing's waic, loo_compare and
# model_weights on inputs the frozen brms fixtures do not reach. Each matrix is one model's log-likelihood (draws x
# observations); loo() is given r_eff = 1, as brms gives it for one-draw chains. Nothing here samples.
suppressPackageStartupMessages({ library(loo); library(jsonlite) })
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("usage: Rscript tests/loo_probes/probe_compare.R <inputs dir> <output json>")
dir <- args[[1]]; out_path <- args[[2]]
enc <- function(x) {
  v <- as.numeric(as.vector(x))
  list(dims = I(as.integer(if (is.null(dim(x))) length(x) else dim(x))),
       values = I(ifelse(is.na(v), ifelse(is.nan(v), "NaN", "NA"), ifelse(is.finite(v), sprintf("%.17g", v), ifelse(v > 0, "Inf", "-Inf")))))
}
index <- read.delim(file.path(dir, "compare_index.tsv"), header = FALSE, sep = "\t", stringsAsFactors = FALSE, quote = "",
                    colClasses = "character", col.names = c("set", "label", "named", "position", "name", "file", "sha256"))
quiet <- function(expr) suppressWarnings(suppressMessages(expr))
probes <- lapply(unique(index$set), function(s) {
  rows <- index[index$set == s, ]
  rows <- rows[order(as.integer(rows$position)), ]
  named <- rows$named[1] == "1"
  lls <- lapply(rows$file, function(f) unname(as.matrix(read.csv(file.path(dir, f), header = FALSE))))
  loos <- lapply(lls, function(ll) quiet(loo(ll, r_eff = rep(1, ncol(ll)))))
  waics <- lapply(lls, function(ll) quiet(waic(ll)))
  if (named) names(loos) <- names(waics) <- rows$name          # unnamed: loo names the rows model1, model2, ...
  by_loo <- quiet(loo_compare(loos))
  by_waic <- quiet(loo_compare(waics))
  weights <- quiet(loo_model_weights(loos, method = "pseudobma", BB = FALSE))
  list(label = rows$label[1], named = named, models = I(rows$name), sha256 = I(rows$sha256),
       waic = unname(lapply(waics, function(w) list(pointwise = enc(w$pointwise), columns = I(colnames(w$pointwise)),
                                             estimates = enc(w$estimates), rows = I(rownames(w$estimates))))),
       loo_estimates = unname(lapply(loos, function(l) enc(l$estimates))),   # in the order of models
       compare_loo = list(table = enc(unclass(by_loo)), rows = I(rownames(by_loo)), columns = I(colnames(by_loo))),
       compare_waic = list(table = enc(unclass(by_waic)), rows = I(rownames(by_waic)), columns = I(colnames(by_waic))),
       pseudobma = list(weights = enc(as.numeric(weights)), models = I(names(weights))))
})
writeLines(toJSON(list(loo = as.character(packageVersion("loo")), R = paste(R.version$major, R.version$minor, sep = "."),
                       probes = probes), auto_unbox = TRUE, digits = I(17), pretty = FALSE), out_path)
cat(sprintf("loo %s: %d comparison probes\n", packageVersion("loo"), length(probes)))
