#!/usr/bin/env Rscript
# loo 2.9.0's loo() and relative_eff() on the deterministic inputs of tests/loo_probes/inputs.py (C07).
#
#   python -m tests.loo_probes.inputs <dir>
#   Rscript tests/loo_probes/probe.R <dir> tests/loo_probes/loo-2.9.0.json
#
# Records loo's outputs (not its code; loo is GPL-3) as the oracle for autobayes.post_processing on inputs the frozen brms
# fixtures do not reach. Nothing here samples.
suppressPackageStartupMessages({ library(loo); library(jsonlite) })
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("usage: Rscript tests/loo_probes/probe.R <inputs dir> <output json>")
dir <- args[[1]]; out_path <- args[[2]]
enc <- function(x) {
  v <- as.numeric(as.vector(x))
  list(dims = I(as.integer(if (is.null(dim(x))) length(x) else dim(x))),
       values = I(ifelse(is.na(v), ifelse(is.nan(v), "NaN", "NA"), ifelse(is.finite(v), sprintf("%.17g", v), ifelse(v > 0, "Inf", "-Inf")))))
}
read_index <- function(name) read.delim(file.path(dir, name), header = FALSE, sep = "\t", stringsAsFactors = FALSE, quote = "",
                                        colClasses = "character")
read_matrix <- function(name) unname(as.matrix(read.csv(file.path(dir, name), header = FALSE, na.strings = "NA")))
loo_index <- read_index("loo_index.tsv"); reff_index <- read_index("relative_eff_index.tsv")
loo_probes <- lapply(seq_len(nrow(loo_index)), function(i) {
  k <- as.integer(loo_index[i, 1])
  ll <- read_matrix(sprintf("loo_%d.csv", k))
  r_eff <- as.numeric(read_matrix(sprintf("loo_%d_r_eff.csv", k)))
  l <- suppressWarnings(suppressMessages(loo(ll, r_eff = r_eff)))
  list(label = loo_index[i, 2], sha256 = loo_index[i, 3], r_eff_sha256 = loo_index[i, 4],
       pointwise = enc(l$pointwise), columns = I(colnames(l$pointwise)), estimates = enc(l$estimates),
       rows = I(rownames(l$estimates)), pareto_k = enc(l$diagnostics$pareto_k), n_eff = enc(l$diagnostics$n_eff))
})
reff_probes <- lapply(seq_len(nrow(reff_index)), function(i) {
  k <- as.integer(reff_index[i, 1]); C <- as.integer(reff_index[i, 3]); n_iter <- as.integer(reff_index[i, 4])
  lik <- read_matrix(sprintf("reff_%d.csv", k))
  list(label = reff_index[i, 2], sha256 = reff_index[i, 5], r_eff = enc(relative_eff(lik, chain_id = rep(seq_len(C), each = n_iter))))
})
writeLines(toJSON(list(loo = as.character(packageVersion("loo")), posterior = as.character(packageVersion("posterior")),
                       R = paste(R.version$major, R.version$minor, sep = "."), loo_probes = loo_probes, relative_eff_probes = reff_probes),
                  auto_unbox = TRUE, digits = I(17), pretty = FALSE), out_path)
cat(sprintf("loo %s: %d loo probes, %d relative_eff probes\n", packageVersion("loo"), length(loo_probes), length(reff_probes)))
