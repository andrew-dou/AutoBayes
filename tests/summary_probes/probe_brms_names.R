#!/usr/bin/env Rscript
# brms 2.23's names and order of its summaries (F07) on model structures the frozen cases do not cover: two grouping factors,
# independent slopes (||), correlations of three and four coefficients, categorical predictors and interactions, a
# distributional formula with several terms, an intercept-only one.
#
#   Rscript tests/summary_probes/probe_brms_names.R tests/summary_probes/brms-2.23.0-names.json [cache dir]
#
# Each model is fit to the data of the frozen case gauss_varint_summaries with brm(algorithm = "fixed_param") (two chains of four
# iterations at Stan's random inits: nothing is sampled), and its labels are recorded: variables(), the row names of fixef() and
# posterior_summary(), each group's dimnames in ranef() and coef(), VarCorr()'s groups, their parts and dimnames, and summary()'s
# tables (their row and column names) and ngrps. Labels only, no numbers (the draws are constant); brms's outputs only, not
# its code (brms is GPL-3). tests/test_summaries.py builds each model in AutoBayes and compares its labels with these.
suppressPackageStartupMessages({ library(brms); library(jsonlite) })
invisible(Sys.setlocale("LC_COLLATE", "C"))
args <- commandArgs(trailingOnly = TRUE)
if (!length(args) %in% 1:2) stop("usage: Rscript tests/summary_probes/probe_brms_names.R <output json> [cache dir for the fits]")
cache <- if (length(args) == 2) args[[2]] else NULL
data_path <- "tests/oracle/fixtures/gauss_varint_summaries/data.csv"
d <- read.csv(data_path, stringsAsFactors = FALSE)
for (col in c("z", "g", "h")) d[[col]] <- factor(d[[col]], levels = sort(unique(d[[col]])))
probes <- list(
  list(id = "two_factors", family = "gaussian", formula = "y ~ x + z + (1 | g) + (1 | h)"),
  list(id = "independent_slopes", family = "gaussian", formula = "y ~ x * z + w + (1 + x + w || g)"),
  list(id = "distributional_terms", family = "gaussian", formula = c("y ~ x + w + (1 + x | g)", "sigma ~ x + z")),
  list(id = "distributional_intercept", family = "gaussian", formula = c("y ~ x + (1 | g)", "sigma ~ 1")),
  list(id = "three_correlated", family = "bernoulli", formula = "yb ~ x + w + (1 + x + w | g) + (1 + x | h)"),
  list(id = "four_correlated", family = "gaussian", formula = "y ~ x + w + cnt + (1 + x + w + cnt | g)"),
  list(id = "population_only", family = "bernoulli", formula = "yb ~ x + z"),
  list(id = "population_only_gaussian", family = "gaussian", formula = "y ~ x")
)
chr <- function(v) I(as.character(v))
labels <- function(x) lapply(dimnames(x), chr)
attempt <- function(f) tryCatch(f(), error = function(e) list(error = conditionMessage(e)))   # brms's refusal, recorded
run <- function(p) {
  f <- if (length(p$formula) == 1) as.formula(p$formula) else do.call(bf, lapply(p$formula, as.formula))
  path <- if (!is.null(cache)) file.path(cache, paste0(p$id, ".rds")) else NULL
  fit <- if (!is.null(path) && file.exists(path)) readRDS(path) else
    suppressWarnings(brm(f, data = d, family = brmsfamily(p$family), algorithm = "fixed_param", chains = 2, iter = 4,
                         warmup = 0, init = 0.5, seed = 1, refresh = 0, silent = 2, backend = "rstan"))
  if (!is.null(path)) saveRDS(fit, path)
  s <- suppressWarnings(summary(fit))
  tables <- intersect(c("fixed", "spec_pars", "cor_pars", "rescor_pars", "splines", "mo", "gp"), names(s))
  list(id = p$id, family = p$family, formula = chr(p$formula), variables = chr(variables(fit)),
       fixef = labels(fixef(fit)), posterior_summary = labels(posterior_summary(fit)),
       ranef = attempt(function() lapply(ranef(fit), labels)), coef = attempt(function() lapply(coef(fit), labels)),
       ranef_groups = attempt(function() chr(names(ranef(fit)))),
       VarCorr = attempt(function() lapply(VarCorr(fit), function(g) lapply(g, labels))),
       VarCorr_groups = attempt(function() chr(names(VarCorr(fit)))),
       VarCorr_parts = attempt(function() lapply(VarCorr(fit), function(g) chr(names(g)))),
       summary = c(lapply(setNames(tables, tables), function(nm) labels(s[[nm]])),
                   list(tables = chr(tables), random = lapply(s$random, labels), random_groups = chr(names(s$random)),
                        ngrps = lapply(s$ngrps, as.integer), ngrps_groups = chr(names(s$ngrps)), nobs = s$nobs)))
}
out <- lapply(probes, run)
writeLines(toJSON(list(brms = as.character(packageVersion("brms")), data = data_path, probes = out), auto_unbox = TRUE,
                  pretty = FALSE, null = "null"), args[[1]])
cat(sprintf("brms %s: %d models\n", packageVersion("brms"), length(out)))
