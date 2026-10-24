#!/usr/bin/env Rscript
# U29 (D93): loo_compare() compares a PSIS-LOO result with a WAIC result without an error or a warning, and names the
# table's columns after the first argument's criterion, so a mixed table reads as a LOO comparison.
#
#   Rscript docs/brms-parity/upstream/repro/U29.R      # exit status 1 while the problem is present
#
# loo only (no brms, nothing sampled): loo() and waic() of two seeded log-likelihood matrices (400 draws x 30
# observations). Expected: an error, or at least a warning, that the results use different criteria.
t_start <- proc.time()[["elapsed"]]
suppressPackageStartupMessages(library(loo))
cat(sprintf("loo %s, R %s\n", packageVersion("loo"), getRversion()))
set.seed(3)
ll1 <- matrix(rnorm(400 * 30, -1, 0.3), 400); ll2 <- matrix(rnorm(400 * 30, -1.1, 0.3), 400)
l1 <- suppressWarnings(loo(ll1, r_eff = rep(1, 30))); w2 <- suppressWarnings(waic(ll2))
cat(sprintf("classes: loo() %s; waic() %s\n", paste(class(l1), collapse = "/"), paste(class(w2), collapse = "/")))
warnings_seen <- character(0)
res <- withCallingHandlers(tryCatch(loo_compare(l1, w2), error = function(e) e),
                           warning = function(w) { warnings_seen <<- c(warnings_seen, conditionMessage(w)); invokeRestart("muffleWarning") })
if (inherits(res, "error")) {
  cat(sprintf("loo_compare(loo, waic): error: %s\n", conditionMessage(res))); present <- FALSE
} else {
  cat("loo_compare(loo, waic) returned:\n"); print(res, simplify = FALSE)
  cat(sprintf("warnings: %s\n", if (length(warnings_seen)) paste(warnings_seen, collapse = "; ") else "none"))
  present <- length(warnings_seen) == 0
}
cat(if (present) "\nPROBLEM PRESENT: loo_compare mixes PSIS-LOO and WAIC without a warning\n" else "\nproblem absent\n")
cat(sprintf("run time %.1f s\n", proc.time()[["elapsed"]] - t_start))
quit(status = if (present) 1 else 0)
