#!/usr/bin/env Rscript
# U30 (D50): loo()'s Monte Carlo standard error of an observation's elpd_loo is NaN once that elpd is below about -372, where
# its square (on the probability scale) underflows, and the total mcse_elpd_loo is then NaN too; elpd_loo, p_loo and k-hat
# are unaffected. The MCSE of a log-likelihood shifted by a constant is the same, so the expected value is the MCSE of the
# shifted row.
#
#   Rscript docs/brms-parity/upstream/repro/U30.R      # exit status 1 while the problem is present
#
# loo only (no brms, nothing sampled): a seeded log-likelihood matrix (1000 draws x 5 observations) whose fifth observation
# lies near -400, and the same matrix with that column shifted by +400 (an exact shift in double precision: the draws are
# rounded to multiples of 2^-40 first). PSIS weights depend on log-likelihood differences only, so every per-observation
# quantity but elpd_loo itself is invariant to the shift.
t_start <- proc.time()[["elapsed"]]
suppressPackageStartupMessages(library(loo))
cat(sprintf("loo %s, R %s\n", packageVersion("loo"), getRversion()))
set.seed(7)
ll <- matrix(rnorm(1000 * 5, -1, 0.4), 1000)
ll[, 5] <- rnorm(1000, -400, 0.4)
ll <- round(ll * 2^40) / 2^40
sh <- ll; sh[, 5] <- sh[, 5] + 400
a <- suppressWarnings(loo(ll, r_eff = rep(1, 5))); b <- suppressWarnings(loo(sh, r_eff = rep(1, 5)))
pa <- a$pointwise; pb <- b$pointwise
cat("observation 5 (log-likelihood near -400):\n")
cat(sprintf("  elpd_loo         %.12g  (shifted row: %.12g, + 400 = %.12g)\n", pa[5, "elpd_loo"], pb[5, "elpd_loo"], pb[5, "elpd_loo"] - 400))
cat(sprintf("  mcse_elpd_loo    %s  (expected, the shifted row's: %.12g)\n", format(pa[5, "mcse_elpd_loo"]), pb[5, "mcse_elpd_loo"]))
cat(sprintf("  p_loo            %.12g  (shifted row: %.12g)\n", pa[5, "p_loo"], pb[5, "p_loo"]))
cat(sprintf("  influence_pareto_k %.12g  (shifted row: %.12g)\n", pa[5, "influence_pareto_k"], pb[5, "influence_pareto_k"]))
mc_line <- function(x) { l <- grep("MCSE of elpd_loo", capture.output(print(x)), value = TRUE); if (length(l)) trimws(l[1]) else "(not printed)" }
cat(sprintf("print(loo) says: %s\nprint(loo of the shifted matrix) says: %s\n", mc_line(a), mc_line(b)))
present <- !is.finite(pa[5, "mcse_elpd_loo"])
cat(if (present) "\nPROBLEM PRESENT: loo's MCSE of elpd_loo is NaN for an observation whose elpd is below about -372\n" else "\nproblem absent\n")
cat(sprintf("run time %.1f s\n", proc.time()[["elapsed"]] - t_start))
quit(status = if (present) 1 else 0)
