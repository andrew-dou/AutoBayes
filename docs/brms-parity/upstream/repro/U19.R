#!/usr/bin/env Rscript
# U19 (D67): the default prior of a Gaussian process's length-scale, inv_gamma(a, b) with 1% of its mass below the smallest
# distance between distinct (scaled) inputs and 1% above the largest, is solved to a tolerance coarser than the 6 decimals
# it is printed (and written into the model) with: at x = (0, 0.5, 1) brms gives inv_gamma(45.631904, 31.399697), where
# the solution 45.6319046088..., 31.3996975373... rounds to 45.631905, 31.399698.
#
#   Rscript docs/brms-parity/upstream/repro/U19.R      # exit status 1 while the problem is present
#
# Stock CRAN brms only (get_prior(); nothing compiled or sampled). Exact: with 1 / l ~ Gamma(a, rate b), P(l > hi) = 0.01
# gives b = qgamma(0.01, a) (here hi = 1, the largest scaled distance) and P(l < lo) = 0.01 gives
# pgamma(b / lo, a) = 0.99, solved for a by uniroot() to 1e-14. The effect on a fit is negligible; it is a rounding
# discrepancy.
t_start <- proc.time()[["elapsed"]]
suppressPackageStartupMessages(library(brms))
cat(sprintf("brms %s, R %s\n", packageVersion("brms"), getRversion()))
present <- FALSE
for (x in list(c(0, 0.5, 1), c(0, 0.48188060200668892, 1), c(0, 0.3, 1))) {
  gp <- get_prior(y ~ gp(x), data = data.frame(y = c(1, 2, 3), x = x))
  printed <- gp$prior[gp$class == "lscale" & gp$prior != ""][1]
  ds <- as.vector(dist(x)); lo <- max(min(ds[ds > 0]) / max(ds), 0.01)
  a <- uniroot(function(a) pgamma(qgamma(0.01, a) / lo, a) - 0.99, c(0.1, 1000), tol = 1e-14)$root
  b <- qgamma(0.01, a)
  exact <- sprintf("inv_gamma(%.6f, %.6f)", a, b)
  cat(sprintf("\nx = (%s)\n  brms   %s\n  exact  %s   (a = %.12f, b = %.12f)\n", paste(format(x, digits = 17), collapse = ", "),
              printed, exact, a, b))
  pv <- as.numeric(regmatches(printed, gregexpr("[0-9.]+", printed))[[1]])
  differs <- any(abs(pv - round(c(a, b), 6)) > 5e-7)
  cat(if (differs) "  brms's printed values are not the correctly rounded solution\n" else "  the same numbers\n")
  if (differs) present <- TRUE
}
cat(if (present) "\nPROBLEM PRESENT: brms's printed length-scale prior is not the correctly rounded solution\n" else "\nproblem absent\n")
cat(sprintf("run time %.1f s\n", proc.time()[["elapsed"]] - t_start))
quit(status = if (present) 1 else 0)
