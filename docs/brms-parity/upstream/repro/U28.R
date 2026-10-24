#!/usr/bin/env Rscript
# U28 (D126): loo's stacking_weights() (behind loo_model_weights()'s default method and brms's model_weights(weights =
# "stacking")) stops its optimiser (stats::constrOptim, a log-barrier method) at its tolerance, short of the optimum of the
# stacking objective: the weights are off by up to 3.7e-4 here and never exactly 0 for a model the optimum excludes.
#
#   Rscript docs/brms-parity/upstream/repro/U28.R      # exit status 1 while the problem is present
#
# loo only (no brms, nothing sampled). Inputs: pointwise elpd matrices drawn with a fixed seed. Exact: the maximiser of
# F(w) = sum_i log(sum_k w_k exp(lpd_ik)) on the simplex, found by EM-style multiplicative updates and then Newton's method
# on the active set to the KKT conditions (gradient g_k = sum_i exp(lpd_ik) / p_i equal to N on the support, at most N off
# it), in double precision.
t_start <- proc.time()[["elapsed"]]
suppressPackageStartupMessages(library(loo))
cat(sprintf("loo %s, R %s\n", packageVersion("loo"), getRversion()))
optimum <- function(lpd) {
  A <- exp(lpd - apply(lpd, 1, max)); N <- nrow(A); K <- ncol(A)
  w <- rep(1 / K, K)
  for (it in 1:20000) { p <- drop(A %*% w); w <- w * colSums(A / p) / N }
  for (round in 1:50) {
    S <- which(w > 1e-12); w[-S] <- 0; w <- w / sum(w)
    for (it in 1:100) {
      p <- drop(A %*% w); g <- colSums(A / p)[S]; H <- -crossprod(A[, S, drop = FALSE] / p)
      M <- rbind(cbind(H, 1), c(rep(1, length(S)), 0))
      step <- solve(M, c(-(g - N), 0))[seq_along(S)]
      wn <- w; wn[S] <- w[S] + step
      if (any(wn[S] < 0)) { w[S] <- pmax(wn[S], 0); w <- w / sum(w); break }
      w <- wn
      if (max(abs(step)) < 1e-17) break
    }
    p <- drop(A %*% w); gall <- colSums(A / p)
    if (all(gall[w == 0] <= N * (1 + 1e-12)) && all(w >= 0)) break
    w[w == 0 & gall > N] <- 1e-6; w <- w / sum(w)
  }
  w
}
objective <- function(lpd, w) { m <- apply(lpd, 1, max); sum(m + log(drop(exp(lpd - m) %*% w))) }
set.seed(20261017)
el <- function(N, centres, sds) sapply(seq_along(centres), function(k) rnorm(N, centres[k], sds[k]))
shared <- rnorm(80, -1.2, 0.6)
sets <- list(
  two_36 = el(36, c(-2, -2.2), c(1.5, 0.8)),
  two_vertex = cbind(rnorm(30, -1, 0.2), rnorm(30, -4, 0.2)),
  two_nearly_equal = cbind(shared, shared + rnorm(80, 0, 1e-3)),
  three_50 = el(50, c(-1.5, -1.6, -1.8), c(0.7, 1.1, 0.4)),
  four_one_dominated = cbind(el(60, c(-1.4, -1.5, -1.6), c(0.6, 0.9, 1.2)), rnorm(60, -6, 0.3)),
  five_200 = el(200, c(-1.3, -1.35, -1.4, -1.5, -1.7), c(0.5, 0.8, 1.0, 0.6, 1.4)))
present <- FALSE
for (nm in names(sets)) {
  lpd <- sets[[nm]]
  wl <- as.numeric(suppressWarnings(suppressMessages(stacking_weights(lpd))))
  wx <- optimum(lpd)
  cat(sprintf("\n%s (N = %d, K = %d)\n  loo   %s\n  exact %s\n", nm, nrow(lpd), ncol(lpd),
              paste(sprintf("%.12f", wl), collapse = " "), paste(sprintf("%.12f", wx), collapse = " ")))
  cat(sprintf("  largest weight difference %.2g; objective at loo's weights %.3g below the optimum's; exact zeros: %d (loo: %d)\n",
              max(abs(wl - wx)), objective(lpd, wx) - objective(lpd, wl), sum(wx == 0), sum(wl == 0)))
  if (objective(lpd, wx) < objective(lpd, wl) - 1e-12) { cat("  the reference optimum is worse than loo's: the script is wrong\n"); quit(status = 2) }
  if (max(abs(wl - wx)) > 1e-9) present <- TRUE
}
cat(if (present) "\nPROBLEM PRESENT: loo's stacking weights stop short of the optimum\n" else "\nproblem absent\n")
cat(sprintf("run time %.1f s\n", proc.time()[["elapsed"]] - t_start))
quit(status = if (present) 1 else 0)
