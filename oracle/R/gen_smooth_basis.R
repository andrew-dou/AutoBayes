#!/usr/bin/env Rscript
# Freeze mgcv's thin-plate regression spline basis for s(x), as brms builds it, on seeded one-dimensional data.
#
#   Rscript oracle/R/gen_smooth_basis.R <out_dir> [<id> ...]      (run from the repository root)
#
# For every data set below (or the ids given) this writes <out_dir>/<id>.json and never overwrites an existing
# file: a changed data set gets a new id. brms 2.23 builds s(x) in brms:::data_sm / brms:::frame_basis_sm as
#
#   sm <- mgcv::smoothCon(s(x, k = k), data, knots = NULL, absorb.cons = TRUE, diagonal.penalty = TRUE)[[1]]
#   re <- mgcv::smooth2random(sm, names(data), type = 2)       # Xs = re$Xf, Zs_1_1 = re$rand[[1]]
#
# and for new data (and for the fitted rows in standata(fit)) brms:::s2rPred(sm, re, newdata), which applies
# mgcv::PredictMat to the stored construction. The file records every stage: the raw tprs construction
# (smooth.construct: shift, knots Xu, UZ, X, S), smoothCon's output (S, S.scale, the constraint's QR, diagRP; its X
# is smooth2random's Xf and rand in another column order), smooth2random's (Xf, rand, trans.U, trans.D), PredictMat
# and s2rPred on the new rows, and s2rPred on the fitted rows (what standata(fit) returns).
# Arrays are {"dims", "values"} with column-major values, as in tests/oracle/fixtures (tests/oracle/fixture.py
# decodes them). Nothing here samples; everything is deterministic given the seeds and the package versions.
suppressPackageStartupMessages({
  library(mgcv)
  library(brms)
  library(jsonlite)
})
options(warn = 1)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("usage: Rscript oracle/R/gen_smooth_basis.R <out_dir> [<id> ...]")
out_dir <- args[[1]]
only <- if (length(args) > 1) args[-1] else NULL
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# -- the data sets ---------------------------------------------------------------------------------------------------
# `x` is generated after set.seed(seed); `k = -1` is mgcv's default (k = 10); `xnew` is where PredictMat is evaluated
# (inside the range, at the ends, and beyond them). `role` is "standard" (ordinary data: the port is held to the
# parity tolerances) or "conditioning" (outlying values, or k close to the number of unique values, so that mgcv's
# own rounding is amplified; see `mgcv_rounding` below).
fixture_x <- function(file) read.csv(file.path("tests/oracle/fixtures/gauss_spline", file))$x
grid_new <- function(x) {                            # extrapolation by a fifth and a tenth of the range
  r <- diff(range(x))
  c(min(x) - 0.2 * r, min(x), quantile(x, c(0.1, 0.37, 0.5, 0.81), names = FALSE), max(x), max(x) + 0.1 * r)
}
sets <- list(
  list(id = "gauss_spline", role = "standard", seed = NA, k = -1,
       what = "the x column of the gauss_spline oracle fixture (data.csv), new rows from its newdata.csv plus the ends",
       x = function() fixture_x("data.csv"), xnew = function(x) c(fixture_x("newdata.csv"), min(x) - 0.5, max(x) + 0.5)),
  list(id = "normal_n50", role = "standard", seed = 101, k = -1, what = "rnorm(50)",
       x = function() rnorm(50)),
  list(id = "uniform_n100_k8", role = "standard", seed = 102, k = 8, what = "runif(100, -2, 7), k = 8",
       x = function() runif(100, -2, 7)),
  list(id = "exponential_n80_k12", role = "standard", seed = 103, k = 12, what = "3 * rexp(80) (skewed spacing), k = 12",
       x = function() 3 * rexp(80)),
  list(id = "ties_rounded_n60", role = "standard", seed = 104, k = -1, what = "round(rnorm(60, 1, 2), 1) (ties)",
       x = function() round(rnorm(60, 1, 2), 1)),
  list(id = "ties_gapped_integers_n70_k6", role = "standard", seed = 105, k = 6,
       what = "70 draws from the integers 1 2 3 5 8 9 13 14 17 20 (ties, uneven gaps), k = 6",
       x = function() sample(c(1, 2, 3, 5, 8, 9, 13, 14, 17, 20), 70, replace = TRUE)),
  list(id = "small_n12_k5", role = "standard", seed = 106, k = 5, what = "rnorm(12), k = 5",
       x = function() rnorm(12)),
  list(id = "k3_n30", role = "standard", seed = 107, k = 3, what = "rnorm(30), k = 3 (smoothCon skips the diagonal reparameterisation)",
       x = function() rnorm(30)),
  list(id = "k4_n30", role = "standard", seed = 108, k = 4, what = "runif(30), k = 4",
       x = function() runif(30)),
  list(id = "unique_equals_k_n20_k6", role = "standard", seed = 109, k = 6,
       what = "20 draws from 6 distinct runif(6, 0, 10) values, k = 6 (exactly k knots: mgcv keeps the full basis)",
       x = function() { v <- runif(6, 0, 10); c(v, sample(v, 14, replace = TRUE)) }),
  list(id = "normal_n150", role = "standard", seed = 110, k = -1, what = "3 + 5 * rnorm(150)",
       x = function() 3 + 5 * rnorm(150)),
  list(id = "offset_mean_n40", role = "standard", seed = 111, k = -1, what = "1000 + rnorm(40) (a large mean to centre)",
       x = function() 1000 + rnorm(40)),
  list(id = "tiny_scale_n40", role = "standard", seed = 112, k = -1, what = "1e-3 * rnorm(40)",
       x = function() 1e-3 * rnorm(40)),
  list(id = "symmetric_full_basis_n50_k10", role = "standard", seed = NA, k = -1,
       what = "rep(1:10, 5): symmetric knots, but exactly k of them, so mgcv keeps the full basis and nothing is ambiguous",
       x = function() rep(1:10, 5)),
  list(id = "outliers_n40", role = "conditioning", seed = 113, k = -1, what = "c(rnorm(37), 25, 26, 40) (outlying values)",
       x = function() c(rnorm(37), 25, 26, 40)),
  list(id = "k_near_unique_n20_k15", role = "conditioning", seed = 114, k = 15, what = "runif(20, -2, 7), k = 15",
       x = function() runif(20, -2, 7)),
  list(id = "unique_equals_k_n25_k10", role = "conditioning", seed = 115, k = -1,
       what = "25 draws from 10 distinct runif(10, 0, 10) values, default k = 10 (exactly k knots)",
       x = function() { v <- runif(10, 0, 10); c(v, sample(v, 15, replace = TRUE)) })
)

# mgcv's own rounding: the construction is exactly invariant under x -> a x + b (a > 0) in exact arithmetic (the
# shift absorbs b, the unit-RMS column scaling and the penalty scaling absorb a), so the largest change of Xf, and of
# Zs after its signed column permutation, under these maps measures how far mgcv's output is from itself.
mgcv_rounding <- function(x, k, Xf, Zs) {
  out <- c(Xf = 0, Zs = 0)
  for (ab in list(c(1.1, 0), c(0.7, 0), c(3.3, 0), c(1, 0.37), c(1.3, -2.1))) {
    xt <- ab[1] * x + ab[2] * sd(x)
    sm2 <- smoothCon(s(x, k = k), data = data.frame(x = xt), knots = NULL, absorb.cons = TRUE, diagonal.penalty = TRUE)[[1]]
    re2 <- smooth2random(sm2, "x", type = 2)
    P <- round(qr.solve(re2$rand[[1]], Zs))
    if (any(rowSums(abs(P)) != 1) || any(colSums(abs(P)) != 1)) stop("no signed permutation between equivalent constructions")
    out["Xf"] <- max(out["Xf"], max(abs(re2$Xf - Xf)))
    out["Zs"] <- max(out["Zs"], max(abs(re2$rand[[1]] %*% P - Zs)))
  }
  out
}

# -- encoders (as oracle/R/gen_fixture.R) ----------------------------------------------------------------------------
enc <- function(x) {
  if (is.null(x)) return(NULL)
  dims <- if (is.null(dim(x))) (if (length(x) == 1) integer(0) else length(x)) else dim(x)
  v <- as.numeric(as.vector(x))
  if (any(!is.finite(v))) stop("non-finite value in a smooth-basis array")
  list(dims = I(as.integer(dims)), values = I(v))
}
vec <- function(x) enc(array(as.numeric(x), length(x)))   # always a vector, even of length one

s2rPred <- brms:::s2rPred
for (st in sets) {
  if (!is.null(only) && !(st$id %in% only)) next
  path <- file.path(out_dir, paste0(st$id, ".json"))
  if (file.exists(path)) {
    cat(sprintf("%s: already frozen (a changed data set needs a new id)\n", st$id))
    next
  }
  if (!is.na(st$seed)) set.seed(st$seed)
  x <- as.numeric(st$x())
  xnew <- as.numeric(if (is.null(st$xnew)) grid_new(x) else st$xnew(x))
  dat <- data.frame(x = x)
  nd <- data.frame(x = xnew)
  spec <- s(x, k = st$k)
  tp <- smooth.construct(spec, dat, NULL)
  sm <- smoothCon(spec, data = dat, knots = NULL, absorb.cons = TRUE, diagonal.penalty = TRUE)[[1]]
  re <- smooth2random(sm, names(dat), type = 2)
  pn <- s2rPred(sm, re, nd)
  pt <- s2rPred(sm, re, dat)
  qrc <- attr(sm, "qrc")
  if (!inherits(qrc, "qr") || !is.null(attr(sm, "indi"))) stop(st$id, ": unexpected constraint representation")
  if (length(re$rand) != 1) stop(st$id, ": expected one random block")
  rnd <- mgcv_rounding(x, st$k, re$Xf, re$rand[[1]])
  out <- list(
    meta = list(id = st$id, role = st$role, what = st$what, seed = if (is.na(st$seed)) NULL else st$seed, k = st$k,
                generator = "oracle/R/gen_smooth_basis.R", mgcv = as.character(packageVersion("mgcv")),
                brms = as.character(packageVersion("brms")), R = paste(R.version$major, R.version$minor, sep = "."),
                platform = R.version$platform, lapack = La_version(),
                calls = paste("smoothCon(s(x, k = k), data, knots = NULL, absorb.cons = TRUE, diagonal.penalty = TRUE);",
                              "smooth2random(sm, 'x', type = 2); brms:::s2rPred(sm, re, newdata)"),
                mgcv_rounding = list(Xf = unname(rnd["Xf"]), Zs = unname(rnd["Zs"]),
                                     how = "max change under x -> a x + b sd(x), (a, b) in (1.1, 0) (0.7, 0) (3.3, 0) (1, 0.37) (1.3, -2.1)")),
    x = vec(x), xnew = vec(xnew),
    tprs = list(bs_dim = tp$bs.dim, shift = vec(tp$shift), Xu = vec(tp$Xu[, 1]), UZ = enc(tp$UZ), X = enc(tp$X),
                S = enc(tp$S[[1]]), rank = tp$rank, null_space_dim = tp$null.space.dim),
    smooth = list(S = enc(sm$S[[1]]), S_scale = vec(sm$S.scale), diagRP = enc(sm$diagRP),
                  qr = enc(qrc$qr), qraux = vec(qrc$qraux), rank = sm$rank, null_space_dim = sm$null.space.dim,
                  df = sm$df),
    random = list(Xf = enc(re$Xf), Zs = enc(re$rand[[1]]), trans_U = enc(re$trans.U), trans_D = vec(re$trans.D),
                  rind = vec(re$rind), pen_ind = vec(re$pen.ind)),
    predict_new = list(PredictMat = enc(PredictMat(sm, nd)), Xf = enc(pn$Xf), Zs = enc(pn$rand[[1]])),
    predict_fitted = list(Xf = enc(pt$Xf), Zs = enc(pt$rand[[1]]))
  )
  js <- toJSON(out, digits = I(17), auto_unbox = TRUE, null = "null", pretty = FALSE)
  back <- fromJSON(js, simplifyVector = FALSE)
  if (!identical(as.numeric(unlist(back$random$Xf$values)), as.numeric(re$Xf)))
    stop(st$id, ": JSON does not round-trip Xf at 17 significant digits")
  if (!identical(as.numeric(unlist(back$x$values)), x)) stop(st$id, ": JSON does not round-trip x")
  writeLines(js, path)
  cat(sprintf("%s: n %d, %d unique, k %d, Zs %d columns -> %s\n", st$id, length(x), nrow(tp$Xu), tp$bs.dim,
              ncol(re$rand[[1]]), path))
}
