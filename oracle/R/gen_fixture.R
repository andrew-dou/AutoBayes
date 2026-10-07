#!/usr/bin/env Rscript
# Generate one brms oracle fixture.
#
#   Rscript oracle/R/gen_fixture.R <fixture_dir>
#
# <fixture_dir> holds the frozen brms-side case (`case_brms.json`, written by oracle/regen.py from the
# case's `brms` section), `data.csv` and optionally `newdata.csv`. This script compiles the brms model
# with rstan and writes `fixture.json`:
#
#   standata        every entry of brms::standata(), with dims (column-major values) and column names
#   stancode_sha256 the hash of the generated Stan program (the program itself is GPL-3 and not stored)
#   declarations    the Stan `parameters` block, one declaration per parameter
#   upars           the map from each unconstrained coordinate to (parameter, element, transform), found
#                   by perturbing constrained values and watching rstan::unconstrain_pars, never assumed
#   points          seeded unconstrained points with constrain_pars(), log_prob() and its gradient, with
#                   and without the Jacobian adjustment
#   predictions     posterior_linpred()/posterior_epred() of a fixed-parameter "fit" whose draws are
#                   exactly those points, on the fitted rows and on newdata, with and without group effects
#   groups          brms's group-level term table with its level labels
#   prior_summary / default_prior   brms's resolved and default prior tables
#
# Nothing here samples, and nothing here evaluates code from the case: the formula is parsed with
# as.formula(), the family is built with brms::brmsfamily() from named arguments, priors with set_prior().
# Everything is deterministic given the case, the data and the package versions.
suppressPackageStartupMessages({
  library(brms)
  library(rstan)
  library(jsonlite)
})
invisible(Sys.setlocale("LC_COLLATE", "C"))          # factor levels sort as Python's sorted() does
options(warn = 1)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("usage: Rscript oracle/R/gen_fixture.R <fixture_dir>")
dir <- args[[1]]
case <- fromJSON(file.path(dir, "case_brms.json"), simplifyVector = FALSE)
`%||%` <- function(a, b) if (is.null(a)) b else a

# -- data ------------------------------------------------------------------------------------------------
read_frame <- function(path) {
  raw <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE, colClasses = "character")
  out <- raw
  for (col in names(raw)) {
    lv <- case$factors[[col]]
    if (!is.null(lv)) {
      lv <- unlist(lv)
      bad <- setdiff(unique(raw[[col]]), lv)
      if (length(bad)) stop(sprintf("column %s has values outside its declared levels: %s", col, paste(bad, collapse = ", ")))
      out[[col]] <- factor(raw[[col]], levels = lv)
    } else {
      num <- suppressWarnings(as.numeric(raw[[col]]))
      out[[col]] <- if (anyNA(num)) raw[[col]] else num   # a label column the case does not use stays text
    }
  }
  out
}
d <- read_frame(file.path(dir, "data.csv"))
nd <- if (file.exists(file.path(dir, "newdata.csv"))) read_frame(file.path(dir, "newdata.csv")) else NULL

# -- model ------------------------------------------------------------------------------------------------
# `formula` is one formula string or a list of them (the first is the response formula, the rest are
# distributional or non-linear parts); `bf_args` are brmsformula() options such as center = FALSE.
parts <- lapply(as.list(unlist(case$formula)), as.formula)
fml <- if (length(parts) == 1 && is.null(case$bf_args)) parts[[1]] else do.call(brms::bf, c(unname(parts), case$bf_args %||% list()))
fam <- do.call(brms::brmsfamily, case$family)        # e.g. {"family": "bernoulli", "link": "probit"}
pri <- if (length(case$priors)) do.call(c, lapply(case$priors, function(p) do.call(brms::set_prior, p))) else NULL
t0 <- Sys.time()
fit0 <- do.call(brm, c(list(formula = fml, data = d, family = fam, prior = pri, chains = 0,
                            backend = "rstan", silent = 2, refresh = 0), case$brm_args %||% list()))
compile_seconds <- as.numeric(Sys.time() - t0, units = "secs")
sf <- fit0$fit
sdat <- standata(fit0)
code <- stancode(fit0)

# -- encoders ---------------------------------------------------------------------------------------------
enc <- function(x, dims = NULL, allow_infinite = FALSE) {
  if (is.null(dims)) dims <- if (is.null(dim(x))) (if (length(x) == 1) integer(0) else length(x)) else dim(x)
  v <- as.numeric(as.vector(x))                       # column-major, as R stores it
  if (any(is.na(v))) stop("missing value in a fixture array")
  if (any(!is.finite(v))) {
    if (!allow_infinite) stop("non-finite value in a fixture array")
    v <- ifelse(is.finite(v), sprintf("%.17g", v), ifelse(v > 0, "Inf", "-Inf"))   # truncation bounds: strings
  }
  out <- list(dims = I(as.integer(dims)), values = I(v))
  cn <- if (length(dim(x)) == 2) dimnames(x)[[2]] else NULL    # colnames() errors on a 1-d array with names
  if (!is.null(cn)) out$colnames <- I(cn)
  out
}
rows <- function(df) {
  df <- as.data.frame(df, stringsAsFactors = FALSE)
  lapply(seq_len(nrow(df)), function(i) lapply(as.list(df[i, , drop = FALSE]), function(v) as.character(v)))
}

# -- unconstrained parameter map by probing ----------------------------------------------------------------
n <- get_num_upars(sf)
u0 <- rep(0.3, n)
base <- constrain_pars(sf, u0)
if (max(abs(unconstrain_pars(sf, base) - u0)) > 1e-10) stop("constrain/unconstrain round trip failed")
upar_map <- vector("list", n)
for (nm in names(base)) {
  v <- base[[nm]]
  for (k in seq_along(v)) {
    cp <- base
    cp[[nm]][k] <- v[k] * 1.5
    u1 <- tryCatch(unconstrain_pars(sf, cp), error = function(e) NULL)
    if (is.null(u1)) next                              # an invalid perturbation: a jointly constrained block
    changed <- which(abs(u1 - u0) > 1e-12)
    if (length(changed) == 0) next                     # a transformed parameter or generated quantity
    du <- u1[changed] - u0[changed]
    if (length(changed) == 1) {
      transform <- if (abs(du - 0.5 * v[k]) < 1e-9) "identity" else if (abs(du - log(1.5)) < 1e-9) "log" else "other"
      if (!is.null(upar_map[[changed]])) {                 # ordered thresholds: a later element also moves it
        upar_map[[changed]]$transform <- "joint"
      } else {
        upar_map[[changed]] <- list(index = changed, par = nm, elem = k, transform = transform, du = du)
      }
    } else {
      for (j in seq_along(changed)) {
        if (is.null(upar_map[[changed[j]]]))
          upar_map[[changed[j]]] <- list(index = changed[j], par = nm, elem = k, transform = "joint", du = du[j])
      }
    }
  }
}
for (i in which(vapply(upar_map, is.null, TRUE))) {    # a jointly constrained block (a Cholesky factor): the
  upar_map[[i]] <- list(index = i, par = NA, elem = NA, transform = "unreached", du = NA)   # adapter needs its own map
}

# -- seeded points: constrained values, log density, gradient --------------------------------------------
set.seed(case$seed)
n_points <- case$n_points %||% 8L
pts <- lapply(seq_len(n_points), function(k) runif(n, -1.2, 1.2))
points <- lapply(pts, function(u) {
  cp <- constrain_pars(sf, u)
  lj <- log_prob(sf, u, adjust_transform = TRUE, gradient = TRUE)
  ln <- log_prob(sf, u, adjust_transform = FALSE, gradient = TRUE)
  vals <- c(lj, ln, attr(lj, "gradient"), attr(ln, "gradient"))
  if (any(!is.finite(vals))) stop("non-finite log density or gradient at a seeded point")
  list(u = enc(u), pars = lapply(setNames(names(cp), names(cp)), function(nm) enc(cp[[nm]], sf@par_dims[[nm]])),
       lp = as.numeric(lj), grad = enc(attr(lj, "gradient")), lp_nojac = as.numeric(ln), grad_nojac = enc(attr(ln, "gradient")))
})

# -- a fixed-parameter "fit" whose draws are exactly the points, for brms's prediction functions ----------
inits <- lapply(pts, function(u) constrain_pars(sf, u))
sf2 <- sampling(sf@stanmodel, data = sdat, algorithm = "Fixed_param", chains = n_points, iter = 1, warmup = 0,
                init = inits, refresh = 0, seed = 1, cores = 1)
fit1 <- fit0
fit1$fit <- sf2
fit1 <- rename_pars(fit1)
dm <- posterior::as_draws_matrix(fit1)
for (k in seq_len(n_points)) {                        # draw k is point k, checked, not assumed
  for (nm in intersect(intersect(c("lprior", "b_Intercept", "sigma"), colnames(dm)), names(inits[[k]]))) {
    if (abs(dm[k, nm] - inits[[k]][[nm]]) > 1e-12) stop(sprintf("draw %d does not reproduce point %d (%s)", k, k, nm))
  }
}
has_groups <- !is.null(fit0$ranef) && nrow(fit0$ranef) > 0
pred_block <- function(newdata) {
  out <- list(linpred = enc(posterior_linpred(fit1, newdata = newdata)),
              epred = enc(posterior_epred(fit1, newdata = newdata)))
  if (has_groups) {
    out$linpred_nogroup <- enc(posterior_linpred(fit1, newdata = newdata, re_formula = NA))
    out$epred_nogroup <- enc(posterior_epred(fit1, newdata = newdata, re_formula = NA))
  }
  out
}
predictions <- list(fitted = pred_block(NULL))
if (!is.null(nd)) predictions$newdata <- pred_block(nd)

# -- tables ------------------------------------------------------------------------------------------------
groups <- list()
if (has_groups) {
  re <- fit0$ranef
  lv <- attr(re, "levels")
  groups <- lapply(seq_len(nrow(re)), function(i) list(
    id = as.integer(re$id[i]), group = as.character(re$group[i]), coef = as.character(re$coef[i]),
    cn = as.integer(re$cn[i]), cor = isTRUE(re$cor[i]), levels = I(as.character(lv[[re$group[i]]]))))
}
declarations <- local({
  lines <- strsplit(code, "\n", fixed = TRUE)[[1]]
  none <- setNames(list(), character(0))              # an empty JSON object, not an empty array
  s <- grep("^parameters \\{", lines)
  if (length(s) != 1) return(none)
  e <- grep("^\\}", lines); e <- min(e[e > s])
  body <- trimws(sub("//.*$", "", lines[(s + 1):(e - 1)]))
  body <- body[nzchar(body)]
  if (!length(body)) return(none)
  nm <- sub("^.*?([A-Za-z_][A-Za-z0-9_]*)\\s*;\\s*$", "\\1", body, perl = TRUE)
  as.list(setNames(body, nm))
})
tf <- tempfile(fileext = ".stan"); writeLines(code, tf)
sha <- function(path) unname(tools::sha256sum(path))

fixture <- list(
  meta = list(
    id = case$id, generator = "oracle/R/gen_fixture.R",
    brms = as.character(packageVersion("brms")), rstan = as.character(packageVersion("rstan")),
    stan = rstan::stan_version(), R = paste(R.version$major, R.version$minor, sep = "."), platform = R.version$platform,
    compile_seconds = round(compile_seconds, 1), stancode_sha256 = sha(tf),
    case_brms_sha256 = sha(file.path(dir, "case_brms.json")), data_sha256 = sha(file.path(dir, "data.csv")),
    newdata_sha256 = if (is.null(nd)) NULL else sha(file.path(dir, "newdata.csv")),
    n_upars = n, n_points = n_points),
  standata = lapply(setNames(names(sdat), names(sdat)), function(nm) enc(sdat[[nm]], allow_infinite = TRUE)),
  declarations = declarations,
  upars = upar_map,
  groups = groups,
  points = points,
  predictions = predictions,
  prior_summary = rows(prior_summary(fit0)),
  default_prior = rows(default_prior(fml, data = d, family = fam))
)
js <- toJSON(fixture, digits = I(17), auto_unbox = TRUE, null = "null", pretty = FALSE)
back <- fromJSON(js, simplifyVector = FALSE)
for (k in seq_len(n_points)) {                        # 17 significant digits round-trip exactly; check, not trust
  if (!identical(back$points[[k]]$lp, points[[k]]$lp)) stop("fixture JSON does not round-trip the log density")
}
writeLines(js, file.path(dir, "fixture.json"))
cat(sprintf("fixture %s: %d unconstrained parameters, %d points, compile %.1f s, stancode %s\n",
            case$id, n, n_points, compile_seconds, substr(fixture$meta$stancode_sha256, 1, 12)))
