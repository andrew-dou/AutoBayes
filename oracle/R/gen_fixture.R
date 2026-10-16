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
# Opt-in per case (Tier C); a case without these keys is generated exactly as before:
#
#   na_columns      numeric columns whose empty fields are missing values (NA), for mi()
#   mvbf            a list of formulas, each (a string, or a list as `formula`) one brmsformula(), added
#                   together into one multivariate model: bf(y ~ mi(xm) + w) + bf(xm | mi() ~ w)
#   rescor          set_rescor(rescor) added to the formula of a multivariate model
#   family$mixture  a list of component families (each as `family`): brms::mixture() of them, with the
#                   family's other entries (e.g. order) as mixture()'s arguments
#   extras          further outputs: "group_scope" (each group term's resp, dpar and nlpar), "responses"
#                   (brms's response names, the last axis of a multivariate model's predictions),
#                   "log_lik", "bayes_R2" and "loo" (the `post` block: log_lik(), bayes_R2(summary = FALSE)
#                   and loo() of the fixed-parameter fit)
#   point_halfwidth the seeded points are uniform on (-h, h) in every unconstrained coordinate (default 1.2)
#
# Opt-in per case (Tier D), each adding to the `post` block; a case without these keys is generated exactly as before:
#
#   extras          "waic": waic() of the fixed-parameter fit (pointwise elpd_waic, p_waic, waic; estimates)
#   loo_compare     the frozen brms section of another case on the same data (its id, seed, n_points, formula, family,
#                   factors, priors): that model is compiled and evaluated on its own seeded points exactly as its own
#                   fixture is, and `post$loo_compare` records loo_compare() of the two fits by "loo" and by "waic", each
#                   model's pointwise elpd_loo and elpd_waic, model_weights() by "loo" and by "pseudobma" (BB = FALSE), and
#                   the other model's log density at its points (to tie it to its own fixture)
#   hypothesis      a list of hypothesis strings: hypothesis() of the fit (class "b", alpha 0.05), its table and draws
#                   (brms evaluates each hypothesis on the draws' coefficient columns, as it parses a prior string)
#   conditional_effects   a list of effects: conditional_effects() of the fit (brms's defaults: posterior_epred,
#                   re_formula = NA, robust summaries, 100 grid points), one table per effect
#
# Nothing here samples, and nothing here evaluates code from the case: the formula is parsed with
# as.formula(), the family is built with brms::brmsfamily() from named arguments, priors with set_prior().
# Everything is deterministic given the case, the data and the package versions; a prediction or
# post-processing function that draws random numbers (brms does for an exact gp() or me() on new data) is
# caught by computing it under two seeds, and stops the generation.
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
na_columns <- unlist(case$na_columns)                 # opt-in: numeric columns with missing values (mi())
extras <- unlist(case$extras)                         # opt-in: further outputs (see the header)
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
    } else if (col %in% na_columns) {                 # an empty field is NA; every other field must be a number
      v <- raw[[col]]
      num <- suppressWarnings(as.numeric(ifelse(v == "", NA_character_, v)))
      if (any(is.na(num) & v != "")) stop(sprintf("column %s has values that are neither numbers nor empty", col))
      out[[col]] <- num
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
if (is.null(case$mvbf)) {
  parts <- lapply(as.list(unlist(case$formula)), as.formula)
  fml <- if (length(parts) == 1 && is.null(case$bf_args)) parts[[1]] else do.call(brms::bf, c(unname(parts), case$bf_args %||% list()))
} else {                                              # opt-in: one brmsformula() per response, added together
  fml <- Reduce(`+`, lapply(case$mvbf, function(f) do.call(brms::bf, unname(lapply(as.list(unlist(f)), as.formula)))))
}
if (!is.null(case$rescor)) {                          # opt-in: a multivariate model's residual correlations
  if (!inherits(fml, "bform")) fml <- brms::bf(fml)
  fml <- fml + brms::set_rescor(isTRUE(case$rescor))
}
fam <- if (is.null(case$family$mixture)) {
  do.call(brms::brmsfamily, case$family)             # e.g. {"family": "bernoulli", "link": "probit"}
} else {                                              # opt-in: {"mixture": [{"family": "gaussian"}, ...], "order": ...}
  do.call(brms::mixture, c(lapply(case$family$mixture, function(f) do.call(brms::brmsfamily, f)),
                           case$family[setdiff(names(case$family), "mixture")]))
}
pri <- if (length(case$priors)) do.call(c, lapply(case$priors, function(p) do.call(brms::set_prior, p))) else NULL
t0 <- Sys.time()
fit0 <- do.call(brm, c(list(formula = fml, data = d, family = fam, prior = pri, chains = 0,
                            backend = "rstan", silent = 2, refresh = 0), case$brm_args %||% list()))
compile_seconds <- as.numeric(Sys.time() - t0, units = "secs")
sf <- fit0$fit
sdat <- standata(fit0)
code <- stancode(fit0)

# -- encoders ---------------------------------------------------------------------------------------------
enc <- function(x, dims = NULL, allow_infinite = FALSE, allow_na = FALSE) {
  if (is.null(dims)) dims <- if (is.null(dim(x))) (if (length(x) == 1) integer(0) else length(x)) else dim(x)
  v <- as.numeric(as.vector(x))                       # column-major, as R stores it
  na <- is.na(v)
  if (any(na) && !allow_na) stop("missing value in a fixture array")
  if (any(!is.finite(v))) {
    if (any(!is.finite(v) & !na) && !allow_infinite) stop("non-finite value in a fixture array")
    # truncation bounds: strings; a missing value (Tier D, allow_na: hypothesis()'s evidence ratio without prior draws) is "NaN"
    v <- ifelse(na, "NaN", ifelse(is.finite(v), sprintf("%.17g", v), ifelse(v > 0, "Inf", "-Inf")))
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
halfwidth <- case$point_halfwidth %||% 1.2            # opt-in: the box (-h, h) the points are drawn from
pts <- lapply(seq_len(n_points), function(k) runif(n, -halfwidth, halfwidth))
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
deterministic <- function(f) {                        # the same draws under two seeds, or the function samples
  set.seed(1); a <- f()
  set.seed(2); b <- f()
  if (!identical(a, b)) stop("a prediction or post-processing function draws random numbers; its output is not deterministic")
  a
}
pred_block <- function(newdata) {
  out <- list(linpred = enc(deterministic(function() posterior_linpred(fit1, newdata = newdata))),
              epred = enc(deterministic(function() posterior_epred(fit1, newdata = newdata))))
  if (has_groups) {
    out$linpred_nogroup <- enc(deterministic(function() posterior_linpred(fit1, newdata = newdata, re_formula = NA)))
    out$epred_nogroup <- enc(deterministic(function() posterior_epred(fit1, newdata = newdata, re_formula = NA)))
  }
  out
}
predictions <- list(fitted = pred_block(NULL))
if (!is.null(nd)) predictions$newdata <- pred_block(nd)

# -- opt-in post-processing of the fixed-parameter fit (C07): pointwise log-likelihood, Bayes R2, PSIS-LOO --------
# brms's loo() on this fit: the relative efficiencies of one-draw chains are NA, which loo() ignores (r_eff = 1),
# and a Pareto tail too short to fit (fewer than five draws in it) leaves raw importance weights and k-hat Inf.
post <- list()
if ("log_lik" %in% extras) post$log_lik <- enc(deterministic(function() log_lik(fit1)))           # (draws, rows)
if ("bayes_R2" %in% extras) post$bayes_R2 <- enc(deterministic(function() bayes_R2(fit1, summary = FALSE)))
if ("loo" %in% extras) {
  lo <- deterministic(function() { l <- loo(fit1); l[c("pointwise", "estimates", "diagnostics")] })
  post$loo <- list(pointwise = enc(lo$pointwise, allow_infinite = TRUE),
                   estimates = enc(lo$estimates), estimate_rows = I(rownames(lo$estimates)),
                   pareto_k = enc(lo$diagnostics$pareto_k, allow_infinite = TRUE),
                   n_eff = enc(lo$diagnostics$n_eff, allow_infinite = TRUE))
}

# -- opt-in post-processing (Tier D): WAIC, a comparison with another case's model, hypotheses, conditional effects --------
enc_frame <- function(df) {                           # a table column by column: numbers as arrays (NA as "NaN"), the rest as text
  df <- as.data.frame(df)
  list(columns = I(names(df)), values = lapply(setNames(names(df), names(df)), function(nm) {
    v <- df[[nm]]
    if (is.numeric(v)) enc(v, dims = length(v), allow_infinite = TRUE, allow_na = TRUE) else I(as.character(v))
  }))
}
if ("waic" %in% extras) {                             # D05: waic() of the fixed-parameter fit
  wa <- deterministic(function() { w <- waic(fit1); w[c("pointwise", "estimates")] })
  post$waic <- list(pointwise = enc(wa$pointwise), estimates = enc(wa$estimates), estimate_rows = I(rownames(wa$estimates)))
}
if (!is.null(case$loo_compare)) {                     # D05: the other case's model on this data, on its own seeded points
  other <- case$loo_compare
  if (!identical(other$factors, case$factors) || length(other$na_columns))
    stop("loo_compare: the other case must declare the same factors and no missing-value columns (it reads the same data)")
  oparts <- lapply(as.list(unlist(other$formula)), as.formula)
  ofml <- if (length(oparts) == 1 && is.null(other$bf_args)) oparts[[1]] else do.call(brms::bf, c(unname(oparts), other$bf_args %||% list()))
  opri <- if (length(other$priors)) do.call(c, lapply(other$priors, function(p) do.call(brms::set_prior, p))) else NULL
  ofit0 <- do.call(brm, c(list(formula = ofml, data = d, family = do.call(brms::brmsfamily, other$family), prior = opri,
                               chains = 0, backend = "rstan", silent = 2, refresh = 0), other$brm_args %||% list()))
  osf <- ofit0$fit
  on <- get_num_upars(osf)
  set.seed(other$seed)                                # as the other case's own fixture draws its points
  ohw <- other$point_halfwidth %||% 1.2
  opts <- lapply(seq_len(other$n_points), function(k) runif(on, -ohw, ohw))
  oinits <- lapply(opts, function(u) constrain_pars(osf, u))
  ofit1 <- ofit0
  ofit1$fit <- sampling(osf@stanmodel, data = standata(ofit0), algorithm = "Fixed_param", chains = other$n_points, iter = 1,
                        warmup = 0, init = oinits, refresh = 0, seed = 1, cores = 1)
  ofit1 <- rename_pars(ofit1)
  odm <- posterior::as_draws_matrix(ofit1)
  for (k in seq_len(other$n_points)) {
    for (nm in intersect(intersect(c("lprior", "b_Intercept", "sigma"), colnames(odm)), names(oinits[[k]]))) {
      if (abs(odm[k, nm] - oinits[[k]][[nm]]) > 1e-12) stop(sprintf("loo_compare: draw %d of the other model does not reproduce its point %d", k, k))
    }
  }
  ids <- c(case$id, other$id)
  cmp <- deterministic(function() {
    a <- add_criterion(fit1, c("loo", "waic"), model_name = ids[1])
    b <- add_criterion(ofit1, c("loo", "waic"), model_name = ids[2])
    list(loo = loo_compare(a, b, criterion = "loo", model_names = ids),
         waic = loo_compare(a, b, criterion = "waic", model_names = ids),
         elpd_loo = cbind(a$criteria$loo$pointwise[, "elpd_loo"], b$criteria$loo$pointwise[, "elpd_loo"]),
         elpd_waic = cbind(a$criteria$waic$pointwise[, "elpd_waic"], b$criteria$waic$pointwise[, "elpd_waic"]),
         weights_loo = model_weights(a, b, weights = "loo", model_names = ids),
         weights_pseudobma = model_weights(a, b, weights = "pseudobma", BB = FALSE, model_names = ids))
  })
  enc_compare <- function(m) list(table = enc(unclass(m)), rows = I(rownames(m)))   # brms's table: best model first
  colnames(cmp$elpd_loo) <- colnames(cmp$elpd_waic) <- ids                         # (rows, 2): this case, then the other
  post$loo_compare <- list(
    models = I(ids), loo = enc_compare(cmp$loo), waic = enc_compare(cmp$waic),
    elpd_loo = enc(cmp$elpd_loo), elpd_waic = enc(cmp$elpd_waic),
    weights_loo = enc(cmp$weights_loo, dims = 2L), weights_pseudobma = enc(cmp$weights_pseudobma, dims = 2L),
    weights_models = I(names(cmp$weights_loo)),
    other_lp = enc(vapply(opts, function(u) as.numeric(log_prob(osf, u, adjust_transform = TRUE)), 0.0), dims = length(opts)))
}
if (length(case$hypothesis)) {                        # D06: brms's defaults (class "b", alpha 0.05, robust = FALSE)
  hy <- deterministic(function() { h <- hypothesis(fit1, unlist(case$hypothesis)); h[c("hypothesis", "samples")] })
  post$hypothesis <- list(table = enc_frame(hy$hypothesis), samples = enc(as.matrix(hy$samples)))   # samples: (draws, hypotheses)
}
if (length(case$conditional_effects)) {               # D06: one table per effect, in the order given
  ce <- deterministic(function() conditional_effects(fit1, effects = unlist(case$conditional_effects)))
  post$conditional_effects <- lapply(ce, enc_frame)
}

# -- tables ------------------------------------------------------------------------------------------------
groups <- list()
if (has_groups) {
  re <- fit0$ranef
  lv <- attr(re, "levels")
  groups <- lapply(seq_len(nrow(re)), function(i) {
    g <- list(
      id = as.integer(re$id[i]), group = as.character(re$group[i]), coef = as.character(re$coef[i]),
      cn = as.integer(re$cn[i]), cor = isTRUE(re$cor[i]), levels = I(as.character(lv[[re$group[i]]])))
    if ("group_scope" %in% extras)                    # opt-in: the response, distributional or non-linear part it is in
      g[c("resp", "dpar", "nlpar")] <- list(as.character(re$resp[i]), as.character(re$dpar[i]), as.character(re$nlpar[i]))
    g
  })
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
if ("responses" %in% extras) {                        # opt-in: the last axis of a multivariate model's predictions
  resp <- as.character(fit0$formula$responses)
  if (!identical(dimnames(posterior_linpred(fit1))[[3]], resp)) stop("brms's responses are not the prediction array's last axis")
  fixture$responses <- I(resp)
}
if (length(post)) fixture$post <- post
js <- toJSON(fixture, digits = I(17), auto_unbox = TRUE, null = "null", pretty = FALSE)
back <- fromJSON(js, simplifyVector = FALSE)
for (k in seq_len(n_points)) {                        # 17 significant digits round-trip exactly; check, not trust
  if (!identical(back$points[[k]]$lp, points[[k]]$lp)) stop("fixture JSON does not round-trip the log density")
}
writeLines(js, file.path(dir, "fixture.json"))
cat(sprintf("fixture %s: %d unconstrained parameters, %d points, compile %.1f s, stancode %s\n",
            case$id, n, n_points, compile_seconds, substr(fixture$meta$stancode_sha256, 1, 12)))
