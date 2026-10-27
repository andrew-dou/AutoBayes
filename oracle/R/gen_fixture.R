#!/usr/bin/env Rscript
# Generate one brms oracle fixture.
#
#   Rscript oracle/R/gen_fixture.R <fixture_dir>
#
# <fixture_dir> holds the frozen brms-side case (`case_brms.json`, written by oracle/regen.py from the
# case's `brms` section), `data.csv`, optionally `newdata.csv` and a Tier F case's `data2_<name>.csv`. This script compiles the brms model
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
# Opt-in per case (Tier E), each adding to the `post` block; a case without these keys is generated exactly as before:
#
#   prior_draws     true, with brm_args sample_prior = "yes": prior_draws() of the fixed-parameter fit, whose prior_* generated
#                   quantities are drawn by Stan's RNG in the fixed-parameter run (seed 1, one chain per point); that run is made
#                   twice and must give the same draws. The seeded points' `pars` leave the prior_* quantities out, because
#                   constrain_pars() draws them anew on every call (everything else it returns must not change between calls).
#                   hypothesis() of such a fit has Savage-Dickey evidence ratios for point hypotheses (brms's density_ratio()
#                   of the posterior and prior draws); a point hypothesis on more than one parameter pairs its parameters'
#                   prior draws through a random permutation (R's RNG), which the two-seed check catches
#   hypothesis_seed hypothesis()'s seed argument (brms calls set.seed() with it): brms permutes the prior draws of every point
#                   hypothesis, which moves a one-parameter evidence ratio (a kernel density at the point, order-free) only in its
#                   last bit; the seed makes that bit reproducible. It would also make a many-parameter point hypothesis
#                   reproducible, but its ratio would then depend on R's RNG, so a case lists one-parameter point hypotheses only
#   extras          "loo_model_weights", with loo_compare: loo_model_weights() of the two fits (stacking, loo's default method)
#                   and model_weights(weights = "stacking"), which must be the same weights
#   dpar_epred      a list of distributional parameters: posterior_epred(dpar = ) of each on the fitted rows, (draws, rows), for a
#                   parameter the points do not carry (gen_extreme_value's xi, which brms derives from tmp_xi, the data, mu and
#                   sigma in the model block)
#   conditional_effects   an interaction effect "x:z" or "x:w" is computed as any effect (brms's default int_conditions: the
#                   levels of a factor, the mean and mean +/- sd of a numeric moderator)
#
# Opt-in per case (Tier F); a case without these keys is generated exactly as before:
#
#   data2           a list of names: each a matrix read from data2_<name>.csv next to data.csv (written by tests/oracle/datasets.py:
#                   a header of column labels after an empty first field, then one line per row, its label first) and handed to
#                   brm() and default_prior() as data2 = list(<name> = <matrix>), for car(W), sar(M) and fcor(M); the meta records
#                   each file's sha256 (data2_sha256). A matrix response needs no key: brms evaluates cbind(m1, m2, m3) in the formula
#                   into the same matrix column (the same Stan program and standata as a column bound beforehand)
#   prediction_refused   a list of prediction kinds ("linpred", "epred") that brms fails to compute for the model (posterior_epred
#                   of sar(type = "error")): each is called once per block and must fail, and brms's error message is recorded as
#                   predictions$<block>$<kind>_refused (and <kind>_nogroup_refused) in place of the array; it stops if brms computes it
#
# Opt-in per case (Tier F, part 2); a case without these keys is generated exactly as before:
#
#   prediction_resp a list of response names: each block also records posterior_linpred / posterior_epred(resp = <name>) (and the
#                   _nogroup kinds) as predictions$<block>$resp$<name>$<kind>, for a multivariate model whose joint predictions
#                   brms refuses (subset(): list linpred and epred in prediction_refused); brms predicts each response on the
#                   rows of its subset, of the fitted data or of newdata
#   summary_draws   {chains, iter, seed, rho, sd, chain_sd}: a second fixed-parameter fit, for summaries, whose draws are seeded
#                   unconstrained points arranged as `chains` chains of `iter` iterations, each chain a stationary AR(1) series
#                   (coefficient rho, marginal sd `sd` in every coordinate) around its own centre (normal, sd chain_sd), so that
#                   brms's Rhat and effective sample sizes are defined and not trivial. Stan's Fixed_param run keeps each chain at
#                   its first draw, so the run's stored draws are then replaced by the seeded ones (rstan's sim$samples, every
#                   parameter brm() saves: brms's exclude_pars() of the model are left out, as brm() leaves them out), lp__ by
#                   rstan's log_prob() with the Jacobian (what a sampler records), and read back through rstan and brms to check
#                   they are exactly the seeded ones; the fit is labelled algorithm "fixed_param" (brm()'s name for such a run).
#                   Recorded as post$summary_draws: the chains and iterations, each draw as a seeded point is (u, the constrained
#                   pars, lp and lp_nojac; chain by chain) and brms's draws array (iterations, chains, variables)
#   summaries       a list of brms functions evaluated on that fit with their defaults, each recorded under its name in the post
#                   block: "fixef", "ranef" and "coef" (by group), "VarCorr" (by group: sd, cor and cov where brms gives them),
#                   "posterior_summary", "summary" (its tables fixed, spec_pars, cor_pars, random by group, and rescor_pars,
#                   splines, mo, gp where present, with nobs, ngrps, chains, iter, warmup, thin and total_ndraws) and "residuals"
#                   (type "ordinary", method "posterior_epred": brms's default method draws from posterior_predict). Each table
#                   or array keeps its labels on every axis (dimnames; [] where brms gives none). The fit is built twice and
#                   every summary must be identical between the two, and under two seeds; the meta records the posterior
#                   package's version (brms's rhat, ess_bulk and ess_tail are posterior's)
#
# Opt-in per case (Tier G, part 1); a case without these keys is generated exactly as before:
#
#   logical         a list of columns written True / False (pandas's bool): each is read as R's logical (TRUE / FALSE; any other
#                   field stops the generation), a logical predictor that brms codes as lgTRUE
#   pandas_categorical   a list of factor columns (each must be in `factors`) that the AutoBayes side receives as a pandas Categorical
#                   whose categories are the declared levels in their order (tests/oracle/fixture.py); brms reads them as every factor
#   newdata_prediction_refused   a list of prediction kinds ("linpred", "epred") that brms fails to compute on newdata only (a factor
#                   made in the formula, factor(o), on new rows lacking one of its levels): on newdata each is called once with and
#                   once without the group terms and must fail, and brms's error message is recorded as predictions$newdata$<kind>_refused
#                   (and <kind>_nogroup_refused) in place of the array; the fitted rows are recorded as usual
#
# Tables (Tier G): a numeric matrix column of a recorded table keeps its dims (brms's conditional_effects() of a model with
# poly(w, 2) carries the model-frame column 'poly(w, 2)', rows by 2); a vector column is encoded as before.
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
logical_columns <- unlist(case$logical)               # opt-in (Tier G): columns written True / False, read as R's logical
if (length(intersect(logical_columns, c(names(case$factors), na_columns)))) stop("a logical column is also a factor or a missing-value column")
pandas_categorical <- unlist(case$pandas_categorical) # opt-in (Tier G): factors the AutoBayes side receives as pandas Categoricals
if (length(setdiff(pandas_categorical, names(case$factors)))) stop("pandas_categorical lists columns that are not declared factors")
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
    } else if (col %in% logical_columns) {            # Tier G: pandas writes a bool as True / False
      v <- raw[[col]]
      if (!all(v %in% c("True", "False"))) stop(sprintf("column %s has values other than True and False", col))
      out[[col]] <- v == "True"
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
data2_names <- unlist(case$data2)                     # opt-in (Tier F): matrices brms receives as data2
read_matrix <- function(path) {                       # labels as text, every entry a number
  raw <- read.csv(path, check.names = FALSE, colClasses = "character", row.names = 1)
  m <- matrix(suppressWarnings(as.numeric(as.matrix(raw))), nrow(raw), ncol(raw), dimnames = list(rownames(raw), colnames(raw)))
  if (anyNA(m)) stop(sprintf("%s has entries that are not numbers", basename(path)))
  m
}
data2 <- if (length(data2_names)) lapply(setNames(data2_names, data2_names), function(nm) read_matrix(file.path(dir, sprintf("data2_%s.csv", nm)))) else NULL
if (!is.null(data2) && !is.null(case$loo_compare)) stop("loo_compare of a case with data2 is not supported")

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
                            backend = "rstan", silent = 2, refresh = 0), if (!is.null(data2)) list(data2 = data2),
                       case$brm_args %||% list()))
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
prior_gq <- isTRUE(case$prior_draws)                  # opt-in (Tier E): a fit with sample_prior = "yes"
if (prior_gq && !identical(case$brm_args$sample_prior, "yes")) stop("prior_draws needs brm_args sample_prior = \"yes\"")
set.seed(case$seed)
n_points <- case$n_points %||% 8L
halfwidth <- case$point_halfwidth %||% 1.2            # opt-in: the box (-h, h) the points are drawn from
pts <- lapply(seq_len(n_points), function(k) runif(n, -halfwidth, halfwidth))
points <- lapply(pts, function(u) {
  cp <- constrain_pars(sf, u)
  if (prior_gq) {                                     # Tier E: the prior_* generated quantities are drawn anew on every call
    keep <- !startsWith(names(cp), "prior_")
    if (!identical(cp[keep], constrain_pars(sf, u)[keep])) stop("constrain_pars() changes between calls beyond the prior draws")
    cp <- cp[keep]
  }
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
refused <- unlist(case$prediction_refused)            # opt-in (Tier F): prediction kinds brms must fail to compute
if (length(setdiff(refused, c("linpred", "epred")))) stop("prediction_refused lists kinds other than linpred and epred")
pred_block <- function(newdata) {
  out <- list(linpred = enc(deterministic(function() posterior_linpred(fit1, newdata = newdata))),
              epred = enc(deterministic(function() posterior_epred(fit1, newdata = newdata))))
  if (has_groups) {
    out$linpred_nogroup <- enc(deterministic(function() posterior_linpred(fit1, newdata = newdata, re_formula = NA)))
    out$epred_nogroup <- enc(deterministic(function() posterior_epred(fit1, newdata = newdata, re_formula = NA)))
  }
  out
}
refusal <- function(f) {                              # brms's error message, or a stop when brms computes it after all
  msg <- tryCatch({ f(); NULL }, error = function(e) conditionMessage(e))
  if (is.null(msg)) stop("brms computed a prediction the case lists in prediction_refused")
  msg
}
if (length(refused)) pred_block <- function(newdata) {   # the kinds listed are recorded as <kind>_refused, brms's message
  fn <- list(linpred = posterior_linpred, epred = posterior_epred)
  run <- function(kind, ...) {
    f <- function() fn[[kind]](fit1, newdata = newdata, ...)
    if (kind %in% refused) refusal(f) else enc(deterministic(f))
  }
  out <- list()
  for (kind in c("linpred", "epred")) out[[if (kind %in% refused) paste0(kind, "_refused") else kind]] <- run(kind)
  if (has_groups) for (kind in c("linpred", "epred"))
    out[[paste0(kind, if (kind %in% refused) "_nogroup_refused" else "_nogroup")]] <- run(kind, re_formula = NA)
  out
}
pred_resp <- unlist(case$prediction_resp)             # opt-in (Tier F, part 2): each response's own predictions (subset())
if (length(pred_resp)) {
  if (is.null(case$mvbf)) stop("prediction_resp needs a multivariate model (mvbf)")
  joint_block <- pred_block
  pred_block <- function(newdata) {                   # predictions$<block>$resp$<name>$<kind>, on the rows of that response
    out <- joint_block(newdata)
    out$resp <- lapply(setNames(pred_resp, pred_resp), function(r) {
      one <- function(f, ...) enc(deterministic(function() f(fit1, newdata = newdata, resp = r, ...)))
      o <- list(linpred = one(posterior_linpred), epred = one(posterior_epred))
      if (has_groups) {
        o$linpred_nogroup <- one(posterior_linpred, re_formula = NA)
        o$epred_nogroup <- one(posterior_epred, re_formula = NA)
      }
      o
    })
    out
  }
}
nd_refused <- unlist(case$newdata_prediction_refused) # opt-in (Tier G): prediction kinds brms fails to compute on newdata only
if (length(nd_refused)) {
  if (length(setdiff(nd_refused, c("linpred", "epred")))) stop("newdata_prediction_refused lists kinds other than linpred and epred")
  if (is.null(nd)) stop("newdata_prediction_refused needs newdata")
  if (length(refused) || length(pred_resp)) stop("newdata_prediction_refused does not combine with prediction_refused or prediction_resp")
}
newdata_refusal_block <- function(newdata) {          # the kinds listed recorded as <kind>_refused, brms's message
  fn <- list(linpred = posterior_linpred, epred = posterior_epred)
  run <- function(kind, ...) {
    f <- function() fn[[kind]](fit1, newdata = newdata, ...)
    if (!(kind %in% nd_refused)) return(enc(deterministic(f)))
    msg <- tryCatch({ f(); NULL }, error = function(e) conditionMessage(e))
    if (is.null(msg)) stop("brms computed a newdata prediction the case lists in newdata_prediction_refused")
    msg
  }
  out <- list()
  for (kind in c("linpred", "epred")) out[[if (kind %in% nd_refused) paste0(kind, "_refused") else kind]] <- run(kind)
  if (has_groups) for (kind in c("linpred", "epred"))
    out[[paste0(kind, if (kind %in% nd_refused) "_nogroup_refused" else "_nogroup")]] <- run(kind, re_formula = NA)
  out
}
predictions <- list(fitted = pred_block(NULL))
if (!is.null(nd)) predictions$newdata <- if (length(nd_refused)) newdata_refusal_block(nd) else pred_block(nd)

# -- opt-in post-processing of the fixed-parameter fit (C07): pointwise log-likelihood, Bayes R2, PSIS-LOO --------
# brms's loo() on this fit: the relative efficiencies of one-draw chains are NA, which loo() ignores (r_eff = 1),
# and a Pareto tail too short to fit (fewer than five draws in it) leaves raw importance weights and k-hat Inf.
post <- list()
if (prior_gq) {                                       # Tier E: the prior draws of the fixed-parameter run, made twice
  sf3 <- sampling(sf@stanmodel, data = sdat, algorithm = "Fixed_param", chains = n_points, iter = 1, warmup = 0,
                  init = inits, refresh = 0, seed = 1, cores = 1)
  fit1b <- fit0
  fit1b$fit <- sf3
  fit1b <- rename_pars(fit1b)
  pd <- prior_draws(fit1)
  if (!identical(pd, prior_draws(fit1b))) stop("two fixed-parameter runs with the same seed give different prior draws")
  if (!nrow(pd) || nrow(pd) != n_points) stop("prior_draws(): expected one row per point")
  post$prior_draws <- enc(as.matrix(pd))              # (draws, variables), brms's names (b for a prior shared by class b)
}
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
    # Tier G: a matrix column (conditional_effects() holds a transform's model-frame column, poly(w, 2), at its means) keeps its dims
    if (is.numeric(v)) enc(v, dims = if (is.matrix(v)) dim(v) else length(v), allow_infinite = TRUE, allow_na = TRUE) else I(as.character(v))
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
  if ("loo_model_weights" %in% extras) {              # Tier E: stacking weights of the two fits
    sw <- deterministic(function() {
      a <- add_criterion(fit1, "loo", model_name = ids[1])
      b <- add_criterion(ofit1, "loo", model_name = ids[2])
      list(lmw = loo_model_weights(a, b, model_names = ids), mw = model_weights(a, b, weights = "stacking", model_names = ids))
    })
    nm <- if (!is.null(names(sw$lmw))) names(sw$lmw) else rownames(sw$lmw)
    if (!identical(as.character(nm), ids)) stop("loo_model_weights(): the weights are not named by the two models in order")
    if (max(abs(as.numeric(sw$lmw) - as.numeric(sw$mw))) > 0) stop("loo_model_weights() and model_weights(weights = \"stacking\") differ")
    post$loo_model_weights <- list(models = I(ids), method = "stacking", weights = enc(as.numeric(sw$lmw), dims = 2L))
  }
}
if ("loo_model_weights" %in% extras && is.null(case$loo_compare)) stop("extras loo_model_weights needs loo_compare")
if (length(case$hypothesis)) {                        # D06: brms's defaults (class "b", alpha 0.05, robust = FALSE)
  hy <- deterministic(function() { h <- hypothesis(fit1, unlist(case$hypothesis), seed = case$hypothesis_seed); h[c("hypothesis", "samples")] })
  post$hypothesis <- list(table = enc_frame(hy$hypothesis), samples = enc(as.matrix(hy$samples)))   # samples: (draws, hypotheses)
}
if (length(case$dpar_epred)) {                        # Tier E: a distributional parameter on the fitted rows, (draws, rows)
  dps <- unlist(case$dpar_epred)
  post$dpar_epred <- lapply(setNames(dps, dps), function(dp) enc(deterministic(function() posterior_epred(fit1, dpar = dp))))
}
if (length(case$conditional_effects)) {               # D06: one table per effect, in the order given
  ce <- deterministic(function() conditional_effects(fit1, effects = unlist(case$conditional_effects)))
  post$conditional_effects <- lapply(ce, enc_frame)
}

# -- opt-in summaries (Tier F, part 2: F07) of a fixed-parameter fit whose draws are seeded chains --------------------------
summaries <- unlist(case$summaries)
summary_kinds <- c("fixef", "ranef", "coef", "VarCorr", "posterior_summary", "summary", "residuals")
if (length(setdiff(summaries, summary_kinds))) stop("summaries lists functions other than ", paste(summary_kinds, collapse = ", "))
if (length(summaries) && is.null(case$summary_draws)) stop("summaries needs summary_draws")
if (!is.null(case$summary_draws)) {
  if (!length(summaries)) stop("summary_draws needs summaries")
  if (prior_gq) stop("summaries of a fit with prior draws are not supported (constrain_pars() draws them anew)")
  spec <- case$summary_draws
  n_chains <- as.integer(spec$chains); n_iter <- as.integer(spec$iter)
  if (n_chains < 2 || n_iter < 4) stop("summary_draws: at least 2 chains of 4 iterations (Rhat splits each chain in two)")
  set.seed(spec$seed)                                 # chain by chain: its centre, then its AR(1) series around it
  U <- array(NA_real_, c(n_iter, n_chains, n))
  for (ch in seq_len(n_chains)) {
    centre <- rnorm(n, 0, spec$chain_sd)
    u <- centre + rnorm(n, 0, spec$sd)
    for (it in seq_len(n_iter)) {
      if (it > 1) u <- centre + spec$rho * (u - centre) + sqrt(1 - spec$rho^2) * rnorm(n, 0, spec$sd)
      U[it, ch, ] <- u
    }
  }
  cps <- lapply(seq_len(n_chains), function(ch) lapply(seq_len(n_iter), function(t) constrain_pars(sf, U[t, ch, ])))
  lps <- lapply(c(TRUE, FALSE), function(adj) apply(U, c(1, 2), function(u) as.numeric(log_prob(sf, u, adjust_transform = adj))))
  if (any(!is.finite(unlist(lps)))) stop("non-finite log density at a summary draw")
  excluded <- unlist(brms:::exclude_pars(fit0))       # the parameters brm() does not save (z_1, L_1, ...)
  summary_fit <- function() {                         # the fixed-parameter run, its draws replaced by the seeded ones
    sfs <- sampling(sf@stanmodel, data = sdat, algorithm = "Fixed_param", chains = n_chains, iter = n_iter, warmup = 0,
                    init = lapply(cps, `[[`, 1), refresh = 0, seed = 1, cores = 1, pars = excluded, include = FALSE)
    kept <- setdiff(names(sfs@sim$dims_oi), "lp__")
    want <- array(NA_real_, c(n_iter, n_chains, length(sfs@sim$fnames_oi)))
    for (ch in seq_len(n_chains)) {
      smp <- sfs@sim$samples[[ch]]
      for (t in seq_len(n_iter)) {
        v <- c(unlist(lapply(kept, function(nm) as.vector(cps[[ch]][[t]][[nm]]))), lps[[1]][t, ch])
        if (length(v) != length(smp)) stop("summary_draws: the constrained draw does not fill the run's parameters")
        want[t, ch, ] <- v
      }
      first <- vapply(smp, function(s) s[1], 0.0)     # the run's own first draw is the chain's first point: the order of the
      P <- length(smp)                                # flattened parameters is the run's, checked, not assumed (lp__ aside)
      if (any(abs(first[-P] - want[1, ch, -P]) > 1e-10 * pmax(1, abs(want[1, ch, -P]))))
        stop("summary_draws: the run's first draw is not the chain's first point (parameter order)")
      for (k in seq_along(smp)) smp[[k]] <- want[, ch, k]
      sfs@sim$samples[[ch]] <- smp
    }
    got <- as.array(sfs)                              # rstan's own read-back: (iterations, chains, parameters)
    if (!identical(dimnames(got)[[3]], sfs@sim$fnames_oi) || !identical(as.vector(got), as.vector(want)))
      stop("summary_draws: rstan does not return the draws written")
    f <- fit0
    f$fit <- sfs
    f <- rename_pars(f)
    f$algorithm <- "fixed_param"                      # brm()'s name for a Fixed_param run (summary() then reads no NUTS diagnostics)
    da <- unclass(posterior::as_draws_array(f))
    for (nm in intersect(c("lprior", "lp__"), dimnames(da)[[3]])) {   # the same draws through brms, checked
      if (!identical(as.vector(da[, , nm]), as.vector(want[, , match(nm, sfs@sim$fnames_oi)])))
        stop(sprintf("summary_draws: brms's draws of %s are not the draws written", nm))
    }
    if (dim(da)[3] != length(sfs@sim$fnames_oi)) stop("summary_draws: brms's draws have other variables than the run")
    f
  }
  enc_labelled <- function(x) {                  # an array or table with its labels on every axis ([] where there are none)
    if (is.data.frame(x)) x <- as.matrix(x)
    out <- enc(unclass(x), dims = dim(x), allow_infinite = TRUE, allow_na = TRUE)
    out$colnames <- NULL
    out$dimnames <- lapply(seq_along(dim(x)), function(k) I(as.character(dimnames(x)[[k]])))
    out
  }
  record_summaries <- function(f) {
    out <- list()
    if ("fixef" %in% summaries) out$fixef <- enc_labelled(fixef(f))
    if ("ranef" %in% summaries) out$ranef <- lapply(ranef(f), enc_labelled)
    if ("coef" %in% summaries) out$coef <- lapply(coef(f), enc_labelled)
    if ("VarCorr" %in% summaries) out$VarCorr <- lapply(VarCorr(f), function(g) lapply(g, enc_labelled))
    if ("posterior_summary" %in% summaries) out$posterior_summary <- enc_labelled(posterior_summary(f))
    if ("summary" %in% summaries) {
      s <- summary(f)
      tables <- intersect(c("fixed", "spec_pars", "cor_pars", "rescor_pars", "splines", "mo", "gp"), names(s))
      out$summary <- c(lapply(setNames(tables, tables), function(nm) enc_labelled(s[[nm]])),
                       list(random = lapply(s$random, enc_labelled), nobs = s$nobs, ngrps = lapply(s$ngrps, as.integer),
                            chains = s$chains, iter = s$iter, warmup = s$warmup, thin = s$thin, total_ndraws = s$total_ndraws))
    }
    if ("residuals" %in% summaries) out$residuals <- enc_labelled(residuals(f, method = "posterior_epred"))
    out
  }
  fa <- summary_fit()
  sums <- deterministic(function() record_summaries(fa))
  if (!identical(sums, record_summaries(summary_fit()))) stop("summaries: two fits of the same draws give different summaries")
  da <- posterior::as_draws_array(fa)
  post$summary_draws <- list(
    chains = n_chains, iter = n_iter, draws = enc_labelled(da),
    points = unlist(lapply(seq_len(n_chains), function(ch) lapply(seq_len(n_iter), function(t) {
      cp <- cps[[ch]][[t]]
      list(u = enc(U[t, ch, ]), pars = lapply(setNames(names(cp), names(cp)), function(nm) enc(cp[[nm]], sf@par_dims[[nm]])),
           lp = lps[[1]][t, ch], lp_nojac = lps[[2]][t, ch])
    })), recursive = FALSE))
  post <- c(post, sums)
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
  default_prior = rows(if (is.null(data2)) default_prior(fml, data = d, family = fam) else default_prior(fml, data = d, family = fam, data2 = data2))
)
if (!is.null(data2))                                  # Tier F: the data2 files, as data.csv's
  fixture$meta$data2_sha256 <- lapply(setNames(data2_names, data2_names), function(nm) sha(file.path(dir, sprintf("data2_%s.csv", nm))))
if (length(summaries))                                # Tier F, part 2: brms's rhat, ess_bulk and ess_tail are the posterior package's
  fixture$meta$posterior <- as.character(packageVersion("posterior"))
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
