#!/usr/bin/env Rscript
# brms's reading of every formula of the syntax corpus.
#
#   Rscript tests/brms_syntax/brms_reading.R            (from the repository root)
#
# Reads tests/brms_syntax/corpus.json and data.csv and, for each entry, builds its brms formula and family from the
# entry's structured description (formula strings through as.formula(), brms's helpers bf(), lf(), nlf(), mvbf(),
# set_rescor(), set_nl(), set_mecor() and brmsfamily(), mixture(), custom_family() called with do.call(); the formulas
# were written for this corpus from brms 2.23's help pages, and no brms source is read or copied). It calls
# make_standata() and brm(..., empty = TRUE), which builds the brmsfit without compiling or sampling, for brms's table
# of group-level terms, and writes tests/brms_syntax/standata.json: per entry its status ("ok" or "error"), brms's error
# and warnings, the Stan data's names and dimensions, its arrays, the design matrices' column names and the group-level
# term table with the level labels. Nothing here samples, compiles Stan or evaluates text as code.
suppressPackageStartupMessages({
  library(brms)
  library(jsonlite)
})
invisible(Sys.setlocale("LC_COLLATE", "C"))          # factor levels sort as Python's sorted() does
dir <- "tests/brms_syntax"
corpus <- fromJSON(file.path(dir, "corpus.json"), simplifyVector = FALSE)
spec <- corpus$data
`%||%` <- function(a, b) if (is.null(a)) b else a

# -- data ------------------------------------------------------------------------------------------------
raw <- read.csv(file.path(dir, spec$csv), stringsAsFactors = FALSE, check.names = FALSE, colClasses = "character")
d <- raw
for (col in names(raw)) {
  v <- raw[[col]]
  if (!is.null(spec$factors[[col]])) {
    d[[col]] <- factor(v, levels = unlist(spec$factors[[col]]))
  } else if (!is.null(spec$category_dtype[[col]])) {
    d[[col]] <- factor(v, levels = unlist(spec$category_dtype[[col]]))
  } else if (!is.null(spec$ordered[[col]])) {
    d[[col]] <- factor(v, levels = unlist(spec$ordered[[col]]), ordered = TRUE)
  } else if (col %in% unlist(spec$logical)) {
    d[[col]] <- as.logical(v)
  } else if (col %in% unlist(spec$text)) {
    d[[col]] <- v
  } else {
    num <- suppressWarnings(as.numeric(ifelse(v == "", NA_character_, v)))
    if (any(is.na(num) & v != "")) stop(sprintf("column %s has values that are neither numbers nor empty", col))
    if (any(is.na(num)) && !(col %in% unlist(spec$na_columns))) stop(sprintf("column %s has unexpected missing values", col))
    d[[col]] <- if (col %in% unlist(spec$integer)) as.integer(num) else num
  }
}
d$ymn <- with(d, cbind(ymn1, ymn2, ymn3))            # multinomial(): a matrix response of counts
d$ydir <- with(d, cbind(p1, p2, p3))                 # dirichlet(): a matrix response of proportions

# data2 objects the entries name: a spatial adjacency over g's levels (car), a row-standardised weight matrix over the
# rows (sar), a fixed residual covariance over the rows (fcor) and a covariance over g's levels (gr(cov = ))
lev <- levels(d$g)
M_car <- matrix(0, 5, 5, dimnames = list(lev, lev))
for (i in 1:4) { M_car[i, i + 1] <- 1; M_car[i + 1, i] <- 1 }
M_sar <- matrix(0, nrow(d), nrow(d))
for (i in seq_len(nrow(d) - 1)) { M_sar[i, i + 1] <- 1; M_sar[i + 1, i] <- 1 }
M_sar <- M_sar / rowSums(M_sar)
V_fcor <- diag(0.5, nrow(d)) + 0.1
A_cov <- 0.5^abs(outer(1:5, 1:5, "-")); dimnames(A_cov) <- list(lev, lev)
data2_objects <- list(M_car = M_car, M_sar = M_sar, V_fcor = V_fcor, A_cov = A_cov)

# -- the brms call of an entry ------------------------------------------------------------------------------
arg <- function(a) {                                  # a formula string becomes a formula; anything else is a value
  if (is.character(a) && length(a) == 1 && grepl("~", a, fixed = TRUE)) as.formula(a) else if (is.list(a)) unlist(a) else a
}
helper <- function(h) {                               # {"fn": "lf", "args": [...], "kwargs": {...}}
  do.call(getExportedValue("brms", h$fn), c(lapply(h$args %||% list(), arg), lapply(h$kwargs %||% list(), arg)))
}
formula_of <- function(b) {
  if (!is.null(b$mvbf)) return(helper(list(fn = "mvbf", args = b$mvbf$args, kwargs = b$mvbf$kwargs)))
  parts <- lapply(as.list(unlist(b$formula)), as.formula)
  fml <- if (length(parts) == 1 && is.null(b$bf_args) && is.null(b$plus)) parts[[1]] else
    do.call(brms::bf, c(unname(parts), lapply(b$bf_args %||% list(), arg)))
  for (h in b$plus %||% list()) fml <- fml + helper(h)
  fml
}
family_of <- function(f) {
  f <- f %||% list(family = "gaussian")
  if (!is.null(f$mixture)) return(do.call(brms::mixture, c(lapply(f$mixture, family_of), f[setdiff(names(f), "mixture")])))
  if (!is.null(f$custom)) return(do.call(brms::custom_family, lapply(f$custom, arg)))
  if (!is.null(f$stats)) return(do.call(getExportedValue("stats", f$stats), f[setdiff(names(f), "stats")]))   # stats::Gamma()
  do.call(brms::brmsfamily, f)
}

# -- encoders ---------------------------------------------------------------------------------------------
enc <- function(x) {
  dims <- if (is.null(dim(x))) (if (length(x) == 1) integer(0) else length(x)) else dim(x)
  v <- suppressWarnings(as.numeric(as.vector(x)))
  v <- ifelse(is.na(v), "NA", ifelse(is.finite(v), sprintf("%.17g", v), ifelse(v > 0, "Inf", "-Inf")))
  out <- list(dims = I(as.integer(dims)), values = I(v))
  cn <- if (length(dim(x)) == 2) colnames(x) else NULL
  if (!is.null(cn)) out$colnames <- I(cn)
  out
}
MAX_ARRAY <- 2000                                     # larger arrays are recorded by their dimensions only

reading <- function(entry) {
  b <- entry$brms
  warns <- character(0)
  out <- list(status = "ok")
  res <- tryCatch(withCallingHandlers({
    fml <- formula_of(b)
    fam <- family_of(b$family)
    data2 <- if (length(b$data2)) data2_objects[unlist(b$data2)] else NULL
    sd <- make_standata(formula = fml, data = d, family = fam, data2 = data2)
    fit <- brm(formula = fml, data = d, family = fam, data2 = data2, empty = TRUE, backend = "rstan", silent = 2)
    list(sd = sd, fit = fit)
  }, warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") }),
  error = function(e) list(error = conditionMessage(e)))
  out$warnings <- I(unique(warns))
  if (!is.null(res$error)) {
    out$status <- "error"
    out$error <- res$error
    return(out)
  }
  sd <- unclass(res$sd)
  sd <- sd[!vapply(sd, is.null, logical(1))]
  out$names <- lapply(sd, function(x) I(as.integer(if (is.null(dim(x))) length(x) else dim(x))))
  keep <- vapply(sd, function(x) is.numeric(x) || is.logical(x), logical(1)) & vapply(sd, length, integer(1)) <= MAX_ARRAY
  out$arrays <- lapply(sd[keep], enc)
  re <- res$fit$ranef
  out$groups <- list()
  if (!is.null(re) && nrow(re) > 0) {
    lv <- attr(re, "levels")
    col <- function(name, i) if (name %in% names(re)) as.character(re[[name]][i]) else ""
    out$groups <- lapply(seq_len(nrow(re)), function(i) list(
      id = as.integer(re$id[i]), group = as.character(re$group[i]), coef = as.character(re$coef[i]),
      cn = as.integer(re$cn[i]), cor = isTRUE(re$cor[i]), resp = col("resp", i), dpar = col("dpar", i),
      nlpar = col("nlpar", i), type = col("type", i), levels = I(as.character(lv[[re$group[i]]]))))
  }
  out$responses <- I(as.character(res$fit$formula$responses %||% character(0)))
  family_record <- function(fam) {                    # the family and its links (link, link_sigma, ...) as brms resolved them
    rec <- list(family = as.character(fam$family),
                links = lapply(Filter(is.character, fam[grep("^link(_|$)", names(fam))]), as.character))
    if (!is.null(fam$mix)) rec$mix <- I(vapply(fam$mix, function(m) as.character(m$family), character(1)))
    rec
  }
  out$family <- if (inherits(res$fit$formula, "mvbrmsformula")) {
    list(resp = lapply(res$fit$formula$forms, function(f) family_record(f$family)))
  } else family_record(res$fit$family)
  out
}

only <- commandArgs(trailingOnly = TRUE)              # optional: the ids to read (default every entry)
result <- list()
for (entry in corpus$entries) {
  if (length(only) && !(entry$id %in% only)) next
  t0 <- Sys.time()
  result[[entry$id]] <- reading(entry)
  cat(sprintf("%-45s %-5s %5.1fs %s\n", entry$id, result[[entry$id]]$status, as.numeric(Sys.time() - t0, units = "secs"),
              substr(gsub("\n", " ", result[[entry$id]]$error %||% ""), 1, 100)))
}
meta <- list(brms = as.character(packageVersion("brms")), R = paste(R.version$major, R.version$minor, sep = "."),
             generator = "tests/brms_syntax/brms_reading.R")
out_path <- file.path(dir, if (length(only)) "standata.partial.json" else "standata.json")
writeLines(toJSON(list(meta = meta, entries = result), auto_unbox = TRUE, null = "null", digits = NA, pretty = FALSE), out_path)
