#!/usr/bin/env Rscript
# U16 (D61, D70): a non-linear formula is passed into the Stan program as written, without a check that Stan has its
# functions: plogis() (which brms's own R-side predictions evaluate), and gp(), mo() and ar() written in the expression
# (brms's terms, read there as calls of functions Stan does not have) produce a program that Stan's compiler rejects, so
# the user gets a Stan semantic error instead of a message from brms (or a translation, plogis to inv_logit).
#
#   Rscript docs/brms-parity/upstream/repro/U16.R      # exit status 1 while the problem is present
#
# Stock CRAN brms and rstan only: make_stancode() writes the program and rstan::stanc() runs Stan's parser and semantic
# check on it (nothing is compiled to C++ or sampled). The generated program is GPL-3 and is not printed; only the
# compiler's one-line diagnosis is. Control: inv_logit(), which Stan has, passes.
t_start <- proc.time()[["elapsed"]]
suppressPackageStartupMessages({ library(brms); library(rstan) })
cat(sprintf("brms %s, rstan %s, StanHeaders %s, R %s\n", packageVersion("brms"), packageVersion("rstan"),
            packageVersion("StanHeaders"), getRversion()))
d <- data.frame(y = c(0.2, 0.5, 0.9, 0.4), x = c(-1, 0, 1, 2), z = c(1L, 2L, 3L, 2L), t = 1:4)
pri <- set_prior("normal(0, 5)", nlpar = "a")
cases <- list(
  list(label = "y ~ inv_logit(a * x)  (control)", f = bf(y ~ inv_logit(a * x), a ~ 1, nl = TRUE), control = TRUE),
  list(label = "y ~ plogis(a * x)", f = bf(y ~ plogis(a * x), a ~ 1, nl = TRUE), control = FALSE),
  list(label = "y ~ a * gp(x)", f = bf(y ~ a * gp(x), a ~ 1, nl = TRUE), control = FALSE),
  list(label = "y ~ a * mo(z)", f = bf(y ~ a * mo(z), a ~ 1, nl = TRUE), control = FALSE),
  list(label = "y ~ a + ar(t)", f = bf(y ~ a + ar(t), a ~ 1, nl = TRUE), control = FALSE))
diagnosis <- function(msg) {                         # the compiler's message without the program's lines
  lines <- trimws(strsplit(msg, "\n")[[1]])
  keep <- lines[grepl("identifier|function|not|expected|declared", lines, ignore.case = TRUE) &
                  !grepl("^[0-9]+:|^-+$|^\\^", lines) & nchar(lines) < 200]
  if (length(keep)) paste(keep, collapse = " | ") else "(no one-line diagnosis)"
}
present <- FALSE; control_ok <- TRUE
for (cs in cases) {
  code <- tryCatch(make_stancode(cs$f, data = d, prior = pri), error = function(e) e)
  if (inherits(code, "error")) {
    cat(sprintf("\n%s\n  brms refuses it: %s\n", cs$label, conditionMessage(code)))
    if (cs$control) control_ok <- FALSE
    next
  }
  chk <- tryCatch({ stanc(model_code = code, verbose = FALSE); "accepted" }, error = function(e) diagnosis(conditionMessage(e)))
  cat(sprintf("\n%s\n  brms writes a program; Stan's compiler: %s\n", cs$label, chk))
  if (cs$control) control_ok <- identical(chk, "accepted") else if (!identical(chk, "accepted")) present <- TRUE
}
if (!control_ok) { cat("\nthe control formula did not pass: the script is wrong\n"); quit(status = 2) }
cat(if (present) "\nPROBLEM PRESENT: brms passes functions Stan does not have into the program instead of refusing them\n" else "\nproblem absent\n")
cat(sprintf("run time %.1f s\n", proc.time()[["elapsed"]] - t_start))
quit(status = if (present) 1 else 0)
