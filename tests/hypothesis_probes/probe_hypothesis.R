#!/usr/bin/env Rscript
# brms 2.23's hypothesis() on recorded draws (D06): its default method, which takes a data frame of draws, one column per
# parameter, as hypothesis() of a fit takes the draws of the parameters of class b under their names without the b_ prefix.
#
#   Rscript tests/hypothesis_probes/probe_hypothesis.R tests/hypothesis_probes/brms-2.23.0-hypothesis.json
#
# Recorded: the draws (seeded here and written with 17 significant digits), and for each list of hypotheses, alpha and
# robust, brms's table (Hypothesis, Estimate, Est.Error, CI.Lower, CI.Upper, Evid.Ratio, Post.Prob, Star) and, for the first
# setting of each set of draws (they do not depend on alpha or robust), its samples;
# for a hypothesis brms refuses, its error message. brms's outputs only, not its code (brms is GPL-3); nothing is sampled
# and no Stan program runs. The Python test (tests/test_hypothesis_effects.py) recomputes every output from the draws.
suppressPackageStartupMessages({ library(brms); library(jsonlite) })
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("usage: Rscript tests/hypothesis_probes/probe_hypothesis.R <output json>")
enc <- function(x) {
  v <- as.numeric(as.vector(x))
  list(dims = I(as.integer(if (is.null(dim(x))) length(x) else dim(x))),
       values = I(ifelse(is.na(v), ifelse(is.nan(v), "NaN", "NA"), ifelse(is.finite(v), sprintf("%.17g", v), ifelse(v > 0, "Inf", "-Inf")))))
}
frame <- function(d) {                                   # the draws as written, read back so that brms sees exactly those
  d <- as.data.frame(lapply(d, function(v) as.numeric(sprintf("%.17g", v))), check.names = FALSE)
  d
}
run <- function(d, hyps, alpha = 0.05, robust = FALSE, samples = FALSE) {
  r <- hypothesis(d, hyps, alpha = alpha, robust = robust)
  h <- r$hypothesis
  out <- list(hypotheses = I(hyps), alpha = alpha, robust = robust, labels = I(as.character(h$Hypothesis)),
              table = lapply(list(Estimate = h$Estimate, Est.Error = h$Est.Error, CI.Lower = h$CI.Lower, CI.Upper = h$CI.Upper,
                                  Evid.Ratio = h$Evid.Ratio, Post.Prob = h$Post.Prob), enc),
              star = I(as.character(h$Star)), columns = I(colnames(h)))
  if (samples) out <- c(out, list(samples = enc(as.matrix(r$samples)), sample_names = I(colnames(r$samples))))
  out
}
refused <- function(d, h) {
  msg <- tryCatch({ suppressWarnings(hypothesis(d, h)); NA_character_ }, error = function(e) conditionMessage(e))
  list(hypothesis = h, error = if (is.na(msg)) NULL else msg)
}
set.seed(20261015)
S <- 40
main <- frame(data.frame(Intercept = rnorm(S, 0.3), x = rnorm(S, 0.2), w = rnorm(S, -0.1, 0.8), zq = rnorm(S, 0.5),
                         zr = rnorm(S, -0.2, 1.3), `x:zq` = rnorm(S, 0, 0.5), `zq:hh2` = rnorm(S, 0.1, 0.4), IxE2 = rnorm(S, 1),
                         check.names = FALSE))
hyps <- c("x > 0", "x + w = 0", "zr < zq", "Intercept < 0", "zq - zr = 0", "x = 1", "x > -1", "2 * x = w", "x - w > 0.5",
          "x + w = zq + zr", "exp(x) - 3 = 0", "x^2 < 1", "-x > 0", "x / w = 1", "(x + w) / 2 > 0", "x:zq > 0",
          "x:zq - zq:hh2 = 0", "x*w = 0", "0 < x", ".5 > x", "x > 0.0", "x = +0", "x - 0 = 0", "((x)) > 0", "x > (0)",
          "-2^2 + x > 0", "2^-1 * x > 0", "abs(x)^2^0.5 > 0.5", "abs(x) > 0", "log(abs(x)) > 0", "sqrt(abs(w)) = 1",
          "x*-1 > 0", "x+-w > 0", "x--w > 0", "-x^2 < 0", "2^x > 1", "x / w / 2 = 0", "x - w - zq = 0", "exp(-x) > 1",
          "Intercept + x * 2 < w", "IxE2 - 1 > Intercept", "x > 0", "  x   +  w  =  0 ", "x>-0", "x=-0")
edge <- list(
  all_positive = frame(data.frame(x = c(1, 2, 3, 4, 5, 6, 7, 8))),
  tiny_negative_20 = frame(data.frame(x = c(-0.0001, 1:19))),
  big_negative_20 = frame(data.frame(x = c(-100, 1:19))),
  big_negative_21 = frame(data.frame(x = c(-100, 1:20))),
  ties = frame(data.frame(x = c(0, 0, 0, 1, 2, -1, 3, 4))),
  all_zero = frame(data.frame(x = rep(0, 8))),
  all_negative = frame(data.frame(x = -(1:40))),
  one_draw = frame(data.frame(x = 2)))
edge_hyps <- c("x > 0", "x < 0", "x = 0", "-x = 0", "-x > 0", "-x < 0")
errors <- c("x >= 0", "x == 0", "x <= 0", "x > 0 > w", "x", "x = 0 = w", "foo > 0", "x != 0", "1e-3 < x", "x > 1L", "1 > 0",
            "2x > 0", "> 0", "x > ", "x ~ 0", "pi > x", "X > 0", "x = 0.", "x[1] > 0")
out <- list(brms = as.character(packageVersion("brms")), R = paste(R.version$major, R.version$minor, sep = "."),
            main = list(draws = lapply(main, enc),
                        runs = list(run(main, hyps, samples = TRUE), run(main, hyps, alpha = 0.1), run(main, hyps, robust = TRUE),
                                    run(main, hyps, alpha = 0.2, robust = TRUE))),
            edge = lapply(edge, function(d) list(draws = lapply(d, enc),
                                                  runs = list(run(d, edge_hyps, samples = TRUE), run(d, edge_hyps, alpha = 0.5),
                                                              run(d, edge_hyps, robust = TRUE)))),
            errors = unname(lapply(errors, function(h) refused(main, h))))
writeLines(toJSON(out, auto_unbox = TRUE, digits = I(17), pretty = FALSE, null = "null"), args[[1]])
cat(sprintf("brms %s: %d hypotheses on %d draws, %d edge sets, %d refused\n", packageVersion("brms"), length(hyps), S,
            length(edge), length(errors)))
