#!/usr/bin/env Rscript
# U14 (D12): a grouping factor g:h is built from the labels of g and h joined with "_", so two different combinations whose
# joined labels coincide, (g = "a_b", h = "c") and (g = "a", h = "b_c"), silently become one group level.
#
#   Rscript docs/brms-parity/upstream/repro/U14.R      # exit status 1 while the problem is present
#
# Stock CRAN brms only (make_standata(); nothing is compiled or sampled). Exact: the data have three distinct (g, h)
# combinations, so (1 | g:h) should have three levels, the rows of different combinations in different levels.
t_start <- proc.time()[["elapsed"]]
suppressPackageStartupMessages(library(brms))
cat(sprintf("brms %s, R %s\n", packageVersion("brms"), getRversion()))
d <- data.frame(y = 1:6, g = c("a_b", "a", "x", "a_b", "a", "x"), h = c("c", "b_c", "y", "c", "b_c", "y"))
sd <- make_standata(y ~ 1 + (1 | g:h), data = d)
combos <- unique(d[, c("g", "h")])
cat(sprintf("distinct (g, h) combinations: %s\n", paste(sprintf("(%s, %s)", combos$g, combos$h), collapse = " ")))
cat(sprintf("brms: N_1 = %d levels; the rows' level indices J_1 = %s\n", sd$N_1, paste(sd$J_1, collapse = " ")))
cat(sprintf("exact: %d levels, rows 1 and 2 (the combinations (a_b, c) and (a, b_c)) in different levels; brms: rows 1 and 2 in levels %d and %d\n",
            nrow(combos), sd$J_1[1], sd$J_1[2]))
present <- sd$N_1 != nrow(combos) || sd$J_1[1] == sd$J_1[2]
cat(if (present) "\nPROBLEM PRESENT: brms merges distinct g:h combinations whose joined labels coincide\n" else "\nproblem absent\n")
cat(sprintf("run time %.1f s\n", proc.time()[["elapsed"]] - t_start))
quit(status = if (present) 1 else 0)
