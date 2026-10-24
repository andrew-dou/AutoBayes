# Upstream log

Problems the brms-parity loop found in brms and in the libraries under it (Stan Math as shipped with rstan's StanHeaders,
loo): places where they compute something wrong or inexactly, or accept input they should refuse. Each entry has a
standalone R reproduction, so that it can be reported or fixed upstream. Nothing here has been posted anywhere.

- `log.json` (schema `autobayes-upstream-log-v1`): one entry per problem, `U01`, `U02`, ...
- `repro/<id>.R`: the reproduction; `repro/<id>.out`: the output of one run.
- `upstream-log.tex` and `upstream-log.pdf`: generated from `log.json` by `tools/upstream_log.py`.

## Running a reproduction

From the repository root, with stock CRAN brms, rstan and loo (no AutoBayes):

```bash
Rscript docs/brms-parity/upstream/repro/U01.R
```

Each script prints what brms (or Stan, or loo) gives next to the exact value, which it computes itself from a closed form
or a stable log-space formula, and exits with status 1 while the problem is present (0 once fixed, 2 if its own control
check fails). Densities and gradients come from `brm(..., chains = 0)` and `rstan::log_prob` / `grad_log_prob`; predictions
from fixed-parameter runs (`algorithm = "fixed_param"`). Nothing samples. Each took 25 to 250 seconds here (one to
four Stan compilations; up to 600 seconds on a loaded machine); the loo and R-side ones take a second.

## Adding an entry

1. Write `repro/U<next>.R` in the same form (header comment: what, how to run, what is exact; a control point where the
   library is right; the exit status), run it once and save `repro/U<next>.out` (`Rscript ... > ....out 2>&1`).
2. Add the entry to `log.json` with every field (`id`, `title`, `component`, `functions`, `versions`, `divergences`,
   `found`, `kind`, `what`, `observed`, `expected`, `status`, `repro`, `upstream`, `fix`, `known_upstream`). Text is
   ASCII; spans in backticks are set as code. Describe the fix from the mathematics, never from brms's, loo's or Stan's
   source, and check the issue trackers read-only for `known_upstream`.
3. Regenerate: `python tools/upstream_log.py --pdf` (the `.tex` is reproduced byte for byte from `log.json`;
   `--check` compares without writing).
