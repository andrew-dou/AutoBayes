# Gaussian processes: `gp(x)` and `gp(x, k, c)`

brms-parity feature C03 (brms's `gp()`).

```python
import autobayes as ab
priors = ab.Priors(intercept=ab.Normal(0, 5), sigma=ab.HalfNormal(1), sdgp=ab.HalfNormal(1), lscale=ab.InverseGamma(3, 1))
exact = ab.gaussian("y ~ gp(ord)", data, priors)                     # the exact Gaussian process
approx = ab.gaussian("y ~ gp(x, k = 10, c = 5/4)", data, priors)     # its Hilbert-space approximation, 10 basis functions
print(approx.describe())
```

## The model

A Gaussian process of one numeric column `x` adds `f(x_i)` to row `i`'s linear predictor, `f` a zero-mean Gaussian process with
the squared-exponential kernel `k(u, u') = sdgp^2 exp(-(u - u')^2 / (2 lscale^2))`. As brms's default `scale = TRUE` does, the
input is divided by `dmax`, the largest distance between the distinct fitted inputs, so that the largest distance is 1 and
`lscale` is measured on that scale. Rows with equal inputs share one value of `f` (brms's default `gr = TRUE`): the process lives
on the `N` distinct inputs `u_1..u_N`, in the order they first appear in the data, and row `i` takes the value at its own input.

- **Exact** (`gp(x)`; Rasmussen and Williams 2006, ch. 2 and 4): `f = chol(K + 1e-12 I) zgp` with `K[j, m] = k(u_j, u_m)` and
  `zgp ~ Normal(0, 1)`, the non-centred form, `1e-12` added to the diagonal as brms adds it. The Cholesky factor depends on the
  order of the inputs, so `zgp_j` means something only together with that order; it is brms's.
- **Approximate** (`gp(x, k = K, c = C)`; Solin and Särkkä 2020; Riutort-Mayol et al. 2023): on `[-L, L]` with Dirichlet
  boundary conditions the Laplacian has the eigenfunctions `phi_j(v) = sin(pi j (v + L) / (2 L)) / sqrt(L)` and eigenvalues
  `lambda_j = (pi j / (2 L))^2`, and `f(u) = sum_{j <= K} phi_j(u - cmean) sqrt(S(sqrt(lambda_j))) zgp_j` with the kernel's
  spectral density `S(w) = sdgp^2 sqrt(2 pi) lscale exp(-lscale^2 w^2 / 2)`. `cmean` is the mean of the distinct scaled
  inputs and `L = C` times their largest distance, which is 1; brms's default is `C = 5/4`. `sqrt(S)` is formed as
  `sdgp (2 pi)^(1/4) sqrt(lscale) exp(-lscale^2 w^2 / 4)`, finite with a finite gradient where `S` itself underflows (D66).
  Its predictions on new rows are deterministic: the basis at their inputs.

`sdgp` is the marginal SD of `f` (on the scale of the linear predictor), `lscale` its length-scale on the scaled input: inputs
`lscale * dmax` apart are correlated `exp(-1/2)`. The process is not centred, so the intercept is not its mean over the rows.

### How each detail was settled

The mathematics is the published one; brms's documentation, its Stan data and its outputs on probe inputs were the oracle, never
its code (brms is GPL-2, this library MIT).

| detail | settled by |
| --- | --- |
| scaling by `dmax`, inputs not centred for the exact GP | `?gp` ("scaled so that the maximum Euclidean distance between two points is 1"); `Xgp_prior` and the exact `Xgp` of both fixtures and of probes are `x / dmax` exactly |
| one value per distinct input, first-appearance order | `Jgp` and `Xgp` of `gauss_gp_exact` (inputs 4, 1, 3, 2) and of a probe with repeated inputs, for both forms |
| centring of the approximate GP | `cmeans` is the mean of the distinct scaled inputs, not of the rows (`gauss_gp_exact`: 0.8333 = the 4 distinct values' mean, where the rows' is 0.759); the basis of a probe with repeated inputs matches only that |
| `L = c` | `Lgp_1 = 1.25` at `c = 5/4` although the fixture's centred inputs reach only 0.53; probes at `c = 1.5` and `c = 2` on skewed inputs give `L = c` |
| the jitter | `1e-12` on the diagonal of `sdgp^2 exp(...)`, as the fixture's case note records brms's Stan program |
| basis, frequencies, spectral density | the papers above; brms's `Xgp` and `slambda` are reproduced within 3e-15 |

## Priors

- `sdgp` (brms class `sdgp`) and `lscale` (brms class `lscale`, on the scaled input) take positive-support priors (`HalfNormal`,
  `HalfStudentT`, `Exponential`, `HalfCauchy`, `Gamma`, `InverseGamma`), both required (D1). They are keyed by the process's
  column, `sdgp={"x": ...}, lscale={"x": ...}` (brms `coef = gpx`), and a single process also takes the bare priors, as the
  monotonic simplex `simo` is keyed (B03): the column names the process uniquely (one `gp()` per column), brms's coefficient name
  `gpx` is derived from it, and the term as written (`gp(x, k = 10, c = 5/4)`) would be an awkward key. Several processes need
  exact-key mappings; nothing is broadcast, as for `sd`.
- `zgp` is standard normal, as brms's.
- `ab.default_priors()` returns brms's defaults: `sdgp` gets a group-level SD's default `student_t(3, 0, scale)`
  (`HalfStudentT(3, scale)`, brms's scale of the response, which `?set_prior` documents and a probe with a large response scale
  confirms); `lscale` gets the tuned inverse gamma below.

### brms's default `lscale` prior

`?set_prior` documents "an informative inverse-gamma prior specifically tuned to the covariates of the Gaussian process", after
Betancourt's case study (Robust Gaussian Processes in Stan, part 3), with a half-normal prior "if tuning fails". Probing
`default_prior()` on synthetic inputs gives the rule:

- the inverse gamma with 1% of its mass below `l` and 1% above `u`, where `u` is the largest distance between distinct scaled
  inputs (1) and `l` the smallest, floored at 0.01 (inputs at a smallest scaled distance of 0.005 or 0.009 give the same prior as
  0.01; 0.011 does not), and each parameter rounded to 6 decimals;
- with two distinct inputs `l = u`, no inverse gamma has those tails, and brms uses `normal(0, 0.5)` (`HalfNormal(0.5)`).

The two conditions leave one equation in the shape `a` (`b = u P^{-1}(a, 0.01)` from the upper tail, then
`Q(a, b / l) = 0.01`), solved by bracketing (`gaussian_process.inverse_gamma_tails`) and rounded by R's rule. It reproduces the
fixtures' `inv_gamma(18.506437, 9.984811)` (`l = 1/3`) and `inv_gamma(1.494197, 0.056607)` (`l = 0.01`) and brms's printed prior
on 307 of 309 probed inputs `x = (0, l, 1)` with `l` from 0.01 to 0.5, and on every other probe (repeated inputs, two
processes, approximate processes, the floor). At `l = 0.5` and `l = 0.48188...` brms prints a sixth decimal one below the
50-digit solution's (its iterative solver stops just short of a rounding boundary the exact solution crosses): divergence D67.

## Using it

- Formula: `gp(col)` or `gp(col, k = K)` / `gp(col, k = K, c = C)` with `K` a positive whole number and `C` a positive number
  or ratio (`5/4`); brms's defaults written out (`cov = "exp_quad"`, `gr = TRUE`, `scale = TRUE`) are accepted. It is not a
  population-level term (no coefficient of class `b`, no design column): `ModelSpec.gp_terms` holds it
  (`formula.GaussianProcess`, whose inputs, `dmax` and `cmean` are known once the data are seen), next to every other term,
  `y ~ w + gp(x) + (1 + w | g)`, `y ~ 0 + gp(t)`, with `weights()`, `mo()` and a distributional formula on `sigma`, `shape` or
  `phi`, for every family but the categorical one.
- Built as a monotonic term is: Bambi is handed the formula without it and the process is added to the parent's linear predictor
  through the family's `transform_backend_eta` hook, before the link; `Model._extract_design` checks the graph (the variables,
  their transforms and dims) and that the linear predictor moves by exactly the process's values when `zgp` moves.
- Names (`model.gp_names["x"]`, also `names["sdgp:x"]` and so on): `sdgp_x` and `lscale_x`, sampled on their logs
  (`sdgp_x_log__`, `lscale_x_log__`), and `zgp_x` with dims `zgp_x_dim` (`"1"`..`"N"`): brms's `sdgp_k`, `lscale_k`, `zgp_k`.
- `value_point(..., sdgp={"x": 0.8}, lscale={"x": 0.3}, zgp={"x": array(N)})` (bare values for a single process).
- `predict()` adds the process on fitted and new rows. An exact GP has values at its fitted inputs only: a new row at a fitted
  input takes that value, any other input is refused (D65). An approximate GP is evaluated at any input inside its boundary,
  `x` in `((cmean - L) dmax, (cmean + L) dmax)`; an input outside is refused (D66).
- `describe()` writes the process out; the target identity holds its inputs, scaling, row index, basis and priors; fit records
  list it under `gaussian_processes` and its priors as `sdgp:x` and `lscale:x`; `prior_table()` writes brms's `sdgp` and
  `lscale` rows (the `lscale` priors on the coefficient rows `gpx`, the class row empty); `prior_predictive()` draws it.

## Refused, naming the construct (D64)

Several covariates in one `gp()`, `by =`, other kernels (`matern32`, `matern52`, `exponential`), `iso`, `gr = FALSE`, `cmc`,
`scale = FALSE`, `c` without `k`, an expression as input (`gp(log(x))`), two processes of one column, `gp()` in an interaction,
a group-level term or a distributional formula, the categorical family, a non-numeric or infinite input column, a column with one
distinct value, and a `c` that leaves a fitted input outside the boundary. An autocorrelation term `ar()`, `ma()`, `arma()` next
to `gp()` is refused too (C04: brms takes the pair, which no frozen case verifies; docs/autocorrelation.md, D70).

## Evidence

The frozen cases `gauss_gp_exact` (`y ~ gp(ord)`, four distinct inputs) and `gauss_gp_approx` (`y ~ gp(x, k = 10, c = 5/4)`, 36
inputs and five new rows) pass G1-G6 against brms 2.23.0: G1 by content (`Jgp`, `Xgp`, `Xgp_prior`, `dmax`, `cmeans`, `slambda`,
`Lgp`, `NBgp`) within 3.0e-15; G2 the log density with and without the Jacobian and its prior part within 6.9e-15 relative and
every gradient coordinate within 8.9e-12 (exact, whose covariance has condition numbers up to 1.2e7 at the seeded points) and
1.6e-15 (approximate); G3 the graph's linear predictor within 4.8e-14; G4 `posterior_linpred` and `posterior_epred` within
5.2e-14 on the fitted rows and 2.2e-16 on new rows; G5 the JAX density within 1.6e-15; G6 the default-prior tables with their
`sdgp` and `lscale` rows. `tests/test_gaussian_process.py` checks the scaling and order against brms's printed Stan data for a
frame with repeated inputs (both forms, two processes), both densities and every gradient coordinate against a transcription of
the mathematics, the build-time check, the refusals, priors, `value_point`, predictions on fitted and new rows, `describe()`,
records, the inverse-gamma tuning against 50-digit values and brms's printed priors, six brms `default_prior()` tables, and that
the gates fail on corrupted fixtures. Nothing samples.
