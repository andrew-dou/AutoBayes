# Scaling: group-level terms from level codes

A model with many rows and many levels builds, evaluates its log density and gradient, and predicts in memory proportional to
the rows times the coefficients (O(N k)), not to the rows times the levels (O(N J)). Every model that built before builds the
same model: the same PyMC graph (free variables, value variables, deterministics, dims and coordinates, in order), the same log
density and gradient bit for bit, the same `target_identity`, `describe()`, names and design arrays (`tools/model_manifest.py`
records all of them; see "How it was checked").

```python
import autobayes as ab
m = ab.bernoulli("y ~ x1 + x2 + (1 + x1 | g)", data, priors)   # N = 1e6 rows, J = 1e4 levels of g: builds in under 1 GB
m.design.group_blocks[("g", "x1")].group_index                # each row's level, int64
m.design.group_blocks[("g", "x1")].covariate                  # x1 per row; None for a varying intercept
m.design.Z                                                     # refused above DENSE_Z_MAX_ENTRIES
```

## What changed

Before, Bambi was handed the whole formula, and formulae built a dense `N x J` int64 indicator per grouping factor (and a
`J x J` identity), which Bambi kept alive; the library then densified each block again (`GroupBlock.Z`), and the target identity
encoded every entry of every dense block as a JSON dictionary (about 290 bytes and 1.35 microseconds per entry). A
`(1 + x1 | g)` Bernoulli model at N = 1e5, J = 1e3 passed 8 GB before its identity was computed.

Now:

- **Bambi is handed the population-level formula** (`ModelSpec.bambi_population_formula`; `bambi_formula` is unchanged, the
  formula with its group-level terms). Before `build()`, the model adds each group-level coefficient Bambi builds to Bambi's
  component as an `IndexGroupTerm` (`autobayes/group_index.py`, a `bambi.terms.GroupSpecificTerm`): the factor's sorted levels,
  each row's 0-based level code, the column the coefficient multiplies as formulae would read it (int64 ones for a varying
  intercept, the frame's column for a slope) and the prior Bambi would have given it, in the order formulae gave the terms.
  Bambi's backend reads only that (`coef[group_index] * predictor`), so it builds the graph it built from formulae's design. The
  same holds for a distributional formula's terms (a shared correlation ID's and its own, `sigma ~ z + (1 | g)`).
  `Model._extract_design` checks every term against the data (levels, codes, column), as it checked formulae's blocks.
- **`GroupBlock` keeps codes and covariates**: `group_index`, `covariate` (a slope's column, None for ones), and for `mm()` the
  members' codes, weights and covariates. `GroupBlock.Z` and `Design.Z` are built on access, exactly the arrays they were
  (zeros, then each row's value added at its level, a multi-membership term's members one after another), and refused above
  `group_index.DENSE_Z_MAX_ENTRIES` = 50,000,000 entries (400 MB of float64) with a message naming the cap. `GroupBlock.dot(v)`
  is `Z @ v` without `Z`; `GroupBlock.row_values()` each row's entries. A block given a dense `Z`
  (`GroupBlock(levels, Z, group_index, column_slice)`, as the non-linear and count-matrix families build theirs) keeps it.
- **The target identity is streamed** (`identity.digest`): the payload's schema-v1 JSON is written into sha256 piece by piece,
  every array in chunks and every group block's dense `Z` row by row from its codes (`identity.LazyZ`), so the bytes hashed,
  and so every digest, are those of v1. Its memory is O(N); its time is still that of hashing the v1 bytes, about 24 bytes per
  entry of each dense block, `24 N sum(J k)` bytes in all, at about 2 GB/s (measured below).
- **Predictions and `log_lik`**: one gather per (member, coefficient) over the draws, `mu += u_ext[..., codes] * covariate`,
  the unseen labels' effects appended to the fitted levels' as codes `J, J + 1, ...` and drawn exactly as before. Bambi adds
  group effects only from formulae's design, which a model no longer has, so every model with group-level terms is predicted by
  the library, as `mm()` and `gr(by = )` always were. `kind="linear"` is bit for bit what it was, in-sample and with new levels;
  `kind="mean"` of a model Bambi used to predict may differ in its last digits (the summation order: at most 3.3e-13 relative on
  the frozen cases but for one entry near zero, an absolute 1.1e-16; 15 of the 52 such cases are unchanged bit for bit); `kind="response"` draws of such a model come from the family's `draw_response` with
  `numpy.random.default_rng(seed)` rather than from Bambi's `posterior_predictive` (D199).

## Memory per term

Per grouping factor: its level codes, int64, 8 N bytes (shared by its coefficients), and its labels as text in the fitted frame.
Per coefficient: its covariate, 8 N bytes for a slope (none for a varying intercept), held by the design and as a constant in
the graph (N values and N codes for each gathered term in PyTensor and in the JAX program); its levels' effects, J per draw.
A multi-membership term: per member its codes, weights and (for a slope) covariates, 8 N bytes each. Predictions hold a few
arrays of draws x rows (the linear predictor and one gathered term at a time).

What stays dense, as it was:
- the non-linear (`nl=True`) and count-matrix (multinomial, Dirichlet-multinomial) models, which build their own designs: per
  formula, a dense block per group-level coefficient and their concatenation, `2 x 8 N sum(J k)` bytes (their graphs gather
  by index already; only their designs are dense). Their identities are streamed like every other.
- `gr(g, cov = A)`'s known covariance, a dense `J x J` matrix and its Cholesky factor (`8 J^2` bytes each), which the model
  reads.

## Measurements (no sampling; one guarded process at a time, 8 GB cap)

`ab.bernoulli("y ~ x1 + x2 + (1 + x1 | g)")` and `ab.gaussian("y ~ x1 + (1 | g1) + (1 | g2)")` (g2 with J / 2 levels), with
`loop-notes/scale-design/measure.py`: the build (time and the process's peak physical footprint), the identity's share of it,
one log density and gradient after compilation on PyTensor and on JAX, and `predict(kind="linear")` and `log_lik` on 100
synthetic draws.

| model | N | J | where | build s (identity s) | build peak GB | logp + dlogp ms, PyTensor / JAX | predict linear s | log_lik s | peak with 100 draws GB |
|---|---|---|---|---|---|---|---|---|---|
| bern | 1e4 | 1e2 | main | 8.9 (2.5) | 1.14 | 0.33 / 0.23 | 0.06 | 0.09 | 1.18 |
| bern | 1e4 | 1e2 | G11 | 5.9 (0.08) | 0.56 | 0.26 / 0.27 | 0.03 | 0.06 | 0.65 |
| bern | 1e5 | 1e3 | main | killed at 8 GB while building | | | | | |
| bern | 1e5 | 1e3 | G11 | 8.4 (2.5) | 0.60 | 2.5 / 0.9 | 0.24 | 0.46 | 0.83 |
| bern | 1e5 | 1e4 | G11 | 26 (20) | 0.60 | 2.9 / 1.2 | 0.25 | 0.48 | 0.85 |
| bern | 1e6 | 1e3 | G11 | 34 (28) | 0.87 | 24 / 7.6 | 3.5 | 6.8 | 4.2 |
| bern | 1e6 | 1e4 | G11 | 204 (197) | 0.81 | 24 / 8.0 | 4.4 | 6.6 | 4.2 |
| gauss | 1e4 | 1e2 | main | 1.9 (1.8) | 0.72 | 0.20 / 0.10 | 0.06 | 0.07 | 0.75 |
| gauss | 1e4 | 1e2 | G11 | 0.21 (0.14) | 0.29 | 0.15 / 0.14 | 0.02 | 0.03 | 0.64 |
| gauss | 1e5 | 1e3 | main | killed at 8 GB while building | | | | | |
| gauss | 1e5 | 1e3 | G11 | 2.5 (2.3) | 0.29 | 1.6 / 1.3 | 0.22 | 0.22 | 0.80 |
| gauss | 1e5 | 1e4 | G11 | 16 (16) | 0.31 | 1.8 / 0.65 | 0.23 | 0.23 | 0.82 |
| gauss | 1e6 | 1e3 | G11 | 23 (22) | 0.47 | 15 / 3.9 | 3.6 | 3.9 | 4.1 |
| gauss | 1e6 | 1e4 | G11 | 150 (149) | 0.56 | 16 / 3.8 | 3.7 | 2.8 | 4.1 |

Peaks are the process's physical footprint (about 0.3 GB of it the imported libraries). The build is now under 1 GB at every
size; the identity is its cost, hashing `24 N sum(J k)` bytes (about 0.45 TB at N = 1e6, J = 1e4 for the Bernoulli model) at
about 2.3 GB/s. The JAX program has no N x J shape at any size. With 100 draws at N = 1e6 a prediction holds arrays of
draws x rows (0.8 GB each), about 3 GB at its peak. main's builds at N = 1e5, J = 1e3 passed the 8 GB cap (formulae's indicators
and the dense blocks; the identity would have needed about 60 GB); the larger sizes were not attempted on main, formulae's int64
indicator alone being 8 to 80 GB. Measured on 3 November 2026 on this machine (shared, swapping), timings indicative.

## How it was checked

`tools/model_manifest.py` records, per model, the target identity, `describe()`, names, the graph's variables, dims and
coordinates, a sha256 of every design array (dtype, shape and bytes: sign bits count), and for every frozen oracle case the log
density and gradient at the fixture's points and at seeded points (with and without the Jacobian, PyTensor and JAX) and
`predict` and `log_lik` on a seeded synthetic posterior. It was recorded on main before the change and compared after it, for
the oracle cases and, through a pytest plugin with the cheap fields, for every model the deterministic suite builds.
`tests/test_index_group_design.py` checks the lazy blocks against formulae's dense design and the model's own construction (sign
bits included), the cap, the streamed digest against the v1 JSON on random payloads, the gathers against a dense reference (new
levels included), Bambi's terms, a guard on Bambi 0.17.2's private layout (`group_index.bambi_layout_problems`: a changed
version or source of the functions this design relies on fails it), and a memory test in a subprocess (N = 2e5, J = 2e3: build,
log density and gradient on both backends, predictions, `log_lik` and new rows with a new level under 0.5 GB of traced memory,
no N x J shape in the JAX program, `Z` refused; the same at J = 2e4 is marked `slow`, as its identity hashes about 190 GB).
