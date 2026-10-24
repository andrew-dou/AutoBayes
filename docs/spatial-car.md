# Spatial CAR terms: `car()`

brms-parity feature F02a (brms's `car(M, gr, type)`, conditional autoregressive spatial effects, for the types `escar`,
`esicar`, `icar` and `bym2`).

```python
import autobayes as ab
priors = ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1),
                   sdcar=ab.HalfNormal(1), rhocar=ab.Beta(1, 1))
model = ab.gaussian('y ~ x + car(W, gr = site, type = "bym2")', data, priors, data2={"W": W})
print(model.describe())
```

`W` is a pandas DataFrame, the 0/1 adjacency of the locations, labelled by location on both axes (brms's `data2 = list(W =
W)` with row names); `site` is the column of each row's location.

## The model

The term adds one effect per location to the linear predictor, `rcar[site_i]`. With the fitted locations (the labels of
`site`, sorted; brms's `Nloc` of them), `W` their adjacency, `D` the diagonal matrix of their numbers of neighbours (brms's
`Nneigh`) and `tau = 1 / sdcar^2`:

- `escar`, the exact sparse proper CAR (brms's default type): `rcar ~ MVN(0, (tau (D - car W))^-1)`, `car` in (0, 1), its log
  density `(Nloc log tau + sum_k log(1 - car lambda_k) - tau (rcar' D rcar - car rcar' W rcar)) / 2` with `lambda` the
  eigenvalues of `D^-1/2 W D^-1/2` (brms's `eigenMcar`), without the constants `log det D / 2` and `Nloc log(2 pi) / 2`, as
  brms writes it;
- `esicar`, the exact sparse intrinsic CAR: `zcar` has `Nloc - 1` free entries, `rcar = (zcar, -sum(zcar))`, log density
  `((Nloc - 1) log tau - tau rcar' (D - W) rcar) / 2`;
- `icar`, the intrinsic CAR in pairwise differences (Morris et al. 2019): `rcar = sdcar zcar`, `zcar`'s log density
  `-sum over the edges (zcar_i - zcar_j)^2 / 2` plus brms's soft sum-to-zero constraint `log Normal(sum(zcar) | 0, 0.001 Nloc)`,
  which dominates the density away from the constraint;
- `bym2` (Riebler et al. 2016): `rcar = sdcar (sqrt(1 - rhocar) nszcar + sqrt(rhocar / s) zcar)`, `zcar` under the ICAR density
  above, `nszcar ~ Normal(0, 1)`, `rhocar` in (0, 1) the spatial share, and `s` the scaling factor (brms's `car_scale`).

The scaling factor is the geometric mean of the marginal variances of the ICAR precision `Q = D - W` under the sum-to-zero
constraint. brms computes it as INLA's `inla.scale.model` does, inverting `Q + eps I` (`eps = max(diag Q) sqrt(machine
epsilon)`) conditioned on `sum = 0`; AutoBayes evaluates that definition exactly, `sum_k v_k v_k' / (lambda_k + eps)` over the
eigenpairs of `Q` but the null vector, without the cancellation of the nearly singular inverse (divergence D140).

## How it is built

Bambi is handed the formula without the term; the term is added to the parent's linear predictor while Bambi builds the
model, through the family's `transform_backend_eta` hook (`likelihoods.monotonic_family`, as for `mo()` and `gp()`), by
`car.car_effect`: `sdcar` (and `car` or `rhocar`) under their priors, the latent vectors on the locations (flat, but
`nszcar`'s standard normal), the CAR log density as a potential (`rcar_lpdf` or `zcar_lpdf`, brms's model block, not its
`lprior`) and `rcar` (the free vector of escar, a deterministic otherwise). When the model is built the graph's `rcar` is
checked against `car.car_values` and the linear predictor's move against `rcar` at each row's location. `Design.car`
(`car.CarDesign`) holds the locations, each row's location (brms's `Jloc - 1`), the edges in brms's order (`edges1`,
`edges2`: the lower triangle, column by column), `Nneigh`, `eigenMcar` and `car_scale`.

## Using it

- Formula: one term, `car(M, gr = g, type = "...")` with R's argument rules (`car(W, site, "icar")`), `type` defaulting to
  `"escar"`, next to the term model's population- and group-level terms and `offset()` (the column of locations may also be a
  grouping factor). The canonical formula writes the term last. `data2={"W": W}` on `ab.gaussian` and `ab.poisson` (and
  `ab.default_priors`) hands the matrix over, through the shared `data2` of `sar()` and `fcor()` (`ModelSpec.data2_names` lists what
  the terms read; `formula.check_data2` refuses an entry no term reads).
- The matrix: every fitted row's location must label `W` (brms's rule); locations `W` has and the rows lack are dropped from
  the graph with their edges, as brms drops them, with a warning; entries other than 0 and 1 are read as 1 with a warning (as
  brms converts them); the columns are the row labels in the same order, or positions.
- Priors: `sdcar` a positive-support prior (brms class `sdcar`; brms's default `student_t(3, 0, scale)` as for `sd`); `car`
  (escar) and `rhocar` (bym2) a `Beta` prior or `Flat()`, the uniform on (0, 1), sampled on Stan's logit (brms's `car` default
  is flat, `rhocar`'s `beta(1, 1)`); each required exactly for the type. `ab.default_priors()` returns brms's defaults and
  `prior_table()` writes the class rows `sdcar` (lb 0), `car` and `rhocar` (lb 0, ub 1).
- `value_point(..., car={"sdcar": 0.8, "rhocar": 0.4, "zcar": ..., "nszcar": ...})`: the type's parameters and latent vectors
  by brms's names (`rcar` for escar, `Nloc - 1` entries of `zcar` for esicar).
- Names: `model.car_names` (`sdcar`, `car`, `rhocar`, `rcar`, `zcar`, `nszcar` as the type has them; the locations' dimension
  `rcar_dim`, esicar's `zcar_dim`); a posterior lists `rcar[s01]`, ... .
- `predict()` adds each draw's `rcar` at the row's location, on the fitted rows and on new rows at fitted locations, with or
  without group effects (brms's `re_formula = NA` keeps the term; probed with a fixed-parameter fit); new locations are
  refused, as brms refuses them ("Cannot handle new locations in CAR models."), and `newdata` needs the column of locations.
- Post-processing: `ab.log_lik()` is each row's density at the predictor with `rcar`, as brms's (probed); `ab.bayes_R2()` and
  `ab.loo()` follow from it and from `predict(kind="mean")`.
- `describe()` writes the locations, the term, its density and its priors; the target identity holds the term, the locations,
  each row's location, the edges, the scaling factor and the priors; fit records list `car` and the priors.

## Refused, naming the construct

Families other than the Gaussian and Poisson (and the Poisson's sqrt link), a distributional formula next to the term or the
term inside one, multivariate and non-linear models, mixtures, addition terms (`weights()`, `cens()`, `trunc()`, `mi()`),
`mo()`, `gp()`, `me()`, `mi()`, `cs()`, `mm()`, `gr(by = )`, autocorrelation terms, `sar()` and `fcor()` beside it, a shrinkage prior on `b`, two
`car()` terms (brms too), `gr = NA` (brms deprecates it), a matrix that is not a labelled square DataFrame, not symmetric
(brms too), with non-finite entries or a non-zero diagonal, a location without neighbours, a graph of several connected
components, fewer than two locations, `prior_predictive()`, `conditional_effects()` and `prior_draws()` (D141); and the
priors `sdcar`, `car`, `rhocar` in any model without the term.

## Evidence

Five frozen cases on the 5 by 5 rook grid of tier F (25 sites, 40 edges, 50 rows): `gauss_car_escar`, `gauss_car_esicar` and
`gauss_car_icar` pass G1-G6 against brms 2.23.0 (log densities within 3.5e-16 relative, gradients within 3.6e-15, the
icar's density dominated by the soft constraint, down to -12020); `gauss_car_bym2` and `poisson_car_bym2` pass G2-G6 and keep
G1 as an expected failure (`"gate": {"gates": ["g1"], ...}`): brms's `car_scale`, 0.5163859009642695, is 1.7e-10 below its own
definition's value 0.5163859011368434, beyond G1's 1e-12, and through it their log densities are within 6.4e-11, their gradients
within 3.1e-9 and the graph's linear predictor within 6.1e-10 relative of brms's (D140). `tests/test_car.py` covers the parsing and refusals, the
locations and graph against brms's Stan data on a 3 by 3 grid (with and without a location), the scaling factor against an
exact rational evaluation of its definition and brms's values, each type's log density and gradient against an independent
transcription (escar against scipy's multivariate normal), the Poisson family, group terms and an offset beside the term, the
JAX density, the build-time graph check, priors and brms's default priors, `value_point`, predictions, `log_lik`,
`bayes_R2`, `loo`, `describe()`, names, the target identity, fit records, the contexts that refuse the term and corrupted
fixtures. Nothing samples.
