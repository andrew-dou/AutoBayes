# M0 decision record: Gaussian varying-intercept model

Date: 2026-09-08. Branch `m0/gaussian-varying-intercept`. Written before implementation,
per the handoff in `IQSS-Research/AUTOBAYES_FIRST_MILESTONE.md`.

## What

A small executable path for `y ~ x + (1 | group)` with a Gaussian likelihood, explicit
priors on every parameter class, an inspectable PyMC model, posterior draws, predictive
draws, CPU inference through PyMC's own NUTS, and a JAX/NumPyro path with explicit device
choice. Package `autobayes/` (Python), tests under `tests/`, one researcher example under
`examples/`, and a short pointer from `SKILL.md`.

## What Bambi already supplies (verified against bambi 0.17.2, pymc 5.28.5, numpyro 0.21.0)

- Formula parsing and design matrices via `formulae`: common matrix `[1, x]`, group
  indicator matrix `Z` (sparse one-hot, levels in sorted order).
- Explicit priors by term name, including a hyperprior on the group SD:
  `{"Intercept": Normal, "x": Normal, "1|g": Normal(0, HalfNormal), "sigma": HalfNormal}`.
  With explicit priors Bambi's auto-scaling is not applied.
- Non-centred group effects by default (`noncentered=True`): free RVs
  `1|g_sigma` (log-transformed) and `1|g_offset` (standard normal), deterministic
  `1|g = 1|g_sigma * 1|g_offset`. This matches brms, which is always non-centred.
- Predictor centring by default (`center_predictors=True`): the model is fit on
  `X - mean(X)` and the sampled `Intercept` is the intercept at the predictor means. After
  sampling Bambi subtracts `mean(X) @ b` from the trace so the *reported* `Intercept` is on
  the raw scale. This matches brms, and it means the prior on `Intercept` applies to the
  centred intercept.
- The PyMC model is reachable as `model.backend.model` after `model.build()`. Reading it
  is not a documented Bambi workflow (the reference pack says so), which is why this
  milestone verifies its log density rather than trusting it.
- `fit(inference_method="numpyro", **kwargs)` forwards `**kwargs` to `pm.sample`, and
  PyMC 5.28.5 forwards `nuts_sampler_kwargs` to `sample_jax_nuts`, which accepts
  `chain_method` and `postprocessing_backend`. So the JAX path and its chain method are
  reachable without calling private entry points.
- `predict(kind="response")` produces posterior predictive draws; new group levels are
  refused unless `sample_new_groups=True`, in which case Bambi implements brms's default
  `sample_new_levels="uncertainty"` (resample an observed level's draws). It does not
  offer brms's `"gaussian"` option (draw from the population distribution).

## What the new layer adds, and why each piece is not already in Bambi

1. **Scope validation with useful errors.** Bambi accepts `(1 + x | g)`, `(1 | g/h)`,
   multiple grouping factors, and much more. M0 claims correctness only for
   `y ~ [1 +] x1 [+ x2 ...] + (1 | g)` with numeric predictors and one grouping factor.
   Anything else is rejected with a message naming the unsupported construct. Reason: a
   formula that parses is not a formula this milestone has verified.
2. **Mandatory explicit priors, by parameter class.** Bambi silently supplies default
   priors for anything unspecified. The layer requires a prior for the intercept, every
   slope, the group SD, and `sigma`, and refuses to build otherwise. Reason: the
   milestone's definition ("explicit priors") and the brms-porting lesson that silent
   defaults are the main source of unreproducible models.
3. **A model description that states what was actually built.** Centring, the
   non-centred parameterisation, the group level to index map, the free variables and
   their transforms, and the exact density. Reason: Bambi's `print(model)` states priors
   but not centring or parameterisation, and the PyMC model's meaning otherwise has to be
   recovered from `pytensor.dprint`.
4. **Explicit device and backend resolution with an honest report.** Bambi and PyMC do
   not manage JAX devices. The layer resolves `device="cpu"|"gpu"` against
   `jax.devices()`, raises when the requested device is absent (never a silent CPU
   fallback), enables float64 on the JAX path before any array is created, sets the host
   device count for parallel CPU chains, and records requested device, actual device,
   dtype, and chain method alongside the trace.
5. **Predictive draws with named semantics.** In-sample and out-of-sample posterior
   predictive draws through Bambi, plus an explicit `new_groups` argument whose options
   are named after what they do (`"error"`, `"resample_observed"`, `"population"`). The
   `"population"` option is implemented in the layer because Bambi lacks it and it is the
   one that samples the fitted population distribution.
6. **A diagnostics panel** with the thresholds from `reference/diagnostics.md`.

Not added: a formula parser, design-matrix builder, prior language, or sampler. Bambi and
PyMC own those.

## Acceptance checks (set before implementation)

Log density and meaning (tests, CPU):

- A1. The joint log density from the Bambi-built PyMC model (`compile_logp`, Jacobian
  included) equals an independently written NumPy/SciPy reference at five random points
  in the value space, to within 1e-8 absolute. The reference is written from the
  mathematical specification (centred X, non-centred group effects, log transforms with
  Jacobians), not from PyMC output.
- A2. Same with `jacobian=False`, to isolate the transform terms.
- A3. The gradient from `compile_dlogp` matches central finite differences of the
  reference at the same points, to within 1e-5 relative.
- A4. The JAX-compiled log density (`get_jaxified_logp`, float64) agrees with the C/NumPy
  path at the same points to within 1e-8.
- A5. Analytic check: with `sigma` and the group SD fixed by `pm.do`, the Hessian of the
  log density in (intercept, slope, offsets) equals the closed-form Gaussian precision.
- A6. Design and coding: the common design matrix is `[1, x - mean(x)]`, the group index
  maps sorted levels to columns of a one-hot `Z`, and `mu` evaluated from the PyMC model
  at a point equals `a_c + (x - xbar) b + sd_g * z[g]` computed by hand.
- A7. Unsupported formulas (`(1 + x | g)`, `(1 | g/h)`, two grouping factors, categorical
  predictors, no intercept `0 +`, missing priors) raise errors naming the construct.
- A8. Predictive meaning: in-sample posterior predictive draws have mean equal to the
  posterior mean of `mu` within Monte Carlo error, and out-of-sample draws for observed
  groups use that group's effect; `new_groups="population"` draws have between-group
  spread consistent with the posterior of the group SD.

Sampling (CPU, modest sizes):

- B1. Conditional-Gaussian check: with the scales fixed via `pm.do`, NUTS posterior
  means of the linear parameters lie within 4 MCSE of the exact conditional posterior
  means, and posterior SDs within 10% of the exact SDs.
- B2. Simulated recovery, n = 200, J = 20: 4 chains, 1000 tune, 1000 draws, on both the
  PyMC and NumPyro CPU paths. Panel thresholds from `reference/diagnostics.md`: zero
  divergences, max R-hat <= 1.01, bulk and tail ESS > 400 total per parameter (the 400
  per chain in the panel is the target for production fits; for a 4x1000 smoke run, 400
  total is the stated threshold). True values inside the 94% HDI for the population
  parameters, and the two backends' posterior means within 4 combined MCSE of each other.
  One recovery is not a calibration claim and is not reported as one.

Device (this machine has no CUDA device):

- C1. `device="gpu"` raises a clear error listing available JAX devices.
- C2. The device report on the CPU JAX path states `cpu`, float64, and the chain method.
- C3. GPU execution is marked **pending** with a reproducible command. Support in code
  is not reported as validation.

## Kill criterion

If the Bambi-built model's log density cannot be reconciled with the reference after
accounting for centring and transforms (A1 fails with an unexplained discrepancy), the
Bambi build is not trustworthy for this milestone, and the fallback is a hand-written
PyMC model for the same formula, with Bambi kept only as a cross-check.

## If a check fails

Repair and re-run. Record the failure in `docs/coordination/STATUS.md`. If B2 fails on
one backend only, the difference is a sampler-path bug and is investigated before any
claim about either backend.
