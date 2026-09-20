# M3c-3: multiple independent varying intercepts

2026-09-20 UTC. Decision frozen before implementation on merged main 9755f1d82ac9ba5aa0fcda86b0d32c0806422cff. The September 20 implementation dispatch accepts the conventions below; feature sampling remains separate.

**Observed.** Clean AutoBayes head `072ae0092a3fb0031608a445a61d1d4f771ee331` exposes at most one group throughout parser/specification, design, priors, predictions and records. Bambi 0.17.2 already loops over group terms, constructs separate noncentered Normal effects, and adds their contributions. Its names and formulae 0.6.2 matrix slices settle the relevant coding questions; **zero builds, tests or fits were executed**. Source hashes and exact locations accompany this proposal.

[brms permits multiple grouping factors](https://paulbuerkner.com/brms/reference/brmsformula.html); [Bambi documents multiple group terms and full-rank group intercept coding](https://bambinos.github.io/bambi/notebooks/getting_started.html). These are blueprints, not evidence of complete brms compatibility.

**Proposed contract.** Support any finite list of distinct simple `(1 | factor)` terms across the existing three families. For row i, add `sum_g tau[g] * z[g][index[g,i]]` to the existing linear predictor; offsets have independent standard Normal priors and each positive scale has its own explicit prior. Effects are independent a priori, generally dependent in the posterior. Crossed factors are additive; no nesting or covariance is inferred.

```python
priors = ab.Priors(
    intercept=ab.Normal(0, 2), b=ab.Normal(0, 1),
    sd={"county": ab.HalfNormal(1), "year": ab.HalfNormal(.5)},
    sigma=ab.HalfNormal(1),
)
m = ab.gaussian("y ~ x + (1 | county) + (1 | year)", df, priors)
# Bernoulli/binomial: omit sigma; binomial retains k | trials(n).
# For 0 + x: omit intercept; existing categorical coding remains unchanged.
m.predict(draws, newdata=new, kind="mean",
          new_groups={"county": "population", "year": "error"}, seed=43)
```

Require exact factor-key coverage in `sd` mappings; validate each against existing scale-prior families. Preserve scalar `sd` for one group only; allow a one-key mapping there. Reject broadcasting a scalar across multiple factors. Preserve population-only refusal of supplied SD priors and all positional `Priors` arguments. `value_point(sd={...}, z={...})` likewise requires exact factor keys and correctly sized offsets; existing scalar/vector single-group calls remain valid.

**Names and representation.** Canonical `spec.groups` follows first formula appearance; `levels_by_group[g]` retains sorted string labels from fitted rows. Expose `design.group_designs[g]` containing levels, J, one-hot Z and row index. Check each Bambi term slice, never require the concatenated Z to sum to one. Keep `design.Z` as concatenated blocks in group order, with explicit slices. Preserve existing zero/one-group fields and `ModelSpec` constructor arguments; ambiguous singular `spec.group`, `spec.levels`, `group_term`, `design.J/levels/group_index` raise on multiple groups, rather than selecting the first or signaling population-only.

Expose `model.group_names[g]`: `1|g_sigma`, `1|g_offset`, `1|g`, coordinate `g__factor_dim`. Keep existing `names` roles unchanged for zero/one group; omit ambiguous singular group roles for multiple groups. Equal labels in different factors occupy separate coordinates. Offsets may still be omitted from returned posterior draws under Bambi's existing default.

**Prediction and records.** Preserve string `new_groups` as one policy applied to every factor; exact-key mappings permit different policies. Default errors name the offending factor and unseen labels. For each posterior draw, generate or resample an effect independently for each `(factor, unseen label)`; repeated labels within that factor share it across rows. Resampling uses that factor's observed levels, never observed combinations of factors. Traverse formula order then sorted unseen labels, preserving single-group seeded behavior. Keep existing known-group Bambi routes and binomial/manual routes; extend the latter to sum factors. Preserve zero/one-group attributes; serialize new per-factor policy/level attributes as JSON strings for netCDF compatibility.

Add ordered `groups` metadata to fit records: factor, levels, J, term/names, SD prior and noncentering. Retain legacy `group`, `J` and prior `sd` for zero/one group; omit ambiguous singular entries for multiple groups and use `sd:factor` prior keys. Summaries/comparisons must include every SD and require identical parameter sets. Preserve requested/effective common centering: raw intercept = centered intercept minus common-column means times coefficients. Never subtract group means; absent population intercept still disables centering and creates no Intercept.

**Refusals.** Reject repeated factors, slopes or `(0 | g)`, `||`, correlation IDs, transformed factors, nesting `/`, grouping interactions `:`, and multimembership syntax. Say precisely that `/` and `:` are unsupported grouping constructions; `(1|g)+(1|h)` is permitted even when observed memberships happen to be nested. Do not combine reused within-parent labels. Apply existing response/predictor/trials conflicts, missingness and minimum-two-fitted-level checks separately to every factor, after binomial zero-trial removal. Pure group-only/no-population-predictor formulas retain current restrictions.

**Implementation/test boundary.** Change `formula.py`, `priors.py`, `model.py`, `predict.py`, `session.py`, public descriptions, support docs/example and tests; no new inference backend. Meaningful checks:

- Two unequal-sized groups across all families, both population-intercept states and centering requests; combine categorical and numeric-by-categorical coding. Include a three-group smoke case.
- Original-input normalized likelihood/prior/Jacobian oracle, PyTensor/JAX values and gradients, and fixed-scale Gaussian precision. Distinct scale priors and offsets catch factor cross-wiring.
- Hand-built posterior arithmetic; mixed known/unseen policies; sharing within factors, independence across factors, row reorder and group-order permutation; binomial zero-trial level removal and response support.
- Names/coordinates, explicit prior coverage, value-point failures, JSON records containing every SD; retain population-only/single-group regressions.

**Accepted compatibility choices.** The implementation dispatch accepts the singular-access refusal and record shape above. Risks are downstream single-group assumptions, matrix ordering, and prior-sensitive scale separation with few/confounded groups.

**Future validation proposal, not authorization.** Freeze one 60-row Gaussian 6×5 crossed table with two observations per cell, priors, named parameters and independent fixed-scale Gaussian checks. Then exactly two CPU fits (PyMC/NumPyro), four chains, 1000 tune/1000 draws, target_accept .99, prespecified seeds/deadline and no retries. Require finite active statistics, zero divergences, Rhat≤1.01, both ESS≥400, BFMI≥.3, no depth saturation, named means within four combined MCSE, and retained-draw arithmetic. This tests one backend/API case, not calibration, all-family sampling or GPU performance.


## Implementation authorization and validation limits

September 20 dispatch SHA256: `0b68cc4bdcf9657f0eb9661491b416d869133d80fd9f3f2ac9d295c31d87f2d9`. Sole branch `m3c3/multiple-intercepts`; PR8 and original planning checkout are excluded. Preserve positional ModelSpec/Priors constructors. Natural-scale points require finite positive SDs. Guard generated names/coordinates against collisions. Known-group predictions are invariant under factor reordering up to summation roundoff; unseen RNG traversal depends on formula order, so unseen values are not promised invariant. Comparison requires identical summary parameter sets.

Focused deterministic commands: at most 300 seconds each and 900 seconds aggregate, with exclusive evidence directories and process-group deadlines. ONE inherited fast suite: 900-second deadline, including its existing CPU sampler fixtures. Do not rerun the full suite; preserve every failure and only repair with remaining focused budget. No feature-specific MCMC, GPU, prior predictive campaign, dependency upgrades or sampler-default changes. Open a focused PR; independent review and merge belong to root. Results below must be recorded as observed, never inferred from this plan.


## Implementation and observed validation (2026-09-20 UTC)

The initial decision-stage source review above used 072ae009. Implementation starts at merged main 9755f1d and preserves the declared contract. Version 0.7.0 adds per-factor specification/design/prior/value/prediction/record support. Generated variables/coordinates are checked for collisions; coefficient/effect names ending `_offset` are refused because pinned Bambi's default result cleaning removes that suffix. No third-party lock entry or sampler default changed.

The ONE inherited fast suite passed 524 tests, 7 deselected, 7 warnings in 501.33 wall seconds, exit 0/no timeout, including existing CPU sampler fixtures. Focused runs preserved their initial 1 and 2 failures, then passed 10 repair checks. A dataclass serialization/hash compatibility correction passed 16 focused copy/design/legacy checks, but independent review found public metadata mutation could invalidate the frozen spec’s hash. The final immutable mapping repair passed 19 focused mutation/copy/design/legacy checks; the reviewed failure and earlier candidate remain preserved. The full suite was not rerun. Aggregate focused time 359.06/900 seconds. Exact commands, source hashes, raw failures, receipts and limits: [m3c3-evidence/README.md](m3c3-evidence/README.md). These are deterministic feature checks plus inherited regression fits, not new multigroup MCMC or scientific calibration. Independent review is in progress; separately frozen CPU acceptance remains future work.

Test mapping: `tests/test_multiple_groups.py` covers original-input graph/design; normalized density/Jacobian/PyTensor/JAX gradients; fixed-scale Gaussian precision/mean; synthetic predictions/records/policies; three-group/order/legacy behavior; prior/value/name/collision/refusal/binomial cases; example; dataclass copy/serialization and immutable grouping metadata with stable hash/lookup. Existing population-only and formula refusal tests now reject duplicate factors rather than the newly supported distinct-factor form.
