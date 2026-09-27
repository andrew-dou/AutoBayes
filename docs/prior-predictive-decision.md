# Prior-predictive foundation — September 27, 2026

Authority: approved foundations roadmap step5. Baseline main485b705e26d04d923c32933fd02b9bbe9c6d6875. Source/design unit only; no implementation, forward draws, MCMC or GPU launched.

## Small first API

Proposed `model.prior_predictive(draws=500, seed=None)` and top-level equivalent return ArviZ InferenceData with prior, prior_predictive and observed_data groups on the existing modeled rows. Use the pinned PyMC forward sampler directly on the existing graph. No backend/device argument, newdata, posterior fit wrapper or arbitrary sampler kwargs in this first slice. It is forward simulation on CPU, not NUTS or a GPU parity claim. Draws must be a positive integer (reject bool); seed supports a nonnegative integer or None with explicit validation.

Preserve graph-native variable names and coordinates, raw observed data and the existing fitted-row filtering. Include seed, requested draw count, software version, generation kind=prior_predictive and a separately named built-target identity metadata field. Do not stamp fit provenance or pretend prior draws are posterior draws. Prior output must not be silently accepted as a bound posterior fit. A target fingerprint includes observed responses for construction provenance even though their values do not condition the forward draws.

Return raw graph parameterization. In particular, do not rename prior to posterior and call existing manual prediction: that helper assumes a raw-scale intercept, whereas graph priors may use a centered intercept. Document this and test response arithmetic with graph design coordinates. Preserve binomial trials and zero-trial fitted-row policy. No automatic rejection sampling, clipping or retry when broad priors generate nonfinite values or unrepresentable Poisson rates; raise a clear failure and preserve it. Do not label failed draws usable.

## Source evidence

Read September27: brms brm reference https://paulbuerkner.com/brms/reference/brm.html documents sample_prior=only ignoring likelihood and requiring proper priors. PyMC stable sample_prior_predictive docs https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample_prior_predictive.html currently describe PyMC6.3.2; do NOT use this as the pinned5.28.5 signature or upgrade dependencies.

Pinned Bambi0.17.2 models.py lines817 onward calls pm.sample_prior_predictive on its graph and returns InferenceData; it does not apply posterior raw-intercept conversion in this method. Pinned PyMC5.28.5 sampling/forward.py exposes draws, random_seed, model, return_inferencedata, samples compatibility. Existing custom Bernoulli/binomial random callbacks use the actual logit parameters and trials. Source hashes recorded alongside this plan.

## Bounded implementation/validation

Write a repository decision in a new isolated worktree before code. Start with deterministic validation/stub tests for arguments, dispatch, output metadata, no model mutation and fail-closed output validation. Then one separately recorded CPU forward-simulation test unit: maximum10 tiny model calls, at most16 draws each and12 rows per model; 300-second combined UTC-or-monotonic cap, zero automatic retries, no MCMC. Cover five families, one grouped/categorical/interacting case, seeded repeatability and observed-response invariance where graph structure/prior remain identical. Check dimensions/support, centering arithmetic against independently reconstructed design, and that group labels/observed rows remain aligned. Broad prior calibration is excluded. Preserve script, hashes, environment, failures and actual result; do not infer distribution calibration from tiny draws.

This is a bounded unit within the authorized predictive workflow, not a broad simulation campaign. If graph parameterization or resource limits prevent reliable implementation, preserve the specific blocker instead of expanding silently. Existing hosted regression CI gates PR/merge after review. Do not alter PR8/F7, data work or dependencies.
