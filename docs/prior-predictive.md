# Prior-predictive draws

After constructing a model with explicit priors:

```python
prior = model.prior_predictive(draws=100, seed=42)
# Equivalent: ab.prior_predictive(model, draws=100, seed=42)
print(prior.prior)
print(prior.prior_predictive)
```

This runs CPU forward simulation through the existing PyMC graph, without MCMC.
The result retains native ArviZ prior, prior_predictive and observed_data groups,
variable names and coordinates. It uses modeled rows (including existing
binomial zero-trial filtering); it does not support newdata or a backend/device
option. Constructing the model still requires valid response columns; the API
is not yet a response-free model builder.

Parameters retain native graph parameterization. A centered graph Intercept is
not a raw-scale posterior intercept. Do not rename prior to posterior to use
posterior helpers. Prior output is explicitly refused as a posterior fit.
The built-target digest records construction inputs, including observed values;
forward simulation does not condition on those response values. Seed and draw
count are recorded. Same-seed reproducibility is only established on the pinned
stack, not across library versions.

Nonfinite draws, invalid count support or mismatched dimensions/observed rows
are refused. Broad priors may cause numerical failures (notably Poisson rates);
there is no clipping, rejection resampling or automatic retry. Inspect the priors.
Errors are not evidence of useful posterior convergence or calibrated priors.

Validation:14 tests passed with exactly9 forward calls,8draws/call,6rows/model,
covering five families, seeded repeatability, response-value invariance and a
categorical/interacting varying-intercept Gaussian arithmetic check. Another17
stub/fit-binding tests passed without forward generation. Source hashes, exact
scripts and outputs are in history/prior-predictive-evidence/. No MCMC, GPU, broad prior
calibration or distributional goodness-of-fit claim follows from these tiny tests.
