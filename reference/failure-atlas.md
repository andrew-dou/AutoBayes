# Failure atlas

Failure modes with their signatures. Match what you are seeing, then read the entry.

Organised by when it bites: environment failures kill the job in seconds, model
failures produce a bad posterior that looks fine.

---

## Environment

These all share a property that makes them expensive: they succeed everywhere except
where it costs money.

### Job dies ~7 seconds in on a missing shared library

**Signature.** `ImportError: libffi.so.8: cannot open shared object file`, or similar,
seconds into an allocation. Works on the login node. Reads like a broken install.

**Cause.** `source activate <env>` does not export `LD_LIBRARY_PATH` on a compute node.
The library is present; the loader cannot see it.

**Fix.** Immediately after activation, in every script:

```bash
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
```

**Why it costs so much.** The job queues, allocates, and dies in seconds — so you
discover it an hour later, having paid for the allocation, and the error names a
library rather than the environment.

### Clean import, dies on the first kernel launch

**Signature.** Imports fine. First GPU operation fails, or silently returns garbage.

**Cause.** The accelerator wheel is built against a newer CUDA than the driver
provides. Importing does not touch the driver; launching does.

**Fix.** Pin the wheel to the driver's CUDA version. Verify with an actual computation,
not an import:

```python
import jax, jax.numpy as jnp
print(jax.devices())
x = jnp.ones((1000, 1000))
print(float((x @ x).sum()))    # a real kernel launch
```

Put this in preflight. An import check passes on a machine that cannot run your model.

### Out-of-memory naming the wrong process

**Signature.** Another job on the node dies with OOM. Yours is fine and using far less
memory than the error implies.

**Cause.** JAX preallocates most of the device at import.

**Fix.**

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
```

### A batch script that runs on the login node

**Signature.** What looks like a hung SSH session, for as long as you allow it.

**Cause.** Tooling that dispatches on file extension executed the script instead of
submitting it.

**Fix.** Keep `.sbatch` on anything meant for `sbatch`. Check the job appears in the
queue before walking away.

### Silent float32

**Signature.** Divergences that look like geometry and do not respond to
reparameterization. Log densities that differ in the sixth digit between runs.

**Cause.** JAX defaults to float32. A hierarchical model accumulates rounding error
through the log density.

**Fix.** `export JAX_ENABLE_X64=true`, and assert `jnp.ones(1).dtype == jnp.float64`
in preflight. This one is insidious because the model works — it just is not the model
you wrote.

### A very long time before the first draw

**Signature.** Minutes of nothing, then normal sampling.

**Cause.** XLA compilation, which is expected once. If it recurs, some shape is
changing between calls and forcing recompilation.

**Fix.** Keep shapes static. Padding to a fixed size beats a shape that varies.

---

## Model geometry

### The funnel

**Signature.** Divergences concentrated where a group-level scale is small. Step size
collapses. Tree depth saturates. E-BFMI low. Pair plot shows the characteristic
narrowing wedge.

**Cause.** The centred parameterization couples a group effect to its scale, so the
posterior's curvature changes by orders of magnitude along the scale axis. No single
step size works everywhere.

**Fix.** Non-centre:

```python
z = pm.Normal("z", 0, 1, dims="group")           # instead of
theta = mu + sigma * z                           # theta ~ N(mu, sigma)
```

**Caveat.** Non-centring is not free — it is *worse* when the data strongly identify
the group effects. Strong data, centred; weak data, non-centred. With mixed group sizes
neither is right everywhere, and the honest answer may be a per-group choice.

**Do not** raise `target_accept` and move on. It masks the signature without fixing the
geometry.

### Multimodality

**Signature.** High R-hat with chains individually stable in separate locations. Trace
plot shows flat lines at different levels rather than a wandering walk. Chain count
matters: it may not appear at four chains and be obvious at forty.

**Cause.** The posterior has genuinely separated modes. Each chain is sampling
correctly, from a different one.

**Diagnosis.** Run many chains from dispersed initializations and cluster the endpoints.
Then evaluate the log density at each mode: a difference of hundreds of nats means one
is dominant and the others are basins the sampler cannot leave; a small difference means
the posterior really is multimodal and reporting a single mean is meaningless.

**Do not** conclude "the sampler is broken." Also do not conclude "the model is
multimodal" without checking that the alternative implementation you are comparing
against actually converged. Both errors are common and they point in opposite
directions.

**Fix.** There is no general one. Reparameterization sometimes merges modes. Additional
prior information sometimes eliminates one. Sometimes the finding is that the model is
non-identified and needs a constraint.

### Non-identified directions

**Signature.** A parameter with enormous posterior variance and low ESS while everything
else is clean. Two parameters with near-perfect posterior correlation. Posterior for
some quantity indistinguishable from its prior.

**Cause.** The data do not constrain that direction. Common with a scale parameter that
can trade off against another, or an intercept absorbed by a group effect.

**Fix.** Constrain it — a sum-to-zero constraint, an anchored reference level, or an
informative prior. Adding data rarely helps, because the direction is unidentified by
the model's structure rather than by sample size.

---

## Implementation

### Gradient returns nan

**Check in this order.** A log of a non-positive number. A scale parameter reaching
zero. A power of a negative number. `0 * inf` in a masked term. Only then suspect a
missing primitive.

Masked terms are the sneaky one: multiplying a masked-out contribution by zero does not
protect you if the contribution is `inf`. Use `pt.switch` to avoid evaluating the
branch, not multiplication to discard it.

### Gradient returns zero, or the sampler will not move

The likelihood is disconnected from the parameters — a common outcome of a masking or
indexing error that drops every observation from a term. Evaluate the log density at two
different parameter vectors. If it does not change, the term is not wired in.

### A non-differentiable primitive

**Signature.** JAX errors on, or returns no gradient for, `betainc`, `erfinv`, or an
iterative solver. Frequently only with respect to one argument — a Student-t CDF
differentiates fine in `mu` and `sigma` and not in `nu`.

**Fix.** See `pymc-to-jax.md`. Short version: replace the special function with
quadrature over a smooth integrand, and only write a custom `Op` after the cheaper
options are gone.

### Fast on CPU, impossibly slow on GPU

**Cause.** A node with no JAX translation forces a fallback, and the graph crosses
between JAX and NumPy on every evaluation.

**Fix.** Find the unjaxified node and register it with `jax_funcify`. If you wrote a
custom `Op`, check that you registered the *gradient* `Op` as well — forgetting that
costs the entire backward pass.

### Elementwise operation fails on a long loop

**Signature.** A build-time error about operand or argument count, triggered by
accumulating a Python list in a loop over time steps and then combining it.

**Cause.** Backends impose a limit on the number of operands to a single fused
elementwise operation, in the low thirties. A Python loop building a random walk hits
it once the panel is long enough — so it works during development and fails when you
extend the data.

**Fix.** Vectorize. `pt.cumsum(increments, axis=0)` replaces the loop, is faster, and
has no length limit. Prefer it from the start rather than after the failure.

---

## Comparison

### A port disagrees with the original

**Do not assume the new implementation is wrong.** Check convergence on *both* first. An
established fit that never converged will disagree with a correct port indefinitely, and
seniority is not evidence.

Then compare at a fixed point rather than comparing posteriors: evaluate both log
densities at the *same* parameter vector. If they differ, it is the likelihood. If they
agree, it is the sampler or the geometry, and posterior comparison was never going to
localize it.

### Two numerical routes to the same quantity disagree slightly

Per-observation differences around `1e-4` in log space are individually negligible and
can accumulate across thousands of observations into a posterior shift — especially
along a poorly identified direction, where a small tilt in the log density moves the
mode a long way.

This means "mathematically equivalent" does not imply "same posterior." If you swapped a
CDF difference for quadrature, or changed a quadrature rule, that is a change to the
model as fitted and belongs in `DECISIONS.md`.

### Predictions disagree while posteriors match

Prediction code is a separate implementation with separate bugs. The usual culprit is
handling of unseen grouping levels — drawing a fresh effect from the population
distribution, versus reusing a known level, versus substituting zero, are three
different models, and the wrong one produces confident, plausible, systematically
overconfident predictions that pass every convergence check.

Validate prediction against held-out data separately from validating the fit.
