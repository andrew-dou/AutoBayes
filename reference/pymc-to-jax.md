# PyMC to JAX: GPU sampling and custom likelihoods

Two things this covers: getting an explicit PyMC model sampling on a GPU through
NumPyro, and what to do when the likelihood you need does not exist or will not
differentiate.

## The stack

```
PyMC model  →  PyTensor graph  →  jax_funcify  →  JAX function  →  NumPyro NUTS  →  XLA  →  GPU
```

PyMC builds a symbolic PyTensor graph. Sampling with `nuts_sampler="numpyro"` walks
that graph and rewrites each node into a JAX function, which NumPyro differentiates
with `jax.grad` and XLA compiles into GPU kernels. Nothing in your model needs to know
about JAX — until a node has no JAX translation, at which point the whole compilation
fails or silently falls back to CPU.

Two consequences worth holding onto:

**Parallelism is across chains.** A single gradient evaluation is one fused kernel
launch. You do not get faster gradients by adding cores; you get throughput by running
many chains at once in the same batched program. Forty chains on one GPU costs little
more wall clock than four, which changes how you think about chain counts — running
enough chains to *detect* multimodality becomes nearly free.

**Compilation happens once, at the first gradient.** A model that takes two minutes
before producing a single draw is usually compiling, not sampling. A model that takes
twenty minutes is usually recompiling — check for shapes that change between calls.

## Getting it running

**Check your PyMC version before copying any sampling snippet, including this one.**
This API moved recently and most tutorials online are now wrong.

```python
import pymc as pm

with model:
    trace = pm.sample(
        draws=2500, tune=2500, chains=40,
        nuts_sampler="numpyro",
        nuts={"target_accept": 0.99, "max_treedepth": 12},
    )
```

Three things that changed and will bite you:

- **`nuts_sampler_kwargs` is deprecated.** Current PyMC wants `nuts={...}`. Any snippet
  passing `nuts_sampler_kwargs=` was written against PyMC 5 and will warn or fail.
- **`nutpie` is now the default `nuts_sampler`**, not the Python NUTS. The choices are
  `"pymc"`, `"nutpie"`, `"blackjax"`, `"numpyro"`. nutpie also has a GPU path via
  `nutpie.compile_pymc_model(model, backend="jax")`, so numpyro is not the only route —
  benchmark both rather than assuming.
- **The docs are on PyMC 6.x.** They are not internally consistent about it: pages on
  the same "stable" tree report 6.3.1, 6.0.1–6.2.0, and the examples site reports
  5.25.1. Trust the installed package over any page.

### The `chain_method` gap

On a single GPU you want all chains as one `vmap`'d program. NumPyro's `chain_method`
controls this: `"vectorized"` collects chains in parallel on one device, while
`"parallel"` maps them across XLA devices and **silently falls back to sequential when
there are not enough devices.** That fallback is the usual explanation for "the GPU port
isn't any faster."

The awkward part: `chain_method` is documented on `pymc.sampling.jax.sample_numpyro_nuts`,
and there is **no documented way to pass it through `pm.sample(nuts_sampler="numpyro")`
on current stable.** This is a real, currently-unresolved gap in the documentation, not
something you are missing. If throughput matters, verify empirically which path you are
on rather than assuming `pm.sample` forwards it — or call `sample_numpyro_nuts`
directly and accept that it is the older entry point.

Environment settings that matter more than they look:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # else JAX takes the whole device at import
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export JAX_ENABLE_X64=true                   # float64; see below
```

**Enable float64.** JAX defaults to float32. A hierarchical model in float32 will
produce divergences that look like a geometry problem and are actually accumulated
rounding in the log density. This costs throughput and is not optional for a model you
intend to trust.

**Pin the wheel to the driver's CUDA version.** A wheel built against a newer CUDA
imports cleanly on a login node and dies on the first kernel launch, which — because
the import succeeded — reads as a model bug. Verify with an actual matmul, not an
import. See `failure-atlas.md`.

## Vectorized chains, briefly

`vmap` turns a function written for one input into one that runs on a batch, without a
loop. Chains are the batch dimension: you write the log density for a single parameter
vector, and forty chains run as one program.

The catch is lockstep. A `vmap`'d program executes the same operations on every batch
element, so data-dependent control flow cannot diverge between chains. In practice
NUTS handles this by padding — every chain in a batch runs to the longest trajectory
in that batch. One chain stuck at maximum tree depth therefore slows down all of them.
This is why tree-depth saturation shows up as a wall-clock problem and not only a
statistical one.

## When the likelihood will not cooperate

This is the section you are probably here for.

### First, do not write a custom Op

A custom PyTensor `Op` is a real cost: it is code that sits between you and every
error message for the rest of the project, it has to be correct in both a NumPy path
and a JAX path, and its gradient has to be right in a way nothing will check for you.
Exhaust these first:

- **An existing distribution covers it.** `pm.Censored`, `pm.Truncated`, `pm.Mixture`
  and `pm.CustomDist` cover a wider range than most people assume. `pm.CustomDist` built
  with the `dist=` form — a graph of PyMC primitives — gets its logp inferred
  automatically and stays inside PyTensor, so it can be JAX-compiled.
- **Reparameterize.** Non-centring, working on the log scale, or integrating a latent
  variable out analytically often removes the offending operation entirely.
- **A different numerical route to the same quantity.** This is the big one, below.
- **The operation is fine and something else is wrong.** `nan` gradients are more often
  a log of a negative number, or a scale hitting zero, than a missing primitive.

**Four documented constraints that decide this for you:**

- **`pm.Censored` is likelihood-only.** The docs are explicit: continuous censored
  distributions "should only be used as likelihoods," because they are
  discrete-continuous mixtures that cannot be sampled properly without a custom step
  sampler. For a censored *latent* quantity, model the uncensored variable and wrap it
  in `Deterministic(clip(...))`. Its `dist` argument must implement `logcdf` and is
  cloned, so it must be unnamed.
- **`pm.Truncated` can fall back to rejection sampling.** If the underlying distribution
  has no `logcdf` and no specialized `Op`, PyMC truncates by rejection with a
  `max_n_steps` cap (default 10,000) that raises `TruncationError`. That is a hard
  blocker on a GPU. Check for a `logcdf` before choosing `pm.Truncated` over a manual
  `pm.Potential` normalization.
- **`pm.Potential` does not affect forward sampling.** It influences probability-based
  sampling only — `pm.sample_prior_predictive` and `pm.sample_posterior_predictive`
  ignore it. If you hand-port a censored likelihood as a `Potential`, budget for
  rebuilding posterior predictive separately, or it will silently be the wrong model.
- **There is no `lccdf` primitive.** PyMC gives you `pm.logcdf`; Stan gives you both
  `_lcdf` and `_lccdf`. Right-censoring therefore needs `log1mexp(logcdf(...))` written
  by hand. *(Inferred from the API surface rather than stated in the docs — verify
  against your PyMC version.)*

For an imputed-versus-integrated-out choice, prefer integrating out. PyMC's own censored
-data example notes the unimputed `pm.Censored` form "scales much better with more
censored data, and converges faster" — the imputed form adds one parameter per censored
row, which dominates gradient cost on GPU.

If none of those apply, continue.

### The failure this pattern solves

Some quantities have a closed form that is exact and non-differentiable, and an
approximation that is differentiable and fast. The canonical case is an
interval-censored likelihood — the probability of an observation landing in `[y_l, y_u]`.

The obvious implementation is a CDF difference. For a Student-t, the CDF goes through
the regularized incomplete beta function, and `betainc` has no derivative with respect
to the degrees-of-freedom parameter in JAX. If `nu` is fixed this never surfaces. If
you are *modelling* `nu` — a regression on the tail weight — the gradient is simply
unavailable, and the usual workaround is finite differences: several extra likelihood
evaluations per gradient step, at every step, for the entire fit.

The alternative is to stop asking for the CDF and integrate the density directly:

```
log P(y_l < x < y_u)  =  log ∫[y_l, y_u] f(x | ν, μ, σ) dx
```

evaluated by Gauss–Legendre quadrature. The density is smooth and differentiable in
every parameter, so JAX's ordinary autodiff handles the backward pass — no custom VJP,
no finite differences. Gauss–Legendre with 64 nodes gives roughly 15 digits on a smooth
integrand, which is far more accuracy than the sampler can use.

The general shape of the trick: **replace an exact-but-non-differentiable special
function with a quadrature approximation of an integral whose integrand is smooth.**
It applies well beyond Student-t censoring.

Two things to check before adopting it:

- **Smoothness across the parameter range.** Gauss–Legendre degrades on integrands with
  kinks or near-singularities. A Student-t stays $C^\infty$ even at low `nu`, so it is
  safe; a density with a boundary singularity is not.
- **You have changed the likelihood.** Quadrature and an analytic CDF are mathematically
  equivalent and numerically different. Per-observation differences around `1e-4` in log
  space are irrelevant individually and can accumulate across thousands of censored
  observations into a posterior shift — particularly along a direction the data barely
  identifies. If you are comparing against a Stan or `brms` fit, this is a real
  candidate explanation for a discrepancy, and the way to settle it is to evaluate both
  implementations' log density at the *same* parameter vector rather than to compare
  posteriors.

### Hoisting work out of the quadrature loop

A worthwhile optimisation, since it costs one refactor and buys a large constant. Split
the log density into the part that depends on the integration variable and the part
that does not:

```
log f(x) = log_norm(ν, σ)  +  log_kernel(x, ν, μ, σ)
```

The normalizing constant carries the `gammaln` calls. Computed once per observation
instead of once per observation per node, it removes 63 of every 64 `gammaln`
evaluations — and, more importantly, shrinks the backward pass through `gammaln` and
`digamma` by the same factor.

### The pattern

Three pieces: a pure JAX function, a PyTensor `Op` wrapping it for the CPU path, and a
`jax_funcify` registration so the GPU path uses the JAX version directly.

```python
import numpy as np
import jax, jax.numpy as jnp
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
from pytensor.link.jax.dispatch import jax_funcify

# --- 1. pure JAX, fully differentiable ---------------------------------------

N_QUAD = 64
_nodes, _weights = np.polynomial.legendre.leggauss(N_QUAD)
GL_NODES = jnp.array(_nodes, dtype=jnp.float64)
GL_WEIGHTS = jnp.array(_weights, dtype=jnp.float64)

def interval_logp(y_l, y_u, nu, mu, sigma):
    """log ∫[y_l, y_u] f(x | nu, mu, sigma) dx, by Gauss-Legendre."""
    half = 0.5 * (y_u - y_l)
    mid = 0.5 * (y_u + y_l)
    x = mid[..., None] + half[..., None] * GL_NODES        # map [-1,1] -> [y_l,y_u]
    logf = _logpdf(x, nu[..., None], mu[..., None], sigma[..., None])
    integral = jnp.sum(GL_WEIGHTS * jnp.exp(logf), axis=-1) * half
    return jnp.log(integral)

# --- 2. PyTensor Op: the CPU / graph-building path ----------------------------

class IntervalLogpOp(Op):
    __props__ = ()

    def make_node(self, y_l, y_u, nu, mu, sigma):
        inputs = [pt.as_tensor_variable(v) for v in (y_l, y_u, nu, mu, sigma)]
        outputs = [pt.dvector()]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        outputs[0][0] = np.asarray(interval_logp(*map(jnp.asarray, inputs)),
                                   dtype="float64")

    def grad(self, inputs, output_gradients):
        (gz,) = output_gradients
        grads = _interval_logp_grad_op(*inputs)
        return [gz * g for g in grads]

interval_logp_op = IntervalLogpOp()

# --- 3. jax_funcify: the GPU path --------------------------------------------

@jax_funcify.register(IntervalLogpOp)
def _(op, **kwargs):
    return interval_logp        # NumPyro differentiates this directly
```

Used in a model as `pm.Potential("censored", interval_logp_op(y_l, y_u, nu, mu, sigma))`.

Notes on each piece:

- **`__props__ = ()`** makes instances compare equal, so PyTensor can deduplicate and
  cache the node. Omitting it causes redundant recompilation.
- **`perform`** is the CPU fallback. It runs during graph construction, shape inference,
  and any non-JAX sampling. It must agree with the JAX version — if the two disagree
  you get a model that samples differently on CPU and GPU, which is a genuinely awful
  bug to chase.
- **`grad`** is needed only for the PyTensor path. A companion gradient `Op` wrapping
  `jax.grad(interval_logp, argnums=...)` is the least-duplication way to write it.
- **`jax_funcify`** is the piece that matters on GPU. Register the gradient `Op` too,
  or the backward pass falls back and you lose everything the port bought you.
  **Return an un-jitted callable** — do not wrap it in `jax.jit` yourself. The point is
  that PyTensor jits the *whole graph* as one unit; a pre-jitted function forces a
  boundary in the middle of it. This is the single most common mistake in the pattern,
  and PyTensor's own extension docs call it out.

### Verifying it

Do not skip this. A custom `Op` with a wrong gradient samples happily and gives a
wrong posterior with clean diagnostics.

1. **Value against a reference.** Compare `perform` output to `scipy` — or to a
   high-precision numerical integral — across the full parameter range, especially the
   tails and the boundaries.
2. **Gradient against finite differences.** Once, offline, at several parameter values.
   This is the check that catches the error that otherwise never surfaces.
3. **CPU/GPU agreement.** Evaluate the log density through both paths at the same
   point. They should agree to floating-point tolerance.
4. **Prior-only fit.** With the likelihood contributing nothing, the sampler must
   recover the prior.

Record the tolerances you accepted. When the model disagrees with something six weeks
later, the first question will be whether the custom `Op` is at fault, and "we checked
it" is not an answer.
