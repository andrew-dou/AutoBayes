---
name: autobayes-custom-op
description: >
  Use when a Bayesian model's likelihood cannot be expressed with built-in
  distributions, or will not differentiate. Symptoms: gradients return nan or zero;
  JAX or NumPyro errors on a non-differentiable primitive such as betainc, erfinv, or
  an iterative solver; a Student-t CDF differentiates in mu and sigma but not in nu; a
  CDF-difference likelihood loses precision; sampling is fine on CPU and impossibly
  slow on GPU; you are considering finite differences inside a sampler; or you need
  interval censoring, a truncated mixture, or a custom integral in PyMC/PyTensor.
  Covers whether to write a custom Op at all, and the make_node / perform / grad /
  jax_funcify pattern if you should.
---

# Custom likelihood operations

You are here because a likelihood will not cooperate. This skill decides whether a
custom operation is the right answer — usually it is not — and gives the pattern if it
is.

## Do not write a custom Op yet

A custom `Op` is a permanent cost. It sits between you and every error message for the
rest of the project, it must be correct on both a NumPy path and a JAX path, and its
gradient must be right in a way nothing will check for you. A wrong gradient samples
happily and produces a wrong posterior with clean diagnostics — the worst failure mode
available.

Work down this list. Stop at the first thing that applies.

**1. Is something else actually wrong?**
`nan` gradients are more often a log of a non-positive number, a scale parameter
reaching zero, or `0 * inf` in a masked term than a missing primitive. Masking is the
sneaky one: multiplying a masked-out contribution by zero does not save you if that
contribution is `inf`. Use `pt.switch` to avoid evaluating the branch rather than
multiplication to discard it. Check this before anything else.

**2. Does an existing distribution cover it?**
`pm.Censored`, `pm.Truncated`, `pm.Mixture` and `pm.CustomDist` cover more than most
people assume. **`pm.CustomDist` built with the `dist=` form — a graph of PyMC
primitives — gets its logp inferred automatically and stays inside PyTensor, so it can
be JAX-compiled.** This is the single most commonly missed option: if you can express
the log density in ordinary PyTensor operations, you are done.

Four documented constraints that often decide this for you:

- **`pm.Censored` is likelihood-only.** Continuous censored distributions "should only
  be used as likelihoods" — they are discrete-continuous mixtures that cannot be
  sampled properly without a custom step sampler. For a censored *latent* quantity,
  model the uncensored variable and wrap it in `Deterministic(clip(...))`.
- **`pm.Truncated` can fall back to rejection sampling** when the underlying
  distribution has no `logcdf`, with a `max_n_steps` cap that raises `TruncationError`.
  That is a hard blocker on GPU. Check for a `logcdf` first.
- **`pm.Potential` does not affect forward sampling.** Prior and posterior predictive
  ignore it. Hand-porting a censored likelihood as a `Potential` means rebuilding
  posterior predictive separately or silently reporting the wrong model.
- **There is no `lccdf` primitive** — right-censoring needs `log1mexp(logcdf(...))`
  written by hand. *(Inferred from the API surface, not documented; verify against your
  version.)*

**3. Can you reparameterize the problem away?**
Non-centring, working on the log scale, or integrating a latent variable out
analytically often removes the offending operation entirely.

**4. Is there a different numerical route to the same quantity?**
Usually yes, and usually this is the answer. See below.

**5. Only now**, write the `Op`.

## The reframe that usually works

Some quantities have a closed form that is exact and non-differentiable, and an
integral form that is approximate and differentiable. Prefer the integral.

The canonical case: the probability of an observation falling in `[y_l, y_u]`. The
obvious implementation is a CDF difference; for a Student-t the CDF runs through the
regularized incomplete beta function, and `betainc` has no derivative with respect to
the degrees-of-freedom parameter in JAX. With `nu` fixed you never see this. If you are
*modelling* `nu`, the gradient is unavailable and the usual workaround is finite
differences — several extra likelihood evaluations per gradient, every step, for the
whole fit.

Instead, integrate the density directly:

```
log P(y_l < x < y_u) = log ∫[y_l, y_u] f(x | ν, μ, σ) dx
```

by Gauss–Legendre quadrature. The density is smooth and differentiable in every
parameter, so ordinary JAX autodiff handles the backward pass — no custom VJP, no
finite differences. Sixty-four nodes gives roughly fifteen digits on a smooth
integrand.

**The general form: replace an exact-but-non-differentiable special function with
quadrature over a smooth integrand.** It generalizes well past censoring.

Two checks before adopting it:

- **Smoothness across the whole parameter range.** Gauss–Legendre degrades on kinks and
  near-singularities. A Student-t stays $C^\infty$ even at low `nu`; a density with a
  boundary singularity does not.
- **You have changed the likelihood.** Quadrature and an analytic CDF are
  mathematically equivalent and numerically different. Per-observation differences
  around `1e-4` in log space are negligible individually and can accumulate across
  thousands of observations into a posterior shift, especially along a poorly
  identified direction. If you are comparing against a Stan or `brms` fit, this is a
  live candidate explanation for a discrepancy. **This is a modelling change and
  belongs in `DECISIONS.md`.**

## Hoist work out of the quadrature loop

Split the log density into the part that depends on the integration variable and the
part that does not:

```
log f(x) = log_norm(ν, σ) + log_kernel(x, ν, μ, σ)
```

The normalizing constant carries the `gammaln` calls. Computed once per observation
rather than once per observation per node, it removes 63 of every 64 `gammaln`
evaluations — and shrinks the backward pass through `gammaln` and `digamma` by the same
factor. One refactor, large constant.

## The pattern

Three pieces: a pure JAX function, a PyTensor `Op` wrapping it for the CPU path, and a
`jax_funcify` registration so the GPU path uses the JAX version directly. The full
worked version is in `reference/pymc-to-jax.md`; the shape is:

```python
import numpy as np
import jax, jax.numpy as jnp
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
from pytensor.link.jax.dispatch import jax_funcify

def interval_logp(y_l, y_u, nu, mu, sigma):
    ...                                    # pure JAX, fully differentiable

class IntervalLogpOp(Op):
    __props__ = ()                         # lets PyTensor dedupe and cache the node

    def make_node(self, y_l, y_u, nu, mu, sigma):
        inputs = [pt.as_tensor_variable(v) for v in (y_l, y_u, nu, mu, sigma)]
        return Apply(self, inputs, [pt.dvector()])

    def perform(self, node, inputs, outputs):          # CPU / graph-building path
        outputs[0][0] = np.asarray(interval_logp(*map(jnp.asarray, inputs)),
                                   dtype="float64")

    def grad(self, inputs, output_gradients):
        (gz,) = output_gradients
        return [gz * g for g in _interval_logp_grad_op(*inputs)]

@jax_funcify.register(IntervalLogpOp)                  # GPU path
def _(op, **kwargs):
    return interval_logp
```

Then `pm.Potential("censored", IntervalLogpOp()(y_l, y_u, nu, mu, sigma))`.

Four things that go wrong:

- **Omitting `__props__ = ()`** — instances stop comparing equal, so PyTensor cannot
  deduplicate the node and recompiles redundantly.
- **`perform` disagreeing with the JAX version** — gives a model that samples
  differently on CPU and GPU. Genuinely awful to chase; test for it explicitly.
- **Forgetting to register the gradient `Op` with `jax_funcify`** — the backward pass
  falls back off the GPU and you lose everything the port bought.
- **Writing the gradient by hand.** Wrap `jax.grad(fn, argnums=...)` in a companion
  `Op` instead. Hand-derived gradients of a quadrature are a bug source with no upside.

## Verify it, and record that you did

A custom `Op` with a wrong gradient gives a wrong posterior with clean diagnostics.
Nothing downstream will catch it.

1. **Value against a reference** — `scipy`, or a high-precision numerical integral,
   across the full parameter range including tails and boundaries.
2. **Gradient against finite differences** — once, offline, at several parameter
   values. This is the check that catches the otherwise-invisible error.
3. **CPU/GPU agreement** — same point through both paths, to floating-point tolerance.
4. **Prior-only fit** — with the likelihood contributing nothing, the sampler must
   recover the prior.

Write the tolerances you accepted into `wiki/infrastructure/`. When the model disagrees
with something six weeks from now, the first question will be whether the custom `Op`
is at fault, and "we checked it" is not an answer.
