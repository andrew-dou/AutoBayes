# The stack, mechanically

What each library in this stack actually is, and the two-to-four internal mechanisms you
need in order to debug it. Not usage documentation — the parts that explain surprising
behaviour.

Verified against official docs, papers, and library source. Where a claim comes from
reading source rather than documentation, it says so. Where behaviour is version-
dependent — and a surprising amount of it is — the version boundary is named.

---

## brms

**What it is.** An R-to-Stan transpiler. It parses an extended `lme4`-style formula into
a symbolic model description, emits a complete Stan program plus a matching data list,
compiles it, and wraps the result in post-processing methods.

**Everything it did is inspectable.** `stancode(fit)` returns the generated Stan program;
`standata(fit)` returns the data list actually passed in (`N`, `Y`, `X`, `Z_*`, `J_*`,
`N_*`, `prior_only`). `make_stancode` / `make_standata` are now aliases of these. If you
disagree with brms about what it fit, diff `stancode(fit)` against your own Stan — that
string is exactly what ran.

**Group-level effects are always non-centred.** brms declares `matrix[M, N] z_id` with a
`std_normal()` prior, plus `vector<lower=0>[M] sd_id` and a Cholesky factor `L_id`, and
forms the effects in transformed parameters as `r_id = scale_r_cor(z_id, sd_id, L_id)`.
There is no centred option. *(From source, `R/stan-predictor.R`.)*

The consequence matters: `sd_*` and `z_*` are sampled, `r_*` is derived. Funnel
pathologies therefore appear as correlation between `sd_*` and `z_*` — **not** in
`r_*`, which is where people look first.

**Distributional parameters carry an inverse link.** `sigma ~ x` becomes its own linear
predictor with the family's inverse link applied before the likelihood. For `sigma` that
is `exp`; for zero-inflation, `logit`. So a `sigma ~ 1` intercept lives on the **log**
scale: `Intercept_sigma = 0` means `sigma = 1`, not `0`, and any prior you set on
`dpar = "sigma"` is a prior on the log scale.

**Unseen grouping levels — read this before trusting any out-of-sample prediction.**
Prediction with `newdata` requires `allow_new_levels = TRUE`; what happens then is
controlled by `sample_new_levels`, and the three options are *not* variations on a
theme.

| Option | What it actually does |
| --- | --- |
| `"uncertainty"` **(default)** | For each posterior draw, picks a random **observed** level and reuses that level's fitted draws. The same donor is used across all coefficients within a draw, to preserve their correlations. |
| `"gaussian"` | Draws from the multivariate normal implied by the fitted `sd_*` and `cor_*` — i.e. the population distribution. Errors on non-Gaussian group-level effects. |
| `"old_levels"` | Picks **one** donor level per new level and copies all of its draws. |

The default is the trap. It resamples *shrunk, partially pooled* posterior draws from a
finite set of observed levels, so it **systematically understates between-level
variance** — and it is worst precisely when few levels were observed, which is when you
most need a new-level prediction to be honest. `"gaussian"` is the option that samples
from the population distribution. `"old_levels"` adds no between-level variance at all;
a new level becomes a literal clone of one observed level, and the answer depends on
which donor the RNG picked. *(From source, `get_new_rdraws()` in
`R/prepare_predictions.R`.)*

Two further consequences of the implementation:

- All three call base R's `sample()`, so **new-level predictions are not reproducible
  across calls without `set.seed()`**.
- Correlations *across different grouping factors* are not preserved — each factor is
  handled in its own loop.

**`NA` in a grouping variable is treated as a new level.** Documented, and silent. A
stray `NA` in a group column quietly activates all of the above rather than erroring.

**`mi()` promotes missing responses to parameters.** Missing entries become a
`vector[Nmi] Ymi` parameter spliced into the response before the likelihood runs. With
`sdy`, the *entire* response becomes latent. This is why discrete variables cannot be
imputed — Stan has no discrete parameters — and why the cost is one extra parameter per
missing cell.

**Censoring is a four-branch conditional** on an integer code, dispatching to `_lpdf`,
`_lccdf` (right), `_lcdf` (left), and `log_diff_exp` of two `_lcdf` calls (interval). A
vectorized fast path exists but is **disabled** when interval censoring, weights, or
truncation are present — vectorized `lpdf` functions return scalars, not vectors. If
your censored model is unexpectedly slow, that is why.

---

## Stan

**What it is.** A statically typed modelling language whose compiler (`stanc3`, in
OCaml) emits a **C++ model class** defining a log density over an *unconstrained* real
vector, differentiated by Stan Math's reverse-mode autodiff and sampled by Stan's NUTS.

Because it emits C++, compile time is C++ compile time and every model change costs a
full recompile.

**Constrained types are transforms plus Jacobians, applied once per leapfrog step.**
Every constrained parameter is transformed to an unconstrained one behind the scenes,
and the log absolute Jacobian determinant of the inverse transform is added to the
accumulated log density.

**Two Jacobian traps, and they are the most common silent-bias bugs in hand-written
Stan:**

- **Constraints on `transformed parameters` do not add a Jacobian.** They are checks
  that convey intent, not transforms. Stan says so explicitly.
- **A nonlinear transform on the left of a `~` needs a manual adjustment.** You write it
  yourself with `jacobian +=`.

**Know which block runs how often.** This is the whole performance model:

| Block | Frequency |
| --- | --- |
| `data` | read once per chain |
| `transformed data` | evaluated once per chain |
| `parameters` | inverse transform + Jacobian, **once per leapfrog step** |
| `transformed parameters` | evaluated **once per leapfrog step**, written once per draw |
| `model` | evaluated **once per leapfrog step** |
| `generated quantities` | evaluated once per draw, no autodiff |

Two rules fall straight out: anything not needed by `target` belongs in `generated
quantities`, and anything parameter-independent belongs in `transformed data`. A large
intermediate parked in `transformed parameters` costs you leapfrog-rate evaluation *and*
per-draw disk.

`generated quantities` runs even when the Metropolis step rejects, because it may
contain random quantities.

**Autodiff is an arena-allocated tape.** The forward pass pushes a node per operation
onto a stack; the reverse sweep walks it backwards calling `chain()`. Peak memory scales
with **the number of autodiff nodes per gradient**, not with parameter count — which is
why a scalar loop over N observations is so much worse than one vectorized `_lpdf` call,
in memory as well as time.

**`~` silently drops constants; `target +=` does not.** Anything needing an absolutely
normalized density — bridge sampling, marginal likelihood, comparing `lp__` across
parameterizations — is wrong if the model used `~`. brms exposes the same switch as
`normalize = FALSE`.

**Version boundary:** the unit simplex transform changed at **2.37**, from stick-breaking
to the softmax of a sum-to-zero vector. Old notes and old unconstrained draws are not
comparable across that line.

---

## PyMC

**What it is.** A front-end that builds a **PyTensor symbolic graph** of `RandomVariable`
Ops, derives a joint log-probability graph from it, and compiles that graph to C, Numba,
or JAX for a sampler to call. PyMC owns almost no numerics of its own.

**Defining a model is graph construction, not computation.** Nothing runs until you ask
for a value.

**The RV/value-variable split is the thing to understand.** Each free random variable is
paired with a *value variable* that is what actually flows through the logp graph, and
constrained variables are automatically transformed. A `HalfNormal` named `s` produces a
value variable `s_log__`, with a registered log transform.

`pm.logp(rv, value)` works on *derived* variables, not only leaf RVs — which is how
`pm.Censored`, `pm.Truncated`, and algebraic transforms get log densities without
hand-written `logp` methods.

**The Jacobian is on by default and it is a real switch.** `compile_logp(...,
jacobian=True)`. Turning it off gives a genuinely different function — appropriate for
MAP, never comparable to what NUTS samples. Do not diff the two numerically.

**`compile_logp` is where the time goes.** Graph rewriting plus backend compilation is
expensive and happens once; the compiled function is then cheap. Slow `pm.sample()`
startup is compilation, not sampling.

**Sampler assignment is automatic and can silently degrade.** NUTS is used where
appropriate; variables it cannot handle get Metropolis or Gibbs variants composed into a
compound step. That is the usual explanation for a model that mixes badly for no visible
reason.

`init` defaults to `jitter+adapt_diag` — a uniform jitter in `[-1, 1]` in
**unconstrained** space, which for a log-scale parameter is a factor of $e^{\pm 1}$.
Models with tight positive-scale priors sometimes fail to initialize for this reason
alone. `cores` defaults to the CPU count **capped at 4**.

---

## PyTensor

**What it is.** A symbolic-array compiler (Theano → Aesara → PyTensor). It builds a
static bipartite DAG of `Variable` and `Apply` nodes, rewrites it through a
priority-ordered optimization database, then hands it to a *linker* that emits C, Numba,
JAX, PyTorch, or MLX.

Debug entry points: `var.owner.op` and `pytensor.dprint(var)`.

**Rewrites run in phases** — canonicalize, stabilize, specialize, then fusion, then
in-place passes. Any rewrite introducing in-place operations must have priority ≥ 50, so
it runs after the destroy handler.

**The JAX mode deliberately disables PyTensor's own fusion.** The source comment is
explicit: fusion is not incompatible with JAX, just undesirable, because **XLA does its
own fusion** and building `Composite` nodes only adds translation overhead.

The consequence is important and widely missed: **the C/Numba and JAX paths run
different rewrite sets.** A small numerical discrepancy between backends is expected
behaviour, not necessarily a bug.

**`jax_funcify` is a `singledispatch` registry** mapping an `Op` to a JAX callable. The
registered function receives the `Op` *instance*, so static attributes are closed over
at trace time, and returns a callable matching `make_node`'s signature.

Two hard constraints, both documented:

- **Only graphs with static shapes can be translated to JAX.**
- Some `Op`s can never be translated — `Eye` is the canonical example, being
  fundamentally a function with dynamic shapes.

When `pm.sample(nuts_sampler="numpyro")` dies with a dispatch error, this registry is
where to look.

**Version boundary:** the default linker is moving from `cvm` to **numba**. Check
`pytensor.config.linker` and `pytensor.compile.mode.get_default_mode()` rather than
assuming which backend a given machine uses.

---

## JAX

**What it is.** Not an array library with extras — a **composable function-transformation
system**. You write a pure function over arrays; `jit` / `grad` / `vmap` / `scan` rewrite
it by tracing it into a `jaxpr`, which XLA compiles to fused device kernels.

**Tracing records operations, not values.** Each argument is wrapped in a tracer. Side
effects happen during the trace but do not appear in the jaxpr — which is why `print()`
fires once, on the first call.

**Traced values can affect control flow only through static attributes** — `shape`,
`dtype` — never through their values. Value-dependent branching raises
`TracerBoolConversionError`. `static_argnums` is the escape hatch, at the price of
recompiling whenever those arguments change.

**Shapes must be static**, so anything whose output size depends on input *values*
(boolean masking, `unique`, dynamic concatenation) is incompatible. The shape/dtype
signature is the compilation cache key: a new shape means a new trace and a new compile.

**`scan` is the loop primitive you want.** Python loops inside `jit` are *unrolled*,
producing enormous XLA computations and long compile times; `scan` compiles to a single
`While`. The carry must hold fixed shape and dtype across iterations.

Critically for likelihoods: **`lax.scan` is fully reverse-mode differentiable, while
`lax.while_loop` and `lax.fori_loop` are forward-mode only.** A state-space likelihood
must be written with `scan` if you want gradients.

**`jit` changes your numerics.** This is documented and worth internalising:

```
jnp.exp(x) / jnp.exp(x)   at x = 100.0
  eager  → nan            (both terms overflow)
  jitted → 1.0            (XLA simplifies to exp(x - x))
```

XLA applies algebraic simplifications whose full list is, by the docs' own admission,
not well documented. So an eager-versus-jitted numerical difference is not automatically
a bug — and neither is it automatically safe.

**`vmap` of a `while_loop` runs in lockstep to the slowest element.** The batched
predicate is reduced with an "any", so every element runs until the last one finishes;
results are masked correctly but the *work* is not saved. Measured: batched iteration
limits of `[10, 10]` took 0.0002 s, `[10, 200000]` took 2.91 s. This is the mechanism
behind vectorized-chain cost below.

---

## NumPyro

**What it is.** Pyro's modelling language — primitives plus effect handlers —
reimplemented on JAX. Its defining engineering trick is an **iterative** NUTS that can be
JIT-compiled end to end, so the whole MCMC loop becomes one XLA program.

**Why iterative NUTS matters.** Standard NUTS builds trajectories with a recursive
`BuildTree`; the Python dispatch overhead per leapfrog step is significant and it blocks
XLA fusion. NumPyro rewrites it iteratively, storing only even-numbered nodes indexed by
bit count to keep memory at $O(\log N)$ rather than $O(N)$.

**Effect handlers are Python-level interceptors that stay transparent to the JAX
tracer**, so they compose with `jit`, `vmap`, and `grad`. Apply them *locally*, within
the function where the model is used — a globally applied handler interacts badly with
the transformations.

**`chain_method` maps onto three JAX transforms**: `"parallel"` → `pmap`, `"sequential"`
→ a Python loop, `"vectorized"` → `vmap`.

**The single biggest gotcha in this whole stack:** with `chain_method="parallel"` and
fewer devices than chains, NumPyro **silently falls back to sequential** with only a
warning. JAX exposes **one CPU device by default**, so the default four chains on a CPU
run *serially* unless you call `numpyro.set_host_device_count(n)` **before any JAX array
is created**. This is the usual explanation for "the JAX backend wasn't faster."

**`"vectorized"` costs the maximum, not the mean.** Vectorized chains share one batched
`while_loop` for tree building, so every chain pays the trajectory length of the *worst*
chain each iteration. One chain stuck at `max_tree_depth = 10` makes all of them do
$2^{10}$ leapfrog steps. That is the statistical problem of tree-depth saturation showing
up as a throughput problem.

`chain_method` is flagged in the docs as experimental and may be removed.

---

## ArviZ

**What it is.** A backend-agnostic container plus diagnostics layer. It normalizes any
sampler's output into labelled, grouped xarray Datasets sharing reserved `chain` and
`draw` dimensions, then computes rank-normalized convergence diagnostics on them.

Groups include `posterior`, `sample_stats`, `log_likelihood`, `posterior_predictive`,
`observed_data`, `prior`, and `predictions`. Note that **`predictions` (out-of-sample) is
deliberately separate from `posterior_predictive` (in-sample)** — new-level forecasts
belong in the former.

**The default `rhat` is `max(bulk R̂, tail R̂)`, both rank-normalized.** Chains are
split in half, rank-normalized with a z-transform, and the folded version (absolute
deviation from the median) gives the tail component. Other methods exist; note that
`method="split"` is the *un*-rank-normalized classic split-R̂ and is **not comparable**
to the default.

**Bulk versus tail ESS is a real distinction.** `method="bulk"` is the rank-normalized
default; `method="tail"` is the minimum of the 5% and 95% quantile ESS. Good bulk ESS
with poor tail ESS means the posterior mean is fine and the interval is not.

**ESS can legitimately exceed the number of draws.** The estimator caps $\hat\tau$ from
below, which ceilings ESS at $n \log_{10} n$ — 14,408 for 4 chains × 1,000 draws. An ESS
above N indicates antithetic behaviour, common for NUTS, and is not a bug; but since it
is also floored by a heuristic, treat very large values as saturated rather than
meaningful.

`rhat` returns `NaN` rather than raising for fewer than 2 chains, fewer than 4 draws, or
non-finite input.

**Version boundary — this one will silently change your figures.** ArviZ 1.0 is a
breaking rewrite splitting into `arviz-base`, `arviz-stats`, and `arviz-plots`.
`InferenceData` is replaced by `xarray.DataTree`. The default credible-interval
probability changes from **0.94 to 0.89**, and the default interval kind from **HDI to
ETI**. WAIC is removed. Any code or caption hard-coding `hdi_3%` / `hdi_97%` or "94%
interval" changes meaning across that boundary — pin the version and state it.

---

## Cross-cutting

**Three Jacobian conventions in one stack.** Stan adds the transform Jacobian
automatically for declared constraints but **not** for `transformed parameters`
constraints or left-hand-side transforms. PyMC adds it automatically and exposes
`jacobian=True/False`. brms never surfaces it and inherits Stan's. Any cross-framework
log-density comparison has to reconcile these before it means anything.

**Two independent "constants dropped" switches.** Stan's `~` versus `target +=`, and
brms's `normalize = FALSE`. Both break bridge sampling and cross-parameterization `lp__`
comparison.

**What actually reaches XLA on the PyMC→JAX path.** PyMC logp graph → PyTensor JAX mode
(rewrites `fast_run` + `jax`, **fusion excluded**) → `jax_funcify` per `Op` → jaxpr →
XLA. Failures at that boundary are almost always an unregistered `Op` or the static-shape
restriction.

**Version drift worth pinning explicitly**, because each of these changes results
silently: Stan's simplex transform at 2.37; PyTensor's default linker moving to Numba;
ArviZ 1.0's container, interval kind, and default probability; brms demoting
`make_stancode` to an alias.
