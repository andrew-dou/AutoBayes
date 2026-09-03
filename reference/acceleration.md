# Acceleration

Making Bayesian posterior estimation faster, and — more importantly — knowing which of
the published speedup numbers survive contact with their own papers.

Most do not. That is the main finding of this document, and the reason it exists.

---

## The honest landscape

Same hardware, same metric, on realistic models: **modern samplers beat NUTS by roughly
2×, occasionally 7×.** Not orders of magnitude.

Every large number in circulation turns out to be one of five things: a GPU-versus-CPU
comparison, a lattice-physics comparison with no Bayesian posterior in it, a
best-case toy target, an asymmetric accounting that charges warmup to one side only, or
a precision change reported as a speedup. The audit at the end of this document works
through each.

This matters practically. If you are choosing a sampler, a 2× improvement is worth
having but is not worth a month of integration risk. If you are writing a paper, quoting
a number you have not traced to its table is how you end up defending someone else's
mistake.

---

## Budget in ESS per second

The quantity that matters is
$\operatorname{Var}[\hat\theta] = \operatorname{Var}_p[f] / \mathrm{ESS}_f$.
Once warmup is long enough that variance dominates bias, maximizing **ESS per second**
minimizes time to acceptable error. Wall clock alone is not a measure of anything.

Report two numbers, always:

- **Implementation-dependent** — ESS/second, on named hardware.
- **Implementation-independent** — ESS per gradient evaluation.

A comparison with only the first is unreviewable, because it confounds the algorithm
with the implementation and the machine. A comparison with only the second hides that
the "faster" method may need hardware you do not have. This taxonomy is posteriordb's,
and it is the citation to reach for when a reviewer asks.

**Fix the ESS estimator across every arm of a comparison.** Cross-package ESS numbers
are not comparable, and mixing them silently manufactures speedups. Report bulk ESS and
tail ESS separately, never a single `n_eff`.

> **Pre-2018 Stan comparisons are invalid.** Stan's reported `n_eff` used to be truncated
> at the number of iterations. Because NUTS is antithetic, ESS can legitimately exceed
> N — so the truncation biased results *against* Stan. Discard or re-run any comparison
> from before that fix, and check that your own ESS estimator does not clamp at the draw
> count.

---

## Many chains changes the rules

On a GPU the parallel axis is chains, and running tens of them costs little more than
running one. Two consequences that are not obvious:

**Warmup dominates, and more chains does not shorten it.** The compute is dominated by
the warmup phase, and adding chains does not let you warm up for fewer iterations. Chain
count buys you precision and diagnosis, not a shorter burn-in.

**The R̂ < 1.01 gate is wrong in this regime.** Classical R̂ carries a persistent-variance
term that only shrinks with a long *sampling* phase, so it can sit above 1.01 in the
many-short-chains regime even when squared error is already small. The fix is
**superchains** — chains sharing an initialization point — and **nested R̂**, which for a
typical tolerance gives a threshold near 1.004. Implemented as `posterior::rhat_nested()`.

If you are running 128 chains and gating on classical R̂ < 1.01, you are measuring the
wrong thing and will run longer than you need to.

**How far does chain scaling go?** The defensible measured claim is that GPU throughput
scales roughly linearly into the **tens to low hundreds** of chains — one V100
measurement saw linear scaling to about 50 chains, rolling off around 75–100 — and the
comparison denominator is *one CPU core*. The widely repeated "thousands of chains
almost as fast as one" is not supported by a primary hardware measurement.

---

## Why NUTS is awkward on a GPU

NUTS builds a trajectory whose length is data-dependent and therefore differs per chain.
Vectorized across chains, that becomes a batched `while_loop` running in lockstep: every
chain pays the trajectory length of the slowest one.

The measured cost of ignoring this is real — synchronizing on trajectory boundaries
leaves as much as a factor of 4 on the table even at batch sizes as low as 30.

This is the actual technical argument for **fixed-trajectory samplers on GPU** (ChEES,
MEADS) rather than for more tensorization of your model. Raggedness is a reason to change
sampler, not a reason to avoid `vmap`.

> A caveat from the people who built these systems: raggedness may be secondary to NUTS's
> other overheads — control flow, switching between trajectory endpoints — and the
> JAX/XLA stack has historically been optimized more for TPUs and large neural networks
> than for GPUs running control-flow-heavy code.

---

## The samplers, honestly

| Sampler | Exact? | Real-model gain vs NUTS | Where it actually wins |
| --- | --- | --- | --- |
| **ChEES-HMC** | yes | 1.3–3.0×, median ≈2× | Fixed trajectories → GPU-friendly batching |
| **MEADS** | yes | **loses** asymptotically on several targets | Transient bias; fast convergence, tuning-free |
| **MCLMC** (unadjusted) | **no** | 2.6–12.5×, median ≈7.5× | Speed, if you accept asymptotic bias |
| **MAMS** | yes | **1.6–2.2×** on real targets | Funnel geometry (>42×, NUTS fails outright) |
| **DR-HMC / DR-G-HMC** | yes | no speedup claimed | Multiscale geometry; robustness, not speed |
| **LAPS** | yes | ~100× wall clock, **9× more gradients** | Buying wall-clock with FLOPs you own |
| **Normalizing-flow MCMC** | varies | no NUTS comparison published | Genuinely multimodal targets with barriers |

Reading this table:

- **MEADS "beats NUTS" is not an asymptotic claim.** On grads-per-ESS it loses to plain
  NUTS on several targets. Its authors say so explicitly, and argue that rapid
  convergence matters more than asymptotic efficiency in many-chain workflows. That is a
  defensible position — just not the same claim.
- **Unadjusted MCLMC is biased.** Its headline metric is gradients to reach a fixed error
  threshold, not a limit you can push past by sampling longer. If you need exactness, the
  adjusted variant (MAMS) is the comparison, and it is a ~2× win.
- **LAPS is a trade, not a free lunch.** On stochastic volatility it spends roughly 9×
  more total gradients than NUTS and converts that into ~100× less wall clock through
  4096-way parallelism. The authors state plainly that sequential algorithms use fewer
  gradient calls. It pays only if the GPU is already yours.
- **Flow-based methods target multimodality**, not ill-conditioning. For a badly
  conditioned unimodal hierarchical posterior they are not the tool, and the main
  software package's paper contains no benchmark at all.

**Framing worth internalizing:** NUTS's path-length rule is one choice among many —
randomized HMC, multinomial HMC, NUTS, and apogee-to-apogee sampling are all special
cases of Gibbs-sampling the tuning parameters. Treating NUTS as the natural baseline
rather than as one point in a design space is a habit, not a fact.

---

## Framework realities

**NumPyro's default serializes your chains.** `chain_method` defaults to `'parallel'`,
which uses `pmap`; on a single GPU the device count is 1, so it **falls back to
sequential** with a warning and continues. You must pass `chain_method='vectorized'`.
This is the single highest-impact configuration line in this document.

**PyMC's default sampler has moved to nutpie.** Any statement that "PyMC's default is its
own Python NUTS" is now wrong. State the PyMC version in every baseline you report.

**Stan does not run NUTS on the GPU.** Its OpenCL support accelerates a set of named
density and GLM functions plus matrix primitives, in CmdStan and its wrappers, behind a
build flag. Chains are not GPU-parallel. This is a structural architecture difference,
not an implementation-quality one — and that is the honest way to describe it.

**BlackJAX has a free lever most people miss:** flattening the parameter pytree into a
single array measured 4.94 s versus 6.35 s (~1.3×) on a horseshoe model.

**nutpie's mass-matrix adaptation is the best-benchmarked thing here** — 114 posteriordb
models — at ~1.3× median wall clock versus Stan in default diagonal mode, ~4× in the
opt-in low-rank mode. Note the denominator is post-filtering: unusually slow models and
any model that diverged under any configuration were removed.

---

## Precision: the part that silently corrupts results

**Float32 breaks Metropolis–Hastings at a computable threshold.** The mantissa is 24
bits, so once the absolute log density exceeds $2^{24} = 16{,}777{,}216$, absolute errors
reach 1 — and the accept/reject step is comparing numbers whose difference is smaller
than their error.

**A pre-flight check falls straight out of that.** With $n$ observations at typical
per-observation log-density magnitude $\ell$, if

$$n \cdot |\ell| \gtrsim 1.7 \times 10^{7}$$

you must use float64 or restructure the accept ratio as
$\sum_i [\log p(y_i \mid \theta') - \log p(y_i \mid \theta)]$, which keeps the
differencing inside the summation.

**Detection heuristic:** log-ratios ending in exactly `.0`, `.5`, or `.25` should arouse
suspicion — that is quantization showing through.

A corroborating datapoint: on one HMM benchmark, float32 produced **282 divergences
against Stan's 42**, while float64 produced **zero**, at 1.5× the per-leapfrog cost.
Divergences are the symptom; precision is the cause. This failure mode looks exactly like
bad geometry and will not respond to reparameterization.

**The float64 cost depends entirely on the card, and the range is enormous:**

| Hardware | FP64 : FP32 |
| --- | --- |
| A100 80GB | 1 : 2 |
| H100 SXM | ≈1 : 2 |
| GeForce (Ampere / Ada) | **1 : 64** |

Never state an x64 penalty without naming the GPU class. On a datacentre card it is
about 2×; on a consumer card it can be 64×.

**One more hidden precision loss:** on NVIDIA GPUs, float32 matmuls default to **TF32**,
which has 10 mantissa bits. If your log density is matmul-dominated — a large design
matrix — then "float32" is not float32. Set
`jax.config.update('jax_default_matmul_precision', 'highest')` before trusting a
gradient.

---

## Measuring anything at all

**Every JAX timing must end in `.block_until_ready()`.** Asynchronous dispatch means an
unblocked timer measures the time to *queue* work, not to do it. The documented example
is 269 µs apparent versus ~4.92 ms actual — an 18× fiction. This is the single largest
source of fake JAX benchmarks.

Report compile time and steady-state time as two separate numbers; the first run always
includes compilation. Note that JAX defaults to 32-bit while NumPy defaults to 64-bit, so
a JAX-versus-Stan comparison that does not mention precision is not a comparison.

**Never report a single-run speedup ratio.** With many chains, run time is an order
statistic over chains and is tail-driven; the within-group variance of two samplers
routinely exceeds the difference between their means. Replicate at least ten runs and
report the distribution of per-run ratios with uncertainty.

Avoid headline min-ESS: it is an extreme order statistic in high dimensions.

**Use the right benchmark suite for the claim.** Inference Gym has ground-truth moments,
so it is what you use for claims about *bias*. posteriordb has realistic applied models
with reference posteriors, so it is what you use for claims about practical performance.
They complement rather than substitute.

**A documented case of a speedup evaporating:** one published sampler's advantage was
traced to reporting ESS for *means only* — its ESS for variances was less than half —
combined with taking its tuning parameter from a NUTS pilot run whose cost was never
charged against it. Mean-only ESS plus uncharged tuning is the standard recipe for a
speedup that is not there.

**Debugging slow runs:** enable `jax_log_compiles`. It distinguishes "the sampler is
slow" from "we are recompiling every iteration," which look identical from outside.
Python loops over containers force a separate binary per unique shape.

---

## Claims audit

Every entry here is a number in wide circulation that does not mean what it is used to
mean. Traced to the source in each case.

**"MAMS is 7× faster than NUTS."** The 7× is a 2D Banana and an ill-conditioned Gaussian.
On real-data and hierarchical targets the paper's own table gives 1.60× (German credit),
1.68× (IRT), 1.96× (stochastic volatility), 2.20× (Brownian). The honest sentence: *for a
typical hierarchical model MAMS is a ~2× win with exactness preserved, and it becomes
decisive only on funnel geometry*, where it exceeds 42× because NUTS fails outright.

**"MCLMC is 10× faster than NUTS."** Asymmetric accounting: the large figures divide
MCLMC with **tuning excluded** by NUTS with **warmup included**. The apples-to-apples
comparison in the same paper is 2.6–12.5×, median ≈7.5× — and MCLMC is asymptotically
biased.

**"MCLMC is 12–32× faster."** Wrong baseline entirely. That figure is versus HMC on a 2D
φ⁴ lattice in field theory. There is no NUTS and no Bayesian posterior in that paper.
"MCLMC is 30× faster than NUTS" is a mash-up of two papers' numbers and appears in
neither.

**"ChEES is orders of magnitude faster than NUTS."** Hardware comparison. The actual
sentence compares ChEES *on a GPU* to NUTS *on a 4-core CPU*. Same hardware, same metric,
the range across its seven targets is 1.3–3.0×.

**"LAPS is 100× faster."** True in wall clock, and the authors explain why: it spends ~9×
more gradients and buys the difference with 4096-way parallelism.

**"MEADS beats NUTS."** Not asymptotically — it loses to plain NUTS on grads-per-ESS on
Banana, IRT, and stochastic volatility. Its win is transient bias.

**"DR-HMC is 5× faster than NUTS."** The 5× baseline is *optimally tuned HMC*. NUTS
appears only as a reference and as a qualitative funnel failure case.

**"NumPyro is 100× / 340× faster than Stan."** Both misquotes. The paper says NumPyro is
~340× faster than **Pyro** and **6× faster than Stan** on that benchmark. The "100×"
conflates three separable factors the same table exposes: CPU implementation 4.5×,
precision 2.4× (a numerics choice, not a speedup), GPU 20.6×. And a footnote undercuts the
clean reading — average ESS was Stan 652, NumPyro-float32 556: fewer effective samples
than Stan on the same run length. The software is also from 2019, predating Stan's
`reduce_sum`.

**"nutpie is 2× faster than Stan."** The docs say ~2×; the paper they point to reports
~1.3× median for the default diagonal mode and ~4× for the opt-in low-rank mode. Neither
is 2×.

**"GPU gives 10–50× for MCMC."** The best-controlled measurement is ~11–13× (P100 versus a
28-core Xeon), and its authors disown it as a benchmark: hyperparameters were hand-tuned
with tuning time not counted, and they note the comparison improves with more chains and
degrades with fewer. **No primary source supports 50×.**

**"GPU Gibbs is >100× faster."** The 139× is against single-threaded R on a laptop.
Against the fair baseline — 16-threaded Scala on dual Xeons — the same paper reports ~35×,
in iterations per second rather than ESS per second.

**"vmap chains are 43× slower than pmap."** CPU-only, from a 2022 issue on a 32-core
machine under WSL2. On a single GPU, `pmap` over one device is meaningless, so the vmap
penalty is one you eat. The mechanism is real; the number does not transfer.

**JAX claims the docs do not support:** `vmap` being faster than a Python loop (the page
makes no efficiency claim at all); fusion giving a specific N× (OpenXLA gives no numbers);
`jit` always helping (for single small CPU ops, NumPy often wins on dispatch overhead);
`jit` preserving numerics (there is a documented section titled otherwise); `scan` saving
memory (its documented benefit is *compile time* — `jax.checkpoint` is the memory lever);
and `PREALLOCATE=false` fixing OOM (the docs say it *increases* fragmentation risk — lower
`XLA_PYTHON_CLIENT_MEM_FRACTION` instead).

---

## Citation hazards

Verified traps, worth hard-coding into any bibliography that touches this area.

- **ChEES has no arXiv version.** PMLR 130:3907–3915 is the only primary source.
- **MEADS is not a paper title** — it is an algorithm inside *Tuning-Free Generalized
  Hamiltonian Monte Carlo* (AISTATS 2022, PMLR 151:7799–7813).
- **MCLMC's theory paper is AABI (PMLR 253), not ICLR.**
- **MAMS *is* published** — NeurIPS 2025, DOI `10.52202/085713-0513`. Sources calling it
  preprint-only are wrong.
- **The unadjusted-tuning preprint (arXiv:2412.08876) has had three titles and a growing
  author list.** Quote the version you read. It has no journal-ref, despite secondary
  sources listing one.
- **MALT, Adaptive MALT, NeuTra, LAPS, BlackJAX, TensorFlow Distributions, and the nutpie
  paper are unpublished preprints.** Do not assign venues.
- **The Stan GPU paper is not a JSS article** despite being typeset like one. Cite
  arXiv:1907.01063.
- **flowMC's JOSS paper contains no benchmark.** Any performance number attributed to it
  is fabricated.
- **There is no TFP paper, no JAX paper, and no XLA paper.** Cite the software entries.
- **BlackJAX's author count is genuinely ambiguous** — 4 in the repo's BibTeX, 22 on arXiv
  v2. Pick one and disclose which.
