# Annotated bibliography

Pointers and summaries. No redistributed PDFs — everything here is a citation plus what
it establishes and when it should change what you do.

**Read the tier labels.** They are the point of this file.

- **[USED]** — demonstrably part of the source program's written record. Cited in its
  documentation, with the specific result that was taken from it.
- **[STANDARD]** — the canonical reference for a technique this pack relies on, listed
  because you will need it. It was *not* necessarily read during the source program;
  treat it as a recommendation, not a provenance claim.

Nobody benefits from a bibliography that launders "probably relevant" into "we used
this." Where a constant is quoted from the literature rather than re-derived, that is
stated.

---

## Sampler theory

**[USED] Neal, R. (2011). "MCMC using Hamiltonian Dynamics."** *Handbook of Markov Chain
Monte Carlo.*
The canonical readable HMC reference: leapfrog integration, the energy function,
scaling behaviour. Start here if you want to understand *why* HMC works rather than how
to call it.

**[USED] Betancourt, M. (2017). "A Conceptual Introduction to Hamiltonian Monte Carlo."**
arXiv:1701.02434.
The typical-set and differential-geometry intuition — why random-walk Metropolis fails
in high dimensions and what HMC does instead. The single most useful thing to read
before debugging a badly behaved sampler, because it turns "the sampler is broken" into
a specific question about geometry.

**[USED] Hoffman, M. & Gelman, A. (2014). "The No-U-Turn Sampler."** *JMLR* 15.
NUTS itself: the U-turn criterion, trajectory doubling, slice sampling for correctness,
and dual averaging for step-size adaptation. Read it when you need to know what
`max_treedepth` and `target_accept` actually control.

**[USED] Beskos, A., Pillai, N., Roberts, G., Sanz-Serna, J. & Stuart, A. (2013).
"Optimal tuning of the hybrid Monte Carlo algorithm."** *Bernoulli* 19.
Source of the $d^{1/4}$ dimensional-scaling result and the ≈0.65–0.8 optimal-acceptance
range. **These constants are quoted from this paper, not re-derived** — if your model's
behaviour depends on them, verify against your own setup.

**[STANDARD] Betancourt, M. & Girolami, M. (2015). "Hamiltonian Monte Carlo for
Hierarchical Models."** arXiv:1312.0906.
The funnel, why it breaks HMC, and why non-centring fixes it — including the part most
often missed, that non-centring is *worse* when the data strongly identify the group
effects. Read before reparameterizing.

**[STANDARD] Papaspiliopoulos, O., Roberts, G. & Sköld, M. (2007). "A general framework
for the parametrization of hierarchical models."** *Statistical Science* 22(1).
The general centred/non-centred framework, of which the funnel fix is one case.

---

## Diagnostics and validation

**[STANDARD] Vehtari, A., Gelman, A., Simpson, D., Carpenter, B. & Bürkner, P.-C.
(2021). "Rank-normalization, folding, and localization: An improved $\hat{R}$."*
*Bayesian Analysis* 16(2).
Split R-hat, rank-normalized R-hat, and bulk-vs-tail ESS — i.e. the diagnostics ArviZ
actually computes. The bulk/tail distinction is the practically important one: fine
bulk ESS with poor tail ESS means your posterior mean is trustworthy and your 95%
interval is not.

**[STANDARD] Gelman, A. & Rubin, D. (1992). "Inference from Iterative Simulation Using
Multiple Sequences."** *Statistical Science* 7(4).
The original R-hat. Cite for provenance; use the 2021 version in practice.

**[STANDARD] Talts, S., Betancourt, M., Simpson, D., Vehtari, A. & Gelman, A. (2018).
"Validating Bayesian Inference Algorithms with Simulation-Based Calibration."**
arXiv:1804.06788.
SBC: fit to data simulated from your own model and check the rank statistics are
uniform. The only routine check that separates an implementation bug from a
specification problem. Worth the cost before trusting a custom likelihood.

**[STANDARD] Vehtari, A., Gelman, A. & Gabry, J. (2017). "Practical Bayesian model
evaluation using leave-one-out cross-validation and WAIC."** *Statistics and Computing*
27(5).
PSIS-LOO, and — more useful in practice — the Pareto-$k$ diagnostic that tells you when
the LOO estimate is not to be trusted.

---

## Software

**[STANDARD] Bürkner, P.-C. (2017). "brms: An R Package for Bayesian Multilevel Models
using Stan."** *Journal of Statistical Software* 80(1).
Needed when porting: the paper documents what the formula syntax expands to, which is
the information a port depends on and which is otherwise only recoverable by reading
generated Stan code.

**[STANDARD] Carpenter, B. et al. (2017). "Stan: A Probabilistic Programming
Language."** *Journal of Statistical Software* 76(1).

**[STANDARD] Phan, D., Pradhan, N. & Jankowiak, M. (2019). "Composable Effects for
Flexible and Accelerated Probabilistic Programming in NumPyro."** arXiv:1912.11554.
The GPU sampling backend behind `nuts_sampler="numpyro"`.

**[STANDARD] PyMC.** Cite the current release rather than the 2016 PyMC3 paper — the
API described there differs substantially from PyMC 5.

**[STANDARD] JAX** (Bradbury et al., 2018, software). Cite the repository.

---

## Posts, docs, and code worth reading

Not papers, and often more immediately useful than papers.

**Betancourt, "Diagnosing Biased Inference with Divergences"** (Stan case studies).
The worked example that teaches you to read a pair plot of divergent transitions. If
you read one thing before your first funnel, read this.

**Stan User's Guide — censoring and truncation chapter.**
The four censoring branches written out explicitly. Useful as the specification to port
*against*, since it states what `brms` generates.

**PyTensor documentation — "Creating a new Op" and the JAX dispatch module.**
The reference for `make_node` / `perform` / `grad` and for `jax_funcify`. Thin, and
close to the only documentation of the JAX registration path that exists.

**BlackJAX repository and sampler gallery.**
Where the newer samplers live. Read the source: the implementations are compact and
more informative than the papers for figuring out whether a sampler fits your model.

**NumPyro examples directory.**
Idiomatic patterns for the GPU path, including chain-vectorization usage.

**Karpathy, "LLM Wiki" gist.**
The compiled-wiki pattern the scaffold's `wiki/` is adapted from. Short, and the source
of the raw-sources-immutable / compiled-knowledge-mutable split.

---

## Unresolved

**Microcanonical / MCLMC samplers (Robnik, Seljak, et al.).** The `adjusted_mclmc`
family in BlackJAX, on $O(d^{1/2})$ scaling grounds versus NUTS. The source program
engaged with this and did not complete the comparison; a speedup figure that circulated
informally was never verified there and is deliberately not quoted here. Promising,
unsettled — if you evaluate it against NUTS on a hierarchical model, that result is
worth publishing.

---

## Contributing

If you add an entry, tag its tier and say what it establishes and which decision it
should change. An entry that only says a paper is "relevant" costs tokens and settles
nothing.
