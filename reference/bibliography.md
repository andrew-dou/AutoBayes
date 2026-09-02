# Annotated bibliography

Pointers and summaries. No redistributed PDFs — citations plus what each source
establishes and when it should change what you do.

Every entry below was verified against a publisher record, DOI, or arXiv abstract.
Where a constant or a claim is commonly mis-attributed, the entry says so.

**Tier labels:**

- **[USED]** — part of the source program's written record, with the specific result
  taken from it.
- **[STANDARD]** — the canonical reference for a technique this pack relies on.
  Verified, and recommended; *not* a claim that the source program read it.

---

## Sampler theory

**[USED] Neal, R. M. (2011).** "MCMC Using Hamiltonian Dynamics." In S. Brooks,
A. Gelman, G. L. Jones & X.-L. Meng (Eds.), *Handbook of Markov Chain Monte Carlo*
(Ch. 5, pp. 113–162). Chapman & Hall/CRC. DOI `10.1201/b10905-6`.

The canonical readable HMC reference: leapfrog integration, the energy function,
scaling behaviour. Start here to understand *why* HMC works rather than how to call it.

> Note the DOI suffix is `-6` but this is **chapter 5** — the publisher counts front
> matter as item 1. Do not write "chapter 6".

**[USED] Betancourt, M. (2017).** "A Conceptual Introduction to Hamiltonian Monte
Carlo." arXiv:1701.02434. DOI `10.48550/arXiv.1701.02434`.

The typical-set and differential-geometry intuition — why random-walk Metropolis fails
in high dimensions and what HMC does instead. The most useful thing to read before
debugging a badly behaved sampler, because it converts "the sampler is broken" into a
specific question about geometry. Never journal-published; v2 (2018) is the version
most people read, cited by convention as 2017.

**[USED] Hoffman, M. D. & Gelman, A. (2014).** "The No-U-Turn Sampler: Adaptively
Setting Path Lengths in Hamiltonian Monte Carlo." *Journal of Machine Learning
Research* **15**(47), 1593–1623. Preprint arXiv:1111.4246.

NUTS itself: the U-turn criterion, trajectory doubling, slice sampling for correctness,
dual averaging for step-size adaptation. Read it when you need to know what
`max_treedepth` and `target_accept` actually control. JMLR mints no DOI.

**[USED] Beskos, A., Pillai, N., Roberts, G., Sanz-Serna, J.-M. & Stuart, A. (2013).**
"Optimal tuning of the hybrid Monte Carlo algorithm." *Bernoulli* **19**(5A),
1501–1534. DOI `10.3150/12-BEJ414`. Preprint arXiv:1001.4460.

Source of the rigorous $d^{1/4}$ dimensional-scaling result: to keep an $O(1)$
acceptance probability as $d \to \infty$, the leapfrog step size scales as
$h = l \cdot d^{-1/4}$, so HMC needs $O(d^{1/4})$ steps to cross the state space.

**Three attribution corrections, because this paper is routinely over-cited:**

1. **It does not claim priority for $d^{-1/4}$.** The paper's own introduction credits
   the heuristic argument to Gupta, Irbäck, Karsch & Petersson (1990), *Physics Letters
   B* 242(3–4), 437–443, and the ≈0.7 numerical result to Chen, Qin & Liu (2000).
   Write "proved rigorously by", not "first shown by".
2. **0.651 is theirs; 0.8 is not.** Theorems 4.1–4.2 give an optimal acceptance
   probability of 0.651. The commonly quoted band "0.65–0.8" is **not** a Beskos et al.
   result — see Betancourt, Byrne & Girolami below. Stan's `adapt_delta = 0.8` default
   is a practical choice, not this theorem.
3. **The result is asymptotic and assumes i.i.d. product-form targets.** It is not a
   theorem about hierarchical or funnel-shaped posteriors — i.e. not about the models
   this pack is mostly concerned with. Useful orientation, not a tuning rule for your
   model.

**[STANDARD] Betancourt, M. J., Byrne, S. & Girolami, M. (2014).** "Optimizing The
Integrator Step Size for Hamiltonian Monte Carlo." arXiv:1411.6669.

The actual source of the relaxed acceptance band: the target of 0.651 "can be relaxed
to $0.6 \lesssim a \lesssim 0.9$ with larger values more robust in practice." Cite this,
not Beskos et al., when justifying a `target_accept` choice.

**[STANDARD] Betancourt, M. & Girolami, M. (2015).** "Hamiltonian Monte Carlo for
Hierarchical Models." In S. K. Upadhyay et al. (Eds.), *Current Trends in Bayesian
Methodology with Applications* (Ch. 4, pp. **119–142**). Chapman & Hall/CRC.
DOI `10.1201/b18502-11`. Preprint: arXiv:1312.0906 (2013, v1 only).

The funnel, why it breaks HMC, and why non-centring fixes it — including the part most
often missed, that non-centring is *worse* when the data strongly identify the group
effects. Read before reparameterizing.

> **The page range "79–101" that circulates widely is wrong.** Publisher metadata gives
> 119–142, consistent with the neighbouring chapters (ch. 3 = 99–118, ch. 5 = 143–162)
> and with the publisher's own "24 pages". Also note the chapter is 2015 and the
> preprint is 2013 — they are often conflated into a nonexistent "2015 arXiv" version.

**[STANDARD] Papaspiliopoulos, O., Roberts, G. O. & Sköld, M. (2007).** "A General
Framework for the Parametrization of Hierarchical Models." *Statistical Science*
**22**(1), 59–73. DOI `10.1214/088342307000000014`. Preprint arXiv:0708.3797.

The general centred/non-centred framework, of which the funnel fix is one case.

**[STANDARD] Neal, R. M. (2003).** "Slice sampling (with discussion)." *The Annals of
Statistics* **31**(3), 705–767. DOI `10.1214/aos/1056562461`.

The canonical source for **Neal's funnel**, defined in §8 ("A demonstration"),
pp. 732–733: $v \sim N(0, 3^2)$ and $x_1..x_9 \mid v \sim N(0, e^v)$, a shape that
"resembles a ten-dimensional funnel." Neal never calls it "Neal's funnel" — the eponym
is later usage. Stan's user guide attributes it here.

---

## Diagnostics and validation

**[STANDARD] Vehtari, A., Gelman, A., Simpson, D., Carpenter, B. & Bürkner, P.-C.
(2021).** "Rank-Normalization, Folding, and Localization: An Improved R̂ for Assessing
Convergence of MCMC (with Discussion)." *Bayesian Analysis* **16**(2), 667–718.
DOI `10.1214/20-BA1221`. Preprint arXiv:1903.08008.

Split R-hat, rank-normalized R-hat, and bulk-vs-tail ESS — the diagnostics ArviZ
actually computes. The bulk/tail distinction is the practically important one: fine
bulk ESS with poor tail ESS means the posterior mean is trustworthy and the 95%
interval is not.

**[STANDARD] Gelman, A. & Rubin, D. B. (1992).** "Inference from Iterative Simulation
Using Multiple Sequences." *Statistical Science* **7**(4), 457–472.
DOI `10.1214/ss/1177011136`.

The original R-hat. Cite for provenance; use the 2021 version in practice.

**[STANDARD] Talts, S., Betancourt, M., Simpson, D., Vehtari, A. & Gelman, A. (2018).**
"Validating Bayesian Inference Algorithms with Simulation-Based Calibration."
arXiv:1804.06788.

SBC: fit to data simulated from your own model and check the rank statistics are
uniform. The only routine check that separates an implementation bug from a
specification problem — worth the cost before trusting a custom likelihood. Still
preprint-only; no journal version exists, and claims of a "2020 published version"
refer to arXiv v2.

**[STANDARD] Vehtari, A., Gelman, A. & Gabry, J. (2017).** "Practical Bayesian model
evaluation using leave-one-out cross-validation and WAIC." *Statistics and Computing*
**27**(5), 1413–1432. DOI `10.1007/s11222-016-9696-4`. Preprint arXiv:1507.04544.

PSIS-LOO, and — more useful in practice — the Pareto-$k$ diagnostic that tells you when
the LOO estimate is not to be trusted. A published erratum exists:
DOI `10.1007/s11222-016-9709-3`.

---

## Software

Cite what each project asks to be cited. Several of these have a commonly used but
now-superseded reference.

**[STANDARD] Bürkner, P.-C. (2017).** "brms: An R Package for Bayesian Multilevel
Models Using Stan." *Journal of Statistical Software* **80**(1), 1–28.
DOI `10.18637/jss.v080.i01`.

Needed when porting: documents what the formula syntax expands to — information a port
depends on and which is otherwise only recoverable by reading generated Stan code.

**[STANDARD] Carpenter, B., Gelman, A., Hoffman, M. D., Lee, D., Goodrich, B.,
Betancourt, M., Brubaker, M., Guo, J., Li, P. & Riddell, A. (2017).** "Stan: A
Probabilistic Programming Language." *Journal of Statistical Software* **76**(1), 1–32.
DOI `10.18637/jss.v076.i01`.

**[STANDARD] Abril-Pla, O., Andreani, V., Carroll, C., Dong, L., Fonnesbeck, C. J.,
Kochurov, M., Kumar, R., Lao, J., Luhmann, C. C., Martin, O. A., Osthege, M.,
Vieira, R., Wiecki, T. & Zinkov, R. (2023).** "PyMC: a modern, and comprehensive
probabilistic programming framework in Python." *PeerJ Computer Science* **9**, e1516.
DOI `10.7717/peerj-cs.1516`.

**This, not the 2016 PyMC3 PeerJ paper**, is the current citation. A version-independent
software DOI also exists: `10.5281/zenodo.4603970`.

**[STANDARD] Phan, D., Pradhan, N. & Jankowiak, M. (2019).** "Composable Effects for
Flexible and Accelerated Probabilistic Programming in NumPyro." arXiv:1912.11554.
NeurIPS 2019 Workshop on Program Transformations for Machine Learning.

The GPU sampling backend behind `nuts_sampler="numpyro"`.

**[STANDARD] Cabezas, A., Corenflos, A., Lao, J. & Louf, R. (2024).** "BlackJAX:
Composable Bayesian inference in JAX." arXiv:2402.10797.

The project's README asks for these four lead authors; the arXiv record lists 22, v2
having expanded the list "to include technical contributors." Follow the project's
preference rather than silently truncating.

**[STANDARD] Martin, O. A., Abril-Pla, O., Deklerk, J., Axen, S. D., Carroll, C.,
Hartikainen, A. & Vehtari, A. (2026).** "ArviZ: a modular and flexible library for
exploratory analysis of Bayesian models." *Journal of Open Source Software* **11**(119),
9889. DOI `10.21105/joss.09889`.

**Supersedes** the 2019 JOSS paper (Kumar et al., 4(33), 1143, DOI
`10.21105/joss.01143`) as what the project asks you to cite. The 2019 record remains
valid as the original release paper.

**[STANDARD] Bradbury, J., Frostig, R., Hawkins, P., Johnson, M. J., Katariya, Y.,
Leary, C., Maclaurin, D., Necula, G., Paszke, A., VanderPlas, J., Wanderman-Milne, S. &
Zhang, Q. (2018).** "JAX: composable transformations of Python+NumPy programs."
Software, `github.com/jax-ml/jax`.

Cite as software, with **the version you actually used** — the version pinned in the
project's own README BibTeX is stale. Note the repository is now `jax-ml/jax`, not
`google/jax`. Companion paper: Frostig, Johnson & Leary (2018), "Compiling machine
learning programs via high-level tracing," SysML.

---

## Posts, docs, and code worth reading

Often more immediately useful than papers. Verified live.

**Vieira, R. (2025). "A rosetta stone for PyMC and Stan."**
`gist.github.com/ricardoV94/c421ddacb3ba7a19ac46efa253ea466c`

The best single source for a Stan→PyMC port, by a PyMC core developer. Resolves the
central structural mismatch — Stan's free-form `target +=` versus PyMC's
one-RV-per-distribution graph — and names the two escape hatches you will need for
censoring (`CustomDist`, `Potential`). GitHub's gist viewer fails to render it; use the
Colab link or the raw URL.

**Kurz, A. S. (2025). "Learn Stan with brms," Parts I–III.**
`solomonkurz.netlify.app`

Reverse-engineers `brms::stancode()` output by rebuilding it from scratch. **Part III is
the one that saves you** — it covers brms's automatic mean-centring, the `transformed
data` block, `0 + Intercept` versus `class = Intercept`, and recovering the original
intercept. The series stops before multilevel models.

**Stan User's Guide — "Truncated or Censored Data."**
`mc-stan.org/docs/stan-users-guide/truncation-censoring.html`

The specification to port *against*. Its two-branch structure — impute the censored
values, or integrate them out — maps exactly onto PyMC's imputed-censored model versus
`pm.Censored`.

**PyTensor — "Adding JAX, Numba and Pytorch support for Ops."**
`pytensor.readthedocs.io/en/latest/extending/creating_a_numba_jax_op.html`

The authoritative source for the custom-`Op` path. The rule people get wrong: the
function returned by `jax_funcify` must be **un-jitted**, so the whole graph jits
together.

**Vieira, R. (2022). "How to wrap a JAX function for use in PyMC."**
`pymc.io/projects/examples/en/latest/howto/wrapping_jax_function.html`

The only source that carries a custom `Op` all the way through to NumPyro sampling —
`make_node`/`perform`/`grad`, a companion gradient `Op`, then `jax_funcify`
registration. The template for any bespoke likelihood.

**`pymc-devs/pymc-resources` and `fehiepsi/rethinking-numpyro`.**

PyMC and NumPyro ports of *Statistical Rethinking*. Paired with Kurz's brms translation
of the same book, these give the same models in R/Stan, brms, PyMC, and NumPyro — the
closest thing that exists to a four-way translation table.

**Karpathy, A. "LLM Wiki" gist.**
`gist.github.com/karpathy/442a6bf555914893e9891c11519de94f`

The compiled-wiki pattern the scaffold's `wiki/` is adapted from.

---

## Known documentation gaps

Searched for and not found. These are findings, not omissions — and collectively they
are the argument for this pack existing.

1. **No systematic brms→PyMC mapping reference exists.** Nothing documents "brms
   feature X becomes PyMC construct Y" across the formula language. Bambi
   *reimplements* the formula layer rather than documenting a translation.
2. **brms's generated parameter naming is undocumented.** `b_`, `Intercept` vs
   `b_Intercept`, `r_1_1`, `sd_1`, `z_1`, `L_1` have no authoritative reference. The
   Stan forum thread asking exactly this was answered with "read `rename_pars.R`."
3. **No walkthrough of brms's generated *multilevel* Stan code.** The `z_1`/`L_1`/`r_1_1`
   block and the Cholesky construction for correlated effects are explained nowhere in
   prose. Read `stancode()`.
4. **Nothing documents what brms emits for `cens()` or `trunc()`.**
5. **`pm.CustomDist` documentation says nothing about JAX/NumPyro compatibility.**
   Whether a `logp=`-defined CustomDist samples under `nuts_sampler="numpyro"` is
   undocumented — an empirical question, not a citable one.
6. **Official PyMC GPU setup documentation is stale.** One wiki page from 2023 with
   internally inconsistent CUDA versions, silent on float64, `chain_method`, and memory.

---

## Contributing

Tag the tier, give a DOI or a working URL, and say what the entry establishes and which
decision it should change. An entry that only says a paper is "relevant" costs tokens
and settles nothing. If you quote a constant, name the theorem it comes from.
