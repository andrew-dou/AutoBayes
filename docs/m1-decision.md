# M1 decision record: Bernoulli varying-intercept model, logit link

Date: 2026-09-09. Branch `m1/bernoulli-varying-intercept`, from `main` at `925554b`. Written
before implementation. Scope is the one family Andrew authorised: a binary outcome with
observations nested in groups, `y ~ x + (1 | group)`, Bernoulli likelihood, logit link.
Not in scope: binomial trials, other links, varying slopes, a generic family layer.

## Mathematical target

For observations $i = 1..n$ with numeric predictors $x_{ip}$, a grouping factor with
sorted levels $j = 0..J-1$ and $g_i$ the level index of observation $i$:

$$\eta_i = a_c + \sum_p (x_{ip} - \bar x_p)\, b_p + s_g\, z_{g_i}, \qquad
y_i \sim \mathrm{Bernoulli}\big(\pi_i\big),\ \pi_i = \mathrm{expit}(\eta_i) = \frac{1}{1 + e^{-\eta_i}}$$

$$z_j \sim \mathrm{Normal}(0, 1),\quad a_c \sim \text{prior}_\text{intercept},\quad
b_p \sim \text{prior}_{b,p},\quad s_g \sim \text{prior}_\text{sd},$$

with $y_i \in \{0, 1\}$ strictly and $\pi_i = P(y_i = 1)$. There is no residual scale.
Predictors are centred by default, as in brms and Bambi, so the prior on the intercept is a
prior on $a_c$, the log-odds of $y = 1$ at the predictor means; the reported `Intercept` is
$a = a_c - \sum_p \bar x_p b_p$, back-transformed after sampling. The group effects are
non-centred. All priors are explicit and live on the logit scale for $a_c$ and $b_p$.

Sampled value space: `Intercept` ($a_c$), one variable per slope, `1|g_sigma_log__`
($\log s_g$, Jacobian $+\log s_g$ in the sampled density), `1|g_offset` ($z$). Log density:

$$\log p = \sum_i \big[y_i \log\mathrm{expit}(\eta_i) + (1 - y_i)\log\mathrm{expit}(-\eta_i)\big]
+ \sum_j \log\phi(z_j) + \log p(a_c) + \sum_p \log p(b_p) + \log p(s_g) + \log s_g .$$

## What Bambi supplies, checked against bambi 0.17.2 source and docs

- `family="bernoulli"`: link `logit` on parameter `p`; a numeric response must be all 0 and
  1 (Bambi raises otherwise) and success is `1`; a boolean response is accepted as 0/1; a
  string response takes the *first sorted level* as success unless `y['level']` is given.
  M1 requires a numeric 0/1 response, so the string trap cannot occur.
- Centring, non-centred group effects, and the prior dictionary work as in M0; the model's
  parent parameter is `p` (a deterministic `p` replaces M0's `mu`), there is no `sigma`.
- `predict(kind="response_params")` returns draws of `p`; `predict(kind="response")` returns
  0/1 draws from the posterior predictive; `sample_new_groups=True` is brms's default
  `sample_new_levels="uncertainty"` (a randomly chosen existing level per draw). brms's
  `"gaussian"` option (draw the new level's effect from the fitted normal) is not offered
  by Bambi and is implemented in this layer, on the logit scale, as `new_groups="population"`.
- brms semantics for the same model: `bernoulli(link = "logit")` is the default link;
  `posterior_epred` returns probabilities, `posterior_predict` returns 0/1 draws with the
  Bernoulli variability included; the intercept prior applies to the centred intercept.

**Known deviation, measured before implementation.** Bambi's backend inverse link computes
`p = sigmoid(eta)` and replaces an exact `0` or `1` by machine epsilon (bambi
`backend/links.py`); PyMC's `Bernoulli(p=...)` then evaluates `log(p)` / `log1p(-p)`.
The sampled density therefore equals the exact Bernoulli-logit density to about 1e-8 for
$\eta \lesssim 15$ and for every negative $\eta$ down to underflow (measured: exact at
$\eta = -40$ and $-300$); for large positive $\eta$ the $y = 0$ term $\log(1 - p)$ loses
precision (total deviation 6e-3 over 80 observations at $\eta \approx 30$ in the probe) and
above $\eta \approx 37$, where $\mathrm{sigmoid}(\eta)$ rounds to 1, it is floored at
$\log \epsilon = -36.04$ per $y = 0$ observation, where the gradient of that term vanishes.
The asymmetry is because $\mathrm{sigmoid}(\eta)$ underflows to 0 only near $\eta = -745$. Bambi's likelihood registry does not accept a `logit_p`
parameterisation, so this cannot be changed without replacing the likelihood. Decision:
reuse the built-in family, state this in `describe()`, pin the regime with tests (A5), and
record the fallback: if a real dataset shows separation (posterior mass at $|\eta| > 20$),
the family is rebuilt by hand with `pm.Bernoulli(logit_p=eta)` on Bambi's design matrices.
For a model with weakly informative logit-scale priors this regime carries negligible mass.

## What the layer adds (delta from M0)

1. `ab.bernoulli(formula, data, priors)`; `Priors` without `sigma` (giving one is an error
   for this family; omitting it is an error for Gaussian). Same formula subset.
2. Response validation: numeric, values exactly in {0, 1}, both values present; booleans
   and strings refused with a message saying how to code them.
3. `describe()` states the link, that the intercept prior is on the log-odds at the
   predictor means, the deterministic `p`, and the epsilon deviation above.
4. `predict(kind="mean")` returns `p` (probabilities); `predict(kind="response")` returns
   `p` and 0/1 draws `y`. New-group semantics as in M0, with the population draw taken on
   the logit scale.
5. `fit_record` carries the family and link; `compare_records` refuses to compare records of
   different families.

Gaussian behaviour is unchanged; the M0 suite must pass untouched.

## Acceptance checks (set before implementation)

Log density and meaning (CPU tests):

- A1. Joint log density (Jacobian included) equals an independent NumPy/SciPy reference
  (written from the equations above, no PyMC/Bambi imports) at five random points with
  $|\eta_i| < 15$, to 1e-8, for centred and uncentred forms, one and two predictors, and an
  intercept-only model.
- A2. Same with `jacobian=False`; the difference is exactly $\log s_g$.
- A3. `compile_dlogp` equals central finite differences of the reference, rtol 1e-5.
- A4. `get_jaxified_logp` equals the PyTensor path to 1e-8 at the same points.
- A5. Extreme logits: at $\eta = \pm 12$, $-40$ and $-300$ the deviation from the exact
  density is below 1e-8; at $\eta = +40$ and $+300$ the log density and gradient are finite
  and the deviation equals $n_0 (\eta - 36.04)$ within 1 % for the $n_0$ zero outcomes
  (documenting the epsilon floor; a Bambi change that removes it will fail this test and
  the record gets updated).
- A6. Design and coding as in M0; `p` from the graph equals $\mathrm{expit}$ of the hand-computed
  $\eta$ to 1e-12.
- A7. Invalid inputs raise errors naming the problem: response values outside {0, 1}, a
  boolean or string response, a response that is all 0 or all 1, a `sigma` prior for this
  family, a missing `sd`, a Gaussian model without `sigma`, and every M0 unsupported formula.
- A8. Predictions: `kind="mean"` gives `p` with shape (chain, draw, n), values in (0, 1),
  equal to $\mathrm{expit}(a + x b + u[g])$ from the trace to 1e-10; `kind="response"` gives
  `y` in {0, 1} whose mean over draws equals the mean of `p` within 4 binomial standard
  errors per observation and 0.01 pooled; observed levels use their effect; unseen levels
  are refused by default; `"population"` draws a fresh logit-scale effect with spread
  consistent with the posterior of $s_g$; `"resample_observed"` reproduces Bambi's draws.

Sampling (CPU, modest size; binary outcomes are less informative than Gaussian ones, so
$n$ is larger):

- B1. Simulated recovery, $n = 600$, $J = 30$, truth $a = -0.5$, $b = 0.8$, $s_g = 0.7$,
  4 chains, 1000 tune, 2000 draws, seed 7, on both the PyMC and NumPyro CPU paths. Panel:
  zero divergences, max R-hat ≤ 1.01, bulk and tail ESS > 400 total. Truth inside the 94 %
  HDI for `Intercept`, `x`, `1|g_sigma`. The two backends' posterior means within 4
  combined MCSE. One recovery is not a calibration claim and will not be reported as one.
  There is no exact-posterior check for this family (no conjugate form); A1-A5 carry the
  correctness burden, B1 the sampler-path burden.

Device:

- C1. `device="gpu"` on this family raises the same `DeviceError` when no GPU is visible;
  C2 the CPU NumPyro report reads `float64`, `parallel`. C3: GPU execution for this family
  is **pending** (not run in this task) and is labelled so.

## Kill criterion

If A1 fails at moderate logits after accounting for centring, the Jacobian and the epsilon
replacement, the Bambi-built likelihood is not the target density and the family is built
by hand with `pm.Bernoulli(logit_p=...)`; Bambi is kept for the design matrices only.

## If a check fails

Repair and re-run; record the failure in `docs/coordination/STATUS.md`. Thresholds are not
loosened. If B1 fails on one backend only, that is a sampler-path bug to investigate before
any claim about either backend.

## Repair record (2026-09-09): the likelihood at large positive logits was wrong, not negligible

**Finding (coordinator review, reproduced).** In the model as merged into the PR, with two
zero and two one observations at $\eta = 40$ the likelihood is $-72.087$ instead of $-80$,
and at $\eta = 300$ it is still $-72.087$ instead of $-600$; the likelihood's contribution
to the intercept gradient vanishes there. This is a parameter-dependent change of the target
density, not a precision footnote, and the earlier A5 acceptance condition asserted the
erroneous deviation instead of refusing it. The statements "negligible under weakly
informative priors" and "watch for separation" are withdrawn: a sampler that can reach the
region samples a different posterior there.

**Constraints found while repairing (all reproduced on this stack, pymc 5.28.5, pytensor
2.38.3, jax 0.10.2).** `pm.Bernoulli(logit_p=eta)` has the correct log density but
`compile_dlogp` returns NaN at $\eta = 40$ and $300$. A `pt.softplus` likelihood lowers to a
branching form whose JAX gradient fails at magnitude 1000. The construction must therefore
be verified on both backends, for values and gradients, rather than chosen by API name.

**Repair acceptance criteria (written before the change).**

- R1. For the actual `ab.bernoulli` model (priors, centring, non-centred coding unchanged),
  `compile_logp` equals the independent stable reference (SciPy `log_expit`) at logits
  $\eta \in \{0, \pm 0.5, \pm 12, \pm 40, \pm 300, \pm 1000, \pm 10^6\}$, for both outcomes,
  to 1e-8 absolute or 1e-12 relative. The reviewer's case (two zeros, two ones) gives
  $-80$ at $\eta = 40$ and $-600$ at $\eta = 300$ for the likelihood term.
- R2. `compile_dlogp` and `jax.grad` of `get_jaxified_logp` equal the analytic gradient
  ($y_i - \mathrm{expit}(\eta_i)$ through the design, plus prior terms) at the same points
  to 1e-8, finite everywhere, on both the PyTensor C path and the JAX path.
- R3. A1-A4 and A6-A8 pass unchanged (moderate regime, design, predictions), and the
  Gaussian suite is untouched.
- R4. The posterior predictive still draws $y \sim \mathrm{Bernoulli}(\mathrm{expit}(\eta))$;
  `predict` semantics and `p` are unchanged.
- R5. B1 is re-run on both backends with the same prespecified thresholds and the evidence
  regenerated; the old A5 assertion of the deviation is deleted, not relaxed.
- R6. `describe()` states the likelihood form that is actually sampled.

**Executed failure history (2026-09-09, this stack).** Two zero and two one observations,
likelihood term only, reference $-80$ at $\eta = 40$, $-600$ at $\eta = 300$, $-2000$ at
$\pm 1000$, gradient $\sum_i (y_i - \mathrm{expit}(\eta))$:

| Construction | PyTensor value / gradient | JAX value / gradient |
| --- | --- | --- |
| Bambi built-in (`p = sigmoid`, eps substitution), as in the PR before repair | $-72.087$ / $0$ at $\eta = 40, 300, \pm 1000, \pm 10^6$ | same |
| `pm.Bernoulli(logit_p=eta)` | exact / **NaN** at $\eta = 40, 300, 1000, 10^6$ | exact / **NaN** at $\pm 1000, \pm 10^6$ |
| `CustomDist` with `y*eta - pt.softplus(eta)` | exact / exact everywhere | exact / **NaN** at $\pm 1000, \pm 10^6$ |
| `CustomDist` with `(x+\lvert x\rvert)/2 + \log1p(e^{-\lvert x\rvert})` written out | exact / exact | exact / **NaN** at $\pm 1000, \pm 10^6$ (PyTensor rewrites it back to `Softplus`) |
| **`CustomDist` with `y*eta - stable_softplus(eta)`** (Softplus subclass, JAX lowering `jax.nn.softplus`) | exact / exact | exact / exact |

Mechanism, from the installed source: `pytensor/link/jax/dispatch/scalar.py` lowers
`Softplus` to nested `jnp.where` branches, one of which is `x + exp(-x)`; at $x \le -710$
that branch overflows and JAX's `where` gradient multiplies its untaken-branch cotangent
by the overflowed derivative, giving NaN. The C path evaluates only the taken branch. The
rewrite `log1p(exp(x)) -> softplus(x)` (`pytensor/tensor/rewriting/math.py`) applies to any
hand-written stable form, so a distinct op is the smallest construction that survives
lowering. The chosen construction is `autobayes/likelihoods.py`; it is verified on the
actual `ab.bernoulli` model (priors, centring, non-centred coding unchanged) at
$\eta \in \{0, \pm 0.5, \pm 12, \pm 40, \pm 300, \pm 1000, \pm 10^6\}$ with both outcomes present,
values and full gradients, PyTensor and JAX (`tests/test_bernoulli.py::test_extreme_logits_density_and_gradient_both_paths`,
`test_reviewer_case_two_zeros_two_ones`).

Consequences for the rest of the record: the "known deviation" paragraph above and the old
A5 are superseded by R1-R6; the `describe()` note now states the sampled likelihood form;
Bambi's built-in `bernoulli` family is no longer used for this model, Bambi's design
matrices, priors, centring, group coding, `predict` machinery and `sample_new_groups`
are. The response-parameter deterministic is now `logit_p` (identity link on the linear
predictor); `p = expit(logit_p)` is computed by `predict`.

**Open review items, not addressed by this repair (for the coordinator).**

1. `new_groups="resample_observed"` inherits Bambi's `sample_new_groups=True`, which draws
   one observed level per posterior draw for *all* unseen levels together, so two distinct
   unseen labels in `newdata` receive the same effect within a draw. `"population"` draws
   an independent effect per unseen label and is unaffected.
2. `validate_data` casts the response with `to_numpy(dtype=float)`, so a complex-valued
   response such as `[0+4j, 1+4j]` passes the 0/1 check after its imaginary part is
   discarded (NumPy warns). A dtype check before the cast would refuse it.

## Follow-up unit (2026-09-09): unseen-label identity in `resample_observed`, complex responses

Written before editing. Scope: the two open review items above, nothing else.

**Intended semantics of `new_groups="resample_observed"`** (brms `sample_new_levels =
"uncertainty"`): for each posterior draw and each *distinct* unseen label in `newdata`,
one observed level is chosen uniformly at random, independently across distinct labels
and across draws, and that draw's fitted effect $u_{\text{chosen}}$ is used. Every row
carrying the same unseen label shares that label's chosen effect within a draw (one
unit, one effect). Two distinct unseen labels may coincide by chance, with probability
$1/J$ per draw, but are never deterministically equal. Observed labels keep their own
effects. The rest of the prediction is unchanged: $\eta = a + x b + u$, Gaussian
$\mu = \eta$, $y \sim \mathrm{Normal}(\mu, \sigma)$; Bernoulli $p = \mathrm{expit}(\eta)$,
$y \sim \mathrm{Bernoulli}(p)$. This is computed in this layer, as `"population"` already
is, instead of delegating to Bambi's `sample_new_groups=True`, which maps every unseen
label to one placeholder group and therefore gives all unseen labels the same effect in
every draw. In-sample predictions and `newdata` with only observed labels still go
through Bambi.

**Intended validation.** A complex-valued column is refused before any cast, for the
response of both families, for predictors, and for `newdata` predictors, with a message
saying so; NumPy's "casting complex values to real discards the imaginary part" warning
must not be the mechanism that reports it. Real integer and float columns are accepted
as before.

**Acceptance criteria.**

- U1 (distinct identity): with a synthetic posterior whose observed effects are distinct
  constants across draws, two distinct unseen labels coincide in a fraction of draws
  near $1/J$ (within Monte Carlo tolerance), never in all draws; each label's effect in
  each draw is one of that draw's observed effects. Under the previous behaviour the
  fraction is 1, so the test separates the two.
- U2 (repeated label): rows with the same unseen label have identical effects in every
  draw, for both families.
- U3 (observed untouched): observed labels in the same `newdata` use their own effects
  exactly, as before.
- U4 (both families, both kinds): Gaussian `mu` and Bernoulli `p` equal the hand-computed
  linear predictor with the chosen effects; `y` draws keep their family's meaning
  (Gaussian residual SD, Bernoulli 0/1).
- U5 (`population` and in-sample unchanged): the existing A8 and M0 prediction tests pass
  unchanged; `seed` makes the result reproducible.
- U6 (validation): complex response (both families), complex predictor, and complex
  `newdata` predictor are refused with a message naming the column and "complex", and no
  NumPy complex-cast warning is emitted; real inputs still pass.
- U7 (regression): the fast suite and the slow CPU sampling suite pass.

**Executed results (2026-09-09).** Before the change, on both families with a real fit
(2 chains x 300 draws) and `newdata` labels `[lvl00, NEW_A, NEW_B, NEW_A]`: NEW_A and NEW_B
identical in 100 % of draws (the collapse); complex responses built with the imaginary
part discarded under a NumPy warning. After the change, J = 6: NEW_A and NEW_B identical in
17.5 % of draws against 1/J = 16.7 %, repeated NEW_A rows identical in 100 % of draws,
complex response/predictor/`newdata` refused with "is complex-valued" before any cast.
Implementation: `autobayes/predict.py::_predict_by_hand` (both `"population"` and
`"resample_observed"` when `newdata` has unseen labels; in-sample and observed-only
`newdata` still through Bambi), `autobayes/formula.py::reject_complex`. Tests:
`tests/test_predict.py` (synthetic posterior with distinct constant effects: U1-U4 exact,
seed reproducibility; real fit: not collapsed), `tests/test_bernoulli.py` (same on `p`
and 0/1 draws; complex refusal for response, predictor, `newdata`, with
`ComplexWarning` turned into an error so a cast cannot be the mechanism),
`tests/test_formula.py` (Gaussian complex refusal, real float32/int accepted).
Suite results are in `docs/coordination/STATUS.md`.

Remaining limitations of this unit: `"resample_observed"` draws levels uniformly, as
brms does; it does not weight by group size. A label that is unseen but present in the
training data under a different spelling is a new label, as before. No change to
`"population"` or to in-sample predictions.
