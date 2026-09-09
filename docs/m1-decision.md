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
$|\eta| \lesssim 15$, loses precision in the $\log(1 - p)$ term as $|\eta|$ grows (total
deviation 6e-3 over 80 observations at $\eta \approx 30$ in the probe), and is floored at
$\log \epsilon = -36.04$ per observation for $|\eta| \gtrsim 37$, where the gradient of the
likelihood term vanishes. Bambi's likelihood registry does not accept a `logit_p`
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
- A5. Extreme logits: at points with $\max|\eta| \le 15$ the deviation from the exact density
  is below 1e-8; at $\eta \approx 40$ the log density and gradient are finite and the
  deviation from the exact density exceeds 1 (documenting the epsilon floor; a Bambi change
  that removes it will fail this test and the record gets updated).
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
