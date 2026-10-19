# `cens()` and `trunc()` beyond the Gaussian family

brms-parity feature E04. B02 and P6 gave the Gaussian family brms's `cens(c)` and `trunc(lb = a, ub = b)`
(docs/addition-terms.md). E04 adds brms's interval censoring `cens(c, y2)`, `cens()` for the Weibull and lognormal families,
and `trunc(ub = b)` for the Poisson and negative binomial families. The Gamma family's `cens()` and a count response's lower
bound are implemented and withheld, each because brms's own numbers on its frozen case are not what an exact
implementation can match (D120, D121).

```python
import autobayes as ab
survival = ab.weibull("ypos | cens(cens_r) ~ x + (1 | g)", data, priors)            # cens_r: 0 observed, 1 right-censored
intervals = ab.lognormal("ypos | cens(cens_i, ypos2) + weights(w) ~ x", data, priors)  # cens_i: -1, 0, 1, 2 (in (ypos, ypos2])
counts = ab.poisson("cnt | trunc(ub = 6) ~ x + (1 | g)", data, priors)                # 0 <= cnt <= 6, both bounds included
```

## What brms does (checked on the frozen points)

- `cens` per row: 0 the log density, 1 the log CCDF (`y` a lower bound), -1 the log CDF (`y` an upper bound), and under
  `cens(c, y2)` 2 the log of `F(y2) - F(y)`: the response lies in `(y, y2]`. brms's standata carries `rcens`, `y2` on the
  interval rows and 0 elsewhere (`Design.rcens`; gate G1 compares it). `posterior_epred` and `posterior_predict` ignore the
  censoring; new rows need neither column. `log_lik()` is each row's censored contribution, times its weight.
- `trunc()` of a count: each row's log mass minus `log P(lb <= Y <= ub)`, both bounds inclusive (`trunc(ub = 6)` subtracts
  `log P(Y <= 6)`; `trunc(lb = 1)` subtracts `log P(Y >= 1)`), and `posterior_epred` of `trunc(ub = b)` is `E[Y | Y <= b]`.
- With `weights()`, each row's whole censored or truncated contribution is weighted (`lognormal_censored_interval_weighted`,
  `poisson_truncated_weighted_post`).

## What is implemented (`autobayes/tails.py`; `FamilySpec.tails`)

Each family's tails are exact on the PyTensor and the JAX path and finite with a finite gradient wherever their value is:

| family | log CDF, log CCDF | interval log mass | truncation log mass |
| --- | --- | --- | --- |
| Weibull | `F = 1 - exp(-exp(u))`, `u = shape (log y - log scale)`: `cloglog_log_p1(u)`, `-exp(u)` | `cloglog_log_mass(u(y), u(y2))` | refused (D122) |
| lognormal | the Normal tails of `log y` (B02) | `normal_log_mass(log y, log y2)` | refused (D122) |
| Gamma (withheld) | `log P(shape, x)`, `log Q(shape, x)`, `x = shape y / mu` | from the tail the interval lies in | refused (D122) |
| Poisson, negative binomial | refused (D122) | refused | `log` of the sum of the masses `0..ub`; a lower bound alone `log1mexp` of those below it |

The Gamma's incomplete gamma functions are PyTensor's values (Cephes) and JAX's (`jax.scipy.special.gammainc`); their
gradient in the shape is this library's own on the PyTensor path (`tails.gammainc_shape_derivative`: the series of `P`
differentiated term by term below `x = shape + 1`, the continued fraction of `Q` differentiated through the Lentz recurrence
above), because PyTensor's is off by up to 0.3% relative (a port of Stan's, stopped at 1e-6). JAX's own derivative is exact.
On the rows a censoring branch does not take, its arguments are a neutral pair (the location and twice it,
`Tails.neutral`), so that no untaken branch meets an infinite derivative. A count response's bounds are integers, at most
1000 (the mass is a sum of that many terms); the upper-bound mass is exact for every mean, the lower-bound one exact for
`lb = 1` (`log(1 - P(0))`) and otherwise to about `1e-16 / P(Y >= lb)` relative.

Predictions: censoring changes none; under `trunc()` `predict(kind="mean")` is the truncated mean (the support's weighted
sum for an upper bound; `mu P(Y >= lb - 1) / P(Y >= lb)` for a Poisson lower bound alone, the shape-plus-one negative binomial
in place of the second `Y` for the negative binomial) and `kind="response"` draws from the truncated distribution by inverting
its CDF (prior-predictive draws too: PyMC's `Truncated` rejection sampler gives up after 10000 steps where the range has
almost no mass).
`ab.log_lik()`, `ab.loo()` and `ab.waic()` read brms's censored and truncated pointwise log likelihood
(`likelihoods.censored_truncated_log_lik`), the Gaussian's included (before E04 its `log_lik()` was the uncensored density,
D123); `ab.bayes_R2()` reads the expected values and the response as brms does. `describe()` writes the interval rows and the
count bounds; the target identity includes `rcens`.

## Withheld (`FamilySpec.parked_additions`, `formula.PARKED_TRUNCATION`)

- **The Gamma family's `cens()`** (D120). With the park lifted, `gamma_censored_interval` passes G1, the G2 log densities,
  G3, G4, G5 and G6 against brms; its gradient in `log(shape)` fails at points 0, 2, 3, 4 and 6, where brms's is off the
  exact (50-digit) gradient by 1.4e-7, 2.8e-5, 2.4e-5, 1.2e-5 and 1.3e-5 relative (point 2: brms -68.5938954721456, exact
  -68.5958155986809, AutoBayes -68.59581559504063). AutoBayes is within 2e-10 of the exact gradient at every point (the rest is
  PyTensor's digamma in the observed rows' density, 1e-10 absolute, as in every Gamma model). Accepting it needs a
  two-reference acceptance of its gradient (P6).
- **A lower bound of a count response** (D121). brms normalises `negbinomial_truncated`'s likelihood by `P(Y >= 1)` (to 4e-16
  on the frozen points) but its `posterior_epred` is `E[Y | Y >= 2]` over a finite support (at point 7, fitted row 27, brms's
  106.236 is the mean over 2..394; `E[Y | Y >= 2]` is 135.97 and `E[Y | Y >= 1]` 133.95). AutoBayes's expected value is the
  mean of the distribution its likelihood truncates, `E[Y | Y >= 1] = mu / (1 - P(0))`. With the park lifted every gate
  passes against brms but G4's expected values, up to 0.499 relative apart. Accepting it needs a two-reference acceptance of
  the expected values (P6).

Refused, naming the construct (D122): `cens()` for the count families and every family other than the four above, `trunc()`
for the Weibull, lognormal, Gamma and every family other than the Gaussian, Poisson and negative binomial, a non-integer or
negative count bound, a count bound above 1000, `cens(c, y2)` with a third argument or with `y2` the codes' column, an interval
row whose `y2` is not above `y`, code 2 without `y2`, and, as before, `cens()` with `trunc()`, per-row bounds, `mi()` with them,
the E06 links (D133), multivariate (D56) and non-linear (D61) formulas, autocorrelation terms (D70) and mixtures.

## Evidence

Oracle cases (G1-G6 against frozen brms fixtures; largest relative differences from brms in the log density, gradient and
expected values): `weibull_censored` (4e-16, 1.1e-9, 4e-16), `lognormal_censored` (6.7e-16, 1.3e-14, 8.1e-16),
`poisson_truncated` (3.4e-16, 2.5e-15, 1.4e-15), and four frozen with this feature: `lognormal_censored_interval_weighted`
(every code, with weights, and `log_lik()` and `loo()`), `poisson_truncated_weighted_post` (with weights, a response at the
bound, and `log_lik()`, `bayes_R2()` and `loo()`), `weibull_censored_interval_post` (every code, `log_lik()`, `bayes_R2()`,
`loo()`; its points on (-0.5, 0.5): on brms's default box brms's own log density is not finite at a seeded point, D120) and
`negbinomial_truncated_ub`. `gamma_censored_interval` and `negbinomial_truncated` stay gated with their parks.
`tests/test_cens_trunc_families.py` checks the shape derivative and every family's tails on both paths against 50- to
120-digit references, the untaken branches at extreme parameters, each model's rows by hand (with weights, formulas on
`sigma` and `shape`, slopes, an offset and `mo()`), the parked cases with their parks lifted against brms and against exact
values, predictions, draws, `log_lik()`, `loo()`, `hypothesis()`, `describe()`, the identity, default priors and every
refusal. Nothing samples.

## With E03's weights() and shape formula

The Weibull family takes `weights()` and a formula on `shape` since E03, and `cens()` since E04. Neither pair has a frozen
brms case; both are checked against SciPy's censored Weibull row by row, all four censoring codes present, to 1e-12
(`tests/test_cens_trunc_families.py::test_weibull_cens_with_weights_and_with_a_shape_formula_equal_scipy`): with `weights()`
each row's term times its weight, with a shape formula a shape per row.
