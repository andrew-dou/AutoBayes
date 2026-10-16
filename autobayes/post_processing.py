"""brms's post-processing of a fit (brms-parity C07, docs/post-processing.md): the pointwise log-likelihood ``log_lik()``,
the Bayesian R-squared ``bayes_R2()`` and approximate leave-one-out cross-validation by Pareto-smoothed importance sampling
``loo()``, each computed from the posterior draws of a model's fit (a ``FitResult`` or its ``InferenceData``) on the fitted
rows. Accepted against brms 2.23 / loo 2.9.0 on the frozen oracle cases ``gauss_varint_post``, ``poisson_varint_post``,
``gauss_varint_post_psis`` and the other ``*_post`` cases (a finite mixture's: ``mixture_gauss3_post``).

``log_lik(model, fit)`` is brms's ``log_lik()``: the log density of each fitted row's response at each draw, from the family's
``FamilySpec.log_likelihood`` (NumPy, ``families.pointwise``) at ``predict(kind="linear")``'s linear predictor and the draw's
auxiliary parameters; under ``weights()`` it is multiplied by the row's weight, as brms multiplies it.

``bayes_R2(model, fit)`` is brms's ``bayes_R2(summary = FALSE)`` (Gelman, Goodrich, Gabry and Vehtari 2019): per draw,
``var(yhat) / (var(yhat) + var(y - yhat))`` over the fitted rows, ``yhat`` the expected response (``predict(kind="mean")``,
brms's ``posterior_epred``; ``n p`` for a binomial model) and ``y - yhat`` the residuals; brms uses these residuals for every
family, the Gaussian's too (not its sigma).

``loo(model, fit)`` is brms's ``loo()``, which hands ``log_lik()`` to the loo package. What it computes, and where each part
comes from (loo is GPL-3 and none of its source is used here):

- Pareto-smoothed importance sampling (Vehtari, Simpson, Gelman, Yao and Gabry 2024, "Pareto smoothed importance sampling",
  JMLR 25(72)), per observation ``i`` with the ratios ``r_s = 1 / p(y_i | theta_s)``: the ``M = ceil(min(S / 5, 3 sqrt(S /
  r_eff)))`` largest are replaced by the quantiles ``F^-1((z - 1/2) / M)``, ``z = 1..M``, of a generalized Pareto
  distribution fitted to their excess over the ``(M + 1)``-th largest (``gpd_fit``: Zhang and Stephens 2009 with the paper's
  weakly informative prior on ``k``), and no smoothed ratio may exceed the largest raw one. The same steps as ArviZ's
  ``psislw`` (Apache-2.0), whose numbers equal loo's on the probes except in the two cases below, settled against loo's
  outputs on probe inputs: the tail is the ``M`` largest draws in order, ties included (ArviZ keeps only the draws strictly
  above the cutoff), and a tail whose lower quartile ties with the cutoff (a fit that does not exist) leaves, like a tail of
  fewer than five draws, the raw ratios and ``k = Inf``; there is no floor at the smallest positive double.
- ``elpd_loo_i = log sum_s w_s p(y_i | theta_s)`` with the normalized smoothed weights ``w``, ``p_loo_i = lpd_i - elpd_loo_i``
  with ``lpd_i = log mean_s p(y_i | theta_s)``, ``looic_i = -2 elpd_loo_i`` (Vehtari, Gelman and Gabry 2017, "Practical
  Bayesian model evaluation using leave-one-out cross-validation and WAIC", Statistics and Computing 27); the estimates are
  the sums and ``SE = sqrt(N var_i)`` (their eq. 23), ``2 SE(elpd_loo)`` for looic.
- ``n_eff_i = r_eff_i / sum_s w_s^2``, the PSIS effective sample size; ``mcse_elpd_loo_i`` the self-normalized importance
  sampling variance ``sum_s w_s^2 (p_s - E_i)^2 / r_eff_i`` of ``E_i = exp(elpd_loo_i)``, carried to the log scale by a
  log-normal approximation, ``sqrt(log(1 + var / E_i^2))`` (loo 2.7.0's documented change "simplify mcse_elpd using log-normal
  approximation"; the numbers settled by probing). It is computed on the scale of ``E_i``, as ``sum_s (w_s p_s / E_i -
  w_s)^2``, so it stays finite where loo's is NaN: an observation with ``elpd_loo_i`` below about -372, where ``E_i^2``
  underflows (divergence D50).
- ``r_eff``: brms passes loo the relative efficiency of ``exp(log_lik)`` per observation, by chain (probed with a
  multi-chain fixed-parameter fit); here ArviZ's ``ess(method="mean")`` over ``S`` draws, which equals loo 2.9.0's
  ``relative_eff`` to 1e-13 on the probes with 16 and 1000 draws per chain. Chains of fewer than four draws (the one-draw
  chains of the oracle's fixed-parameter fits among them) give ``r_eff = 1``, as loo replaces its NA by 1 (divergence D51 for
  very short or strongly autocorrelated chains).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import logsumexp

POINTWISE_COLUMNS = ("elpd_loo", "mcse_elpd_loo", "p_loo", "looic", "influence_pareto_k")   # brms's loo()$pointwise
ESTIMATE_ROWS = ("elpd_loo", "p_loo", "looic")                                              # brms's loo()$estimates
_MIN_TAIL = 5                    # a Pareto tail of fewer draws is not fitted: raw ratios, k = Inf
_MIN_CHAIN_DRAWS = 4             # chains shorter than this have no relative efficiency: r_eff = 1


def _refuse_measurement(model, what: str) -> None:
    """C06: ``what`` is not verified for a model with ``me()``, ``mi()`` or missing response values (brms's log_lik() of such
    a model leaves out the noisy values' and the imputed values' densities, which no frozen case checks here; divergence D74)."""
    specs = [sub.spec for sub in model.submodels.values()] if getattr(model, "responses", None) else [model.spec]
    if any(getattr(s, "me_terms", ()) or getattr(s, "mi_terms", ()) or getattr(s, "missing_columns", ()) for s in specs):
        raise ValueError(f"{what} of a model with me() or mi() is not offered in this slice (docs/missing-and-measurement-error.md, "
                         "divergence D74)")


def log_lik(model, fit_result) -> xr.DataArray:
    """brms's ``log_lik()``: the log density of each fitted row's response at each posterior draw, ``(chain, draw,
    __obs__)``, times the row's weight under ``weights()``. ``fit_result`` is a ``FitResult`` or its ``InferenceData``
    (checked against the model like ``predict()``'s)."""
    from autobayes.families import family_of
    from autobayes.predict import _with_dpars, predict

    _refuse_measurement(model, "log_lik()")
    fam = family_of(model)
    if fam.log_likelihood is None:
        raise ValueError(f"log_lik() is not offered for the {model.family} family")
    linear = predict(model, fit_result, kind="linear")
    eta = linear["eta"].transpose("chain", "draw", "__obs__", ...).to_numpy()
    post = getattr(fit_result, "idata", fit_result).posterior
    if model.spec.dpars:                                       # each distributional parameter's per-row values (B01)
        post = _with_dpars(model, post, model.data)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        ll = np.asarray(fam.log_likelihood(model, post, eta), dtype=float)
    ll = np.broadcast_to(ll, eta.shape[:3]).copy()
    if model.design.weights is not None:                       # brms multiplies each row's log density by its weight (B02)
        ll = ll * model.design.weights
    if np.isnan(ll).any():
        raise ValueError("log_lik() is NaN for some draws and rows; inspect the posterior draws")
    out = xr.DataArray(ll, dims=("chain", "draw", "__obs__"),
                       coords={"chain": linear["chain"].values, "draw": linear["draw"].values, "__obs__": np.arange(ll.shape[2])},
                       name="log_lik")
    out.attrs["fit_binding_status"] = linear.attrs.get("fit_binding_status", "")
    return out


def bayes_R2(model, fit_result) -> xr.DataArray:
    """brms's ``bayes_R2(summary = FALSE)``: per draw, ``var(yhat) / (var(yhat) + var(y - yhat))`` over the fitted rows, with
    ``yhat`` the expected response (``predict(kind="mean")``; the expected count ``n p`` for a binomial model; for the
    cumulative family the expected category ``sum_k k P(y = k)`` against the 1-based category, as brms computes it),
    ``(chain, draw)``. Refused for the categorical family, as brms refuses it."""
    from autobayes.families import family_of
    from autobayes.predict import predict

    _refuse_measurement(model, "bayes_R2()")
    fam = family_of(model)
    if fam.per_category:
        raise ValueError(f"bayes_R2() is not defined for the {model.family} family's unordered categories (brms: \"'bayes_R2' is "
                         "not defined for unordered categorical models\")")
    ds = predict(model, fit_result, kind="mean")
    if fam.thresholds:                                         # B05: the expected 1-based category against the category
        p = ds[fam.mean_name].transpose("chain", "draw", "__obs__", "category").to_numpy()
        yhat = (p * np.arange(1, p.shape[-1] + 1)).sum(axis=-1)
    else:
        yhat = (ds["expected"] if fam.has_trials else ds[fam.mean_name]).transpose("chain", "draw", "__obs__").to_numpy()
    y = np.asarray(model.design.y, dtype=float)
    var_fit = np.var(yhat, axis=-1, ddof=1)
    var_res = np.var(y - yhat, axis=-1, ddof=1)
    out = xr.DataArray(var_fit / (var_fit + var_res), dims=("chain", "draw"),
                       coords={"chain": ds["chain"].values, "draw": ds["draw"].values}, name="R2")
    out.attrs["fit_binding_status"] = ds.attrs.get("fit_binding_status", "")
    return out


# -- PSIS --------------------------------------------------------------------------------------------------------
def gpd_fit(x) -> tuple[float, float]:
    """The generalized Pareto shape ``k`` and scale ``sigma`` of the exceedances ``x`` (ascending, non-negative), by Zhang and
    Stephens's (2009, Technometrics 51(3)) empirical Bayes estimate with ``30 + floor(sqrt(n))`` grid points and the PSIS
    paper's weakly informative prior on ``k``, ``(n k + 10 * 0.5) / (n + 10)`` (Vehtari et al. 2024): the steps and constants
    of ArviZ's ``_gpdfit`` (Apache-2.0), which the tests compare it with. ``(nan, nan)`` when the estimate does not exist
    (a lower quartile or largest value of zero, or no grid point with a finite weight), where ArviZ's returns the prior's
    ``5 / (n + 10)`` and loo's k-hat is Inf (settled by probing)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    quartile = x[int(n / 4 + 0.5) - 1]
    if not (n > 0 and quartile > 0 and np.isfinite(x[-1])):
        return math.nan, math.nan
    m = 30 + int(math.sqrt(n))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        theta = 1.0 / x[-1] + (1.0 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))) / (3.0 * quartile)
        k = np.log1p(-theta[:, None] * x).mean(axis=1)
        profile = n * (np.log(-theta / k) - k - 1.0)                      # profile log-likelihood of each theta
        weights = 1.0 / np.exp(profile[None, :] - profile[:, None]).sum(axis=1)
    keep = weights >= 10 * np.finfo(float).eps                            # NaN weights are dropped too
    if not keep.any():
        return math.nan, math.nan
    weights = weights[keep] / weights[keep].sum()
    theta_hat = float(np.sum(theta[keep] * weights))
    k_hat = float(np.log1p(-theta_hat * x).mean())
    sigma = -k_hat / theta_hat
    return (n * k_hat + 10 * 0.5) / (n + 10), sigma


def gpd_quantile(p, k: float, sigma: float) -> np.ndarray:
    """The generalized Pareto quantile function at ``p`` in (0, 1): ``sigma (exp(-k log(1 - p)) - 1) / k``, and
    ``-sigma log(1 - p)`` at ``k = 0``."""
    p = np.asarray(p, dtype=float)
    if abs(k) < np.finfo(float).eps:
        return -sigma * np.log1p(-p)
    return sigma * np.expm1(-k * np.log1p(-p)) / k


def psis(log_ratios, r_eff: float = 1.0) -> tuple[np.ndarray, float]:
    """Pareto-smoothed importance sampling of one observation's log importance ratios (``-log_lik`` for LOO), ``(S,)``:
    the normalized smoothed log weights and the Pareto shape estimate ``k`` (``inf`` where the tail was not fitted; module
    docstring)."""
    x = np.asarray(log_ratios, dtype=float)
    S = x.size
    x = x - x.max()
    M = int(math.ceil(min(0.2 * S, 3.0 * math.sqrt(S / r_eff))))
    k = math.inf
    if M >= _MIN_TAIL and M < S:
        order = np.argsort(x, kind="stable")
        tail, cutoff = order[-M:], x[order[-M - 1]]
        with np.errstate(under="ignore"):
            k_fit, sigma = gpd_fit(np.exp(x[tail]) - math.exp(cutoff))
        if np.isfinite(k_fit) and np.isfinite(sigma) and sigma > 0:
            k = k_fit
            with np.errstate(divide="ignore"):
                x[tail] = np.log(gpd_quantile((np.arange(1, M + 1) - 0.5) / M, k, sigma) + math.exp(cutoff))
            x = np.minimum(x, 0.0)                               # no smoothed ratio above the largest raw one
    return x - logsumexp(x), k


def relative_eff(log_lik: xr.DataArray) -> np.ndarray:
    """brms's ``r_eff``: the relative efficiency ``ESS / S`` of each observation's likelihood ``exp(log_lik)`` across the
    chains, with ArviZ's ``ess(method="mean")``; 1 where it does not exist (chains of fewer than four draws, a constant
    likelihood), as loo replaces an NA by 1."""
    import arviz as az

    ll = log_lik.transpose("chain", "draw", "__obs__")
    chains, draws, n = ll.shape
    if draws < _MIN_CHAIN_DRAWS:
        return np.ones(n)
    with np.errstate(under="ignore", invalid="ignore", divide="ignore"):
        lik = xr.Dataset({"lik": (("chain", "draw", "__obs__"), np.exp(ll.to_numpy()))})
        ess = az.ess(lik, method="mean")["lik"].to_numpy()
    r = ess / (chains * draws)
    return np.where(np.isfinite(r) & (r > 0), r, 1.0)


@dataclass(frozen=True)
class Loo:
    """brms's ``loo()`` of a fit (loo 2.9.0's ``psis_loo``): ``pointwise`` one row per fitted row with brms's columns
    ``elpd_loo``, ``mcse_elpd_loo``, ``p_loo``, ``looic`` and ``influence_pareto_k``; ``estimates`` brms's rows ``elpd_loo``,
    ``p_loo``, ``looic`` with columns ``Estimate`` and ``SE``; ``pareto_k`` and ``n_eff`` the diagnostics; ``r_eff`` the
    relative efficiencies used; ``n_draws`` the number of posterior draws ``S``."""

    pointwise: pd.DataFrame
    estimates: pd.DataFrame
    pareto_k: np.ndarray
    n_eff: np.ndarray
    r_eff: np.ndarray
    n_draws: int

    @property
    def elpd_loo(self) -> float:
        return float(self.estimates.loc["elpd_loo", "Estimate"])

    @property
    def se_elpd_loo(self) -> float:
        return float(self.estimates.loc["elpd_loo", "SE"])

    @property
    def p_loo(self) -> float:
        return float(self.estimates.loc["p_loo", "Estimate"])

    @property
    def looic(self) -> float:
        return float(self.estimates.loc["looic", "Estimate"])

    @property
    def pareto_k_threshold(self) -> float:
        """loo's sample-size-dependent threshold ``min(1 - 1 / log10(S), 0.7)``: above it the estimate is not reliable."""
        return min(1.0 - 1.0 / math.log10(self.n_draws), 0.7)

    def __str__(self) -> str:
        bad = int(np.sum(self.pareto_k > self.pareto_k_threshold))
        lines = [f"PSIS-LOO of {len(self.pointwise)} observations from {self.n_draws} posterior draws",
                 self.estimates.to_string(float_format=lambda v: f"{v:.1f}"),
                 f"Pareto k above {self.pareto_k_threshold:.2f} (not reliable): {bad} of {len(self.pareto_k)}"]
        return "\n".join(lines)


def loo_from_log_lik(log_lik: xr.DataArray, r_eff=None) -> Loo:
    """``loo()`` of a pointwise log-likelihood ``(chain, draw, __obs__)`` (``log_lik()``'s output): PSIS-LOO with the
    relative efficiencies ``r_eff`` (a positive number or one per observation; ``None``: ``relative_eff(log_lik)``, brms's
    default; NaN entries become 1, as loo makes them)."""
    ll = log_lik.transpose("chain", "draw", "__obs__")
    chains, draws, n = ll.shape
    S = chains * draws
    if r_eff is None:
        r = relative_eff(ll)
    else:
        r = np.broadcast_to(np.asarray(r_eff, dtype=float), (n,)).copy()
        r[np.isnan(r)] = 1.0
        if not (np.isfinite(r).all() and (r > 0).all()):
            raise ValueError("r_eff must be positive and finite (one number, or one per observation)")
    values = ll.to_numpy().reshape(S, n)
    if not np.isfinite(values).all():
        raise ValueError("loo() needs a finite log-likelihood at every draw and row")
    pointwise = np.empty((n, len(POINTWISE_COLUMNS)))
    n_eff = np.empty(n)
    for i in range(n):
        lli = values[:, i]
        lw, k = psis(-lli, r[i])
        elpd = float(logsumexp(lw + lli))
        lpd = float(logsumexp(lli)) - math.log(S)
        w2 = np.exp(2.0 * lw)
        n_eff[i] = r[i] / w2.sum()
        # sum_s w_s^2 (p_s - E_i)^2 / E_i^2 with E_i = exp(elpd_loo_i), as the squares of w_s p_s / E_i - w_s: both terms in
        # [0, 1] (the w_s p_s / E_i sum to one), so nothing overflows or underflows to NaN where E_i^2 is below the doubles
        terms = np.exp(lw + lli - elpd) - np.exp(lw)
        mcse = math.sqrt(math.log1p(float(np.sum(terms * terms)) / r[i]))
        pointwise[i] = (elpd, mcse, lpd - elpd, -2.0 * elpd, k)
    pw = pd.DataFrame(pointwise, columns=list(POINTWISE_COLUMNS), index=pd.RangeIndex(n, name="row"))

    def se(column):
        return math.sqrt(n * float(np.var(pw[column].to_numpy(), ddof=1))) if n > 1 else math.nan

    estimates = pd.DataFrame({"Estimate": [pw["elpd_loo"].sum(), pw["p_loo"].sum(), pw["looic"].sum()],
                              "SE": [se("elpd_loo"), se("p_loo"), se("looic")]}, index=list(ESTIMATE_ROWS))
    return Loo(pointwise=pw, estimates=estimates, pareto_k=pw["influence_pareto_k"].to_numpy().copy(), n_eff=n_eff, r_eff=r,
               n_draws=S)


def loo(model, fit_result, *, r_eff=None) -> Loo:
    """brms's ``loo()``: PSIS-LOO cross-validation of the fitted rows from ``log_lik(model, fit_result)`` (module docstring).
    ``r_eff``: ``None`` (brms's default) computes each observation's relative efficiency from the chains; a number, or one
    per observation, is used as given (``r_eff=1`` is loo's own default for a log-likelihood matrix)."""
    return loo_from_log_lik(log_lik(model, fit_result), r_eff=r_eff)
