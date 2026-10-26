"""brms's post-processing of a fit (brms-parity C07, docs/post-processing.md): the pointwise log-likelihood ``log_lik()``,
the Bayesian R-squared ``bayes_R2()`` and approximate leave-one-out cross-validation by Pareto-smoothed importance sampling
``loo()``, each computed from the posterior draws of a model's fit (a ``FitResult`` or its ``InferenceData``) on the fitted
rows. Accepted against brms 2.23 / loo 2.9.0 on the frozen oracle cases ``gauss_varint_post``, ``poisson_varint_post``,
``gauss_varint_post_psis`` and the other ``*_post`` cases (a finite mixture's: ``mixture_gauss3_post``).

``log_lik(model, fit)`` is brms's ``log_lik()``: the log density of each fitted row's response at each draw, from the family's
``FamilySpec.log_likelihood`` (NumPy, ``families.pointwise``) at ``predict(kind="linear")``'s linear predictor and the draw's
auxiliary parameters; under ``weights()`` it is multiplied by the row's weight, as brms multiplies it. With a residual
correlation term (F03b, ``unstr()``, ``cosy()``) it is each row's density given its group's other rows (``timecor.log_lik``),
as brms's ``log_lik()`` of a non-factorised model is. Under ``cens()`` a
censored row's log likelihood is its log CDF, log CCDF or interval log mass, and under ``trunc()`` every row's is less the
log mass of the allowed range (E04, ``likelihoods.censored_truncated_log_lik``), as brms's ``log_lik()`` writes them. A SAR
model's rows are not independent (F02b, docs/spatial-sar.md): its ``log_lik()`` is each row's density given every other row
(``sar.sar_log_lik``), as brms's, and ``loo()`` and ``waic()`` read it; its ``bayes_R2()`` is refused (divergence D143).

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

D05 adds brms's ``waic()``, ``loo_compare()`` and ``model_weights()``, accepted against brms 2.23 / loo 2.9.0 on the frozen
cases ``gauss_loo_compare_x`` and ``gauss_loo_compare_xw`` and against loo's and brms's outputs on recorded probe inputs
(``tests/loo_probes``); no loo or brms source is used:

- ``waic(model, fit)``, the widely applicable information criterion (Watanabe 2010, JMLR 11; the WAIC section of Vehtari,
  Gelman and Gabry 2017): ``lpd_i = log mean_s p(y_i | theta_s)``, ``p_waic_i`` the sample variance (divisor ``S - 1``)
  of ``log p(y_i | theta_s)`` over the draws, ``elpd_waic_i = lpd_i - p_waic_i``, ``waic_i = -2 elpd_waic_i``; estimates
  and SEs as for loo.
- ``loo_compare(results)``: the models best first (the largest ``elpd``; ties keep the order given), each with ``elpd_diff``,
  the sum over observations of its pointwise elpd less the best model's, and ``se_diff = sqrt(N var_i)`` of those pointwise
  differences (Vehtari, Gelman and Gabry 2017, on model comparison), then its own estimates and SEs: loo's eight columns.
- ``model_weights(results, weights)``: brms's ``weights = "loo"`` and ``"waic"``, Akaike weights ``exp(-dIC_k / 2) / sum_j
  exp(-dIC_j / 2)`` of the information criteria, which are ``exp(elpd_k) / sum_j exp(elpd_j)``; and ``"pseudobma"`` with
  ``BB=False``, the pseudo-BMA weights of Yao, Vehtari, Simpson and Gelman (2018, Bayesian Analysis 13(3)), the same formula
  on ``elpd_loo``.

E05 adds the stacking weights of the same paper (its eq. 4, the weights maximising the leave-one-out log score of the
combined predictive distribution, ``sum_i log sum_k w_k exp(elpd_loo_k,i)`` over the simplex), brms's default
``model_weights(weights = "stacking")`` and loo's ``loo_model_weights()``: ``stacking_weights`` computes the exact optimum
(an active-set Newton method on the simplex, to the Karush-Kuhn-Tucker conditions in double precision), where loo's
``stats::constrOptim`` stops at its tolerance. On the frozen case ``gauss_loo_compare_x_stacking`` brms's weights are off the
exact optimum by 1.04e-8, more than the oracle's 1e-9, so the weights are accepted against two references (P6, divergence
D126): brms where brms is exact, and the optimum solved in 50-digit arithmetic (``oracle/reference.py``) where it is not.
``"stacking"`` is ``model_weights``'s default, as it is brms's.
"""
from __future__ import annotations

import hashlib
import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

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
        raise ValueError(f"{what} of a model with me() or mi() is not offered (docs/missing-and-measurement-error.md, "
                         "divergence D74)")


def log_lik(model, fit_result) -> xr.DataArray:
    """brms's ``log_lik()``: the log density of each fitted row's response at each posterior draw, ``(chain, draw,
    __obs__)``, times the row's weight under ``weights()``. ``fit_result`` is a ``FitResult`` or its ``InferenceData``
    (checked against the model like ``predict()``'s)."""
    from autobayes.families import family_of
    from autobayes.predict import _with_dpars, predict

    _refuse_unsupported(model, "log_lik()")                 # me() / mi() (D74) and multivariate models (D95)
    fam = family_of(model)
    if fam.log_likelihood is None:
        raise ValueError(f"log_lik() is not offered for the {model.family} family")
    linear = predict(model, fit_result, kind="linear")
    eta = linear["eta"].transpose("chain", "draw", "__obs__", ...).to_numpy()
    post = getattr(fit_result, "idata", fit_result).posterior
    if model.spec.dpars:                                       # each distributional parameter's per-row values (B01)
        post = _with_dpars(model, post, model.data)
    sar = getattr(model.design, "sar", None)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        if sar is not None:                                    # F02b: each row given every other row
            from autobayes.sar import sar_log_lik
            ll = sar_log_lik(model.design.y, eta, post[model.names["sigma"]].to_numpy(), post[sar.parameter].to_numpy(), sar)
        elif getattr(model.spec, "timecor", None) is not None:   # F03b: each row's density given its group's other rows
            from autobayes.timecor import log_lik as timecor_log_lik
            ll = timecor_log_lik(model, post, eta)
        else:
            ll = np.asarray(fam.log_likelihood(model, post, eta), dtype=float)
    ll = np.broadcast_to(ll, eta.shape[:3]).copy()
    if (model.design.cens is not None or model.design.lb is not None) and "cens" not in fam.own_additions:   # E04: brms's censored /
                                                                                       # truncated log_lik (F05: cox's is its own)
        from autobayes.likelihoods import censored_truncated_log_lik
        ll = np.asarray(censored_truncated_log_lik(model, post, eta, ll), dtype=float)
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
    ``(chain, draw)``. Refused for the categorical family, as brms refuses it. With a residual correlation term (F03b) the
    rows are paired as given, where brms 2.23 pairs the response with ``posterior_epred`` in its rows sorted by group and time
    (divergence D147)."""
    from autobayes.families import family_of
    from autobayes.predict import predict

    _refuse_unsupported(model, "bayes_R2()")                 # me() / mi() (D74) and multivariate models (D95)
    if getattr(model.design, "sar", None) is not None:       # F02b: no frozen case records brms's R2 of a SAR model
        raise ValueError(f"bayes_R2() of a model with the SAR term {model.spec.sar.term!r} is not offered: no frozen brms case verifies "
                         "it (brms's posterior_epred, which it reads, fails for the error model; docs/spatial-sar.md, divergence D143)")
    fam = family_of(model)
    if fam.simplex is not None:                                # F01b: a K-column response on the simplex
        raise ValueError(f"bayes_R2() is not offered for the {model.family} family: its response is K proportions per row, for which "
                         "no frozen case records brms's R2 (docs/dirichlet-logistic-normal.md, divergence "
                         f"{fam.simplex.divergence})")
    if fam.per_category:
        raise ValueError(f"bayes_R2() is not defined for the {model.family} family's unordered categories (brms: \"'bayes_R2' is "
                         "not defined for unordered categorical models\")")
    if fam.hurdle_category is not None:                        # E01: no frozen case records brms's R2 of a hurdle ordinal model
        raise ValueError(f"bayes_R2() of the {model.family} family is not offered: how brms scores the hurdle category against the "
                         "ordered ones is not verified by a frozen case (docs/bounded-families.md, divergence D109)")
    if "mean" not in fam.prediction_kinds:                     # F05: a family without expected values (cox, divergence D154)
        raise ValueError(f"bayes_R2() is not offered for the {model.family} family: it needs the expected response; {fam.kind_error}")
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
    relative efficiencies used; ``n_draws`` the number of posterior draws ``S``; ``response`` a fingerprint of the response
    the model was fit to (D05: ``loo_compare`` warns when they differ, as brms's does), ``None`` for a bare log-likelihood."""

    pointwise: pd.DataFrame
    estimates: pd.DataFrame
    pareto_k: np.ndarray
    n_eff: np.ndarray
    r_eff: np.ndarray
    n_draws: int
    response: str | None = None

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
    return replace(loo_from_log_lik(log_lik(model, fit_result), r_eff=r_eff), response=response_fingerprint(model))


# -- D05: WAIC, model comparison and model weights -----------------------------------------------------------------
WAIC_COLUMNS = ("elpd_waic", "p_waic", "waic")      # brms's waic()$pointwise columns, and its $estimates rows
P_WAIC_LIMIT = 0.4                                  # loo flags p_waic_i above this and recommends PSIS-LOO there
MODEL_WEIGHTS = ("loo", "waic", "pseudobma")        # brms's model_weights(weights = ...) offered here
# E05: weights implemented and withheld, each with the reason its refusal carries: none since the stacking weights were
# accepted against two references (P6; docs/post-processing.md, divergence D126).
PARKED_WEIGHTS = MappingProxyType({})


def response_fingerprint(model) -> str | None:
    """A fingerprint (SHA-256) of the response the model was fit to: its fitted rows' values and, for a binomial model, their
    trials. ``loo_compare`` and ``model_weights`` warn when the models' fingerprints differ, as brms's ``loo_compare`` does
    when the fits' ``yhash`` attributes differ (probed); ``None`` when the model holds no single response."""
    design = getattr(model, "design", None)
    parts = [getattr(design, "y", None), getattr(design, "trials", None)]
    if parts[0] is None:
        return None
    h = hashlib.sha256()
    for part in parts:
        if part is None:
            h.update(b"none;")
            continue
        a = np.asarray(part)
        if a.dtype.kind in "biuf":
            h.update(np.ascontiguousarray(a, dtype="<f8").tobytes())
        else:
            h.update("\x1f".join(map(str, a.ravel().tolist())).encode("utf-8"))
        h.update(f";{a.shape};".encode())
    return h.hexdigest()


def _refuse_unsupported(model, what: str) -> None:
    """What ``log_lik()`` does not take here is refused naming ``what``: a model with ``me()`` or ``mi()`` (D74), a
    multivariate model (C01), whose log-likelihood brms sums over the responses or keeps per response, and a model with
    ``fcor()`` (F03a, D145)."""
    _refuse_measurement(model, what)
    if getattr(model, "responses", None):
        raise ValueError(f"{what} of a multivariate model is not offered (docs/post-processing.md, divergence D95)")
    term = getattr(getattr(model, "spec", None), "fcor", None)
    if term is not None:                                     # F03a: brms's log_lik conditions each row on all the others
        raise ValueError(f"{what} of a model with {term.term} is not offered: brms's log_lik() of the joint normal is each row's "
                         "density given every other row, which no frozen case checks here (docs/fcor.md, divergence D145)")


def _estimates(pw: pd.DataFrame, rows) -> pd.DataFrame:
    """loo's estimates table: each column's sum and ``SE = sqrt(N var)`` (NaN for a single observation, loo's NA)."""
    n = len(pw)

    def se(column):
        return math.sqrt(n * float(np.var(pw[column].to_numpy(), ddof=1))) if n > 1 else math.nan

    return pd.DataFrame({"Estimate": [float(pw[r].sum()) for r in rows], "SE": [se(r) for r in rows]}, index=list(rows))


@dataclass(frozen=True)
class Waic:
    """brms's ``waic()`` of a fit (loo 2.9.0's ``waic``): ``pointwise`` one row per fitted row with brms's columns
    ``elpd_waic``, ``p_waic``, ``waic``; ``estimates`` brms's rows ``elpd_waic``, ``p_waic``, ``waic`` with columns ``Estimate``
    and ``SE``; ``n_draws`` the number of posterior draws ``S``; ``response`` as ``Loo.response``."""

    pointwise: pd.DataFrame
    estimates: pd.DataFrame
    n_draws: int
    response: str | None = None

    @property
    def elpd_waic(self) -> float:
        return float(self.estimates.loc["elpd_waic", "Estimate"])

    @property
    def se_elpd_waic(self) -> float:
        return float(self.estimates.loc["elpd_waic", "SE"])

    @property
    def p_waic(self) -> float:
        return float(self.estimates.loc["p_waic", "Estimate"])

    @property
    def waic(self) -> float:
        return float(self.estimates.loc["waic", "Estimate"])

    def __str__(self) -> str:
        high = int(np.sum(self.pointwise["p_waic"].to_numpy() > P_WAIC_LIMIT))
        lines = [f"WAIC of {len(self.pointwise)} observations from {self.n_draws} posterior draws",
                 self.estimates.to_string(float_format=lambda v: f"{v:.1f}")]
        if high:
            lines.append(f"p_waic above {P_WAIC_LIMIT}: {high} of {len(self.pointwise)} (loo recommends PSIS-LOO there)")
        return "\n".join(lines)


def waic_from_log_lik(log_lik: xr.DataArray) -> Waic:
    """``waic()`` of a pointwise log-likelihood ``(chain, draw, __obs__)`` (``log_lik()``'s output; module docstring)."""
    ll = log_lik.transpose("chain", "draw", "__obs__")
    chains, draws, n = ll.shape
    S = chains * draws
    if S < 2:
        raise ValueError("waic() needs at least two posterior draws: p_waic is the variance of the log-likelihood across them")
    values = ll.to_numpy().reshape(S, n)
    if not np.isfinite(values).all():
        raise ValueError("waic() needs a finite log-likelihood at every draw and row")
    lpd = logsumexp(values, axis=0) - math.log(S)
    p_waic = np.var(values, axis=0, ddof=1)
    elpd = lpd - p_waic
    pw = pd.DataFrame({"elpd_waic": elpd, "p_waic": p_waic, "waic": -2.0 * elpd}, index=pd.RangeIndex(n, name="row"))
    return Waic(pointwise=pw, estimates=_estimates(pw, WAIC_COLUMNS), n_draws=S)


def waic(model, fit_result) -> Waic:
    """brms's ``waic()``: the widely applicable information criterion of the fitted rows from ``log_lik(model, fit_result)``
    (module docstring)."""
    _refuse_unsupported(model, "waic()")
    return replace(waic_from_log_lik(log_lik(model, fit_result)), response=response_fingerprint(model))


def _criterion_results(results, what: str) -> tuple[list[str], list, str]:
    """The names, results and criterion (``"loo"`` or ``"waic"``) of ``loo_compare``'s and ``model_weights``' argument: a
    mapping ``{name: result}`` or a sequence of results, named ``model1``, ``model2``, ... as loo names them; at least two,
    all ``Loo`` or all ``Waic``, on the same number of observations."""
    if isinstance(results, (Loo, Waic)):
        raise ValueError(f"{what} needs at least two models: pass a mapping {{name: result}} or a list of results")
    if isinstance(results, Mapping):
        names, items = [str(k) for k in results], list(results.values())
    else:
        items = list(results)
        names = [f"model{i}" for i in range(1, len(items) + 1)]
    if len(items) < 2:
        raise ValueError(f"{what} needs at least two models (loo: \"'loo_compare' requires at least two models\")")
    bad = [name for name, r in zip(names, items) if not isinstance(r, (Loo, Waic))]
    if bad:
        raise ValueError(f"{what} takes the results of ab.loo() or ab.waic(); {bad} are not (compute each model's criterion first)")
    kinds = {type(r) for r in items}
    if len(kinds) > 1:
        raise ValueError(f"{what} of PSIS-LOO and WAIC results together is refused: compare like with like (divergence D93)")
    counts = {name: len(r.pointwise) for name, r in zip(names, items)}
    if len(set(counts.values())) > 1:
        raise ValueError(f"{what}: all models must have the same number of observations, but they have {counts}")
    if next(iter(counts.values())) < 2:
        raise ValueError(f"{what} needs at least two observations: se_diff is a variance across them")
    fingerprints = {r.response for r in items}
    if None not in fingerprints and len(fingerprints) > 1:
        warnings.warn(f"{what}: not all models have the same response (their fingerprints differ); the comparison is likely "
                      "invalid", UserWarning, stacklevel=3)
    return names, items, ("loo" if kinds == {Loo} else "waic")


def loo_compare(results) -> pd.DataFrame:
    """brms's (and loo 2.9.0's) ``loo_compare()`` of several models' ``ab.loo()`` or ``ab.waic()`` results, given as a mapping
    ``{name: result}`` (or a sequence, named ``model1``, ``model2``, ... as loo names them): one row per model, best first
    (the largest ``elpd``; ties keep the order given), with loo's columns ``elpd_diff``, ``se_diff``, ``elpd_<c>``,
    ``se_elpd_<c>``, ``p_<c>``, ``se_p_<c>``, ``<ic>``, ``se_<ic>`` (``<c>`` ``loo`` or ``waic``, ``<ic>`` ``looic`` or
    ``waic``); ``elpd_diff`` and ``se_diff`` are those of each model's pointwise elpd less the best model's (module
    docstring). Results of both criteria together, models on different numbers of observations and a single model are
    refused; models fit to different responses are compared with a warning, as brms compares them."""
    names, items, crit = _criterion_results(results, "loo_compare()")
    ic = "looic" if crit == "loo" else "waic"
    elpd = f"elpd_{crit}"
    pointwise = np.column_stack([r.pointwise[elpd].to_numpy() for r in items])
    totals = np.array([r.estimates.loc[elpd, "Estimate"] for r in items], dtype=float)
    order = np.argsort(-totals, kind="stable")
    best = pointwise[:, order[0]]
    n = pointwise.shape[0]
    rows = []
    for k in order:
        diff = pointwise[:, k] - best
        est = items[k].estimates
        rows.append([float(diff.sum()), math.sqrt(n * float(np.var(diff, ddof=1))),
                     est.loc[elpd, "Estimate"], est.loc[elpd, "SE"], est.loc[f"p_{crit}", "Estimate"], est.loc[f"p_{crit}", "SE"],
                     est.loc[ic, "Estimate"], est.loc[ic, "SE"]])
    columns = ["elpd_diff", "se_diff", elpd, f"se_{elpd}", f"p_{crit}", f"se_p_{crit}", ic, f"se_{ic}"]
    return pd.DataFrame(rows, columns=columns, index=pd.Index([names[k] for k in order], name="model"), dtype=float)


_KKT_TOL = 1e-14                 # stacking: the relative KKT residual at which the Newton iterations stop


def stacking_weights(lpd) -> np.ndarray:
    """The stacking weights of Yao, Vehtari, Simpson and Gelman (2018, eq. 4) of a ``(N, K)`` matrix of pointwise elpd (one
    column per model, loo's ``lpd_point``): the exact maximiser over the simplex of ``F(w) = sum_i log sum_k w_k
    exp(lpd_ik)``. ``F`` is concave; its gradient is ``g_k = sum_i a_ik / s_i`` (``a_ik = exp(lpd_ik - max_k lpd_ik)``, ``s =
    a w``), and since ``sum_k w_k g_k = N`` the optimum is where ``g_k = N`` for every model with a positive weight and ``g_k
    <= N`` for the others (the Karush-Kuhn-Tucker conditions). An active-set Newton method reaches it: Newton steps on the face
    of the simplex spanned by the models with positive weight (``F``'s Hessian restricted to the face, a backtracking line
    search, a model leaving the face when its weight reaches 0), then the model with the largest ``g_k > N`` joins the face,
    until no model does; the iterations stop at a relative residual of ``1e-14`` (or where no step improves ``F`` in double
    precision). Where the optimum is not unique (models with proportional ``exp(lpd)`` columns), one of the optima is
    returned."""
    lpd = np.asarray(lpd, dtype=float)
    if lpd.ndim != 2 or lpd.shape[0] < 1 or lpd.shape[1] < 1:
        raise ValueError(f"stacking needs a (observations, models) matrix of pointwise elpd, got shape {lpd.shape}")
    if not np.isfinite(lpd).all():
        raise ValueError("stacking needs finite pointwise elpd")
    N, K = lpd.shape
    if K == 1:
        return np.ones(1)
    a = np.exp(lpd - lpd.max(axis=1, keepdims=True))

    def objective(w):
        s = a @ w
        return -math.inf if (s <= 0).any() else math.fsum(np.log(s))

    w = np.full(K, 1.0 / K)
    active = np.ones(K, dtype=bool)

    def longest(d):                                   # the step to the face's boundary along d, and the weight that reaches it
        ratios = np.full(K, np.inf)
        down = (d < 0) & active
        ratios[down] = -w[down] / d[down]
        k = int(np.argmin(ratios))
        return min(1.0, float(ratios[k])), (k if ratios[k] <= 1.0 else None)

    for _ in range(100 * K):
        s = a @ w
        g = a.T @ (1.0 / s)
        idx = np.flatnonzero(active)
        residual = float(np.max(np.abs(g[idx] - N))) / N
        if residual <= _KKT_TOL or len(idx) == 1:
            joining = np.flatnonzero(~active & (g > N * (1.0 + _KKT_TOL)))
            if not len(joining):
                break
            active[joining[np.argmax(g[joining])]] = True
            idx = np.flatnonzero(active)
        B = a[:, idx] / s[:, None]
        m = len(idx)
        Z = np.vstack([np.eye(m - 1), -np.ones((1, m - 1))])            # directions within the face: weights summing to 0
        BZ = B @ Z
        u = np.linalg.lstsq(BZ.T @ BZ, Z.T @ g[idx], rcond=None)[0]     # Newton: -Z'HZ u = Z'g with H = -B'B
        d = np.zeros(K)
        d[idx] = Z @ u
        if not (d @ g > 0):                                               # no ascent (a degenerate face): the projected gradient
            d[idx] = g[idx] - g[idx].mean()
        t_max, hit = longest(d)
        if t_max <= 0.0:                                                  # a weight at 0 that the step would make negative
            d[idx] = g[idx] - g[idx].mean()
            t_max, hit = longest(d)
        f0, t = objective(w), t_max
        while t > 1e-30 and not objective(w + t * d) >= f0:
            t /= 2.0
        if t <= 1e-30:
            break                                                         # no step improves F: as exact as double precision allows
        w = w + t * d
        if hit is not None and t == t_max:                                # that weight reaches 0 and leaves the face
            w[hit] = 0.0
            active[hit] = False
        w = np.maximum(w, 0.0)
        w /= w.sum()
    else:
        raise RuntimeError("stacking: the active-set Newton iterations did not converge")
    w[~active] = 0.0
    return w / w.sum()


def loo_model_weights(results) -> pd.Series:
    """loo's ``loo_model_weights()`` with its default ``method = "stacking"`` of several models' ``ab.loo()`` results (as
    ``model_weights`` takes them), which brms's ``model_weights(weights = "stacking")`` is: ``model_weights(results,
    "stacking")``, the exact optimum where loo's optimiser stops near it (divergence D126)."""
    return model_weights(results, "stacking")


def model_weights(results, weights: str = "stacking", *, BB: bool | None = None) -> pd.Series:
    """brms's ``model_weights()`` of several models' results (a mapping ``{name: result}`` or a sequence, as ``loo_compare``
    takes them), in the order given: ``weights="loo"`` (``ab.loo()`` results) and ``weights="waic"`` (``ab.waic()``
    results) are brms's Akaike weights of looic and WAIC, ``exp(elpd_k) / sum_j exp(elpd_j)``; ``weights="pseudobma"`` with
    ``BB=False`` the pseudo-BMA weights of ``ab.loo()`` results, the same numbers (module docstring); ``weights="stacking"``
    (``ab.loo()`` results; E05), the default as in brms, the exact stacking weights of their pointwise ``elpd_loo``
    (``stacking_weights``; divergence D126). ``"bma"``, ``"kfold"`` and pseudo-BMA with the Bayesian bootstrap (brms's default
    ``BB = TRUE``, which draws random numbers) are refused (divergence D92)."""
    if weights in PARKED_WEIGHTS:
        raise ValueError(f"model_weights(weights={weights!r}) is withheld: {PARKED_WEIGHTS[weights]}")
    if weights in ("bma", "kfold"):
        raise ValueError(f"model_weights(weights={weights!r}) is not offered (divergence D92): brms's "
                         + {"bma": "Bayesian model averaging needs the marginal likelihood (bridge sampling)",
                            "kfold": "K-fold cross-validation refits the model"}[weights]
                         + f"; offered: {', '.join(MODEL_WEIGHTS)}")
    if weights == "stacking":                                             # E05 (D126)
        if BB is not None:
            raise ValueError("BB applies to weights='pseudobma' only")
        names, items, crit = _criterion_results(results, "model_weights()")
        if crit != "loo":
            raise ValueError("model_weights(weights='stacking') takes the results of ab.loo(), not of ab.waic()")
        w = stacking_weights(np.column_stack([r.pointwise["elpd_loo"].to_numpy() for r in items]))
        return pd.Series(w, index=pd.Index(names, name="model"), name="stacking")
    if weights not in MODEL_WEIGHTS:
        raise ValueError(f"model_weights(weights=...) must be one of {MODEL_WEIGHTS}, got {weights!r}")
    if weights == "pseudobma" and BB is not False:
        raise ValueError("model_weights(weights='pseudobma') needs BB=False: pseudo-BMA with the Bayesian bootstrap (brms's "
                         "default BB = TRUE) draws random numbers (divergence D92)")
    if weights != "pseudobma" and BB is not None:
        raise ValueError("BB applies to weights='pseudobma' only")
    names, items, crit = _criterion_results(results, "model_weights()")
    if crit != ("waic" if weights == "waic" else "loo"):
        raise ValueError(f"model_weights(weights={weights!r}) takes the results of "
                         f"{'ab.waic()' if weights == 'waic' else 'ab.loo()'}, not of {'ab.loo()' if crit == 'loo' else 'ab.waic()'}")
    elpd = np.array([r.estimates.loc[f"elpd_{crit}", "Estimate"] for r in items], dtype=float)
    w = np.exp(elpd - elpd.max())
    return pd.Series(w / w.sum(), index=pd.Index(names, name="model"), name=weights)
