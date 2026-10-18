"""Generalized extreme value response (brms ``gen_extreme_value()``, brms-parity E02, docs/continuous-families.md): real
responses with location ``mu`` (identity link), scale ``sigma`` and shape ``xi``.

The density is brms's (the vignette "Parameterization of Response Distributions in brms"): ``f(y) = t(y)^(xi + 1) exp(-t(y)) /
sigma`` with ``t(y) = (1 + xi (y - mu) / sigma)^(-1 / xi)`` (the Gumbel's ``exp(-(y - mu) / sigma)`` at ``xi = 0``); ``E[y] = mu
+ sigma (Gamma(1 - xi) - 1) / xi`` for ``xi < 1`` (``mu + sigma * 0.5772...``, Euler's constant, at ``xi = 0``), infinite for
``xi >= 1``. brms does not sample ``xi``: it declares the unbounded ``tmp_xi``, puts the prior of class ``xi`` on it, and derives
``xi = lb + (ub - lb) inv_logit(tmp_xi)`` in its model block, with ``lb`` and ``ub`` the smaller and the larger of ``-1 / max(x)``
and ``-1 / min(x)`` for ``x = (y - mu) / sigma`` over the fitted rows, which keeps every fitted row inside the support
(``likelihoods.gen_extreme_value_xi``; found from the frozen case's points, where it reproduces brms's ``posterior_epred(dpar
= "xi")`` to the last bit). So does this library: ``Priors(xi=...)`` is the location prior of ``tmp_xi`` (brms's default
``normal(0, 2.5)``), ``value_point(tmp_xi=...)`` its value, and the graph's deterministic ``xi`` (``model.names["xi"]``) the
derived shape, which a fit's posterior holds and ``predict()``, ``log_lik()`` and the response draws read. ``sigma`` takes
``Priors(sigma=...)``. The likelihood is ``likelihoods.gen_extreme_value_family``. Checked against brms in the oracle case
``gen_extreme_value_varint`` (gate G3 compares the graph's ``xi`` with brms's record), not assumed. brms 2.23 warns that the
family is deprecated. Other links, distributional formulas and the addition terms are refused (D112); ``E[y]`` for ``xi >= 1``
differs from brms's (D114).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln

from autobayes.families.spec import (NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, ExtraParameter, FamilySpec, auxiliary,
                                     extra_draws, scale_line)
from autobayes.formula import validate_finite_response
from autobayes.priors import HalfStudentT, Normal

XI = ExtraParameter(name="xi", meaning="shape", support="real", default_prior=lambda: Normal(0, 2.5), parameter="tmp_xi",
                    divergence="D112")

EULER_GAMMA = 0.57721566490153286061                  # -digamma(1), the Gumbel's mean offset
# zeta(k) for k = 2..12: lgamma(1 - xi) = gamma xi + sum_k zeta(k) xi^k / k for |xi| < 1 (Abramowitz and Stegun 6.1.33)
_ZETA = (math.pi ** 2 / 6, 1.2020569031595942854, math.pi ** 4 / 90, 1.0369277551433699263, math.pi ** 6 / 945,
         1.0083492773819228268, math.pi ** 8 / 9450, 1.0020083928260822144, math.pi ** 10 / 93555, 1.0004941886041194646,
         691 * math.pi ** 12 / 638512875)
_SERIES = 0.01                                        # |xi| below which lgamma(1 - xi) / xi is its series (error < 1e-24)


def gev_mean_offset(xi):
    """``(Gamma(1 - xi) - 1) / xi`` (NumPy), the GEV mean's offset from ``mu`` in units of ``sigma``: Euler's constant at ``xi =
    0``, ``inf`` for ``xi >= 1``. Near 0 it is ``expm1(L) / xi`` with ``L = lgamma(1 - xi)`` summed from its series, so that it is
    exact as ``xi`` passes 0 (``1 - xi`` rounds away ``xi``'s low digits there; brms's ``(gamma(1 - xi) - 1) / xi`` loses them,
    and is NaN at 0)."""
    xi = np.asarray(xi, dtype=float)
    small = np.abs(xi) < _SERIES
    s = np.where(small, xi, 0.0)
    poly = np.zeros(xi.shape)
    for k in range(len(_ZETA) + 1, 1, -1):             # Horner's scheme in xi
        poly = poly * s + _ZETA[k - 2] / k
    ratio = EULER_GAMMA + poly * s                     # L / xi = gamma + sum_k zeta(k) xi^(k - 1) / k
    L = ratio * s
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        near = ratio * np.where(L == 0.0, 1.0, np.expm1(L) / np.where(L == 0.0, 1.0, L))
        lgam = gammaln(np.where(xi < 1.0, 1.0 - xi, 1.0))
        far = np.expm1(lgam) / np.where(small, 1.0, xi)
    return np.where(xi >= 1.0, np.inf, np.where(small, near, far))


def _bambi_family():
    from autobayes.likelihoods import gen_extreme_value_family

    return gen_extreme_value_family()


def _validate_response(spec, data) -> None:
    validate_finite_response(spec.response, data[spec.response], "gen_extreme_value")


def _expected_response(model, post, eta):
    """``mu + sigma (Gamma(1 - xi) - 1) / xi`` per draw and row with the posterior's derived ``xi``; ``inf`` where ``xi >= 1``,
    whose mean is infinite (brms's ``posterior_epred`` evaluates the formula there, a finite number; D114)."""
    return np.asarray(eta, dtype=float) + auxiliary(model, post, "sigma") * gev_mean_offset(extra_draws(model, post, "xi"))


def _draw_response(model, post, eta, mean, df, rng):
    """GEV draws of location ``mu`` (the linear predictor), scale ``sigma`` and the derived ``xi``."""
    from autobayes.likelihoods import gen_extreme_value_random

    mu = np.asarray(eta, dtype=float)
    return gen_extreme_value_random(mu, np.broadcast_to(auxiliary(model, post, "sigma"), mu.shape),
                                    np.broadcast_to(extra_draws(model, post, "xi"), mu.shape), rng=rng)


def _log_likelihood(model, post, eta):
    """brms's log_lik (C07): the GEV log density of ``y_i`` with location ``eta_i``, sigma and the derived xi per draw."""
    from autobayes.families.pointwise import gen_extreme_value

    return gen_extreme_value(model.design.y, eta, auxiliary(model, post, "sigma"), extra_draws(model, post, "xi"))


def _likelihood_lines(model) -> list[str]:
    return [scale_line(model), XI.prior_line(model), "", "Likelihood",
            "  y_i ~ GEV(mu_i, sigma, xi): log f(y) = -log sigma - (1 + xi) w - exp(-w), w = log1p(xi z) / xi, z = (y - mu) / sigma",
            "  (the Gumbel's w = z at xi = 0), brms's gen_extreme_value(). xi = lb + (ub - lb) inv_logit(tmp_xi), lb and ub the",
            "  smaller and the larger of -1 / max(x) and -1 / min(x), x_i = (y_i - mu_i) / sigma over the fitted rows (brms's derivation),",
            "  so 1 + xi x_i > 0 on every fitted row; xi is the graph's deterministic 'xi'. E[y_i] = mu_i + sigma (Gamma(1 - xi) - 1) / xi",
            "  for xi < 1 (infinite for xi >= 1). sigma is sampled as log(sigma), tmp_xi as it is."]


GEN_EXTREME_VALUE = FamilySpec(
    name="gen_extreme_value", tag="generalized extreme value", parent="mu", link="identity", link_scale=None, mean_name="mean",
    response_kind="continuous", response_dtype=None,
    scale_parameter="sigma", bambi_scale_parameter="sigma", fixed_parameters=NO_FIXED_PARAMETERS,
    bambi_family=_bambi_family,
    validate_response=_validate_response,
    response_warnings=lambda spec, df: [],
    response_upper_bound=lambda model: None,
    inverse_link=lambda eta: eta,
    expected_response=_expected_response,
    draw_response=_draw_response,
    manual_prediction=True, prediction_kinds=STANDARD_PREDICTION_KINDS,
    kind_error="kind must be 'mean', 'response' or 'linear'",
    scale_rule="sigma is required for the gen_extreme_value family; shape and phi do not apply",
    scale_missing_error="the gen_extreme_value family needs a prior on the scale sigma: pass sigma=...",
    linpred_symbol="mu", scale_phrase="response (the location)", intercept_meaning="location",
    mean_lines=lambda model: ["  mu_i = the linear predictor (identity link), the location; E[y_i] = mu_i + sigma (Gamma(1 - xi) - 1) / xi "
                              "(brms posterior_epred)."],
    likelihood_lines=_likelihood_lines,
    centred_intercept_note=(),
    deterministic_note="mu = linear predictor, the location; xi = the derived shape; the expected response is computed by predict()",
    default_prior_link="identity",                                  # brms reads y as given
    default_scale_prior=lambda scale: HalfStudentT(3, scale),       # brms sigma: student_t(3, 0, scale), lb 0
    extra_parameters=(XI,),                                         # E02: brms's xi (on tmp_xi), default normal(0, 2.5)
    log_likelihood=_log_likelihood,                                 # C07: log_lik(), checked against the model's own density
)
