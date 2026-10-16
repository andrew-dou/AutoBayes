"""Skew-normal response (brms ``skew_normal()``, brms-parity D01, docs/continuous-families.md): real responses, identity link
on the mean ``mu``, standard deviation ``sigma`` and skewness ``alpha``.

brms parameterises Azzalini's skew normal ``SN(xi, omega, alpha)`` by its mean and standard deviation: with
``delta = alpha / sqrt(1 + alpha^2)``, ``omega = sigma / sqrt(1 - 2 delta^2 / pi)`` and ``xi = mu - omega delta sqrt(2 / pi)``,
``E[y] = mu`` and ``SD[y] = sigma`` whatever ``alpha``; ``alpha = 0`` is the Normal. The likelihood is
``likelihoods.skew_normal_family`` (its own density, ``log 2 - log omega + log phi(z) + log Phi(alpha z)``, with a stable
``log Phi``); ``sigma`` takes ``Priors(sigma=...)`` (sampled as ``sigma_log__``) and ``alpha`` ``Priors(alpha=...)``, a
location prior (brms class ``alpha``, default ``normal(0, 4)``; sampled as it is). Checked against brms in the oracle case
``skew_normal_varint``, not assumed. No distributional formula on ``sigma`` or ``alpha`` and no ``weights()``: no frozen case
verifies them (D79).
"""
from __future__ import annotations

import numpy as np

from autobayes.families.spec import (NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, ExtraParameter, FamilySpec, auxiliary,
                                     extra_draws, scale_line)
from autobayes.formula import validate_numeric_response
from autobayes.priors import HalfStudentT, Normal

ALPHA = ExtraParameter(name="alpha", meaning="skewness", support="real", default_prior=lambda: Normal(0, 4))


def _bambi_family():
    from autobayes.likelihoods import skew_normal_family

    return skew_normal_family()


def _validate_response(spec, data) -> None:
    validate_numeric_response(spec.response, data[spec.response], "skew_normal")


def _draw_response(model, post, eta, mean, df, rng):
    """Skew-normal draws with mean ``mu``, SD ``sigma`` and skewness ``alpha`` (``likelihoods.skew_normal_random``)."""
    from autobayes.likelihoods import skew_normal_random

    mu = np.asarray(eta, dtype=float)
    sigma, alpha = auxiliary(model, post, "sigma"), extra_draws(model, post, "alpha")
    return skew_normal_random(mu, np.broadcast_to(sigma, mu.shape), np.broadcast_to(alpha, mu.shape), rng=rng)


def _log_likelihood(model, post, eta):
    """brms's log_lik (C07): the skew-normal log density of ``y_i`` with mean ``eta_i``, sigma and alpha per draw."""
    from autobayes.families.pointwise import skew_normal

    return skew_normal(model.design.y, eta, auxiliary(model, post, "sigma"), extra_draws(model, post, "alpha"))


def _likelihood_lines(model) -> list[str]:
    return [scale_line(model), ALPHA.prior_line(model), "", "Likelihood",
            "  y_i ~ SkewNormal(mean mu_i, SD sigma, skewness alpha): Azzalini's SN(xi_i, omega, alpha) with",
            "  delta = alpha / sqrt(1 + alpha^2), omega = sigma / sqrt(1 - 2 delta^2 / pi), xi_i = mu_i - omega delta sqrt(2 / pi);",
            "  log f(y) = log 2 - log omega + log phi(z) + log Phi(alpha z), z = (y - xi_i) / omega (brms's skew_normal).",
            "  sigma is the SD of y, sampled as log(sigma); alpha = 0 is the Normal; alpha is sampled as it is."]


SKEW_NORMAL = FamilySpec(
    name="skew_normal", tag="skew normal", parent="mu", link="identity", link_scale=None, mean_name="mu",
    response_kind="continuous", response_dtype=None,
    scale_parameter="sigma", bambi_scale_parameter="sigma", fixed_parameters=NO_FIXED_PARAMETERS,
    bambi_family=_bambi_family,
    validate_response=_validate_response,
    response_warnings=lambda spec, df: [],
    response_upper_bound=lambda model: None,
    inverse_link=lambda eta: eta,
    expected_response=None,
    draw_response=_draw_response,
    manual_prediction=True, prediction_kinds=STANDARD_PREDICTION_KINDS,
    kind_error="kind must be 'mean', 'response' or 'linear'",
    scale_rule="sigma is required for the skew_normal family; shape and phi do not apply",
    scale_missing_error="the skew_normal family needs a prior on the standard deviation sigma: pass sigma=...",
    linpred_symbol="mu", scale_phrase="response", intercept_meaning="expected response",
    mean_lines=lambda model: ["  E[y_i] = mu_i, the linear predictor (identity link); SD[y_i] = sigma whatever the skewness alpha."],
    likelihood_lines=_likelihood_lines,
    centred_intercept_note=(),
    deterministic_note="mu = linear predictor, the expected response",
    default_prior_link="identity",                                  # brms reads y as given
    default_scale_prior=lambda scale: HalfStudentT(3, scale),       # brms sigma: student_t(3, 0, scale), lb 0
    extra_parameters=(ALPHA,),                                      # D01: brms's alpha, default normal(0, 4)
    log_likelihood=_log_likelihood,                                 # C07: log_lik(), checked against the model's own density
)
