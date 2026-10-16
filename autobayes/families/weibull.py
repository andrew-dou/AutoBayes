"""Weibull response (brms ``weibull()``, brms-parity D01, docs/continuous-families.md): strictly positive reals, log link on
the mean ``mu``, shape parameter ``shape``.

brms parameterises the Weibull by its mean: ``weibull_lpdf(y | shape, mu / Gamma(1 + 1 / shape))``, the Weibull of shape
``shape`` and scale ``s = mu / Gamma(1 + 1 / shape)``, so ``E[y] = mu`` and ``Var[y] = s^2 Gamma(1 + 2 / shape) - mu^2``.
Bambi's ``weibull`` family is that parameterisation (``beta = mu / gamma(1 + 1 / alpha)`` handed to PyMC's
``Weibull(alpha, beta)``) with the shape named ``alpha``, so the prior is ``Priors(shape=...)`` (brms class ``shape``) and the
sampled variable ``alpha_log__`` (``model.names["shape"]`` is ``"alpha"``), as for the Gamma family. Checked against brms in
the oracle case ``weibull_varint``, not assumed. No distributional formula on ``shape`` and no ``weights()``: no frozen case
verifies them (D79).
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln

from autobayes.families.spec import NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, FamilySpec, auxiliary, exp_mean, scale_line
from autobayes.formula import validate_positive_response
from autobayes.priors import Gamma


def _bambi_family():
    from bambi.defaults import get_builtin_family

    family = get_builtin_family("weibull")             # PyMC's Weibull(alpha, beta = mu / gamma(1 + 1 / alpha)): brms's form
    family.link = {"mu": "log", "alpha": "log"}        # brms's default link on mu is log
    return family


def _validate_response(spec, data) -> None:
    validate_positive_response(spec.response, data[spec.response], "Weibull")


def _inverse_link(eta):
    return exp_mean(eta, "Weibull expected response")


def _draw_response(model, post, eta, mean, df, rng):
    """``Weibull(shape, scale = mu / Gamma(1 + 1 / shape))`` draws: numpy's standard Weibull times the scale."""
    shape = auxiliary(model, post, "shape")
    scale = mean * np.exp(-gammaln(1.0 + 1.0 / shape))
    return scale * rng.weibull(np.broadcast_to(shape, np.shape(mean)))


def _log_likelihood(model, post, eta):
    """brms's log_lik (C07): ``log Weibull(y_i | shape, exp(eta_i) / Gamma(1 + 1 / shape))``, shape per draw."""
    from autobayes.families.pointwise import weibull

    return weibull(model.design.y, eta, auxiliary(model, post, "shape"))


def _likelihood_lines(model) -> list[str]:
    return [scale_line(model), "", "Likelihood",
            "  y_i ~ Weibull(shape, scale = mu_i / Gamma(1 + 1 / shape)); mu_i = exp(eta_i); E[y_i] = mu_i",
            "  brms's parameterisation weibull_lpdf(y | shape, mu / tgamma(1 + 1 / shape)), through PyMC's Weibull with",
            "  alpha = shape, sampled as log(shape).",
            "  Responses are strictly positive finite reals."]


WEIBULL = FamilySpec(
    name="weibull", tag="Weibull", parent="mu", link="log", link_scale="log", mean_name="mu",
    response_kind="continuous", response_dtype=None,
    scale_parameter="shape", bambi_scale_parameter="alpha", fixed_parameters=NO_FIXED_PARAMETERS,
    bambi_family=_bambi_family,
    validate_response=_validate_response,
    response_warnings=lambda spec, df: [],
    response_upper_bound=lambda model: None,
    inverse_link=_inverse_link,
    expected_response=None,
    draw_response=_draw_response,
    manual_prediction=True, prediction_kinds=STANDARD_PREDICTION_KINDS,
    kind_error="kind must be 'mean', 'response' or 'linear'",
    scale_rule="shape is required for the Weibull family; sigma does not apply",
    scale_missing_error="the Weibull family needs a prior on the shape parameter: pass shape=...",
    linpred_symbol="eta", scale_phrase="log mean", intercept_meaning="log expected response",
    mean_lines=lambda model: ["  E[y_i] = mu_i = exp(eta_i); eta is the log expected response (brms's mean parameterisation)."],
    likelihood_lines=_likelihood_lines,
    centred_intercept_note=(),
    deterministic_note="mu = exp(linear predictor); mu = exp(eta), expected response",
    default_prior_link="log",                                       # brms reads log(y) under the log link
    default_scale_prior=lambda scale: Gamma(0.01, 0.01),            # brms shape: gamma(0.01, 0.01), lb 0
    log_likelihood=_log_likelihood,                                 # C07: log_lik(), checked against the model's own density
)
