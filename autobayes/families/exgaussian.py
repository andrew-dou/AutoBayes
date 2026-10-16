"""Ex-Gaussian response (brms ``exgaussian()``, brms-parity D01, docs/continuous-families.md): real responses, identity link
on the mean ``mu``, the Normal component's SD ``sigma`` and the exponential component's mean ``beta``.

``y = n + e`` with ``n ~ Normal(mu - beta, sigma)`` and ``e`` an independent exponential of mean ``beta``, so ``E[y] = mu`` and
``Var[y] = sigma^2 + beta^2``: brms's ``exp_mod_normal_lpdf(y | mu - beta, sigma, 1 / beta)``. The likelihood is
``likelihoods.exgaussian_family`` (its own density, with a stable ``log Phi``); ``sigma`` takes ``Priors(sigma=...)`` (sampled
as ``sigma_log__``) and ``beta`` ``Priors(beta=...)``, a positive-support prior (brms class ``beta``, default
``gamma(1, 0.1)``; sampled as ``beta_log__``). Checked against brms in the oracle case ``exgaussian_varint``, not assumed. No
distributional formula on ``sigma`` or ``beta`` and no ``weights()``: no frozen case verifies them (D79).
"""
from __future__ import annotations

import numpy as np

from autobayes.families.spec import (NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, ExtraParameter, FamilySpec, auxiliary,
                                     extra_draws, scale_line)
from autobayes.formula import validate_numeric_response
from autobayes.priors import Gamma, HalfStudentT

BETA = ExtraParameter(name="beta", meaning="mean of the exponential component", support="positive",
                      default_prior=lambda: Gamma(1, 0.1))


def _bambi_family():
    from autobayes.likelihoods import exgaussian_family

    return exgaussian_family()


def _validate_response(spec, data) -> None:
    validate_numeric_response(spec.response, data[spec.response], "exgaussian")


def _draw_response(model, post, eta, mean, df, rng):
    """``Normal(mu - beta, sigma)`` plus an exponential of mean ``beta``, per draw and row."""
    mu = np.asarray(eta, dtype=float)
    sigma, beta = np.broadcast_to(auxiliary(model, post, "sigma"), mu.shape), np.broadcast_to(extra_draws(model, post, "beta"), mu.shape)
    return rng.normal(mu - beta, sigma) + rng.exponential(beta)


def _log_likelihood(model, post, eta):
    """brms's log_lik (C07): the ex-Gaussian log density of ``y_i`` with mean ``eta_i``, sigma and beta per draw."""
    from autobayes.families.pointwise import exgaussian

    return exgaussian(model.design.y, eta, auxiliary(model, post, "sigma"), extra_draws(model, post, "beta"))


def _likelihood_lines(model) -> list[str]:
    return [scale_line(model), BETA.prior_line(model), "", "Likelihood",
            "  y_i ~ ExGaussian(mean mu_i, sigma, beta): Normal(mu_i - beta, sigma) plus an exponential of mean beta;",
            "  E[y_i] = mu_i, Var[y_i] = sigma^2 + beta^2; brms's exp_mod_normal_lpdf(y | mu - beta, sigma, 1 / beta).",
            "  sigma and beta are sampled as their logs."]


EXGAUSSIAN = FamilySpec(
    name="exgaussian", tag="ex-Gaussian", parent="mu", link="identity", link_scale=None, mean_name="mu",
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
    scale_rule="sigma is required for the exgaussian family; shape and phi do not apply",
    scale_missing_error="the exgaussian family needs a prior on the SD of its Normal component sigma: pass sigma=...",
    linpred_symbol="mu", scale_phrase="response", intercept_meaning="expected response",
    mean_lines=lambda model: ["  E[y_i] = mu_i, the linear predictor (identity link); the Normal component's mean is mu_i - beta."],
    likelihood_lines=_likelihood_lines,
    centred_intercept_note=(),
    deterministic_note="mu = linear predictor, the expected response",
    default_prior_link="identity",                                  # brms reads y as given
    default_scale_prior=lambda scale: HalfStudentT(3, scale),       # brms sigma: student_t(3, 0, scale), lb 0
    extra_parameters=(BETA,),                                       # D01: brms's beta, default gamma(1, 0.1)
    log_likelihood=_log_likelihood,                                 # C07: log_lik(), checked against the model's own density
)
