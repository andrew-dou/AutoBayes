"""Shifted lognormal response (brms ``shifted_lognormal()``, brms-parity D01, docs/continuous-families.md): strictly positive
reals, identity link on the log-scale location ``mu``, scale ``sigma`` and shift ``ndt`` (the non-decision time).

``log(y - ndt) ~ Normal(mu, sigma)``, brms's ``lognormal_lpdf(y - ndt | mu, sigma)``: ``E[y] = exp(mu + sigma^2 / 2) + ndt``.
``ndt`` is brms's ``real<lower=0,upper=min_Y>`` with ``min_Y`` the smallest response, sampled on Stan's lower-upper coordinate
(PyMC's ``ndt_interval__``) under ``Priors(ndt=...)`` truncated to (0, min y) and renormalised there as brms does
(``Prior.to_bambi_bounded``); brms's default is ``uniform(0, min_Y)``, ``ab.Uniform(0, "min_Y")``. The likelihood is
``likelihoods.shifted_lognormal_family``; ``sigma`` takes ``Priors(sigma=...)``. ``predict(kind="mean")`` returns brms's
``posterior_epred`` ``exp(mu + sigma^2 / 2) + ndt`` as ``mean``; ``kind="linear"`` returns ``mu``. Checked against brms in the
oracle case ``shifted_lognormal_varint``, not assumed. No distributional formula on ``sigma`` or ``ndt`` and no ``weights()``:
no frozen case verifies them (D79).
"""
from __future__ import annotations

import numpy as np

from autobayes.families.spec import (NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, ExtraParameter, FamilySpec, auxiliary,
                                     exp_mean, extra_draws, scale_line)
from autobayes.formula import validate_positive_response
from autobayes.priors import MIN_Y, HalfStudentT, Uniform

NDT = ExtraParameter(name="ndt", meaning="non-decision time", support="below_min_y", default_prior=lambda: Uniform(0, MIN_Y))


def _bambi_family():
    from autobayes.likelihoods import shifted_lognormal_family

    return shifted_lognormal_family()


def _validate_response(spec, data) -> None:
    validate_positive_response(spec.response, data[spec.response], "shifted_lognormal")


def _expected_response(model, post, eta):
    """brms's ``posterior_epred`` ``exp(mu + sigma^2 / 2) + ndt`` per draw, refused rather than clipped when it overflows."""
    return exp_mean(eta + 0.5 * auxiliary(model, post, "sigma") ** 2, "shifted lognormal expected response") + extra_draws(model, post, "ndt")


def _draw_response(model, post, eta, mean, df, rng):
    """``ndt + LogNormal(mu, sigma)`` draws with ``mu`` the linear predictor."""
    mu = np.asarray(eta, dtype=float)
    return extra_draws(model, post, "ndt") + rng.lognormal(mu, np.broadcast_to(auxiliary(model, post, "sigma"), mu.shape))


def _log_likelihood(model, post, eta):
    """brms's log_lik (C07): ``log LogNormal(y_i - ndt | mu_i, sigma)``, sigma and ndt per draw."""
    from autobayes.families.pointwise import shifted_lognormal

    return shifted_lognormal(model.design.y, eta, auxiliary(model, post, "sigma"), extra_draws(model, post, "ndt"))


def _likelihood_lines(model) -> list[str]:
    return [scale_line(model), NDT.prior_line(model), "", "Likelihood",
            "  y_i ~ ndt + LogNormal(mu_i, sigma), i.e. log(y_i - ndt) ~ Normal(mu_i, sigma); brms's lognormal_lpdf(y - ndt | mu, sigma)",
            "  E[y_i] = exp(mu_i + sigma^2 / 2) + ndt; sigma is the SD of log(y - ndt), sampled as log(sigma); ndt lies in",
            "  (0, min(y)), sampled as log(ndt / (min(y) - ndt)). predict(kind='mean') returns E[y_i]; kind='linear' returns mu_i.",
            "  Responses are strictly positive finite reals."]


SHIFTED_LOGNORMAL = FamilySpec(
    name="shifted_lognormal", tag="shifted lognormal", parent="mu", link="identity", link_scale="log", mean_name="mean",
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
    scale_rule="sigma is required for the shifted_lognormal family; shape and phi do not apply",
    scale_missing_error="the shifted_lognormal family needs a prior on the SD of log(y - ndt): pass sigma=...",
    linpred_symbol="mu", scale_phrase="log response", intercept_meaning="expected log(y - ndt)",
    mean_lines=lambda model: ["  log(y_i - ndt) ~ Normal(mu_i, sigma): mu is the location of log(y - ndt), so the coefficients act on it;",
                              "  E[y_i] = exp(mu_i + sigma^2 / 2) + ndt (brms posterior_epred)."],
    likelihood_lines=_likelihood_lines,
    centred_intercept_note=(),
    deterministic_note="mu = linear predictor (log-scale location); the expected response exp(mu + sigma^2 / 2) + ndt is computed by predict()",
    default_prior_link="log",                                       # brms reads log(y), as for the lognormal
    default_scale_prior=lambda scale: HalfStudentT(3, scale),       # brms sigma: student_t(3, 0, scale), lb 0
    extra_parameters=(NDT,),                                        # D01: brms's ndt, default uniform(0, min_Y)
    log_likelihood=_log_likelihood,                                 # C07: log_lik(), checked against the model's own density
)
