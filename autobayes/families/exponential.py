"""Exponential response (brms ``exponential()``, brms-parity D01, docs/continuous-families.md): strictly positive reals, log
link on the mean ``mu``, no auxiliary parameter.

brms's density is ``exponential_lpdf(y | 1 / mu)``, the exponential of rate ``1 / mu``: ``E[y] = mu``, ``Var[y] = mu^2``.
Bambi's ``exponential`` family is that parameterisation (``lam = 1 / mu`` handed to PyMC's ``Exponential``). Checked against
brms in the oracle case ``exponential_varint``, not assumed. No ``weights()``: no frozen case verifies it (D79).
"""
from __future__ import annotations

from autobayes.families.spec import NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, FamilySpec, exp_mean
from autobayes.formula import validate_positive_response


def _bambi_family():
    from bambi.defaults import get_builtin_family

    family = get_builtin_family("exponential")         # PyMC's Exponential(lam = 1 / mu): brms's form
    family.link = {"mu": "log"}                        # brms's default link
    return family


def _validate_response(spec, data) -> None:
    validate_positive_response(spec.response, data[spec.response], "exponential")


def _inverse_link(eta):
    return exp_mean(eta, "exponential expected response")


def _draw_response(model, post, eta, mean, df, rng):
    """``Exponential(rate = 1 / mu)`` draws: numpy's exponential of scale ``mu``."""
    return rng.exponential(mean)


def _log_likelihood(model, post, eta):
    """brms's log_lik (C07): ``log Exponential(y_i | 1 / exp(eta_i))`` = ``-eta_i - y_i exp(-eta_i)``."""
    from autobayes.families.pointwise import exponential_log

    return exponential_log(model.design.y, eta)


def _likelihood_lines(model) -> list[str]:
    return ["", "Likelihood",
            "  y_i ~ Exponential(rate = 1 / mu_i); mu_i = exp(eta_i); E[y_i] = mu_i, Var[y_i] = mu_i^2",
            "  brms's exponential_lpdf(y | inv(mu)), through PyMC's Exponential; no auxiliary parameter.",
            "  Responses are strictly positive finite reals."]


EXPONENTIAL = FamilySpec(
    name="exponential", tag="exponential", parent="mu", link="log", link_scale="log", mean_name="mu",
    response_kind="continuous", response_dtype=None,
    scale_parameter=None, bambi_scale_parameter=None, fixed_parameters=NO_FIXED_PARAMETERS,
    bambi_family=_bambi_family,
    validate_response=_validate_response,
    response_warnings=lambda spec, df: [],
    response_upper_bound=lambda model: None,
    inverse_link=_inverse_link,
    expected_response=None,
    draw_response=_draw_response,
    manual_prediction=True, prediction_kinds=STANDARD_PREDICTION_KINDS,
    kind_error="kind must be 'mean', 'response' or 'linear'",
    scale_rule="the exponential family has no auxiliary parameter; sigma, shape and phi do not apply",
    scale_missing_error=None,
    linpred_symbol="eta", scale_phrase="log mean", intercept_meaning="log expected response",
    mean_lines=lambda model: ["  E[y_i] = mu_i = exp(eta_i); eta is the log expected response; Var[y_i] = mu_i^2."],
    likelihood_lines=_likelihood_lines,
    centred_intercept_note=(),
    deterministic_note="mu = exp(linear predictor); mu = exp(eta), expected response",
    default_prior_link="log",                                       # brms reads log(y) under the log link
    default_scale_prior=None,                                       # no auxiliary parameter
    log_likelihood=_log_likelihood,                                 # C07: log_lik(), checked against the model's own density
)
