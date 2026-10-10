"""Lognormal response (brms ``lognormal()``): identity link on the log-scale location ``mu`` (the linear predictor),
scale ``sigma``.

``log(y) ~ Normal(mu, sigma)``, brms's ``lognormal_lpdf(y | mu, sigma)``: ``E[y] = exp(mu + sigma^2 / 2)``, the median
is ``exp(mu)`` and ``Var[y] = (exp(sigma^2) - 1) exp(2 mu + sigma^2)``. The coefficients act on ``log y`` and the centred
intercept is the expected log response at the predictor means. Bambi has no plain lognormal family, so
``likelihoods.lognormal_family`` wraps PyMC's ``LogNormal(mu, sigma)`` the way Bambi's ``gaussian`` wraps ``Normal``; the
prior is ``Priors(sigma=...)`` (brms class ``sigma``) and the sampled variable ``sigma_log__``. ``predict(kind="mean")``
returns brms's ``posterior_epred`` ``exp(mu + sigma^2 / 2)`` under the name ``mean`` (not ``exp(mu)``, the median, and not
``mu``, which is the location); ``kind="linear"`` returns ``mu``. Checked against brms in the oracle case
``lognormal_varint``, not assumed.
"""
from __future__ import annotations

from autobayes.families.spec import NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, FamilySpec, exp_mean
from autobayes.formula import validate_positive_response
from autobayes.priors import HalfStudentT, describe_prior


def _bambi_family():
    from autobayes.likelihoods import lognormal_family

    return lognormal_family()


def _validate_response(spec, data) -> None:
    validate_positive_response(spec.response, data[spec.response], "lognormal")


def _expected_response(model, post, eta):
    """brms's ``posterior_epred`` ``exp(mu + sigma^2 / 2)`` per draw, refused rather than clipped when it overflows."""
    sigma = post[model.names["sigma"]].to_numpy()
    return exp_mean(eta + 0.5 * sigma[..., None] ** 2, "lognormal expected response")


def _draw_response(model, post, eta, mean, df, rng):
    """``LogNormal(mu, sigma)`` draws with ``mu`` the linear predictor (numpy's ``lognormal(mean, sigma)`` of ``log y``)."""
    sigma = post[model.names["sigma"]].to_numpy()
    return rng.lognormal(eta, sigma[..., None])


def _likelihood_lines(model) -> list[str]:
    return [f"  sigma ~ {describe_prior(model.priors.sigma, lb=0)}", "", "Likelihood",
            "  y_i ~ LogNormal(mu_i, sigma), i.e. log(y_i) ~ Normal(mu_i, sigma); brms's lognormal_lpdf(y | mu, sigma)",
            "  E[y_i] = exp(mu_i + sigma^2 / 2), median exp(mu_i), Var[y_i] = (exp(sigma^2) - 1) exp(2 mu_i + sigma^2);",
            "  sigma is the SD of log y, sampled as log(sigma). predict(kind='mean') returns E[y_i]; kind='linear' returns mu_i.",
            "  Responses are strictly positive finite reals."]


LOGNORMAL = FamilySpec(
    name="lognormal", tag="lognormal", parent="mu", link="identity", link_scale="log", mean_name="mean",
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
    scale_rule="sigma is required for the lognormal family; no other auxiliary parameter applies",
    scale_missing_error="the lognormal family needs a prior on the SD of log y: pass sigma=...",
    linpred_symbol="mu", scale_phrase="log response", intercept_meaning="expected log response",
    mean_lines=lambda model: ["  log(y_i) ~ Normal(mu_i, sigma): mu is the location of log y, so the coefficients act on log y;",
                              "  E[y_i] = exp(mu_i + sigma^2 / 2) (brms posterior_epred), median(y_i) = exp(mu_i)."],
    likelihood_lines=_likelihood_lines,
    centred_intercept_note=(),
    deterministic_note="mu = linear predictor (log-scale location); the expected response exp(mu + sigma^2 / 2) is computed by predict()",
    default_prior_link="log",                                       # brms's log-scale family: it reads log(y)
    default_scale_prior=lambda scale: HalfStudentT(3, scale),       # brms sigma: student_t(3, 0, scale), lb 0
    addition_terms=frozenset({"weights"}),                      # B02: weights(), verified against the frozen case lognormal_weighted
)
