"""Hurdle Gamma responses (brms ``hurdle_gamma``): non-negative reals, log link on the mean ``mu`` of the Gamma part, its shape
``shape`` and the hurdle probability ``hu`` (brms-parity D03, docs/zero-inflated-hurdle.md).

A row is an exact zero with probability ``hu`` and a ``Gamma(shape, rate = shape / mu)`` value otherwise (brms's
``gamma_lpdf(y | shape, shape / mu)``, ``mu = exp(eta)``): the density is ``hu`` at 0 and ``(1 - hu) Gamma(y | shape, shape /
mu)`` above, ``E[y] = (1 - hu) mu`` (brms's ``posterior_epred``). The likelihood is ``likelihoods.hurdle_gamma_family``: ``eta``
kept as the parent ``log_mu`` under the identity link (brms's hurdle_gamma has the log link by default, unlike its Gamma
family, D9), the shape under Bambi's name ``alpha`` and ``hu`` under brms's. Checked against brms in the oracle case
``hurdle_gamma_varint``, not assumed.
"""
from __future__ import annotations

import numpy as np

from autobayes.families.spec import NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, FamilySpec, auxiliary, exp_mean, scale_line
from autobayes.formula import validate_nonnegative_response
from autobayes.priors import Beta, Gamma, describe_prior


def _bambi_family():
    from autobayes.likelihoods import hurdle_gamma_family

    return hurdle_gamma_family()


def _validate_response(spec, data) -> None:
    validate_nonnegative_response(spec.response, data[spec.response], "hurdle Gamma")


def _response_warnings(spec, df) -> list[str]:
    if (df[spec.response].to_numpy(dtype=float) == 0).all():
        return ["every response is zero: the Gamma part has no data, so its mean and shape are informed by their priors alone; "
                "check sensitivity to the explicit priors"]
    return []


def _inverse_link(eta):
    return exp_mean(eta, "hurdle Gamma positive-part mean")


def _expected_response(model, post, eta):
    """brms's ``posterior_epred`` ``(1 - hu) mu`` with ``mu = exp(eta)``, per draw; refused rather than clipped when ``mu``
    overflows."""
    hu = post[model.names["hu"]].to_numpy()[..., None]
    return (1.0 - hu) * exp_mean(eta, "hurdle Gamma expected response")


def _draw_response(model, post, eta, mean, df, rng):
    """brms's ``posterior_predict``: zero where a uniform draw falls below ``hu``, a ``Gamma(shape, shape / mu)`` draw otherwise
    (numpy's ``gamma(shape, scale = mu / shape)``)."""
    mu = exp_mean(eta, "hurdle Gamma positive-part mean")
    hu, shape = post[model.names["hu"]].to_numpy()[..., None], auxiliary(model, post, "shape")
    zero = rng.uniform(size=mu.shape) < hu
    return np.where(zero, 0.0, rng.gamma(shape, mu / shape))


def _log_likelihood(model, post, eta):
    """brms's log_lik (C07): ``hurdle_gamma_lpdf(y_i | shape, shape / exp(eta_i), hu)``, shape and hu per draw."""
    from autobayes.families.pointwise import hurdle_gamma_log

    return hurdle_gamma_log(model.design.y, eta, auxiliary(model, post, "shape"), post[model.names["hu"]].to_numpy()[..., None])


def _likelihood_lines(model) -> list[str]:
    return [scale_line(model), f"  hu ~ {describe_prior(model.priors.hu, lb=0, ub=1)}", "", "Likelihood",
            "  y_i ~ HurdleGamma(mu_i, shape, hu); mu_i = exp(eta_i): an exact zero with probability hu,",
            "  Gamma(shape, rate = shape / mu_i) otherwise; the density is hu at 0 and (1 - hu) Gamma(y_i | shape, shape / mu_i) above",
            "  (brms's hurdle_gamma); E[y_i] = (1 - hu) mu_i. shape is sampled as log(shape) (PyMC's alpha); hu is brms's hurdle",
            "  probability, sampled as logit(hu).",
            "  Responses are non-negative finite reals: exact zeros are the zero part."]


HURDLE_GAMMA = FamilySpec(
    name="hurdle_gamma", tag="hurdle Gamma", parent="log_mu", link="log", link_scale="log", mean_name="mean",
    response_kind="continuous", response_dtype=None,
    scale_parameter="shape", bambi_scale_parameter="alpha", fixed_parameters=NO_FIXED_PARAMETERS,
    bambi_family=_bambi_family,
    validate_response=_validate_response,
    response_warnings=_response_warnings,
    response_upper_bound=lambda model: None,
    inverse_link=_inverse_link,
    expected_response=_expected_response,
    draw_response=_draw_response,
    manual_prediction=True, prediction_kinds=STANDARD_PREDICTION_KINDS,
    kind_error="kind must be 'mean', 'response' or 'linear'",
    scale_rule="shape is required for the hurdle Gamma family; sigma and phi do not apply (its hurdle probability is hu)",
    scale_missing_error="the hurdle_gamma family needs a prior on the shape parameter: pass shape=...",
    linpred_symbol="eta", scale_phrase="log positive-part mean", intercept_meaning="log mean of the Gamma part",
    mean_lines=lambda model: ["  mu_i = exp(eta_i) is the mean of the Gamma part (eta is its log); E[y_i] = (1 - hu) mu_i;",
                              "  Var[y_i | y_i > 0] = mu_i^2 / shape."],
    likelihood_lines=_likelihood_lines,
    centred_intercept_note=(),
    deterministic_note="log_mu = linear predictor (the log of the Gamma part's mean); E[y] = (1 - hu) exp(log_mu) is computed by predict()",
    default_prior_link="log",                                       # brms reads log(y), zeros as 0.1
    default_scale_prior=lambda scale: Gamma(0.01, 0.01),            # brms shape: gamma(0.01, 0.01), lb 0
    zero_parameter="hu",
    default_zero_prior=lambda: Beta(1, 1),                          # brms hu: beta(1, 1), lb 0, ub 1
    addition_terms=frozenset({"weights"}),                          # D03: weights(), each row's log likelihood times its weight
    log_likelihood=_log_likelihood,                                 # C07: log_lik(), checked against the model's own density
)
