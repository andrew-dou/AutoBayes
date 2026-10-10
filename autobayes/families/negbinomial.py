"""Negative binomial count response (brms ``negbinomial``): log link on ``mu``, overdispersion parameter ``shape``.

brms's ``neg_binomial_2`` parameterisation: ``E[y] = mu``, ``Var[y] = mu + mu^2 / shape``, Poisson as ``shape``
grows. Bambi's ``negativebinomial`` family is PyMC's ``NegativeBinomial(mu, alpha)`` with ``alpha`` that same
``shape``; the prior is given as ``Priors(shape=...)`` (brms class ``shape``) and the sampled variable is
``alpha_log__`` (``model.names["shape"]`` is ``"alpha"``). Equality of the two densities and their gradients is
checked against brms in the oracle case ``negbinomial_varint``, not assumed.
"""
from __future__ import annotations

import numpy as np

from autobayes.families.spec import NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, FamilySpec, exp_mean
from autobayes.formula import check_counts
from autobayes.priors import InverseGamma, describe_prior


def _validate_response(spec, data) -> None:
    check_counts(f"response {spec.response!r}", data[spec.response], domain="poisson")


def _response_warnings(spec, df) -> list[str]:
    if (df[spec.response] == 0).all():
        return ["all observed counts are zero; the shape parameter is then informed by its prior alone; "
                "check sensitivity to the explicit log-mean and shape priors"]
    return []


def _inverse_link(eta):
    return exp_mean(eta, "negative binomial expected count")


def _draw_response(model, post, eta, mean, df, rng):
    """``NegativeBinomial(mu, shape)`` draws: numpy's ``negative_binomial(n = shape, p = shape / (shape + mu))``
    has mean ``mu`` and variance ``mu + mu^2 / shape``. Means beyond the integer generator's range are refused."""
    limit = np.iinfo(np.int64).max - 10 * np.sqrt(float(np.iinfo(np.int64).max))
    if np.any(mean >= limit):
        raise ValueError("negative binomial response prediction exceeds the integer random-generator limit")
    shape = post[model.names["shape"]].to_numpy()[..., None]
    return rng.negative_binomial(shape, shape / (shape + mean)).astype(np.int64)


def _likelihood_lines(model) -> list[str]:
    return [f"  shape ~ {describe_prior(model.priors.shape, lb=0)}", "", "Likelihood",
            "  y_i ~ NegativeBinomial(mu_i, shape); mu_i = exp(eta_i); Var[y_i] = mu_i + mu_i^2 / shape",
            "  brms's neg_binomial_2 parameterisation: shape is the inverse overdispersion (Poisson as shape -> infinity);",
            "  PyMC's NegativeBinomial(mu, alpha) with alpha = shape, sampled as log(shape).",
            "  Input counts are integers from 0 through 1e6; zero-count rows are retained.",
            "  No exposure/offset or zero-inflation is modeled."]


NEGBINOMIAL = FamilySpec(
    name="negbinomial", tag="negative binomial", parent="mu", link="log", link_scale="log", mean_name="mu",
    response_kind="count", response_dtype=np.int64,
    scale_parameter="shape", bambi_scale_parameter="alpha", fixed_parameters=NO_FIXED_PARAMETERS,
    bambi_family=lambda: "negativebinomial",
    validate_response=_validate_response,
    response_warnings=_response_warnings,
    response_upper_bound=lambda model: None,
    inverse_link=_inverse_link,
    expected_response=None,
    draw_response=_draw_response,
    manual_prediction=True, prediction_kinds=STANDARD_PREDICTION_KINDS,
    kind_error="kind must be 'mean', 'response' or 'linear'",
    scale_rule="shape is required for the negative binomial family; sigma does not apply",
    scale_missing_error="the negative binomial family needs a prior on the shape (overdispersion) parameter: pass shape=...",
    linpred_symbol="eta", scale_phrase="log mean", intercept_meaning="log expected count",
    mean_lines=lambda model: ["  E[y_i] = mu_i = exp(eta_i); eta is log expected count; Var[y_i] = mu_i + mu_i^2 / shape."],
    likelihood_lines=_likelihood_lines,
    centred_intercept_note=(),
    deterministic_note="mu = exp(linear predictor); mu = exp(eta), expected count",
    default_prior_link="log",                                       # brms reads log(y), zeros as 0.1
    default_scale_prior=lambda scale: InverseGamma(0.4, 0.3),       # brms negbinomial()$prior: shape inv_gamma(0.4, 0.3), lb 0
    addition_terms=frozenset({"weights"}),                      # B02: weights(), verified against the frozen case negbinomial_weighted
)
