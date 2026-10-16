"""Geometric count response (brms ``geometric()``, brms-parity D02, docs/discrete-families.md): log link on the mean ``mu``, no
auxiliary parameter.

brms's geometric is the negative binomial of mean ``mu`` and shape 1, ``neg_binomial_2(y | mu, 1)``: ``P(y) = mu^y / (1 +
mu)^(y + 1)`` on ``0, 1, 2, ...``, ``E[y] = mu``, ``Var[y] = mu + mu^2``. The likelihood is ``likelihoods.geometric_family``
(PyMC's ``NegativeBinomial(mu, alpha = 1)``; PyMC's own ``Geometric`` counts trials from 1 and is a different model). Checked
against brms in the oracle case ``geometric_varint``, not assumed. No ``weights()``, no other link: no frozen case verifies
them (D80, D81).
"""
from __future__ import annotations

import numpy as np

from autobayes.families.spec import NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, FamilySpec, exp_mean
from autobayes.formula import check_counts


def _bambi_family():
    from autobayes.likelihoods import geometric_family

    return geometric_family()


def _validate_response(spec, data) -> None:
    check_counts(f"response {spec.response!r}", data[spec.response], domain="poisson")


def _response_warnings(spec, df) -> list[str]:
    if (df[spec.response] == 0).all():
        return ["all observed counts are zero; check sensitivity to the explicit log-mean priors"]
    return []


def _inverse_link(eta):
    return exp_mean(eta, "geometric expected count")


def _draw_response(model, post, eta, mean, df, rng):
    """``NegativeBinomial(mu, 1)`` draws: numpy's ``negative_binomial(n = 1, p = 1 / (1 + mu))``, failures before the first
    success, mean ``mu``. Means beyond the integer generator's range are refused."""
    limit = np.iinfo(np.int64).max - 10 * np.sqrt(float(np.iinfo(np.int64).max))
    if np.any(mean >= limit):
        raise ValueError("geometric response prediction exceeds the integer random-generator limit")
    return rng.negative_binomial(1.0, 1.0 / (1.0 + mean)).astype(np.int64)


def _log_likelihood(model, post, eta):
    """brms's log_lik (C07): ``log NegBinomial2(y_i | exp(eta_i), 1)``."""
    from autobayes.families.pointwise import neg_binomial_2_log

    return neg_binomial_2_log(model.design.y, eta, 1.0)


def _likelihood_lines(model) -> list[str]:
    return ["", "Likelihood",
            "  y_i ~ Geometric(mu_i) = NegativeBinomial(mu_i, shape = 1); mu_i = exp(eta_i); P(y_i = k) = mu_i^k / (1 + mu_i)^(k + 1)",
            "  E[y_i] = mu_i, Var[y_i] = mu_i + mu_i^2; brms's neg_binomial_2(y | mu, 1), through PyMC's NegativeBinomial(mu, alpha = 1);",
            "  no auxiliary parameter.",
            "  Input counts are integers from 0 through 1e6; zero-count rows are retained."]


GEOMETRIC = FamilySpec(
    name="geometric", tag="geometric", parent="mu", link="log", link_scale="log", mean_name="mu",
    response_kind="count", response_dtype=np.int64,
    scale_parameter=None, bambi_scale_parameter=None, fixed_parameters=NO_FIXED_PARAMETERS,
    bambi_family=_bambi_family,
    validate_response=_validate_response,
    response_warnings=_response_warnings,
    response_upper_bound=lambda model: None,
    inverse_link=_inverse_link,
    expected_response=None,
    draw_response=_draw_response,
    manual_prediction=True, prediction_kinds=STANDARD_PREDICTION_KINDS,
    kind_error="kind must be 'mean', 'response' or 'linear'",
    scale_rule="the geometric family has no auxiliary parameter; sigma, shape and phi do not apply",
    scale_missing_error=None,
    linpred_symbol="eta", scale_phrase="log mean", intercept_meaning="log expected count",
    mean_lines=lambda model: ["  E[y_i] = mu_i = exp(eta_i); eta is log expected count; Var[y_i] = mu_i + mu_i^2."],
    likelihood_lines=_likelihood_lines,
    centred_intercept_note=(),
    deterministic_note="mu = exp(linear predictor); mu = exp(eta), expected count",
    default_prior_link="log",                                       # brms reads log(y), zeros as 0.1 (probed)
    default_scale_prior=None,                                       # no auxiliary parameter
    log_likelihood=_log_likelihood,                                 # C07: log_lik(), checked against the model's own density
)
