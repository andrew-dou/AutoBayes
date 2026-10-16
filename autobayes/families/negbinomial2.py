"""Negative binomial count response parameterised by its dispersion (brms ``negbinomial2()``, brms-parity D02,
docs/discrete-families.md): log link on the mean ``mu``, dispersion ``sigma``.

brms's ``negbinomial2`` is the negative binomial of ``negbinomial`` with ``shape = 1 / sigma``: ``neg_binomial_2(y | mu,
1 / sigma)``, ``E[y] = mu``, ``Var[y] = mu + sigma mu^2``, Poisson as ``sigma`` goes to 0. brms declares ``real<lower=0>
sigma`` (class ``sigma``, log link, default prior ``gamma(0.4, 0.3)``, which is the distribution of ``1 / shape`` under
``negbinomial``'s default ``inv_gamma(0.4, 0.3)`` on ``shape``). The likelihood is ``likelihoods.negbinomial2_family``
(PyMC's ``NegativeBinomial(mu, alpha = 1 / sigma)``); ``sigma`` takes ``Priors(sigma=...)``, a positive-support prior, and
is sampled as ``sigma_log__``. Checked against brms in the oracle case ``negbinomial2_varint``, not assumed. No
distributional formula on ``sigma``, no ``weights()`` and no other link: no frozen case verifies them (D80, D81).
"""
from __future__ import annotations

import numpy as np

from autobayes.families.spec import NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, FamilySpec, auxiliary, exp_mean, scale_line
from autobayes.formula import check_counts
from autobayes.priors import Gamma


def _bambi_family():
    from autobayes.likelihoods import negbinomial2_family

    return negbinomial2_family()


def _validate_response(spec, data) -> None:
    check_counts(f"response {spec.response!r}", data[spec.response], domain="poisson")


def _response_warnings(spec, df) -> list[str]:
    if (df[spec.response] == 0).all():
        return ["all observed counts are zero; the dispersion sigma is then informed by its prior alone; "
                "check sensitivity to the explicit log-mean and sigma priors"]
    return []


def _inverse_link(eta):
    return exp_mean(eta, "negbinomial2 expected count")


def _draw_response(model, post, eta, mean, df, rng):
    """``NegativeBinomial(mu, 1 / sigma)`` draws: numpy's ``negative_binomial(n = shape, p = shape / (shape + mu))`` with
    ``shape = 1 / sigma`` has mean ``mu`` and variance ``mu + sigma mu^2``. Means beyond the integer generator's range are
    refused."""
    limit = np.iinfo(np.int64).max - 10 * np.sqrt(float(np.iinfo(np.int64).max))
    if np.any(mean >= limit):
        raise ValueError("negbinomial2 response prediction exceeds the integer random-generator limit")
    shape = 1.0 / auxiliary(model, post, "sigma")
    return rng.negative_binomial(shape, shape / (shape + mean)).astype(np.int64)


def _log_likelihood(model, post, eta):
    """brms's log_lik (C07): ``log NegBinomial2(y_i | exp(eta_i), 1 / sigma)``, sigma per draw."""
    from autobayes.families.pointwise import neg_binomial_2_log

    return neg_binomial_2_log(model.design.y, eta, 1.0 / auxiliary(model, post, "sigma"))


def _likelihood_lines(model) -> list[str]:
    return [scale_line(model), "", "Likelihood",
            "  y_i ~ NegativeBinomial(mu_i, shape = 1 / sigma); mu_i = exp(eta_i); Var[y_i] = mu_i + sigma mu_i^2",
            "  brms's negbinomial2: sigma is the dispersion, brms's negbinomial's 1 / shape (Poisson as sigma -> 0);",
            "  PyMC's NegativeBinomial(mu, alpha) with alpha = 1 / sigma, sigma sampled as log(sigma).",
            "  Input counts are integers from 0 through 1e6; zero-count rows are retained."]


NEGBINOMIAL2 = FamilySpec(
    name="negbinomial2", tag="negative binomial (dispersion)", parent="mu", link="log", link_scale="log", mean_name="mu",
    response_kind="count", response_dtype=np.int64,
    scale_parameter="sigma", bambi_scale_parameter="sigma", fixed_parameters=NO_FIXED_PARAMETERS,
    bambi_family=_bambi_family,
    validate_response=_validate_response,
    response_warnings=_response_warnings,
    response_upper_bound=lambda model: None,
    inverse_link=_inverse_link,
    expected_response=None,
    draw_response=_draw_response,
    manual_prediction=True, prediction_kinds=STANDARD_PREDICTION_KINDS,
    kind_error="kind must be 'mean', 'response' or 'linear'",
    scale_rule="sigma (the dispersion, brms's 1 / shape) is required for the negbinomial2 family; shape and phi do not apply",
    scale_missing_error=("the negbinomial2 family needs a prior on the dispersion sigma (brms's negbinomial2: Var[y] = mu + sigma "
                         "mu^2, sigma = 1 / shape of negbinomial): pass sigma=..."),
    linpred_symbol="eta", scale_phrase="log mean", intercept_meaning="log expected count",
    mean_lines=lambda model: ["  E[y_i] = mu_i = exp(eta_i); eta is log expected count; Var[y_i] = mu_i + sigma mu_i^2."],
    likelihood_lines=_likelihood_lines,
    centred_intercept_note=(),
    deterministic_note="mu = exp(linear predictor); mu = exp(eta), expected count",
    default_prior_link="log",                                       # brms reads log(y), zeros as 0.1 (probed)
    default_scale_prior=lambda scale: Gamma(0.4, 0.3),              # brms negbinomial2 sigma: gamma(0.4, 0.3), lb 0 (probed)
    log_likelihood=_log_likelihood,                                 # C07: log_lik(), checked against the model's own density
)
