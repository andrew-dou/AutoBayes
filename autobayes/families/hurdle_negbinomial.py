"""Hurdle negative binomial counts (brms ``hurdle_negbinomial``): log link on the mean ``mu`` of the untruncated negative
binomial part, its shape ``shape`` and the hurdle probability ``hu`` (brms-parity D03, docs/zero-inflated-hurdle.md).

A row is zero with probability ``hu`` and a zero-truncated ``NB(mu, shape)`` count otherwise: ``P(y = 0) = hu``, ``P(y = k) =
(1 - hu) NB(k | mu, shape) / (1 - NB(0 | mu, shape))`` for ``k > 0`` with ``NB(0 | mu, shape) = (shape / (mu + shape))^shape``,
``E[y] = (1 - hu) mu / (1 - NB(0 | mu, shape))`` (brms's ``posterior_epred``). The likelihood is
``likelihoods.hurdle_negbinomial_family``: ``eta = log(mu)`` as the parent ``log_mu`` under the identity link, the shape
under Bambi's name ``alpha`` and ``hu`` under brms's. Response draws are exact zero-truncated negative binomial draws
(``likelihoods.zero_truncated_negbinomial_draws``), where brms's ``posterior_predict`` draws are not that distribution
(divergence D86).
Checked against brms in the oracle case ``hurdle_negbinomial_varint``, not assumed.
"""
from __future__ import annotations

import numpy as np

from autobayes.families.spec import NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, FamilySpec, auxiliary, exp_mean, scale_line
from autobayes.formula import check_counts
from autobayes.priors import Beta, InverseGamma, describe_prior


def _bambi_family():
    from autobayes.likelihoods import hurdle_negbinomial_family

    return hurdle_negbinomial_family()


def _validate_response(spec, data) -> None:
    check_counts(f"response {spec.response!r}", data[spec.response], domain="poisson")


def _response_warnings(spec, df) -> list[str]:
    if (df[spec.response] == 0).all():
        return ["all observed counts are zero: the count part has no data, so its mean and shape are informed by their priors "
                "alone; check sensitivity to the explicit priors"]
    return []


def _inverse_link(eta):
    return exp_mean(eta, "hurdle negative binomial count-part mean")


def positive_mean(mu, shape):
    """``E[y | y > 0] = mu / (1 - NB(0 | mu, shape))``, ``1 - NB(0)`` as ``-expm1(-shape log1p(mu / shape))`` (exact for small
    ``mu``); its limit 1 where ``mu`` has underflowed to 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(mu > 0, mu / -np.expm1(-shape * np.log1p(mu / shape)), 1.0)


def _expected_response(model, post, eta):
    """brms's ``posterior_epred`` ``(1 - hu) mu / (1 - (shape / (mu + shape))^shape)`` with ``mu = exp(eta)``, per draw;
    refused rather than clipped when ``mu`` overflows."""
    hu = post[model.names["hu"]].to_numpy()[..., None]
    mu = exp_mean(eta, "hurdle negative binomial expected count")
    return (1.0 - hu) * positive_mean(mu, auxiliary(model, post, "shape"))


def _draw_response(model, post, eta, mean, df, rng):
    """Zero where a uniform draw falls below ``hu`` (brms's ``posterior_predict``), a zero-truncated ``NB(mu, shape)`` draw
    otherwise (``likelihoods.zero_truncated_negbinomial_draws``, exact; D86). Means beyond the integer generator's range are
    refused."""
    from autobayes.likelihoods import zero_truncated_negbinomial_draws

    mu = exp_mean(eta, "hurdle negative binomial count-part mean")
    limit = np.iinfo(np.int64).max - 10 * np.sqrt(float(np.iinfo(np.int64).max))
    if np.any(mu >= limit):
        raise ValueError("hurdle negative binomial response prediction exceeds the integer random-generator limit")
    hu, shape = post[model.names["hu"]].to_numpy()[..., None], auxiliary(model, post, "shape")
    zero = rng.uniform(size=mu.shape) < hu
    return np.where(zero, 0, zero_truncated_negbinomial_draws(mu, np.broadcast_to(shape, mu.shape), rng)).astype(np.int64)


def _log_likelihood(model, post, eta):
    """brms's log_lik (C07): ``hurdle_neg_binomial_log_lpmf(y_i | eta_i, shape, hu)``, shape and hu per draw."""
    from autobayes.families.pointwise import hurdle_negbinomial_log

    return hurdle_negbinomial_log(model.design.y, eta, auxiliary(model, post, "shape"), post[model.names["hu"]].to_numpy()[..., None])


def _likelihood_lines(model) -> list[str]:
    return [scale_line(model), f"  hu ~ {describe_prior(model.priors.hu, lb=0, ub=1)}", "", "Likelihood",
            "  y_i ~ HurdleNegBinomial(mu_i, shape, hu); mu_i = exp(eta_i): zero with probability hu, a zero-truncated",
            "  NegativeBinomial(mu_i, shape) otherwise (brms's neg_binomial_2: Var = mu_i + mu_i^2 / shape before truncation);",
            "  P(y_i = 0) = hu; P(y_i = k) = (1 - hu) NB(k | mu_i, shape) / (1 - NB(0 | mu_i, shape)) for k > 0 with",
            "  NB(0 | mu_i, shape) = (shape / (mu_i + shape))^shape (brms's hurdle_negbinomial);",
            "  E[y_i] = (1 - hu) mu_i / (1 - NB(0 | mu_i, shape)). hu is brms's hurdle probability, sampled as logit(hu); shape is",
            "  sampled as log(shape) (PyMC's alpha).",
            "  Input counts are integers from 0 through 1e6; zero-count rows are retained.",
            ("  The exposure enters through the offset term (mu_i = exp(offset_i) * exp(eta_i - offset_i))."
             if model.spec.offset is not None else "  No exposure/offset is modeled.")]


HURDLE_NEGBINOMIAL = FamilySpec(
    name="hurdle_negbinomial", tag="hurdle negative binomial", parent="log_mu", link="log", link_scale="log", mean_name="mean",
    response_kind="count", response_dtype=np.int64,
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
    scale_rule="shape is required for the hurdle negative binomial family; sigma and phi do not apply (its hurdle probability is hu)",
    scale_missing_error="the hurdle_negbinomial family needs a prior on the shape (overdispersion) parameter: pass shape=...",
    linpred_symbol="eta", scale_phrase="log count-part mean", intercept_meaning="log mean of the (untruncated) negative binomial part",
    mean_lines=lambda model: ["  mu_i = exp(eta_i) is the mean of the untruncated negative binomial part (eta is its log);",
                              "  E[y_i] = (1 - hu) mu_i / (1 - (shape / (mu_i + shape))^shape)."],
    likelihood_lines=_likelihood_lines,
    centred_intercept_note=(),
    deterministic_note=("log_mu = linear predictor (the log of the negative binomial part's mean); E[y] = (1 - hu) mu / (1 - "
                        "(shape / (mu + shape))^shape) with mu = exp(log_mu) is computed by predict()"),
    default_prior_link="log",                                       # brms reads log(y), zeros as 0.1
    default_scale_prior=lambda scale: InverseGamma(0.4, 0.3),       # brms shape: inv_gamma(0.4, 0.3), lb 0
    zero_parameter="hu",
    default_zero_prior=lambda: Beta(1, 1),                          # brms hu: beta(1, 1), lb 0, ub 1
    addition_terms=frozenset({"weights"}),                          # D03: weights(), each row's log likelihood times its weight
    log_likelihood=_log_likelihood,                                 # C07: log_lik(), checked against the model's own density
)
