"""Zero-inflated negative binomial counts (brms ``zero_inflated_negbinomial``): log link on the mean ``mu`` of the negative
binomial part, its shape ``shape`` and the zero-inflation probability ``zi`` (brms-parity D03, docs/zero-inflated-hurdle.md).

A row is a structural zero with probability ``zi`` and a ``NB(mu, shape)`` count otherwise (brms's ``neg_binomial_2``: mean
``mu``, variance ``mu + mu^2 / shape``): ``P(y = 0) = zi + (1 - zi) (shape / (mu + shape))^shape``, ``P(y = k) = (1 - zi)
NB(k | mu, shape)`` for ``k > 0``, ``E[y] = (1 - zi) mu`` (brms's ``posterior_epred``). The likelihood is
``likelihoods.zero_inflated_negbinomial_family``: the linear predictor ``eta = log(mu)`` kept as Bambi's parent ``log_mu``
under the identity link, the shape under Bambi's name ``alpha`` (as for the negative binomial family; ``Priors(shape=...)``,
sampled as ``alpha_log__``) and ``zi`` under brms's (``Priors(zi=...)``, a ``Beta`` prior, sampled as ``zi_logodds__``).
Checked against brms in the oracle case ``zero_inflated_negbinomial_varint``, not assumed. E03
(docs/contexts-newer-families.md): ``zi`` takes a formula of its own under brms's logit link (``["zcnt ~ x", "zi ~ z"]``, one
probability per row, ``zi_i = inv_logit(eta_zi,i)``, its intercept's default prior ``logistic(0, 1)``;
``zero_inflated_negbinomial_distributional_zi``; Bambi's inverse logit, which saturates where brms's logit-scale density does not,
D117); a formula on ``shape`` stays refused (D84).
"""
from __future__ import annotations

import numpy as np

from autobayes.families.spec import (NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, DistributionalFormula, FamilySpec, auxiliary,
                                     exp_mean, is_distributional, parameter_symbol, scale_line, zero_line)
from autobayes.formula import check_counts
from autobayes.priors import Beta, InverseGamma, Logistic


def _bambi_family():
    from autobayes.likelihoods import zero_inflated_negbinomial_family

    return zero_inflated_negbinomial_family()


def _validate_response(spec, data) -> None:
    check_counts(f"response {spec.response!r}", data[spec.response], domain="poisson")


def _response_warnings(spec, df) -> list[str]:
    if (df[spec.response] == 0).all():
        return ["all observed counts are zero: the zero-inflation probability, the mean and the shape of the negative binomial "
                "part are not separately identified (only P(y = 0) is); check sensitivity to the explicit priors"]
    return []


def _inverse_link(eta):
    return exp_mean(eta, "zero-inflated negative binomial count-part mean")


def _expected_response(model, post, eta):
    """brms's ``posterior_epred`` ``(1 - zi) mu`` with ``mu = exp(eta)``, per draw; refused rather than clipped when ``mu``
    overflows."""
    zi = auxiliary(model, post, "zi")                               # one per row with its own formula (E03)
    return (1.0 - zi) * exp_mean(eta, "zero-inflated negative binomial expected count")


def _draw_response(model, post, eta, mean, df, rng):
    """brms's ``posterior_predict``: zero where a uniform draw falls below ``zi``, a ``NB(mu, shape)`` draw otherwise (numpy's
    ``negative_binomial(shape, shape / (shape + mu))``). Means beyond the integer generator's range are refused."""
    mu = exp_mean(eta, "zero-inflated negative binomial count-part mean")
    limit = np.iinfo(np.int64).max - 10 * np.sqrt(float(np.iinfo(np.int64).max))
    if np.any(mu >= limit):
        raise ValueError("zero-inflated negative binomial response prediction exceeds the integer random-generator limit")
    zi, shape = auxiliary(model, post, "zi"), auxiliary(model, post, "shape")
    zero = rng.uniform(size=mu.shape) < zi
    return np.where(zero, 0, rng.negative_binomial(shape, shape / (shape + mu))).astype(np.int64)


def _log_likelihood(model, post, eta):
    """brms's log_lik (C07): ``zero_inflated_neg_binomial_log_lpmf(y_i | eta_i, shape, zi)``, shape and zi per draw."""
    from autobayes.families.pointwise import zero_inflated_negbinomial_log

    return zero_inflated_negbinomial_log(model.design.y, eta, auxiliary(model, post, "shape"), auxiliary(model, post, "zi"))


def _likelihood_lines(model) -> list[str]:
    zi = parameter_symbol(model, "zi")                              # zi_i with its own formula (E03)
    return [scale_line(model), zero_line(model), "", "Likelihood",
            f"  y_i ~ ZeroInflatedNegBinomial(mu_i, shape, {zi}); mu_i = exp(eta_i): a structural zero with probability {zi},",
            "  NegativeBinomial(mu_i, shape) otherwise (brms's neg_binomial_2: Var = mu_i + mu_i^2 / shape);",
            f"  P(y_i = 0) = {zi} + (1 - {zi}) (shape / (mu_i + shape))^shape; P(y_i = k) = (1 - {zi}) NB(k | mu_i, shape) for k > 0",
            f"  (brms's zero_inflated_negbinomial); E[y_i] = (1 - {zi}) mu_i. zi is brms's zero-inflation probability, "
            + ("one per row from its formula (above);" if is_distributional(model, "zi") else "sampled as"),
            ("  " if is_distributional(model, "zi") else "  logit(zi); ") + "shape is sampled as log(shape) (PyMC's alpha).",
            "  Input counts are integers from 0 through 1e6; zero-count rows are retained.",
            ("  The exposure enters through the offset term (mu_i = exp(offset_i) * exp(eta_i - offset_i))."
             if model.spec.offset is not None else "  No exposure/offset is modeled.")]


ZERO_INFLATED_NEGBINOMIAL = FamilySpec(
    name="zero_inflated_negbinomial", tag="zero-inflated negative binomial", parent="log_mu", link="log", link_scale="log",
    mean_name="mean",
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
    scale_rule="shape is required for the zero-inflated negative binomial family; sigma and phi do not apply (its zero-inflation "
               "probability is zi)",
    scale_missing_error="the zero_inflated_negbinomial family needs a prior on the shape (overdispersion) parameter: pass shape=...",
    linpred_symbol="eta", scale_phrase="log count-part mean", intercept_meaning="log mean of the negative binomial part",
    mean_lines=lambda model: ["  mu_i = exp(eta_i) is the mean of the negative binomial part (eta is its log); E[y_i] = "
                              f"(1 - {parameter_symbol(model, 'zi')}) mu_i."],
    likelihood_lines=_likelihood_lines,
    centred_intercept_note=(),
    deterministic_note=("log_mu = linear predictor (the log of the negative binomial part's mean); E[y] = (1 - zi) exp(log_mu) is "
                        "computed by predict()"),
    default_prior_link="log",                                       # brms reads log(y), zeros as 0.1
    default_scale_prior=lambda scale: InverseGamma(0.4, 0.3),       # brms shape: inv_gamma(0.4, 0.3), lb 0
    zero_parameter="zi",
    default_zero_prior=lambda: Beta(1, 1),                          # brms zi: beta(1, 1), lb 0, ub 1
    addition_terms=frozenset({"weights"}),                          # D03: weights(), each row's log likelihood times its weight
    log_likelihood=_log_likelihood,                                 # C07: log_lik(), checked against the model's own density
    # E03: zi ~ z under brms's logit link (zero_inflated_negbinomial_distributional_zi); its intercept's default logistic(0, 1)
    distributional_formulas=(DistributionalFormula("zi", "logit", lambda: Logistic(0, 1)),),
)
