"""Zero-inflated Beta proportions (brms ``zero_inflated_beta``): proportions in [0, 1), logit link on the mean ``mu`` of the
Beta part, its precision ``phi`` and the zero-inflation probability ``zi`` (brms-parity D03, docs/zero-inflated-hurdle.md).

A row is an exact zero with probability ``zi`` and a ``Beta(mu phi, (1 - mu) phi)`` proportion otherwise, ``mu =
inv_logit(eta)``: the density is ``zi`` at 0 and ``(1 - zi) Beta(y | mu phi, (1 - mu) phi)`` on (0, 1), ``E[y] = (1 - zi) mu``
(brms's ``posterior_epred``). The likelihood is ``likelihoods.zero_inflated_beta_family``: ``eta`` kept as the parent
``logit_mu`` under the identity link, the Beta part the Beta family's (``1 - mu`` as ``expit(-eta)``, D13), ``phi`` and ``zi``
under brms's names. The logit link only (D84). Checked against brms in the oracle case ``zero_inflated_beta_varint``.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from autobayes.families.spec import NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, FamilySpec, auxiliary, scale_line
from autobayes.formula import validate_zero_proportion_response
from autobayes.priors import Beta, Gamma, describe_prior

_INTERCEPT_NOTE = (
    "  expit(a_c) is the Beta part's mean at that proportion-weighted predictor; it is NOT the",
    "  average expected proportion across the levels (the link is nonlinear).",
)


def _bambi_family():
    from autobayes.likelihoods import zero_inflated_beta_family

    return zero_inflated_beta_family()


def _validate_response(spec, data) -> None:
    validate_zero_proportion_response(spec.response, data[spec.response], "zero-inflated Beta")


def _response_warnings(spec, df) -> list[str]:
    y = df[spec.response].to_numpy(dtype=float)
    if (y == 0).all():
        return ["every response is zero: the Beta part has no data, so its mean and precision are informed by their priors alone; "
                "check sensitivity to the explicit priors"]
    return []


def _expected_response(model, post, eta):
    """brms's ``posterior_epred`` ``(1 - zi) mu`` with ``mu = expit(eta)``, per draw."""
    zi = post[model.names["zi"]].to_numpy()[..., None]
    return (1.0 - zi) * expit(eta)


def _draw_response(model, post, eta, mean, df, rng):
    """brms's ``posterior_predict``: zero where a uniform draw falls below ``zi``, a ``Beta(mu phi, (1 - mu) phi)`` draw
    otherwise (``1 - mu`` as ``expit(-eta)``); a shape parameter that underflows to zero is refused, as for the Beta family."""
    phi, zi = auxiliary(model, post, "phi"), post[model.names["zi"]].to_numpy()[..., None]
    alpha, beta = expit(eta) * phi, expit(-eta) * phi
    if not (np.all(alpha > 0) and np.all(beta > 0)):
        raise ValueError("zero-inflated Beta response prediction has a shape parameter that underflowed to zero; inspect the "
                         "logit-scale draws")
    zero = rng.uniform(size=np.shape(eta)) < zi
    return np.where(zero, 0.0, rng.beta(alpha, beta))


def _log_likelihood(model, post, eta):
    """brms's log_lik (C07): ``zero_inflated_beta_lpdf(y_i | expit(eta_i), phi, zi)``, phi and zi per draw."""
    from autobayes.families.pointwise import zero_inflated_beta_logit

    return zero_inflated_beta_logit(model.design.y, eta, auxiliary(model, post, "phi"), post[model.names["zi"]].to_numpy()[..., None])


def _likelihood_lines(model) -> list[str]:
    return [scale_line(model), f"  zi ~ {describe_prior(model.priors.zi, lb=0, ub=1)}", "", "Likelihood",
            "  y_i ~ ZeroInflatedBeta(mu_i, phi, zi); mu_i = expit(eta_i): an exact zero with probability zi,",
            "  Beta(mu_i phi, (1 - mu_i) phi) otherwise; the density is zi at 0 and (1 - zi) Beta(y_i | mu_i phi, (1 - mu_i) phi)",
            "  on (0, 1) (brms's zero_inflated_beta); E[y_i] = (1 - zi) mu_i. 1 - mu_i is computed as expit(-eta_i); phi is the",
            "  Beta part's precision, sampled as log(phi); zi is brms's zero-inflation probability, sampled as logit(zi).",
            "  Responses are proportions in [0, 1): exact zeros are the zero part; an exact 1 is refused."]


ZERO_INFLATED_BETA = FamilySpec(
    name="zero_inflated_beta", tag="zero-inflated Beta", parent="logit_mu", link="logit", link_scale="logit", mean_name="mean",
    response_kind="continuous", response_dtype=None,
    scale_parameter="phi", bambi_scale_parameter="phi", fixed_parameters=NO_FIXED_PARAMETERS,
    bambi_family=_bambi_family,
    validate_response=_validate_response,
    response_warnings=_response_warnings,
    response_upper_bound=lambda model: None,
    inverse_link=expit,
    expected_response=_expected_response,
    draw_response=_draw_response,
    manual_prediction=True, prediction_kinds=STANDARD_PREDICTION_KINDS,
    kind_error="kind must be 'mean', 'response' or 'linear'",
    scale_rule="phi is required for the zero-inflated Beta family; sigma and shape do not apply (its zero-inflation probability is zi)",
    scale_missing_error="the zero_inflated_beta family needs a prior on the precision parameter phi: pass phi=...",
    linpred_symbol="eta", scale_phrase="log-odds", intercept_meaning="logit of the Beta part's mean",
    mean_lines=lambda model: ["  mu_i = expit(eta_i) is the mean of the Beta part (eta is on the logit scale); E[y_i] = (1 - zi) mu_i;",
                              "  phi is the Beta part's precision: Var[y_i | y_i > 0] = mu_i (1 - mu_i) / (1 + phi)."],
    likelihood_lines=_likelihood_lines,
    centred_intercept_note=_INTERCEPT_NOTE,
    deterministic_note=("logit_mu = linear predictor (the Beta part's logit mean); E[y] = (1 - zi) expit(logit_mu) is computed by "
                        "predict()"),
    default_prior_link=None,                                        # logit: brms's defaults are student_t(3, 0, 2.5)
    default_scale_prior=lambda scale: Gamma(0.01, 0.01),            # brms phi: gamma(0.01, 0.01), lb 0
    zero_parameter="zi",
    default_zero_prior=lambda: Beta(1, 1),                          # brms zi: beta(1, 1), lb 0, ub 1
    addition_terms=frozenset({"weights"}),                          # D03: weights(), each row's log likelihood times its weight
    log_likelihood=_log_likelihood,                                 # C07: log_lik(), checked against the model's own density
)
