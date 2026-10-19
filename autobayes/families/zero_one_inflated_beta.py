"""Zero-one-inflated Beta proportions (brms ``zero_one_inflated_beta``): proportions in [0, 1], logit link on the mean ``mu`` of
the Beta part, its precision ``phi``, the zero-one-inflation probability ``zoi`` and the conditional one-inflation probability
``coi`` (brms-parity E01, docs/bounded-families.md).

A row is an exact 0 or 1 with probability ``zoi`` (a 1 with probability ``coi`` given that) and a ``Beta(mu phi, (1 - mu) phi)``
proportion otherwise, ``mu = inv_logit(eta)``: the density is ``zoi (1 - coi)`` at 0, ``zoi coi`` at 1 and ``(1 - zoi) Beta(y |
mu phi, (1 - mu) phi)`` on (0, 1), ``E[y] = zoi coi + (1 - zoi) mu`` (brms's ``posterior_epred``). The likelihood is
``likelihoods.zero_one_inflated_beta_family``: ``eta`` kept as the parent ``logit_mu`` under the identity link, the Beta part the
Beta family's (``1 - mu`` as ``expit(-eta)``, D13), ``phi`` and the two probabilities under brms's names; ``zoi`` and ``coi`` are
the family's ``extra_parameters`` of support ``"unit"`` (brms's ``real<lower=0,upper=1>``, sampled as ``zoi_logodds__`` and
``coi_logodds__``, Stan's logit). The logit link only (D108). Checked against brms in the oracle case
``zero_one_inflated_beta_varint``, not assumed.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from autobayes.families.spec import (NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, ExtraParameter, FamilySpec, auxiliary,
                                     extra_draws, scale_line)
from autobayes.formula import validate_unit_interval_response
from autobayes.priors import Beta, Gamma

ZOI = ExtraParameter(name="zoi", meaning="zero-one-inflation probability", support="unit", default_prior=lambda: Beta(1, 1))
COI = ExtraParameter(name="coi", meaning="conditional one-inflation probability", support="unit", default_prior=lambda: Beta(1, 1))

_INTERCEPT_NOTE = (
    "  expit(a_c) is the Beta part's mean at that proportion-weighted predictor; it is NOT the",
    "  average expected proportion across the levels (the link is nonlinear).",
)


def _bambi_family():
    from autobayes.likelihoods import zero_one_inflated_beta_family

    return zero_one_inflated_beta_family()


def _validate_response(spec, data) -> None:
    validate_unit_interval_response(spec.response, data[spec.response], "zero-one-inflated Beta")


def _response_warnings(spec, df) -> list[str]:
    y = df[spec.response].to_numpy(dtype=float)
    if ((y == 0) | (y == 1)).all():
        return ["every response is an exact 0 or 1: the Beta part has no data, so its mean and precision are informed by their "
                "priors alone; check sensitivity to the explicit priors"]
    return []


def _expected_response(model, post, eta):
    """brms's ``posterior_epred`` ``zoi coi + (1 - zoi) mu`` with ``mu = expit(eta)``, per draw."""
    zoi, coi = extra_draws(model, post, "zoi"), extra_draws(model, post, "coi")
    return zoi * coi + (1.0 - zoi) * expit(eta)


def _draw_response(model, post, eta, mean, df, rng):
    """A 0 or 1 where a uniform draw falls below ``zoi`` (a 1 where a second one falls below ``coi``), a ``Beta(mu phi, (1 - mu)
    phi)`` draw otherwise (``1 - mu`` as ``expit(-eta)``); a shape parameter that underflows to zero is refused, as for the Beta
    family."""
    phi, zoi, coi = auxiliary(model, post, "phi"), extra_draws(model, post, "zoi"), extra_draws(model, post, "coi")
    alpha, beta = expit(eta) * phi, expit(-eta) * phi
    if not (np.all(alpha > 0) and np.all(beta > 0)):
        raise ValueError("zero-one-inflated Beta response prediction has a shape parameter that underflowed to zero; inspect the "
                         "logit-scale draws")
    edge = rng.uniform(size=np.shape(eta)) < zoi
    one = rng.uniform(size=np.shape(eta)) < coi
    return np.where(edge, np.where(one, 1.0, 0.0), rng.beta(alpha, beta))


def _log_likelihood(model, post, eta):
    """brms's log_lik (C07): ``zero_one_inflated_beta_lpdf(y_i | expit(eta_i), phi, zoi, coi)``, phi, zoi and coi per draw."""
    from autobayes.families.pointwise import zero_one_inflated_beta_logit

    return zero_one_inflated_beta_logit(model.design.y, eta, auxiliary(model, post, "phi"), extra_draws(model, post, "zoi"),
                                        extra_draws(model, post, "coi"))


def _likelihood_lines(model) -> list[str]:
    return [scale_line(model), ZOI.prior_line(model), COI.prior_line(model), "", "Likelihood",
            "  y_i ~ ZeroOneInflatedBeta(mu_i, phi, zoi, coi); mu_i = expit(eta_i): an exact 0 or 1 with probability zoi (a 1 with",
            "  probability coi given that), Beta(mu_i phi, (1 - mu_i) phi) otherwise; the density is zoi (1 - coi) at 0, zoi coi at 1",
            "  and (1 - zoi) Beta(y_i | mu_i phi, (1 - mu_i) phi) on (0, 1) (brms's zero_one_inflated_beta); E[y_i] = zoi coi +",
            "  (1 - zoi) mu_i. 1 - mu_i is computed as expit(-eta_i); phi is the Beta part's precision, sampled as log(phi); zoi and",
            "  coi are brms's probabilities, sampled as logit(zoi) and logit(coi). Responses are proportions in [0, 1]."]


ZERO_ONE_INFLATED_BETA = FamilySpec(
    name="zero_one_inflated_beta", tag="zero-one-inflated Beta", parent="logit_mu", link="logit", link_scale="logit",
    mean_name="mean", response_kind="continuous", response_dtype=None,
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
    scale_rule=("phi is required for the zero-one-inflated Beta family; sigma and shape do not apply (its probabilities are zoi "
                "and coi)"),
    scale_missing_error="the zero_one_inflated_beta family needs a prior on the precision parameter phi: pass phi=...",
    linpred_symbol="eta", scale_phrase="log-odds", intercept_meaning="logit of the Beta part's mean",
    mean_lines=lambda model: ["  mu_i = expit(eta_i) is the mean of the Beta part (eta is on the logit scale); E[y_i] = zoi coi + "
                              "(1 - zoi) mu_i;",
                              "  phi is the Beta part's precision: Var[y_i | 0 < y_i < 1] = mu_i (1 - mu_i) / (1 + phi)."],
    likelihood_lines=_likelihood_lines,
    centred_intercept_note=_INTERCEPT_NOTE,
    deterministic_note=("logit_mu = linear predictor (the Beta part's logit mean); E[y] = zoi coi + (1 - zoi) expit(logit_mu) is "
                        "computed by predict()"),
    default_prior_link=None,                                        # logit: brms's defaults are student_t(3, 0, 2.5)
    default_scale_prior=lambda scale: Gamma(0.01, 0.01),            # brms phi: gamma(0.01, 0.01), lb 0
    extra_parameters=(ZOI, COI),                                    # E01: brms's zoi and coi, beta(1, 1), lb 0, ub 1
    addition_terms=frozenset({"weights"}),                          # E01: weights(), each row's log likelihood times its weight
    log_likelihood=_log_likelihood,                                 # C07: log_lik(), checked against the model's own density
)
