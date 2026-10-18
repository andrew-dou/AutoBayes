"""Extended-support Beta proportions (brms ``xbeta``, Kosmidis and Zeileis 2024): proportions in [0, 1], logit link on the mean
``mu`` of the underlying Beta, its precision ``phi`` and the exceedance ``kappa`` (brms-parity E01, docs/bounded-families.md).

``B ~ Beta(mu phi, (1 - mu) phi)`` with ``mu = inv_logit(eta)`` is stretched to ``(1 + 2 kappa) B - kappa`` on ``(-kappa, 1 +
kappa)`` and censored to [0, 1], so one process gives the exact 0s, the exact 1s and the proportions in between: with ``c =
kappa / (1 + 2 kappa)``, ``P(y = 0) = I_c(mu phi, (1 - mu) phi)``, ``P(y = 1) = I_c((1 - mu) phi, mu phi)`` (``I`` the regularized
incomplete Beta function) and the density inside is ``Beta((y + kappa) / (1 + 2 kappa) | mu phi, (1 - mu) phi) / (1 + 2 kappa)``.
``E[y]`` (brms's ``posterior_epred``) is the censored variable's mean, ``1 - I_d(a, b) + (1 + 2 kappa) mu (I_d(a + 1, b) - I_c(a +
1, b)) - kappa (I_d(a, b) - I_c(a, b))`` with ``d = 1 - c``, not ``mu``. The likelihood is ``likelihoods.xbeta_family`` (the
masses through ``likelihoods.shape_betainc``, differentiable in the shapes on both paths); ``kappa`` is the family's extra
parameter of support ``"positive"`` (brms's ``real<lower=0>``, sampled as ``kappa_log__``). The logit link only (D108). Checked
against brms in the oracle case ``xbeta_varint``, not assumed.
"""
from __future__ import annotations

import numpy as np
from scipy.special import betainc, expit

from autobayes.families.spec import (NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, ExtraParameter, FamilySpec, auxiliary,
                                     extra_draws, scale_line)
from autobayes.formula import validate_unit_interval_response
from autobayes.priors import Gamma

KAPPA = ExtraParameter(name="kappa", meaning="exceedance (the stretch of the Beta beyond [0, 1])", support="positive",
                       default_prior=lambda: Gamma(0.01, 0.01))

_INTERCEPT_NOTE = (
    "  expit(a_c) is the underlying Beta's mean at that proportion-weighted predictor; it is NOT the",
    "  average expected proportion across the levels (the link is nonlinear, and E[y] is not mu).",
)


def _bambi_family():
    from autobayes.likelihoods import xbeta_family

    return xbeta_family()


def _validate_response(spec, data) -> None:
    validate_unit_interval_response(spec.response, data[spec.response], "xbeta")


def censored_mean(eta, phi, kappa):
    """``E[y]`` of brms's ``xbeta``: the mean of ``(1 + 2 kappa) B - kappa`` censored to [0, 1], ``B ~ Beta(a, b)``, ``a = expit(eta)
    phi``, ``b = expit(-eta) phi``. With ``c = kappa / (1 + 2 kappa)`` and ``d = 1 - c`` it is ``P(B > d) + int_c^d ((1 + 2 kappa) u
    - kappa) Beta(u | a, b) du``, and ``int u Beta(u | a, b) du = mu I(a + 1, b)``: so ``I_c(b, a) + (1 + 2 kappa) mu (I_d(a + 1, b)
    - I_c(a + 1, b)) - kappa (I_d(a, b) - I_c(a, b))``, the upper tail as ``I_c(b, a)`` (no subtraction from 1)."""
    eta = np.asarray(eta, dtype=float)
    mu = expit(eta)
    a, b = mu * phi, expit(-eta) * phi
    c = kappa / (1.0 + 2.0 * kappa)
    d = (1.0 + kappa) / (1.0 + 2.0 * kappa)
    return (betainc(b, a, c) + (1.0 + 2.0 * kappa) * mu * (betainc(a + 1.0, b, d) - betainc(a + 1.0, b, c))
            - kappa * (betainc(a, b, d) - betainc(a, b, c)))


def _expected_response(model, post, eta):
    """brms's ``posterior_epred``: the censored mean ``censored_mean`` per draw and row."""
    return censored_mean(eta, auxiliary(model, post, "phi"), extra_draws(model, post, "kappa"))


def _draw_response(model, post, eta, mean, df, rng):
    """``(1 + 2 kappa) B - kappa`` censored to [0, 1], ``B ~ Beta(mu phi, (1 - mu) phi)`` (``1 - mu`` as ``expit(-eta)``); a shape
    parameter that underflows to zero is refused, as for the Beta family."""
    phi, kappa = auxiliary(model, post, "phi"), extra_draws(model, post, "kappa")
    alpha, beta = expit(eta) * phi, expit(-eta) * phi
    if not (np.all(alpha > 0) and np.all(beta > 0)):
        raise ValueError("xbeta response prediction has a shape parameter that underflowed to zero; inspect the logit-scale draws")
    return np.clip((1.0 + 2.0 * kappa) * rng.beta(alpha, beta) - kappa, 0.0, 1.0)


def _log_likelihood(model, post, eta):
    """brms's log_lik (C07): ``xbeta_lpdf(y_i | expit(eta_i), phi, kappa)``, phi and kappa per draw."""
    from autobayes.families.pointwise import xbeta_logit

    return xbeta_logit(model.design.y, eta, auxiliary(model, post, "phi"), extra_draws(model, post, "kappa"))


def _likelihood_lines(model) -> list[str]:
    return [scale_line(model), KAPPA.prior_line(model), "", "Likelihood",
            "  y_i ~ XBeta(mu_i, phi, kappa) (brms's xbeta, Kosmidis and Zeileis 2024): B_i ~ Beta(mu_i phi, (1 - mu_i) phi) with",
            "  mu_i = expit(eta_i), stretched to (1 + 2 kappa) B_i - kappa and censored to [0, 1]; with c = kappa / (1 + 2 kappa),",
            "  P(y_i = 0) = I_c(mu_i phi, (1 - mu_i) phi), P(y_i = 1) = I_c((1 - mu_i) phi, mu_i phi) (I the regularized incomplete",
            "  Beta function) and the density on (0, 1) is Beta((y_i + kappa) / (1 + 2 kappa) | mu_i phi, (1 - mu_i) phi) / (1 + 2 kappa).",
            "  E[y_i] is the censored variable's mean, not mu_i. phi and kappa are sampled as their logs; 1 - mu_i is computed as",
            "  expit(-eta_i). Responses are proportions in [0, 1], exact 0s and 1s from the same process as the others."]


XBETA = FamilySpec(
    name="xbeta", tag="extended-support Beta", parent="logit_mu", link="logit", link_scale="logit", mean_name="mean",
    response_kind="continuous", response_dtype=None,
    scale_parameter="phi", bambi_scale_parameter="phi", fixed_parameters=NO_FIXED_PARAMETERS,
    bambi_family=_bambi_family,
    validate_response=_validate_response,
    response_warnings=lambda spec, df: [],
    response_upper_bound=lambda model: None,
    inverse_link=expit,
    expected_response=_expected_response,
    draw_response=_draw_response,
    manual_prediction=True, prediction_kinds=STANDARD_PREDICTION_KINDS,
    kind_error="kind must be 'mean', 'response' or 'linear'",
    scale_rule="phi is required for the xbeta family; sigma and shape do not apply (its exceedance is kappa)",
    scale_missing_error="the xbeta family needs a prior on the precision parameter phi: pass phi=...",
    linpred_symbol="eta", scale_phrase="log-odds", intercept_meaning="logit of the underlying Beta's mean",
    mean_lines=lambda model: ["  mu_i = expit(eta_i) is the mean of the underlying Beta B_i (eta is on the logit scale); phi is its",
                              "  precision and kappa how far it is stretched beyond [0, 1]; E[y_i], the mean of the censored variable, is",
                              "  computed from mu_i, phi and kappa by predict()."],
    likelihood_lines=_likelihood_lines,
    centred_intercept_note=_INTERCEPT_NOTE,
    deterministic_note=("logit_mu = linear predictor (the underlying Beta's logit mean); E[y], the censored mean, is computed by "
                        "predict()"),
    default_prior_link=None,                                        # logit: brms's defaults are student_t(3, 0, 2.5)
    default_scale_prior=lambda scale: Gamma(0.01, 0.01),            # brms phi: gamma(0.01, 0.01), lb 0
    extra_parameters=(KAPPA,),                                      # E01: brms's kappa, gamma(0.01, 0.01), lb 0
    addition_terms=frozenset({"weights"}),                          # E01: weights(), each row's log likelihood times its weight
    log_likelihood=_log_likelihood,                                 # C07: log_lik(), checked against the model's own density
)
