"""Beta response (brms ``brmsfamily("Beta")``): proportions strictly inside (0, 1), logit link on the mean ``mu``,
precision ``phi``.

brms's parameterisation is ``beta_lpdf(y | mu * phi, (1 - mu) * phi)`` with ``mu = inv_logit(eta)``: ``E[y] = mu``,
``Var[y] = mu (1 - mu) / (1 + phi)``. The likelihood is ``likelihoods.beta_logit_family``: PyMC's ``Beta(alpha, beta)`` with
``alpha = expit(eta) phi`` and ``beta = expit(-eta) phi``, the linear predictor kept as Bambi's parent ``logit_mu`` under
the identity link (as the Bernoulli and binomial families do) and the precision named ``phi`` as in brms, so the prior
is ``Priors(phi=...)`` (brms class ``phi``) and the sampled variable ``phi_log__``. Bambi's own ``beta`` family (``mu``,
``kappa``; ``kappa`` is brms's ``phi``) is not used because its deterministic ``mu`` would hold ``expit(eta)`` rather than
the linear predictor. Equality with brms is checked in the oracle case ``beta_varint``, not assumed. ``phi`` may have its own
formula under a log link (B01, docs/distributional.md; ``beta_distributional_phi``).
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from autobayes.families.spec import NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, FamilySpec, auxiliary, scale_line, scale_symbol
from autobayes.formula import validate_proportion_response
from autobayes.priors import Gamma, StudentT

_INTERCEPT_NOTE = (
    "  expit(a_c) is the expected proportion at that proportion-weighted predictor; it is NOT the",
    "  average expected proportion across the levels (the link is nonlinear).",
)


def _bambi_family():
    from autobayes.likelihoods import beta_logit_family

    return beta_logit_family()


def _validate_response(spec, data) -> None:
    validate_proportion_response(spec.response, data[spec.response], "Beta")


def _draw_response(model, post, eta, mean, df, rng):
    """``Beta(mu phi, (1 - mu) phi)`` draws with ``1 - mu`` as ``expit(-eta)``; a shape parameter that underflows to
    zero (|eta| beyond about 745) is refused rather than handed to the generator."""
    phi = auxiliary(model, post, "phi")                        # per draw, or per draw and row (B01)
    alpha, beta = expit(eta) * phi, expit(-eta) * phi
    if not (np.all(alpha > 0) and np.all(beta > 0)):
        raise ValueError("Beta response prediction has a shape parameter that underflowed to zero; inspect the logit-scale draws")
    return rng.beta(alpha, beta)


def _likelihood_lines(model) -> list[str]:
    s = scale_symbol(model)                                    # "phi", or "phi_i" with its own formula (B01)
    return [scale_line(model), "", "Likelihood",
            f"  y_i ~ Beta(mu_i {s}, (1 - mu_i) {s}); mu_i = expit(eta_i); E[y_i] = mu_i, Var[y_i] = mu_i (1 - mu_i) / (1 + {s})",
            "  brms's parameterisation beta_lpdf(y | mu * phi, (1 - mu) * phi) through PyMC's Beta(alpha, beta), with",
            "  1 - mu computed as expit(-eta); " + ("phi is the precision, sampled as log(phi)." if s == "phi" else "phi_i is row i's precision."),
            "  Responses are proportions strictly inside (0, 1); an exact 0 or 1 is refused."]


BETA = FamilySpec(
    name="beta", tag="Beta", parent="logit_mu", link="logit", link_scale="logit", mean_name="mu",
    response_kind="continuous", response_dtype=None,
    scale_parameter="phi", bambi_scale_parameter="phi", fixed_parameters=NO_FIXED_PARAMETERS,
    bambi_family=_bambi_family,
    validate_response=_validate_response,
    response_warnings=lambda spec, df: [],
    response_upper_bound=lambda model: None,
    inverse_link=expit,
    expected_response=None,
    draw_response=_draw_response,
    manual_prediction=True, prediction_kinds=STANDARD_PREDICTION_KINDS,
    kind_error="kind must be 'mean', 'response' or 'linear' ('probability' is for binomial models; here 'mean' is mu)",
    scale_rule="phi is required for the Beta family; sigma and shape do not apply",
    scale_missing_error="the Beta family needs a prior on the precision parameter phi: pass phi=...",
    linpred_symbol="eta", scale_phrase="log-odds", intercept_meaning="logit of the expected proportion",
    mean_lines=lambda model: ["  E[y_i] = mu_i = expit(eta_i) = 1 / (1 + exp(-eta_i)); eta is on the logit (log-odds) scale;",
                              f"  Var[y_i] = mu_i (1 - mu_i) / (1 + {scale_symbol(model)}): {scale_symbol(model)} is the precision."],
    likelihood_lines=_likelihood_lines,
    centred_intercept_note=_INTERCEPT_NOTE,
    deterministic_note="logit_mu = linear predictor; mu = expit(logit_mu), the expected proportion, is computed by predict()",
    default_prior_link=None,                                        # logit: brms's defaults are student_t(3, 0, 2.5)
    default_scale_prior=lambda scale: Gamma(0.01, 0.01),            # brms phi: gamma(0.01, 0.01), lb 0
    addition_terms=frozenset({"weights"}),                      # B02: weights(), verified against the frozen case beta_weighted
    distributional_link="log",                                      # B01: phi ~ z, brms's log link (beta_distributional_phi)
    default_distributional_intercept=lambda scale: StudentT(3, 0, 2.5),   # brms Intercept_phi: student_t(3, 0, 2.5) under the log link
    monotonic=True,                                                 # B03: mo(x), verified against beta_monotonic
)
