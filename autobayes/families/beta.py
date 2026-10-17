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

E06 (docs/links.md): brms's ``Beta("probit")`` (``mu = Phi(eta)``) and ``Beta("cloglog")`` (``mu = 1 - exp(-exp(eta))``) are
``_link("probit")`` and ``_link("cloglog")``, reached through ``BETA.links`` (``ab.beta(..., link="probit")``): the likelihood
``likelihoods.beta_link_family`` with ``mu`` and ``1 - mu`` each computed directly, the linear predictor the parent
``probit_mu`` / ``cloglog_mu``.
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache, partial
from types import MappingProxyType

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


def _log_likelihood(model, post, eta):
    """brms's log_lik (C07): ``log Beta(y_i | mu_i phi, (1 - mu_i) phi)``, ``1 - mu_i`` as ``expit(-eta_i)`` (D13)."""
    from autobayes.families.pointwise import beta_logit

    return beta_logit(model.design.y, eta, auxiliary(model, post, "phi"))


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
    log_likelihood=_log_likelihood,                                 # C07: log_lik(), checked against the model's own density (test_post_processing.py)
    links=MappingProxyType({"probit": lambda: _link("probit"), "cloglog": lambda: _link("cloglog")}),   # E06 (beta_probit, beta_cloglog)
)


# -- E06: the probit and cloglog links ----------------------------------------------------------------------------------------
def _link_bambi_family(link: str):
    from autobayes.likelihoods import beta_link_family

    return beta_link_family(link)


def _link_draw_response(link: str, model, post, eta, mean, df, rng):
    """``Beta(mu phi, (1 - mu) phi)`` draws with ``1 - mu`` computed directly (``links.COMPLEMENTS``); a shape parameter that
    underflows to zero is refused rather than handed to the generator."""
    from autobayes.families.links import COMPLEMENTS

    phi = auxiliary(model, post, "phi")
    alpha, beta = mean * phi, COMPLEMENTS[link](eta) * phi
    if not (np.all(alpha > 0) and np.all(beta > 0)):
        raise ValueError(f"Beta response prediction has a shape parameter that underflowed to zero; inspect the {link}-scale draws")
    return rng.beta(alpha, beta)


def _link_log_likelihood(link: str, model, post, eta):
    """brms's log_lik (C07): ``log Beta(y_i | mu_i phi, (1 - mu_i) phi)`` under the link, ``mu`` and ``1 - mu`` computed directly."""
    from autobayes.families.pointwise import beta_link

    return beta_link(model.design.y, eta, auxiliary(model, post, "phi"), link)


def _link_likelihood_lines(link: str, model) -> list[str]:
    from autobayes.families.links import LINKS

    L = LINKS[link]
    complement = "Phi(-eta_i)" if link == "probit" else "exp(-exp(eta_i))"
    return [scale_line(model), "", "Likelihood",
            f"  y_i ~ Beta(mu_i phi, (1 - mu_i) phi); mu_i = {L.probability} (brms's Beta(\"{link}\")); E[y_i] = mu_i,",
            "  Var[y_i] = mu_i (1 - mu_i) / (1 + phi); brms's parameterisation beta_lpdf(y | mu * phi, (1 - mu) * phi) with",
            f"  1 - mu_i computed as {complement}, never by subtraction, and the log-gamma of a shape parameter below 1e-10",
            "  from its logarithm, so the density is finite for every finite eta (autobayes/likelihoods.py); phi is the",
            "  precision, sampled as log(phi). Responses are proportions strictly inside (0, 1)."]


def _link_check_formula(link: str, spec) -> None:
    from autobayes.families.links import check_link_formula

    check_link_formula(spec, _link(link))


@lru_cache(maxsize=None)
def _link(link: str) -> FamilySpec:
    """The Beta family's spec under brms's probit or cloglog link (E06), verified against the frozen cases ``beta_probit`` and
    ``beta_cloglog``. ``weights()``, a formula on ``phi``, ``mo()`` and the other special terms are refused under them
    (``check_formula``, divergence D133)."""
    from autobayes.families.links import LINKS

    L = LINKS[link]
    others = {"logit": lambda: BETA, **{o: partial(_link, o) for o in ("probit", "cloglog") if o != link}}
    return replace(
        BETA, parent=f"{link}_mu", link=link, link_scale=link, bambi_family=partial(_link_bambi_family, link),
        inverse_link=L.inverse_link, draw_response=partial(_link_draw_response, link),
        scale_phrase=L.scale, intercept_meaning=f"{L.scale} of the expected proportion",
        mean_lines=lambda model: [f"  E[y_i] = mu_i = {L.probability}{L.probability_gloss}; eta is on the {L.scale} scale;",
                                  "  Var[y_i] = mu_i (1 - mu_i) / (1 + phi): phi is the precision."],
        likelihood_lines=partial(_link_likelihood_lines, link),
        centred_intercept_note=(f"  {L.inverse_of_intercept} is the expected proportion at that proportion-weighted predictor; it is NOT the",
                                "  average expected proportion across the levels (the link is nonlinear)."),
        deterministic_note=(f"{link}_mu = linear predictor; mu = {L.probability.replace('eta_i', f'{link}_mu')}, the expected "
                            "proportion, is computed by predict()"),
        addition_terms=frozenset(), distributional_link=None, default_distributional_intercept=None, monotonic=False,   # D133
        log_likelihood=partial(_link_log_likelihood, link),
        check_formula=partial(_link_check_formula, link),
        links=MappingProxyType(others),
    )
