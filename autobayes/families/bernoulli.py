"""Bernoulli response (M1): a strictly 0/1 outcome under the logit link, or the probit or cloglog link on request
(brms-parity A12 and P6, docs/links.md), each with a stable hand-written likelihood from ``autobayes.likelihoods``.

One spec per link: ``_spec(link)`` builds it, the registry holds the logit spec ``BERNOULLI``, and the other links' specs
are reached through ``FamilySpec.links`` (``get_family("bernoulli", "probit")``, ``ab.bernoulli(..., link="cloglog")``).
The linear predictor is the parent ``logit_p`` / ``probit_p`` / ``cloglog_p``; ``p`` is its inverse link. brms's cauchit
link is not offered.
"""
from __future__ import annotations

from functools import lru_cache, partial
from types import MappingProxyType

import numpy as np

from autobayes.families.links import LINKS, PARKED_LINKS
from autobayes.families.spec import NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, FamilySpec
from autobayes.formula import validate_binary_response


def _bambi_family(link: str):
    from autobayes.likelihoods import bernoulli_link_family, bernoulli_logit_family

    return bernoulli_logit_family() if link == "logit" else bernoulli_link_family(link)   # stable likelihoods; see likelihoods.py


def _validate_response(spec, data) -> None:
    validate_binary_response(spec.response, data[spec.response])


def _draw_response(model, post, eta, mean, df, rng):
    return rng.binomial(1, mean).astype(np.int64)


_LIKELIHOOD_LINES = {
    "logit": [
        "", "Likelihood", "  y_i ~ Bernoulli(p_i), sampled as log p(y_i | eta_i) = y_i * eta_i - softplus(eta_i)",
        "  with a softplus whose value and gradient are exact for every finite eta on both the",
        "  PyTensor and the JAX path (autobayes/likelihoods.py; Bambi's built-in bernoulli family",
        "  is not used because its likelihood is floored at machine epsilon for |eta| > ~37).",
    ],
    "probit": [
        "", "Likelihood", "  y_i ~ Bernoulli(p_i), sampled as log p(y_i | eta_i) = y_i log Phi(eta_i) + (1 - y_i) log Phi(-eta_i)",
        "  with a Normal log-CDF whose value and gradient are finite for every finite eta on both the",
        "  PyTensor and the JAX path (autobayes/likelihoods.py; Stan's Phi(eta) is exactly 0 below",
        "  eta = -37.5 and exactly 1 above 8.25, where brms's bernoulli_lpmf(y | Phi(eta)) is -inf).",
    ],
    "cloglog": [
        "", "Likelihood", "  y_i ~ Bernoulli(p_i), sampled as log p(y_i | eta_i) = y_i log(1 - exp(-exp(eta_i))) - (1 - y_i) exp(eta_i)",
        "  with log(1 - exp(-exp(eta))) as log1mexp(-exp(eta)), finite with a finite gradient for every finite eta on both",
        "  the PyTensor and the JAX path (autobayes/likelihoods.py; brms's bernoulli_lpmf(y | inv_cloglog(eta)) forms",
        "  1 - inv_cloglog(eta) by subtraction, which loses eps * exp(exp(eta)) relative and is 0 above eta = 3.6).",
    ],
}


@lru_cache(maxsize=None)
def _spec(link: str) -> FamilySpec:
    L = LINKS[link]
    return FamilySpec(
        name="bernoulli", tag="M1", parent=L.parent, link=link, link_scale=link, mean_name="p",
        response_kind="binary", response_dtype=np.int64,
        scale_parameter=None, bambi_scale_parameter=None, fixed_parameters=NO_FIXED_PARAMETERS,
        bambi_family=partial(_bambi_family, link),
        validate_response=_validate_response,
        response_warnings=lambda spec, df: [],
        response_upper_bound=lambda model: 1,
        inverse_link=L.inverse_link,
        expected_response=None,
        draw_response=_draw_response,
        manual_prediction=False, prediction_kinds=STANDARD_PREDICTION_KINDS,
        kind_error="kind must be 'mean', 'response' or 'linear' ('probability' is for binomial models; here 'mean' is p)",
        scale_rule="sigma and shape must be omitted for Bernoulli (no auxiliary parameter)",
        scale_missing_error=None,
        linpred_symbol="eta", scale_phrase=L.scale, intercept_meaning=f"{L.of_probability} y = 1",
        mean_lines=lambda model: [f"  P(y_i = 1) = p_i = {L.probability}{L.probability_gloss}; eta is on the {L.scale} scale."],
        likelihood_lines=lambda model: list(_LIKELIHOOD_LINES[link]),
        centred_intercept_note=L.centred_intercept_note,
        deterministic_note=f"{L.parent} = linear predictor; p = {L.probability.replace('eta_i', L.parent)} are computed by predict()",
        default_prior_link=None,                                    # logit, probit and cloglog: brms's defaults are student_t(3, 0, 2.5)
        default_scale_prior=None,
        links=MappingProxyType({other: partial(_spec, other) for other in LINKS if other != link}),
        parked_links=PARKED_LINKS,
        addition_terms=frozenset({"weights"}),                  # B02: weights(), verified against the frozen case bernoulli_weighted (logit; the probit link's likelihood is weighted the same way)
        monotonic=True,                                             # B03: mo(x), verified against bernoulli_monotonic (logit; under probit the term enters the same linear predictor)
    )


BERNOULLI = _spec("logit")
