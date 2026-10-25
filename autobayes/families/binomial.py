"""Binomial response (M2): ``successes | trials(n)`` under the logit link, or the probit or cloglog link on request
(brms-parity A12 and P6, docs/links.md), each with a stable likelihood from ``autobayes.likelihoods``. Zero-trial rows
are dropped by the model before the design is built.

One spec per link, as for the Bernoulli family: ``_spec(link)`` builds it, the registry holds the logit spec
``BINOMIAL``, the other links' specs are reached through ``FamilySpec.links``; brms's cauchit link is not offered."""
from __future__ import annotations

from functools import lru_cache, partial
from types import MappingProxyType

import numpy as np

from autobayes.families.links import LINKS, PARKED_LINKS
from autobayes.families.spec import NO_FIXED_PARAMETERS, FamilySpec
from autobayes.formula import binomial_counts


def _bambi_family(link: str):
    from autobayes.likelihoods import binomial_link_family, binomial_logit_family

    return binomial_logit_family() if link == "logit" else binomial_link_family(link)


def _validate_response(spec, data) -> None:
    binomial_counts(spec, data)


def _response_warnings(spec, df) -> list[str]:
    k, n = df[spec.response].to_numpy(), df[spec.trials].to_numpy()
    if not (np.all(k == 0) or np.all(k == n)):
        return []
    which = "zero successes" if np.all(k == 0) else "all successes"
    return [
        (f"every fitted row has {which}. There is no population intercept; any separating direction "
         "depends on the design. Check coefficient-prior sensitivity, without assuming an intercept direction.")
        if not spec.intercept else
        f"every fitted row has {which}: the data are separated along the intercept "
        "direction, the likelihood is monotone there, and the posterior in that direction "
        "is set by the intercept prior's tail and the data's extent. Check prior sensitivity "
        "(refit with a wider and a narrower intercept prior and report the movement)."
    ]


def _draw_response(model, post, eta, mean, df, rng):
    trials = df[model.spec.trials].to_numpy(dtype=np.int64)
    return rng.binomial(trials, mean).astype(np.int64)


def _log_likelihood(link: str, model, post, eta):
    """brms's log_lik (C07): ``log Binomial(y_i | n_i, p_i)``, ``p_i`` the link's inverse at ``eta_i``, from the link's stable
    log probabilities."""
    from autobayes.families.pointwise import binomial

    return binomial(model.design.y, model.design.trials, eta, link)


def _mean_lines(link: str, model) -> list[str]:
    s, L = model.spec, LINKS[link]
    return [f"  P(success) = p_i = {L.probability}; {s.response}_i ~ Binomial({s.trials}_i, p_i); eta is on the {L.scale} scale."]


def _likelihood_lines(link: str, model) -> list[str]:
    s = model.spec
    if link == "logit":
        return [
            "", "Likelihood",
            f"  log p({s.response}_i | {s.trials}_i, eta_i) = log C({s.trials}_i, {s.response}_i)"
            f" - {s.response}_i softplus(-eta_i) - ({s.trials}_i - {s.response}_i) softplus(eta_i)",
            "  with log C computed inside the graph from the evaluated value and a softplus whose gradient",
            "  is finite on both the PyTensor and the JAX path. Validated for counts",
            "  up to 1e6 per row at logits to 1e6 against a 60-digit reference; larger counts are refused.",
            "  Zero-trial rows are dropped before design construction; centering, when effective, uses only fitted rows.",
        ]
    k, n = s.response, s.trials
    counts = {"probit": f" + {k}_i log Phi(eta_i) + ({n}_i - {k}_i) log Phi(-eta_i)",
              "cloglog": f" + {k}_i log(1 - exp(-exp(eta_i))) - ({n}_i - {k}_i) exp(eta_i)"}[link]
    return [
        "", "Likelihood",
        f"  log p({k}_i | {n}_i, eta_i) = log C({n}_i, {k}_i)" + counts,
        "  with log C computed inside the graph from the evaluated value (as for the logit link) and log probabilities",
        "  whose value and gradient are finite for every finite eta on both the PyTensor and the JAX path",
        "  (autobayes/likelihoods.py); counts above 1e6 per row are refused as for the logit link.",
        "  Zero-trial rows are dropped before design construction; centering, when effective, uses only fitted rows.",
    ]


@lru_cache(maxsize=None)
def _spec(link: str) -> FamilySpec:
    L = LINKS[link]
    return FamilySpec(
        name="binomial", tag="binomial", parent=L.parent, link=link, link_scale=link, mean_name="p",
        response_kind="binomial", response_dtype=np.int64,
        scale_parameter=None, bambi_scale_parameter=None, fixed_parameters=NO_FIXED_PARAMETERS,
        bambi_family=partial(_bambi_family, link),
        validate_response=_validate_response,
        response_warnings=_response_warnings,
        response_upper_bound=lambda model: model.design.trials,
        inverse_link=L.inverse_link,
        expected_response=None,
        draw_response=_draw_response,
        manual_prediction=True, prediction_kinds=frozenset({"probability", "mean", "response", "linear"}),
        kind_error="kind must be 'probability', 'mean', 'response' or 'linear' for a binomial model",
        scale_rule="sigma and shape must be omitted for binomial (no auxiliary parameter)",
        scale_missing_error=None,
        linpred_symbol="eta", scale_phrase=L.scale, intercept_meaning=L.trial_success,
        mean_lines=partial(_mean_lines, link),
        likelihood_lines=partial(_likelihood_lines, link),
        centred_intercept_note=L.centred_intercept_note,
        deterministic_note=f"{L.parent} = linear predictor; p = {L.probability.replace('eta_i', L.parent)}, expected = n p and count draws are computed by predict()",
        default_prior_link=None,                                    # logit, probit and cloglog: brms's defaults are student_t(3, 0, 2.5)
        default_scale_prior=None,
        links=MappingProxyType({other: partial(_spec, other) for other in LINKS if other != link}),
        parked_links=PARKED_LINKS,
        addition_terms=frozenset({"weights"}),                  # B02: weights(), verified against the frozen case binomial_weighted (logit; the probit link's likelihood is weighted the same way)
        monotonic=True,                                             # B03: mo(x), verified against binomial_monotonic (logit; under probit the term enters the same linear predictor)
        log_likelihood=partial(_log_likelihood, link),                # C07: log_lik(), verified against binomial_weighted_post (logit)
    )


BINOMIAL = _spec("logit")
