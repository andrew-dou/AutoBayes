"""Binomial response (M2): ``successes | trials(n)`` with a logit link on ``logit_p``, the stable likelihood of
``autobayes.likelihoods``. Zero-trial rows are dropped by the model before the design is built."""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from autobayes.families.spec import LOGIT_INTERCEPT_NOTE, NO_FIXED_PARAMETERS, FamilySpec
from autobayes.formula import binomial_counts


def _bambi_family():
    from autobayes.likelihoods import binomial_logit_family

    return binomial_logit_family()


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


def _draw_response(model, post, mean, df, rng):
    trials = df[model.spec.trials].to_numpy(dtype=np.int64)
    return rng.binomial(trials, mean).astype(np.int64)


def _mean_lines(model) -> list[str]:
    s = model.spec
    return [f"  P(success) = p_i = expit(eta_i); {s.response}_i ~ Binomial({s.trials}_i, p_i); eta is on the log-odds scale."]


def _likelihood_lines(model) -> list[str]:
    s = model.spec
    return [
        "", "Likelihood",
        f"  log p({s.response}_i | {s.trials}_i, eta_i) = log C({s.trials}_i, {s.response}_i)"
        f" - {s.response}_i softplus(-eta_i) - ({s.trials}_i - {s.response}_i) softplus(eta_i)",
        "  with log C computed inside the graph from the evaluated value and a softplus whose gradient",
        "  is finite on both the PyTensor and the JAX path. Validated (docs/m2-decision.md) for counts",
        "  up to 1e6 per row at logits to 1e6 against a 60-digit reference; larger counts are refused.",
        "  Zero-trial rows are dropped before design construction; centering, when effective, uses only fitted rows.",
    ]


BINOMIAL = FamilySpec(
    name="binomial", tag="M2", parent="logit_p", link="logit", link_scale="logit", mean_name="p",
    response_kind="binomial", response_dtype=np.int64,
    scale_parameter=None, bambi_scale_parameter=None, fixed_parameters=NO_FIXED_PARAMETERS,
    bambi_family=_bambi_family,
    validate_response=_validate_response,
    response_warnings=_response_warnings,
    response_upper_bound=lambda model: model.design.trials,
    inverse_link=expit,
    draw_response=_draw_response,
    manual_prediction=True, prediction_kinds=frozenset({"probability", "mean", "response", "linear"}),
    kind_error="kind must be 'probability', 'mean', 'response' or 'linear' for a binomial model",
    scale_rule="sigma and shape must be omitted for binomial (no auxiliary parameter)",
    scale_missing_error=None,
    linpred_symbol="eta", scale_phrase="log-odds", intercept_meaning="per-trial success log-odds",
    mean_lines=_mean_lines,
    likelihood_lines=_likelihood_lines,
    centred_intercept_note=LOGIT_INTERCEPT_NOTE,
    deterministic_note="logit_p = linear predictor; p = expit(logit_p), expected = n p and count draws are computed by predict()",
)
