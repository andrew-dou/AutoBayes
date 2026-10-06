"""Bernoulli response (M1): logit link on ``logit_p``, the stable hand-written likelihood of ``autobayes.likelihoods``."""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from autobayes.families.spec import LOGIT_INTERCEPT_NOTE, NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, FamilySpec
from autobayes.formula import validate_binary_response


def _bambi_family():
    from autobayes.likelihoods import bernoulli_logit_family

    return bernoulli_logit_family()   # stable likelihood; see likelihoods.py


def _validate_response(spec, data) -> None:
    validate_binary_response(spec.response, data[spec.response])


def _draw_response(model, post, mean, df, rng):
    return rng.binomial(1, mean).astype(np.int64)


BERNOULLI = FamilySpec(
    name="bernoulli", tag="M1", parent="logit_p", link="logit", link_scale="logit", mean_name="p",
    response_kind="binary", response_dtype=np.int64,
    scale_parameter=None, bambi_scale_parameter=None, fixed_parameters=NO_FIXED_PARAMETERS,
    bambi_family=_bambi_family,
    validate_response=_validate_response,
    response_warnings=lambda spec, df: [],
    response_upper_bound=lambda model: 1,
    inverse_link=expit,
    draw_response=_draw_response,
    manual_prediction=False, prediction_kinds=STANDARD_PREDICTION_KINDS,
    kind_error="kind must be 'mean', 'response' or 'linear' ('probability' is for binomial models; here 'mean' is p)",
    scale_rule="sigma and shape must be omitted for Bernoulli (no auxiliary parameter)",
    scale_missing_error=None,
    linpred_symbol="eta", scale_phrase="log-odds", intercept_meaning="log-odds of y = 1",
    mean_lines=lambda model: ["  P(y_i = 1) = p_i = expit(eta_i) = 1 / (1 + exp(-eta_i)); eta is on the log-odds scale."],
    likelihood_lines=lambda model: [
        "", "Likelihood", "  y_i ~ Bernoulli(p_i), sampled as log p(y_i | eta_i) = y_i * eta_i - softplus(eta_i)",
        "  with a softplus whose value and gradient are exact for every finite eta on both the",
        "  PyTensor and the JAX path (autobayes/likelihoods.py; Bambi's built-in bernoulli family",
        "  is not used because its likelihood is floored at machine epsilon for |eta| > ~37).",
    ],
    centred_intercept_note=LOGIT_INTERCEPT_NOTE,
    deterministic_note="logit_p = linear predictor; p = expit(logit_p) are computed by predict()",
)
