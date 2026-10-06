"""Poisson count response: log link on ``mu``, no auxiliary parameter."""
from __future__ import annotations

import numpy as np

from autobayes.families.spec import NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, FamilySpec
from autobayes.formula import check_counts


def _validate_response(spec, data) -> None:
    check_counts(f"response {spec.response!r}", data[spec.response], domain="poisson")


def _response_warnings(spec, df) -> list[str]:
    if (df[spec.response] == 0).all():
        return ["all observed counts are zero; check sensitivity to the explicit log-mean priors"]
    return []


def _inverse_link(eta):
    with np.errstate(over="ignore", invalid="ignore"):
        expected = np.exp(eta)
    if not np.isfinite(expected).all():
        raise ValueError("Poisson expected count overflow/nonfinite prediction; inspect the log-mean draws")
    return expected


def _draw_response(model, post, mean, df, rng):
    limit = np.iinfo(np.int64).max - 10 * np.sqrt(float(np.iinfo(np.int64).max))
    if np.any(mean >= limit):
        raise ValueError("Poisson response prediction exceeds the integer random-generator limit")
    return rng.poisson(mean)


POISSON = FamilySpec(
    name="poisson", tag="Poisson", parent="mu", link="log", link_scale="log", mean_name="mu",
    response_kind="count", response_dtype=np.int64,
    scale_parameter=None, fixed_parameters=NO_FIXED_PARAMETERS,
    bambi_family=lambda: "poisson",
    validate_response=_validate_response,
    response_warnings=_response_warnings,
    response_upper_bound=lambda model: None,
    inverse_link=_inverse_link,
    draw_response=_draw_response,
    manual_prediction=True, prediction_kinds=STANDARD_PREDICTION_KINDS,
    kind_error="kind must be 'mean', 'response' or 'linear'",
    sigma_rule="sigma must be omitted for Poisson",
    sigma_missing_error=None,
    linpred_symbol="eta", scale_phrase="log mean", intercept_meaning="log expected count",
    mean_lines=lambda model: ["  E[y_i] = mu_i = exp(eta_i); eta is log expected count."],
    likelihood_lines=lambda model: ["", "Likelihood", "  y_i ~ Poisson(mu_i); mu_i = exp(eta_i)",
                                    "  Input counts are integers from 0 through 1e6; zero-count rows are retained.",
                                    "  No exposure/offset, overdispersion or zero-inflation is modeled."],
    centred_intercept_note=(),
    deterministic_note="mu = exp(linear predictor); mu = exp(eta), expected count",
)
