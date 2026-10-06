"""Student-t response with a fixed, user-supplied ``nu > 1``: identity link on ``mu``, scale ``sigma``."""
from __future__ import annotations

from types import MappingProxyType

import numpy as np

from autobayes.families.spec import STANDARD_PREDICTION_KINDS, FamilySpec
from autobayes.formula import validate_numeric_response


def check_nu(nu) -> float:
    """The fixed degrees of freedom: a finite real scalar greater than one."""
    if isinstance(nu, (bool, np.bool_)) or not isinstance(nu, (int, float, np.integer, np.floating)) or not np.isfinite(nu) or nu <= 1:
        raise ValueError("student requires a fixed finite scalar nu > 1")
    return float(nu)


def _validate_response(spec, data) -> None:
    validate_numeric_response(spec.response, data[spec.response], "Student-t")


def _draw_response(model, post, mean, df, rng):
    sigma = post[model.names["sigma"]].to_numpy()
    return mean + sigma[..., None] * rng.standard_t(model.nu, size=mean.shape)


def _likelihood_lines(model) -> list[str]:
    return [f"  sigma ~ {model.priors.sigma}", f"  nu = {model.nu:g} (fixed, not sampled)", "", "Likelihood",
            "  y_i ~ StudentT(nu, mu_i, sigma); E[y_i] = mu_i",
            "  sigma is scale, not residual SD; variance = sigma^2 * nu/(nu-2) for nu>2",
            "  variance is infinite for 1<nu<=2. Learned nu is not supported."]


STUDENT = FamilySpec(
    name="student", tag="Student-t", parent="mu", link="identity", link_scale=None, mean_name="mu",
    response_kind="continuous", response_dtype=None,
    scale_parameter="sigma", bambi_scale_parameter="sigma", fixed_parameters=MappingProxyType({"nu": check_nu}),
    bambi_family=lambda: "t",
    validate_response=_validate_response,
    response_warnings=lambda spec, df: [],
    response_upper_bound=lambda model: None,
    inverse_link=lambda eta: eta,
    draw_response=_draw_response,
    manual_prediction=False, prediction_kinds=STANDARD_PREDICTION_KINDS,
    kind_error="kind must be 'mean', 'response' or 'linear'",
    scale_rule="sigma is required for the Student-t family; shape does not apply (it is the negative binomial and Gamma families' parameter)",
    scale_missing_error="the Student-t family needs a prior on the scale: pass sigma=...",
    linpred_symbol="mu", scale_phrase="response", intercept_meaning="expected response",
    mean_lines=lambda model: [],
    likelihood_lines=_likelihood_lines,
    centred_intercept_note=(),
    deterministic_note="mu = linear predictor",
)
