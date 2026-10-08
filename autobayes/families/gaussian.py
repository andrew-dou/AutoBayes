"""Gaussian response (M0): identity link on ``mu``, residual SD ``sigma``."""
from __future__ import annotations

from autobayes.families.spec import NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, FamilySpec
from autobayes.formula import validate_numeric_response
from autobayes.priors import HalfStudentT, describe_prior


def _validate_response(spec, data) -> None:
    validate_numeric_response(spec.response, data[spec.response], "Gaussian")


def _draw_response(model, post, eta, mean, df, rng):
    sigma = post[model.names["sigma"]].to_numpy()
    return rng.normal(mean, sigma[..., None])


def _likelihood_lines(model) -> list[str]:
    return [f"  sigma ~ {describe_prior(model.priors.sigma, lb=0)}", "", "Likelihood", "  y_i ~ Normal(mu_i, sigma)"]


GAUSSIAN = FamilySpec(
    name="gaussian", tag="M0", parent="mu", link="identity", link_scale=None, mean_name="mu",
    response_kind="continuous", response_dtype=None,
    scale_parameter="sigma", bambi_scale_parameter="sigma", fixed_parameters=NO_FIXED_PARAMETERS,
    bambi_family=lambda: "gaussian",
    validate_response=_validate_response,
    response_warnings=lambda spec, df: [],
    response_upper_bound=lambda model: None,
    inverse_link=lambda eta: eta,
    expected_response=None,
    draw_response=_draw_response,
    manual_prediction=False, prediction_kinds=STANDARD_PREDICTION_KINDS,
    kind_error="kind must be 'mean', 'response' or 'linear'",
    scale_rule="sigma is required for the Gaussian family; shape does not apply (it is the negative binomial and Gamma families' parameter)",
    scale_missing_error="the Gaussian family needs a prior on the residual SD: pass sigma=...",
    linpred_symbol="mu", scale_phrase="response", intercept_meaning="expected response",
    mean_lines=lambda model: [],
    likelihood_lines=_likelihood_lines,
    centred_intercept_note=(),
    deterministic_note="mu = linear predictor",
    default_prior_link="identity",                                  # brms reads y as given: student_t(3, round(median(y), 1), max(2.5, round(mad(y), 1)))
    default_scale_prior=lambda scale: HalfStudentT(3, scale),       # brms sigma: student_t(3, 0, scale), lb 0
)
