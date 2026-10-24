"""Student-t response: identity link on ``mu``, scale ``sigma``, and degrees of freedom ``nu`` either fixed by the user
(``nu > 1``) or learned under its own prior truncated to ``nu > 1`` (brms-parity A13, docs/student-learned-nu.md). ``sigma``
may have its own formula under a log link (B01, docs/distributional.md)."""
from __future__ import annotations

from types import MappingProxyType

import numpy as np

from autobayes.families.spec import STANDARD_PREDICTION_KINDS, FamilySpec, auxiliary, scale_line, scale_symbol
from autobayes.formula import validate_numeric_response
from autobayes.priors import Gamma, HalfStudentT, StudentT, describe_prior

NU_LOWER = 1.0                    # brms's real<lower=1> nu: a learned nu's prior is truncated here, and it is sampled as log(nu - 1)


def check_nu(nu) -> float:
    """The fixed degrees of freedom: a finite real scalar greater than one."""
    if isinstance(nu, (bool, np.bool_)) or not isinstance(nu, (int, float, np.integer, np.floating)) or not np.isfinite(nu) or nu <= 1:
        raise ValueError("student requires a fixed finite scalar nu > 1")
    return float(nu)


def _validate_response(spec, data) -> None:
    validate_numeric_response(spec.response, data[spec.response], "Student-t")


def _scale(model, post, df):
    """Each draw's and row's scale: sigma per draw, or per draw and row (B01); under se() (F06a) each row's known standard error
    alone (se(col): no sigma) or with sigma in quadrature (se(col, sigma = TRUE)), the rows of ``df`` (None: the fitted rows)."""
    from autobayes.se_rate import has_scale, row_scale, standard_errors

    return row_scale(model, auxiliary(model, post, "sigma") if has_scale(model) else None, standard_errors(model, df))


def _draw_response(model, post, eta, mean, df, rng):
    sigma = _scale(model, post, df)                            # per draw, or per draw and row (B01; F06a's se())
    if model.nu is None:                                       # learned: each posterior draw's own degrees of freedom
        nu = post[model.names["nu"]].to_numpy()[..., None]
        return mean + sigma * rng.standard_t(nu, size=mean.shape)
    return mean + sigma * rng.standard_t(model.nu, size=mean.shape)


def _log_likelihood(model, post, eta):
    """brms's log_lik (C07): ``log StudentT(y_i | nu, mu_i, sigma)``, a learned nu per draw (A13)."""
    from autobayes.families.pointwise import student_t

    nu = post[model.names["nu"]].to_numpy()[..., None] if model.nu is None else model.nu
    return student_t(model.design.y, nu, eta, _scale(model, post, None))      # F06a: se()'s row scale


def _time_correlation():
    from autobayes.timecor import STUDENT_TIME            # lazily, as autobayes.timecor imports the formula module

    return STUDENT_TIME


def _likelihood_lines(model) -> list[str]:
    from autobayes.se_rate import has_scale, scale_text

    sigma = scale_text(model, scale_symbol(model))             # "sigma", or "sigma_i" with its own formula (B01); F06a: se()'s
    head = [scale_line(model)] if has_scale(model) else []     # (no sigma under se(col) without sigma = TRUE)
    if model.nu is None:                                       # learned (A13)
        return [*head,
                f"  nu ~ {describe_prior(model.priors.nu, lb=NU_LOWER)}, truncated to nu > {NU_LOWER:g} and renormalised there (brms's real<lower=1> nu:",
                f"       the prior minus its log probability of nu > {NU_LOWER:g}); nu is learned, sampled as log(nu - {NU_LOWER:g})",
                "", "Likelihood",
                f"  y_i ~ StudentT(nu, mu_i, {sigma}); E[y_i] = mu_i",
                "  sigma is scale, not residual SD; variance = sigma^2 * nu/(nu-2) for nu>2",
                "  variance is infinite for 1<nu<=2."]
    return [*head, f"  nu = {model.nu:g} (fixed, not sampled)", "", "Likelihood",
            f"  y_i ~ StudentT(nu, mu_i, {sigma}); E[y_i] = mu_i",
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
    expected_response=None,
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
    learnable_parameters=MappingProxyType({"nu": NU_LOWER}),
    default_prior_link="identity",                                  # brms reads y as given, as for the Gaussian
    default_scale_prior=lambda scale: HalfStudentT(3, scale),       # brms sigma: student_t(3, 0, scale), lb 0
    default_learned_priors=MappingProxyType({"nu": lambda: Gamma(2, 0.1)}),   # brms nu: gamma(2, 0.1), lb 1
    addition_terms=frozenset({"weights", "se"}),                # B02: weights(), verified against the frozen case student_weighted; F06a: se() (student_se_sigma)
    se_parameter="sigma",                                           # F06a: se(col) is the row's sigma, or adds to it in quadrature
    distributional_link="log",                                      # B01: sigma ~ z, brms's log link (student_distributional_sigma)
    default_distributional_intercept=lambda scale: StudentT(3, 0, scale),   # brms Intercept_sigma: student_t(3, 0, scale of log y)
    monotonic=True,                                                 # B03: mo(x), verified against student_monotonic
    log_likelihood=_log_likelihood,                                 # C07: log_lik(), checked against the model's own density (test_post_processing.py)
    time_correlation=_time_correlation,                             # F03b: cosy(), verified against student_cosy and student_cosy_post
)
