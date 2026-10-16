"""Gaussian response (M0): identity link on ``mu``, residual SD ``sigma`` (or ``sigma``'s own formula under a log link, B01)."""
from __future__ import annotations

from autobayes.families.spec import NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, FamilySpec, auxiliary, scale_line, scale_symbol
from autobayes.formula import validate_numeric_response
from autobayes.priors import HalfStudentT, StudentT


def _validate_response(spec, data) -> None:
    validate_numeric_response(spec.response, data[spec.response], "Gaussian")


def _draw_response(model, post, eta, mean, df, rng):
    return rng.normal(mean, auxiliary(model, post, "sigma"))       # sigma per draw, or per draw and row (B01)


def _log_likelihood(model, post, eta):
    """brms's log_lik (C07): ``log Normal(y_i | mu_i, sigma)``, sigma per draw (or per draw and row, B01)."""
    from autobayes.families.pointwise import normal

    return normal(model.design.y, eta, auxiliary(model, post, "sigma"))


def _likelihood_lines(model) -> list[str]:
    return [scale_line(model), "", "Likelihood", f"  y_i ~ Normal(mu_i, {scale_symbol(model)})"]


def _tails():
    from autobayes.likelihoods import NORMAL_TAILS        # lazily, as the families that build their likelihood import it

    return NORMAL_TAILS


def _nonlinear_likelihood(model, mu, scales):
    """A non-linear model's likelihood (C02, docs/nonlinear.md): ``y ~ Normal(mu, sigma)`` with ``mu`` the non-linear predictor
    under the identity link, brms's ``normal_lpdf(Y | mu, sigma)``."""
    import pymc as pm

    return pm.Normal(model.names["y"], mu=mu, sigma=scales["sigma"], observed=model.design.y, dims="__obs__")


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
    addition_terms=frozenset({"weights", "cens", "trunc", "mi"}),   # B02: weights(); cens() and trunc() since P6 (two references, D24, D25); mi() (C06)
    tails=_tails,
    distributional_link="log",                                      # B01: sigma ~ z, brms's log link
    default_distributional_intercept=lambda scale: StudentT(3, 0, scale),   # brms Intercept_sigma: student_t(3, 0, scale of log y)
    monotonic=True,                                                 # B03: mo(x), verified against gauss_monotonic
    log_likelihood=_log_likelihood,                                 # C07: log_lik(), verified against gauss_varint_post
    multivariate=True,                                              # C01: mvbind(y, y2) ~ ..., verified against gauss_mv_rescor and gauss_mv_norescor
    nonlinear_likelihood=_nonlinear_likelihood,                     # C02: nl = TRUE, verified against gauss_nonlinear_exp and _terms
    autocorrelation=True,                                           # C04: ar(), ma(), arma(), verified against gauss_ar1, gauss_arma11
    measurement_error=True,                                         # C06: me(x, sdx), verified against gauss_me
    missing_values=True,                                            # C06: mi(col) with col | mi() ~ ..., verified against gauss_mi
    shrinkage=True,                                                 # D07: horseshoe() and R2D2() on b, verified against gauss_horseshoe and gauss_r2d2
)
