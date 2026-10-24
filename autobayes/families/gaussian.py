"""Gaussian response (M0): identity link on ``mu``, residual SD ``sigma`` (or ``sigma``'s own formula under a log link, B01)."""
from __future__ import annotations

from autobayes.families.spec import NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, FamilySpec, auxiliary, scale_line, scale_symbol
from autobayes.formula import validate_numeric_response
from autobayes.priors import HalfStudentT, StudentT


def _validate_response(spec, data) -> None:
    validate_numeric_response(spec.response, data[spec.response], "Gaussian")


def _scale(model, post, df):
    """Each draw's and row's scale: sigma per draw, or per draw and row (B01); under se() (F06a) each row's known standard error
    alone (se(col): no sigma) or with sigma in quadrature (se(col, sigma = TRUE)), the rows of ``df`` (None: the fitted rows)."""
    from autobayes.se_rate import has_scale, row_scale, standard_errors

    return row_scale(model, auxiliary(model, post, "sigma") if has_scale(model) else None, standard_errors(model, df))


def _draw_response(model, post, eta, mean, df, rng):
    return rng.normal(mean, _scale(model, post, df))


def _log_likelihood(model, post, eta):
    """brms's log_lik (C07): ``log Normal(y_i | mu_i, sigma)``, sigma per draw (or per draw and row, B01); under se() (F06a) the
    row's scale is its standard error, or sqrt(sigma^2 + se_i^2)."""
    from autobayes.families.pointwise import normal

    return normal(model.design.y, eta, _scale(model, post, None))


def _likelihood_lines(model) -> list[str]:
    from autobayes.se_rate import has_scale, scale_text

    term = getattr(model.spec, "fcor", None)                        # F03a: the rows' joint normal with the fixed covariance
    if term is not None:
        n = model.design.n
        return [scale_line(model), "", "Likelihood",
                f"  y ~ MultiNormal(mu, sigma^2 * {term.matrix}) over the {n} fitted rows: {term.term}, brms's normal_fcor_hom; {term.matrix} is",
                f"  the known {n} x {n} residual covariance from data2 (brms's Mfcor), read in data order; its Cholesky factor L (brms's",
                "  Lfcor) gives z = L^-1 (y - mu) / sigma and log p(y) = sum_i (-z_i^2 / 2 - log sigma - log L_ii - log(2 pi) / 2)"]
    return ([scale_line(model)] if has_scale(model) else []) + ["", "Likelihood",
                                                                f"  y_i ~ Normal(mu_i, {scale_text(model, scale_symbol(model))})"]


def _tails():
    from autobayes.likelihoods import NORMAL_TAILS        # lazily, as the families that build their likelihood import it

    return NORMAL_TAILS


def _time_correlation():
    from autobayes.timecor import NORMAL_TIME             # lazily, as autobayes.timecor imports the formula module

    return NORMAL_TIME


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
    addition_terms=frozenset({"weights", "cens", "trunc", "mi", "se"}),   # B02: weights(); cens() and trunc() since P6 (two references, D24, D25); mi() (C06); se() (F06a: gauss_se_meta, gauss_se_sigma)
    tails=_tails,
    se_parameter="sigma",                                           # F06a: se(col) is the row's sigma, or adds to it in quadrature
    distributional_link="log",                                      # B01: sigma ~ z, brms's log link
    default_distributional_intercept=lambda scale: StudentT(3, 0, scale),   # brms Intercept_sigma: student_t(3, 0, scale of log y)
    monotonic=True,                                                 # B03: mo(x), verified against gauss_monotonic
    log_likelihood=_log_likelihood,                                 # C07: log_lik(), verified against gauss_varint_post
    multivariate=True,                                              # C01: mvbind(y, y2) ~ ..., verified against gauss_mv_rescor and gauss_mv_norescor
    nonlinear_likelihood=_nonlinear_likelihood,                     # C02: nl = TRUE, verified against gauss_nonlinear_exp and _terms
    autocorrelation=True,                                           # C04: ar(), ma(), arma(), verified against gauss_ar1, gauss_arma11
    sar=True,                                                       # F02b: sar(M, type = ), verified against gauss_sar_lag, gauss_sar_error
    fcor=True,                                                      # F03a: fcor(M), verified against gauss_fcor and gauss_fcor_varint
    car=True,                                                       # F02a: car(), verified against gauss_car_escar, _esicar, _icar, _bym2
    time_correlation=_time_correlation,                             # F03b: unstr(), cosy(), verified against gauss_unstr, gauss_cosy
    measurement_error=True,                                         # C06: me(x, sdx), verified against gauss_me
    missing_values=True,                                            # C06: mi(col) with col | mi() ~ ..., verified against gauss_mi
    shrinkage=True,                                                 # D07: horseshoe() and R2D2() on b, verified against gauss_horseshoe and gauss_r2d2
)
