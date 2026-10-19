"""Lognormal response (brms ``lognormal()``): identity link on the log-scale location ``mu`` (the linear predictor),
scale ``sigma``.

``log(y) ~ Normal(mu, sigma)``, brms's ``lognormal_lpdf(y | mu, sigma)``: ``E[y] = exp(mu + sigma^2 / 2)``, the median
is ``exp(mu)`` and ``Var[y] = (exp(sigma^2) - 1) exp(2 mu + sigma^2)``. The coefficients act on ``log y`` and the centred
intercept is the expected log response at the predictor means. Bambi has no plain lognormal family, so
``likelihoods.lognormal_family`` wraps PyMC's ``LogNormal(mu, sigma)`` the way Bambi's ``gaussian`` wraps ``Normal``; the
prior is ``Priors(sigma=...)`` (brms class ``sigma``) and the sampled variable ``sigma_log__``. ``predict(kind="mean")``
returns brms's ``posterior_epred`` ``exp(mu + sigma^2 / 2)`` under the name ``mean`` (not ``exp(mu)``, the median, and not
``mu``, which is the location); ``kind="linear"`` returns ``mu``. Checked against brms in the oracle case
``lognormal_varint``, not assumed. ``sigma`` may have its own formula under a log link (B01, docs/distributional.md;
``lognormal_distributional_sigma``), and the mean then uses each row's ``sigma``. E04 (docs/cens-trunc-families.md):
``cens(c)`` and ``cens(c, y2)``, the Normal tails of ``log y`` (``autobayes.tails``; ``lognormal_censored``).
"""
from __future__ import annotations

from autobayes.families.spec import NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, FamilySpec, auxiliary, exp_mean, scale_line, scale_symbol
from autobayes.formula import validate_positive_response
from autobayes.priors import HalfStudentT, StudentT


def _bambi_family():
    from autobayes.likelihoods import lognormal_family

    return lognormal_family()


def _validate_response(spec, data) -> None:
    validate_positive_response(spec.response, data[spec.response], "lognormal")


def _expected_response(model, post, eta):
    """brms's ``posterior_epred`` ``exp(mu + sigma^2 / 2)`` per draw (and row, when sigma has its own formula, B01), refused
    rather than clipped when it overflows."""
    return exp_mean(eta + 0.5 * auxiliary(model, post, "sigma") ** 2, "lognormal expected response")


def _draw_response(model, post, eta, mean, df, rng):
    """``LogNormal(mu, sigma)`` draws with ``mu`` the linear predictor (numpy's ``lognormal(mean, sigma)`` of ``log y``)."""
    return rng.lognormal(eta, auxiliary(model, post, "sigma"))


def _log_likelihood(model, post, eta):
    """brms's log_lik (C07): ``log LogNormal(y_i | mu_i, sigma)``, sigma per draw (or per draw and row, B01)."""
    from autobayes.families.pointwise import lognormal

    return lognormal(model.design.y, eta, auxiliary(model, post, "sigma"))


def _tails():
    from autobayes.tails import LOGNORMAL_TAILS

    return LOGNORMAL_TAILS


def _likelihood_lines(model) -> list[str]:
    s = scale_symbol(model)                                    # "sigma", or "sigma_i" with its own formula (B01)
    return [scale_line(model), "", "Likelihood",
            f"  y_i ~ LogNormal(mu_i, {s}), i.e. log(y_i) ~ Normal(mu_i, {s}); brms's lognormal_lpdf(y | mu, sigma)",
            f"  E[y_i] = exp(mu_i + {s}^2 / 2), median exp(mu_i), Var[y_i] = (exp({s}^2) - 1) exp(2 mu_i + {s}^2);",
            ("  sigma is the SD of log y, sampled as log(sigma)." if s == "sigma" else "  sigma_i is the SD of log y_i.")
            + " predict(kind='mean') returns E[y_i]; kind='linear' returns mu_i.",
            "  Responses are strictly positive finite reals."]


LOGNORMAL = FamilySpec(
    name="lognormal", tag="lognormal", parent="mu", link="identity", link_scale="log", mean_name="mean",
    response_kind="continuous", response_dtype=None,
    scale_parameter="sigma", bambi_scale_parameter="sigma", fixed_parameters=NO_FIXED_PARAMETERS,
    bambi_family=_bambi_family,
    validate_response=_validate_response,
    response_warnings=lambda spec, df: [],
    response_upper_bound=lambda model: None,
    inverse_link=lambda eta: eta,
    expected_response=_expected_response,
    draw_response=_draw_response,
    manual_prediction=True, prediction_kinds=STANDARD_PREDICTION_KINDS,
    kind_error="kind must be 'mean', 'response' or 'linear'",
    scale_rule="sigma is required for the lognormal family; no other auxiliary parameter applies",
    scale_missing_error="the lognormal family needs a prior on the SD of log y: pass sigma=...",
    linpred_symbol="mu", scale_phrase="log response", intercept_meaning="expected log response",
    mean_lines=lambda model: [f"  log(y_i) ~ Normal(mu_i, {scale_symbol(model)}): mu is the location of log y, so the coefficients act on log y;",
                              f"  E[y_i] = exp(mu_i + {scale_symbol(model)}^2 / 2) (brms posterior_epred), median(y_i) = exp(mu_i)."],
    likelihood_lines=_likelihood_lines,
    centred_intercept_note=(),
    deterministic_note="mu = linear predictor (log-scale location); the expected response exp(mu + sigma^2 / 2) is computed by predict()",
    default_prior_link="log",                                       # brms's log-scale family: it reads log(y)
    default_scale_prior=lambda scale: HalfStudentT(3, scale),       # brms sigma: student_t(3, 0, scale), lb 0
    addition_terms=frozenset({"weights", "cens"}),              # B02: weights(), verified against the frozen case lognormal_weighted; E04: cens() (lognormal_censored)
    tails=_tails,
    distributional_link="log",                                      # B01: sigma ~ z, brms's log link (lognormal_distributional_sigma)
    default_distributional_intercept=lambda scale: StudentT(3, 0, scale),   # brms Intercept_sigma: student_t(3, 0, scale of log y)
    monotonic=True,                                                 # B03: mo(x), verified against lognormal_monotonic
    log_likelihood=_log_likelihood,                                 # C07: log_lik(), checked against the model's own density (test_post_processing.py)
)
