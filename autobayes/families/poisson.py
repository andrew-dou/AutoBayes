"""Poisson count response: log link on ``mu``, no auxiliary parameter; or brms's ``poisson("sqrt")`` (E06, docs/links.md),
``mu = eta^2``, whose spec ``_sqrt()`` is reached through ``POISSON.links`` (``ab.poisson(..., link="sqrt")``). E04
(docs/cens-trunc-families.md): under the log link, ``trunc()`` with integer, inclusive bounds (``autobayes.tails``;
``poisson_truncated``), a lower bound's expected value ``E[Y | Y >= lb]`` (accepted on the negative binomial's frozen case
against two references, P6, D121; no frozen Poisson case has a lower bound)."""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from types import MappingProxyType

import numpy as np

from autobayes.families.spec import NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, FamilySpec, exp_mean
from autobayes.formula import check_counts


def _validate_response(spec, data) -> None:
    check_counts(f"response {spec.response!r}", data[spec.response], domain="poisson")


def _response_warnings(spec, df) -> list[str]:
    if (df[spec.response] == 0).all():
        return ["all observed counts are zero; check sensitivity to the explicit log-mean priors"]
    return []


def _inverse_link(eta):
    return exp_mean(eta, "Poisson expected count")


def _draw_response(model, post, eta, mean, df, rng):
    limit = np.iinfo(np.int64).max - 10 * np.sqrt(float(np.iinfo(np.int64).max))
    if np.any(mean >= limit):
        raise ValueError("Poisson response prediction exceeds the integer random-generator limit")
    return rng.poisson(mean)


def _tails():
    from autobayes.tails import POISSON_TAILS

    return POISSON_TAILS


def _log_likelihood(model, post, eta):
    """brms's log_lik (C07): ``log Poisson(y_i | exp(eta_i))``; under rate() (F06a) at the mean ``exp(eta_i) denom_i``."""
    from autobayes.families.pointwise import poisson_log
    from autobayes.se_rate import denominators

    denom = denominators(model, None)
    return poisson_log(model.design.y, eta if denom is None else eta + np.log(denom))


def _likelihood_lines(model) -> list[str]:
    rate = model.spec.addition("rate")
    if rate is not None:                                       # F06a: brms's rate(col)
        mean = f"  y_i ~ Poisson(mu_i * {rate.column}_i); mu_i = exp(eta_i), {rate.column}_i the row's denominator (brms's rate())"
    else:
        mean = "  y_i ~ Poisson(mu_i); mu_i = exp(eta_i)"
    return ["", "Likelihood", mean,
            "  Input counts are integers from 0 through 1e6; zero-count rows are retained.",
            ("  The exposure enters through the offset term (mu_i = exp(offset_i) * exp(eta_i - offset_i)); "
             "no overdispersion or zero-inflation is modeled.") if model.spec.offset is not None else
            "  No overdispersion or zero-inflation is modeled." if rate is not None else
            "  No exposure/offset, overdispersion or zero-inflation is modeled."]


POISSON = FamilySpec(
    name="poisson", tag="Poisson", parent="mu", link="log", link_scale="log", mean_name="mu",
    response_kind="count", response_dtype=np.int64,
    scale_parameter=None, bambi_scale_parameter=None, fixed_parameters=NO_FIXED_PARAMETERS,
    bambi_family=lambda: "poisson",
    validate_response=_validate_response,
    response_warnings=_response_warnings,
    response_upper_bound=lambda model: None,
    inverse_link=_inverse_link,
    expected_response=None,
    draw_response=_draw_response,
    manual_prediction=True, prediction_kinds=STANDARD_PREDICTION_KINDS,
    kind_error="kind must be 'mean', 'response' or 'linear'",
    scale_rule="sigma and shape must be omitted for Poisson (no auxiliary parameter)",
    scale_missing_error=None,
    linpred_symbol="eta", scale_phrase="log mean", intercept_meaning="log expected count",
    mean_lines=lambda model: (["  E[y_i] = mu_i = exp(eta_i); eta is log expected count."] if model.spec.addition("rate") is None else
                              [f"  E[y_i] = exp(eta_i) * {model.spec.addition('rate').column}_i; eta is the log expected count per unit of "
                               "the denominator (brms's rate())."]),
    likelihood_lines=_likelihood_lines,
    centred_intercept_note=(),
    deterministic_note="mu = exp(linear predictor); mu = exp(eta), expected count",
    default_prior_link="log",                                       # brms reads log(y), zeros as 0.1
    default_scale_prior=None,
    addition_terms=frozenset({"weights", "trunc", "rate"}),     # B02: weights(), verified against the frozen case poisson_weighted; E04: trunc() (poisson_truncated); F06a: rate() (poisson_rate)
    tails=_tails,
    monotonic=True,                                                 # B03: mo(x), verified against poisson_monotonic
    log_likelihood=_log_likelihood,                                 # C07: log_lik(), verified against poisson_varint_post
    links=MappingProxyType({"sqrt": lambda: _sqrt()}),              # E06: brms's sqrt link (poisson_sqrt)
    car=True,                                                       # F02a: car(), verified against poisson_car_bym2 (log link)
)


# -- E06: the sqrt link -------------------------------------------------------------------------------------------------------
def _sqrt_bambi_family():
    from autobayes.likelihoods import poisson_sqrt_family

    return poisson_sqrt_family()


def _sqrt_inverse_link(eta):
    from autobayes.families.links import square_mean

    return square_mean(eta, "Poisson expected count")


def _sqrt_log_likelihood(model, post, eta):
    """brms's log_lik (C07): ``log Poisson(y_i | eta_i^2)``."""
    from autobayes.families.pointwise import poisson_sqrt

    return poisson_sqrt(model.design.y, eta)


def _sqrt_check_formula(spec) -> None:
    from autobayes.families.links import check_link_formula

    check_link_formula(spec, _sqrt())


@lru_cache(maxsize=None)
def _sqrt() -> FamilySpec:
    """The Poisson family's spec under brms's sqrt link (E06): ``mu = eta^2`` for every real ``eta``, the linear predictor kept as
    the parent ``sqrt_mu``; verified against the frozen case ``poisson_sqrt``. ``weights()``, ``mo()`` and the other special terms
    are refused under it (``check_formula``, divergence D133)."""
    return replace(
        POISSON, parent="sqrt_mu", link="sqrt", link_scale="sqrt", bambi_family=_sqrt_bambi_family, inverse_link=_sqrt_inverse_link,
        scale_phrase="square-root mean", intercept_meaning="square root of the expected count",
        mean_lines=lambda model: ["  E[y_i] = mu_i = eta_i^2; eta is the square root of the expected count (brms's sqrt link: any real",
                                  "  eta gives a valid mean, and eta and -eta the same one)."],
        likelihood_lines=lambda model: ["", "Likelihood", "  y_i ~ Poisson(mu_i); mu_i = eta_i^2 (brms's poisson(\"sqrt\"))",
                                        "  sampled as 2 y_i log|eta_i| - eta_i^2 - log(y_i!) (autobayes/likelihoods.py; -inf at eta_i = 0",
                                        "  for y_i > 0, as Stan's poisson_lpmf(y | 0)).",
                                        "  Input counts are integers from 0 through 1e6; zero-count rows are retained.",
                                        ("  The offset enters the linear predictor eta, under the square root."
                                         if model.spec.offset is not None else "  No offset, overdispersion or zero-inflation is modeled.")],
        deterministic_note="sqrt_mu = linear predictor; mu = sqrt_mu^2, the expected count, is computed by predict()",
        default_prior_link="sqrt",                                  # brms reads sqrt(y) (zeros as they are)
        addition_terms=frozenset(), monotonic=False,                # D133: not verified under the sqrt link
        log_likelihood=_sqrt_log_likelihood,
        check_formula=_sqrt_check_formula,
        links=MappingProxyType({"log": lambda: POISSON}),
    )
