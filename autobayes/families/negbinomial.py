"""Negative binomial count response (brms ``negbinomial``): log link on ``mu``, overdispersion parameter ``shape``.

brms's ``neg_binomial_2`` parameterisation: ``E[y] = mu``, ``Var[y] = mu + mu^2 / shape``, Poisson as ``shape``
grows. Bambi's ``negativebinomial`` family is PyMC's ``NegativeBinomial(mu, alpha)`` with ``alpha`` that same
``shape``; the prior is given as ``Priors(shape=...)`` (brms class ``shape``) and the sampled variable is
``alpha_log__`` (``model.names["shape"]`` is ``"alpha"``). Equality of the two densities and their gradients is
checked against brms in the oracle case ``negbinomial_varint``, not assumed. ``shape`` may have its own formula under a log
link (B01, docs/distributional.md; ``negbinomial_distributional_shape``), Bambi's distributional ``alpha``.

E06 (docs/links.md): brms's ``negbinomial("identity")``, ``mu = eta`` (a positive linear predictor), is ``_identity()``, reached
through ``NEGBINOMIAL.links`` (``ab.negbinomial(..., link="identity")``).

E04 (docs/cens-trunc-families.md): under the log link, ``trunc(ub = b)`` with an integer, inclusive bound (``autobayes.tails``);
a lower bound, ``negbinomial_truncated``'s ``trunc(lb = 1)``, is implemented and withheld (D121).
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from types import MappingProxyType

import numpy as np

from autobayes.families.spec import NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, FamilySpec, auxiliary, exp_mean, scale_line, scale_symbol
from autobayes.formula import check_counts
from autobayes.priors import InverseGamma, StudentT


def _validate_response(spec, data) -> None:
    check_counts(f"response {spec.response!r}", data[spec.response], domain="poisson")


def _response_warnings(spec, df) -> list[str]:
    if (df[spec.response] == 0).all():
        return ["all observed counts are zero; the shape parameter is then informed by its prior alone; "
                "check sensitivity to the explicit log-mean and shape priors"]
    return []


def _inverse_link(eta):
    return exp_mean(eta, "negative binomial expected count")


def _draw_response(model, post, eta, mean, df, rng):
    """``NegativeBinomial(mu, shape)`` draws: numpy's ``negative_binomial(n = shape, p = shape / (shape + mu))``
    has mean ``mu`` and variance ``mu + mu^2 / shape``. Means beyond the integer generator's range are refused."""
    limit = np.iinfo(np.int64).max - 10 * np.sqrt(float(np.iinfo(np.int64).max))
    if np.any(mean >= limit):
        raise ValueError("negative binomial response prediction exceeds the integer random-generator limit")
    shape = auxiliary(model, post, "shape")                    # per draw, or per draw and row (B01)
    shape = _rate_shape(model, shape, df)                      # F06a: times each row's denominator under rate()
    return rng.negative_binomial(shape, shape / (shape + mean)).astype(np.int64)


def _rate_shape(model, shape, df):
    """The shape of each row: ``shape``, or under rate() (F06a) ``shape * denom_i``, the rows of ``df`` (None: the fitted rows)."""
    from autobayes.se_rate import denominators

    denom = denominators(model, df)
    return shape if denom is None else shape * denom


def _tails():
    from autobayes.tails import NEGBINOMIAL_TAILS

    return NEGBINOMIAL_TAILS


def _log_likelihood(model, post, eta):
    """brms's log_lik (C07): ``log NegBinomial2(y_i | exp(eta_i), shape)``, shape per draw (or per draw and row, B01); under rate()
    (F06a) at the mean ``exp(eta_i) denom_i`` and the shape ``shape * denom_i``."""
    from autobayes.families.pointwise import neg_binomial_2_log
    from autobayes.se_rate import denominators

    denom = denominators(model, None)
    shape = _rate_shape(model, auxiliary(model, post, "shape"), None)
    return neg_binomial_2_log(model.design.y, eta if denom is None else eta + np.log(denom), shape)


def _likelihood_lines(model) -> list[str]:
    s = scale_symbol(model)                                    # "shape", or "shape_i" with its own formula (B01)
    rate = model.spec.addition("rate")
    if rate is not None:                                       # F06a: brms's rate(col): mean and shape times the denominator
        d = f"{rate.column}_i"
        head = [f"  y_i ~ NegativeBinomial(mu_i * {d}, {s} * {d}); mu_i = exp(eta_i), {d} the row's denominator (brms's rate());",
                f"  Var[y_i] = {d} (mu_i + mu_i^2 / {s})"]
    else:
        head = [f"  y_i ~ NegativeBinomial(mu_i, {s}); mu_i = exp(eta_i); Var[y_i] = mu_i + mu_i^2 / {s}"]
    return [scale_line(model), "", "Likelihood", *head,
            "  brms's neg_binomial_2 parameterisation: shape is the inverse overdispersion (Poisson as shape -> infinity);",
            "  PyMC's NegativeBinomial(mu, alpha) with alpha = shape, " + ("sampled as log(shape)." if s == "shape" else "one shape_i per row."),
            "  Input counts are integers from 0 through 1e6; zero-count rows are retained.",
            "  No zero-inflation is modeled." if rate is not None else "  No exposure/offset or zero-inflation is modeled."]


NEGBINOMIAL = FamilySpec(
    name="negbinomial", tag="negative binomial", parent="mu", link="log", link_scale="log", mean_name="mu",
    response_kind="count", response_dtype=np.int64,
    scale_parameter="shape", bambi_scale_parameter="alpha", fixed_parameters=NO_FIXED_PARAMETERS,
    bambi_family=lambda: "negativebinomial",
    validate_response=_validate_response,
    response_warnings=_response_warnings,
    response_upper_bound=lambda model: None,
    inverse_link=_inverse_link,
    expected_response=None,
    draw_response=_draw_response,
    manual_prediction=True, prediction_kinds=STANDARD_PREDICTION_KINDS,
    kind_error="kind must be 'mean', 'response' or 'linear'",
    scale_rule="shape is required for the negative binomial family; sigma does not apply",
    scale_missing_error="the negative binomial family needs a prior on the shape (overdispersion) parameter: pass shape=...",
    linpred_symbol="eta", scale_phrase="log mean", intercept_meaning="log expected count",
    mean_lines=lambda model: ([f"  E[y_i] = mu_i = exp(eta_i); eta is log expected count; Var[y_i] = mu_i + mu_i^2 / {scale_symbol(model)}."]
                              if model.spec.addition("rate") is None else
                              [f"  E[y_i] = exp(eta_i) * {model.spec.addition('rate').column}_i; eta is the log expected count per unit of "
                               "the denominator (brms's rate())."]),
    likelihood_lines=_likelihood_lines,
    centred_intercept_note=(),
    deterministic_note="mu = exp(linear predictor); mu = exp(eta), expected count",
    default_prior_link="log",                                       # brms reads log(y), zeros as 0.1
    default_scale_prior=lambda scale: InverseGamma(0.4, 0.3),       # brms negbinomial()$prior: shape inv_gamma(0.4, 0.3), lb 0
    addition_terms=frozenset({"weights", "trunc", "rate"}),     # B02: weights(), verified against the frozen case negbinomial_weighted; E04: trunc() (negbinomial_truncated, the lower bound withheld: D121); F06a: rate() (negbinomial_rate)
    rate_parameters=("alpha",),                                     # F06a: rate() multiplies the shape (Bambi's alpha) by the denominator too
    tails=_tails,
    distributional_link="log",                                      # B01: shape ~ z, brms's log link (negbinomial_distributional_shape)
    default_distributional_intercept=lambda scale: StudentT(3, 0, 2.5),   # brms Intercept_shape: student_t(3, 0, 2.5) under the log link
    monotonic=True,                                                 # B03: mo(x), verified against negbinomial_monotonic
    log_likelihood=_log_likelihood,                                 # C07: log_lik(), checked against the model's own density (test_post_processing.py)
    links=MappingProxyType({"identity": lambda: _identity()}),      # E06: brms's identity link (negbinomial_identity)
)


# -- E06: the identity link ---------------------------------------------------------------------------------------------------
def _identity_bambi_family():
    from autobayes.likelihoods import negbinomial_identity_family

    return negbinomial_identity_family()


def _identity_inverse_link(eta):
    from autobayes.families.links import positive_mean

    return positive_mean(eta, "identity", "negative binomial expected count")


def _identity_log_likelihood(model, post, eta):
    """brms's log_lik (C07): ``log NegBinomial2(y_i | eta_i, shape)``, ``-inf`` where ``eta_i <= 0``."""
    from autobayes.families.pointwise import neg_binomial_2_identity

    return neg_binomial_2_identity(model.design.y, eta, auxiliary(model, post, "shape"))


def _identity_likelihood_lines(model) -> list[str]:
    return [scale_line(model), "", "Likelihood",
            "  y_i ~ NegativeBinomial(mu_i, shape); mu_i = eta_i (brms's negbinomial(\"identity\")); Var[y_i] = mu_i + mu_i^2 / shape",
            "  brms's neg_binomial_2 parameterisation, written as its log form at log(mu_i) (autobayes/likelihoods.py); a row",
            "  with eta_i <= 0 has no valid mean and the log density is -inf there (brms's Stan rejects it).",
            "  Input counts are integers from 0 through 1e6; zero-count rows are retained."]


def _identity_check_formula(spec) -> None:
    from autobayes.families.links import check_link_formula

    check_link_formula(spec, _identity())


@lru_cache(maxsize=None)
def _identity() -> FamilySpec:
    """The negative binomial family's spec under brms's identity link (E06): ``mu = eta``, positive where the model is defined;
    verified against the frozen case ``negbinomial_identity``. ``weights()``, a formula on ``shape``, ``mo()`` and the other special
    terms are refused under it (``check_formula``, divergence D133)."""
    return replace(
        NEGBINOMIAL, link="identity", link_scale=None, bambi_family=_identity_bambi_family, inverse_link=_identity_inverse_link,
        scale_phrase="expected count", intercept_meaning="expected count",
        mean_lines=lambda model: ["  E[y_i] = mu_i = eta_i, which must be positive; Var[y_i] = mu_i + mu_i^2 / shape."],
        likelihood_lines=_identity_likelihood_lines,
        deterministic_note="mu = linear predictor = the expected count",
        default_prior_link="identity",                              # brms reads y as it is
        addition_terms=frozenset(), distributional_link=None, default_distributional_intercept=None, monotonic=False,   # D133
        log_likelihood=_identity_log_likelihood,
        check_formula=_identity_check_formula,
        links=MappingProxyType({"log": lambda: NEGBINOMIAL}),
    )
