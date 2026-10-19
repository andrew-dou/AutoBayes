"""Gamma response (brms ``brmsfamily("Gamma", link = "log")``): strictly positive reals, log link on the mean ``mu``,
shape parameter ``shape``.

brms's parameterisation is ``gamma_lpdf(y | shape, shape / mu)``: ``E[y] = mu``, ``Var[y] = mu^2 / shape``. Bambi's
``gamma`` family takes the prior for ``alpha`` (= ``shape``) and hands PyMC ``Gamma(mu, sigma = mu / sqrt(alpha))``,
which PyMC turns back into ``alpha = mu^2 / sigma^2`` and ``beta = mu / sigma^2 = alpha / mu``; that round trip's
rounding is inside the oracle tolerance, checked against brms in the case ``gamma_varint``, not assumed. The
sampled variable is ``alpha_log__`` (``model.names["shape"]`` is ``"alpha"``). This library's Gamma link is ``log``;
brms's default is ``inverse`` (docs/brms-parity/divergences.md, D9). ``shape`` may have its own formula under a log link
(B01, docs/distributional.md; ``gamma_distributional_shape``), Bambi's distributional ``alpha``.

E06 (docs/links.md): brms's ``Gamma("inverse")`` (``mu = 1 / eta``, parent ``inverse_mu``) and ``Gamma("identity")`` (``mu = eta``,
parent ``mu``), each needing a positive linear predictor, are ``_link("inverse")`` and ``_link("identity")``, reached through
``GAMMA.links`` (``ab.gamma(..., link="inverse")``); the default stays ``log`` (D9).

E04 (docs/cens-trunc-families.md): under the log link, ``cens(c)`` and ``cens(c, y2)`` through the regularized incomplete gamma
function with an exact gradient in the shape (``autobayes.tails``), implemented and withheld (``parked_additions``): brms's own
gradient in the shape is not exact on the frozen case ``gamma_censored_interval`` (divergence D120).
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache, partial
from types import MappingProxyType

from autobayes.families.spec import NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, FamilySpec, auxiliary, exp_mean, scale_line, scale_symbol
from autobayes.formula import validate_positive_response
from autobayes.priors import Gamma, StudentT


CENSORING_PARK = (
    "cens() is implemented for the Gamma family (log CDF, log CCDF and interval log mass through the regularized incomplete gamma "
    "function, with an exact gradient in the shape) and withheld: brms's gradient in the shape is not exact. On the frozen case "
    "gamma_censored_interval, against a 50-digit evaluation (mpmath) of the same log density, brms's gradient in log(shape) is "
    "off by 1.4e-7, 2.8e-5, 2.4e-5, 1.2e-5 and 1.3e-5 relative at points 0, 2, 3, 4 and 6 (GRAD_REL 1e-7; at point 2 brms "
    "-68.5938954721456, exact -68.5958155986809), where AutoBayes is within 2e-10; every other gate passes, so the case needs a "
    "two-reference acceptance (P6) of its gradient (docs/cens-trunc-families.md, divergence D120)")


def _bambi_family():
    from bambi.defaults import get_builtin_family

    family = get_builtin_family("gamma")               # PyMC's Gamma through Bambi's alpha -> (mu, sigma) transform
    family.link = {"mu": "log", "alpha": "log"}        # brms's default Gamma link is inverse; this library uses log (D9)
    return family


def _validate_response(spec, data) -> None:
    validate_positive_response(spec.response, data[spec.response], "Gamma")


def _inverse_link(eta):
    return exp_mean(eta, "Gamma expected response")


def _draw_response(model, post, eta, mean, df, rng):
    """``Gamma(shape, rate = shape / mu)`` draws: numpy's ``gamma(shape, scale = mu / shape)``."""
    shape = auxiliary(model, post, "shape")                    # per draw, or per draw and row (B01)
    return rng.gamma(shape, mean / shape)


def _log_likelihood(model, post, eta):
    """brms's log_lik (C07): ``log Gamma(y_i | shape, shape / exp(eta_i))``, shape per draw (or per draw and row, B01)."""
    from autobayes.families.pointwise import gamma_log

    return gamma_log(model.design.y, eta, auxiliary(model, post, "shape"))


def _tails():
    from autobayes.tails import GAMMA_TAILS

    return GAMMA_TAILS


def _likelihood_lines(model) -> list[str]:
    s = scale_symbol(model)                                    # "shape", or "shape_i" with its own formula (B01)
    return [scale_line(model), "", "Likelihood",
            f"  y_i ~ Gamma({s}, rate = {s} / mu_i); mu_i = exp(eta_i); E[y_i] = mu_i, Var[y_i] = mu_i^2 / {s}",
            "  brms's parameterisation gamma_lpdf(y | shape, shape / mu), through PyMC's Gamma with alpha = shape,",
            ("  sampled as log(shape)." if s == "shape" else "  one shape_i per row.")
            + " The link is log; brms's default Gamma link is inverse (divergence D9).",
            "  Responses are strictly positive finite reals."]


GAMMA = FamilySpec(
    name="gamma", tag="Gamma", parent="mu", link="log", link_scale="log", mean_name="mu",
    response_kind="continuous", response_dtype=None,
    scale_parameter="shape", bambi_scale_parameter="alpha", fixed_parameters=NO_FIXED_PARAMETERS,
    bambi_family=_bambi_family,
    validate_response=_validate_response,
    response_warnings=lambda spec, df: [],
    response_upper_bound=lambda model: None,
    inverse_link=_inverse_link,
    expected_response=None,
    draw_response=_draw_response,
    manual_prediction=True, prediction_kinds=STANDARD_PREDICTION_KINDS,
    kind_error="kind must be 'mean', 'response' or 'linear'",
    scale_rule="shape is required for the Gamma family; sigma does not apply",
    scale_missing_error="the Gamma family needs a prior on the shape parameter: pass shape=...",
    linpred_symbol="eta", scale_phrase="log mean", intercept_meaning="log expected response",
    mean_lines=lambda model: [f"  E[y_i] = mu_i = exp(eta_i); eta is the log expected response; Var[y_i] = mu_i^2 / {scale_symbol(model)}."],
    likelihood_lines=_likelihood_lines,
    centred_intercept_note=(),
    deterministic_note="mu = exp(linear predictor); mu = exp(eta), expected response",
    default_prior_link="log",                                       # brms reads log(y) under the log link
    default_scale_prior=lambda scale: Gamma(0.01, 0.01),            # brms shape: gamma(0.01, 0.01), lb 0
    addition_terms=frozenset({"weights", "cens"}),              # B02: weights(), verified against the frozen case gamma_weighted; E04: cens(), withheld below
    tails=_tails,
    parked_additions=MappingProxyType({"cens": CENSORING_PARK}),    # E04: implemented, withheld (gamma_censored_interval, D120)
    distributional_link="log",                                      # B01: shape ~ z, brms's log link (gamma_distributional_shape)
    default_distributional_intercept=lambda scale: StudentT(3, 0, 2.5),   # brms Intercept_shape: student_t(3, 0, 2.5) under the log link
    monotonic=True,                                                 # B03: mo(x), verified against gamma_monotonic
    log_likelihood=_log_likelihood,                                 # C07: log_lik(), checked against the model's own density (test_post_processing.py)
    links=MappingProxyType({"inverse": lambda: _link("inverse"), "identity": lambda: _link("identity")}),   # E06 (gamma_inverse, gamma_identity)
)


# -- E06: the inverse and identity links --------------------------------------------------------------------------------------
_LINK_WORDS = {   # link -> (parent, mean as a formula in eta, what eta is, the rate as a formula)
    "inverse": ("inverse_mu", "1 / eta_i", "the inverse of the expected response", "shape eta_i"),
    "identity": ("mu", "eta_i", "the expected response", "shape / eta_i"),
}


def _link_bambi_family(link: str):
    from autobayes.likelihoods import gamma_link_family

    return gamma_link_family(link)


def _link_inverse(link: str, eta):
    from autobayes.families.links import positive_mean

    return positive_mean(eta, link, "Gamma expected response")


def _link_log_likelihood(link: str, model, post, eta):
    """brms's log_lik (C07): ``log Gamma(y_i | shape, shape / mu_i)`` under the link, ``-inf`` where ``eta_i <= 0``."""
    from autobayes.families.pointwise import gamma_link

    return gamma_link(model.design.y, eta, auxiliary(model, post, "shape"), link)


def _link_likelihood_lines(link: str, model) -> list[str]:
    _, mean, _, rate = _LINK_WORDS[link]
    return [scale_line(model), "", "Likelihood",
            f"  y_i ~ Gamma(shape, rate = shape / mu_i); mu_i = {mean} (brms's Gamma(\"{link}\")); E[y_i] = mu_i, Var[y_i] = mu_i^2 / shape",
            f"  brms's parameterisation gamma_lpdf(y | shape, shape / mu), written with the rate {rate} (autobayes/likelihoods.py),",
            "  sampled as log(shape); a row with eta_i <= 0 has no valid mean and the log density is -inf there (brms's Stan",
            "  rejects it). Responses are strictly positive finite reals."]


def _link_check_formula(link: str, spec) -> None:
    from autobayes.families.links import check_link_formula

    check_link_formula(spec, _link(link))


@lru_cache(maxsize=None)
def _link(link: str) -> FamilySpec:
    """The Gamma family's spec under brms's inverse or identity link (E06): the mean ``1 / eta`` or ``eta``, positive where the
    model is defined; verified against the frozen cases ``gamma_inverse`` and ``gamma_identity``. ``weights()``, a formula on
    ``shape``, ``mo()`` and the other special terms are refused under them (``check_formula``, divergence D133)."""
    parent, mean, meaning, _ = _LINK_WORDS[link]
    others = {"log": lambda: GAMMA, **{o: partial(_link, o) for o in _LINK_WORDS if o != link}}
    return replace(
        GAMMA, parent=parent, link=link, link_scale=None if link == "identity" else link,
        bambi_family=partial(_link_bambi_family, link), inverse_link=partial(_link_inverse, link),
        scale_phrase="expected response" if link == "identity" else "inverse mean",
        intercept_meaning=meaning.replace("the ", "", 1),
        mean_lines=lambda model: [f"  E[y_i] = mu_i = {mean}; eta is {meaning}, which must be positive; Var[y_i] = mu_i^2 / shape."],
        likelihood_lines=partial(_link_likelihood_lines, link),
        deterministic_note=(f"{parent} = linear predictor; mu = 1 / {parent}, the expected response, is computed by predict()"
                            if link == "inverse" else "mu = linear predictor = the expected response"),
        default_prior_link=link,                                    # brms reads 1 / y (zeros as 0.1) or y as it is
        addition_terms=frozenset(), distributional_link=None, default_distributional_intercept=None, monotonic=False,   # D133
        parked_additions=MappingProxyType({}),                      # E04: cens() is refused under these links (D133), not withheld
        log_likelihood=partial(_link_log_likelihood, link),
        check_formula=partial(_link_check_formula, link),
        links=MappingProxyType(others),
    )
