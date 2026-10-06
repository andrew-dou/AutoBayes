"""Gamma response (brms ``brmsfamily("Gamma", link = "log")``): strictly positive reals, log link on the mean ``mu``,
shape parameter ``shape``.

brms's parameterisation is ``gamma_lpdf(y | shape, shape / mu)``: ``E[y] = mu``, ``Var[y] = mu^2 / shape``. Bambi's
``gamma`` family takes the prior for ``alpha`` (= ``shape``) and hands PyMC ``Gamma(mu, sigma = mu / sqrt(alpha))``,
which PyMC turns back into ``alpha = mu^2 / sigma^2`` and ``beta = mu / sigma^2 = alpha / mu``; that round trip's
rounding is inside the oracle tolerance, checked against brms in the case ``gamma_varint``, not assumed. The
sampled variable is ``alpha_log__`` (``model.names["shape"]`` is ``"alpha"``). This library's Gamma link is ``log``;
brms's default is ``inverse`` (docs/brms-parity/divergences.md, D9).
"""
from __future__ import annotations

from autobayes.families.spec import NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, FamilySpec, exp_mean
from autobayes.formula import validate_positive_response


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
    shape = post[model.names["shape"]].to_numpy()[..., None]
    return rng.gamma(shape, mean / shape)


def _likelihood_lines(model) -> list[str]:
    return [f"  shape ~ {model.priors.shape}", "", "Likelihood",
            "  y_i ~ Gamma(shape, rate = shape / mu_i); mu_i = exp(eta_i); E[y_i] = mu_i, Var[y_i] = mu_i^2 / shape",
            "  brms's parameterisation gamma_lpdf(y | shape, shape / mu), through PyMC's Gamma with alpha = shape,",
            "  sampled as log(shape). The link is log; brms's default Gamma link is inverse (divergence D9).",
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
    mean_lines=lambda model: ["  E[y_i] = mu_i = exp(eta_i); eta is the log expected response; Var[y_i] = mu_i^2 / shape."],
    likelihood_lines=_likelihood_lines,
    centred_intercept_note=(),
    deterministic_note="mu = exp(linear predictor); mu = exp(eta), expected response",
)
