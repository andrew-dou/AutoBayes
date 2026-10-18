"""Frechet response (brms ``frechet()``, brms-parity E02, docs/continuous-families.md): strictly positive reals, log link on
the mean ``mu``, shape ``nu > 1``.

The density is brms's (the vignette "Parameterization of Response Distributions in brms"): ``f(y) = nu / s (y / s)^(-1 - nu)
exp(-(y / s)^-nu)`` with the scale ``s = mu / Gamma(1 - 1 / nu)``, so that ``E[y] = mu = exp(eta)`` (brms's
``posterior_epred``). ``nu`` is brms's ``real<lower=1> nu``: learned, as the Student-t ``nu`` is (A13), under
``Priors(nu=...)`` truncated to ``nu > 1`` and renormalised there (brms class ``nu``, default ``gamma(2, 0.1)``; sampled as
``log(nu - 1)``, ``nu_interval__``), or fixed, ``ab.frechet(..., nu=3)`` (any finite ``nu > 1``; brms's ``bf(nu = 3)``), through
the same fixed-parameter path the Student-t family's verified fixed ``nu`` takes. The likelihood is
``likelihoods.frechet_family``, the linear predictor as the parent ``log_mu``. Checked against brms in the oracle case
``frechet_varint``, not assumed. Other links, a distributional formula on ``nu`` and the addition terms are refused (D112).
"""
from __future__ import annotations

from types import MappingProxyType

import numpy as np

from autobayes.families.spec import STANDARD_PREDICTION_KINDS, FamilySpec, exp_mean
from autobayes.formula import validate_positive_response
from autobayes.priors import Gamma, describe_prior

NU_LOWER = 1.0                    # brms's real<lower=1> nu: a learned nu's prior is truncated here, sampled as log(nu - 1)


def check_nu(nu) -> float:
    """The fixed Frechet shape: a finite real scalar greater than one (brms's ``real<lower=1> nu``; the mean needs it)."""
    if isinstance(nu, (bool, np.bool_)) or not isinstance(nu, (int, float, np.integer, np.floating)) or not np.isfinite(nu) or nu <= 1:
        raise ValueError("frechet requires a fixed finite scalar nu > 1 (or nu=None to learn it)")
    return float(nu)


def _nu(model, post):
    return post[model.names["nu"]].to_numpy()[..., None] if model.nu is None else model.nu


def _bambi_family():
    from autobayes.likelihoods import frechet_family

    return frechet_family()


def _validate_response(spec, data) -> None:
    validate_positive_response(spec.response, data[spec.response], "frechet")


def _inverse_link(eta):
    return exp_mean(eta, "Frechet expected response")


def _draw_response(model, post, eta, mean, df, rng):
    """Frechet draws of mean ``mu = exp(eta)`` and shape ``nu`` per draw and row (``likelihoods.frechet_random``)."""
    from autobayes.likelihoods import frechet_random

    eta = np.asarray(eta, dtype=float)
    return frechet_random(eta, np.broadcast_to(_nu(model, post), eta.shape), rng=rng)


def _log_likelihood(model, post, eta):
    """brms's log_lik (C07): the Frechet log density of ``y_i`` with ``mu_i = exp(eta_i)`` and nu per draw (or fixed)."""
    from autobayes.families.pointwise import frechet

    return frechet(model.design.y, eta, _nu(model, post))


def _likelihood_lines(model) -> list[str]:
    if model.nu is None:
        nu = [f"  nu ~ {describe_prior(model.priors.nu, lb=NU_LOWER)}, truncated to nu > {NU_LOWER:g} and renormalised there (brms's "
              f"real<lower=1> nu); nu is learned, sampled as log(nu - {NU_LOWER:g})"]
    else:
        nu = [f"  nu = {model.nu:g} (fixed, not sampled)"]
    return [*nu, "", "Likelihood",
            "  y_i ~ Frechet(scale s_i, shape nu): log f(y) = log nu - log s - (1 + nu) log(y / s) - (y / s)^-nu, brms's frechet();",
            "  s_i = mu_i / Gamma(1 - 1 / nu), mu_i = exp(eta_i), so E[y_i] = mu_i. Responses are strictly positive finite reals."]


FRECHET = FamilySpec(
    name="frechet", tag="Frechet", parent="log_mu", link="log", link_scale="log", mean_name="mu",
    response_kind="continuous", response_dtype=None,
    scale_parameter=None, bambi_scale_parameter=None, fixed_parameters=MappingProxyType({"nu": check_nu}),
    bambi_family=_bambi_family,
    validate_response=_validate_response,
    response_warnings=lambda spec, df: [],
    response_upper_bound=lambda model: None,
    inverse_link=_inverse_link,
    expected_response=None,
    draw_response=_draw_response,
    manual_prediction=True, prediction_kinds=STANDARD_PREDICTION_KINDS,
    kind_error="kind must be 'mean', 'response' or 'linear'",
    scale_rule="sigma, shape and phi do not apply to the frechet family (its shape is nu)",
    scale_missing_error=None,
    linpred_symbol="eta", scale_phrase="log mean", intercept_meaning="log expected response",
    mean_lines=lambda model: ["  E[y_i] = mu_i = exp(eta_i); eta is the log expected response (brms posterior_epred)."],
    likelihood_lines=_likelihood_lines,
    centred_intercept_note=(),
    deterministic_note="log_mu = linear predictor eta; mu = exp(eta), the expected response, is computed by predict()",
    learnable_parameters=MappingProxyType({"nu": NU_LOWER}),          # E02: nu learned when not fixed, as the Student-t nu (A13)
    default_prior_link="log",                                       # brms reads log(y) under the log link
    default_scale_prior=None,
    default_learned_priors=MappingProxyType({"nu": lambda: Gamma(2, 0.1)}),   # brms nu: gamma(2, 0.1), lb 1
    log_likelihood=_log_likelihood,                                 # C07: log_lik(), checked against the model's own density
)
