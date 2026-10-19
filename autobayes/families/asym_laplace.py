"""Asymmetric Laplace response (brms ``asym_laplace()``, brms-parity E02, docs/continuous-families.md): real responses with
location ``mu`` (identity link), scale ``sigma`` and the quantile ``p``, learned or fixed: quantile regression.

The density is brms's (the vignette "Parameterization of Response Distributions in brms"): ``f(y) = p (1 - p) / sigma
exp(-rho_p((y - mu) / sigma))`` with ``rho_p(u) = u (p - I(u < 0))``, so that ``P(y < mu) = p``: ``mu`` is the ``p``-quantile;
``E[y] = mu + sigma (1 - 2 p) / (p (1 - p))`` (brms's ``posterior_epred``). ``sigma`` takes ``Priors(sigma=...)``. The quantile
is brms's ``real<lower=0,upper=1> quantile``, learned under ``Priors(quantile=...)``, a prior on (0, 1) (brms class
``quantile``, default ``beta(1, 1)``; sampled as ``quantile_logodds__``), or fixed, ``ab.asym_laplace(..., quantile=0.25)``
(brms's ``bf(y ~ ..., quantile = 0.25)``, which writes it into its Stan data and has no quantile parameter or prior): then
``FamilySpec.with_fixed`` gives the family's spec without the extra parameter and with ``quantile`` a fixed parameter,
checked strictly inside (0, 1) (brms accepts 0 and 1, where the density is 0 everywhere; D115). The likelihood is
``likelihoods.asym_laplace_family``. Checked against brms in the oracle cases ``asym_laplace_varint`` and
``asym_laplace_quantile``, not assumed. Distributional formulas on ``sigma`` or ``quantile`` and the addition terms are
refused (D112).
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from types import MappingProxyType

import numpy as np

from autobayes.families.spec import (NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, ExtraParameter, FamilySpec, auxiliary,
                                     extra_draws, scale_line)
from autobayes.formula import validate_finite_response
from autobayes.priors import Beta, HalfStudentT

QUANTILE = ExtraParameter(name="quantile", meaning="quantile", support="unit", default_prior=lambda: Beta(1, 1),
                          divergence="D112")


def check_quantile(value) -> float:
    """A fixed quantile (brms's ``bf(quantile = p)``): a finite real scalar strictly between 0 and 1."""
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value) or not 0 < value < 1):
        raise ValueError(f"quantile={value!r}: asym_laplace's fixed quantile must be a finite number strictly between 0 and 1 "
                         "(at 0 or 1 the density p (1 - p) / sigma ... is 0 everywhere; divergence D115)")
    return float(value)


def quantile_of(model, post):
    """The quantile per draw with a row axis (learned) or the model's fixed value (a scalar)."""
    fixed = getattr(model, "quantile", None)
    return fixed if fixed is not None else extra_draws(model, post, "quantile")


def _bambi_family():
    from autobayes.likelihoods import asym_laplace_family

    return asym_laplace_family()


def _validate_response(spec, data) -> None:
    validate_finite_response(spec.response, data[spec.response], "asym_laplace")


def _expected_response(model, post, eta):
    """brms's ``posterior_epred``: ``mu + sigma (1 - 2 p) / (p (1 - p))`` per draw and row."""
    p = quantile_of(model, post)
    return np.asarray(eta, dtype=float) + auxiliary(model, post, "sigma") * (1.0 - 2.0 * p) / (p * (1.0 - p))


def _draw_response(model, post, eta, mean, df, rng):
    """Asymmetric Laplace draws of location ``mu`` (the linear predictor), scale ``sigma`` and quantile ``p``."""
    from autobayes.likelihoods import asym_laplace_random

    mu = np.asarray(eta, dtype=float)
    return asym_laplace_random(mu, np.broadcast_to(auxiliary(model, post, "sigma"), mu.shape),
                               np.broadcast_to(quantile_of(model, post), mu.shape), rng=rng)


def _log_likelihood(model, post, eta):
    """brms's log_lik (C07): the asymmetric Laplace log density of ``y_i`` with location ``eta_i``, sigma and the quantile."""
    from autobayes.families.pointwise import asym_laplace

    return asym_laplace(model.design.y, eta, auxiliary(model, post, "sigma"), quantile_of(model, post))


def _likelihood_lines(model) -> list[str]:
    fixed = getattr(model, "quantile", None)
    q = (f"  quantile = {fixed:g} (fixed, not sampled; brms's bf(quantile = {fixed:g}))" if fixed is not None
         else QUANTILE.prior_line(model))
    return [scale_line(model), q, "", "Likelihood",
            "  y_i ~ AsymmetricLaplace(mu_i, sigma, p): log f(y) = log p + log(1 - p) - log sigma - rho_p((y - mu) / sigma),",
            "  rho_p(u) = u (p - I(u < 0)), brms's asym_laplace(); P(y_i < mu_i) = p, so mu_i is the p-quantile of y_i;",
            "  E[y_i] = mu_i + sigma (1 - 2 p) / (p (1 - p)). sigma is sampled as log(sigma)"
            + ("." if fixed is not None else ", the quantile as logit(quantile).")]


ASYM_LAPLACE = FamilySpec(
    name="asym_laplace", tag="asymmetric Laplace", parent="mu", link="identity", link_scale=None, mean_name="mean",
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
    scale_rule="sigma is required for the asym_laplace family; shape and phi do not apply",
    scale_missing_error="the asym_laplace family needs a prior on the scale sigma: pass sigma=...",
    linpred_symbol="mu", scale_phrase="response (the p-quantile)", intercept_meaning="p-quantile of the response",
    mean_lines=lambda model: ["  mu_i = the linear predictor (identity link), the p-quantile of y_i; E[y_i] = mu_i + sigma (1 - 2 p) / "
                              "(p (1 - p)) (brms posterior_epred)."],
    likelihood_lines=_likelihood_lines,
    centred_intercept_note=(),
    deterministic_note="mu = linear predictor, the p-quantile; the expected response is computed by predict()",
    default_prior_link="identity",                                  # brms reads y as given
    default_scale_prior=lambda scale: HalfStudentT(3, scale),       # brms sigma: student_t(3, 0, scale), lb 0
    extra_parameters=(QUANTILE,),                                   # E02: brms's quantile, default beta(1, 1)
    fixable=MappingProxyType({"quantile": lambda: fixed_quantile()}),   # E02: bf(quantile = p), asym_laplace_quantile
    log_likelihood=_log_likelihood,                                 # C07: log_lik(), checked against the model's own density
)


@lru_cache(maxsize=None)
def fixed_quantile() -> FamilySpec:
    """The family with its quantile fixed (brms's ``bf(quantile = p)``): no extra parameter, ``quantile`` a fixed parameter
    whose value the model holds (``model.quantile``) and Bambi takes as a float64 constant."""
    return replace(ASYM_LAPLACE, extra_parameters=(), fixed_parameters=MappingProxyType({"quantile": check_quantile}))
