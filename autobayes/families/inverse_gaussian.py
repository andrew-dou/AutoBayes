"""Inverse Gaussian response (brms ``inverse.gaussian()``, brms-parity E02, docs/continuous-families.md): strictly positive
reals with mean ``mu`` and shape ``shape``, under brms's default link ``1/mu^2`` or its log link.

The density is brms's (the vignette "Parameterization of Response Distributions in brms"): ``f(y) = sqrt(shape / (2 pi
y^3)) exp(-shape (y - mu)^2 / (2 mu^2 y))``, ``E[y] = mu``, ``Var[y] = mu^3 / shape``. Under ``1/mu^2`` the linear predictor
is ``eta = 1 / mu^2``, so ``mu = 1 / sqrt(eta)`` exists for ``eta > 0`` only: the density is ``-inf`` elsewhere (brms's
``inv_sqrt`` of a negative number is NaN and its density an error) and ``predict()`` refuses such draws; a model with this
link usually carries an offset or priors that keep ``eta`` positive (the frozen case ``inverse_gaussian_varint`` carries
``offset(eoff)``). Under ``log``, ``mu = exp(eta)``. The likelihood is ``likelihoods.inverse_gaussian_family(link)``, the
linear predictor as the parent ``inverse_squared_mu`` or ``log_mu``; ``shape`` takes ``Priors(shape=...)`` (brms class
``shape``, default ``gamma(0.01, 0.01)``; Bambi's ``alpha``, as for the Gamma family, sampled as ``alpha_log__``). Checked
against brms in the oracle cases ``inverse_gaussian_varint`` and ``inverse_gaussian_log``, not assumed. brms's
``inverse`` and ``identity`` links, a distributional formula on ``shape`` and the addition terms are refused (D112).
"""
from __future__ import annotations

from functools import lru_cache, partial
from types import MappingProxyType

import numpy as np

from autobayes.families.spec import NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, FamilySpec, auxiliary, exp_mean, scale_line
from autobayes.formula import validate_positive_response
from autobayes.priors import Gamma


def _validate_response(spec, data) -> None:
    validate_positive_response(spec.response, data[spec.response], "inverse.gaussian")


def _draw_response(model, post, eta, mean, df, rng):
    """Inverse Gaussian draws of mean ``mu`` and shape ``shape`` per draw and row (``likelihoods.inverse_gaussian_random``)."""
    from autobayes.likelihoods import inverse_gaussian_random

    mu = np.asarray(mean, dtype=float)
    return inverse_gaussian_random(mu, np.broadcast_to(auxiliary(model, post, "shape"), mu.shape), rng=rng)


def _log_likelihood(link: str):
    def log_likelihood(model, post, eta):
        """brms's log_lik (C07): the inverse Gaussian log density of ``y_i`` with ``1 / mu_i`` from ``eta_i`` under the link,
        ``-inf`` where the 1/mu^2 link's ``eta_i <= 0``."""
        from autobayes.families.pointwise import inverse_gaussian

        eta = np.asarray(eta, dtype=float)
        if link == "log":
            inv_mu = np.exp(-eta)
        else:
            inv_mu = np.sqrt(np.where(eta > 0, eta, 1.0))
        ll = inverse_gaussian(model.design.y, inv_mu, auxiliary(model, post, "shape"))
        return ll if link == "log" else np.where(eta > 0, ll, -np.inf)
    return log_likelihood


def _likelihood_lines(link: str):
    def lines(model) -> list[str]:
        inverse = ("mu_i = 1 / sqrt(eta_i) (brms's 1/mu^2 link: eta_i = 1 / mu_i^2, which needs eta_i > 0; the density is -inf "
                   "where eta_i <= 0)" if link != "log" else "mu_i = exp(eta_i) (log link)")
        return [scale_line(model), "", "Likelihood",
                "  y_i ~ InverseGaussian(mean mu_i, shape): log f(y) = 0.5 log(shape / (2 pi y^3)) - shape (y - mu)^2 / (2 mu^2 y)",
                f"  {inverse}; E[y_i] = mu_i, Var[y_i] = mu_i^3 / shape",
                "  brms's inverse.gaussian(); shape is sampled as log(shape) (Bambi's alpha). Responses are strictly positive finite reals."]
    return lines


LINKS = ("1/mu^2", "log")                                          # brms's default link first; brms's inverse and identity links are refused (D112)


@lru_cache(maxsize=None)
def _spec(link: str) -> FamilySpec:
    log = link == "log"
    parent = "log_mu" if log else "inverse_squared_mu"

    def inverse_link(eta):
        from autobayes.likelihoods import inverse_gaussian_mean

        return exp_mean(eta, "inverse Gaussian expected response") if log else inverse_gaussian_mean(eta, link)

    def bambi_family():
        from autobayes.likelihoods import inverse_gaussian_family

        return inverse_gaussian_family(link)

    return FamilySpec(
        name="inverse_gaussian", tag="inverse Gaussian", parent=parent, link=link, link_scale=link, mean_name="mu",
        response_kind="continuous", response_dtype=None,
        scale_parameter="shape", bambi_scale_parameter="alpha", fixed_parameters=NO_FIXED_PARAMETERS,
        bambi_family=bambi_family,
        validate_response=_validate_response,
        response_warnings=lambda spec, df: [],
        response_upper_bound=lambda model: None,
        inverse_link=inverse_link,
        expected_response=None,
        draw_response=_draw_response,
        manual_prediction=True, prediction_kinds=STANDARD_PREDICTION_KINDS,
        kind_error="kind must be 'mean', 'response' or 'linear'",
        scale_rule="shape is required for the inverse_gaussian family; sigma and phi do not apply",
        scale_missing_error="the inverse_gaussian family needs a prior on the shape parameter: pass shape=...",
        linpred_symbol="eta", scale_phrase="log mean" if log else "1/mu^2",
        intercept_meaning="log expected response" if log else "1 / mu^2 (the inverse squared expected response)",
        mean_lines=lambda model: [("  E[y_i] = mu_i = exp(eta_i); eta is the log expected response." if log else
                                   "  E[y_i] = mu_i = 1 / sqrt(eta_i); eta = 1 / mu^2 (brms's default link), defined for eta > 0.")],
        likelihood_lines=_likelihood_lines(link),
        centred_intercept_note=(),
        deterministic_note=f"{parent} = linear predictor eta; the expected response mu is computed by predict()",
        default_prior_link=link,                                      # brms reads the response through the link (1 / y^2, log y)
        default_scale_prior=lambda scale: Gamma(0.01, 0.01),          # brms shape: gamma(0.01, 0.01), lb 0
        links=MappingProxyType({other: partial(_spec, other) for other in LINKS if other != link}),
        log_likelihood=_log_likelihood(link),                         # C07: log_lik(), checked against the model's own density
    )


INVERSE_GAUSSIAN = _spec("1/mu^2")                                    # brms's default link; _spec("log") through FamilySpec.links
