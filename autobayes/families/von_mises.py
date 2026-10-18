"""von Mises response (brms ``von_mises()``, brms-parity E02, docs/continuous-families.md): angles in [-pi, pi] with mean
direction ``mu`` under brms's ``tan_half`` link and precision ``kappa``.

The density is brms's (the vignette "Parameterization of Response Distributions in brms"): ``f(y) = exp(kappa cos(y - mu)) /
(2 pi I0(kappa))``. The ``tan_half`` link is ``eta = tan(mu / 2)``, so ``mu = 2 atan(eta)`` lies in (-pi, pi) for every
finite ``eta``; ``E[y]`` as brms's ``posterior_epred`` reports it is ``mu``. ``log I0`` is ``likelihoods.log_bessel_i0``,
exact for every ``kappa`` (Stan's ``von_mises_lpdf``, and so brms's density, raises an overflow error above ``kappa = 713.9``;
D113). The response must lie in [-pi, pi], as brms requires. ``kappa`` takes ``Priors(kappa=...)``, a positive-support
prior (brms class ``kappa``, default ``gamma(2, 0.01)``; sampled as ``kappa_log__``). brms's default prior of the centred
intercept is ``student_t(1, 0, 1)`` under this link, whatever the data, and its group-level SDs' scale 2.5 (probed). Checked
against brms in the oracle case ``von_mises_varint``, not assumed. brms's ``identity`` link, a distributional formula on
``kappa`` and the addition terms are refused (D112).
"""
from __future__ import annotations

import numpy as np

from autobayes.families.spec import NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, ExtraParameter, FamilySpec, extra_draws
from autobayes.formula import validate_angle_response
from autobayes.priors import Gamma, StudentT

KAPPA = ExtraParameter(name="kappa", meaning="precision", support="positive", default_prior=lambda: Gamma(2, 0.01),
                       divergence="D112")


def _bambi_family():
    from autobayes.likelihoods import von_mises_family

    return von_mises_family()


def _validate_response(spec, data) -> None:
    validate_angle_response(spec.response, data[spec.response], "von_mises")


def inverse_tan_half(eta):
    """brms's ``tan_half`` inverse link (NumPy): ``mu = 2 atan(eta)``, in (-pi, pi)."""
    return 2.0 * np.arctan(np.asarray(eta, dtype=float))


def _draw_response(model, post, eta, mean, df, rng):
    """von Mises draws of mean direction ``mu`` and precision ``kappa`` per draw and row, on [-pi, pi]."""
    from autobayes.likelihoods import von_mises_random

    mu = np.asarray(mean, dtype=float)
    return von_mises_random(mu, np.broadcast_to(extra_draws(model, post, "kappa"), mu.shape), rng=rng)


def _log_likelihood(model, post, eta):
    """brms's log_lik (C07): the von Mises log density of ``y_i`` with ``mu_i = 2 atan(eta_i)`` and kappa per draw."""
    from autobayes.families.pointwise import von_mises

    return von_mises(model.design.y, inverse_tan_half(eta), extra_draws(model, post, "kappa"))


def _likelihood_lines(model) -> list[str]:
    return [KAPPA.prior_line(model), "", "Likelihood",
            "  y_i ~ VonMises(mu_i, kappa): log f(y) = kappa cos(y - mu) - log(2 pi) - log I0(kappa), brms's von_mises();",
            "  mu_i = 2 atan(eta_i) (brms's tan_half link: eta = tan(mu / 2)), the mean direction in (-pi, pi); kappa is sampled",
            "  as log(kappa); log I0 is exact for every kappa. Responses are angles in [-pi, pi] (radians)."]


VON_MISES = FamilySpec(
    name="von_mises", tag="von Mises", parent="tan_half_mu", link="tan_half", link_scale="tan_half", mean_name="mu",
    response_kind="continuous", response_dtype=None,
    scale_parameter=None, bambi_scale_parameter=None, fixed_parameters=NO_FIXED_PARAMETERS,
    bambi_family=_bambi_family,
    validate_response=_validate_response,
    response_warnings=lambda spec, df: [],
    response_upper_bound=lambda model: None,
    inverse_link=inverse_tan_half,
    expected_response=None,
    draw_response=_draw_response,
    manual_prediction=True, prediction_kinds=STANDARD_PREDICTION_KINDS,
    kind_error="kind must be 'mean', 'response' or 'linear'",
    scale_rule="sigma, shape and phi do not apply to the von_mises family (its precision is kappa)",
    scale_missing_error=None,
    linpred_symbol="eta", scale_phrase="tan_half (tan(mu / 2))", intercept_meaning="tan of half the mean direction",
    mean_lines=lambda model: ["  E[y_i] = mu_i = 2 atan(eta_i), the mean direction (brms's posterior_epred); eta = tan(mu / 2)."],
    likelihood_lines=_likelihood_lines,
    centred_intercept_note=(),
    deterministic_note="tan_half_mu = linear predictor eta; the mean direction mu = 2 atan(eta) is computed by predict()",
    default_prior_link=None,                                        # brms reads no response under tan_half: location 0, scale 2.5
    default_scale_prior=None,
    default_intercept_prior=lambda location, scale: StudentT(1, 0, 1),   # brms's Intercept under tan_half: student_t(1, 0, 1)
    extra_parameters=(KAPPA,),                                      # E02: brms's kappa, default gamma(2, 0.01)
    log_likelihood=_log_likelihood,                                 # C07: log_lik(), checked against the model's own density
)
