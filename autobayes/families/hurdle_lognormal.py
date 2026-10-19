"""Hurdle lognormal responses (brms ``hurdle_lognormal``): non-negative reals, identity link on the log-scale location ``mu``
of the lognormal part (the linear predictor), its scale ``sigma`` and the hurdle probability ``hu`` (brms-parity D03,
docs/zero-inflated-hurdle.md).

A row is an exact zero with probability ``hu`` and a ``LogNormal(mu, sigma)`` value otherwise: the density is ``hu`` at 0 and
``(1 - hu) LogNormal(y | mu, sigma)`` above, ``E[y] = (1 - hu) exp(mu + sigma^2 / 2)`` (brms's ``posterior_epred``). The
likelihood is ``likelihoods.hurdle_lognormal_family``: ``mu`` as the parent under the identity link, the lognormal part
PyMC's ``LogNormal`` (the lognormal family's), ``sigma`` and ``hu`` under brms's names. ``predict(kind="mean")`` returns the
expected response as ``mean``; ``kind="linear"`` returns ``mu``. Checked against brms in the oracle case
``hurdle_lognormal_varint``, not assumed. E03 (docs/contexts-newer-families.md): ``hu`` takes a formula of its own under brms's
logit link (``["zpos ~ x", "hu ~ z"]``, one probability per row, ``hu_i = inv_logit(eta_hu,i)``, its intercept's default prior
``logistic(0, 1)``; ``hurdle_lognormal_distributional_hu``; Bambi's inverse logit, D117); a formula on ``sigma`` stays refused (D84).
"""
from __future__ import annotations

import numpy as np

from autobayes.families.spec import (NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, DistributionalFormula, FamilySpec, auxiliary,
                                     exp_mean, is_distributional, parameter_symbol, scale_line, zero_line)
from autobayes.formula import validate_nonnegative_response
from autobayes.priors import Beta, HalfStudentT, Logistic


def _bambi_family():
    from autobayes.likelihoods import hurdle_lognormal_family

    return hurdle_lognormal_family()


def _validate_response(spec, data) -> None:
    validate_nonnegative_response(spec.response, data[spec.response], "hurdle lognormal")


def _response_warnings(spec, df) -> list[str]:
    if (df[spec.response].to_numpy(dtype=float) == 0).all():
        return ["every response is zero: the lognormal part has no data, so its location and scale are informed by their priors "
                "alone; check sensitivity to the explicit priors"]
    return []


def _expected_response(model, post, eta):
    """brms's ``posterior_epred`` ``(1 - hu) exp(mu + sigma^2 / 2)``, per draw; refused rather than clipped when it overflows."""
    hu = auxiliary(model, post, "hu")                               # one per row with its own formula (E03)
    return (1.0 - hu) * exp_mean(eta + 0.5 * auxiliary(model, post, "sigma") ** 2, "hurdle lognormal expected response")


def _draw_response(model, post, eta, mean, df, rng):
    """brms's ``posterior_predict``: zero where a uniform draw falls below ``hu``, a ``LogNormal(mu, sigma)`` draw otherwise."""
    hu, sigma = auxiliary(model, post, "hu"), auxiliary(model, post, "sigma")
    zero = rng.uniform(size=np.shape(eta)) < hu
    return np.where(zero, 0.0, rng.lognormal(eta, sigma))


def _log_likelihood(model, post, eta):
    """brms's log_lik (C07): ``hurdle_lognormal_lpdf(y_i | mu_i, sigma, hu)``, sigma and hu per draw."""
    from autobayes.families.pointwise import hurdle_lognormal

    return hurdle_lognormal(model.design.y, eta, auxiliary(model, post, "sigma"), auxiliary(model, post, "hu"))


def _likelihood_lines(model) -> list[str]:
    hu = parameter_symbol(model, "hu")                              # hu_i with its own formula (E03)
    return [scale_line(model), zero_line(model), "", "Likelihood",
            f"  y_i ~ HurdleLogNormal(mu_i, sigma, {hu}): an exact zero with probability {hu}, LogNormal(mu_i, sigma) otherwise",
            f"  (log(y_i) ~ Normal(mu_i, sigma)); the density is {hu} at 0 and (1 - {hu}) LogNormal(y_i | mu_i, sigma) above (brms's",
            f"  hurdle_lognormal); E[y_i] = (1 - {hu}) exp(mu_i + sigma^2 / 2). sigma is the SD of log y in the lognormal part, sampled",
            "  as log(sigma); hu is brms's hurdle probability, "
            + ("one per row from its formula (above)." if is_distributional(model, "hu") else "sampled as logit(hu).")
            + " predict(kind='mean') returns E[y_i];",
            "  kind='linear' returns mu_i.",
            "  Responses are non-negative finite reals: exact zeros are the zero part."]


HURDLE_LOGNORMAL = FamilySpec(
    name="hurdle_lognormal", tag="hurdle lognormal", parent="mu", link="identity", link_scale="log", mean_name="mean",
    response_kind="continuous", response_dtype=None,
    scale_parameter="sigma", bambi_scale_parameter="sigma", fixed_parameters=NO_FIXED_PARAMETERS,
    bambi_family=_bambi_family,
    validate_response=_validate_response,
    response_warnings=_response_warnings,
    response_upper_bound=lambda model: None,
    inverse_link=lambda eta: eta,
    expected_response=_expected_response,
    draw_response=_draw_response,
    manual_prediction=True, prediction_kinds=STANDARD_PREDICTION_KINDS,
    kind_error="kind must be 'mean', 'response' or 'linear'",
    scale_rule="sigma is required for the hurdle lognormal family; shape and phi do not apply (its hurdle probability is hu)",
    scale_missing_error="the hurdle_lognormal family needs a prior on the SD of log y: pass sigma=...",
    linpred_symbol="mu", scale_phrase="log response", intercept_meaning="expected log response of the lognormal part",
    mean_lines=lambda model: ["  log(y_i) ~ Normal(mu_i, sigma) in the lognormal part: mu is the location of log y, so the coefficients act",
                              f"  on log y there; E[y_i] = (1 - {parameter_symbol(model, 'hu')}) exp(mu_i + sigma^2 / 2) (brms posterior_epred)."],
    likelihood_lines=_likelihood_lines,
    centred_intercept_note=(),
    deterministic_note=("mu = linear predictor (the lognormal part's log-scale location); E[y] = (1 - hu) exp(mu + sigma^2 / 2) is "
                        "computed by predict()"),
    default_prior_link="log",                                       # brms's log-scale family: it reads log(y), zeros as 0.1
    default_scale_prior=lambda scale: HalfStudentT(3, scale),       # brms sigma: student_t(3, 0, scale), lb 0
    zero_parameter="hu",
    default_zero_prior=lambda: Beta(1, 1),                          # brms hu: beta(1, 1), lb 0, ub 1
    addition_terms=frozenset({"weights"}),                          # D03: weights(), each row's log likelihood times its weight
    log_likelihood=_log_likelihood,                                 # C07: log_lik(), checked against the model's own density
    # E03: hu ~ z under brms's logit link (hurdle_lognormal_distributional_hu); its intercept's default logistic(0, 1)
    distributional_formulas=(DistributionalFormula("hu", "logit", lambda: Logistic(0, 1)),),
)
