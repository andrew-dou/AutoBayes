"""Wiener diffusion model of response times and decisions (brms ``wiener()``, brms-parity F04, docs/wiener.md): ``rt | dec(dec)
~ ...``, a response time ``rt`` with its decision ``dec`` (1 the upper boundary, 0 the lower one), the identity link on the drift
rate ``mu``, and three parameters: the boundary separation ``bs`` (``real<lower=0>``; brms's default ``gamma(1, 1)``), the
non-decision time ``ndt`` (``real<lower=0,upper=min_Y>``, as the shifted lognormal's, D01; default ``uniform(0, min_Y)``) and
the relative starting point ``bias`` (``real<lower=0,upper=1>``; default ``beta(1, 1)``).

The likelihood is ``autobayes.wiener_diffusion.wiener_family``, the first-passage time density of Navarro and Fuss (2009) in brms's
parameterisation (Stan's ``wiener_lpdf``), built per model because it reads each fitted row's decision (``bambi_setup``).
``bs`` takes a formula of its own under brms's log link (``["rt | dec(dec) ~ x", "bs ~ z"]``, its intercept's default prior
``normal(-0.6, 1.3)``; ``wiener_distributional_bs``). ``predict(kind="mean")`` returns the mean response time ``ndt + E[T]``
(``autobayes.wiener_diffusion.mean_decision_time``) as ``mean``, which is not brms's ``posterior_epred`` (divergence D148);
``kind="linear"`` returns ``mu``; draws of the response are refused (D149); ``ab.log_lik`` and ``ab.loo`` are offered from the
density, where brms needs RWiener (D150). Checked against brms in the oracle cases ``wiener_varint`` and
``wiener_distributional_bs``. Formulas on ``ndt`` and ``bias``, other links and the other addition terms and contexts are refused
(D151).
"""
from __future__ import annotations

import numpy as np

from autobayes.families.spec import (NO_FIXED_PARAMETERS, DistributionalFormula, ExtraParameter, FamilySpec, extra_draws,
                                     is_distributional, parameter_symbol)
from autobayes.formula import FormulaError, validate_positive_response
from autobayes.priors import MIN_Y, Beta, Gamma, Normal, Uniform
from autobayes.wiener_diffusion import DRAWS_REFUSED

BS = ExtraParameter(name="bs", meaning="boundary separation", support="positive", default_prior=lambda: Gamma(1, 1),
                    divergence="D151")
NDT = ExtraParameter(name="ndt", meaning="non-decision time", support="below_min_y", default_prior=lambda: Uniform(0, MIN_Y),
                     divergence="D151")
BIAS = ExtraParameter(name="bias", meaning="relative starting point", support="unit", default_prior=lambda: Beta(1, 1),
                      divergence="D151")


def _validate_response(spec, data) -> None:
    """brms's checks: the decisions are given (``dec()``, 'Addition argument 'dec' is required for family 'wiener''), and the
    response times are strictly positive finite reals ('Family 'wiener' requires response greater than 0')."""
    if spec.addition("dec") is None:
        raise FormulaError(f"{spec.formula!r}: the wiener family needs each response time's decision, 'rt | dec(dec) ~ ...' with "
                           "dec 1 for the upper boundary and 0 for the lower one (brms: \"Addition argument 'dec' is required for "
                           "family 'wiener'\")")
    validate_positive_response(spec.response, data[spec.response], "wiener")


def _bambi_setup(model, bambi_priors):
    """The likelihood of this model's fitted rows: ``wiener_family`` with each row's decision (brms's ``dec``), the priors and
    the centring as the model set them."""
    from autobayes.formula import addition_data
    from autobayes.wiener_diffusion import wiener_family

    return wiener_family(addition_data(model.spec, model.data)["dec"] == 1), bambi_priors, model._bambi_centring


def _expected_response(model, post, eta):
    """``E[rt] = ndt + E[T]`` per draw and row (``autobayes.wiener_diffusion.expected_response_time``), ``bs`` per row under its own
    formula; refused rather than returned where it is not finite."""
    from autobayes.wiener_diffusion import expected_response_time

    mean = expected_response_time(np.asarray(eta, dtype=float), extra_draws(model, post, "bs"), extra_draws(model, post, "ndt"),
                                  extra_draws(model, post, "bias"))
    if not np.isfinite(mean).all():
        raise ValueError("wiener expected response time is nonfinite for some draws and rows; inspect the posterior draws")
    return mean


def _draw_response(model, post, eta, mean, df, rng):
    """Not offered (``prediction_kinds`` has no ``"response"``): ``autobayes.wiener_diffusion.DRAWS_REFUSED``."""
    raise ValueError(DRAWS_REFUSED)


def _log_likelihood(model, post, eta):
    """brms's log_lik (C07), which brms itself computes only with RWiener: each fitted row's Wiener log density at its decision,
    from the model's own density (``autobayes.wiener_diffusion.wiener_log_density`` in NumPy)."""
    from autobayes.wiener_diffusion import wiener_log_density

    return wiener_log_density(model.design.y, model.design.dec == 1, np.asarray(eta, dtype=float), extra_draws(model, post, "bs"),
                              extra_draws(model, post, "ndt"), extra_draws(model, post, "bias"))


def _likelihood_lines(model) -> list[str]:
    a = parameter_symbol(model, "bs")
    return [BS.prior_line(model), NDT.prior_line(model), BIAS.prior_line(model), "", "Likelihood",
            f"  (rt_i, dec_i) ~ Wiener(boundary separation {a}, non-decision time ndt, relative starting point bias, drift mu_i):",
            f"  rt_i - ndt is the first time a Wiener process with drift mu_i and unit diffusion, started at {a} * bias between 0",
            f"  and {a}, reaches a boundary, dec_i which one (1 the upper, 0 the lower); Stan's wiener_lpdf(rt_i | {a}, ndt, bias, mu_i)",
            f"  for dec_i = 1 and wiener_lpdf(rt_i | {a}, ndt, 1 - bias, -mu_i) for dec_i = 0 (brms's wiener()), the Navarro-Fuss",
            "  series summed to double precision. ndt lies in (0, min(rt)) and bias in (0, 1); " +
            ("bs is its formula's exp(eta_bs,i) (above)." if is_distributional(model, "bs") else "bs is sampled as log(bs)."),
            "  Response times are strictly positive finite reals."]


def _mean_lines(model) -> list[str]:
    a = parameter_symbol(model, "bs")
    return ["  mu_i is the drift rate (identity link): the coefficients act on it.",
            f"  E[rt_i] = ndt + E[T_i], E[T_i] = ({a} P_i(upper) - {a} bias) / mu_i the mean decision time over both boundaries",
            f"  ({a}^2 bias (1 - bias) at mu_i = 0), P_i(upper) = (1 - exp(-2 mu_i {a} bias)) / (1 - exp(-2 mu_i {a})); predict(kind='mean')",
            "  returns it, which is not brms's posterior_epred (divergence D148)."]


WIENER = FamilySpec(
    name="wiener", tag="Wiener diffusion", parent="mu", link="identity", link_scale=None, mean_name="mean",
    response_kind="continuous", response_dtype=None,
    scale_parameter=None, bambi_scale_parameter=None, fixed_parameters=NO_FIXED_PARAMETERS,
    bambi_family=lambda: None,                                      # built per model by bambi_setup (it reads the decisions)
    validate_response=_validate_response,
    response_warnings=lambda spec, df: [],
    response_upper_bound=lambda model: None,
    inverse_link=lambda eta: eta,
    expected_response=_expected_response,
    draw_response=_draw_response,
    manual_prediction=True, prediction_kinds=frozenset({"mean", "linear"}),
    kind_error=f"kind must be 'mean' or 'linear' for the wiener family; {DRAWS_REFUSED}",
    scale_rule="sigma, shape and phi do not apply to the wiener family (its parameters are bs, ndt and bias)",
    scale_missing_error=None,
    linpred_symbol="mu", scale_phrase="drift rate", intercept_meaning="drift rate",
    mean_lines=_mean_lines,
    likelihood_lines=_likelihood_lines,
    centred_intercept_note=(),
    deterministic_note="mu = linear predictor, the drift rate; the mean response time ndt + E[T] is computed by predict()",
    default_prior_link="identity",                                  # brms reads rt as given (probed: median and MAD of rt)
    default_scale_prior=None,
    extra_parameters=(BS, NDT, BIAS),                               # F04: brms's bs, ndt, bias, in brms's order
    addition_terms=frozenset({"dec"}),                              # F04: brms's dec(), the decision of each row
    bambi_setup=_bambi_setup,
    log_likelihood=_log_likelihood,                                 # C07: log_lik(), from the model's own density (D150)
    # F04: bs ~ z under brms's log link (wiener_distributional_bs); its intercept's default normal(-0.6, 1.3), whatever the data
    distributional_formulas=(DistributionalFormula("bs", "log", lambda: Normal(-0.6, 1.3)),),
)
