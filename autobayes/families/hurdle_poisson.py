"""Hurdle Poisson counts (brms ``hurdle_poisson``): log link on the mean ``mu`` of the untruncated Poisson part, and the
hurdle probability ``hu`` (brms-parity B06, docs/zero-inflated-hurdle.md).

A row is zero with probability ``hu`` and a zero-truncated ``Poisson(mu)`` count otherwise: ``P(y = 0) = hu``,
``P(y = k) = (1 - hu) Poisson(k | mu) / (1 - exp(-mu))`` for ``k > 0``, ``E[y] = (1 - hu) mu / (1 - exp(-mu))`` (brms's
``posterior_epred``). The likelihood is ``likelihoods.hurdle_poisson_family``: the linear predictor ``eta = log(mu)`` kept
as Bambi's parent ``log_mu`` under the identity link, and ``hu`` named and parameterised as brms's (``Priors(hu=...)``, a
``Beta`` prior; sampled as ``hu_logodds__``), not PyMC's ``psi = 1 - hu``. ``predict(kind="mean")`` returns brms's
expected count as ``mean``; ``kind="linear"`` returns ``eta``. Checked against brms in the oracle case
``hurdle_poisson_varint``, not assumed.
"""
from __future__ import annotations

import numpy as np

from autobayes.families.spec import NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, FamilySpec, exp_mean
from autobayes.formula import check_counts
from autobayes.priors import Beta, describe_prior


def _bambi_family():
    from autobayes.likelihoods import hurdle_poisson_family

    return hurdle_poisson_family()


def _validate_response(spec, data) -> None:
    check_counts(f"response {spec.response!r}", data[spec.response], domain="poisson")


def _response_warnings(spec, df) -> list[str]:
    if (df[spec.response] == 0).all():
        return ["all observed counts are zero: the count part has no data, so the Poisson mean is informed by its prior "
                "alone; check sensitivity to the explicit priors"]
    return []


def _inverse_link(eta):
    return exp_mean(eta, "hurdle Poisson count-part mean")


def _expected_response(model, post, eta):
    """brms's ``posterior_epred`` ``(1 - hu) mu / (1 - exp(-mu))`` with ``mu = exp(eta)``, per draw, the ratio computed as
    ``mu / -expm1(-mu)`` (its limit 1 where ``mu`` underflows to 0); refused rather than clipped when ``mu`` overflows."""
    hu = post[model.names["hu"]].to_numpy()[..., None]
    mu = exp_mean(eta, "hurdle Poisson expected count")
    with np.errstate(divide="ignore", invalid="ignore"):
        truncated = np.where(mu > 0, mu / -np.expm1(-mu), 1.0)
    return (1.0 - hu) * truncated


def _draw_response(model, post, eta, mean, df, rng):
    """brms's ``posterior_predict_hurdle_poisson``: zero where a uniform draw falls below ``hu``, a zero-truncated
    ``Poisson(mu)`` draw otherwise (``likelihoods.zero_truncated_poisson_draws``). Means beyond the integer generator's
    range are refused."""
    from autobayes.likelihoods import zero_truncated_poisson_draws

    mu = exp_mean(eta, "hurdle Poisson count-part mean")
    limit = np.iinfo(np.int64).max - 10 * np.sqrt(float(np.iinfo(np.int64).max))
    if np.any(mu >= limit):
        raise ValueError("hurdle Poisson response prediction exceeds the integer random-generator limit")
    hu = post[model.names["hu"]].to_numpy()[..., None]
    zero = rng.uniform(size=mu.shape) < hu
    return np.where(zero, 0, zero_truncated_poisson_draws(mu, rng)).astype(np.int64)


def _likelihood_lines(model) -> list[str]:
    return [f"  hu ~ {describe_prior(model.priors.hu, lb=0, ub=1)}", "", "Likelihood",
            "  y_i ~ HurdlePoisson(mu_i, hu); mu_i = exp(eta_i): zero with probability hu, a zero-truncated Poisson(mu_i) otherwise",
            "  P(y_i = 0) = hu; P(y_i = k) = (1 - hu) Poisson(k | mu_i) / (1 - exp(-mu_i)) for k > 0 (brms's hurdle_poisson);",
            "  E[y_i] = (1 - hu) mu_i / (1 - exp(-mu_i)). hu is brms's hurdle probability (not PyMC's psi = 1 - hu), sampled as logit(hu).",
            "  Input counts are integers from 0 through 1e6; zero-count rows are retained.",
            ("  The exposure enters through the offset term (mu_i = exp(offset_i) * exp(eta_i - offset_i)); no overdispersion "
             "is modeled.") if model.spec.offset is not None else "  No exposure/offset or overdispersion is modeled."]


HURDLE_POISSON = FamilySpec(
    name="hurdle_poisson", tag="hurdle Poisson", parent="log_mu", link="log", link_scale="log", mean_name="mean",
    response_kind="count", response_dtype=np.int64,
    scale_parameter=None, bambi_scale_parameter=None, fixed_parameters=NO_FIXED_PARAMETERS,
    bambi_family=_bambi_family,
    validate_response=_validate_response,
    response_warnings=_response_warnings,
    response_upper_bound=lambda model: None,
    inverse_link=_inverse_link,
    expected_response=_expected_response,
    draw_response=_draw_response,
    manual_prediction=True, prediction_kinds=STANDARD_PREDICTION_KINDS,
    kind_error="kind must be 'mean', 'response' or 'linear'",
    scale_rule="sigma, shape and phi do not apply to the hurdle Poisson family; its auxiliary parameter is hu",
    scale_missing_error=None,
    linpred_symbol="eta", scale_phrase="log Poisson-mean", intercept_meaning="log mean of the (untruncated) Poisson part",
    mean_lines=lambda model: ["  mu_i = exp(eta_i) is the mean of the untruncated Poisson part (eta is its log);",
                              "  E[y_i] = (1 - hu) mu_i / (1 - exp(-mu_i))."],
    likelihood_lines=_likelihood_lines,
    centred_intercept_note=(),
    deterministic_note=("log_mu = linear predictor (the log of the Poisson part's mean); E[y] = (1 - hu) mu / (1 - exp(-mu)) "
                        "with mu = exp(log_mu) is computed by predict()"),
    default_prior_link="log",                                       # brms reads log(y), zeros as 0.1
    default_scale_prior=None,
    zero_parameter="hu",
    default_zero_prior=lambda: Beta(1, 1),                          # brms hu: beta(1, 1), lb 0, ub 1
    monotonic=True,                                                 # B03: mo(x), verified against hurdle_poisson_monotonic
)
