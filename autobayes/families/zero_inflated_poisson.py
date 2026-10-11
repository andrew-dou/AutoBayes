"""Zero-inflated Poisson counts (brms ``zero_inflated_poisson``): log link on the Poisson part's mean ``mu``, and the
zero-inflation probability ``zi`` (brms-parity B06, docs/zero-inflated-hurdle.md).

A row is a structural zero with probability ``zi`` and a ``Poisson(mu)`` count otherwise: ``P(y = 0) = zi + (1 - zi)
exp(-mu)``, ``P(y = k) = (1 - zi) Poisson(k | mu)`` for ``k > 0``, ``E[y] = (1 - zi) mu`` (brms's ``posterior_epred``). The
likelihood is ``likelihoods.zero_inflated_poisson_family``: the linear predictor ``eta = log(mu)`` kept as Bambi's parent
``log_mu`` under the identity link, and ``zi`` named and parameterised as brms's (``Priors(zi=...)``, a ``Beta`` prior;
sampled as ``zi_logodds__``), not PyMC's ``psi = 1 - zi``. ``predict(kind="mean")`` returns ``(1 - zi) mu`` as ``mean``;
``kind="linear"`` returns ``eta``. Checked against brms in the oracle case ``zero_inflated_poisson_varint``, not assumed.
"""
from __future__ import annotations

import numpy as np

from autobayes.families.spec import NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, FamilySpec, exp_mean
from autobayes.formula import check_counts
from autobayes.priors import Beta, describe_prior


def _bambi_family():
    from autobayes.likelihoods import zero_inflated_poisson_family

    return zero_inflated_poisson_family()


def _validate_response(spec, data) -> None:
    check_counts(f"response {spec.response!r}", data[spec.response], domain="poisson")


def _response_warnings(spec, df) -> list[str]:
    if (df[spec.response] == 0).all():
        return ["all observed counts are zero: the zero-inflation probability and the Poisson mean are not separately "
                "identified (only P(y = 0) = zi + (1 - zi) exp(-mu) is); check sensitivity to the explicit priors"]
    return []


def _inverse_link(eta):
    return exp_mean(eta, "zero-inflated Poisson count-part mean")


def _expected_response(model, post, eta):
    """brms's ``posterior_epred`` ``(1 - zi) mu`` with ``mu = exp(eta)``, per draw; refused rather than clipped when ``mu``
    overflows."""
    zi = post[model.names["zi"]].to_numpy()[..., None]
    return (1.0 - zi) * exp_mean(eta, "zero-inflated Poisson expected count")


def _draw_response(model, post, eta, mean, df, rng):
    """brms's ``posterior_predict_zero_inflated_poisson``: zero where a uniform draw falls below ``zi``, a ``Poisson(mu)``
    draw otherwise. Means beyond the integer generator's range are refused."""
    mu = exp_mean(eta, "zero-inflated Poisson count-part mean")
    limit = np.iinfo(np.int64).max - 10 * np.sqrt(float(np.iinfo(np.int64).max))
    if np.any(mu >= limit):
        raise ValueError("zero-inflated Poisson response prediction exceeds the integer random-generator limit")
    zi = post[model.names["zi"]].to_numpy()[..., None]
    zero = rng.uniform(size=mu.shape) < zi
    return np.where(zero, 0, rng.poisson(mu)).astype(np.int64)


def _likelihood_lines(model) -> list[str]:
    return [f"  zi ~ {describe_prior(model.priors.zi, lb=0, ub=1)}", "", "Likelihood",
            "  y_i ~ ZeroInflatedPoisson(mu_i, zi); mu_i = exp(eta_i): a structural zero with probability zi, Poisson(mu_i) otherwise",
            "  P(y_i = 0) = zi + (1 - zi) exp(-mu_i); P(y_i = k) = (1 - zi) Poisson(k | mu_i) for k > 0 (brms's zero_inflated_poisson);",
            "  E[y_i] = (1 - zi) mu_i. zi is brms's zero-inflation probability (not PyMC's psi = 1 - zi), sampled as logit(zi).",
            "  Input counts are integers from 0 through 1e6; zero-count rows are retained.",
            ("  The exposure enters through the offset term (mu_i = exp(offset_i) * exp(eta_i - offset_i)); no overdispersion "
             "is modeled.") if model.spec.offset is not None else "  No exposure/offset or overdispersion is modeled."]


ZERO_INFLATED_POISSON = FamilySpec(
    name="zero_inflated_poisson", tag="zero-inflated Poisson", parent="log_mu", link="log", link_scale="log", mean_name="mean",
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
    scale_rule="sigma, shape and phi do not apply to the zero-inflated Poisson family; its auxiliary parameter is zi",
    scale_missing_error=None,
    linpred_symbol="eta", scale_phrase="log Poisson-mean", intercept_meaning="log mean of the Poisson part",
    mean_lines=lambda model: ["  mu_i = exp(eta_i) is the mean of the Poisson part (eta is its log); E[y_i] = (1 - zi) mu_i."],
    likelihood_lines=_likelihood_lines,
    centred_intercept_note=(),
    deterministic_note="log_mu = linear predictor (the log of the Poisson part's mean); E[y] = (1 - zi) exp(log_mu) is computed by predict()",
    default_prior_link="log",                                       # brms reads log(y), zeros as 0.1
    default_scale_prior=None,
    zero_parameter="zi",
    default_zero_prior=lambda: Beta(1, 1),                          # brms zi: beta(1, 1), lb 0, ub 1
    monotonic=True,                                                 # B03: mo(x), verified against zero_inflated_poisson_monotonic
)
