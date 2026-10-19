"""Zero-inflated beta-binomial successes (brms ``zero_inflated_beta_binomial``): ``successes | trials(n)`` under the logit link on
the mean success probability ``mu``, the precision ``phi`` and the zero-inflation probability ``zi`` (brms-parity E01,
docs/bounded-families.md).

A row is a structural zero with probability ``zi`` and a ``BetaBinomial(n, mu phi, (1 - mu) phi)`` count otherwise, ``mu =
inv_logit(eta)``: ``P(y = 0) = zi + (1 - zi) BetaBinomial(0 | n, mu phi, (1 - mu) phi)``, ``P(y = k) = (1 - zi) BetaBinomial(k |
n, ...)`` for ``k > 0``, ``E[y] = (1 - zi) n mu`` (brms's ``posterior_epred``). The likelihood is
``likelihoods.zero_inflated_beta_binomial_family``: ``eta`` kept as the parent ``logit_mu`` under the identity link, the
beta-binomial part the beta-binomial family's (PyMC's ``BetaBinomial``, ``1 - mu`` as ``expit(-eta)``, D02), ``phi`` and ``zi``
under brms's names. Trials are handled as for the binomial family (zero-trial rows dropped, divergence D4); ``predict(kind="mean")``
returns brms's expected count ``(1 - zi) n mu`` as ``expected``, as the zero-inflated binomial's is. The logit link only (D108).
Checked against brms in the oracle case ``zero_inflated_beta_binomial_varint``, not assumed.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from autobayes.families.spec import NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, FamilySpec, auxiliary, scale_line
from autobayes.families.zero_inflated_binomial import _response_warnings
from autobayes.formula import binomial_counts
from autobayes.priors import Beta, Gamma, describe_prior

_INTERCEPT_NOTE = (
    "  expit(a_c) is the beta-binomial part's mean success probability at that proportion-weighted predictor; it is NOT",
    "  the average probability across the levels (the link is nonlinear).",
)


def _bambi_family():
    from autobayes.likelihoods import zero_inflated_beta_binomial_family

    return zero_inflated_beta_binomial_family()


def _validate_response(spec, data) -> None:
    binomial_counts(spec, data)


def _expected_response(model, post, eta):
    """The per-trial mean ``(1 - zi) mu`` with ``mu = expit(eta)``, per draw: ``predict()`` multiplies it by each row's trials
    into brms's ``posterior_epred`` ``(1 - zi) n mu``, named ``expected`` (the family's ``mean_name``)."""
    zi = post[model.names["zi"]].to_numpy()[..., None]
    return (1.0 - zi) * expit(eta)


def _draw_response(model, post, eta, mean, df, rng):
    """Zero where a uniform draw falls below ``zi``, otherwise ``Binomial(n_i, p)`` with ``p ~ Beta(mu phi, (1 - mu) phi)``
    (``1 - mu`` as ``expit(-eta)``); a shape that underflows to zero is refused, as for the beta-binomial family."""
    phi, zi = auxiliary(model, post, "phi"), post[model.names["zi"]].to_numpy()[..., None]
    alpha, beta = expit(eta) * phi, expit(-eta) * phi
    if not (np.all(alpha > 0) and np.all(beta > 0)):
        raise ValueError("zero-inflated beta-binomial response prediction has a shape parameter that underflowed to zero; inspect "
                         "the logit-scale draws")
    trials = df[model.spec.trials].to_numpy(dtype=np.int64)
    zero = rng.uniform(size=np.shape(eta)) < zi
    return np.where(zero, 0, rng.binomial(trials, rng.beta(alpha, beta))).astype(np.int64)


def _log_likelihood(model, post, eta):
    """brms's log_lik (C07): ``zero_inflated_beta_binomial_lpmf(y_i | n_i, expit(eta_i), phi, zi)``, phi and zi per draw."""
    from autobayes.families.pointwise import zero_inflated_beta_binomial_logit

    return zero_inflated_beta_binomial_logit(model.design.y, model.design.trials, eta, auxiliary(model, post, "phi"),
                                             post[model.names["zi"]].to_numpy()[..., None])


def _likelihood_lines(model) -> list[str]:
    k, n = model.spec.response, model.spec.trials
    return [scale_line(model), f"  zi ~ {describe_prior(model.priors.zi, lb=0, ub=1)}", "", "Likelihood",
            f"  {k}_i ~ ZeroInflatedBetaBinomial({n}_i, mu_i, phi, zi); mu_i = expit(eta_i): a structural zero with probability zi,",
            f"  BetaBinomial({n}_i, mu_i phi, (1 - mu_i) phi) otherwise; P({k}_i = 0) = zi + (1 - zi) BetaBinomial(0 | {n}_i, mu_i phi,",
            f"  (1 - mu_i) phi); P({k}_i = j) = (1 - zi) BetaBinomial(j | {n}_i, mu_i phi, (1 - mu_i) phi) for j > 0 (brms's",
            f"  zero_inflated_beta_binomial); E[{k}_i] = (1 - zi) {n}_i mu_i. phi is the precision of the success probability (the",
            "  zero-inflated binomial as phi -> infinity), sampled as log(phi); 1 - mu is computed as expit(-eta); zi is brms's",
            "  zero-inflation probability, sampled as logit(zi). Counts up to 1e6 per row; zero-trial rows are dropped before design",
            "  construction (as for the binomial)."]


ZERO_INFLATED_BETA_BINOMIAL = FamilySpec(
    name="zero_inflated_beta_binomial", tag="zero-inflated beta-binomial", parent="logit_mu", link="logit", link_scale="logit",
    mean_name="expected",
    response_kind="binomial", response_dtype=np.int64,
    scale_parameter="phi", bambi_scale_parameter="phi", fixed_parameters=NO_FIXED_PARAMETERS,
    bambi_family=_bambi_family,
    validate_response=_validate_response,
    response_warnings=_response_warnings,
    response_upper_bound=lambda model: model.design.trials,
    inverse_link=expit,
    expected_response=_expected_response,
    draw_response=_draw_response,
    manual_prediction=True, prediction_kinds=STANDARD_PREDICTION_KINDS,
    kind_error=("kind must be 'mean', 'response' or 'linear' for a zero-inflated beta-binomial model ('mean' is brms's expected "
                "count (1 - zi) n mu, named expected)"),
    scale_rule="phi (the precision) is required for the zero_inflated_beta_binomial family; sigma and shape do not apply (its "
               "zero-inflation probability is zi)",
    scale_missing_error="the zero_inflated_beta_binomial family needs a prior on the precision parameter phi: pass phi=...",
    linpred_symbol="eta", scale_phrase="log-odds", intercept_meaning="log-odds of the beta-binomial part's mean success probability",
    mean_lines=lambda model: [f"  mu_i = expit(eta_i) is the mean success probability of the beta-binomial part (eta is its log-odds); "
                              f"E[{model.spec.response}_i] = (1 - zi) {model.spec.trials}_i mu_i."],
    likelihood_lines=_likelihood_lines,
    centred_intercept_note=_INTERCEPT_NOTE,
    deterministic_note=("logit_mu = linear predictor (the beta-binomial part's log-odds); expected = (1 - zi) n expit(logit_mu) and "
                        "count draws are computed by predict()"),
    default_prior_link=None,                                        # logit: brms's defaults are student_t(3, 0, 2.5)
    default_scale_prior=lambda scale: Gamma(0.01, 0.01),            # brms phi: gamma(0.01, 0.01), lb 0
    zero_parameter="zi",
    default_zero_prior=lambda: Beta(1, 1),                          # brms zi: beta(1, 1), lb 0, ub 1
    addition_terms=frozenset({"weights"}),                          # E01: weights(), each row's log likelihood times its weight
    log_likelihood=_log_likelihood,                                 # C07: log_lik(), checked against the model's own density
)
