"""Zero-inflated binomial successes (brms ``zero_inflated_binomial``): ``successes | trials(n)`` under the logit link and the
zero-inflation probability ``zi`` (brms-parity D03, docs/zero-inflated-hurdle.md).

A row is a structural zero with probability ``zi`` and a ``Binomial(n, p)`` count otherwise, ``p = inv_logit(eta)``:
``P(y = 0) = zi + (1 - zi) (1 - p)^n``, ``P(y = k) = (1 - zi) Binomial(k | n, p)`` for ``k > 0``, ``E[y] = (1 - zi) n p``
(brms's ``posterior_epred``). The likelihood is ``likelihoods.zero_inflated_binomial_family``: ``eta`` kept as the parent
``logit_p`` under the identity link, the binomial part the M2 likelihood (``binomial_logit_logp``) and ``zi`` under brms's
name. Trials are handled as for the binomial family (zero-trial rows dropped, divergence D4). ``predict(kind="mean")``
returns brms's expected count ``(1 - zi) n p`` as ``expected``: the family's per-trial mean ``(1 - zi) p`` times each row's
trials, as the binomial family's ``expected`` is. The logit link only (other links: divergence D84). Checked against brms
in the oracle case ``zero_inflated_binomial_varint``, not assumed.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from autobayes.families.links import LINKS
from autobayes.families.spec import NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, FamilySpec
from autobayes.formula import binomial_counts
from autobayes.priors import Beta, describe_prior


def _bambi_family():
    from autobayes.likelihoods import zero_inflated_binomial_family

    return zero_inflated_binomial_family()


def _validate_response(spec, data) -> None:
    binomial_counts(spec, data)


def _response_warnings(spec, df) -> list[str]:
    k, n = df[spec.response].to_numpy(), df[spec.trials].to_numpy()
    if np.all(k == 0):
        return ["every fitted row has zero successes: the zero-inflation probability and the binomial part are not separately "
                "identified (only P(y = 0) = zi + (1 - zi) (1 - p)^n is); check sensitivity to the explicit priors"]
    if np.all(k == n):
        return ["every fitted row has all successes: the data are separated along the binomial part's linear predictor, the "
                "likelihood is monotone there, and zi is informed by its prior and the absence of zeros; check prior sensitivity"]
    return []


def _expected_response(model, post, eta):
    """The per-trial mean ``(1 - zi) p`` with ``p = expit(eta)``, per draw: ``predict()`` multiplies it by each row's trials
    into brms's ``posterior_epred`` ``(1 - zi) n p``, named ``expected`` (the family's ``mean_name``)."""
    zi = post[model.names["zi"]].to_numpy()[..., None]
    return (1.0 - zi) * expit(eta)


def _draw_response(model, post, eta, mean, df, rng):
    """brms's ``posterior_predict``: zero where a uniform draw falls below ``zi``, a ``Binomial(n_i, expit(eta_i))`` draw
    otherwise."""
    trials = df[model.spec.trials].to_numpy(dtype=np.int64)
    zi = post[model.names["zi"]].to_numpy()[..., None]
    zero = rng.uniform(size=np.shape(eta)) < zi
    return np.where(zero, 0, rng.binomial(trials, expit(eta))).astype(np.int64)


def _log_likelihood(model, post, eta):
    """brms's log_lik (C07): ``zero_inflated_binomial_logit_lpmf(y_i | n_i, eta_i, zi)``, zi per draw."""
    from autobayes.families.pointwise import zero_inflated_binomial_logit

    return zero_inflated_binomial_logit(model.design.y, model.design.trials, eta, post[model.names["zi"]].to_numpy()[..., None])


def _likelihood_lines(model) -> list[str]:
    k, n = model.spec.response, model.spec.trials
    return [f"  zi ~ {describe_prior(model.priors.zi, lb=0, ub=1)}", "", "Likelihood",
            f"  {k}_i ~ ZeroInflatedBinomial({n}_i, p_i, zi); p_i = expit(eta_i): a structural zero with probability zi,",
            f"  Binomial({n}_i, p_i) otherwise; P({k}_i = 0) = zi + (1 - zi) (1 - p_i)^{n}_i; P({k}_i = j) = (1 - zi) Binomial(j | {n}_i, p_i)",
            f"  for j > 0 (brms's zero_inflated_binomial); E[{k}_i] = (1 - zi) {n}_i p_i. zi is brms's zero-inflation probability, sampled",
            "  as logit(zi). The binomial part's log density is the binomial family's (log C computed inside the graph, softplus",
            "  terms exact for every finite eta); counts above 1e6 per row are refused.",
            "  Zero-trial rows are dropped before design construction; centering, when effective, uses only fitted rows."]


ZERO_INFLATED_BINOMIAL = FamilySpec(
    name="zero_inflated_binomial", tag="zero-inflated binomial", parent="logit_p", link="logit", link_scale="logit",
    mean_name="expected",
    response_kind="binomial", response_dtype=np.int64,
    scale_parameter=None, bambi_scale_parameter=None, fixed_parameters=NO_FIXED_PARAMETERS,
    bambi_family=_bambi_family,
    validate_response=_validate_response,
    response_warnings=_response_warnings,
    response_upper_bound=lambda model: model.design.trials,
    inverse_link=expit,
    expected_response=_expected_response,
    draw_response=_draw_response,
    manual_prediction=True, prediction_kinds=STANDARD_PREDICTION_KINDS,
    kind_error=("kind must be 'mean', 'response' or 'linear' for a zero-inflated binomial model ('mean' is brms's expected count "
                "(1 - zi) n p, named expected)"),
    scale_rule="sigma, shape and phi do not apply to the zero-inflated binomial family; its auxiliary parameter is zi",
    scale_missing_error=None,
    linpred_symbol="eta", scale_phrase="log-odds", intercept_meaning="per-trial success log-odds of the binomial part",
    mean_lines=lambda model: ["  p_i = expit(eta_i) is the success probability of the binomial part (eta is its log-odds); "
                              f"E[{model.spec.response}_i] = (1 - zi) {model.spec.trials}_i p_i."],
    likelihood_lines=_likelihood_lines,
    centred_intercept_note=LINKS["logit"].centred_intercept_note,
    deterministic_note=("logit_p = linear predictor (the binomial part's log-odds); expected = (1 - zi) n expit(logit_p) and count "
                        "draws are computed by predict()"),
    default_prior_link=None,                                        # logit: brms's defaults are student_t(3, 0, 2.5)
    default_scale_prior=None,
    zero_parameter="zi",
    default_zero_prior=lambda: Beta(1, 1),                          # brms zi: beta(1, 1), lb 0, ub 1
    addition_terms=frozenset({"weights"}),                          # D03: weights(), each row's log likelihood times its weight
    log_likelihood=_log_likelihood,                                 # C07: log_lik(), checked against the model's own density
)
