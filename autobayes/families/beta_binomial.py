"""Beta-binomial response (brms ``beta_binomial()``, brms-parity D02, docs/discrete-families.md): ``successes | trials(n)``,
logit link on the mean success probability ``mu``, precision ``phi``.

brms's parameterisation is ``beta_binomial(y | n, mu phi, (1 - mu) phi)`` with ``mu = inv_logit(eta)``: each row's success
probability is ``Beta(mu phi, (1 - mu) phi)``, so ``E[y] = n mu`` and ``Var[y] = n mu (1 - mu) (n + phi) / (1 + phi)``, the
binomial as ``phi`` grows. The likelihood is ``likelihoods.beta_binomial_logit_family``: PyMC's ``BetaBinomial`` with
``alpha = expit(eta) phi`` and ``beta = expit(-eta) phi``, the linear predictor kept as the parent ``logit_mu`` under the identity
link and the precision named ``phi`` as in brms (``Priors(phi=...)``, brms class ``phi``, default ``gamma(0.01, 0.01)``; sampled
as ``phi_log__``). The response and the trials are validated, cast and dropped where the trials are zero exactly as the
binomial family's (D4); ``predict(kind="mean")`` returns ``mu`` with ``expected = n mu`` (brms's ``posterior_epred``) and
``kind="probability"`` returns ``mu`` alone, as for the binomial (D5; both D82, with PyMC's accuracy at a large ``phi``).
Checked against brms in the oracle case ``beta_binomial_varint``, not assumed. No distributional formula on ``phi``, no
``weights()`` and no other link: no frozen case verifies them (D80, D81).
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from autobayes.families.binomial import _response_warnings as _binomial_warnings
from autobayes.families.spec import NO_FIXED_PARAMETERS, FamilySpec, auxiliary, scale_line
from autobayes.formula import binomial_counts
from autobayes.priors import Gamma

_INTERCEPT_NOTE = (
    "  expit(a_c) is the mean success probability at that proportion-weighted predictor; it is NOT the",
    "  average probability across the levels (the link is nonlinear).",
)


def _bambi_family():
    from autobayes.likelihoods import beta_binomial_logit_family

    return beta_binomial_logit_family()


def _validate_response(spec, data) -> None:
    binomial_counts(spec, data)


def _draw_response(model, post, eta, mean, df, rng):
    """``Binomial(n, p)`` with ``p ~ Beta(mu phi, (1 - mu) phi)`` per draw and row, ``1 - mu`` as ``expit(-eta)``; a shape that
    underflows to zero (|eta| beyond about 745) is refused rather than handed to the generator."""
    phi = auxiliary(model, post, "phi")
    alpha, beta = expit(eta) * phi, expit(-eta) * phi
    if not (np.all(alpha > 0) and np.all(beta > 0)):
        raise ValueError("beta-binomial response prediction has a shape parameter that underflowed to zero; inspect the logit-scale "
                         "draws")
    trials = df[model.spec.trials].to_numpy(dtype=np.int64)
    return rng.binomial(trials, rng.beta(alpha, beta)).astype(np.int64)


def _log_likelihood(model, post, eta):
    """brms's log_lik (C07): ``log BetaBinomial(y_i | n_i, mu_i phi, (1 - mu_i) phi)``, phi per draw."""
    from autobayes.families.pointwise import beta_binomial_logit

    return beta_binomial_logit(model.design.y, model.design.trials, eta, auxiliary(model, post, "phi"))


def _mean_lines(model) -> list[str]:
    s = model.spec
    return [f"  mean success probability mu_i = expit(eta_i) = 1 / (1 + exp(-eta_i)); {s.response}_i ~ BetaBinomial({s.trials}_i, "
            "mu_i phi, (1 - mu_i) phi);",
            f"  eta is on the log-odds scale; E[{s.response}_i] = {s.trials}_i mu_i."]


def _likelihood_lines(model) -> list[str]:
    s = model.spec
    k, n = s.response, s.trials
    return [scale_line(model), "", "Likelihood",
            f"  {k}_i ~ BetaBinomial({n}_i, mu_i phi, (1 - mu_i) phi): {n}_i trials whose success probability is Beta(mu_i phi, (1 - mu_i) phi);",
            f"  log p({k}_i) = log C({n}_i, {k}_i) + log B({k}_i + mu_i phi, {n}_i - {k}_i + (1 - mu_i) phi) - log B(mu_i phi, (1 - mu_i) phi);",
            f"  Var[{k}_i] = {n}_i mu_i (1 - mu_i) ({n}_i + phi) / (1 + phi): phi is the precision (the binomial as phi -> infinity),",
            "  sampled as log(phi); 1 - mu is computed as expit(-eta). brms's beta_binomial, through PyMC's BetaBinomial.",
            "  Counts up to 1e6 per row; zero-trial rows are dropped before design construction (as for the binomial)."]


BETA_BINOMIAL = FamilySpec(
    name="beta_binomial", tag="beta-binomial", parent="logit_mu", link="logit", link_scale="logit", mean_name="mu",
    response_kind="binomial", response_dtype=np.int64,
    scale_parameter="phi", bambi_scale_parameter="phi", fixed_parameters=NO_FIXED_PARAMETERS,
    bambi_family=_bambi_family,
    validate_response=_validate_response,
    response_warnings=_binomial_warnings,
    response_upper_bound=lambda model: model.design.trials,
    inverse_link=expit,
    expected_response=None,
    draw_response=_draw_response,
    manual_prediction=True, prediction_kinds=frozenset({"probability", "mean", "response", "linear"}),
    kind_error="kind must be 'probability', 'mean', 'response' or 'linear' for a beta-binomial model",
    scale_rule="phi (the precision) is required for the beta_binomial family; sigma and shape do not apply",
    scale_missing_error="the beta_binomial family needs a prior on the precision parameter phi: pass phi=...",
    linpred_symbol="eta", scale_phrase="log-odds", intercept_meaning="log-odds of the mean success probability",
    mean_lines=_mean_lines,
    likelihood_lines=_likelihood_lines,
    centred_intercept_note=_INTERCEPT_NOTE,
    deterministic_note="logit_mu = linear predictor; mu = expit(logit_mu), expected = n mu and count draws are computed by predict()",
    default_prior_link=None,                                        # logit: brms's defaults are student_t(3, 0, 2.5) (probed)
    default_scale_prior=lambda scale: Gamma(0.01, 0.01),            # brms beta_binomial phi: gamma(0.01, 0.01), lb 0 (probed)
    log_likelihood=_log_likelihood,                                 # C07: log_lik(), checked against the model's own density
)
