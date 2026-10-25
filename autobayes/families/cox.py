"""brms's ``cox()`` proportional hazards model (brms-parity F05, docs/cox.md): ``time | cens(c) ~ ...`` with right censoring
(``cens`` 0 an event, 1 right-censored; without ``cens()`` every row is an event), brms's log link on the relative hazard
``mu = exp(eta)``, and the baseline hazard of ``autobayes/baseline_hazard.py``: a cubic M-spline basis of the time with the
simplex ``sbhaz`` of its weights (brms class ``sbhaz``, ``Priors(sbhaz=ab.Dirichlet(1))``, brms's default ``dirichlet(1)``), set
by ``bhaz(df = )``, ``bhaz(knots = )`` and ``bhaz(Boundary.knots = )``.

The likelihood is ``autobayes.baseline_hazard.cox_family``, built per model because it reads the fitted rows' bases and events
(``bambi_setup``). brms refuses ``posterior_epred`` and ``posterior_predict`` for cox: ``predict(kind="linear")`` returns ``eta``
(brms's ``posterior_linpred``) and ``kind="relative_hazard"`` ``mu = exp(eta)`` (brms's ``posterior_linpred(transform = TRUE)``);
the expected time and draws of it are refused (D154). ``ab.log_lik`` and ``ab.loo`` are brms's. Checked against brms in the
oracle cases ``cox_varint``, ``cox_uncensored_default_priors``, ``cox_bhaz_df``, ``cox_bhaz_knots`` and ``cox_varint_post``.
Interval censoring and ``trunc()`` (which brms computes wrongly, D153), left censoring, ``weights()``, a stratified ``bhaz(gr =
)``, other links and the contexts no frozen case verifies are refused (D155).
"""
from __future__ import annotations

from types import MappingProxyType

import numpy as np

from autobayes.baseline_hazard import DRAWS_REFUSED, TRUNCATION_REFUSED, WEIGHTS_REFUSED, check_formula, describe_lines
from autobayes.families.spec import NO_FIXED_PARAMETERS, FamilySpec, exp_mean
from autobayes.formula import validate_positive_response

KIND_ERROR = ("kind must be 'linear' or 'relative_hazard' for the cox family: brms refuses posterior_epred and posterior_predict "
              "for cox (\"Cannot compute expected values of the posterior predictive distribution for family 'cox'.\"), and the "
              "expected time and draws of it are not offered here (docs/cox.md, divergence D154)")


def _validate_response(spec, data) -> None:
    """The times are strictly positive finite reals."""
    validate_positive_response(spec.response, data[spec.response], "cox")


def _bambi_setup(model, bambi_priors):
    """The likelihood of this model's fitted rows: ``cox_family`` with their bases and events and the Dirichlet prior's
    concentrations; the priors and the centring as the model set them."""
    from autobayes.baseline_hazard import cox_family
    from autobayes.priors import dirichlet_concentration

    block = model._bhaz_design
    return cox_family(block, dirichlet_concentration(model.priors.sbhaz, block.K)), bambi_priors, model._bambi_centring


def _inverse_link(eta):
    """``mu = exp(eta)``, the relative hazard (brms's ``posterior_linpred(transform = TRUE)``)."""
    return exp_mean(eta, "cox relative hazard")


def _draw_response(model, post, eta, mean, df, rng):
    """Not offered (``prediction_kinds`` has no ``"response"``): ``autobayes.baseline_hazard.DRAWS_REFUSED``."""
    raise ValueError(DRAWS_REFUSED)


def _log_likelihood(model, post, eta):
    """brms's log_lik (C07): each fitted row's log likelihood per draw, from the bases and the draws of ``sbhaz``."""
    from autobayes.baseline_hazard import SBHAZ, cox_log_lik

    block = model.design.bhaz
    sbhaz = post[SBHAZ].transpose("chain", "draw", ...).to_numpy()
    return cox_log_lik(np.asarray(eta, dtype=float), block.Zbhaz, block.Zcbhaz, sbhaz, block.event)


COX = FamilySpec(
    name="cox", tag="Cox proportional hazards", parent="log_mu", link="log", link_scale="log", mean_name="mu",
    response_kind="continuous", response_dtype=None,
    scale_parameter=None, bambi_scale_parameter=None, fixed_parameters=NO_FIXED_PARAMETERS,
    bambi_family=lambda: None,                                      # built per model by bambi_setup (it reads the bases)
    validate_response=_validate_response,
    response_warnings=lambda spec, df: [],
    response_upper_bound=lambda model: None,
    inverse_link=_inverse_link,
    expected_response=None,
    draw_response=_draw_response,
    manual_prediction=True, prediction_kinds=frozenset({"linear", "relative_hazard"}),
    kind_error=KIND_ERROR,
    scale_rule="sigma, shape and phi do not apply to the cox family (its baseline hazard is the simplex sbhaz)",
    scale_missing_error=None,
    linpred_symbol="eta", scale_phrase="log relative hazard", intercept_meaning="log relative hazard",
    mean_lines=lambda model: ["  eta_i is the log relative hazard: mu_i = exp(eta_i) multiplies the baseline hazard (brms's mu); "
                              "predict(kind='linear') returns eta,",
                              "  kind='relative_hazard' mu. The expected time and draws of it are not offered (brms refuses "
                              "posterior_epred and posterior_predict for cox; divergence D154)."],
    likelihood_lines=describe_lines,
    centred_intercept_note=(),
    deterministic_note="log_mu = linear predictor eta, the log relative hazard; mu = exp(eta) is computed by predict()",
    default_prior_link="log",                                       # brms reads log(time) for the Intercept's default (G6)
    default_scale_prior=None,
    addition_terms=frozenset({"cens", "bhaz"}),                     # F05: right censoring and the baseline hazard's settings
    own_additions=frozenset({"cens", "bhaz"}),                      # read by the cox likelihood itself
    refused_additions=MappingProxyType({"trunc": f"the truncation of a cox model is not offered: {TRUNCATION_REFUSED}",
                                        "weights": WEIGHTS_REFUSED}),
    check_formula=check_formula,                                    # F05: interval censoring (D153)
    bambi_setup=_bambi_setup,
    log_likelihood=_log_likelihood,                                 # C07: log_lik(), brms's (cox_varint_post)
    baseline_hazard=True,
)
