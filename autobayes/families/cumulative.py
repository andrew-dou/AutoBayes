"""Ordinal responses: brms's ``cumulative("logit")`` family (brms-parity B05, docs/ordinal-categorical.md).

The response has K ordered categories (positive integers ``1..K`` or an ordered pandas Categorical; brms's ``Y`` is the
1-based category). With the linear predictor ``mu`` (brms's ``mu``: the population-level terms on the raw design and the
group effects, without an intercept) and the K - 1 increasing thresholds ``tau`` (brms's ``b_Intercept``),
``P(y <= k) = expit(tau_k - mu)``. brms samples the thresholds of the centred design, ``Intercept = tau - sum_p mean(x_p)
b_p``, an ``ordered[K - 1]`` vector on Stan's ordered coordinates, with the intercept prior on every threshold; so does this
family (``likelihoods.ordered_thresholds``), and the likelihood (``likelihoods.cumulative_logit_family``, brms's
``ordered_logistic_lpmf`` in a stable form) reads ``b_Intercept = Intercept + sum_p mean(x_p) b_p``, a deterministic of the
graph. ``predict(kind="linear")`` returns ``mu`` (brms's ``posterior_linpred``), ``kind="mean"`` the category
probabilities ``p`` with a ``category`` dimension (brms's ``posterior_epred``) and ``kind="response"`` draws 1-based
categories. Checked against brms in the oracle cases ``cumulative_varint`` and ``cumulative_poponly_categorical``, not
assumed.

D04: brms's ``cumulative("probit")``, ``P(y <= k) = Phi(tau_k - mu)`` (brms's ``ordered_probit_lpmf``), is built by
``_probit()``: the same thresholds and graph with the likelihood ``likelihoods.ordinal_family("cumulative_probit", ...)``. Its
log density, predictions and defaults equal brms's on the oracle case ``cumulative_probit_varint``, but brms's own gradient
there (Stan's ``ordered_probit_lpmf``) is off the 50-digit value by up to 6.5e-5 relative at 86 of its 88 entries, where this
one is within 4e-15. The case is accepted by two-reference acceptance (P6) against the 50-digit reference of the model
(``oracle/reference.py``, ``tests/oracle/references/cumulative_probit_varint.json``), so the link is offered (``links``,
divergence D88). E06: brms's ``cumulative("cloglog")``, ``P(y <= k) = 1 - exp(-exp(tau_k - mu))``, is built by ``_cloglog()``, the
same graph with the likelihood ``likelihoods.ordinal_family("cumulative_cloglog", ...)`` (``likelihoods.cloglog_log_mass``),
offered and accepted on ``cumulative_cloglog_narrow``. On the wider ``cumulative_cloglog`` its design, predictions, JAX density
and defaults equal brms's and its log density and gradient equal the 50-digit value (to 2.4e-16 and 7e-17 relative), while brms's
own log density there is off by 5.9e-7 relative at one point (it forms the last category's probability 1 - inv_cloglog by
subtraction), so that case keeps G2 as an expected failure (docs/links.md, divergences D38, D132). brms's other cumulative links
are withheld (divergence D38).
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from types import MappingProxyType

import numpy as np
from scipy.special import expit, ndtr

from autobayes.families.spec import NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, FamilySpec
from autobayes.formula import FormulaError, ordinal_categories, pop_term_design

# brms's ordinal links that no frozen case verifies here, with the reason FamilySpec.with_link gives (divergence D38)
PARKED_ORDINAL_LINKS = MappingProxyType({
    link: f"brms's {link} link of the ordinal families is not verified against a frozen brms case here (docs/ordinal-categorical.md, "
          "divergence D38)" for link in ("probit_approx", "cloglog", "cauchit")})
# the cumulative family's own: its cloglog link (E06, _cloglog) is offered since cumulative_cloglog_narrow, a case on which brms's
# arithmetic is exact (on cumulative_cloglog's wider box it is not, D132); its other links stay parked (D38)
_CUMULATIVE_PARKED_LINKS = MappingProxyType({k: v for k, v in PARKED_ORDINAL_LINKS.items() if k != "cloglog"})


def _validate_response(spec, data) -> None:
    ordinal_categories(spec.response, data[spec.response])


def _response_categories(spec, column):
    return ordinal_categories(spec.response, column)


def _check_formula(spec) -> None:
    if not spec.intercept:
        raise FormulaError(f"{spec.formula!r}: the cumulative family's thresholds are its intercepts, so the intercept cannot "
                           "be removed ('0 +' or '- 1'); brms refuses it too ('Cannot remove the intercept in an ordinal model')")


def threshold_labels(model) -> list[str]:
    """brms's names of the thresholds, ``Intercept[1]``, ..., ``Intercept[K - 1]``: ``"1"``, ..., ``"K - 1"``."""
    return [str(k) for k in range(1, len(model.categories))]


def _bambi_setup(model, bambi_priors):
    """Bambi is handed the formula with formulae's intercept (so categorical predictors are treatment coded, as brms codes
    them) made to add nothing (``likelihoods.no_intercept``), the design uncentred (``mu`` is brms's linear predictor on the
    raw design), and the thresholds ``Intercept`` as the family's auxiliary parameter under the intercept prior; when the
    model centres, the likelihood adds the column means times the coefficients to them (brms's ``b_Intercept``)."""
    import bambi as bmb

    from autobayes.likelihoods import cumulative_logit_family, no_intercept, ordered_thresholds

    spec = model.spec
    columns = [pop_term_design(t, model.data, spec.categoricals) for t in spec.bambi_terms]   # a mo() term is not centred (B03)
    X = np.concatenate(columns, axis=1) if columns else np.zeros((len(model.data), 0))
    means = X.mean(0) if model.effective_center_predictors else np.zeros(X.shape[1])
    prior = model.priors.intercept
    labels = threshold_labels(model)
    dim = model.names["threshold_dim"]

    def thresholds(name, **kwargs):
        return ordered_thresholds(name, prior.name, dict(prior.params), labels, dim)

    terms = {k: v for k, v in bambi_priors.items() if k != "Intercept"}
    priors = {"mu": {**terms, "Intercept": bmb.Prior("NoIntercept", dist=no_intercept)},
              "Intercept": bmb.Prior("OrderedThresholds", dist=thresholds)}
    return cumulative_logit_family([t.name for t in spec.bambi_terms], means, dim), priors, False


def _expected_response(model, post, eta):
    """brms's ``posterior_epred``: ``P(y = k)`` per draw, row and category, from the draws of ``b_Intercept``."""
    from autobayes.likelihoods import cumulative_probabilities

    tau = post[model.names["thresholds"]].transpose("chain", "draw", model.names["threshold_dim"]).to_numpy()
    return cumulative_probabilities(eta, tau[..., None, :])


def _draw_response(model, post, eta, mean, df, rng):
    from autobayes.likelihoods import draw_categories

    return draw_categories(mean, rng)


def _log_likelihood(model, post, eta):
    """brms's log_lik (C07): ``log P(y_i | mu_i, b_Intercept)`` of the cumulative logit model, the thresholds per draw."""
    from autobayes.families.pointwise import ordered_logistic

    tau = post[model.names["thresholds"]].transpose("chain", "draw", model.names["threshold_dim"]).to_numpy()
    return ordered_logistic(model.design.y, eta, tau[..., None, :])


def _categories(model) -> str:
    """The categories in order, each with its 1-based code when the label is not the code itself."""
    if all(c == str(k) for k, c in enumerate(model.categories, start=1)):
        return " < ".join(model.categories)
    return " < ".join(f"{c} ({k})" for k, c in enumerate(model.categories, start=1))


def _mean_lines(model) -> list[str]:
    K = len(model.categories)
    return [f"  P(y_i <= k) = expit(b_Intercept[k] - mu_i), k = 1..{K - 1}, for the {K} ordered categories {_categories(model)};",
            "  P(y_i = k) is the difference of consecutive ones (brms's posterior_epred). mu has no intercept: the thresholds",
            "  b_Intercept[k] take its place, and a larger mu moves probability towards the higher categories."]


def _intercept_lines(model) -> list[str]:
    K = len(model.categories)
    if model.design.centered:
        return ["  Predictors are centred for the thresholds, as brms centres them: the sampled thresholds are",
                f"  Intercept[k] = b_Intercept[k] - sum_p mean(x_p) * b_p, k = 1..{K - 1} (brms's Intercept, an ordered vector on",
                "  Stan's ordered coordinates), the thresholds at the predictor means; the prior on the intercept applies to each",
                "  Intercept[k], and P(y_i <= k) = expit(Intercept[k] - (mu_i - sum_p mean(x_p) * b_p)). The graph's",
                "  b_Intercept (brms's b_Intercept, the thresholds of the raw design) is what predict() uses; the posterior keeps",
                "  both, as brms does."]
    return ["  Predictors are not centred (center_predictors=False): the sampled thresholds Intercept[k] are the thresholds of",
            f"  the raw design, k = 1..{K - 1}, and b_Intercept equals them."]


def _likelihood_lines(model) -> list[str]:
    return ["", "Likelihood",
            "  y_i ~ OrderedLogistic(mu_i, b_Intercept): brms's ordered_logistic_lpmf (cumulative(\"logit\"), disc = 1), written",
            "  as -softplus(mu_i - b_Intercept[y]) - softplus(b_Intercept[y - 1] - mu_i) + log(1 - exp(-(b_Intercept[y] -",
            "  b_Intercept[y - 1]))) with the end terms dropped for the first and last category: finite with a finite gradient",
            "  for every finite mu (autobayes/likelihoods.py). The response is fitted as its 1-based category (brms's Y)."]


CUMULATIVE = FamilySpec(
    name="cumulative", tag="cumulative (ordinal)", parent="mu", link="logit", link_scale="logit", mean_name="p",
    response_kind="ordinal", response_dtype=np.int64,
    scale_parameter=None, bambi_scale_parameter=None, fixed_parameters=NO_FIXED_PARAMETERS,
    bambi_family=lambda: None,                                      # built per model by bambi_setup
    validate_response=_validate_response,
    response_warnings=lambda spec, df: [],
    response_upper_bound=lambda model: len(model.categories),
    inverse_link=expit,                                             # P(y <= k) = expit(tau_k - mu)
    expected_response=_expected_response,
    draw_response=_draw_response,
    manual_prediction=True, prediction_kinds=STANDARD_PREDICTION_KINDS,
    kind_error="kind must be 'mean' (the category probabilities), 'response' (1-based category draws) or 'linear' (mu)",
    scale_rule="sigma, shape and phi do not apply to the cumulative family; its thresholds are the intercept",
    scale_missing_error=None,
    linpred_symbol="mu", scale_phrase="logit (latent)", intercept_meaning="thresholds",
    mean_lines=_mean_lines,
    likelihood_lines=_likelihood_lines,
    centred_intercept_note=(),
    deterministic_note="mu = linear predictor (no intercept); b_Intercept = Intercept + sum_p mean(x_p) * b_p, the thresholds",
    default_prior_link=None,                                        # brms's Intercept default is student_t(3, 0, 2.5)
    default_scale_prior=None,
    response_categories=_response_categories,
    thresholds=True,
    bambi_setup=_bambi_setup,
    check_formula=_check_formula,
    intercept_lines=_intercept_lines,
    monotonic=True,                                                 # B03: mo(x), verified against cumulative_monotonic
    log_likelihood=_log_likelihood,                                 # C07: log_lik(), verified against cumulative_varint_post
    links=MappingProxyType({"probit": lambda: _probit(),           # D04: the probit link, accepted against brms and P6's reference (D88)
                            "cloglog": lambda: _cloglog()}),        # E06: accepted against brms on cumulative_cloglog_narrow (D132)
    parked_links=_CUMULATIVE_PARKED_LINKS,
    addition_terms=frozenset({"thres"}),                            # F06b: thres(gr = z), the logit link's (_grouped)
    grouped_thresholds=lambda: _grouped(),
)


def _refuse_grouped_thresholds(spec, link: str) -> None:
    """F06b: grouped thresholds are verified for the logit link only (divergence D158)."""
    a = spec.addition("thres")
    if a is not None:
        raise FormulaError(f"{spec.formula!r}: {a.term!r} with the cumulative family's {link} link is not offered: grouped thresholds "
                           "are verified for the logit link only (docs/grouped-thresholds.md, divergence D158)")


def _probit_check_formula(spec) -> None:
    _check_formula(spec)
    _refuse_grouped_thresholds(spec, "probit")


# -- D04: the probit link ------------------------------------------------------------------------------------------------
def _probit_mean_lines(model) -> list[str]:
    K = len(model.categories)
    return [f"  P(y_i <= k) = Phi(b_Intercept[k] - mu_i), k = 1..{K - 1}, Phi the standard Normal CDF, for the {K} ordered "
            f"categories {_categories(model)};",
            "  P(y_i = k) is the difference of consecutive ones (brms's posterior_epred). mu has no intercept: the thresholds",
            "  b_Intercept[k] take its place, and a larger mu moves probability towards the higher categories."]


def _probit_likelihood_lines(model) -> list[str]:
    return ["", "Likelihood",
            "  y_i ~ OrderedProbit(mu_i, b_Intercept): brms's ordered_probit_lpmf (cumulative(\"probit\"), disc = 1), written",
            "  as log Phi(b_Intercept[1] - mu_i) for the first category, log Phi(mu_i - b_Intercept[K - 1]) for the last and the",
            "  log Normal mass between b_Intercept[y - 1] - mu_i and b_Intercept[y] - mu_i otherwise, from the tail where it",
            "  lies (likelihoods.normal_log_mass): finite with a finite gradient for every finite mu (autobayes/likelihoods.py).",
            "  The response is fitted as its 1-based category (brms's Y)."]


@lru_cache(maxsize=None)
def _probit() -> FamilySpec:
    """The cumulative family's spec under the probit link (D04): CUMULATIVE's, with the probit likelihood, probabilities and
    words; ``CUMULATIVE.links["probit"]`` (divergence D88: accepted on ``cumulative_probit_varint`` with P6's reference)."""
    from autobayes.families.ordinal import ordinal_bambi_setup, ordinal_expected_response, ordinal_log_likelihood

    return replace(
        CUMULATIVE, link="probit", link_scale="probit", inverse_link=ndtr,       # P(y <= k) = Phi(tau_k - mu)
        expected_response=ordinal_expected_response("cumulative_probit"),
        scale_phrase="probit (latent)", mean_lines=_probit_mean_lines, likelihood_lines=_probit_likelihood_lines,
        bambi_setup=ordinal_bambi_setup("cumulative_probit", ordered=True),
        log_likelihood=ordinal_log_likelihood("cumulative_probit"),
        links=MappingProxyType({"logit": lambda: CUMULATIVE, "cloglog": lambda: _cloglog()}),
        check_formula=_probit_check_formula, addition_terms=frozenset(), grouped_thresholds=None,   # F06b: logit only (D158)
    )


# -- E06: the cloglog link -----------------------------------------------------------------------------------------------------
def _cloglog_mean_lines(model) -> list[str]:
    K = len(model.categories)
    return [f"  P(y_i <= k) = 1 - exp(-exp(b_Intercept[k] - mu_i)), k = 1..{K - 1}, the inverse complementary log-log, for the {K} "
            f"ordered categories {_categories(model)};",
            "  P(y_i = k) is the difference of consecutive ones (brms's posterior_epred). mu has no intercept: the thresholds",
            "  b_Intercept[k] take its place, and a larger mu moves probability towards the higher categories."]


def _cloglog_likelihood_lines(model) -> list[str]:
    return ["", "Likelihood",
            "  y_i ~ cumulative(\"cloglog\")(mu_i, b_Intercept), disc = 1: with F(x) = 1 - exp(-exp(x)) and d_k = b_Intercept[k] - mu_i,",
            "  log F(d_1) for the first category, -exp(d_{K-1}) for the last and log(exp(-exp(d_{y-1})) (1 - exp(-(exp(d_y) -",
            "  exp(d_{y-1}))))) otherwise (likelihoods.cloglog_log_mass), never a difference of probabilities: finite with a",
            "  finite gradient for every finite mu (autobayes/likelihoods.py). The response is fitted as its 1-based category",
            "  (brms's Y)."]


def _cloglog_check_formula(spec) -> None:
    from autobayes.families.links import check_link_formula

    _check_formula(spec)
    _refuse_grouped_thresholds(spec, "cloglog")                     # F06b: logit only (D158)
    check_link_formula(spec, _cloglog())


@lru_cache(maxsize=None)
def _cloglog() -> FamilySpec:
    """The cumulative family's spec under brms's cloglog link (E06): CUMULATIVE's, with the cloglog likelihood, probabilities and
    words; ``CUMULATIVE.links["cloglog"]`` (divergence D132), accepted on ``cumulative_cloglog_narrow``; ``tests/test_links_more.py``
    also checks it against a 50-digit reference.
    ``mo()`` and the other special terms would be refused under it (``check_formula``, divergence D133)."""
    from autobayes.families.links import inverse_cloglog
    from autobayes.families.ordinal import ordinal_bambi_setup, ordinal_expected_response, ordinal_log_likelihood

    return replace(
        CUMULATIVE, link="cloglog", link_scale="cloglog", inverse_link=inverse_cloglog,   # P(y <= k) = 1 - exp(-exp(tau_k - mu))
        expected_response=ordinal_expected_response("cumulative_cloglog"),
        scale_phrase="complementary log-log (latent)", mean_lines=_cloglog_mean_lines, likelihood_lines=_cloglog_likelihood_lines,
        bambi_setup=ordinal_bambi_setup("cumulative_cloglog", ordered=True),
        log_likelihood=ordinal_log_likelihood("cumulative_cloglog"),
        monotonic=False, check_formula=_cloglog_check_formula,       # D133
        links=MappingProxyType({"logit": lambda: CUMULATIVE, "probit": lambda: _probit()}),
        addition_terms=frozenset(), grouped_thresholds=None,         # F06b: logit only (D158)
    )


# -- F06b: grouped thresholds, thres(gr = z) -------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _grouped() -> FamilySpec:
    """The cumulative logit family's spec with brms's grouped thresholds ``thres(gr = z)`` and ``thres(K, gr = z)`` (F06b,
    ``autobayes/grouped_thresholds.py``, docs/grouped-thresholds.md): one ordered set of thresholds per level of ``z``, the
    design uncentred, each row's likelihood, expected values and log_lik from its level's thresholds;
    ``CUMULATIVE.grouped_thresholds``, which ``Model`` takes for a formula with the term. Checked against brms on
    ``cumulative_thres_gr`` and ``cumulative_thres_k_gr``."""
    from autobayes import grouped_thresholds as gt

    return replace(
        CUMULATIVE, expected_response=gt.expected_response, log_likelihood=gt.log_likelihood, bambi_setup=gt.bambi_setup,
        mean_lines=gt.mean_lines, likelihood_lines=gt.likelihood_lines, intercept_lines=gt.intercept_lines,
        deterministic_note="mu = linear predictor (no intercept); b_Intercept = Intercept, the thresholds of every level (not centred)",
        monotonic=False, links=MappingProxyType({}), parked_links=MappingProxyType({}), grouped_thresholds=None,
    )
