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
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from autobayes.families.spec import NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, FamilySpec
from autobayes.formula import FormulaError, ordinal_categories, pop_term_design


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
    columns = [pop_term_design(t, model.data, spec.categoricals) for t in spec.pop_terms]
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
    return cumulative_logit_family([t.name for t in spec.pop_terms], means, dim), priors, False


def _expected_response(model, post, eta):
    """brms's ``posterior_epred``: ``P(y = k)`` per draw, row and category, from the draws of ``b_Intercept``."""
    from autobayes.likelihoods import cumulative_probabilities

    tau = post[model.names["thresholds"]].transpose("chain", "draw", model.names["threshold_dim"]).to_numpy()
    return cumulative_probabilities(eta, tau[..., None, :])


def _draw_response(model, post, eta, mean, df, rng):
    from autobayes.likelihoods import draw_categories

    return draw_categories(mean, rng)


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
)
