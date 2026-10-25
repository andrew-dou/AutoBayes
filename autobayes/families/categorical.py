"""Unordered categorical responses: brms's ``categorical()`` family (brms-parity B05, docs/ordinal-categorical.md).

The response has K categories (labels, sorted as brms's ``as.factor`` sorts them; the first is the reference; brms's
``Y`` is the 1-based category). Each non-reference category ``k`` has its own linear predictor ``mu_k`` (brms's
``mu<k>``: its own centred intercept and coefficients on the same design) and ``P(y = k)`` is the softmax of
``(0, mu_2, ..., mu_K)``. Bambi's categorical family builds exactly that coefficient layout (every term with a trailing
category axis, the design centred, the intercepts uncentred after sampling); ``likelihoods.categorical_logit_family``
keeps the K - 1 linear predictors as the parent ``mu`` under the identity link and uses brms's
``categorical_logit_lpmf`` as a stable log-softmax. ``predict(kind="linear")`` returns ``mu`` with a ``category``
dimension over the non-reference categories (brms's ``posterior_linpred``), ``kind="mean"`` the probabilities ``p`` over
all K (brms's ``posterior_epred``), ``kind="response"`` 1-based category draws. Group-level terms and the offset are not
taken in this slice (divergence D37). Checked against brms in the oracle cases ``categorical_poponly`` and
``categorical_poponly_categorical``, not assumed.
"""
from __future__ import annotations

import numpy as np

from autobayes.families.spec import NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, FamilySpec
from autobayes.formula import FormulaError, categorical_categories


def _validate_response(spec, data) -> None:
    from autobayes.default_priors import brms_dpar_names

    categories, _, _ = categorical_categories(spec.response, data[spec.response])
    brms_dpar_names(categories[1:])                                 # refuses categories brms would name alike


def _response_categories(spec, column):
    return categorical_categories(spec.response, column)


def _check_formula(spec) -> None:
    if spec.group_terms:
        raise FormulaError(f"{spec.formula!r}: the categorical family takes population-level terms only; a "
                           f"group-level term ('{spec.group_terms[0].canonical}') would give every non-reference category its "
                           "own group effects and SD (brms's group term in each mu<k>), which is not verified here (divergence D37)")
    if spec.offset is not None:
        raise FormulaError(f"{spec.formula!r}: the categorical family takes no offset term (brms adds it to "
                           "every non-reference category's linear predictor; divergence D37)")


def _bambi_setup(model, bambi_priors):
    from autobayes.likelihoods import categorical_logit_family

    family = categorical_logit_family(model.categories, model.names["category_dim"])
    return family, bambi_priors, model.effective_center_predictors


def _inverse_link(eta):
    """brms's ``posterior_epred``: the softmax of ``(0, mu)`` over the categories, ``(..., K - 1)`` -> ``(..., K)``."""
    from autobayes.likelihoods import categorical_probabilities

    return categorical_probabilities(eta)


def _draw_response(model, post, eta, mean, df, rng):
    from autobayes.likelihoods import draw_categories

    return draw_categories(mean, rng)


def _log_likelihood(model, post, eta):
    """brms's log_lik (C07): ``log softmax(0, mu_i)[y_i]``; ``eta`` has the non-reference categories as its last axis."""
    from autobayes.families.pointwise import categorical_logit

    return categorical_logit(model.design.y, eta)


def _mean_lines(model) -> list[str]:
    ref, others = model.categories[0], model.categories[1:]
    from autobayes.default_priors import brms_dpar_names

    return [f"  One such linear predictor mu_i[k] per non-reference category k = {', '.join(others)}, each with its own intercept",
            f"  and coefficients (a_c[k], b[k]; brms's {', '.join(brms_dpar_names(others))}); the reference category {ref!r} has the",
            "  linear predictor 0, and P(y_i = k) = exp(mu_i[k]) / (1 + sum_j exp(mu_i[j])) (brms's posterior_epred).",
            f"  Each coefficient of category k is on the log-odds scale of k against {ref!r}."]


def _likelihood_lines(model) -> list[str]:
    cats = ", ".join(f"{c} ({k})" for k, c in enumerate(model.categories, start=1))
    return ["", "Likelihood",
            "  y_i ~ Categorical(softmax(0, mu_i)): brms's categorical_logit_lpmf, written as mu_i[y_i] - log_sum_exp(0, mu_i)",
            f"  (autobayes/likelihoods.py). Categories, sorted as brms sorts them, the first the reference: {cats}; the response",
            "  is fitted as its 1-based category (brms's Y)."]


CATEGORICAL = FamilySpec(
    name="categorical", tag="categorical", parent="mu", link="logit", link_scale="logit", mean_name="p",
    response_kind="categorical", response_dtype=np.int64,
    scale_parameter=None, bambi_scale_parameter=None, fixed_parameters=NO_FIXED_PARAMETERS,
    bambi_family=lambda: None,                                      # built per model by bambi_setup
    validate_response=_validate_response,
    response_warnings=lambda spec, df: [],
    response_upper_bound=lambda model: len(model.categories),
    inverse_link=_inverse_link,
    expected_response=None,
    draw_response=_draw_response,
    manual_prediction=True, prediction_kinds=STANDARD_PREDICTION_KINDS,
    kind_error="kind must be 'mean' (the category probabilities), 'response' (1-based category draws) or 'linear' (mu)",
    scale_rule="sigma, shape and phi do not apply to the categorical family (no auxiliary parameter)",
    scale_missing_error=None,
    linpred_symbol="mu", scale_phrase="log-odds (against the reference category)",
    intercept_meaning="log-odds of category k against the reference category",
    mean_lines=_mean_lines,
    likelihood_lines=_likelihood_lines,
    centred_intercept_note=("  The centred intercepts a_c[k] are at the proportion-weighted predictor, not the average over levels.",),
    deterministic_note="mu = the linear predictors of the non-reference categories; p = softmax(0, mu) is computed by predict()",
    default_prior_link=None,                                        # brms's Intercept default is student_t(3, 0, 2.5) per category
    default_scale_prior=None,
    response_categories=_response_categories,
    per_category=True,
    bambi_setup=_bambi_setup,
    check_formula=_check_formula,
    log_likelihood=_log_likelihood,                                 # C07: log_lik(), verified against categorical_poponly_post
)
