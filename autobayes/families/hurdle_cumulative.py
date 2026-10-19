"""Hurdle ordinal responses: brms's ``hurdle_cumulative("logit")`` family (brms-parity E01, docs/bounded-families.md).

The response is brms's: whole numbers ``0..K``, where 0 is the hurdle category and ``1..K`` the K ordered categories of a
cumulative logit model, or an ordered pandas Categorical, whose first declared category is the hurdle (``formula.
hurdle_ordinal_categories``, brms 2.23's coding of both, probed). ``P(y = 0) = hu`` and ``P(y = k) = (1 - hu) (F(tau_k - mu) -
F(tau_{k-1} - mu))`` for ``k = 1..K``, with ``F`` the logistic CDF, the linear predictor ``mu`` without an intercept and the K - 1
ordered thresholds ``tau`` (brms's ``b_Intercept``); brms samples the thresholds of the centred design, ``Intercept = tau - sum_p
mean(x_p) b_p``, an ``ordered[K - 1]`` vector on Stan's ordered coordinates under the intercept prior, as for the cumulative
family (``cumulative.py``), and the hurdle probability ``hu`` (brms's ``real<lower=0,upper=1>``, the family's ``zero_parameter``,
sampled as ``hu_logodds__``). The likelihood is ``likelihoods.hurdle_cumulative_logit_family``. ``model.categories`` holds the
ordinal part's categories (so the thresholds are ``len(model.categories) - 1`` as for the cumulative family) and
``model.hurdle_category`` the hurdle's label; ``predict(kind="mean")`` returns the K + 1 probabilities, the hurdle first
(brms's ``posterior_epred`` over categories 0..K), and ``kind="response"`` draws codes 0..K (brms's ``Y``). Checked against brms
in the oracle case ``hurdle_cumulative_varint``, not assumed. Only the logit link, ``disc = 1`` and brms's flexible thresholds are
offered (divergence D109); ``weights()``, ``mo()``, distributional formulas and the other contexts are refused (D108).
"""
from __future__ import annotations

from types import MappingProxyType

import numpy as np
from scipy.special import expit

from autobayes.families.cumulative import _categories, _intercept_lines, threshold_labels
from autobayes.families.spec import NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, FamilySpec
from autobayes.formula import FormulaError, hurdle_ordinal_categories, pop_term_design
from autobayes.priors import Beta, describe_prior

# brms's links of hurdle_cumulative() besides logit, withheld here (divergence D109)
_PARKED_LINKS = MappingProxyType({
    link: f"brms's {link} link of the hurdle_cumulative family is not verified against a frozen brms case here "
          "(docs/bounded-families.md, divergence D109)" for link in ("probit", "probit_approx", "cloglog", "cauchit")})


def _validate_response(spec, data) -> None:
    hurdle_ordinal_categories(spec.response, data[spec.response])


def _response_categories(spec, column):
    categories, codes, notes, _ = hurdle_ordinal_categories(spec.response, column)
    return categories, codes, notes


def _hurdle_category(spec, column) -> str:
    return hurdle_ordinal_categories(spec.response, column)[3]


def _check_formula(spec) -> None:
    if not spec.intercept:
        raise FormulaError(f"{spec.formula!r}: the hurdle_cumulative family's thresholds are its intercepts, so the intercept cannot "
                           "be removed ('0 +' or '- 1'); brms refuses it too ('Cannot remove the intercept in an ordinal model')")


def _bambi_setup(model, bambi_priors):
    """As the cumulative family's (``cumulative._bambi_setup``): formulae's intercept made to add nothing, the design uncentred,
    the thresholds ``Intercept`` an auxiliary parameter under the intercept prior; and ``hu`` under its own prior."""
    import bambi as bmb

    from autobayes.likelihoods import hurdle_cumulative_logit_family, no_intercept, ordered_thresholds

    spec = model.spec
    columns = [pop_term_design(t, model.data, spec.categoricals) for t in spec.bambi_terms]
    X = np.concatenate(columns, axis=1) if columns else np.zeros((len(model.data), 0))
    means = X.mean(0) if model.effective_center_predictors else np.zeros(X.shape[1])
    prior = model.priors.intercept
    labels = threshold_labels(model)
    dim = model.names["threshold_dim"]

    def thresholds(name, **kwargs):
        return ordered_thresholds(name, prior.name, dict(prior.params), labels, dim)

    terms = {k: v for k, v in bambi_priors.items() if k not in ("Intercept", "hu")}
    priors = {"mu": {**terms, "Intercept": bmb.Prior("NoIntercept", dist=no_intercept)},
              "Intercept": bmb.Prior("OrderedThresholds", dist=thresholds), "hu": bambi_priors["hu"]}
    return hurdle_cumulative_logit_family([t.name for t in spec.bambi_terms], means, dim), priors, False


def _thresholds(model, post):
    return post[model.names["thresholds"]].transpose("chain", "draw", model.names["threshold_dim"]).to_numpy()


def _expected_response(model, post, eta):
    """brms's ``posterior_epred``: ``P(y = k)`` per draw, row and category ``k = 0..K``, the hurdle first, from the draws of
    ``b_Intercept`` and ``hu``."""
    from autobayes.likelihoods import hurdle_cumulative_probabilities

    hu = post[model.names["hu"]].to_numpy()[..., None]
    return hurdle_cumulative_probabilities(eta, _thresholds(model, post)[..., None, :], hu)


def _draw_response(model, post, eta, mean, df, rng):
    """Codes ``0..K`` (brms's ``Y``) drawn from the K + 1 probabilities, the hurdle's first."""
    from autobayes.likelihoods import draw_categories

    return draw_categories(mean, rng) - 1


def _log_likelihood(model, post, eta):
    """brms's log_lik (C07): ``log P(y_i | mu_i, b_Intercept, hu)`` of the hurdle cumulative logit model, per draw."""
    from autobayes.families.pointwise import hurdle_ordered_logistic

    return hurdle_ordered_logistic(model.design.y, eta, _thresholds(model, post)[..., None, :], post[model.names["hu"]].to_numpy()[..., None])


def _mean_lines(model) -> list[str]:
    K = len(model.categories)
    return [f"  P(y_i = 0) = hu, the hurdle category {model.hurdle_category!r}; for the {K} ordered categories {_categories(model)},",
            f"  P(y_i <= k | y_i > 0) = expit(b_Intercept[k] - mu_i), k = 1..{K - 1}, and P(y_i = k) = (1 - hu) times the difference of",
            "  consecutive ones (brms's posterior_epred, categories 0..K). mu has no intercept: the thresholds b_Intercept[k] take",
            "  its place, and a larger mu moves probability towards the higher categories."]


def _likelihood_lines(model) -> list[str]:
    return [f"  hu ~ {describe_prior(model.priors.hu, lb=0, ub=1)}", "", "Likelihood",
            "  y_i ~ HurdleCumulative(mu_i, b_Intercept, hu): brms's hurdle_cumulative(\"logit\") (disc = 1); log P(y_i = 0) = log(hu),",
            "  log P(y_i = k) = log(1 - hu) - softplus(mu_i - b_Intercept[k]) - softplus(b_Intercept[k - 1] - mu_i) + log(1 -",
            "  exp(-(b_Intercept[k] - b_Intercept[k - 1]))) with the end terms dropped for the first and last category: finite with",
            "  a finite gradient for every finite mu (autobayes/likelihoods.py). hu is brms's hurdle probability, sampled as logit(hu).",
            "  The response is fitted as brms's Y: 0 for the hurdle category, k for the k-th ordered category."]


HURDLE_CUMULATIVE = FamilySpec(
    name="hurdle_cumulative", tag="hurdle cumulative (ordinal)", parent="mu", link="logit", link_scale="logit", mean_name="p",
    response_kind="ordinal", response_dtype=np.int64,
    scale_parameter=None, bambi_scale_parameter=None, fixed_parameters=NO_FIXED_PARAMETERS,
    bambi_family=lambda: None,                                      # built per model by bambi_setup
    validate_response=_validate_response,
    response_warnings=lambda spec, df: [],
    response_upper_bound=lambda model: len(model.categories),
    inverse_link=expit,                                             # P(y <= k | y > 0) = expit(tau_k - mu)
    expected_response=_expected_response,
    draw_response=_draw_response,
    manual_prediction=True, prediction_kinds=STANDARD_PREDICTION_KINDS,
    kind_error=("kind must be 'mean' (the probabilities of the hurdle and the ordered categories), 'response' (codes 0..K) or "
                "'linear' (mu)"),
    scale_rule="sigma, shape and phi do not apply to the hurdle_cumulative family; its thresholds are the intercept and its "
               "hurdle probability is hu",
    scale_missing_error=None,
    linpred_symbol="mu", scale_phrase="logit (latent)", intercept_meaning="thresholds",
    mean_lines=_mean_lines,
    likelihood_lines=_likelihood_lines,
    centred_intercept_note=(),
    deterministic_note="mu = linear predictor (no intercept); b_Intercept = Intercept + sum_p mean(x_p) * b_p, the thresholds",
    default_prior_link=None,                                        # brms's Intercept default is student_t(3, 0, 2.5)
    default_scale_prior=None,
    parked_links=_PARKED_LINKS,
    zero_parameter="hu",
    default_zero_prior=lambda: Beta(1, 1),                          # brms hu: beta(1, 1), lb 0, ub 1
    hurdle_category=_hurdle_category,
    response_categories=_response_categories,
    thresholds=True,
    bambi_setup=_bambi_setup,
    check_formula=_check_formula,
    intercept_lines=_intercept_lines,
    log_likelihood=_log_likelihood,                                 # C07: log_lik(), checked against the model's own density
)
