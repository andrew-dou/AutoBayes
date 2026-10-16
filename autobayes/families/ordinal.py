"""brms's sequential and adjacent-category ordinal families, ``sratio("logit")``, ``cratio("logit")`` and ``acat("logit")``,
with category-specific effects ``cs(x)`` (brms-parity D04, docs/ordinal-categorical.md); and the pieces they share with the
cumulative family's probit link (``cumulative.py``).

The response is coded as for the cumulative family (``cumulative.py``: positive integers ``1..K`` or an ordered pandas
Categorical, fitted as brms's 1-based ``Y``), ``mu`` is the linear predictor without an intercept and the K - 1 thresholds
take the intercept's place: brms samples them for the centred design, ``Intercept = b_Intercept - sum_p mean(x_p) b_p``, under
the intercept prior on every threshold, but as an unordered ``vector[nthres]`` (no ordering constraint, sampled as it is), and
reports ``b_Intercept``, the thresholds of the raw design, which the likelihood reads. With ``d_k = b_Intercept[k] - mu`` (minus
``x * bcs[k]`` for each category-specific effect, brms's ``Xcs bcs``):

- ``sratio``, the stopping ratio: ``P(y = k | y >= k) = expit(d_k)``;
- ``cratio``, the continuation ratio: ``P(y > k | y >= k) = expit(-d_k)``, under the logit link the same probabilities as
  ``sratio``'s (the logistic is symmetric; brms's two parameterisations agree there, as the frozen cases show);
- ``acat``, adjacent categories: ``log(P(y = k + 1) / P(y = k)) = -d_k``.

The likelihood is ``likelihoods.ordinal_family`` with the stable rows of ``likelihoods.ordinal_logp_rows``; ``predict()``,
``log_lik()`` and the response draws use ``likelihoods.ordinal_log_probabilities``. Checked against brms in the oracle cases
``sratio_varint``, ``cratio_varint``, ``acat_varint`` and ``acat_cs``, not assumed. Only the logit link (brms's others are
withheld, ``PARKED_ORDINAL_LINKS``), ``disc = 1`` and brms's flexible thresholds are offered (divergence D38).
"""
from __future__ import annotations

from functools import lru_cache
from types import MappingProxyType

import numpy as np
from scipy.special import expit

from autobayes.families.cumulative import PARKED_ORDINAL_LINKS, _categories, _response_categories, _validate_response, threshold_labels
from autobayes.families.spec import NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, FamilySpec
from autobayes.formula import FormulaError, pop_term_design

# brms offers every ordinal link for these families; the logit link alone is verified here (divergence D38)
_PARKED_LINKS = MappingProxyType({**PARKED_ORDINAL_LINKS, "probit": PARKED_ORDINAL_LINKS["cloglog"].replace("cloglog", "probit")})


# -- shared with the cumulative family's probit link -----------------------------------------------------------------------
def ordinal_bambi_setup(kind: str, *, ordered: bool):
    """The family's ``bambi_setup`` (``cumulative._bambi_setup``'s construction): Bambi is handed the formula with formulae's
    intercept made to add nothing, the design uncentred, and the thresholds ``Intercept`` as the family's auxiliary parameter
    under the intercept prior, ordered (Stan's ordered coordinates, ``likelihoods.ordered_thresholds``) or not
    (``likelihoods.unordered_thresholds``); the likelihood is ``likelihoods.ordinal_family(kind, ...)``, which reads the
    category-specific effects through ``Model._add_category_specific_effects`` when the formula has ``cs()`` terms."""
    def setup(model, bambi_priors):
        import bambi as bmb

        from autobayes.likelihoods import no_intercept, ordered_thresholds, ordinal_family, unordered_thresholds

        spec = model.spec
        columns = [pop_term_design(t, model.data, spec.categoricals) for t in spec.bambi_terms]   # cs() and mo() terms are not centred
        X = np.concatenate(columns, axis=1) if columns else np.zeros((len(model.data), 0))
        means = X.mean(0) if model.effective_center_predictors else np.zeros(X.shape[1])
        prior = model.priors.intercept
        labels = threshold_labels(model)
        dim = model.names["threshold_dim"]
        make = ordered_thresholds if ordered else unordered_thresholds

        def thresholds(name, **kwargs):
            return make(name, prior.name, dict(prior.params), labels, dim)

        terms = {k: v for k, v in bambi_priors.items() if k != "Intercept"}
        priors = {"mu": {**terms, "Intercept": bmb.Prior("NoIntercept", dist=no_intercept)},
                  "Intercept": bmb.Prior("OrderedThresholds" if ordered else "Thresholds", dist=thresholds)}
        cs = model._add_category_specific_effects if spec.cs_terms else None
        return ordinal_family(kind, [t.name for t in spec.bambi_terms], means, dim, cs), priors, False

    return setup


def _gaps(model, post, eta):
    """``d = b_Intercept - eta`` per draw, row and threshold, from the draws of ``b_Intercept``; ``eta`` is ``(chain, draw, row)``,
    or ``(chain, draw, row, threshold)`` with category-specific effects."""
    from autobayes.likelihoods import ordinal_gaps

    tau = post[model.names["thresholds"]].transpose("chain", "draw", model.names["threshold_dim"]).to_numpy()
    return ordinal_gaps(np.asarray(eta, dtype=float), tau[..., None, :], bool(model.spec.cs_terms))


def ordinal_expected_response(kind: str):
    """brms's ``posterior_epred``: ``P(y = k)`` per draw, row and category, the exponential of the stable log probabilities."""
    def expected(model, post, eta):
        from autobayes.likelihoods import ordinal_log_probabilities

        return np.exp(ordinal_log_probabilities(kind, _gaps(model, post, eta)))

    return expected


def ordinal_log_likelihood(kind: str):
    """brms's log_lik (C07): ``log P(y_i | mu_i, b_Intercept)`` per draw and row."""
    def log_likelihood(model, post, eta):
        from autobayes.families.pointwise import ordinal

        return ordinal(model.design.y, _gaps(model, post, eta), kind)

    return log_likelihood


def _draw_response(model, post, eta, mean, df, rng):
    from autobayes.likelihoods import draw_categories

    return draw_categories(mean, rng)


# -- sratio, cratio, acat -----------------------------------------------------------------------------------------------
def _predictor(model) -> str:
    """How ``describe()`` writes the predictor of threshold k: ``mu_i``, or ``mu_i[k]`` with category-specific effects."""
    return "mu_i[k]" if model.spec.cs_terms else "mu_i"


_MEANS = {
    "sratio": ("P(y_i = k | y_i >= k) = expit(b_Intercept[k] - {p}), k = 1..{k1} (the stopping ratio),",
               "  P(y_i = k) = expit(b_Intercept[k] - {p}) * prod_{{j<k}} (1 - expit(b_Intercept[j] - mu_i[j])) (brms's posterior_epred)."),
    "cratio": ("P(y_i > k | y_i >= k) = expit({p} - b_Intercept[k]), k = 1..{k1} (the continuation ratio),",
               "  P(y_i = k) = (1 - expit({p} - b_Intercept[k])) * prod_{{j<k}} expit(mu_i[j] - b_Intercept[j]) (brms's "
               "posterior_epred); under the logit link these are sratio's probabilities."),
    "acat": ("log(P(y_i = k + 1) / P(y_i = k)) = {p} - b_Intercept[k], k = 1..{k1} (adjacent categories),",
             "  P(y_i = k) is proportional to exp(sum_{{j<k}} (mu_i[j] - b_Intercept[j])) (brms's posterior_epred)."),
}

_ROWS = {
    "sratio": ("brms's sratio_logit_lpmf", "-softplus({p} - b_Intercept[y]) [y < K] - sum_{{j<y}} softplus(b_Intercept[j] - mu_i[j])"),
    "cratio": ("brms's cratio_logit_lpmf", "-softplus({p} - b_Intercept[y]) [y < K] - sum_{{j<y}} softplus(b_Intercept[j] - mu_i[j])"),
    "acat": ("brms's acat_logit_lpmf", "s_y - log_sum_exp(s_1, ..., s_K), s_1 = 0, s_k = sum_{{j<k}} (mu_i[j] - b_Intercept[j])"),
}


def _mean_lines(name: str):
    def lines(model) -> list[str]:
        K = len(model.categories)
        first, second = _MEANS[name]
        out = ["  " + first.format(p=_predictor(model), k1=K - 1), f"  for the {K} ordered categories {_categories(model)};",
               second.format(p=_predictor(model))]
        out += ["  mu has no intercept: the thresholds b_Intercept[k] take its place, and a larger mu moves probability towards the",
                "  higher categories."]
        if model.spec.cs_terms:
            cs = " + ".join(f"x_{t.operands[0]},i * bcs_{t.operands[0]}[k]" for t in model.spec.cs_terms)
            out += [f"  mu_i[k] = mu_i + {cs}: each category-specific effect cs() has one coefficient per threshold (brms's bcs; its",
                    "  column is never centred), the threshold's own predictor (brms's posterior_linpred, one column per threshold)."]
        else:
            out += ["  mu_i[k] = mu_i for every threshold (no category-specific effect)."]
        return out

    return lines


def _likelihood_lines(name: str):
    def lines(model) -> list[str]:
        what, row = _ROWS[name]
        return ["", "Likelihood",
                f"  y_i ~ {name}(\"logit\") (disc = 1), {what}, written as",
                f"    log P(y_i = y) = {row.format(p=_predictor(model))},",
                "  a sum of stable terms, finite with a finite gradient for every finite mu (autobayes/likelihoods.py). The response",
                "  is fitted as its 1-based category (brms's Y)."]

    return lines


def _check_formula(name: str):
    def check(spec) -> None:
        if not spec.intercept:
            raise FormulaError(f"{spec.formula!r}: the {name} family's thresholds are its intercepts, so the intercept cannot be "
                               "removed ('0 +' or '- 1'); brms refuses it too ('Cannot remove the intercept in an ordinal model')")

    return check


def _intercept_lines(model) -> list[str]:
    K = len(model.categories)
    if model.design.centered:
        return ["  Predictors are centred for the thresholds, as brms centres them: the sampled thresholds are",
                f"  Intercept[k] = b_Intercept[k] - sum_p mean(x_p) * b_p, k = 1..{K - 1} (brms's Intercept, an unordered vector",
                "  sampled as it is, with no ordering constraint), the thresholds at the predictor means; the prior on the",
                "  intercept applies to each Intercept[k]. The graph's b_Intercept (brms's b_Intercept, the thresholds of the raw",
                "  design) is what the likelihood and predict() use; the posterior keeps both, as brms does."]
    return ["  Predictors are not centred (center_predictors=False): the sampled thresholds Intercept[k] are the thresholds of",
            f"  the raw design, k = 1..{K - 1}, an unordered vector, and b_Intercept equals them."]


@lru_cache(maxsize=None)
def _spec(name: str) -> FamilySpec:
    kind = f"{name}_logit"
    return FamilySpec(
        name=name, tag=f"{name} (ordinal)", parent="mu", link="logit", link_scale="logit", mean_name="p",
        response_kind="ordinal", response_dtype=np.int64,
        scale_parameter=None, bambi_scale_parameter=None, fixed_parameters=NO_FIXED_PARAMETERS,
        bambi_family=lambda: None,                                      # built per model by bambi_setup
        validate_response=_validate_response,
        response_warnings=lambda spec, df: [],
        response_upper_bound=lambda model: len(model.categories),
        inverse_link=expit,
        expected_response=ordinal_expected_response(kind),
        draw_response=_draw_response,
        manual_prediction=True, prediction_kinds=STANDARD_PREDICTION_KINDS,
        kind_error="kind must be 'mean' (the category probabilities), 'response' (1-based category draws) or 'linear' (mu)",
        scale_rule=f"sigma, shape and phi do not apply to the {name} family; its thresholds are the intercept",
        scale_missing_error=None,
        linpred_symbol="mu", scale_phrase="logit (latent)", intercept_meaning="thresholds",
        mean_lines=_mean_lines(name),
        likelihood_lines=_likelihood_lines(name),
        centred_intercept_note=(),
        deterministic_note="mu = linear predictor (no intercept); b_Intercept = Intercept + sum_p mean(x_p) * b_p, the thresholds",
        default_prior_link=None,                                        # brms's Intercept default is student_t(3, 0, 2.5)
        default_scale_prior=None,
        parked_links=_PARKED_LINKS,
        response_categories=_response_categories,
        thresholds=True,
        ordered_thresholds=False,                                       # brms's vector[nthres] Intercept
        category_specific=True,                                         # cs(x): verified against acat_cs (the same term for sratio, cratio)
        bambi_setup=ordinal_bambi_setup(kind, ordered=False),
        check_formula=_check_formula(name),
        intercept_lines=_intercept_lines,
        log_likelihood=ordinal_log_likelihood(kind),
    )


SRATIO, CRATIO, ACAT = _spec("sratio"), _spec("cratio"), _spec("acat")
