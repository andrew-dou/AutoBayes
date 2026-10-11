"""Finite mixtures: brms's ``mixture(gaussian, ..., gaussian)`` (brms-parity C05, docs/mixtures.md).

A mixture of K Gaussian components with the same formula: component k has its own linear predictor ``mu_k`` (brms's
``mu<k>``: its own centred intercept and coefficients on the same design), its own residual SD ``sigma_k`` (brms's
``sigma<k>``), and the mixing proportions ``theta`` (brms's ``simplex[K] theta``) weigh the components:
``y_i ~ sum_k theta_k Normal(mu_k,i, sigma_k)``. Under brms's default ``order = "mu"`` the components are identified by
their intercepts, ``Intercept_mu1 < ... < Intercept_muK`` (brms's ``ordered[K] ordered_Intercept``, Stan's ``ordered``
transform). The graph reuses three earlier pieces: Bambi's categorical family's per-category design (B05: every coefficient
with a trailing axis, here over the components; ``likelihoods.gaussian_mixture_family``), B05's ordered vector for the
intercepts (``likelihoods.ordered_thresholds``, with one prior per component) and B03's stick-breaking simplex for ``theta``
(``monotonic.simplex_rv``). The priors are one ``Priors`` per component, the Gaussian model's own (``intercept``, ``b``,
``sigma``), and ``theta`` (``Priors(components=[...], theta=ab.Dirichlet(...))``).

Offered: Gaussian components (any number, at least two), ``order = "mu"``, one formula for every component and
population-level terms of every kind the term model has (numeric, declared categorical, interactions, transforms).
Refused, naming the construct: other component families (divergence D52), ``order = "none"`` (D53), formulas per component
or on ``sigma<k>`` and ``theta`` (D54), group-level terms, offsets, addition terms and ``mo()`` (D55); ``0 +`` and
``center_predictors=False``, which leave no intercept to order, as brms refuses them. Checked against brms in the oracle
cases ``mixture_gauss_gauss``, ``mixture_gauss3_categorical`` and ``mixture_gauss_default_priors``, not assumed.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from autobayes.families.spec import NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, FamilySpec
from autobayes.formula import FormulaError, validate_numeric_response

COMPONENT_FAMILIES = ("gaussian",)        # the component families offered (brms takes any, and mixes them: divergence D52)
ORDER = "mu"                              # the identification offered: brms's default for components of one family (D53)
DOC = "docs/mixtures.md"


# -- the components and their names -------------------------------------------------------------------------------------
def check_mixture(formula, families, order, priors, center_predictors: bool) -> tuple[str, ...]:
    """The component families of ``mixture(families, order=order)``, checked before the formula is parsed: a list of at
    least two registered families, each one this library offers as a component (D52), brms's ``order = "mu"`` (``None``, the
    default for components of one family, ``"mu"`` or ``True``; ``"none"`` and ``False`` are D53), one formula string (a list
    would be brms's ``bf(y ~ x, mu2 ~ z, theta1 ~ w)``, D54), centred predictors (brms orders the centred intercepts and
    refuses ``center = FALSE`` under ``order = "mu"``), and, when ``priors`` gives its components, one per family."""
    from autobayes.families import get_family

    if not isinstance(formula, str):
        raise FormulaError(
            f"{formula!r}: a mixture takes one formula, the same for every component; a list would give components or "
            "their sigma<k> and theta formulas of their own (brms's bf(y ~ x, mu2 ~ z, theta1 ~ w)), which is not offered "
            f"here ({DOC}, divergence D54)")
    if families is None:
        raise ValueError("a mixture needs its component families: families=['gaussian', 'gaussian'] (brms's "
                         "mixture(gaussian, gaussian))")
    if isinstance(families, str) or not isinstance(families, Sequence) or not all(isinstance(f, str) for f in families):
        raise ValueError(f"families must be a list of family names, one per component (['gaussian', 'gaussian']), got {families!r}")
    families = tuple(families)
    if len(families) < 2:
        raise ValueError(f"a mixture has at least two components, got {list(families)} (brms: 'Expecting at least 2 mixture "
                         "components.')")
    for name in families:
        get_family(name)                                          # an unknown family is refused with the registry's list
        if name not in COMPONENT_FAMILIES:
            raise ValueError(f"a {name} mixture component is not offered: the components are {list(COMPONENT_FAMILIES)} in "
                             f"this slice (brms mixes any families; {DOC}, divergence D52)")
    if not (order is None or order is True or (isinstance(order, str) and order == ORDER)):
        if order is False or (isinstance(order, str) and order == "none"):
            raise ValueError(f"order={order!r} (brms's order = 'none': components not identified by their intercepts) is not "
                             f"offered; the components are ordered by their intercepts, brms's default order = 'mu' ({DOC}, "
                             "divergence D53)")
        raise ValueError(f"order must be 'mu' (brms's default for components of one family) or None, got {order!r}")
    if not center_predictors:
        raise FormulaError(
            "center_predictors=False is not available for a mixture: its components are identified by ordering their "
            "intercepts at the predictor means (brms's ordered_Intercept), and brms refuses it too ('Identifying mixture "
            "components via ordering requires population-level intercepts to be present.')")
    given = getattr(priors, "components", None)
    if isinstance(given, (list, tuple)) and len(given) != len(families):
        from autobayes.priors import PriorError

        raise PriorError(f"the mixture has {len(families)} components {list(families)}; Priors(components=[...]) gives "
                         f"{len(given)}, one Priors per component in the same order")
    return families


def roles(K: int) -> dict:
    """PyMC names of a mixture of ``K`` components (``Model.mixture_names``): ``"dpars"`` brms's names of the components'
    linear predictors (``mu1``, ...), ``"sigma"`` each component's scale (``sigma1``, ...; brms's names), ``"theta"`` the
    mixing proportions, ``"coordinate"`` the component dimension of the intercepts, coefficients and ``theta``
    (``mixture_dim``) and ``"labels"`` its coordinates (``"1"``..``"K"``)."""
    labels = tuple(str(k) for k in range(1, K + 1))
    return {"dpars": tuple(f"mu{k}" for k in labels), "sigma": tuple(f"sigma{k}" for k in labels), "theta": "theta",
            "coordinate": "mixture_dim", "labels": labels}


def _check_formula(spec) -> None:
    if not spec.intercept:
        raise FormulaError(f"{spec.formula!r}: a mixture's components are identified by ordering their intercepts (brms's "
                           "order = 'mu'), so the intercept cannot be removed ('0 +' or '- 1'); brms refuses it too")
    if spec.group_terms:
        raise FormulaError(f"{spec.formula!r}: a mixture takes population-level terms only in this slice; a group-level term "
                           f"('{spec.group_terms[0].canonical}') would give every component its own group effects and SD (brms's "
                           f"group term in each mu<k>), which is not verified here ({DOC}, divergence D55)")
    if spec.offset is not None:
        raise FormulaError(f"{spec.formula!r}: a mixture takes no offset term in this slice (brms adds it to every component's "
                           f"linear predictor; {DOC}, divergence D55)")
    if getattr(spec, "gp_terms", ()):
        raise FormulaError(f"{spec.formula!r}: a mixture takes no Gaussian process in this slice ('{spec.gp_terms[0].term}' would "
                           f"give every component its own process, brms's gp() in each mu<k>, which is not verified here; {DOC}, "
                           "divergence D55)")


def _validate_response(spec, data) -> None:
    validate_numeric_response(spec.response, data[spec.response], "Gaussian mixture")


# -- the graph ------------------------------------------------------------------------------------------------------------
def _bambi_setup(model, bambi_priors):
    """Bambi is handed the formula with its own intercept, centred design and Bambi's categorical family's per-category
    layout over the components (``likelihoods.gaussian_mixture_family``); the intercept is B05's ordered vector with each
    component's own prior (``likelihoods.ordered_thresholds``: brms's ``ordered_Intercept``), and the likelihood creates
    each ``sigma<k>`` and ``theta`` under their priors. The term priors come from ``Priors.to_bambi``, one value per
    component."""
    import bambi as bmb

    from autobayes.likelihoods import gaussian_mixture_family, ordered_thresholds
    from autobayes.priors import dirichlet_concentration

    r, comps = model.mixture_names, model.priors.components
    first = comps[0].intercept                                    # one distribution family for every component (validated)
    params = {k: np.array([float(c.intercept.params[k]) for c in comps]) for k in first.params}

    def intercepts(name, dims=None, **kwargs):
        return ordered_thresholds(name, first.name, params, r["labels"], r["coordinate"])

    priors = {**bambi_priors, "Intercept": bmb.Prior("OrderedIntercepts", dist=intercepts)}
    family = gaussian_mixture_family(r["labels"], r["coordinate"], list(zip(r["sigma"], (c.sigma for c in comps))), r["theta"],
                                     dirichlet_concentration(model.priors.theta, len(comps)))
    return family, priors, model.effective_center_predictors


def check_graph(model) -> None:
    """The mixture's graph as built, checked: the intercept a free vector of K entries over the component coordinate sampled
    on Stan's ``ordered`` coordinates, every coefficient with the component axis last, each ``sigma<k>`` a free scalar on
    the log scale, ``theta`` a free simplex of K on its K - 1 stick-breaking coordinates, nothing else free; and the
    graph's ``mu`` at a point with a distinct value per component equal to ``a_c[k] + (X - mean(X)) b[:, k]`` computed from
    the design."""
    from pymc.distributions.transforms import Ordered

    from autobayes.monotonic import StickBreakingTransform

    pmm, r, n = model.pymc_model, model.mixture_names, model.names
    K, dim = len(model.components), r["coordinate"]
    if tuple(str(v) for v in pmm.coords.get(dim, ())) != r["labels"]:
        raise RuntimeError(f"PyMC coordinate {dim!r} is not the components {list(r['labels'])}")
    intercept = pmm[n["intercept"]]
    value = pmm.rvs_to_values[intercept]
    if (intercept not in pmm.free_RVs or value.name != f"{n['intercept']}_ordered__" or tuple(value.type.shape) != (K,)
            or not isinstance(pmm.rvs_to_transforms[intercept], Ordered)
            or tuple(pmm.named_vars_to_dims.get(n["intercept"], ())) != (dim,)):
        raise RuntimeError(f"{n['intercept']!r} is not a free vector of {K} over {dim!r} on Stan's ordered coordinates")
    for term in model.spec.bambi_terms:
        dims = tuple(pmm.named_vars_to_dims.get(term.name, ()))
        if pmm[term.name] not in pmm.free_RVs or dims != ((f"{term.name}_dim",) if term.levels else ()) + (dim,):
            raise RuntimeError(f"the coefficients of {term.name!r} have dims {dims}; expected the component axis {dim!r} last")
    for scale in r["sigma"]:
        if pmm[scale] not in pmm.free_RVs or pmm[scale].type.ndim != 0 or pmm.rvs_to_values[pmm[scale]].name != f"{scale}_log__":
            raise RuntimeError(f"{scale!r} is not a free positive scalar sampled on the log scale")
    theta = pmm[r["theta"]]
    tv = pmm.rvs_to_values[theta]
    if (theta not in pmm.free_RVs or tv.name != f"{r['theta']}_stickbreaking__" or tuple(tv.type.shape) != (K - 1,)
            or not isinstance(pmm.rvs_to_transforms[theta], StickBreakingTransform)
            or tuple(pmm.named_vars_to_dims.get(r["theta"], ())) != (dim,)):
        raise RuntimeError(f"{r['theta']!r} is not a free simplex of {K} over {dim!r} on Stan's stick-breaking coordinates")
    expected = {f"{n['intercept']}_ordered__", *(t.name for t in model.spec.bambi_terms), *(f"{s}_log__" for s in r["sigma"]),
                f"{r['theta']}_stickbreaking__"}
    if {v.name for v in pmm.value_vars} != expected:
        raise RuntimeError(f"the mixture's free variables are {sorted(v.name for v in pmm.value_vars)}; expected {sorted(expected)}")
    d = model.design
    a = np.linspace(-1.0, 1.0, K) * 0.7 + 0.1                     # increasing, distinct per component
    B = 0.3 + 0.1 * np.arange(d.X_raw.shape[1])[:, None] - 0.25 * np.arange(K)[None, :]
    point = value_point(model, intercept=a, b=B, sigma=np.linspace(0.5, 1.5, K), theta=np.full(K, 1.0 / K))
    fn = pmm.compile_fn(pmm.replace_rvs_by_values([pmm[n["linpred"]]]), inputs=pmm.value_vars, point_fn=True,
                        on_unused_input="ignore", mode="FAST_COMPILE")
    mu = np.asarray(fn(point)[0], dtype=float)
    want = a + (d.X_raw - d.x_means) @ B
    if mu.shape != want.shape or not np.allclose(mu, want, rtol=1e-12, atol=1e-12):
        raise RuntimeError(f"the graph's {n['linpred']!r} is not a_c[k] + (X - mean(X)) b[:, k] per component")


def value_point(model, *, intercept=None, b=None, sigma=None, theta=None, **others) -> dict[str, np.ndarray]:
    """``Model.value_point`` of a mixture: ``intercept`` the K centred intercepts (brms's ``Intercept_mu<k>``), strictly
    increasing, stored on Stan's ordered coordinates (``Intercept_ordered__``); ``b`` the coefficients, ``(columns, K)``, one
    column per component; ``sigma`` the K residual SDs (brms's ``sigma<k>``), each stored as its log; ``theta`` the mixing
    proportions, a simplex of K, stored on Stan's stick-breaking coordinates. Every other argument does not apply."""
    from autobayes.monotonic import check_simplex, stick_breaking_free

    given = sorted(k for k, v in others.items() if v is not None)
    if given:
        raise ValueError(f"{given} do not apply to a mixture: give intercept (K increasing centred intercepts), b (columns x K), "
                         "sigma (K values) and theta (a simplex of K)")
    r, n, spec = model.mixture_names, model.names, model.spec
    K, P = len(model.components), len(spec.columns)
    a = np.atleast_1d(np.asarray(intercept if intercept is not None else np.nan, dtype=float))
    if a.shape != (K,) or not np.isfinite(a).all() or not (np.diff(a) > 0).all():
        raise ValueError(f"intercept must be the {K} components' centred intercepts, finite and strictly increasing (brms's "
                         "ordered_Intercept, order = 'mu')")
    B = np.asarray(b if b is not None else np.zeros((P, K)), dtype=float)
    if B.shape == (K,) and P == 1:
        B = B[None, :]
    if B.shape != (P, K) or not np.isfinite(B).all():
        raise ValueError(f"b must be an array of shape ({P}, {K}): one row per design column {list(spec.columns)}, one column "
                         "per component")
    s = np.atleast_1d(np.asarray(sigma if sigma is not None else np.nan, dtype=float))
    if s.shape != (K,) or not np.isfinite(s).all() or not (s > 0).all():
        raise ValueError(f"sigma must be the {K} components' residual SDs, finite and positive (brms's sigma1..sigma{K})")
    if theta is None:
        raise ValueError(f"theta (the mixing proportions, a simplex of {K}) is required for a mixture")
    point = {f"{n['intercept']}_ordered__": np.r_[a[0], np.log(np.diff(a))]}
    i = 0
    for term in spec.bambi_terms:
        k = len(term.columns)
        point[term.name] = np.array(B[i:i + k]) if term.levels else np.array(B[i])
        i += k
    for name, value in zip(r["sigma"], s):
        point[f"{name}_log__"] = np.asarray(np.log(value))
    point[f"{r['theta']}_stickbreaking__"] = stick_breaking_free(check_simplex(theta, K, "theta"))
    expected = {v.name for v in model.pymc_model.value_vars}
    if set(point) != expected:
        raise RuntimeError(f"value names {sorted(point)} do not match PyMC's {sorted(expected)}")
    return point


# -- prediction -----------------------------------------------------------------------------------------------------------
def _draws(model, post):
    """(chain, draw, K) posterior draws of theta and of the components' sigmas."""
    r = model.mixture_names
    theta = post[r["theta"]].transpose("chain", "draw", r["coordinate"]).to_numpy()
    sigma = np.stack([post[s].to_numpy() for s in r["sigma"]], axis=-1)
    return theta, sigma


def _expected_response(model, post, eta):
    """brms's ``posterior_epred``: ``sum_k theta_k mu_k`` per draw and row, ``eta`` the components' linear predictors
    ``(chain, draw, row, K)``."""
    from autobayes.likelihoods import mixture_mean

    theta, _ = _draws(model, post)
    return mixture_mean(eta, theta[..., None, :])


def _draw_response(model, post, eta, mean, df, rng):
    """brms's ``posterior_predict``: a component by theta per draw and row, then its Normal."""
    from autobayes.likelihoods import gaussian_mixture_random

    theta, sigma = _draws(model, post)
    return gaussian_mixture_random(eta, sigma[..., None, :], theta[..., None, :], rng)


# -- description ----------------------------------------------------------------------------------------------------------
def default_prior_labels(model) -> list[str]:
    """``Model.default_prior_labels`` of a mixture: ``mu<k>:intercept``, ``mu<k>:b:<column>`` and ``sigma<k>`` per component,
    then ``theta``, where ``ab.default_priors`` gave the prior."""
    found = []
    for k, c in enumerate(model.priors.components, start=1):
        found += [(f"mu{k}:intercept", c.intercept)] + [(f"mu{k}:b:{col}", p) for col, p in c.coefficient_priors(model.spec).items()]
        found += [(f"sigma{k}", c.sigma)]
    found += [("theta", model.priors.theta)]
    return [label for label, prior in found if getattr(prior, "source", "user") == "default"]


def prior_labels(model) -> dict[str, object]:
    """Every prior of a mixture by the label fit records use: ``mu<k>:intercept``, ``mu<k>:b:<column>``, ``sigma<k>``, then
    ``theta``."""
    out = {}
    for k, c in enumerate(model.priors.components, start=1):
        out[f"mu{k}:intercept"] = c.intercept
        out.update({f"mu{k}:b:{col}": p for col, p in c.coefficient_priors(model.spec).items()})
        out[f"sigma{k}"] = c.sigma
    out["theta"] = model.priors.theta
    return out


def describe(model) -> str:
    """``Model.describe()`` of a mixture: the components' linear predictors and their identification, the mixing
    proportions, the likelihood, each component's priors and theta's, and the variables sampled."""
    from autobayes.model import _linear_terms
    from autobayes.priors import describe_prior, dirichlet_concentration

    s, d, pr, n, r = model.spec, model.design, model.priors, model.names, model.mixture_names
    K, fams = len(model.components), ", ".join(model.components)
    xs = " + ".join(_linear_terms(s, d.centered))
    ks = ", ".join(r["dpars"])

    def each(symbol, sep):                                   # a_c[1] < a_c[2] < a_c[3], or a_c[1] < ... < a_c[K]
        return sep.join(f"{symbol}[{k}]" for k in range(1, K + 1)) if K <= 4 else f"{symbol}[1]{sep}...{sep}{symbol}[{K}]"

    lines = [f"AutoBayes mixture model of {K} Gaussian components (population-level only)",
             f"  formula   : {s.formula}"]
    if s.canonical_formula != s.formula:
        lines += [f"  expanded  : {s.canonical_formula}   (canonical form handed to Bambi)"]
    lines += [f"  family    : mixture({fams}), order = '{ORDER}' (brms's mixture() and its default for components of one family), "
              "identity link",
              f"  data      : n = {d.n} observations, no group-level term (population-level model)"]
    for t in s.transform_terms:
        tr, col = t.transform, d.X_raw[:, s.columns.index(t.name)]
        lines += [f"  {t.name:<10}: in-formula transform of {tr.column!r}, {tr.expression(f'{tr.column}_i')}, range {col.min():.4f} to "
                  f"{col.max():.4f} over the fitted rows; recomputed on new rows; centred like any column"]
    for cat in s.categoricals:
        labels = model.data[cat.name].astype(str).to_numpy()
        props = ", ".join(f"{l} {np.mean(labels == l):.3f}" for l in cat.levels)
        lines += [f"  {cat.name:<10}: categorical, {'full-level coding' if cat.full_rank else 'treatment coding'}, reference "
                  f"{cat.reference!r}; levels (sorted) {', '.join(cat.levels)}; coefficients b_{cat.name}[{', '.join(cat.coefficients)}] "
                  f"per component; proportions among fitted rows: {props}"]
    lines += ["", "Components",
              f"  mu_k,i = a_c[k]{' + ' + xs if xs else ''}, k = 1..{K}: one linear predictor per component (brms's {ks}), each",
              "  with its own intercept a_c[k] and coefficients b[k] (every b_<term> above has one value per component) on the",
              "  same design.",
              "  Predictors are centred at their sample means, as brms centres them: a_c[k] is component k's mean at the predictor",
              "  means (brms's Intercept_mu<k>), where the prior on it applies; the posterior reports the raw-scale intercept",
              "  a[k] = a_c[k] - sum_p mean(x_p) * b_p[k] (brms's b_mu<k>_Intercept), computed after sampling.",
              f"  Identification (order = '{ORDER}'): {each('a_c', ' < ')} (brms's ordered[{K}] ordered_Intercept), sampled on Stan's",
              "  ordered coordinates (a_c[1], log(a_c[k] - a_c[k - 1])); component k is the one with the k-th smallest centred",
              "  intercept.",
              f"  Mixing proportions theta = ({each('theta', ', ')}), positive and summing to 1 (brms's simplex[{K}] theta), on",
              "  Stan's stick-breaking coordinates; sigma_k (brms's sigma<k>) is component k's residual SD.",
              "", "Likelihood",
              "  y_i ~ sum_k theta[k] * Normal(mu_k,i, sigma_k): row i's log density is log_sum_exp_k(log theta[k] +",
              "  normal_lpdf(y_i | mu_k,i, sigma_k)), brms's log_mix (autobayes/likelihoods.py).",
              "  E[y_i] = sum_k theta[k] * mu_k,i (brms's posterior_epred, predict(kind='mean')); predict(kind='linear') returns every",
              "  mu_k with a component dimension, and predict(kind='response') draws a component by theta, then its Normal.",
              "", "Priors"]
    for k, c in enumerate(pr.components, start=1):                 # within a component's block: its a_c[k], b[k], sigma_k
        lines += [f"  component {k} (brms's dpar mu{k} and class sigma{k}):",
                  f"    a_c ~ {describe_prior(c.intercept)}"]
        lines += [f"    b_{col} ~ {describe_prior(p)}" for col, p in c.coefficient_priors(s).items()]
        lines += [f"    sigma{k} ~ {describe_prior(c.sigma, lb=0)}"]
    con = ", ".join(f"{v:g}" for v in dirichlet_concentration(pr.theta, K))
    lines += [f"  theta ~ {describe_prior(pr.theta)}   (brms class theta; concentrations {con})",
              "  The intercepts' priors are added at the ordered a_c[k] as brms adds them (each component's density, not",
              "  renormalised to the ordered region)."]
    defaults = default_prior_labels(model)
    if defaults:
        from autobayes.default_priors import BRMS_VERSION

        lines += ["", "Default priors (ab.default_priors(); docs/default-priors.md)",
                  f"  {', '.join(defaults)}: brms {BRMS_VERSION}'s default_prior() for this formula, data and mixture, marked",
                  "  (brms default) above with brms's spelling: each component the Gaussian family's defaults, theta dirichlet(1);",
                  "  a flat prior is improper: it adds nothing to the log density, and prior_predictive() is refused."]
    for w in model.warnings:
        lines += ["", "WARNING: " + w]
    lines += ["", "PyMC variables sampled (value space)"]
    roles = {f"{n['intercept']}_ordered__": f"a_c[1..{K}], the components' centred intercepts, on Stan's ordered coordinates "
                                            f"(dims {r['coordinate']})"}
    for t in s.bambi_terms:
        roles[t.name] = (f"b_{t.name}[{', '.join(t.levels)}] (matrix; dims {t.name}_dim x {r['coordinate']})" if t.levels
                         else f"b_{t.name}[1..{K}] (vector; dims {r['coordinate']})")
    roles.update({f"{scale}_log__": f"log({scale})" for scale in r["sigma"]})
    roles[f"{r['theta']}_stickbreaking__"] = (f"{r['theta']}[1..{K}] on Stan's stick-breaking simplex coordinates ({K - 1} "
                                              f"entr{'y' if K == 2 else 'ies'}; brms's theta); dims {r['coordinate']}")
    for v in model.pymc_model.value_vars:
        lines.append(f"  {v.name:<24} {roles.get(v.name, f'b_{v.name}')}")
    lines.append(f"Deterministic: {n['linpred']} = the components' linear predictors mu_k,i (rows x {K}); the mixture's mean "
                 "and response draws are computed by predict()")
    return "\n".join(lines)


def _mean_lines(model) -> list[str]:
    return ["  E[y_i] = sum_k theta[k] * mu_k,i, the components' means weighed by the mixing proportions (brms's posterior_epred)."]


def _likelihood_lines(model) -> list[str]:
    return ["", "Likelihood", "  y_i ~ sum_k theta[k] * Normal(mu_k,i, sigma_k) (brms's log_mix of normal_lpdf)"]


MIXTURE = FamilySpec(
    name="mixture", tag="mixture", parent="mu", link="identity", link_scale=None, mean_name="mean",
    response_kind="continuous", response_dtype=None,
    scale_parameter=None, bambi_scale_parameter=None, fixed_parameters=NO_FIXED_PARAMETERS,   # sigma<k> per component instead
    bambi_family=lambda: None,                                      # built per model by bambi_setup
    validate_response=_validate_response,
    response_warnings=lambda spec, df: [],
    response_upper_bound=lambda model: None,
    inverse_link=lambda eta: eta,                                   # each Gaussian component's mean is its mu_k
    expected_response=_expected_response,
    draw_response=_draw_response,
    manual_prediction=True, prediction_kinds=STANDARD_PREDICTION_KINDS,
    kind_error="kind must be 'mean' (the mixture's expected response), 'response' (draws) or 'linear' (every component's mu)",
    scale_rule="a mixture takes sigma as one residual SD per component, sigma=[s1, ..., sK]; shape and phi do not apply",
    scale_missing_error=None,
    linpred_symbol="mu", scale_phrase="response", intercept_meaning="expected response of component k",
    mean_lines=_mean_lines,
    likelihood_lines=_likelihood_lines,
    centred_intercept_note=(),
    deterministic_note="mu = the components' linear predictors (rows x K); the mixture's mean is computed by predict()",
    default_prior_link=None,                                        # per component: the component family's own defaults
    default_scale_prior=None,
    bambi_setup=_bambi_setup,
    check_formula=_check_formula,
    mixture=True,
)
