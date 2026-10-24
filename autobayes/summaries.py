"""brms's summaries of a fit's draws (brms-parity F07, docs/summaries.md): ``fixef()``, ``ranef()``, ``coef()``, ``VarCorr()``,
``posterior_summary()``, the ``summary()`` tables and ``residuals(type = "ordinary", method = "posterior_epred")``, each a
pandas DataFrame (or a mapping of them) named, ordered and labelled as brms 2.23 labels its own. Accepted against brms on the
frozen oracle cases ``gauss_varint_summaries``, ``bernoulli_slope_correlated_summaries`` and
``gauss_distributional_sigma_summaries`` (four chains of fifty seeded draws each); brms's names and order for further model
structures (two grouping factors, independent slopes, three and four correlated coefficients, categorical predictors and
interactions, a formula of sigma with several terms or none, no group-level terms) were recorded from brms
(``tests/summary_probes/brms-2.23.0-names.json``).

The draws are first named as brms names them (``brms_draws``), in brms's order: ``b_Intercept`` (the raw-scale intercept) and
each distributional formula's ``b_<dpar>_Intercept``, then the coefficients ``b_<name>`` and ``b_<dpar>_<name>``; every
group-level SD ``sd_<group>__<coef>``, factor by factor; every correlation ``cor_<group>__<coef>__<coef>``, the upper triangle
column by column; ``sigma``; the intercepts of the centred designs, ``Intercept`` and ``Intercept_<dpar>``; the group-level
effects ``r_<group>[<level>,<coef>]``, coefficient by coefficient; the log prior density ``lprior`` and the log density
``lp__`` of each draw, both from the model's own graph at the draw (``log_densities``). Every summary is the mean, the SD
(divisor ``S - 1``) and the 2.5 % and 97.5 % quantiles (R's type 7) over all the draws, the chains together (``summarise``);
``summary()`` adds ``Rhat``, ``Bulk_ESS`` and ``Tail_ESS`` (``autobayes.convergence``, posterior 1.6.1's).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd

from autobayes import convergence

STATS = ("Estimate", "Est.Error", "Q2.5", "Q97.5")                        # posterior_summary(), fixef(), ranef(), ...
SUMMARY_COLUMNS = ("Estimate", "Est.Error", "l-95% CI", "u-95% CI", "Rhat", "Bulk_ESS", "Tail_ESS")   # summary()'s tables
POP_KINDS = ("numeric", "categorical", "interaction", "transform")      # population-level terms with brms names of class b
# the parts of a parsed formula the summaries take; every other field of ModelSpec (and of GroupTerm) must be empty, so a
# construct that adds one is refused by name until its brms names are verified here
SPEC_FIELDS = frozenset({"formula", "response", "intercept", "levels_by_group", "trials", "rhs", "categoricals", "pop_terms",
                         "group_terms", "dpars"})
GROUP_FIELDS = frozenset({"factor", "coefs", "correlated"})


def summarise(draws) -> np.ndarray:
    """brms's ``posterior_summary()`` of draws on the first axis: ``(..., 4)``, the mean, the SD (divisor ``S - 1``) and the 2.5 %
    and 97.5 % quantiles (R's type 7)."""
    draws = np.asarray(draws, dtype=float)
    return np.stack([draws.mean(axis=0), draws.std(axis=0, ddof=1), *np.quantile(draws, [0.025, 0.975], axis=0)], axis=-1)


def _refuse(what: str, construct: str):
    raise ValueError(f"{what} of {construct} is not offered: no frozen brms case or probe verifies brms's names for it "
                     "(docs/summaries.md, divergence D162)")


def refuse_unsupported(model, what: str) -> None:
    """``what`` (one of the summaries) takes the model structures whose brms names are verified (module docstring): a
    univariate, linear Gaussian or Bernoulli model with an intercept and centred predictors, population-level terms with
    design columns (numeric, categorical, interactions, transforms), group-level terms ``(1 | g)``, ``(1 + x | g)`` and ``(1 + x
    || g)`` of grouping factors read from one column, and a formula of the Gaussian's ``sigma`` with an intercept and
    population-level terms. Everything else is refused naming the construct (divergence D162)."""
    if getattr(model, "responses", None):
        _refuse(what, "a multivariate model")
    spec = model.spec
    if getattr(spec, "is_nonlinear", False):
        _refuse(what, "a non-linear model (nl=True)")
    if not model.family_spec.summaries:                       # FamilySpec.summaries: the families a frozen case verifies
        from autobayes.families import FAMILIES

        offering = [name for name, fam in FAMILIES.items() if fam.summaries]
        _refuse(what, f"the {model.family} family (the summaries take the {' and '.join(offering)} families)")
    if not spec.intercept:
        _refuse(what, "a formula without an intercept")
    if not model.center_predictors:
        _refuse(what, "a model with center_predictors=False")
    if getattr(model, "shrinkage_names", None):
        _refuse(what, "a shrinkage prior on b")
    for s in (spec, *spec.dpars):
        for f in dataclasses.fields(s):
            value = getattr(s, f.name)
            if f.name not in SPEC_FIELDS and value is not None and value != () and value != {}:
                _refuse(what, f"a formula with {f.name} ({value!r})")
        for t in s.pop_terms:
            if t.kind not in POP_KINDS:
                _refuse(what, f"the {t.kind} term {t.name!r}")
    for d in spec.dpars:                                      # the Gaussian's sigma: its only distributional parameter
        if not d.intercept or d.group_terms:
            _refuse(what, f"the distributional formula {d.formula!r} (the summaries take a formula of sigma with an intercept and "
                          "population-level terms)")
    for t in spec.group_terms:                                 # (a factor is in one term: the formula parser refuses another)
        extra = [f.name for f in dataclasses.fields(t) if f.name not in GROUP_FIELDS and getattr(t, f.name)]
        if extra or t.derived:
            _refuse(what, f"the group-level term {t.canonical} ({', '.join(extra) or 'a grouping factor of two columns'})")


# -- brms's names of the draws ----------------------------------------------------------------------------------------
def _values(posterior, name: str, j=None) -> np.ndarray:
    arr = posterior[name].transpose("chain", "draw", ...).to_numpy()
    return arr if j is None else arr[..., j]


def _population(model, posterior):
    """The response formula's raw-scale intercept and each distributional formula's (``{"Intercept": ..., "sigma_Intercept":
    ...}``), the coefficients by brms name (``{"x": ..., "zq": ..., "sigma_x": ...}``), each ``(chain, draw)``, and each formula's
    coefficients in its design-column order, ``(chain, draw, columns)`` or None (keyed ``""`` for the response formula)."""
    from autobayes.default_priors import brms_coefficient_names

    spec = model.spec
    names = brms_coefficient_names(spec)
    intercepts = {"Intercept": _values(posterior, model.names["intercept"])}
    slopes = {names[column]: _values(posterior, variable, j) for column, variable, j in model.coefficients}
    in_order = {"": np.stack(list(slopes.values()), axis=-1) if slopes else None}
    for d in spec.dpars:
        roles, dnames = model.dpar_names[d.response], brms_coefficient_names(d)
        intercepts[f"{d.response}_Intercept"] = _values(posterior, roles["intercept"])
        own = {f"{d.response}_{dnames[column]}": _values(posterior, roles[f"b:{term}"], j) for column, term, j in d.column_terms}
        slopes.update(own)
        in_order[d.response] = np.stack(list(own.values()), axis=-1) if own else None
    return intercepts, slopes, in_order


def _centred(intercept, b, x_means):
    """The intercept of the centred design (brms's ``Intercept``): the raw-scale intercept plus ``mean(X) b``."""
    return intercept + (b @ x_means if b is not None else 0.0)


def _group_effects(model, posterior):
    """``{factor: {coef: (sd (chain, draw), r (chain, draw, levels))}}`` by brms name, in formula order, and each factor's levels."""
    from autobayes.default_priors import brms_rename

    out, levels = {}, {}
    for (g, coef_name), roles in model.group_coefficient_names.items():
        out.setdefault(g, {})[brms_rename(coef_name)] = (_values(posterior, roles["sd"]), _values(posterior, roles["group_effect"]))
        levels[g] = [str(v) for v in model.design.group_blocks[(g, coef_name)].levels]
    return out, levels


def _correlations(model, posterior):
    """``{factor: (coefs, cor (chain, draw, M, M), L (chain, draw, M, M))}`` of each correlated term."""
    from autobayes.default_priors import brms_rename

    return {g: ([brms_rename(c) for c in model.correlation_coefs(g)], _values(posterior, roles["cor"]), _values(posterior, roles["L"]))
            for g, roles in model.correlation_names.items()}


def _pairs(M: int) -> list[tuple[int, int]]:
    """brms's order of the correlations of ``M`` coefficients: the upper triangle column by column ((1, 2), (1, 3), (2, 3),
    (1, 4), ...)."""
    return [(a, b) for b in range(M) for a in range(b)]


def log_densities(model, posterior) -> tuple[np.ndarray, np.ndarray]:
    """``(lprior, lp__)``, each ``(chain, draw)``, of the model's own graph at each draw of ``posterior``: brms's ``lprior``, the
    log prior density (the priors of the centred design's intercept, the coefficients, the group-level SDs and correlations,
    and sigma or its formula's intercept and coefficients; not the standardised group-level effects), and brms's ``lp__``, the
    log density with the Jacobian, what a sampler records. Each draw is mapped back to the sampled coordinates: the centred
    intercepts (``_centred``), the standardised group-level effects ``z = L^-1 (r / sd)`` (``r / sd`` without a correlation),
    then ``Model.value_point``."""
    spec, design = model.spec, model.design
    intercepts, _, in_order = _population(model, posterior)
    groups = _group_effects(model, posterior)[0]
    cors = _correlations(model, posterior)
    offsets = {roles["offset"] for roles in model.group_coefficient_names.values()}
    pmm = model.pymc_model
    lp_fn = pmm.compile_logp()
    lprior_fn = pmm.compile_logp(vars=[rv for rv in pmm.free_RVs if rv.name not in offsets], jacobian=False)
    shape = intercepts["Intercept"].shape
    lprior, lp = np.empty(shape), np.empty(shape)
    for idx in np.ndindex(*shape):
        b = in_order[""][idx] if in_order[""] is not None else np.zeros(0)
        natural = {"intercept": float(_centred(intercepts["Intercept"][idx], b, design.x_means)), "b": b}
        if spec.group_terms:
            natural.update(sd={}, z={})
            for t in spec.group_terms:
                coefs = groups[t.factor]
                sds = np.array([sd[idx] for sd, _ in coefs.values()])
                z = np.stack([r[idx] for _, r in coefs.values()]) / sds[:, None]         # (coefs, levels)
                if t.factor in cors:
                    natural.setdefault("L", {})[t.factor] = cors[t.factor][2][idx]
                    z = np.linalg.solve(cors[t.factor][2][idx], z)
                natural["sd"][t.factor] = dict(zip(t.coefs, sds))
                natural["z"][t.factor] = dict(zip(t.coefs, z))
        for d in spec.dpars:                                     # the Gaussian's sigma (refuse_unsupported)
            bd = in_order[d.response][idx] if in_order[d.response] is not None else np.zeros(0)
            natural[d.response] = {"b": bd, "intercept": float(_centred(intercepts[f"{d.response}_Intercept"][idx], bd,
                                                                        design.dpars[d.response].x_means))}
        if "sigma" in model.names and not spec.dpars:
            natural["sigma"] = float(_values(posterior, model.names["sigma"])[idx])
        point = model.value_point(**natural)
        lprior[idx], lp[idx] = float(lprior_fn(point)), float(lp_fn(point))
    return lprior, lp


def _named_draws(model, posterior, *, densities: bool = True) -> dict[str, np.ndarray]:
    spec, design = model.spec, model.design
    intercepts, slopes, in_order = _population(model, posterior)
    groups, levels = _group_effects(model, posterior)
    cors = _correlations(model, posterior)
    out = {f"b_{name}": v for name, v in {**intercepts, **slopes}.items()}
    for g, coefs in groups.items():
        out.update({f"sd_{g}__{c}": sd for c, (sd, _) in coefs.items()})
    for g, (coefs, cor, _) in cors.items():
        out.update({f"cor_{g}__{coefs[a]}__{coefs[b]}": cor[..., a, b] for a, b in _pairs(len(coefs))})
    if "sigma" in model.names and not spec.dpars:
        out["sigma"] = _values(posterior, model.names["sigma"])
    out["Intercept"] = _centred(intercepts["Intercept"], in_order[""], design.x_means)
    for d in spec.dpars:
        out[f"Intercept_{d.response}"] = _centred(intercepts[f"{d.response}_Intercept"], in_order[d.response],
                                                  design.dpars[d.response].x_means)
    for g, coefs in groups.items():
        for c, (_, r) in coefs.items():
            out.update({f"r_{g}[{level},{c}]": r[..., k] for k, level in enumerate(levels[g])})
    if densities:
        out["lprior"], out["lp__"] = log_densities(model, posterior)
    return out


def _posterior(model, fit_result, what: str):
    from autobayes.identity import check_fit_binding

    refuse_unsupported(model, what)
    check_fit_binding(model, fit_result, allow_unverified=True)
    return getattr(fit_result, "idata", fit_result).posterior


def brms_draws(model, fit_result) -> dict[str, np.ndarray]:
    """The draws of ``fit_result`` (a ``FitResult`` or its ``InferenceData``, checked against the model as ``predict()``
    checks it) by brms's variable names and in brms's order (module docstring), each ``(chain, draw)``: the variables of brms's
    ``as_draws_array()`` of the same fit."""
    return _named_draws(model, _posterior(model, fit_result, "brms_draws()"))


# -- the summaries ----------------------------------------------------------------------------------------------------
def _stack(arrays) -> np.ndarray:
    """``(S, k)``: each ``(chain, draw)`` array flattened chain by chain, side by side."""
    return np.stack([np.asarray(a, dtype=float).reshape(-1) for a in arrays], axis=1)


def _table(rows, arrays) -> pd.DataFrame:
    return pd.DataFrame(summarise(_stack(arrays)), index=pd.Index(list(rows)), columns=list(STATS))


def _three_way(index, coefs, arrays_by_coef) -> pd.DataFrame:
    """brms's ``[index, statistic, coef]`` array as a DataFrame: one row per entry of ``index``, the columns ``(coef,
    statistic)``; ``arrays_by_coef[c]`` holds one ``(chain, draw)`` array per row."""
    values = np.concatenate([summarise(_stack(arrays_by_coef[c])) for c in coefs], axis=1)
    return pd.DataFrame(values, index=pd.Index(list(index)), columns=pd.MultiIndex.from_product([list(coefs), list(STATS)]))


def posterior_summary(model, fit_result) -> pd.DataFrame:
    """brms's ``posterior_summary()``: one row per variable of ``brms_draws`` (brms's names and order, ``lprior`` and ``lp__``
    last), the columns ``Estimate``, ``Est.Error``, ``Q2.5``, ``Q97.5``."""
    draws = _named_draws(model, _posterior(model, fit_result, "posterior_summary()"))
    return _table(draws, draws.values())


def fixef(model, fit_result) -> pd.DataFrame:
    """brms's ``fixef()``: the population-level coefficients, one row per brms name (``Intercept``, each distributional
    formula's ``sigma_Intercept``, then the coefficients ``x``, ``zq``, ``x:zq``, ..., ``sigma_x``), the columns ``Estimate``,
    ``Est.Error``, ``Q2.5``, ``Q97.5``."""
    intercepts, slopes, _ = _population(model, _posterior(model, fit_result, "fixef()"))
    coefs = {**intercepts, **slopes}
    return _table(coefs, coefs.values())


def ranef(model, fit_result) -> dict[str, pd.DataFrame]:
    """brms's ``ranef()``: for each grouping factor (in formula order), its levels' group-level effects, one row per level and
    the columns ``(coef, statistic)``: brms's ``[level, statistic, coef]`` array, so that ``ranef(model, fit)["g"]["x"]`` is brms's
    ``ranef(fit)$g[, , "x"]``. A model without group-level terms is refused, as brms refuses it."""
    posterior = _posterior(model, fit_result, "ranef()")
    if not model.spec.group_terms:
        raise ValueError("ranef(): the model has no group-level terms (brms: \"The model does not contain group-level effects.\")")
    groups, levels = _group_effects(model, posterior)
    return {g: _three_way(levels[g], list(coefs), {c: [r[..., k] for k in range(len(levels[g]))] for c, (_, r) in coefs.items()})
            for g, coefs in groups.items()}


def coef(model, fit_result) -> dict[str, pd.DataFrame]:
    """brms's ``coef()``: for each grouping factor, each level's coefficients, laid out as ``ranef()``'s: the population-level
    coefficient plus the level's group-level effect for the factor's coefficients, first and in the factor's order, then the
    population-level coefficient alone for every other row of ``fixef()``, in its order (brms's order; probed). A model without
    group-level terms is refused, as brms refuses it."""
    posterior = _posterior(model, fit_result, "coef()")
    if not model.spec.group_terms:
        raise ValueError("coef(): the model has no group-level terms (brms: \"No group-level effects detected. Call method 'fixef' "
                         "to access population-level effects.\")")
    intercepts, slopes, _ = _population(model, posterior)
    population = {**intercepts, **slopes}
    groups, levels = _group_effects(model, posterior)
    out = {}
    for g, coefs in groups.items():
        order = [*coefs, *(c for c in population if c not in coefs)]
        per = {c: [population[c] + (coefs[c][1][..., k] if c in coefs else 0.0) for k in range(len(levels[g]))] for c in order}
        out[g] = _three_way(levels[g], order, per)
    return out


def VarCorr(model, fit_result) -> dict[str, dict[str, pd.DataFrame]]:   # noqa: N802 - brms's name
    """brms's ``VarCorr()``: for each grouping factor (in formula order), ``{"sd": DataFrame}`` with one row per coefficient and,
    for a correlated term, ``"cor"`` and ``"cov"`` (``sd_a sd_b cor_ab``), each laid out as ``ranef()``'s with the coefficients as
    the rows and as the first level of the columns (1 and ``sd^2`` on the diagonal); then ``"residual__"``, ``{"sd": ...}`` of
    sigma with the row name ``""``, for a Gaussian model without a formula of sigma. A model with neither is refused, as brms
    refuses it."""
    posterior = _posterior(model, fit_result, "VarCorr()")
    groups = _group_effects(model, posterior)[0]
    cors = _correlations(model, posterior)
    out = {}
    for g, coefs in groups.items():
        names, sds = list(coefs), [sd for sd, _ in coefs.values()]
        out[g] = {"sd": _table(names, sds)}
        if g in cors:
            cor = cors[g][1]
            out[g]["cor"] = _three_way(names, names, {c: [cor[..., a, k] for a in range(len(names))] for k, c in enumerate(names)})
            out[g]["cov"] = _three_way(names, names, {c: [sds[a] * sds[k] * cor[..., a, k] for a in range(len(names))]
                                                      for k, c in enumerate(names)})
    if "sigma" in model.names and not model.spec.dpars:
        out["residual__"] = {"sd": _table([""], [_values(posterior, model.names["sigma"])])}
    if not out:
        raise ValueError("VarCorr(): the model has no group-level terms and no residual SD (brms: \"The model does not contain "
                         "covariance matrices.\")")
    return out


def _summary_table(rows: dict) -> pd.DataFrame:
    """One of ``summary()``'s tables: ``rows`` maps each row name to its ``(chain, draw)`` draws."""
    arrays = list(rows.values())
    diagnostics = [[convergence.rhat(a), convergence.ess_bulk(a), convergence.ess_tail(a)] for a in arrays]
    values = np.concatenate([summarise(_stack(arrays)), np.array(diagnostics)], axis=1) if arrays else np.zeros((0, 7))
    return pd.DataFrame(values, index=pd.Index(list(rows)), columns=list(SUMMARY_COLUMNS))


@dataclass(frozen=True)
class Summary:
    """brms's ``summary()`` of a fit: the tables ``fixed`` (the population-level coefficients, ``fixef()``'s rows),
    ``spec_pars`` (the family's parameters without a formula of their own: the Gaussian's ``sigma``), ``cor_pars``
    (autocorrelation parameters: none in the models taken) and ``random`` (by grouping factor in formula order: ``sd(<coef>)``,
    then ``cor(<coef>,<coef>)`` in brms's order), each with the columns ``Estimate``, ``Est.Error``, ``l-95% CI``, ``u-95% CI``
    (the 2.5 % and 97.5 % quantiles), ``Rhat``, ``Bulk_ESS``, ``Tail_ESS``; ``nobs``; ``ngrps``, the levels of each grouping
    factor; ``chains``, ``iter`` and ``warmup`` (per chain), ``thin`` and ``total_ndraws`` (the post-warmup draws); and
    ``formula``, ``family`` and ``links``, which printing shows."""

    fixed: pd.DataFrame
    spec_pars: pd.DataFrame
    cor_pars: pd.DataFrame
    random: dict
    nobs: int
    ngrps: dict
    chains: int
    iter: int
    warmup: int
    thin: int
    total_ndraws: int
    formula: tuple
    family: str
    links: dict

    def __str__(self) -> str:
        def block(title, frame):
            return [title, frame.to_string(float_format=lambda v: f"{v:.2f}"), ""] if len(frame) else []

        lines = [f" Family: {self.family}", "  Links: " + "; ".join(f"{k} = {v}" for k, v in self.links.items()),
                 "Formula: " + "\n         ".join(self.formula), f"   Data: {self.nobs} observations",
                 f"  Draws: {self.chains} chains, each with iter = {self.iter}; warmup = {self.warmup}; thin = {self.thin};",
                 f"         total post-warmup draws = {self.total_ndraws}", ""]
        if self.random:
            lines.append("Multilevel Hyperparameters:")
            for g, frame in self.random.items():
                lines += block(f"~{g} (Number of levels: {self.ngrps[g]})", frame)
        lines += block("Regression Coefficients:", self.fixed)
        lines += block("Further Distributional Parameters:", self.spec_pars)
        return "\n".join(lines).rstrip() + "\n"


def _draw_counts(fit_result, posterior) -> tuple[int, int, int]:
    """``(chains, iter, warmup)``, ``iter`` and ``warmup`` per chain as brms counts them: a ``FitResult``'s tuning steps are its
    warmup, and an ``InferenceData`` from PyMC records them as its posterior's ``tuning_steps``; draws with neither had none (as
    a fixed-parameter run has none)."""
    chains, draws = int(posterior.sizes["chain"]), int(posterior.sizes["draw"])
    warmup = getattr(fit_result, "tune", None)
    warmup = int(posterior.attrs.get("tuning_steps", 0) if warmup is None else warmup)
    return chains, draws + warmup, warmup


def summary(model, fit_result) -> Summary:
    """brms's ``summary()`` of the fit (``Summary``): its tables of the draws (``brms_draws``' names), with posterior 1.6.1's
    ``Rhat``, ``Bulk_ESS`` and ``Tail_ESS`` (``autobayes.convergence``) of each row."""
    posterior = _posterior(model, fit_result, "summary()")
    draws = _named_draws(model, posterior, densities=False)
    intercepts, slopes, _ = _population(model, posterior)
    random = {}
    for g, coefs in _group_effects(model, posterior)[0].items():
        names = list(coefs)
        rows = {f"sd({c})": draws[f"sd_{g}__{c}"] for c in names}
        if g in model.correlation_names:
            rows.update({f"cor({names[a]},{names[b]})": draws[f"cor_{g}__{names[a]}__{names[b]}"] for a, b in _pairs(len(names))})
        random[g] = _summary_table(rows)
    chains, iters, warmup = _draw_counts(fit_result, posterior)
    links = {"mu": model.link, **{d.response: model.design.dpars[d.response].link for d in model.spec.dpars}}
    if "sigma" in draws:
        links["sigma"] = "identity"
    return Summary(fixed=_summary_table({n: draws[f"b_{n}"] for n in {**intercepts, **slopes}}),
                   spec_pars=_summary_table({"sigma": draws["sigma"]} if "sigma" in draws else {}), cor_pars=_summary_table({}),
                   random=random, nobs=int(model.design.n),
                   ngrps={t.factor: len(model.spec.levels_by_group[t.factor]) for t in model.spec.group_terms}, chains=chains,
                   iter=iters, warmup=warmup, thin=1, total_ndraws=chains * (iters - warmup),
                   formula=(model.spec.formula, *(d.formula for d in model.spec.dpars)), family=model.family, links=links)


def residuals(model, fit_result, *, type: str = "ordinary", method: str = "posterior_epred") -> pd.DataFrame:   # noqa: A002
    """brms's ``residuals(type = "ordinary", method = "posterior_epred")``: for each fitted row, in data order and without row
    names (as brms's), the summary (``Estimate``, ``Est.Error``, ``Q2.5``, ``Q97.5``) over the draws of ``y -
    posterior_epred`` (``predict(kind="mean")``). brms's ``"pearson"`` type (deprecated) and its default method
    ``"posterior_predict"`` (random draws) are refused (divergence D163)."""
    from autobayes.families import family_of
    from autobayes.predict import predict

    _posterior(model, fit_result, "residuals()")
    if type != "ordinary" or method != "posterior_epred":
        raise ValueError(f"residuals(type={type!r}, method={method!r}) is not offered: only type='ordinary' with "
                         "method='posterior_epred' (brms's default method draws from posterior_predict, and its 'pearson' type is "
                         "deprecated; docs/summaries.md, divergence D163)")
    mean = predict(model, fit_result, kind="mean")[family_of(model).mean_name].transpose("chain", "draw", "__obs__").to_numpy()
    return pd.DataFrame(summarise(np.asarray(model.design.y, dtype=float) - mean.reshape(-1, mean.shape[-1])), columns=list(STATS))
