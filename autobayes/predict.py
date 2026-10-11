"""Posterior draws of the mean and posterior predictive draws, with named semantics.

``kind="mean"`` returns draws of the expected response: ``mu`` for the Gaussian family,
``p = P(y = 1)`` for the Bernoulli family, for the binomial family ``p`` together with
``expected = n p`` (brms ``posterior_epred``, which is ``mu * trials`` there), and for the
lognormal family ``mean = exp(mu + sigma^2 / 2)`` (brms ``posterior_epred``; ``mu`` is the
log-scale location, which ``kind="linear"`` returns).
``kind="response"`` also returns draws of a new observation ``y``: ``Normal(mu, sigma)``,
``Bernoulli(p)`` or ``Binomial(n, p)``, so ``y`` is an integer for the discrete families and
carries the outcome's own variability (brms ``posterior_predict``). ``kind="probability"``
(binomial only) returns ``p`` alone and needs no trials (brms ``posterior_linpred(transform =
TRUE)``). Binomial predictions are computed in this layer from the posterior; for
``newdata``, ``mean`` and ``response`` need the trials column named in the formula, and a row
with zero trials gives ``expected = 0`` and ``y = 0``.

A model with response addition terms (B02, docs/addition-terms.md) is predicted in this layer: ``weights()`` and
``cens()`` change nothing here (brms's ``posterior_epred`` and ``posterior_predict`` ignore them, and new rows need
neither column), while under ``trunc()`` the mean is the truncated distribution's mean and ``y`` is drawn from the
truncated distribution, as brms does.

A model with a monotonic term ``mo(x)`` (B03, docs/monotonic.md) is predicted in this layer: each term adds its coefficient
times ``D * sum(simplex[:Xmo_i])`` per draw and row, with every row's value coded as the fitted rows were (a value outside
the fitted coding is refused, as brms refuses it).

A model whose auxiliary parameter has its own formula (B01, docs/distributional.md: ``["y ~ x", "sigma ~ z"]``) is
predicted in this layer too: ``kind="linear"`` and ``kind="mean"`` keep their meaning (the linear predictor of ``mu`` and
the expected response, brms's ``posterior_linpred`` and ``posterior_epred``), with each row's own ``sigma`` wherever the
family needs it (the lognormal's mean, every ``kind="response"`` draw); ``dpar="sigma"`` returns the parameter itself,
its linear predictor under ``kind="linear"`` (brms ``posterior_linpred(dpar = "sigma")``, named ``eta``) or its values
under ``kind="mean"`` (brms ``posterior_epred(dpar = "sigma")``, named ``sigma``). Its intercept draws are on the raw scale,
as Bambi reports them after sampling.

A finite mixture (C05, docs/mixtures.md) is predicted in this layer, population-level only: ``kind="linear"`` returns every
component's linear predictor as ``eta`` with a ``component`` dimension (brms's ``posterior_linpred``), ``kind="mean"`` the
mixture's expected response ``sum_k theta_k mu_k`` as ``mean`` (brms's ``posterior_epred``) and ``kind="response"`` draws, per
draw and row, a component by theta and then its Normal (brms's ``posterior_predict``), seeded.

``new_groups`` decides what happens when ``newdata`` contains a level of the grouping
factor that was not in the training data (a population-only model has no grouping factor: it
needs no such column, its result carries ``new_groups = "not_applicable"``, and any value other
than the default ``"error"`` is refused rather than ignored):

- ``"error"`` (default): refuse. The safe default.
- ``"resample_observed"``: brms's default ``sample_new_levels="uncertainty"``: for each
  posterior draw and each distinct unseen label, reuse the fitted effect of a uniformly
  chosen observed level (chosen independently per label and per draw; rows sharing a
  label share the choice). With varying slopes, the whole effect vector of the chosen
  level is reused: its intercept and slope effects come from the same level, as brms does.
  This understates between-level variance because it resamples
  shrunk draws from a finite set of levels. Computed here rather than through Bambi's
  ``sample_new_groups=True``, which gives every unseen label the same effect in a draw.
- ``"population"``: brms's ``sample_new_levels="gaussian"``: for each posterior draw,
  draw a fresh effect from ``Normal(0, sd_group)`` (on the linear-predictor scale, i.e. the
  logit scale for Bernoulli), independently for each coefficient of the factor (intercept
  and every varying slope, each from its own ``Normal(0, sd)``); for a correlated term
  ``(1 + x | g)`` the level's whole effect vector comes from the implied multivariate normal,
  ``diag(sd) L z`` with ``z`` standard normal and ``L`` the draw's Cholesky factor. This is the population
  distribution the model actually fit and is implemented here because Bambi does not offer it.
"""
from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np
import pandas as pd
import xarray as xr

from autobayes.families import family_of
from autobayes.families.spec import DISTRIBUTIONAL_INVERSE_LINKS
from autobayes.formula import FormulaError, categorical_labels, check_counts, group_labels, pop_term_design, reject_complex

_NEW_GROUPS = {"error", "resample_observed", "population"}


def raw_design(model, df: pd.DataFrame) -> np.ndarray:
    """The (n, P) raw design of ``df`` in ``model.coefficients`` order: each population-level term's own
    block (``pop_term_design``) side by side: numeric columns as given, one 0/1 indicator per coefficient
    level of a categorical term (reference rows all zero), the numeric column times each indicator for an
    ``x:z`` interaction, the product column of ``x:w``, the indicator product of each cell of ``z:h``."""
    blocks = [pop_term_design(term, df, model.spec.categoricals) for term in model.spec.pop_terms]
    return np.concatenate(blocks, axis=1) if blocks else np.zeros((len(df), 0))


def expected_response(fam, model, post, eta):
    """``E[y]`` from the linear predictor: the family's inverse link, or its own function when the mean also depends
    on an auxiliary parameter (the lognormal's ``exp(mu + sigma^2 / 2)``)."""
    return fam.inverse_link(eta) if fam.expected_response is None else fam.expected_response(model, post, eta)


def _coefficient_draws(post, model, axis: str | None = None):
    """(chain, draw, P) posterior draws of the coefficients in design-column order; (chain, draw, P, K - 1) for the
    categorical family (B05), whose every coefficient has the non-reference categories as its last axis, and (chain, draw,
    P, K) for a mixture (C05), whose coefficients have the component dimension ``axis`` last."""
    category = axis or model.names.get("category_dim")
    parts = []
    for term in model.spec.bambi_terms:                         # a monotonic term has no design column (B03)
        if category is None:
            arr = post[term.name].to_numpy()
            parts.append(arr if term.levels else arr[..., None])
        else:
            arr = post[term.name].transpose("chain", "draw", ..., category).to_numpy()
            parts.append(arr if term.levels else arr[..., None, :])
    return np.concatenate(parts, axis=-1 if category is None else -2) if parts else None


def monotonic_contribution(model, post, df: pd.DataFrame) -> np.ndarray | float:
    """(chain, draw, row) draws of the monotonic terms' part of the linear predictor (B03) on the rows of ``df`` (the fitted
    frame, or checked new rows): each term's coefficient times ``mo()`` of its simplex draws at the rows' codes, which
    ``formula.Monotonic.codes`` reads with the fitted coding and refuses outside it. 0 without a monotonic term."""
    from autobayes.monotonic import monotonic_values

    out = 0.0
    for column, roles in model.mo_names.items():
        codes = model.spec.monotonic(column).codes(df[column], where="newdata column")
        simplex = post[roles["simplex"]].transpose("chain", "draw", roles["coordinate"]).to_numpy()
        out = out + post[roles["b"]].to_numpy()[..., None] * monotonic_values(simplex, codes)
    return out


def dpar_linear_predictor(model, post, df: pd.DataFrame, name: str) -> np.ndarray:
    """(chain, draw, row) draws of distributional parameter ``name``'s linear predictor (B01) on the rows of ``df`` (the
    fitted frame, or checked new rows): its raw-scale intercept draws plus its raw design times its coefficient draws, as
    the response formula's is computed here."""
    ds, roles = model.spec.dpar(name), model.dpar_names[name]
    shape = (post.sizes["chain"], post.sizes["draw"])
    a = post[roles["intercept"]].to_numpy() if ds.intercept else np.zeros(shape)
    eta = np.repeat(a[..., None], len(df), axis=-1)
    if ds.pop_terms:
        X = np.concatenate([pop_term_design(t, df, ds.categoricals) for t in ds.pop_terms], axis=1)
        b = np.concatenate([post[roles[f"b:{t.name}"]].to_numpy() if t.levels else post[roles[f"b:{t.name}"]].to_numpy()[..., None]
                            for t in ds.pop_terms], axis=-1)
        eta = eta + np.einsum("cdp,np->cdn", b, X)
    return eta


def dpar_values(model, post, df: pd.DataFrame, name: str) -> np.ndarray:
    """(chain, draw, row) draws of distributional parameter ``name`` itself (B01): the inverse link of its linear
    predictor, refused where it is not a positive finite number."""
    return DISTRIBUTIONAL_INVERSE_LINKS[model.design.dpars[name].link](dpar_linear_predictor(model, post, df, name), name)


def _with_dpars(model, post, df: pd.DataFrame):
    """The posterior with each distributional parameter's per-row draws under its PyMC name (B01), the form in which the
    families' mean and response functions read an auxiliary parameter (``families.spec.auxiliary``)."""
    return post.assign({model.dpar_names[name]["value"]: (("chain", "draw", "__row__"), dpar_values(model, post, df, name))
                        for name in model.spec.dpar_names})


def _check_newdata(model, newdata: pd.DataFrame, *, need_trials: bool = False, include_group_effects: bool = True) -> pd.DataFrame:
    s = model.spec
    operands = [c for t in s.group_terms for c in t.operands] if include_group_effects else []   # g and h, never g_h
    cols = [*s.model_columns] + ([s.offset.column] if s.offset is not None else []) + operands   # every formula's columns (B01)
    cols = list(dict.fromkeys(cols))
    if need_trials:
        if s.trials not in newdata.columns:
            raise FormulaError(
                f"newdata lacks the trials column {s.trials!r}, which kind='mean' and kind='response' need "
                "for a binomial model (kind='probability' does not)."
            )
        cols.append(s.trials)
    missing = [c for c in cols if c not in newdata.columns]
    if missing:
        hint = (f" (the offset term {s.offset.term!r} needs {s.offset.column!r} on every row to predict)"
                if s.offset is not None and s.offset.column in missing else "")
        raise FormulaError(f"newdata lacks columns {missing}{hint}")
    df = newdata[cols].copy()
    if df.isna().any().any():
        raise FormulaError("newdata contains missing values")
    categorical_names = {c.name for c in s.all_categoricals}
    monotonic_columns = {m.column for m in s.monotonics}
    for c in s.model_columns:                                    # numeric predictors and the raw columns of transforms
        if c in categorical_names or (c in monotonic_columns and c not in s.predictors):
            continue                                             # a monotonic predictor's values are checked by its coding (B03)
        reject_complex(f"newdata column {c!r}", df[c])
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise FormulaError(f"newdata column {c!r} must be numeric")
    if s.offset is not None:
        s.offset.values(df, where="newdata column")               # numeric, finite and (for log) positive, or refused
    for t in (*s.transform_terms, *(t for d in s.dpars for t in d.transform_terms)):   # the evaluated columns, recomputed from the raw ones
        df[t.name] = t.transform.values(df, where="newdata column")
    for m in s.monotonics:                                       # whole numbers in the fitted range, or a fitted category (B03)
        m.codes(df[m.column], where="newdata column")
    for cat in s.all_categoricals:
        labels, _ = categorical_labels(f"newdata column {cat.name!r}", df[cat.name])
        unseen = sorted(set(labels.tolist()) - set(cat.levels))
        if unseen:
            raise FormulaError(
                f"newdata has levels of the categorical predictor {cat.name!r} that were not in the fitted rows: "
                f"{unseen}; fitted levels {list(cat.levels)} (reference {cat.reference!r}). A categorical "
                "predictor has a coefficient only for its fitted levels; it is not a grouping factor, so "
                "new_groups does not apply to it."
            )
        df[cat.name] = pd.Categorical(labels, categories=list(cat.categories), ordered=True)
    if need_trials:
        df[s.trials] = check_counts(f"newdata trials {s.trials!r}", df[s.trials])
    for t in (s.group_terms if include_group_effects else ()):   # labels as text; a derived factor's column is built here
        df[t.column] = group_labels(t.factor, df)
    return df.reset_index(drop=True)


def _predict_dpar(model, idata, df: pd.DataFrame | None, kind: str, name: str) -> xr.Dataset:
    """``predict(..., dpar=name)`` (B01): the distributional parameter's linear predictor (``kind="linear"``, ``eta``) or
    its values (``kind="mean"``, under its brms name), per draw and row; its formula has no group-level terms, so the group
    policies change nothing here and nothing is random (``predict()`` writes the policy attributes)."""
    if df is None:
        df = model.data
    post = idata.posterior
    eta = dpar_linear_predictor(model, post, df, name)
    if kind == "linear":
        if not np.isfinite(eta).all():
            raise ValueError(f"the linear predictor of {name} contains nonfinite draws")
        out = {"eta": eta}
    else:
        out = {name: DISTRIBUTIONAL_INVERSE_LINKS[model.design.dpars[name].link](eta, name)}
    dims = ("chain", "draw", "__obs__")
    ds = xr.Dataset({k: (dims, v) for k, v in out.items()},
                    coords={"chain": post["chain"], "draw": post["draw"], "__obs__": np.arange(len(df))})
    ds.attrs["dpar"] = name
    return ds


def _predict_by_hand(model, idata, df: pd.DataFrame | None, kind: str, seed, policies: dict[str, str], *, include_group_effects=True) -> xr.Dataset:
    """Linear predictor by hand from the posterior, with one effect per distinct unseen label.

    ``df=None`` means the fitted rows (in-sample). The offset term, when the formula has one, is added as
    written (brms keeps it under ``re_formula = NA`` too). Each group-level coefficient adds its effect at the
    row's level times its covariate (one for the varying intercept, the raw slope column for a varying
    slope). ``"population"``: a fresh ``Normal(0, sd)`` draw per unseen label, posterior draw and
    coefficient, which for a correlated term is a draw ``z`` per coefficient turned into the effect vector
    ``diag(sd) L z`` (the same random numbers as the independent case, where ``L`` is the identity).
    ``"resample_observed"``: the effects of one uniformly chosen observed level per label
    and posterior draw, all coefficients from that same level. In both cases every row with the same
    label shares that label's effects.
    """
    if df is None:
        df = model.data
    post = idata.posterior
    n = model.names
    fam = family_of(model)
    rng = np.random.default_rng(seed)
    b = _coefficient_draws(post, model)                         # (chain, draw, P) or None
    X = raw_design(model, df)                                   # raw scale: a is raw-scale intercept
    if fam.per_category:                                        # B05: (chain, draw, n, K - 1), one predictor per non-reference category
        a = (post[n["intercept"]].transpose("chain", "draw", n["category_dim"]).to_numpy() if model.spec.intercept else
             np.zeros((post.sizes["chain"], post.sizes["draw"], len(model.categories) - 1)))
        mu = np.repeat(a[..., None, :], len(df), axis=-2)
        if b is not None:
            mu = mu + np.einsum("cdpk,np->cdnk", b, X)
    else:                                                       # the cumulative thresholds are not part of mu (B05)
        a = (post[n["intercept"]].to_numpy() if model.spec.intercept and not fam.thresholds else
             np.zeros((post.sizes["chain"], post.sizes["draw"])))                     # (chain, draw) raw-scale intercept
        mu = np.repeat(a[..., None], len(df), axis=-1)
        if b is not None:
            mu = mu + np.einsum("cdp,np->cdn", b, X)
    if model.spec.offset is not None:                           # the offset enters as written, with or without group effects
        mu = mu + model.spec.offset.values(df)
    mu = mu + monotonic_contribution(model, post, df)           # the monotonic terms (B03), population-level as well
    new_levels = {}
    for term in (model.spec.group_terms if include_group_effects else ()):
        g = term.factor
        levels = list(model.design.group_designs[g].levels)
        labels = df[term.column]
        new_levels[g] = sorted(set(labels) - set(levels))
        # (sd draws, effect draws, covariate column) per coefficient: ones for the intercept, the raw column for a slope
        coefs = []
        for coef in term.coefs:
            names = model.group_coefficient_names[(g, coef)]
            covariate = np.ones(len(df)) if coef == "Intercept" else df[coef].to_numpy(dtype=float)
            coefs.append((post[names["sd"]].to_numpy(), post[names["group_effect"]].to_numpy(), covariate))
        # a correlated term (A02) drawing unseen levels from the population: the draws' Cholesky factors,
        # (chain, draw, M, M), rows and columns selected by coefficient label in the term's order
        L = None
        if term.has_correlation and policies[g] == "population" and new_levels[g]:
            cn = model.correlation_names[g]
            rows = {cn["coordinate"]: list(term.coefs), cn["coordinate_T"]: list(term.coefs)}
            L = post[cn["L"]].sel(rows).transpose("chain", "draw", cn["coordinate"], cn["coordinate_T"]).to_numpy()
        new_effects = {}
        for lvl in new_levels[g]:
            if policies[g] == "population":              # each coefficient from its own Normal(0, sd); correlated: diag(sd) L z
                zs = [rng.normal(0.0, 1.0, size=sd.shape) for sd, _, _ in coefs]
                if L is None:
                    new_effects[lvl] = [z * sd for z, (sd, _, _) in zip(zs, coefs)]
                else:
                    new_effects[lvl] = [sd * sum(L[..., c, k] * zs[k] for k in range(c + 1)) for c, (sd, _, _) in enumerate(coefs)]
            else:                                        # one observed level per draw; its whole effect vector is reused
                chosen = rng.integers(0, len(levels), size=coefs[0][0].shape)
                new_effects[lvl] = [np.take_along_axis(u, chosen[..., None], axis=-1)[..., 0] for _, u, _ in coefs]
        for i, lvl in enumerate(labels):
            for k, (_, u, covariate) in enumerate(coefs):
                mu[..., i] += (new_effects[lvl][k] if lvl in new_effects else u[..., levels.index(lvl)]) * covariate[i]
    dims = ("chain", "draw", "__obs__")
    coords = {"chain": post["chain"], "draw": post["draw"], "__obs__": np.arange(len(df))}
    if kind == "linear":
        if not np.isfinite(mu).all():
            raise ValueError("linear predictor contains nonfinite draws")
        if fam.per_category:                                     # B05: brms's mu<k> of the non-reference categories
            out = {"eta": (dims + ("category",), mu)}
            coords["category"] = list(model.categories[1:])
        else:
            out = {"eta": (dims, mu)}
    else:                                                        # mu held the linear predictor eta
        trunc = model.spec.addition("trunc")
        if model.spec.dpars and (kind == "response" or fam.expected_response is not None or trunc is not None):
            post = _with_dpars(model, post, df)                  # each row's own sigma, shape or phi (B01)
        if trunc is None:
            mean = expected_response(fam, model, post, mu)
        else:                                                    # brms's posterior_epred of a truncated model: the truncated mean (B02)
            lb, ub = (-np.inf if trunc.lb is None else trunc.lb), (np.inf if trunc.ub is None else trunc.ub)
            mean = fam.tails().truncated_mean(model, post, mu, lb, ub)
        if model.categories:                                     # B05: the probability of every category
            out = {fam.mean_name: (dims + ("category",), mean)}
            coords["category"] = list(model.categories)
        else:
            out = {fam.mean_name: (dims, mean)}
        if fam.has_trials and kind in ("mean", "response"):      # binomial: expected = n p beside p
            out["expected"] = (dims, mean * df[model.spec.trials].to_numpy(dtype=np.int64))
        if kind == "response":                                   # brms's posterior_predict ignores weights and censoring
            out["y"] = (dims, fam.draw_response(model, post, mu, mean, df, rng) if trunc is None
                        else fam.tails().draw_truncated(model, post, mu, lb, ub, rng))   # and draws from the truncated distribution
    ds = xr.Dataset(out, coords=coords)
    if model.categories:                                         # what the 1-based categories of y are (B05)
        ds.attrs["categories_json"] = json.dumps(list(model.categories))
    _prediction_attrs(model, ds, policies, new_levels, in_sample=False, manual=True, include_group_effects=include_group_effects)
    return ds


def _predict_mixture(model, idata, df: pd.DataFrame | None, kind: str, seed) -> xr.Dataset:
    """A mixture's predictions (C05, docs/mixtures.md), population-level only: each component's linear predictor from the
    raw-scale intercept draws (as Bambi reports them after sampling) and the raw design, ``(chain, draw, row, K)``;
    ``kind="linear"`` returns it as ``eta`` with a ``component`` dimension (brms's ``posterior_linpred``), ``kind="mean"`` the
    mixture's expected response ``sum_k theta_k mu_k`` under the family's mean name (brms's ``posterior_epred``) and
    ``kind="response"`` adds ``y``, a component by theta per draw and row, then its Normal (brms's ``posterior_predict``),
    seeded."""
    if df is None:
        df = model.data
    post, n, fam, r = idata.posterior, model.names, family_of(model), model.mixture_names
    a = post[n["intercept"]].transpose("chain", "draw", r["coordinate"]).to_numpy()            # (chain, draw, K)
    eta = np.repeat(a[..., None, :], len(df), axis=-2)
    b = _coefficient_draws(post, model, axis=r["coordinate"])                                  # (chain, draw, P, K) or None
    if b is not None:
        eta = eta + np.einsum("cdpk,np->cdnk", b, raw_design(model, df))
    dims = ("chain", "draw", "__obs__")
    coords = {"chain": post["chain"], "draw": post["draw"], "__obs__": np.arange(len(df))}
    if kind == "linear":
        if not np.isfinite(eta).all():
            raise ValueError("linear predictor contains nonfinite draws")
        out = {"eta": (dims + ("component",), eta)}
        coords["component"] = list(r["labels"])
    else:
        mean = expected_response(fam, model, post, eta)
        out = {fam.mean_name: (dims, mean)}
        if kind == "response":
            out["y"] = (dims, fam.draw_response(model, post, eta, mean, df, np.random.default_rng(seed)))
    return xr.Dataset(out, coords=coords)


def _group_policies(model, new_groups):
    groups = model.spec.groups
    if not groups and new_groups != "error":
        raise ValueError("this model has no group-level term; new_groups does not apply, so leave it at its default")
    if isinstance(new_groups, Mapping):
        if set(new_groups) != set(groups):
            raise ValueError(f"new_groups must cover exactly the grouping factors {list(groups)}")
        policies = {g: new_groups[g] for g in groups}
    else:
        policies = {g: new_groups for g in groups}
    values = policies.values() if groups else [new_groups]
    if any(not isinstance(value, str) or value not in _NEW_GROUPS for value in values):
        raise ValueError(f"new_groups must be one of {sorted(_NEW_GROUPS)} or an exact-key factor mapping")
    return policies


def _prediction_attrs(model, ds, policies, new_levels, *, in_sample, manual, include_group_effects=True):
    groups = model.spec.groups
    ds.attrs["include_group_effects"] = int(bool(include_group_effects))   # an integer flag: netCDF attributes cannot be boolean
    if not groups or not include_group_effects:
        ds.attrs.update(new_groups="not_applicable", new_levels=[])
    elif len(groups) == 1:
        g = groups[0]
        ds.attrs["new_groups"] = "in_sample" if in_sample else policies[g]
        if manual:
            ds.attrs["new_levels"] = new_levels[g]
    else:
        ds.attrs["new_groups"] = "in_sample" if in_sample else "per_factor"
        ds.attrs["new_groups_by_factor_json"] = json.dumps(policies)
        ds.attrs["new_levels_by_factor_json"] = json.dumps(new_levels)


def predict(model, fit_result, *, newdata: pd.DataFrame | None = None, kind: str = "response",
            new_groups: str | Mapping[str, str] = "error", seed: int | None = None, include_group_effects: bool = True,
            dpar: str | None = None) -> xr.Dataset:
    """Expected response or new response draws, with explicit policies for unseen factors.

    A string applies to every factor; a mapping covers every factor exactly. RNG traversal
    follows factor formula order, then sorted unseen labels. Equal labels in different
    factors are independent identities. Common intercept draws are already on the raw scale.
    ``dpar`` names a distributional parameter with its own formula (B01): ``kind="linear"`` then returns its linear
    predictor and ``kind="mean"`` its values (brms's ``dpar =``); ``kind="response"`` does not apply.
    """
    fam = family_of(model)                                      # the family under the model's link
    if kind not in fam.prediction_kinds:
        raise ValueError(fam.kind_error)
    if dpar is not None:                                        # a distributional parameter itself (B01)
        if dpar not in model.spec.dpar_names:
            raise ValueError(f"dpar must name a distributional parameter with its own formula, {list(model.spec.dpar_names)}; "
                             f"got {dpar!r}" if model.spec.dpars else
                             f"this model has no distributional formula; dpar does not apply (got {dpar!r})")
        if kind not in ("linear", "mean"):
            raise ValueError(f"dpar={dpar!r} takes kind='linear' (its linear predictor) or kind='mean' (its values); a new "
                             "observation is drawn for the response only")
    if not isinstance(include_group_effects, (bool, np.bool_)):
        raise ValueError("include_group_effects must be boolean")
    if not include_group_effects and new_groups != "error":
        raise ValueError("new_groups must stay at its default when group effects are excluded")
    policies = _group_policies(model, new_groups)
    from autobayes.identity import check_fit_binding
    binding = check_fit_binding(model, fit_result, allow_unverified=True)
    idata = getattr(fit_result, "idata", fit_result)
    df = (None if newdata is None else _check_newdata(
        model, newdata, need_trials=(fam.has_trials and kind in ("mean", "response")), include_group_effects=include_group_effects))
    new_levels = {t.factor: ([] if df is None else sorted(set(df[t.column]) - set(model.design.group_designs[t.factor].levels)))
                  for t in (model.spec.group_terms if include_group_effects else ())}
    for g, unseen in new_levels.items():
        if unseen and policies[g] == "error":
            raise FormulaError(
                f"newdata has levels of {g!r} not seen in training: {unseen}. "
                "Pass new_groups='population' to draw their effects from Normal(0, sd), or "
                "new_groups='resample_observed' for brms's default 'uncertainty' behaviour; "
                "an exact-key mapping can set the policy separately for each factor.")
    manual = (not include_group_effects or kind == "linear" or fam.manual_prediction or any(new_levels.values())
              or bool(model.spec.additions)                    # addition terms (B02): the truncated mean and draws are this layer's
              or bool(model.spec.dpars)                        # distributional formulas (B01): each row's own sigma is this layer's
              or bool(model.spec.mo_terms))                    # monotonic terms (B03): Bambi's predictions do not know them
    if dpar is not None:
        ds = _predict_dpar(model, idata, df, kind, dpar)
    elif fam.mixture:                                           # C05: every component's linear predictor, theta's mean and draws
        ds = _predict_mixture(model, idata, df, kind, seed)
    elif manual:
        ds = _predict_by_hand(model, idata, df, kind, seed, policies, include_group_effects=include_group_effects)
    else:
        bambi_kind = "response" if kind == "response" else "response_params"
        out = model.bambi_model.predict(idata, kind=bambi_kind, data=df, inplace=False, random_seed=seed)
        mean_name = model.names["mean"]
        mu = expected_response(fam, model, out.posterior, out.posterior[model.names["linpred"]])
        data = {mean_name: mu.rename({mu.dims[-1]: "__obs__"}) if mu.dims[-1] != "__obs__" else mu}
        if kind == "response":
            y = out.posterior_predictive[model.names["y"]]
            if fam.response_dtype is not None:
                y = y.astype(fam.response_dtype)
            data["y"] = y.rename({y.dims[-1]: "__obs__"}) if y.dims[-1] != "__obs__" else y
        ds = xr.Dataset({k: v.drop_vars([c for c in v.coords if c not in ("chain", "draw")]) for k, v in data.items()})
        ds = ds.assign_coords(__obs__=np.arange(ds.sizes["__obs__"]))
    _prediction_attrs(model, ds, policies, new_levels, in_sample=df is None, manual=manual, include_group_effects=include_group_effects)
    ds.attrs["fit_binding_status"] = binding
    if kind == "linear":
        ds.attrs.update(prediction_scale="linear", link=model.link if dpar is None else model.design.dpars[dpar].link)
    return ds
