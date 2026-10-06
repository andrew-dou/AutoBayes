"""Posterior draws of the mean and posterior predictive draws, with named semantics.

``kind="mean"`` returns draws of the expected response: ``mu`` for the Gaussian family,
``p = P(y = 1)`` for the Bernoulli family, and for the binomial family ``p`` together with
``expected = n p`` (brms ``posterior_epred``, which is ``mu * trials`` there).
``kind="response"`` also returns draws of a new observation ``y``: ``Normal(mu, sigma)``,
``Bernoulli(p)`` or ``Binomial(n, p)``, so ``y`` is an integer for the discrete families and
carries the outcome's own variability (brms ``posterior_predict``). ``kind="probability"``
(binomial only) returns ``p`` alone and needs no trials (brms ``posterior_linpred(transform =
TRUE)``). Binomial predictions are computed in this layer from the posterior; for
``newdata``, ``mean`` and ``response`` need the trials column named in the formula, and a row
with zero trials gives ``expected = 0`` and ``y = 0``.

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
  and every varying slope, each from its own ``Normal(0, sd)``). This is the population
  distribution the model actually fit and is implemented here because Bambi does not offer it.
"""
from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np
import pandas as pd
import xarray as xr

from autobayes.families import get_family
from autobayes.formula import FormulaError, categorical_labels, check_counts, pop_term_design, reject_complex

_NEW_GROUPS = {"error", "resample_observed", "population"}


def raw_design(model, df: pd.DataFrame) -> np.ndarray:
    """The (n, P) raw design of ``df`` in ``model.coefficients`` order: each population-level term's own
    block (``pop_term_design``) side by side: numeric columns as given, one 0/1 indicator per coefficient
    level of a categorical term (reference rows all zero), the numeric column times each indicator for an
    interaction."""
    blocks = [pop_term_design(term, df) for term in model.spec.pop_terms]
    return np.concatenate(blocks, axis=1) if blocks else np.zeros((len(df), 0))


def _coefficient_draws(post, model):
    """(chain, draw, P) posterior draws of the coefficients in design-column order."""
    parts = []
    for term in model.spec.pop_terms:
        arr = post[term.name].to_numpy()
        parts.append(arr if term.levels else arr[..., None])
    return np.concatenate(parts, axis=-1) if parts else None


def _check_newdata(model, newdata: pd.DataFrame, *, need_trials: bool = False, include_group_effects: bool = True) -> pd.DataFrame:
    s = model.spec
    cols = [*s.predictors, *s.categorical_names] + (list(s.groups) if include_group_effects else [])
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
        raise FormulaError(f"newdata lacks columns {missing}")
    df = newdata[cols].copy()
    if df.isna().any().any():
        raise FormulaError("newdata contains missing values")
    for c in s.predictors:
        reject_complex(f"newdata column {c!r}", df[c])
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise FormulaError(f"newdata column {c!r} must be numeric")
    for cat in s.categoricals:
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
    for group in (s.groups if include_group_effects else ()):
        df[group] = df[group].astype(str)
    return df.reset_index(drop=True)


def _predict_by_hand(model, idata, df: pd.DataFrame | None, kind: str, seed, policies: dict[str, str], *, include_group_effects=True) -> xr.Dataset:
    """Linear predictor by hand from the posterior, with one effect per distinct unseen label.

    ``df=None`` means the fitted rows (in-sample). Each group-level coefficient adds its effect at the
    row's level times its covariate (one for the varying intercept, the raw slope column for a varying
    slope). ``"population"``: a fresh ``Normal(0, sd)`` draw per unseen label, posterior draw and
    coefficient. ``"resample_observed"``: the effects of one uniformly chosen observed level per label
    and posterior draw, all coefficients from that same level. In both cases every row with the same
    label shares that label's effects.
    """
    if df is None:
        df = model.data
    post = idata.posterior
    n = model.names
    rng = np.random.default_rng(seed)
    a = (post[n["intercept"]].to_numpy() if model.spec.intercept else
         np.zeros((post.sizes["chain"], post.sizes["draw"])))                         # (chain, draw) raw-scale intercept
    b = _coefficient_draws(post, model)                         # (chain, draw, P) or None
    X = raw_design(model, df)                                   # raw scale: a is raw-scale intercept
    mu = np.repeat(a[..., None], len(df), axis=-1)
    if b is not None:
        mu = mu + np.einsum("cdp,np->cdn", b, X)
    new_levels = {}
    for term in (model.spec.group_terms if include_group_effects else ()):
        g = term.factor
        levels = list(model.design.group_designs[g].levels)
        new_levels[g] = sorted(set(df[g]) - set(levels))
        # (sd draws, effect draws, covariate column) per coefficient: ones for the intercept, the raw column for a slope
        coefs = []
        for coef in term.coefs:
            names = model.group_coefficient_names[(g, coef)]
            covariate = np.ones(len(df)) if coef == "Intercept" else df[coef].to_numpy(dtype=float)
            coefs.append((post[names["sd"]].to_numpy(), post[names["group_effect"]].to_numpy(), covariate))
        new_effects = {}
        for lvl in new_levels[g]:
            if policies[g] == "population":              # each coefficient from its own Normal(0, sd)
                new_effects[lvl] = [rng.normal(0.0, 1.0, size=sd.shape) * sd for sd, _, _ in coefs]
            else:                                        # one observed level per draw; its whole effect vector is reused
                chosen = rng.integers(0, len(levels), size=coefs[0][0].shape)
                new_effects[lvl] = [np.take_along_axis(u, chosen[..., None], axis=-1)[..., 0] for _, u, _ in coefs]
        for i, lvl in enumerate(df[g]):
            for k, (_, u, covariate) in enumerate(coefs):
                mu[..., i] += (new_effects[lvl][k] if lvl in new_effects else u[..., levels.index(lvl)]) * covariate[i]
    dims = ("chain", "draw", "__obs__")
    if kind == "linear":
        if not np.isfinite(mu).all():
            raise ValueError("linear predictor contains nonfinite draws")
        out = {"eta": (dims, mu)}
    else:                                                        # mu held the linear predictor eta
        fam = get_family(model.family)
        mean = fam.inverse_link(mu)
        out = {fam.mean_name: (dims, mean)}
        if fam.has_trials and kind in ("mean", "response"):      # binomial: expected = n p beside p
            out["expected"] = (dims, mean * df[model.spec.trials].to_numpy(dtype=np.int64))
        if kind == "response":
            out["y"] = (dims, fam.draw_response(model, post, mean, df, rng))
    ds = xr.Dataset(out, coords={"chain": post["chain"], "draw": post["draw"], "__obs__": np.arange(len(df))})
    _prediction_attrs(model, ds, policies, new_levels, in_sample=False, manual=True, include_group_effects=include_group_effects)
    return ds


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
            new_groups: str | Mapping[str, str] = "error", seed: int | None = None, include_group_effects: bool = True) -> xr.Dataset:
    """Expected response or new response draws, with explicit policies for unseen factors.

    A string applies to every factor; a mapping covers every factor exactly. RNG traversal
    follows factor formula order, then sorted unseen labels. Equal labels in different
    factors are independent identities. Common intercept draws are already on the raw scale.
    """
    fam = get_family(model.family)
    if kind not in fam.prediction_kinds:
        raise ValueError(fam.kind_error)
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
    new_levels = {g: ([] if df is None else sorted(set(df[g]) - set(model.design.group_designs[g].levels)))
                  for g in (model.spec.groups if include_group_effects else ())}
    for g, unseen in new_levels.items():
        if unseen and policies[g] == "error":
            raise FormulaError(
                f"newdata has levels of {g!r} not seen in training: {unseen}. "
                "Pass new_groups='population' to draw their effects from Normal(0, sd), or "
                "new_groups='resample_observed' for brms's default 'uncertainty' behaviour; "
                "an exact-key mapping can set the policy separately for each factor.")
    manual = not include_group_effects or kind == "linear" or fam.manual_prediction or any(new_levels.values())
    if manual:
        ds = _predict_by_hand(model, idata, df, kind, seed, policies, include_group_effects=include_group_effects)
    else:
        bambi_kind = "response" if kind == "response" else "response_params"
        out = model.bambi_model.predict(idata, kind=bambi_kind, data=df, inplace=False, random_seed=seed)
        mean_name = model.names["mean"]
        mu = fam.inverse_link(out.posterior[model.names["linpred"]])
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
        ds.attrs.update(prediction_scale="linear", link=model.link)
    return ds
