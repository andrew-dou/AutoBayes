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
factor that was not in the training data:

- ``"error"`` (default): refuse. The safe default.
- ``"resample_observed"``: brms's default ``sample_new_levels="uncertainty"``: for each
  posterior draw and each distinct unseen label, reuse the fitted effect of a uniformly
  chosen observed level (chosen independently per label and per draw; rows sharing a
  label share the choice). This understates between-level variance because it resamples
  shrunk draws from a finite set of levels. Computed here rather than through Bambi's
  ``sample_new_groups=True``, which gives every unseen label the same effect in a draw.
- ``"population"``: brms's ``sample_new_levels="gaussian"``: for each posterior draw,
  draw a fresh effect from ``Normal(0, sd_group)`` (on the linear-predictor scale, i.e. the
  logit scale for Bernoulli). This is the population distribution the model actually fit
  and is implemented here because Bambi does not offer it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from scipy.special import expit

from autobayes.formula import FormulaError, categorical_labels, check_counts, reject_complex

_NEW_GROUPS = {"error", "resample_observed", "population"}


def raw_design(model, df: pd.DataFrame) -> np.ndarray:
    """The (n, P) raw design of ``df`` in ``model.coefficients`` order: numeric columns as given,
    one 0/1 indicator per coefficient level of a categorical term (reference rows all zero)."""
    cols = []
    for term in model.spec.terms:
        if term in model.spec.categorical_names:
            cat = model.spec.categorical(term)
            labels = df[term].astype(str).to_numpy()
            cols.extend((labels == lvl).astype(float) for lvl in cat.coefficients)
        elif term in model.spec.interaction_names:
            xn, zn = model.spec.interaction_of(term)
            labels = df[zn].astype(str).to_numpy()
            xv = df[xn].to_numpy(dtype=float)
            cols.extend(xv * (labels == lvl) for lvl in model.spec.levels_of(term))
        else:
            cols.append(df[term].to_numpy(dtype=float))
    return np.column_stack(cols) if cols else np.zeros((len(df), 0))


def _coefficient_draws(post, model):
    """(chain, draw, P) posterior draws of the coefficients in design-column order."""
    parts = []
    for term in model.spec.terms:
        arr = post[term].to_numpy()
        parts.append(arr if model.spec.levels_of(term) else arr[..., None])
    return np.concatenate(parts, axis=-1) if parts else None


def _check_newdata(model, newdata: pd.DataFrame, *, need_trials: bool = False) -> pd.DataFrame:
    s = model.spec
    cols = [*s.predictors, *s.categorical_names, s.group]
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
    df[s.group] = df[s.group].astype(str)
    return df.reset_index(drop=True)


def _predict_by_hand(model, idata, df: pd.DataFrame | None, kind: str, seed, new_groups: str) -> xr.Dataset:
    """Linear predictor by hand from the posterior, with one effect per distinct unseen label.

    ``df=None`` means the fitted rows (in-sample). ``"population"``: a fresh ``Normal(0, sd)``
    draw per unseen label and posterior draw. ``"resample_observed"``: the effect of a
    uniformly chosen observed level per label and posterior draw. In both cases every row
    with the same label shares that label's effect.
    """
    if df is None:
        df = model.data
    post = idata.posterior
    n = model.names
    rng = np.random.default_rng(seed)
    a = post[n["intercept"]].to_numpy()                         # (chain, draw) raw-scale intercept
    b = _coefficient_draws(post, model)                         # (chain, draw, P) or None
    sd = post[n["sd"]].to_numpy()
    u = post[n["group_effect"]].to_numpy()                      # (chain, draw, J)
    levels = list(model.design.levels)
    X = raw_design(model, df)                                   # raw scale: a is raw-scale intercept
    mu = np.repeat(a[..., None], len(df), axis=-1)
    if b is not None:
        mu = mu + np.einsum("cdp,np->cdn", b, X)
    new_levels = sorted(set(df[model.spec.group]) - set(levels))
    new_effects = {}
    for lvl in new_levels:                                      # one effect per label per draw
        if new_groups == "population":
            new_effects[lvl] = rng.normal(0.0, 1.0, size=sd.shape) * sd
        else:                                                   # resample_observed
            chosen = rng.integers(0, len(levels), size=sd.shape)
            new_effects[lvl] = np.take_along_axis(u, chosen[..., None], axis=-1)[..., 0]
    for i, lvl in enumerate(df[model.spec.group]):
        if lvl in new_effects:
            mu[..., i] += new_effects[lvl]
        else:
            mu[..., i] += u[..., levels.index(lvl)]
    if model.family == "gaussian":
        out = {"mu": (("chain", "draw", "__obs__"), mu)}
        if kind == "response":
            sigma = post[n["sigma"]].to_numpy()
            out["y"] = (("chain", "draw", "__obs__"), rng.normal(mu, sigma[..., None]))
    elif model.family == "bernoulli":
        prob = expit(mu)                                         # mu held the linear predictor eta
        out = {"p": (("chain", "draw", "__obs__"), prob)}
        if kind == "response":
            out["y"] = (("chain", "draw", "__obs__"), rng.binomial(1, prob).astype(np.int64))
    else:                                                        # binomial: p, expected = n p, count draws
        prob = expit(mu)
        out = {"p": (("chain", "draw", "__obs__"), prob)}
        if kind in ("mean", "response"):
            trials = df[model.spec.trials].to_numpy(dtype=np.int64)
            out["expected"] = (("chain", "draw", "__obs__"), prob * trials)
            if kind == "response":
                out["y"] = (("chain", "draw", "__obs__"), rng.binomial(trials, prob).astype(np.int64))
    ds = xr.Dataset(out, coords={"chain": post["chain"], "draw": post["draw"], "__obs__": np.arange(len(df))})
    ds.attrs["new_groups"] = new_groups
    ds.attrs["new_levels"] = new_levels
    return ds


def predict(model, fit_result, *, newdata: pd.DataFrame | None = None, kind: str = "response",
            new_groups: str = "error", seed: int | None = None) -> xr.Dataset:
    """Return an ``xarray.Dataset`` with the mean (``mu`` or ``p``, plus ``expected`` for binomial)
    and, for ``kind="response"``, ``y``; ``kind="probability"`` (binomial) gives ``p`` only."""
    if model.family == "binomial":
        if kind not in {"probability", "mean", "response"}:
            raise ValueError("kind must be 'probability', 'mean' or 'response' for a binomial model")
    elif kind not in {"mean", "response"}:
        raise ValueError("kind must be 'mean' or 'response'" + (" ('probability' is for binomial models; here 'mean' is p)" if model.family == "bernoulli" else ""))
    if new_groups not in _NEW_GROUPS:
        raise ValueError(f"new_groups must be one of {sorted(_NEW_GROUPS)}")
    idata = getattr(fit_result, "idata", fit_result)
    if model.family == "binomial":
        df = None if newdata is None else _check_newdata(model, newdata, need_trials=(kind != "probability"))
        if df is not None:
            unseen = sorted(set(df[model.spec.group]) - set(model.design.levels))
            if unseen and new_groups == "error":
                raise FormulaError(
                    f"newdata has levels of {model.spec.group!r} not seen in training: {unseen}. "
                    "Pass new_groups='population' to draw their effects from Normal(0, sd), or "
                    "new_groups='resample_observed' for brms's default 'uncertainty' behaviour."
                )
        ds = _predict_by_hand(model, idata, df, kind, seed, new_groups)
        if df is None:
            ds.attrs["new_groups"] = "in_sample"
        return ds
    df = None if newdata is None else _check_newdata(model, newdata)

    if df is not None:
        unseen = sorted(set(df[model.spec.group]) - set(model.design.levels))
        if unseen and new_groups == "error":
            raise FormulaError(
                f"newdata has levels of {model.spec.group!r} not seen in training: {unseen}. "
                "Pass new_groups='population' to draw their effects from Normal(0, sd), or "
                "new_groups='resample_observed' for brms's default 'uncertainty' behaviour."
            )
        if unseen:                                              # "population" or "resample_observed"
            return _predict_by_hand(model, idata, df, kind, seed, new_groups)

    bambi_kind = "response" if kind == "response" else "response_params"
    out = model.bambi_model.predict(idata, kind=bambi_kind, data=df, inplace=False, random_seed=seed)
    mean_name = model.names["mean"]
    mu = out.posterior[model.names["linpred"]]
    if model.family == "bernoulli":
        mu = expit(mu)                                          # logit_p -> p = P(y = 1)
    data = {mean_name: mu.rename({mu.dims[-1]: "__obs__"}) if mu.dims[-1] != "__obs__" else mu}
    if kind == "response":
        y = out.posterior_predictive[model.names["y"]]
        if model.family == "bernoulli":
            y = y.astype(np.int64)
        data["y"] = y.rename({y.dims[-1]: "__obs__"}) if y.dims[-1] != "__obs__" else y
    ds = xr.Dataset({k: v.drop_vars([c for c in v.coords if c not in ("chain", "draw")]) for k, v in data.items()})
    ds = ds.assign_coords(__obs__=np.arange(ds.sizes["__obs__"]))
    ds.attrs["new_groups"] = new_groups if df is not None else "in_sample"
    return ds
