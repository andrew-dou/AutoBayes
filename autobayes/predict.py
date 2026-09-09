"""Posterior draws of the mean and posterior predictive draws, with named semantics.

``kind="mean"`` returns draws of ``mu`` (the expected response); ``kind="response"``
also returns draws of a new observation ``y ~ Normal(mu, sigma)``.

``new_groups`` decides what happens when ``newdata`` contains a level of the grouping
factor that was not in the training data:

- ``"error"`` (default): refuse. The safe default.
- ``"resample_observed"``: Bambi's ``sample_new_groups=True``, which is brms's default
  ``sample_new_levels="uncertainty"``: for each posterior draw, reuse the fitted effect
  of a randomly chosen observed level. This understates between-level variance because
  it resamples shrunk draws from a finite set of levels.
- ``"population"``: brms's ``sample_new_levels="gaussian"``: for each posterior draw,
  draw a fresh effect from ``Normal(0, sd_group)``. This is the population distribution
  the model actually fit and is implemented here because Bambi does not offer it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from autobayes.formula import FormulaError

_NEW_GROUPS = {"error", "resample_observed", "population"}


def _check_newdata(model, newdata: pd.DataFrame) -> pd.DataFrame:
    s = model.spec
    cols = [*s.predictors, s.group]
    missing = [c for c in cols if c not in newdata.columns]
    if missing:
        raise FormulaError(f"newdata lacks columns {missing}")
    df = newdata[cols].copy()
    if df.isna().any().any():
        raise FormulaError("newdata contains missing values")
    for c in s.predictors:
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise FormulaError(f"newdata column {c!r} must be numeric")
    df[s.group] = df[s.group].astype(str)
    return df.reset_index(drop=True)


def _population_predict(model, idata, df: pd.DataFrame, kind: str, seed) -> xr.Dataset:
    """Linear predictor by hand from the posterior, drawing new-level effects from Normal(0, sd)."""
    post = idata.posterior
    n = model.names
    rng = np.random.default_rng(seed)
    a = post[n["intercept"]].to_numpy()                         # (chain, draw) raw-scale intercept
    b = np.stack([post[p].to_numpy() for p in model.spec.predictors], axis=-1) if model.spec.predictors else None
    sd = post[n["sd"]].to_numpy()
    sigma = post[n["sigma"]].to_numpy()
    u = post[n["group_effect"]].to_numpy()                      # (chain, draw, J)
    levels = list(model.design.levels)
    X = df[list(model.spec.predictors)].to_numpy(dtype=float)  # raw scale: a is raw-scale intercept
    mu = np.repeat(a[..., None], len(df), axis=-1)
    if b is not None:
        mu = mu + np.einsum("cdp,np->cdn", b, X)
    new_levels = sorted(set(df[model.spec.group]) - set(levels))
    new_effects = {lvl: rng.normal(0.0, 1.0, size=sd.shape) * sd for lvl in new_levels}
    for i, lvl in enumerate(df[model.spec.group]):
        if lvl in new_effects:
            mu[..., i] += new_effects[lvl]
        else:
            mu[..., i] += u[..., levels.index(lvl)]
    out = {"mu": (("chain", "draw", "__obs__"), mu)}
    if kind == "response":
        out["y"] = (("chain", "draw", "__obs__"), rng.normal(mu, sigma[..., None]))
    ds = xr.Dataset(out, coords={"chain": post["chain"], "draw": post["draw"], "__obs__": np.arange(len(df))})
    ds.attrs["new_groups"] = "population"
    ds.attrs["new_levels"] = new_levels
    return ds


def predict(model, fit_result, *, newdata: pd.DataFrame | None = None, kind: str = "response",
            new_groups: str = "error", seed: int | None = None) -> xr.Dataset:
    """Return an ``xarray.Dataset`` with ``mu`` and, for ``kind="response"``, ``y``."""
    if kind not in {"mean", "response"}:
        raise ValueError("kind must be 'mean' or 'response'")
    if new_groups not in _NEW_GROUPS:
        raise ValueError(f"new_groups must be one of {sorted(_NEW_GROUPS)}")
    idata = getattr(fit_result, "idata", fit_result)
    df = None if newdata is None else _check_newdata(model, newdata)

    if df is not None:
        unseen = sorted(set(df[model.spec.group]) - set(model.design.levels))
        if unseen and new_groups == "error":
            raise FormulaError(
                f"newdata has levels of {model.spec.group!r} not seen in training: {unseen}. "
                "Pass new_groups='population' to draw their effects from Normal(0, sd), or "
                "new_groups='resample_observed' for brms's default 'uncertainty' behaviour."
            )
        if new_groups == "population" and unseen:
            return _population_predict(model, idata, df, kind, seed)

    bambi_kind = "response" if kind == "response" else "response_params"
    out = model.bambi_model.predict(
        idata, kind=bambi_kind, data=df, inplace=False,
        sample_new_groups=(new_groups == "resample_observed"), random_seed=seed,
    )
    mu = out.posterior[model.names["mu"]]
    data = {"mu": mu.rename({mu.dims[-1]: "__obs__"}) if mu.dims[-1] != "__obs__" else mu}
    if kind == "response":
        y = out.posterior_predictive[model.names["y"]]
        data["y"] = y.rename({y.dims[-1]: "__obs__"}) if y.dims[-1] != "__obs__" else y
    ds = xr.Dataset({k: v.drop_vars([c for c in v.coords if c not in ("chain", "draw")]) for k, v in data.items()})
    ds = ds.assign_coords(__obs__=np.arange(ds.sizes["__obs__"]))
    ds.attrs["new_groups"] = new_groups if df is not None else "in_sample"
    return ds
