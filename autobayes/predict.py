"""Posterior draws of the mean and posterior predictive draws, with named semantics.

``kind="mean"`` returns draws of the expected response: ``mu`` for the Gaussian family,
``p = P(y = 1)`` for the Bernoulli family, for the binomial family ``p`` together with
``expected = n p`` (brms ``posterior_epred``, which is ``mu * trials`` there), and for the
lognormal family ``mean = exp(mu + sigma^2 / 2)`` (brms ``posterior_epred``; ``mu`` is the
log-scale location, which ``kind="linear"`` returns).
``kind="response"`` also returns draws of a new observation ``y``: ``Normal(mu, sigma)``,
``Bernoulli(p)`` or ``Binomial(n, p)``, so ``y`` is an integer for the discrete families and
carries the outcome's own variability (brms ``posterior_predict``). ``kind="probability"``
(binomial, and since D02 beta-binomial) returns the success probability alone (``p``; the beta-binomial's mean ``mu``) and
needs no trials (brms ``posterior_linpred(transform = TRUE)``). Binomial predictions are computed in this layer from the
posterior; for ``newdata``, ``mean`` and ``response`` need the trials column named in the formula, and a row with zero
trials gives ``expected = 0`` and ``y = 0``.

A model with response addition terms (B02, docs/addition-terms.md) is predicted in this layer: ``weights()`` and
``cens()`` change nothing here (brms's ``posterior_epred`` and ``posterior_predict`` ignore them, and new rows need
neither column), while under ``trunc()`` the mean is the truncated distribution's mean and ``y`` is drawn from the
truncated distribution, as brms does.

A model with a monotonic term ``mo(x)`` (B03, docs/monotonic.md) is predicted in this layer: each term adds its coefficient
times ``D * sum(simplex[:Xmo_i])`` per draw and row, with every row's value coded as the fitted rows were (a value outside
the fitted coding is refused, as brms refuses it).

A model with a Gaussian process ``gp(x)`` (C03, docs/gaussian-processes.md) is predicted in this layer: per draw, the exact
GP's values at its distinct fitted inputs, taken at each row's input (a new row must take a fitted input: brms draws a value
at any other input from the conditional normal, which is not offered here, divergence D65), or the approximate GP's basis at
each row's input times ``sqrt(S) * zgp`` (deterministic, as brms's; an input outside the basis's boundary is refused, D66).

A model whose auxiliary parameter has its own formula (B01, docs/distributional.md: ``["y ~ x", "sigma ~ z"]``) is
predicted in this layer too: ``kind="linear"`` and ``kind="mean"`` keep their meaning (the linear predictor of ``mu`` and
the expected response, brms's ``posterior_linpred`` and ``posterior_epred``), with each row's own ``sigma`` wherever the
family needs it (the lognormal's mean, every ``kind="response"`` draw); ``dpar="sigma"`` returns the parameter itself,
its linear predictor under ``kind="linear"`` (brms ``posterior_linpred(dpar = "sigma")``, named ``eta``) or its values
under ``kind="mean"`` (brms ``posterior_epred(dpar = "sigma")``, named ``sigma``). Its intercept draws are on the raw scale,
as Bambi reports them after sampling.

A model with an ``me(x, sdx)`` or ``mi(col)`` term (C06, docs/missing-and-measurement-error.md) is predicted in this layer: on
the fitted rows the term adds its coefficient times the draws' latent values (brms's ``Xme``) or the column completed by the
draws' missing values (brms's ``Yl``); on new rows an ``mi()`` column must be observed and an ``me()`` term is refused (brms draws
the latent values of new rows; divergence D74).

A finite mixture (C05, docs/mixtures.md) is predicted in this layer, population-level only: ``kind="linear"`` returns every
component's linear predictor as ``eta`` with a ``component`` dimension (brms's ``posterior_linpred``), ``kind="mean"`` the
mixture's expected response ``sum_k theta_k mu_k`` as ``mean`` (brms's ``posterior_epred``) and ``kind="response"`` draws, per
draw and row, a component by theta and then its Normal (brms's ``posterior_predict``), seeded.

A model with an autocorrelation term (C04, docs/autocorrelation.md: brms's ``ar()``, ``ma()``, ``arma()``) is predicted in
this layer on its fitted rows: the ARMA term is added to the linear predictor from the rows' observed residuals along each
series, as brms's ``posterior_linpred`` and ``posterior_epred`` add it (without group effects, the residuals are those of
the predictor without them, as under brms's ``re_formula = NA``), and ``kind="response"`` draws around that predictor. New
rows are refused: brms computes the term on them from their own responses (divergence D71).

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
from autobayes.formula import (FormulaError, by_levels_of, categorical_labels, check_counts, group_labels, mm_weights, pop_term_design,
                               reject_complex)

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


def gp_contribution(model, post, df: pd.DataFrame) -> np.ndarray | float:
    """(chain, draw, row) draws of the Gaussian processes' part of the linear predictor (C03) on the rows of ``df`` (the fitted
    frame, or checked new rows): an exact GP's values at its distinct fitted inputs from each draw's ``sdgp``, ``lscale`` and
    ``zgp``, taken at each row's input, which must be a fitted input (divergence D65); an approximate GP's basis at each row's
    input, which must lie inside its boundary (D66), times ``sqrt(S) * zgp``. 0 without a Gaussian process."""
    from autobayes.gaussian_process import approximate_values, exact_values

    out = 0.0
    for column, roles in model.gp_names.items():
        g = model.spec.gp(column)
        sdgp, lscale = post[roles["sdgp"]].to_numpy(), post[roles["lscale"]].to_numpy()
        z = post[roles["zgp"]].transpose("chain", "draw", roles["coordinate"]).to_numpy()
        if g.approximate:
            out = out + approximate_values(g.basis(df[column], where="newdata column"), g.slambda, sdgp, lscale, z)
        else:
            out = out + exact_values(g.scaled_inputs, sdgp, lscale, z)[..., g.index(df[column], where="newdata column")]
    return out


def autocor_predictor(model, post, eta: np.ndarray) -> np.ndarray:
    """(chain, draw, row) draws of the linear predictor with an autocorrelation term (C04) on the fitted rows: ``eta``, the
    predictor without the term, plus brms's ARMA term on the observed residuals ``y - eta`` along each series
    (``autocor.arma_values``, brms's row-by-row recursion), with each draw's ``ar`` and ``ma``."""
    from autobayes.autocor import arma_values

    roles, coefficients = model.autocor_names, {}
    for part, _ in model.spec.autocor.parts:
        coefficients[part] = post[roles[part]].transpose("chain", "draw", roles[f"{part}_coordinate"]).to_numpy()
    return arma_values(eta, model.design.y, coefficients.get("ar"), coefficients.get("ma"), model.design.autocor)


def measurement_contribution(model, post, df: pd.DataFrame, *, in_sample: bool) -> np.ndarray | float:
    """(chain, draw, row) draws of the ``me()`` and ``mi()`` terms' part of the linear predictor (C06,
    docs/missing-and-measurement-error.md) on the rows of ``df``. On the fitted rows (``in_sample``) an ``me(x, sdx)`` term adds
    its coefficient times the latent values ``meanme + sdme * zme`` of each draw (brms's ``Xme``), and an ``mi(col)`` term its
    coefficient times the column, observed or, at a missing row, the draw's ``Ymi_col`` (brms's ``Yl``). On new rows an
    ``mi(col)`` term reads the column, which must be observed (``_check_newdata``), and an ``me()`` term is refused: brms draws
    the latent values of new rows (divergence D74). 0 without either term."""
    out = 0.0
    for t in getattr(model.spec, "me_terms", ()):
        if not in_sample:
            raise FormulaError(f"predict(newdata=...) of a model with {t.name!r} is not offered: brms draws the latent values of "
                               "new rows, which no fixed rule reproduces; predict on the fitted rows (newdata=None) "
                               "(docs/missing-and-measurement-error.md, divergence D74)")
        roles = model.me_names[t.operands[0]]
        latent = (post[roles["mean"]].transpose("chain", "draw").to_numpy()[..., None]
                  + post[roles["sd"]].transpose("chain", "draw").to_numpy()[..., None]
                  * post[roles["z"]].transpose("chain", "draw", ...).to_numpy())
        out = out + post[roles["b"]].transpose("chain", "draw").to_numpy()[..., None] * latent
    for t in getattr(model.spec, "mi_terms", ()):
        col = t.operands[0]
        coefficient = post[t.name].transpose("chain", "draw").to_numpy()[..., None]
        if in_sample:
            block, roles = model.design.missing[col], model.missing_names.get(col)
            values = np.broadcast_to(block.filled, coefficient.shape[:2] + block.filled.shape).copy()
            if roles is not None:
                values[..., block.index] = post[roles["values"]].transpose("chain", "draw", roles["coordinate"]).to_numpy()
        else:
            values = df[col].to_numpy(dtype=float)
        out = out + coefficient * values
    return out


def dpar_linear_predictor(model, post, df: pd.DataFrame, name: str, *, include_group_effects: bool = True) -> np.ndarray:
    """(chain, draw, row) draws of distributional parameter ``name``'s linear predictor (B01) on the rows of ``df`` (the
    fitted frame, or checked new rows): its raw-scale intercept draws plus its raw design times its coefficient draws, as
    the response formula's is computed here; with its group-level terms (D08: a correlation ID shared with the response
    formula's term) each row adds the effect of its level, unless ``include_group_effects`` is False. A level the fit has not
    seen is refused here (``predict()`` refuses new levels of a shared ID's factor first)."""
    ds, roles = model.spec.dpar(name), model.dpar_names[name]
    shape = (post.sizes["chain"], post.sizes["draw"])
    a = post[roles["intercept"]].to_numpy() if ds.intercept else np.zeros(shape)
    eta = np.repeat(a[..., None], len(df), axis=-1)
    if ds.pop_terms:
        X = np.concatenate([pop_term_design(t, df, ds.categoricals) for t in ds.pop_terms], axis=1)
        b = np.concatenate([post[roles[f"b:{t.name}"]].to_numpy() if t.levels else post[roles[f"b:{t.name}"]].to_numpy()[..., None]
                            for t in ds.pop_terms], axis=-1)
        eta = eta + np.einsum("cdp,np->cdn", b, X)
    for t in (ds.group_terms if include_group_effects else ()):   # D08: a shared ID's varying intercept of the parameter
        levels = list(model.design.dpars[name].group_blocks[(t.factor, "Intercept")].levels)
        labels = group_labels(t.factor, df).to_numpy()
        unseen = sorted(set(labels) - set(levels))
        if unseen:
            raise FormulaError(f"levels {unseen} of {t.factor!r} were not seen in training; {name}'s formula has no effect for them "
                               "(new levels of a factor whose correlation ID is shared are not offered; divergence D106)")
        u = post[model.dpar_group_names[(name, t.factor, "Intercept")]["group_effect"]].transpose("chain", "draw", ...).to_numpy()
        eta = eta + u[..., [levels.index(l) for l in labels]]
    return eta


def dpar_values(model, post, df: pd.DataFrame, name: str, *, include_group_effects: bool = True) -> np.ndarray:
    """(chain, draw, row) draws of distributional parameter ``name`` itself (B01): the inverse link of its linear
    predictor, refused where it is not a positive finite number."""
    return DISTRIBUTIONAL_INVERSE_LINKS[model.design.dpars[name].link](
        dpar_linear_predictor(model, post, df, name, include_group_effects=include_group_effects), name)


def _with_dpars(model, post, df: pd.DataFrame, *, include_group_effects: bool = True):
    """The posterior with each distributional parameter's per-row draws under its PyMC name (B01), the form in which the
    families' mean and response functions read an auxiliary parameter (``families.spec.auxiliary``)."""
    return post.assign({model.dpar_names[name]["value"]: (("chain", "draw", "__row__"),
                                                         dpar_values(model, post, df, name, include_group_effects=include_group_effects))
                        for name in model.spec.dpar_names})


def term_labels(term, df: pd.DataFrame) -> list[np.ndarray]:
    """The level labels a group-level term reads on the rows of ``df``: one array, or one per member of a multi-membership
    term (D08)."""
    if term.is_mm:
        return [group_labels(c, df).to_numpy() for c in term.members]
    return [df[term.column].to_numpy()]


def _check_newdata(model, newdata: pd.DataFrame, *, need_trials: bool = False, include_group_effects: bool = True) -> pd.DataFrame:
    s = model.spec
    operands = [c for t in s.group_terms for c in t.data_columns] if include_group_effects else []   # g and h, never g_h; D08:
    operands += [c for d in s.dpars for t in d.group_terms for c in t.data_columns] if include_group_effects else []   # mm(), by
    cols = [*s.model_columns] + ([s.offset.column] if s.offset is not None else []) + operands   # every formula's columns (B01)
    cols = list(dict.fromkeys(cols))
    if need_trials:
        if s.trials not in newdata.columns:
            raise FormulaError(
                f"newdata lacks the trials column {s.trials!r}, which kind='mean' and kind='response' need "
                "for a model with trials, binomial or beta-binomial (kind='probability' does not)."
            )
        cols.append(s.trials)
    missing = [c for c in cols if c not in newdata.columns]
    if missing:
        hint = (f" (the offset term {s.offset.term!r} needs {s.offset.column!r} on every row to predict)"
                if s.offset is not None and s.offset.column in missing else "")
        raise FormulaError(f"newdata lacks columns {missing}{hint}")
    df = newdata[cols].copy()
    if getattr(s, "me_terms", ()):                               # C06: brms draws the latent values of new rows
        raise FormulaError(f"predict(newdata=...) of a model with {s.me_terms[0].name!r} is not offered: brms draws the latent values "
                           "of new rows, which no fixed rule reproduces; predict on the fitted rows (newdata=None) "
                           "(docs/missing-and-measurement-error.md, divergence D74)")
    for t in getattr(s, "mi_terms", ()):                         # C06: an mi() predictor is imputed on the fitted rows only
        if df[t.operands[0]].isna().any():
            raise FormulaError(f"newdata column {t.operands[0]!r}, read by {t.name!r}, has missing values: on new rows an mi() "
                               "predictor must be observed (the model imputes the fitted rows only; "
                               "docs/missing-and-measurement-error.md, divergence D74)")
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
        if t.is_mm:                                              # D08: the member columns, and the weights checked
            for c in t.members:
                df[c] = group_labels(c, df)
            mm_weights(t, df, where="newdata column")
            continue
        df[t.column] = group_labels(t.factor, df)
        if t.by is not None:                                     # D08: a fitted level keeps its fitted by level
            block = model.design.group_blocks[(t.factor, "Intercept")]
            fitted = {level: block.by_levels[block.by_index[j]] for j, level in enumerate(block.levels)}
            for level, b in by_levels_of(t, df, where="newdata column").items():
                if level in fitted and fitted[level] != b:
                    raise FormulaError(f"newdata gives the level {level!r} of {t.factor!r} the {t.by!r} level {b!r}; it was fitted "
                                       f"within {fitted[level]!r} (gr({t.factor}, by = {t.by}))")
            df[t.by] = group_labels(t.by, df)
    return df.reset_index(drop=True)


def _predict_dpar(model, idata, df: pd.DataFrame | None, kind: str, name: str, *, include_group_effects: bool = True) -> xr.Dataset:
    """``predict(..., dpar=name)`` (B01): the distributional parameter's linear predictor (``kind="linear"``, ``eta``) or
    its values (``kind="mean"``, under its brms name), per draw and row; its formula's group-level terms (D08, a shared
    correlation ID's) add the effects of fitted levels only (``predict()`` refuses new levels of such a factor first), so the
    group policies change nothing here and nothing is random (``predict()`` writes the policy attributes)."""
    if df is None:
        df = model.data
    post = idata.posterior
    eta = dpar_linear_predictor(model, post, df, name, include_group_effects=include_group_effects)
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
    in_sample = df is None                                      # the fitted rows: me() and mi() terms read their draws there (C06)
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
    mu = mu + gp_contribution(model, post, df)                  # the Gaussian processes (C03), population-level as well
    mu = mu + measurement_contribution(model, post, df, in_sample=in_sample)   # the me() and mi() terms (C06)
    new_levels = {}
    for term in (model.spec.group_terms if include_group_effects else ()):
        g = term.factor
        levels = list(model.design.group_designs[g].levels)
        members = term_labels(term, df)                         # D08: one label array per member of a multi-membership term
        weights = mm_weights(term, df) if term.is_mm else np.ones((len(df), 1))
        labels = members[0]
        new_levels[g] = sorted(set().union(*(set(m) for m in members)) - set(levels))
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
        for member, w in zip(members, weights.T):                # one member, weight 1, except for a multi-membership term (D08)
            for i, lvl in enumerate(member):
                for k, (_, u, covariate) in enumerate(coefs):
                    effect = (new_effects[lvl][k] if lvl in new_effects else u[..., levels.index(lvl)]) * covariate[i]
                    mu[..., i] += effect if not term.is_mm else w[i] * effect
    if model.spec.autocor is not None:                          # C04: the ARMA term on the fitted rows' residuals, added last
        mu = autocor_predictor(model, post, mu)
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
            post = _with_dpars(model, post, df, include_group_effects=include_group_effects)   # each row's own sigma (B01)
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
            dpar: str | None = None, nlpar: str | None = None) -> xr.Dataset:
    """Expected response or new response draws, with explicit policies for unseen factors.

    A string applies to every factor; a mapping covers every factor exactly. RNG traversal
    follows factor formula order, then sorted unseen labels. Equal labels in different
    factors are independent identities. Common intercept draws are already on the raw scale.
    ``dpar`` names a distributional parameter with its own formula (B01): ``kind="linear"`` then returns its linear
    predictor and ``kind="mean"`` its values (brms's ``dpar =``); ``kind="response"`` does not apply.
    A non-linear model (C02, docs/nonlinear.md) is predicted by ``nonlinear.predict_nonlinear``; ``nlpar`` names one of its
    non-linear parameters, whose values ``kind="linear"`` and ``kind="mean"`` then return (brms's ``nlpar =``).
    """
    if getattr(getattr(model, "spec", None), "is_nonlinear", False):    # a non-linear model (C02)
        from autobayes.nonlinear import predict_nonlinear

        return predict_nonlinear(model, fit_result, newdata=newdata, kind=kind, new_groups=new_groups, seed=seed,
                                 include_group_effects=include_group_effects, dpar=dpar, nlpar=nlpar)
    if nlpar is not None:
        raise ValueError(f"nlpar applies to a non-linear model (nl=True; docs/nonlinear.md); this model has no non-linear parameter "
                         f"(got {nlpar!r})")
    if getattr(model, "responses", None):                       # C01: a multivariate model predicts response by response
        from autobayes.multivariate import predict_multivariate

        return predict_multivariate(model, fit_result, newdata=newdata, kind=kind, new_groups=new_groups, seed=seed,
                                    include_group_effects=include_group_effects, dpar=dpar)
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
    if model.spec.autocor is not None and newdata is not None:  # C04: brms reads the term off the new rows' own responses (D71)
        raise FormulaError(
            f"predictions on new data are not offered for a model with the autocorrelation term {model.spec.autocor.term!r}: the "
            "term needs each row's lagged residuals, which new rows without their responses do not have (brms computes it from "
            "the new rows' own responses, sorted by series and time, as a series of their own; docs/autocorrelation.md, "
            "divergence D71); predict on the fitted rows (newdata=None)"
        )
    from autobayes.identity import check_fit_binding
    binding = check_fit_binding(model, fit_result, allow_unverified=True)
    idata = getattr(fit_result, "idata", fit_result)
    df = (None if newdata is None else _check_newdata(
        model, newdata, need_trials=(fam.has_trials and kind in ("mean", "response")), include_group_effects=include_group_effects))
    new_levels = {t.factor: ([] if df is None else sorted(set().union(*(set(m) for m in term_labels(t, df)))
                                                          - set(model.design.group_designs[t.factor].levels)))
                  for t in (model.spec.group_terms if include_group_effects else ())}
    for g, unseen in new_levels.items():
        t = model.spec.group_term_of(g)
        if unseen and policies[g] != "error" and (t.by is not None or t.id is not None):   # D08
            raise FormulaError(
                f"newdata has levels of {g!r} not seen in training: {unseen}; new levels of "
                + (f"'{t.canonical}' (one SD per level of {t.by!r})" if t.by is not None else
                   f"a factor whose correlation ID {t.id!r} is shared with a distributional formula")
                + " are not offered in this slice (docs/multi-membership-and-by.md, divergences D105, D106)")
        if unseen and policies[g] == "error":
            raise FormulaError(
                f"newdata has levels of {g!r} not seen in training: {unseen}. "
                "Pass new_groups='population' to draw their effects from Normal(0, sd), or "
                "new_groups='resample_observed' for brms's default 'uncertainty' behaviour; "
                "an exact-key mapping can set the policy separately for each factor.")
    manual = (not include_group_effects or kind == "linear" or fam.manual_prediction or any(new_levels.values())
              or bool(model.spec.additions)                    # addition terms (B02): the truncated mean and draws are this layer's
              or bool(model.spec.dpars)                        # distributional formulas (B01): each row's own sigma is this layer's
              or bool(model.spec.mo_terms)                     # monotonic terms (B03): Bambi's predictions do not know them
              or bool(model.spec.gp_terms)                     # nor the Gaussian processes (C03)
              or model.spec.autocor is not None                # an autocorrelation term (C04): nor this one
              or bool(model.spec.me_terms) or bool(model.spec.mi_terms)    # nor the me() and mi() terms (C06)
              or bool(model.spec.custom_group_terms))          # nor the mm() and gr(by = ) terms (D08)
    if dpar is not None:
        ds = _predict_dpar(model, idata, df, kind, dpar, include_group_effects=include_group_effects)
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
