"""brms's ``conditional_effects()`` with its defaults (brms-parity D06, docs/post-processing.md): the expected response
along one predictor with the others held fixed and the group-level effects left out. Accepted against brms 2.23 on the
frozen oracle cases ``gauss_hypothesis_effects`` and ``bernoulli_hypothesis_effects``; none of brms's source is used.

For each effect, a predictor of the population-level formula: the grid is ``resolution`` (100) evenly spaced values from the
fitted rows' minimum to their maximum for a numeric predictor (R's ``seq(min, max, length.out = 100)``) and the levels of a
categorical one, reference first. Every other numeric predictor is held at its mean over the fitted rows, every other
categorical predictor at its reference level (brms's first level), the response at its mean (brms writes it into the
grid, where it changes nothing), and the grouping factors are missing: brms's default ``re_formula = NA`` leaves the
group-level effects out. At each grid row brms's ``posterior_epred`` (``predict(kind="mean", include_group_effects=False)``)
is summarised over the draws as brms's default ``robust = TRUE`` summarises it (settled on the frozen cases): ``estimate__``
the median, ``se__`` the MAD (``1.4826 median |e - median e|``), ``lower__`` and ``upper__`` the ``(1 - prob) / 2`` and
``1 - (1 - prob) / 2`` quantiles (R's type 7).

Only what the frozen cases and the prediction gates verify is offered: models whose expected response is one value per row
(not the binomial family, whose grid needs trials, nor the cumulative, categorical or mixture families), without offsets,
``mo()``, ``gp()``, ``me()``, ``mi()``, autocorrelation, addition terms or distributional formulas; anything else is refused
naming the construct (divergence D99). brms's other arguments (``conditions``, ``int_conditions``, ``re_formula``, ``method``,
``robust = FALSE``, interaction effects such as ``"x:z"``, plots) are not offered (divergence D98).
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from autobayes.hypothesis import MAD_CONSTANT, refuse_unsupported

def _refuse_unsupported_effects(model) -> None:
    """The models ``conditional_effects()`` takes (module docstring): anything else is refused naming the construct."""
    refuse_unsupported(model, "conditional_effects()")
    fam, s = model.family_spec, model.spec
    if fam.has_trials:
        raise ValueError(f"conditional_effects() of a {model.family} model is not offered: its expected response needs the "
                         "trials of each grid row, which no frozen case checks (docs/post-processing.md, divergence D99)")
    constructs = [
        (s.offset is not None, lambda: f"the offset {s.offset.term!r}"),
        (bool(s.mo_terms), lambda: f"the monotonic term {s.mo_terms[0].name!r}"),
        (bool(s.gp_terms), lambda: f"the Gaussian process {s.gp_terms[0].term!r}"),
        (bool(getattr(s, "me_terms", ())), lambda: f"the measurement-error term {s.me_terms[0].name!r}"),
        (bool(getattr(s, "mi_terms", ())) or bool(getattr(s, "missing_columns", ())), lambda: "missing values modelled by mi()"),
        (s.autocor is not None, lambda: f"the autocorrelation term {s.autocor.term!r}"),
        (bool(s.dpars), lambda: f"a distributional formula for {', '.join(s.dpar_names)}"),
        (bool(s.additions), lambda: f"the addition term(s) {', '.join(a.term for a in s.additions)}"),
    ]
    for present, construct in constructs:
        if present:
            raise ValueError(f"conditional_effects() of a model with {construct()} is not offered: no frozen case checks how brms "
                             "holds it on the grid (docs/post-processing.md, divergence D99)")


def _mean(values) -> float:
    v = np.asarray(values, dtype=float)
    return math.fsum(v) / v.size


def _check_effects(model, effects) -> tuple[str, ...]:
    s = model.spec
    if isinstance(effects, str):
        effects = (effects,)
    if not isinstance(effects, Sequence) or not effects or not all(isinstance(e, str) for e in effects):
        raise ValueError("effects must be a predictor's name or a non-empty list of them, like ['x', 'z'] (brms's default of "
                         "every main effect and two-way interaction is not offered; divergence D98)")
    groups = {c for t in s.group_terms for c in t.operands}
    for e in effects:
        if ":" in e:
            raise ValueError(f"the interaction effect {e!r} is not offered: name one predictor per effect (divergence D98)")
        if e in groups and e not in s.pop_columns:
            raise ValueError(f"{e!r} is a grouping factor; conditional_effects() leaves the group-level effects out (brms's "
                             "re_formula = NA) and takes the population-level predictors only (divergence D98)")
        if e not in s.pop_columns:
            raise ValueError(f"{e!r} is not a predictor of the population-level formula; its predictors are {list(s.pop_columns)}")
    if len(set(effects)) != len(effects):
        raise ValueError(f"effects {list(effects)} name a predictor twice")
    return tuple(effects)


def _grid_value(model, column: str):
    """The value a predictor other than the effect is held at: a numeric predictor's mean over the fitted rows, a
    categorical one's reference level."""
    categoricals = {c.name: c for c in model.spec.categoricals}
    if column in categoricals:
        return categoricals[column].categories[0]
    return _mean(model.data[column])


def conditional_effects(model, fit_result, effects, *, resolution: int = 100, prob: float = 0.95) -> dict[str, pd.DataFrame]:
    """brms's ``conditional_effects(fit, effects)`` with its defaults (module docstring): for each effect, a predictor of
    the population-level formula, a ``pandas.DataFrame`` with brms's columns: the effect, the response and every other
    variable of the model (the grouping factors missing, ``None``) as held on the grid, then ``cond__`` (``"1"``),
    ``effect1__`` (the effect's grid values), ``estimate__``, ``se__``, ``lower__`` and ``upper__``. ``fit_result`` is a
    ``FitResult`` or its ``InferenceData``, checked against the model as ``predict()`` checks it. Returns a dict keyed by
    effect, in the order given."""
    from autobayes.predict import predict

    _refuse_unsupported_effects(model)
    effects = _check_effects(model, effects)
    if isinstance(resolution, bool) or not isinstance(resolution, (int, np.integer)) or resolution < 2:
        raise ValueError(f"resolution must be a whole number of grid points, at least 2, got {resolution!r}")
    if isinstance(prob, bool) or not isinstance(prob, (int, float, np.floating)) or not 0 < float(prob) < 1:
        raise ValueError(f"prob must be a number strictly between 0 and 1, got {prob!r}")
    s = model.spec
    categoricals = {c.name: c for c in s.categoricals}
    groups = [c for t in s.group_terms for c in t.operands if c not in s.pop_columns]
    variables = list(dict.fromkeys([s.response, *s.pop_columns, *groups]))
    held = {s.response: _mean(model.design.y)}
    held.update({c: _grid_value(model, c) for c in s.pop_columns})
    held.update({g: None for g in groups})
    probs = ((1.0 - prob) / 2.0, 1.0 - (1.0 - prob) / 2.0)
    out = {}
    for effect in effects:
        if effect in categoricals:
            grid = list(categoricals[effect].categories)
        else:
            values = np.asarray(model.data[effect], dtype=float)
            grid = np.linspace(values.min(), values.max(), int(resolution))
        n = len(grid)
        newdata = pd.DataFrame({c: (grid if c == effect else [held[c]] * n) for c in s.pop_columns})
        ds = predict(model, fit_result, newdata=newdata, kind="mean", include_group_effects=False)
        draws = ds[model.names["mean"]].transpose("chain", "draw", "__obs__").to_numpy().reshape(-1, n)
        estimate = np.median(draws, axis=0)
        se = MAD_CONSTANT * np.median(np.abs(draws - estimate), axis=0)
        lower, upper = np.quantile(draws, probs, axis=0)
        columns = {effect: grid}
        columns.update({c: [held[c]] * n for c in variables if c != effect})
        columns.update({"cond__": ["1"] * n, "effect1__": grid, "estimate__": estimate, "se__": se, "lower__": lower,
                        "upper__": upper})
        table = pd.DataFrame(columns)
        table.attrs["fit_binding_status"] = ds.attrs.get("fit_binding_status", "")
        out[effect] = table
    return out
