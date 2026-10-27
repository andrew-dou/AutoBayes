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

An interaction effect ``"a:b"`` (E05; frozen cases ``gauss_hypothesis_prior_effects`` and ``bernoulli_hypothesis_prior_effects``,
grids probed in ``tests/hypothesis_probes/brms-2.23.0-effects-grid.json``) is two distinct population-level predictors, with
brms's default ``int_conditions``: ``a`` on its grid as above, ``b`` at each of its levels (a categorical moderator) or at
its mean minus its SD, its mean and its mean plus its SD (a numeric one, the SD with divisor ``n - 1``); one row per pair,
``b`` varying fastest. The table's columns are ``a``, ``b``, the response and the other variables as held, then ``cond__``,
``effect1__`` (``a``), ``effect2__`` (``b``'s level, or the numeric value as brms labels it: R's
``as.character(round(value, 2))``, so ``"1.6"``, ``"0"``, ``"1e+05"``; ``b``'s column keeps the exact value) and the
summaries. brms accepts any pair of the model's variables, whether or not the formula has their interaction.

Only what the frozen cases and the prediction gates verify is offered: models whose expected response is one value per row
(not the binomial family, whose grid needs trials, nor the cumulative, categorical or mixture families), without offsets,
``mo()``, ``gp()``, ``me()``, ``mi()``, autocorrelation, ``fcor()``, addition terms or distributional formulas; anything else is refused
naming the construct (divergence D99). brms's other arguments (``conditions``, ``int_conditions``, ``re_formula``, ``method``,
``robust = FALSE``, interactions of three predictors, surface plots, plots) are not offered (divergence D98).
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
    if "mean" not in fam.prediction_kinds:                     # F05: a family without expected values (cox, divergence D154)
        raise ValueError(f"conditional_effects() of a {model.family} model is not offered: it needs the expected response; "
                         f"{fam.kind_error}")
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
        (getattr(s, "sar", None) is not None, lambda: f"the SAR term {s.sar.term!r}"),                          # F02b (D143)
        (getattr(s, "fcor", None) is not None, lambda: f"the fixed residual covariance {s.fcor.term!r}"),   # F03a
        (getattr(s, "car", None) is not None, lambda: f"the CAR term {s.car.term!r}"),      # F02a
        (getattr(s, "timecor", None) is not None, lambda: f"the residual correlation term {s.timecor.term!r}"),   # F03b
        (bool(s.dpars), lambda: f"a distributional formula for {', '.join(s.dpar_names)}"),
        (bool(s.additions), lambda: f"the addition term(s) {', '.join(a.term for a in s.additions)}"),
        # G04: a factor or logical made in the formula, whose model-frame column brms holds as a factor (divergence D182)
        (any(t.transform.form in ("factor", "logical") for t in getattr(s, "expression_terms", ())),
         lambda: f"the term {next(t.basis for t in s.expression_terms if t.transform.form in ('factor', 'logical'))!r}"),
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
        raise ValueError("effects must be a predictor's name, an interaction of two ('x:z'), or a non-empty list of them, like "
                         "['x', 'x:z'] (brms's default of every main effect and two-way interaction is not offered; divergence D98)")
    groups = {c for t in s.group_terms for c in t.operands}
    for e in effects:
        parts = e.split(":")
        if len(parts) > 2:
            raise ValueError(f"the effect {e!r} interacts {len(parts)} predictors; brms's conditional_effects() takes one or two "
                             "(divergence D98)")
        for v in parts:
            if v in groups and v not in s.pop_columns:
                raise ValueError(f"{v!r} is a grouping factor; conditional_effects() leaves the group-level effects out (brms's "
                                 "re_formula = NA) and takes the population-level predictors only (divergence D98)")
            if v not in s.pop_columns:
                raise ValueError(f"{v!r} is not a predictor of the population-level formula; its predictors are {list(s.pop_columns)}")
        if len(parts) == 2 and parts[0] == parts[1]:
            raise ValueError(f"the interaction effect {e!r} names {parts[0]!r} twice")
    keys = [frozenset(e.split(":")) for e in effects]
    repeated = [k for i, k in enumerate(keys) if k in keys[:i]]
    if repeated:
        if len(repeated[0]) == 1:
            raise ValueError(f"effects {list(effects)} name a predictor twice")
        raise ValueError(f"effects {list(effects)} name a pair of predictors twice (brms drops the repeat with a warning; "
                         "divergence D98)")
    return tuple(effects)


def r_round(value: float, digits: int = 2) -> float:
    """R's ``round(value, digits)`` as its outputs show it (``tests/hypothesis_probes/brms-2.23.0-effects-grid.json``, 3313
    values, ties and near-ties among them): of the two candidates ``floor(value 10^d) / 10^d`` and ``ceil(value 10^d) /
    10^d``, the one closer to ``value`` in double arithmetic, the even one when they are equally close (so ``round(0.005, 2)``
    is 0, where the correctly rounded value is 0.01)."""
    p = 10.0 ** digits
    lo_i, hi_i = math.floor(value * p), math.ceil(value * p)
    lo, hi = lo_i / p, hi_i / p
    if lo_i == hi_i or value - lo < hi - value:
        return lo
    if hi - value < value - lo:
        return hi
    return lo if lo_i % 2 == 0 else hi


def r_number_label(value: float) -> str:
    """R's ``as.character(round(value, 2))``: the value rounded to two decimals as R rounds (``r_round``), written with at
    most 15 significant digits, in fixed notation unless the scientific one is shorter (R's rule with ``scipen = 0``:
    ``1e+05``, ``123456.79``), and negative zero as ``0``."""
    r = r_round(float(value), 2)
    if r == 0:
        return "0"
    digits = next(k for k in range(1, 16) if float(f"{r:.{k - 1}e}") == r) if float(f"{r:.14e}") == r else 15
    mantissa, exponent = f"{r:.{digits - 1}e}".split("e")
    exponent = int(exponent)
    scientific = f"{mantissa}e{'-' if exponent < 0 else '+'}{abs(exponent):02d}"
    decimals = max(0, digits - 1 - exponent)
    fixed = f"{r:.{decimals}f}"
    return fixed if len(fixed) <= len(scientific) else scientific


def _moderator(model, column: str):
    """brms's default ``int_conditions`` of a moderator: its levels (categorical), or its mean minus its SD, its mean and
    its mean plus its SD (numeric, the SD with divisor ``n - 1``), with brms's labels for ``effect2__``."""
    categoricals = {c.name: c for c in model.spec.categoricals}
    if column in categoricals:
        levels = list(categoricals[column].categories)
        return levels, [str(v) for v in levels]
    values = np.asarray(model.data[column], dtype=float)
    mean, sd = _mean(values), float(np.std(values, ddof=1))
    grid = [mean - sd, mean, mean + sd]
    return grid, [r_number_label(v) for v in grid]


def _frame_columns(spec) -> dict[str, object]:
    """G04: the model-frame column of each in-formula expression term (``scale(x)``, ``poly(w, 2)``, ``log(xpos + 1)``), keyed by R's
    name of it (the term as R deparses it), with its first design column's term, which evaluates it on rows of the raw columns:
    brms's ``conditional_effects()`` tables carry these columns, held at the transform of the held values
    (``gauss_scale_poly_post``). A05's transforms (``log(xpos)``, ``I(x^2)``) keep D06's tables, without the column."""
    from autobayes import expressions

    out = {}
    for t in getattr(spec, "expression_terms", ()):
        out.setdefault(expressions.deparse(expressions.parse(t.basis), r_numbers=True), t)
    return out


def _variables(spec, groups) -> list[str]:
    """The columns of a ``conditional_effects()`` table between the effect and ``cond__``, in brms's order: the response, each
    population-level term's model-frame column in formula order (a column, or a transform's own, ``scale(x)``), the raw columns
    the terms read, then the grouping factors."""
    frame = {t.name: name for name, t in _frame_columns(spec).items()}     # the first column of each transform term
    terms = []
    for t in spec.pop_terms:
        if t.kind == "transform":
            terms += [frame[t.name]] if t.name in frame else []
        else:
            terms += [t.name] if t.kind in ("numeric", "categorical") else list(t.operands)
    return list(dict.fromkeys([spec.response, *terms, *spec.pop_columns, *groups]))


def _frame_value(t, held: dict):
    """A transform term's model-frame value at the held raw values: a number, or an array (``scale()``, ``poly()``: R's matrix)."""
    row = pd.DataFrame({c: [held[c]] for c in t.operands})
    value = t.transform.frame_value(row)[0]
    return np.array(value, dtype=float) if np.ndim(value) else float(value)


def _grid_value(model, column: str):
    """The value a predictor other than the effect is held at: a numeric predictor's mean over the fitted rows, a
    categorical one's reference level."""
    categoricals = {c.name: c for c in model.spec.categoricals}
    if column in categoricals:
        return categoricals[column].categories[0]
    return _mean(model.data[column])


def conditional_effects(model, fit_result, effects, *, resolution: int = 100, prob: float = 0.95) -> dict[str, pd.DataFrame]:
    """brms's ``conditional_effects(fit, effects)`` with its defaults (module docstring): for each effect, a predictor of
    the population-level formula or an interaction of two (``"x:z"``, E05), a ``pandas.DataFrame`` with brms's columns: the
    effect's predictors, the response and every other variable of the model (the grouping factors missing, ``None``) as held
    on the grid, then ``cond__`` (``"1"``), ``effect1__`` (the effect's grid values), for an interaction ``effect2__`` (the
    moderator's levels or labelled values), ``estimate__``, ``se__``, ``lower__`` and ``upper__``. ``fit_result`` is a
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
    variables = _variables(s, groups)
    held = {s.response: _mean(model.design.y)}
    held.update({c: _grid_value(model, c) for c in s.pop_columns})
    held.update({name: _frame_value(t, held) for name, t in _frame_columns(s).items()})   # G04: held at the held raw values
    held.update({g: None for g in groups})
    probs = ((1.0 - prob) / 2.0, 1.0 - (1.0 - prob) / 2.0)
    out = {}
    for effect in effects:
        first, *second = effect.split(":")
        if first in categoricals:
            grid = list(categoricals[first].categories)
        else:
            values = np.asarray(model.data[first], dtype=float)
            grid = list(np.linspace(values.min(), values.max(), int(resolution)))
        if second:                                                   # E05: an interaction, the moderator fastest
            moderator, labels = _moderator(model, second[0])
            grid, mod_values, mod_labels = ([v for v in grid for _ in moderator], moderator * len(grid), labels * len(grid))
        n = len(grid)
        varying = {first: grid, **({second[0]: mod_values} if second else {})}
        newdata = pd.DataFrame({c: (varying[c] if c in varying else [held[c]] * n) for c in s.pop_columns})
        ds = predict(model, fit_result, newdata=newdata, kind="mean", include_group_effects=False)
        draws = ds[model.names["mean"]].transpose("chain", "draw", "__obs__").to_numpy().reshape(-1, n)
        estimate = np.median(draws, axis=0)
        se = MAD_CONSTANT * np.median(np.abs(draws - estimate), axis=0)
        lower, upper = np.quantile(draws, probs, axis=0)
        grid_column = grid if first in categoricals else np.asarray(grid, dtype=float)
        columns = {first: grid_column, **({second[0]: mod_values} if second else {})}
        columns.update({c: [held[c]] * n for c in variables if c not in varying})
        columns.update({"cond__": ["1"] * n, "effect1__": grid_column})
        if second:
            columns["effect2__"] = mod_labels
        columns.update({"estimate__": estimate, "se__": se, "lower__": lower, "upper__": upper})
        table = pd.DataFrame(columns)
        table.attrs["fit_binding_status"] = ds.attrs.get("fit_binding_status", "")
        out[effect] = table
    return out
