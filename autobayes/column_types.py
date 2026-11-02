"""Columns whose dtype declares their levels (D172, docs/column-types.md).

brms codes a factor predictor with treatment contrasts against its first level, an ordered factor with R's polynomial
contrasts and a logical as one column ``<name>TRUE`` with ``FALSE`` the reference. A pandas Categorical declares its levels
and their order as an R factor does, and a boolean column declares its two values, so such a population-level predictor needs no
``categorical=`` entry: ``typed_categorical`` adds one for every numeric main effect of the formula whose column is an undeclared
pandas Categorical or boolean column, the ``DtypeLevels`` marker, and ``dtype_coding`` codes the column once the data are seen:
the fitted levels in the dtype's order (the Categorical's categories; ``FALSE``, ``TRUE``), the first the reference; an ordered
Categorical takes R's polynomial contrasts as ``ordered()`` does (F08, D168). A declared boolean column (``categorical={"lg":
False}``) is read as brms reads a logical too, its labels ``FALSE`` and ``TRUE`` (``logical_labels``), against the declared
reference. Text and numeric columns still need the declaration (their levels would depend on sorting or guessing).

Who calls it: ``Model`` and ``MultivariateModel`` before they parse the formula, and ``default_priors`` (and its multivariate
form) the same way, so that a model and its default priors read the columns alike. The non-linear and count-matrix models do not:
there such a column is refused as before, naming the declaration to write.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

LOGICAL_LEVELS = ("FALSE", "TRUE")     # brms's labels of a logical predictor, FALSE the reference (its column <name>TRUE)

# describe()'s wording of the level order of a treatment-coded factor
LEVEL_ORDER = {"": "sorted", "category": "the pandas Categorical's order", "bool": "a boolean column, as brms reads a logical"}


@dataclass(frozen=True)
class DtypeLevels:
    """The ``categorical=`` entry ``typed_categorical`` writes for an undeclared column whose dtype declares its levels: ``kind``
    is ``"category"`` (a pandas Categorical) or ``"bool"``. The reference and the level order come from the column once the data
    are seen (``formula.Categorical.from_dtype``, ``dtype_coding``)."""

    kind: str


def logical_labels(label: str, s: pd.Series) -> np.ndarray:
    """A boolean column's labels as brms reads a logical predictor: ``"FALSE"`` and ``"TRUE"``; a missing value is refused."""
    from autobayes.formula import FormulaError

    values = s.astype(object).to_numpy()
    if pd.isna(values).any():
        raise FormulaError(f"{label} has missing values; drop or impute them explicitly before fitting.")
    return np.array([LOGICAL_LEVELS[int(bool(v))] for v in values], dtype=object)


def declared_reference(name: str, ref):
    """The reference a declaration gives a boolean column, as its label: ``False`` or ``"FALSE"`` is ``"FALSE"``, ``True`` or
    ``"TRUE"`` is ``"TRUE"``; anything else is refused (``None`` is left for the declaration's own refusal)."""
    from autobayes.formula import FormulaError

    if ref is None or isinstance(ref, DtypeLevels):
        return ref
    if isinstance(ref, (bool, np.bool_)):
        return LOGICAL_LEVELS[int(bool(ref))]
    if isinstance(ref, str) and ref in LOGICAL_LEVELS:
        return ref
    raise FormulaError(f"categorical predictor {name!r} is a boolean column, whose levels are FALSE and TRUE as brms reads a logical: "
                       f"its reference is False or True, got {ref!r} (docs/column-types.md)")


def typed_categorical(data, categorical, parse):
    """``categorical`` with the declarations the data's dtypes make (module docstring): each numeric main effect of the formula
    and each variable of an interaction (``parse(categorical)``, the formula parsed without data: a ``ModelSpec``, its
    distributional formulas' too, or a ``MultivariateSpec``, every response's) whose column is a pandas Categorical or a boolean
    column gets a ``DtypeLevels``
    entry, and a declared boolean column its reference as a label (``declared_reference``). ``categorical`` itself when nothing
    changes, or when ``data`` is not a data frame or ``categorical`` not a mapping (the constructors refuse those)."""
    if not isinstance(data, pd.DataFrame) or not (categorical is None or isinstance(categorical, Mapping)):
        return categorical
    given = dict(categorical or {})
    declared = {k: declared_reference(k, v) if k in data.columns and pd.api.types.is_bool_dtype(data[k]) else v
                for k, v in given.items()}
    spec = parse(declared or None)
    specs = list(spec.specs.values()) if getattr(spec, "specs", None) is not None else [spec]
    typed = {}
    # the numeric main effects, and the variables of an interaction without their main effect (coded by R's rule, term_algebra)
    names = [p for s in specs for f in (s, *s.dpars) for p in f.predictors]
    names += [v for s in specs for f in (s, *s.dpars) for t in f.pop_terms if t.kind == "product" for v in t.product.operands]
    for name in dict.fromkeys(names):
        column = data[name] if name in data.columns else None
        if column is not None and pd.api.types.is_bool_dtype(column):
            typed[name] = DtypeLevels("bool")
        elif column is not None and isinstance(column.dtype, pd.CategoricalDtype):
            typed[name] = DtypeLevels("category")
    if not typed and declared == given:
        return categorical
    return {**declared, **typed}


def dtype_coding(cat, s: pd.Series, labels: np.ndarray, unused: tuple[str, ...]):
    """The ``Categorical`` of an undeclared typed column (``cat.from_dtype``) once the data are seen: its fitted levels in the dtype's
    order, the Categorical's categories or ``FALSE``, ``TRUE``, the first the reference (brms drops a level no fitted row takes, so
    the first level that occurs is the reference); an ordered Categorical is coded with R's polynomial contrasts over them (F08,
    ``formula._poly_categorical``), unless it is the full-rank factor of a formula without an intercept, which R codes with one
    indicator per level."""
    from autobayes.formula import Categorical, _poly_categorical, _scalar

    observed = set(labels.tolist())
    order = LOGICAL_LEVELS if cat.from_dtype == "bool" else tuple(
        _scalar(f"categorical predictor {cat.name!r}", c, where="declared category") for c in s.cat.categories)
    levels = tuple(level for level in order if level in observed)
    if cat.from_dtype == "category" and s.dtype.ordered and not cat.full_rank:
        return _poly_categorical(replace(cat, reference=levels[0]), s, labels, unused)
    return Categorical(name=cat.name, reference=levels[0], levels=levels, unused=unused, full_rank=cat.full_rank,
                       from_dtype=cat.from_dtype)
