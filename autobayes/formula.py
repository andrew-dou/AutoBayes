"""The M0 formula grammar and its validation.

Supported::

    response ~ [1 +] predictor [+ predictor ...] + (1 | group)

Predictors must be numeric columns; the grouping variable may be any column and is
coded as a factor with levels in sorted order. Bambi (via formulae) accepts a much
larger language; this module exists so that a formula outside the verified subset is
refused with an error naming the construct, rather than silently built.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_IDENT = r"[A-Za-z_][A-Za-z0-9_.]*"
_FORMULA_RE = re.compile(rf"^\s*({_IDENT})\s*~\s*(.+?)\s*$", re.S)
_GROUP_RE = re.compile(rf"^\(\s*(.+?)\s*\|\s*(.+?)\s*\)$", re.S)


class FormulaError(ValueError):
    """A formula outside the M0 subset, or one that does not match the data."""


@dataclass(frozen=True)
class ModelSpec:
    formula: str
    response: str
    predictors: tuple[str, ...]
    group: str
    intercept: bool = True
    levels: tuple[str, ...] = field(default_factory=tuple)

    @property
    def group_term(self) -> str:
        return f"1|{self.group}"


def _split_top_level(rhs: str, sep: str = "+") -> list[str]:
    """Split on ``sep`` outside parentheses."""
    parts, depth, buf = [], 0, []
    for ch in rhs:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                raise FormulaError(f"unbalanced parentheses in {rhs!r}")
        if ch == sep and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if depth != 0:
        raise FormulaError(f"unbalanced parentheses in {rhs!r}")
    parts.append("".join(buf).strip())
    return parts


def _reject_group_term(inner: str, factor: str, piece: str) -> None:
    if "||" in piece:
        raise FormulaError(
            f"{piece!r}: the '||' (uncorrelated varying effects) syntax is not supported "
            "in M0; only a single varying intercept '(1 | group)' is."
        )
    if "/" in factor or ":" in factor:
        raise FormulaError(
            f"{piece!r}: nested or crossed grouping factors ('g/h', 'g:h') are not "
            "supported in M0; only a single grouping factor '(1 | group)' is."
        )
    if not re.fullmatch(_IDENT, factor):
        raise FormulaError(f"{piece!r}: grouping factor must be a single column name")
    if inner != "1":
        raise FormulaError(
            f"{piece!r}: varying slopes are not supported in M0; only a varying "
            f"intercept '(1 | {factor})' is."
        )


def parse_formula(formula: str, data: pd.DataFrame | None = None) -> ModelSpec:
    """Parse and validate an M0 formula, optionally against a data frame."""
    m = _FORMULA_RE.match(formula)
    if not m:
        raise FormulaError(
            f"{formula!r}: expected 'response ~ terms'. M0 supports "
            "'y ~ [1 +] x1 [+ x2 ...] + (1 | group)'."
        )
    response, rhs = m.group(1), m.group(2)
    if "-" in rhs:
        raise FormulaError(
            f"{formula!r}: '-' is not supported in M0 (this includes '- 1' to drop the "
            "intercept, which is required)."
        )

    predictors: list[str] = []
    groups: list[str] = []
    saw_intercept_token = False
    for piece in _split_top_level(rhs):
        if piece == "":
            raise FormulaError(f"{formula!r}: empty term")
        if piece == "1":
            saw_intercept_token = True
            continue
        if piece == "0":
            raise FormulaError(
                f"{formula!r}: '0 +' (no intercept) is not supported in M0; the model "
                "requires a population-level intercept."
            )
        gm = _GROUP_RE.match(piece)
        if gm:
            inner, factor = gm.group(1).strip(), gm.group(2).strip()
            _reject_group_term(inner, factor, piece)
            groups.append(factor)
            continue
        if piece.startswith("("):
            raise FormulaError(f"{piece!r}: parenthesised term is not a '(1 | group)' term")
        if "|" in piece:
            raise FormulaError(f"{piece!r}: a '|' term must be written '(1 | group)'")
        if "*" in piece or ":" in piece:
            raise FormulaError(
                f"{piece!r}: interactions ('*', ':') are not supported in M0."
            )
        if "(" in piece or ")" in piece:
            raise FormulaError(
                f"{piece!r}: function calls and transformations inside the formula are "
                "not supported in M0; transform the column in the data frame instead."
            )
        if not re.fullmatch(_IDENT, piece):
            raise FormulaError(f"{piece!r}: not a column name")
        if piece == response:
            raise FormulaError(f"{piece!r}: the response cannot also be a predictor")
        if piece in predictors:
            raise FormulaError(f"{piece!r}: duplicated predictor")
        predictors.append(piece)

    if len(groups) == 0:
        raise FormulaError(
            f"{formula!r}: M0 requires exactly one varying-intercept term '(1 | group)'."
        )
    if len(groups) > 1:
        raise FormulaError(
            f"{formula!r}: more than one grouping factor ({groups}) is not supported in "
            "M0; use a single '(1 | group)' term."
        )
    if len(predictors) == 0 and not saw_intercept_token:
        raise FormulaError(
            f"{formula!r}: no population-level predictor; write 'y ~ 1 + (1 | group)' "
            "for an intercept-only model."
        )
    group = groups[0]
    if group in predictors or group == response:
        raise FormulaError(f"{group!r}: the grouping factor cannot also be a predictor or the response")

    spec = ModelSpec(
        formula=formula.strip(), response=response, predictors=tuple(predictors),
        group=group, intercept=True,
    )
    if data is not None:
        spec = validate_data(spec, data)
    return spec


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def validate_data(spec: ModelSpec, data: pd.DataFrame) -> ModelSpec:
    """Check columns exist, are numeric where required, and have no missing values."""
    needed = [spec.response, *spec.predictors, spec.group]
    missing = [c for c in needed if c not in data.columns]
    if missing:
        raise FormulaError(f"columns not in data: {missing}")
    for c in (spec.response, *spec.predictors):
        if not _is_numeric(data[c]):
            raise FormulaError(
                f"column {c!r} has dtype {data[c].dtype}; M0 supports numeric predictors "
                "and a numeric response only (categorical predictors are not supported)."
            )
    na = data[needed].isna().any()
    if na.any():
        raise FormulaError(
            f"missing values in columns {list(na[na].index)}; M0 does not handle missing "
            "data. Drop or impute them explicitly before fitting."
        )
    if len(data) < 2:
        raise FormulaError("need at least two observations")
    levels = tuple(str(v) for v in sorted(data[spec.group].astype(str).unique()))
    if len(levels) < 2:
        raise FormulaError(
            f"grouping factor {spec.group!r} has a single level; a varying intercept "
            "needs at least two."
        )
    for c in spec.predictors:
        if np.isclose(np.std(data[c].to_numpy(dtype=float)), 0.0):
            raise FormulaError(f"predictor {c!r} is constant")
    return ModelSpec(
        formula=spec.formula, response=spec.response, predictors=spec.predictors,
        group=spec.group, intercept=True, levels=levels,
    )
