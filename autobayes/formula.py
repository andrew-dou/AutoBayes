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


def parse_formula(formula: str, data: pd.DataFrame | None = None, family: str = "gaussian") -> ModelSpec:
    """Parse and validate a formula in the supported subset, optionally against a data frame."""
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
        spec = validate_data(spec, data, family)
    return spec


def reject_complex(label: str, s: pd.Series) -> None:
    """Refuse a complex-valued column before anything casts it (a cast would silently drop
    the imaginary part and pandas counts complex as numeric)."""
    if pd.api.types.is_complex_dtype(s):
        raise FormulaError(f"{label} is complex-valued ({s.dtype}); only real numeric columns are supported.")


def _is_numeric(s: pd.Series) -> bool:
    return (pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
            and not pd.api.types.is_complex_dtype(s))


def _validate_binary_response(name: str, s: pd.Series) -> None:
    reject_complex(f"response {name!r}", s)
    if pd.api.types.is_bool_dtype(s):
        raise FormulaError(
            f"response {name!r} is boolean; code it as integers 0/1 explicitly "
            f"(data[{name!r}].astype(int)) so that 1 is unambiguously the modelled outcome."
        )
    if not pd.api.types.is_numeric_dtype(s):
        raise FormulaError(
            f"response {name!r} has dtype {s.dtype}; the Bernoulli family needs a numeric 0/1 "
            "column with 1 the modelled outcome. Recode labels explicitly rather than relying on "
            "the sorted first level."
        )
    vals = set(np.unique(s.dropna().to_numpy(dtype=float)))
    if not vals <= {0.0, 1.0}:
        raise FormulaError(
            f"response {name!r} takes values outside {{0, 1}}: {sorted(vals - {0.0, 1.0})[:5]}. "
            "The Bernoulli family needs exactly 0/1 (binomial trials are not supported in M1)."
        )
    if vals != {0.0, 1.0}:
        raise FormulaError(
            f"response {name!r} is constant ({sorted(vals)}); a Bernoulli model needs both outcomes."
        )


def validate_data(spec: ModelSpec, data: pd.DataFrame, family: str = "gaussian") -> ModelSpec:
    """Check columns exist, are numeric where required, and have no missing values."""
    needed = [spec.response, *spec.predictors, spec.group]
    missing = [c for c in needed if c not in data.columns]
    if missing:
        raise FormulaError(f"columns not in data: {missing}")
    for c in spec.predictors:
        reject_complex(f"column {c!r}", data[c])
        if not _is_numeric(data[c]):
            raise FormulaError(
                f"column {c!r} has dtype {data[c].dtype}; only numeric predictors are "
                "supported (categorical predictors are not supported)."
            )
    reject_complex(f"response {spec.response!r}", data[spec.response])
    if family == "bernoulli":
        _validate_binary_response(spec.response, data[spec.response])
    elif not _is_numeric(data[spec.response]):
        raise FormulaError(
            f"column {spec.response!r} has dtype {data[spec.response].dtype}; the Gaussian "
            "family needs a numeric response."
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
