"""The formula grammar and its validation.

Supported::

    response ~ [1 +] predictor [+ predictor ...] [+ x:z ...] + (1 | group)               (Gaussian, Bernoulli)
    successes | trials(n) ~ [1 +] predictor [+ predictor ...] [+ x:z ...] + (1 | group)  (binomial, brms syntax)

with ``x:z`` (or the shorthand ``x*z`` = ``x + z + x:z``) one interaction between a numeric term
and a declared categorical term whose main effects are both present (milestone M3a-2).

Predictors are numeric columns, or categorical columns declared with ``categorical={column:
reference_level}`` (treatment coding, milestone M3a-1); the grouping variable may be any
column and is coded as a factor with levels in sorted order. Bambi (via formulae) accepts a
much larger language; this module exists so that a formula outside the verified subset is
refused with an error naming the construct, rather than silently built.
"""
from __future__ import annotations

import re
import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_IDENT = r"[A-Za-z_][A-Za-z0-9_.]*"
_FORMULA_RE = re.compile(rf"^\s*({_IDENT})\s*~\s*(.+?)\s*$", re.S)
_TRIALS_FORMULA_RE = re.compile(rf"^\s*({_IDENT})\s*\|\s*trials\(\s*({_IDENT})\s*\)\s*~\s*(.+?)\s*$", re.S)
_BAMBI_PROPORTION_RE = re.compile(r"^\s*(p|prop|proportion)\s*\(", re.S)
_GROUP_RE = re.compile(rf"^\(\s*(.+?)\s*\|\s*(.+?)\s*\)$", re.S)

COUNT_CAP = 10**6   # the validated domain for binomial counts per row (docs/m2-decision.md, repair record)


class FormulaError(ValueError):
    """A formula outside the M0 subset, or one that does not match the data."""


@dataclass(frozen=True)
class Categorical:
    """One treatment-coded categorical predictor (docs/m3a1-decision.md).

    ``levels`` are the distinct string labels present in the fitted rows in ``sorted`` order
    (empty until the data are seen); ``reference`` is the declared reference label, which
    must be one of them; ``coefficients`` are the levels without the reference, in order,
    one dummy column and one coefficient each; ``unused`` lists a pandas Categorical's
    declared categories that do not occur in the fitted rows.
    """

    name: str
    reference: str
    levels: tuple[str, ...] = ()
    unused: tuple[str, ...] = ()

    @property
    def coefficients(self) -> tuple[str, ...]:
        return tuple(l for l in self.levels if l != self.reference)

    @property
    def categories(self) -> tuple[str, ...]:
        """Reference first, then the coefficient levels: the ordered Categorical handed to formulae."""
        return (self.reference, *self.coefficients)

    @property
    def columns(self) -> tuple[str, ...]:
        """Design-column labels, Bambi/formulae's own: ``z[level]`` per coefficient level."""
        return tuple(f"{self.name}[{l}]" for l in self.coefficients)


@dataclass(frozen=True)
class ModelSpec:
    formula: str
    response: str
    predictors: tuple[str, ...]        # numeric predictor terms, in formula order
    group: str
    intercept: bool = True
    levels: tuple[str, ...] = field(default_factory=tuple)
    trials: str | None = None          # binomial only: the trials column of ``successes | trials(n)``
    rhs: str = ""                      # the CANONICAL right-hand side handed to Bambi (docs/m3a2-decision.md)
    terms: tuple[str, ...] = ()        # every term in canonical order: main effects, then interactions ``x:z``
    categoricals: tuple[Categorical, ...] = ()   # the declared categorical terms, in formula order
    interactions: tuple[tuple[str, str], ...] = ()   # (numeric term, categorical term) per interaction, canonical order

    def __post_init__(self):
        if not self.terms:                      # numeric-only construction (M0-M2 call sites)
            object.__setattr__(self, "terms", tuple(self.predictors))

    @property
    def group_term(self) -> str:
        return f"1|{self.group}"

    @property
    def categorical_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.categoricals)

    def categorical(self, name: str) -> Categorical:
        for c in self.categoricals:
            if c.name == name:
                return c
        raise KeyError(name)

    @property
    def interaction_names(self) -> tuple[str, ...]:
        return tuple(f"{x}:{z}" for x, z in self.interactions)

    def interaction_of(self, term: str) -> tuple[str, str]:
        for x, z in self.interactions:
            if f"{x}:{z}" == term:
                return x, z
        raise KeyError(term)

    def is_vector_term(self, term: str) -> bool:
        """A categorical or interaction term: one coefficient per non-reference level."""
        return term in self.categorical_names or term in self.interaction_names

    def levels_of(self, term: str) -> tuple[str, ...]:
        """Coefficient levels of a vector-valued term: a categorical term's non-reference levels,
        or those of an interaction's categorical side. ``()`` for a numeric term."""
        if term in self.categorical_names:
            return self.categorical(term).coefficients
        if term in self.interaction_names:
            return self.categorical(self.interaction_of(term)[1]).coefficients
        return ()

    def term_columns(self, term: str) -> tuple[str, ...]:
        """Design-column labels of one term, Bambi/formulae's own: ``x``; ``z[l]``; ``x:z[l]``."""
        levels = self.levels_of(term)
        return tuple(f"{term}[{l}]" for l in levels) if levels else (term,)

    @property
    def columns(self) -> tuple[str, ...]:
        """Design-column labels in design order: canonical term order, a categorical term
        expanded into ``z[level]`` dummies (reference omitted), an interaction into ``x:z[level]``."""
        return tuple(col for term in self.terms for col in self.term_columns(term))

    @property
    def column_terms(self) -> tuple[tuple[str, str, int | None], ...]:
        """``(column label, term, index within the term or None for a numeric term)`` per column."""
        out: list[tuple[str, str, int | None]] = []
        for term in self.terms:
            cols = self.term_columns(term)
            if self.levels_of(term):
                out.extend((col, term, j) for j, col in enumerate(cols))
            else:
                out.append((term, term, None))
        return tuple(out)

    @property
    def canonical_formula(self) -> str:
        """The formula as expanded: main effects, then interactions spelled ``numeric:categorical``."""
        lhs = self.response if self.trials is None else f"{self.response} | trials({self.trials})"
        return f"{lhs} ~ {self.rhs}"

    @property
    def bambi_formula(self) -> str:
        """What is handed to Bambi: the canonical right-hand side; brms's ``k | trials(n)`` becomes
        Bambi's ``p(k, n)``."""
        if self.trials is None:
            return f"{self.response} ~ {self.rhs}"
        return f"p({self.response}, {self.trials}) ~ {self.rhs}"


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


def _normalise_categorical(categorical, terms: list[str], response: str, group: str, trials: str | None,
                           formula: str) -> tuple[Categorical, ...]:
    if categorical is None:
        return ()
    if not isinstance(categorical, Mapping):
        raise FormulaError(
            "categorical must be a mapping {column: reference_level}, for example categorical={'z': 'a'}"
        )
    out: list[Categorical] = []
    for name, ref in categorical.items():
        if not isinstance(name, str):
            raise FormulaError(f"categorical keys must be column names, got {name!r}")
        if name == response:
            raise FormulaError(f"{name!r} is the response; only predictor terms can be declared categorical")
        if trials is not None and name == trials:
            raise FormulaError(f"{name!r} is the trials column; only predictor terms can be declared categorical")
        if name == group:
            raise FormulaError(
                f"{name!r} is the grouping factor of '(1 | {group})', which is already a factor with a "
                "varying intercept; only fixed-effect predictor terms are declared categorical"
            )
        if name not in terms:
            raise FormulaError(
                f"{name!r} is declared categorical but is not a term of {formula!r}; add it to the formula"
            )
        if ref is None:
            raise FormulaError(
                f"categorical predictor {name!r} needs an explicit reference level: categorical={{{name!r}: <level>}}"
            )
        out.append(Categorical(name=name, reference=_scalar(f"categorical predictor {name!r}", ref, where="reference level")))
    return tuple(c for t in terms for c in out if c.name == t)          # formula order


_FLOAT_EXACT = 2 ** 53          # float64 represents every integer up to here exactly


def _label(v) -> str:
    """The string label of one categorical value (docs/m3a1-decision.md, repair record).

    Integers of any width and sign map to their exact decimal string; floats only when finite,
    integral and at most 2**53 in magnitude; strings are themselves. Booleans, complex numbers,
    bytes, missing values and every other type are refused. No float intermediate anywhere."""
    if isinstance(v, (bool, np.bool_)):
        raise FormulaError("boolean values are not accepted as categorical labels; recode them explicitly")
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            raise FormulaError(f"{v!r} is not a finite categorical value")
        if float(v) != math.floor(float(v)):
            raise FormulaError(f"categorical values must be integral or labels, got {v!r}")
        if abs(float(v)) > _FLOAT_EXACT:
            raise FormulaError(
                f"float categorical value {v!r} exceeds 2**53 in magnitude, beyond which float64 does not represent "
                "integers exactly; store large codes as integers (int64/uint64/object) or as strings"
            )
        return str(int(v))
    if isinstance(v, str):
        return str(v)
    if isinstance(v, (complex, np.complexfloating)):
        raise FormulaError(f"complex value {v!r} is not accepted as a categorical label")
    if v is None or v is pd.NA or v is pd.NaT:
        raise FormulaError("missing value is not accepted as a categorical label")
    raise FormulaError(f"unsupported categorical value type {type(v).__name__} ({v!r}); use integers or strings")


def _kind(v) -> str:
    return "text" if isinstance(v, str) else "number"


def categorical_labels(label: str, s: pd.Series) -> tuple[np.ndarray, tuple[str, ...]]:
    """String labels of a categorical predictor column, checked scalar by scalar before any
    coding (``_label``), and the column's declared-but-absent categories (pandas Categorical
    only). Containers are read as Python objects, which is exact for every integer width and
    for nullable types; a column mixing text labels with numeric codes is refused as ambiguous."""
    reject_complex(label, s)
    if pd.api.types.is_bool_dtype(s):
        raise FormulaError(
            f"{label} is boolean; a categorical predictor needs explicit labels or integer codes "
            "(recode it, for example data[col].map({False: 'no', True: 'yes'}))"
        )
    declared: tuple[str, ...] = ()
    if isinstance(s.dtype, pd.CategoricalDtype):
        declared = tuple(_scalar(label, c, where="declared category") for c in s.cat.categories)
        if len(set(declared)) != len(declared):
            raise FormulaError(f"{label} has declared categories that collide after normalisation: {list(declared)}")
        if len({_kind(c) for c in s.cat.categories}) > 1:
            raise FormulaError(f"{label} mixes text labels and numeric codes in its declared categories; recode it explicitly")
    values = s.astype(object)
    na = pd.isna(values.to_numpy())
    if na.any():
        raise FormulaError(f"{label} has missing values; drop or impute them explicitly before fitting.")
    raw = values.tolist()
    labels = np.array([_scalar(label, v) for v in raw], dtype=object)
    if len({_kind(v) for v in raw}) > 1:
        raise FormulaError(
            f"{label} mixes text labels and numeric codes (for example a string '1' next to an integer 1), which "
            "would silently merge under label normalisation; recode the column to one kind"
        )
    unused = tuple(c for c in declared if c not in set(labels.tolist()))
    return labels, unused


def _scalar(label: str, v, *, where: str = "value") -> str:
    try:
        return _label(v)
    except FormulaError as e:
        raise FormulaError(f"{label}: {where} {v!r} refused: {e}") from None


def parse_formula(formula: str, data: pd.DataFrame | None = None, family: str = "gaussian",
                  categorical: Mapping[str, object] | None = None) -> ModelSpec:
    """Parse and validate a formula in the supported subset, optionally against a data frame.

    ``categorical`` maps a predictor column to its reference level (treatment coding); every
    other predictor term must be numeric.
    """
    trials = None
    if _BAMBI_PROPORTION_RE.match(formula):
        raise FormulaError(
            f"{formula!r}: Bambi's 'p(successes, trials)' left-hand side is not accepted; write the brms "
            "form 'successes | trials(n) ~ ...' (ab.binomial)."
        )
    mt = _TRIALS_FORMULA_RE.match(formula)
    if mt:
        if family != "binomial":
            raise FormulaError(
                f"{formula!r}: 'successes | trials(n)' is the binomial family's left-hand side; use "
                "ab.binomial, or drop '| trials(...)' for a Gaussian or Bernoulli response."
            )
        response, trials, rhs = mt.group(1), mt.group(2), mt.group(3)
        if trials == response:
            raise FormulaError(f"{trials!r}: the trials column cannot also be the response")
    else:
        if "~" in formula and "|" in formula.split("~", 1)[0]:
            raise FormulaError(
                f"{formula!r}: a '|' on the left-hand side must be exactly 'successes | trials(n)'."
            )
        m = _FORMULA_RE.match(formula)
        if not m:
            raise FormulaError(
                f"{formula!r}: expected 'response ~ terms'. Supported: "
                "'y ~ [1 +] x1 [+ x2 ...] + (1 | group)' and, for ab.binomial, "
                "'successes | trials(n) ~ [1 +] x1 [+ x2 ...] + (1 | group)'."
            )
        if family == "binomial":
            raise FormulaError(
                f"{formula!r}: the binomial family needs 'successes | trials(n) ~ ...' (brms syntax); "
                "counts without a trials column are not a binomial model."
            )
        response, rhs = m.group(1), m.group(2)
    if "-" in rhs:
        raise FormulaError(
            f"{formula!r}: '-' is not supported in M0 (this includes '- 1' to drop the "
            "intercept, which is required)."
        )

    predictors: list[str] = []
    groups: list[str] = []
    raw_interactions: list[tuple[str, str, str]] = []      # (left, right, piece as written)
    saw_intercept_token = False

    def _main(name: str, piece: str) -> None:
        if not re.fullmatch(_IDENT, name):
            raise FormulaError(f"{piece!r}: {name!r} is not a column name")
        if name == response:
            raise FormulaError(f"{name!r}: the response cannot also be a predictor")
        if name in predictors:
            raise FormulaError(
                f"{name!r}: duplicated predictor" + (f" after expanding {piece!r} (a*b means a + b + a:b; write "
                                                     f"'a + b + a:b' or 'a*b', not both)" if piece != name else "")
            )
        predictors.append(name)

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
        if "(" in piece or ")" in piece:
            raise FormulaError(
                f"{piece!r}: function calls and transformations inside the formula are "
                "not supported in M0; transform the column in the data frame instead."
            )
        if "*" in piece or ":" in piece:
            sep = "*" if "*" in piece else ":"
            if "*" in piece and ":" in piece:
                raise FormulaError(f"{piece!r}: mixing '*' and ':' in one term is not supported; write 'x*z' or 'x + z + x:z'")
            factors = [f.strip() for f in piece.split(sep)]
            if len(factors) != 2 or any(f == "" for f in factors):
                raise FormulaError(
                    f"{piece!r}: only a two-way interaction between one numeric and one categorical term is "
                    "supported in this slice (docs/m3a2-decision.md); higher-order interactions are not."
                )
            left, right = factors
            if left == right:
                raise FormulaError(f"{piece!r}: a term cannot interact with itself")
            if sep == "*":                                          # a*b is shorthand for a + b + a:b
                _main(left, piece)
                _main(right, piece)
            raw_interactions.append((left, right, piece))
            continue
        _main(piece, piece)

    for left, right, piece in raw_interactions:              # both main effects are required in this slice
        for side in (left, right):
            if side not in predictors and re.fullmatch(_IDENT, side) and side not in groups:
                raise FormulaError(
                    f"{piece!r}: the main effect {side!r} is missing. This slice requires both main effects of an "
                    f"interaction ('{left} + {right} + {left}:{right}' or '{left}*{right}'): omitting one is a different "
                    "model (the level intercepts or the reference slope are forced to zero and formulae changes the "
                    "interaction's coding) and is not part of the verified subset (docs/m3a2-decision.md)."
                )
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
    if trials is not None and (trials in predictors or trials == group):
        raise FormulaError(f"{trials!r}: the trials column cannot also be a predictor or the grouping factor")

    categoricals = _normalise_categorical(categorical, predictors, response, group, trials, formula)
    cat_names = {c.name for c in categoricals}
    interactions: list[tuple[str, str]] = []
    for left, right, piece in raw_interactions:
        for side in (left, right):
            if not re.fullmatch(_IDENT, side):
                raise FormulaError(f"{piece!r}: {side!r} is not a column name")
            if side == group:
                raise FormulaError(f"{piece!r}: the grouping factor {group!r} cannot enter a fixed interaction")
            if side == response or (trials is not None and side == trials):
                raise FormulaError(f"{piece!r}: the response or trials column cannot enter an interaction")
        cats = [s for s in (left, right) if s in cat_names]
        nums = [s for s in (left, right) if s not in cat_names]
        if len(cats) != 1:
            kind = "two categorical terms" if len(cats) == 2 else "two numeric terms"
            raise FormulaError(
                f"{piece!r}: an interaction between {kind} is not supported in this slice; only one numeric "
                "by one declared categorical term is (docs/m3a2-decision.md). A categorical side must be "
                "declared in categorical=."
            )
        x, z = nums[0], cats[0]
        if (x, z) in interactions:
            raise FormulaError(f"{piece!r}: the interaction '{x}:{z}' appears more than once after expansion")
        interactions.append((x, z))
    canonical_rhs = " + ".join([*predictors, *(f"{x}:{z}" for x, z in interactions), f"(1 | {group})"])
    spec = ModelSpec(
        formula=formula.strip(), response=response,
        predictors=tuple(p for p in predictors if p not in cat_names),
        group=group, intercept=True, trials=trials, rhs=canonical_rhs,
        terms=(*predictors, *(f"{x}:{z}" for x, z in interactions)), categoricals=categoricals,
        interactions=tuple(interactions),
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


def check_counts(label: str, s: pd.Series, *, cap: int = COUNT_CAP) -> np.ndarray:
    """A count column: real numeric, finite, integral, in ``[0, cap]``. Checked before any cast;
    returns the values as ``int64``."""
    reject_complex(label, s)
    if pd.api.types.is_bool_dtype(s):
        raise FormulaError(f"{label} is boolean; counts must be integers")
    if not pd.api.types.is_numeric_dtype(s):
        raise FormulaError(f"{label} has dtype {s.dtype}; counts must be numeric integers")
    v = s.to_numpy()
    if not np.all(np.isfinite(v.astype(float))):
        bad = v[~np.isfinite(v.astype(float))][0]
        raise FormulaError(f"{label} has a non-finite value ({bad!r}); counts must be finite integers")
    vf = v.astype(float)
    if not np.all(np.mod(vf, 1.0) == 0.0):
        bad = vf[np.mod(vf, 1.0) != 0.0][0]
        raise FormulaError(f"{label} has a non-integral value ({bad!r}); counts must be whole numbers")
    if np.any(vf < 0):
        raise FormulaError(f"{label} has a negative value ({vf[vf < 0][0]!r}); counts must be non-negative")
    if np.any(vf > cap):
        raise FormulaError(
            f"{label} has a value above the tested limit {cap} ({vf[vf > cap][0]:.0f}); larger counts per row "
            "are outside the validated domain (the float64 density and gradient checks in docs/m2-decision.md) "
            "and are refused rather than assumed. Note that splitting a row into several rows changes the "
            "unweighted centring and therefore what the centred-intercept prior refers to."
        )
    return vf.astype(np.int64)


def binomial_counts(spec: ModelSpec, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Validated ``(successes, trials)`` for a binomial formula, without dropping any row.

    Refuses ``successes > trials``, a zero-trial row with successes, and a data set with no
    trials at all. Rows with ``trials = 0`` and ``successes = 0`` pass; the model drops and
    reports them (they carry no likelihood and would otherwise enter the centring means).
    """
    na = data[[spec.response, spec.trials]].isna().any()
    if na.any():
        raise FormulaError(
            f"missing values in columns {list(na[na].index)}; drop or impute them explicitly before fitting."
        )
    k = check_counts(f"response {spec.response!r}", data[spec.response])
    n = check_counts(f"trials {spec.trials!r}", data[spec.trials])
    impossible = np.flatnonzero((n == 0) & (k > 0))
    if impossible.size:
        raise FormulaError(f"row {int(impossible[0])}: successes with zero trials is impossible data")
    over = np.flatnonzero(k > n)
    if over.size:
        i = int(over[0])
        raise FormulaError(
            f"row {i}: successes {spec.response!r} = {k[i]} exceed trials {spec.trials!r} = {n[i]}"
        )
    if not np.any(n > 0):
        raise FormulaError(
            f"no trials in the data ({spec.trials!r} is zero everywhere); a prior-only fit must be requested "
            "explicitly, it is not a binomial fit"
        )
    return k, n


def validate_data(spec: ModelSpec, data: pd.DataFrame, family: str = "gaussian") -> ModelSpec:
    """Check columns exist, are numeric where required, and have no missing values."""
    if (spec.trials is not None) != (family == "binomial"):
        raise FormulaError("a trials column is present exactly for the binomial family")
    needed = [spec.response, *spec.predictors, *spec.categorical_names, spec.group] + ([spec.trials] if spec.trials else [])
    missing = [c for c in needed if c not in data.columns]
    if missing:
        raise FormulaError(f"columns not in data: {missing}")
    for c in spec.predictors:
        reject_complex(f"column {c!r}", data[c])
        if not _is_numeric(data[c]):
            raise FormulaError(
                f"column {c!r} has dtype {data[c].dtype}; a predictor is numeric unless it is declared "
                f"categorical with its reference level, for example categorical={{{c!r}: <level>}} "
                "(treatment coding; docs/m3a1-decision.md)."
            )
    categoricals = []
    for cat in spec.categoricals:
        labels, unused = categorical_labels(f"categorical predictor {cat.name!r}", data[cat.name])
        levels = tuple(sorted(set(labels.tolist())))
        if len(levels) < 2:
            raise FormulaError(f"predictor {cat.name!r} is constant (single level {levels[0]!r} in the fitted rows)")
        if cat.reference not in levels:
            raise FormulaError(
                f"reference level {cat.reference!r} of {cat.name!r} does not occur in the fitted rows; "
                f"observed levels: {list(levels)}"
            )
        categoricals.append(Categorical(name=cat.name, reference=cat.reference, levels=levels, unused=unused))
    reject_complex(f"response {spec.response!r}", data[spec.response])
    if family == "bernoulli":
        _validate_binary_response(spec.response, data[spec.response])
    elif family == "binomial":
        binomial_counts(spec, data)
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
        group=spec.group, intercept=True, levels=levels, trials=spec.trials, rhs=spec.rhs,
        terms=spec.terms, categoricals=tuple(categoricals), interactions=spec.interactions,
    )
