"""Autocorrelation terms: brms's ``ar()``, ``ma()`` and ``arma()`` in the regression formulation (C04,
docs/autocorrelation.md).

brms 2.23 writes an ARMA term with ``cov = FALSE`` (its default) as a recursion over the residuals of the linear
predictor. The rows form series (the levels of ``gr``; one series without it), each ordered by ``time`` (the data order
without it), and brms's Stan data are sorted by series and then time. With ``eta_i`` row ``i``'s linear predictor
without the ARMA terms (every population-level, group-level, monotonic and offset term) and ``E_i[k]`` the residual
``e`` of the row ``k`` places before ``i`` in its own series (0 when the series has no such row; brms's ``Err``, filled
through ``J_lag``):

    mu_i = eta_i + sum_{k <= q} ma[k] E_i[k] + sum_{k <= p} ar[k] E_i[k],    e_i = y_i - eta_i - sum_{k <= q} ma[k] E_i[k]

so ``e`` is the residual of the predictor with its MA part (``y - eta`` when ``q = 0``), and the AR part reads the same
lagged residuals. Lags count rows of a series, not units of ``time``: a gap in ``time`` is one step, as in brms. ``ar``
and ``ma`` are unbounded vectors (``vector[Kar] ar``, ``vector[Kma] ma``), flat by default.

Without an MA part the terms are lagged differences of data and predictor and vectorise. With one, the residuals solve a
linear recursion along each series: ``arma_effect`` writes it as a ``pytensor.scan`` over the positions in a series with
every series at once (``AutocorDesign.cells``: the ``(T, G)`` table of each series' rows in time order), which PyTensor's
JAX backend lowers to ``lax.scan``. ``arma_values`` is the same model in NumPy written as brms writes it, row by row along
brms's sorted rows with ``J_lag``: the model checks its graph against it when it is built, and ``predict()`` uses it.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from autobayes.formula import _IDENT, FormulaError, _split_top_level, categorical_labels, reject_complex

# brms's other autocorrelation structures, refused by name (divergence D69); sar() and fcor() are taken (F02b, autobayes.sar;
# F03a, autobayes.fcor)
UNSUPPORTED_STRUCTURES = {
    "cosy": "compound symmetry, cosy()", "car": "spatial conditional autoregression, car()",
    "unstr": "an unstructured residual covariance, unstr()",
}
_CALL_RE = re.compile(r"^(ar|ma|arma)\s*\((.*)\)$", re.S)
_OTHER_RE = re.compile(r"^(cosy|car|unstr)\s*\(", re.S)
_FORMALS = {"ar": ("time", "gr", "p", "cov"), "ma": ("time", "gr", "q", "cov"), "arma": ("time", "gr", "p", "q", "cov")}
_DEFAULT_ORDER = {"ar": (1, 0), "ma": (0, 1), "arma": (1, 1)}       # brms's defaults of (p, q) for each call


@dataclass(frozen=True)
class Autocor:
    """A formula's autocorrelation term, brms's ``ar(time, gr, p)``, ``ma(time, gr, q)`` or ``arma(time, gr, p, q)``
    with ``cov = FALSE`` (C04). It is not a population-level term: it has no design column and no ``b`` coefficient,
    but its own coefficients ``ar`` (``p`` of them, brms's ``Kar``) and ``ma`` (``q``, brms's ``Kma``) with the prior
    classes ``ar`` and ``ma``. ``call`` is the function as written, ``time`` the column ordering each series (None: the
    data order) and ``gr`` the column whose levels are the series (None: one series), as brms's defaults ``NA`` read."""

    call: str
    time: str | None
    gr: str | None
    p: int
    q: int

    @property
    def term(self) -> str:
        """The term as the canonical formula writes it: ``arma(time = t, gr = g, p = 1, q = 1)``."""
        args = ([f"time = {self.time}"] if self.time is not None else []) + ([f"gr = {self.gr}"] if self.gr is not None else [])
        args += [f"p = {self.p}"] if self.call in ("ar", "arma") else []
        args += [f"q = {self.q}"] if self.call in ("ma", "arma") else []
        return f"{self.call}({', '.join(args)})"

    @property
    def columns(self) -> tuple[str, ...]:
        """The data columns the term reads: its time column and its series column, when given."""
        return tuple(c for c in (self.time, self.gr) if c is not None)

    @property
    def max_lag(self) -> int:
        """brms's ``max_lag``: the most rows back a residual is read from, ``max(p, q)``."""
        return max(self.p, self.q)

    @property
    def parts(self) -> tuple[tuple[str, int], ...]:
        """``("ar", p)`` and ``("ma", q)`` for the parts the term has (a positive order)."""
        return tuple((part, order) for part, order in (("ar", self.p), ("ma", self.q)) if order > 0)


def _order(value: str, name: str, piece: str) -> int:
    """``p`` or ``q`` as written: a non-negative whole number (R's ``2``, ``2L`` or ``2.0``)."""
    text = value[:-1] if re.fullmatch(r"\d+L", value) else value
    try:
        number = float(text)
    except ValueError:
        number = math.nan
    if not (math.isfinite(number) and number >= 0 and number == math.floor(number)):
        what = "Autoregressive" if name == "p" else "Moving-average"
        raise FormulaError(f"{piece!r}: {name} = {value}: {what} order must be a non-negative integer (brms's rule too)")
    return int(number)


def parse_autocor(piece: str, formula: str) -> Autocor | None:
    """brms's autocorrelation term ``piece`` as written in ``formula``: an ``Autocor``, or None when ``piece`` is not one.
    The arguments match R's rules for ``ar(time = NA, gr = NA, p = 1, cov = FALSE)``, ``ma(time = NA, gr = NA, q = 1, cov =
    FALSE)`` and ``arma(time = NA, gr = NA, p = 1, q = 1, cov = FALSE)``: named ones first, then the positional ones in
    order. Refused, naming the construct: brms's other autocorrelation structures (D69), ``cov = TRUE`` (D68), a ``time``
    or ``gr`` that is not one column name (D70), an argument the function does not have, and ``p = q = 0`` (brms refuses
    it too)."""
    other = _OTHER_RE.match(piece)
    if other:
        raise FormulaError(
            f"{piece!r}: brms's {UNSUPPORTED_STRUCTURES[other.group(1)]} is not offered here; the autocorrelation terms "
            "supported are ar(), ma() and arma() in brms's regression formulation (docs/autocorrelation.md, divergence D69)"
        )
    m = _CALL_RE.match(piece)
    if not m:
        return None
    call, inside = m.group(1), m.group(2).strip()
    formals = _FORMALS[call]
    named, positional = {}, []
    for arg in (_split_top_level(inside, ",") if inside else []):
        arg = arg.strip()
        km = re.fullmatch(rf"({_IDENT})\s*=(?!=)\s*(.+)", arg, re.S)
        if km is None:
            positional.append(arg)
            continue
        name, value = km.group(1), km.group(2).strip()
        if name not in formals:
            raise FormulaError(f"{piece!r}: {call}() has no argument {name!r}; brms's {call}() takes {', '.join(formals)}")
        if name in named:
            raise FormulaError(f"{piece!r}: the argument {name!r} is given twice")
        named[name] = value
    rest = [f for f in formals if f not in named]
    if len(positional) > len(rest):
        raise FormulaError(f"{piece!r}: too many arguments; brms's {call}() takes {', '.join(formals)}")
    named.update(zip(rest, positional))
    cov = named.get("cov", "FALSE")
    if cov in ("TRUE", "T"):
        raise FormulaError(
            f"{piece!r}: cov = TRUE, brms's covariance formulation of the ARMA term (the residuals' joint normal "
            "covariance matrix within each series, with |ar| < 1), is not offered here; the regression formulation "
            "cov = FALSE, brms's default, is (docs/autocorrelation.md, divergence D68)"
        )
    if cov not in ("FALSE", "F"):
        raise FormulaError(f"{piece!r}: cov must be TRUE or FALSE, got {cov!r}")
    columns = {}
    for name in ("time", "gr"):
        value = named.get(name, "NA")
        if value == "NA":
            columns[name] = None
        elif re.fullmatch(_IDENT, value):
            columns[name] = value
        else:
            hint = (" (brms's gr = g:h groups the series by two columns; make one column of the combined labels and "
                    "name it)" if name == "gr" else "")
            raise FormulaError(
                f"{piece!r}: {name} = {value}: {name} must be one column name here{hint} (docs/autocorrelation.md, "
                "divergence D70)"
            )
    p_default, q_default = _DEFAULT_ORDER[call]
    p = _order(named["p"], "p", piece) if "p" in named else p_default
    q = _order(named["q"], "q", piece) if "q" in named else q_default
    if p + q == 0:
        raise FormulaError(f"{piece!r}: at least one of p and q must be greater than zero (brms's rule too)")
    if columns["time"] is not None and columns["time"] == columns["gr"]:
        raise FormulaError(f"{piece!r}: the time column cannot also be the series column gr")
    return Autocor(call=call, time=columns["time"], gr=columns["gr"], p=p, q=q)


def check_autocor(term: Autocor, piece: str, *, response: str, trials: str | None, fam, additions, formula: str) -> None:
    """The refusals of an autocorrelation term that depend on the rest of the formula: its columns may be predictors or
    grouping factors, but not the response or the trials column; the family must take the term (``FamilySpec.
    autocorrelation``: the Gaussian family, verified against frozen brms cases; a mixture, C05, as brms refuses it too);
    response addition terms are not combined with it (D70)."""
    from autobayes.families import FAMILIES

    for column in term.columns:
        if column == response or (trials is not None and column == trials):
            raise FormulaError(f"{piece!r}: the {'response' if column == response else 'trials column'} {column!r} cannot be "
                               "the time or series column of an autocorrelation term")
    if fam.mixture:                                          # C05: brms refuses it too (not a divergence)
        raise FormulaError(
            f"{formula!r}: an autocorrelation term in a mixture ({term.term}) is not supported, as in brms ('Explicit covariance "
            "terms can only be specified on mu': a mixture's linear predictors are mu1, mu2, ...); docs/autocorrelation.md"
        )
    if not fam.autocorrelation:
        offering = [name for name, other in FAMILIES.items() if other.autocorrelation]
        raise FormulaError(
            f"{formula!r}: the autocorrelation term {term.term!r} is not available for the {fam.name} family; brms's ARMA "
            f"terms are verified here for the {', '.join(offering) or 'no'} famil{'y' if len(offering) == 1 else 'ies'} "
            "(docs/autocorrelation.md, divergence D70)"
        )
    if additions:
        raise FormulaError(
            f"{formula!r}: the autocorrelation term {term.term!r} with the response addition term(s) "
            f"{', '.join(a.term for a in additions)} is not supported in this slice (docs/autocorrelation.md, divergence D70)"
        )


# -- the series of the fitted rows ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class AutocorDesign:
    """The series structure of an autocorrelation term on the fitted rows (C04). ``series`` are the series' labels,
    sorted (``("1",)`` without ``gr``); ``series_index`` (n,) each fitted row's series and ``position`` (n,) its 0-based
    place in its series in time order, both in data order; ``order`` (n,) the fitted rows as brms's Stan data list them,
    sorted by series and then time (data row per Stan row); ``J_lag`` (n,) brms's ``J_lag`` on those rows: for Stan row
    ``i``, how many of the residuals of rows ``i``, ``i - 1``, ... ``i + 1 - max_lag`` belong to row ``i + 1``'s series,
    i.e. the lags available to row ``i + 1`` (0 for the last row)."""

    term: str
    p: int
    q: int
    series: tuple[str, ...]
    series_index: np.ndarray
    position: np.ndarray
    order: np.ndarray
    J_lag: np.ndarray

    @property
    def n(self) -> int:
        return int(self.order.size)

    @property
    def max_lag(self) -> int:
        return max(self.p, self.q)

    @property
    def lengths(self) -> np.ndarray:
        """The number of rows of each series."""
        return np.bincount(self.series_index, minlength=len(self.series))

    @property
    def T(self) -> int:
        """The length of the longest series."""
        return int(self.lengths.max()) if self.n else 0

    @property
    def cells(self) -> np.ndarray:
        """``(T, G)``: the fitted row at each position (rows) of each series (columns), ``n`` where a series is shorter."""
        out = np.full((self.T, len(self.series)), self.n, dtype=np.int64)
        out[self.position, self.series_index] = np.arange(self.n)
        return out


def series_layout(term: Autocor, df: pd.DataFrame) -> AutocorDesign:
    """The series of ``term`` on the rows of ``df`` (the fitted frame), as brms's ``order_data()`` and ``data_ac()``
    build them: the rows sorted by series label and then by time, ``J_lag`` on the sorted rows. The series column is read
    as labels (text, or whole numbers; no missing values) and the time column must be numeric and finite (brms orders any
    R vector, text as text: ``"10"`` before ``"9"``; D70); two rows of a series at the same time are refused, as brms
    refuses them."""
    n = len(df)
    for column in term.columns:
        if column not in df.columns:
            raise FormulaError(f"columns not in data: [{column!r}] (the autocorrelation term {term.term!r} reads it)")
    if term.gr is None:
        labels = np.full(n, "1", dtype=object)
    else:
        labels, _ = categorical_labels(f"series column {term.gr!r} of {term.term!r}", df[term.gr])
    if term.time is None:
        time = np.arange(n, dtype=float)
    else:
        s = df[term.time]
        reject_complex(f"time column {term.time!r}", s)
        if not (pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)):
            raise FormulaError(
                f"time column {term.time!r} of {term.term!r} has dtype {s.dtype}; it must be numeric (brms orders any column, "
                "text as text, so that '10' comes before '9'; compute a numeric time and name it: docs/autocorrelation.md, "
                "divergence D70)"
            )
        time = s.to_numpy(dtype=float)
        if not np.isfinite(time).all():
            raise FormulaError(f"time column {term.time!r} of {term.term!r} has missing or non-finite values")
    series = tuple(sorted(set(labels.tolist())))
    lookup = {label: k for k, label in enumerate(series)}
    series_index = np.array([lookup[label] for label in labels], dtype=np.int64)
    order = np.lexsort((time, series_index))                     # by series, then by time (stable: no ties remain)
    s_sorted, t_sorted = series_index[order], time[order]
    tie = np.flatnonzero((s_sorted[1:] == s_sorted[:-1]) & (t_sorted[1:] == t_sorted[:-1]))
    if tie.size:
        i, j = order[tie[0]], order[tie[0] + 1]
        where = f"series {series[s_sorted[tie[0]]]!r}" if term.gr is not None else "the series"
        raise FormulaError(f"{term.term!r}: rows {int(i)} and {int(j)} of {where} have the same time {float(t_sorted[tie[0]])!r}; "
                           "time points within a series must be unique (brms's rule too)")
    starts = np.r_[True, s_sorted[1:] != s_sorted[:-1]]
    run_start = np.maximum.accumulate(np.where(starts, np.arange(n), 0))
    pos_sorted = np.arange(n) - run_start
    position = np.empty(n, dtype=np.int64)
    position[order] = pos_sorted
    J_lag = np.zeros(n, dtype=np.int64)
    J_lag[:-1] = np.minimum(term.max_lag, pos_sorted[1:])
    return AutocorDesign(term=term.term, p=term.p, q=term.q, series=series, series_index=series_index, position=position,
                         order=order.astype(np.int64), J_lag=J_lag)


# -- the model ---------------------------------------------------------------------------------------------------------
def arma_values(eta, y, ar, ma, design: AutocorDesign) -> np.ndarray:
    """The linear predictor with the ARMA terms (NumPy), row by row along brms's sorted rows as brms writes it: for each
    row, the MA part from the lagged residuals, then the row's residual, then the AR part. ``eta`` is ``(..., n)``, the
    predictor without the terms on the fitted rows in data order; ``y`` the ``(n,)`` responses; ``ar`` ``(..., p)`` and
    ``ma`` ``(..., q)`` per draw (``None`` for a part the term lacks). Returns ``(..., n)`` in data order."""
    eta = np.asarray(eta, dtype=float)
    lead, n = eta.shape[:-1], eta.shape[-1]
    if n != design.n:
        raise ValueError(f"eta has {n} rows; the autocorrelation term has {design.n} fitted rows")
    coefficients = {}
    for part, order, value in (("ar", design.p, ar), ("ma", design.q, ma)):
        if order:
            arr = np.asarray(value, dtype=float)
            if arr.shape[-1:] != (order,):
                raise ValueError(f"{part} must have {order} entries on its last axis, got shape {arr.shape}")
            coefficients[part] = np.broadcast_to(arr, lead + (order,))
    order, J_lag = design.order, design.J_lag
    mu = np.array(eta[..., order], dtype=float)
    ys = np.asarray(y, dtype=float)[order]
    err = np.zeros(lead + (n,))
    for i in range(n):
        lags = int(J_lag[i - 1]) if i else 0                       # brms's Err[i, 1..J_lag[i - 1]]: err[i - 1], err[i - 2], ...
        if "ma" in coefficients:
            for k in range(min(lags, design.q)):
                mu[..., i] += coefficients["ma"][..., k] * err[..., i - 1 - k]
        err[..., i] = ys[i] - mu[..., i]
        if "ar" in coefficients:
            for k in range(min(lags, design.p)):
                mu[..., i] += coefficients["ar"][..., k] * err[..., i - 1 - k]
    out = np.empty_like(mu)
    out[..., order] = mu
    return out


def arma_effect(eta, y, ar, ma, design: AutocorDesign):
    """The linear predictor with the ARMA terms as a PyTensor graph: ``eta`` the ``(n,)`` predictor without them (data
    order), ``y`` the responses, ``ar`` / ``ma`` the coefficient vectors (``None`` for a part the term lacks). The rows'
    residuals are laid out by position and series (``design.cells``, a zero where a series is shorter); with an MA part
    they solve ``e_t = r_t - sum_k ma[k] e_{t-k}`` by a scan over the positions, all series at once, with a series' cells
    past its end held at zero; each lag ``k`` is then that table shifted ``k`` positions down, read back at every row."""
    import pytensor
    import pytensor.tensor as pt

    G, T, n = len(design.series), design.T, design.n
    r = pt.as_tensor(np.asarray(y, dtype=np.float64)) - eta
    cells = design.cells
    E = pt.concatenate([r, pt.zeros((1,), dtype="float64")])[cells]            # (T, G) residuals before the MA part
    if design.q:
        valid = (cells < n).astype(np.float64)

        def step(r_t, valid_t, previous, ma_):                                   # previous[k] = e_{t-1-k}, k < q
            e_t = valid_t * (r_t - pt.dot(ma_, previous))
            return pt.concatenate([e_t[None, :], previous[:-1]], axis=0)

        lags = pytensor.scan(step, sequences=[E, pt.as_tensor(valid)], outputs_info=[pt.zeros((design.q, G), dtype="float64")],
                             non_sequences=[ma], strict=True, return_updates=False)
        E = lags[:, 0, :]
    flat = design.position * G + design.series_index                            # each row's cell in the flattened table
    out = eta
    for k in range(1, min(design.max_lag, T - 1) + 1):                           # a lag no series reaches adds nothing
        lagged = pt.concatenate([pt.zeros((k, G), dtype="float64"), E[:-k]], axis=0).reshape((-1,))[flat]
        coefficient = None
        if k <= design.p:
            coefficient = ar[k - 1]
        if k <= design.q:
            coefficient = ma[k - 1] if coefficient is None else coefficient + ma[k - 1]
        out = out + coefficient * lagged
    return out


def coefficient_vector(name: str, prior, dims: str):
    """The vector ``ar`` or ``ma`` under its location prior (``Normal``, ``StudentT`` or ``Flat``; brms's ``vector[K] ar``
    with the prior on each entry), a PyMC variable named ``name`` with dims ``dims`` (whose coordinate must exist),
    unconstrained. Must be called inside a PyMC model context. Parameters are handed over as float64 (a Python 0.5 would
    become a float32 constant)."""
    import pymc as pm

    return getattr(pm, prior.name)(name, **{k: np.float64(v) for k, v in prior.params.items()}, dims=dims)


def autocor_family(base, add_effects):
    """The Bambi family ``base`` (a built-in family's name or a family object) whose parent linear predictor gains the
    ARMA terms (C04): Bambi passes the parent's linear predictor through the family's ``transform_backend_eta`` inside
    the model it builds, and ``add_effects(eta)`` creates ``ar`` / ``ma`` there and returns the predictor with the terms.
    The family's own transform (a monotonic term's, ``likelihoods.monotonic_family``) runs first, so that the residuals
    are those of every other term, as in brms's Stan code."""
    import copy

    from bambi.defaults import get_builtin_family

    family = get_builtin_family(base) if isinstance(base, str) else copy.copy(base)
    own = getattr(family, "transform_backend_eta", None)

    def transform_backend_eta(eta, kwargs):
        return add_effects(eta if own is None else own(eta, kwargs))

    family.transform_backend_eta = transform_backend_eta
    return family


def probe_coefficients(order: int, scale: float) -> np.ndarray:
    """Distinct moderate coefficients ``scale / k`` with alternating signs, for the build-time check of the graph."""
    return np.array([scale * (-1.0) ** k / (k + 1.0) for k in range(order)])
