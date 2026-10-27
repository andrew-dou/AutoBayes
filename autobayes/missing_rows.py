"""Rows with missing values, dropped on request (D174, docs/missing-rows.md).

brms drops every row with a missing value in a variable the model reads, except the variables modelled with ``mi()``, and warns
'Rows containing NAs were excluded from the model.'. Here such data are refused unless the constructor is given ``dropna=True``
(the keyword pandas's ``DataFrame.dropna`` and Bambi's ``Model`` use for the same act); then the rows with a missing value in a
column the model reads (``columns_read``; not the columns whose missing values are parameters, C06) are dropped before anything
else reads the data, the rest are kept in data order, a warning gives their number and positions, and the model records the
positions (``model.dropped_na_rows``, 0-based positions in the data frame as given). The model is then the model of the rows that
remain: ``N``, the centring of the predictors, the factor levels, the default priors' location and scale (``default_priors(...,
dropna=True)`` drops the same rows), and the rows of ``predict()``, ``log_lik``, ``loo`` and ``bayes_R2``. New rows with missing
values are refused, naming the columns.

Not offered: dropping rows from a model whose terms are built over the data's rows as a whole (an autocorrelation or residual
correlation term's series, ``sar()``'s and ``fcor()``'s matrices from ``data2``, ``car()``'s locations): no frozen case verifies
what brms does to them (divergence D177). Without missing values ``dropna=True`` changes nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BRMS_WARNING = "Rows containing NAs were excluded from the model."


def check_dropna(value) -> bool:
    """``dropna`` as given to a constructor: True or False, nothing else."""
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"dropna must be True or False, got {value!r} (docs/missing-rows.md)")
    return bool(value)


def columns_read(spec) -> list[str]:
    """The data columns a univariate model of ``spec`` (a ``ModelSpec``) reads, in the order ``Model`` selects them: the response
    (a ``cbind()`` response's columns), the population-level terms' columns, the grouping factors' columns, the trials, the
    offset, the addition terms' columns, each distributional formula's columns, and an autocorrelation, CAR or residual
    correlation term's columns."""
    operands = [c for t in spec.group_terms for c in t.data_columns]
    cols = ([*(getattr(spec, "response_columns", ()) or (spec.response,)), *spec.pop_columns] + operands
            + ([spec.trials] if spec.trials else []) + (list(spec.offset.columns) if spec.offset is not None else [])
            + [c for a in spec.additions for c in a.columns]
            + [c for d in spec.dpars for c in d.pop_columns]
            + [c for d in spec.dpars for t in d.group_terms for c in t.data_columns])
    for term in (spec.autocor, spec.car, spec.timecor):
        cols += list(term.columns) if term is not None else []
    return list(dict.fromkeys(cols))


def row_terms(spec) -> list[str]:
    """The terms of ``spec`` built over the data's rows as a whole, which rows cannot be dropped from (module docstring)."""
    return [t.term for t in (spec.autocor, spec.timecor, spec.sar, spec.fcor, spec.car) if t is not None]


def drop_missing(frame: pd.DataFrame, columns, terms=()) -> tuple[pd.DataFrame, tuple[int, ...], list[str]]:
    """``frame`` without its rows with a missing value in ``columns`` (those of them it has), the rest in data order and
    renumbered, the positions of the rows dropped and the columns that hold the missing values; ``frame`` itself, ``()`` and
    ``[]`` when no row is dropped. Refused when every row would go, and when ``terms`` (``row_terms``, or a multivariate model's
    ``subset()``) names a term the rows cannot be dropped from."""
    from autobayes.formula import FormulaError

    columns = [c for c in columns if c in frame.columns]
    na = frame[columns].isna()
    missing = na.any(axis=1).to_numpy()
    rows = np.flatnonzero(missing)
    if rows.size == 0:
        return frame, (), []
    where = [c for c in columns if na[c].any()]
    if terms:
        raise FormulaError(f"dropna=True would drop {rows.size} row(s) with missing values in {where} from a model with {terms[0]}, "
                           "whose data are over the data's rows as a whole: dropping rows there is not offered (no frozen case "
                           "verifies what brms does; docs/missing-rows.md, divergence D177); drop the rows (and the matching rows "
                           "of any data2 matrix) in the data first")
    if rows.size == len(frame):
        raise FormulaError(f"every row has a missing value in {where}: dropna=True would leave no row to fit")
    return frame.iloc[np.flatnonzero(~missing)].reset_index(drop=True), tuple(int(i) for i in rows), where


def drop_missing_multivariate(spec, data: pd.DataFrame) -> tuple[pd.DataFrame, tuple[int, ...], list[str]]:
    """``drop_missing`` for a multivariate model (``MultivariateSpec``): one set of rows for every response, those with a missing
    value in a column some response's formula reads, except the responses whose missing values are parameters (``| mi()``, C06:
    brms keeps a row whose only missing values are such a variable's); refused for a model with ``subset()`` (F06c), whose
    responses have their own rows, and with a term over the data's rows."""
    columns = [c for r in spec.responses for c in columns_read(spec.specs[r]) if c not in spec.imputed]
    terms = ([f"subset({c})" for c in spec.subsets.values()]
             + [t for r in spec.responses for t in row_terms(spec.specs[r])])
    return drop_missing(data.reset_index(drop=True), list(dict.fromkeys(columns)), terms)


def dropped_warning(rows: tuple[int, ...], n: int, where) -> str:
    """The warning of a drop (brms's 'Rows containing NAs were excluded from the model.'), with the number and positions."""
    shown = list(rows)[:10]
    return (f"{BRMS_WARNING} {len(rows)} of {n} rows have a missing value in {list(where)} and were dropped (dropna=True; positions "
            f"{shown}{'...' if len(rows) > 10 else ''}, model.dropped_na_rows); the model is fitted to the other {n - len(rows)} rows "
            "in data order")
