"""brms's ``gr(g, cov = A)``: a known covariance of a grouping factor's levels, read from ``data2`` (G05b, docs/gr-cov-dist.md).

The levels' effects of a term ``(1 | gr(g, cov = A))`` have covariance ``sd^2 A``: ``r = sd * Lcov z`` with ``Lcov`` the lower
Cholesky factor of ``A`` on the factor's levels (brms's ``Lcov_m``). With several coefficients, ``(1 + x | gr(g, cov = A))``,
``vec(r) = (Lcov kron diag(sd) L) vec(z)``, level by level, so that ``vec(r)`` has covariance ``A kron Sigma``; coefficient
``c``'s effects over the levels are ``Lcov`` times the effects the term would have without ``cov`` (divergence D185: brms 2.23.0
drops the negative entries of ``Lcov`` there). This module reads and checks ``A``; the model builds the effects
(``Model._add_custom_group_effects``).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from autobayes.formula import FormulaError, categorical_labels


@dataclass(frozen=True)
class GroupCovariance:
    """The known covariance of one ``gr(g, cov = A)`` term on the factor's fitted levels: ``matrix`` the ``data2`` name, ``levels``
    the levels in the model's (sorted) order, ``A`` the matrix on them in that order, ``L`` its lower Cholesky factor (brms's
    ``Lcov_m``) and ``unused`` the labels of the ``data2`` matrix that are no fitted level (dropped, as brms drops them)."""
    factor: str
    matrix: str
    levels: tuple[str, ...]
    A: np.ndarray
    L: np.ndarray
    unused: tuple[str, ...] = ()


def known_covariance(term, levels: tuple[str, ...], data2) -> tuple[GroupCovariance, list[str]]:
    """The known covariance of ``term`` (a ``GroupTerm`` with ``cov``) on its fitted ``levels``, and the warnings to record.

    ``data2[term.cov]`` must be a square pandas DataFrame whose index labels the levels (brms: "Row or column names are required
    for within-group covariance matrices."), its columns the same labels in the same order or positions, holding finite real
    numbers; it must be symmetric (brms: "Within-group covariance matrices must be symmetric."), label every fitted level (brms:
    "Levels of the within-group covariance matrix for 'g' do not match names of the grouping levels.") and be positive definite on
    them (brms: "Within-group covariance matrices must be positive definite."). Labels that are no fitted level are dropped with a
    warning (brms drops them silently)."""
    name = term.cov
    what = f"{term.canonical}"
    if not isinstance(data2, Mapping) or name not in data2:
        raise FormulaError(f"{what} reads the known covariance {name!r} from data2: pass data2={{{name!r}: <a pandas DataFrame "
                           "labelled by the levels of the factor on both axes>} (brms: \"Object '" + name + "' was not found in "
                           "'data2'.\"; docs/gr-cov-dist.md)")
    M = data2[name]
    if not isinstance(M, pd.DataFrame) or M.index.equals(pd.RangeIndex(len(M))) or M.shape[0] != M.shape[1]:
        raise FormulaError(f"data2[{name!r}] must be a square pandas DataFrame whose index labels the levels of {term.factor!r} (brms: "
                           "'Row or column names are required for within-group covariance matrices.')")
    rows, _ = categorical_labels(f"the row labels of data2[{name!r}]", pd.Series(list(M.index)))
    if not M.columns.equals(pd.RangeIndex(M.shape[1])):        # the columns by label, the rows' in their order
        cols, _ = categorical_labels(f"the column labels of data2[{name!r}]", pd.Series(list(M.columns)))
        if cols.tolist() != rows.tolist():
            raise FormulaError(f"data2[{name!r}]: the columns must be labelled as the rows, in the same order (or by position)")
    if len(set(rows.tolist())) != len(rows):
        raise FormulaError(f"data2[{name!r}]: each level must label one row")
    values = M.to_numpy()
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values) or not np.isfinite(values.astype(float)).all():
        raise FormulaError(f"data2[{name!r}] must hold finite real numbers")
    values = values.astype(float)
    if not np.allclose(values, values.T, rtol=0.0, atol=1e-12 * max(1.0, float(np.max(np.abs(values))))):
        raise FormulaError(f"data2[{name!r}] must be symmetric (brms: 'Within-group covariance matrices must be symmetric.')")
    position = {label: i for i, label in enumerate(rows.tolist())}
    missing = [level for level in levels if level not in position]
    if missing:
        raise FormulaError(f"{what}: the levels {missing[:5]} of {term.factor!r} are not labels of data2[{name!r}] (brms: 'Levels of "
                           f"the within-group covariance matrix for '{term.factor}' do not match names of the grouping levels.')")
    index = [position[level] for level in levels]
    A = 0.5 * (values + values.T)[np.ix_(index, index)]
    try:
        L = np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        L = None
    if L is None or not np.linalg.eigvalsh(A).min() > 0:
        raise FormulaError(f"data2[{name!r}] on the levels of {term.factor!r} is not positive definite (brms: 'Within-group "
                           "covariance matrices must be positive definite.')")
    unused = tuple(label for label in rows.tolist() if label not in set(levels))
    notes = ([f"data2[{name!r}]: the labels {list(unused[:5])}{' ...' if len(unused) > 5 else ''} are no fitted level of "
              f"{term.factor!r}; their rows and columns are dropped (brms drops them too)"] if unused else [])
    return GroupCovariance(term.factor, name, tuple(levels), A, L, unused), notes


def check_dropped_levels(term, given: pd.Series, levels: tuple[str, ...]) -> None:
    """With ``dropna=True`` (D174): refused when the rows dropped for their missing values hold every row of a level of the
    factor of ``term`` (``given``, the factor's column as given, missing labels left out): the levels the known covariance covers
    would then depend on another column's missing values, which no frozen case verifies (divergence D186)."""
    lost = sorted(set(given.dropna().astype(str)) - set(levels))
    if lost:
        raise FormulaError(f"{term.canonical}: dropna=True drops every row of the level(s) {lost[:5]} of {term.factor!r}, so the levels "
                           f"of the known covariance {term.cov!r} would change with the missing values; dropna=True with gr(cov = ) is "
                           "offered when every level keeps a row (docs/gr-cov-dist.md, divergence D186)")
