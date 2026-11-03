"""Group-level terms from integer level codes (docs/scaling.md): what lets a model with many rows and many levels build,
evaluate and predict in memory proportional to the rows times the coefficients rather than the rows times the levels.

Bambi is handed the population-level formula only. Each group-level coefficient Bambi builds (``1|g``, ``x|g``, a
distributional formula's ``sigma_1|g``) is added to the Bambi component's terms before ``build()`` as an ``IndexGroupTerm``:
a ``bambi.terms.GroupSpecificTerm`` that holds the factor's sorted levels, each row's 0-based level code and the column the
coefficient multiplies (formulae's own column: int64 ones for a varying intercept, the frame's column as given for a slope),
which is all Bambi's PyMC backend reads, so it builds the graph it builds from formulae's dense design: the same variables,
names, dims, coordinates and log density. ``GroupBlock.Z`` and ``Design.Z`` are built from the codes on access, as the
arrays they always were, and refused above ``DENSE_Z_MAX_ENTRIES`` entries.

The design relies on Bambi 0.17.2's private layout (``bambi_layout_problems`` lists what it checks; a test fails when any of
it changes): the frontend ``DistributionalComponent.terms`` dict and its ``group_specific_terms``, the backend's
``build_group_specific_terms`` (which reads ``predictor``, ``group_index`` and ``data`` of each term and gathers
``coef[group_index] * predictor``) and ``bambi.backend.terms.GroupSpecificTerm`` (which reads ``name``, ``alias``, ``coords``,
``prior`` and ``hyperprior_alias``).
"""
from __future__ import annotations

import hashlib
import inspect

import numpy as np
import pandas as pd

# The largest dense group-level design ``GroupBlock.Z`` or ``Design.Z`` builds on access (rows times columns): 50 million
# float64 entries, 400 MB. The model never needs the dense arrays; above this they are refused (docs/scaling.md).
DENSE_Z_MAX_ENTRIES = 50_000_000

BAMBI_VERSION = "0.17.2"


def check_dense_size(rows: int, columns: int, what: str) -> None:
    """Refuse a dense group-level design of more than ``DENSE_Z_MAX_ENTRIES`` entries."""
    if rows * columns > DENSE_Z_MAX_ENTRIES:
        raise ValueError(f"{what} would be a dense {rows} x {columns} array ({rows * columns:,} entries, more than "
                         f"group_index.DENSE_Z_MAX_ENTRIES = {DENSE_Z_MAX_ENTRIES:,}); the model keeps each block's level codes "
                         "and covariates instead (GroupBlock.group_index, .covariate, .memberships, .weights, .covariates; "
                         "docs/scaling.md)")


def level_codes(labels, levels) -> np.ndarray:
    """The 0-based position of each label among ``levels`` (int64), every label one of them."""
    codes = pd.Index(list(levels)).get_indexer(np.asarray(labels, dtype=object)).astype(np.int64)
    if codes.size and codes.min() < 0:
        raise RuntimeError(f"labels {sorted(set(np.asarray(labels, dtype=object)[codes < 0].tolist()))[:5]} are not among the "
                           "fitted levels")
    return codes


def bambi_formula_value(frame: pd.DataFrame, column: str | None) -> np.ndarray:
    """The column a group-level coefficient multiplies, as formulae evaluates it from the frame Bambi is handed: int64 ones
    for the varying intercept (``column`` None), the column's values as stored for a slope (formulae's ``Series.values``)."""
    if column is None:
        return np.ones(len(frame), dtype=int)
    return frame[column].values


def _bambi_term_class():
    from bambi.terms import GroupSpecificTerm

    class IndexGroupTerm(GroupSpecificTerm):
        """A Bambi group-specific term from level codes: ``label`` its formulae name (``1|g``, ``x|g``), ``levels`` the
        factor's sorted levels (Bambi's ``groups``), ``codes`` each row's level (Bambi's ``group_index``), ``predictor`` the
        column the coefficient multiplies, ``prior`` its Bambi prior, ``prefix`` the component's (a distributional formula's
        ``sigma``). No dense indicator exists: ``grouper`` and ``term`` are not offered."""

        def __init__(self, label: str, kind: str, levels, codes: np.ndarray, predictor: np.ndarray, prior, prefix: str = ""):
            self._hyperprior_alias = {}
            self._label, self._kind = label, kind
            self._groups = [str(v) for v in levels]
            self.group_index = codes
            self._predictor = predictor
            self.prior = prior
            self.prefix = prefix
            self._data = None

        @property
        def term(self):
            raise AttributeError("an IndexGroupTerm has no formulae term (group_index.IndexGroupTerm, docs/scaling.md)")

        @property
        def name(self):
            return f"{self.prefix}_{self._label}" if self.prefix else self._label

        @property
        def kind(self):
            return self._kind

        @property
        def shape(self):
            return (len(self.group_index), len(self._groups))

        @property
        def categorical(self):
            return False

        @property
        def groups(self):
            return self._groups

        @property
        def levels(self):
            return self._groups

        @property
        def predictor(self):
            return self._predictor

        @property
        def grouper(self):
            raise AttributeError("an IndexGroupTerm keeps level codes (group_index), not formulae's dense indicator")

    return IndexGroupTerm


_TERM_CLASS = None


def _term_class():
    """``IndexGroupTerm``, defined on first use (Bambi is imported lazily) under this module's name."""
    global _TERM_CLASS
    if _TERM_CLASS is None:
        _TERM_CLASS = _bambi_term_class()
        _TERM_CLASS.__module__, _TERM_CLASS.__qualname__ = __name__, "IndexGroupTerm"
    return _TERM_CLASS


def __getattr__(name: str):
    """``group_index.IndexGroupTerm`` (PEP 562), so that pickle finds the class of a term it restores."""
    if name == "IndexGroupTerm":
        return _term_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def index_group_term(label: str, kind: str, levels, codes, predictor, prior, prefix: str = ""):
    """An ``IndexGroupTerm``."""
    return _term_class()(label, kind, levels, codes, predictor, prior, prefix)


def bambi_prior(priors: dict, name: str):
    """The prior Bambi would give the group-specific term ``name`` from the prior dict it was handed: a copy of the entry (Bambi
    deep-copies the dict), its ``auto_scale`` off (Bambi's ``prepare_prior`` for a prior it is given)."""
    from copy import deepcopy

    prior = deepcopy(priors[name])
    prior.auto_scale = False
    return prior


# -- the private layout of Bambi this design relies on ---------------------------------------------------------------------
# sha256 of the source of each function or class whose behaviour IndexGroupTerm depends on, in Bambi 0.17.2
_BAMBI_SOURCES = {
    "bambi.backend.model_components.DistributionalComponent.build":
        "0144767ddcd8bcd99565899ff5eca69558fc4ceda9b303e7395226c711434f12",
    "bambi.backend.model_components.DistributionalComponent.build_group_specific_terms":
        "20a488b39bb9dedb9454e9c290c58890174d5f6b5c96ee0412abbf6871743e1b",
    "bambi.backend.terms.GroupSpecificTerm":
        "d5913e05f2c151377fc3e563cb958374fffd8b048631d29dc8027379ce3e5d19",
    "bambi.terms.group_specific.GroupSpecificTerm":
        "d2d544a126df12840849d5cafe082ef86fdb8018056a3aca3d977d01974a2c8c",
    "bambi.model_components.DistributionalComponent.group_specific_terms":
        "87508f04bcfc7310952dfa43bdf6a6b29f68ff8a128f00b98967453e274477af",
}


def _source_object(path: str):
    import importlib

    parts = path.split(".")
    for k in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:k]))
        except ImportError:
            continue
        for attr in parts[k:]:
            obj = getattr(obj, attr)
        return obj.fget if isinstance(obj, property) else obj
    raise ImportError(path)


def source_hash(path: str) -> str:
    return hashlib.sha256(inspect.getsource(_source_object(path)).encode("utf-8")).hexdigest()


def bambi_layout_problems() -> list[str]:
    """What differs from the Bambi this design was written against (empty when nothing does): its version, the sources of
    the functions and classes ``IndexGroupTerm`` relies on, and Bambi's ``SPARSE_DOT`` setting (off: the backend gathers)."""
    import bambi
    from bambi.config import config

    problems = []
    if bambi.__version__ != BAMBI_VERSION:
        problems.append(f"Bambi is {bambi.__version__}, not {BAMBI_VERSION}")
    for path, expected in _BAMBI_SOURCES.items():
        try:
            got = source_hash(path)
        except Exception as e:                         # noqa: BLE001 - a moved or renamed object is a problem to report
            problems.append(f"{path}: {type(e).__name__}: {e}")
            continue
        if got != expected:
            problems.append(f"{path} changed (source sha256 {got})")
    if config["SPARSE_DOT"]:
        problems.append("bambi.config['SPARSE_DOT'] is on: the backend would multiply a sparse Z the terms do not hold")
    return problems
