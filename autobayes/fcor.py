"""brms's fixed residual covariance ``fcor(M)`` (F03a, docs/fcor.md).

brms 2.23's ``fcor(M)`` gives the residuals of a Gaussian model a known covariance matrix ``M`` over the fitted rows, scaled
by the residual variance: with ``mu`` the linear predictor (every population- and group-level term),

    y ~ MultiNormal(mu, sigma^2 M),

brms's ``normal_fcor_hom(Y | mu, sigma, Lfcor)`` with ``Lfcor = cholesky_decompose(Mfcor)`` computed once from the data. ``M``
is data, handed over by name in ``data2`` (brms's ``data2 = list(M = ...)``), ``N`` by ``N`` for the ``N`` fitted rows in data
order (brms reads the matrix by position: its row and column names are not used); a vector stands for the diagonal matrix of
those variances, as in brms. brms refuses a matrix of another size and an asymmetric one, and Stan's ``cholesky_decompose``
one that is not positive definite: so does ``fcor_design``. The term has no parameter of its own and no prior class.

The log density is written row by row. With ``L`` the lower Cholesky factor of ``M`` and ``z = L^-1 (y - mu) / sigma`` (a
triangular solve),

    log p(y | mu, sigma) = sum_i [ -z_i^2 / 2 - log(sigma) - log(L_ii) - log(2 pi) / 2 ],

the ``i``-th term being the density of ``y_i`` given the rows before it (a sequential factorisation of the multivariate
normal), which is the elementwise log density PyMC's likelihood holds. It is not brms's ``log_lik()``, which conditions each
row on all the others: ``log_lik()`` and what reads it are refused for the term (divergence D145). ``fcor_values`` is the
same density in NumPy, from the dense covariance (``-(N log(2 pi) + log det S + r' S^-1 r) / 2``): the model checks its graph
against it when it is built, and the tests against SciPy.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from autobayes.formula import _IDENT, FormulaError, _split_top_level, data2_mapping

DOC = "docs/fcor.md"
_CALL_RE = re.compile(r"^fcor\s*\((.*)\)$", re.S)
_HALF_LOG_2PI = np.float64(0.5 * math.log(2.0 * math.pi))
SYMMETRY_TOLERANCE = 100 * np.finfo(np.float64).eps        # relative to M's largest entry: rounding, not asymmetry


@dataclass(frozen=True)
class Fcor:
    """A formula's ``fcor(M)`` term (F03a): ``matrix`` is the name of the covariance matrix in ``data2``. It is not a
    population-level term: no design column, no coefficient, no prior."""

    matrix: str

    @property
    def term(self) -> str:
        """The term as the canonical formula writes it: ``fcor(V)``."""
        return f"fcor({self.matrix})"


def parse_fcor(piece: str) -> Fcor:
    """brms's ``fcor(M)`` as written (``piece`` starts ``fcor(``): one argument, ``M`` or ``M = <name>``, the name of a
    matrix in ``data2``. Refused, naming the construct: no argument (brms: "Argument 'M' is missing in fcor()"), another
    argument or a second one (brms's ``fcor()`` takes ``M`` alone), and an ``M`` that is not a name (an expression)."""
    m = _CALL_RE.match(piece)
    if m is None:
        raise FormulaError(f"{piece!r}: brms's fixed residual covariance is written fcor(M), M the name of a matrix in data2 ({DOC})")
    inside = m.group(1).strip()
    args = [a.strip() for a in _split_top_level(inside, ",")] if inside else []
    if not args:
        raise FormulaError(f"{piece!r}: fcor() needs the matrix M, the name of a matrix in data2 (brms: \"Argument 'M' is missing "
                           f"in fcor()\"; {DOC})")
    if len(args) > 1:
        raise FormulaError(f"{piece!r}: fcor() takes one argument, M, the name of a matrix in data2 (brms's fcor(M); {DOC})")
    arg = args[0]
    named = re.fullmatch(rf"({_IDENT})\s*=(?!=)\s*(.+)", arg, re.S)
    if named is not None:
        if named.group(1) != "M":
            raise FormulaError(f"{piece!r}: fcor() has no argument {named.group(1)!r}; brms's fcor() takes M ({DOC})")
        arg = named.group(2).strip()
    if not re.fullmatch(_IDENT, arg):
        raise FormulaError(f"{piece!r}: M = {arg}: fcor() reads its matrix from data2 by name; compute the matrix, put it in "
                           f"data2 and name it (data2={{'M': matrix}}, fcor(M); {DOC})")
    return Fcor(arg)


def check_fcor(spec, fam, pieces, formula: str) -> None:
    """The refusals of ``fcor()`` that depend on the rest of the formula, ``spec`` (the parsed ``ModelSpec``, its term in
    ``spec.fcor``; ``pieces`` every ``fcor()`` piece as written): one term per formula; the family must take it
    (``FamilySpec.fcor``: the Gaussian family, verified against frozen brms cases; a mixture, as brms refuses it too); and
    none of the constructs no frozen case verifies with it (divergence D144): response addition terms (brms refuses
    ``weights()`` with it too), an ARMA term, a SAR term, ``mo()``, ``gp()``, ``me()``, ``mi()``, ``mm()``, ``gr(by = )`` and
    ``cs()``."""
    from autobayes.families import FAMILIES

    term = spec.fcor
    if len(pieces) > 1:
        raise FormulaError(f"{formula!r}: {pieces[0]!r} and {pieces[1]!r}: one fcor() term per formula (brms: 'Can only model one "
                           f"term per autocorrelation class'; {DOC})")
    if fam.mixture:                                          # brms refuses it too (not a divergence)
        raise FormulaError(f"{formula!r}: {term.term} in a mixture is not supported, as in brms ('Explicit covariance terms can "
                           f"only be specified on mu': a mixture's linear predictors are mu1, mu2, ...; {DOC})")
    if not fam.fcor:
        offering = [name for name, other in FAMILIES.items() if other.fcor]
        raise FormulaError(
            f"{formula!r}: {term.term} is not available for the {fam.name} family; the fixed residual covariance is verified "
            f"here for the {', '.join(offering) or 'no'} famil{'y' if len(offering) == 1 else 'ies'} (brms also takes it for "
            f"student(), whose Student-t version is not verified here, and refuses it for the others: 'FCOR terms are not "
            f"implemented for this family'; {DOC}, divergence D144)")
    if spec.additions:
        raise FormulaError(f"{formula!r}: {term.term} with the response addition term(s) {', '.join(a.term for a in spec.additions)} "
                           f"is not supported (brms refuses weights() with it too: 'Invalid addition arguments for this model'; "
                           f"{DOC}, divergence D144)")
    others = ([f"the autocorrelation term {spec.autocor.term!r}"] if spec.autocor is not None else [])
    others += [f"the SAR term {spec.sar.term!r}"] if spec.sar is not None else []          # F02b
    others += [f"the monotonic term {t.name!r}" for t in spec.mo_terms]
    others += [f"the Gaussian process {g.term!r}" for g in spec.gp_terms]
    others += [f"the measurement-error term {t.name!r}" for t in spec.me_terms]
    others += [f"the missing-value term {t.name!r}" for t in spec.mi_terms]
    others += [f"the group term '{t.canonical}'" for t in spec.custom_group_terms]
    others += [f"the category-specific term {t.name!r}" for t in spec.cs_terms]
    if others:
        raise FormulaError(f"{formula!r}: {term.term} next to {others[0]} is not supported in this slice: no frozen brms case "
                           f"verifies the pair ({DOC}, divergence D144)")


# -- the matrix (data2 itself is checked by formula.data2_mapping and formula.check_data2, shared with sar()) ------------------
@dataclass(frozen=True)
class FcorDesign:
    """The covariance of an ``fcor(M)`` term on the fitted rows (F03a): ``M`` the ``(n, n)`` matrix as given (brms's
    ``Mfcor``; a vector given as its diagonal), ``L`` its lower Cholesky factor (brms's ``Lfcor``), both in data order."""

    term: str
    name: str
    M: np.ndarray
    L: np.ndarray

    @property
    def n(self) -> int:
        return int(self.M.shape[0])

    @property
    def log_diagonal(self) -> np.ndarray:
        """``log(L_ii)``, each row's share of ``log det(M) / 2``."""
        return np.log(np.diag(self.L))


def fcor_design(term: Fcor, data2, n: int) -> FcorDesign:
    """The matrix ``term`` names, read from ``data2`` (a mapping of names to matrices: a ``pandas.DataFrame``, a NumPy array
    or nested lists) for ``n`` fitted rows, checked as brms and Stan check it: numeric and finite, a vector of ``n``
    variances (the diagonal, as brms reads a vector) or an ``n`` by ``n`` matrix (brms: "Dimensions of 'M' for FCOR terms must
    be equal to the number of observations"), symmetric to ``SYMMETRY_TOLERANCE`` of its largest entry (brms: "'M' for FCOR
    terms must be symmetric") and positive definite (Stan's ``cholesky_decompose``). Labels are not read: row ``i`` of the
    matrix is the ``i``-th fitted row, as in brms."""
    name = term.matrix
    data2_mapping(data2)
    if data2 is None or name not in data2:
        held = sorted(data2) if data2 is not None else []
        raise FormulaError(f"{term.term} reads the matrix {name!r} from data2, which " + (f"holds {held}" if held else "is not given")
                           + f": pass data2={{{name!r}: matrix}} (fcor() reads its matrix from data2; brms: \"Object '{name}' was not "
                           f"found in 'data2'\"; {DOC})")
    value = data2[name]
    arr = np.asarray(value.to_numpy() if hasattr(value, "to_numpy") else value)
    if arr.dtype.kind not in "iuf":
        raise FormulaError(f"data2[{name!r}] of {term.term} has dtype {arr.dtype}; the covariance matrix must be numeric ({DOC})")
    arr = arr.astype(np.float64)
    if arr.ndim == 1:                                        # brms: a vector gives the variances, the covariances zero
        arr = np.diag(arr)
    if arr.ndim != 2 or arr.shape != (n, n):
        raise FormulaError(f"data2[{name!r}] of {term.term} has shape {arr.shape}; it must be {n} x {n}, one row and column per fitted "
                           f"row, or a vector of {n} variances (brms: \"Dimensions of 'M' for FCOR terms must be equal to the number "
                           f"of observations\"; {DOC})")
    if not np.isfinite(arr).all():
        raise FormulaError(f"data2[{name!r}] of {term.term} has missing or non-finite entries ({DOC})")
    if np.max(np.abs(arr - arr.T)) > SYMMETRY_TOLERANCE * np.max(np.abs(arr)):
        raise FormulaError(f"data2[{name!r}] of {term.term} is not symmetric (brms: \"'M' for FCOR terms must be symmetric\"; {DOC})")
    try:
        L = np.linalg.cholesky(arr)
    except np.linalg.LinAlgError:
        raise FormulaError(f"data2[{name!r}] of {term.term} is not positive definite: it has no Cholesky factor (Stan's "
                           f"cholesky_decompose refuses it when brms runs the model; {DOC})") from None
    return FcorDesign(term=term.term, name=name, M=arr, L=L)


# -- the density --------------------------------------------------------------------------------------------------------
def fcor_logp(value, mu, sigma, design: FcorDesign):
    """Each row's term of ``log MultiNormal(value | mu, sigma^2 M)`` as a PyTensor graph (module docstring): ``(n,)``, summing
    to the log density, with ``sigma`` the residual SD (a scalar)."""
    import pytensor.tensor as pt
    from pytensor.tensor.slinalg import solve_triangular

    z = solve_triangular(pt.as_tensor(design.L), value - mu, lower=True) / sigma
    return -0.5 * pt.square(z) - pt.log(sigma) - pt.as_tensor(design.log_diagonal) - _HALF_LOG_2PI


def fcor_values(y, mu, sigma, design: FcorDesign) -> np.ndarray:
    """``log MultiNormal(y | mu, sigma^2 M)`` in NumPy from the dense covariance ``S = sigma^2 M``, per draw: ``mu`` is
    ``(..., n)`` and ``sigma`` broadcasts against its leading axes; returns ``(...)``."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    r = np.asarray(y, dtype=float) - mu
    _, logdet = np.linalg.slogdet(design.M)
    solved = np.linalg.solve(design.M, r.reshape(-1, design.n).T).T.reshape(r.shape)        # S^-1 r = M^-1 r / sigma^2
    quadratic = np.sum(r * solved, axis=-1)
    return -0.5 * (design.n * math.log(2.0 * math.pi) + logdet + 2.0 * design.n * np.log(sigma) + quadratic / sigma ** 2)


def fcor_draws(mu, sigma, design: FcorDesign, rng) -> np.ndarray:
    """Draws of ``y ~ MultiNormal(mu, sigma^2 M)``: ``mu + sigma * L z``, ``z`` standard normal from ``rng`` in ``mu``'s shape
    (``(..., n)``; ``sigma`` broadcasts against the leading axes, or against ``mu`` when it already has a row axis)."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    z = rng.standard_normal(mu.shape)
    return mu + (sigma if sigma.ndim == mu.ndim else sigma[..., None]) * (z @ design.L.T)


def fcor_family(base, design: FcorDesign):
    """The Bambi family ``base`` (a built-in family's name or a family object; the Gaussian) whose likelihood is
    ``MultiNormal(mu, sigma^2 M)`` over the fitted rows: a ``pymc.CustomDist`` with the rows' terms of ``fcor_logp`` as its
    elementwise log density and ``fcor_draws`` as its random draws. Everything else about the family (its parameters, link,
    priors and the hooks on its linear predictor) is the base family's."""
    import copy

    import bambi as bmb
    import pymc as pm
    from bambi.defaults import get_builtin_family

    family = get_builtin_family(base) if isinstance(base, str) else copy.copy(base)
    lk = family.likelihood
    if tuple(lk.params) != ("mu", "sigma"):
        raise ValueError(f"fcor() needs the Gaussian likelihood's parameters (mu, sigma); this family's are {tuple(lk.params)}")

    def logp(value, mu, sigma):
        return fcor_logp(value, mu, sigma, design)

    def random(mu, sigma, rng=None, size=None):
        return fcor_draws(np.broadcast_to(mu, np.shape(mu) if size is None else tuple(size)), sigma, design, rng)

    class FcorNormal:
        """``y ~ MultiNormal(mu, sigma^2 M)`` over the fitted rows (``fcor()``), as the likelihood Bambi builds."""

        def __new__(cls, name, *, mu, sigma, observed=None, dims=None):
            return pm.CustomDist(name, mu, sigma, logp=logp, random=random, observed=observed, dims=dims)

    family.likelihood = bmb.Likelihood(lk.name, params=lk.params, parent=lk.parent, dist=FcorNormal)
    return family
