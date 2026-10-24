"""brms's residual correlation structures over time within groups: ``unstr(time, gr)`` and ``cosy(time, gr)`` (F03b,
docs/unstr-cosy.md).

The rows form groups, the levels of ``gr``, each ordered by ``time``; brms's Stan data list them sorted by group and then
time (``N_tg`` groups, ``begin_tg``, ``end_tg``, ``nobs_tg``). The residuals ``e = y - mu`` of different groups are
independent, and those of one group are jointly

    e_g ~ MultiNormal(0, sigma^2 Cor_g)                (Gaussian)
    e_g ~ MultiStudentT(nu, 0, sigma^2 Cor_g)          (Student-t: the scale matrix sigma^2 Cor_g, one nu)

so that each row is marginally the family's own ``Normal(mu_i, sigma)`` or ``StudentT(nu, mu_i, sigma)``. ``Cor_g`` is

- ``unstr``: the rows and columns of the group's times in one correlation matrix over every distinct time of the fitted rows
  (brms's ``Cortime``, ``n_unique_t`` by ``n_unique_t``, the times sorted; ``Jtime_tg`` each group's time indices). It is
  sampled through its Cholesky factor ``Lcortime`` (``cholesky_factor_corr``, Stan's ``cholesky_corr`` coordinates) under
  ``lkj_corr_cholesky(eta)`` (prior class ``cortime``, brms's default ``lkj(1)``). A group with every time uses ``Lcortime``
  itself; another the Cholesky factor of its rows and columns of ``Cortime``.
- ``cosy``: compound symmetry, every pair of a group's rows correlated ``cosy`` (``real<lower=0,upper=1>``, prior class
  ``cosy``, brms's default flat: uniform on (0, 1)); time only orders the rows. brms's ``Lcortime`` is the Cholesky factor of
  the compound-symmetric matrix of the largest group's size, and a group of ``k`` rows uses its leading ``k`` by ``k`` block.

The log density is evaluated per pattern (the groups that share their time indices share one Cholesky factor): the
residuals of those groups, solved against the factor, give each group's quadratic form ``q`` and log determinant, and the
family's kernel (``NORMAL_TIME``, ``STUDENT_TIME``; ``FamilySpec.time_correlation``) turns them into the multivariate
density. brms's ``log_lik()`` of such a model is each row's log density given its group's other rows (the leave-one-out
predictive of Buerkner, Gabry and Vehtari 2021, "Efficient leave-one-out cross-validation for Bayesian non-factorized normal
and Student-t models", Computational Statistics 36): with ``P = (sigma^2 Cor_g)^-1``, ``g = P e`` and ``c = diag(P)``, row
``i`` is ``Normal(e_i | e_i - g_i / c_i, 1 / sqrt(c_i))``; for the Student-t, the conditional of the multivariate t,
``StudentT(nu + k - 1, e_i - g_i / c_i, sqrt((nu + q_-i) / (nu + k - 1) / c_i))`` with ``q_-i = e' P e - g_i^2 / c_i`` the
quadratic form of the group's other rows.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from autobayes.formula import _IDENT, FormulaError, _split_top_level, categorical_labels, reject_complex

DOC = "docs/unstr-cosy.md"
CONTEXTS = "D146"                 # the contexts no frozen case verifies (new rows' predictions among them), refused by name
STRUCTURES = {"unstr": "an unstructured residual correlation over the times", "cosy": "compound symmetry"}
_CALL_RE = re.compile(r"^(unstr|cosy)\s*\((.*)\)$", re.S)
_FORMALS = ("time", "gr")
_HALF_LOG_2PI = np.float64(0.5 * math.log(2.0 * math.pi))


@dataclass(frozen=True)
class TimeCor:
    """A formula's residual correlation term, brms's ``unstr(time, gr)`` or ``cosy(time, gr)`` (F03b). Like an ARMA term it
    has no design column and no ``b`` coefficient: ``unstr`` has the correlation matrix ``Cortime`` (prior class
    ``cortime``), ``cosy`` the correlation ``cosy`` (prior class ``cosy``). ``time`` orders each group's rows and ``gr``'s
    levels are the groups."""

    call: str
    time: str
    gr: str

    @property
    def term(self) -> str:
        """The term as the canonical formula writes it: ``unstr(time = tt, gr = subj)``."""
        return f"{self.call}(time = {self.time}, gr = {self.gr})"

    @property
    def columns(self) -> tuple[str, str]:
        """The data columns the term reads: its time column and its grouping column."""
        return (self.time, self.gr)

    @property
    def prior_class(self) -> str:
        """brms's prior class of the term's parameter: ``cortime`` (the correlation matrix of ``unstr``) or ``cosy``."""
        return "cortime" if self.call == "unstr" else "cosy"


def parse_timecor(piece: str, formula: str) -> TimeCor:
    """brms's ``unstr(time, gr)`` or ``cosy(time, gr)`` as written in ``formula`` (``piece`` starts with one of them). The
    arguments follow R's rules, named ones first, then the positional ones in order; both must name one column each.
    Refused, naming the construct: an argument the function does not have, one given twice, too many, a missing ``time`` or
    ``gr`` (brms's ``cosy()`` defaults to the data order and one group, which no frozen case verifies: divergence D146), a
    ``gr`` of two columns (``g:h``) or an expression, and ``time`` equal to ``gr``."""
    m = _CALL_RE.match(piece)
    if m is None:
        raise FormulaError(f"{piece!r}: a residual correlation term is written unstr(time, gr) or cosy(time, gr) ({DOC})")
    call, inside = m.group(1), m.group(2).strip()
    named, positional = {}, []
    for arg in (_split_top_level(inside, ",") if inside else []):
        arg = arg.strip()
        km = re.fullmatch(rf"({_IDENT})\s*=(?!=)\s*(.+)", arg, re.S)
        if km is None:
            positional.append(arg)
            continue
        name, value = km.group(1), km.group(2).strip()
        if name not in _FORMALS:
            raise FormulaError(f"{piece!r}: {call}() has no argument {name!r}; brms's {call}() takes time, gr")
        if name in named:
            raise FormulaError(f"{piece!r}: the argument {name!r} is given twice")
        named[name] = value
    rest = [f for f in _FORMALS if f not in named]
    if len(positional) > len(rest):
        raise FormulaError(f"{piece!r}: too many arguments; brms's {call}() takes time, gr")
    named.update(zip(rest, positional))
    for name in _FORMALS:
        value = named.get(name, "NA")
        if value == "NA":
            raise FormulaError(f"{piece!r}: {call}() needs its {name} column here (brms's cosy() defaults to the data order and "
                               f"one group; no frozen case verifies it): name it, {call}(time = <column>, gr = <column>) ({DOC}, "
                               f"divergence {CONTEXTS})")
        if not re.fullmatch(_IDENT, value):
            hint = (" (brms's gr = g:h groups the rows by two columns; make one column of the combined labels and name it)"
                    if name == "gr" else "")
            raise FormulaError(f"{piece!r}: {name} = {value}: {name} must be one column name here{hint} ({DOC}, divergence "
                               f"{CONTEXTS})")
    if named["time"] == named["gr"]:
        raise FormulaError(f"{piece!r}: the time column cannot also be the grouping column gr")
    return TimeCor(call=call, time=named["time"], gr=named["gr"])


def check_timecor(term: TimeCor, piece: str, *, response: str, trials: str | None, fam, additions, formula: str,
                  others: list[str]) -> None:
    """The refusals of a residual correlation term that depend on the rest of the formula: its columns may be predictors or
    grouping factors but not the response or the trials column; the family must take the term (``FamilySpec.
    time_correlation``: the Gaussian and Student-t families, verified against frozen brms cases; a mixture, which brms
    refuses too: its linear predictors are ``mu1``, ``mu2``, ...); no response addition term and none of ``others`` (the
    Gaussian processes, ``me()`` / ``mi()`` terms, autocorrelation, SAR, CAR and ``fcor()`` terms of the formula) next to it
    (divergence D146)."""
    from autobayes.families import FAMILIES

    for column in term.columns:
        if column == response or (trials is not None and column == trials):
            raise FormulaError(f"{piece!r}: the {'response' if column == response else 'trials column'} {column!r} cannot be the "
                               "time or grouping column of a residual correlation term")
    if fam.mixture:
        raise FormulaError(f"{formula!r}: a residual correlation term in a mixture ({term.term}) is not supported, as in brms "
                           f"('Explicit covariance terms can only be specified on mu'); {DOC}")
    if fam.time_correlation is None:
        offering = [name for name, other in FAMILIES.items() if other.time_correlation is not None]
        raise FormulaError(f"{formula!r}: the residual correlation term {term.term!r} is not available for the {fam.name} family; "
                           f"brms's {term.call}() is verified here for the {', '.join(offering)} families ({DOC}, divergence "
                           f"{CONTEXTS})")
    if additions:
        raise FormulaError(f"{formula!r}: the residual correlation term {term.term!r} with the response addition term(s) "
                           f"{', '.join(a.term for a in additions)} is not supported in this slice: no frozen brms case verifies "
                           f"the pair ({DOC}, divergence {CONTEXTS})")
    if others:
        raise FormulaError(f"{formula!r}: the residual correlation term {term.term!r} next to {others[0]} is not supported in this "
                           f"slice: no frozen brms case verifies the pair ({DOC}, divergence {CONTEXTS})")


# -- the groups of the fitted rows -------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TimeCorDesign:
    """The groups of a residual correlation term on the fitted rows (F03b). ``groups`` are the groups' labels, sorted;
    ``group_index`` (n,) each fitted row's group and ``index`` (n,) its row of the correlation matrix (``unstr``: its time's
    0-based index among ``times``, the fitted rows' distinct times sorted; ``cosy``: its 0-based position in its group in time
    order), both in data order; ``order`` (n,) the fitted rows as brms's Stan data list them, sorted by group and then time
    (data row per Stan row); ``nobs`` (N_tg,) each group's number of rows in group order; ``patterns`` the groups that share
    their rows of the correlation matrix: ``(index tuple, rows)`` with ``rows`` the ``(groups, k)`` data rows of each such
    group in time order, in the order of the patterns' first groups."""

    term: str
    call: str
    groups: tuple[str, ...]
    times: tuple[float, ...]
    group_index: np.ndarray
    index: np.ndarray
    order: np.ndarray
    nobs: np.ndarray
    patterns: tuple

    @property
    def n(self) -> int:
        return int(self.order.size)

    @property
    def dim(self) -> int:
        """The size of the correlation matrix: ``unstr``'s number of distinct times (brms's ``n_unique_t``), ``cosy``'s largest
        group (brms's ``max(nobs_tg)``)."""
        return len(self.times) if self.call == "unstr" else int(self.nobs.max())

    @property
    def begin(self) -> np.ndarray:
        """Each group's first Stan row, 0-based (brms's ``begin_tg - 1``)."""
        return np.concatenate([[0], np.cumsum(self.nobs)[:-1]]).astype(np.int64)

    @property
    def Jtime(self) -> np.ndarray:
        """brms's ``Jtime_tg`` (``unstr``): ``(N_tg, max(nobs))``, each group's 1-based time indices in time order, then 0."""
        out = np.zeros((len(self.groups), int(self.nobs.max())), dtype=np.int64)
        for g, (b, k) in enumerate(zip(self.begin, self.nobs)):
            out[g, :k] = self.index[self.order[b:b + k]] + 1
        return out

    @property
    def labels(self) -> list[str]:
        """The coordinates of the correlation matrix's rows: ``unstr``'s times as text (the shortest decimal that reads back
        as the time), ``cosy``'s positions ``"1"``..``"dim"``."""
        if self.call == "unstr":
            return [np.format_float_positional(t, trim="-") for t in self.times]
        return [str(k) for k in range(1, self.dim + 1)]


def time_layout(term: TimeCor, df: pd.DataFrame) -> TimeCorDesign:
    """The groups of ``term`` on the rows of ``df`` (the fitted frame), as brms's Stan data list them: the rows sorted by group
    label and then by time. The grouping column is read as labels (text, or whole numbers; no missing values), the time column
    must be numeric and finite (brms orders any R vector, text as text; divergence D146), and two rows of a group at the same
    time are refused, as brms refuses them."""
    n = len(df)
    for column in term.columns:
        if column not in df.columns:
            raise FormulaError(f"columns not in data: [{column!r}] (the residual correlation term {term.term!r} reads it)")
    labels, _ = categorical_labels(f"grouping column {term.gr!r} of {term.term!r}", df[term.gr])
    s = df[term.time]
    reject_complex(f"time column {term.time!r}", s)
    if not (pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)):
        raise FormulaError(f"time column {term.time!r} of {term.term!r} has dtype {s.dtype}; it must be numeric (brms orders any "
                           f"column, text as text; compute a numeric time and name it: {DOC}, divergence {CONTEXTS})")
    time = s.to_numpy(dtype=float)
    if not np.isfinite(time).all():
        raise FormulaError(f"time column {term.time!r} of {term.term!r} has missing or non-finite values")
    groups = tuple(sorted(set(labels.tolist())))
    lookup = {label: k for k, label in enumerate(groups)}
    group_index = np.array([lookup[label] for label in labels], dtype=np.int64)
    order = np.lexsort((time, group_index)).astype(np.int64)        # by group, then by time
    g_sorted, t_sorted = group_index[order], time[order]
    tie = np.flatnonzero((g_sorted[1:] == g_sorted[:-1]) & (t_sorted[1:] == t_sorted[:-1]))
    if tie.size:
        i, j = order[tie[0]], order[tie[0] + 1]
        raise FormulaError(f"{term.term!r}: rows {int(i)} and {int(j)} of group {groups[g_sorted[tie[0]]]!r} have the same time "
                           f"{float(t_sorted[tie[0]])!r}; time points within a group must be unique (brms's rule too)")
    nobs = np.bincount(group_index, minlength=len(groups)).astype(np.int64)
    starts = np.r_[True, g_sorted[1:] != g_sorted[:-1]]
    run_start = np.maximum.accumulate(np.where(starts, np.arange(n), 0))
    position = np.empty(n, dtype=np.int64)
    position[order] = np.arange(n) - run_start
    if term.call == "unstr":
        times = tuple(float(t) for t in np.unique(time))
        index = np.searchsorted(np.asarray(times), time).astype(np.int64)
    else:
        times, index = (), position
    found: dict[tuple, list] = {}
    begin = 0
    for k in nobs:
        rows = order[begin:begin + k]
        found.setdefault(tuple(int(i) for i in index[rows]), []).append(rows)
        begin += k
    patterns = tuple((key, np.array(rows, dtype=np.int64)) for key, rows in found.items())
    return TimeCorDesign(term=term.term, call=term.call, groups=groups, times=times, group_index=group_index, index=index,
                         order=order, nobs=nobs, patterns=patterns)


# -- the correlation's Cholesky factor -----------------------------------------------------------------------------------
def cosy_cholesky_numpy(rho, dim: int) -> np.ndarray:
    """The Cholesky factor of the ``dim`` by ``dim`` compound-symmetric correlation matrix ``(1 - rho) I + rho 1 1'``
    (NumPy, any leading shape of ``rho``): diagonal ``d_j = sqrt((1 - rho)(1 + j rho) / (1 + (j - 1) rho))`` and every entry
    below the diagonal of column ``j`` ``rho (1 - rho) / ((1 + (j - 1) rho) d_j)``, ``j = 0..dim-1`` (the ratios of the
    leading minors ``(1 - rho)^(k - 1) (1 + (k - 1) rho)``)."""
    rho = np.asarray(rho, dtype=float)[..., None]
    j = np.arange(dim, dtype=float)
    den = 1.0 + (j - 1.0) * rho
    d = np.sqrt((1.0 - rho) * (1.0 + j * rho) / den)
    c = rho * (1.0 - rho) / (den * d)
    below = np.tril(np.ones((dim, dim)), -1)
    return below * c[..., None, :] + d[..., None, :] * np.eye(dim)


def cosy_cholesky(rho, dim: int):
    """``cosy_cholesky_numpy`` as a PyTensor graph of the scalar ``rho``."""
    import pytensor.tensor as pt

    j = np.arange(dim, dtype=np.float64)
    den = 1.0 + (j - 1.0) * rho
    d = pt.sqrt((1.0 - rho) * (1.0 + j * rho) / den)
    c = rho * (1.0 - rho) / (den * d)
    return np.tril(np.ones((dim, dim)), -1) * c[None, :] + pt.diag(d)


def _is_leading(idx: tuple) -> bool:
    return idx == tuple(range(len(idx)))


# -- the families' kernels -----------------------------------------------------------------------------------------------
class NormalTime:
    """The Gaussian family's kernel (``FamilySpec.time_correlation``): ``e_g ~ MultiNormal(0, sigma^2 Cor_g)``."""

    name = "MultiNormal"
    parameters: tuple[str, ...] = ()

    @staticmethod
    def logp(q, k: int, groups: int, logdet):
        """The log density of ``groups`` groups of ``k`` rows (PyTensor): ``q`` their quadratic forms ``e' Sigma^-1 e``,
        ``logdet`` ``log det(chol(Sigma))``, the same for each."""
        import pytensor.tensor as pt

        return -0.5 * pt.sum(q) - np.float64(groups) * (logdet + np.float64(k) * _HALF_LOG_2PI)

    @staticmethod
    def parameter_draws(model, post) -> tuple:
        """The draws of the kernel's parameters besides mu and sigma: none."""
        return ()

    @staticmethod
    def loo(g, c, q, k: int):
        """Each row's log density given its group's other rows (NumPy), from ``g = P e``, ``c = diag(P)`` and the group's
        quadratic form ``q``: ``Normal(e_i | e_i - g_i / c_i, 1 / sqrt(c_i))``, whose deviation is ``g_i / c_i``."""
        return -_HALF_LOG_2PI + 0.5 * np.log(c) - 0.5 * g * g / c

    @staticmethod
    def scale_draws(rng, shape):
        """The factors the standard-normal draws are multiplied by: ones."""
        return np.ones(shape)


class StudentTime:
    """The Student-t family's kernel: ``e_g ~ MultiStudentT(nu, 0, sigma^2 Cor_g)``, the scale matrix ``sigma^2 Cor_g`` (each
    row marginally ``StudentT(nu, mu_i, sigma)``)."""

    name = "MultiStudentT"
    parameters: tuple[str, ...] = ("nu",)

    @staticmethod
    def logp(q, k: int, groups: int, logdet, nu):
        """As ``NormalTime.logp``, the multivariate t's: ``lgamma((nu + k) / 2) - lgamma(nu / 2) - k / 2 log(nu pi) - logdet -
        (nu + k) / 2 log(1 + q / nu)`` per group."""
        import pytensor.tensor as pt

        half = 0.5 * (nu + np.float64(k))
        const = pt.gammaln(half) - pt.gammaln(0.5 * nu) - np.float64(0.5 * k) * pt.log(nu * np.float64(math.pi)) - logdet
        return pt.sum(-half * pt.log1p(q / nu)) + np.float64(groups) * const

    @staticmethod
    def parameter_draws(model, post) -> tuple:
        """The draws of nu, ``(chain, draw)``, or the model's fixed nu."""
        return (post[model.names["nu"]].to_numpy() if model.nu is None else np.float64(model.nu),)

    @staticmethod
    def loo(g, c, q, k: int, nu):
        """The conditional of the multivariate t: ``StudentT(nu + k - 1, e_i - g_i / c_i, sqrt((nu + q_-i) / (nu + k - 1) /
        c_i))`` with ``q_-i = q - g_i^2 / c_i``."""
        from scipy.special import gammaln

        nu = np.asarray(nu, dtype=float)[..., None, None]
        df = nu + (k - 1.0)
        rest = q[..., None] - g * g / c
        scale2 = (nu + rest) / df / c
        z2 = (g / c) ** 2 / scale2
        return (gammaln(0.5 * (df + 1.0)) - gammaln(0.5 * df) - 0.5 * np.log(df * math.pi) - 0.5 * np.log(scale2)
                - 0.5 * (df + 1.0) * np.log1p(z2 / df))

    @staticmethod
    def scale_draws(rng, shape, nu):
        """``1 / sqrt(w)``, ``w ~ chi^2(nu) / nu``, one per group: a multivariate normal draw times it is a multivariate t."""
        nu = np.broadcast_to(np.asarray(nu, dtype=float)[..., None], shape[:-1])     # per draw, the same for each group
        return np.broadcast_to(1.0 / np.sqrt(rng.chisquare(nu)[..., None] / nu[..., None]), shape)


NORMAL_TIME = NormalTime()
STUDENT_TIME = StudentTime()


# -- the model's graph ---------------------------------------------------------------------------------------------------
def log_density(y, mu, sigma, extra, value, design: TimeCorDesign, kernel):
    """The log density of the fitted responses ``y`` (PyTensor): ``mu`` the ``(n,)`` linear predictor (data order), ``sigma``
    the scale, ``extra`` the kernel's other parameters (the Student-t's nu), ``value`` the term's parameter (``unstr``: the
    Cholesky factor ``Lcortime``; ``cosy``: the correlation ``cosy``). Per pattern: the factor of its rows of the correlation
    (the leading block of ``Lcortime``, which is ``Lcortime`` itself for a group with every time, or the Cholesky factor of
    its rows and columns of ``Cortime``), the groups' residuals solved against it, and the kernel's density of their
    quadratic forms."""
    import pytensor.tensor as pt
    from pytensor.tensor.slinalg import cholesky, solve_triangular

    L = value if design.call == "unstr" else cosy_cholesky(value, design.dim)
    r = pt.as_tensor_variable(y) - mu
    cor, total, log_sigma = None, 0.0, pt.log(sigma)
    for idx, rows in design.patterns:
        k = len(idx)
        if _is_leading(idx):
            Lp = L[:k, :k]
        else:
            if cor is None:
                cor = L @ L.T
            sel = np.asarray(idx, dtype=np.int64)
            Lp = cholesky(cor[sel][:, sel], lower=True)
        Z = solve_triangular(Lp, r[rows].T, lower=True)                      # (k, groups)
        q = pt.sum(Z * Z, axis=0) / (sigma * sigma)
        logdet = pt.sum(pt.log(pt.diagonal(Lp))) + np.float64(k) * log_sigma
        total = total + kernel.logp(q, k, rows.shape[0], logdet, *extra)
    return total


def _block_factors(L: np.ndarray, design: TimeCorDesign):
    """Per pattern (NumPy): ``(idx, rows, factor, correlation)``, the factor of its rows of the correlation and the rows and
    columns of the correlation, each with the leading shape of ``L``."""
    C = L @ np.swapaxes(L, -1, -2)
    for idx, rows in design.patterns:
        sel = np.asarray(idx, dtype=np.int64)
        Cp = C[..., sel[:, None], sel[None, :]]
        Lp = L[..., :len(idx), :len(idx)] if _is_leading(idx) else np.linalg.cholesky(Cp)
        yield idx, rows, Lp, Cp


def structure_draws(model, post) -> np.ndarray:
    """The draws of the Cholesky factor of the correlation, ``(chain, draw, dim, dim)``: ``unstr``'s ``Lcortime``, ``cosy``'s
    factor of its draws of ``cosy``."""
    roles, design = model.timecor_names, model.design.timecor
    if design.call == "unstr":
        return post[roles["L"]].transpose("chain", "draw", roles["coordinate"], roles["coordinate_T"]).to_numpy()
    return cosy_cholesky_numpy(post[roles["cosy"]].to_numpy(), design.dim)


def log_lik(model, post, eta: np.ndarray) -> np.ndarray:
    """brms's ``log_lik()`` (C07) of a model with the term, ``(chain, draw, row)`` in data order: each row's log density given
    its group's other rows at the draws' linear predictor ``eta`` (``(chain, draw, row)``), scale, correlation and kernel
    parameters (the module docstring)."""
    design, kernel = model.design.timecor, model.family_spec.time_correlation()
    sigma = post[model.names["sigma"]].to_numpy()
    extra = kernel.parameter_draws(model, post)
    y = np.asarray(model.design.y, dtype=float)
    out = np.empty(eta.shape, dtype=float)
    for idx, rows, _, Cp in _block_factors(structure_draws(model, post), design):
        P = np.linalg.inv(sigma[..., None, None] ** 2 * Cp)                    # (chain, draw, k, k)
        e = y[rows] - eta[..., rows]                                         # (chain, draw, groups, k)
        g = np.einsum("...ij,...gj->...gi", P, e)
        c = np.diagonal(P, axis1=-2, axis2=-1)[..., None, :]
        q = np.sum(e * g, axis=-1)
        out[..., rows] = kernel.loo(g, c, q, len(idx), *extra)
    return out


def draw_rows(mu, sigma, extra, L, design: TimeCorDesign, kernel, rng) -> np.ndarray:
    """New responses (NumPy), every group's jointly: ``mu + sigma L_p z`` (times the kernel's scale draws, the Student-t's
    ``1 / sqrt(w)``) with ``z`` standard normal, ``L_p`` the factor of the group's rows of the correlation. ``mu`` is ``(...,
    n)``, ``sigma`` and each of ``extra`` ``(...)``, ``L`` ``(..., dim, dim)``."""
    mu = np.asarray(mu, dtype=float)
    lead = mu.shape[:-1]
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), lead)
    out = np.empty_like(mu)
    for idx, rows, Lp, _ in _block_factors(np.broadcast_to(L, lead + np.shape(L)[-2:]), design):
        shape = lead + rows.shape
        z = rng.standard_normal(shape)
        e = np.einsum("...ij,...gj->...gi", Lp, z) * sigma[..., None, None]
        out[..., rows] = mu[..., rows] + e * kernel.scale_draws(rng, shape, *extra)
    return out


def draw_response(model, post, mu: np.ndarray, rng) -> np.ndarray:
    """``predict(kind="response")`` of a model with the term (brms's ``posterior_predict``): per draw, each group's rows drawn
    jointly around the draw's ``mu`` (``(chain, draw, row)``, the fitted rows)."""
    kernel = model.family_spec.time_correlation()
    sigma = post[model.names["sigma"]].to_numpy()
    return draw_rows(mu, sigma, kernel.parameter_draws(model, post), structure_draws(model, post), model.design.timecor, kernel, rng)


def lkj_cholesky_variable(name: str, eta: float, dim: int, dims: tuple[str, str]):
    """``unstr``'s Cholesky factor ``Lcortime`` under ``lkj_corr_cholesky(eta)`` (brms's prior class ``cortime``), a PyMC
    variable sampled on Stan's ``cholesky_corr`` coordinates (``<name>_cholesky_corr__``) whose initial point is the identity.
    Must be called inside a PyMC model context whose coordinates hold ``dims``."""
    import pymc as pm
    import pytensor.tensor as pt

    from autobayes.correlation import CholeskyCorrTransform, lkj_cholesky_logp, lkj_cholesky_rng

    eta = float(eta)

    def logp(value, eta_):
        return lkj_cholesky_logp(value, eta, dim)

    def random(eta_, rng=None, size=None):
        return lkj_cholesky_rng(eta, dim, rng, size)

    return pm.CustomDist(name, np.float64(eta), logp=logp, random=random, signature=f"()->({dim},{dim})",
                         transform=CholeskyCorrTransform(dim), dims=dims, support_point=lambda rv, size, eta_: pt.eye(dim))


def cosy_variable(name: str, prior):
    """``cosy``, brms's ``real<lower=0,upper=1>`` under its prior: ``Flat()`` (brms's default, uniform on (0, 1), PyMC's
    ``Uniform(0, 1)``, sampled as ``<name>_interval__``) or ``Beta(a, b)`` (sampled as ``<name>_logodds__``), both Stan's
    logit coordinate with its Jacobian."""
    import pymc as pm

    if prior.name == "Flat":
        return pm.Uniform(name, lower=np.float64(0.0), upper=np.float64(1.0))
    return pm.Beta(name, **{k: np.float64(v) for k, v in prior.params.items()})


def timecor_family(base, model):
    """The Bambi family ``base`` (a built-in family's name or a family object) whose likelihood is the joint density of each
    group's rows (F03b): Bambi calls it with the family's parameters by name inside the model it builds, and it creates the
    term's parameter there (``Lcortime`` and the deterministic ``Cortime``, or ``cosy``) and the response as one
    ``CustomDist`` over every fitted row, whose log density is ``log_density`` and whose draws are ``draw_rows``."""
    import copy

    import bambi as bmb
    import pymc as pm
    from bambi.defaults import get_builtin_family

    family = get_builtin_family(base) if isinstance(base, str) else copy.copy(base)
    lk = family.likelihood
    kernel, design, roles = model.family_spec.time_correlation(), model._timecor_design, model.timecor_names
    prior = model.priors.timecor_priors(model.spec)[model.spec.timecor.prior_class]
    keys = ("mu", "sigma", *kernel.parameters)

    class TimeCorDist:
        """The fitted responses jointly: every group's rows ``MultiNormal`` / ``MultiStudentT`` around their ``mu``."""

        def __new__(cls, name, *, observed=None, dims=None, **params):
            if set(params) != set(keys):
                raise RuntimeError(f"Bambi handed the likelihood {sorted(params)}; the residual correlation term needs {sorted(keys)}")
            pmm = pm.modelcontext(None)
            if design.call == "unstr":
                for dim in (roles["coordinate"], roles["coordinate_T"]):
                    pmm.add_coord(dim, design.labels)
                L = lkj_cholesky_variable(roles["L"], prior.params["eta"], design.dim, (roles["coordinate"], roles["coordinate_T"]))
                pm.Deterministic(roles["cor"], L @ L.T, dims=(roles["coordinate"], roles["coordinate_T"]))
                value, core = L, "(t,t)"
            else:
                value, core = cosy_variable(roles["cosy"], prior), "()"

            def logp(y, mu, sigma, *rest):
                *extra, v = rest
                return log_density(y, mu, sigma, extra, v, design, kernel)

            def random(mu, sigma, *rest, rng=None, size=None):
                if size not in (None, ()):
                    raise ValueError(f"the residual correlation term draws one response vector at a time (size {size!r})")
                *extra, v = rest
                L_ = v if design.call == "unstr" else cosy_cholesky_numpy(v, design.dim)
                return draw_rows(mu, sigma, extra, L_, design, kernel, rng)

            signature = "(n)," + "()," * (1 + len(kernel.parameters)) + core + "->(n)"   # mu, sigma, nu, the parameter
            return pm.CustomDist(name, *(params[k] for k in keys), value, logp=logp, random=random, signature=signature,
                                 observed=observed, dims=dims)

    family.likelihood = bmb.Likelihood(lk.name, params=lk.params, parent=lk.parent, dist=TimeCorDist)
    return family
