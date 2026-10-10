"""Thin-plate regression spline smooths ``s(x)``: the basis brms gets from mgcv, for one numeric covariate.

brms (2.23) builds ``s(x)`` with mgcv (1.9): ``smoothCon(s(x), data, absorb.cons = TRUE, diagonal.penalty = TRUE)``
and then ``smooth2random(sm, type = 2)``. The fixed part ``Xf`` is brms's ``Xs`` (coefficients ``bs``, class ``b``)
and the random part is ``Zs_1_1`` (``s_1_1 = sds_1 * zs_1_1``, ``zs_1_1 ~ std_normal()``). For new data brms evaluates
``PredictMat`` on the construction stored with the fit and applies smooth2random's transformation
(``brms:::s2rPred``); ``standata(fit)`` takes the same route for the fitted rows. This module ports that pipeline for
``bs = "tp"`` with one covariate and the default penalty order (``m = 2``), each step after the mgcv function named:

1. ``smooth.construct.tp.smooth.spec`` / C ``construct_tprs`` (Wood 2003, JRSS-B 65:95). x is centred by its mean
   (``shift``); the sorted unique centred values are the knots ``Xu``; ``E = eta(|x_i - x_j|)`` with
   ``eta(r) = r^3 / 12`` and ``T = [1, x]``. The ``k`` eigenvalues of ``E`` largest in magnitude come from mgcv's
   own Lanczos iteration (C ``Rlanczos``), which stops at a tolerance of ``eps^0.7``: its vectors are
   approximations (about 1e-9 from the exact eigenvectors for k = 10), and brms's basis is built from those
   approximations, so :func:`mgcv_lanczos` reproduces the iteration (start vector, re-orthogonalisation,
   convergence test and stopping point) rather than calling an eigensolver; with exactly ``k`` knots mgcv skips the
   decomposition (``U = I``, ``U D`` read as ``E``). The null space of ``T'U`` comes from
   mgcv's ``QT`` Householder factorisation; ``X = [U D Z, T]`` (Ritz values ``D``, not ``E U Z``) with every
   column scaled to unit root mean square over the data rows; ``S = Z' D Z`` scaled alike; ``UZ`` maps
   ``[eta(|x - Xu|), 1, x]`` to the basis for prediction.
2. ``smoothCon``: the constraint ``C = colMeans(X)``; the penalty divided by ``norm(S, "O") / norm(X, "I")^2``; the
   constraint absorbed by LINPACK's Householder QR of ``C'`` (``qr.qty``); then ``nat.param(type = 2,
   unit.fnorm = FALSE)``: ``eigen(S)``, range columns scaled to unit penalty, the null column scaled to the mean
   squared norm of the range columns (skipped, as in mgcv, when the penalty is already diagonal: ``k = 3``).
3. ``smooth2random``: the null-space column is ``Xs``; the range columns are ``Zs``.
4. ``PredictMat`` (C ``predict_tprs``) and ``brms:::s2rPred`` for new values of x.

What is and is not reproduced exactly:

- ``Xs`` (and its prior on ``bs``) is not invariant to a reparameterisation, so it is reproduced exactly, sign
  included: LAPACK's ``dsyevr`` returns the penalty's null vector with a positive last component here, because that
  vector is the last coordinate up to the constraint's rounding residue.
- ``Zs`` is reproduced up to an orthogonal matrix ``R`` (in practice a signed permutation): mgcv's eigenvector signs
  and the order ``eigen()`` gives the tied unit penalties are LAPACK details, and ``zs ~ N(0, sds^2 I)`` is invariant
  under them. Callers compare ``Zs`` with brms's through an estimated ``R`` and check that it is orthogonal.
- Agreement is limited by mgcv's own rounding, which grows with the spread of the Ritz values
  (:attr:`TPRS.ritz_ratio`): about 1e-13 for the default ``k = 10`` on ordinary data; outlying covariate values or a
  ``k`` close to the number of unique values amplify it, and mgcv's output then moves by as much under the maps
  ``x -> a x + b`` that leave the construction unchanged in exact arithmetic.
- When more than ``k`` knots are symmetric about their centre (equally spaced, or integer values with every level
  present), the eigenvector of the most negative eigenvalue is antisymmetric, the first ``QT`` pivot is zero in exact
  arithmetic and mgcv's reflection follows the sign of its rounding noise: brms's basis is then a rounding accident
  that no other implementation can reproduce. :func:`tprs_construct` refuses such inputs with
  :class:`IndeterminateBasisError`, as it does a tie in magnitude at the truncation. (With exactly ``k`` knots there
  is no eigen-decomposition and nothing is ambiguous.)

Not ported (refused): more than ``MAX_KNOTS`` unique values (mgcv subsamples knots with R's RNG), ``m`` other than
2, user-supplied knots, ``by`` variables, other basis types. Nothing here depends on the model.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh_tridiagonal

#: mgcv's ``xt$max.knots`` default: with more unique values mgcv subsamples the knots with R's RNG (not ported).
MAX_KNOTS = 2000
#: The stopping tolerance ``construct_tprs`` passes to ``Rlanczos``: ``.Machine$double.eps^0.7``. Identified from
#: mgcv's output (every one of 269 seeded constructions, including 30 chosen to separate it from 1e-11).
LANCZOS_TOL = float(np.finfo(float).eps ** 0.7)
#: Penalty null-space dimension for ``d = 1``, ``m = 2`` (the polynomials ``1, x``).
NULL_DIM = 2
#: mgcv's default basis dimension for one covariate: ``M + 8``.
DEFAULT_K = NULL_DIM + 8
#: A ``QT`` pivot below this fraction of its row norm has a sign set by rounding noise (exactly zero in exact
#: arithmetic for symmetric knots); data-driven pivots are many orders of magnitude larger.
PIVOT_TOL = 1e-10
#: Two Ritz values whose magnitudes the truncation compares and that agree to this relative precision make the
#: choice of the ``k`` largest a matter of rounding.
TIE_TOL = 1e-8


class IndeterminateBasisError(ValueError):
    """mgcv's basis for these covariate values is decided by rounding noise, so brms's cannot be reproduced."""


# -- R's arithmetic ----------------------------------------------------------------------------------------------
def r_mean(x) -> float:
    """R's ``mean()`` of a double vector: a sequential sum divided by n, then one correction pass (summary.c).

    R accumulates in ``long double``, which is ``double`` on the aarch64 builds the oracle runs on; on x86-64 builds
    the two can differ in the last bit, which moves nothing here beyond rounding."""
    x = np.asarray(x, dtype=float)
    s = np.cumsum(x)[-1] / x.size                     # cumsum adds sequentially, as R's loop does
    if math.isfinite(s):
        s = s + np.cumsum(x - s)[-1] / x.size
    return float(s)


def r_col_means(X: np.ndarray) -> np.ndarray:
    """R's ``colMeans()``: a sequential sum down each column divided by the number of rows."""
    return np.cumsum(X, axis=0)[-1] / X.shape[0]


# -- mgcv's Lanczos iteration (C Rlanczos) -------------------------------------------------------------------------
def lanczos_start(n: int) -> np.ndarray:
    """``Rlanczos``'s start vector: a fixed linear congruential sequence (``jran = (106 jran + 1283) mod 6075`` from
    ``jran = 1``) mapped to ``jran / 6075 - 0.5`` and normalised."""
    jran = 1
    b = np.empty(n)
    for i in range(n):
        jran = (jran * 106 + 1283) % 6075
        b[i] = jran / 6075.0 - 0.5
    return b / math.sqrt(float(b @ b))


@dataclass(frozen=True)
class LanczosResult:
    """The output of :func:`mgcv_lanczos`: mgcv's ``(D, U)`` and how the iteration ended."""

    values: np.ndarray        # the Ritz values, in mgcv's order: positive ones descending, then the negative ones
    vectors: np.ndarray       # n x k Ritz vectors (signs arbitrary)
    iterations: int           # mgcv's iteration count (slanczos()$iter)
    converged: bool           # False when the test never passed and mgcv fell back to the k largest values
    stop_margins: tuple       # |log(rho / tol)| at every convergence test (rho: the worst decisive error estimate)
    tie_gap: float            # the smallest relative gap between the magnitudes the truncation compared


def mgcv_lanczos(A: np.ndarray, k: int, tol: float = LANCZOS_TOL) -> LanczosResult:
    """mgcv's ``Rlanczos(A, k, lm = -1, tol)``: the ``k`` eigenvalues of the symmetric ``A`` largest in magnitude and
    their Ritz vectors, by the same iteration mgcv runs (so the same approximations, not exact eigenvectors).

    The recursion re-orthogonalises fully, twice, against every earlier vector; the tridiagonal ``T_j`` is solved every
    ``f_check = min(max(10, k // 2), max(1, n // 10))`` iterations once ``j >= k`` (and at ``j = n - 1``). With the
    eigenvalues of ``T_j`` sorted downwards and error estimates ``err_i = |b_j v_{j,i}|``, the test walks inwards from
    both ends taking the larger magnitude, and requires every error it reads to be at most ``tol * max(|d_0|, |d_j|)``.
    It reads ``err[ni]`` (an element from the top) when it takes a value from the negative end; that is mgcv's code
    (``err[j - ni]`` would be the error of that value) and it decides when mgcv stops, so it is reproduced. If no test
    passes, mgcv keeps ``m = k`` and returns the ``k`` algebraically largest values."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if A.shape != (n, n) or not 1 <= k <= n:
        raise ValueError(f"need a square matrix with at least k = {k} rows, got {A.shape}")
    f_check = min(max(10, k // 2), max(1, n // 10))
    q = [lanczos_start(n)]
    a = np.zeros(n)
    b = np.zeros(n)
    margins = []
    tie_gap = math.inf
    d = v = None
    for j in range(n):
        z = A @ q[j]
        a[j] = float(q[j] @ z)
        if j == 0:
            z = z - a[j] * q[j]
        else:
            z = z - (a[j] * q[j] + b[j - 1] * q[j - 1])
            for _ in range(2):                                  # mgcv re-orthogonalises twice
                for i in range(j + 1):
                    z = z + (-float(z @ q[i])) * q[i]
        b[j] = math.sqrt(float(z @ z))
        if j < n - 1:
            if b[j] == 0.0:
                raise IndeterminateBasisError("the Lanczos recursion broke down (an exactly invariant Krylov subspace); "
                                              "mgcv would divide by zero here")
            q.append(z / b[j])
        if not ((j >= k and j % f_check == 0) or j == n - 1):
            continue
        if j > 0:
            d, v = eigh_tridiagonal(a[:j + 1], b[:j])
        else:
            d, v = a[:1].copy(), np.ones((1, 1))
        d, v = d[::-1], v[:, ::-1]                              # descending, as mgcv_trisymeig returns them
        if j < k:
            continue
        norm_t = max(abs(d[0]), abs(d[j]))
        err = np.abs(b[j] * v[j, :])
        max_err = norm_t * tol
        pi = ni = 0
        converged = True
        rho = 0.0
        while pi + ni < k:
            top, bottom = abs(d[pi]), abs(d[j - ni])
            tie_gap = min(tie_gap, abs(top - bottom) / max(top, bottom, 1e-300))
            if top >= bottom:
                rho = max(rho, err[pi] / norm_t)
                if err[pi] > max_err:
                    converged = False
                    break
                pi += 1
            else:
                rho = max(rho, err[ni] / norm_t)
                if err[ni] > max_err:
                    converged = False
                    break
                ni += 1
        margins.append(abs(math.log(max(rho, 1e-300) / tol)))
        if converged:
            jj = j + 1
            Q = np.array(q[:jj]).T
            idx = list(range(pi)) + [jj - (ni + pi - kk) for kk in range(pi, pi + ni)]
            return LanczosResult(d[idx].copy(), Q @ v[:, idx], jj, True, tuple(margins), tie_gap)
    Q = np.array(q[:n]).T                                       # never converged (k == n): the k largest values
    return LanczosResult(d[:k].copy(), Q @ v[:, :k], n, False, tuple(margins), tie_gap)


def mgcv_qt(A: np.ndarray) -> tuple[np.ndarray, tuple]:
    """mgcv's ``QT``: an orthogonal ``Q`` with ``A Q = [0, R]`` (``A`` is r x c, r < c), built from Householder
    reflections applied from the right, row by row: row ``i`` (columns ``0 .. c - i - 1``) is reflected onto its last
    active column with ``w = u + sign(u_last) |u| e_last``. Returns ``Q`` and each pivot ``|u_last| / |u|``."""
    A = np.array(A, dtype=float)
    r, c = A.shape
    Q = np.eye(c)
    pivots = []
    for i in range(r):
        L = c - i
        u = A[i, :L].copy()
        scale = float(np.abs(u).max())
        if scale == 0.0:
            raise IndeterminateBasisError("a row of T'U is zero")
        u = u / scale
        nrm = math.sqrt(float(u @ u))
        pivots.append(abs(u[L - 1]) / nrm)
        if u[L - 1] < 0.0:
            nrm = -nrm
        u[L - 1] += nrm
        H = np.eye(c)
        H[:L, :L] -= (2.0 / float(u @ u)) * np.outer(u, u)
        A = A @ H
        Q = Q @ H
    return Q, tuple(pivots)


def tps_eta(r: np.ndarray) -> np.ndarray:
    """The thin-plate radial function for ``d = 1``, ``m = 2``: ``Gamma(-3/2) / (2^4 sqrt(pi)) r^3 = r^3 / 12``."""
    r = np.asarray(r, dtype=float)
    return r * r * r / 12.0


# -- smooth.construct.tp.smooth.spec ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class TPRS:
    """mgcv's ``tprs.smooth`` object for one covariate (``smooth.construct(s(x, k), data, NULL)``)."""

    k: int                    # basis dimension (bs.dim)
    shift: float              # the mean subtracted from x
    Xu: np.ndarray            # the knots: sorted unique centred values
    UZ: np.ndarray            # (len(Xu) + 2) x k: [eta(|x - Xu|), 1, x - shift] @ UZ is the basis at x
    X: np.ndarray             # n x k model matrix: k - 2 penalised columns, then 1 and x (unit RMS columns)
    S: np.ndarray             # k x k penalty
    lanczos: LanczosResult | None   # None when there are exactly k knots (mgcv then keeps the full basis)
    qt_pivots: tuple
    spectrum: np.ndarray      # the Ritz values D (or, with exactly k knots, the eigenvalues of E)

    @property
    def rank(self) -> int:
        return self.k - NULL_DIM

    @property
    def ritz_ratio(self) -> float:
        """``max |D| / min |D|``: mgcv's own rounding in the basis grows roughly in proportion to it."""
        d = np.abs(self.spectrum)
        return float(d.max() / d.min())


def _covariate(x, what: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{what} must be a non-empty one-dimensional array, got shape {arr.shape}")
    if arr.dtype.kind not in "biuf":
        raise ValueError(f"{what} must be numeric, got dtype {arr.dtype}")
    arr = arr.astype(float)
    if not np.isfinite(arr).all():
        raise ValueError(f"{what} has missing or non-finite values")
    return arr


def tprs_construct(x, k: int | None = None) -> TPRS:
    """``smooth.construct.tp.smooth.spec(s(x, k = k), data, knots = NULL)`` for one numeric covariate, ``m = 2``.

    ``k = None`` (or -1, mgcv's default) gives ``k = 10``; ``k < 3`` is raised to 3 with mgcv's warning. Refused:
    fewer unique values than ``k`` (mgcv's error), more than ``MAX_KNOTS`` unique values (mgcv subsamples them with
    R's RNG), and inputs whose mgcv basis is decided by rounding noise (:class:`IndeterminateBasisError`)."""
    x = _covariate(x, "x")
    if k is None or k == -1:
        k = DEFAULT_K
    if isinstance(k, bool) or int(k) != k:
        raise ValueError(f"k must be an integer, got {k!r}")
    k = int(k)
    if k < NULL_DIM + 1:
        warnings.warn("basis dimension, k, increased to minimum possible", UserWarning, stacklevel=2)
        k = NULL_DIM + 1
    n = x.size
    shift = r_mean(x)
    xs = x - shift
    Xu, index = np.unique(xs, return_inverse=True)
    nk = Xu.size
    if nk > MAX_KNOTS:
        raise NotImplementedError(f"s(x) with more than {MAX_KNOTS} unique values: mgcv then subsamples the knots with "
                                  f"R's random number generator, which is not ported ({nk} unique values)")
    if nk < k:
        raise ValueError("A term has fewer unique covariate combinations than specified maximum degrees of freedom "
                         f"({nk} unique values, k = {k})")
    E = tps_eta(np.abs(Xu[:, None] - Xu[None, :]))
    T = np.column_stack([np.ones(nk), Xu])
    if nk == k:
        # exactly k knots: mgcv keeps the full basis (U = I, no eigen-decomposition), so X = [E Z, T], S = Z'EZ
        lz = None
        U, DZ = np.eye(nk), E
        spectrum = np.linalg.eigvalsh(E)
    else:
        lz = mgcv_lanczos(E, k)
        if lz.tie_gap <= TIE_TOL:
            raise IndeterminateBasisError(f"two eigenvalues of the thin-plate matrix tie in magnitude at the truncation to "
                                          f"k = {k} (relative gap {lz.tie_gap:.1e}); which one mgcv keeps is decided by rounding")
        U, DZ = lz.vectors, np.diag(lz.values)
        spectrum = lz.values
    Q, pivots = mgcv_qt(T.T @ U)
    if min(pivots) <= PIVOT_TOL:
        raise IndeterminateBasisError(
            f"a pivot of mgcv's null-space factorisation for s(x) is zero up to rounding ({min(pivots):.1e}), as it is when "
            "the knots are symmetric about their centre (for example equally spaced or integer values with every level "
            "present); mgcv's basis then follows the sign of that rounding noise and brms's cannot be reproduced")
    Zk = Q[:, :k - NULL_DIM]
    Xknots = np.column_stack([U @ (DZ @ Zk), T])
    X = Xknots[index]
    scale = np.sqrt(np.mean(X * X, axis=0))                   # unit root mean square over the data rows
    X = X / scale
    S = np.zeros((k, k))
    S[:k - NULL_DIM, :k - NULL_DIM] = Zk.T @ (DZ @ Zk)
    S = S / np.outer(scale, scale)
    S = (S + S.T) / 2.0                                       # as smooth.construct.tp.smooth.spec does
    UZ = np.zeros((nk + NULL_DIM, k))
    UZ[:nk, :k - NULL_DIM] = U @ Zk
    UZ[nk:, k - NULL_DIM:] = np.eye(NULL_DIM)
    UZ = UZ / scale
    return TPRS(k=k, shift=shift, Xu=Xu, UZ=UZ, X=X, S=S, lanczos=lz, qt_pivots=pivots, spectrum=spectrum)


def tprs_predict(tp: TPRS, x) -> np.ndarray:
    """``Predict.matrix.tprs.smooth`` (C ``predict_tprs``): the raw k-column basis at new values of x."""
    xs = _covariate(x, "x") - tp.shift
    nk = tp.Xu.size
    B = tps_eta(np.abs(xs[:, None] - tp.Xu[None, :]))
    return B @ tp.UZ[:nk] + np.column_stack([np.ones(xs.size), xs]) @ tp.UZ[nk:]


# -- smoothCon(absorb.cons = TRUE, diagonal.penalty = TRUE) and smooth2random(type = 2) ------------------------------
def _linpack_reflector(c: np.ndarray) -> np.ndarray:
    """LINPACK ``dqrdc2`` on one column ``c``: the Householder vector ``u`` (``qraux = u[0]``) of ``qr(t(C))``."""
    nrm = float(np.linalg.norm(c))
    if nrm == 0.0:
        raise ValueError("the sum-to-zero constraint is zero")
    if c[0] != 0.0:
        nrm = math.copysign(nrm, c[0])
    u = c / nrm
    u[0] += 1.0
    return u


def _qty(u: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """``qr.qty`` for the one-reflector QR: ``Q'Y = Y - u (u'Y) / u[0]``, column by column."""
    return Y - np.outer(u, (u @ Y) / u[0])


@dataclass(frozen=True)
class SmoothBasis:
    """``s(x)`` as brms hands it to Stan: ``Xs`` (fixed, ``bs``) and ``Zs`` (random, ``zs``), with the construction
    needed to evaluate both at new values of x (``predict``)."""

    tprs: TPRS
    reflector: np.ndarray            # the absorbed constraint (LINPACK Householder vector of colMeans(X))
    penalty_scale: float             # smoothCon's S.scale
    reparam: np.ndarray | None       # nat.param's P (smoothCon's diagRP); None when the penalty was already diagonal
    trans_U: np.ndarray              # smooth2random's trans.U and trans.D
    trans_D: np.ndarray
    rank: int                        # penalised columns (knots_1 in brms's standata)
    Xs: np.ndarray                   # n x 1 fixed part on the data the basis was built from
    Zs: np.ndarray                   # n x rank random part on the same rows

    @property
    def k(self) -> int:
        return self.tprs.k

    def predict_matrix(self, x) -> np.ndarray:
        """``mgcv::PredictMat(sm, data)``: the basis at new values of x with the constraint absorbed and (unless
        ``k = 3``) ``diagRP`` applied, in smoothCon's column order (penalised columns by decreasing penalty
        eigenvalue, then the null column). Matches mgcv's up to the signs of the penalised columns."""
        X = _qty(self.reflector, tprs_predict(self.tprs, x).T)[1:, :].T
        return X if self.reparam is None else X @ self.reparam

    def predict(self, x) -> tuple[np.ndarray, np.ndarray]:
        """``(Xs, Zs)`` at new values of x: ``PredictMat`` on the stored construction, then smooth2random's map
        (``brms:::s2rPred``). On the construction's own rows this is what ``standata(fit)`` returns; it differs from
        ``Xs``/``Zs`` by mgcv's Lanczos residual (``E U != U D``), about 1e-10 for k = 10."""
        X = (self.predict_matrix(x) @ self.trans_U) * self.trans_D
        return X[:, self.rank:], X[:, :self.rank]


def _sym_eigen_desc(S: np.ndarray, null_from: int) -> tuple[np.ndarray, np.ndarray]:
    """``eigen(S, symmetric = TRUE)`` with R's descending order; null vectors (index >= ``null_from``) signed to a
    positive last component, as LAPACK's dsyevr leaves the penalty null vector here (see the module docstring)."""
    w, V = np.linalg.eigh(S)
    w, V = w[::-1].copy(), V[:, ::-1].copy()
    for j in range(null_from, V.shape[1]):
        if V[-1, j] < 0.0:
            V[:, j] = -V[:, j]
    return w, V


def _nat_param(X: np.ndarray, S: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """``nat.param(X, S, rank, type = 2, unit.fnorm = FALSE)``: returns the reparameterised ``X`` and ``P``."""
    w, V = _sym_eigen_desc(S, rank)
    p = X.shape[1]
    E = np.ones(p)
    E[:rank] = np.sqrt(w[:rank])
    XV = X @ V
    col = np.sum(XV * XV, axis=0) / E ** 2
    av = float(np.mean(col[:rank]))
    for i in range(rank, p):
        E[i] = math.sqrt(col[i] / av)
    return XV / E, V / E


def smooth_basis(x, k: int | None = None) -> SmoothBasis:
    """brms's ``Xs`` and ``Zs`` for ``s(x, k = k)`` on the values ``x``: ``smoothCon(..., absorb.cons = TRUE,
    diagonal.penalty = TRUE)`` then ``smooth2random(type = 2)``, as ``brms:::data_sm`` builds them when fitting."""
    tp = tprs_construct(x, k)
    X, S, kk = tp.X, tp.S, tp.k
    rank = tp.rank
    # smoothCon: the constraint, the penalty scale, the absorption
    C = r_col_means(X)
    ma_xx = float(np.max(np.sum(np.abs(X), axis=1))) ** 2
    ma_s = float(np.max(np.sum(np.abs(S), axis=0))) / ma_xx
    S = S / ma_s
    u = _linpack_reflector(C)
    Xa = _qty(u, X.T)[1:, :].T
    Sa = _qty(u, _qty(u, S)[1:, :].T)[1:, :].T
    p = kk - 1
    # diagonal.penalty: nat.param unless S is already diag(S11, ..., S11, 0, ...)
    s11 = Sa[0, 0]
    target = np.diag(np.r_[np.full(rank, s11), np.zeros(p - rank)])
    if np.max(np.abs(Sa - target)) > abs(s11) * np.finfo(float).eps ** 0.8:
        Xd, P = _nat_param(Xa, Sa, rank)
        Sd = np.diag(np.r_[np.ones(rank), np.zeros(p - rank)])
    else:
        Xd, P, Sd = Xa, None, Sa
    # smooth2random(type = 2): eigen of the (now diagonal) penalty; the null part is fixed, the range part random
    if np.count_nonzero(Sd - np.diag(np.diag(Sd))) == 0:
        order = np.argsort(-np.diag(Sd), kind="stable")
        w = np.diag(Sd)[order]
        U = np.eye(p)[:, order]
    else:                                                       # k = 3: nat.param was skipped
        # smooth2random flips every vector when the first one starts negative; after that flip the null vector has
        # a positive last entry (LAPACK's 2 x 2 dlaev2 branch), which _sym_eigen_desc imposes directly. The signs
        # of the range vectors are immaterial (zs is isotropic).
        w, U = _sym_eigen_desc(Sd, rank)
    D = 1.0 / np.sqrt(np.r_[w[:rank], np.ones(p - rank)])
    Xr = (Xd @ U) * D
    return SmoothBasis(tprs=tp, reflector=u, penalty_scale=ma_s, reparam=P, trans_U=U, trans_D=D, rank=rank,
                       Xs=Xr[:, rank:], Zs=Xr[:, :rank])
