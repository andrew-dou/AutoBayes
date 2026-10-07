"""Correlated group-level effects: the Cholesky factor of a correlation matrix under an LKJ prior, in Stan's terms.

brms gives the coefficients of ``(1 + x | g)`` a correlation matrix through its Cholesky factor ``L_m``
(``cholesky_factor_corr[M_m]``) with the prior ``lkj_corr_cholesky_lpdf(L_m | eta)``, and samples it on Stan's
unconstrained ``cholesky_corr`` coordinates: ``M (M - 1) / 2`` canonical partial correlations through ``tanh``,
read row by row below the diagonal (``(1, 0), (2, 0), (2, 1), ...``). This module is that construction for
PyMC: ``cholesky_corr_constrain`` / ``cholesky_corr_free`` are Stan's ``cholesky_corr_constrain`` /
``cholesky_corr_free`` (NumPy), ``CholeskyCorrTransform`` the same map as a PyMC value transform (with Stan's
Jacobian, so that the Jacobian-adjusted density and every gradient coordinate are comparable with
``rstan::log_prob``), ``lkj_cholesky_logp`` Stan's density with its normalising constant, ``lkj_cholesky_rng`` its forward draws (for prior
predictive draws), and ``cholesky_corr_rv`` the PyMC random variable that carries them. Nothing here depends on the
model.
"""
from __future__ import annotations

import numpy as np
import pytensor.tensor as pt
from scipy.special import betaln

from pymc.logprob.transforms import Transform


def n_free(M: int) -> int:
    """The number of unconstrained coordinates of an ``M x M`` correlation Cholesky factor, ``M (M - 1) / 2``."""
    return M * (M - 1) // 2


def lower_indices(M: int) -> list[tuple[int, int]]:
    """Stan's order of the strictly-lower entries: row by row, ``(1, 0), (2, 0), (2, 1), (3, 0), ...``."""
    return [(i, j) for i in range(1, M) for j in range(i)]


# -- Stan's cholesky_corr transform, NumPy ---------------------------------------------------------------------
def cholesky_corr_constrain(y, M: int) -> tuple[np.ndarray, float]:
    """Stan's ``cholesky_corr_constrain``: the unconstrained ``y`` (``M (M - 1) / 2`` entries) to the lower-triangular
    ``L`` with unit-norm rows, and the log Jacobian of the map (``log |dL / dy|`` over the strictly-lower entries)."""
    y = np.asarray(y, dtype=float).reshape(-1)
    if y.shape != (n_free(M),):
        raise ValueError(f"y must have {n_free(M)} entries for M = {M}, got {y.shape}")
    z = np.tanh(y)
    lp = float(np.sum(np.log1p(-z ** 2)))
    L = np.zeros((M, M))
    L[0, 0] = 1.0
    k = 0
    for i in range(1, M):
        L[i, 0] = z[k]
        k += 1
        s = L[i, 0] ** 2
        for j in range(1, i):
            lp += 0.5 * np.log1p(-s)
            L[i, j] = z[k] * np.sqrt(1.0 - s)
            k += 1
            s += L[i, j] ** 2
        L[i, i] = np.sqrt(1.0 - s)
    return L, lp


def cholesky_corr_free(L) -> np.ndarray:
    """Stan's ``cholesky_corr_free``: the unconstrained coordinates of a correlation Cholesky factor ``L``."""
    L = np.asarray(L, dtype=float)
    M = L.shape[0]
    if L.shape != (M, M):
        raise ValueError(f"L must be square, got {L.shape}")
    z = []
    for i in range(1, M):
        z.append(L[i, 0])
        s = L[i, 0] ** 2
        for j in range(1, i):
            z.append(L[i, j] / np.sqrt(1.0 - s))
            s += L[i, j] ** 2
    z = np.asarray(z, dtype=float)
    if z.size and not (np.abs(z) < 1).all():
        raise ValueError("L is not the Cholesky factor of a correlation matrix (a partial correlation is not inside (-1, 1))")
    return np.arctanh(z)


def check_correlation_cholesky(L, M: int, what: str) -> np.ndarray:
    """``L`` as a float array, refused unless it is an ``M x M`` lower-triangular matrix with positive diagonal and
    unit-norm rows (the Cholesky factor of a correlation matrix), within rounding."""
    arr = np.asarray(L, dtype=float)
    if arr.shape != (M, M) or not np.isfinite(arr).all():
        raise ValueError(f"{what} must be a finite {M} x {M} matrix, got shape {arr.shape}")
    if not np.allclose(np.triu(arr, 1), 0.0):
        raise ValueError(f"{what} must be lower triangular (the Cholesky factor of the correlation matrix)")
    if not (np.diag(arr) > 0).all():
        raise ValueError(f"{what} must have a positive diagonal")
    if not np.allclose((arr ** 2).sum(1), 1.0, atol=1e-10):
        raise ValueError(f"{what} must have unit-norm rows (L L^T must have unit diagonal)")
    return arr


# -- the same map as a PyMC value transform --------------------------------------------------------------------
class CholeskyCorrTransform(Transform):
    """Stan's ``cholesky_corr`` transform as a PyMC transform: the value variable holds the ``M (M - 1) / 2``
    unconstrained coordinates, ``backward`` builds ``L`` from them, ``forward`` recovers them from ``L`` and
    ``log_jac_det`` is Stan's Jacobian term, so the sampled space and the Jacobian-adjusted density are brms's."""

    name = "cholesky_corr"
    ndim_supp = 2

    def __init__(self, M: int):
        if int(M) < 2:
            raise ValueError("a correlation Cholesky factor needs at least two coefficients")
        self.M = int(M)

    def backward(self, value, *inputs):
        M = self.M
        z = pt.tanh(value)
        L = pt.zeros((M, M))
        L = pt.set_subtensor(L[0, 0], 1.0)
        k = 0
        for i in range(1, M):
            L = pt.set_subtensor(L[i, 0], z[..., k])
            k += 1
            s = z[..., k - 1] ** 2
            for j in range(1, i):
                L = pt.set_subtensor(L[i, j], z[..., k] * pt.sqrt(1.0 - s))
                k += 1
                s = s + L[i, j] ** 2
            L = pt.set_subtensor(L[i, i], pt.sqrt(1.0 - s))
        return L

    def forward(self, value, *inputs):
        M = self.M
        z = []
        for i in range(1, M):
            z.append(value[..., i, 0])
            s = value[..., i, 0] ** 2
            for j in range(1, i):
                z.append(value[..., i, j] / pt.sqrt(1.0 - s))
                s = s + value[..., i, j] ** 2
        return pt.arctanh(pt.stack(z, axis=-1))

    def log_jac_det(self, value, *inputs):
        M = self.M
        z = pt.tanh(value)
        lp = pt.sum(pt.log1p(-z ** 2), axis=-1)
        k = 0
        for i in range(1, M):
            s = z[..., k] ** 2
            k += 1
            for j in range(1, i):
                lp = lp + 0.5 * pt.log1p(-s)
                s = s + (z[..., k] * pt.sqrt(1.0 - s)) ** 2
                k += 1
        return lp


# -- the LKJ density on the Cholesky factor ----------------------------------------------------------------------
def lkj_log_normalizer(eta: float, M: int) -> float:
    """``log c_M(eta)``, where the LKJ density of a correlation matrix is ``det(R)^(eta - 1) / c_M(eta)``
    (Lewandowski, Kurowicka and Joe 2009, the normalising constant written with Beta functions):
    ``c_M = 2^(sum_k (2 eta - 2 + M - k)(M - k)) prod_k B(eta + (M - k - 1) / 2, eta + (M - k - 1) / 2)^(M - k)``
    over ``k = 1..M-1``. For ``M = 2`` the single correlation is ``2 Beta(eta, eta) - 1``."""
    eta, M = float(eta), int(M)
    ks = np.arange(1, M)
    a = eta + (M - ks - 1) / 2.0
    return float(np.sum((2.0 * eta - 2.0 + M - ks) * (M - ks) * np.log(2.0) + (M - ks) * betaln(a, a)))


def lkj_cholesky_logp(L, eta: float, M: int):
    """Stan's ``lkj_corr_cholesky_lpdf(L | eta)`` with its constant: ``-log c_M(eta) + sum_{d=2}^{M} (M - d + 2 eta - 2) log L[d, d]``,
    the LKJ density of ``R = L L^T`` times the Jacobian of ``L -> R``. ``L`` is a PyTensor ``(M, M)`` matrix."""
    d = np.arange(2, int(M) + 1)
    coef = pt.as_tensor(np.asarray(int(M) - d + 2.0 * float(eta) - 2.0, dtype=float))
    diag = pt.diagonal(L, axis1=-2, axis2=-1)[..., 1:]
    return pt.sum(coef * pt.log(diag), axis=-1) - lkj_log_normalizer(eta, M)


def lkj_cholesky_rng(eta: float, M: int, rng: np.random.Generator, size=None) -> np.ndarray:
    """Draws of the Cholesky factor under LKJ(eta): canonical partial correlations ``2 Beta(a_c, a_c) - 1`` with
    ``a_c = eta + (M - 2 - c) / 2`` for the entries of column ``c`` (the C-vine construction of Lewandowski et al.,
    Stan's ``lkj_corr_cholesky_rng``), assembled as ``cholesky_corr_constrain`` assembles ``L`` from its
    partial correlations."""
    M = int(M)
    size = () if size is None else tuple(np.atleast_1d(size).astype(int))
    out = np.zeros(size + (M, M))
    out[..., 0, 0] = 1.0
    for i in range(1, M):
        s = np.zeros(size)
        for j in range(i):
            a = float(eta) + (M - 2 - j) / 2.0
            cpc = 2.0 * rng.beta(a, a, size=size) - 1.0
            out[..., i, j] = cpc * np.sqrt(1.0 - s)
            s = s + out[..., i, j] ** 2
        out[..., i, i] = np.sqrt(1.0 - s)
    return out


def cholesky_corr_rv(name: str, eta: float, M: int):
    """The PyMC random variable ``L`` (``M x M``, brms's ``L_m``) under ``lkj_corr_cholesky(eta)`` together with Stan's
    ``cholesky_corr`` transform and the value variable it is sampled through, ``<name>_cholesky_corr__`` (``M (M - 1) / 2``
    entries). The variable is not registered in any model: ``Model`` places it in the graph Bambi built
    (``pymc.model.fgraph.model_free_rv``)."""
    import pymc as pm

    M = int(M)
    if M < 2:
        raise ValueError("a correlation matrix needs at least two coefficients")
    eta = float(eta)
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("eta must be a positive finite number")

    def logp(value, eta_):
        return lkj_cholesky_logp(value, eta, M)

    def random(eta_, rng=None, size=None):
        return lkj_cholesky_rng(eta, M, rng if rng is not None else np.random.default_rng(), size)

    rv = pm.CustomDist.dist(eta, logp=logp, random=random, signature=f"()->({M},{M})", class_name=f"CustomDist_{name}")
    rv.name = name
    transform = CholeskyCorrTransform(M)
    value = transform.forward(rv).type()
    value.name = f"{name}_{transform.name}__"
    return rv, transform, value
