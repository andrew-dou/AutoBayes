"""Gaussian processes, brms's ``gp()`` (C03, docs/gaussian-processes.md): the exact GP, its Hilbert-space approximation and the
default prior of the length-scale.

Written from the published mathematics; brms's documentation, its Stan data and its outputs on probe inputs are the oracle
it is checked against, never its code. One numeric input ``x`` per term, scaled as brms's default ``scale = TRUE`` scales it:
divided by ``dmax``, the largest distance between the distinct fitted inputs, so that the largest distance is 1
(``formula.GaussianProcess`` keeps the scaling). The kernel is the squared-exponential (exponentiated-quadratic) one,
``k(x, x') = sdgp^2 exp(-(x - x')^2 / (2 lscale^2))``.

- The exact GP (Rasmussen and Williams 2006, ch. 2 and 4): ``f`` at the ``N`` distinct scaled inputs is multivariate normal
  with covariance ``K``; it is written non-centred, ``f = chol(K + 1e-12 I) z`` with ``z`` standard normal (brms's ``zgp``)
  and ``1e-12`` added to the diagonal (brms's jitter), the distinct inputs in the order they first appear in the data (brms's
  order, which fixes ``chol`` and so what each ``z_j`` means).
- The Hilbert-space approximation (Solin and Särkkä 2020; Riutort-Mayol et al. 2023): on ``[-L, L]`` with Dirichlet boundary
  conditions the Laplacian's eigenfunctions are ``phi_j(x) = sin(pi j (x + L) / (2 L)) / sqrt(L)`` with eigenvalues
  ``lambda_j = (pi j / (2 L))^2``, and ``f(x) = sum_{j <= k} phi_j(x) sqrt(S(sqrt(lambda_j))) z_j`` with ``S`` the kernel's
  spectral density, in one dimension ``S(w) = sdgp^2 sqrt(2 pi) lscale exp(-lscale^2 w^2 / 2)``. The basis is evaluated at
  the scaled inputs less the mean of the distinct scaled inputs (brms's ``cmeans``), and ``L = c`` times the largest distance
  between scaled inputs, which is 1 (brms's ``Lgp``). ``sqrt(S)`` is formed as the product
  ``sdgp (2 pi)^(1/4) sqrt(lscale) exp(-lscale^2 w^2 / 4)``, whose every factor and gradient is finite where ``S`` itself
  underflows to 0 (and ``d sqrt(S) / dS`` is infinite).
- The default prior of ``lscale`` (brms's documentation of ``set_prior``: an inverse gamma "specifically tuned to the
  covariates of the Gaussian process", after Betancourt's case study "Robust Gaussian Processes in Stan", part 3): the
  inverse gamma with 1% of its mass below the smallest distance ``l`` between distinct scaled inputs and 1% above the
  largest, ``u = 1``; brms floors ``l`` at 0.01 and prints each parameter rounded to 6 decimals (its rule, read off
  ``default_prior()`` on probe inputs). With two distinct inputs ``l = u`` and no inverse gamma has those tails: brms's
  documented fallback is a half-normal, ``normal(0, 0.5)``. ``inverse_gamma_tails`` solves the two conditions exactly.

G09 (docs/gaussian-processes.md, "Several inputs, by, other kernels and scale = FALSE"): the exact GP of several inputs, one
kernel of the Euclidean distance between the scaled input rows (isotropic) or, with ``iso = FALSE`` and the squared exponential,
the product of one-dimensional kernels with a length-scale each (Stan's ARD kernel); the Matern 3/2, Matern 5/2 and exponential
kernels of the Stan Functions Reference, ``k(r) = (1 + sqrt(3) r) exp(-sqrt(3) r)``, ``(1 + sqrt(5) r + 5 r^2 / 3)
exp(-sqrt(5) r)`` and ``exp(-r)`` at ``r = d / lscale``; and their one-dimensional spectral densities for the approximate GP
(Rasmussen and Williams 2006, eq. 4.15 at nu = 3/2, 5/2 and 1/2): ``sdgp^2 12 sqrt(3) l (3 + l^2 w^2)^-2``,
``sdgp^2 (400 sqrt(5) / 3) l (5 + l^2 w^2)^-3`` and ``sdgp^2 2 l (1 + l^2 w^2)^-1``, whose square roots are formed directly.
"""
from __future__ import annotations

import math

import numpy as np

JITTER = 1e-12                  # added to the exact GP's covariance diagonal before its Cholesky factor (brms's)
TAIL = 0.01                     # the default lscale prior's mass below the smallest and above the largest distance
FLOOR = 0.01                    # brms's floor on the smallest distance (of scaled inputs) the default lscale prior is tuned to
FALLBACK_SCALE = 0.5            # brms's half-normal lscale prior when no inverse gamma has those tails (two distinct inputs)
DIGITS = 6                      # the decimals brms rounds the tuned inverse gamma's parameters to
_SQRT_SQRT_2PI = (2.0 * math.pi) ** 0.25
KERNELS = ("exp_quad", "matern32", "matern52", "exponential")   # brms's cov = (G09; exp_quad is C03's)
_SQRT3, _SQRT5 = math.sqrt(3.0), math.sqrt(5.0)
# sqrt of each Matern spectral density's constant in one dimension: S(w) = sdgp^2 C l (nu2 + l^2 w^2)^(-p) (G09)
_SPD = {"matern32": (math.sqrt(12.0 * _SQRT3), 3.0, 2.0), "matern52": (math.sqrt(400.0 * _SQRT5 / 3.0), 5.0, 3.0),
        "exponential": (math.sqrt(2.0), 1.0, 1.0)}


# -- NumPy ----------------------------------------------------------------------------------------------------------------
def sqrt_eigenvalues(L: float, k: int) -> np.ndarray:
    """``sqrt(lambda_j) = pi j / (2 L)``, ``j = 1..k`` (brms's ``slambda``): the square roots of the first ``k`` eigenvalues
    of the Laplacian on ``[-L, L]`` with Dirichlet boundary conditions."""
    return np.pi * np.arange(1, int(k) + 1, dtype=float) / (2.0 * float(L))


def basis(x, L: float, k: int) -> np.ndarray:
    """``(n, k)``: the eigenfunctions ``phi_j(x) = sin(pi j (x + L) / (2 L)) / sqrt(L)`` at the centred, scaled inputs ``x``
    (brms's ``Xgp`` of an approximate GP)."""
    x = np.asarray(x, dtype=float).reshape(-1)
    j = np.arange(1, int(k) + 1, dtype=float)
    return np.sin(np.pi * j[None, :] * (x[:, None] + float(L)) / (2.0 * float(L))) / np.sqrt(float(L))


def sqrt_spectral_density(slambda, sdgp, lscale, cov: str = "exp_quad") -> np.ndarray:
    """``sqrt(S(w))`` of the kernel ``cov`` in one dimension at the frequencies ``slambda`` (``(k,)``), for ``sdgp`` and
    ``lscale`` of any common shape ``s``: ``(*s, k)``. The squared exponential's is C03's; G09 adds the Matern 3/2, Matern 5/2
    and exponential kernels' (module docstring), ``sdgp sqrt(C l) (nu2 + l^2 w^2)^(-p / 2)``."""
    w = np.asarray(slambda, dtype=float)
    s = np.asarray(sdgp, dtype=float)[..., None]
    l = np.asarray(lscale, dtype=float)[..., None]
    if cov == "exp_quad":
        return s * _SQRT_SQRT_2PI * np.sqrt(l) * np.exp(-(l * l) * (w * w) / 4.0)
    root, nu2, p = _SPD[cov]
    return s * root * np.sqrt(l) * (nu2 + (l * l) * (w * w)) ** (-p / 2.0)


def squared_differences(x) -> np.ndarray:
    """``(D, N, N)``: the squared differences ``(x_i[d] - x_j[d])^2`` of the distinct scaled inputs ``x`` (``(N,)`` for one
    input, ``(N, D)`` for several), dimension by dimension (G09)."""
    x = np.asarray(x, dtype=float)
    x = x.reshape(x.shape[0], -1)
    return np.stack([np.subtract.outer(x[:, d], x[:, d]) ** 2 for d in range(x.shape[1])])


def _matern(r, cov, exp):
    """The one-dimensional kernel ``cov`` (not the squared exponential) at ``r = d / lscale``, with ``exp`` NumPy's or
    PyTensor's (G09)."""
    if cov == "matern32":
        a = np.float64(_SQRT3) * r
        return (np.float64(1.0) + a) * exp(-a)
    if cov == "matern52":
        a = np.float64(_SQRT5) * r
        return (np.float64(1.0) + a + a * a / np.float64(3.0)) * exp(-a)
    return exp(-r)                                            # the exponential kernel


def covariance(x, sdgp, lscale, cov: str = "exp_quad") -> np.ndarray:
    """``K + 1e-12 I`` of the exact GP at the distinct scaled inputs ``x`` (``(N,)`` or ``(N, D)``) for draws of ``sdgp`` of
    shape ``s`` and ``lscale`` of shape ``s`` (one length-scale) or a list of ``D`` arrays of shape ``s`` (``iso = FALSE``,
    the squared exponential only: the product of one-dimensional kernels): ``(*s, N, N)``. One length-scale: the kernel of the
    Euclidean distance (G09; C03's one input is the case ``D = 1``)."""
    sq = squared_differences(x)
    s = np.asarray(sdgp, dtype=float)[..., None, None]
    if isinstance(lscale, (list, tuple)):
        if cov != "exp_quad":
            raise ValueError(f"one length-scale per input is offered for the squared-exponential kernel only, not {cov!r}")
        K = s * s
        for d, l in enumerate(lscale):
            l = np.asarray(l, dtype=float)[..., None, None]
            K = K * np.exp(-sq[d] / (2.0 * l * l))
    else:
        l = np.asarray(lscale, dtype=float)[..., None, None]
        if cov == "exp_quad":
            K = s * s * np.exp(-sq.sum(0) / (2.0 * l * l))
        else:
            K = s * s * _matern(np.sqrt(sq.sum(0)) / l, cov, np.exp)
    return K + JITTER * np.eye(sq.shape[1])


def exact_values(x, sdgp, lscale, z, cov: str = "exp_quad") -> np.ndarray:
    """``f = chol(sdgp^2 exp(-d^2 / (2 lscale^2)) + 1e-12 I) z`` at the distinct scaled inputs ``x`` (``(N,)``), for draws of
    ``sdgp`` and ``lscale`` of shape ``s`` and of ``z`` of shape ``(*s, N)``: ``(*s, N)``. G09: ``x`` may be ``(N, D)``,
    ``cov`` another kernel and ``lscale`` a list of one length-scale per input (``covariance``)."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1 and cov == "exp_quad" and not isinstance(lscale, (list, tuple)):   # C03's one input, as it was written
        s = np.asarray(sdgp, dtype=float)[..., None, None]
        l = np.asarray(lscale, dtype=float)[..., None, None]
        d2 = np.subtract.outer(x, x) ** 2
        K = s * s * np.exp(-d2 / (2.0 * l * l)) + JITTER * np.eye(x.size)
    else:
        K = covariance(x, sdgp, lscale, cov)
    return (np.linalg.cholesky(K) @ np.asarray(z, dtype=float)[..., None])[..., 0]


def approximate_values(phi, slambda, sdgp, lscale, z, cov: str = "exp_quad") -> np.ndarray:
    """``f = phi (sqrt(S(slambda)) * z)`` at the rows of the basis ``phi`` (``(n, k)``), for draws of ``sdgp`` and ``lscale``
    of shape ``s`` and of ``z`` of shape ``(*s, k)``: ``(*s, n)``; ``cov`` the kernel (G09)."""
    beta = sqrt_spectral_density(slambda, sdgp, lscale, cov) * np.asarray(z, dtype=float)
    return np.einsum("...k,nk->...n", beta, np.asarray(phi, dtype=float))


# -- PyTensor -------------------------------------------------------------------------------------------------------------
def exact_effect(x, sdgp, lscale, z, cov: str = "exp_quad"):
    """``exact_values`` in PyTensor: ``x`` the distinct scaled inputs (NumPy), ``sdgp`` and ``lscale`` scalar variables, ``z``
    the ``(N,)`` standardised values. G09: ``x`` may be ``(N, D)`` (the kernel of the Euclidean distance), ``cov`` another
    kernel, and ``lscale`` a list of ``D`` scalar variables (``iso = FALSE``, the squared exponential only: the product of
    one-dimensional kernels, as brms writes it)."""
    import pytensor.tensor as pt

    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1 and cov == "exp_quad" and not isinstance(lscale, (list, tuple)):   # C03's one input, as it was written
        d2 = np.subtract.outer(x, x) ** 2                     # float64 constants throughout (a Python 2.0 would be float32)
        K = pt.sqr(sdgp) * pt.exp(-d2 / (np.float64(2.0) * pt.sqr(lscale))) + np.float64(JITTER) * np.eye(x.size)
        return pt.dot(pt.linalg.cholesky(K, lower=True), z)
    sq = squared_differences(x)
    if isinstance(lscale, (list, tuple)):
        if cov != "exp_quad":
            raise ValueError(f"one length-scale per input is offered for the squared-exponential kernel only, not {cov!r}")
        K = pt.sqr(sdgp)
        for d, l in enumerate(lscale):
            K = K * pt.exp(-sq[d] / (np.float64(2.0) * pt.sqr(l)))
    elif cov == "exp_quad":
        K = pt.sqr(sdgp) * pt.exp(-sq.sum(0) / (np.float64(2.0) * pt.sqr(lscale)))
    else:
        K = pt.sqr(sdgp) * _matern(np.sqrt(sq.sum(0)) / lscale, cov, pt.exp)
    K = K + np.float64(JITTER) * np.eye(sq.shape[1])
    return pt.dot(pt.linalg.cholesky(K, lower=True), z)


def approximate_effect(phi, slambda, sdgp, lscale, z, cov: str = "exp_quad"):
    """``approximate_values`` in PyTensor: ``phi`` the ``(N, k)`` basis at the distinct inputs and ``slambda`` the ``(k,)``
    frequencies (NumPy), ``sdgp`` and ``lscale`` scalar variables, ``z`` the ``(k,)`` standardised coefficients; ``cov`` the
    kernel (G09)."""
    import pytensor.tensor as pt

    w2 = np.asarray(slambda, dtype=np.float64) ** 2
    if cov == "exp_quad":
        root = sdgp * np.float64(_SQRT_SQRT_2PI) * pt.sqrt(lscale) * pt.exp(-pt.sqr(lscale) * w2 / np.float64(4.0))
    else:
        c, nu2, p = _SPD[cov]
        root = sdgp * np.float64(c) * pt.sqrt(lscale) * (np.float64(nu2) + pt.sqr(lscale) * w2) ** np.float64(-p / 2.0)
    return pt.dot(np.asarray(phi, dtype=np.float64), root * z)


def positive_rv(name: str, prior):
    """``sdgp`` or ``lscale`` under its positive-support prior (``HalfNormal``, ``HalfStudentT``, ``Exponential``,
    ``HalfCauchy``, ``Gamma``, ``InverseGamma``): a scalar PyMC variable named ``name``, sampled as its log (PyMC's default
    transform, Stan's for ``real<lower=0>``). Must be called inside a PyMC model context."""
    import pymc as pm

    return getattr(pm, prior.name)(name, **{k: np.float64(v) for k, v in prior.params.items()})


# -- the default prior of lscale ------------------------------------------------------------------------------------------
def inverse_gamma_tails(lower: float, upper: float, tail: float = TAIL) -> tuple[float, float]:
    """The shape ``a`` and scale ``b`` of the inverse gamma with ``P(X < lower) = P(X > upper) = tail``, solved exactly.

    With ``Y = 1 / X ~ Gamma(a, rate b)``: ``P(X > upper) = P(a, b / upper)`` (the regularised lower incomplete gamma), so
    ``b = upper P^{-1}(a, tail)``; then ``P(X < lower) = Q(a, b / lower)`` must be ``tail``, which leaves one equation in
    ``a``, ``g(a) = Q(a, (upper / lower) P^{-1}(a, tail)) - tail``, decreasing from ``1 - 2 tail`` (``a -> 0``) to ``-tail``
    (``a -> inf``) for ``lower < upper``; it is solved by bracketing on ``log a``."""
    from scipy.optimize import brentq
    from scipy.special import gammaincc, gammaincinv

    lower, upper = float(lower), float(upper)
    if not (0.0 < lower < upper and math.isfinite(upper)):
        raise ValueError(f"the tails of an inverse gamma need 0 < lower < upper, got {lower!r} and {upper!r}")
    ratio = upper / lower

    def g(log_a):
        a = math.exp(log_a)
        return float(gammaincc(a, ratio * gammaincinv(a, tail))) - tail

    lo, hi = math.log(1e-3), math.log(1e4)
    while g(hi) > 0.0:                                        # lower close to upper: a large shape (never for brms's bounds)
        lo, hi = hi, hi + math.log(1e2)
    log_a = brentq(g, lo, hi, xtol=1e-15, rtol=4 * np.finfo(float).eps, maxiter=500)
    a = math.exp(log_a)
    return a, upper * float(gammaincinv(a, tail))


def smallest_distance(x) -> float:
    """The smallest distance between two of the distinct values ``x`` (at least two); G09: between two of the distinct rows of
    ``x`` (``(N, D)``), Euclidean."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        v = np.sort(x)
        return float(np.min(np.diff(v)))
    d = np.sqrt(squared_differences(x).sum(0))
    return float(np.min(d[~np.eye(len(x), dtype=bool)]))


def largest_distance(x) -> float:
    """The largest Euclidean distance between two of the rows of ``x`` (``(N,)`` or ``(N, D)``; G09: brms's ``dmax``)."""
    return float(np.sqrt(squared_differences(x).sum(0)).max())
