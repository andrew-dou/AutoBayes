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


def sqrt_spectral_density(slambda, sdgp, lscale) -> np.ndarray:
    """``sqrt(S(w))`` of the squared-exponential kernel in one dimension at the frequencies ``slambda`` (``(k,)``), for
    ``sdgp`` and ``lscale`` of any common shape ``s``: ``(*s, k)``."""
    w = np.asarray(slambda, dtype=float)
    s = np.asarray(sdgp, dtype=float)[..., None]
    l = np.asarray(lscale, dtype=float)[..., None]
    return s * _SQRT_SQRT_2PI * np.sqrt(l) * np.exp(-(l * l) * (w * w) / 4.0)


def exact_values(x, sdgp, lscale, z) -> np.ndarray:
    """``f = chol(sdgp^2 exp(-d^2 / (2 lscale^2)) + 1e-12 I) z`` at the distinct scaled inputs ``x`` (``(N,)``), for draws of
    ``sdgp`` and ``lscale`` of shape ``s`` and of ``z`` of shape ``(*s, N)``: ``(*s, N)``."""
    x = np.asarray(x, dtype=float).reshape(-1)
    s = np.asarray(sdgp, dtype=float)[..., None, None]
    l = np.asarray(lscale, dtype=float)[..., None, None]
    d2 = np.subtract.outer(x, x) ** 2
    K = s * s * np.exp(-d2 / (2.0 * l * l)) + JITTER * np.eye(x.size)
    return (np.linalg.cholesky(K) @ np.asarray(z, dtype=float)[..., None])[..., 0]


def approximate_values(phi, slambda, sdgp, lscale, z) -> np.ndarray:
    """``f = phi (sqrt(S(slambda)) * z)`` at the rows of the basis ``phi`` (``(n, k)``), for draws of ``sdgp`` and ``lscale``
    of shape ``s`` and of ``z`` of shape ``(*s, k)``: ``(*s, n)``."""
    beta = sqrt_spectral_density(slambda, sdgp, lscale) * np.asarray(z, dtype=float)
    return np.einsum("...k,nk->...n", beta, np.asarray(phi, dtype=float))


# -- PyTensor -------------------------------------------------------------------------------------------------------------
def exact_effect(x, sdgp, lscale, z):
    """``exact_values`` in PyTensor: ``x`` the distinct scaled inputs (NumPy), ``sdgp`` and ``lscale`` scalar variables, ``z``
    the ``(N,)`` standardised values."""
    import pytensor.tensor as pt

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    d2 = np.subtract.outer(x, x) ** 2                         # float64 constants throughout (a Python 2.0 would be float32)
    K = pt.sqr(sdgp) * pt.exp(-d2 / (np.float64(2.0) * pt.sqr(lscale))) + np.float64(JITTER) * np.eye(x.size)
    return pt.dot(pt.linalg.cholesky(K, lower=True), z)


def approximate_effect(phi, slambda, sdgp, lscale, z):
    """``approximate_values`` in PyTensor: ``phi`` the ``(N, k)`` basis at the distinct inputs and ``slambda`` the ``(k,)``
    frequencies (NumPy), ``sdgp`` and ``lscale`` scalar variables, ``z`` the ``(k,)`` standardised coefficients."""
    import pytensor.tensor as pt

    w2 = np.asarray(slambda, dtype=np.float64) ** 2
    root = sdgp * np.float64(_SQRT_SQRT_2PI) * pt.sqrt(lscale) * pt.exp(-pt.sqr(lscale) * w2 / np.float64(4.0))
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
    """The smallest distance between two of the distinct values ``x`` (at least two)."""
    v = np.sort(np.asarray(x, dtype=float).reshape(-1))
    return float(np.min(np.diff(v)))
