"""Monotonic effects, brms's ``mo()`` (B03, docs/monotonic.md): the simplex on Stan's coordinates and the effect it builds.

brms's term ``mo(x)`` of an ordinal predictor coded ``0..D`` (``formula.Monotonic``, brms's ``Xmo``; ``D`` is its
``Jmo``) has a coefficient ``bsp`` and a simplex ``simo`` (``simplex[D]``, a share per step between consecutive values)
under a Dirichlet prior, and adds ``bsp * mo(simo, x_i)`` to the linear predictor with ``mo(simo, i) = D * sum(simo[1:i])``
(0 for ``i = 0``): ``bsp`` is the average change per step, ``bsp * D`` the change from the lowest value to the highest.

Stan samples a simplex on its stick-breaking coordinates (Stan 2.32's ``simplex_constrain``, the Stan behind the frozen
fixtures): ``D - 1`` unconstrained values ``y`` with ``z_k = inv_logit(y_k - log(D - k))`` (1-based ``k``),
``x_k = (1 - x_1 - ... - x_{k-1}) z_k``, ``x_D`` the rest of the stick, and the log Jacobian
``sum_k [log(stick_k) + log z_k + log(1 - z_k)]``. PyMC's own simplex transform is another map (log-ratios to the
geometric mean, with its own Jacobian), on which the same density has other coordinates and another gradient, so
``StickBreakingTransform`` is Stan's map as a PyMC transform, as ``correlation.CholeskyCorrTransform`` is for
``cholesky_corr``: the value variable holds Stan's coordinates and the Jacobian-adjusted density and every gradient
coordinate are comparable with ``rstan::log_prob``. ``stick_breaking_constrain`` / ``stick_breaking_free`` are the same map
in NumPy. The map is evaluated in log space (``log(1 - z) = -softplus(y - log(D - k))``), which equals Stan's
subtraction ``stick -= x_k`` up to rounding and stays exact where that subtraction loses the rest of the stick.
Nothing here depends on the model.
"""
from __future__ import annotations

import numpy as np
import pytensor.tensor as pt
from pymc.logprob.transforms import Transform

SIMPLEX_TOLERANCE = 1e-8          # Stan's check_simplex tolerance on |sum - 1| (CONSTRAINT_TOLERANCE)


def _offsets(D: int) -> np.ndarray:
    """``log(D - k)`` for ``k = 1..D-1``: Stan's shift, which makes ``y = 0`` the uniform simplex."""
    return np.log(np.arange(D - 1, 0, -1, dtype=float))


# -- Stan's simplex transform, NumPy ---------------------------------------------------------------------------
def stick_breaking_constrain(y, D: int) -> tuple[np.ndarray, float]:
    """Stan's ``simplex_constrain``: the ``D - 1`` unconstrained ``y`` to the simplex ``x`` (``D`` entries) and the log
    Jacobian of the map."""
    y = np.asarray(y, dtype=float).reshape(-1)
    if y.shape != (D - 1,):
        raise ValueError(f"y must have {D - 1} entries for a simplex of {D}, got {y.shape}")
    adj = y - _offsets(D)
    log_z, log_1mz = -np.logaddexp(0.0, -adj), -np.logaddexp(0.0, adj)
    log_stick = np.concatenate([[0.0], np.cumsum(log_1mz)])
    x = np.exp(np.concatenate([log_stick[:-1] + log_z, log_stick[-1:]]))
    return x, float(np.sum(log_stick[:-1] + log_z + log_1mz))


def stick_breaking_free(x) -> np.ndarray:
    """Stan's ``simplex_free``: the unconstrained coordinates of the simplex ``x``, ``y_k = logit(x_k / (x_k + ... + x_D)) +
    log(D - k)``, written as ``log(x_k) - log(x_{k+1} + ... + x_D) + log(D - k)``."""
    x = np.asarray(x, dtype=float).reshape(-1)
    rest = np.cumsum(x[::-1])[::-1]                         # x_k + ... + x_D
    return np.log(x[:-1]) - np.log(rest[1:]) + _offsets(x.size)


def check_simplex(x, D: int, what: str) -> np.ndarray:
    """``x`` as a float array, refused unless it is a simplex of ``D`` entries: finite, strictly positive (Stan's free
    transform takes the logarithm of each entry) and summing to 1 within Stan's tolerance."""
    arr = np.asarray(x, dtype=float)
    if arr.shape != (D,) or not np.isfinite(arr).all():
        raise ValueError(f"{what} must be a simplex of {D} finite entries, got shape {arr.shape}")
    if not (arr > 0).all():
        raise ValueError(f"{what} must have strictly positive entries (a simplex on Stan's coordinates)")
    if abs(arr.sum() - 1.0) > SIMPLEX_TOLERANCE:
        raise ValueError(f"{what} must sum to 1 (within {SIMPLEX_TOLERANCE:g}), got {arr.sum()!r}")
    return arr


# -- the same map as a PyMC value transform --------------------------------------------------------------------
class StickBreakingTransform(Transform):
    """Stan's simplex transform as a PyMC transform for a simplex of ``D`` entries: the value variable holds the ``D - 1``
    stick-breaking coordinates, ``backward`` builds the simplex, ``forward`` recovers the coordinates and ``log_jac_det``
    is Stan's Jacobian term, so the sampled space and the Jacobian-adjusted density are brms's."""

    name = "stickbreaking"
    ndim_supp = 1

    def __init__(self, D: int):
        if int(D) < 2:
            raise ValueError("a simplex on stick-breaking coordinates needs at least two entries")
        self.D = int(D)

    def _logs(self, value):
        from autobayes.likelihoods import stable_softplus   # a softplus whose JAX gradient is finite everywhere

        adj = value - pt.as_tensor(_offsets(self.D))
        log_z, log_1mz = -stable_softplus(-adj), -stable_softplus(adj)
        log_stick = pt.concatenate([pt.zeros_like(log_1mz[..., :1]), pt.cumsum(log_1mz, axis=-1)], axis=-1)
        return log_z, log_1mz, log_stick

    def backward(self, value, *inputs):
        log_z, _, log_stick = self._logs(value)
        return pt.exp(pt.concatenate([log_stick[..., :-1] + log_z, log_stick[..., -1:]], axis=-1))

    def forward(self, value, *inputs):
        rest = pt.cumsum(value[..., ::-1], axis=-1)[..., ::-1]
        return pt.log(value[..., :-1]) - pt.log(rest[..., 1:]) + pt.as_tensor(_offsets(self.D))

    def log_jac_det(self, value, *inputs):
        log_z, log_1mz, log_stick = self._logs(value)
        return pt.sum(log_stick[..., :-1] + log_z + log_1mz, axis=-1)


def simplex_rv(name: str, concentration, dims: str):
    """brms's ``simo`` with ``dirichlet_lpdf(simo | concentration)``: a PyMC ``Dirichlet`` named ``name`` with dims ``dims``
    (its coordinate must exist), sampled on Stan's stick-breaking coordinates, ``<name>_stickbreaking__``. Must be called
    inside a PyMC model context."""
    import pymc as pm

    a = np.asarray(concentration, dtype=np.float64)         # float64: a Python 2.0 would become a float32 constant
    return pm.Dirichlet(name, a=a, default_transform=StickBreakingTransform(a.size), dims=dims)


def coefficient_rv(name: str, prior):
    """The monotonic coefficient (brms's ``bsp``) under its location prior (``Normal``, ``StudentT`` or ``Flat``), a scalar
    PyMC variable named ``name``. Must be called inside a PyMC model context."""
    import pymc as pm

    return getattr(pm, prior.name)(name, **{k: np.float64(v) for k, v in prior.params.items()})


# -- the effect ----------------------------------------------------------------------------------------------
def monotonic_effect(simplex, codes, D: int):
    """``mo(simo, x_i) = D * sum(simo[1:x_i])`` per row (PyTensor), ``codes`` the rows' values coded ``0..D``: the
    cumulative shares with a leading zero, picked at each row's code."""
    cumulative = pt.concatenate([pt.zeros_like(simplex[..., :1]), pt.cumsum(simplex, axis=-1)], axis=-1)
    return np.float64(D) * cumulative[np.asarray(codes, dtype=np.int64)]


def monotonic_values(simplex, codes) -> np.ndarray:
    """``mo()`` per draw and row (NumPy): ``simplex`` is ``(..., D)``, ``codes`` ``(n,)``; the result is ``(..., n)``."""
    s = np.asarray(simplex, dtype=float)
    cumulative = np.concatenate([np.zeros(s.shape[:-1] + (1,)), np.cumsum(s, axis=-1)], axis=-1)
    return s.shape[-1] * cumulative[..., np.asarray(codes, dtype=np.int64)]
