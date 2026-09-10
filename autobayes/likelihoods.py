"""The Bernoulli-logit likelihood that is actually sampled, and why it is written by hand.

Bambi's built-in ``bernoulli`` family passes ``p = sigmoid(eta)`` with an exact 0 or 1
replaced by machine epsilon into ``pm.Bernoulli(p=...)``. Measured on this stack (pymc
5.28.5, pytensor 2.38.3): with two zero and two one observations the likelihood is
-72.087 instead of -80 at ``eta = 40`` and still -72.087 instead of -600 at ``eta = 300``
(and the same floor for large negative ``eta``), with the likelihood's contribution to
the gradient vanishing. That is a parameter-dependent change of the target density.

``pm.Bernoulli(logit_p=eta)`` has the right density but its gradient is NaN on the
PyTensor path at ``eta = 40`` (the untaken ``log1p(-p)`` branch is ``-inf`` and
``0 * inf`` is NaN). A ``pt.softplus`` likelihood is right on the PyTensor path but its
JAX lowering is a nested ``jnp.where`` whose untaken branch ``x + exp(-x)`` overflows, and
JAX's ``where`` gradient turns that into NaN at magnitude 1000. Writing the stable form
``(x + |x|)/2 + log1p(exp(-|x|))`` does not help: PyTensor rewrites ``log1p(exp(.))`` back
into its ``Softplus`` op before lowering.

So the log likelihood is ``y * eta - softplus(eta)`` with ``softplus`` a subclass of
PyTensor's ``Softplus`` (same value, C code and gradient ``sigmoid(eta)`` on the PyTensor
path) whose JAX lowering is ``jax.nn.softplus`` (``logaddexp(x, 0)``, stable value and
gradient for every finite ``x``). The subclass is a distinct op, so no rewrite replaces it
with the plain one. Verified in ``tests/test_bernoulli.py`` on both paths at logits up to
``1e6``, both signs, both outcomes (checks R1 and R2 in ``docs/m1-decision.md``).
"""
from __future__ import annotations

import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor.scalar.basic import upgrade_to_float
from pytensor.scalar.math import Softplus
from pytensor.tensor.elemwise import Elemwise
from scipy.special import expit


class StableSoftplus(Softplus):
    """``log(1 + exp(x))``: PyTensor's ``Softplus`` with a gradient-safe JAX lowering."""

    nfunc_spec = None


stable_softplus = Elemwise(StableSoftplus(upgrade_to_float, name="stable_softplus"))

try:  # the JAX lowering; absent JAX, the PyTensor path is all there is
    import jax
    from pytensor.link.jax.dispatch import jax_funcify

    @jax_funcify.register(StableSoftplus)
    def _jax_funcify_stable_softplus(op, **kwargs):
        return jax.nn.softplus
except ImportError:  # pragma: no cover
    pass


def bernoulli_logit_logp(value, logit_p):
    """``log Bernoulli(value | expit(logit_p))`` written to stay exact and differentiable."""
    return value * logit_p - stable_softplus(logit_p)


def bernoulli_logit_random(logit_p, rng=None, size=None):
    return rng.binomial(1, expit(logit_p), size=size).astype(np.int64)


class BernoulliLogit:
    """A PyMC distribution-like object Bambi can use as a likelihood: ``y ~ Bernoulli(expit(logit_p))``."""

    def __new__(cls, name, logit_p, **kwargs):
        return pm.CustomDist(name, logit_p, logp=bernoulli_logit_logp, random=bernoulli_logit_random,
                             dtype="int64", **kwargs)

    @staticmethod
    def dist(logit_p, **kwargs):
        return pm.CustomDist.dist(logit_p, logp=bernoulli_logit_logp, random=bernoulli_logit_random,
                                  dtype="int64", **kwargs)


def bernoulli_logit_family():
    """Bambi family: identity link on ``logit_p`` (the linear predictor), the likelihood above."""
    import bambi as bmb

    likelihood = bmb.Likelihood("BernoulliLogit", params=["logit_p"], parent="logit_p", dist=BernoulliLogit)
    return bmb.Family("bernoulli_logit", likelihood, link="identity")


# ---- binomial with a logit link (milestone M2) ---------------------------------------------
# log p(k | n, eta) = log C(n, k) - k softplus(-eta) - (n - k) softplus(eta). Both likelihood
# terms are a count times a non-positive stable quantity and are added, never subtracted, so
# nothing cancels at k = 0 or k = n; d/deta = k expit(-eta) - (n - k) expit(eta). Bambi's
# built-in binomial (eps-floored p) and pm.Binomial(logit_p=) were measured first and fail
# the tolerance on 49 and 60 of 130 grid points (docs/m2-decision.md).
#
# The coefficient is a constant in eta and is computed as DATA, per row, below, never as a
# float64 difference of three gammaln values: near a binomial mode the true log density is
# O(-10) while gammaln(n + 1) is ~n log n, so that difference carried an absolute error of
# ~1e-6 at n = 1e9 (repair record in docs/m2-decision.md). Even with an exact coefficient the
# likelihood terms are ~n ln 2 in magnitude near the mode, whose float64 spacing bounds the
# accuracy of the combined value; that is why counts are capped (formula.COUNT_CAP) at the
# largest tested value that meets the tolerance on both paths with a margin.

_EXACT_M = 1000          # exact summation of the coefficient for min(k, n - k) <= _EXACT_M
_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


def _stirlerr(x: float) -> float:
    """lgamma(x + 1) - (x + 1/2) log x + x - log(2 pi) / 2: the Stirling remainder. Series for
    x >= 16 (remainder below 1e-17 relative), lgamma below (moderate magnitudes, ~1e-15)."""
    if x < 16:
        return math.lgamma(x + 1.0) - (x + 0.5) * math.log(x) + x - _HALF_LOG_2PI
    x2 = x * x
    return (1.0 / 12 - (1.0 / 360 - (1.0 / 1260 - (1.0 / 1680 - 1.0 / 1188 / x2) / x2) / x2) / x2) / x


def log_binomial_coefficient(n, k):
    """log C(n, k) per row, accurate to ~1e-12 absolute for n <= 1e7 (measured against a 60-digit
    reference; docs/m2-decision.md). Exact summation sum_{j=1..m} log((n - m + j) / j) with
    m = min(k, n - k) when m <= 1000; otherwise n H(k/n) + c(n, k) with the Stirling
    remainders, which does not form any quantity larger than the result itself."""
    n = np.asarray(n, dtype=np.int64)
    k = np.asarray(k, dtype=np.int64)
    out = np.zeros(n.shape, dtype=float)
    for i, (ni, ki) in enumerate(zip(n.ravel(), k.ravel())):
        ni, ki = int(ni), int(ki)
        m = min(ki, ni - ki)
        if m <= 0:
            continue
        if m <= _EXACT_M:
            j = np.arange(1, m + 1, dtype=float)
            out.flat[i] = float(np.sum(np.log((ni - m + j) / j)))
        else:
            q = ki / ni
            entropy = -(q * math.log(q) + (1.0 - q) * math.log1p(-q))
            c = 0.5 * math.log(ni / (2.0 * math.pi * ki * (ni - ki))) + _stirlerr(ni) - _stirlerr(ki) - _stirlerr(ni - ki)
            out.flat[i] = ni * entropy + c
    return out


def binomial_logit_logp(value, logit_p, n, logc):
    return logc - value * stable_softplus(-logit_p) - (n - value) * stable_softplus(logit_p)


def binomial_logit_random(logit_p, n, logc, rng=None, size=None):
    return rng.binomial(n, expit(logit_p), size=size).astype(np.int64)


class BinomialLogit:
    """``k ~ Binomial(n, expit(logit_p))`` with the stable log density above; ``logc`` is the
    per-row ``log C(n, k)`` computed as data by ``log_binomial_coefficient``."""

    def __new__(cls, name, logit_p, n, logc, **kwargs):
        return pm.CustomDist(name, logit_p, n, logc, logp=binomial_logit_logp, random=binomial_logit_random,
                             dtype="int64", **kwargs)

    @staticmethod
    def dist(logit_p, n, logc, **kwargs):
        return pm.CustomDist.dist(logit_p, n, logc, logp=binomial_logit_logp, random=binomial_logit_random,
                                  dtype="int64", **kwargs)


def binomial_logit_family():
    """Bambi family for a two-column ``p(k, n)`` response: identity link on ``logit_p`` (the
    linear predictor); ``n`` reaches the likelihood the way Bambi's own binomial passes it,
    and the accurate coefficient rides along as a second data input."""
    import bambi as bmb

    class BinomialLogitFamily(bmb.Family):
        @staticmethod
        def transform_backend_kwargs(kwargs):
            observed = kwargs.pop("observed")
            k = np.asarray(observed[:, 0]).squeeze().astype(np.int64)
            n = np.asarray(observed[:, 1]).squeeze().astype(np.int64)
            kwargs["observed"] = k
            kwargs["n"] = n
            kwargs["logc"] = log_binomial_coefficient(n, k)
            return kwargs

    likelihood = bmb.Likelihood("BinomialLogit", params=["logit_p"], parent="logit_p", dist=BinomialLogit)
    return BinomialLogitFamily("binomial_logit", likelihood, link="identity")
