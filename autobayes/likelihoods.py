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
# log p(v | n, eta) = log C(n, v) - v softplus(-eta) - (n - v) softplus(eta) for integer v in
# [0, n], and -inf otherwise. Both likelihood terms are a count times a non-positive stable
# quantity and are added, never subtracted, so nothing cancels at v = 0 or v = n;
# d/deta = v expit(-eta) - (n - v) expit(eta). Bambi's built-in binomial (eps-floored p) and
# pm.Binomial(logit_p=) were measured first and fail the tolerance on 49 and 60 of 130 grid
# points (docs/m2-decision.md).
#
# The coefficient is a function of the EVALUATED value, computed inside the graph, so that
# the exposed RV is one normalised distribution whose logp and random agree for every
# outcome (correction record in docs/m2-decision.md: a fixed observed-outcome coefficient
# gave masses summing to 0.75 or 1.5 for n = 2). It is never formed as a float64 difference
# of three gammaln values of magnitude ~n log n: near a binomial mode the true log density
# is O(-10), and that difference carried ~1e-6 absolute error at n = 1e9 (repair record).
# Measured accuracy of the symbolic decomposition below against a 60-digit reference over the
# acceptance grid, identical on both paths: absolute error <= 1.2e-10 at n <= 1e6 (one float64
# ulp of the coefficient's own magnitude, worst at v = n/4) and 9.4e-10 at n = 1e7
# (docs/m2-evidence/correction/coefficient_accuracy.json). Even with an exact coefficient the
# likelihood terms are ~n ln 2 near the mode, whose float64 spacing bounds the combined value;
# that is why counts are capped (formula.COUNT_CAP) at the largest tested value that meets the
# tolerance on both paths with a margin.

_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


def _stirlerr_sym(x):
    """lgamma(x + 1) - (x + 1/2) log x + x - log(2 pi) / 2 for x >= 1: series for x >= 16, gammaln
    below. Both branches are finite for every x >= 1, so the switch never sees NaN."""
    x = pt.as_tensor_variable(x)
    small = pt.gammaln(x + 1.0) - (x + 0.5) * pt.log(x) + x - _HALF_LOG_2PI
    x2 = x * x
    series = (1.0 / 12 - (1.0 / 360 - (1.0 / 1260 - (1.0 / 1680 - 1.0 / 1188 / x2) / x2) / x2) / x2) / x
    return pt.switch(x < 16.0, small, series)


def log_binomial_coefficient_sym(n, v):
    """log C(n, v) as a graph in the evaluated value v: 0 at v in {0, n}; otherwise
    n H(v/n) + log(n / (2 pi v (n - v))) / 2 + stirlerr(n) - stirlerr(v) - stirlerr(n - v).
    The interior formula is evaluated on v and n - v clipped to [1, n], so no branch produces
    NaN or inf; the selection is a switch on the actual value. Callers mask invalid v."""
    n = pt.cast(pt.as_tensor_variable(n), "float64")
    v = pt.cast(pt.as_tensor_variable(v), "float64")
    a = pt.clip(v, 1.0, n)                 # = v         when 0 < v < n
    b = pt.clip(n - v, 1.0, n)             # = n - v     when 0 < v < n
    q, r = a / n, b / n
    entropy = -(q * pt.log(q) + r * pt.log(r))
    c = 0.5 * pt.log(n / (2.0 * math.pi * a * b)) + _stirlerr_sym(n) - _stirlerr_sym(a) - _stirlerr_sym(b)
    interior = n * entropy + c
    boundary = pt.or_(pt.eq(v, 0.0), pt.eq(v, n))
    return pt.switch(boundary, 0.0, interior)


def binomial_logit_logp(value, logit_p, n):
    v = pt.cast(pt.as_tensor_variable(value), "float64")
    nf = pt.cast(pt.as_tensor_variable(n), "float64")
    valid = pt.and_(pt.and_(pt.ge(v, 0.0), pt.le(v, nf)), pt.eq(v, pt.floor(v)))
    v_ok = pt.clip(pt.floor(v), 0.0, nf)   # finite arguments in the untaken branch
    body = log_binomial_coefficient_sym(nf, v_ok) - v_ok * stable_softplus(-logit_p) - (nf - v_ok) * stable_softplus(logit_p)
    return pt.switch(valid, body, -np.inf)


def binomial_logit_random(logit_p, n, rng=None, size=None):
    return rng.binomial(n, expit(logit_p), size=size).astype(np.int64)


class BinomialLogit:
    """``k ~ Binomial(n, expit(logit_p))`` with the stable, value-dependent log density above."""

    def __new__(cls, name, logit_p, n, **kwargs):
        return pm.CustomDist(name, logit_p, n, logp=binomial_logit_logp, random=binomial_logit_random,
                             dtype="int64", **kwargs)

    @staticmethod
    def dist(logit_p, n, **kwargs):
        return pm.CustomDist.dist(logit_p, n, logp=binomial_logit_logp, random=binomial_logit_random,
                                  dtype="int64", **kwargs)


def binomial_logit_family():
    """Bambi family for a two-column ``p(k, n)`` response: identity link on ``logit_p`` (the
    linear predictor); ``n`` reaches the likelihood the way Bambi's own binomial passes it."""
    import bambi as bmb

    class BinomialLogitFamily(bmb.Family):
        @staticmethod
        def transform_backend_kwargs(kwargs):
            observed = kwargs.pop("observed")
            kwargs["observed"] = np.asarray(observed[:, 0]).squeeze().astype(np.int64)
            kwargs["n"] = np.asarray(observed[:, 1]).squeeze().astype(np.int64)
            return kwargs

    likelihood = bmb.Likelihood("BinomialLogit", params=["logit_p"], parent="logit_p", dist=BinomialLogit)
    return BinomialLogitFamily("binomial_logit", likelihood, link="identity")
