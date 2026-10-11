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
from collections.abc import Callable, Mapping
from dataclasses import dataclass

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


# ---- lognormal (brms-parity A10) ----------------------------------------------------------------
# Bambi has no plain lognormal family (its ``hurdle_lognormal`` is a different model), so the family is
# PyMC's own ``LogNormal(mu, sigma)`` wrapped exactly as Bambi wraps ``Normal`` for ``gaussian``: the
# identity link on the location ``mu`` (the linear predictor) and ``sigma`` as the auxiliary parameter
# whose prior comes from the prior dict. ``lognormal_lpdf(y | mu, sigma)`` in Stan and PyMC's density are
# the same expression; equality with brms is checked in the oracle case ``lognormal_varint``.
def lognormal_family():
    """Bambi family for ``log(y) ~ Normal(mu, sigma)``: ``mu`` is the linear predictor, ``sigma`` the SD of ``log y``."""
    import bambi as bmb

    likelihood = bmb.Likelihood("LogNormal", params=["mu", "sigma"], parent="mu", dist=pm.LogNormal)
    return bmb.Family("lognormal", likelihood, link={"mu": "identity", "sigma": "log"})


# ---- Beta with a logit link (brms-parity A11) ----------------------------------------------------
# brms: ``beta_lpdf(Y | mu * phi, (1 - mu) * phi)`` with ``mu = inv_logit(eta)``. PyMC's ``Beta(alpha, beta)`` is the
# same density. ``1 - mu`` is computed as ``sigmoid(-eta)`` rather than by subtraction: the same number up to rounding
# (the oracle case ``beta_varint`` agrees to its tolerance), and it keeps ``beta > 0`` for ``|eta|`` up to about 745,
# where brms's subtraction returns exactly zero beyond ``eta`` of about 37 (docs/brms-parity/divergences.md, D13).
# Bambi's own ``beta`` family (mean ``mu`` and ``kappa``, which is brms's ``phi``) would name its deterministic ``mu``
# and hold ``expit(eta)``; this family keeps the linear predictor as the parent ``logit_mu`` under the identity link,
# as the Bernoulli and binomial families above do, and names the precision ``phi`` as brms does.
def _beta_shape_parameters(logit_mu, phi):
    return pt.sigmoid(logit_mu) * phi, pt.sigmoid(-logit_mu) * phi


class BetaLogit:
    """``y ~ Beta(mu phi, (1 - mu) phi)`` with ``mu = expit(logit_mu)``."""

    def __new__(cls, name, logit_mu, phi, **kwargs):
        alpha, beta = _beta_shape_parameters(logit_mu, phi)
        return pm.Beta(name, alpha=alpha, beta=beta, **kwargs)

    @staticmethod
    def dist(logit_mu, phi, **kwargs):
        alpha, beta = _beta_shape_parameters(logit_mu, phi)
        return pm.Beta.dist(alpha=alpha, beta=beta, **kwargs)


def beta_logit_family():
    """Bambi family: identity link on ``logit_mu`` (the linear predictor), the precision ``phi`` with its own prior."""
    import bambi as bmb

    likelihood = bmb.Likelihood("BetaLogit", params=["logit_mu", "phi"], parent="logit_mu", dist=BetaLogit)
    return bmb.Family("beta_logit", likelihood, link={"logit_mu": "identity", "phi": "log"})


# ---- the probit (brms-parity A12) and cloglog (P6) links for the Bernoulli and binomial families --------------
# brms: bernoulli_lpmf(Y | Phi(eta)) for link = "probit" and bernoulli_lpmf(Y | inv_cloglog(eta)) with
# inv_cloglog(eta) = 1 - exp(-exp(eta)) for link = "cloglog" (binomial_lpmf likewise). Stan's Phi(eta) is
# exactly 0 below eta = -37.5 and exactly 1 above 8.25, and 1 - exp(-exp(eta)) is exactly 0 below eta of
# about -37, so brms's forms give log(0) = -inf for the outcome the link makes improbable at large |eta|,
# with a NaN gradient. Here log P(y = 1 | eta) and log P(y = 0 | eta) are written so that value and gradient
# are finite for every finite eta on both the PyTensor and the JAX path, out of ops that have both C code
# and a JAX lowering (erfc, log1p, log1mexp, exp, switch, clip; erfcx, which PyMC's normal_lcdf uses, has
# neither on this stack):
#   probit   log P(y = 1) = log Phi(eta),  log P(y = 0) = log Phi(-eta)           (log_normal_cdf)
#   cloglog  log P(y = 1) = log(1 - exp(-exp(eta))) = log1mexp(-exp(eta)),  log P(y = 0) = -exp(eta)
# Every piecewise branch is evaluated on its argument clipped into the branch's own domain, so the branches
# a switch does not take are finite too, and their zero-weighted gradients never become 0 * inf (the NaN
# mechanism described above for pm.Bernoulli(logit_p=)). Clipping uses clip and switch, never maximum or
# minimum, whose JAX gradient at a tie is split in half while PyTensor's is not. Measured in
# tests/test_links.py against scipy's log_ndtr (probit, |eta| <= 300) and a 50-digit Decimal reference (cloglog,
# |eta| <= 40) on both paths.
#
# The cloglog forms below are exact where brms's own gradient is not (Stan forms 1 - theta by subtraction, losing
# eps * exp(exp(eta)) relative for a y = 0 row at eta above about 2.7), so the link was parked until P6 accepted it
# against two references: brms where brms is accurate, a committed 50-digit reference where it is not (docs/links.md,
# divergence D18). The hurdle Poisson (B06, below) uses cloglog_log_p1 for what it also is, log P(count > 0) =
# log(1 - exp(-exp(eta))) of a Poisson count.

_SQRT2 = math.sqrt(2.0)
_LOG_PHI_LOWER, _LOG_PHI_UPPER = -20.0, -1.0


def log_normal_cdf(z):
    """``log Phi(z)``, the standard Normal log-CDF, finite with a finite gradient for every finite ``z``.

    ``z >= -1``: ``log1p(-erfc(z / sqrt 2) / 2)`` (``1 - Phi(z)`` is at most 0.84 there and tiny for large ``z``,
    where ``log1p`` keeps the answer exact); ``-20 <= z < -1``: ``log(erfc(-z / sqrt 2) / 2)`` (``erfc`` is
    accurate relative to its value down to ``erfc(14.1) = 5e-89``); ``z < -20``: the asymptotic series of the
    Mills ratio, ``-z^2 / 2 - log(-z) - log(2 pi) / 2 + log(1 - 1/z^2 + 3/z^4 - 15/z^6 + ...)`` through the term
    ``-34459425/z^18`` (1.3e-16 at ``z = -20``; the first omitted term, ``654729075/z^20``, is 6e-18 there and smaller
    beyond; ``z`` is clipped at -1e18, where the value is -5e35, so that ``z^2`` stays finite).
    """
    z = pt.as_tensor_variable(z)
    z_hi = pt.switch(z < _LOG_PHI_UPPER, _LOG_PHI_UPPER, z)
    hi = pt.log1p(-0.5 * pt.erfc(z_hi / _SQRT2))
    z_mid = pt.clip(z, _LOG_PHI_LOWER, _LOG_PHI_UPPER)
    mid = pt.log(0.5 * pt.erfc(-z_mid / _SQRT2))
    z_lo = pt.clip(z, -1e18, _LOG_PHI_LOWER)
    w = 1.0 / (z_lo * z_lo)
    series = -1.0 + w * (3.0 + w * (-15.0 + w * (105.0 + w * (-945.0 + w * (10395.0 + w * (-135135.0 + w * (2027025.0 - w * 34459425.0)))))))
    lo = -0.5 * z_lo * z_lo - pt.log(-z_lo) - _HALF_LOG_2PI + pt.log1p(w * series)
    return pt.switch(z < _LOG_PHI_LOWER, lo, pt.switch(z < _LOG_PHI_UPPER, mid, hi))


_CLOGLOG_LOWER, _CLOGLOG_UPPER, _CLOGLOG_EXP_CAP = -20.0, 40.0, 700.0


def cloglog_log_p1(eta):
    """``log(1 - exp(-exp(eta)))``, log P(y = 1) under the cloglog link, finite with a finite gradient for every
    finite ``eta``: ``log1mexp(-exp(eta))`` for ``-20 <= eta <= 40`` (``exp(eta)`` from 2e-9 to 2e17; above 40
    the value is 0 to double precision and the gradient has underflowed to 0), and below -20
    ``eta + log1p(-exp(eta) / 2)``, from ``log(1 - e^-x) = log x + log(1 - x/2 + x^2/6 - ...)`` with
    ``x = exp(eta) < 2e-9`` (the omitted term is below 1e-18)."""
    eta = pt.as_tensor_variable(eta)
    e_mid = pt.clip(eta, _CLOGLOG_LOWER, _CLOGLOG_UPPER)
    mid = pt.log1mexp(-pt.exp(e_mid))                     # pytensor's log1mexp(x) is log(1 - exp(x)) for x <= 0
    e_lo = pt.switch(eta < _CLOGLOG_LOWER, eta, _CLOGLOG_LOWER)
    lo = e_lo + pt.log1p(-0.5 * pt.exp(e_lo))
    return pt.switch(eta < _CLOGLOG_LOWER, lo, mid)


def cloglog_log_p0(eta):
    """``-exp(eta)``, log P(y = 0) under the cloglog link, evaluated at ``min(eta, 700)`` so that it stays finite
    (about -1e304) where ``exp`` would overflow to -inf, as brms's form does; a sampler never reaches it."""
    eta = pt.as_tensor_variable(eta)
    return -pt.exp(pt.switch(eta > _CLOGLOG_EXP_CAP, _CLOGLOG_EXP_CAP, eta))


LINK_LOG_PROBABILITIES = {                                 # link -> (log P(y = 1 | eta), log P(y = 0 | eta))
    "probit": (log_normal_cdf, lambda eta: log_normal_cdf(-eta)),
    "cloglog": (cloglog_log_p1, cloglog_log_p0),
}


def _inverse_link(link: str):
    from autobayes.families.links import INVERSE_LINKS        # NumPy inverse links: ndtr, -expm1(-exp(eta))

    return INVERSE_LINKS[link]


def _bernoulli_link_dist(link: str):
    """Bambi hands the parent under its own name (``probit_p=eta``), as it hands ``logit_p`` to ``BernoulliLogit``."""
    log_p1, log_p0 = LINK_LOG_PROBABILITIES[link]
    inverse = _inverse_link(link)
    parent = f"{link}_p"

    def logp(value, eta):
        return pt.switch(pt.eq(value, 1), log_p1(eta), log_p0(eta))

    def random(eta, rng=None, size=None):
        return rng.binomial(1, inverse(eta), size=size).astype(np.int64)

    class BernoulliLink:
        """``y ~ Bernoulli(p)`` with ``p`` the link's inverse of ``eta``, the stable log probabilities above."""

        def __new__(cls, name, **kwargs):
            eta = kwargs.pop(parent)
            return pm.CustomDist(name, eta, logp=logp, random=random, dtype="int64", **kwargs)

        @staticmethod
        def dist(**kwargs):
            eta = kwargs.pop(parent)
            return pm.CustomDist.dist(eta, logp=logp, random=random, dtype="int64", **kwargs)

    return BernoulliLink


def bernoulli_link_family(link: str):
    """Bambi family for the Bernoulli response under the probit or the cloglog link: the identity link on the parent
    ``<link>_p``, which is the linear predictor (as ``bernoulli_logit_family`` keeps ``logit_p``), and the likelihood of
    ``LINK_LOG_PROBABILITIES[link]``."""
    import bambi as bmb

    parent = f"{link}_p"
    likelihood = bmb.Likelihood(f"Bernoulli{link.capitalize()}", params=[parent], parent=parent, dist=_bernoulli_link_dist(link))
    return bmb.Family(f"bernoulli_{link}", likelihood, link="identity")


def _binomial_link_dist(link: str):
    log_p1, log_p0 = LINK_LOG_PROBABILITIES[link]
    inverse = _inverse_link(link)

    def logp(value, eta, n):
        v = pt.cast(pt.as_tensor_variable(value), "float64")
        nf = pt.cast(pt.as_tensor_variable(n), "float64")
        valid = pt.and_(pt.and_(pt.ge(v, 0.0), pt.le(v, nf)), pt.eq(v, pt.floor(v)))
        v_ok = pt.clip(pt.floor(v), 0.0, nf)
        body = log_binomial_coefficient_sym(nf, v_ok) + v_ok * log_p1(eta) + (nf - v_ok) * log_p0(eta)
        return pt.switch(valid, body, -np.inf)

    def random(eta, n, rng=None, size=None):
        return rng.binomial(n, inverse(eta), size=size).astype(np.int64)

    parent = f"{link}_p"

    class BinomialLink:
        """``k ~ Binomial(n, p)`` with ``p`` the link's inverse of ``eta``: the coefficient of ``binomial_logit_logp``
        and the link's stable log probabilities, counts times non-positive finite terms."""

        def __new__(cls, name, n, **kwargs):
            eta = kwargs.pop(parent)
            return pm.CustomDist(name, eta, n, logp=logp, random=random, dtype="int64", **kwargs)

        @staticmethod
        def dist(n, **kwargs):
            eta = kwargs.pop(parent)
            return pm.CustomDist.dist(eta, n, logp=logp, random=random, dtype="int64", **kwargs)

    return BinomialLink


def binomial_link_family(link: str):
    """Bambi family for the binomial response ``p(k, n)`` under the probit or the cloglog link, built as
    ``binomial_logit_family`` is (the two-column response split in ``transform_backend_kwargs``)."""
    import bambi as bmb

    class BinomialLinkFamily(bmb.Family):
        @staticmethod
        def transform_backend_kwargs(kwargs):
            observed = kwargs.pop("observed")
            kwargs["observed"] = np.asarray(observed[:, 0]).squeeze().astype(np.int64)
            kwargs["n"] = np.asarray(observed[:, 1]).squeeze().astype(np.int64)
            return kwargs

    parent = f"{link}_p"
    likelihood = bmb.Likelihood(f"Binomial{link.capitalize()}", params=[parent], parent=parent, dist=_binomial_link_dist(link))
    return BinomialLinkFamily(f"binomial_{link}", likelihood, link="identity")


# ---- zero-inflated and hurdle Poisson (brms-parity B06) ---------------------------------------------------------------
# brms 2.23, with eta the linear predictor of the Poisson part (log link, mu = exp(eta)) and zi, hu in (0, 1):
#   zero_inflated_poisson_log_lpmf(y | eta, zi) = log_sum_exp(log(zi), log1m(zi) + poisson_log_lpmf(0 | eta))   y = 0
#                                                 log1m(zi) + poisson_log_lpmf(y | eta)                          y > 0
#   hurdle_poisson_log_lpmf(y | eta, hu)        = log(hu)                                                         y = 0
#                                                 log1m(hu) + poisson_log_lpmf(y | eta) - log1m_exp(-exp(eta))   y > 0
# with poisson_log_lpmf(y | eta) = y eta - exp(eta) - lgamma(y + 1). The families keep the linear predictor as the parent
# ``log_mu`` under the identity link (as the Beta keeps ``logit_mu``), so the likelihood sees eta itself, never
# log(exp(eta)); and they are parameterised by brms's zi and hu, not by PyMC's psi = 1 - zi (``ZeroInflatedPoisson``,
# ``HurdlePoisson``), so that the prior, the sampled coordinate logit(zi) and its gradient are brms's. The forms:
# - the y = 0 row of the zero-inflated Poisson is a two-term log_sum_exp written as hi + softplus(lo - hi), the larger term
#   chosen by a switch (at a tie each term gets half the gradient on both paths) and the softplus the stable one above
#   (PyTensor's own Softplus lowers to JAX as a where whose untaken branch overflows); log P_Poisson(0 | eta) = -exp(eta)
#   is evaluated at eta capped at 700, beyond which exp(-mu) is exactly 0 and the row is log(zi) exactly, where brms's
#   -exp(eta) overflows to -inf above eta = 709.78: the row keeps its value but its gradient is NaN (0 * inf);
# - the hurdle's zero truncation log(1 - exp(-mu)) is ``cloglog_log_p1(eta)`` above (the same function: the cloglog link's
#   log P(y = 1)), log(-expm1(-mu)) where that is exact and eta + log1p(-mu / 2) below eta = -20, where brms's
#   log1m_exp(-exp(eta)) has an infinite gradient once exp(eta) is subnormal (below eta = -709.78) and is -inf, the
#   row's log density +inf with a NaN gradient, below eta of about -745 (both seen with rstan::log_prob);
# - the y > 0 terms are evaluated at eta on y > 0 rows only (at 0 on the others), so that a branch the switch does not
#   take is finite with a finite gradient and never turns a zero adjoint into 0 * inf = NaN, on either path.
# A row's value is then finite with a finite gradient for every finite eta, except a y > 0 row beyond eta = 709.78, where
# the true log density, about -exp(eta), is below the most negative double and is -inf with gradient -inf, as brms's
# is; never NaN. Measured in tests/test_zero_inflated_hurdle.py against a 50-digit reference on both paths.

_POISSON_ZERO_CAP = 700.0     # exp(700) = 1.0e304: exp(-mu) is exactly 0 from here on (it underflows at mu = 745)


def _log_sum_exp2(a, b):
    """``log(exp(a) + exp(b))`` as ``hi + softplus(lo - hi)``, exact for every finite ``a`` and ``b``."""
    hi = pt.switch(a > b, a, b)
    lo = pt.switch(a > b, b, a)
    return hi + stable_softplus(lo - hi)


def poisson_log_lpmf(value, eta):
    """``log Poisson(value | exp(eta))`` = ``value * eta - exp(eta) - lgamma(value + 1)`` (Stan's ``poisson_log_lpmf``)."""
    y = pt.cast(value, "float64")
    return y * eta - pt.exp(eta) - pt.gammaln(y + 1.0)


def zero_inflated_poisson_logp(value, log_mu, zi):
    """brms's ``zero_inflated_poisson_log_lpmf(value | log_mu, zi)`` in the stable form described above; ``-inf`` for a
    negative value."""
    is_zero = pt.eq(value, 0)
    capped = pt.switch(log_mu > _POISSON_ZERO_CAP, _POISSON_ZERO_CAP, log_mu)
    zero = _log_sum_exp2(pt.log(zi), pt.log1p(-zi) - pt.exp(capped))
    positive = pt.log1p(-zi) + poisson_log_lpmf(value, pt.switch(is_zero, 0.0, log_mu))
    return pt.switch(pt.lt(value, 0), -np.inf, pt.switch(is_zero, zero, positive))


def hurdle_poisson_logp(value, log_mu, hu):
    """brms's ``hurdle_poisson_log_lpmf(value | log_mu, hu)`` in the stable form described above; ``-inf`` for a negative
    value."""
    is_zero = pt.eq(value, 0)
    eta = pt.switch(is_zero, 0.0, log_mu)
    positive = pt.log1p(-hu) + poisson_log_lpmf(value, eta) - cloglog_log_p1(eta)
    return pt.switch(pt.lt(value, 0), -np.inf, pt.switch(is_zero, pt.log(hu), positive))


def zero_truncated_poisson_draws(mu, rng):
    """Draws of a Poisson(mu) count conditioned on being positive, one per entry of ``mu``, as brms's
    ``posterior_predict_hurdle_poisson`` makes them: the first point of a unit-rate Poisson process on [0, mu], given that
    there is one, sits at ``t = -log(1 - U (1 - exp(-mu)))`` (here ``-log1p(U expm1(-mu))``, exact for small mu), and the
    count is 1 plus the Poisson(mu - t) points after it."""
    mu = np.asarray(mu, dtype=float)
    t = -np.log1p(rng.uniform(size=mu.shape) * np.expm1(-mu))
    return 1 + rng.poisson(np.clip(mu - t, 0.0, None))


def zero_inflated_poisson_random(log_mu, zi, rng=None, size=None):
    """brms's ``posterior_predict_zero_inflated_poisson``: a structural zero where a uniform draw falls below zi, a
    Poisson(exp(log_mu)) draw otherwise."""
    shape = size if size is not None else np.broadcast_shapes(np.shape(log_mu), np.shape(zi))
    zero = rng.uniform(size=shape) < zi
    return np.where(zero, 0, rng.poisson(np.exp(log_mu), size=shape)).astype(np.int64)


def hurdle_poisson_random(log_mu, hu, rng=None, size=None):
    """brms's ``posterior_predict_hurdle_poisson``: zero where a uniform draw falls below hu, a zero-truncated
    Poisson(exp(log_mu)) draw otherwise."""
    shape = size if size is not None else np.broadcast_shapes(np.shape(log_mu), np.shape(hu))
    zero = rng.uniform(size=shape) < hu
    return np.where(zero, 0, zero_truncated_poisson_draws(np.broadcast_to(np.exp(log_mu), shape), rng)).astype(np.int64)


class ZeroInflatedPoissonLog:
    """``y ~ ZeroInflatedPoisson(exp(log_mu), zi)``: zero with probability zi, Poisson(exp(log_mu)) otherwise."""

    def __new__(cls, name, log_mu, zi, **kwargs):
        return pm.CustomDist(name, log_mu, zi, logp=zero_inflated_poisson_logp, random=zero_inflated_poisson_random,
                             dtype="int64", **kwargs)

    @staticmethod
    def dist(log_mu, zi, **kwargs):
        return pm.CustomDist.dist(log_mu, zi, logp=zero_inflated_poisson_logp, random=zero_inflated_poisson_random,
                                  dtype="int64", **kwargs)


class HurdlePoissonLog:
    """``y ~ HurdlePoisson(exp(log_mu), hu)``: zero with probability hu, a zero-truncated Poisson(exp(log_mu)) otherwise."""

    def __new__(cls, name, log_mu, hu, **kwargs):
        return pm.CustomDist(name, log_mu, hu, logp=hurdle_poisson_logp, random=hurdle_poisson_random, dtype="int64", **kwargs)

    @staticmethod
    def dist(log_mu, hu, **kwargs):
        return pm.CustomDist.dist(log_mu, hu, logp=hurdle_poisson_logp, random=hurdle_poisson_random, dtype="int64", **kwargs)


def zero_inflated_poisson_family():
    """Bambi family: identity link on ``log_mu`` (the linear predictor), the zero-inflation probability ``zi`` with its own
    prior, the likelihood ``zero_inflated_poisson_logp``."""
    import bambi as bmb

    likelihood = bmb.Likelihood("ZeroInflatedPoissonLog", params=["log_mu", "zi"], parent="log_mu", dist=ZeroInflatedPoissonLog)
    return bmb.Family("zero_inflated_poisson_log", likelihood, link={"log_mu": "identity", "zi": "logit"})


def hurdle_poisson_family():
    """Bambi family: identity link on ``log_mu`` (the linear predictor), the hurdle probability ``hu`` with its own prior,
    the likelihood ``hurdle_poisson_logp``."""
    import bambi as bmb

    likelihood = bmb.Likelihood("HurdlePoissonLog", params=["log_mu", "hu"], parent="log_mu", dist=HurdlePoissonLog)
    return bmb.Family("hurdle_poisson_log", likelihood, link={"log_mu": "identity", "hu": "logit"})


# ---- brms's response addition terms (brms-parity B02, docs/addition-terms.md) --------------------------------------
# weights(w): brms adds weights[n] * family_lpdf(Y[n] | ...) row by row (its stan_log_lik_weights), so the likelihood here
# is the family's own elementwise log density times the row's weight, whatever the family's likelihood is (PyMC's own
# distribution or one of the hand-written ones above). cens(c) (-1 left, 0 none, 1 right) replaces a censored row's
# density by log P(Y <= y) or log P(Y > y), and trunc(lb, ub) subtracts the log mass between the bounds from every row,
# both through the family's Tails (Gaussian: NORMAL_TAILS); with weights, a row's whole log likelihood is weighted, as
# brms writes it for every combination accepted here (brms leaves the truncation mass unweighted only for cens() with
# trunc(), which is refused). Both were withheld until P6 (brms's own gradients of their frozen fixtures, and its
# truncated mean, are not exact everywhere; they are accepted against two references, D24, D25). PyMC's Normal log-CDF
# goes through erfcx, which has no JAX lowering on this stack, so the Gaussian's tails are built from log_normal_cdf above
# and erf, finite with a finite gradient on both paths for every finite argument, every piecewise branch evaluated on
# arguments clipped into its own domain.
#
# The likelihood is a pm.CustomDist whose draws are the family's own distribution (the truncated one under trunc(): brms's
# posterior_predict draws from the truncated distribution and ignores censoring and weights) and whose log density is the
# weighted, censored or truncated one, so model.logp(), the JAX density and pointwise log likelihoods all carry them.

def normal_log_cdf(value, mu, sigma):
    """``log P(Y <= value)`` for ``Y ~ Normal(mu, sigma)``: ``log Phi((value - mu) / sigma)``."""
    return log_normal_cdf((value - mu) / sigma)


def normal_log_ccdf(value, mu, sigma):
    """``log P(Y > value)`` for ``Y ~ Normal(mu, sigma)``: ``log Phi((mu - value) / sigma)``, exact in the upper tail
    (Stan's normal_lccdf forms ``1 - erf`` by subtraction there)."""
    return log_normal_cdf((mu - value) / sigma)


def normal_log_mass(lower, upper, mu, sigma):
    """``log P(lower < Y <= upper)`` for ``Y ~ Normal(mu, sigma)`` and finite ``lower < upper``. With ``a``, ``b`` the
    standardised bounds: above zero (``a > 0``) ``log Q(a) + log1mexp(log Q(b) - log Q(a))`` from the upper tail; below it
    (``b < 0``) ``log Phi(b) + log1mexp(log Phi(a) - log Phi(b))``; across it ``log((erf(b / sqrt 2) - erf(a / sqrt 2)) / 2)``,
    a sum of two non-negative terms. Each branch is evaluated on bounds replaced by a fixed valid pair where it is not
    taken, so that no untaken branch is infinite or NaN. Bounds closer than about 1e-6 SD keep about ``eps / (b - a)``
    relative accuracy in a tail (brms's ``log_diff_exp`` of two ``normal_lcdf`` loses more)."""
    a, b = (lower - mu) / sigma, (upper - mu) / sigma
    above, below = a > 0.0, b < 0.0
    a_hi, b_hi = pt.switch(above, a, 1.0), pt.switch(above, b, 2.0)
    lq_a, lq_b = log_normal_cdf(-a_hi), log_normal_cdf(-b_hi)
    upper_tail = lq_a + pt.log1mexp(lq_b - lq_a)
    a_lo, b_lo = pt.switch(below, a, -2.0), pt.switch(below, b, -1.0)
    lp_a, lp_b = log_normal_cdf(a_lo), log_normal_cdf(b_lo)
    lower_tail = lp_b + pt.log1mexp(lp_a - lp_b)
    across = pt.or_(above, below)
    a_mid, b_mid = pt.switch(across, -1.0, a), pt.switch(across, 1.0, b)
    middle = pt.log(0.5 * (pt.erf(b_mid / _SQRT2) - pt.erf(a_mid / _SQRT2)))
    return pt.switch(above, upper_tail, pt.switch(below, lower_tail, middle))


def _log_phi(x):
    with np.errstate(over="ignore"):
        return -0.5 * np.square(x) - _HALF_LOG_2PI


def _upper_mills(a, b):
    """``(phi(a) - phi(b)) / (Q(a) - Q(b))`` for ``0 < a < b <= inf``, in logs so that neither difference cancels in the
    tail (``log phi(b) - log phi(a)`` is formed as ``-(b - a)(b + a) / 2``; for bounds closer than about 1e-6 SD the
    difference of the two ``log Q`` keeps about ``eps / (b - a)`` relative accuracy)."""
    from scipy.special import log_ndtr

    lq_a, lq_b = log_ndtr(-a), log_ndtr(-b)
    with np.errstate(invalid="ignore", over="ignore"):
        dphi = np.where(np.isinf(b), -np.inf, -0.5 * (b - a) * (b + a))
        return np.exp(_log_phi(a) - lq_a) * -np.expm1(dphi) / -np.expm1(lq_b - lq_a)


def standard_truncated_mean(a, b):
    """``E[Z | a < Z < b]`` for a standard Normal ``Z`` and ``a < b`` (either may be infinite), exact in both tails: from the
    upper tail when ``a > 0``, by symmetry when ``b < 0``, directly across zero (``Phi(b) - Phi(a)`` as a sum of erfs).
    brms's posterior_epred forms ``(dnorm(a) - dnorm(b)) / (pnorm(b) - pnorm(a))`` by subtraction, which loses all its
    digits beyond about ``a = 8``."""
    from scipy.special import erf

    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    out = np.empty(a.shape)
    above, below = a > 0, b < 0
    middle = ~(above | below)
    out[above] = _upper_mills(a[above], b[above])
    out[below] = -_upper_mills(-b[below], -a[below])
    am, bm = a[middle], b[middle]
    with np.errstate(under="ignore"):
        num = np.exp(_log_phi(am)) - np.exp(_log_phi(bm))
    out[middle] = num / (0.5 * (erf(bm / _SQRT2) - erf(am / _SQRT2)))
    return out


def _normal_truncated_mean(model, post, eta, lb, ub):
    from autobayes.families.spec import auxiliary          # sigma per draw, or per draw and row (B01)

    sigma = auxiliary(model, post, "sigma")
    return eta + sigma * standard_truncated_mean((lb - eta) / sigma, (ub - eta) / sigma)


def _normal_draw_truncated(model, post, eta, lb, ub, rng):
    from scipy.stats import truncnorm

    from autobayes.families.spec import auxiliary

    sigma = np.broadcast_to(auxiliary(model, post, "sigma"), eta.shape)
    # size= keeps a leading chain axis of length one, which scipy squeezes away when it infers the shape itself
    return truncnorm.rvs((lb - eta) / sigma, (ub - eta) / sigma, loc=eta, scale=sigma, size=eta.shape, random_state=rng)


@dataclass(frozen=True)
class Tails:
    """What cens() and trunc() need from a family (B02): ``log_cdf(value, **params)``, ``log_ccdf(value, **params)`` and
    ``log_mass(lower, upper, **params)`` as PyTensor graphs of the likelihood's parameters by Bambi's names, finite with a
    finite gradient; ``truncated(lower, upper, size=, **params)`` the PyMC distribution truncated to the bounds (``None``
    for an open side), for draws; ``truncated_mean(model, post, eta, lb, ub)`` and ``draw_truncated(model, post, eta, lb,
    ub, rng)`` brms's posterior_epred and posterior_predict of a truncated model, from posterior draws (NumPy)."""

    log_cdf: Callable
    log_ccdf: Callable
    log_mass: Callable
    truncated: Callable
    truncated_mean: Callable
    draw_truncated: Callable


NORMAL_TAILS = Tails(
    log_cdf=normal_log_cdf, log_ccdf=normal_log_ccdf, log_mass=normal_log_mass,
    truncated=lambda lower, upper, *, size=None, mu, sigma: pm.TruncatedNormal.dist(mu=mu, sigma=sigma, lower=lower, upper=upper, size=size),
    truncated_mean=_normal_truncated_mean, draw_truncated=_normal_draw_truncated,
)


def _addition_dist(base_dist, data: Mapping[str, np.ndarray], tails: Tails | None):
    """The likelihood class Bambi calls with the family's parameters by name: ``base_dist`` (the family's PyMC
    distribution, or its hand-written class) with the addition terms of ``data`` (``formula.addition_data`` on the fitted
    rows) applied to its log density."""
    weights, cens = data.get("weights"), data.get("cens")
    lower = None if "lb" not in data or np.isneginf(data["lb"][0]) else float(data["lb"][0])   # numeric bounds, one per model
    upper = None if "ub" not in data or np.isposinf(data["ub"][0]) else float(data["ub"][0])
    truncated = lower is not None or upper is not None
    if (cens is not None or truncated) and tails is None:
        raise ValueError("cens() and trunc() need the family's tails (FamilySpec.tails)")

    def forms(keys):
        def draw(*args):
            *params, size = args
            kw = dict(zip(keys, params))
            return tails.truncated(lower, upper, size=size, **kw) if truncated else base_dist.dist(**kw, size=size)

        def logp(value, *params):
            kw = dict(zip(keys, params))
            ll = pm.logp(base_dist.dist(**kw), value)
            if cens is not None:                               # brms: lpdf if 0, lccdf if 1, lcdf if -1
                ll = pt.switch(pt.eq(cens, 1), tails.log_ccdf(value, **kw), pt.switch(pt.eq(cens, -1), tails.log_cdf(value, **kw), ll))
            if truncated:                                      # minus the log mass between the bounds, every row
                ll = ll - (tails.log_mass(lower, upper, **kw) if lower is not None and upper is not None
                           else tails.log_ccdf(lower, **kw) if lower is not None else tails.log_cdf(upper, **kw))
            return ll if weights is None else weights * ll    # each row's log likelihood times its weight
        return draw, logp

    class AdditionDist:
        """``y ~ family(...)`` with brms's addition terms on its log density; draws are the family's own (truncated)."""

        def __new__(cls, name, *, observed=None, dims=None, **params):
            keys = tuple(params)
            draw, logp = forms(keys)
            return pm.CustomDist(name, *(params[k] for k in keys), dist=draw, logp=logp, observed=observed, dims=dims)

        @staticmethod
        def dist(**params):
            return base_dist.dist(**params)

    return AdditionDist


def addition_family(base, family_spec, data: Mapping[str, np.ndarray]):
    """The Bambi family ``base`` (``FamilySpec.bambi_family()``: a built-in family's name or a family object) with brms's
    response addition terms applied to its likelihood; ``data`` is ``formula.addition_data`` on the fitted rows and
    ``family_spec`` the model's spec, whose ``tails`` give the censored and truncated forms. Everything else about the
    family (its parameters, links, priors, response handling) is the base family's own."""
    import copy

    import bambi as bmb
    from bambi.backend.utils import get_distribution_from_likelihood
    from bambi.defaults import get_builtin_family

    family = get_builtin_family(base) if isinstance(base, str) else copy.copy(base)
    lk = family.likelihood
    tails = family_spec.tails() if family_spec.tails is not None else None
    family.likelihood = bmb.Likelihood(lk.name, params=lk.params, parent=lk.parent,
                                       dist=_addition_dist(get_distribution_from_likelihood(lk), data, tails))
    return family


def monotonic_family(base, add_effects: Callable):
    """The Bambi family ``base`` (a built-in family's name or a family object) whose parent linear predictor gains brms's
    monotonic effects (B03, docs/monotonic.md): Bambi passes the parent's linear predictor through the family's
    ``transform_backend_eta`` inside the model it builds, before the link, and ``add_effects(eta)`` creates the terms'
    variables there and returns ``eta`` plus their contribution. A family's own ``transform_backend_eta`` runs after it;
    everything else about the family is the base family's."""
    import copy

    from bambi.defaults import get_builtin_family

    family = get_builtin_family(base) if isinstance(base, str) else copy.copy(base)
    own = getattr(family, "transform_backend_eta", None)

    def transform_backend_eta(eta, kwargs):
        eta = add_effects(eta)
        return eta if own is None else own(eta, kwargs)

    family.transform_backend_eta = transform_backend_eta
    return family


# ---- ordinal (cumulative) and categorical responses (brms-parity B05) --------------------------------------------------
# brms 2.23, with eta (brms's mu) the linear predictor and the response coded 1..K (brms's Y):
#   cumulative("logit"), disc = 1, no category-specific effects: ordered_logistic_lpmf(y | eta, tau), Stan's ordered
#   logistic with P(y <= k) = expit(tau_k - eta) for the K - 1 thresholds tau (brms's b_Intercept on the raw design):
#     y = 1         log expit(tau_1 - eta)                   = -softplus(eta - tau_1)
#     y = K         log expit(eta - tau_{K-1})               = -softplus(tau_{K-1} - eta)
#     1 < y < K     log(expit(tau_y - eta) - expit(tau_{y-1} - eta))
#                   = -softplus(eta - tau_y) - softplus(tau_{y-1} - eta) + log1mexp(tau_{y-1} - tau_y)
#   which is Stan's log_inv_logit_diff written out (expit(a) - expit(b) = (e^a - e^b) / ((1 + e^a)(1 + e^b))): two stable
#   softplus terms and the log of one minus the exponential of minus the gap between the two thresholds, never a
#   difference of probabilities, so a row is finite with a finite gradient for every finite eta;
#   categorical(), reference category 1: categorical_logit_lpmf(y | (0, eta_2, ..., eta_K)) = eta_y - log_sum_exp(0, eta).
# Each row reads its own thresholds (or category) through a one-hot selection on an index clipped into range, so the
# branch a switch does not take is evaluated at a valid index and stays finite (its zero adjoint never becomes 0 * inf).
# The positions along the last axis are a cumulative sum of ones, not an arange: JAX's arange lowering takes only
# constant or bare-shape endpoints (a model's category axis is neither once Bambi has built it), and the int8 endpoints
# of PyTensor's own take_along_axis overflow under NumPy 2's promotion rules once a run-time length exceeds 127. The
# thresholds reach the likelihood as brms's b_Intercept: the sampled Intercept (brms's thresholds of the centred design,
# on Stan's ordered coordinates) plus sum_p mean(x_p) b_p, a deterministic of the graph.

def _select(x, index):
    """``x[..., index]`` row by row: the entry of the last axis of ``x`` at ``index`` (``x``'s leading shape), by a one-hot
    switch on the positions ``0, 1, ...`` of that axis."""
    positions = pt.cumsum(pt.ones_like(x), axis=-1) - 1.0
    return pt.sum(pt.switch(pt.eq(index[..., None], positions), x, 0.0), axis=-1)


def ordered_logistic_logp(value, eta, thresholds):
    """``log P(y = value | eta, thresholds)`` of brms's ``cumulative("logit")`` (Stan's ``ordered_logistic_lpmf``) in the
    stable form above, for a 1-based category ``value``; ``-inf`` outside ``1..K``."""
    y = pt.cast(value, "int64")
    k = thresholds.shape[-1]
    thr = thresholds + pt.zeros_like(eta)[..., None]                   # one row of thresholds per observation
    upper = _select(thr, pt.clip(y - 1, 0, k - 1))
    lower = _select(thr, pt.clip(y - 2, 0, k - 1))
    has_upper, has_lower = pt.le(y, k), pt.ge(y, 2)
    both = pt.and_(has_upper, has_lower)
    out = (pt.switch(has_upper, -stable_softplus(eta - upper), 0.0)
           + pt.switch(has_lower, -stable_softplus(lower - eta), 0.0)
           + pt.switch(both, pt.log1mexp(pt.switch(both, lower - upper, -1.0)), 0.0))
    return pt.switch(pt.and_(pt.ge(y, 1), pt.le(y, k + 1)), out, -np.inf)


def cumulative_probabilities(eta, thresholds):
    """``P(y = k)``, ``k = 1..K``, under the cumulative logit model (brms's ``posterior_epred``): the differences of
    ``P(y <= k) = expit(tau_k - eta)`` with ``P(y <= 0) = 0`` and ``P(y <= K) = 1``, each evaluated as the exponential of
    the likelihood's stable log form, ``log expit(tau_k - eta) + log expit(eta - tau_{k-1}) + log(1 - exp(tau_{k-1} -
    tau_k))``, so that a category between two close thresholds keeps its relative precision (brms subtracts the
    cumulative probabilities, which loses it there; the two agree to rounding elsewhere). ``eta`` is ``(...)``,
    ``thresholds`` ``(..., K - 1)`` broadcastable against it; the result is ``(..., K)``."""
    d = np.asarray(thresholds, dtype=float) - np.asarray(eta, dtype=float)[..., None]        # tau_k - eta
    end = np.ones(d.shape[:-1] + (1,))
    hi = np.concatenate([d, np.inf * end], axis=-1)                                           # tau_k - eta, tau_K = inf
    lo = np.concatenate([-np.inf * end, d], axis=-1)                                          # tau_{k-1} - eta, tau_0 = -inf
    gap = lo - hi                                                                             # minus the gap, <= 0
    with np.errstate(divide="ignore", invalid="ignore"):
        log1mexp = np.where(gap > -math.log(2.0), np.log(-np.expm1(gap)), np.log1p(-np.exp(gap)))
    return np.exp(-np.logaddexp(0.0, -hi) - np.logaddexp(0.0, lo) + log1mexp)


def _log_sum_exp(x):
    """``log(sum(exp(x)))`` over the last axis as ``max + log(sum(exp(x - max)))``: exact for every finite ``x``, with a
    finite gradient on both paths. (PyTensor rewrites its own ``logsumexp`` into ``max + log(sum(switch(isinf(max),
    exp(max), exp(x - max))))``, whose untaken branch overflows once the maximum exceeds 709.78, and JAX's gradient of that
    switch is then NaN; here the rewrite, if it fires, sees a maximum of exactly 0.)"""
    m = pt.max(x, axis=-1, keepdims=True)
    return m[..., 0] + pt.log(pt.sum(pt.exp(x - m), axis=-1))


def categorical_logit_logp(value, eta):
    """``log P(y = value | eta)`` of brms's ``categorical()`` (Stan's ``categorical_logit_lpmf`` on ``(0, eta)``): ``eta``
    holds the linear predictors of categories ``2..K``, the reference category's is 0; ``-inf`` outside ``1..K``."""
    y = pt.cast(value, "int64")
    full = pt.concatenate([pt.zeros_like(eta[..., :1]), eta], axis=-1)
    k = eta.shape[-1]                                                  # K - 1: the reference category's predictor is 0
    picked = pt.switch(pt.eq(y, 1), 0.0, _select(eta, pt.clip(y - 2, 0, k - 1)))
    return pt.switch(pt.and_(pt.ge(y, 1), pt.le(y, k + 1)), picked - _log_sum_exp(full), -np.inf)


def categorical_probabilities(eta):
    """``P(y = k)``, ``k = 1..K``: the softmax of ``(0, eta_2, ..., eta_K)`` (brms's ``posterior_epred``); ``eta`` is
    ``(..., K - 1)`` and the result ``(..., K)``."""
    eta = np.asarray(eta, dtype=float)
    full = np.concatenate([np.zeros(eta.shape[:-1] + (1,)), eta], axis=-1)
    full = full - full.max(axis=-1, keepdims=True)
    e = np.exp(full)
    return e / e.sum(axis=-1, keepdims=True)


def draw_categories(probabilities, rng, size=None):
    """1-based category draws from ``probabilities`` ``(..., K)`` (rows summing to one): the number of cumulative
    probabilities a uniform draw exceeds, plus one, as brms's ``posterior_predict`` for these families draws them."""
    p = np.asarray(probabilities, dtype=float)
    shape = tuple(size) if size is not None else p.shape[:-1]
    cum = np.cumsum(np.broadcast_to(p, shape + p.shape[-1:]), axis=-1)[..., :-1]
    u = rng.uniform(size=shape)
    return (1 + (u[..., None] > cum).sum(axis=-1)).astype(np.int64)


def ordered_logistic_random(eta, thresholds, rng=None, size=None):
    return draw_categories(cumulative_probabilities(eta, thresholds), rng, size)


def categorical_logit_random(eta, rng=None, size=None):
    return draw_categories(categorical_probabilities(eta), rng, size)


_LOCATION_DRAWS = {                                   # location prior -> NumPy draws (the thresholds' forward sampler)
    "Normal": lambda rng, shape, mu, sigma: rng.normal(mu, sigma, size=shape),
    "StudentT": lambda rng, shape, nu, mu, sigma: mu + sigma * rng.standard_t(nu, size=shape),
}


def ordered_thresholds(name, prior_name: str, params: dict, labels, dim: str):
    """brms's ``ordered[nthres] Intercept`` with the same prior on every threshold (``Priors(intercept=...)``, brms's
    class ``Intercept``), as a PyMC variable named ``name`` with dims ``dim`` (coordinates ``labels``) sampled on Stan's
    ``ordered`` coordinates (PyMC's ``ordered`` transform: ``(tau_1, log(tau_2 - tau_1), ...)``, log Jacobian the sum of
    the log gaps). Its density is the prior's density summed over the thresholds as brms adds it (``lprior +=
    normal_lpdf(Intercept | ...)``), not renormalised to the ordered region; its forward draws are the prior's independent
    draws sorted, which is that density's distribution (the order statistics), so prior-predictive thresholds are
    ordered. Must be called inside a PyMC model context."""
    n = len(labels)
    names = list(params)
    # The first input only carries the number of thresholds (a flat prior has no parameter to carry it); then the prior's
    # parameters, one entry per threshold, in float64 (a Python 3.0 would otherwise become a float32 constant).
    values = [np.zeros(n)] + [np.full(n, np.float64(params[k])) for k in names]
    base = getattr(pm, prior_name)

    def logp(value, anchor, *ps):
        return pt.sum(pm.logp(base.dist(**dict(zip(names, ps))), value), axis=-1)

    def random(anchor, *ps, rng=None, size=None):
        if prior_name not in _LOCATION_DRAWS:
            raise NotImplementedError(f"a {prior_name} prior on the thresholds has no forward draws")
        core = np.broadcast_shapes(np.shape(anchor), *(np.shape(p) for p in ps))
        shape = (tuple(size) + core[-1:]) if size is not None else core
        return np.sort(_LOCATION_DRAWS[prior_name](rng, shape, *ps), axis=-1)

    start = float(params.get("mu", 0.0)) + np.arange(n) - (n - 1) / 2.0         # an ordered starting point at the location

    def support_point(rv, size, anchor, *ps):
        # the default initial point (not an initval, which pymc.model.fgraph refuses: A02's correlated terms edit the graph)
        return pt.broadcast_to(pt.as_tensor(start), rv.shape)

    model = pm.modelcontext(None)
    if dim not in model.coords:
        model.add_coord(dim, list(labels))
    return pm.CustomDist(name, *values, logp=logp, random=random, support_point=support_point,
                         signature=",".join(["(k)"] * len(values)) + "->(k)", transform=pm.distributions.transforms.ordered,
                         dims=dim)


def cumulative_logit_family(terms, means, dim: str):
    """Bambi family for brms's ``cumulative("logit")``: the linear predictor ``mu`` (brms's ``mu``, without thresholds) as
    the parent under the identity link and the thresholds ``Intercept`` (built by ``ordered_thresholds`` from the prior
    dict) as an auxiliary parameter. The likelihood adds ``sum_p means[p] b_p`` to the sampled thresholds, ``terms`` being
    the population-level terms in design order (PyMC variables of those names, scalar or one entry per column), and
    registers the result as the deterministic ``b_Intercept`` (brms's thresholds of the raw design), which is what
    ``ordered_logistic_logp`` reads with ``mu`` on the raw design: brms's likelihood with its centred thresholds."""
    import bambi as bmb

    means = np.asarray(means, dtype=np.float64)
    names = tuple(terms)

    class CumulativeLogit:
        """``y ~ OrderedLogistic(mu, b_Intercept)``, ``b_Intercept = Intercept + sum_p mean(x_p) b_p``."""

        def __new__(cls, name, mu, Intercept, **kwargs):
            model = pm.modelcontext(None)
            thresholds = Intercept
            if names:
                b = pt.concatenate([pt.atleast_1d(model[term]) for term in names])
                thresholds = Intercept + pt.dot(means, b)
            thresholds = pm.Deterministic("b_Intercept", thresholds, dims=dim)
            return pm.CustomDist(name, mu, thresholds, logp=ordered_logistic_logp, random=ordered_logistic_random,
                                 signature="(),(k)->()", dtype="int64", **kwargs)

    class CumulativeLogitFamily(bmb.Family):
        SUPPORTED_LINKS = {"mu": ["identity"], "Intercept": ["identity"]}

    likelihood = bmb.Likelihood("CumulativeLogit", params=["mu", "Intercept"], parent="mu", dist=CumulativeLogit)
    return CumulativeLogitFamily("cumulative_logit", likelihood, link={"mu": "identity", "Intercept": "identity"})


def no_intercept(name, **kwargs):
    """A Bambi intercept prior that adds nothing (and no variable) to the linear predictor: the cumulative family's
    formula keeps formulae's intercept, so that categorical predictors are treatment coded as brms codes them, while
    the thresholds take the intercept's place."""
    return pt.constant(np.float64(0.0))


def categorical_logit_family(categories, reduced_dim: str):
    """Bambi family for brms's ``categorical()``: Bambi's categorical family (one coefficient per non-reference category
    on every term, centred design, the intercept uncentred after sampling) with the linear predictors ``mu`` of the
    non-reference categories as the parent under the identity link (``(n, K - 1)``, brms's ``mu<k>``) and the
    likelihood ``categorical_logit_logp`` on the 1-based categories. ``categories`` are the response's categories in
    order, the first the reference; ``reduced_dim`` the coordinate of the non-reference ones."""
    import bambi as bmb
    from bambi.families.univariate import Categorical as BambiCategorical

    others = list(categories[1:])

    class CategoricalLogit:
        """``y ~ Categorical(softmax(0, mu))``."""

        def __new__(cls, name, mu, **kwargs):
            return pm.CustomDist(name, mu, logp=categorical_logit_logp, random=categorical_logit_random,
                                 signature="(k)->()", dtype="int64", **kwargs)

    class CategoricalLogitFamily(BambiCategorical):
        SUPPORTED_LINKS = {"mu": ["identity"]}

        def get_data(self, response):
            return np.asarray(response.term.data).squeeze().astype(np.int64)

        def get_levels(self, response):
            return list(others)

        def get_coords(self, response):
            return {reduced_dim: list(others)}

        def get_reference(self, response):
            return categories[0]

        @staticmethod
        def transform_backend_eta(eta, kwargs):
            return eta

    likelihood = bmb.Likelihood("CategoricalLogit", params=["mu"], parent="mu", dist=CategoricalLogit)
    return CategoricalLogitFamily("categorical_logit", likelihood, link={"mu": "identity"})
