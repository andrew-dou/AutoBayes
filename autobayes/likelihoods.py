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
``1e6``, both signs, both outcomes (checks R1 and R2 in ``docs/history/m1-decision.md``).
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
except ImportError:  # pragma: no cover - JAX is an optional extra; the pinned test environment always has it
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
# points (docs/history/m2-decision.md).
#
# The coefficient is a function of the EVALUATED value, computed inside the graph, so that
# the exposed RV is one normalised distribution whose logp and random agree for every
# outcome (correction record in docs/history/m2-decision.md: a fixed observed-outcome coefficient
# gave masses summing to 0.75 or 1.5 for n = 2). It is never formed as a float64 difference
# of three gammaln values of magnitude ~n log n: near a binomial mode the true log density
# is O(-10), and that difference carried ~1e-6 absolute error at n = 1e9 (repair record).
# Measured accuracy of the symbolic decomposition below against a 60-digit reference over the
# acceptance grid, identical on both paths: absolute error <= 1.2e-10 at n <= 1e6 (one float64
# ulp of the coefficient's own magnitude, worst at v = n/4) and 9.4e-10 at n = 1e7
# (docs/history/m2-evidence/correction/coefficient_accuracy.json). Even with an exact coefficient the
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


# ---- more zero-inflated and hurdle families (brms-parity D03, docs/zero-inflated-hurdle.md) ----------------------------
# brms 2.23's published densities ("Parameterization of Response Distributions in brms"), with eta the linear predictor of
# the count, proportion or positive part, and zi, hu in (0, 1) brms's zero-inflation and hurdle probabilities:
#   zero_inflated_negbinomial(eta, shape, zi)  y = 0: log(zi + (1 - zi) NB(0 | mu, shape)); y > 0: log1m(zi) + log NB(y | mu, shape)
#   hurdle_negbinomial(eta, shape, hu)         y = 0: log(hu);  y > 0: log1m(hu) + log NB(y | mu, shape) - log(1 - NB(0 | mu, shape))
#   zero_inflated_binomial(eta, zi; n)         y = 0: log(zi + (1 - zi) (1 - p)^n); y > 0: log1m(zi) + log Binomial(y | n, p)
#   zero_inflated_beta(eta, phi, zi)           y = 0: log(zi);  0 < y < 1: log1m(zi) + log Beta(y | mu phi, (1 - mu) phi)
#   hurdle_gamma(eta, shape, hu)               y = 0: log(hu);  y > 0: log1m(hu) + log Gamma(y | shape, shape / mu)
#   hurdle_lognormal(mu, sigma, hu)            y = 0: log(hu);  y > 0: log1m(hu) + log LogNormal(y | mu, sigma)
# with mu = exp(eta) (log link), p and mu = inv_logit(eta) (logit link) and NB brms's neg_binomial_2 (mean mu, variance
# mu + mu^2 / shape). Each family keeps its linear predictor as Bambi's parent under the identity link, as B06's do, and
# brms's names for its parameters (shape is Bambi's alpha, as for the negative binomial and Gamma families). The forms:
# - the negative binomial in eta: log NB(y | eta, shape) = log C(y + shape - 1, y) - shape softplus(eta - log shape)
#   - y softplus(log shape - eta), two non-positive terms added, so nothing cancels at large or small mu, the softplus the
#   stable one above; log C as lgamma(shape + y) - lgamma(shape) written through Stirling's remainder (``_log_rising``),
#   whose terms are of the size of the result for every shape, where the difference of two lgamma of size shape log(shape)
#   loses all its digits as shape grows (the Poisson limit);
# - its zero mass log NB(0 | eta, shape) = -shape softplus(eta - log shape) and the hurdle's log(1 - NB(0)) =
#   log(1 - exp(-exp(t))) with t = log shape + log softplus(eta - log shape), which is ``cloglog_log_p1(t)`` (B06's hurdle
#   truncation, the Poisson's t = eta in the limit shape -> infinity), log softplus written as x + log1p(-exp(x) / 2) below
#   x = -20 so that it never takes log(0);
# - the zero-inflated y = 0 rows are B06's two-term log-sum-exp; the y > 0 terms of every family are evaluated on y > 0
#   rows only (eta at 0 and the response at an interior value elsewhere, never at y = 0, where the Beta, Gamma and
#   lognormal densities are infinite), so that a branch a switch does not take is finite with a finite gradient.
# The Beta part is the Beta family's (``_beta_shape_parameters``, D13), the lognormal part PyMC's LogNormal (the lognormal
# family's), the Gamma part ``shape log(shape) - lgamma(shape) - shape eta + (shape - 1) log y - shape y exp(-eta)`` written in
# eta (no log(exp(eta))); the binomial part ``binomial_logit_logp`` (M2). Checked against brms in the oracle cases
# zero_inflated_negbinomial_varint, hurdle_negbinomial_varint, zero_inflated_binomial_varint, zero_inflated_beta_varint,
# hurdle_gamma_varint and hurdle_lognormal_varint; in tests/test_zero_inflated_hurdle_more.py the negative binomial and
# binomial rows against 50-digit references (value, and gradient on both paths: PyTensor's goes through its digamma, the
# gradient of gammaln, which is 1.5e-10 off relative between 1 and 12, JAX's is exact to rounding) and the others against SciPy.

_LOG_SOFTPLUS_LOWER = -20.0
_STIRLING_SWITCH = 16.0
_F64_HALF, _F64_ONE = np.float64(0.5), np.float64(1.0)


def log_softplus(x):
    """``log(log(1 + exp(x)))``, finite with a finite gradient for every finite ``x``: ``log(softplus(x))`` from ``x = -20``
    (``softplus`` is at least 2e-9 there) and ``x + log1p(-exp(x) / 2)`` below, from ``log(1 + e^x) = e^x (1 - e^x / 2 +
    e^2x / 3 - ...)`` (the omitted term is ``5 e^2x / 24``, below 1e-18 there)."""
    x = pt.as_tensor_variable(x)
    low = x < _LOG_SOFTPLUS_LOWER
    x_lo = pt.switch(low, x, _LOG_SOFTPLUS_LOWER)
    x_hi = pt.switch(low, _LOG_SOFTPLUS_LOWER, x)
    return pt.switch(low, x_lo + pt.log1p(-_F64_HALF * pt.exp(x_lo)), pt.log(stable_softplus(x_hi)))


def _stirling_remainder(x):
    """``lgamma(x) - ((x - 1/2) log x - x + log(2 pi) / 2)`` for ``x > 0``: ``_stirlerr_sym``'s two branches, each evaluated
    on ``x`` moved into its own range, so that the series is never evaluated at a tiny ``x``, where its powers of ``1 / x``
    overflow."""
    x = pt.as_tensor_variable(x)
    small = x < _STIRLING_SWITCH
    x_lo = pt.switch(small, x, _STIRLING_SWITCH)
    x_hi = pt.switch(small, _STIRLING_SWITCH, x)
    lo = pt.gammaln(x_lo + _F64_ONE) - (x_lo + _F64_HALF) * pt.log(x_lo) + x_lo - _HALF_LOG_2PI
    x2 = x_hi * x_hi
    hi = (1.0 / 12 - (1.0 / 360 - (1.0 / 1260 - (1.0 / 1680 - 1.0 / 1188 / x2) / x2) / x2) / x2) / x_hi
    return pt.switch(small, lo, hi)


def _log_rising(shape, y):
    """``lgamma(shape + y) - lgamma(shape)`` for ``shape > 0`` and ``y >= 0``, written as ``(shape - 1/2) log1p(y / shape) +
    y log(shape + y) - y`` plus the difference of the two Stirling remainders (Stirling's formula for each lgamma): every
    term is at most of the size of the result, whatever ``shape``."""
    b = shape + y
    return ((shape - _F64_HALF) * pt.log1p(y / shape) + y * pt.log(b) - y
            + _stirling_remainder(b) - _stirling_remainder(shape))


def log_nb_coefficient(y, shape):
    """``log C(y + shape - 1, y)`` = ``lgamma(y + shape) - lgamma(shape) - lgamma(y + 1)``: exactly 0 at ``y = 0``, else
    ``_log_rising``, evaluated at ``y = 1`` on the zero rows so that no branch divides by zero."""
    y = pt.as_tensor_variable(y)
    zero = pt.eq(y, 0.0)
    y_pos = pt.switch(zero, _F64_ONE, y)
    return pt.switch(zero, 0.0, _log_rising(shape, y_pos) - pt.gammaln(y_pos + _F64_ONE))


def neg_binomial_2_log_lpmf(value, eta, shape):
    """brms's ``neg_binomial_2_log_lpmf(value | eta, shape)``: ``log C(y + shape - 1, y) - shape softplus(eta - log shape) - y
    softplus(log shape - eta)``, i.e. ``shape log(shape / (mu + shape)) + y log(mu / (mu + shape))`` with ``mu = exp(eta)``."""
    y = pt.cast(value, "float64")
    log_shape = pt.log(shape)
    return log_nb_coefficient(y, shape) - shape * stable_softplus(eta - log_shape) - y * stable_softplus(log_shape - eta)


def neg_binomial_log_zero(eta, shape):
    """``log NB(0 | exp(eta), shape)`` = ``shape log(shape / (mu + shape))`` = ``-shape softplus(eta - log shape)``."""
    return -shape * stable_softplus(eta - pt.log(shape))


def neg_binomial_log_positive(eta, shape):
    """``log(1 - NB(0 | exp(eta), shape))``, the hurdle negative binomial's zero truncation: ``cloglog_log_p1(t)`` with
    ``t = log(-log NB(0)) = log shape + log softplus(eta - log shape)``, finite with a finite gradient for every finite
    ``eta`` and ``shape``."""
    log_shape = pt.log(shape)
    return cloglog_log_p1(log_shape + log_softplus(eta - log_shape))


def zero_inflated_negbinomial_logp(value, log_mu, alpha, zi):
    """brms's ``zero_inflated_neg_binomial_log_lpmf(value | log_mu, shape, zi)`` with ``alpha`` the shape, in the stable form
    above; ``-inf`` for a negative value."""
    is_zero = pt.eq(value, 0)
    zero = _log_sum_exp2(pt.log(zi), pt.log1p(-zi) + neg_binomial_log_zero(log_mu, alpha))
    positive = pt.log1p(-zi) + neg_binomial_2_log_lpmf(value, pt.switch(is_zero, 0.0, log_mu), alpha)
    return pt.switch(pt.lt(value, 0), -np.inf, pt.switch(is_zero, zero, positive))


def hurdle_negbinomial_logp(value, log_mu, alpha, hu):
    """brms's ``hurdle_neg_binomial_log_lpmf(value | log_mu, shape, hu)`` with ``alpha`` the shape, in the stable form above;
    ``-inf`` for a negative value."""
    is_zero = pt.eq(value, 0)
    eta = pt.switch(is_zero, 0.0, log_mu)
    positive = pt.log1p(-hu) + neg_binomial_2_log_lpmf(value, eta, alpha) - neg_binomial_log_positive(eta, alpha)
    return pt.switch(pt.lt(value, 0), -np.inf, pt.switch(is_zero, pt.log(hu), positive))


def zero_inflated_binomial_logp(value, logit_p, zi, n):
    """brms's ``zero_inflated_binomial_logit_lpmf(value | n, logit_p, zi)``: the zero row ``log(zi + (1 - zi) exp(-n
    softplus(eta)))`` as B06's log-sum-exp, the others ``log1m(zi)`` plus ``binomial_logit_logp`` (``-inf`` outside 0..n)."""
    nf = pt.cast(pt.as_tensor_variable(n), "float64")
    zero = _log_sum_exp2(pt.log(zi), pt.log1p(-zi) - nf * stable_softplus(logit_p))
    positive = pt.log1p(-zi) + binomial_logit_logp(value, logit_p, n)
    return pt.switch(pt.eq(value, 0), zero, positive)


def zero_inflated_beta_logp(value, logit_mu, phi, zi):
    """brms's ``zero_inflated_beta_lpdf(value | mu, phi, zi)`` with ``mu = inv_logit(logit_mu)``: ``log(zi)`` at 0, else
    ``log1m(zi)`` plus the Beta family's density; ``-inf`` outside [0, 1) (at 1 the Beta density itself is infinite for
    ``(1 - mu) phi < 1``, and 1 is outside the family's support, as brms's data check says)."""
    is_zero = pt.eq(value, 0)
    inside = pt.and_(pt.gt(value, 0), pt.lt(value, 1))
    # the Beta part at eta = 0 and y = 1/2 on the other rows: PyMC's Beta checks alpha > 0 and beta > 0 over all rows at
    # once, and expit(-eta) phi underflows to 0 beyond |eta| = 745, which on a zero row would void the whole likelihood
    a, b = _beta_shape_parameters(pt.switch(inside, logit_mu, 0.0), phi)
    positive = pt.log1p(-zi) + pm.logp(pm.Beta.dist(alpha=a, beta=b), pt.switch(inside, value, _F64_HALF))
    return pt.switch(is_zero, pt.log(zi), pt.switch(inside, positive, -np.inf))


def gamma_log_lpdf(value, eta, shape):
    """``log Gamma(value | shape, rate = shape / exp(eta))`` (brms's ``gamma_lpdf(y | shape, shape / mu)``) written in ``eta``."""
    return (shape * pt.log(shape) - pt.gammaln(shape) - shape * eta + (shape - _F64_ONE) * pt.log(value)
            - shape * value * pt.exp(-eta))


def hurdle_gamma_logp(value, log_mu, alpha, hu):
    """brms's ``hurdle_gamma_lpdf(value | shape, shape / mu, hu)`` with ``alpha`` the shape: ``log(hu)`` at 0, else ``log1m(hu)``
    plus the Gamma density; ``-inf`` for a negative value."""
    is_zero = pt.eq(value, 0)
    y = pt.switch(is_zero, _F64_ONE, value)
    positive = pt.log1p(-hu) + gamma_log_lpdf(y, pt.switch(is_zero, 0.0, log_mu), alpha)
    return pt.switch(pt.lt(value, 0), -np.inf, pt.switch(is_zero, pt.log(hu), positive))


def hurdle_lognormal_logp(value, mu, sigma, hu):
    """brms's ``hurdle_lognormal_lpdf(value | mu, sigma, hu)``: ``log(hu)`` at 0, else ``log1m(hu)`` plus PyMC's LogNormal
    density (``-inf`` for a negative value)."""
    is_zero = pt.eq(value, 0)
    positive = pt.log1p(-hu) + pm.logp(pm.LogNormal.dist(mu=mu, sigma=sigma), pt.switch(is_zero, _F64_ONE, value))
    return pt.switch(is_zero, pt.log(hu), positive)


_POISSON_DRAW_LIMIT = np.iinfo(np.int64).max - 10 * np.sqrt(float(np.iinfo(np.int64).max))   # numpy's Poisson generator's range


def zero_truncated_negbinomial_draws(mu, shape, rng):
    """Draws of a ``NB(mu, shape)`` count (brms's ``neg_binomial_2``) conditioned on being positive, one per entry, exact for
    every ``mu >= 0`` and ``shape > 0`` in O(1) per draw. The count is Poisson(lambda) with lambda ~ Gamma(shape, rate =
    shape / mu); given ``y > 0`` it is a zero-truncated Poisson(lambda) with lambda drawn from its conditional density
    ``Gamma(lambda | shape, shape / mu) (1 - exp(-lambda)) / P(y > 0)``. Writing ``1 - exp(-lambda)`` as ``int_0^1 lambda
    exp(-u lambda) du`` makes that density the lambda-marginal of ``(lambda, u)`` with density proportional to ``lambda^shape
    exp(-lambda (c + u))`` on ``u`` in (0, 1), ``c = shape / mu``: ``u`` has the CDF ``(1 - ((c + u) / c)^-shape) / P(y > 0)``,
    which inverts in closed form, and ``lambda | u ~ Gamma(shape + 1, rate = c + u)``. With ``V`` uniform the rate is
    ``c (1 - V P(y > 0))^(-1 / shape)``, so ``lambda = Gamma(shape + 1) * (mu / shape) (1 - V P(y > 0))^(1 / shape)``, the
    power formed as ``exp(log1p(-V P(y > 0)) / shape)`` and ``P(y > 0) = -expm1(-shape log1p(mu / shape))``, exact for small
    ``mu`` and small ``shape``; the count is then ``zero_truncated_poisson_draws(lambda)``. 1 where ``mu`` has underflowed to
    0 (the limit). A non-positive or non-finite shape, and a lambda beyond the integer generator's range, are refused."""
    mu, shape = np.broadcast_arrays(np.asarray(mu, dtype=float), np.asarray(shape, dtype=float))
    if not (np.all(np.isfinite(shape)) and np.all(shape > 0) and np.all(np.isfinite(mu)) and np.all(mu >= 0)):
        raise ValueError("zero-truncated negative binomial draws need a finite mean mu >= 0 and a finite shape > 0")
    live = mu > 0
    m, s = np.where(live, mu, 1.0), np.where(live, shape, 1.0)
    positive = -np.expm1(-s * np.log1p(m / s))                        # P(y > 0)
    v = rng.uniform(size=mu.shape)
    with np.errstate(over="ignore"):
        scale = np.exp(np.log(m) - np.log(s) + np.log1p(-v * positive) / s)
        lam = rng.gamma(s + 1.0, scale)
    if not (np.all(np.isfinite(lam)) and np.all(lam < _POISSON_DRAW_LIMIT)):
        raise ValueError("zero-truncated negative binomial draws exceed the integer random-generator limit; inspect the mean and "
                         "shape draws")
    return np.where(live, zero_truncated_poisson_draws(lam, rng), 1).astype(np.int64)


def _zero_part(zero_probability, shape, rng):
    """Where a draw is the zero part: a uniform draw below zi or hu, as brms's posterior_predict decides it."""
    return rng.uniform(size=shape) < zero_probability


def zero_inflated_negbinomial_random(log_mu, alpha, zi, rng=None, size=None):
    shape = size if size is not None else np.broadcast_shapes(np.shape(log_mu), np.shape(alpha), np.shape(zi))
    zero = _zero_part(zi, shape, rng)
    mu = np.broadcast_to(np.exp(log_mu), shape)
    return np.where(zero, 0, rng.negative_binomial(alpha, alpha / (alpha + mu), size=shape)).astype(np.int64)


def hurdle_negbinomial_random(log_mu, alpha, hu, rng=None, size=None):
    shape = size if size is not None else np.broadcast_shapes(np.shape(log_mu), np.shape(alpha), np.shape(hu))
    zero = _zero_part(hu, shape, rng)
    counts = zero_truncated_negbinomial_draws(np.broadcast_to(np.exp(log_mu), shape), np.broadcast_to(alpha, shape), rng)
    return np.where(zero, 0, counts).astype(np.int64)


def zero_inflated_binomial_random(logit_p, zi, n, rng=None, size=None):
    shape = size if size is not None else np.broadcast_shapes(np.shape(logit_p), np.shape(zi), np.shape(n))
    zero = _zero_part(zi, shape, rng)
    return np.where(zero, 0, rng.binomial(n, expit(logit_p), size=shape)).astype(np.int64)


def zero_inflated_beta_random(logit_mu, phi, zi, rng=None, size=None):
    shape = size if size is not None else np.broadcast_shapes(np.shape(logit_mu), np.shape(phi), np.shape(zi))
    zero = _zero_part(zi, shape, rng)
    return np.where(zero, 0.0, rng.beta(expit(logit_mu) * phi, expit(-np.asarray(logit_mu)) * phi, size=shape))


def hurdle_gamma_random(log_mu, alpha, hu, rng=None, size=None):
    shape = size if size is not None else np.broadcast_shapes(np.shape(log_mu), np.shape(alpha), np.shape(hu))
    zero = _zero_part(hu, shape, rng)
    return np.where(zero, 0.0, rng.gamma(alpha, np.exp(log_mu) / alpha, size=shape))


def hurdle_lognormal_random(mu, sigma, hu, rng=None, size=None):
    shape = size if size is not None else np.broadcast_shapes(np.shape(mu), np.shape(sigma), np.shape(hu))
    zero = _zero_part(hu, shape, rng)
    return np.where(zero, 0.0, rng.lognormal(mu, sigma, size=shape))


def _zero_family(name: str, label: str, params: tuple[str, ...], links: dict, logp, random, dtype: str | None = None,
                 trials: bool = False):
    """A Bambi family whose likelihood is ``logp`` / ``random`` (a ``pm.CustomDist``) of the parent ``params[0]`` and the
    auxiliary parameters ``params[1:]``, each with its prior from the prior dict; ``dtype="int64"`` for counts. ``trials``:
    a two-column ``p(k, n)`` response whose trials reach the likelihood as ``n``, as ``binomial_logit_family`` hands them."""
    import bambi as bmb

    names = (*params, "n") if trials else params
    extra = {"dtype": dtype} if dtype is not None else {}

    class Dist:
        def __new__(cls, name_, **kwargs):
            args = [kwargs.pop(p) for p in names]
            return pm.CustomDist(name_, *args, logp=logp, random=random, **extra, **kwargs)

        @staticmethod
        def dist(**kwargs):
            args = [kwargs.pop(p) for p in names]
            return pm.CustomDist.dist(*args, logp=logp, random=random, **extra, **kwargs)

    Dist.__name__ = label
    likelihood = bmb.Likelihood(label, params=list(params), parent=params[0], dist=Dist)
    if not trials:
        return bmb.Family(name, likelihood, link=links)

    class TrialsFamily(bmb.Family):
        @staticmethod
        def transform_backend_kwargs(kwargs):
            observed = kwargs.pop("observed")
            kwargs["observed"] = np.asarray(observed[:, 0]).squeeze().astype(np.int64)
            kwargs["n"] = np.asarray(observed[:, 1]).squeeze().astype(np.int64)
            return kwargs

    return TrialsFamily(name, likelihood, link=links)


def zero_inflated_negbinomial_family():
    """Bambi family of brms's ``zero_inflated_negbinomial()``: the identity link on ``log_mu`` (the linear predictor), the
    shape (Bambi's ``alpha``) and ``zi`` with their priors."""
    return _zero_family("zero_inflated_negbinomial_log", "ZeroInflatedNegBinomialLog", ("log_mu", "alpha", "zi"),
                        {"log_mu": "identity", "alpha": "log", "zi": "logit"}, zero_inflated_negbinomial_logp,
                        zero_inflated_negbinomial_random, dtype="int64")


def hurdle_negbinomial_family():
    """Bambi family of brms's ``hurdle_negbinomial()``: as the zero-inflated one, with ``hu``."""
    return _zero_family("hurdle_negbinomial_log", "HurdleNegBinomialLog", ("log_mu", "alpha", "hu"),
                        {"log_mu": "identity", "alpha": "log", "hu": "logit"}, hurdle_negbinomial_logp,
                        hurdle_negbinomial_random, dtype="int64")


def zero_inflated_binomial_family():
    """Bambi family of brms's ``zero_inflated_binomial()`` for a two-column ``p(k, n)`` response: the identity link on
    ``logit_p`` (the linear predictor) and ``zi`` with its prior."""
    return _zero_family("zero_inflated_binomial_logit", "ZeroInflatedBinomialLogit", ("logit_p", "zi"),
                        {"logit_p": "identity", "zi": "logit"}, zero_inflated_binomial_logp, zero_inflated_binomial_random,
                        dtype="int64", trials=True)


def zero_inflated_beta_family():
    """Bambi family of brms's ``zero_inflated_beta()``: the identity link on ``logit_mu`` (the linear predictor), the
    precision ``phi`` and ``zi`` with their priors."""
    return _zero_family("zero_inflated_beta_logit", "ZeroInflatedBetaLogit", ("logit_mu", "phi", "zi"),
                        {"logit_mu": "identity", "phi": "log", "zi": "logit"}, zero_inflated_beta_logp, zero_inflated_beta_random)


def hurdle_gamma_family():
    """Bambi family of brms's ``hurdle_gamma()`` (log link): the identity link on ``log_mu`` (the linear predictor), the shape
    (Bambi's ``alpha``) and ``hu`` with their priors."""
    return _zero_family("hurdle_gamma_log", "HurdleGammaLog", ("log_mu", "alpha", "hu"),
                        {"log_mu": "identity", "alpha": "log", "hu": "logit"}, hurdle_gamma_logp, hurdle_gamma_random)


def hurdle_lognormal_family():
    """Bambi family of brms's ``hurdle_lognormal()``: the identity link on the log-scale location ``mu`` (the linear
    predictor), ``sigma`` and ``hu`` with their priors."""
    return _zero_family("hurdle_lognormal_location", "HurdleLogNormal", ("mu", "sigma", "hu"),
                        {"mu": "identity", "sigma": "log", "hu": "logit"}, hurdle_lognormal_logp, hurdle_lognormal_random)


# ---- skew normal, ex-Gaussian and shifted lognormal (brms-parity D01, docs/continuous-families.md) --------------------
# brms 2.23 parameterises the skew normal and the ex-Gaussian by their mean mu (the linear predictor, identity link) and the
# shifted lognormal by the location of log(y - ndt); the densities are the published ones (brms's vignette
# "Parameterization of Response Distributions in brms"; Azzalini 1985 for the skew normal) in those parameters, checked
# against brms's log_prob on the frozen cases skew_normal_varint, exgaussian_varint and shifted_lognormal_varint:
#   skew_normal(mu, sigma, alpha): Azzalini's SN(xi, omega, alpha) with mean mu and standard deviation sigma,
#     delta = alpha / sqrt(1 + alpha^2), omega = sigma / sqrt(1 - 2 delta^2 / pi), xi = mu - omega delta sqrt(2 / pi),
#     log f(y) = log 2 - log omega + log phi(z) + log Phi(alpha z) with z = (y - xi) / omega;
#   exgaussian(mu, sigma, beta): Normal(mu - beta, sigma) plus an independent exponential of mean beta, so E[y] = mu,
#     log f(y) = -log beta + (mu - beta - y) / beta + sigma^2 / (2 beta^2) + log Phi((y - mu + beta) / sigma - sigma / beta);
#   shifted_lognormal(mu, sigma, ndt): log(y - ndt) ~ Normal(mu, sigma) with 0 < ndt < min(y), PyMC's LogNormal density
#     at y - ndt.
# log Phi is log_normal_cdf above, exact with a finite gradient for every finite argument on both paths; brms (Stan) writes
# it log(erfc(-x / sqrt 2)) - log 2, which is -inf once erfc underflows (x below about -37.5) and equal to it elsewhere to
# rounding. PyMC's own SkewNormal (log(1 + erf(.)), which cancels in the lower tail) and ExGaussian (which switches to a
# Normal approximation when nu < 0.05 sigma, through a log-CDF that has no JAX lowering here) are not used. Every constant
# is a float64 (PyTensor would make a Python 2.0 a float32 constant).
_LOG_2_OVER_SQRT_2PI = np.float64(math.log(2.0) - 0.5 * math.log(2.0 * math.pi))
_TWO_OVER_PI = np.float64(2.0 / math.pi)
_SQRT_2_OVER_PI = np.float64(math.sqrt(2.0 / math.pi))
_ONE, _HALF = np.float64(1.0), np.float64(0.5)


def skew_normal_location_scale(mu, sigma, alpha):
    """Azzalini's location ``xi`` and scale ``omega`` of the skew normal with mean ``mu``, SD ``sigma`` and skewness ``alpha``
    (works on PyTensor variables and NumPy arrays alike)."""
    sqrt = np.sqrt if all(isinstance(a, (int, float, np.ndarray, np.generic)) for a in (mu, sigma, alpha)) else pt.sqrt
    delta = alpha / sqrt(_ONE + alpha * alpha)
    omega = sigma / sqrt(_ONE - _TWO_OVER_PI * delta * delta)
    return mu - omega * delta * _SQRT_2_OVER_PI, omega, delta


def skew_normal_logp(value, mu, sigma, alpha):
    """brms's ``skew_normal`` log density at ``value`` (mean ``mu``, SD ``sigma``, skewness ``alpha``)."""
    xi, omega, _ = skew_normal_location_scale(mu, sigma, alpha)
    z = (value - xi) / omega
    return _LOG_2_OVER_SQRT_2PI - pt.log(omega) - _HALF * z * z + log_normal_cdf(alpha * z)


def skew_normal_random(mu, sigma, alpha, rng=None, size=None):
    """Skew-normal draws: ``xi + omega (delta |u0| + sqrt(1 - delta^2) u1)`` with ``u0``, ``u1`` standard normal."""
    xi, omega, delta = skew_normal_location_scale(np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float),
                                                  np.asarray(alpha, dtype=float))
    shape = size if size is not None else np.broadcast_shapes(np.shape(xi), np.shape(omega))
    u0, u1 = np.abs(rng.standard_normal(shape)), rng.standard_normal(shape)
    return xi + omega * (delta * u0 + np.sqrt(1.0 - delta * delta) * u1)


def exgaussian_logp(value, mu, sigma, beta):
    """brms's ``exgaussian`` log density at ``value``: Stan's ``exp_mod_normal_lpdf(value | mu - beta, sigma, 1 / beta)``."""
    r = sigma / beta
    return -pt.log(beta) + (mu - beta - value) / beta + _HALF * r * r + log_normal_cdf((value - mu + beta) / sigma - r)


def exgaussian_random(mu, sigma, beta, rng=None, size=None):
    """Ex-Gaussian draws: ``Normal(mu - beta, sigma)`` plus an exponential of mean ``beta``."""
    shape = size if size is not None else np.broadcast_shapes(np.shape(mu), np.shape(sigma), np.shape(beta))
    return rng.normal(np.asarray(mu) - np.asarray(beta), sigma, size=shape) + rng.exponential(beta, size=shape)


def shifted_lognormal_logp(value, mu, sigma, ndt):
    """brms's ``shifted_lognormal`` log density at ``value``: PyMC's ``LogNormal(mu, sigma)`` at ``value - ndt`` (``-inf``
    where ``value <= ndt``, which the bound ``ndt < min(y)`` excludes for the fitted rows)."""
    return pm.logp(pm.LogNormal.dist(mu, sigma), value - ndt)


def shifted_lognormal_random(mu, sigma, ndt, rng=None, size=None):
    """Shifted lognormal draws: ``ndt + exp(Normal(mu, sigma))``."""
    shape = size if size is not None else np.broadcast_shapes(np.shape(mu), np.shape(sigma), np.shape(ndt))
    return np.asarray(ndt) + rng.lognormal(mu, sigma, size=shape)


def _mean_family(name: str, label: str, params: tuple[str, ...], links: dict, logp, random):
    """A Bambi family whose likelihood is ``logp`` / ``random`` (a ``pm.CustomDist``) of the parent ``mu`` and the
    auxiliary parameters ``params[1:]``, each with its prior from the prior dict."""
    import bambi as bmb

    class Dist:
        def __new__(cls, name_, **kwargs):
            args = [kwargs.pop(p) for p in params]
            return pm.CustomDist(name_, *args, logp=logp, random=random, **kwargs)

        @staticmethod
        def dist(**kwargs):
            args = [kwargs.pop(p) for p in params]
            return pm.CustomDist.dist(*args, logp=logp, random=random, **kwargs)

    Dist.__name__ = label
    likelihood = bmb.Likelihood(label, params=list(params), parent=params[0], dist=Dist)
    return bmb.Family(name, likelihood, link=links)


def skew_normal_family():
    """Bambi family of brms's ``skew_normal()``: the identity link on the mean ``mu``, ``sigma`` and ``alpha`` with their
    priors."""
    return _mean_family("skew_normal", "SkewNormalMean", ("mu", "sigma", "alpha"),
                        {"mu": "identity", "sigma": "log", "alpha": "identity"}, skew_normal_logp, skew_normal_random)


def exgaussian_family():
    """Bambi family of brms's ``exgaussian()``: the identity link on the mean ``mu``, ``sigma`` and ``beta`` with their
    priors."""
    return _mean_family("exgaussian", "ExGaussianMean", ("mu", "sigma", "beta"),
                        {"mu": "identity", "sigma": "log", "beta": "log"}, exgaussian_logp, exgaussian_random)


def shifted_lognormal_family():
    """Bambi family of brms's ``shifted_lognormal()``: the identity link on the log-scale location ``mu``, ``sigma`` and the
    shift ``ndt`` with their priors (``ndt``'s bounded by the model, ``ExtraParameter.bambi_prior``)."""
    return _mean_family("shifted_lognormal", "ShiftedLogNormal", ("mu", "sigma", "ndt"),
                        {"mu": "identity", "sigma": "log", "ndt": "log"}, shifted_lognormal_logp, shifted_lognormal_random)


# ---- inverse Gaussian, von Mises, asymmetric Laplace, Frechet, generalized extreme value (brms-parity E02) ---------------
# docs/continuous-families.md. The densities are the published ones in brms's parameterisations (brms's vignette
# "Parameterization of Response Distributions in brms"), checked against brms's log_prob on the frozen cases
# inverse_gaussian_varint, inverse_gaussian_log, von_mises_varint, asym_laplace_varint, asym_laplace_quantile,
# frechet_varint and gen_extreme_value_varint, not assumed. Each likelihood takes the linear predictor eta as Bambi's
# parent under Bambi's identity link and applies brms's link itself, so the graph's parent deterministic is eta (brms's
# posterior_linpred):
#   inverse Gaussian(mu, shape): 0.5 log(shape / (2 pi y^3)) - shape (y - mu)^2 / (2 mu^2 y), written in 1 / mu as
#     -shape (y / mu - 1)^2 / (2 y); brms's default link 1/mu^2 (1 / mu = sqrt(eta), defined for eta > 0 only: -inf
#     elsewhere, where Stan's inv_sqrt is NaN and brms's density an error) and its log link (1 / mu = exp(-eta));
#   von Mises(mu, kappa): kappa cos(y - mu) - log(2 pi) - log I0(kappa), mu = 2 atan(eta) (brms's tan_half link), with
#     log I0 exact for every kappa (log_bessel_i0: Stan's density raises an overflow above kappa = 713.9);
#   asymmetric Laplace(mu, sigma, p): log p + log(1 - p) - log sigma - rho_p((y - mu) / sigma), rho_p(u) = u (p - I(u < 0));
#   Frechet(mu, nu): log nu - log s - (1 + nu) log(y / s) - (y / s)^-nu with log s = eta - lgamma(1 - 1 / nu), mu = exp(eta);
#   generalized extreme value(mu, sigma, xi): -log sigma - (1 + xi) w - exp(-w), w = log1p(xi z) / xi with z = (y - mu) /
#     sigma (the Gumbel's z at xi = 0), written as z g(xi z) with g(u) = log1p(u) / u by its series near 0, so that the
#     density and its gradient are exact as xi passes 0; brms samples the unbounded tmp_xi and derives xi = lb + (ub - lb)
#     inv_logit(tmp_xi) with lb and ub the smaller and the larger of -1 / max(x) and -1 / min(x), x = (y - mu) / sigma over
#     the fitted rows (found from the fixture's points: brms's posterior_epred(dpar = "xi") equals it to the last bit), so
#     that 1 + xi x > 0 on every fitted row; gen_extreme_value_family registers it as the deterministic "xi".
# Every constant is a float64 (PyTensor would make a Python 2.0 a float32 constant).
_LOG_2PI = np.float64(math.log(2.0 * math.pi))
_HALF_LOG_2PI_64 = np.float64(0.5 * math.log(2.0 * math.pi))
_ONE_AND_HALF = np.float64(1.5)
_I0_SWITCH = np.float64(500.0)                        # log I0 from i0 below, from the asymptotic series at and above
_ZERO_64 = np.float64(0.0)


def _i0_series(terms: int) -> tuple[np.float64, ...]:
    """The coefficients ``a_j = ((2j - 1)!!)^2 / (j! 8^j)`` of the asymptotic series ``I0(k) ~ exp(k) / sqrt(2 pi k) (1 +
    sum_j a_j / k^j)`` (Abramowitz and Stegun 9.7.1), ``a_j = a_{j-1} (2j - 1)^2 / (8 j)``."""
    out, a = [], 1.0
    for j in range(1, terms + 1):
        a *= (2 * j - 1) ** 2 / (8.0 * j)
        out.append(np.float64(a))
    return tuple(out)


_I0_COEFFICIENTS = _i0_series(8)                      # at k = 500 the ninth term is below 1e-23 of the sum


def log_bessel_i0e(kappa):
    """``log I0(kappa) - kappa`` for ``kappa >= 0``, the log of the exponentially scaled Bessel function: ``log(i0(kappa)) -
    kappa`` below 500 (PyTensor's ``i0``, whose JAX lowering is ``jax.scipy.special.i0``; ``i0`` itself overflows past 709.8)
    and ``-log(2 pi kappa) / 2 + log1p(sum_j a_j / kappa^j)`` at and above 500, exact there to rounding. Each branch is evaluated
    on its argument switched into its own range (not clipped by minimum and maximum, whose JAX gradient splits in two at the
    tie), so that the branch a switch does not take stays finite with a zero gradient."""
    below = pt.lt(kappa, _I0_SWITCH)
    small = pt.switch(below, kappa, _I0_SWITCH)
    large = pt.switch(below, _I0_SWITCH, kappa)
    inv = _ONE / large
    tail = pt.zeros_like(large)
    for a in reversed(_I0_COEFFICIENTS):                # Horner's scheme in 1 / kappa
        tail = (tail + a) * inv
    asymptotic = -_HALF * pt.log(large) - _HALF_LOG_2PI_64 + pt.log1p(tail)
    return pt.switch(below, pt.log(pt.i0(small)) - small, asymptotic)


def log_bessel_i0(kappa):
    """``log I0(kappa)`` for ``kappa >= 0``: ``kappa + log_bessel_i0e(kappa)``, exact for every kappa."""
    return kappa + log_bessel_i0e(kappa)


def _inverse_gaussian_logp(value, inv_mu, shape):
    """The inverse Gaussian log density at ``value`` with mean ``1 / inv_mu`` and shape ``shape``."""
    d = value * inv_mu - _ONE
    return (_HALF * pt.log(shape) - _HALF_LOG_2PI_64 - _ONE_AND_HALF * pt.log(value)
            - shape * d * d / (np.float64(2.0) * value))


def inverse_gaussian_inverse_squared_logp(value, eta, shape):
    """brms's ``inverse.gaussian()`` under its default link ``1/mu^2``: ``1 / mu = sqrt(eta)``; ``-inf`` where ``eta <= 0``
    (no mean; brms's density is an error there), with the branch not taken evaluated at ``eta = 1``."""
    positive = pt.gt(eta, 0.0)
    return pt.switch(positive, _inverse_gaussian_logp(value, pt.sqrt(pt.switch(positive, eta, _ONE)), shape), -np.inf)


def inverse_gaussian_log_logp(value, eta, shape):
    """brms's ``inverse.gaussian("log")``: ``1 / mu = exp(-eta)``."""
    return _inverse_gaussian_logp(value, pt.exp(-eta), shape)


def inverse_gaussian_mean(eta, link: str):
    """The inverse Gaussian mean ``mu`` from the linear predictor (NumPy): ``1 / sqrt(eta)`` under ``1/mu^2``, refused where
    ``eta <= 0`` (no mean exists; brms's ``posterior_epred`` would be NaN there), ``exp(eta)`` under ``log``."""
    eta = np.asarray(eta, dtype=float)
    if link == "log":
        with np.errstate(over="ignore"):
            return np.exp(eta)
    if not (eta > 0).all():
        raise ValueError("the inverse.gaussian family's 1/mu^2 link needs a positive linear predictor: some draws and rows have "
                         "eta <= 0, where mu = 1 / sqrt(eta) does not exist; inspect the draws (or use link='log')")
    return 1.0 / np.sqrt(eta)


def inverse_gaussian_random(mu, shape, rng=None, size=None):
    """Inverse Gaussian draws of mean ``mu`` and shape ``shape`` (NumPy's ``wald``)."""
    return rng.wald(np.asarray(mu, dtype=float), np.asarray(shape, dtype=float), size=size)


def von_mises_logp(value, eta, kappa):
    """brms's ``von_mises()``: the von Mises log density at ``value`` with ``mu = 2 atan(eta)`` (the tan_half link),
    ``kappa cos(y - mu) - log(2 pi) - log I0(kappa)`` written as ``-2 kappa sin^2((y - mu) / 2) - log(2 pi) - (log I0(kappa) -
    kappa)``, which forms no difference of two numbers of the size of kappa (at kappa = 1e6 the first form loses 1e-10)."""
    mu = np.float64(2.0) * pt.arctan(eta)
    half = pt.sin(_HALF * (value - mu))
    return np.float64(-2.0) * kappa * half * half - _LOG_2PI - log_bessel_i0e(kappa)


def von_mises_random(mu, kappa, rng=None, size=None):
    """von Mises draws of mean direction ``mu`` and precision ``kappa`` on [-pi, pi] (NumPy's ``vonmises``)."""
    return rng.vonmises(np.asarray(mu, dtype=float), np.asarray(kappa, dtype=float), size=size)


def asym_laplace_logp(value, mu, sigma, quantile):
    """brms's ``asym_laplace()``: ``log p + log(1 - p) - log sigma - rho_p((value - mu) / sigma)``."""
    z = (value - mu) / sigma
    rho = pt.switch(pt.lt(z, 0.0), z * (quantile - _ONE), z * quantile)
    return pt.log(quantile) + pt.log1p(-quantile) - pt.log(sigma) - rho


def asym_laplace_random(mu, sigma, quantile, rng=None, size=None):
    """Asymmetric Laplace draws: below ``mu`` with probability ``p``, an exponential of rate ``(1 - p) / sigma`` under it;
    above it an exponential of rate ``p / sigma`` (the density's two halves)."""
    mu, sigma, p = (np.asarray(a, dtype=float) for a in (mu, sigma, quantile))
    shape = size if size is not None else np.broadcast_shapes(mu.shape, sigma.shape, p.shape)
    e, below = rng.exponential(1.0, size=shape), rng.uniform(size=shape) < p
    return mu + sigma * np.where(below, -e / (1.0 - p), e / p)


def frechet_logp(value, eta, nu):
    """brms's ``frechet()``: the Frechet log density with scale ``s = mu / Gamma(1 - 1 / nu)``, ``mu = exp(eta)``."""
    log_s = eta - pt.gammaln(_ONE - _ONE / nu)
    r = pt.log(value) - log_s
    return pt.log(nu) - log_s - (_ONE + nu) * r - pt.exp(-nu * r)


def frechet_random(eta, nu, rng=None, size=None):
    """Frechet draws ``s E^(-1 / nu)`` with ``E`` a standard exponential and ``s = exp(eta) / Gamma(1 - 1 / nu)``."""
    from scipy.special import gammaln

    eta, nu = np.asarray(eta, dtype=float), np.asarray(nu, dtype=float)
    shape = size if size is not None else np.broadcast_shapes(eta.shape, nu.shape)
    return np.exp(eta - gammaln(1.0 - 1.0 / nu) - np.log(rng.exponential(1.0, size=shape)) / nu)


_G_SERIES = 1e-3                                      # g(u) = log1p(u) / u by its series below |u| = 1e-3 (error < 2e-22)


def _log1p_over(u):
    """``g(u) = log1p(u) / u`` with ``g(0) = 1``: the series ``1 - u/2 + u^2/3 - ... + u^6/7`` for ``|u| < 1e-3`` and
    ``log1p(u) / u`` elsewhere (``u > -1``), each branch on an argument it is defined at."""
    near = pt.lt(pt.abs(u), _G_SERIES)
    s = pt.clip(u, -_G_SERIES, _G_SERIES)
    series = pt.zeros_like(s)
    for k in range(7, 0, -1):                          # 1/1 - s/2 + s^2/3 - ...: Horner's scheme
        series = np.float64(1.0 / k) - s * series
    far = pt.switch(near, np.float64(0.5), u)
    return pt.switch(near, series, pt.log1p(far) / far)


def gen_extreme_value_logp(value, mu, sigma, xi):
    """brms's ``gen_extreme_value()``: ``-log sigma - (1 + xi) w - exp(-w)`` with ``w = log1p(xi z) / xi``, ``z = (value -
    mu) / sigma``; ``-inf`` where ``1 + xi z <= 0`` (outside the support, which brms's xi excludes on the fitted rows)."""
    z = (value - mu) / sigma
    u = xi * z
    inside = pt.gt(u, -_ONE)
    w = z * _log1p_over(pt.switch(inside, u, _ZERO_64))
    return pt.switch(inside, -pt.log(sigma) - (_ONE + xi) * w - pt.exp(-w), -np.inf)


def gen_extreme_value_xi(tmp_xi, y, mu, sigma):
    """brms's ``xi`` from its sampled ``tmp_xi`` (PyTensor): ``lb + (ub - lb) inv_logit(tmp_xi)`` with ``lb`` and ``ub`` the
    smaller and the larger of ``-1 / max(x)`` and ``-1 / min(x)``, ``x = (y - mu) / sigma`` over the fitted rows."""
    x = (y - mu) / sigma
    a, b = -_ONE / pt.max(x), -_ONE / pt.min(x)
    lb, ub = pt.minimum(a, b), pt.maximum(a, b)
    return lb + (ub - lb) * pt.sigmoid(tmp_xi)


def gen_extreme_value_random(mu, sigma, xi, rng=None, size=None):
    """GEV draws ``mu + sigma (E^-xi - 1) / xi`` with ``E`` a standard exponential (``-log E``, the Gumbel's, at ``xi = 0``),
    written as ``-log(E) expm1(a) / a`` with ``a = -xi log E``."""
    mu, sigma, xi = (np.asarray(v, dtype=float) for v in (mu, sigma, xi))
    shape = size if size is not None else np.broadcast_shapes(mu.shape, sigma.shape, xi.shape)
    log_e = np.log(rng.exponential(1.0, size=shape))
    a = -xi * log_e
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(a == 0.0, 1.0, np.expm1(a) / np.where(a == 0.0, 1.0, a))
    return mu - sigma * log_e * ratio


GEV_XI = "xi"                                          # the graph's deterministic of brms's derived xi


def inverse_gaussian_family(link: str):
    """Bambi family of brms's ``inverse.gaussian()`` under ``link`` (``"1/mu^2"``, the default, or ``"log"``): the linear
    predictor as the parent (``inverse_squared_mu`` or ``log_mu``) under Bambi's identity link, and the shape (Bambi's
    ``alpha``, as for the Gamma family) with its prior."""
    parent, logp = (("log_mu", inverse_gaussian_log_logp) if link == "log" else
                    ("inverse_squared_mu", inverse_gaussian_inverse_squared_logp))

    def random(eta, shape, rng=None, size=None):
        eta = np.asarray(eta, dtype=float)
        mu = np.exp(eta) if link == "log" else 1.0 / np.sqrt(np.where(eta > 0, eta, np.nan))
        return inverse_gaussian_random(mu, shape, rng=rng, size=size)

    return _mean_family("inverse_gaussian", "InverseGaussian", (parent, "alpha"), {parent: "identity", "alpha": "log"}, logp, random)


def von_mises_family():
    """Bambi family of brms's ``von_mises()``: the linear predictor ``tan_half_mu`` (``tan(mu / 2)``) under Bambi's identity
    link, and ``kappa`` with its prior."""
    def random(eta, kappa, rng=None, size=None):
        return von_mises_random(2.0 * np.arctan(np.asarray(eta, dtype=float)), kappa, rng=rng, size=size)

    return _mean_family("von_mises", "VonMisesTanHalf", ("tan_half_mu", "kappa"), {"tan_half_mu": "identity", "kappa": "log"},
                        von_mises_logp, random)


def asym_laplace_family():
    """Bambi family of brms's ``asym_laplace()``: the identity link on the location ``mu``, ``sigma`` and the ``quantile``
    (its prior, or a float64 constant when the model fixes it, brms's ``bf(quantile = p)``)."""
    return _mean_family("asym_laplace", "AsymmetricLaplace", ("mu", "sigma", "quantile"),
                        {"mu": "identity", "sigma": "log", "quantile": "logit"}, asym_laplace_logp, asym_laplace_random)


def frechet_family():
    """Bambi family of brms's ``frechet()``: the linear predictor ``log_mu`` under Bambi's identity link, and ``nu`` (learned
    under its prior truncated at 1, as the Student-t nu, or a float64 constant when fixed)."""
    return _mean_family("frechet", "Frechet", ("log_mu", "nu"), {"log_mu": "identity", "nu": "log"}, frechet_logp, frechet_random)


def gen_extreme_value_family():
    """Bambi family of brms's ``gen_extreme_value()``: the identity link on the location ``mu``, ``sigma`` and brms's unscaled
    shape ``tmp_xi`` with its prior; building the likelihood registers ``xi`` (``gen_extreme_value_xi`` of the observed
    response) as the deterministic ``GEV_XI``, which the density takes."""
    import bambi as bmb

    params = ("mu", "sigma", "tmp_xi")

    class GenExtremeValue:
        def __new__(cls, name_, **kwargs):
            mu, sigma, tmp_xi = (kwargs.pop(p) for p in params)
            y = pt.as_tensor_variable(np.asarray(kwargs["observed"], dtype=np.float64))
            xi = pm.Deterministic(GEV_XI, gen_extreme_value_xi(tmp_xi, y, mu, sigma))
            return pm.CustomDist(name_, mu, sigma, xi, logp=gen_extreme_value_logp, random=gen_extreme_value_random, **kwargs)

        @staticmethod
        def dist(**kwargs):
            raise NotImplementedError("the generalized extreme value family's xi is derived from the fitted rows; its "
                                      "predictions are assembled by predict() (manual_prediction)")

    likelihood = bmb.Likelihood("GenExtremeValue", params=list(params), parent="mu", dist=GenExtremeValue)
    return bmb.Family("gen_extreme_value", likelihood, link={"mu": "identity", "sigma": "log", "tmp_xi": "identity"})


# ---- beta-binomial, geometric and negbinomial2 (brms-parity D02, docs/discrete-families.md) -----------------------------
# The densities are the published ones in brms's parameterisations (brms's vignette "Parameterization of Response
# Distributions in brms"), checked against brms's log_prob on the frozen cases beta_binomial_varint, geometric_varint and
# negbinomial2_varint, not assumed:
#   beta_binomial(mu, phi) with trials n: BetaBinomial(n, mu phi, (1 - mu) phi), mu = inv_logit(eta), E[y] = n mu; PyMC's
#     BetaBinomial, with 1 - mu as expit(-eta) as for the Beta family (D13). PyMC's log mass, log C(n, k) + log B(k + a,
#     n - k + b) - log B(a, b) through gammaln, is within 2e-9 absolute (3e-10 relative) of a 50-digit reference at n = 1e6,
#     the largest count accepted (formula.COUNT_CAP), and within 1e-15 at the fixture's counts; its log-gamma differences lose
#     about eps * phi * log(phi) as the precision grows (1.5e-9 at phi = 1e6, 1.4e-7 at 1e8; divergence D82);
#   geometric(mu): the negative binomial of mean mu and shape 1 (Var[y] = mu + mu^2), brms's neg_binomial_2(mu, 1);
#   negbinomial2(mu, sigma): the negative binomial of mean mu and shape 1 / sigma (Var[y] = mu + sigma mu^2), brms's
#     neg_binomial_2(mu, 1 / sigma);
# the last two through PyMC's NegativeBinomial(mu, alpha), as the negbinomial family (Bambi's) does. Constants are float64.
class BetaBinomialLogit:
    """``k ~ BetaBinomial(n, mu phi, (1 - mu) phi)`` with ``mu = expit(logit_mu)``."""

    def __new__(cls, name, logit_mu, phi, n, **kwargs):
        alpha, beta = _beta_shape_parameters(logit_mu, phi)
        return pm.BetaBinomial(name, n=n, alpha=alpha, beta=beta, **kwargs)

    @staticmethod
    def dist(logit_mu, phi, n, **kwargs):
        alpha, beta = _beta_shape_parameters(logit_mu, phi)
        return pm.BetaBinomial.dist(n=n, alpha=alpha, beta=beta, **kwargs)


def beta_binomial_logit_family():
    """Bambi family of brms's ``beta_binomial()`` for a two-column ``p(k, n)`` response: the identity link on ``logit_mu``
    (the linear predictor), the precision ``phi`` with its own prior; ``n`` reaches the likelihood as the binomial's does."""
    import bambi as bmb

    class BetaBinomialLogitFamily(bmb.Family):
        @staticmethod
        def transform_backend_kwargs(kwargs):
            observed = kwargs.pop("observed")
            kwargs["observed"] = np.asarray(observed[:, 0]).squeeze().astype(np.int64)
            kwargs["n"] = np.asarray(observed[:, 1]).squeeze().astype(np.int64)
            return kwargs

    likelihood = bmb.Likelihood("BetaBinomialLogit", params=["logit_mu", "phi"], parent="logit_mu", dist=BetaBinomialLogit)
    return BetaBinomialLogitFamily("beta_binomial_logit", likelihood, link={"logit_mu": "identity", "phi": "log"})


_UNIT_SHAPE = np.float64(1.0)


class Geometric:
    """``y ~ NegativeBinomial(mu, 1)``: brms's ``geometric()``, PyMC's ``NegativeBinomial`` with ``alpha = 1``."""

    def __new__(cls, name, mu, **kwargs):
        return pm.NegativeBinomial(name, mu=mu, alpha=_UNIT_SHAPE, **kwargs)

    @staticmethod
    def dist(mu, **kwargs):
        return pm.NegativeBinomial.dist(mu=mu, alpha=_UNIT_SHAPE, **kwargs)


def geometric_family():
    """Bambi family of brms's ``geometric()``: the log link on the mean ``mu``, no auxiliary parameter."""
    import bambi as bmb

    likelihood = bmb.Likelihood("Geometric", params=["mu"], parent="mu", dist=Geometric)
    return bmb.Family("geometric", likelihood, link={"mu": "log"})


class NegBinomial2:
    """``y ~ NegativeBinomial(mu, 1 / sigma)``: brms's ``negbinomial2()``, PyMC's ``NegativeBinomial`` with
    ``alpha = 1 / sigma``."""

    def __new__(cls, name, mu, sigma, **kwargs):
        return pm.NegativeBinomial(name, mu=mu, alpha=_UNIT_SHAPE / sigma, **kwargs)

    @staticmethod
    def dist(mu, sigma, **kwargs):
        return pm.NegativeBinomial.dist(mu=mu, alpha=_UNIT_SHAPE / sigma, **kwargs)


def negbinomial2_family():
    """Bambi family of brms's ``negbinomial2()``: the log link on the mean ``mu``, the dispersion ``sigma`` (brms's
    ``1 / shape``) with its own prior."""
    import bambi as bmb

    likelihood = bmb.Likelihood("NegBinomial2", params=["mu", "sigma"], parent="mu", dist=NegBinomial2)
    return bmb.Family("negbinomial2", likelihood, link={"mu": "log", "sigma": "log"})


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
    ub, rng)`` brms's posterior_epred and posterior_predict of a truncated model, from posterior draws (NumPy).

    E04 (docs/cens-trunc-families.md; ``autobayes.tails``): a family that takes only one of the two terms leaves the other's
    fields ``None``. ``log_truncation(lower, upper, **params)``, where set, is the log mass of the allowed range (``None`` for
    an open side) in place of the continuous reading ``log P(lower < Y <= upper)`` from the three functions above (a count
    family's inclusive ``log P(lower <= Y <= upper)``); ``neutral(**params)`` a pair of response values ``v < w`` well inside
    the support, substituted on the rows a censoring branch does not take (the Gaussian's tails need none); and
    ``numpy_tails(model, post, eta)`` the NumPy forms ``log_cdf(v)``, ``log_ccdf(v)``, ``log_mass(v, w)`` and
    ``log_truncation(lower, upper)`` at each draw and row, from which ``censored_truncated_log_lik`` writes brms's pointwise
    log likelihood."""

    log_cdf: Callable | None = None
    log_ccdf: Callable | None = None
    log_mass: Callable | None = None
    truncated: Callable | None = None
    truncated_mean: Callable | None = None
    draw_truncated: Callable | None = None
    log_truncation: Callable | None = None
    neutral: Callable | None = None
    numpy_tails: Callable | None = None


def _normal_numpy_tails(model, post, eta):
    """The Gaussian's NumPy tails (E04: ``log_lik()`` of a censored or truncated model)."""
    from types import SimpleNamespace

    from scipy.special import log_ndtr

    from autobayes.families.spec import auxiliary
    from autobayes.tails import np_normal_log_mass

    sigma = auxiliary(model, post, "sigma")
    return SimpleNamespace(log_cdf=lambda v: log_ndtr((v - eta) / sigma), log_ccdf=lambda v: log_ndtr((eta - v) / sigma),
                           log_mass=lambda lo, hi: np_normal_log_mass((lo - eta) / sigma, (hi - eta) / sigma))


NORMAL_TAILS = Tails(
    log_cdf=normal_log_cdf, log_ccdf=normal_log_ccdf, log_mass=normal_log_mass,
    truncated=lambda lower, upper, *, size=None, mu, sigma: pm.TruncatedNormal.dist(mu=mu, sigma=sigma, lower=lower, upper=upper, size=size),
    truncated_mean=_normal_truncated_mean, draw_truncated=_normal_draw_truncated,
    numpy_tails=_normal_numpy_tails,
)


def censored_truncated_log_lik(model, post, eta, ll):
    """brms's pointwise log likelihood of a model with ``cens()`` or ``trunc()`` (E04), from the family's density per draw and
    row ``ll`` (``(chain, draw, row)``, NumPy): a censored row's log CDF (-1), log CCDF (1) or interval log mass (2) in place
    of its density, and every row of a truncated model minus the log mass of the allowed range; ``ll`` itself otherwise."""
    from autobayes.families import family_of

    d = model.design
    if d.cens is None and d.lb is None:
        return ll
    np_tails = family_of(model).tails().numpy_tails(model, post, eta)
    y = d.y
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        if d.cens is not None:
            out = np.where(d.cens == 1, np_tails.log_ccdf(y), np.where(d.cens == -1, np_tails.log_cdf(y), ll))
            if d.rcens is not None:                            # the interval's mass, (y, y2]; any valid pair on the other rows
                upper = np.where(d.cens == 2, d.rcens, y + np.abs(y) + 1.0)
                out = np.where(d.cens == 2, np_tails.log_mass(y, upper), out)
            ll = out
        if d.lb is not None:
            lower = None if np.isneginf(d.lb[0]) else float(d.lb[0])
            upper = None if np.isposinf(d.ub[0]) else float(d.ub[0])
            if getattr(np_tails, "log_truncation", None) is not None:
                mass = np_tails.log_truncation(lower, upper)
            elif lower is not None and upper is not None:
                mass = np_tails.log_mass(np.full(y.shape, lower), np.full(y.shape, upper))
            elif lower is not None:
                mass = np_tails.log_ccdf(np.full(y.shape, lower))
            else:
                mass = np_tails.log_cdf(np.full(y.shape, upper))
            ll = ll - mass
    return ll


def _addition_dist(base_dist, data: Mapping[str, np.ndarray], tails: Tails | None, family_spec=None):
    """The likelihood class Bambi calls with the family's parameters by name: ``base_dist`` (the family's PyMC
    distribution, or its hand-written class) with the addition terms of ``data`` (``formula.addition_data`` on the fitted
    rows) applied to its log density. F06a: ``se()`` and ``rate()`` give the distribution its parameters per row
    (``se_rate.likelihood_parameters``, by ``family_spec``'s Bambi names) before anything else reads them, its draws included."""
    from autobayes.se_rate import likelihood_parameters

    weights, cens, rcens = data.get("weights"), data.get("cens"), data.get("rcens")
    lower = None if "lb" not in data or np.isneginf(data["lb"][0]) else float(data["lb"][0])   # numeric bounds, one per model
    upper = None if "ub" not in data or np.isposinf(data["ub"][0]) else float(data["ub"][0])
    truncated = lower is not None or upper is not None
    if (cens is not None or truncated) and tails is None:
        raise ValueError("cens() and trunc() need the family's tails (FamilySpec.tails)")

    def forms(keys):
        def draw(*args):
            *params, size = args
            kw = likelihood_parameters(dict(zip(keys, params)), data, family_spec)
            return tails.truncated(lower, upper, size=size, **kw) if truncated else base_dist.dist(**kw, size=size)

        def logp(value, *params):
            kw = likelihood_parameters(dict(zip(keys, params)), data, family_spec)
            ll = pm.logp(base_dist.dist(**kw), value)
            if cens is not None:                               # brms: lpdf if 0, lccdf if 1, lcdf if -1
                right, left = value, value
                if tails.neutral is not None:                  # E04: each branch at its own rows, a neutral value elsewhere
                    v, _ = tails.neutral(**kw)
                    right, left = pt.switch(pt.eq(cens, 1), value, v), pt.switch(pt.eq(cens, -1), value, v)
                if rcens is not None:                          # E04, cens(c, y2): the log mass of (y, y2] if 2
                    interval = pt.eq(cens, 2)
                    if tails.neutral is not None:
                        v, w = tails.neutral(**kw)
                        lo, hi = pt.switch(interval, value, v), pt.switch(interval, rcens, w)
                    else:
                        lo, hi = value, pt.switch(interval, rcens, value + pt.abs(value) + 1.0)
                    ll = pt.switch(interval, tails.log_mass(lo, hi, **kw), ll)
                ll = pt.switch(pt.eq(cens, 1), tails.log_ccdf(right, **kw), pt.switch(pt.eq(cens, -1), tails.log_cdf(left, **kw), ll))
            if truncated:                                      # minus the log mass between the bounds, every row
                if tails.log_truncation is not None:           # E04: a count family's inclusive log P(lb <= Y <= ub)
                    ll = ll - tails.log_truncation(lower, upper, **kw)
                else:
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
                                       dist=_addition_dist(get_distribution_from_likelihood(lk), data, tails, family_spec))
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


_ORDERED_FAMILY_CLASSES: dict[type, type] = {}


def _ordered_auxiliary_parameters(family):
    """The family's parameters other than the parent, in the order its likelihood lists them: a set-like ``dict`` keys
    view (``in``, ``-``, ``&`` work as on Bambi's set), whose iteration order is fixed."""
    lk = family.likelihood
    return {p: None for p in lk.params if p != lk.parent}.keys()


def ordered_family(base):
    """The Bambi family ``base`` (a built-in family's name or a family object) whose auxiliary parameters come in the order
    its likelihood lists them (F09). Bambi's ``Family.auxiliary_parameters`` is a ``set`` of the parameters' names, and
    ``bmb.Model`` creates their PyMC variables in its iteration order, which for strings depends on ``PYTHONHASHSEED``: a
    model with two or more auxiliary parameters (a Student-t model's sigma and nu, a hurdle model's shape and hu, the hurdle
    cumulative family's thresholds and hu) had its free variables, the posterior's variables and the layout of the
    sampler's unconstrained vector in another order in each process, and ``describe()`` listed them so. Everything else
    about the family is the base family's (its class is a subclass of the base family's class)."""
    import copy

    from bambi.defaults import get_builtin_family

    family = get_builtin_family(base) if isinstance(base, str) else copy.copy(base)
    cls = type(family)
    ordered = _ORDERED_FAMILY_CLASSES.get(cls)
    if ordered is None:
        ordered = type(cls.__name__, (cls,), {"auxiliary_parameters": property(_ordered_auxiliary_parameters),
                                              "__module__": cls.__module__, "__qualname__": cls.__qualname__})
        _ORDERED_FAMILY_CLASSES[cls] = ordered
    family.__class__ = ordered
    return family


def missing_response_family(base, index, values_name: str, coordinate: str):
    """The Bambi family ``base`` whose response has missing values that are parameters of the model (C06, brms's
    ``y | mi() ~ ...``, docs/missing-and-measurement-error.md): Bambi is handed the response with a placeholder at the missing
    rows ``index`` (0-based), and its likelihood is built here as a ``CustomDist`` that evaluates the family's density at the
    completed response, brms's ``Yl = Y`` with ``Yl[Jmi] = Ymi``: the free variable ``values_name`` (brms's ``Ymi``) on the
    coordinate ``coordinate`` (the missing rows), under brms's flat prior (PyMC's ``Flat``), takes the place of the
    placeholders, which the density never reads. Draws are the family's own on every row."""
    import copy

    import bambi as bmb
    from bambi.backend.utils import get_distribution_from_likelihood
    from bambi.defaults import get_builtin_family

    family = get_builtin_family(base) if isinstance(base, str) else copy.copy(base)
    lk = family.likelihood
    base_dist = get_distribution_from_likelihood(lk)
    rows = np.asarray(index, dtype=np.int64)

    class MissingDist:
        """``y ~ family(...)`` over the observed values and the missing ones, ``values_name``, together."""

        def __new__(cls, name, *, observed=None, dims=None, **params):
            model = pm.modelcontext(None)
            model.add_coord(coordinate, [int(i) for i in rows])
            missing = pm.Flat(values_name, dims=coordinate)
            keys = tuple(params)

            def logp(value, *args):
                *ps, ymi = args
                return pm.logp(base_dist.dist(**dict(zip(keys, ps))), pt.set_subtensor(value[rows], ymi))

            def draw(*args):
                *ps, _, size = args
                return base_dist.dist(**dict(zip(keys, ps)), size=size)

            return pm.CustomDist(name, *(params[k] for k in keys), missing, dist=draw, logp=logp, observed=observed, dims=dims)

        @staticmethod
        def dist(**params):
            return base_dist.dist(**params)

    family.likelihood = bmb.Likelihood(lk.name, params=lk.params, parent=lk.parent, dist=MissingDist)
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


_ORDERED_REJECTION_ROUNDS = 1000          # forward draws of ordered entries under different priors: rounds of rejection (C05)


def ordered_thresholds(name, prior_name: str, params: dict, labels, dim: str):
    """brms's ``ordered[nthres] Intercept`` with the same prior on every threshold (``Priors(intercept=...)``, brms's
    class ``Intercept``), as a PyMC variable named ``name`` with dims ``dim`` (coordinates ``labels``) sampled on Stan's
    ``ordered`` coordinates (PyMC's ``ordered`` transform: ``(tau_1, log(tau_2 - tau_1), ...)``, log Jacobian the sum of
    the log gaps). Its density is the prior's density summed over the thresholds as brms adds it (``lprior +=
    normal_lpdf(Intercept | ...)``), not renormalised to the ordered region; its forward draws are the prior's independent
    draws sorted, which is that density's distribution (the order statistics), so prior-predictive thresholds are
    ordered. Must be called inside a PyMC model context.

    A parameter may also be given one value per entry (C05, a mixture's ``ordered_Intercept``: brms's ``Intercept_mu<k>``
    each under its own prior of one distribution family, ``normal(-1, 2)`` and ``normal(1, 2)``): the density is then each
    entry's own prior summed, and the forward draws are the entries' independent draws kept where they come out
    strictly increasing (rejection, the product density restricted to the ordered region; sorting would be another
    distribution once the priors differ), refused after ``_ORDERED_REJECTION_ROUNDS`` rounds."""
    n = len(labels)
    names = list(params)
    # The first input only carries the number of thresholds (a flat prior has no parameter to carry it); then the prior's
    # parameters, one entry per threshold, in float64 (a Python 3.0 would otherwise become a float32 constant).
    values = [np.zeros(n)] + [np.broadcast_to(np.asarray(params[k], dtype=np.float64), (n,)).copy() for k in names]
    base = getattr(pm, prior_name)
    same = all(np.all(v == v[0]) for v in values[1:])        # one prior for every entry: the sorted draws are exact

    def logp(value, anchor, *ps):
        return pt.sum(pm.logp(base.dist(**dict(zip(names, ps))), value), axis=-1)

    def random(anchor, *ps, rng=None, size=None):
        if prior_name not in _LOCATION_DRAWS:
            raise NotImplementedError(f"a {prior_name} prior on the thresholds has no forward draws")
        core = np.broadcast_shapes(np.shape(anchor), *(np.shape(p) for p in ps))
        shape = (tuple(size) + core[-1:]) if size is not None else core
        if same:
            return np.sort(_LOCATION_DRAWS[prior_name](rng, shape, *ps), axis=-1)
        out = _LOCATION_DRAWS[prior_name](rng, shape, *ps)
        for _ in range(_ORDERED_REJECTION_ROUNDS):
            bad = ~(np.diff(out, axis=-1) > 0).all(axis=-1)
            if not bad.any():
                return out
            out = np.where(bad[..., None], _LOCATION_DRAWS[prior_name](rng, shape, *ps), out)
        raise ValueError(f"forward draws of {name}: after {_ORDERED_REJECTION_ROUNDS} rounds the entries' priors still give "
                         "unordered draws; they put almost no mass on the ordered region (inspect the priors)")

    mu = params.get("mu", 0.0)                                                   # one location, or one per entry (C05)
    start = (float(mu) if np.ndim(mu) == 0 else float(np.mean(mu))) + np.arange(n) - (n - 1) / 2.0   # ordered, at the location

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


# ---- more ordinal models (brms-parity D04, docs/ordinal-categorical.md) ------------------------------------------------------
# brms 2.23, with the 1-based category y, the K - 1 thresholds tau (brms's b_Intercept on the raw design), the linear
# predictor eta (brms's mu) and disc = 1, each family written through the gaps d_k = tau_k - eta_k (eta_k = eta, or with
# category-specific effects cs(x) the threshold's own eta + Xcs bcs[, k]; brms subtracts Xcs bcs from the thresholds, which
# is the same), from the published definitions (Agresti, Categorical Data Analysis, ch. 8; Buerkner and Vuorre 2019):
#   cumulative("probit"): P(y <= k) = Phi(d_k), so P(y = k) = Phi(d_k) - Phi(d_{k-1}) (d_0 = -inf, d_K = inf); the first row
#     is log Phi(d_1), the last log Phi(-d_{K-1}) and a middle one the log Normal mass between d_{k-1} and d_k, each from
#     log_normal_cdf / normal_log_mass above, never a difference of probabilities;
#   sratio("logit"), the stopping ratio: P(y = k | y >= k) = expit(d_k), so log P(y = k) = log expit(d_k) + sum_{j<k}
#     log(1 - expit(d_j)) = -softplus(-d_k) [k < K] - sum_{j<k} softplus(d_j);
#   cratio("logit"), the continuation ratio: P(y > k | y >= k) = expit(eta_k - tau_k) = expit(-d_k), so log P(y = k) =
#     log(1 - expit(-d_k)) [k < K] + sum_{j<k} log expit(-d_j): the same row as sratio's, the logistic being symmetric;
#   acat("logit"), adjacent categories: log(P(y = k + 1) / P(y = k)) = eta_k - tau_k = -d_k, so log P(y = k) = s_k -
#     log_sum_exp(s_1, ..., s_K) with s_1 = 0, s_k = -sum_{j<k} d_j, the largest s factored out of the sum.
#   cumulative("cloglog") (E06): P(y <= k) = F(d_k) with F(x) = 1 - exp(-exp(x)), so the first row is log F(d_1) =
#     cloglog_log_p1(d_1), the last log(1 - F(d_{K-1})) = -exp(d_{K-1}) = cloglog_log_p0(d_{K-1}) and a middle one the log of
#     F(d_y) - F(d_{y-1}) = exp(-exp(lo)) (1 - exp(-t)), t = exp(hi) - exp(lo), written cloglog_log_mass(lo, hi) below, never a
#     difference of probabilities (brms subtracts two inv_cloglog values, which loses the mass where both are near 1; D132).
# Each is a sum of stable non-positive terms, finite with a finite gradient for every finite eta and tau on the PyTensor path
# and under JAX's autodiff; a row's thresholds are selected by the one-hot switch of _select, as for the cumulative logit row.
ORDINAL_KINDS = ("cumulative_probit", "sratio_logit", "cratio_logit", "acat_logit", "cumulative_cloglog")
_ZERO, _ONE = np.float64(0.0), np.float64(1.0)
_CLOGLOG_MASS_LOWER = -20.0


def cloglog_log_mass(lower, upper):
    """``log(F(upper) - F(lower))`` with ``F(x) = 1 - exp(-exp(x))`` (the cumulative cloglog model's probability of a middle
    category, E06) for ``lower < upper``, finite with a finite gradient for every finite pair. From ``upper = -20`` to 700:
    ``-exp(lower) + log(1 - exp(-t))`` with ``t = exp(upper) - exp(lower)`` formed as ``exp(upper) (-expm1(lower - upper))``
    (never a difference of two exponentials) and ``log(1 - exp(-t))`` as ``log1mexp(-t)``; above ``upper = 700``, where ``exp``
    overflows, ``t`` exceeds 745 for every ``lower < upper`` that double precision can hold, so that term is exactly 0 and the
    row is ``-exp(lower)``, with ``lower`` taken at 700 at most (the finite stand-in ``cloglog_log_p0`` uses). Below ``upper =
    -20``: ``upper + log(1 - exp(lower - upper)) + log1p(-(exp(upper) + exp(lower)) / 2)``, from ``F(x) = e^x - e^2x / 2 + ...``
    (the omitted term is below 1e-17 relative there). Every branch is evaluated on arguments moved into its own range where
    it is not taken."""
    lower, upper = pt.as_tensor_variable(lower), pt.as_tensor_variable(upper)
    low = upper < _CLOGLOG_MASS_LOWER
    cap = upper > _CLOGLOG_EXP_CAP
    skip = pt.or_(low, cap)
    hi = pt.switch(skip, _ONE, upper)
    lo = pt.switch(skip, _ZERO, lower)
    t = pt.exp(hi) * -pt.expm1(lo - hi)
    regular = -pt.exp(pt.switch(lower > _CLOGLOG_EXP_CAP, _CLOGLOG_EXP_CAP, lower)) + pt.switch(cap, _ZERO, pt.log1mexp(-t))
    u_lo = pt.switch(low, upper, _CLOGLOG_MASS_LOWER - _ONE)
    l_lo = pt.switch(low, lower, _CLOGLOG_MASS_LOWER - 2.0)
    tail = u_lo + pt.log1mexp(l_lo - u_lo) + pt.log1p(-_F64_HALF * (pt.exp(u_lo) + pt.exp(l_lo)))
    return pt.switch(low, tail, regular)


def ordinal_logp_rows(kind: str, value, d):
    """``log P(y = value)`` of the ordinal model ``kind`` (``ORDINAL_KINDS``) from the gaps ``d = tau - eta`` (``(..., K -
    1)``, PyTensor), for 1-based categories ``value``; ``-inf`` outside ``1..K``."""
    y = pt.cast(value, "int64")
    k = d.shape[-1]
    has_upper, has_lower = pt.le(y, k), pt.ge(y, 2)
    both = pt.and_(has_upper, has_lower)
    upper = _select(d, pt.clip(y - 1, 0, k - 1))                         # d_y, the row's own gap
    lower = _select(d, pt.clip(y - 2, 0, k - 1))                         # d_{y-1}
    if kind == "cumulative_probit":
        middle = normal_log_mass(pt.switch(both, lower, -1.0), pt.switch(both, upper, 1.0), _ZERO, _ONE)
        out = pt.switch(both, middle, pt.switch(has_upper, log_normal_cdf(upper), log_normal_cdf(-lower)))
    elif kind in ("sratio_logit", "cratio_logit"):
        passed = _select(pt.cumsum(stable_softplus(d), axis=-1), pt.clip(y - 2, 0, k - 1))   # sum_{j<y} softplus(d_j)
        out = pt.switch(has_upper, -stable_softplus(-upper), 0.0) - pt.switch(has_lower, passed, 0.0)
    elif kind == "acat_logit":
        full = pt.concatenate([pt.zeros_like(d[..., :1]), -pt.cumsum(d, axis=-1)], axis=-1)       # s_1 = 0, s_k
        m = pt.max(full, axis=-1, keepdims=True)
        out = (_select(full, pt.clip(y - 1, 0, k)) - m[..., 0]) - pt.log(pt.sum(pt.exp(full - m), axis=-1))
    elif kind == "cumulative_cloglog":                                    # E06
        middle = cloglog_log_mass(pt.switch(both, lower, -1.0), pt.switch(both, upper, 1.0))
        out = pt.switch(both, middle, pt.switch(has_upper, cloglog_log_p1(upper), cloglog_log_p0(lower)))
    else:
        raise ValueError(f"kind must be one of {ORDINAL_KINDS}, got {kind!r}")
    return pt.switch(pt.and_(pt.ge(y, 1), pt.le(y, k + 1)), out, -np.inf)


def ordinal_log_probabilities(kind: str, d):
    """``log P(y = k)``, ``k = 1..K``, of the ordinal model ``kind`` from the gaps ``d = tau - eta`` (``(..., K - 1)``, NumPy),
    ``(..., K)``: the rows of ``ordinal_logp_rows`` for every category (brms's ``posterior_epred`` is their exponential)."""
    from scipy.special import erf, log_ndtr

    def log1mexp(a):                                                        # log(1 - exp(a)), a <= 0
        return np.where(a > -math.log(2.0), np.log(-np.expm1(a)), np.log1p(-np.exp(a)))

    d = np.asarray(d, dtype=float)
    end = np.ones(d.shape[:-1] + (1,))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if kind == "cumulative_probit":                                     # the log Normal mass between lo and hi, as
            hi = np.concatenate([d, np.inf * end], axis=-1)                # normal_log_mass forms it: d_k, d_K = inf
            lo = np.concatenate([-np.inf * end, d], axis=-1)               # d_{k-1}, d_0 = -inf
            upper_tail = log_ndtr(-lo) + log1mexp(log_ndtr(-hi) - log_ndtr(-lo))     # both gaps above 0
            lower_tail = log_ndtr(hi) + log1mexp(log_ndtr(lo) - log_ndtr(hi))        # both below 0
            across = np.log(0.5 * (erf(hi / _SQRT2) - erf(lo / _SQRT2)))            # a sum of two non-negative terms
            return np.where(lo > 0, upper_tail, np.where(hi < 0, lower_tail, across))
        if kind in ("sratio_logit", "cratio_logit"):
            sp = np.logaddexp(0.0, d)                                       # softplus(d_j) = -log(1 - expit(d_j))
            stop = np.concatenate([-np.logaddexp(0.0, -d), 0.0 * end], axis=-1)
            passed = np.concatenate([0.0 * end, np.cumsum(sp, axis=-1)], axis=-1)
            return stop - passed
        if kind == "acat_logit":
            full = np.concatenate([0.0 * end, -np.cumsum(d, axis=-1)], axis=-1)
            m = full.max(axis=-1, keepdims=True)
            return (full - m) - np.log(np.exp(full - m).sum(axis=-1, keepdims=True))
        if kind == "cumulative_cloglog":                                    # E06: cloglog_log_mass's forms, row by row
            hi = np.concatenate([d, np.inf * end], axis=-1)                # d_k, d_K = inf
            lo = np.concatenate([-np.inf * end, d], axis=-1)               # d_{k-1}, d_0 = -inf
            first = np.where(hi < -20.0, hi + np.log1p(-0.5 * np.exp(np.minimum(hi, -20.0))),
                             np.log(-np.expm1(-np.exp(np.clip(hi, -20.0, 40.0)))))           # log F(d_1), cloglog_log_p1
            last = -np.exp(np.minimum(lo, 700.0))                                             # log(1 - F(d_{K-1}))
            skip = (hi < -20.0) | (hi > 700.0)
            h, l = np.where(skip, 1.0, hi), np.where(skip, 0.0, lo)
            regular = -np.exp(np.minimum(lo, 700.0)) + np.where(hi > 700.0, 0.0, log1mexp(np.exp(h) * np.expm1(l - h)))
            tail = hi + log1mexp(lo - hi) + np.log1p(-0.5 * (np.exp(hi) + np.exp(lo)))
            middle = np.where(hi < -20.0, tail, regular)
            return np.where(np.isneginf(lo), first, np.where(np.isposinf(hi), last, middle))
    raise ValueError(f"kind must be one of {ORDINAL_KINDS}, got {kind!r}")


def ordinal_gaps(eta, thresholds, per_threshold: bool):
    """``d = tau - eta``, ``(..., K - 1)``: ``eta`` one value per row (``per_threshold=False``, ``(...)``) or one per row and
    threshold (``cs()``, ``(..., K - 1)``), ``thresholds`` ``(..., K - 1)`` broadcasting against it (NumPy or PyTensor)."""
    return thresholds - (eta if per_threshold else eta[..., None])


def _ordinal_dist(kind: str, per_threshold: bool):
    """The log density and the forward draws of the ordinal model ``kind`` for ``pm.CustomDist`` with the parameters (``eta``,
    the thresholds)."""
    def logp(value, eta, thresholds):
        return ordinal_logp_rows(kind, value, ordinal_gaps(eta, thresholds, per_threshold))

    def random(eta, thresholds, rng=None, size=None):
        d = ordinal_gaps(np.asarray(eta, dtype=float), np.asarray(thresholds, dtype=float), per_threshold)
        return draw_categories(np.exp(ordinal_log_probabilities(kind, d)), rng, size)

    return logp, random


def unordered_thresholds(name, prior_name: str, params: dict, labels, dim: str):
    """brms's unordered ``vector[nthres] Intercept`` (sratio, cratio, acat; D04) with the same prior on every threshold
    (``Priors(intercept=...)``, brms's class ``Intercept``): a PyMC variable named ``name`` with dims ``dim`` (coordinates
    ``labels``), sampled as it is (no transform, as brms's), its density the prior's summed over the thresholds and its forward
    draws the prior's independent draws. Must be called inside a PyMC model context."""
    model = pm.modelcontext(None)
    if dim not in model.coords:
        model.add_coord(dim, list(labels))
    return getattr(pm, prior_name)(name, **{k: np.float64(v) for k, v in params.items()}, dims=dim)


def ordinal_family(kind: str, terms, means, dim: str, add_category_specific: Callable | None = None):
    """Bambi family for brms's ``cumulative("probit")``, ``sratio("logit")``, ``cratio("logit")`` and ``acat("logit")`` (D04),
    built as ``cumulative_logit_family`` is: the linear predictor ``mu`` (no thresholds) as the parent under the identity link,
    the thresholds ``Intercept`` (``ordered_thresholds`` or ``unordered_thresholds``) as an auxiliary parameter, and
    ``b_Intercept = Intercept + sum_p means[p] b_p`` (brms's thresholds of the raw design) a deterministic the likelihood
    reads. ``add_category_specific(mu)``, given for a model with category-specific effects ``cs(x)``, creates their
    coefficients inside the model being built and returns the ``(n, K - 1)`` predictor of every threshold (brms's ``mu`` plus
    ``Xcs bcs``), which the likelihood then reads instead of ``mu``."""
    import bambi as bmb

    if kind not in ORDINAL_KINDS:
        raise ValueError(f"kind must be one of {ORDINAL_KINDS}, got {kind!r}")
    means = np.asarray(means, dtype=np.float64)
    names = tuple(terms)
    per_threshold = add_category_specific is not None
    logp, random = _ordinal_dist(kind, per_threshold)

    class OrdinalLikelihood:
        """``y ~ Ordinal(mu, b_Intercept)`` of ``kind``, ``b_Intercept = Intercept + sum_p mean(x_p) b_p``."""

        def __new__(cls, name, mu, Intercept, **kwargs):
            model = pm.modelcontext(None)
            thresholds = Intercept
            if names:
                b = pt.concatenate([pt.atleast_1d(model[term]) for term in names])
                thresholds = Intercept + pt.dot(means, b)
            thresholds = pm.Deterministic("b_Intercept", thresholds, dims=dim)
            eta = add_category_specific(mu) if per_threshold else mu
            return pm.CustomDist(name, eta, thresholds, logp=logp, random=random,
                                 signature=("(k)" if per_threshold else "()") + ",(k)->()", dtype="int64", **kwargs)

    class OrdinalFamily(bmb.Family):
        SUPPORTED_LINKS = {"mu": ["identity"], "Intercept": ["identity"]}

    likelihood = bmb.Likelihood(f"Ordinal_{kind}", params=["mu", "Intercept"], parent="mu", dist=OrdinalLikelihood)
    return OrdinalFamily(f"ordinal_{kind}", likelihood, link={"mu": "identity", "Intercept": "identity"})


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


# ---- finite mixtures of Gaussian components (brms-parity C05, docs/mixtures.md) ----------------------------------------------
# brms 2.23, mixture(gaussian, ..., gaussian) under its default order = "mu": component k has its own linear predictor
# mu_k = Intercept_mu<k> + Xc b_mu<k> (the intercepts are brms's ordered[K] ordered_Intercept, Intercept_mu1 < ... <
# Intercept_muK, on Stan's ordered coordinates; Xc the design centred on its means) and its own sigma<k>, and the mixing
# proportions are brms's simplex[K] theta with a Dirichlet prior (con_theta). Each row's log density is
#   log_mix(theta, normal_lpdf(y | mu_1, sigma1), ..., normal_lpdf(y | mu_K, sigmaK))
#     = log_sum_exp_k(log(theta_k) + normal_lpdf(y | mu_k, sigma_k)),
# the sum written as _log_sum_exp above (the maximum subtracted inside), finite with a finite gradient for every finite
# argument on both paths. The graph is Bambi's categorical family's per-category design (B05: every coefficient, the
# intercept included, with a trailing axis, here over the components, and the design centred), the intercept B05's
# ordered vector (ordered_thresholds, one prior per component) and theta B03's stick-breaking simplex.

def gaussian_mixture_logp(value, mu, sigma, theta):
    """``log sum_k theta_k Normal(value | mu_k, sigma_k)`` per row (brms's ``log_mix`` of ``normal_lpdf``): ``mu`` is
    ``(..., K)``, ``sigma`` and ``theta`` ``(K,)`` (or broadcastable against ``mu``)."""
    z = (value[..., None] - mu) / sigma
    return _log_sum_exp(pt.log(theta) - np.float64(0.5) * z * z - pt.log(sigma) - np.float64(_HALF_LOG_2PI))


def gaussian_mixture_random(mu, sigma, theta, rng=None, size=None):
    """Draws as brms's ``posterior_predict`` makes them for a mixture: per draw and row a component by ``theta`` (brms's
    ``sample_mixture_ids``), then that component's ``Normal(mu_k, sigma_k)``. ``mu`` is ``(..., K)``; ``sigma`` and ``theta``
    broadcast against it."""
    mu, sigma, theta = (np.asarray(v, dtype=float) for v in (mu, sigma, theta))
    K = mu.shape[-1]
    shape = tuple(size) if size is not None else np.broadcast_shapes(mu.shape[:-1], sigma.shape[:-1], theta.shape[:-1])
    k = draw_categories(np.broadcast_to(theta, shape + (K,)), rng) - 1

    def pick(a):
        return np.take_along_axis(np.broadcast_to(a, shape + (K,)), k[..., None], axis=-1)[..., 0]

    return rng.normal(pick(mu), pick(sigma))


def mixture_mean(mu, theta):
    """brms's ``posterior_epred`` of a Gaussian mixture: ``sum_k theta_k mu_k``, accumulated in component order as brms's
    ``posterior_epred_mixture`` adds it. ``mu`` is ``(..., K)`` and ``theta`` broadcasts against it."""
    mu = np.asarray(mu, dtype=float)
    theta = np.broadcast_to(np.asarray(theta, dtype=float), mu.shape)
    out = np.zeros(mu.shape[:-1])
    for k in range(mu.shape[-1]):
        out = out + theta[..., k] * mu[..., k]
    return out


def component_prior(name: str, params: Mapping[str, np.ndarray]):
    """Bambi's prior of one population-level term of a mixture (C05): the PyMC distribution ``name`` with one value of each
    parameter per (coefficient, component), ``params`` arrays of the term's coefficient shape (``(K,)`` for a scalar term,
    ``(levels, K)`` for a vector term), built under the dims Bambi gives the term (its levels, then the components). Bambi's
    own array-valued priors cannot carry this: it repeats a one-dimensional parameter over the components."""
    import bambi as bmb

    arrays = {k: np.asarray(v, dtype=np.float64) for k, v in params.items()}

    def dist(label, dims=None, **kwargs):
        return getattr(pm, name)(label, dims=dims, **arrays)

    return bmb.Prior(name, dist=dist)


def gaussian_mixture_family(labels, dim: str, scales, theta_name: str, concentration):
    """Bambi family for brms's ``mixture(gaussian, ..., gaussian)`` (C05): Bambi's categorical family, whose every coefficient
    has a trailing axis over the response's levels, here the components (coordinate ``dim``, ``labels`` "1".."K"; B05's
    per-category design, the design centred, the intercept uncentred after sampling), with the components' linear
    predictors ``mu`` (``(n, K)``, brms's ``mu<k>``) as the parent under the identity link. The likelihood creates each
    component's scale under its prior (``scales``: ``(name, Prior)`` in component order, brms's ``sigma<k>``, a positive
    variable on the log scale as Stan's ``real<lower=0>``) and the mixing proportions ``theta_name`` (brms's ``theta``), a
    Dirichlet of ``concentration`` on Stan's stick-breaking coordinates (B03's ``simplex_rv``), in that order, then
    ``gaussian_mixture_logp`` on the response as a number."""
    import bambi as bmb
    from bambi.families.univariate import Categorical as BambiCategorical

    from autobayes.monotonic import simplex_rv

    labels, scales = list(labels), list(scales)

    class GaussianMixture:
        """``y ~ sum_k theta_k Normal(mu_k, sigma_k)``."""

        def __new__(cls, name, mu, **kwargs):
            sigmas = [getattr(pm, prior.name)(scale, **{k: np.float64(v) for k, v in prior.params.items()}) for scale, prior in scales]
            theta = simplex_rv(theta_name, concentration, dim)
            return pm.CustomDist(name, mu, pt.stack(sigmas), theta, logp=gaussian_mixture_logp, random=gaussian_mixture_random,
                                 signature="(k),(k),(k)->()", **kwargs)

    class GaussianMixtureFamily(BambiCategorical):
        SUPPORTED_LINKS = {"mu": ["identity"]}

        def get_data(self, response):
            return np.asarray(response.term.data, dtype=float).squeeze()

        def get_levels(self, response):
            return list(labels)

        def get_coords(self, response):
            return {dim: list(labels)}

        def get_reference(self, response):
            return None

        @staticmethod
        def transform_backend_eta(eta, kwargs):
            # one row per observation even for 'y ~ 1', whose linear predictors are the intercepts alone, (1, K), as Bambi's
            # categorical family makes them (multiplying by ones changes no value)
            return np.ones((np.shape(kwargs["observed"])[0], 1)) * eta

    likelihood = bmb.Likelihood("GaussianMixture", params=["mu"], parent="mu", dist=GaussianMixture)
    return GaussianMixtureFamily("gaussian_mixture", likelihood, link={"mu": "identity"})


# ---- more links for the Tier A families (brms-parity E06, docs/links.md) ----------------------------------------------------
# brms 2.23 under the links its documentation lists for these families (brmsfamily()), with eta the linear predictor (the offset
# included) and the densities of brms's vignette "Parameterization of Response Distributions in brms":
#   poisson("sqrt")          y ~ Poisson(mu), mu = eta^2 for every real eta (brms squares the linear predictor)
#   negbinomial("identity")  y ~ NegBinomial2(mu, shape), mu = eta
#   Gamma("inverse")         y ~ Gamma(shape, rate = shape / mu), mu = 1 / eta
#   Gamma("identity")        the same with mu = eta
#   Beta("probit")           y ~ Beta(mu phi, (1 - mu) phi), mu = Phi(eta)
#   Beta("cloglog")          the same with mu = 1 - exp(-exp(eta))
#   cumulative("cloglog")    P(y <= k) = 1 - exp(-exp(tau_k - eta))   (ORDINAL_KINDS above)
# Each family keeps eta itself as Bambi's parent under the identity link (sqrt_mu, inverse_mu, probit_mu, cloglog_mu; mu under
# the identity links, where the linear predictor is the mean), as the Beta logit family keeps logit_mu, so that the graph's
# deterministic is brms's posterior_linpred. The forms:
# - Poisson: y log(eta^2) - eta^2 - lgamma(y + 1) as 2 y log|eta| - eta^2 - lgamma(y + 1), the first term 0 on y = 0 rows
#   (evaluated at eta = 1 there, so that eta = 0 never reaches log); -inf at eta = 0 for y > 0, as Stan's poisson_lpmf(y | 0).
# - The identity and inverse links need eta > 0 (a positive mean): a row with eta <= 0 has log density -inf and gradient 0
#   (its terms evaluated at eta = 1), where brms's Stan rejects the draw (gamma_lpdf and neg_binomial_2_lpmf throw on a
#   non-positive rate or mean; brms's log_prob raises there). Elsewhere the Gamma density is shape log(rate) - lgamma(shape) +
#   (shape - 1) log y - rate y with log(rate) = log(shape) + log(eta) and rate = shape eta (inverse), or log(shape) - log(eta)
#   and shape / eta (identity); the negative binomial is neg_binomial_2_log_lpmf above at log(mu) = log(eta), its stable
#   softplus form and Stirling coefficient.
# - Beta: log Beta(y | a, b) = (a - 1) log y + (b - 1) log1p(-y) - (lgamma(a) + lgamma(b) - lgamma(a + b)) with a = mu phi and
#   b = (1 - mu) phi, mu and 1 - mu each computed directly, never by subtraction: Phi(eta) = erfc(-eta / sqrt 2) / 2 and
#   Phi(-eta) for probit, -expm1(-exp(eta)) and exp(-exp(eta)) for cloglog. Where a shape parameter is below 1e-10 (1 - mu
#   underflows at eta of about 6.6 under cloglog and 38 under probit) its lgamma is -log(x) - Euler's gamma x from log(x) =
#   log(phi) + log(mu) or log(phi) + log(1 - mu) (log_normal_cdf, cloglog_log_p1, cloglog_log_p0), so the density stays finite
#   with a finite gradient for every finite eta (cloglog: below exp's cap of 700) on both paths; brms forms 1 - mu by
#   subtraction, which is exactly 0 above eta = 8.25 (probit) and about 3.6 (cloglog), where its beta_lpdf rejects (D132).
_EULER_GAMMA = np.float64(0.5772156649015329)
_LOG_TINY_SHAPE = np.float64(math.log(1e-10))


def poisson_sqrt_logp(value, sqrt_mu):
    """brms's ``poisson_lpmf(y | square(eta))`` as ``2 y log|eta| - eta^2 - lgamma(y + 1)``."""
    y = pt.cast(value, "float64")
    positive = pt.gt(y, 0.0)
    eta_pos = pt.switch(positive, sqrt_mu, _F64_ONE)
    return pt.switch(positive, 2.0 * y * pt.log(pt.abs(eta_pos)), 0.0) - pt.sqr(sqrt_mu) - pt.gammaln(y + _F64_ONE)


def poisson_sqrt_random(sqrt_mu, rng=None, size=None):
    mu = np.square(np.asarray(sqrt_mu, dtype=float))
    return rng.poisson(mu, size=size).astype(np.int64)


def _positive_mean(eta):
    """(whether ``eta > 0``, ``eta`` there and 1 elsewhere): the identity and inverse links' rows outside the support are
    evaluated at a valid value and replaced by ``-inf`` afterwards."""
    ok = pt.gt(eta, 0.0)
    return ok, pt.switch(ok, eta, _F64_ONE)


def gamma_link_logp(link: str):
    """``log Gamma(y | shape, shape / mu)`` under the inverse (``mu = 1 / eta``) or the identity (``mu = eta``) link, for
    ``pm.CustomDist`` with the parameters (``eta``, ``alpha``); ``-inf`` where ``eta <= 0``."""
    sign = {"inverse": 1.0, "identity": -1.0}[link]

    def logp(value, eta, alpha):
        ok, e = _positive_mean(eta)
        rate = alpha * e if link == "inverse" else alpha / e
        body = (alpha * (pt.log(alpha) + sign * pt.log(e)) - pt.gammaln(alpha) + (alpha - _F64_ONE) * pt.log(value)
                - rate * value)
        return pt.switch(ok, body, -np.inf)

    return logp


def _check_positive_draw_mean(eta, link: str):
    """Forward draws (``prior_predictive()``) at a linear predictor that gives no valid mean are refused naming the link (D134)."""
    if not np.all(np.asarray(eta) > 0):
        raise ValueError(f"the {link} link gives a non-positive mean (linear predictor <= 0) for some draws and rows; it needs a "
                         "positive linear predictor (divergence D134)")


def gamma_link_random(link: str):
    def random(eta, alpha, rng=None, size=None):
        eta, alpha = np.asarray(eta, dtype=float), np.asarray(alpha, dtype=float)
        _check_positive_draw_mean(eta, link)
        mu = 1.0 / eta if link == "inverse" else eta
        return rng.gamma(alpha, mu / alpha, size=size)

    return random


def negbinomial_identity_logp(value, mu, alpha):
    """brms's ``neg_binomial_2_lpmf(y | eta, shape)`` (mean ``eta``) as ``neg_binomial_2_log_lpmf`` at ``log(eta)``; ``-inf``
    where ``eta <= 0``."""
    ok, e = _positive_mean(mu)
    return pt.switch(ok, neg_binomial_2_log_lpmf(value, pt.log(e), alpha), -np.inf)


def negbinomial_identity_random(mu, alpha, rng=None, size=None):
    mu, alpha = np.asarray(mu, dtype=float), np.asarray(alpha, dtype=float)
    _check_positive_draw_mean(mu, "identity")
    return rng.negative_binomial(alpha, alpha / (alpha + mu), size=size).astype(np.int64)


BETA_LINKS = ("probit", "cloglog")


def _beta_link_parts(link: str, eta):
    """(``mu``, ``1 - mu``, ``log mu``, ``log(1 - mu)``) of the Beta family under ``link`` (PyTensor), each computed directly."""
    if link == "probit":
        return (_F64_HALF * pt.erfc(-eta / _SQRT2), _F64_HALF * pt.erfc(eta / _SQRT2), log_normal_cdf(eta), log_normal_cdf(-eta))
    if link == "cloglog":
        e = pt.exp(pt.switch(eta > _CLOGLOG_EXP_CAP, _CLOGLOG_EXP_CAP, eta))
        return -pt.expm1(-e), pt.exp(-e), cloglog_log_p1(eta), cloglog_log_p0(eta)
    raise ValueError(f"link must be one of {BETA_LINKS}, got {link!r}")


def _lgamma_of_shape(x, log_x):
    """``lgamma(x)`` of a Beta shape parameter ``x > 0`` given also as ``log_x``: ``gammaln(x)`` from 1e-10 up, and below it
    ``-log_x - gamma x`` (``lgamma(x) = -log x - gamma x + pi^2 x^2 / 12 - ...``, the omitted term below 1e-20), which stays
    finite where ``x`` itself has underflowed to 0."""
    tiny = log_x < _LOG_TINY_SHAPE
    x_reg = pt.switch(tiny, _F64_ONE, x)
    log_tiny = pt.switch(tiny, log_x, _LOG_TINY_SHAPE)
    return pt.switch(tiny, -log_tiny - _EULER_GAMMA * pt.exp(log_tiny), pt.gammaln(x_reg))


def beta_link_logp(link: str):
    """``log Beta(y | mu phi, (1 - mu) phi)`` under the probit or the cloglog link, for ``pm.CustomDist`` with the parameters
    (``eta``, ``phi``): finite with a finite gradient for every finite ``eta`` (the comment above); ``-inf`` outside (0, 1)."""
    def logp(value, eta, phi):
        mu, cm, log_mu, log_cm = _beta_link_parts(link, eta)
        a, b = mu * phi, cm * phi
        log_phi = pt.log(phi)
        inside = pt.and_(pt.gt(value, 0.0), pt.lt(value, 1.0))
        y = pt.switch(inside, value, _F64_HALF)
        body = ((a - _F64_ONE) * pt.log(y) + (b - _F64_ONE) * pt.log1p(-y)
                - (_lgamma_of_shape(a, log_phi + log_mu) + _lgamma_of_shape(b, log_phi + log_cm) - pt.gammaln(a + b)))
        return pt.switch(inside, body, -np.inf)

    return logp


def beta_link_random(link: str):
    from autobayes.families.links import INVERSE_LINKS, COMPLEMENTS

    def random(eta, phi, rng=None, size=None):
        eta, phi = np.asarray(eta, dtype=float), np.asarray(phi, dtype=float)
        return rng.beta(INVERSE_LINKS[link](eta) * phi, COMPLEMENTS[link](eta) * phi, size=size)

    return random


def poisson_sqrt_family():
    """Bambi family of brms's ``poisson("sqrt")``: the identity link on ``sqrt_mu`` (the linear predictor), no auxiliary
    parameter."""
    return _zero_family("poisson_sqrt", "PoissonSqrt", ("sqrt_mu",), {"sqrt_mu": "identity"}, poisson_sqrt_logp,
                        poisson_sqrt_random, dtype="int64")             # _zero_family's builder: a CustomDist with an int64 dtype


def negbinomial_identity_family():
    """Bambi family of brms's ``negbinomial("identity")``: the identity link on the mean ``mu`` (the linear predictor), the
    shape (Bambi's ``alpha``) with its prior."""
    return _zero_family("negbinomial_identity", "NegBinomialIdentity", ("mu", "alpha"), {"mu": "identity", "alpha": "log"},
                        negbinomial_identity_logp, negbinomial_identity_random, dtype="int64")


def gamma_link_family(link: str):
    """Bambi family of brms's ``Gamma("inverse")`` (parent ``inverse_mu``) or ``Gamma("identity")`` (parent ``mu``): the identity
    link on the parent (the linear predictor), the shape (Bambi's ``alpha``) with its prior."""
    parent = {"inverse": "inverse_mu", "identity": "mu"}[link]
    return _mean_family(f"gamma_{link}", f"Gamma{link.capitalize()}", (parent, "alpha"), {parent: "identity", "alpha": "log"},
                        gamma_link_logp(link), gamma_link_random(link))


def beta_link_family(link: str):
    """Bambi family of brms's ``Beta("probit")`` or ``Beta("cloglog")``: the identity link on ``<link>_mu`` (the linear
    predictor), the precision ``phi`` with its prior."""
    parent = f"{link}_mu"
    return _mean_family(f"beta_{link}", f"Beta{link.capitalize()}", (parent, "phi"), {parent: "identity", "phi": "log"},
                        beta_link_logp(link), beta_link_random(link))


# ---- bounded families (brms-parity E01, docs/bounded-families.md) ---------------------------------------------------------
# brms 2.23's densities ("Parameterization of Response Distributions in brms"; Kosmidis and Zeileis 2024 for xbeta), with eta the
# linear predictor, mu = inv_logit(eta) under the logit link, the Beta shapes a = mu phi and b = (1 - mu) phi (1 - mu computed as
# expit(-eta), D13), and the probabilities in (0, 1):
#   zero_one_inflated_beta(eta, phi, zoi, coi)   y = 0: log(zoi) + log1m(coi); y = 1: log(zoi) + log(coi);
#                                                0 < y < 1: log1m(zoi) + log Beta(y | a, b)
#   zero_inflated_beta_binomial(eta, phi, zi; n) y = 0: log(zi + (1 - zi) BetaBinomial(0 | n, a, b));
#                                                y > 0: log1m(zi) + log BetaBinomial(y | n, a, b)
#   xbeta(eta, phi, kappa)                       Y = (1 + 2 kappa) B - kappa censored to [0, 1], B ~ Beta(a, b): with
#                                                c = kappa / (1 + 2 kappa), y = 0: log I_c(a, b); y = 1: log I_c(b, a) (the
#                                                upper tail 1 - I_{1-c}(a, b) by the Beta's symmetry, never a subtraction);
#                                                0 < y < 1: log Beta((y + kappa) / (1 + 2 kappa) | a, b) - log1p(2 kappa)
#   hurdle_cumulative(eta, tau, hu)              y = 0: log(hu); y = k in 1..K: log1m(hu) + the cumulative logit row of B05
# where I_x(a, b) is the regularized incomplete Beta function. Each term of a branch is evaluated on its own rows only (the
# others at fixed interior arguments: shapes 1, x = 1/2), so that a branch a switch does not take is finite with a finite
# gradient (an underflowed I_c on a row that is not a zero row would otherwise turn a zero adjoint into 0 * inf = NaN). The
# zero-inflated beta-binomial's zero row is B06's two-term log-sum-exp; its BetaBinomial is PyMC's, as the beta-binomial
# family's (D02). Checked against brms in the oracle cases zero_one_inflated_beta_varint, zero_inflated_beta_binomial_varint,
# xbeta_varint and hurdle_cumulative_varint; in tests/test_bounded_families.py against SciPy and a 50-digit reference.
#
# I_x(a, b) and its gradient in a and b: PyTensor's betainc (Cephes' incbet in C) with PyTensor's gradient (Boik and
# Robison-Cox's continued fraction for the shape derivatives, a scalar loop with C code) on the PyTensor path. JAX's betainc
# has no derivative in a and b and PyTensor lowers no scalar loop to JAX, so ``shape_betainc`` is a distinct op (as
# ``stable_softplus`` is) whose JAX lowering is ``jax.scipy.special.betainc`` with a custom JVP: the derivative in x is the
# Beta density, those in a and b the continued fraction of Boik and Robison-Cox (1998, Journal of Statistical Software 3(1)),
# I_x(p, q) = K A_n / B_n with K = x^p (1 - x)^(q - 1) / (p B(p, q)), differentiated through the recurrences of A_n and B_n,
# with the coefficients' own derivatives by forward-mode autodiff, rescaled every step and stopped when both derivatives change
# by less than 1e-14 relative (at most 1000 steps; NaN if they do not converge), for x <= p / (p + q), and through I_x(a, b) =
# 1 - I_{1-x}(b, a) above. B(p, q) is formed from gammaln: jax.scipy.special.betaln is off by 7e-8 relative at B(10, 3.3).
from pytensor.scalar.basic import upgrade_to_float_no_complex  # noqa: E402
from pytensor.scalar.math import BetaInc  # noqa: E402


class ShapeDifferentiableBetaInc(BetaInc):
    """PyTensor's regularized incomplete Beta function ``betainc(a, b, x)`` (same value, C code and gradient) as a distinct op
    whose JAX lowering is differentiable in ``a`` and ``b`` as well as in ``x``."""

    nfunc_spec = None


shape_betainc = Elemwise(ShapeDifferentiableBetaInc(upgrade_to_float_no_complex, name="shape_betainc"))

_BETAINC_MAX_STEPS = 1000
_BETAINC_TOL = 1e-14

try:  # the JAX lowering; absent JAX, the PyTensor path is all there is
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import betainc as _jax_betainc, digamma as _jax_digamma, gammaln as _jax_gammaln
    from pytensor.link.jax.dispatch import jax_funcify

    def _jax_betaln(a, b):
        return _jax_gammaln(a) + _jax_gammaln(b) - _jax_gammaln(a + b)

    def _cf_coefficients(p, q, x, n):
        """The n-th numerator and denominator coefficients of Boik and Robison-Cox's continued fraction for I_x(p, q)."""
        f = q * x / (p * (1.0 - x))
        p2n = p + 2.0 * n
        first = p * f * (q - 1.0) / (q * (p + 1.0))
        later = ((p * p * f * f * (n - 1.0) / (q * q)) * (p + q + n - 2.0) * (p + n - 1.0) * (q - n)
                 / ((p2n - 3.0) * (p2n - 2.0) ** 2 * (p2n - 1.0)))
        a = jnp.where(n == 1.0, first, later)
        b = (2.0 * (p * f + 2.0 * q) * n * (n + p - 1.0) + p * q * (p - 2.0 - p * f)) / (q * (p2n - 2.0) * p2n)
        return a, b

    def _cf_shape_derivatives(p, q, x):
        """``(dI/dp, dI/dq)`` of ``I_x(p, q)`` for ``x <= p / (p + q)``, by the continued fraction above."""
        one, zero = jnp.ones_like(p), jnp.zeros_like(p)
        K = jnp.exp(p * jnp.log(x) + (q - 1.0) * jnp.log1p(-x) - jnp.log(p) - _jax_betaln(p, q))
        dlogK_p = jnp.log(x) - 1.0 / p + _jax_digamma(p + q) - _jax_digamma(p)
        dlogK_q = jnp.log1p(-x) + _jax_digamma(p + q) - _jax_digamma(q)
        keys = ("A2", "A1", "B2", "B1", "Ap2", "Ap1", "Bp2", "Bp1", "Aq2", "Aq1", "Bq2", "Bq1")
        start = dict(zip(keys, (one, one, zero, one, zero, zero, zero, zero, zero, zero, zero, zero)))
        state = dict(n=jnp.asarray(1.0, dtype=p.dtype), dp=zero, dq=zero, done=jnp.zeros(p.shape, dtype=bool), **start)

        def more(s):
            return (s["n"] <= _BETAINC_MAX_STEPS) & ~jnp.all(s["done"])

        def step(s):
            n = s["n"]
            (a, b), (ap, bp) = jax.jvp(lambda v: _cf_coefficients(v, q, x, n), (p,), (one,))
            (_, _), (aq, bq) = jax.jvp(lambda v: _cf_coefficients(p, v, x, n), (q,), (one,))
            A, B = a * s["A2"] + b * s["A1"], a * s["B2"] + b * s["B1"]
            Ap = ap * s["A2"] + a * s["Ap2"] + bp * s["A1"] + b * s["Ap1"]
            Bp = ap * s["B2"] + a * s["Bp2"] + bp * s["B1"] + b * s["Bp1"]
            Aq = aq * s["A2"] + a * s["Aq2"] + bq * s["A1"] + b * s["Aq1"]
            Bq = aq * s["B2"] + a * s["Bq2"] + bq * s["B1"] + b * s["Bq1"]
            F = A / B
            dp = K * (F * dlogK_p + (Ap - F * Bp) / B)
            dq = K * (F * dlogK_q + (Aq - F * Bq) / B)
            close = ((n >= 3.0) & (jnp.abs(dp - s["dp"]) <= _BETAINC_TOL * jnp.maximum(jnp.abs(dp), 1e-300))
                     & (jnp.abs(dq - s["dq"]) <= _BETAINC_TOL * jnp.maximum(jnp.abs(dq), 1e-300)))
            r = 1.0 / B                                   # rescaled: A / B and its derivatives are unchanged
            new = dict(A2=s["A1"] * r, A1=A * r, B2=s["B1"] * r, B1=one, Ap2=s["Ap1"] * r, Ap1=Ap * r, Bp2=s["Bp1"] * r,
                       Bp1=Bp * r, Aq2=s["Aq1"] * r, Aq1=Aq * r, Bq2=s["Bq1"] * r, Bq1=Bq * r, dp=dp, dq=dq)
            out = {k: jnp.where(s["done"], s[k], v) for k, v in new.items()}
            return dict(n=n + 1.0, done=s["done"] | close, **out)

        s = jax.lax.while_loop(more, step, state)
        return jnp.where(s["done"], s["dp"], jnp.nan), jnp.where(s["done"], s["dq"], jnp.nan)

    def betainc_shape_derivatives(a, b, x):
        """``(dI_x(a, b)/da, dI_x(a, b)/db)`` in JAX, for ``0 < x < 1``: the continued fraction on ``x <= a / (a + b)``, and
        through ``I_x(a, b) = 1 - I_{1-x}(b, a)`` above (the derivative in ``a`` is then minus the one in the second argument
        there)."""
        a, b, x = jnp.broadcast_arrays(jnp.asarray(a), jnp.asarray(b), jnp.asarray(x))
        flip = x > a / (a + b)
        dp, dq = _cf_shape_derivatives(jnp.where(flip, b, a), jnp.where(flip, a, b), jnp.where(flip, 1.0 - x, x))
        return jnp.where(flip, -dq, dp), jnp.where(flip, -dp, dq)

    @jax.custom_jvp
    def jax_shape_betainc(a, b, x):
        """``jax.scipy.special.betainc`` differentiable in all three arguments."""
        return _jax_betainc(a, b, x)

    @jax_shape_betainc.defjvp
    def _jax_shape_betainc_jvp(primals, tangents):
        a, b, x = primals
        ta, tb, tx = tangents
        da, db = betainc_shape_derivatives(a, b, x)
        dx = jnp.exp((a - 1.0) * jnp.log(x) + (b - 1.0) * jnp.log1p(-x) - _jax_betaln(a, b))
        return jax_shape_betainc(a, b, x), da * ta + db * tb + dx * tx

    @jax_funcify.register(ShapeDifferentiableBetaInc)
    def _jax_funcify_shape_betainc(op, **kwargs):
        return jax_shape_betainc
except ImportError:  # pragma: no cover - JAX is an optional extra; the pinned test environment always has it
    pass


_UNIT_SHAPE_HALF = np.float64(0.5)


def _beta_shapes_on(rows, logit_mu, phi):
    """The Beta shapes ``(expit(eta) phi, expit(-eta) phi)`` on ``rows``, and 1 on every other row."""
    a, b = _beta_shape_parameters(logit_mu, phi)
    return pt.switch(rows, a, _F64_ONE), pt.switch(rows, b, _F64_ONE)


def zero_one_inflated_beta_logp(value, logit_mu, phi, zoi, coi):
    """brms's ``zero_one_inflated_beta_lpdf(value | mu, phi, zoi, coi)`` with ``mu = inv_logit(logit_mu)``: ``log(zoi) +
    log1m(coi)`` at 0, ``log(zoi) + log(coi)`` at 1, ``log1m(zoi)`` plus the Beta family's density inside; ``-inf`` outside
    [0, 1]."""
    is_zero, is_one = pt.eq(value, 0), pt.eq(value, 1)
    inside = pt.and_(pt.gt(value, 0), pt.lt(value, 1))
    a, b = _beta_shapes_on(inside, logit_mu, phi)
    positive = pt.log1p(-zoi) + pm.logp(pm.Beta.dist(alpha=a, beta=b), pt.switch(inside, value, _UNIT_SHAPE_HALF))
    return pt.switch(is_zero, pt.log(zoi) + pt.log1p(-coi),
                     pt.switch(is_one, pt.log(zoi) + pt.log(coi), pt.switch(inside, positive, -np.inf)))


def zero_one_inflated_beta_random(logit_mu, phi, zoi, coi, rng=None, size=None):
    """A 0 or 1 where a uniform draw falls below zoi (a 1 where a second one falls below coi), a ``Beta(mu phi, (1 - mu) phi)``
    draw otherwise."""
    shape = size if size is not None else np.broadcast_shapes(np.shape(logit_mu), np.shape(phi), np.shape(zoi), np.shape(coi))
    edge = _zero_part(zoi, shape, rng)
    one = rng.uniform(size=shape) < coi
    inner = rng.beta(expit(logit_mu) * phi, expit(-np.asarray(logit_mu)) * phi, size=shape)
    return np.where(edge, np.where(one, 1.0, 0.0), inner)


def zero_inflated_beta_binomial_logp(value, logit_mu, phi, zi, n):
    """brms's ``zero_inflated_beta_binomial_lpmf(value | n, mu, phi, zi)`` with ``mu = inv_logit(logit_mu)``: the zero row
    ``log(zi + (1 - zi) BetaBinomial(0 | n, a, b))`` as B06's log-sum-exp, the others ``log1m(zi)`` plus PyMC's BetaBinomial
    (``-inf`` outside 0..n)."""
    a, b = _beta_shape_parameters(logit_mu, phi)
    mass = pm.logp(pm.BetaBinomial.dist(n=n, alpha=a, beta=b), value)
    zero = _log_sum_exp2(pt.log(zi), pt.log1p(-zi) + mass)
    return pt.switch(pt.eq(value, 0), zero, pt.log1p(-zi) + mass)


def zero_inflated_beta_binomial_random(logit_mu, phi, zi, n, rng=None, size=None):
    shape = size if size is not None else np.broadcast_shapes(np.shape(logit_mu), np.shape(phi), np.shape(zi), np.shape(n))
    zero = _zero_part(zi, shape, rng)
    p = rng.beta(expit(logit_mu) * phi, expit(-np.asarray(logit_mu)) * phi, size=shape)
    return np.where(zero, 0, rng.binomial(n, p)).astype(np.int64)


def xbeta_logp(value, logit_mu, phi, kappa):
    """brms's ``xbeta_lpdf(value | mu, phi, kappa)`` (Kosmidis and Zeileis 2024): with ``c = kappa / (1 + 2 kappa)``, ``log
    I_c(a, b)`` at 0, ``log I_c(b, a)`` at 1 and ``log Beta((value + kappa) / (1 + 2 kappa) | a, b) - log1p(2 kappa)`` inside;
    ``-inf`` outside [0, 1]."""
    is_zero, is_one = pt.eq(value, 0), pt.eq(value, 1)
    inside = pt.and_(pt.gt(value, 0), pt.lt(value, 1))
    c = kappa / (_F64_ONE + 2.0 * kappa)
    a0, b0 = _beta_shapes_on(is_zero, logit_mu, phi)
    a1, b1 = _beta_shapes_on(is_one, logit_mu, phi)
    zero = pt.log(shape_betainc(a0, b0, pt.switch(is_zero, c, _UNIT_SHAPE_HALF)))
    one = pt.log(shape_betainc(b1, a1, pt.switch(is_one, c, _UNIT_SHAPE_HALF)))
    a, b = _beta_shapes_on(inside, logit_mu, phi)
    u = (pt.switch(inside, value, _UNIT_SHAPE_HALF) + kappa) / (_F64_ONE + 2.0 * kappa)
    density = pm.logp(pm.Beta.dist(alpha=a, beta=b), u) - pt.log1p(2.0 * kappa)
    return pt.switch(is_zero, zero, pt.switch(is_one, one, pt.switch(inside, density, -np.inf)))


def xbeta_random(logit_mu, phi, kappa, rng=None, size=None):
    """``(1 + 2 kappa) B - kappa`` censored to [0, 1], ``B ~ Beta(mu phi, (1 - mu) phi)``."""
    shape = size if size is not None else np.broadcast_shapes(np.shape(logit_mu), np.shape(phi), np.shape(kappa))
    draws = rng.beta(expit(logit_mu) * phi, expit(-np.asarray(logit_mu)) * phi, size=shape)
    return np.clip((1.0 + 2.0 * np.asarray(kappa)) * draws - kappa, 0.0, 1.0)


def zero_one_inflated_beta_family():
    """Bambi family of brms's ``zero_one_inflated_beta()``: the identity link on ``logit_mu`` (the linear predictor), the precision
    ``phi`` and the probabilities ``zoi`` and ``coi`` with their priors."""
    return _zero_family("zero_one_inflated_beta_logit", "ZeroOneInflatedBetaLogit", ("logit_mu", "phi", "zoi", "coi"),
                        {"logit_mu": "identity", "phi": "log", "zoi": "logit", "coi": "logit"}, zero_one_inflated_beta_logp,
                        zero_one_inflated_beta_random)


def zero_inflated_beta_binomial_family():
    """Bambi family of brms's ``zero_inflated_beta_binomial()`` for a two-column ``p(k, n)`` response: the identity link on
    ``logit_mu`` (the linear predictor), the precision ``phi`` and ``zi`` with their priors."""
    return _zero_family("zero_inflated_beta_binomial_logit", "ZeroInflatedBetaBinomialLogit", ("logit_mu", "phi", "zi"),
                        {"logit_mu": "identity", "phi": "log", "zi": "logit"}, zero_inflated_beta_binomial_logp,
                        zero_inflated_beta_binomial_random, dtype="int64", trials=True)


def xbeta_family():
    """Bambi family of brms's ``xbeta()``: the identity link on ``logit_mu`` (the linear predictor), the precision ``phi`` and
    ``kappa`` with their priors."""
    return _zero_family("xbeta_logit", "XBetaLogit", ("logit_mu", "phi", "kappa"),
                        {"logit_mu": "identity", "phi": "log", "kappa": "log"}, xbeta_logp, xbeta_random)


def hurdle_ordered_logistic_logp(value, eta, thresholds, hu):
    """brms's ``hurdle_cumulative_logit_lpmf(value | eta, hu, thresholds)``: ``log(hu)`` at 0 and ``log1m(hu)`` plus the
    cumulative logit row (``ordered_logistic_logp``, read at category 1 on the zero rows) at the categories ``1..K``; ``-inf``
    outside ``0..K``."""
    y = pt.cast(value, "int64")
    is_zero = pt.eq(y, 0)
    ordinal = ordered_logistic_logp(pt.switch(is_zero, 1, y), eta, thresholds)
    return pt.switch(pt.lt(y, 0), -np.inf, pt.switch(is_zero, pt.log(hu), pt.log1p(-hu) + ordinal))


def hurdle_cumulative_probabilities(eta, thresholds, hu):
    """``P(y = k)``, ``k = 0..K``: ``hu`` for the hurdle category and ``(1 - hu)`` times the cumulative logit probabilities
    (``cumulative_probabilities``) of ``1..K``; ``(..., K + 1)``."""
    ordinal = cumulative_probabilities(eta, thresholds)
    hu = np.broadcast_to(np.asarray(hu, dtype=float), ordinal.shape[:-1])[..., None]
    return np.concatenate([hu, (1.0 - hu) * ordinal], axis=-1)


def hurdle_ordered_logistic_random(eta, thresholds, hu, rng=None, size=None):
    """0 where a uniform draw falls below hu, otherwise a category ``1..K`` of the cumulative logit model."""
    shape = tuple(size) if size is not None else np.broadcast_shapes(np.shape(eta), np.shape(thresholds)[:-1], np.shape(hu))
    zero = _zero_part(hu, shape, rng)
    return np.where(zero, 0, draw_categories(cumulative_probabilities(eta, thresholds), rng, shape)).astype(np.int64)


def hurdle_cumulative_logit_family(terms, means, dim: str):
    """Bambi family for brms's ``hurdle_cumulative("logit")``, built as ``cumulative_logit_family`` is (the linear predictor
    ``mu`` as the parent under the identity link, the thresholds ``Intercept`` an auxiliary parameter and ``b_Intercept =
    Intercept + sum_p means[p] b_p`` the deterministic the likelihood reads), with the hurdle probability ``hu`` (brms's name,
    its own prior, sampled as ``hu_logodds__``) and the likelihood ``hurdle_ordered_logistic_logp``."""
    import bambi as bmb

    means = np.asarray(means, dtype=np.float64)
    names = tuple(terms)

    class HurdleCumulativeLogit:
        """``y ~ HurdleOrderedLogistic(mu, b_Intercept, hu)``, ``b_Intercept = Intercept + sum_p mean(x_p) b_p``."""

        def __new__(cls, name, mu, Intercept, hu, **kwargs):
            model = pm.modelcontext(None)
            thresholds = Intercept
            if names:
                b = pt.concatenate([pt.atleast_1d(model[term]) for term in names])
                thresholds = Intercept + pt.dot(means, b)
            thresholds = pm.Deterministic("b_Intercept", thresholds, dims=dim)
            return pm.CustomDist(name, mu, thresholds, hu, logp=hurdle_ordered_logistic_logp, random=hurdle_ordered_logistic_random,
                                 signature="(),(k),()->()", dtype="int64", **kwargs)

    class HurdleCumulativeLogitFamily(bmb.Family):
        SUPPORTED_LINKS = {"mu": ["identity"], "Intercept": ["identity"], "hu": ["logit"]}

    likelihood = bmb.Likelihood("HurdleCumulativeLogit", params=["mu", "Intercept", "hu"], parent="mu", dist=HurdleCumulativeLogit)
    return HurdleCumulativeLogitFamily("hurdle_cumulative_logit", likelihood,
                                       link={"mu": "identity", "Intercept": "identity", "hu": "logit"})
