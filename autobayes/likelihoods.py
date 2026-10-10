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


# ---- the probit link for the Bernoulli and binomial families (brms-parity A12), and the withheld cloglog ------
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
# Only the probit link is offered (families/links.py): the cloglog forms below are exact where brms's own
# gradient is not (Stan forms 1 - theta by subtraction, losing eps * exp(exp(eta)) relative for a y = 0 row at
# eta above about 2.7), so the frozen brms fixture cannot certify the link to the oracle's tolerance; they are
# kept, tested and reached only from the tests until a fixture can (docs/links.md, divergence D18). The hurdle Poisson
# (B06, below) uses cloglog_log_p1 for what it also is, log P(count > 0) = log(1 - exp(-exp(eta))) of a Poisson count.

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
    """Bambi family for the Bernoulli response under the probit link (or the withheld cloglog link, reached only from
    the tests): the identity link on the parent ``<link>_p``, which is the linear predictor (as
    ``bernoulli_logit_family`` keeps ``logit_p``), and the likelihood of ``LINK_LOG_PROBABILITIES[link]``."""
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
    """Bambi family for the binomial response ``p(k, n)`` under the probit link (or the withheld cloglog link), built
    as ``binomial_logit_family`` is (the two-column response split in ``transform_backend_kwargs``)."""
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
