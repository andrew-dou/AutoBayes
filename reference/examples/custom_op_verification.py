"""Executable version of the custom-Op pattern in reference/pymc-to-jax.md.

Run it:  python3 reference/examples/custom_op_verification.py
Needs:   pymc, pytensor, jax, scipy  (verified against pymc 5.28.2 / jax 0.9.1)

This is the verification ladder the reference doc tells you to run, as code. A custom
Op with a wrong gradient samples happily and produces a wrong posterior with clean
diagnostics, so none of these checks are optional:

  1. value against scipy's analytic CDF difference
  2. gradient against finite differences
  3. CPU (perform) vs JAX (jax_funcify) agreement, values and gradients
  4. usable inside a live PyMC model

It also characterizes the accuracy limit of the quadrature, which is the part people
get wrong: what degrades Gauss-Legendre here is the width of the integration interval
relative to the scale, NOT heavy tails. See the table it prints.

Copy this file, swap in your own density, and keep the checks.
"""
import warnings; warnings.filterwarnings("ignore")
import os
os.environ["JAX_ENABLE_X64"] = "true"

import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import pytensor
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
from pytensor.link.jax.dispatch import jax_funcify
from scipy import stats, integrate

# --- 1. pure JAX, fully differentiable -------------------------------------
N_QUAD = 64
_nodes, _weights = np.polynomial.legendre.leggauss(N_QUAD)
GL_NODES = jnp.array(_nodes, dtype=jnp.float64)
GL_WEIGHTS = jnp.array(_weights, dtype=jnp.float64)


def _logpdf(x, nu, mu, sigma):
    t = (x - mu) / sigma
    return (
        jax.scipy.special.gammaln((nu + 1) / 2)
        - jax.scipy.special.gammaln(nu / 2)
        - 0.5 * jnp.log(nu * jnp.pi)
        - jnp.log(sigma)
        - ((nu + 1) / 2) * jnp.log1p(t ** 2 / nu)
    )


def interval_logp(y_l, y_u, nu, mu, sigma):
    half = 0.5 * (y_u - y_l)
    mid = 0.5 * (y_u + y_l)
    x = mid[..., None] + half[..., None] * GL_NODES
    logf = _logpdf(x, nu[..., None], mu[..., None], sigma[..., None])
    integral = jnp.sum(GL_WEIGHTS * jnp.exp(logf), axis=-1) * half
    return jnp.log(integral)


# --- 2. PyTensor Ops -------------------------------------------------------
class IntervalLogpGradOp(Op):
    __props__ = ()

    def make_node(self, y_l, y_u, nu, mu, sigma):
        inputs = [pt.as_tensor_variable(v) for v in (y_l, y_u, nu, mu, sigma)]
        return Apply(self, inputs, [pt.dvector() for _ in range(3)])

    def perform(self, node, inputs, outputs):
        y_l, y_u, nu, mu, sigma = map(jnp.asarray, inputs)
        g = jax.jacobian(lambda n, m, s: interval_logp(y_l, y_u, n, m, s).sum(),
                         argnums=(0, 1, 2))(nu, mu, sigma)
        for i, gi in enumerate(g):
            outputs[i][0] = np.asarray(gi, dtype="float64")


_interval_logp_grad_op = IntervalLogpGradOp()


class IntervalLogpOp(Op):
    __props__ = ()

    def make_node(self, y_l, y_u, nu, mu, sigma):
        inputs = [pt.as_tensor_variable(v) for v in (y_l, y_u, nu, mu, sigma)]
        return Apply(self, inputs, [pt.dvector()])

    def perform(self, node, inputs, outputs):
        outputs[0][0] = np.asarray(interval_logp(*map(jnp.asarray, inputs)),
                                   dtype="float64")

    def grad(self, inputs, output_gradients):
        (gz,) = output_gradients
        g_nu, g_mu, g_sigma = _interval_logp_grad_op(*inputs)
        disc = pytensor.gradient.grad_not_implemented
        return [disc(self, 0, inputs[0]), disc(self, 1, inputs[1]),
                gz * g_nu, gz * g_mu, gz * g_sigma]


interval_logp_op = IntervalLogpOp()


@jax_funcify.register(IntervalLogpOp)
def _jax_interval_logp(op, **kwargs):
    return interval_logp


@jax_funcify.register(IntervalLogpGradOp)
def _jax_interval_logp_grad(op, **kwargs):
    def impl(y_l, y_u, nu, mu, sigma):
        return jax.jacobian(lambda n, m, s: interval_logp(y_l, y_u, n, m, s).sum(),
                            argnums=(0, 1, 2))(nu, mu, sigma)
    return impl


# ===========================================================================
ok = True
def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")

y_l = np.array([-1.0, 0.2, -3.0, 0.0])
y_u = np.array([ 1.0, 0.9,  2.0, 0.5])
nu  = np.array([ 3.0, 7.5, 30.0, 4.2])
mu  = np.array([ 0.0, 0.5, -0.4, 0.1])
sig = np.array([ 1.0, 0.3,  2.0, 0.7])

print("\n1. value vs scipy")
got = np.asarray(interval_logp(*map(jnp.asarray, (y_l, y_u, nu, mu, sig))))
exp = np.log(stats.t.cdf(y_u, nu, mu, sig) - stats.t.cdf(y_l, nu, mu, sig))
check("quadrature matches analytic CDF difference", np.allclose(got, exp, atol=1e-10),
      f"max abs diff {np.max(np.abs(got - exp)):.2e}")

def _err(lo, hi, nu_, n=N_QUAD):
    nd, wt = np.polynomial.legendre.leggauss(n)
    half, mid = 0.5 * (hi - lo), 0.5 * (hi + lo)
    lf = _logpdf(mid + half * jnp.array(nd), nu_, 0.0, 1.0)
    q = float(jnp.log(jnp.sum(jnp.array(wt) * jnp.exp(lf)) * half))
    return abs(q - float(np.log(stats.t.cdf(hi, nu_, 0, 1) - stats.t.cdf(lo, nu_, 0, 1))))

print("\n1b. accuracy limit — what actually degrades Gauss-Legendre")
print("   interval width relative to sigma IS the driver (nu=2.1, GL-64):")
for w in (10, 20, 30, 60, 100):
    print(f"      width/sigma={w:4d}   err={_err(-w / 2, w / 2, 2.1):.2e}")
print("   degrees of freedom are NOT (width/sigma fixed at 10):")
for nu_ in (1.5, 2.1, 10.0, 100.0):
    print(f"      nu={nu_:6.1f}         err={_err(-5, 5, nu_):.2e}")
print("   node count recovers it (width/sigma=60, nu=2.1):")
for n in (64, 128, 256, 512):
    print(f"      N_QUAD={n:4d}      err={_err(-30, 30, 2.1, n):.2e}")
check("GL-64 is exact within ~10 sigma", _err(-5, 5, 2.1) < 1e-12,
      f"err {_err(-5, 5, 2.1):.2e}")
check("and degrades past it, as documented", _err(-30, 30, 2.1) > 1e-6,
      f"err at 60 sigma {_err(-30, 30, 2.1):.2e} — raise N_QUAD or split the interval")

print("\n2. gradient vs finite differences")
def fd(f, x, i, h=1e-6):
    xp, xm = x.copy(), x.copy(); xp[i] += h; xm[i] -= h
    return (f(xp) - f(xm)) / (2 * h)

for k, (nm, arr) in enumerate([("nu", nu), ("mu", mu), ("sigma", sig)]):
    ana = np.asarray(jax.jacobian(
        lambda n, m, s: interval_logp(jnp.asarray(y_l), jnp.asarray(y_u), n, m, s).sum(),
        argnums=k)(jnp.asarray(nu), jnp.asarray(mu), jnp.asarray(sig)))
    num = np.array([fd(lambda a, kk=k: float(interval_logp(
        jnp.asarray(y_l), jnp.asarray(y_u),
        jnp.asarray(a if kk == 0 else nu), jnp.asarray(a if kk == 1 else mu),
        jnp.asarray(a if kk == 2 else sig)).sum()), arr, i) for i in range(len(arr))])
    check(f"d/d{nm}", np.allclose(ana, num, rtol=1e-5, atol=1e-7),
          f"max rel err {np.max(np.abs((ana - num) / (num + 1e-12))):.2e}")

print("\n3. CPU (perform) vs JAX (jax_funcify)")
args = [pt.dvector(n) for n in ("y_l", "y_u", "nu", "mu", "sigma")]
out = interval_logp_op(*args)
f_c = pytensor.function(args, out, mode="FAST_RUN")
f_j = pytensor.function(args, out, mode="JAX")
r_c, r_j = f_c(y_l, y_u, nu, mu, sig), f_j(y_l, y_u, nu, mu, sig)
check("both backends compile and agree", np.allclose(r_c, r_j, atol=1e-12),
      f"max abs diff {np.max(np.abs(np.asarray(r_c) - np.asarray(r_j))):.2e}")

gsum = pt.grad(out.sum(), args[2:])
g_c = pytensor.function(args, gsum, mode="FAST_RUN")(y_l, y_u, nu, mu, sig)
g_j = pytensor.function(args, gsum, mode="JAX")(y_l, y_u, nu, mu, sig)
check("gradients agree across backends",
      all(np.allclose(a, b, atol=1e-10) for a, b in zip(g_c, g_j)))

print("\n4. inside a PyMC model")
import pymc as pm
with pm.Model() as m:
    mu_p = pm.Normal("mu_p", 0, 1)
    pm.Potential("censored", interval_logp_op(
        pt.as_tensor(y_l), pt.as_tensor(y_u), pt.as_tensor(nu),
        pt.full_like(pt.as_tensor(mu), 1.0) * mu_p, pt.as_tensor(sig)))
lp = m.compile_logp()
v0, v1 = lp({"mu_p": 0.0}), lp({"mu_p": 0.3})
check("model logp evaluates and is finite", np.isfinite(v0) and np.isfinite(v1))
check("logp actually depends on the parameter", abs(v0 - v1) > 1e-8,
      f"delta {abs(v0 - v1):.4f}")
dlp = m.compile_dlogp()({"mu_p": 0.2})
check("model dlogp is finite", np.all(np.isfinite(dlp)), f"{np.asarray(dlp).ravel()}")

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
raise SystemExit(0 if ok else 1)
