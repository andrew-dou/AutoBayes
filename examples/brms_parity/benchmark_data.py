#!/usr/bin/env python
"""Simulated data sets for the AutoBayes-versus-brms benchmark (examples/brms_parity/README.md).

    python examples/brms_parity/benchmark_data.py <out_dir>
    python examples/brms_parity/benchmark_data.py <out_dir> --scaling N [models ...]

Writes one CSV per model. Every numeric predictor is standardised to mean 0 and SD 1, so that a centred intercept and
the raw-scale intercept coincide and both libraries' intercepts are directly comparable. Without ``--scaling`` it writes
B1-B18, unchanged. With ``--scaling N`` it writes only the scaling models S1-S5 (SCALING below) with N rows each, under
the same file names at every N (put each size in its own directory). Their seeds continue the scheme: 20261015 + k, k the
model's position after B18 (S1 is k = 18), the same seed at every N.

The scaling models have J = min(N / 100, J_MAX) groups. J_MAX = 1,000 is the memory cap: AutoBayes builds its design
through Bambi and formulae, and formulae keeps a dense int64 indicator matrix of N x J for every group-level term (8 N J
bytes: 80 GB a term at N = 1M and J = 10,000, 8 GB at J = 1,000). So J = N / 100 up to N = 100,000 and J = 1,000 at
N = 1,000,000.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.special import gamma as gamma_fn


def _std(v):
    return (v - v.mean()) / v.std()


def logistic_glmm(rng, n=2000, J=50):
    """B1: y ~ x + (1 + x | g), Bernoulli, correlated varying intercepts and slopes."""
    g = rng.integers(0, J, n)
    x = _std(rng.normal(size=n))
    L = np.linalg.cholesky(np.array([[1.0, 0.3], [0.3, 1.0]]))
    u = (L @ rng.normal(size=(2, J))).T * np.array([0.8, 0.4])
    y = rng.binomial(1, expit(-0.4 + 0.9 * x + u[g, 0] + u[g, 1] * x))
    return pd.DataFrame({"y": y, "x": x, "g": [f"g{k:03d}" for k in g]})


def gaussian_glmm(rng, n=20000, J=200):
    """B2: y ~ x + z + (1 + x | g), Gaussian, the same structure on ten times the rows."""
    g = rng.integers(0, J, n)
    x, z = _std(rng.normal(size=n)), _std(rng.normal(size=n))
    L = np.linalg.cholesky(np.array([[1.0, -0.2], [-0.2, 1.0]]))
    u = (L @ rng.normal(size=(2, J))).T * np.array([1.0, 0.5])
    y = 2.0 + 0.7 * x - 0.4 * z + u[g, 0] + u[g, 1] * x + rng.normal(0, 1.5, n)
    return pd.DataFrame({"y": y, "x": x, "z": z, "g": [f"g{k:03d}" for k in g]})


def ordinal_glmm(rng, n=1500, J=30, cuts=(-1.5, -0.3, 0.6, 1.8)):
    """B3: y ~ x + (1 | g), cumulative logit with five ordered categories."""
    g = rng.integers(0, J, n)
    x = _std(rng.normal(size=n))
    eta = 0.8 * x + rng.normal(0, 0.7, J)[g]
    latent = eta + rng.logistic(size=n)
    y = 1 + np.searchsorted(np.asarray(cuts), latent)
    return pd.DataFrame({"y": y, "x": x, "g": [f"g{k:03d}" for k in g]})


def zip_glmm(rng, n=2000, J=40):
    """B4: y ~ x + (1 | g), zero-inflated Poisson with zi = 0.3."""
    g = rng.integers(0, J, n)
    x = _std(rng.normal(size=n))
    mu = np.exp(0.6 + 0.5 * x + rng.normal(0, 0.5, J)[g])
    y = np.where(rng.random(n) < 0.3, 0, rng.poisson(mu))
    return pd.DataFrame({"y": y, "x": x, "g": [f"g{k:03d}" for k in g]})


def weibull_glmm(rng, n=2000, J=40):
    """B5: y ~ x + (1 | g), Weibull with shape 1.5 and brms's mean parameterisation (log link on the mean)."""
    g = rng.integers(0, J, n)
    x = _std(rng.normal(size=n))
    mu = np.exp(1.0 + 0.4 * x + rng.normal(0, 0.4, J)[g])
    shape = 1.5
    scale = mu / gamma_fn(1 + 1 / shape)
    return pd.DataFrame({"y": scale * rng.weibull(shape, n), "x": x, "g": [f"g{k:03d}" for k in g]})


def zinb_glmm(rng, n=2000, J=40):
    """B6: y ~ x + (1 | g), zero-inflated negative binomial with shape 2 and zi = 0.25."""
    g = rng.integers(0, J, n)
    x = _std(rng.normal(size=n))
    mu = np.exp(0.8 + 0.5 * x + rng.normal(0, 0.5, J)[g])
    counts = rng.negative_binomial(2.0, 2.0 / (2.0 + mu))
    return pd.DataFrame({"y": np.where(rng.random(n) < 0.25, 0, counts), "x": x, "g": [f"g{k:03d}" for k in g]})


def sratio_glmm(rng, n=1500, J=30, cuts=(-1.0, -0.2, 0.5, 1.2)):
    """B7: y ~ x + (1 | g), the sequential-ratio (stopping) model with five categories: stop at k with probability
    inv_logit(cut_k - eta)."""
    g = rng.integers(0, J, n)
    x = _std(rng.normal(size=n))
    eta = 0.7 * x + rng.normal(0, 0.6, J)[g]
    y = np.full(n, len(cuts) + 1)
    alive = np.ones(n, dtype=bool)
    for k, c in enumerate(cuts, start=1):
        stop = alive & (rng.random(n) < expit(c - eta))
        y[stop] = k
        alive &= ~stop
    return pd.DataFrame({"y": y, "x": x, "g": [f"g{k:03d}" for k in g]})


def horseshoe_regression(rng, n=400, p=20):
    """B8: y ~ x1 + ... + x20, Gaussian, three non-zero coefficients, under a horseshoe prior."""
    X = np.column_stack([_std(rng.normal(size=n)) for _ in range(p)])
    beta = np.zeros(p)
    beta[:3] = (1.0, -0.7, 0.5)
    out = pd.DataFrame(X, columns=[f"x{j + 1}" for j in range(p)])
    out.insert(0, "y", 0.5 + X @ beta + rng.normal(0, 1.0, n))
    return out


def betabinomial_glmm(rng, n=2000, J=40):
    """B9: k | trials(size) ~ x + (1 | g), beta-binomial with precision phi = 20."""
    g = rng.integers(0, J, n)
    x = _std(rng.normal(size=n))
    mu = expit(-0.3 + 0.6 * x + rng.normal(0, 0.5, J)[g])
    size = rng.integers(5, 31, n)
    k = rng.binomial(size, rng.beta(mu * 20.0, (1 - mu) * 20.0))
    return pd.DataFrame({"k": k, "size": size, "x": x, "g": [f"g{k_:03d}" for k_ in g]})


def multimembership(rng, n=1000, J=30):
    """B10: y ~ x + (1 | mm(g1, g2)), Gaussian, each row a member of two groups with equal weights."""
    g1, g2 = rng.integers(0, J, n), rng.integers(0, J, n)
    x = _std(rng.normal(size=n))
    u = rng.normal(0, 0.8, J)
    y = 1.0 + 0.5 * x + 0.5 * (u[g1] + u[g2]) + rng.normal(0, 1.0, n)
    return pd.DataFrame({"y": y, "x": x, "g1": [f"g{k:03d}" for k in g1], "g2": [f"g{k:03d}" for k in g2]})


def gauss_mixture(rng, n=1000):
    """B11: y ~ x, a two-component Gaussian mixture (brms's mixture(gaussian, gaussian), ordered intercepts), theta = (0.35,
    0.65)."""
    x = _std(rng.normal(size=n))
    first = rng.random(n) < 0.35
    y = np.where(first, -1.5 + 0.4 * x + rng.normal(0, 0.6, n), 1.5 + 0.8 * x + rng.normal(0, 1.0, n))
    return pd.DataFrame({"y": y, "x": x})


def gp_exact(rng, n=200):
    """B12: y ~ gp(x), the exact Gaussian process (squared-exponential kernel) on 200 distinct inputs."""
    x = _std(rng.uniform(-3, 3, n))
    return pd.DataFrame({"y": np.sin(2.0 * x) + 0.3 * x + rng.normal(0, 0.3, n), "x": x})


def ar1_panel(rng, J=40, T=50):
    """B13: y ~ x + ar(time = t, gr = g, p = 1), Gaussian, AR(1) residuals (coefficient 0.6) within each of 40 series of 50
    times, rows sorted by series and time."""
    x = _std(rng.normal(size=J * T))
    e = np.empty(J * T)
    for j in range(J):
        prev = 0.0
        for t in range(T):
            prev = 0.6 * prev + rng.normal(0, 0.5)
            e[j * T + t] = prev
    return pd.DataFrame({"y": 1.0 + 0.8 * x + e, "x": x, "t": np.tile(np.arange(T), J),
                         "g": [f"g{k:03d}" for k in np.repeat(np.arange(J), T)]})


def nonlinear_growth(rng, n=1000, J=20):
    """B14: y ~ a * exp(b * x), a ~ 1 + (1 | g), b ~ 1 (brms's nl = TRUE), Gaussian, a varying amplitude."""
    g = rng.integers(0, J, n)
    x = _std(rng.uniform(0, 3, n))
    a = 2.0 + rng.normal(0, 0.3, J)[g]
    return pd.DataFrame({"y": a * np.exp(0.5 * x) + rng.normal(0, 0.4, n), "x": x, "g": [f"g{k:03d}" for k in g]})


def mv_rescor(rng, n=1000, J=30):
    """B15: mvbind(y1, y2) ~ x + (1 | g), Gaussian with a residual correlation of 0.5 (set_rescor(TRUE))."""
    g = rng.integers(0, J, n)
    x = _std(rng.normal(size=n))
    e = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.4], [0.4, 0.64]], n)
    u1, u2 = rng.normal(0, 0.5, J), rng.normal(0, 0.3, J)
    return pd.DataFrame({"y1": 0.5 + 1.0 * x + u1[g] + e[:, 0], "y2": -0.3 + 0.4 * x + u2[g] + e[:, 1], "x": x,
                         "g": [f"g{k:03d}" for k in g]})


def monotonic_gauss(rng, n=1500, J=30):
    """B16: y ~ x + mo(edu) + (1 | g), Gaussian, a monotonic effect of a five-level ordinal predictor (steps 0.5, 0.2, 0.2,
    0.1 of a total 1.2)."""
    g = rng.integers(0, J, n)
    x = _std(rng.normal(size=n))
    edu = rng.integers(1, 6, n)
    effect = 1.2 * np.cumsum([0.0, 0.5, 0.2, 0.2, 0.1])
    y = 0.5 + 0.6 * x + effect[edu - 1] + rng.normal(0, 0.5, J)[g] + rng.normal(0, 1.0, n)
    return pd.DataFrame({"y": y, "x": x, "edu": edu, "g": [f"g{k:03d}" for k in g]})


def weibull_censored(rng, n=2000, J=40):
    """B17: y | cens(cens_r) ~ x + (1 | g), Weibull (shape 1.5, log link on the mean) with about 30% of the times
    right-censored by independent uniform censoring times (cens_r = 1)."""
    g = rng.integers(0, J, n)
    x = _std(rng.normal(size=n))
    mu = np.exp(1.0 + 0.4 * x + rng.normal(0, 0.4, J)[g])
    t = mu / gamma_fn(1 + 1 / 1.5) * rng.weibull(1.5, n)
    c = rng.uniform(0, 4.5 * np.median(t), n)
    return pd.DataFrame({"y": np.minimum(t, c), "cens_r": (c < t).astype(int), "x": x, "g": [f"g{k:03d}" for k in g]})


def distributional_sigma(rng, n=2000, J=40):
    """B18: y ~ x + (1 | g), sigma ~ z, Gaussian with a log-linear residual SD."""
    g = rng.integers(0, J, n)
    x, z = _std(rng.normal(size=n)), _std(rng.normal(size=n))
    y = 1.0 + 0.5 * x + rng.normal(0, 0.5, J)[g] + rng.normal(size=n) * np.exp(-0.2 + 0.5 * z)
    return pd.DataFrame({"y": y, "x": x, "z": z, "g": [f"g{k:03d}" for k in g]})


MODELS = {"B1_logistic_glmm": logistic_glmm, "B2_gaussian_glmm": gaussian_glmm,
          "B3_ordinal_glmm": ordinal_glmm, "B4_zip_glmm": zip_glmm,
          "B5_weibull_glmm": weibull_glmm, "B6_zinb_glmm": zinb_glmm, "B7_sratio_glmm": sratio_glmm,
          "B8_horseshoe_regression": horseshoe_regression, "B9_betabinomial_glmm": betabinomial_glmm,
          "B10_multimembership": multimembership,
          # since 21 October 2026 (seeds 20261015 + 10, ...; B1-B10 unchanged)
          "B11_gauss_mixture": gauss_mixture, "B12_gp_exact": gp_exact, "B13_ar1_panel": ar1_panel,
          "B14_nonlinear_growth": nonlinear_growth, "B15_mv_rescor": mv_rescor, "B16_monotonic": monotonic_gauss,
          "B17_weibull_censored": weibull_censored, "B18_distributional_sigma": distributional_sigma}


J_MAX = 1000


def _groups(n):
    return max(min(n // 100, J_MAX), 2)


def _labels(idx):
    return [f"g{k:04d}" for k in idx]


def scaling_logistic(rng, n):
    """S1: y ~ x1 + x2 + (1 + x1 | g), Bernoulli, correlated varying intercepts and slopes."""
    J = _groups(n)
    g = rng.integers(0, J, n)
    x1, x2 = _std(rng.normal(size=n)), _std(rng.normal(size=n))
    L = np.linalg.cholesky(np.array([[1.0, 0.3], [0.3, 1.0]]))
    u = (L @ rng.normal(size=(2, J))).T * np.array([0.8, 0.4])
    y = rng.binomial(1, expit(-0.4 + 0.9 * x1 - 0.5 * x2 + u[g, 0] + u[g, 1] * x1))
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "g": _labels(g)})


def scaling_gaussian(rng, n):
    """S2: y ~ x1 + x2 + (1 + x1 + x2 | g), Gaussian, three correlated group-level coefficients."""
    J = _groups(n)
    g = rng.integers(0, J, n)
    x1, x2 = _std(rng.normal(size=n)), _std(rng.normal(size=n))
    R = np.array([[1.0, -0.2, 0.3], [-0.2, 1.0, 0.1], [0.3, 0.1, 1.0]])
    u = (np.linalg.cholesky(R) @ rng.normal(size=(3, J))).T * np.array([1.0, 0.5, 0.3])
    y = 2.0 + 0.7 * x1 - 0.4 * x2 + u[g, 0] + u[g, 1] * x1 + u[g, 2] * x2 + rng.normal(0, 1.5, n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "g": _labels(g)})


def scaling_poisson_crossed(rng, n):
    """S3: y ~ x1 + (1 | g1) + (1 | g2), Poisson, two crossed groupings of J and J / 2 levels."""
    J1 = _groups(n)
    J2 = max(J1 // 2, 2)
    g1, g2 = rng.integers(0, J1, n), rng.integers(0, J2, n)
    x1 = _std(rng.normal(size=n))
    y = rng.poisson(np.exp(0.5 + 0.4 * x1 + rng.normal(0, 0.5, J1)[g1] + rng.normal(0, 0.3, J2)[g2]))
    return pd.DataFrame({"y": y, "x1": x1, "g1": _labels(g1), "g2": _labels(g2)})


def scaling_ordinal(rng, n, cuts=(-1.5, -0.3, 0.6, 1.8)):
    """S4: y ~ x1 + x2 + (1 | g), cumulative logit with five ordered categories."""
    J = _groups(n)
    g = rng.integers(0, J, n)
    x1, x2 = _std(rng.normal(size=n)), _std(rng.normal(size=n))
    latent = 0.8 * x1 - 0.3 * x2 + rng.normal(0, 0.7, J)[g] + rng.logistic(size=n)
    return pd.DataFrame({"y": 1 + np.searchsorted(np.asarray(cuts), latent), "x1": x1, "x2": x2, "g": _labels(g)})


def scaling_gp(rng, n):
    """S5: y ~ gp(x, k = 30), the Hilbert-space approximate GP (brms's default c = 5/4) on n distinct inputs."""
    x = _std(rng.uniform(-3, 3, n))
    return pd.DataFrame({"y": np.sin(2.0 * x) + 0.3 * x + rng.normal(0, 0.3, n), "x": x})


SCALING = {"S1_logistic_glmm": scaling_logistic, "S2_gaussian_slopes": scaling_gaussian,
           "S3_poisson_crossed": scaling_poisson_crossed, "S4_ordinal_glmm": scaling_ordinal, "S5_gp_approx": scaling_gp}


def main(out_dir: str, scaling: int | None = None, only=None) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if scaling is None:
        for k, (name, make) in enumerate(MODELS.items()):
            make(np.random.default_rng(20261015 + k)).to_csv(out / f"{name}.csv", index=False, float_format="%.10g")
        return
    for k, (name, make) in enumerate(SCALING.items(), start=len(MODELS)):
        if only and name not in only:
            continue
        make(np.random.default_rng(20261015 + k), scaling).to_csv(out / f"{name}.csv", index=False, float_format="%.10g")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir")
    ap.add_argument("models", nargs="*", help="with --scaling: the scaling models to write (default all five)")
    ap.add_argument("--scaling", type=int, metavar="N", help="write the scaling models S1-S5 with N rows each, not B1-B18")
    a = ap.parse_args()
    if a.models and a.scaling is None:
        ap.error("model names are taken with --scaling only")
    main(a.out_dir, a.scaling, set(a.models) or None)
