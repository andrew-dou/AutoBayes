#!/usr/bin/env python
"""Simulated data sets for the AutoBayes-versus-brms benchmark (examples/brms_parity/README.md).

    python examples/brms_parity/benchmark_data.py <out_dir>

Writes one CSV per model. Every numeric predictor is standardised to mean 0 and SD 1, so that a centred intercept and
the raw-scale intercept coincide and both libraries' intercepts are directly comparable.
"""
import sys
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


MODELS = {"B1_logistic_glmm": logistic_glmm, "B2_gaussian_glmm": gaussian_glmm,
          "B3_ordinal_glmm": ordinal_glmm, "B4_zip_glmm": zip_glmm,
          "B5_weibull_glmm": weibull_glmm, "B6_zinb_glmm": zinb_glmm, "B7_sratio_glmm": sratio_glmm,
          "B8_horseshoe_regression": horseshoe_regression, "B9_betabinomial_glmm": betabinomial_glmm,
          "B10_multimembership": multimembership}


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for k, (name, make) in enumerate(MODELS.items()):
        make(np.random.default_rng(20261015 + k)).to_csv(out / f"{name}.csv", index=False, float_format="%.10g")


if __name__ == "__main__":
    main(sys.argv[1])
