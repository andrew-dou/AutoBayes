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


MODELS = {"B1_logistic_glmm": logistic_glmm, "B2_gaussian_glmm": gaussian_glmm,
          "B3_ordinal_glmm": ordinal_glmm, "B4_zip_glmm": zip_glmm}


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for k, (name, make) in enumerate(MODELS.items()):
        make(np.random.default_rng(20261015 + k)).to_csv(out / f"{name}.csv", index=False, float_format="%.10g")


if __name__ == "__main__":
    main(sys.argv[1])
