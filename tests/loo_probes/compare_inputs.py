"""Deterministic probe inputs for loo 2.9.0's ``waic()``, ``loo_compare()`` and ``loo_model_weights()`` (D05,
docs/post-processing.md): sets of pointwise log-likelihood matrices, one per model, on which ``probe_compare.R`` recorded
loo's outputs in ``loo-compare-2.9.0.json``.

Regenerate the record (R with loo 2.9.0; nothing is sampled):

    python -m tests.loo_probes.compare_inputs <scratch dir>     # writes the inputs as CSV files and an index
    Rscript tests/loo_probes/probe_compare.R <scratch dir> tests/loo_probes/loo-compare-2.9.0.json

Each matrix is a Gaussian model's log-likelihood ``log N(y_i | b_s x_i, s_s)`` at draws ``b_s``, ``s_s`` of a slope and a
scale, rounded to six decimals: the rounded values are the same on every platform (the last bit of ``log`` is not), so the
SHA-256 of each input, recorded with loo's outputs, checks that ``tests/test_loo_compare.py`` rebuilds exactly what loo was
run on. The sets reach what the frozen brms fixtures do not: three to five models, models given unnamed (loo names them
``model1``, ``model2``, ...), exact ties in elpd (loo keeps the order given), raw importance weights (8 draws) and fitted
Pareto tails (1000 and 4000 draws), log-likelihoods near -800 (WAIC's ``lpd`` must not underflow), nearly equal models
(weights near 1/3) and models far apart (a weight that underflows to zero).
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

DECIMALS = 6


def _gaussian_ll(rng, y, x, S: int, slope: float, noise: float, sigma: float, shift: float = 0.0) -> np.ndarray:
    """``(S, N)``: ``log N(y_i | b_s x_i, s_s) + shift`` with ``b_s = slope + noise z_s`` and ``s_s = sigma exp(0.1 z'_s)``."""
    b = slope + noise * rng.standard_normal(S)
    s = sigma * np.exp(0.1 * rng.standard_normal(S))
    mu = b[:, None] * x[None, :]
    ll = -0.5 * np.log(2.0 * np.pi * s[:, None] ** 2) - 0.5 * ((y[None, :] - mu) / s[:, None]) ** 2 + shift
    return np.round(ll, DECIMALS)


def compare_inputs() -> dict[str, tuple[bool, dict[str, np.ndarray]]]:
    """``{label: (named, {model: log_lik (S, N)})}``: ``named`` False where loo is given the models unnamed (its rows are then
    ``model1``, ``model2``, ...; the keys here are those names)."""
    rng = np.random.default_rng(20261015)
    out = {}

    def data(N):
        x = np.round(rng.uniform(-2.0, 2.0, N), DECIMALS)
        return np.round(0.7 * x + rng.standard_normal(N), DECIMALS), x

    y, x = data(20)
    out["three S=100"] = (False, {f"model{k}": _gaussian_ll(rng, y, x, 100, slope, 0.15, sigma)
                                  for k, (slope, sigma) in enumerate(((0.7, 1.0), (0.4, 1.1), (0.8, 0.9)), start=1)})
    y, x = data(12)
    better, worse = _gaussian_ll(rng, y, x, 100, 0.7, 0.2, 1.0), _gaussian_ll(rng, y, x, 100, 0.2, 0.2, 1.2)
    out["ties S=100"] = (True, {"a": worse, "b": better, "c": better.copy(), "d": worse.copy()})
    y, x = data(36)
    out["two S=8"] = (True, {"x": _gaussian_ll(rng, y, x, 8, 0.7, 0.3, 1.0), "xw": _gaussian_ll(rng, y, x, 8, 0.5, 0.3, 1.0)})
    y, x = data(30)
    out["five S=1000"] = (True, {f"e{k}": _gaussian_ll(rng, y, x, 1000, slope, noise, sigma)
                                 for k, (slope, noise, sigma) in enumerate(((0.3, 0.1, 1.2), (0.7, 0.1, 1.0), (0.9, 0.3, 1.0),
                                                                            (0.0, 0.05, 1.3), (0.7, 0.4, 0.8)), start=1)})
    y, x = data(6)
    out["very negative S=100"] = (True, {"p": _gaussian_ll(rng, y, x, 100, 0.7, 0.2, 1.0, shift=-800.0),
                                         "q": _gaussian_ll(rng, y, x, 100, 0.6, 0.2, 1.0, shift=-805.0)})
    y, x = data(10)
    out["close S=4000"] = (True, {name: _gaussian_ll(rng, y, x, 4000, 0.7, 0.2, 1.0) for name in ("u", "v", "w")})
    y, x = data(40)
    out["far apart S=100"] = (True, {"near": _gaussian_ll(rng, y, x, 100, 0.7, 0.1, 1.0),
                                     "far": _gaussian_ll(rng, y, x, 100, -8.0, 0.1, 0.3)})
    return out


def sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype="<f8").tobytes()).hexdigest()


def _csv(path: Path, a: np.ndarray) -> None:
    """A matrix as CSV with 17 significant digits, which R reads back exactly."""
    path.write_text("\n".join(",".join(f"{v:.17g}" for v in row) for row in np.atleast_2d(a)) + "\n")


def write(directory: str) -> None:
    """Each matrix as CSV for ``probe_compare.R`` and an index: set, label, named, model position, name, file, SHA-256."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    lines = []
    for k, (label, (named, models)) in enumerate(compare_inputs().items()):
        for j, (name, ll) in enumerate(models.items(), start=1):
            path = f"compare_{k}_{j}.csv"
            _csv(d / path, ll)
            lines.append(f"{k}\t{label}\t{int(named)}\t{j}\t{name}\t{path}\t{sha256(ll)}")
    (d / "compare_index.tsv").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    write(sys.argv[1])
