"""Deterministic probe inputs for loo 2.9.0 (C07, docs/post-processing.md): pointwise log-likelihood matrices and
likelihood arrays on which ``probe.R`` recorded loo's ``loo()`` and ``relative_eff()`` in ``loo-2.9.0.json``.

Regenerate the record (R with loo 2.9.0; nothing is sampled):

    python -m tests.loo_probes.inputs <scratch dir>        # writes the inputs as CSV files
    Rscript tests/loo_probes/probe.R <scratch dir> tests/loo_probes/loo-2.9.0.json

``tests/test_post_processing.py`` rebuilds the inputs here, checks their fingerprints (``input-fingerprints.json``) and
compares ``autobayes.post_processing`` with loo's outputs. numpy's normal and Student-t draws go through ``exp``/``log``,
whose last bit differs between platforms' builds, so the exact SHA-256 in the record (of the inputs loo was run on, on the
recording platform) holds only there; the fingerprint, the SHA-256 of the inputs rounded to 8 decimals, holds on every
platform and still changes if the generator does, and loo's outputs are compared to 1e-9, far above that last bit. The probes settle what neither ArviZ nor the papers fix: the tail when
draws tie at its cutoff, a tail whose fit does not exist, log ratios beyond the double range, the relative efficiencies.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


def _tied_column(rng, S: int, z: int) -> np.ndarray:
    """A log-likelihood column whose log ratios ``-log_lik`` have ``z`` draws tied with the ``(M + 1)``-th largest (the tail's
    cutoff) and ``M - z`` above it."""
    M = int(math.ceil(min(0.2 * S, 3.0 * math.sqrt(S))))
    body = rng.uniform(-6.0, -0.5, S - M - 1)
    above = np.sort(rng.uniform(0.1, 3.0, M - z))
    x = np.concatenate([body, np.zeros(z + 1), above])
    return -rng.permutation(x)


def loo_inputs() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``{label: (log_lik (S, N), r_eff (N,))}``; ``r_eff`` NaN where loo is to replace it by 1."""
    rng = np.random.default_rng(20261011)
    out = {}
    N = 6
    scales = np.linspace(0.2, 3.0, N)
    for S in (8, 24, 25, 64, 100, 1000, 4000):
        ll = rng.normal(-1.0, 1.0, (S, N)) * scales
        out[f"normal S={S}"] = (ll, np.ones(N))
        out[f"normal S={S} r_eff"] = (ll, np.linspace(0.3, 1.4, N))
    for S in (100, 1000, 4000):
        out[f"heavy S={S}"] = (-2.0 * np.abs(rng.standard_t(1.5, (S, N))), np.ones(N))
    for S in (25, 100, 1000):
        M = int(math.ceil(min(0.2 * S, 3.0 * math.sqrt(S))))
        q = int(M / 4 + 0.5)
        zs = sorted({max(0, z) for z in (0, 1, q - 2, q - 1, q, q + 1, M - 1, M)})
        cols = [_tied_column(rng, S, z) for z in zs]
        dup = np.concatenate([rng.uniform(-6.0, -0.5, S - M), np.resize(np.sort(rng.uniform(0.1, 3.0, -(-M // 3))), M)])
        cols.append(-rng.permutation(dup))
        out[f"ties S={S}"] = (np.column_stack(cols), np.ones(len(cols)))
    S = 100
    const = np.full((S, 3), -1.0)
    const[:, 1] = np.linspace(-3.0, -1.0, S)
    out["constant S=100"] = (const, np.ones(3))
    out["very negative S=100"] = (rng.normal(-800.0, 30.0, (S, N)), np.ones(N))
    out["negative S=100"] = (rng.normal(-380.0, 3.0, (S, N)), np.ones(N))
    out["wide S=100"] = (rng.normal(0.0, 40.0, (S, N)), np.ones(N))
    ll = rng.normal(-1.0, 1.0, (4000, N))
    out["r_eff NaN S=4000"] = (ll, np.array([np.nan, np.nan, np.nan, 0.5, 0.7, 1.2]))
    return out


def relative_eff_inputs() -> dict[str, np.ndarray]:
    """``{label: likelihood (chains, draws, N)}``: AR(1) chains with autocorrelations from -0.5 to 0.97, one per column."""
    rng = np.random.default_rng(1011)
    rhos = np.array([-0.5, 0.0, 0.3, 0.6, 0.9, 0.97])
    out = {}
    for C, I in ((1, 1000), (2, 1000), (4, 1000), (4, 16), (2, 100)):
        z = np.empty((C, I, rhos.size))
        z[:, 0] = rng.normal(size=(C, rhos.size))
        for t in range(1, I):
            z[:, t] = rhos * z[:, t - 1] + np.sqrt(1.0 - rhos ** 2) * rng.normal(size=(C, rhos.size))
        out[f"C={C} I={I}"] = np.exp(-0.5 * (z + 0.3 * np.arange(1, C + 1)[:, None, None]) ** 2)
    return out


def sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype="<f8").tobytes()).hexdigest()


FINGERPRINTS = Path(__file__).with_name("input-fingerprints.json")


def fingerprint(a: np.ndarray) -> str:
    """The SHA-256 of an input rounded to 8 decimals: the same on every platform (see the module docstring)."""
    return sha256(np.round(np.asarray(a, dtype=float), 8))


def _csv(path: Path, a: np.ndarray) -> None:
    """A matrix as CSV with 17 significant digits and ``NA`` for NaN, which R reads back exactly."""
    path.write_text("\n".join(",".join("NA" if np.isnan(v) else f"{v:.17g}" for v in row) for row in np.atleast_2d(a)) + "\n")


def write(directory: str) -> None:
    """Each input as CSV for ``probe.R`` (a likelihood array as the (chains x draws, N) matrix, chain by chain) and an index
    of labels, shapes and hashes."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    loo_lines, reff_lines = [], []
    for k, (label, (ll, r)) in enumerate(loo_inputs().items()):
        _csv(d / f"loo_{k}.csv", ll)
        _csv(d / f"loo_{k}_r_eff.csv", r)
        loo_lines.append(f"{k}\t{label}\t{sha256(ll)}\t{sha256(r)}")
    for k, (label, lik) in enumerate(relative_eff_inputs().items()):
        C, I, N = lik.shape
        _csv(d / f"reff_{k}.csv", lik.reshape(C * I, N))
        reff_lines.append(f"{k}\t{label}\t{C}\t{I}\t{sha256(lik)}")
    (d / "loo_index.tsv").write_text("\n".join(loo_lines) + "\n")
    (d / "relative_eff_index.tsv").write_text("\n".join(reff_lines) + "\n")
    prints = {"loo": {label: [fingerprint(ll), fingerprint(r)] for label, (ll, r) in loo_inputs().items()},
              "relative_eff": {label: fingerprint(lik) for label, lik in relative_eff_inputs().items()}}
    FINGERPRINTS.write_text(json.dumps(prints, indent=1) + "\n")


if __name__ == "__main__":
    write(sys.argv[1])
