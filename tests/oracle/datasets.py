"""Seeded data for oracle cases. A case names a generator and its arguments; `oracle/regen.py` writes the
frame to `data.csv` once, and both brms and AutoBayes then read that file, so the two sides see the same
doubles. Values are rounded to four decimals so that the decimal text parses to the same double in R and
in Python.

Every frame has the same columns: numeric `x`, `w`; categorical `z` (p, q, r); grouping factors `g` (six
levels) and `h` (four); responses `y` (Gaussian), `yb` (0/1), `k` successes out of `size` trials (never
zero trials), and `cnt` (Poisson counts). A case uses the columns its formula names.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

LEVELS = {"z": ("p", "q", "r"), "g": ("a", "b", "c", "d", "e", "f"), "h": ("h1", "h2", "h3", "h4")}
_WEIGHTS = {"z": (0.5, 0.3, 0.2), "g": (0.25, 0.2, 0.2, 0.15, 0.1, 0.1), "h": (0.4, 0.3, 0.2, 0.1)}
_Z_EFFECT = {"p": 0.0, "q": 0.5, "r": -0.6}


def _labels(rng, name: str, n: int) -> np.ndarray:
    """Unbalanced labels with every level present (the first rows carry one level each, then shuffled)."""
    levels = LEVELS[name]
    lab = rng.choice(levels, size=n, p=_WEIGHTS[name]).astype(object)
    lab[: len(levels)] = levels
    rng.shuffle(lab)
    return lab


def baseline(seed: int, n: int = 36) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z, g, h = _labels(rng, "z", n), _labels(rng, "g", n), _labels(rng, "h", n)
    x = np.round(rng.normal(0.0, 1.0, n), 4)
    w = np.round(rng.normal(0.5, 1.5, n), 4)
    ug = dict(zip(LEVELS["g"], rng.normal(0.0, 0.7, len(LEVELS["g"]))))
    uh = dict(zip(LEVELS["h"], rng.normal(0.0, 0.4, len(LEVELS["h"]))))
    ug_row = np.array([ug[v] for v in g])
    eta = (0.4 + 0.8 * x - 0.3 * w + np.array([_Z_EFFECT[v] for v in z]) + 0.3 * x * (z == "q")
           + ug_row + np.array([uh[v] for v in h]))
    size = rng.integers(1, 13, n)
    return pd.DataFrame({
        "x": x, "w": w, "z": z, "g": g, "h": h,
        "y": np.round(eta + rng.normal(0.0, 0.8, n), 4),
        "yb": rng.binomial(1, expit(eta)).astype(np.int64),
        "k": rng.binomial(size, expit(eta - 0.2)).astype(np.int64),
        "size": size.astype(np.int64),
        "cnt": rng.poisson(np.exp(np.clip(0.3 + 0.5 * x + 0.5 * ug_row, -5.0, 4.0))).astype(np.int64),
    })


def newdata(seed: int, n: int = 5) -> pd.DataFrame:
    """Rows to predict on, with fitted levels only (unseen levels need stochastic policies on both sides)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "x": np.round(rng.normal(0.0, 1.0, n), 4), "w": np.round(rng.normal(0.5, 1.5, n), 4),
        "z": rng.choice(LEVELS["z"], size=n).astype(object), "g": rng.choice(LEVELS["g"], size=n).astype(object),
        "h": rng.choice(LEVELS["h"], size=n).astype(object), "size": rng.integers(1, 13, n).astype(np.int64),
    })


GENERATORS = {"baseline": baseline}


def make(spec: dict) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """The data and newdata frames of a case's `data` section."""
    gen = GENERATORS[spec["generator"]]
    data = gen(int(spec["seed"]), int(spec.get("n", 36)))
    nd = spec.get("newdata")
    return data, (newdata(int(nd["seed"]), int(nd.get("n", 5))) if nd else None)


def write_csv(frame: pd.DataFrame, path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")
