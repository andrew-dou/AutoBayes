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


def newdata(seed: int, n: int = 5, *, tier_a: bool = False, pairs_from: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rows to predict on, with fitted levels only (unseen levels need stochastic policies on both sides).
    ``pairs_from`` copies (g, h) from random rows of the fitted frame, so a nested factor ``g:h`` sees only fitted
    combinations (brms refuses new levels of a grouping factor by default)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "x": np.round(rng.normal(0.0, 1.0, n), 4), "w": np.round(rng.normal(0.5, 1.5, n), 4),
        "z": rng.choice(LEVELS["z"], size=n).astype(object), "g": rng.choice(LEVELS["g"], size=n).astype(object),
        "h": rng.choice(LEVELS["h"], size=n).astype(object), "size": rng.integers(1, 13, n).astype(np.int64),
    })
    if tier_a:                                                  # drawn after the shared columns, which stay as above
        df["xpos"] = np.round(np.exp(rng.normal(0.0, 0.5, n)), 4)
        df["exposure"] = np.round(np.exp(rng.normal(1.0, 0.4, n)), 4)
    if pairs_from is not None:
        rows = rng.integers(0, len(pairs_from), n)
        df["g"] = pairs_from["g"].to_numpy(dtype=object)[rows]
        df["h"] = pairs_from["h"].to_numpy(dtype=object)[rows]
    return df


def tier_a(seed: int, n: int = 36) -> pd.DataFrame:
    """`baseline` plus the columns Tier A features need: positive `xpos` and `ypos` (log/Gamma/lognormal),
    `exposure` for an offset, a proportion `prop` in (0, 1) for Beta, overdispersed counts `ynb`. Built from
    its own draws after `baseline`'s so the shared columns equal `baseline(seed, n)` exactly."""
    df = baseline(seed, n)
    rng = np.random.default_rng(seed + 7919)
    eta = 0.3 + 0.4 * df["x"].to_numpy()
    df["xpos"] = np.round(np.exp(rng.normal(0.0, 0.5, n)), 4)
    df["ypos"] = np.round(np.exp(eta + rng.normal(0.0, 0.5, n)), 4)
    df["exposure"] = np.round(np.exp(rng.normal(1.0, 0.4, n)), 4)
    df["prop"] = np.round(np.clip(rng.beta(2.0 + 2.0 * (eta - eta.min()), 3.0, n), 0.02, 0.98), 4)
    df["ynb"] = rng.negative_binomial(2.0, 1.0 / (1.0 + np.exp(eta)), n).astype(np.int64)
    return df


GENERATORS = {"baseline": baseline, "tier_a": tier_a}


def make(spec: dict) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """The data and newdata frames of a case's `data` section."""
    gen = GENERATORS[spec["generator"]]
    data = gen(int(spec["seed"]), int(spec.get("n", 36)))
    nd = spec.get("newdata")
    if not nd:
        return data, None
    return data, newdata(int(nd["seed"]), int(nd.get("n", 5)), tier_a=spec["generator"] == "tier_a",
                         pairs_from=data if nd.get("pairs_from_data") else None)


def write_csv(frame: pd.DataFrame, path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")
