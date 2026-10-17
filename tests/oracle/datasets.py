"""Seeded data for oracle cases. A case names a generator and its arguments; `oracle/regen.py` writes the
frame to `data.csv` once, and both brms and AutoBayes then read that file, so the two sides see the same
doubles. Values are rounded to four decimals so that the decimal text parses to the same double in R and
in Python.

Every frame has the same columns: numeric `x`, `w`; categorical `z` (p, q, r); grouping factors `g` (six
levels) and `h` (four); responses `y` (Gaussian), `yb` (0/1), `k` successes out of `size` trials (never
zero trials), and `cnt` (Poisson counts). A case uses the columns its formula names. The tier generators add
columns after these; only `tier_c`'s `xm` (which `tier_d` and `tier_e` inherit) has missing values (`NA_COLUMNS`): empty
fields in the CSV, NaN in Python, and NA in R for a case that declares the column in `brms.na_columns`.
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


def newdata(seed: int, n: int = 5, *, tier_a: bool = False, pairs_from: pd.DataFrame | None = None,
            tier_b: bool = False, tier_c: bool = False, tier_d: bool = False, tier_e: bool = False) -> pd.DataFrame:
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
    if tier_b:                                                  # addition-term and monotonic columns brms needs on new rows
        df["w"] = np.round(rng.uniform(0.5, 2.0, n), 4)
        df["ord"] = rng.integers(1, 5, n).astype(np.int64)
        df["cens_ind"] = np.zeros(n, dtype=np.int64)
    if pairs_from is not None:
        rows = rng.integers(0, len(pairs_from), n)
        df["g"] = pairs_from["g"].to_numpy(dtype=object)[rows]
        df["h"] = pairs_from["h"].to_numpy(dtype=object)[rows]
    if tier_c:                                                  # mi() and me() predictors on new rows (xm observed), drawn
        df["xm"] = np.round(rng.normal(0.6, 0.7, n), 4)         # last so that every column above stays as it was
        df["xobs"] = np.round(rng.normal(0.0, 1.0, n), 4)
        df["sdx"] = np.round(rng.uniform(0.2, 0.6, n), 4)
    if tier_d:                                                  # mm() memberships and weights, gr(by =)'s factor; drawn last
        df["g2"] = rng.choice(LEVELS["g"], size=n).astype(object)
        df["wmm1"] = np.round(rng.uniform(0.2, 0.8, n), 4)
        df["wmm2"] = np.round(1.0 - df["wmm1"].to_numpy(), 4)
        df["gby"] = _by_g(df["g"].to_numpy(dtype=object))
    if tier_e:                                                  # cens() indicators and bounds, the offset; drawn last
        df["cens_r"] = np.zeros(n, dtype=np.int64)
        df["cens_i"] = np.zeros(n, dtype=np.int64)
        df["ypos2"] = np.round(np.exp(rng.normal(0.5, 0.5, n)), 4)
        df["eoff"] = np.round(rng.uniform(9.0, 10.0, n), 4)
    return df


def _by_g(g: np.ndarray) -> np.ndarray:
    """gr(g, by = gby)'s factor: a function of `g` (brms requires the levels of `g` nested in those of `by`)."""
    return np.array(["u" if v in ("a", "b", "c") else "v" for v in g], dtype=object)


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


def tier_b(seed: int, n: int = 36) -> pd.DataFrame:
    """`tier_a` plus the columns Tier B features need: an ordinal predictor/response `ord` (1..4), a three-level
    categorical response `cat3`, a censoring indicator `cens_ind` (-1 left, 0 none, 1 right), positive weights `w`
    and zero-inflated counts `zcnt`. Drawn after `tier_a`'s so the shared columns equal `tier_a(seed, n)` exactly."""
    df = tier_a(seed, n)
    rng = np.random.default_rng(seed + 104729)
    eta = 0.3 + 0.5 * df["x"].to_numpy()
    cuts = np.array([-0.8, 0.2, 1.2])
    latent = eta + rng.logistic(0.0, 1.0, n)
    df["ord"] = (1 + (latent[:, None] > cuts[None, :]).sum(1)).astype(np.int64)
    logits = np.column_stack([np.zeros(n), 0.4 * df["x"].to_numpy() + 0.2, -0.3 * df["x"].to_numpy() - 0.1])
    probs = np.exp(logits - logits.max(1, keepdims=True)); probs /= probs.sum(1, keepdims=True)
    df["cat3"] = np.array([f"c{1 + rng.choice(3, p=pr)}" for pr in probs], dtype=object)
    df["cens_ind"] = rng.choice([-1, 0, 1], size=n, p=[0.15, 0.7, 0.15]).astype(np.int64)
    df["w"] = np.round(rng.uniform(0.5, 2.0, n), 4)
    counts = rng.poisson(np.exp(np.clip(eta, -5.0, 4.0)))
    df["zcnt"] = np.where(rng.uniform(size=n) < 0.3, 0, counts).astype(np.int64)
    for col, levels in (("ord", (1, 2, 3, 4)), ("cat3", ("c1", "c2", "c3"))):          # every level present
        vals = df[col].to_numpy(dtype=object)
        for i, lvl in enumerate(levels):
            if lvl not in set(vals.tolist()):
                vals[i] = lvl
        df[col] = vals.astype(np.int64) if col == "ord" else vals
    return df


def tier_c(seed: int, n: int = 36) -> pd.DataFrame:
    """`tier_b` plus the columns Tier C features need: a time index `t` (1, 2, ... in row order within each level of
    `g`) for autocorrelation terms, a second Gaussian response `y2` that shares part of `y`'s noise (multivariate
    models), `xm`, related to `x` and `w`, with a fifth of its values missing (mi()), and `xobs`, `x` observed with
    known noise standard deviations `sdx` (me()). Drawn after `tier_b`'s so the shared columns equal `tier_b(seed, n)`
    exactly."""
    df = tier_b(seed, n)
    rng = np.random.default_rng(seed + 1299709)
    g = df["g"].to_numpy(dtype=object)
    t = np.zeros(n, dtype=np.int64)
    for lvl in LEVELS["g"]:
        rows = np.flatnonzero(g == lvl)
        t[rows] = np.arange(1, rows.size + 1)
    df["t"] = t
    x, w, y = (df[c].to_numpy(dtype=float) for c in ("x", "w", "y"))
    df["y2"] = np.round(0.2 - 0.5 * x + 0.5 * (y - y.mean()) + rng.normal(0.0, 0.6, n), 4)
    xm = np.round(0.5 * x + 0.4 * w + rng.normal(0.0, 0.5, n), 4)
    xm[rng.permutation(n)[: max(2, n // 5)]] = np.nan
    df["xm"] = xm
    sdx = np.round(rng.uniform(0.2, 0.6, n), 4)
    df["xobs"] = np.round(x + rng.normal(0.0, 1.0, n) * sdx, 4)
    df["sdx"] = sdx
    return df


def tier_d(seed: int, n: int = 36) -> pd.DataFrame:
    """`tier_c` plus the columns Tier D features need: zero-inflated binomial successes `zk` out of `size`, a proportion
    `zprop` in [0, 1) with exact zeros (zero-inflated Beta), a non-negative `zpos` with exact zeros (hurdle Gamma and
    lognormal), a second membership `g2` with `g`'s levels and membership weights `wmm1`, `wmm2 = 1 - wmm1` (mm()), and
    `gby`, a factor of which `g` is nested in (gr(g, by = gby)). Drawn after `tier_c`'s so the shared columns equal
    `tier_c(seed, n)` exactly (its `xm` keeps its missing values)."""
    df = tier_c(seed, n)
    rng = np.random.default_rng(seed + 15485863)
    eta = 0.3 + 0.5 * df["x"].to_numpy()
    size = df["size"].to_numpy()
    successes = rng.binomial(size, expit(eta - 0.2))
    df["zk"] = np.where(rng.uniform(size=n) < 0.3, 0, successes).astype(np.int64)
    df["zprop"] = np.where(rng.uniform(size=n) < 0.25, 0.0, df["prop"].to_numpy())
    df["zpos"] = np.where(rng.uniform(size=n) < 0.25, 0.0, df["ypos"].to_numpy())
    df["g2"] = _labels(rng, "g", n)
    df["wmm1"] = np.round(rng.uniform(0.2, 0.8, n), 4)
    df["wmm2"] = np.round(1.0 - df["wmm1"].to_numpy(), 4)
    df["gby"] = _by_g(df["g"].to_numpy(dtype=object))
    return df


def _every_level(values: np.ndarray, levels) -> np.ndarray:
    """``values`` with each of ``levels`` present: a missing level replaces the first row's value, then the next row's."""
    vals = values.copy()
    for i, lvl in enumerate(levels):
        if lvl not in set(vals.tolist()):
            vals[i] = lvl
    return vals


def tier_e(seed: int, n: int = 36) -> pd.DataFrame:
    """`tier_d` plus the columns Tier E features need: a proportion `zoprop` in [0, 1] with exact zeros and ones (zero-one
    inflated and extended-support Beta), an ordinal `ordh` in 0..4 whose 0 is the hurdle category (brms's hurdle_cumulative
    models 1..4 as ordinal and 0 as the hurdle), an angle `yang` in (-pi, pi) (von Mises), a right-censoring indicator
    `cens_r` (0 none, 1 right), an interval-censoring indicator `cens_i` (-1 left, 0 none, 1 right, 2 interval) with upper
    bounds `ypos2` > `ypos`, zero-truncated counts `ytr` >= 1 (trunc(lb = 1)) and an offset `eoff` in (9, 10) that keeps a
    linear predictor positive at every seeded point (identity, inverse and 1/mu^2 links of positive means). Drawn after
    `tier_d`'s so the shared columns equal `tier_d(seed, n)` exactly (its `xm` keeps its missing values)."""
    df = tier_d(seed, n)
    rng = np.random.default_rng(seed + 32452843)
    x = df["x"].to_numpy()
    eta = 0.3 + 0.5 * x
    u = rng.uniform(size=n)
    zoprop = np.where(u < 0.15, 0.0, np.where(u > 0.85, 1.0, df["prop"].to_numpy()))
    df["zoprop"] = _every_level(zoprop, (0.0, 1.0))
    ordh = np.where(rng.uniform(size=n) < 0.25, 0, df["ord"].to_numpy())
    df["ordh"] = _every_level(ordh, (0, 1, 2, 3, 4)).astype(np.int64)
    df["yang"] = np.clip(np.round(rng.vonmises(2.0 * np.arctan(eta), 2.0), 4), -3.1415, 3.1415)
    df["cens_r"] = _every_level((rng.uniform(size=n) < 0.25).astype(np.int64), (0, 1)).astype(np.int64)
    cens_i = rng.choice([-1, 0, 1, 2], size=n, p=[0.1, 0.5, 0.1, 0.3])
    df["cens_i"] = _every_level(cens_i, (-1, 0, 1, 2)).astype(np.int64)
    df["ypos2"] = np.round(df["ypos"].to_numpy() * rng.uniform(1.2, 2.0, n), 4)
    draws = rng.negative_binomial(2.0, (1.0 / (1.0 + np.exp(eta)))[:, None], size=(n, 64))
    first = np.argmax(draws > 0, axis=1)                        # the first positive draw of each row (zero-truncated)
    df["ytr"] = np.maximum(draws[np.arange(n), first], 1).astype(np.int64)
    df["eoff"] = np.round(rng.uniform(9.0, 10.0, n), 4)
    return df


GENERATORS = {"baseline": baseline, "tier_a": tier_a, "tier_b": tier_b, "tier_c": tier_c, "tier_d": tier_d, "tier_e": tier_e}
NA_COLUMNS = {"tier_c": ("xm",), "tier_d": ("xm",), "tier_e": ("xm",)}   # the only columns with missing values, by design


def make(spec: dict) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """The data and newdata frames of a case's `data` section."""
    gen = GENERATORS[spec["generator"]]
    data = gen(int(spec["seed"]), int(spec.get("n", 36)))
    nd = spec.get("newdata")
    if not nd:
        return data, None
    tier = spec["generator"]
    return data, newdata(int(nd["seed"]), int(nd.get("n", 5)), tier_a=tier in ("tier_a", "tier_b", "tier_c", "tier_d", "tier_e"),
                         pairs_from=data if nd.get("pairs_from_data") else None,
                         tier_b=tier in ("tier_b", "tier_c", "tier_d", "tier_e"), tier_c=tier in ("tier_c", "tier_d", "tier_e"),
                         tier_d=tier in ("tier_d", "tier_e"), tier_e=tier == "tier_e")


def write_csv(frame: pd.DataFrame, path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")
