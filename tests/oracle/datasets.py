"""Seeded data for oracle cases. A case names a generator and its arguments; `oracle/regen.py` writes the
frame to `data.csv` once, and both brms and AutoBayes then read that file, so the two sides see the same
doubles. Values are rounded to four decimals so that the decimal text parses to the same double in R and
in Python.

Every frame has the same columns: numeric `x`, `w`; categorical `z` (p, q, r); grouping factors `g` (six
levels) and `h` (four); responses `y` (Gaussian), `yb` (0/1), `k` successes out of `size` trials (never
zero trials), and `cnt` (Poisson counts). A case uses the columns its formula names. The tier generators add
columns after these; only `tier_c`'s `xm` (which `tier_d`, `tier_e` and `tier_f` inherit) has missing values (`NA_COLUMNS`):
empty fields in the CSV, NaN in Python, and NA in R for a case that declares the column in `brms.na_columns`. A Tier F case may
also name matrices that brms receives as ``data2`` (`data2`, `DATA2`), each written next to `data.csv` as `data2_<name>.csv`.
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
            tier_b: bool = False, tier_c: bool = False, tier_d: bool = False, tier_e: bool = False,
            tier_f: bool = False) -> pd.DataFrame:
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
    if tier_f:                                                  # car()'s sites (all fitted) and wiener's dec() and times; drawn last
        df["site"] = rng.choice(np.array(SITES, dtype=object), size=n)
        df["rt"] = np.round(0.35 + rng.gamma(2.0, 0.25, n), 4)
        df["dec"] = rng.integers(0, 2, n).astype(np.int64)
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


GRID = 5                                                        # tier_f: the sites of a GRID by GRID lattice, row by row
SITES = tuple(f"s{k:02d}" for k in range(1, GRID * GRID + 1))


def tier_f(seed: int, n: int = 36) -> pd.DataFrame:
    """`tier_e` plus the columns Tier F features need: counts `m1`, `m2`, `m3` of three categories summing to `size` (brms's
    multinomial and dirichlet_multinomial, ``cbind(m1, m2, m3) | trials(size)``), a simplex `s1`, `s2`, `s3` strictly inside it
    (every part at least 0.0199, ``s3 = 1 - s1 - s2`` to four decimals; dirichlet and logistic_normal), a site `site` of the 5 by 5
    grid (`SITES`; each block of 25 rows holds every site once, in shuffled order: car(), sar()), a panel of subjects `subj`
    (``u01``, ...) by times `tt` (1 to 4; rows in subject and time order, balanced when n is a multiple of 4: unstr(), cosy()) and
    response times `rt` (above 0.35) with decisions `dec` (1 upper, 0 lower boundary; wiener). Drawn after `tier_e`'s so the
    shared columns equal `tier_e(seed, n)` exactly (its `xm` keeps its missing values)."""
    df = tier_e(seed, n)
    rng = np.random.default_rng(seed + 49979687)
    x = df["x"].to_numpy()
    logits = np.column_stack([np.zeros(n), 0.4 * x + 0.2, -0.3 * x - 0.1])
    probs = np.exp(logits - logits.max(1, keepdims=True)); probs /= probs.sum(1, keepdims=True)
    counts = np.array([rng.multinomial(int(s), p) for s, p in zip(df["size"].to_numpy(), probs)])
    for k in range(3):
        df[f"m{k + 1}"] = counts[:, k].astype(np.int64)
    simplex = 0.02 + 0.94 * np.array([rng.dirichlet(4.0 * p + 1.0) for p in probs])
    s1, s2 = np.round(simplex[:, 0], 4), np.round(simplex[:, 1], 4)
    df["s1"], df["s2"], df["s3"] = s1, s2, np.round(1.0 - s1 - s2, 4)
    blocks = -(-n // len(SITES))
    df["site"] = np.concatenate([rng.permutation(np.array(SITES, dtype=object)) for _ in range(blocks)])[:n]
    df["subj"] = np.array([f"u{1 + i // 4:02d}" for i in range(n)], dtype=object)
    df["tt"] = (np.arange(n) % 4 + 1).astype(np.int64)
    df["rt"] = np.round(0.35 + rng.gamma(2.0, 0.25, n), 4)
    df["dec"] = _every_level(rng.binomial(1, expit(0.8 * x)).astype(np.int64), (0, 1)).astype(np.int64)
    return df


def rook_adjacency() -> pd.DataFrame:
    """The 0/1 rook adjacency of `SITES` on their grid (1 for two sites that share an edge), labelled by site on both axes."""
    W = np.zeros((len(SITES), len(SITES)), dtype=np.int64)
    for a in range(len(SITES)):
        i, j = divmod(a, GRID)
        if i + 1 < GRID:
            W[a, a + GRID] = W[a + GRID, a] = 1
        if j + 1 < GRID:
            W[a, a + 1] = W[a + 1, a] = 1
    return pd.DataFrame(W, index=list(SITES), columns=list(SITES))


def _rook_symmetric_normalised(frame: pd.DataFrame) -> pd.DataFrame:
    """sar()'s weights, an N by N matrix: the rook adjacency W of the rows' sites normalised symmetrically,
    D^(-1/2) W D^(-1/2) with D the numbers of neighbours, rows and columns in data order and labelled by site (written to
    twelve decimals); every row must have a site of its own. It has the eigenvalues of the row-standardised D^(-1) W (in
    [-1, 1]), which brms cannot take: R's eigen() of that non-symmetric matrix returns the repeated eigenvalue 0 as complex
    pairs (|Im| about 4e-17) and rstan refuses brms's complex eigenMsar."""
    sites = list(frame["site"])
    if len(set(sites)) != len(sites):
        raise ValueError("rook_symmetric_normalised needs one row per site")
    W = rook_adjacency().loc[sites, sites].to_numpy(dtype=float)
    d = W.sum(1)
    return pd.DataFrame(W / np.sqrt(np.outer(d, d)), index=sites, columns=sites)


def _ar1_covariance(frame: pd.DataFrame) -> pd.DataFrame:
    """fcor()'s fixed covariance of the rows, s_i s_j 0.5^|i - j| with standard deviations s_i cycling through 0.8, 0.9, ...,
    1.2 (positive definite), labelled by row number."""
    n = len(frame)
    s = np.round(0.8 + 0.1 * (np.arange(n) % 5), 4)
    lag = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    labels = [str(i + 1) for i in range(n)]
    return pd.DataFrame(np.outer(s, s) * 0.5 ** lag, index=labels, columns=labels)


DATA2 = {"rook_adjacency": lambda frame: rook_adjacency(), "rook_symmetric_normalised": _rook_symmetric_normalised,
         "ar1_covariance": _ar1_covariance}


def data2(spec: dict, frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The matrices a case hands brms as ``data2`` (Tier F), by name, from its data section's ``"data2": {name: kind}`` (a kind
    of `DATA2`, built from the case's data frame); {} for a case without."""
    return {name: DATA2[kind](frame) for name, kind in spec.get("data2", {}).items()}


def _fixed12(v: float) -> str:
    """``v`` to twelve decimal places, trailing zeros dropped: R reads such text as Python does (the digits form an integer below
    2^53 and 10^12 is exact, so both make one correctly rounded division), which the shortest repr of a double, up to 17
    significant digits, does not guarantee (R's parser rounds 0.11000000000000001 to 0.11)."""
    text = f"{v:.12f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def write_matrix(matrix: pd.DataFrame, path) -> None:
    """A data2 matrix as CSV: a header of column labels after an empty first field, then one line per row, its label first;
    integers as integers and doubles to twelve decimal places (`_fixed12`), the text both sides read. Read back by
    `tests.oracle.fixture.read_matrix` and by `oracle/R/gen_fixture.R`."""
    values = matrix.to_numpy()
    fmt = (lambda v: str(int(v))) if np.issubdtype(values.dtype, np.integer) else (lambda v: _fixed12(float(v)))
    lines = [",".join(["", *map(str, matrix.columns)])]
    lines += [",".join([str(label), *map(fmt, row)]) for label, row in zip(matrix.index, values)]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


GENERATORS = {"baseline": baseline, "tier_a": tier_a, "tier_b": tier_b, "tier_c": tier_c, "tier_d": tier_d, "tier_e": tier_e,
              "tier_f": tier_f}
NA_COLUMNS = {"tier_c": ("xm",), "tier_d": ("xm",), "tier_e": ("xm",), "tier_f": ("xm",)}   # the only columns with missing values, by design


def make(spec: dict) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """The data and newdata frames of a case's `data` section."""
    gen = GENERATORS[spec["generator"]]
    data = gen(int(spec["seed"]), int(spec.get("n", 36)))
    nd = spec.get("newdata")
    if not nd:
        return data, None
    tier = spec["generator"]
    later = ("tier_e", "tier_f")
    return data, newdata(int(nd["seed"]), int(nd.get("n", 5)), tier_a=tier in ("tier_a", "tier_b", "tier_c", "tier_d", *later),
                         pairs_from=data if nd.get("pairs_from_data") else None,
                         tier_b=tier in ("tier_b", "tier_c", "tier_d", *later), tier_c=tier in ("tier_c", "tier_d", *later),
                         tier_d=tier in ("tier_d", *later), tier_e=tier in later, tier_f=tier == "tier_f")


def write_csv(frame: pd.DataFrame, path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")
