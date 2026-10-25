"""The brms syntax audit: a corpus of brms formulas, brms's reading of each and what AutoBayes does with it.

``corpus.json`` lists one entry per brms formula: the brms call (an R formula or ``bf()`` expression and a family,
written from brms 2.23's help pages and vignettes, never from its source), the AutoBayes translation (constructor,
formula string or list, keyword arguments), brms's reading of it (``make_standata()`` on ``data.csv``: the Stan data
names, the design columns and the group-level terms; the arrays themselves are in ``standata.json``) and the AutoBayes
result. ``brms_reading.R`` writes the brms side (it needs R and brms); ``python -m tests.brms_syntax.audit`` writes the
AutoBayes side from the library as it stands; ``tests/test_brms_syntax_corpus.py`` checks that the library still does
what the corpus records.

Outcomes (the ``autobayes.outcome`` of an entry):

- ``match``: AutoBayes builds the model and its reading equals brms's Stan data (``compare``: the population-level
  design columns by content, the group-level terms by the partition of the rows and the covariate column, the level
  labels, and the addition and special-term data the model carries).
- ``differs``: AutoBayes builds a model whose reading differs from brms's in a way a divergence row documents.
- ``misread``: AutoBayes builds a model whose reading differs from brms's, silently.
- ``refused_named``: AutoBayes refuses with a deliberate error whose message names the construct.
- ``refused_unclear``: AutoBayes refuses with a deliberate error that does not name the construct, or names it wrongly.
- ``crash``: an exception that is not one of AutoBayes's deliberate refusals (raised inside a dependency, or a
  ``KeyError`` / ``AttributeError`` / ``TypeError`` of the library's own code).

When brms itself refuses the formula, ``brms.status`` is ``"error"``; AutoBayes refusing it too is then the expected
behaviour, and accepting it is recorded as ``misread`` (an extraneous acceptance).
"""
from __future__ import annotations

import json
import math
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus.json"
STANDATA = HERE / "standata.json"
DATA_CSV = HERE / "data.csv"
N_ROWS = 40
SEED = 20261024

# columns R reads as factors with these levels (C-locale order, as R's factor() sorts them in the oracle's locale)
FACTORS = {
    "f": ["a", "b", "c"], "f2": ["u", "v"], "g": ["g1", "g2", "g3", "g4", "g5"], "g2": ["g1", "g2", "g3", "g4", "g5"],
    "h": ["h1", "h2", "h3"], "ycat": ["c1", "c2", "c3"], "gby": ["u", "v"],
}
ORDERED = {"fo": ["lo", "mid", "hi"]}       # an ordered factor (R: factor(ordered = TRUE); pandas: an ordered Categorical)
LOGICAL = ["lgl", "sub"]                    # R logical, pandas bool
NA_COLUMNS = ["xm"]                         # numeric with missing values (an empty CSV field, R's NA, pandas NaN)
CATEGORY_DTYPE = {"fcat": ["a", "b", "c"],   # the same labels as f, held as an (unordered) pandas Categorical
                  "fextra": ["a", "b", "c", "d"]}   # and with a declared category no row takes


def _balanced(rng, labels, n):
    out = np.array([labels[i % len(labels)] for i in range(n)], dtype=object)
    rng.shuffle(out)
    return out


def make_data(seed: int = SEED, n: int = N_ROWS) -> pd.DataFrame:
    """The corpus's data: one frame with every column some entry names, rounded to four decimals so that the CSV text
    parses to the same double in R and in Python. Rows are sorted by ``g`` and then the time ``t`` (1..8 within ``g``), so
    that brms's reordering of an autocorrelation model's rows is the identity."""
    if n != 40:
        raise ValueError("the corpus's frame has 40 rows: five levels of g with eight time points each")
    rng = np.random.default_rng(seed)
    r4 = lambda v: np.round(np.asarray(v, dtype=float), 4)
    g = np.repeat([f"g{i}" for i in range(1, 6)], 8).astype(object)
    t = np.tile(np.arange(1, 9), 5)
    shift = {f"g{i}": f"g{i % 5 + 1}" for i in range(1, 6)}
    g2 = np.array([shift[v] for v in g], dtype=object)                 # a second member for mm(g, g2): never g itself
    h = _balanced(rng, ["h1", "h2", "h3"], n)
    f = _balanced(rng, ["a", "b", "c"], n)
    f2 = _balanced(rng, ["u", "v"], n)
    fo = _balanced(rng, ["lo", "mid", "hi"], n)
    lgl = _balanced(rng, [True, False], n).astype(bool)
    x, z, w = r4(rng.normal(0, 1, n)), r4(rng.normal(0, 1, n)), r4(rng.normal(0.5, 1, n))
    xdot = r4(rng.normal(0, 1, n))
    xp = r4(np.exp(rng.normal(0, 0.5, n)))
    o = _balanced(rng, [0, 1, 2, 3], n).astype(np.int64)
    ug = dict(zip([f"g{i}" for i in range(1, 6)], rng.normal(0, 0.6, 5)))
    eta = 0.4 + 0.6 * x - 0.4 * z + np.array([ug[v] for v in g]) + 0.3 * (f == "b") - 0.3 * (f == "c")
    y = r4(1.0 + eta + rng.normal(0, 0.7, n))
    y2 = r4(0.5 - 0.3 * x + 0.5 * z + rng.normal(0, 0.8, n))
    ypos = r4(np.exp(0.2 + 0.3 * eta + rng.normal(0, 0.3, n)))
    yprop = r4(np.clip(expit(0.3 * eta + rng.normal(0, 0.6, n)), 0.02, 0.98))
    ycount = rng.poisson(np.exp(0.5 + 0.3 * eta)).astype(np.int64)
    ybin = np.where(np.arange(n) < 2, np.arange(n) % 2, rng.binomial(1, expit(eta))).astype(np.int64)
    ntr = rng.integers(5, 16, n).astype(np.int64)
    k = rng.binomial(ntr, expit(0.3 * eta)).astype(np.int64)
    latent = eta + rng.logistic(0, 1, n)
    yord = (np.searchsorted(np.quantile(latent, [0.25, 0.5, 0.75]), latent) + 1).astype(np.int64)
    ycat = _balanced(rng, ["c1", "c2", "c3"], n)
    yang = r4(np.clip(np.angle(np.exp(1j * (0.3 * x + rng.normal(0, 0.5, n)))), -3.1, 3.1))
    yz = np.where(rng.uniform(size=n) < 0.3, 0, rng.poisson(np.exp(0.6 + 0.3 * x))).astype(np.int64)
    yz[:2] = [0, 3]
    cc = np.where(np.arange(n) % 7 == 3, -1, np.where(np.arange(n) % 7 == 5, 1, 0)).astype(np.int64)
    ccs = np.array([{-1: "left", 0: "none", 1: "right"}[c] for c in cc], dtype=object)
    ci = np.where(np.arange(n) % 5 == 2, 2, 0).astype(np.int64)
    y2c = r4(y + rng.uniform(0.2, 1.0, n))
    lbcol = r4(y - rng.uniform(0.5, 2.0, n))
    sdx = r4(rng.uniform(0.1, 0.5, n))
    xobs = r4(x + rng.normal(0, 1, n) * sdx)
    sey = r4(rng.uniform(0.2, 0.6, n))
    wt = r4(rng.uniform(0.5, 2.0, n))
    off = r4(rng.uniform(0.5, 2.0, n))
    expo = r4(rng.uniform(1.0, 5.0, n))
    xm = r4(0.5 * z + rng.normal(0, 0.5, n))
    xm[[3, 11, 19, 27, 35]] = np.nan
    w1, w2 = r4(rng.uniform(0.2, 1.0, n)), r4(rng.uniform(0.2, 1.0, n))
    x1, x2 = r4(rng.normal(0, 1, n)), r4(rng.normal(0, 1, n))
    dec = _balanced(rng, [0, 1], n).astype(np.int64)
    rt = r4(0.3 + rng.gamma(2.0, 0.2, n))
    sub = _balanced(rng, [True, True, False], n).astype(bool)
    ysurv = r4(rng.weibull(1.5, n) * np.exp(0.3 * x) + 0.05)
    ymn1 = rng.binomial(ntr, 0.3)
    ymn2 = rng.binomial(ntr - ymn1, 0.5)
    ymn3 = ntr - ymn1 - ymn2
    p = rng.dirichlet([2.0, 3.0, 4.0], n)
    p = np.round(p, 4)
    p[:, 2] = np.round(1.0 - p[:, 0] - p[:, 1], 4)
    gnum = (np.arange(n) % 12 + 1).astype(np.int64)
    yordgap = np.where(yord == 3, 4, yord).astype(np.int64)                         # ordinal codes 1, 2, 4: category 3 unobserved
    gby = np.array(["u" if v in ("g1", "g2") else "v" for v in g], dtype=object)   # constant within g, for gr(g, by = gby)
    xg_of = dict(zip([f"g{i}" for i in range(1, 6)], r4(rng.normal(0, 1, 5))))     # constant within g, for me(..., gr = g)
    sdxg_of = dict(zip([f"g{i}" for i in range(1, 6)], r4(rng.uniform(0.1, 0.4, 5))))
    xg = np.array([xg_of[v] for v in g])
    sdxg = np.array([sdxg_of[v] for v in g])
    df = pd.DataFrame({
        "y": y, "y2": y2, "ypos": ypos, "yprop": yprop, "ycount": ycount, "ybin": ybin, "k": k, "ntr": ntr,
        "yord": yord, "ycat": ycat, "yang": yang, "yz": yz, "ysurv": ysurv, "rt": rt, "dec": dec,
        "x": x, "z": z, "w": w, "x.dot": xdot, "xp": xp, "o": o, "fo": fo, "f": f, "fcat": f.copy(), "fextra": f.copy(), "f2": f2,
        "lgl": lgl, "g": g, "g2": g2, "h": h, "t": t, "gnum": gnum, "yordgap": yordgap, "gby": gby, "xg": xg, "sdxg": sdxg,
        "cc": cc, "ccs": ccs, "ci": ci, "y2c": y2c, "lbcol": lbcol, "sdx": sdx, "xobs": xobs, "sey": sey,
        "wt": wt, "off": off, "expo": expo, "xm": xm, "w1": w1, "w2": w2, "x1": x1, "x2": x2, "sub": sub,
        "idx": np.arange(1, n + 1, dtype=np.int64),
        "ymn1": ymn1.astype(np.int64), "ymn2": ymn2.astype(np.int64), "ymn3": ymn3.astype(np.int64),
        "p1": p[:, 0], "p2": p[:, 1], "p3": p[:, 2],
    })
    return df


def python_frame(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """The frame as an AutoBayes user holds it: text labels as ``str``, ``fo`` an ordered pandas Categorical (an R ordered
    factor), ``fcat`` an unordered one (an R factor), ``lgl``/``sub`` booleans and ``xm`` with NaN."""
    df = make_data() if df is None else df.copy()
    for col, levels in ORDERED.items():
        df[col] = pd.Categorical(df[col], categories=levels, ordered=True)
    for col, levels in CATEGORY_DTYPE.items():
        df[col] = pd.Categorical(df[col], categories=levels, ordered=False)
    return df


def write_csv(df: pd.DataFrame, path: Path = DATA_CSV) -> None:
    out = df.copy()
    for col in LOGICAL:
        out[col] = np.where(out[col], "TRUE", "FALSE")
    out.to_csv(path, index=False, na_rep="", float_format="%.4f", lineterminator="\n")


def data_spec() -> dict:
    """What ``brms_reading.R`` needs to read ``data.csv`` as brms's data frame."""
    df = make_data()
    integer = [c for c in df.columns if pd.api.types.is_integer_dtype(df[c])]
    return {"generator": "tests.brms_syntax.audit.make_data", "seed": SEED, "n": N_ROWS, "csv": "data.csv",
            "factors": FACTORS, "ordered": ORDERED, "logical": LOGICAL, "na_columns": NA_COLUMNS,
            "category_dtype": CATEGORY_DTYPE, "text": ["ccs"], "integer": integer}


# -- the brms side ------------------------------------------------------------------------------------------
def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def decode(obj) -> np.ndarray:
    """A ``{dims, values}`` array of ``standata.json`` (R's column-major order)."""
    dims = [int(d) for d in obj["dims"]]
    values = np.asarray([math.nan if v in ("NA", "NaN") else v for v in obj["values"]], dtype=float)
    return values.reshape(dims, order="F") if dims else values.reshape(())



# -- the AutoBayes side -------------------------------------------------------------------------------------
DEFAULT_PRIOR_KWARGS = ("categorical", "nu", "link", "center_predictors", "families", "order", "rescor", "nl", "imputed",
                        "quantile")
DESIGN_ABS = 1e-12            # tests/oracle/tolerances.py's G1 tolerance


def family_of(entry: dict) -> str:
    a = entry["autobayes"]
    return a["kwargs"].get("family", "gaussian") if a["constructor"] == "Model" else a["constructor"]


def build(entry: dict, df: pd.DataFrame):
    """Build the entry's AutoBayes model as a user would: ``ab.default_priors`` for the formula (unless the entry gives
    explicit priors), then ``ab.<constructor>(formula, data, priors, **kwargs)``. Returns ``(model, stage)``; an exception
    propagates with ``stage`` set on it ("default_priors" or "constructor")."""
    import autobayes as ab

    a = entry["autobayes"]
    kwargs = dict(a["kwargs"])
    stage = "default_priors"
    try:
        if a["priors"] == "default":
            pk = {k: v for k, v in kwargs.items() if k in DEFAULT_PRIOR_KWARGS}
            priors = ab.default_priors(a["formula"], df, family=family_of(entry), **pk)
        else:
            from tests.oracle.adapter import priors_from_spec
            priors = priors_from_spec(a["priors"])
        stage = "constructor"
        model = getattr(ab, a["constructor"])(a["formula"], df, priors, **kwargs)
    except Exception as exc:
        exc.stage = stage
        raise
    return model, stage


def error_record(exc: BaseException) -> dict:
    """What a refusal or crash looks like: the exception type, its message, where it was raised and how to read it.
    ``mechanism`` is ``"refusal"`` for one of the library's deliberate errors (``FormulaError``, ``PriorError``,
    ``ValueError`` or ``NotImplementedError`` raised in ``autobayes/``), ``"keyword"`` for a constructor that has no such
    keyword argument (Python's ``TypeError``), and ``"crash"`` for everything else."""
    from autobayes.formula import FormulaError
    from autobayes.priors import PriorError

    frames = traceback.extract_tb(exc.__traceback__)
    last = frames[-1] if frames else None
    where = "" if last is None else f"{Path(last.filename).name}:{last.lineno}"
    in_library = last is not None and f"{Path('autobayes')}/" in last.filename.replace("\\", "/") and "site-packages" not in last.filename
    msg = str(exc)
    if isinstance(exc, TypeError) and "unexpected keyword argument" in msg:
        mechanism = "keyword"
    elif in_library and isinstance(exc, (FormulaError, PriorError, ValueError, NotImplementedError)) \
            and not isinstance(exc, (KeyError, IndexError)):
        mechanism = "refusal"
    else:
        mechanism = "crash"
    return {"type": type(exc).__name__, "message": msg, "raised_at": where, "stage": getattr(exc, "stage", ""),
            "mechanism": mechanism}


# -- comparing the AutoBayes model with brms's Stan data ------------------------------------------------------
class Reading:
    """brms's Stan data for one scope of the model (the whole model, or one response of a multivariate one: keys with the
    ``_<resp>`` suffix shown without it), with the keys the comparison has used."""

    def __init__(self, sd: dict, names: dict, groups: list, used: set, resp: str = ""):
        self.sd, self.names, self.groups, self.used, self.resp = sd, names, groups, used, resp

    def key(self, k: str) -> str:
        return f"{k}_{self.resp}" if self.resp else k

    def has(self, k: str) -> bool:
        return self.key(k) in self.names

    def get(self, k: str, raw: bool = False):
        key = k if raw else self.key(k)
        self.used.add(key)
        if key not in self.sd:
            raise KeyError(f"brms's {key} is not recorded in standata.json (too large or not numeric)")
        return self.sd[key]


def _close(a, b) -> bool:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return False
    both_inf = np.isinf(a) & np.isinf(b) & (np.sign(a) == np.sign(b))
    with np.errstate(invalid="ignore"):
        ok = both_inf | (np.abs(a - b) <= DESIGN_ABS)
    return bool(np.all(ok))


def match_design(X: np.ndarray, colnames, XA: np.ndarray, labels, intercept: bool, what: str, issues: list) -> None:
    """AutoBayes's design columns ``XA`` (no intercept column) against brms's ``X``, by content: each AutoBayes column must
    equal exactly one brms column, and brms's remaining columns must be its intercept column (when ``intercept``) only."""
    X = np.asarray(X, dtype=float)
    X = X[:, None] if X.ndim == 1 else X
    XA = np.asarray(XA, dtype=float)
    names = list(colnames) if colnames is not None else [str(k) for k in range(X.shape[1])]
    if X.shape[0] != XA.shape[0]:
        issues.append(f"{what}: brms has {X.shape[0]} rows, AutoBayes {XA.shape[0]}")
        return
    taken = set()
    for c in range(XA.shape[1]):
        hits = [k for k in range(X.shape[1]) if k not in taken and _close(X[:, k], XA[:, c])]
        if not hits:
            issues.append(f"{what}: AutoBayes column {labels[c]!r} matches no brms column {names}")
            continue
        taken.add(hits[0])
    rest = [k for k in range(X.shape[1]) if k not in taken]
    if intercept:
        ones = [k for k in rest if _close(X[:, k], np.ones(X.shape[0]))]
        if len(ones) != 1:
            issues.append(f"{what}: AutoBayes has an intercept; brms's columns of ones among {[names[k] for k in rest]}: {len(ones)}")
        else:
            rest.remove(ones[0])
    if rest:
        issues.append(f"{what}: brms columns without an AutoBayes counterpart: {[names[k] for k in rest]}")


def _zkey(grp: dict, resp: str = "") -> str:
    scope = [p for p in (grp.get("resp") or resp, grp.get("dpar") if grp.get("dpar") not in ("", "mu") else "", grp.get("nlpar"))
             if p]
    return "_".join(["Z", str(grp["id"]), *scope, str(grp["cn"])])


def _jkey(grp: dict, resp: str = "") -> str:
    r = grp.get("resp") or resp
    return f"J_{grp['id']}_{r}" if r else f"J_{grp['id']}"


def match_groups(blocks: dict, terms, reading: Reading, scope: dict, issues: list, notes: list, what: str = "") -> None:
    """AutoBayes's group blocks ``{(factor, coef): GroupBlock}`` against brms's group-level terms in ``scope`` (``dpar`` /
    ``nlpar``): each block must meet exactly one brms coefficient whose ``J`` induces its partition of the rows and whose
    ``Z`` is its covariate column (``W``/``J_k`` per member of a multi-membership term), with the same level labels; every
    brms coefficient of the scope must be met; correlated on both sides or neither."""
    groups = [g for g in reading.groups if (g.get("dpar") or "") == scope.get("dpar", "") and (g.get("nlpar") or "") == scope.get("nlpar", "")
              and (not reading.resp or g.get("resp") == reading.resp)]
    used, ids = set(), {}
    for (factor, coef), block in blocks.items():
        n = block.Z.shape[0]
        found = []
        for grp in groups:
            key = (grp["id"], grp["cn"])
            if key in used:
                continue
            if block.memberships:                                    # mm(): J_m_k, W_m_k, Z_m_c_k per member k
                ok, pairs = True, []
                members = sum(1 for k in range(1, 50) if f"J_{grp['id']}_{k}" in reading.names)
                if members != len(block.memberships):
                    continue
                for k, (mem, wt) in enumerate(zip(block.memberships, block.weights), start=1):
                    J = np.asarray(reading.get(f"J_{grp['id']}_{k}", raw=True)).astype(int) - 1
                    W = np.asarray(reading.get(f"W_{grp['id']}_{k}", raw=True), dtype=float)
                    Zc = np.asarray(reading.get(f"Z_{grp['id']}_{grp['cn']}_{k}", raw=True), dtype=float)
                    pairs += list(zip(J.tolist(), np.asarray(mem).tolist()))
                    ok &= _close(W, wt) and _close(Zc, np.ones(n))
                m = dict(pairs)
                if ok and len(m) == len(set(m.values())) and all(m[j] == a for j, a in pairs):
                    found.append((grp, [m.get(j, -1) for j in range(len(grp["levels"]))]))
                continue
            jk = _jkey(grp, reading.resp)
            if jk not in reading.names:
                continue
            J = np.asarray(reading.get(jk, raw=True)).astype(int) - 1
            gidx = np.asarray(block.group_index)
            if J.shape != gidx.shape:
                continue
            m = {}
            for j, a in zip(J.tolist(), gidx.tolist()):
                if m.setdefault(j, a) != a:
                    break
            else:
                if len(set(m.values())) != len(m):
                    continue
                Zc = np.asarray(reading.get(_zkey(grp, reading.resp), raw=True), dtype=float)
                covariate = block.Z[np.arange(n), gidx]
                if _close(Zc, covariate):
                    found.append((grp, [m.get(j, -1) for j in range(len(grp["levels"]))]))
        if len(found) != 1:
            issues.append(f"{what}group-level coefficient {coef!r} of {factor!r}: {len(found)} brms group-level terms have its "
                          f"partition of the rows and its covariate column")
            continue
        grp, m = found[0]
        used.add((grp["id"], grp["cn"]))
        ids.setdefault(factor, set()).add(grp["id"])
        mine = [block.levels[i] if i >= 0 else None for i in m]
        if mine != list(grp["levels"]):
            issues.append(f"{what}{factor!r}: brms's levels {grp['levels']} are AutoBayes's {mine}")
        elif list(block.levels) != list(grp["levels"]):
            notes.append(f"{what}{factor!r}: the same levels in another order (brms {grp['levels']}, AutoBayes {list(block.levels)})")
        if grp["coef"] != coef:
            notes.append(f"{what}{factor!r}: brms names the coefficient {grp['coef']!r}, AutoBayes {coef!r}")
        for k in (f"N_{grp['id']}", f"M_{grp['id']}", f"NC_{grp['id']}"):
            reading.used.add(k)
        if block.by_levels:                                          # gr(g, by = gby): Nby_m, Jby_m
            Jby = np.asarray(reading.get(f"Jby_{grp['id']}", raw=True)).astype(int) - 1
            if int(reading.get(f"Nby_{grp['id']}", raw=True)) != len(block.by_levels) or \
                    [block.by_index[i] for i in m] != Jby.tolist():
                issues.append(f"{what}{factor!r}: brms's by-levels (Jby) differ from AutoBayes's")
    for grp in groups:
        if (grp["id"], grp["cn"]) not in used:
            issues.append(f"{what}brms group-level coefficient {grp['coef']!r} of {grp['group']!r} (id {grp['id']}) has no AutoBayes counterpart")
    for t in terms:
        brms_cor = {bool(g["cor"]) for g in groups if g["id"] in ids.get(t.factor, set())}
        M = sum(1 for g in groups if g["id"] in ids.get(t.factor, set()))
        if M > 1 and brms_cor == {True} and not (t.has_correlation or t.id is not None):
            issues.append(f"{what}{t.factor!r}: brms correlates the term's {M} coefficients; AutoBayes's {t.canonical!r} does not")
        if M > 1 and brms_cor == {False} and t.has_correlation:
            issues.append(f"{what}{t.factor!r}: AutoBayes correlates {t.canonical!r}; brms does not")


def _y_matches(Y, y) -> bool:
    Y, y = np.asarray(Y, dtype=float), np.asarray(y, dtype=float)
    if Y.shape != y.shape:
        return False
    miss = np.isnan(y)                                               # mi(): brms holds Inf at a missing response value
    return bool(np.all(np.isinf(Y[miss])) and _close(Y[~miss], y[~miss]))


def compare_univariate(model, reading: Reading, issues: list, notes: list, what: str = "") -> None:
    d, fam, spec = model.design, model.family_spec, model.spec
    if int(reading.get("N")) != d.n:
        issues.append(f"{what}brms N = {int(reading.get('N'))}, AutoBayes n = {d.n}")
    y = d.y
    if d.missing and spec.response in d.missing:
        y = d.missing[spec.response].values
    if not _y_matches(reading.get("Y"), y):
        issues.append(f"{what}the response differs from brms's Y")
    if getattr(fam, "per_category", False) or getattr(fam, "mixture", False):
        designs = sorted(k[2:] for k in reading.names if k.startswith("X_mu"))
        if not designs:
            issues.append(f"{what}brms has no per-category or per-component designs X_mu<k>")
        for dp in designs:
            match_design(reading.get(f"X_{dp}", raw=True), None, d.X_raw, d.columns, spec.intercept, f"{what}X_{dp}", issues)
            reading.used.update({f"K_{dp}", f"Kc_{dp}"})
        if getattr(fam, "per_category", False) and int(reading.get("ncat")) != len(model.categories):
            issues.append(f"{what}brms has {int(reading.get('ncat'))} categories, AutoBayes {len(model.categories)}")
        if getattr(fam, "mixture", False):
            reading.used.add("con_theta")
    else:
        thresholds = bool(getattr(fam, "thresholds", False))
        colnames = reading.sd.get("__colnames__", {}).get(reading.key("X"))
        if d.X_raw.shape[1] and reading.has("Kc") != bool(model.effective_center_predictors):   # brms's Kc: a centred design
            issues.append(f"{what}brms {'centres' if reading.has('Kc') else 'does not centre'} the population-level design; "
                          f"AutoBayes {'does' if model.effective_center_predictors else 'does not'}")
        match_design(reading.get("X"), colnames, d.X_raw, d.columns, spec.intercept and not thresholds, f"{what}X", issues)
        reading.used.update({reading.key("K"), reading.key("Kc")})
        if thresholds:
            grouped = getattr(d, "thresholds", None)
            if grouped is not None:                                  # F06b: thres(gr = z): ngrthres, nthres per level, Jthres
                if (int(reading.get("ngrthres")) != len(grouped.levels) or not _close(reading.get("nthres"), np.asarray(grouped.nthres))
                        or not _close(reading.get("Jthres"), grouped.jthres())):
                    issues.append(f"{what}brms's grouped thresholds (ngrthres, nthres, Jthres) differ from AutoBayes's {grouped.term}")
            elif int(reading.get("nthres")) != len(model.categories) - 1:
                issues.append(f"{what}brms has {int(reading.get('nthres'))} thresholds, AutoBayes {len(model.categories) - 1}")
            if reading.has("disc") and float(reading.get("disc")) != 1.0:
                issues.append(f"{what}brms's disc is {float(reading.get('disc'))}")
    for name, dd in d.dpars.items():                                # a distributional parameter's own formula
        key = f"X_{name}"
        if not reading.has(key):
            issues.append(f"{what}AutoBayes gives {name} its own formula; brms has no {key}")
            continue
        match_design(reading.get(key), None, dd.X_raw, dd.columns, spec.dpar(name).intercept, f"{what}{key}", issues)
        if dd.X_raw.shape[1] and reading.has(f"Kc_{name}") != bool(dd.centered):
            issues.append(f"{what}brms {'centres' if reading.has(f'Kc_{name}') else 'does not centre'} {key}; AutoBayes "
                          f"{'does' if dd.centered else 'does not'}")
        reading.used.update({reading.key(f"K_{name}"), reading.key(f"Kc_{name}")})
        if dd.group_blocks:
            match_group_scope(model, dd.group_blocks, (), reading, {"dpar": name}, issues, notes, what)
    match_group_scope(model, d.group_blocks, spec.group_terms, reading, {}, issues, notes, what)
    # addition terms and offsets
    for mine, key in ((d.trials, "trials"), (d.offset, "offsets"), (d.weights, "weights"), (d.cens, "cens"), (d.rcens, "rcens"),
                      (d.lb, "lb"), (d.ub, "ub"), (getattr(d, "dec", None), "dec"),     # F04: dec(col), the wiener decisions
                      (getattr(d, "se", None), "se"), (getattr(d, "denom", None), "denom")):   # F06a: se(col), rate(col)
        if mine is None:
            continue
        if not reading.has(key):
            issues.append(f"{what}AutoBayes has {key}; brms's Stan data has none")
        elif not _close(reading.get(key), mine):
            issues.append(f"{what}{key} differ from brms's")
    se = spec.addition("se") if hasattr(spec, "addition") else None   # F06a: se(col) without sigma = TRUE, brms's data sigma = 0
    if se is not None and not se.sigma:
        if not reading.has("sigma") or float(reading.get("sigma")) != 0.0 or "sigma" in model.names:
            issues.append(f"{what}brms fixes sigma at 0 in its Stan data under {se.term}; AutoBayes's model "
                          + ("learns sigma" if "sigma" in model.names else "does not match it"))
    for column, block in d.monotonic.items():                       # mo(): Xmo_k, Jmo
        ks = [k for k in range(1, 50) if reading.has(f"Xmo_{k}") and _close(reading.sd.get(reading.key(f"Xmo_{k}")), block.codes)]
        if len(ks) != 1:
            issues.append(f"{what}{block.term!r}: {len(ks)} brms monotonic terms carry its codes")
            continue
        reading.used.update({reading.key(f"Xmo_{ks[0]}"), reading.key(f"con_simo_{ks[0]}"), reading.key("Imo"), reading.key("Ksp")})
        if int(np.atleast_1d(reading.get("Jmo"))[ks[0] - 1]) != block.D:
            issues.append(f"{what}{block.term!r}: brms has {int(np.atleast_1d(reading.get('Jmo'))[ks[0] - 1])} steps, AutoBayes {block.D}")
    if d.category_specific:                                          # cs(): Xcs, Kcs
        if not reading.has("Xcs"):
            issues.append(f"{what}AutoBayes has cs() terms; brms has no Xcs")
        else:
            match_design(reading.get("Xcs"), None, np.column_stack(list(d.category_specific.values())),
                         list(d.category_specific), False, f"{what}Xcs", issues)
            reading.used.add(reading.key("Kcs"))
    for column, block in d.gp.items():                              # gp(): Jgp_k, Xgp_k, dmax_k, cmeans_k, ...
        g = spec.gp(column)
        ks = [k for k in range(1, 20) if reading.has(f"Jgp_{k}") and _close(reading.sd.get(reading.key(f"Jgp_{k}")), block.index + 1)]
        if len(ks) != 1:
            issues.append(f"{what}{block.term!r}: {len(ks)} brms Gaussian processes have its row index Jgp")
            continue
        k = ks[0]
        mine = {f"Xgp_prior_{k}": block.inputs[:, None], f"dmax_{k}": g.dmax, f"cmeans_{k}": g.cmean,
                f"Xgp_{k}": block.basis if g.approximate else block.inputs[:, None]}
        if g.approximate:
            mine.update({f"NBgp_{k}": g.k, f"Lgp_{k}": g.L, f"slambda_{k}": block.slambda[:, None]})
        for key, value in mine.items():
            if not reading.has(key) or not _close(reading.get(key), value):
                issues.append(f"{what}{block.term!r}: brms's {key} differs from AutoBayes's")
        reading.used.update({reading.key(f"Jgp_{k}"), reading.key(f"Kgp_{k}"), reading.key(f"Dgp_{k}"), reading.key(f"Nsubgp_{k}")})
        if int(reading.get(f"Kgp_{k}")) != 1 or int(reading.get(f"Dgp_{k}")) != 1:
            issues.append(f"{what}{block.term!r}: brms's GP has Kgp, Dgp {int(reading.get(f'Kgp_{k}'))}, {int(reading.get(f'Dgp_{k}'))}")
    for column, block in d.measurement_error.items():               # me(): Xn_k, noise_k
        ks = [k for k in range(1, 20) if reading.has(f"Xn_{k}") and _close(reading.sd.get(reading.key(f"Xn_{k}")), block.Xn)]
        if len(ks) != 1 or not _close(reading.get(f"noise_{ks[0]}"), block.noise):
            issues.append(f"{what}{block.term!r}: brms's Xn / noise differ from AutoBayes's")
            continue
        reading.used.update({reading.key(f"Xn_{ks[0]}"), reading.key(f"Mme_{ks[0]}"), reading.key(f"NCme_{ks[0]}"), reading.key("Ksp")})
    for t in spec.pop_terms:                                         # mi(col) terms: brms's special terms (Ksp)
        if t.kind == "missing":
            reading.used.add(reading.key("Ksp"))
    if spec.response in d.missing:                                   # a response with missing values: Jmi, Nmi
        idx = d.missing[spec.response].index
        if int(reading.get("Nmi")) != idx.size or not _close(reading.get("Jmi"), idx + 1):
            issues.append(f"{what}brms's missing rows (Jmi) differ from AutoBayes's")
        reading.used.update({reading.key("lbmi"), reading.key("ubmi")})
    if spec.autocor is not None:                                     # ar()/ma()/arma(): Kar, Kma, J_lag
        ac = d.autocor
        if int(reading.get("Kar")) != ac.p or int(reading.get("Kma")) != ac.q:
            issues.append(f"{what}brms has Kar, Kma = {int(reading.get('Kar'))}, {int(reading.get('Kma'))}; AutoBayes p, q = {ac.p}, {ac.q}")
        if not _close(reading.get("J_lag"), ac.J_lag) or not np.array_equal(ac.order, np.arange(ac.n)):
            issues.append(f"{what}brms's J_lag differs from AutoBayes's (the data are in brms's series and time order)")
    if getattr(spec, "timecor", None) is not None:                   # unstr()/cosy() (F03b): N_tg, begin_tg, end_tg, nobs_tg,
        tc = d.timecor                                               # and unstr()'s n_unique_t, n_unique_cortime, Jtime_tg
        mine = {"N_tg": len(tc.groups), "nobs_tg": tc.nobs, "begin_tg": tc.begin + 1, "end_tg": tc.begin + tc.nobs}
        if tc.call == "unstr":
            mine.update(n_unique_t=tc.dim, n_unique_cortime=tc.dim * (tc.dim - 1) // 2, Jtime_tg=tc.Jtime)
        for key, value in mine.items():
            if not reading.has(key) or not _close(reading.get(key), value):
                issues.append(f"{what}brms's {key} differs from AutoBayes's groups of {spec.timecor.term}")
        if not np.array_equal(tc.order, np.arange(tc.n)):
            issues.append(f"{what}the data are not in brms's group and time order")
    bhaz = getattr(d, "bhaz", None)                                  # F05: a cox model's Zbhaz, Zcbhaz, Kbhaz and con_sbhaz
    if bhaz is not None:
        from autobayes.priors import dirichlet_concentration
        con = dirichlet_concentration(model.priors.sbhaz, bhaz.K)
        for key, mine in (("Zbhaz", bhaz.Zbhaz), ("Zcbhaz", bhaz.Zcbhaz), ("Kbhaz", bhaz.K), ("con_sbhaz", con)):
            if not reading.has(key) or not _close(reading.get(key), mine):
                issues.append(f"{what}brms's {key} differs from AutoBayes's baseline hazard")
    q = getattr(model, "quantile", None)
    if q is not None:
        if not reading.has("quantile") or not _close(reading.get("quantile"), q):
            issues.append(f"{what}brms's fixed quantile differs from AutoBayes's {q}")


def match_group_scope(model, blocks, terms, reading, scope, issues, notes, what):
    match_groups(dict(blocks), terms, reading, scope, issues, notes, what)


def compare(model, entry_id: str, store: dict) -> tuple[list[str], list[str]]:
    try:
        return _compare(model, entry_id, store)
    except KeyError as exc:                                          # a key the model needs that brms's Stan data lacks
        return [f"brms's Stan data lacks what the AutoBayes model has: {exc.args[0]}"], []


def _compare(model, entry_id: str, store: dict) -> tuple[list[str], list[str]]:
    """``(issues, notes)``: the differences between the AutoBayes model's reading and brms's Stan data (empty: the same
    reading), and differences of names or level order only."""
    rec = store["entries"][entry_id]
    sd = {k: decode(v) for k, v in rec.get("arrays", {}).items()}
    sd["__colnames__"] = {k: v.get("colnames") for k, v in rec.get("arrays", {}).items()}
    names = rec.get("names", {})
    groups = rec.get("groups", [])
    used = {"prior_only"}
    issues, notes = [], []
    if getattr(model, "responses", None) and hasattr(model, "submodels"):          # a multivariate model
        used.update({"N", "nresp"})
        if "nrescor" in names:
            used.add("nrescor")
            if int(sd["nrescor"]) != (1 if model.rescor else 0) * len(model.responses) * (len(model.responses) - 1) // 2:
                issues.append(f"brms has nrescor = {int(sd['nrescor'])}; AutoBayes rescor = {model.rescor}")
        elif model.rescor:
            issues.append("AutoBayes correlates the residuals; brms's Stan data has no nrescor")
        for r, sub in model.submodels.items():
            compare_univariate(sub, Reading(sd, names, groups, used, resp=r), issues, notes, what=f"[{r}] ")
    elif getattr(getattr(model, "spec", None), "is_nonlinear", False):            # a non-linear model
        d = model.design
        reading = Reading(sd, names, groups, used)
        if int(reading.get("N")) != d.n or not _close(reading.get("Y"), d.y):
            issues.append("the response differs from brms's")
        cks = sorted(k for k in names if k.startswith("C_"))
        for name, col in d.covariates.items():
            hits = [k for k in cks if _close(sd.get(k), col)]
            if len(hits) != 1:
                issues.append(f"covariate {name!r}: {len(hits)} brms covariates C_k carry it")
            used.update(hits)
        for name, nd in d.nlpars.items():
            key = f"X_{name}"
            if key not in names:
                issues.append(f"AutoBayes's non-linear parameter {name!r} has no brms {key}")
                continue
            match_design(reading.get(key), None, nd.X[:, int(nd.intercept):], nd.columns[int(nd.intercept):], nd.intercept,
                         key, issues)
            used.add(f"K_{name}")
            match_groups(dict(nd.group_blocks), (), reading, {"nlpar": name}, issues, notes, what=f"[{name}] ")
    else:
        compare_univariate(model, Reading(sd, names, groups, used), issues, notes)
    leftover = [k for k in names if k not in used]
    if leftover:
        issues.append(f"brms's Stan data has {leftover}, which the AutoBayes model does not carry")
    issues += compare_family(model, rec.get("family") or {})
    return issues, notes


def compare_family(model, fam: dict) -> list[str]:
    """brms's family and links (``link``, and ``link_<dpar>`` of each distributional parameter with a formula) against the
    model's: the Stan data do not carry them, and a different default link is a different model on the same data."""
    if not fam:
        return []
    if "resp" in fam:                                                # a multivariate model: each response's family
        return [f"[{r}] {issue}" for r, sub in model.submodels.items() for issue in compare_family(sub, fam["resp"].get(r, {}))] \
            if hasattr(model, "submodels") else ["brms reads a multivariate model; AutoBayes's is univariate"]
    issues = []
    theirs = fam["family"].replace(".", "_")
    if theirs == "mixture":
        mine = list(getattr(model, "components", ()))
        if mine != list(fam.get("mix", [])):
            issues.append(f"brms's mixture components are {fam.get('mix')}; AutoBayes's {mine}")
        return issues
    subs = list(model.submodels.values()) if hasattr(model, "submodels") else [model]
    for sub in subs:
        if sub.family != theirs:
            issues.append(f"brms's family is {theirs!r}; AutoBayes's {sub.family!r}")
        link = sub.family_spec.link
        if fam["links"].get("link") != link:
            issues.append(f"brms's link is {fam['links'].get('link')!r}; AutoBayes's {link!r}")
        for name, dd in getattr(sub.design, "dpars", {}).items():
            if fam["links"].get(f"link_{name}") != dd.link:
                issues.append(f"brms's link of {name} is {fam['links'].get(f'link_{name}')!r}; AutoBayes's {dd.link!r}")
    return issues


# -- one entry, and the corpus ---------------------------------------------------------------------------------
OUTCOMES = ("match", "differs", "misread", "refused_named", "refused_unclear", "crash")


def run_entry(entry: dict, df: pd.DataFrame, store: dict) -> dict:
    """Build the entry's model and compare its reading with brms's (``compare``), or record the refusal."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model, _ = build(entry, df)
        except Exception as exc:                                     # noqa: BLE001 - every failure is recorded
            return {"status": "error", **error_record(exc)}
    out = {"status": "built", "model": type(model).__name__, "model_warnings": list(getattr(model, "warnings", []))}
    lib = sorted({str(w.message).splitlines()[0][:200] for w in caught if "autobayes" in str(getattr(w, "filename", ""))})
    if lib:
        out["warnings"] = lib
    if store["entries"][entry["id"]]["status"] != "ok":
        out["issues"], out["notes"] = ["brms refuses this formula; AutoBayes builds a model"], []
    else:
        out["issues"], out["notes"] = compare(model, entry["id"], store)
    return out


def classify(entry: dict, result: dict) -> str:
    """The entry's outcome (module docstring) from its result and its review: ``tokens`` (what a message naming the
    construct contains), ``unclear`` (why a refusal that quotes the construct still misnames it) and ``documented_by`` (the
    divergence row of a documented difference)."""
    if result["status"] == "built":
        if not result["issues"]:
            return "match"
        return "differs" if entry.get("documented_by") else "misread"
    if result["mechanism"] == "crash":
        return "crash"
    if entry.get("unclear"):
        return "refused_unclear"
    tokens = entry.get("tokens")
    if tokens is None:
        return "unreviewed"
    msg = result["message"].lower()
    return "refused_named" if any(t.lower() in msg for t in tokens) else "refused_unclear"


def gap(entry: dict, outcome: str) -> str | None:
    """``"defect"`` for behaviour that should change (a silent misreading, an unclear refusal, a crash), ``"capability"``
    for a brms construct AutoBayes refuses by name, ``"divergence"`` for a documented difference, None otherwise (the same
    reading as brms's, or a refusal of what brms refuses too)."""
    brms_ok = entry["brms_reading"]["status"] == "ok"
    if outcome in ("misread", "refused_unclear", "crash"):
        return "defect" if brms_ok or outcome != "refused_unclear" else None
    if outcome == "refused_named":
        return "capability" if brms_ok else None
    if outcome == "differs":
        return "divergence"
    return None


def brms_summary(rec: dict) -> dict:
    """The readable part of brms's reading: status, error, warnings, the Stan data's names, the design columns and the
    group-level terms."""
    out = {"status": rec["status"]}
    if rec["status"] != "ok":
        out["error"] = rec["error"]
    if rec.get("warnings"):
        out["warnings"] = rec["warnings"]
    if rec["status"] == "ok":
        out["stan_data"] = sorted(k for k in rec["names"] if k != "prior_only")
        out["design_columns"] = {k: v["colnames"] for k, v in rec.get("arrays", {}).items() if v.get("colnames")}
        out["group_terms"] = [f"{g['group']}:{g['coef']}" + (f" ({g['resp'] or g['dpar'] or g['nlpar']})" if (g["resp"] or g["dpar"] or g["nlpar"]) else "")
                              + (" cor" if g["cor"] else "") for g in rec.get("groups", [])]
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    corpus = load_json(CORPUS)
    store = load_json(STANDATA)
    df = python_frame()
    only = set(argv)
    for entry in corpus["entries"]:
        if only and entry["id"] not in only:
            continue
        entry["brms_reading"] = brms_summary(store["entries"][entry["id"]])
        result = run_entry(entry, df, store)
        result["outcome"] = classify(entry, result)
        result["gap"] = gap(entry, result["outcome"])
        if result["outcome"] == "refused_unclear":
            result["pin"] = result["message"][:80]
        entry["result"] = result
        print(f"{entry['id']:45s} {result['outcome']:16s} {(result.get('message') or '; '.join(result.get('issues', [])))[:150]}")
    if not only:
        from collections import Counter
        corpus["summary"] = {"outcomes": dict(Counter(e["result"]["outcome"] for e in corpus["entries"])),
                             "gaps": dict(Counter(e["result"]["gap"] or "none" for e in corpus["entries"])),
                             "brms_refuses": sum(e["brms_reading"]["status"] != "ok" for e in corpus["entries"])}
        CORPUS.write_text(json.dumps(corpus, indent=1, ensure_ascii=False) + "\n")
        print(corpus["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
