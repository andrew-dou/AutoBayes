"""Put a brms fixture and an AutoBayes model in correspondence, by name and by array content, never by position.

brms's design columns are matched to AutoBayes's by content (`match_columns`), its group-level terms by the
partition of the rows they induce (`match_groups`), and its unconstrained coordinates by the probed
(parameter, element, transform) table in the fixture (`coordinate_map`). What cannot be matched raises
`OracleMismatch`, which is also what the gates raise, so a test fails rather than compares the wrong things.

Conventions this relies on, each checked where it is used: brms's `b` is the design without the intercept
column exactly when brms carries the separate (centred) `Intercept` parameter; `z_m` is an (M_m, N_m) array
of coefficient-by-level standardised effects; `r_m_c` the corresponding effects on the natural scale.
"""
from __future__ import annotations

from dataclasses import dataclass

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

import autobayes as ab
from tests.oracle.fixture import Fixture, Point
from tests.oracle.tolerances import DESIGN_ABS


class OracleMismatch(AssertionError):
    """The AutoBayes model and the brms fixture disagree, or cannot be put in correspondence."""


def coefficients(pars: dict) -> np.ndarray:
    """brms's ``b``; rstan omits a zero-length parameter from constrain_pars(), so an intercept-only model has none."""
    return np.atleast_1d(pars["b"]) if "b" in pars else np.zeros(0)


# -- the AutoBayes side of a case ---------------------------------------------------------------------------
def _prior(v):
    if v is None:
        return None
    if isinstance(v, dict):
        return {k: _prior(x) for k, x in v.items()}
    name, *args = v
    return getattr(ab, name)(*args)


def priors_from_spec(spec: dict) -> ab.Priors:
    """``{"intercept": ["Normal", 0, 5], "b": ["Normal", 0, 2] | {term: ...}, "sd": [...] | {factor: ...}, "sigma": [...]}``."""
    return ab.Priors(intercept=_prior(spec.get("intercept")), b=_prior(spec.get("b")),
                     sd=_prior(spec.get("sd")), sigma=_prior(spec.get("sigma")))


def build_model(case: dict, data: pd.DataFrame) -> ab.Model:
    spec = case["autobayes"]
    constructor = getattr(ab, spec["constructor"])
    return constructor(spec["formula"], data, priors_from_spec(spec["priors"]), **spec.get("kwargs", {}))


# -- design columns -----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ColumnMap:
    """brms X column of each AutoBayes design column (in ``model.spec.columns`` order), brms's intercept
    column if any, and whether brms carries the separate centred ``Intercept`` parameter."""
    x_column: tuple[int, ...]
    intercept_column: int | None
    centred_intercept: bool

    def b_index(self, x_col: int) -> int:
        """Index into brms's ``b`` of design column ``x_col``."""
        if self.centred_intercept:
            if x_col == self.intercept_column:
                raise OracleMismatch("the intercept column has no entry in b when brms centres")
            return x_col - (1 if x_col > self.intercept_column else 0)
        return x_col

    def x_of_b(self, b_idx: int) -> int:
        if self.centred_intercept:
            return b_idx + (1 if b_idx >= self.intercept_column else 0)
        return b_idx


def match_columns(fixture: Fixture, model: ab.Model) -> ColumnMap:
    X = fixture.standata["X"]
    names = fixture.colnames("X") or [str(k) for k in range(X.shape[1])]
    XA = model.design.X_raw
    if X.shape[0] != XA.shape[0]:
        raise OracleMismatch(f"brms has {X.shape[0]} rows, AutoBayes {XA.shape[0]}")
    taken, cols = set(), []
    for c, label in enumerate(model.spec.columns):
        hits = [k for k in range(X.shape[1]) if k not in taken and np.max(np.abs(X[:, k] - XA[:, c])) <= DESIGN_ABS]
        if len(hits) != 1:
            raise OracleMismatch(f"design column {label!r} matches {len(hits)} brms columns {[names[k] for k in hits]} within {DESIGN_ABS}")
        cols.append(hits[0])
        taken.add(hits[0])
    rest = [k for k in range(X.shape[1]) if k not in taken]
    intercept_col = None
    if model.spec.intercept:
        ones = [k for k in rest if np.max(np.abs(X[:, k] - 1.0)) <= DESIGN_ABS]
        if len(ones) != 1:
            raise OracleMismatch(f"expected one brms intercept column, found {[names[k] for k in ones]}")
        intercept_col = ones[0]
        rest.remove(intercept_col)
    if rest:
        raise OracleMismatch(f"brms design columns without an AutoBayes counterpart: {[names[k] for k in rest]}")
    pars = fixture.points[0].pars
    centred = "Intercept" in pars
    if centred and intercept_col is None:
        raise OracleMismatch("brms has a centred Intercept parameter but AutoBayes has no population intercept")
    expected_b = X.shape[1] - (1 if centred else 0)
    if coefficients(pars).size != expected_b:
        raise OracleMismatch(f"brms b has {coefficients(pars).size} entries; expected {expected_b} from {X.shape[1]} design columns")
    if "Kc" in fixture.standata and int(fixture.standata["Kc"]) != expected_b:
        raise OracleMismatch(f"brms Kc = {int(fixture.standata['Kc'])} but b has {expected_b} entries")
    return ColumnMap(tuple(cols), intercept_col, centred)


# -- group-level terms --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GroupMap:
    brms_id: dict[str, int]              # AutoBayes factor -> brms group id m (of sd_m, z_m, J_m)
    cn: dict[str, int]                   # 1-based coefficient number of the varying intercept in that group
    level_of: dict[str, np.ndarray]      # level_of[g][j] = AutoBayes level index of brms level j (0-based)


def match_groups(fixture: Fixture, model: ab.Model) -> GroupMap:
    ids, cns, levels_of, used = {}, {}, {}, set()
    for g, block in model.design.group_designs.items():
        gidx = block.group_index
        found = []
        for grp in fixture.groups:
            if grp["coef"] != "Intercept" or grp["id"] in used:
                continue
            J = fixture.standata[f"J_{grp['id']}"].astype(int) - 1
            if J.shape != gidx.shape or int(J.max()) + 1 != block.J:
                continue
            m = np.full(block.J, -1)
            for j in range(block.J):
                vals = np.unique(gidx[J == j])
                if vals.size != 1:
                    break
                m[j] = vals[0]
            else:
                if len(np.unique(m)) == block.J:
                    found.append((grp, m))
        if len(found) != 1:
            raise OracleMismatch(f"grouping factor {g!r}: {len(found)} brms group terms induce its partition of the rows")
        grp, m = found[0]
        if [block.levels[i] for i in m] != list(grp["levels"]):
            raise OracleMismatch(f"{g!r}: brms levels {grp['levels']} vs AutoBayes {[block.levels[i] for i in m]}")
        Z = fixture.standata[f"Z_{grp['id']}_{grp['cn']}"]
        if Z.shape != (block.Z.shape[0],) or np.max(np.abs(Z - 1.0)) > DESIGN_ABS:
            raise OracleMismatch(f"{g!r}: brms Z_{grp['id']}_{grp['cn']} is not the varying intercept's column of ones")
        used.add(grp["id"])
        ids[g], cns[g], levels_of[g] = int(grp["id"]), int(grp["cn"]), m
    unmatched = [(grp["group"], grp["coef"]) for grp in fixture.groups if grp["id"] not in used]
    if unmatched:
        raise OracleMismatch(f"brms group-level terms without an AutoBayes counterpart: {unmatched}")
    return GroupMap(ids, cns, levels_of)


# -- points -------------------------------------------------------------------------------------------------
def natural_point(model: ab.Model, point: Point, cmap: ColumnMap, gmap: GroupMap) -> dict:
    """AutoBayes's natural-scale arguments (intercept, b, sd, z, sigma) at a brms point."""
    pars = point.pars
    bb = coefficients(pars)
    b = np.array([bb[cmap.b_index(k)] for k in cmap.x_column], dtype=float)
    intercept = None
    if model.spec.intercept:
        intercept = float(pars["Intercept"]) if cmap.centred_intercept else float(bb[cmap.b_index(cmap.intercept_column)])
    sd, z = {}, {}
    for g in model.spec.groups:
        m, cn = gmap.brms_id[g], gmap.cn[g]
        sd[g] = float(np.atleast_1d(pars[f"sd_{m}"])[cn - 1])
        zb = np.atleast_2d(pars[f"z_{m}"])                     # (M_m, N_m), brms level order
        za = np.empty(zb.shape[1])
        za[gmap.level_of[g]] = zb[cn - 1]
        z[g] = za
    sigma = float(pars["sigma"]) if "sigma" in pars else None
    return dict(intercept=intercept, b=b, sd=sd or None, z=z or None, sigma=sigma)


def value_point(model: ab.Model, point: Point, cmap: ColumnMap, gmap: GroupMap) -> dict:
    return model.value_point(**natural_point(model, point, cmap, gmap))


def value_layout(model: ab.Model, point: dict) -> dict[str, slice]:
    """Where each value variable sits in PyMC's flattened gradient (value_vars order, each raveled)."""
    out, i = {}, 0
    for v in model.pymc_model.value_vars:
        size = int(np.asarray(point[v.name]).size)
        out[v.name] = slice(i, i + size)
        i += size
    return out


def coordinate_map(fixture: Fixture, model: ab.Model, cmap: ColumnMap, gmap: GroupMap) -> list[tuple[int, str, int]]:
    """For each brms unconstrained coordinate: (its 0-based index, the AutoBayes value variable, the flat
    index within that variable), with the transform checked on both sides."""
    names = model.names
    by_id = {m: g for g, m in gmap.brms_id.items()}
    pars = fixture.points[0].pars
    out = []
    for e in fixture.upars:
        par, elem, transform, idx = e["par"], int(e["elem"]) - 1, e["transform"], int(e["index"]) - 1
        if par == "b":
            k = cmap.x_of_b(elem)
            if cmap.intercept_column is not None and k == cmap.intercept_column:
                target, expect = (names["intercept"], 0), "identity"
            else:
                _, term, j = model.coefficients[cmap.x_column.index(k)]
                target, expect = (term, 0 if j is None else j), "identity"
        elif par == "Intercept":
            target, expect = (names["intercept"], 0), "identity"
        elif par == "sigma":
            target, expect = (f"{names['sigma']}_log__", 0), "log"
        elif par.startswith("sd_") and int(par[3:]) in by_id:
            g = by_id[int(par[3:])]
            if elem != gmap.cn[g] - 1:
                raise OracleMismatch(f"{par}[{elem + 1}] is not the SD of the varying intercept of {g!r}")
            target, expect = (f"{model.group_names[g]['sd']}_log__", 0), "log"
        elif par.startswith("z_") and int(par[2:]) in by_id:
            g = by_id[int(par[2:])]
            M = pars[par].shape[0]
            c, j = elem % M, elem // M                          # column-major flat index of (M, N)
            if c != gmap.cn[g] - 1:
                raise OracleMismatch(f"{par} coefficient {c + 1} is not the varying intercept of {g!r}")
            target, expect = (model.group_names[g]["offset"], int(gmap.level_of[g][j])), "identity"
        else:
            raise OracleMismatch(f"no adapter for brms parameter {par!r} ({fixture.declarations.get(par, 'undeclared')})")
        if transform != expect:
            raise OracleMismatch(f"brms transform of {par}[{elem + 1}] is {transform!r}; AutoBayes uses {expect!r}")
        out.append((idx, *target))
    return out


# -- draws --------------------------------------------------------------------------------------------------
def draws(fixture: Fixture, model: ab.Model, cmap: ColumnMap, gmap: GroupMap) -> az.InferenceData:
    """The fixture's points as one chain of posterior draws under AutoBayes's names: the raw-scale intercept
    (brms ``b_Intercept``), coefficients by term, group SDs and effects by level, sigma."""
    pts = fixture.points
    names = model.names
    data, coords = {}, {"chain": [0], "draw": np.arange(len(pts))}
    if model.spec.intercept:
        a = [float(p.pars["b_Intercept"]) if cmap.centred_intercept else float(coefficients(p.pars)[cmap.b_index(cmap.intercept_column)]) for p in pts]
        data[names["intercept"]] = (("chain", "draw"), np.array(a)[None, :])
    for term in model.spec.terms:
        positions = [(c, j) for c, (_, t, j) in enumerate(model.coefficients) if t == term]
        if model.spec.is_vector_term(term):
            vals = np.empty((len(pts), len(positions)))
            for d, p in enumerate(pts):
                for c, j in positions:
                    vals[d, j] = coefficients(p.pars)[cmap.b_index(cmap.x_column[c])]
            data[term] = (("chain", "draw", f"{term}_dim"), vals[None])
            coords[f"{term}_dim"] = list(model.spec.levels_of(term))
        else:
            (c, _), = positions
            data[term] = (("chain", "draw"), np.array([coefficients(p.pars)[cmap.b_index(cmap.x_column[c])] for p in pts])[None, :])
    for g, roles in model.group_names.items():
        m, cn = gmap.brms_id[g], gmap.cn[g]
        sd = np.array([np.atleast_1d(p.pars[f"sd_{m}"])[cn - 1] for p in pts])
        r = np.empty((len(pts), len(gmap.level_of[g])))
        for d, p in enumerate(pts):
            r[d, gmap.level_of[g]] = np.atleast_1d(p.pars[f"r_{m}_{cn}"])
        data[roles["sd"]] = (("chain", "draw"), sd[None, :])
        data[roles["group_effect"]] = (("chain", "draw", roles["coordinate"]), r[None])
        coords[roles["coordinate"]] = list(model.design.group_designs[g].levels)
    if "sigma" in names and "sigma" in pts[0].pars:
        data[names["sigma"]] = (("chain", "draw"), np.array([float(p.pars["sigma"]) for p in pts])[None, :])
    return az.InferenceData(posterior=xr.Dataset(data, coords=coords))
