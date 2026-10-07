"""Put a brms fixture and an AutoBayes model in correspondence, by name and by array content, never by position.

brms's design columns are matched to AutoBayes's by content (`match_columns`), its group-level terms by the
partition of the rows they induce (`match_groups`), and its unconstrained coordinates by the probed
(parameter, element, transform) table in the fixture (`coordinate_map`). What cannot be matched raises
`OracleMismatch`, which is also what the gates raise, so a test fails rather than compares the wrong things.

Conventions this relies on, each checked where it is used: brms's `b` is the design without the intercept
column exactly when brms carries the separate (centred) `Intercept` parameter; `z_m` is an (M_m, N_m) array
of coefficient-by-level standardised effects; `r_m_c` the corresponding effects on the natural scale. A
group-level coefficient `c` of brms group `m` (the varying intercept, or a varying slope of `(1 + x || g)`) is
matched to the AutoBayes block `(factor, coef)` by the partition of the rows `J_m` induces and by the content
of `Z_m_c`, which is the coefficient's covariate column (ones for the intercept, `x` for a slope). A correlated
group (`cor = TRUE`, a declared `cholesky_factor_corr` `L_m`; `(1 + x | g)`, A02) must meet a correlated AutoBayes
term and vice versa; its `L_m` is AutoBayes's `L` (in the same coefficient order, or the oracle stops), and its
unconstrained coordinate(s) are Stan's `cholesky_corr` coordinates, which `coordinate_map` recomputes from `L_m`
and checks against the fixture's `u` before mapping them to `L|g_cholesky_corr__`.
"""
from __future__ import annotations

from dataclasses import dataclass

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

import autobayes as ab
from autobayes.correlation import cholesky_corr_free
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
    """``{"intercept": ["Normal", 0, 5], "b": ["Normal", 0, 2] | {term: ...}, "sd": [...] | {factor: ...}, "cor": {factor: ["LKJ", eta]},
    "sigma": [...], "shape": [...], "phi": [...]}``."""
    return ab.Priors(intercept=_prior(spec.get("intercept")), b=_prior(spec.get("b")),
                     sd=_prior(spec.get("sd")), cor=_prior(spec.get("cor")), sigma=_prior(spec.get("sigma")),
                     shape=_prior(spec.get("shape")), phi=_prior(spec.get("phi")))


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
    coef_cn: dict[tuple[str, str], int]  # (factor, coefficient) -> 1-based coefficient number c of sd_m[c], z_m[c], r_m_c


def match_groups(fixture: Fixture, model: ab.Model) -> GroupMap:
    """Match every AutoBayes group block ``(factor, coef)`` to one brms group coefficient ``(id, cn)``: the one
    whose ``J_id`` induces the block's partition of the rows and whose ``Z_id_cn`` equals the block's covariate
    column (ones for the varying intercept, the slope's column otherwise); then check the coefficient's brms
    label, the level order, that every brms coefficient was used, and that each group is correlated on both
    sides or on neither."""
    ids, cns, levels_of, coef_cn, used = {}, {}, {}, {}, set()
    n = model.design.n
    for (g, coef), block in model.design.group_blocks.items():
        gidx = block.group_index
        covariate = block.Z[np.arange(n), gidx]          # the coefficient's covariate per row (brms Z_m_c)
        found = []
        for grp in fixture.groups:
            key = (int(grp["id"]), int(grp["cn"]))
            if key in used:
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
                if len(np.unique(m)) != block.J:
                    continue
                Z = fixture.standata[f"Z_{grp['id']}_{grp['cn']}"]
                if Z.shape == (n,) and np.max(np.abs(Z - covariate)) <= DESIGN_ABS:
                    found.append((grp, m))
        if len(found) != 1:
            raise OracleMismatch(
                f"group-level coefficient {coef!r} of {g!r}: {len(found)} brms group terms induce its partition of "
                f"the rows and carry its covariate column within {DESIGN_ABS}")
        grp, m = found[0]
        if grp["coef"] != coef:
            raise OracleMismatch(f"{g!r}: the brms coefficient matched by content is labelled {grp['coef']!r}, AutoBayes has {coef!r}")
        if [block.levels[i] for i in m] != list(grp["levels"]):
            raise OracleMismatch(f"{g!r}: brms levels {grp['levels']} vs AutoBayes {[block.levels[i] for i in m]}")
        if g in ids and ids[g] != int(grp["id"]):
            raise OracleMismatch(f"{g!r}: its coefficients match different brms groups ({ids[g]} and {grp['id']})")
        used.add((int(grp["id"]), int(grp["cn"])))
        ids[g], levels_of[g], coef_cn[(g, coef)] = int(grp["id"]), m, int(grp["cn"])
        if coef == "Intercept":
            cns[g] = int(grp["cn"])
    unmatched = [(grp["group"], grp["coef"]) for grp in fixture.groups if (int(grp["id"]), int(grp["cn"])) not in used]
    if unmatched:
        raise OracleMismatch(f"brms group-level terms without an AutoBayes counterpart: {unmatched}")
    for t in model.spec.group_terms:
        m = ids[t.factor]
        M = int(fixture.standata[f"M_{m}"])
        if M != len(t.coefs):
            raise OracleMismatch(f"{t.factor!r}: brms group {m} has M_{m} = {M} coefficients, AutoBayes {len(t.coefs)}")
        # brms's standata carries NC_m = M_m (M_m - 1) / 2 whether or not the effects are correlated; what says
        # they are is the term's `cor` flag and a declared Cholesky factor L_m (A02): both must agree with the
        # AutoBayes term, which is correlated exactly when it has two or more coefficients written with '|'.
        correlated = any(bool(grp["cor"]) for grp in fixture.groups if int(grp["id"]) == m)
        declared = f"L_{m}" in fixture.declarations
        if M > 1 and correlated != declared:
            raise OracleMismatch(f"{t.factor!r}: brms group {m} has cor = {correlated} but L_{m} is {'' if declared else 'not '}declared")
        if M > 1 and (correlated or declared) != t.has_correlation:
            raise OracleMismatch(
                f"{t.factor!r}: brms group {m} has {M} coefficients with cor = {correlated}"
                f"{' and a correlation matrix L_' + str(m) if declared else ''}; AutoBayes's term '{t.canonical}' is "
                f"{'correlated' if t.has_correlation else 'independent'}")
    return GroupMap(ids, cns, levels_of, coef_cn)


def correlation_order(model: ab.Model, gmap: GroupMap, factor: str) -> list[int]:
    """brms's 0-based coefficient numbers in the AutoBayes coefficient order of a correlated factor: the permutation
    that takes brms's ``L_m`` / ``Cor_m`` rows and columns to AutoBayes's ``L`` / ``cor``."""
    return [gmap.coef_cn[(factor, c)] - 1 for c in model.spec.group_term_of(factor).coefs]


def brms_cholesky(model: ab.Model, gmap: GroupMap, factor: str, pars: dict) -> np.ndarray:
    """brms's ``L_m`` of a correlated factor in AutoBayes's coefficient order: ``L_m`` itself when the orders agree,
    otherwise the Cholesky factor of the permuted ``Cor_m`` (a permuted Cholesky factor is not one)."""
    m = gmap.brms_id[factor]
    L = np.atleast_2d(pars[f"L_{m}"])
    perm = correlation_order(model, gmap, factor)
    if perm == list(range(len(perm))):
        return L
    return np.linalg.cholesky((L @ L.T)[np.ix_(perm, perm)])


# -- points -------------------------------------------------------------------------------------------------
def natural_point(model: ab.Model, point: Point, cmap: ColumnMap, gmap: GroupMap) -> dict:
    """AutoBayes's natural-scale arguments (intercept, b, sd, z, L, sigma, shape, phi) at a brms point. An intercept-only
    factor's ``sd`` and ``z`` are the scalar and the vector; a factor with varying slopes takes the
    coefficient-keyed mappings ``{"Intercept": ..., "x": ...}`` (``Model.value_point``); a correlated factor's ``L`` is
    brms's ``L_m`` (A02). ``sigma``, ``shape`` and
    ``phi`` are brms's parameters of those names (the Gaussian/Student-t/lognormal scale; the negative binomial/Gamma
    shape; the Beta precision)."""
    pars = point.pars
    bb = coefficients(pars)
    b = np.array([bb[cmap.b_index(k)] for k in cmap.x_column], dtype=float)
    intercept = None
    if model.spec.intercept:
        intercept = float(pars["Intercept"]) if cmap.centred_intercept else float(bb[cmap.b_index(cmap.intercept_column)])
    sd, z = {}, {}
    for t in model.spec.group_terms:
        g = t.factor
        m = gmap.brms_id[g]
        sd_m, zb = np.atleast_1d(pars[f"sd_{m}"]), np.atleast_2d(pars[f"z_{m}"])    # z_m is (M_m, N_m), brms level order
        per_sd, per_z = {}, {}
        for coef in t.coefs:
            cn = gmap.coef_cn[(g, coef)]
            za = np.empty(zb.shape[1])
            za[gmap.level_of[g]] = zb[cn - 1]
            per_sd[coef], per_z[coef] = float(sd_m[cn - 1]), za
        if t.slopes:
            sd[g], z[g] = per_sd, per_z
        else:
            sd[g], z[g] = per_sd["Intercept"], per_z["Intercept"]
    L = {t.factor: brms_cholesky(model, gmap, t.factor, pars) for t in model.spec.correlated_terms}
    sigma = float(pars["sigma"]) if "sigma" in pars else None
    shape = float(pars["shape"]) if "shape" in pars else None
    phi = float(pars["phi"]) if "phi" in pars else None
    return dict(intercept=intercept, b=b, sd=sd or None, z=z or None, L=L or None, sigma=sigma, shape=shape, phi=phi)


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
    by_cn = {(g, cn): coef for (g, coef), cn in gmap.coef_cn.items()}     # (factor, brms coefficient number) -> coefficient
    roles = model.group_coefficient_names
    pars = fixture.points[0].pars
    seen_L: dict[str, int] = {}                                            # L_m coordinates met so far, in index order
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
        elif par == "shape":                                   # negative binomial / Gamma: PyMC's alpha, sampled on the log scale
            target, expect = (f"{names['shape']}_log__", 0), "log"
        elif par == "phi":                                     # Beta precision, sampled on the log scale
            target, expect = (f"{names['phi']}_log__", 0), "log"
        elif par.startswith("sd_") and int(par[3:]) in by_id:
            g = by_id[int(par[3:])]
            if (g, elem + 1) not in by_cn:
                raise OracleMismatch(f"{par}[{elem + 1}] is not the SD of a group-level coefficient of {g!r}")
            target, expect = (f"{roles[(g, by_cn[(g, elem + 1)])]['sd']}_log__", 0), "log"
        elif par.startswith("z_") and int(par[2:]) in by_id:
            g = by_id[int(par[2:])]
            M = pars[par].shape[0]
            c, j = elem % M, elem // M                          # column-major flat index of (M, N)
            if (g, c + 1) not in by_cn:
                raise OracleMismatch(f"{par} coefficient {c + 1} is not a group-level coefficient of {g!r}")
            target, expect = (roles[(g, by_cn[(g, c + 1)])]["offset"], int(gmap.level_of[g][j])), "identity"
        elif par.startswith("L_") and int(par[2:]) in by_id:     # the Cholesky factor of a correlated factor (A02)
            g = by_id[int(par[2:])]
            if g not in model.correlation_names:
                raise OracleMismatch(f"{par} is brms's correlation Cholesky factor of {g!r}, whose AutoBayes term is not correlated")
            perm = correlation_order(model, gmap, g)
            if perm != list(range(len(perm))):
                raise OracleMismatch(
                    f"{g!r}: brms orders the coefficients as {[by_cn[(g, c + 1)] for c in range(len(perm))]}, AutoBayes as "
                    f"{list(model.spec.group_term_of(g).coefs)}; the cholesky_corr coordinates of {par} are not comparable")
            k = seen_L.get(g, 0)
            seen_L[g] = k + 1
            # brms samples L_m on Stan's cholesky_corr coordinates (the probing saw a non-elementwise change, 'other'):
            # recompute coordinate k from L_m at every point and require it to be the fixture's u, so that the
            # AutoBayes transform is Stan's transform coordinate by coordinate before the gradient is compared.
            for q, pt_ in enumerate(fixture.points):
                u = cholesky_corr_free(np.atleast_2d(pt_.pars[par]))
                if k >= u.size or abs(u[k] - pt_.u[idx]) > 1e-10:
                    raise OracleMismatch(
                        f"point {q}: Stan's cholesky_corr coordinate {k} of {par} is {u[k] if k < u.size else 'absent'!r}; "
                        f"brms's unconstrained value is {pt_.u[idx]!r}")
            target, expect = (f"{model.correlation_names[g]['L']}_cholesky_corr__", k), "other"
        else:
            raise OracleMismatch(f"no adapter for brms parameter {par!r} ({fixture.declarations.get(par, 'undeclared')})")
        if transform != expect:
            raise OracleMismatch(f"brms transform of {par}[{elem + 1}] is {transform!r}; AutoBayes uses {expect!r}")
        out.append((idx, *target))
    return out


# -- draws --------------------------------------------------------------------------------------------------
def draws(fixture: Fixture, model: ab.Model, cmap: ColumnMap, gmap: GroupMap) -> az.InferenceData:
    """The fixture's points as one chain of posterior draws under AutoBayes's names: the raw-scale intercept
    (brms ``b_Intercept``), coefficients by term, group SDs and effects by level for every group-level
    coefficient (``1|g`` from ``r_m_1``, ``x|g`` from the slope's ``r_m_c``), the Cholesky factor and correlation
    matrix of a correlated factor (``L|g`` from ``L_m``, ``cor|g`` from ``L_m L_m^T``), sigma, shape or phi."""
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
    for (g, coef), roles in model.group_coefficient_names.items():
        m, cn = gmap.brms_id[g], gmap.coef_cn[(g, coef)]
        sd = np.array([np.atleast_1d(p.pars[f"sd_{m}"])[cn - 1] for p in pts])
        r = np.empty((len(pts), len(gmap.level_of[g])))
        for d, p in enumerate(pts):
            r[d, gmap.level_of[g]] = np.atleast_1d(p.pars[f"r_{m}_{cn}"])
        data[roles["sd"]] = (("chain", "draw"), sd[None, :])
        data[roles["group_effect"]] = (("chain", "draw", roles["coordinate"]), r[None])
        coords[roles["coordinate"]] = list(model.design.group_blocks[(g, coef)].levels)
    for g, roles in model.correlation_names.items():             # L and the correlation matrix of a correlated factor (A02)
        L = np.stack([brms_cholesky(model, gmap, g, p.pars) for p in pts])
        dims = ("chain", "draw", roles["coordinate"], roles["coordinate_T"])
        data[roles["L"]] = (dims, L[None])
        data[roles["cor"]] = (dims, (L @ np.swapaxes(L, -1, -2))[None])
        coords[roles["coordinate"]] = coords[roles["coordinate_T"]] = list(model.spec.group_term_of(g).coefs)
    if "sigma" in names and "sigma" in pts[0].pars:
        data[names["sigma"]] = (("chain", "draw"), np.array([float(p.pars["sigma"]) for p in pts])[None, :])
    if "shape" in names and "shape" in pts[0].pars:
        data[names["shape"]] = (("chain", "draw"), np.array([float(p.pars["shape"]) for p in pts])[None, :])
    if "phi" in names and "phi" in pts[0].pars:
        data[names["phi"]] = (("chain", "draw"), np.array([float(p.pars["phi"]) for p in pts])[None, :])
    return az.InferenceData(posterior=xr.Dataset(data, coords=coords))
