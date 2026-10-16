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

The ordinal and categorical families (B05): brms's cumulative model has no intercept column in `X` and its `Intercept`
is the `ordered[nthres]` thresholds, matched to AutoBayes's thresholds on Stan's `ordered` coordinates (recomputed from
`Intercept` at every point and checked against the fixture's `u`, as for `cholesky_corr`); brms's categorical model has
one design `X_mu<k>` and parameters `b_mu<k>`, `Intercept_mu<k>` per non-reference category, matched to the trailing
category axis of AutoBayes's coefficients by brms's name of each category's linear predictor (`brms_dpar_names`).

A distributional parameter's own formula (B01, brms's `bf(y ~ ..., sigma ~ z)`) has its own design `X_sigma`, matched by
content to the model's `Design.dpars["sigma"]` as `X` is (`match_distributional_columns`; its map rides on the response
formula's `ColumnMap.distributional`, which is not B05's `ColumnMap.dpars`: brms calls both kinds of linear predictor
dpars, but a categorical model's `mu<k>` share the response formula's design while `sigma`'s formula has its own), and
its own `Intercept_sigma` (centred when brms centres) and `b_sigma`, mapped to the formula's PyMC variables
(`Model.dpar_names`); brms reports the raw-scale intercept as `b_sigma_Intercept`.

A monotonic term `mo(x)` (B03) is brms's special term `k`: its codes `Xmo_k` and number of steps `Jmo[k]` are matched by
content to the model's `Design.monotonic` (`match_monotonic`, on `ColumnMap.monotonic`), its coefficient is `bsp[k]` and its
simplex `simo_k`. The probe cannot move one entry of a simplex alone (`unreached`): those coordinates are given to the declared
simplexes in declaration order and Stan's stick-breaking coordinates are recomputed from `simo_k` at every point and checked
against the fixture's `u` before they are mapped to `simo_x_stickbreaking__`, as for `cholesky_corr`.

A finite mixture (C05, brms's `mixture(gaussian, gaussian)`) has one design `X_mu<k>`, `b_mu<k>` and `Intercept_mu<k>` per
component, matched like the categorical model's per-category designs (`ColumnMap.dpars` holds brms's `mu1`, ..., with
`ColumnMap.mixture`); its intercepts are brms's `ordered[K] ordered_Intercept`, whose Stan `ordered` coordinates are
recomputed at every point and checked against the fixture's `u` (as for the cumulative thresholds) before they are mapped to
`Intercept_ordered__`; each `sigma<k>` is the PyMC variable of that name on the log scale; and `theta`, brms's `simplex[K]`,
takes the unreached coordinates as a monotonic term's simplex does, its stick-breaking coordinates recomputed and checked
the same way before they are mapped to `theta_stickbreaking__`.

A multivariate model (C01, brms's `mvbind(y, y2) ~ ...`) is put in correspondence response by response: `ResponseView` shows
the fixture as brms's univariate model of one response (`X_y` as `X`, `b_y` as `b`, `r_1_y_1` as `r_1_1`, its group terms by
`groups[*].resp`), which the matching above applies to with that response's own model (`MultivariateModel.submodels`); the
names it maps to are the response's in the multivariate graph (`y_Intercept`). The residual correlation `Lrescor` is mapped
as `L_m` is: Stan's `cholesky_corr` coordinates recomputed from `Lrescor` at every point and checked against `u`.

A Gaussian process `gp(x)` (C03) is brms's GP `k`: its row index `Jgp_k` and its `Xgp_k` (the distinct scaled inputs of an
exact GP, the basis at them of an approximate one, `NBgp_k` basis functions) are matched by content to the model's
`Design.gp` (`match_gp`, on `ColumnMap.gp`), which fixes the order of `zgp_k`; `sdgp_k` and `lscale_k` (one each: one
covariate, no `by`) map to `sdgp_x_log__` and `lscale_x_log__`, `zgp_k` to `zgp_x`.
A non-linear model (C02, brms's `nl = TRUE`) has no design `X`: its covariates `C_k` are matched by content to the model's
covariates, each non-linear parameter's never-centred `X_<nlpar>` column by column to its design, its `b_<nlpar>` to the
parameter's coefficient variables, and its group-level terms, which brms numbers as groups of their own and labels with the
parameter (`Z_<id>_<nlpar>_<k>`, `r_<id>_<nlpar>_<k>`; the `nlpar` column of the group table), within the parameter
(`match_nonlinear_columns`, `match_nonlinear_groups`, `nonlinear_coordinate_map`, `nonlinear_draws`; the maps key by parameter).

An autocorrelation term (C04, `ar()`, `ma()`, `arma()`): brms's Stan data list the rows sorted by series and time
(`order_data()`), while its predictions are in data order. `in_data_order` derives that permutation from the data's series
and time columns as R's `order()` sorts them, requires brms's `Y` to be the response under it, and puts every per-row
standata array back in data order, so that every matcher compares rows by content as for any other model; `J_lag` stays
on brms's rows and `check_autocor_design` checks it, `Kar` and `Kma`, and each Stan row's series and position against the
model's series. brms's `ar` and `ma` (`vector[Kar]`, `vector[Kma]`, unbounded) are the model's `ar` and `ma`.

C06: an `me(x, sdx)` or `mi(col)` term is one of brms's special terms (`Ksp`); the adapter maps `bsp[1]` to the model's one such
term (`ColumnMap.special`, `match_special`; brms's order of several is not verified here). An `me()` term is brms's
measurement-error group 1, whose `Xn_1` and `noise_1` are matched by content to `Design.measurement_error` and whose `meanme_1`,
`sdme_1` (log scale) and `zme_1` (one per row) map to `meanme_x`, `sdme_x_log__` and `zme_x` (`ColumnMap.measurement`). A
response's missing values, brms's `Ymi_<col>` (`Y` holds Inf at the 1-based rows `Jmi`), map to the model's `Ymi_<col>`, one
variable of the multivariate graph shared by the responses that hold it; `ResponseView` shows a response's `Ksp_y`, `bsp_y`,
`Jmi_xm`, `Nmi_xm`, `lbmi_xm`, `ubmi_xm` and `Ymi_xm` under brms's univariate names, and the multivariate point and draws carry
`Ymi_<col>` at the top level.

D08: a multi-membership term (brms's `J_m_k`, `W_m_k`, `Z_m_c_k` for each member `k`, no `J_m`) is matched by one bijection of
the levels over every member and row and by its weights (`_match_mm`); a `gr(g, by = gby)` term by brms's `Nby_m` and `Jby_m`
(`_check_by`), its `sd_m` an `M_m x Nby_m` matrix whose entry `(c, b)` maps to the by-level `b` of the coefficient's SD; a
correlation ID shared with a distributional formula is one brms group whose coefficients labelled with the parameter
(`groups[*].dpar`, `Z_m_<dpar>_c`, `r_m_<dpar>_c`) map to `Model.dpar_group_names` (`GroupMap.dpar_cn`), its `L_m` over all of
them in the order `correlation_order` gives (three coefficients: the probe calls the later `cholesky_corr` coordinates
'joint', each still recomputed from `L_m` and checked against `u`).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from functools import cached_property

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

import autobayes as ab
from autobayes.correlation import cholesky_corr_free
from autobayes.monotonic import stick_breaking_free
from tests.oracle.fixture import Fixture, Point
from tests.oracle.tolerances import DESIGN_ABS, abs_err


class OracleMismatch(AssertionError):
    """The AutoBayes model and the brms fixture disagree, or cannot be put in correspondence."""


def coefficients(pars: dict, name: str = "b") -> np.ndarray:
    """brms's ``b`` (``b_mu<k>`` of a categorical model's category, B05); rstan omits a zero-length parameter from
    constrain_pars(), so an intercept-only model has none."""
    return np.atleast_1d(pars[name]) if name in pars else np.zeros(0)


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
    "sigma": [...], "shape": [...], "phi": [...], "nu": [...], "zi": ["Beta", a, b], "hu": ["Beta", a, b],
    "simo": {column: ["Dirichlet", a, ...]}}``; an auxiliary parameter with its own formula (B01) takes ``{"intercept": [...],
    "b": [...]}``; ``simo`` is the simplex prior of each monotonic term (B03); a mixture (C05) takes ``{"components": [spec, ...],
    "theta": ["Dirichlet", a, ...]}``, each component's spec of this same form; ``nlpars`` each non-linear parameter's
    ``{"b": [...] | {term: ...}, "sd": [...] | {factor: ...}}`` (C02); ``ar`` and ``ma`` the priors of an
    autocorrelation term's coefficients (C04)."""
    # C01, a multivariate model: {"resp": {response: {<the classes above>}}, "cor": {factor: ["LKJ", eta]}, "rescor": ["LKJ", eta]}
    if "resp" in spec:
        return ab.Priors(resp={r: priors_from_spec(sub) for r, sub in spec["resp"].items()}, cor=_prior(spec.get("cor")),
                         rescor=_prior(spec.get("rescor")))
    return ab.Priors(intercept=_prior(spec.get("intercept")), b=_prior(spec.get("b")),
                     sd=_prior(spec.get("sd")), cor=_prior(spec.get("cor")), sigma=_prior(spec.get("sigma")),
                     shape=_prior(spec.get("shape")), phi=_prior(spec.get("phi")), nu=_prior(spec.get("nu")),
                     zi=_prior(spec.get("zi")), hu=_prior(spec.get("hu")), simo=_prior(spec.get("simo")),
                     # C05: a mixture's priors, one spec of this form per component, and theta's Dirichlet
                     components=[priors_from_spec(c) for c in spec["components"]] if "components" in spec else None,
                     theta=_prior(spec.get("theta")),
                     sdgp=_prior(spec.get("sdgp")), lscale=_prior(spec.get("lscale")),   # C03: {column: [...]} or one prior
                     nlpars=_prior(spec.get("nlpars")),                                 # C02: {nlpar: {"b": ..., "sd": ...}}
                     ar=_prior(spec.get("ar")), ma=_prior(spec.get("ma")),
                     # C06: a measurement-error term's latent values, {"meanme": [...], "sdme": [...]} (bare, or by column)
                     meanme=_prior(spec.get("meanme")), sdme=_prior(spec.get("sdme")))


def build_model(case: dict, data: pd.DataFrame) -> ab.Model:
    """The case's AutoBayes model on ``data``. ``"priors": "default"`` builds it under brms's default priors,
    ``ab.default_priors`` for the same formula, data, family and keyword arguments (A07): the case whose brms side has
    no priors (``"priors": []``)."""
    spec = case["autobayes"]
    constructor = getattr(ab, spec["constructor"])
    kwargs = spec.get("kwargs", {})
    if spec["priors"] == "default":
        priors = ab.default_priors(spec["formula"], data, family=spec.get("family", spec["constructor"]), **kwargs)
    else:
        priors = priors_from_spec(spec["priors"])
    return constructor(spec["formula"], data, priors, **kwargs)


# -- design columns -----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ColumnMap:
    """brms X column of each AutoBayes design column (in ``model.spec.columns`` order), brms's intercept
    column if any, and whether brms carries the separate centred ``Intercept`` parameter. ``thresholds``: brms's
    ``Intercept`` is the cumulative model's ``ordered`` thresholds and ``X`` has no intercept column (B05); ``dpars``: the
    categorical model's brms name of each non-reference category's linear predictor (``muc2``, ...), in AutoBayes's
    category order, each with its own ``X_<dpar>``, ``b_<dpar>`` and ``Intercept_<dpar>`` mapped alike (B05).
    ``distributional``: the ``ColumnMap`` of each auxiliary parameter's own formula (B01, ``X_sigma`` of ``sigma ~ z``) by
    parameter name, a design of its own matched to ``Design.dpars``, unlike the categorical ``dpars``, which share the
    response formula's design."""
    x_column: tuple[int, ...]
    intercept_column: int | None
    centred_intercept: bool
    thresholds: bool = False
    dpars: tuple[str, ...] = ()
    distributional: dict = None           # B01: the column map of each auxiliary parameter's own formula (X_sigma), by name
    monotonic: dict = None                # B03: brms's special-term number k (Xmo_k, bsp[k], simo_k) of each monotonic term, by column
    mixture: bool = False                 # C05: dpars are a mixture's components (brms's mu<k>), not a categorical model's categories
    gp: dict = None                       # C03: brms's GP number k (Xgp_k, sdgp_k, lscale_k, zgp_k) of each Gaussian process, by column
    nlpars: dict = None                   # C02: brms's X_<nlpar> column of each design column of each non-linear parameter, by name
    covariates: dict = None               # C02: brms's covariate number k (C_k) of each covariate of the non-linear formula, by name
    special: tuple = ()                   # C06: the model's me() / mi() term, by name, as brms's bsp[1] (match_special)
    measurement: dict = None              # C06: brms's measurement-error group k (Xn_k, meanme_k, sdme_k, zme_k) of each me() term, by noisy column

    def b_index(self, x_col: int) -> int:
        """Index into brms's ``b`` of design column ``x_col``."""
        if self.thresholds:                                   # no intercept column in X: b has one entry per column
            return x_col
        if self.centred_intercept:
            if x_col == self.intercept_column:
                raise OracleMismatch("the intercept column has no entry in b when brms centres")
            return x_col - (1 if x_col > self.intercept_column else 0)
        return x_col

    def x_of_b(self, b_idx: int) -> int:
        if self.thresholds:
            return b_idx
        if self.centred_intercept:
            return b_idx + (1 if b_idx >= self.intercept_column else 0)
        return b_idx


def _match_design(fixture: Fixture, model: ab.Model, X_name: str = "X", b_name: str = "b", intercept_name: str = "Intercept",
                  Kc_name: str = "Kc", thresholds: bool = False) -> ColumnMap:
    """Match AutoBayes's design columns to brms's design ``X_name`` by content (the population-level design ``X``, or a
    categorical model's ``X_mu<k>``), with ``b_name`` and ``intercept_name`` brms's coefficients and centred intercept of
    that design. ``thresholds``: the cumulative model's ``X``, which has no intercept column (B05)."""
    X = fixture.standata[X_name]
    names = fixture.colnames(X_name) or [str(k) for k in range(X.shape[1])]
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
    if thresholds:                                            # B05: brms's ordinal X has no intercept column, its Intercept
        declared = fixture.declarations.get(intercept_name, "")  # is the vector of ordered thresholds
        if "nthres" not in fixture.standata or not declared.startswith("ordered["):
            raise OracleMismatch(f"AutoBayes's cumulative model has thresholds but brms declares {intercept_name!r} as {declared!r}")
    elif model.spec.intercept:
        ones = [k for k in rest if np.max(np.abs(X[:, k] - 1.0)) <= DESIGN_ABS]
        if len(ones) != 1:
            raise OracleMismatch(f"expected one brms intercept column, found {[names[k] for k in ones]}")
        intercept_col = ones[0]
        rest.remove(intercept_col)
    if rest:
        raise OracleMismatch(f"brms design columns without an AutoBayes counterpart: {[names[k] for k in rest]}")
    pars = fixture.points[0].pars
    centred = intercept_name in pars
    if centred and intercept_col is None and not thresholds:
        raise OracleMismatch("brms has a centred Intercept parameter but AutoBayes has no population intercept")
    expected_b = X.shape[1] - (1 if centred and not thresholds else 0)
    if coefficients(pars, b_name).size != expected_b:
        raise OracleMismatch(f"brms {b_name} has {coefficients(pars, b_name).size} entries; expected {expected_b} from {X.shape[1]} design columns")
    if Kc_name in fixture.standata and int(fixture.standata[Kc_name]) != expected_b:
        raise OracleMismatch(f"brms {Kc_name} = {int(fixture.standata[Kc_name])} but {b_name} has {expected_b} entries")
    return ColumnMap(tuple(cols), intercept_col, centred, thresholds=thresholds)


def match_columns(fixture: Fixture, model: ab.Model) -> ColumnMap:
    if is_multivariate(model):                                 # C01: each response's design, by its view of the fixture
        return match_response_columns(fixture, model)
    if is_nonlinear(model):                                   # C02: the covariates and each non-linear parameter's X_<nlpar>
        return match_nonlinear_columns(fixture, model)
    fam = model.family_spec
    if fam.mixture:                                            # C05: one brms design per component, matched as B05's categories'
        return match_mixture(fixture, model)
    if not fam.per_category:
        cmap = _with_distributional(fixture, model, _match_design(fixture, model, thresholds=fam.thresholds))
        return replace(cmap, monotonic=match_monotonic(fixture, model), gp=match_gp(fixture, model), **match_special(fixture, model))
    # B05, the categorical family: one brms design per non-reference category, each matched like X; all must map alike
    from autobayes.default_priors import brms_dpar_names

    dpars = tuple(brms_dpar_names(model.categories[1:]))
    designs = sorted(k[2:] for k in fixture.standata if k.startswith("X_"))
    if sorted(dpars) != designs:
        raise OracleMismatch(f"brms has the category designs {designs}; AutoBayes's non-reference categories "
                             f"{list(model.categories[1:])} are brms's {sorted(dpars)}")
    maps = [_match_design(fixture, model, f"X_{dp}", f"b_{dp}", f"Intercept_{dp}", f"Kc_{dp}") for dp in dpars]
    if any(m != maps[0] for m in maps):
        raise OracleMismatch(f"brms's category designs {list(dpars)} do not map alike onto the AutoBayes design")
    return ColumnMap(maps[0].x_column, maps[0].intercept_column, maps[0].centred_intercept, dpars=dpars,
                     monotonic=match_monotonic(fixture, model))


def match_mixture(fixture: Fixture, model: ab.Model) -> ColumnMap:
    """A mixture's column map (C05): brms's ``X_mu<k>`` of every component, each matched by content like the categorical
    model's per-category designs (``_match_design`` with ``b_mu<k>``, ``Intercept_mu<k>``, ``Kc_mu<k>``), all mapping alike;
    brms must have exactly the model's components, ordered intercepts ``ordered[K] ordered_Intercept`` (``order = "mu"``)
    and the mixing proportions ``simplex[K] theta`` with ``con_theta``, and no other design."""
    dpars, K = tuple(model.mixture_names["dpars"]), len(model.components)
    designs = sorted(k[2:] for k in fixture.standata if k.startswith("X_"))
    if sorted(dpars) != designs:
        raise OracleMismatch(f"brms has the designs {designs}; the AutoBayes mixture's {K} components are brms's {sorted(dpars)}")
    for name, declared in (("ordered_Intercept", f"ordered[{K}]"), ("theta", f"simplex[{K}]")):
        if not fixture.declarations.get(name, "").startswith(declared):
            raise OracleMismatch(f"brms declares {name!r} as {fixture.declarations.get(name)!r}; the AutoBayes mixture of {K} "
                                 f"components has {declared} {name}")
    if "con_theta" not in fixture.standata or np.atleast_1d(fixture.standata["con_theta"]).shape != (K,):
        raise OracleMismatch(f"brms's con_theta is not the {K} concentrations of the mixing proportions")
    maps = [_match_design(fixture, model, f"X_{dp}", f"b_{dp}", f"Intercept_{dp}", f"Kc_{dp}") for dp in dpars]
    if any(m != maps[0] for m in maps):
        raise OracleMismatch(f"brms's component designs {list(dpars)} do not map alike onto the AutoBayes design")
    if not maps[0].centred_intercept:
        raise OracleMismatch("brms's mixture components have no centred Intercept_mu<k>; AutoBayes orders the centred intercepts")
    return ColumnMap(maps[0].x_column, maps[0].intercept_column, True, dpars=dpars, mixture=True)


def match_monotonic(fixture: Fixture, model: ab.Model) -> dict[str, int]:
    """brms's special-term number ``k`` of each monotonic term ``mo(x)`` (B03), by the term's column: the one whose ``Xmo_k``
    equals the term's codes row by row (within ``DESIGN_ABS``) and whose ``Jmo[k]`` is its number of steps. Every brms
    monotonic variable must be matched, and ``Ksp`` and ``Imo`` must both be the number of terms (each special term is then
    one ``mo()`` variable, so ``bsp[k]`` and ``simo_k`` belong to the term of ``Xmo_k``); brms's special terms without an
    AutoBayes monotonic term, and the reverse, are a mismatch."""
    sd, blocks = fixture.standata, model.design.monotonic
    brms = sorted(int(key[len("Xmo_"):]) for key in sd if key.startswith("Xmo_"))
    special = len(getattr(model.spec, "me_terms", ())) + len(getattr(model.spec, "mi_terms", ()))   # C06: me() / mi() terms
    if not blocks:
        if brms or ("Ksp" in sd and (not special or int(sd["Ksp"]) != special)) or (special and "Ksp" not in sd):
            raise OracleMismatch(f"brms has special terms (Ksp, Xmo_{brms}); the AutoBayes model has no monotonic term"
                                 + (f" and {special} me() / mi() term(s)" if special else ""))
        return {}
    counts = {"Ksp": int(sd["Ksp"]) if "Ksp" in sd else None, "Imo": int(sd["Imo"]) if "Imo" in sd else None, "Xmo_k": len(brms)}
    if any(v != len(blocks) for v in counts.values()):
        raise OracleMismatch(f"brms has {counts} special and monotonic terms; AutoBayes has {len(blocks)} monotonic terms")
    Jmo = np.atleast_1d(sd["Jmo"]).astype(int)
    out, used = {}, set()
    for column, block in blocks.items():
        hits = [k for k in brms if k not in used and sd[f"Xmo_{k}"].shape == block.codes.shape
                and abs_err(sd[f"Xmo_{k}"], block.codes) <= DESIGN_ABS and int(Jmo[k - 1]) == block.D]
        if len(hits) != 1:
            raise OracleMismatch(f"monotonic term {block.term!r}: {len(hits)} brms monotonic variables carry its codes and its "
                                 f"{block.D} steps (Xmo_k, Jmo[k])")
        out[column] = hits[0]
        used.add(hits[0])
    return out


def match_gp(fixture: Fixture, model: ab.Model) -> dict[str, int]:
    """brms's GP number ``k`` of each Gaussian process ``gp(x)`` (C03), by the process's column: the one whose ``Jgp_k`` is the
    process's row index (1-based) row by row and whose ``Xgp_k`` is its distinct scaled inputs, in order (an exact GP, without
    ``NBgp_k``) or its basis at them (an approximate GP, ``NBgp_k`` basis functions), within ``DESIGN_ABS``; ``Kgp_k`` and
    ``Dgp_k`` must be 1 (one covariate, no ``by``) and ``Nsubgp_k`` the number of distinct inputs. brms's Gaussian processes
    without an AutoBayes counterpart, and the reverse, are a mismatch. The content match fixes the order of the distinct
    inputs, which is the order of ``zgp_k`` (and, for the exact GP, what its Cholesky factor is)."""
    sd, blocks = fixture.standata, model.design.gp
    brms = sorted(int(m.group(1)) for m in (re.fullmatch(r"Xgp_(\d+)", key) for key in sd) if m)
    if len(brms) != len(blocks):
        raise OracleMismatch(f"brms has {len(brms)} Gaussian processes (Xgp_{brms}); the AutoBayes model has {len(blocks)}")
    out, used = {}, set()
    for column, block in blocks.items():
        g = model.spec.gp(column)

        def carries(k):
            X, J = sd[f"Xgp_{k}"], sd.get(f"Jgp_{k}")
            if J is None or J.shape != block.index.shape or abs_err(J, block.index + 1) > DESIGN_ABS:
                return False
            if g.approximate:
                return (f"NBgp_{k}" in sd and int(sd[f"NBgp_{k}"]) == g.k and X.shape == block.basis.shape
                        and abs_err(X, block.basis) <= DESIGN_ABS)
            return f"NBgp_{k}" not in sd and X.shape == (block.inputs.size, 1) and abs_err(X[:, 0], block.inputs) <= DESIGN_ABS
        hits = [k for k in brms if k not in used and carries(k)]
        if len(hits) != 1:
            raise OracleMismatch(f"Gaussian process {g.term!r}: {len(hits)} brms Gaussian processes carry its row index and its "
                                 f"{'basis' if g.approximate else 'distinct scaled inputs'} (Jgp_k, Xgp_k) within {DESIGN_ABS}")
        k = hits[0]
        for name, want in ((f"Kgp_{k}", 1), (f"Dgp_{k}", 1), (f"Nsubgp_{k}", block.inputs.size)):
            if name not in sd or int(sd[name]) != want:
                raise OracleMismatch(f"Gaussian process {g.term!r}: brms {name} is {sd.get(name)!r}, expected {want}")
        out[column] = k
        used.add(k)
    return out


def match_special(fixture: Fixture, model: ab.Model) -> dict:
    """C06: brms's special terms of the model's ``me()`` and ``mi()`` terms, as ``ColumnMap`` fields. brms's ``Ksp`` must be
    their number and the adapter maps ``bsp[1]`` for one such term (brms's order of several is not verified here). An
    ``me(x, sdx)`` term is brms's measurement-error group 1 (``Mme_1 = 1`` latent variable, ``NCme_1 = 0`` correlations),
    whose noisy values ``Xn_1`` and known standard deviations ``noise_1`` must equal ``Design.measurement_error[x]`` row by
    row; brms's measurement-error groups without an AutoBayes ``me()`` term, and the reverse, are a mismatch."""
    sd, spec = fixture.standata, model.spec
    terms = (*getattr(spec, "me_terms", ()), *getattr(spec, "mi_terms", ()))
    groups = sorted(int(key[len("Mme_"):]) for key in sd if key.startswith("Mme_") and key[len("Mme_"):].isdigit())
    if not terms:
        if groups:
            raise OracleMismatch(f"brms has measurement-error groups {groups}; the AutoBayes model has no me() term")
        return {"special": (), "measurement": None}
    if len(terms) != 1:
        raise OracleMismatch(f"the AutoBayes model has the special terms {[t.name for t in terms]}; brms's order of several in bsp "
                             "is not verified by this adapter")
    if "Ksp" not in sd or int(sd["Ksp"]) != 1 or any(key.startswith("Xmo_") for key in sd):
        raise OracleMismatch(f"brms has Ksp = {sd.get('Ksp')} special terms; the AutoBayes model has {terms[0].name!r} alone")
    measurement = {}
    for t in getattr(spec, "me_terms", ()):
        x, block = t.operands[0], model.design.measurement_error[t.operands[0]]
        if groups != [1] or int(sd["Mme_1"]) != 1 or int(sd.get("NCme_1", 0)) != 0:
            raise OracleMismatch(f"brms's measurement-error groups {groups} (Mme_1 = {sd.get('Mme_1')}, NCme_1 = {sd.get('NCme_1')}) "
                                 f"are not the one latent variable of {t.name!r}")
        for key, mine in (("Xn_1", block.Xn), ("noise_1", block.noise)):
            if key not in sd or sd[key].shape != mine.shape or abs_err(sd[key], mine) > DESIGN_ABS:
                raise OracleMismatch(f"brms's {key} is not the {'noisy values' if key == 'Xn_1' else 'known noise'} of {t.name!r}")
        measurement[x] = 1
    if groups and not measurement:
        raise OracleMismatch(f"brms has measurement-error groups {groups}; the AutoBayes model has no me() term")
    return {"special": (terms[0].name,), "measurement": measurement or None}


def _with_distributional(fixture: Fixture, model: ab.Model, cmap: ColumnMap) -> ColumnMap:
    """``cmap`` with the column map of each auxiliary parameter's own formula (B01): a formula on either side must exist
    on both (brms's ``X_<parameter>`` of the family's auxiliary parameter), and each is matched by content."""
    scale = model.family_spec.scale_parameter
    if scale is not None and f"X_{scale}" in fixture.standata and scale not in model.spec.dpar_names:
        raise OracleMismatch(f"brms gives {scale} its own formula (standata X_{scale}); the AutoBayes model has none")
    return replace(cmap, distributional={name: match_distributional_columns(fixture, model, name) for name in model.spec.dpar_names})


def match_distributional_columns(fixture: Fixture, model: ab.Model, name: str) -> ColumnMap:
    """The column map of distributional parameter ``name``'s formula (B01): brms's ``X_<name>`` matched by content to
    ``model.design.dpars[name].X_raw`` column by column, its intercept column, and whether brms carries the separate
    (centred) ``Intercept_<name>``, with ``b_<name>`` and ``Kc_<name>`` checked against them, as ``match_columns`` does
    for the response formula's ``X``."""
    key = f"X_{name}"
    if key not in fixture.standata:
        raise OracleMismatch(f"the AutoBayes model gives {name} its own formula; brms's standata has no {key}")
    X = fixture.standata[key]
    labels = fixture.colnames(key) or [str(k) for k in range(X.shape[1])]
    ds, XA = model.spec.dpar(name), model.design.dpars[name].X_raw
    if X.shape[0] != XA.shape[0]:
        raise OracleMismatch(f"brms {key} has {X.shape[0]} rows, AutoBayes {XA.shape[0]}")
    taken, cols = set(), []
    for c, label in enumerate(ds.columns):
        hits = [k for k in range(X.shape[1]) if k not in taken and np.max(np.abs(X[:, k] - XA[:, c])) <= DESIGN_ABS]
        if len(hits) != 1:
            raise OracleMismatch(f"{name}'s design column {label!r} matches {len(hits)} brms {key} columns {[labels[k] for k in hits]} within {DESIGN_ABS}")
        cols.append(hits[0])
        taken.add(hits[0])
    rest = [k for k in range(X.shape[1]) if k not in taken]
    intercept_col = None
    if ds.intercept:
        ones = [k for k in rest if np.max(np.abs(X[:, k] - 1.0)) <= DESIGN_ABS]
        if len(ones) != 1:
            raise OracleMismatch(f"expected one intercept column in brms {key}, found {[labels[k] for k in ones]}")
        intercept_col = ones[0]
        rest.remove(intercept_col)
    if rest:
        raise OracleMismatch(f"brms {key} columns without an AutoBayes counterpart: {[labels[k] for k in rest]}")
    pars = fixture.points[0].pars
    centred = f"Intercept_{name}" in pars
    if centred and intercept_col is None:
        raise OracleMismatch(f"brms has a centred Intercept_{name} but AutoBayes's formula of {name} has no intercept")
    expected_b = X.shape[1] - (1 if centred else 0)
    got = coefficients(pars, f"b_{name}").size
    if got != expected_b:
        raise OracleMismatch(f"brms b_{name} has {got} entries; expected {expected_b} from {X.shape[1]} {key} columns")
    if f"Kc_{name}" in fixture.standata and int(fixture.standata[f"Kc_{name}"]) != expected_b:
        raise OracleMismatch(f"brms Kc_{name} = {int(fixture.standata[f'Kc_{name}'])} but b_{name} has {expected_b} entries")
    if ds.columns and centred != model.design.dpars[name].centered:     # moot without a column to centre (sigma ~ 1)
        raise OracleMismatch(f"brms {'centres' if centred else 'does not centre'} the formula of {name}; AutoBayes "
                             f"{'does' if model.design.dpars[name].centered else 'does not'}")
    return ColumnMap(tuple(cols), intercept_col, centred)


# -- group-level terms --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GroupMap:
    brms_id: dict[str, int]              # AutoBayes factor -> brms group id m (of sd_m, z_m, J_m)
    cn: dict[str, int]                   # 1-based coefficient number of the varying intercept in that group
    level_of: dict[str, np.ndarray]      # level_of[g][j] = AutoBayes level index of brms level j (0-based)
    coef_cn: dict[tuple[str, str], int]  # (factor, coefficient) -> 1-based coefficient number c of sd_m[c], z_m[c], r_m_c
    # D08: a distributional formula's coefficient of a shared correlation ID, (parameter, factor, coefficient) -> its 1-based
    # coefficient number in the factor's brms group (sd_m[c], z_m[c], r_m_<parameter>_c, Z_m_<parameter>_c)
    dpar_cn: dict = None


def match_groups(fixture: Fixture, model: ab.Model) -> GroupMap:
    """Match every AutoBayes group block ``(factor, coef)`` to one brms group coefficient ``(id, cn)``: the one
    whose ``J_id`` induces the block's partition of the rows and whose ``Z_id_cn`` equals the block's covariate
    column (ones for the varying intercept, the slope's column otherwise); then check the coefficient's brms
    label, the level order, that every brms coefficient was used, and that each group is correlated on both
    sides or on neither."""
    if is_multivariate(model):                                 # C01: each response's group terms, by its view of the fixture
        return match_response_groups(fixture, model)
    if is_nonlinear(model):                                   # C02: each non-linear parameter's group-level terms
        return match_nonlinear_groups(fixture, model)
    ids, cns, levels_of, coef_cn, used = {}, {}, {}, {}, set()
    n = model.design.n
    for (g, coef), block in model.design.group_blocks.items():
        if block.memberships:                            # D08: a multi-membership term, brms's J_m_k, W_m_k, Z_m_c_k
            found = _match_mm(fixture, block, used)
            if len(found) != 1:
                raise OracleMismatch(f"multi-membership group {g!r}: {len(found)} brms group terms carry its memberships, weights and "
                                     f"covariates (J_m_k, W_m_k, Z_m_c_k) within {DESIGN_ABS}")
            grp, m = found[0]
            if grp["coef"] != coef or [block.levels[i] for i in m] != list(grp["levels"]):
                raise OracleMismatch(f"{g!r}: brms coefficient {grp['coef']!r} with levels {grp['levels']}; AutoBayes {coef!r} with "
                                     f"{[block.levels[i] for i in m]}")
            used.add((int(grp["id"]), int(grp["cn"])))
            ids[g], levels_of[g], coef_cn[(g, coef)], cns[g] = int(grp["id"]), m, int(grp["cn"]), int(grp["cn"])
            continue
        gidx = block.group_index
        covariate = block.Z[np.arange(n), gidx]          # the coefficient's covariate per row (brms Z_m_c)
        found = []
        for grp in fixture.groups:
            key = (int(grp["id"]), int(grp["cn"]))
            if key in used or grp.get("dpar") or f"J_{grp['id']}" not in fixture.standata:   # D08: a distributional formula's
                continue                                                                      # coefficients; mm's J_m_k
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
    dpar_cn = _match_dpar_groups(fixture, model, ids, levels_of, used)   # D08: a shared ID's distributional coefficients
    unmatched = [(grp["group"], grp["coef"]) for grp in fixture.groups if (int(grp["id"]), int(grp["cn"])) not in used]
    if unmatched:
        raise OracleMismatch(f"brms group-level terms without an AutoBayes counterpart: {unmatched}")
    for t in model.spec.group_terms:
        m = ids[t.factor]
        M = int(fixture.standata[f"M_{m}"])
        shared = model.spec.shared_id_of(t.factor) if hasattr(model.spec, "shared_id_of") else None
        coefs = shared.labels if shared is not None else t.coefs   # D08: a shared ID's group spans the formulas
        if M != len(coefs):
            raise OracleMismatch(f"{t.factor!r}: brms group {m} has M_{m} = {M} coefficients, AutoBayes {len(coefs)}")
        _check_by(fixture, model, t, m, levels_of[t.factor])            # D08: gr(by = )'s Nby_m and Jby_m
        # brms's standata carries NC_m = M_m (M_m - 1) / 2 whether or not the effects are correlated; what says
        # they are is the term's `cor` flag and a declared Cholesky factor L_m (A02): both must agree with the
        # AutoBayes term, which is correlated exactly when it has two or more coefficients written with '|'.
        correlated = any(bool(grp["cor"]) for grp in fixture.groups if int(grp["id"]) == m)
        declared = f"L_{m}" in fixture.declarations
        if M > 1 and correlated != declared:
            raise OracleMismatch(f"{t.factor!r}: brms group {m} has cor = {correlated} but L_{m} is {'' if declared else 'not '}declared")
        if M > 1 and (correlated or declared) != (t.has_correlation or shared is not None):
            raise OracleMismatch(
                f"{t.factor!r}: brms group {m} has {M} coefficients with cor = {correlated}"
                f"{' and a correlation matrix L_' + str(m) if declared else ''}; AutoBayes's term '{t.canonical}' is "
                f"{'correlated' if t.has_correlation else 'independent'}")
    return GroupMap(ids, cns, levels_of, coef_cn, dpar_cn or None)


def _level_map(pairs, J: int) -> np.ndarray | None:
    """D08: the AutoBayes level of each brms level from the (brms level, AutoBayes level) pairs of every row, or None when
    the pairs are not one bijection of ``J`` levels."""
    m = np.full(J, -1)
    for b, a in pairs:
        if b < 0 or b >= J or (m[b] not in (-1, a)):
            return None
        m[b] = a
    return m if (m >= 0).all() and len(np.unique(m)) == J else None


def _match_mm(fixture: Fixture, block, used) -> list:
    """D08: the brms group coefficients that carry a multi-membership block: brms's member indices ``J_m_k`` give one
    bijection of the levels with the block's memberships over every member and row, brms's weights ``W_m_k`` are the block's
    and ``Z_m_c_k`` its covariate (ones, the varying intercept), for every member ``k``."""
    sd, out, K = fixture.standata, [], len(block.memberships)
    for grp in fixture.groups:
        m_id, cn = int(grp["id"]), int(grp["cn"])
        if (m_id, cn) in used or grp.get("dpar") or f"J_{m_id}_{K}" not in sd or f"J_{m_id}_{K + 1}" in sd:
            continue
        J = [sd[f"J_{m_id}_{k}"].astype(int) - 1 for k in range(1, K + 1)]
        if any(j.shape != block.group_index.shape for j in J):
            continue
        m = _level_map([(int(b), int(a)) for j, idx in zip(J, block.memberships) for b, a in zip(j, idx)], block.J)
        if m is None:
            continue
        same = all(f"W_{m_id}_{k}" in sd and abs_err(sd[f"W_{m_id}_{k}"], w) <= DESIGN_ABS
                   and f"Z_{m_id}_{cn}_{k}" in sd and abs_err(sd[f"Z_{m_id}_{cn}_{k}"], np.ones(block.group_index.shape)) <= DESIGN_ABS
                   for k, w in zip(range(1, K + 1), block.weights))
        if same:
            out.append((grp, m))
    return out


def _check_by(fixture: Fixture, model, t, m: int, level_of: np.ndarray) -> None:
    """D08: a ``gr(g, by = gby)`` term against brms's ``Nby_m`` (the number of by levels) and ``Jby_m`` (each brms level's
    1-based by level): the block's by levels in the same order (brms's factor levels, sorted here) and each level's by level;
    a term without ``by`` has neither."""
    sd = fixture.standata
    block = model.design.group_blocks.get((t.factor, "Intercept"))
    if t.by is None:
        if f"Jby_{m}" in sd or f"Nby_{m}" in sd:
            raise OracleMismatch(f"brms group {m} has by levels (Nby_{m}, Jby_{m}); AutoBayes's '{t.canonical}' has no by column")
        return
    if f"Nby_{m}" not in sd or int(sd[f"Nby_{m}"]) != len(block.by_levels):
        raise OracleMismatch(f"brms Nby_{m} = {sd.get(f'Nby_{m}')}; '{t.canonical}' has the by levels {list(block.by_levels)}")
    jby = np.atleast_1d(sd[f"Jby_{m}"]).astype(int) - 1
    if jby.shape != level_of.shape or not np.array_equal(jby, block.by_index[level_of]):
        raise OracleMismatch(f"brms Jby_{m} = {(jby + 1).tolist()}; '{t.canonical}' gives brms's levels the by levels "
                             f"{(block.by_index[level_of] + 1).tolist()} (by levels {list(block.by_levels)})")


def _match_dpar_groups(fixture: Fixture, model, ids: dict, levels_of: dict, used: set) -> dict:
    """D08: each group-level coefficient of a distributional formula (a correlation ID shared with the response formula's
    term of its factor) is brms's coefficient of the same group ``m`` labelled with the parameter (``groups[*].dpar``), whose
    ``Z_m_<parameter>_c`` is its covariate (ones) and whose level order is the response formula's term's."""
    out = {}
    for name, dd in getattr(model.design, "dpars", {}).items():
        for (g, coef), block in getattr(dd, "group_blocks", {}).items():
            hits = [grp for grp in fixture.groups if grp.get("dpar") == name and grp["coef"] == coef and int(grp["id"]) == ids.get(g)
                    and (int(grp["id"]), int(grp["cn"])) not in used]
            hits = [grp for grp in hits if f"Z_{grp['id']}_{name}_{grp['cn']}" in fixture.standata
                    and abs_err(fixture.standata[f"Z_{grp['id']}_{name}_{grp['cn']}"], block.Z[np.arange(block.Z.shape[0]), block.group_index]) <= DESIGN_ABS
                    and [block.levels[i] for i in levels_of[g]] == list(grp["levels"])]
            if len(hits) != 1:
                raise OracleMismatch(f"the coefficient {coef!r} of {g!r} in {name}'s formula: {len(hits)} brms coefficients of group "
                                     f"{ids.get(g)} with dpar = {name!r} carry its covariate and levels")
            used.add((int(hits[0]["id"]), int(hits[0]["cn"])))
            out[(name, g, coef)] = int(hits[0]["cn"])
    unclaimed = [(grp["group"], grp["coef"], grp["dpar"]) for grp in fixture.groups if grp.get("dpar") and (int(grp["id"]), int(grp["cn"])) not in used]
    if unclaimed:
        raise OracleMismatch(f"brms group-level coefficients of distributional formulas without an AutoBayes counterpart: {unclaimed}")
    return out


def correlation_order(model: ab.Model, gmap: GroupMap, factor: str) -> list[int]:
    """brms's 0-based coefficient numbers in the AutoBayes coefficient order of a correlated factor: the permutation
    that takes brms's ``L_m`` / ``Cor_m`` rows and columns to AutoBayes's ``L`` / ``cor``. A correlation ID shared with a
    distributional formula (D08) spans the response formula's coefficients and then the distributional formula's."""
    shared = model.spec.shared_id_of(factor) if hasattr(model.spec, "shared_id_of") else None
    if shared is not None:
        return [(gmap.coef_cn[(factor, c)] if not dpar else gmap.dpar_cn[(dpar, factor, c)]) - 1 for dpar, c in shared.coefs]
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
    """AutoBayes's natural-scale arguments (intercept, b, sd, z, L, sigma, shape, phi, nu) at a brms point (the cumulative
    model's intercept is brms's vector of thresholds; a categorical model's intercept has one entry per non-reference
    category and its b one column per such category, B05). An intercept-only
    factor's ``sd`` and ``z`` are the scalar and the vector; a factor with varying slopes takes the
    coefficient-keyed mappings ``{"Intercept": ..., "x": ...}`` (``Model.value_point``); a correlated factor's ``L`` is
    brms's ``L_m`` (A02). ``sigma``, ``shape`` and
    ``phi`` are brms's parameters of those names (the Gaussian/Student-t/lognormal scale; the negative binomial/Gamma
    shape; the Beta precision); ``nu`` is brms's ``nu`` when the model learns it (a fixed nu is no value variable here,
    though brms reports it among its parameters); ``zi`` and ``hu`` are brms's zero-inflation and hurdle probabilities (B06);
    ``bsp`` and ``simo`` each monotonic term's coefficient and simplex, by column (B03); ``ar`` and ``ma`` an autocorrelation
    term's coefficients, brms's vectors of those names (C04)."""
    if is_multivariate(model):                                 # C01: each response's arguments from its view of the point, and Lrescor
        return {"resp": {r: natural_point(sub, cmap.views[r].point(point), cmap.maps[r], gmap.maps[r]) for r, sub in model.submodels.items()},
                **({"Lrescor": np.atleast_2d(point.pars["Lrescor"])} if model.rescor else {}),
                # C06: the model's missing values, brms's Ymi_<col>, shared by the responses that hold them
                **({"Ymi": {c: np.atleast_1d(point.pars[f"Ymi_{c}"]).astype(float) for c in model.missing_names}}
                   if getattr(model, "missing_names", {}) else {})}
    pars = point.pars
    intercept = None
    if cmap.dpars:                                             # B05, categorical: one column of b and one intercept per category
        bs = [coefficients(pars, f"b_{dp}") for dp in cmap.dpars]
        b = np.array([[bb[cmap.b_index(k)] for bb in bs] for k in cmap.x_column], dtype=float).reshape(len(cmap.x_column), len(bs))
        if model.spec.intercept:
            intercept = np.array([float(pars[f"Intercept_{dp}"]) if cmap.centred_intercept else float(bb[cmap.b_index(cmap.intercept_column)])
                                  for dp, bb in zip(cmap.dpars, bs)])
    else:
        bb = coefficients(pars)
        b = np.array([bb[cmap.b_index(k)] for k in cmap.x_column], dtype=float)
        if model.spec.intercept and cmap.thresholds:           # B05: brms's ordered thresholds, a vector
            intercept = np.atleast_1d(pars["Intercept"]).astype(float)
        elif model.spec.intercept:
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
            per_sd[coef], per_z[coef] = float(np.ravel(sd_m)[cn - 1]) if t.by is None else None, za
            if t.by is not None:                               # D08: brms's sd_m is (M_m, Nby_m), one SD per by level (in order)
                per_sd[coef] = np.atleast_2d(pars[f"sd_{m}"])[cn - 1].astype(float)
        if t.slopes:
            sd[g], z[g] = per_sd, per_z
        else:
            sd[g], z[g] = per_sd["Intercept"], per_z["Intercept"]
    L = {g: brms_cholesky(model, gmap, g, pars) for g in model.correlation_names}   # D08: a shared ID's too
    sigma = float(pars["sigma"]) if "sigma" in pars else None
    shape = float(pars["shape"]) if "shape" in pars else None
    phi = float(pars["phi"]) if "phi" in pars else None
    formulas = {}                                          # B01: an auxiliary parameter's own formula: its intercept (as sampled) and b
    for name, dmap in (cmap.distributional or {}).items():
        bd = coefficients(pars, f"b_{name}")
        ds = model.spec.dpar(name)
        formulas[name] = {"b": np.array([bd[dmap.b_index(k)] for k in dmap.x_column], dtype=float)}
        if ds.intercept:
            formulas[name]["intercept"] = (float(pars[f"Intercept_{name}"]) if dmap.centred_intercept
                                           else float(bd[dmap.b_index(dmap.intercept_column)]))
        if ds.group_terms:                                 # D08: a shared ID's coefficient: brms's sd_m[c] and z_m[c] of its group
            formulas[name]["sd"], formulas[name]["z"] = {}, {}
            for t in ds.group_terms:
                m, cn = gmap.brms_id[t.factor], gmap.dpar_cn[(name, t.factor, "Intercept")]
                za = np.empty(np.atleast_2d(pars[f"z_{m}"]).shape[1])
                za[gmap.level_of[t.factor]] = np.atleast_2d(pars[f"z_{m}"])[cn - 1]
                formulas[name]["sd"][t.factor], formulas[name]["z"][t.factor] = float(np.atleast_1d(pars[f"sd_{m}"])[cn - 1]), za
    sigma, shape, phi = formulas.get("sigma", sigma), formulas.get("shape", shape), formulas.get("phi", phi)
    nu = float(pars["nu"]) if "nu" in model.names else None
    zi = float(pars["zi"]) if "zi" in pars else None
    hu = float(pars["hu"]) if "hu" in pars else None
    bsp = simo = None
    if cmap.monotonic:                                     # B03: brms's bsp[k] and simo_k of each monotonic term
        bsp = {c: float(np.atleast_1d(pars["bsp"])[k - 1]) for c, k in cmap.monotonic.items()}
        simo = {c: np.atleast_1d(pars[f"simo_{k}"]).astype(float) for c, k in cmap.monotonic.items()}
    theta = None
    if cmap.mixture:                                       # C05: brms's sigma<k> and the mixing proportions theta
        sigma = np.array([float(pars[name]) for name in model.mixture_names["sigma"]])
        theta = np.atleast_1d(pars["theta"]).astype(float)
    sdgp = lscale = zgp = None
    if cmap.gp:                                            # C03: brms's sdgp_k, lscale_k and zgp_k of each Gaussian process
        sdgp = {c: float(np.ravel(pars[f"sdgp_{k}"])[0]) for c, k in cmap.gp.items()}
        lscale = {c: float(np.ravel(pars[f"lscale_{k}"])[0]) for c, k in cmap.gp.items()}
        zgp = {c: np.ravel(pars[f"zgp_{k}"]).astype(float) for c, k in cmap.gp.items()}
    ar, ma = (np.atleast_1d(pars[part]).astype(float) if part in model.autocor_names else None for part in ("ar", "ma"))   # C04
    special = {}
    if cmap.special:                                       # C06: the me() / mi() term's coefficient, brms's bsp[1], by its column
        terms = {t.name: t for t in model.spec.pop_terms}
        bsp = {terms[name].operands[0]: float(np.atleast_1d(pars["bsp"])[k]) for k, name in enumerate(cmap.special)}
    if cmap.measurement:                                   # C06: an me() term's meanme_k, sdme_k and zme_k, by noisy column
        special = {cls: {x: (np.atleast_1d(pars[f"{cls}_{k}"]).astype(float) if cls == "zme" else float(np.atleast_1d(pars[f"{cls}_{k}"])[0]))
                         for x, k in cmap.measurement.items()} for cls in ("meanme", "sdme", "zme")}
    return dict(intercept=intercept, b=b, sd=sd or None, z=z or None, L=L or None, sigma=sigma, shape=shape, phi=phi, nu=nu,
                zi=zi, hu=hu, bsp=bsp, simo=simo, theta=theta, sdgp=sdgp, lscale=lscale, zgp=zgp, ar=ar, ma=ma, **special)


def value_point(model: ab.Model, point: Point, cmap: ColumnMap, gmap: GroupMap) -> dict:
    if is_nonlinear(model):                                   # C02
        return model.value_point(**nonlinear_natural_point(model, point, cmap, gmap))
    return model.value_point(**natural_point(model, point, cmap, gmap))


def value_layout(model: ab.Model, point: dict) -> dict[str, slice]:
    """Where each value variable sits in PyMC's flattened gradient (value_vars order, each raveled)."""
    out, i = {}, 0
    for v in model.pymc_model.value_vars:
        size = int(np.asarray(point[v.name]).size)
        out[v.name] = slice(i, i + size)
        i += size
    return out


# brms's probe (oracle/R/gen_fixture.R) puts every unconstrained coordinate at 0.3 and moves one constrained value to 1.5
# times its value there; for real<lower=1> nu sampled as log(nu - 1), the probed step du is therefore this number exactly.
_NU0 = 1.0 + math.exp(0.3)
LOG_NU_MINUS_ONE_STEP = math.log(1.5 * _NU0 - 1.0) - 0.3
# For real<lower=0,upper=1> zi or hu (B06) sampled as logit(zi), from zi = inv_logit(0.3) to 1.5 times that.
_P0 = 1.0 / (1.0 + math.exp(-0.3))
LOGIT_STEP = math.log(1.5 * _P0 / (1.0 - 1.5 * _P0)) - 0.3


def coordinate_map(fixture: Fixture, model: ab.Model, cmap: ColumnMap, gmap: GroupMap) -> list[tuple[int, str, int]]:
    """For each brms unconstrained coordinate: (its 0-based index, the AutoBayes value variable, the flat
    index within that variable), with the transform checked on both sides."""
    if is_multivariate(model):                                 # C01: each response's coordinates, then Lrescor's
        return response_coordinate_map(fixture, model, cmap, gmap)
    if is_nonlinear(model):                                   # C02
        return nonlinear_coordinate_map(fixture, model, cmap, gmap)
    names = model.names
    by_id = {m: g for g, m in gmap.brms_id.items()}
    by_cn = {(g, cn): coef for (g, coef), cn in gmap.coef_cn.items()}     # (factor, brms coefficient number) -> coefficient
    by_dpar_cn = {(g, cn): (name, g, coef) for (name, g, coef), cn in (gmap.dpar_cn or {}).items()}   # D08: a shared ID's
    roles = model.group_coefficient_names
    pars = fixture.points[0].pars
    seen_L: dict[str, int] = {}                                            # L_m coordinates met so far, in index order
    seen_thresholds = 0                                                    # Intercept coordinates met so far (B05)
    others = len(cmap.dpars)
    simplex_coordinates = _simplex_coordinates(fixture, model, cmap)       # B03: the coordinates the probe could not reach
    by_k = {k: c for c, k in (cmap.monotonic or {}).items()}               # B03: brms's special-term number -> column
    by_gp = {k: c for c, k in (cmap.gp or {}).items()}                     # C03: brms's GP number -> column
    out = []
    for e in fixture.upars:
        idx, transform = int(e["index"]) - 1, e["transform"]
        if transform == "unreached":                                        # B03: a simplex coordinate of a monotonic term
            if idx not in simplex_coordinates:
                raise OracleMismatch(f"brms's unconstrained coordinate {idx + 1} was not reached by the probe and is no simplex "
                                     "coordinate of a monotonic term")
            out.append((idx, *simplex_coordinates[idx]))
            continue
        par, elem = e["par"], int(e["elem"]) - 1
        if cmap.dpars and par.startswith("b_") and par[2:] in cmap.dpars:          # B05: a coefficient of category k
            k = cmap.dpars.index(par[2:])
            col = cmap.x_of_b(elem)
            if cmap.intercept_column is not None and col == cmap.intercept_column:
                target, expect = (names["intercept"], k), "identity"
            else:
                _, term, j = model.coefficients[cmap.x_column.index(col)]
                target, expect = (term, (0 if j is None else j) * others + k), "identity"   # (levels, categories), C order
        elif cmap.dpars and par.startswith("Intercept_") and par[len("Intercept_"):] in cmap.dpars:   # B05: its intercept
            target, expect = (names["intercept"], cmap.dpars.index(par[len("Intercept_"):])), "identity"
        elif par == "ordered_Intercept" and cmap.mixture:     # C05: a mixture's intercepts, brms's ordered[K] ordered_Intercept
            k = seen_thresholds
            seen_thresholds += 1
            # sampled on Stan's ordered coordinates, as the cumulative thresholds are (below): recompute coordinate k from
            # ordered_Intercept at every point and require it to be the fixture's u before the gradient is compared
            for q, pt_ in enumerate(fixture.points):
                tau = np.atleast_1d(pt_.pars["ordered_Intercept"]).astype(float)
                u = np.r_[tau[0], np.log(np.diff(tau))]
                if k >= u.size or abs(u[k] - pt_.u[idx]) > 1e-10:
                    raise OracleMismatch(f"point {q}: Stan's ordered coordinate {k} of ordered_Intercept is "
                                         f"{u[k] if k < u.size else 'absent'!r}; brms's unconstrained value is {pt_.u[idx]!r}")
            target, expect = (f"{names['intercept']}_ordered__", k), "joint"
        elif cmap.mixture and par in model.mixture_names["sigma"]:   # C05: brms's real<lower=0> sigma<k>, on the log scale
            target, expect = (f"{names[par]}_log__", 0), "log"
        elif par == "Intercept" and cmap.thresholds:          # B05: the cumulative thresholds, brms's ordered[nthres]
            k = seen_thresholds
            seen_thresholds += 1
            # brms samples them on Stan's ordered coordinates (the probe saw each move a later one: 'joint'): recompute
            # coordinate k from Intercept at every point and require it to be the fixture's u, so that the AutoBayes
            # transform is Stan's coordinate by coordinate before the gradient is compared.
            for q, pt_ in enumerate(fixture.points):
                tau = np.atleast_1d(pt_.pars["Intercept"]).astype(float)
                u = np.r_[tau[0], np.log(np.diff(tau))]
                if k >= u.size or abs(u[k] - pt_.u[idx]) > 1e-10:
                    raise OracleMismatch(f"point {q}: Stan's ordered coordinate {k} of Intercept is "
                                         f"{u[k] if k < u.size else 'absent'!r}; brms's unconstrained value is {pt_.u[idx]!r}")
            target, expect = (f"{names['intercept']}_ordered__", k), "joint"
        elif par == "b":
            k = cmap.x_of_b(elem)
            if cmap.intercept_column is not None and k == cmap.intercept_column:
                target, expect = (names["intercept"], 0), "identity"
            else:
                _, term, j = model.coefficients[cmap.x_column.index(k)]
                target, expect = (term, 0 if j is None else j), "identity"
        elif par == "Intercept":
            target, expect = (names["intercept"], 0), "identity"
        elif par == "bsp" and cmap.special:                    # C06: the coefficient of the model's me() / mi() term
            if elem >= len(cmap.special):
                raise OracleMismatch(f"bsp[{elem + 1}] is not the coefficient of an me() or mi() term of the AutoBayes model")
            target, expect = (cmap.special[elem], 0), "identity"
        elif par.startswith(("meanme_", "sdme_", "zme_")) and par.rsplit("_", 1)[1].isdigit():   # C06: an me() term's latent values
            cls, k = par.rsplit("_", 1)
            x = next((x for x, kk in (cmap.measurement or {}).items() if kk == int(k)), None)
            if x is None:
                raise OracleMismatch(f"{par} belongs to brms's measurement-error group {k}, which no me() term of the AutoBayes model is")
            roles = model.me_names[x]
            if cls == "zme":
                target, expect = (roles["z"], elem), "identity"
            elif elem != 0:
                raise OracleMismatch(f"{par}[{elem + 1}]: the AutoBayes me() term has one latent variable")
            else:
                target, expect = ((roles["mean"], 0), "identity") if cls == "meanme" else ((f"{roles['sd']}_log__", 0), "log")
        elif par == "Ymi":                                     # C06: the response's own missing values (brms's Ymi_<response>)
            roles = getattr(model, "missing_names", {}).get(model.spec.response)
            if roles is None:
                raise OracleMismatch(f"brms samples missing values of {model.spec.response!r}; the AutoBayes model holds none")
            target, expect = (roles["values"], elem), "identity"
        elif par == "bsp":                                     # B03: the coefficient of brms's special term elem + 1, a monotonic term
            if elem + 1 not in by_k:
                raise OracleMismatch(f"bsp[{elem + 1}] is not the coefficient of a monotonic term of the AutoBayes model")
            target, expect = (model.mo_names[by_k[elem + 1]]["b"], 0), "identity"
        elif par in ("ar", "ma"):                              # C04: brms's vector[Kar] ar / vector[Kma] ma, unbounded
            if par not in model.autocor_names:
                raise OracleMismatch(f"brms samples {par} but the AutoBayes model has no {par} coefficients")
            target, expect = (model.autocor_names[par], elem), "identity"
        elif par == "sigma":
            target, expect = (f"{names['sigma']}_log__", 0), "log"
        elif par == "shape":                                   # negative binomial / Gamma: PyMC's alpha, sampled on the log scale
            target, expect = (f"{names['shape']}_log__", 0), "log"
        elif par == "phi":                                     # Beta precision, sampled on the log scale
            target, expect = (f"{names['phi']}_log__", 0), "log"
        elif par.startswith("Intercept_") and par[len("Intercept_"):] in (cmap.distributional or {}):   # B01: brms's centred
            target, expect = (model.dpar_names[par[len("Intercept_"):]]["intercept"], 0), "identity"       # Intercept_sigma
        elif par.startswith("b_") and par[2:] in (cmap.distributional or {}):                           # B01: brms's b_sigma
            name = par[2:]
            dmap, ds = cmap.distributional[name], model.spec.dpar(name)
            k = dmap.x_of_b(elem)
            if dmap.intercept_column is not None and k == dmap.intercept_column:
                target, expect = (model.dpar_names[name]["intercept"], 0), "identity"
            else:
                _, term, j = ds.column_terms[dmap.x_column.index(k)]
                target, expect = (model.dpar_names[name][f"b:{term}"], 0 if j is None else j), "identity"
        elif par == "nu":                                      # learned Student-t nu, brms real<lower=1> nu: PyMC's interval log(nu - 1)
            if "nu" not in names:
                raise OracleMismatch("brms samples nu but the AutoBayes model does not learn it")
            if abs(float(e["du"]) - LOG_NU_MINUS_ONE_STEP) > 1e-9:
                raise OracleMismatch(f"brms's probe moved the coordinate of nu by {e['du']!r}; log(nu - 1) moves it by {LOG_NU_MINUS_ONE_STEP!r}")
            target, expect = (f"{names['nu']}_interval__", 0), "other"
        elif par in ("zi", "hu"):                              # the zero-inflation / hurdle probability (B06), brms
            if par not in names:                               # real<lower=0,upper=1>: Stan's logit, PyMC's log-odds
                raise OracleMismatch(f"brms samples {par} but the AutoBayes model has no {par}")
            if abs(float(e["du"]) - LOGIT_STEP) > 1e-9:
                raise OracleMismatch(f"brms's probe moved the coordinate of {par} by {e['du']!r}; logit({par}) moves it by {LOGIT_STEP!r}")
            for q, pt_ in enumerate(fixture.points):          # and at every point, brms's coordinate is logit of its value
                p = float(pt_.pars[par])
                if abs(math.log(p / (1.0 - p)) - pt_.u[idx]) > 1e-10:
                    raise OracleMismatch(f"point {q}: logit({par}) is {math.log(p / (1.0 - p))!r}; brms's unconstrained value is {pt_.u[idx]!r}")
            target, expect = (f"{names[par]}_logodds__", 0), "other"
        elif par.startswith("sd_") and int(par[3:]) in by_id:
            g = by_id[int(par[3:])]
            M = int(fixture.standata[f"M_{par[3:]}"])
            c, b = elem % M, elem // M                          # D08: gr(by = )'s sd_m is (M_m, Nby_m), column-major
            dpar = by_dpar_cn.get((g, c + 1))                   # D08: a shared ID's distributional coefficient
            if dpar is not None:
                target, expect = (f"{model.dpar_group_names[dpar]['sd']}_log__", 0), "log"
            elif (g, c + 1) not in by_cn:
                raise OracleMismatch(f"{par}[{elem + 1}] is not the SD of a group-level coefficient of {g!r}")
            else:
                target, expect = (f"{roles[(g, by_cn[(g, c + 1)])]['sd']}_log__", b), "log"
        elif par.startswith("z_") and int(par[2:]) in by_id:
            g = by_id[int(par[2:])]
            M = pars[par].shape[0]
            c, j = elem % M, elem // M                          # column-major flat index of (M, N)
            dpar = by_dpar_cn.get((g, c + 1))                   # D08: a shared ID's distributional coefficient
            if dpar is not None:
                target, expect = (model.dpar_group_names[dpar]["offset"], int(gmap.level_of[g][j])), "identity"
            elif (g, c + 1) not in by_cn:
                raise OracleMismatch(f"{par} coefficient {c + 1} is not a group-level coefficient of {g!r}")
            else:
                target, expect = (roles[(g, by_cn[(g, c + 1)])]["offset"], int(gmap.level_of[g][j])), "identity"
        elif par.startswith("L_") and int(par[2:]) in by_id:     # the Cholesky factor of a correlated factor (A02)
            g = by_id[int(par[2:])]
            if g not in model.correlation_names:
                raise OracleMismatch(f"{par} is brms's correlation Cholesky factor of {g!r}, whose AutoBayes term is not correlated")
            perm = correlation_order(model, gmap, g)
            if perm != list(range(len(perm))):
                raise OracleMismatch(
                    f"{g!r}: brms orders the coefficients as {[by_cn.get((g, c + 1), by_dpar_cn.get((g, c + 1))) for c in range(len(perm))]}, "
                    f"AutoBayes as {list(model.correlation_coefs(g))}; the cholesky_corr coordinates of {par} are not comparable")
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
            # a factor of three or more coefficients (D08's shared ID): the probe also sees later coordinates move ('joint')
            target, expect = (f"{model.correlation_names[g]['L']}_cholesky_corr__", k), ("joint" if transform == "joint" and len(perm) > 2 else "other")
        elif (gp := re.fullmatch(r"(sdgp|lscale|zgp)_(\d+)", par)) and int(gp.group(2)) in by_gp:   # C03: a Gaussian process's
            roles = model.gp_names[by_gp[int(gp.group(2))]]
            if gp.group(1) == "zgp":                           # zgp_k in the order of the distinct inputs or basis functions
                target, expect = (roles["zgp"], elem), "identity"
            elif elem != 0:
                raise OracleMismatch(f"{par}[{elem + 1}]: brms has several {gp.group(1)} for one Gaussian process; AutoBayes has one")
            else:                                              # sdgp_k, lscale_k: real<lower=0>, sampled on the log scale
                target, expect = (f"{roles[gp.group(1)]}_log__", 0), "log"
        else:
            raise OracleMismatch(f"no adapter for brms parameter {par!r} ({fixture.declarations.get(par, 'undeclared')})")
        if transform != expect:
            raise OracleMismatch(f"brms transform of {par}[{elem + 1}] is {transform!r}; AutoBayes uses {expect!r}")
        out.append((idx, *target))
    return out


def _simplex_coordinates(fixture: Fixture, model: ab.Model, cmap: ColumnMap) -> dict[int, tuple[str, int]]:
    """B03: brms's simplex ``simo_k`` (``simplex[Jmo[k]]``) has ``Jmo[k] - 1`` unconstrained coordinates, none of which the
    probe can move alone (changing one entry leaves the simplex: ``unreached``). Give each declared simplex, in declaration
    order, the next ``Jmo[k] - 1`` unreached coordinates, then recompute Stan's stick-breaking coordinates from ``simo_k`` at
    every point and require them to be the fixture's ``u`` there, so that the AutoBayes transform is Stan's coordinate by
    coordinate before the gradient is compared. Returns (value variable, index) by brms's 0-based coordinate."""
    unreached = [int(e["index"]) - 1 for e in fixture.upars if e["transform"] == "unreached"]
    by_k = {k: c for c, k in (cmap.monotonic or {}).items()}
    out, pos = {}, 0
    for name, declaration in fixture.declarations.items():
        if not declaration.startswith("simplex["):
            continue
        if cmap.mixture and name == "theta":                  # C05: a mixture's mixing proportions, simplex[K] theta
            size = len(model.components) - 1
            index = unreached[pos:pos + size]
            pos += size
            if len(index) != size:
                raise OracleMismatch(f"brms's simplex {name!r} needs {size} unreached coordinates; {len(index)} are left")
            for q, pt_ in enumerate(fixture.points):
                u = stick_breaking_free(np.atleast_1d(pt_.pars[name]))
                if np.max(np.abs(u - pt_.u[index])) > 1e-10:
                    raise OracleMismatch(f"point {q}: Stan's stick-breaking coordinates of {name} are {u!r}; brms's unconstrained "
                                         f"values at {[i + 1 for i in index]} are {pt_.u[index]!r}")
            for j, i in enumerate(index):
                out[i] = (f"{model.mixture_names['theta']}_stickbreaking__", j)
            continue
        k = int(name[len("simo_"):]) if name.startswith("simo_") and name[len("simo_"):].isdigit() else None
        if k not in by_k:
            raise OracleMismatch(f"brms's simplex {name!r} ({declaration}) is not the simplex of a monotonic term of the AutoBayes model")
        column = by_k[k]
        size = model.design.monotonic[column].D - 1
        index = unreached[pos:pos + size]
        pos += size
        if len(index) != size:
            raise OracleMismatch(f"brms's simplex {name!r} needs {size} unreached coordinates; {len(index)} are left")
        for q, pt_ in enumerate(fixture.points):
            u = stick_breaking_free(np.atleast_1d(pt_.pars[name]))
            if np.max(np.abs(u - pt_.u[index])) > 1e-10:
                raise OracleMismatch(f"point {q}: Stan's stick-breaking coordinates of {name} are {u!r}; brms's unconstrained values "
                                     f"at {[i + 1 for i in index]} are {pt_.u[index]!r}")
        for j, i in enumerate(index):
            out[i] = (f"{model.mo_names[column]['simplex']}_stickbreaking__", j)
    if pos != len(unreached):
        raise OracleMismatch(f"brms has {len(unreached) - pos} unreached unconstrained coordinates that belong to no simplex")
    return out


# -- draws --------------------------------------------------------------------------------------------------
def draws(fixture: Fixture, model: ab.Model, cmap: ColumnMap, gmap: GroupMap) -> az.InferenceData:
    """The fixture's points as one chain of posterior draws under AutoBayes's names: the raw-scale intercept
    (brms ``b_Intercept``), coefficients by term, group SDs and effects by level for every group-level
    coefficient (``1|g`` from ``r_m_1``, ``x|g`` from the slope's ``r_m_c``), the Cholesky factor and correlation
    matrix of a correlated factor (``L|g`` from ``L_m``, ``cor|g`` from ``L_m L_m^T``), sigma, shape or phi, a learned nu,
    and zi or hu (B06); the cumulative thresholds as brms's ``Intercept`` and ``b_Intercept``, which a fit's posterior
    keeps under those names, and a categorical model's intercepts and coefficients per category (B05); each monotonic term's
    coefficient and simplex (B03)."""
    if is_multivariate(model):                                 # C01: each response's draws under its names, Lrescor and rescor
        return response_draws(fixture, model, cmap, gmap)
    if is_nonlinear(model):                                   # C02
        return nonlinear_draws(fixture, model, cmap, gmap)
    pts = fixture.points
    names = model.names
    data, coords = {}, {"chain": [0], "draw": np.arange(len(pts))}
    if cmap.mixture:                                           # C05: the components' intercepts and coefficients as categories',
        roles = model.mixture_names                            # each sigma<k> and theta
        for s in roles["sigma"]:
            data[names[s]] = (("chain", "draw"), np.array([float(p.pars[s]) for p in pts])[None, :])
        data[roles["theta"]] = (("chain", "draw", roles["coordinate"]), np.stack([np.atleast_1d(p.pars["theta"]) for p in pts])[None])
        return _category_draws(fixture, model, cmap, data, coords, axis=roles["coordinate"], labels=roles["labels"])
    if cmap.dpars:                                             # B05: the categorical family, one column per category
        return _category_draws(fixture, model, cmap, data, coords)
    if model.spec.intercept and cmap.thresholds:               # B05: the thresholds, centred (Intercept) and raw (b_Intercept)
        dims = ("chain", "draw", names["threshold_dim"])
        data[names["intercept"]] = (dims, np.stack([np.atleast_1d(p.pars["Intercept"]) for p in pts])[None])
        data[names["thresholds"]] = (dims, np.stack([np.atleast_1d(p.pars["b_Intercept"]) for p in pts])[None])
        coords[names["threshold_dim"]] = [str(k) for k in range(1, len(model.categories))]
    elif model.spec.intercept:
        a = [float(p.pars["b_Intercept"]) if cmap.centred_intercept else float(coefficients(p.pars)[cmap.b_index(cmap.intercept_column)]) for p in pts]
        data[names["intercept"]] = (("chain", "draw"), np.array(a)[None, :])
    for column, k in (cmap.monotonic or {}).items():            # B03: each monotonic term's coefficient bsp[k] and simplex simo_k
        roles = model.mo_names[column]
        data[roles["b"]] = (("chain", "draw"), np.array([float(np.atleast_1d(p.pars["bsp"])[k - 1]) for p in pts])[None, :])
        data[roles["simplex"]] = (("chain", "draw", roles["coordinate"]), np.stack([np.atleast_1d(p.pars[f"simo_{k}"]) for p in pts])[None])
        coords[roles["coordinate"]] = [str(j) for j in range(1, model.design.monotonic[column].D + 1)]
    for column, k in (cmap.gp or {}).items():                   # C03: each Gaussian process's sdgp_k, lscale_k and zgp_k
        roles = model.gp_names[column]
        for key in ("sdgp", "lscale"):
            data[roles[key]] = (("chain", "draw"), np.array([float(np.ravel(p.pars[f"{key}_{k}"])[0]) for p in pts])[None, :])
        zs = np.stack([np.ravel(p.pars[f"zgp_{k}"]).astype(float) for p in pts])
        data[roles["zgp"]] = (("chain", "draw", roles["coordinate"]), zs[None])
        coords[roles["coordinate"]] = [str(j) for j in range(1, zs.shape[1] + 1)]
    for k, name in enumerate(cmap.special or ()):               # C06: the me() / mi() term's coefficient, brms's bsp[1]
        data[name] = (("chain", "draw"), np.array([float(np.atleast_1d(p.pars["bsp"])[k]) for p in pts])[None, :])
    for x, k in (cmap.measurement or {}).items():               # C06: an me() term's meanme_k, sdme_k and zme_k
        roles = model.me_names[x]
        for cls, key in (("meanme", "mean"), ("sdme", "sd")):
            data[roles[key]] = (("chain", "draw"), np.array([float(np.atleast_1d(p.pars[f"{cls}_{k}"])[0]) for p in pts])[None, :])
        data[roles["z"]] = (("chain", "draw", "__obs__"), np.stack([np.atleast_1d(p.pars[f"zme_{k}"]) for p in pts])[None])
        coords["__obs__"] = np.arange(model.design.n)
    for term in (t.name for t in model.spec.bambi_terms):       # the terms with design columns (a monotonic term has none)
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
        r = np.empty((len(pts), len(gmap.level_of[g])))
        for d, p in enumerate(pts):
            r[d, gmap.level_of[g]] = np.atleast_1d(p.pars[f"r_{m}_{cn}"])
        if "by_coordinate" in roles:                             # D08: gr(by = ): brms's sd_m row c, one SD per by level
            sd = np.stack([np.atleast_2d(p.pars[f"sd_{m}"])[cn - 1] for p in pts])
            data[roles["sd"]] = (("chain", "draw", roles["by_coordinate"]), sd[None])
            coords[roles["by_coordinate"]] = list(model.design.group_blocks[(g, coef)].by_levels)
        else:
            sd = np.array([np.atleast_1d(p.pars[f"sd_{m}"])[cn - 1] for p in pts])
            data[roles["sd"]] = (("chain", "draw"), sd[None, :])
        data[roles["group_effect"]] = (("chain", "draw", roles["coordinate"]), r[None])
        coords[roles["coordinate"]] = list(model.design.group_blocks[(g, coef)].levels)
    for (name, g, coef), roles in getattr(model, "dpar_group_names", {}).items():   # D08: a shared ID's distributional coefficients
        m, cn = gmap.brms_id[g], gmap.dpar_cn[(name, g, coef)]
        r = np.empty((len(pts), len(gmap.level_of[g])))
        for d, p in enumerate(pts):
            r[d, gmap.level_of[g]] = np.atleast_1d(p.pars[f"r_{m}_{name}_{cn}"])
        data[roles["sd"]] = (("chain", "draw"), np.array([np.atleast_1d(p.pars[f"sd_{m}"])[cn - 1] for p in pts])[None, :])
        data[roles["group_effect"]] = (("chain", "draw", roles["coordinate"]), r[None])
    for g, roles in model.correlation_names.items():             # L and the correlation matrix of a correlated factor (A02)
        L = np.stack([brms_cholesky(model, gmap, g, p.pars) for p in pts])
        dims = ("chain", "draw", roles["coordinate"], roles["coordinate_T"])
        data[roles["L"]] = (dims, L[None])
        data[roles["cor"]] = (dims, (L @ np.swapaxes(L, -1, -2))[None])
        coords[roles["coordinate"]] = coords[roles["coordinate_T"]] = list(model.correlation_coefs(g))
    if "sigma" in names and "sigma" in pts[0].pars:
        data[names["sigma"]] = (("chain", "draw"), np.array([float(p.pars["sigma"]) for p in pts])[None, :])
    if "shape" in names and "shape" in pts[0].pars:
        data[names["shape"]] = (("chain", "draw"), np.array([float(p.pars["shape"]) for p in pts])[None, :])
    if "phi" in names and "phi" in pts[0].pars:
        data[names["phi"]] = (("chain", "draw"), np.array([float(p.pars["phi"]) for p in pts])[None, :])
    if "nu" in names and "nu" in pts[0].pars:
        data[names["nu"]] = (("chain", "draw"), np.array([float(p.pars["nu"]) for p in pts])[None, :])
    for par in ("zi", "hu"):                                    # the zero-inflation / hurdle probability (B06)
        if par in names and par in pts[0].pars:
            data[names[par]] = (("chain", "draw"), np.array([float(p.pars[par]) for p in pts])[None, :])
    for part, order in (model.spec.autocor.parts if model.spec.autocor is not None else ()):   # C04: brms's ar and ma
        roles = model.autocor_names
        data[roles[part]] = (("chain", "draw", roles[f"{part}_coordinate"]), np.stack([np.atleast_1d(p.pars[part]) for p in pts])[None])
        coords[roles[f"{part}_coordinate"]] = [str(k) for k in range(1, order + 1)]
    for name, dmap in (cmap.distributional or {}).items():       # B01: an auxiliary parameter's own formula, raw-scale intercept and b
        ds, roles = model.spec.dpar(name), model.dpar_names[name]
        if ds.intercept:
            a = [float(p.pars[f"b_{name}_Intercept"]) if dmap.centred_intercept
                 else float(coefficients(p.pars, f"b_{name}")[dmap.b_index(dmap.intercept_column)]) for p in pts]
            data[roles["intercept"]] = (("chain", "draw"), np.array(a)[None, :])
        for t in ds.pop_terms:
            positions = [(c, j) for c, (_, term, j) in enumerate(ds.column_terms) if term == t.name]
            vals = np.array([[coefficients(p.pars, f"b_{name}")[dmap.b_index(dmap.x_column[c])] for c, _ in positions] for p in pts])
            var = roles[f"b:{t.name}"]
            if t.is_vector:
                data[var] = (("chain", "draw", f"{var}_dim"), vals[None])
                coords[f"{var}_dim"] = list(t.levels)
            else:
                data[var] = (("chain", "draw"), vals[:, 0][None, :])
    return az.InferenceData(posterior=xr.Dataset(data, coords=coords))


def _category_draws(fixture: Fixture, model: ab.Model, cmap: ColumnMap, data: dict, coords: dict, axis: str | None = None,
                    labels=None) -> az.InferenceData:
    """``draws`` for the categorical family (B05): the raw-scale intercept of every non-reference category (brms
    ``b_mu<k>_Intercept``, as Bambi's posterior holds it after uncentring) and every coefficient with the categories as its
    last axis, brms's ``b_mu<k>`` in column k; the family takes population-level terms only. A mixture's components (C05)
    take the same layout over the dimension ``axis`` with coordinates ``labels``."""
    pts, names, category = fixture.points, model.names, axis or model.names["category_dim"]
    coords[category] = list(labels) if labels is not None else list(model.categories[1:])
    if model.spec.intercept:
        a = np.array([[float(p.pars[f"b_{dp}_Intercept"]) if cmap.centred_intercept
                       else float(coefficients(p.pars, f"b_{dp}")[cmap.b_index(cmap.intercept_column)]) for dp in cmap.dpars]
                      for p in pts])
        data[names["intercept"]] = (("chain", "draw", category), a[None])
    for term in model.spec.terms:
        positions = [(c, j) for c, (_, t, j) in enumerate(model.coefficients) if t == term]
        vals = np.array([[[coefficients(p.pars, f"b_{dp}")[cmap.b_index(cmap.x_column[c])] for dp in cmap.dpars] for c, _ in positions]
                         for p in pts])                                        # (draw, columns of the term, categories)
        if model.spec.is_vector_term(term):
            data[term] = (("chain", "draw", f"{term}_dim", category), vals[None])
            coords[f"{term}_dim"] = list(model.spec.levels_of(term))
        else:
            data[term] = (("chain", "draw", category), vals[None, :, 0, :])
    return az.InferenceData(posterior=xr.Dataset(data, coords=coords))


# -- multivariate models (C01) ------------------------------------------------------------------------------
def is_multivariate(model) -> bool:
    """Whether ``model`` is a multivariate model (``ab.MultivariateModel``, brms's ``mvbind(y, y2) ~ ...``)."""
    return bool(getattr(model, "responses", ()))


class ResponseView(Fixture):
    """A multivariate fixture seen as brms's univariate model of one of its responses (C01), so that the univariate matching
    above applies to each response unchanged. brms names a response's data and parameters after it (``N_y``, ``Y_y``, ``K_y``,
    ``Kc_y``, ``X_y``, ``offsets_y``, ``J_1_y``, ``Z_1_y_1``; ``b_y``, ``Intercept_y``, ``sigma_y``, ``b_y_Intercept``,
    ``r_1_y_1``) and gives its group terms (``groups[*].resp``) global ids ``m``, whose ``N_m``, ``M_m``, ``NC_m``, ``sd_m``,
    ``z_m``, ``L_m`` and ``Cor_m`` the view keeps. ``data_names`` and ``par_names`` map brms's names to the univariate ones; the
    unconstrained coordinates keep their indices into the joint model, and a prediction is the response's slice of the last
    axis (``responses``)."""

    def __init__(self, parent: Fixture, response: str):
        super().__init__(parent.raw)
        self.parent, self.response = parent, response
        r, groups = response, [g for g in parent.groups if g.get("resp") == response]
        # C06: a response's special terms (Ksp_y, bsp_y: its mi() term) and its own missing values (Jmi_xm, Nmi_xm, lbmi_xm,
        # ubmi_xm; Ymi_xm), under brms's univariate names
        self.data_names = {f"{base}_{r}": base for base in ("N", "Y", "K", "Kc", "X", "offsets", "Ksp", "Jmi", "Nmi", "lbmi", "ubmi")}
        self.par_names = {f"b_{r}": "b", f"Intercept_{r}": "Intercept", f"sigma_{r}": "sigma", f"b_{r}_Intercept": "b_Intercept",
                          f"bsp_{r}": "bsp", f"Ymi_{r}": "Ymi"}
        for g in groups:
            m, cn = int(g["id"]), int(g["cn"])
            self.data_names.update({f"J_{m}_{r}": f"J_{m}", f"Z_{m}_{r}_{cn}": f"Z_{m}_{cn}", **{f"{k}_{m}": f"{k}_{m}" for k in ("N", "M", "NC")}})
            self.par_names.update({f"r_{m}_{r}_{cn}": f"r_{m}_{cn}", **{f"{k}_{m}": f"{k}_{m}" for k in ("sd", "z", "L", "Cor", "cor")}})

    @cached_property
    def standata(self) -> dict[str, np.ndarray]:
        sd = self.parent.standata
        return {new: sd[old] for old, new in self.data_names.items() if old in sd}

    def colnames(self, name: str) -> list[str] | None:
        old = next((o for o, n in self.data_names.items() if n == name and o in self.parent.raw["standata"]), None)
        return None if old is None else self.parent.colnames(old)

    def point(self, p: Point) -> Point:
        """The response's view of a point: its parameters under the univariate names; ``u``, the log density and gradient are
        the joint model's."""
        return replace(p, pars={new: p.pars[old] for old, new in self.par_names.items() if old in p.pars})

    @cached_property
    def points(self) -> list[Point]:
        return [self.point(p) for p in self.parent.points]

    @property
    def upars(self) -> list[dict]:
        return [{**e, "par": self.par_names[e["par"]]} for e in self.parent.upars if e["par"] in self.par_names]

    @property
    def groups(self) -> list[dict]:
        return [g for g in self.parent.groups if g.get("resp") == self.response]

    @property
    def declarations(self) -> dict[str, str]:
        return {self.par_names[k]: v for k, v in self.parent.declarations.items() if k in self.par_names}

    def prediction(self, block: str, kind: str) -> np.ndarray | None:
        full = self.parent.prediction(block, kind)
        return None if full is None else full[..., list(self.parent.raw["responses"]).index(self.response)]


@dataclass(frozen=True)
class ResponseMaps:
    """A multivariate model's correspondence (C01): each response's view of the fixture (``views``) and its univariate map
    (``maps``: a ``ColumnMap`` from ``match_columns``, a ``GroupMap`` from ``match_groups``)."""
    views: dict
    maps: dict


_GLOBAL_DATA = ("N", "nresp", "nrescor", "prior_only")        # a multivariate model's data that no single response owns


def _response_views(fixture: Fixture, model) -> dict[str, ResponseView]:
    """Each response's view, after checking that brms's responses are the model's, in order (the last axis of the predictions),
    that every group term belongs to a response, and that every entry of brms's data is a response's or ``_GLOBAL_DATA``."""
    responses = list(fixture.raw.get("responses") or [])
    if responses != list(model.responses):
        raise OracleMismatch(f"brms's responses are {responses}; the AutoBayes model's {list(model.responses)}")
    stray = [(g["group"], g["coef"]) for g in fixture.groups if g.get("resp") not in responses]
    if stray:
        raise OracleMismatch(f"brms group-level terms of no response (groups[*].resp): {stray}")
    views = {r: ResponseView(fixture, r) for r in responses}
    claimed = {old for v in views.values() for old in v.data_names}
    unclaimed = sorted(k for k in fixture.standata if k not in claimed and k not in _GLOBAL_DATA)
    if unclaimed:
        raise OracleMismatch(f"brms data without an AutoBayes counterpart in the multivariate model: {unclaimed}")
    return views


def match_response_columns(fixture: Fixture, model) -> ResponseMaps:
    views = _response_views(fixture, model)
    return ResponseMaps(views, {r: match_columns(views[r], sub) for r, sub in model.submodels.items()})


def match_response_groups(fixture: Fixture, model) -> ResponseMaps:
    views = _response_views(fixture, model)
    return ResponseMaps(views, {r: match_groups(views[r], sub) for r, sub in model.submodels.items()})


def response_coordinate_map(fixture: Fixture, model, cmap: ResponseMaps, gmap: ResponseMaps) -> list[tuple[int, str, int]]:
    """``coordinate_map`` of a multivariate model (C01): each response's coordinates from its view, renamed to the multivariate
    graph's variables, and those of ``Lrescor``, brms's ``cholesky_factor_corr[nresp]``, which are Stan's ``cholesky_corr``
    coordinates recomputed from ``Lrescor`` at every point and checked against the fixture's ``u`` (as ``L_m``'s)."""
    out = []
    for r, sub in model.submodels.items():                     # C06: the missing values keep brms's name, Ymi_<col> (value_name)
        out += [(idx, model.value_name(r, var), j) for idx, var, j in coordinate_map(cmap.views[r], sub, cmap.maps[r], gmap.maps[r])]
    k = 0
    for e in fixture.upars:
        if e["par"] != "Lrescor":
            continue
        if not model.rescor:
            raise OracleMismatch("brms samples Lrescor (a residual correlation) but the AutoBayes model has rescor=False")
        idx = int(e["index"]) - 1
        if e["transform"] != "other":
            raise OracleMismatch(f"brms transform of Lrescor[{e['elem']}] is {e['transform']!r}; AutoBayes uses 'other' (cholesky_corr)")
        for q, pt_ in enumerate(fixture.points):
            u = cholesky_corr_free(np.atleast_2d(pt_.pars["Lrescor"]))
            if k >= u.size or abs(u[k] - pt_.u[idx]) > 1e-10:
                raise OracleMismatch(f"point {q}: Stan's cholesky_corr coordinate {k} of Lrescor is {u[k] if k < u.size else 'absent'!r}; "
                                     f"brms's unconstrained value is {pt_.u[idx]!r}")
        out.append((idx, "Lrescor_cholesky_corr__", k))
        k += 1
    mapped = {idx for idx, _, _ in out}
    missing = [(e["par"], e["elem"]) for e in fixture.upars if int(e["index"]) - 1 not in mapped]
    if missing:
        raise OracleMismatch(f"brms's unconstrained coordinates without an AutoBayes counterpart: {missing}")
    return sorted(out)


def response_draws(fixture: Fixture, model, cmap: ResponseMaps, gmap: ResponseMaps) -> az.InferenceData:
    """``draws`` of a multivariate model (C01): each response's draws from its view under its names in the multivariate graph,
    and ``Lrescor`` with ``rescor = Lrescor Lrescor^T`` on the response coordinates."""
    from autobayes.multivariate import RESPONSE_DIM, RESPONSE_DIM_T

    parts = []
    for r, sub in model.submodels.items():
        post = draws(cmap.views[r], sub, cmap.maps[r], gmap.maps[r]).posterior
        parts.append(post.rename({v: model.variable(r, v) for v in post.data_vars}))
    post = xr.merge(parts)
    for col, roles in getattr(model, "missing_names", {}).items():   # C06: the missing values, brms's Ymi_<col>, on their rows
        rows = model.submodels[col].design.missing[col].index
        post = post.assign({roles["values"]: (("chain", "draw", roles["coordinate"]),
                                              np.stack([np.atleast_1d(p.pars[f"Ymi_{col}"]) for p in fixture.points])[None])})
        post = post.assign_coords({roles["coordinate"]: rows})
    if model.rescor:
        L = np.stack([np.atleast_2d(p.pars["Lrescor"]) for p in fixture.points])
        dims = ("chain", "draw", RESPONSE_DIM, RESPONSE_DIM_T)
        post = post.assign({"Lrescor": (dims, L[None]), "rescor": (dims, (L @ np.swapaxes(L, -1, -2))[None])})
        post = post.assign_coords({RESPONSE_DIM: list(model.responses), RESPONSE_DIM_T: list(model.responses)})
    return az.InferenceData(posterior=post)


# -- non-linear models (C02) ---------------------------------------------------------------------------------------
# brms's nl = TRUE: no design X; one design X_<nlpar>, its coefficients b_<nlpar> and K_<nlpar> per non-linear parameter
# (never centred: the intercept is a column of X_<nlpar> and an entry of b_<nlpar>), the covariates of the non-linear
# formula as C_1, C_2, ..., and each parameter's group-level terms as groups of their own, whose Z_<id>_<nlpar>_<k>, r_<id>_
# <nlpar>_<k> carry the parameter's name and whose rows of brms's group table carry it as `nlpar` (the fixture's extras
# "group_scope"). Everything is matched by content, never by position: covariates to C_k, design columns to X_<nlpar>'s
# columns, group-level coefficients to (J_<id>, Z_<id>_<nlpar>_<k>) within the parameter's own groups.
def is_nonlinear(model) -> bool:
    """A non-linear model (``ab.gaussian(..., nl=True)``, C02); a multivariate model (C01) has no single ``spec``."""
    return bool(getattr(getattr(model, "spec", None), "is_nonlinear", False))


@dataclass(frozen=True)
class NlparMap:
    """brms's ``X_<nlpar>`` column of each design column of one non-linear parameter (C02), in its design's column order."""
    x_column: tuple[int, ...]


def match_nonlinear_columns(fixture: Fixture, model: ab.Model) -> ColumnMap:
    """The column map of a non-linear model (C02): each covariate of the non-linear formula to the one brms ``C_k`` with its
    values, and each non-linear parameter's design columns, one by one, to the columns of brms's ``X_<nlpar>`` by content,
    with ``K_<nlpar>`` and ``b_<nlpar>`` checked against them. brms's linear design ``X``, a brms covariate or ``X_<nlpar>``
    without an AutoBayes counterpart, and the reverse, are mismatches."""
    sd, design = fixture.standata, model.design
    if "X" in sd:
        raise OracleMismatch("brms has the linear design X; the AutoBayes model is non-linear")
    brms_c = sorted(int(k[2:]) for k in sd if k.startswith("C_") and k[2:].isdigit())
    covariates, used = {}, set()
    for name, values in design.covariates.items():
        hits = [k for k in brms_c if k not in used and sd[f"C_{k}"].shape == values.shape and abs_err(sd[f"C_{k}"], values) <= DESIGN_ABS]
        if len(hits) != 1:
            raise OracleMismatch(f"covariate {name!r} of the non-linear formula matches {len(hits)} brms covariates C_k within {DESIGN_ABS}")
        covariates[name] = hits[0]
        used.add(hits[0])
    if used != set(brms_c):
        raise OracleMismatch(f"brms covariates without an AutoBayes counterpart: {[f'C_{k}' for k in sorted(set(brms_c) - used)]}")
    brms_nl = sorted(k[2:] for k in sd if k.startswith("X_"))
    if brms_nl != sorted(design.nlpars):
        raise OracleMismatch(f"brms has the non-linear parameter designs {['X_' + k for k in brms_nl]}; AutoBayes has the parameters "
                             f"{sorted(design.nlpars)}")
    pars, nlpars = fixture.points[0].pars, {}
    for name, nd in design.nlpars.items():
        X = np.atleast_2d(sd[f"X_{name}"]).reshape(nd.X.shape[0], -1)
        labels = fixture.colnames(f"X_{name}") or [str(k) for k in range(X.shape[1])]
        if X.shape != nd.X.shape:
            raise OracleMismatch(f"brms X_{name} is {X.shape}; AutoBayes's design of {name!r} is {nd.X.shape} ({list(nd.columns)})")
        taken, cols = set(), []
        for c, label in enumerate(nd.columns):
            hits = [k for k in range(X.shape[1]) if k not in taken and np.max(np.abs(X[:, k] - nd.X[:, c])) <= DESIGN_ABS]
            if len(hits) != 1:
                raise OracleMismatch(f"{name}'s design column {label!r} matches {len(hits)} brms X_{name} columns "
                                     f"{[labels[k] for k in hits]} within {DESIGN_ABS}")
            cols.append(hits[0])
            taken.add(hits[0])
        if int(sd[f"K_{name}"]) != X.shape[1] or coefficients(pars, f"b_{name}").size != X.shape[1]:
            raise OracleMismatch(f"brms K_{name} = {int(sd[f'K_{name}'])} and b_{name} has {coefficients(pars, f'b_{name}').size} "
                                 f"entries for {X.shape[1]} columns of X_{name}")
        if f"Intercept_{name}" in pars or f"Kc_{name}" in sd:
            raise OracleMismatch(f"brms centres the formula of {name} (Intercept_{name}, Kc_{name}); it never centres a non-linear "
                                 "parameter's formula, and AutoBayes does not")
        nlpars[name] = NlparMap(tuple(cols))
    return ColumnMap((), None, False, nlpars=nlpars, covariates=covariates)


def match_nonlinear_groups(fixture: Fixture, model: ab.Model) -> GroupMap:
    """The group map of a non-linear model (C02), keyed by parameter: every group-level coefficient ``(factor, coef)`` of
    parameter ``name``'s formula to the one brms group coefficient ``(id, cn)`` of that parameter (brms's group table, column
    ``nlpar``) whose ``J_<id>`` induces its partition of the rows and whose ``Z_<id>_<name>_<cn>`` is its covariate column; then
    the label, the level order, ``M_<id>``, the correlation flags and that every brms coefficient was used, as ``match_groups``
    checks them. Keys: ``brms_id[(name, factor)]``, ``level_of[(name, factor)]``, ``coef_cn[(name, factor, coef)]``."""
    if fixture.groups and any("nlpar" not in grp for grp in fixture.groups):
        raise OracleMismatch("brms's group table has no nlpar column; freeze the case with extras: ['group_scope']")
    ids, cns, levels_of, coef_cn, used = {}, {}, {}, {}, set()
    n = model.design.n
    for name, nd in model.design.nlpars.items():
        for (g, coef), block in nd.group_blocks.items():
            gidx = block.group_index
            covariate = block.Z[np.arange(n), gidx]
            found = []
            for grp in fixture.groups:
                key = (int(grp["id"]), int(grp["cn"]))
                if grp["nlpar"] != name or key in used:
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
                    Z = fixture.standata.get(f"Z_{grp['id']}_{name}_{grp['cn']}")
                    if Z is not None and Z.shape == (n,) and np.max(np.abs(Z - covariate)) <= DESIGN_ABS:
                        found.append((grp, m))
            if len(found) != 1:
                raise OracleMismatch(f"group-level coefficient {coef!r} of {g!r} in {name!r}: {len(found)} brms group terms of {name} "
                                     f"induce its partition of the rows and carry its covariate column within {DESIGN_ABS}")
            grp, m = found[0]
            if grp["coef"] != coef:
                raise OracleMismatch(f"{g!r} in {name!r}: the brms coefficient matched by content is labelled {grp['coef']!r}, "
                                     f"AutoBayes has {coef!r}")
            if [block.levels[i] for i in m] != list(grp["levels"]):
                raise OracleMismatch(f"{g!r} in {name!r}: brms levels {grp['levels']} vs AutoBayes {[block.levels[i] for i in m]}")
            if (name, g) in ids and ids[(name, g)] != int(grp["id"]):
                raise OracleMismatch(f"{g!r} in {name!r}: its coefficients match different brms groups ({ids[(name, g)]} and {grp['id']})")
            used.add((int(grp["id"]), int(grp["cn"])))
            ids[(name, g)], levels_of[(name, g)], coef_cn[(name, g, coef)] = int(grp["id"]), m, int(grp["cn"])
            if coef == "Intercept":
                cns[(name, g)] = int(grp["cn"])
    unmatched = [(grp["nlpar"], grp["group"], grp["coef"]) for grp in fixture.groups if (int(grp["id"]), int(grp["cn"])) not in used]
    if unmatched:
        raise OracleMismatch(f"brms group-level terms without an AutoBayes counterpart: {unmatched}")
    for name, t in model.spec.nlpar_group_terms:
        m = ids[(name, t.factor)]
        M = int(fixture.standata[f"M_{m}"])
        if M != len(t.coefs):
            raise OracleMismatch(f"{t.factor!r} in {name!r}: brms group {m} has M_{m} = {M} coefficients, AutoBayes {len(t.coefs)}")
        declared = f"L_{m}" in fixture.declarations
        if M > 1 and declared != t.has_correlation:
            raise OracleMismatch(f"{t.factor!r} in {name!r}: brms group {m} {'declares' if declared else 'has no'} L_{m}; AutoBayes's "
                                 f"term '{t.canonical}' is {'correlated' if t.has_correlation else 'independent'}")
    return GroupMap(ids, cns, levels_of, coef_cn)


def nonlinear_cholesky(model: ab.Model, gmap: GroupMap, name: str, factor: str, pars: dict) -> np.ndarray:
    """brms's ``L_<id>`` of parameter ``name``'s correlated term on ``factor`` in AutoBayes's coefficient order (C02)."""
    m = gmap.brms_id[(name, factor)]
    L = np.atleast_2d(pars[f"L_{m}"])
    perm = [gmap.coef_cn[(name, factor, c)] - 1 for c in model.spec.nlpar(name).group_term_of(factor).coefs]
    if perm == list(range(len(perm))):
        return L
    return np.linalg.cholesky((L @ L.T)[np.ix_(perm, perm)])


def nonlinear_natural_point(model: ab.Model, point: Point, cmap: ColumnMap, gmap: GroupMap) -> dict:
    """``NonlinearModel.value_point``'s arguments at a brms point (C02): per parameter ``b`` (brms's ``b_<nlpar>`` in the
    AutoBayes design's column order), ``sd`` and ``z`` of its group-level terms (``sd_<id>``, ``z_<id>`` in AutoBayes's level
    order) and ``L`` of its correlated ones; the family's auxiliary parameter by its brms name."""
    pars, out = point.pars, {}
    for d in model.spec.nlpars:
        name = d.response
        bb = coefficients(pars, f"b_{name}")
        entry = {"b": np.array([bb[k] for k in cmap.nlpars[name].x_column], dtype=float)}
        sd, z, L = {}, {}, {}
        for t in d.group_terms:
            m = gmap.brms_id[(name, t.factor)]
            sd_m, zb = np.atleast_1d(pars[f"sd_{m}"]), np.atleast_2d(pars[f"z_{m}"])
            per_sd, per_z = {}, {}
            for coef in t.coefs:
                cn = gmap.coef_cn[(name, t.factor, coef)]
                za = np.empty(zb.shape[1])
                za[gmap.level_of[(name, t.factor)]] = zb[cn - 1]
                per_sd[coef], per_z[coef] = float(sd_m[cn - 1]), za
            sd[t.factor], z[t.factor] = (per_sd, per_z) if t.slopes else (per_sd["Intercept"], per_z["Intercept"])
            if t.has_correlation:
                L[t.factor] = nonlinear_cholesky(model, gmap, name, t.factor, pars)
        if sd:
            entry.update(sd=sd, z=z)
        if L:
            entry["L"] = L
        out[name] = entry
    scale = model.family_spec.scale_parameter
    return {"nlpars": out, **({scale: float(pars[scale])} if scale is not None else {})}


def nonlinear_coordinate_map(fixture: Fixture, model: ab.Model, cmap: ColumnMap, gmap: GroupMap) -> list[tuple[int, str, int]]:
    """``coordinate_map`` of a non-linear model (C02): ``b_<nlpar>[k]`` to the variable and index of the AutoBayes design
    column brms's column k matched, ``sd_<id>`` / ``z_<id>`` / ``L_<id>`` to the parameter's group-level variables (``L_<id>``
    on Stan's ``cholesky_corr`` coordinates, recomputed from ``L_<id>`` at every point and checked against the fixture's
    ``u``), the auxiliary parameter to its log coordinate; each transform checked on both sides."""
    names, scale = model.names, model.family_spec.scale_parameter
    by_id = {m: key for key, m in gmap.brms_id.items()}                     # brms group id -> (parameter, factor)
    by_cn = {(name, g, cn): coef for (name, g, coef), cn in gmap.coef_cn.items()}
    pars = fixture.points[0].pars
    seen_L: dict[tuple[str, str], int] = {}
    out = []
    for e in fixture.upars:
        idx, transform = int(e["index"]) - 1, e["transform"]
        par, elem = e["par"], (int(e["elem"]) - 1 if e["elem"] is not None else None)
        if par is None:
            raise OracleMismatch(f"brms's unconstrained coordinate {idx + 1} was not reached by the probe")
        if par.startswith("b_") and par[2:] in (cmap.nlpars or {}):
            name = par[2:]
            col = cmap.nlpars[name].x_column.index(elem)
            _, var, j = model.nlpar_coefficients[name][col]
            target, expect = (var, 0 if j is None else j), "identity"
        elif par == scale:
            target, expect = (f"{names[scale]}_log__", 0), "log"
        elif par.startswith("sd_") and par[3:].isdigit() and int(par[3:]) in by_id:
            name, g = by_id[int(par[3:])]
            if (name, g, elem + 1) not in by_cn:
                raise OracleMismatch(f"{par}[{elem + 1}] is not the SD of a group-level coefficient of {g!r} in {name!r}")
            target, expect = (f"{model.nlpar_group_names[(name, g, by_cn[(name, g, elem + 1)])]['sd']}_log__", 0), "log"
        elif par.startswith("z_") and par[2:].isdigit() and int(par[2:]) in by_id:
            name, g = by_id[int(par[2:])]
            M = pars[par].shape[0]
            c, j = elem % M, elem // M                                   # column-major flat index of (M, N)
            if (name, g, c + 1) not in by_cn:
                raise OracleMismatch(f"{par} coefficient {c + 1} is not a group-level coefficient of {g!r} in {name!r}")
            target = (model.nlpar_group_names[(name, g, by_cn[(name, g, c + 1)])]["offset"], int(gmap.level_of[(name, g)][j]))
            expect = "identity"
        elif par.startswith("L_") and par[2:].isdigit() and int(par[2:]) in by_id:
            name, g = by_id[int(par[2:])]
            if (name, g) not in model.nlpar_correlation_names:
                raise OracleMismatch(f"{par} is brms's correlation Cholesky factor of {g!r} in {name!r}, whose AutoBayes term is not correlated")
            coefs = list(model.spec.nlpar(name).group_term_of(g).coefs)
            if [gmap.coef_cn[(name, g, c)] for c in coefs] != list(range(1, len(coefs) + 1)):
                raise OracleMismatch(f"{g!r} in {name!r}: brms orders the coefficients otherwise than AutoBayes {coefs}; the "
                                     f"cholesky_corr coordinates of {par} are not comparable")
            k = seen_L.get((name, g), 0)
            seen_L[(name, g)] = k + 1
            for q, pt_ in enumerate(fixture.points):
                u = cholesky_corr_free(np.atleast_2d(pt_.pars[par]))
                if k >= u.size or abs(u[k] - pt_.u[idx]) > 1e-10:
                    raise OracleMismatch(f"point {q}: Stan's cholesky_corr coordinate {k} of {par} is "
                                         f"{u[k] if k < u.size else 'absent'!r}; brms's unconstrained value is {pt_.u[idx]!r}")
            target, expect = (f"{model.nlpar_correlation_names[(name, g)]['L']}_cholesky_corr__", k), "other"
        else:
            raise OracleMismatch(f"no adapter for brms parameter {par!r} ({fixture.declarations.get(par, 'undeclared')}) of a "
                                 "non-linear model")
        if transform != expect:
            raise OracleMismatch(f"brms transform of {par}[{elem + 1}] is {transform!r}; AutoBayes uses {expect!r}")
        out.append((idx, *target))
    return out


def nonlinear_draws(fixture: Fixture, model: ab.Model, cmap: ColumnMap, gmap: GroupMap) -> az.InferenceData:
    """``draws`` of a non-linear model (C02): every parameter's coefficients under AutoBayes's names (``a_Intercept``, the
    vector ``a_z`` over ``a_z_dim``) from brms's ``b_<nlpar>``, its group SDs and effects by level (``a_1|g`` from
    ``r_<id>_<nlpar>_<cn>``), the Cholesky factor and correlation matrix of a correlated term, and the auxiliary parameter."""
    pts = fixture.points
    data, coords = {}, {"chain": [0], "draw": np.arange(len(pts))}
    for d in model.spec.nlpars:
        name, nmap = d.response, cmap.nlpars[d.response]
        rows = model.nlpar_coefficients[name]
        values = np.array([[coefficients(p.pars, f"b_{name}")[k] for k in nmap.x_column] for p in pts])   # (draw, column)
        for var in dict.fromkeys(v for _, v, _ in rows):
            positions = [(c, j) for c, (_, v, j) in enumerate(rows) if v == var]
            if positions[0][1] is None:
                data[var] = (("chain", "draw"), values[:, positions[0][0]][None, :])
            else:
                term = next(t for t in d.pop_terms if model.nlpar_names[name][f"b:{t.name}"] == var)
                vec = np.empty((len(pts), len(positions)))
                for c, j in positions:
                    vec[:, j] = values[:, c]
                data[var] = (("chain", "draw", f"{var}_dim"), vec[None])
                coords[f"{var}_dim"] = list(term.levels)
        for t in d.group_terms:
            m = gmap.brms_id[(name, t.factor)]
            for coef in t.coefs:
                roles, cn = model.nlpar_group_names[(name, t.factor, coef)], gmap.coef_cn[(name, t.factor, coef)]
                level_of = gmap.level_of[(name, t.factor)]
                r = np.empty((len(pts), len(level_of)))
                for q, p in enumerate(pts):
                    r[q, level_of] = np.atleast_1d(p.pars[f"r_{m}_{name}_{cn}"])
                data[roles["sd"]] = (("chain", "draw"), np.array([np.atleast_1d(p.pars[f"sd_{m}"])[cn - 1] for p in pts])[None, :])
                data[roles["group_effect"]] = (("chain", "draw", roles["coordinate"]), r[None])
                coords[roles["coordinate"]] = list(model.design.nlpars[name].group_blocks[(t.factor, coef)].levels)
            if t.has_correlation:
                croles = model.nlpar_correlation_names[(name, t.factor)]
                L = np.stack([nonlinear_cholesky(model, gmap, name, t.factor, p.pars) for p in pts])
                dims = ("chain", "draw", croles["coordinate"], croles["coordinate_T"])
                data[croles["L"]] = (dims, L[None])
                data[croles["cor"]] = (dims, (L @ np.swapaxes(L, -1, -2))[None])
                coords[croles["coordinate"]] = coords[croles["coordinate_T"]] = list(t.coefs)
    scale = model.family_spec.scale_parameter
    if scale is not None:
        data[model.names[scale]] = (("chain", "draw"), np.array([float(p.pars[scale]) for p in pts])[None, :])
    return az.InferenceData(posterior=xr.Dataset(data, coords=coords))


def brms_nlpar_values(fixture: Fixture, point: Point, name: str) -> np.ndarray:
    """brms's own linear predictor of non-linear parameter ``name`` at a point (C02), from its Stan data and parameters alone:
    ``X_<name> b_<name>`` plus, for every group coefficient of ``name`` in brms's group table, ``r_<id>_<name>_<cn>[J_<id>]``
    times ``Z_<id>_<name>_<cn>`` (brms's ``nlp_<name>``)."""
    sd = fixture.standata
    X = np.asarray(sd[f"X_{name}"], dtype=float).reshape(int(sd["N"]), -1)
    eta = X @ coefficients(point.pars, f"b_{name}")
    for grp in fixture.groups:
        if grp.get("nlpar") != name:
            continue
        m, cn = int(grp["id"]), int(grp["cn"])
        J = sd[f"J_{m}"].astype(int) - 1
        eta = eta + np.atleast_1d(point.pars[f"r_{m}_{name}_{cn}"])[J] * sd[f"Z_{m}_{name}_{cn}"]
    return eta


# -- autocorrelation terms (C04) ----------------------------------------------------------------------------
# brms's Stan data names of per-row arrays: the response, every design, the group indices and covariates, the offsets and the
# addition-term data, the monotonic codes (J_lag, brms's lags on its sorted rows, is checked on those rows instead)
_PER_ROW = re.compile(r"^(Y|X|X_\w+|Z_\d+_\d+|J_\d+|offsets|weights|cens|lb|ub|trials|Xmo_\d+)$")


def brms_row_order(case: dict, data: pd.DataFrame, term) -> np.ndarray:
    """brms's ``order_data()`` of an autocorrelation model: the data rows sorted by the series column and then by the time
    column (data row per Stan row), as R's ``order(gr, time)`` sorts them: a declared factor by its levels' order (the case's
    ``factors``), numbers numerically, anything else as C-locale text; the data order stands in for an absent time and a
    constant for an absent series column. Two rows of a series at the same time cannot be ordered: brms refuses them."""
    n = len(data)

    def key(column):
        levels = case["brms"].get("factors", {}).get(column)
        values = data[column]
        if levels is not None:
            return np.array([list(levels).index(str(v)) for v in values], dtype=float)
        if pd.api.types.is_numeric_dtype(values):
            return values.to_numpy(dtype=float)
        return np.unique(values.astype(str).to_numpy(), return_inverse=True)[1].astype(float)

    series = np.zeros(n) if term.gr is None else key(term.gr)
    time = np.arange(n, dtype=float) if term.time is None else key(term.time)
    order = np.lexsort((time, series))
    if np.any((series[order][1:] == series[order][:-1]) & (time[order][1:] == time[order][:-1])):
        raise OracleMismatch(f"{term.term}: two rows of a series share a time point; brms cannot order them")
    return order


class _DataOrderFixture(Fixture):
    """A fixture of an autocorrelation model whose per-row Stan data are back in data order (``in_data_order``):
    ``stan_order`` is brms's permutation (data row per Stan row); every other entry, ``J_lag`` included, is brms's own."""

    def __init__(self, raw: dict, standata: dict, stan_order: np.ndarray):
        super().__init__(raw)
        self._standata, self.stan_order = standata, stan_order

    @property
    def standata(self) -> dict[str, np.ndarray]:
        return self._standata


def in_data_order(case: dict, fixture: Fixture, model: ab.Model, data: pd.DataFrame) -> Fixture:
    """``fixture`` with its per-row Stan data in data order when the model has an autocorrelation term (C04), else itself.
    brms sorts such a model's rows by series and time (``brms_row_order``, derived from the data); brms's ``Y`` must be the
    response under that permutation, row for row, and every per-row array (``_PER_ROW``) is put back through it, so the
    matchers compare rows by content as for any other model and a wrong permutation fails them. An array of ``N`` rows
    that is not a known per-row array stops the oracle rather than being left sorted or moved blindly."""
    term = getattr(getattr(model, "spec", None), "autocor", None)   # a multivariate or non-linear model has none
    if term is None:
        return fixture
    order = brms_row_order(case, data, term)
    sd = fixture.standata
    N = int(sd["N"])
    if order.size != N:
        raise OracleMismatch(f"brms has N = {N} rows, the data {order.size}")
    y = data[model.spec.response].to_numpy(dtype=float)
    if abs_err(sd["Y"], y[order]) > DESIGN_ABS:
        raise OracleMismatch(f"brms's Y is not the response sorted by {term.columns} (brms's order_data()): it differs by "
                             f"{abs_err(sd['Y'], y[order])}")
    unsorted = {}
    for key, value in sd.items():
        if key == "J_lag" or np.ndim(value) == 0 or value.shape[0] != N:
            unsorted[key] = value
            continue
        if not _PER_ROW.match(key):
            raise OracleMismatch(f"brms's standata {key!r} has {N} rows but is not a known per-row array; cannot put it in data order")
        back = np.empty_like(value)
        back[order] = value
        unsorted[key] = back
    return _DataOrderFixture(fixture.raw, unsorted, order)


def check_autocor_design(fixture: Fixture, model: ab.Model) -> None:
    """G1 of an autocorrelation term (C04): brms's ``Kar`` and ``Kma`` are the term's ``p`` and ``q``; along brms's sorted
    rows each row either continues the previous row's series one position later or starts a series (a change of brms's
    series label) at position 0, in the model's series (``Design.autocor``); and brms's ``J_lag`` is, row by row, the lags
    those positions give the next row, ``min(max(p, q), position)``. A model and a fixture of which only one has ARMA data
    are a mismatch."""
    sd, term = fixture.standata, getattr(getattr(model, "spec", None), "autocor", None)
    brms_has = [k for k in ("Kar", "Kma", "J_lag") if k in sd]
    if term is None:
        if brms_has:
            raise OracleMismatch(f"brms has the ARMA data {brms_has}; the AutoBayes model has no autocorrelation term")
        return
    if len(brms_has) != 3:
        raise OracleMismatch(f"the AutoBayes model has {term.term}; brms's standata lacks {sorted({'Kar', 'Kma', 'J_lag'} - set(brms_has))}")
    if (int(sd["Kar"]), int(sd["Kma"])) != (term.p, term.q):
        raise OracleMismatch(f"brms has Kar = {int(sd['Kar'])}, Kma = {int(sd['Kma'])}; {term.term} has p = {term.p}, q = {term.q}")
    order, design = getattr(fixture, "stan_order", None), model.design.autocor
    if order is None:
        raise OracleMismatch("the fixture of an autocorrelation model must be compared in data order (in_data_order)")
    series, position = design.series_index[order], design.position[order]
    labels = (np.zeros(order.size) if term.gr is None else model.data[term.gr].astype(str).to_numpy()[order])
    for i in range(order.size):
        starts = i == 0 or labels[i] != labels[i - 1]
        if starts and (position[i] != 0 or (i and series[i] == series[i - 1])):
            raise OracleMismatch(f"Stan row {i + 1} starts brms's series {labels[i]!r}; AutoBayes puts it at position {position[i]} "
                                 f"of series {design.series[series[i]]!r}")
        if not starts and (series[i] != series[i - 1] or position[i] != position[i - 1] + 1):
            raise OracleMismatch(f"Stan row {i + 1} follows row {i} in brms's series {labels[i]!r}; AutoBayes puts it at position "
                                 f"{position[i]} of series {design.series[series[i]]!r}")
    expected = np.r_[np.minimum(term.max_lag, position[1:]), 0]
    J_lag = np.asarray(sd["J_lag"], dtype=float).ravel()
    if J_lag.shape != expected.shape or abs_err(J_lag, expected) > 0:
        raise OracleMismatch(f"brms's J_lag {J_lag.astype(int).tolist()} differs from the lags of AutoBayes's series positions "
                             f"{expected.tolist()}")
