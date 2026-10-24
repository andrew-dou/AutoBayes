"""Shrinkage priors on the population-level coefficients: brms's ``horseshoe()`` and ``R2D2()`` on class ``b`` (D07,
docs/shrinkage-priors.md).

Both are joint priors on the whole vector of coefficients ``b`` (every column of the design but the intercept), built
non-centred as brms builds them: ``b = zb * sdb`` with ``zb ~ Normal(0, 1)`` per coefficient and ``sdb`` the coefficient's
scale, a function of the prior's own parameters. ``b`` is then a transformed parameter (PyMC deterministics, one per
term, under the term's name as every coefficient variable), and the sampled coordinates are ``zb`` and the scales' own.

The regularised horseshoe (Piironen & Vehtari 2017, Electronic Journal of Statistics 11, 5018-5051), with brms's
autoscaling by the residual SD ``sigma``::

    hs_local[k] ~ half-Student-t(df, 0, 1)                    local scales, one per coefficient
    hs_global   ~ half-Student-t(df_global, 0, scale_global * sigma)
    hs_slab     ~ inverse-gamma(df_slab / 2, df_slab / 2),    c2 = scale_slab^2 * hs_slab
    sdb[k] = hs_global * sqrt(c2 * hs_local[k]^2 / (c2 + hs_global^2 * hs_local[k]^2))

The R2D2 prior (Zhang, Naughton, Bondell & Reich 2022, JASA 117, 862-874), with brms's autoscaling::

    R2D2_R2  ~ Beta(mean_R2 * prec_R2, (1 - mean_R2) * prec_R2)
    R2D2_phi ~ Dirichlet(cons_D2)                             a simplex over the coefficients
    R2D2_tau2 = sigma^2 * R2D2_R2 / (1 - R2D2_R2),            sdb[k] = sqrt(R2D2_phi[k] * R2D2_tau2)

The PyMC names are brms's Stan names (``zb``, ``hs_global``, ``hs_slab``, ``hs_local``, ``R2D2_R2``, ``R2D2_phi``, and the
deterministics ``sdb`` and ``R2D2_tau2``), the scales are sampled on their logs, ``R2D2_R2`` on its log-odds and
``R2D2_phi`` on Stan's stick-breaking coordinates (``monotonic.StickBreakingTransform``), so the sampled space and its
Jacobian are brms's. The vectors are in the model's design-column order (``Model.coefficients``) on the coordinate
``zb_dim``. Implemented from brms's documentation (``?horseshoe``, ``?R2D2``) and the papers above, and checked against
brms's generated program and ``rstan::log_prob`` on the frozen cases ``gauss_horseshoe`` and ``gauss_r2d2``; no brms
source is used.

What is offered is what those cases verify: the whole class ``b`` of a univariate Gaussian formula with an intercept and
population-level terms that are columns of the design (numeric, categorical, interactions, transforms), centred, with
brms's ``autoscale = TRUE``. Everything else is refused naming the construct (``check_prior``, ``check_model``).
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from autobayes.priors import Prior, PriorError, Priors

SHRINKAGE = frozenset({"Horseshoe", "R2D2"})
DOC = "docs/shrinkage-priors.md"
ZB, SDB, COORDINATE = "zb", "sdb", "zb_dim"
HORSESHOE_ARGUMENTS = ("df", "scale_global", "df_global", "scale_slab", "df_slab")   # brms's order
R2D2_ARGUMENTS = ("mean_R2", "prec_R2", "cons_D2")
_BRMS_NAME = {"Horseshoe": "horseshoe", "R2D2": "R2D2"}


def is_shrinkage(prior) -> bool:
    """Whether ``prior`` is a shrinkage prior (``Horseshoe`` or ``R2D2``)."""
    return isinstance(prior, Prior) and prior.name in SHRINKAGE


def _positive(fn: str, name: str, value) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise PriorError(f"{fn}: {name} must be a positive finite number, got {value!r}")
    value = float(value)
    if not (np.isfinite(value) and value > 0):
        raise PriorError(f"{fn}: {name} must be a positive finite number, got {value!r}")
    return value


def _brms_only_defaults(fn: str, *, par_ratio=None, autoscale=True, main=False) -> None:
    """brms's arguments that only their defaults are offered for, each refused naming it."""
    if par_ratio is not None:
        raise PriorError(f"{fn}(par_ratio = ...) is not offered: brms then replaces scale_global by par_ratio / sqrt(N); give "
                         f"scale_global yourself ({DOC}, divergence D101)")
    if autoscale is not True:
        raise PriorError(f"{fn}(autoscale = FALSE) is not offered: the frozen cases verify brms's autoscale = TRUE, the scale of "
                         f"the prior multiplied by the residual SD sigma ({DOC}, divergence D101)")
    if main is not False:
        raise PriorError(f"{fn}(main = TRUE) is not offered: it joins the prior across several parameter classes, and only "
                         f"class b takes it here ({DOC}, divergence D100)")


def Horseshoe(df: float = 1, scale_global: float = 1, df_global: float = 1, scale_slab: float = 2, df_slab: float = 4,
              par_ratio=None, autoscale: bool = True, main: bool = False) -> Prior:
    """brms's regularised horseshoe ``horseshoe(df, scale_global, df_global, scale_slab, df_slab)`` on the whole class ``b``
    (``Priors(b=ab.Horseshoe(...))``), with brms's defaults and argument order: ``df`` the degrees of freedom of the local
    scales' half-Student-t, ``scale_global`` and ``df_global`` the global scale's (its scale multiplied by the residual SD
    ``sigma``, brms's ``autoscale = TRUE``), ``scale_slab`` and ``df_slab`` the slab's (Piironen & Vehtari 2017). brms's
    ``par_ratio``, ``autoscale = FALSE`` and ``main = TRUE`` are refused (divergences D100, D101). See docs/shrinkage-priors.md."""
    _brms_only_defaults("horseshoe", par_ratio=par_ratio, autoscale=autoscale, main=main)
    values = (df, scale_global, df_global, scale_slab, df_slab)
    return Prior("Horseshoe", {k: _positive("horseshoe", k, v) for k, v in zip(HORSESHOE_ARGUMENTS, values)})


def R2D2(mean_R2: float = 0.5, prec_R2: float = 2, cons_D2=0.5, autoscale: bool = True, main: bool = False) -> Prior:
    """brms's ``R2D2(mean_R2, prec_R2, cons_D2)`` on the whole class ``b`` (``Priors(b=ab.R2D2(...))``), with brms's defaults:
    a Beta prior on the coefficient of determination R^2 with mean ``mean_R2`` (in (0, 1)) and precision ``prec_R2``, and a
    Dirichlet prior with concentrations ``cons_D2`` on the split of the explained variance over the coefficients: one
    positive number for every coefficient, or one per coefficient in the model's design-column order (``Model.coefficients``,
    as ``describe()`` lists them); the scale is ``sigma``'s (brms's ``autoscale = TRUE``; Zhang et al. 2022). brms's
    ``autoscale = FALSE`` and ``main = TRUE`` are refused (D100, D101). See docs/shrinkage-priors.md."""
    _brms_only_defaults("R2D2", autoscale=autoscale, main=main)
    mean = _positive("R2D2", "mean_R2", mean_R2)
    if not mean < 1:
        raise PriorError(f"R2D2: mean_R2 is the mean of a Beta prior on R^2 and must lie in (0, 1), got {mean!r}")
    params = {"mean_R2": mean, "prec_R2": _positive("R2D2", "prec_R2", prec_R2)}
    if isinstance(cons_D2, (list, tuple, np.ndarray)):
        values = np.asarray(cons_D2, dtype=object).reshape(-1)
        if values.size == 0:
            raise PriorError("R2D2: cons_D2 needs at least one concentration")
        params.update({f"cons_D2[{k}]": _positive("R2D2", "cons_D2", v) for k, v in enumerate(values, start=1)})
    else:
        params["cons_D2"] = _positive("R2D2", "cons_D2", cons_D2)
    return Prior("R2D2", params)


def concentration(prior: Prior, K: int) -> np.ndarray:
    """The ``K`` Dirichlet concentrations of an ``R2D2`` prior on a model of ``K`` coefficients (brms's ``R2D2_cons_D2``): one
    number repeated, or the vector given, which must have ``K`` entries."""
    if "cons_D2" in prior.params:
        return np.full(int(K), float(prior.params["cons_D2"]))
    values = [float(v) for k, v in prior.params.items() if k.startswith("cons_D2[")]
    if len(values) != int(K):
        raise PriorError(f"{prior} gives {len(values)} concentrations cons_D2 for a model of {K} coefficients; give {K}, one per "
                         "coefficient in the design-column order describe() lists, or one number for all")
    return np.asarray(values, dtype=float)


def brms_spelling(prior: Prior) -> str:
    """The prior in brms's syntax with every argument named, as brms's prior table shows a user's string:
    ``horseshoe(df = 3, scale_global = 0.5, df_global = 1, scale_slab = 2, df_slab = 4)``, ``R2D2(mean_R2 = 0.4, prec_R2 = 3,
    cons_D2 = 0.7)``; a vector of concentrations is ``cons_D2 = c(0.7, 0.5, ...)``."""
    from autobayes.default_priors import r_number

    if prior.name == "Horseshoe":
        parts = [f"{k} = {r_number(prior.params[k])}" for k in HORSESHOE_ARGUMENTS]
    else:
        parts = [f"{k} = {r_number(prior.params[k])}" for k in ("mean_R2", "prec_R2")]
        if "cons_D2" in prior.params:
            parts.append(f"cons_D2 = {r_number(prior.params['cons_D2'])}")
        else:
            parts.append("cons_D2 = c(" + ", ".join(r_number(v) for k, v in prior.params.items() if k.startswith("cons_D2[")) + ")")
    return f"{_BRMS_NAME[prior.name]}({', '.join(parts)})"


def names(prior) -> dict[str, str]:
    """The PyMC names of a shrinkage prior's variables (brms's Stan names): ``zb``, ``sdb`` and the coordinate ``zb_dim`` of
    the coefficients, and ``hs_global``, ``hs_slab``, ``hs_local`` (horseshoe) or ``R2D2_R2``, ``R2D2_phi``, ``R2D2_tau2``
    (R2D2); ``lprior`` lists the free variables whose densities brms sums into ``lprior`` (the others' are in its model
    block). Empty for any other prior."""
    if not is_shrinkage(prior):
        return {}
    out = {"prior": prior.name, "zb": ZB, "sdb": SDB, "coordinate": COORDINATE}
    if prior.name == "Horseshoe":
        out.update(hs_global="hs_global", hs_slab="hs_slab", hs_local="hs_local")
        out["lprior"] = ("hs_global", "hs_slab")
    else:
        out.update(R2D2_R2="R2D2_R2", R2D2_phi="R2D2_phi", R2D2_tau2="R2D2_tau2")
        out["lprior"] = ("R2D2_R2",)
    return out


# -- what is offered (the refusals) --------------------------------------------------------------------------------------
def _found(priors) -> list[tuple[str, Prior]]:
    """Every shrinkage prior anywhere in ``priors`` but its top-level ``b``, labelled by where it is."""
    out = []

    def walk(value, label):
        if is_shrinkage(value):
            out.append((label, value))
        elif isinstance(value, Mapping):
            for k, v in value.items():
                walk(v, f"{label}[{k!r}]")
        elif isinstance(value, (list, tuple)):
            for k, v in enumerate(value):
                walk(v, f"{label}[{k}]")
        elif isinstance(value, Priors):
            for name in value.__dataclass_fields__:
                walk(getattr(value, name), f"{label}.{name}")

    for name in priors.__dataclass_fields__:
        value = getattr(priors, name)
        if name == "b" and is_shrinkage(value):
            continue
        walk(value, name)
    return out


def check_prior(priors) -> None:
    """Refuse a shrinkage prior anywhere but as the bare ``b`` of a model's priors: on a single term or coefficient (a mapping
    under ``b``), on another class (``intercept``, ``sd``, ``sigma``, ...), on a distributional formula's ``b``, a mixture's
    component, a response of a multivariate model or a non-linear parameter."""
    found = _found(priors)
    if not found:
        return
    label, prior = found[0]
    what = (f"{prior.name}() applies to the whole class b of the response formula only (Priors(b=ab.{prior.name}(...))); "
            f"found at {label}")
    if label.startswith("b["):
        why = "a shrinkage prior on one term or coefficient is not offered: it is a joint prior over every coefficient of class b"
    elif label.startswith("sigma[") or label.startswith(("shape[", "phi[")):
        why = "a distributional formula's coefficients are refused (brms does not autoscale them; no frozen case verifies them)"
    elif label.startswith("components"):
        why = "a mixture's components are refused (no frozen case verifies them)"
    elif label.startswith("resp"):
        why = "a multivariate model's responses are refused (no frozen case verifies them)"
    elif label.startswith("nlpars"):
        why = "a non-linear parameter's coefficients are refused (no frozen case verifies them)"
    else:
        why = ("brms also offers it on sd, sds, sdgp, ar, ma, sderr and sdcar (and across classes with main = TRUE); none of "
               "those is offered here")
    raise PriorError(f"{what}: {why} ({DOC}, divergence D100)")


def check_model(prior: Prior, spec, fam) -> None:
    """Refuse every model a shrinkage prior on ``b`` is not verified for, naming the construct: a family without
    ``FamilySpec.shrinkage`` (the Gaussian has it: brms autoscales by its sigma), a formula without population-level terms or
    without an intercept, and group-level terms, an offset, response addition terms, distributional formulas, monotonic,
    measurement-error and missing-value terms (brms's class bsp, which brms's shrinkage would include), Gaussian processes and
    autocorrelation terms. ``spec`` is the parsed formula (the data need not be seen)."""
    fn = f"{prior.name}() on b"
    if not getattr(fam, "shrinkage", False):
        raise PriorError(f"{fn} is offered for the Gaussian family only, where the frozen cases verify it (brms scales it by the "
                         f"residual SD sigma); the {fam.name} family is refused ({DOC}, divergence D102)")
    if not spec.pop_terms:
        raise PriorError(f"{fn}: the formula '{spec.formula}' has no population-level coefficient for it to shrink (brms refuses "
                         "it too: no parameter of class b)")
    if not spec.intercept:
        raise PriorError(f"{fn}: a formula without an intercept is not offered (brms's b then holds every column of the "
                         f"design; no frozen case verifies it) ({DOC}, divergence D102)")
    refused = []
    if spec.group_terms:
        refused.append(f"group-level terms {[t.canonical for t in spec.group_terms]}")
    if spec.offset is not None:
        refused.append(f"an offset ({spec.offset.term})")
    if spec.additions:
        refused.append(f"response addition terms {[a.term for a in spec.additions]}")
    if spec.dpars:
        refused.append(f"distributional formulas for {list(spec.dpar_names)}")
    special = [t.name for t in spec.pop_terms if t.is_monotonic or t.kind in ("measurement_error", "missing")]
    if special:
        refused.append(f"the special terms {special} (brms's class bsp, whose coefficients brms's shrinkage includes)")
    if spec.gp_terms:
        refused.append(f"Gaussian processes {[g.term for g in spec.gp_terms]}")
    if spec.autocor is not None:
        refused.append(f"an autocorrelation term {spec.autocor.term}")
    if getattr(spec, "sar", None) is not None:                # F02b: not verified with it either (D143)
        refused.append(f"a SAR term {spec.sar.term}")
    if refused:
        raise PriorError(f"{fn} is offered with population-level terms that are columns of the design only; this formula has "
                         + "; ".join(refused) + f", which no frozen case verifies with it ({DOC}, divergence D102)")


def check_coefficients(prior: Prior, columns) -> None:
    """With the data seen: the coefficients' number suits the prior (``R2D2``'s simplex needs two; a vector of
    concentrations one per coefficient)."""
    K = len(columns)
    if prior.name == "R2D2":
        if K < 2:
            raise PriorError(f"R2D2() on b with a single coefficient ({list(columns)}) is not offered: brms's R2D2_phi is then a "
                             f"simplex of one entry, which this library does not build ({DOC}, divergence D102)")
        concentration(prior, K)


# -- the graph -----------------------------------------------------------------------------------------------------------
def _f64(x):
    return np.float64(x)   # PyTensor makes a Python number that float32 represents exactly a float32 constant (#45)


def build(prior: Prior, columns, scale):
    """The coefficients ``b = zb * sdb`` (a vector over ``columns``, PyTensor) and the variables they are built from, created
    in the PyMC model being built: ``scale`` is the residual SD's PyMC variable (brms's autoscale)."""
    import pymc as pm
    import pytensor.tensor as pt

    model = pm.modelcontext(None)
    if COORDINATE in model.coords:
        raise RuntimeError(f"the coordinate {COORDINATE!r} of a shrinkage prior's coefficients already exists in the graph")
    model.add_coord(COORDINATE, list(columns))
    K = len(columns)
    zb = pm.Normal(ZB, mu=_f64(0.0), sigma=_f64(1.0), shape=(K,), dims=COORDINATE)
    p = {k: float(v) for k, v in prior.params.items()}
    if prior.name == "Horseshoe":
        hs_global = pm.HalfStudentT("hs_global", nu=_f64(p["df_global"]), sigma=_f64(p["scale_global"]) * scale)
        hs_slab = pm.InverseGamma("hs_slab", alpha=_f64(0.5 * p["df_slab"]), beta=_f64(0.5 * p["df_slab"]))
        hs_local = pm.HalfStudentT("hs_local", nu=_f64(p["df"]), sigma=_f64(1.0), shape=(K,), dims=COORDINATE)
        c2 = _f64(p["scale_slab"] ** 2) * hs_slab
        lambda2 = pt.square(hs_local)
        sdb = hs_global * pt.sqrt(c2 * lambda2 / (c2 + pt.square(hs_global) * lambda2))
    else:
        from autobayes.monotonic import simplex_rv

        mean, prec = p["mean_R2"], p["prec_R2"]
        R2 = pm.Beta("R2D2_R2", alpha=_f64(mean * prec), beta=_f64((1.0 - mean) * prec))
        phi = simplex_rv("R2D2_phi", concentration(prior, K), COORDINATE)
        tau2 = pm.Deterministic("R2D2_tau2", pt.square(scale) * R2 / (_f64(1.0) - R2))
        sdb = pt.sqrt(phi * tau2)
    sdb = pm.Deterministic(SDB, sdb, dims=COORDINATE)
    return zb * sdb


def bambi_priors(prior: Prior, spec, fam) -> dict:
    """Bambi's prior of each population-level term under a shrinkage prior: a ``bmb.Prior`` whose ``dist`` returns the term's
    coefficients as a deterministic named after the term (its slice of ``b = zb * sdb``), the shared variables built once per
    PyMC model by the first term Bambi builds. The residual SD ``sigma`` exists by then: Bambi builds the family's auxiliary
    parameters before the linear predictor's terms."""
    import bambi as bmb
    import pymc as pm

    check_coefficients(prior, spec.columns)
    columns = list(spec.columns)
    built: dict = {}

    def coefficients():
        model = pm.modelcontext(None)
        if built.get("model") is not model:
            name = fam.bambi_scale_parameter
            if name not in model.named_vars:
                raise RuntimeError(f"{prior.name}() on b is scaled by {name!r}, which Bambi has not built before the coefficients")
            built.clear()
            built.update(model=model, b=build(prior, columns, model[name]))
        return built["b"]

    out = {}
    for term in spec.bambi_terms:
        index = np.array([columns.index(c) for c in term.columns], dtype=np.int64)

        def dist(name, dims=None, _index=index, _vector=term.is_vector, **params):
            if params:
                raise RuntimeError(f"Bambi passed the parameters {sorted(params)} to the shrinkage coefficients of {name!r}")
            b = coefficients()
            return pm.Deterministic(name, b[_index] if _vector else b[_index[0]], dims=dims)

        out[term.name] = bmb.Prior(prior.name, dist=dist)
    return out


def check_graph(model) -> None:
    """The built graph against the prior: ``zb`` a free standard normal vector over the coefficients, the scales free and on
    brms's coordinates (logs; R2's log-odds; phi's stick-breaking coordinates), ``sdb`` a deterministic and every term's
    coefficient variable a deterministic; and, at the graph's initial point with a moved ``zb``, the term variables equal
    ``zb * sdb`` column by column."""
    from autobayes.monotonic import StickBreakingTransform

    pmm, roles = model.pymc_model, model.shrinkage_names
    K = len(model.spec.columns)
    zb = pmm[roles["zb"]]
    if zb not in pmm.free_RVs or tuple(zb.type.shape) != (K,) or pmm.rvs_to_values[zb].name != roles["zb"]:
        raise RuntimeError(f"{roles['zb']!r} is not a free vector of the {K} coefficients sampled as it is")
    if tuple(str(c) for c in pmm.coords[roles["coordinate"]]) != tuple(model.spec.columns):
        raise RuntimeError(f"the coordinate {roles['coordinate']!r} is not the design columns {list(model.spec.columns)}")
    coordinates = {"hs_global": "log", "hs_slab": "log", "hs_local": "log", "R2D2_R2": "logodds", "R2D2_phi": "stickbreaking"}
    for key, suffix in coordinates.items():
        if key not in roles:
            continue
        rv = pmm[roles[key]]
        if rv not in pmm.free_RVs or pmm.rvs_to_values[rv].name != f"{roles[key]}_{suffix}__":
            raise RuntimeError(f"{roles[key]!r} is not a free variable sampled as {roles[key]}_{suffix}__")
    if "R2D2_phi" in roles and not isinstance(pmm.rvs_to_transforms[pmm[roles["R2D2_phi"]]], StickBreakingTransform):
        raise RuntimeError(f"{roles['R2D2_phi']!r} is not sampled on Stan's stick-breaking coordinates")
    terms = [t.name for t in model.spec.bambi_terms]
    for name in [roles["sdb"], *terms, *([roles["R2D2_tau2"]] if "R2D2_tau2" in roles else [])]:
        if pmm[name] not in pmm.deterministics:
            raise RuntimeError(f"{name!r} is not a deterministic of the graph under a shrinkage prior on b")
    fn = pmm.compile_fn(pmm.replace_rvs_by_values([pmm[roles["sdb"]], *(pmm[t] for t in terms)]), inputs=pmm.value_vars,
                        point_fn=True, on_unused_input="ignore", mode="FAST_COMPILE")
    point = pmm.initial_point()
    point[roles["zb"]] = np.linspace(-1.0, 1.0, K) + 0.25
    sdb, *coefs = (np.atleast_1d(np.asarray(v, dtype=float)) for v in fn(point))
    got = np.concatenate(coefs) if coefs else np.zeros(0)
    if got.shape != (K,) or not np.allclose(got, point[roles["zb"]] * sdb, rtol=1e-12, atol=0.0):
        raise RuntimeError("the graph's coefficients are not zb * sdb in design-column order")


# -- points ---------------------------------------------------------------------------------------------------------------
def value_entries(model, shrinkage) -> dict[str, np.ndarray]:
    """``Model.value_point``'s entries of a shrinkage prior from natural-scale values keyed by brms's names, each vector in the
    model's design-column order: ``{"zb": (K,), "hs_global": >0, "hs_slab": >0, "hs_local": (K,) > 0}`` for the horseshoe,
    ``{"zb": (K,), "R2D2_R2": in (0, 1), "R2D2_phi": a simplex of K}`` for R2D2; the scales stored as their logs, R2 as its
    log-odds and phi on Stan's stick-breaking coordinates."""
    from autobayes.monotonic import check_simplex, stick_breaking_free

    roles = model.shrinkage_names
    if not roles:
        if shrinkage is not None:
            raise ValueError("this model has no shrinkage prior on b; shrinkage does not apply")
        return {}
    K = len(model.spec.columns)
    keys = ("zb", "hs_global", "hs_slab", "hs_local") if roles["prior"] == "Horseshoe" else ("zb", "R2D2_R2", "R2D2_phi")
    if not isinstance(shrinkage, Mapping) or set(shrinkage) != set(keys):
        raise ValueError(f"shrinkage must be a mapping over exactly {list(keys)} (the {roles['prior']} prior's variables, "
                         f"vectors in the design-column order {list(model.spec.columns)}); b is then omitted, a transformed "
                         "parameter zb * sdb")

    def vector(key, positive=False):
        arr = np.asarray(shrinkage[key], dtype=float)
        if arr.shape != (K,) or not np.isfinite(arr).all() or (positive and (arr <= 0).any()):
            raise ValueError(f"{key} must have {K} finite{' positive' if positive else ''} entries, one per design column")
        return arr

    def scalar(key, low=0.0, high=np.inf):
        arr = np.asarray(shrinkage[key], dtype=float)
        if arr.shape != () or not np.isfinite(arr) or not low < float(arr) < high:
            raise ValueError(f"{key} must be a finite scalar in ({low:g}, {high:g})")
        return float(arr)

    out = {roles["zb"]: vector("zb")}
    if roles["prior"] == "Horseshoe":
        out[f"{roles['hs_global']}_log__"] = np.asarray(np.log(scalar("hs_global")))
        out[f"{roles['hs_slab']}_log__"] = np.asarray(np.log(scalar("hs_slab")))
        out[f"{roles['hs_local']}_log__"] = np.log(vector("hs_local", positive=True))
    else:
        r2 = scalar("R2D2_R2", 0.0, 1.0)
        out[f"{roles['R2D2_R2']}_logodds__"] = np.asarray(np.log(r2) - np.log1p(-r2))
        out[f"{roles['R2D2_phi']}_stickbreaking__"] = stick_breaking_free(check_simplex(shrinkage["R2D2_phi"], K, "R2D2_phi"))
    return out


# -- describe() -----------------------------------------------------------------------------------------------------------
def describe_lines(model) -> list[str]:
    """``describe()``'s prior lines of a shrinkage prior on b, in place of one line per coefficient."""
    from autobayes.default_priors import r_number

    prior, roles, s = model.priors.b, model.shrinkage_names, model.spec
    p = {k: float(v) for k, v in prior.params.items()}
    columns = ", ".join(s.columns)
    lines = [f"  b ~ {prior}   (brms {brms_spelling(prior)} on the whole class b, autoscale = TRUE: scaled by sigma)",
             f"    b_k = zb[k] * sdb[k] for each coefficient k = {columns} (brms's b, a transformed parameter; non-centred)",
             "    zb[k] ~ Normal(0, 1)   (brms's zb)"]
    if prior.name == "Horseshoe":
        lines += [f"    sdb[k] = hs_global * sqrt(c2 * hs_local[k]^2 / (c2 + hs_global^2 * hs_local[k]^2)), c2 = scale_slab^2 * hs_slab "
                  f"= {r_number(p['scale_slab'] ** 2)} * hs_slab   (the regularised horseshoe, Piironen & Vehtari 2017)",
                  f"    hs_local[k] ~ HalfStudentT(nu={p['df']:g}, sigma=1)   (brms's hs_local, the local scales)",
                  f"    hs_global ~ HalfStudentT(nu={p['df_global']:g}, sigma={p['scale_global']:g} * sigma)   (brms's hs_global, the "
                  "global scale: scale_global times the residual SD)",
                  f"    hs_slab ~ InverseGamma(alpha={0.5 * p['df_slab']:g}, beta={0.5 * p['df_slab']:g})   (brms's hs_slab, df_slab / 2 "
                  "both)"]
    else:
        con = ", ".join(f"{v:g}" for v in concentration(prior, len(s.columns)))
        mean, prec = p["mean_R2"], p["prec_R2"]
        lines += ["    sdb[k] = sqrt(R2D2_phi[k] * R2D2_tau2), R2D2_tau2 = sigma^2 * R2D2_R2 / (1 - R2D2_R2)   (the R2D2 prior, Zhang "
                  "et al. 2022: the explained variance R2D2_tau2 split over the coefficients)",
                  f"    R2D2_R2 ~ Beta(alpha={mean * prec:g}, beta={(1 - mean) * prec:g})   (brms's R2D2_R2: mean mean_R2 = {mean:g}, "
                  f"precision prec_R2 = {prec:g})",
                  f"    R2D2_phi ~ Dirichlet({con})   (brms's R2D2_phi, a simplex over the coefficients in the order above)"]
    return lines


def value_roles(model) -> dict[str, str]:
    """``describe()``'s value-space lines of a shrinkage prior's variables."""
    roles, s = model.shrinkage_names, model.spec
    K, columns = len(s.columns), ", ".join(s.columns)
    out = {roles["zb"]: f"zb[{columns}], the standardised coefficients (brms's zb; vector, dims {roles['coordinate']})"}
    if roles["prior"] == "Horseshoe":
        out[f"{roles['hs_global']}_log__"] = "log(hs_global), the global scale (brms's hs_global)"
        out[f"{roles['hs_slab']}_log__"] = "log(hs_slab), the slab's scale (brms's hs_slab)"
        out[f"{roles['hs_local']}_log__"] = f"log(hs_local[{columns}]), the local scales (brms's hs_local; dims {roles['coordinate']})"
    else:
        out[f"{roles['R2D2_R2']}_logodds__"] = "logit(R2D2_R2), the prior's R^2 (brms's R2D2_R2)"
        out[f"{roles['R2D2_phi']}_stickbreaking__"] = (f"R2D2_phi[{columns}] on Stan's stick-breaking simplex coordinates ({K - 1} "
                                                        f"entries; brms's R2D2_phi); dims {roles['coordinate']}")
    return out


def deterministic_note(model) -> str:
    """``describe()``'s note on the deterministics of a shrinkage prior."""
    roles = model.shrinkage_names
    terms = ", ".join(t.name for t in model.spec.bambi_terms)
    return (f"; {terms} = b = zb * sdb (brms's b), {roles['sdb']} = the coefficients' scales (brms's sdb)"
            + (f", {roles['R2D2_tau2']} = sigma^2 * R2D2_R2 / (1 - R2D2_R2)" if "R2D2_tau2" in roles else ""))
