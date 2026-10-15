"""The oracle gates, as functions of a `Bundle` (case, fixture, data, AutoBayes model and the two maps).
Each raises `OracleMismatch` naming the point and the quantities that differ; a passing gate returns None.

G1 `check_design`        standata equals the model's design, response, trials, offsets, group coding and addition-term
                         data (weights, cens, lb, ub; B02), each distributional formula's X_<dpar> (B01), each
                         monotonic term's codes, number of steps and Dirichlet concentrations (Xmo_k, Jmo, con_simo_k; B03),
                         and a mixture's component designs X_mu<k> and theta's concentrations (con_theta; C05)
G2 `check_logp`          log density with and without the Jacobian, and its prior part, at every point
G2 `check_gradient`      every gradient coordinate, mapped through the probed parameter table
G3 `check_derived`       the graph's linear predictor, group effects and correlation matrices, and the raw-scale intercept
                         (the cumulative thresholds and a categorical model's per-category intercepts, B05); an
                         auxiliary parameter's own formula: its per-row values and raw-scale intercept (B01)
G4 `check_predictions`   predict() on the fixture's draws equals posterior_linpred/posterior_epred
G5 `check_jax`           the JAX-compiled density equals the PyTensor one at every point
G6 `check_default_prior` brms's default-prior table has the classes this model has, and `ab.default_priors()` for the
                         same formula, data and family, written as brms's table, equals it row by row (A07)
C07 `check_post`         for a case that lists post-processing functions: `ab.log_lik`, `ab.bayes_R2` and `ab.loo` on the
                         fixture's draws, held as brms's fixed-parameter fit holds them (one chain per point), equal the
                         fixture's `post` block (brms's log_lik(), bayes_R2() and loo()), every non-finite entry exactly

A case that opts in to two-reference acceptance (P6, its gate field names a committed high-precision reference and the
quantity kinds that may fall back on it: "gradient", "epred") has G2's gradient and G4's expected values checked by
`two_reference`: brms where brms is accurate, the reference where it is not; every other check is unchanged.
"""
from __future__ import annotations

import re
from functools import cached_property
from itertools import zip_longest

import numpy as np

import autobayes as ab
from autobayes.priors import dirichlet_concentration

from tests.oracle import adapter
from tests.oracle import fixture as fx
from tests.oracle import two_reference
from tests.oracle.adapter import OracleMismatch
from tests.oracle.tolerances import DESIGN_ABS, GRAD_REL, LOGP_REL, PRED_REL, abs_err, rel_err


def prior_names(model) -> list[str]:
    """The free variables whose densities brms sums into ``lprior``: intercept, coefficients, sigma, shape or phi, a learned nu,
    zi or hu (B06), the SD of every group-level coefficient (not the standardised offsets, which brms adds in the model block) and the
    correlation Cholesky factor of every correlated group term (``lkj_corr_cholesky_lpdf(L_m | eta)``); for an auxiliary
    parameter with its own formula (B01), that formula's intercept and coefficients (brms's ``Intercept_sigma``, ``b_sigma``);
    the simplex of every monotonic term (``dirichlet_lpdf(simo_k | con_simo_k)``; its coefficient is among the terms, B03);
    a mixture's ``sigma<k>`` and its mixing proportions (``dirichlet_lpdf(theta | con_theta)``; its ordered intercepts are the
    intercept, each component's density as brms adds it, C05)."""
    names = model.names
    out = [names["intercept"]] if model.spec.intercept else []
    out += list(model.spec.terms)
    distributional = model.spec.dpar_names                  # the auxiliary parameters with a formula of their own (B01)
    if "sigma" in names and "sigma" not in distributional:
        out.append(names["sigma"])
    if "shape" in names and "shape" not in distributional:   # negative binomial / Gamma shape (PyMC's alpha)
        out.append(names["shape"])
    if "phi" in names and "phi" not in distributional:       # Beta precision
        out.append(names["phi"])
    out += [v for name in distributional for k, v in model.dpar_names[name].items() if k != "value"]   # their formulas (B01)
    if "nu" in names:                                      # a learned Student-t nu, its prior truncated to nu > 1 (A13)
        out.append(names["nu"])
    out += [names[par] for par in ("zi", "hu") if par in names]   # the zero-inflation / hurdle probability (B06)
    out += [roles["sd"] for roles in model.group_coefficient_names.values()]
    out += [roles["L"] for roles in model.correlation_names.values()]
    out += [roles["simplex"] for roles in model.mo_names.values()]
    mixture = getattr(model, "mixture_names", {})          # C05: each component's sigma<k> and theta
    out += [*mixture["sigma"], mixture["theta"]] if mixture else []
    return out


class Bundle:
    def __init__(self, case_id: str, fixture: fx.Fixture | None = None, model=None):
        self.case = fx.load_case(case_id)
        self.fixture = fixture or fx.Fixture.load(case_id)
        d = fx.fixture_dir(case_id)
        self.data = fx.read_frame(self.case, d / "data.csv")
        self.newdata = fx.read_frame(self.case, d / "newdata.csv") if (d / "newdata.csv").exists() else None
        self.model = model or adapter.build_model(self.case, self.data)
        # C07: the post-processing functions a case names (``"post": ["log_lik", "bayes_R2", "loo"]`` in its autobayes
        # section, each proposed as ``ab.<name>(model, idata)``) must exist, so that such a case fails until they do; the
        # gate that compares them with the fixture's ``post`` block (brms's log_lik(), bayes_R2(), loo()) lands with them
        self.post = {name: getattr(ab, name) for name in self.case.get("autobayes", {}).get("post", ())}
        self.cmap = adapter.match_columns(self.fixture, self.model)
        self.gmap = adapter.match_groups(self.fixture, self.model)

    def with_fixture(self, fixture: fx.Fixture) -> "Bundle":
        """The same case and model against another fixture (used to show that a corrupted fixture fails)."""
        return Bundle(self.case["id"], fixture=fixture, model=self.model)

    def points(self):
        for k, p in enumerate(self.fixture.points):
            yield k, p, adapter.value_point(self.model, p, self.cmap, self.gmap)

    @cached_property
    def logp(self):
        return self.model.pymc_model.compile_logp()

    @cached_property
    def logp_nojac(self):
        return self.model.pymc_model.compile_logp(jacobian=False)

    @cached_property
    def dlogp(self):
        return self.model.pymc_model.compile_dlogp()

    @cached_property
    def dlogp_nojac(self):
        return self.model.pymc_model.compile_dlogp(jacobian=False)

    @cached_property
    def lprior(self):
        rvs = [self.model.pymc_model[name] for name in prior_names(self.model)]
        return self.model.pymc_model.compile_logp(vars=rvs, jacobian=False)

    @cached_property
    def reference(self):
        """P6: the committed high-precision reference of a case that opts in to two-reference acceptance (its gate field
        names one, ``two_reference.load`` validates the opt-in), else None."""
        return two_reference.load(self.case)


# -- G1 -----------------------------------------------------------------------------------------------------
def check_design(b: Bundle) -> None:
    sd, d = b.fixture.standata, b.model.design
    if int(sd["N"]) != d.n:
        raise OracleMismatch(f"brms N = {int(sd['N'])}, AutoBayes n = {d.n}")
    # B05: the cumulative model's X has no intercept column (its thresholds take its place); the categorical model has one
    # design per non-reference category
    designs = [f"X_{dp}" for dp in b.cmap.dpars] or ["X"]
    intercept_column = int(b.model.spec.intercept and not b.cmap.thresholds)
    for name in designs:
        if sd[name].shape != (d.n, d.X_raw.shape[1] + intercept_column):
            raise OracleMismatch(f"brms {name} is {sd[name].shape}; AutoBayes has {d.X_raw.shape[1]} design columns and intercept={b.model.spec.intercept}")
    if b.cmap.thresholds and int(sd["nthres"]) != len(b.model.categories) - 1:
        raise OracleMismatch(f"brms has nthres = {int(sd['nthres'])}; AutoBayes has {len(b.model.categories)} categories")
    if b.cmap.dpars and not b.cmap.mixture and int(sd["ncat"]) != len(b.model.categories):   # a mixture's dpars are components
        raise OracleMismatch(f"brms has ncat = {int(sd['ncat'])}; AutoBayes has {len(b.model.categories)} categories")
    if b.cmap.mixture:                                           # C05: K component designs (matched by content) and con_theta
        K = len(b.model.components)
        if len(b.cmap.dpars) != K:
            raise OracleMismatch(f"brms has {len(b.cmap.dpars)} component designs; the AutoBayes mixture has {K} components")
        con = dirichlet_concentration(b.model.priors.theta, K)
        theirs = np.atleast_1d(sd["con_theta"])
        if theirs.shape != con.shape or abs_err(theirs, con) > DESIGN_ABS:
            raise OracleMismatch(f"brms con_theta is {theirs!r}; the prior of theta gives {con!r}")
    err = abs_err(sd["Y"], d.y)
    if err > DESIGN_ABS:
        raise OracleMismatch(f"response differs by {err}")
    if d.trials is not None:
        if "trials" not in sd:
            raise OracleMismatch("AutoBayes has trials but brms standata has none")
        err = abs_err(sd["trials"], d.trials)
        if err > DESIGN_ABS:
            raise OracleMismatch(f"trials differ by {err}")
    elif "trials" in sd:
        raise OracleMismatch("brms has trials but AutoBayes does not")
    if d.offset is not None:                                     # brms's offsets: the offset term's values per row
        if "offsets" not in sd:
            raise OracleMismatch("AutoBayes has an offset term but brms standata has no offsets")
        err = abs_err(sd["offsets"], d.offset)
        if err > DESIGN_ABS:
            raise OracleMismatch(f"offsets differ by {err}")
    elif "offsets" in sd:
        raise OracleMismatch("brms has offsets but AutoBayes has no offset term")
    for key in ("weights", "cens", "lb", "ub"):                  # brms's addition-term data (B02): weights(), cens(), trunc() bounds
        mine = getattr(d, key)
        if mine is None:
            if key in sd:
                raise OracleMismatch(f"brms has {key} but AutoBayes does not")
            continue
        if key not in sd:
            raise OracleMismatch(f"AutoBayes has {key} but brms standata has none")
        theirs, mine = np.asarray(sd[key], dtype=float), np.asarray(mine, dtype=float)
        if theirs.shape != mine.shape:
            raise OracleMismatch(f"brms {key} has shape {theirs.shape}, AutoBayes {mine.shape}")
        inf = np.isinf(theirs)                                   # an open truncation bound: brms's Inf where the model has inf, same sign
        if not np.array_equal(theirs[inf], mine[inf]) or np.isinf(mine[~inf]).any():
            raise OracleMismatch(f"{key}: brms's infinite entries (rows {np.flatnonzero(inf)[:5].tolist()}...) differ from AutoBayes's")
        err = abs_err(theirs[~inf], mine[~inf])
        if not err <= DESIGN_ABS:
            raise OracleMismatch(f"{key} differ by {err}")
    for name, dd in d.dpars.items():                            # an auxiliary parameter's own formula (B01): X_sigma
        key = f"X_{name}"
        if key not in sd:
            raise OracleMismatch(f"AutoBayes gives {name} its own formula but brms standata has no {key}")
        if sd[key].shape != (d.n, dd.X_raw.shape[1] + int(b.model.spec.dpar(name).intercept)):
            raise OracleMismatch(f"brms {key} is {sd[key].shape}; AutoBayes has {dd.X_raw.shape[1]} design columns for {name} "
                                 f"and intercept={b.model.spec.dpar(name).intercept}")
    for column, k in (b.cmap.monotonic or {}).items():          # B03: Xmo_k and Jmo[k] were matched by content; the prior's
        block = d.monotonic[column]                              # concentrations are brms's con_simo_k
        con = dirichlet_concentration(b.model.priors.simo_priors(b.model.spec)[column], block.D)
        theirs = np.atleast_1d(sd[f"con_simo_{k}"])
        if theirs.shape != con.shape or abs_err(theirs, con) > DESIGN_ABS:
            raise OracleMismatch(f"brms con_simo_{k} is {theirs!r}; the prior of {block.term!r} gives {con!r}")
    # the design columns (every formula's), the group partitions and the monotonic codes were matched by content when the
    # maps were built
    adapter.match_columns(b.fixture, b.model)
    adapter.match_groups(b.fixture, b.model)


# -- G2 -----------------------------------------------------------------------------------------------------
def check_logp(b: Bundle) -> None:
    for k, p, vp in b.points():
        a, an, pri = float(b.logp(vp)), float(b.logp_nojac(vp)), float(b.lprior(vp))
        pri_b = float(p.pars["lprior"]) if "lprior" in p.pars else None
        bad = rel_err(a, p.lp) > LOGP_REL or rel_err(an, p.lp_nojac) > LOGP_REL
        bad = bad or (pri_b is not None and rel_err(pri, pri_b) > LOGP_REL)
        if bad:
            raise OracleMismatch(
                f"point {k}: log density AutoBayes {a!r} vs brms {p.lp!r}; without Jacobian {an!r} vs {p.lp_nojac!r}; "
                f"prior part {pri!r} vs {pri_b!r}; remainder {an - pri!r} vs {None if pri_b is None else p.lp_nojac - pri_b!r}")


def check_gradient(b: Bundle) -> None:
    if b.reference is not None and "gradient" in b.reference.fallback:      # P6: brms where it is accurate, the reference where not
        return two_reference.check_gradient(b)
    coords = adapter.coordinate_map(b.fixture, b.model, b.cmap, b.gmap)
    for k, p, vp in b.points():
        layout = adapter.value_layout(b.model, vp)
        total = sum(s.stop - s.start for s in layout.values())
        if len(coords) != total or p.grad.size != total:
            raise OracleMismatch(f"brms has {len(coords)} unconstrained coordinates, AutoBayes {total}")
        ga, gan = np.asarray(b.dlogp(vp), dtype=float), np.asarray(b.dlogp_nojac(vp), dtype=float)
        hit = np.zeros(total, dtype=bool)
        for i, var, idx in coords:
            pos = layout[var].start + idx
            hit[pos] = True
            if rel_err(ga[pos], p.grad[i]) > GRAD_REL or rel_err(gan[pos], p.grad_nojac[i]) > GRAD_REL:
                raise OracleMismatch(
                    f"point {k}: gradient of {var}[{idx}] AutoBayes {ga[pos]!r} vs brms {p.grad[i]!r} "
                    f"(without Jacobian {gan[pos]!r} vs {p.grad_nojac[i]!r})")
        if not hit.all():
            missing = [v for v, s in layout.items() if not hit[s].all()]
            raise OracleMismatch(f"AutoBayes coordinates without a brms counterpart: {missing}")


# -- G3 -----------------------------------------------------------------------------------------------------
def check_derived(b: Bundle) -> None:
    pmm, names, d = b.model.pymc_model, b.model.names, b.model.design
    groups = list(b.model.group_coefficient_names.items())          # every group-level coefficient: intercepts and slopes
    cors = list(b.model.correlation_names.items())                  # the correlation matrix of every correlated term (A02)
    distributional = list(b.model.dpar_names.items())               # each auxiliary parameter's own formula (B01): its per-row values
    dets = ([pmm[names["linpred"]]] + [pmm[roles["group_effect"]] for _, roles in groups] + [pmm[roles["cor"]] for _, roles in cors]
            + [pmm[roles["value"]] for _, roles in distributional])
    if b.cmap.thresholds:                                           # B05: the graph's b_Intercept, brms's raw thresholds (the last)
        dets.append(pmm[names["thresholds"]])
    fn = pmm.compile_fn(pmm.replace_rvs_by_values(dets), inputs=pmm.value_vars, point_fn=True, on_unused_input="ignore")
    # Under a log link Bambi's parent deterministic holds the mean exp(eta), which is brms's epred when the parent is the
    # expected response (Poisson, negative binomial, Gamma: names["linpred"] == names["mean"] == "mu"); the zero-inflated
    # and hurdle families (B06) keep eta itself as their parent log_mu, whose brms counterpart is posterior_linpred.
    holds_mean = b.model.link == "log" and names["linpred"] == names["mean"]
    ref = b.fixture.prediction("fitted", "epred" if holds_mean else "linpred")
    for k, p, vp in b.points():
        out = fn(vp)
        # B05: a categorical model's linear predictor has one column per non-reference category, brms's (rows, categories)
        lin = np.asarray(out[0], dtype=float) if b.cmap.dpars else np.asarray(out[0], dtype=float).ravel()
        err = rel_err(lin, ref[k])
        if err > PRED_REL:
            raise OracleMismatch(f"point {k}: the graph's {names['linpred']!r} differs from brms by {err}")
        for ((g, coef), _), eff in zip(groups, out[1:1 + len(groups)]):
            m, cn = b.gmap.brms_id[g], b.gmap.coef_cn[(g, coef)]
            r = np.asarray(eff, dtype=float).ravel()[b.gmap.level_of[g]]
            err = rel_err(r, np.atleast_1d(p.pars[f"r_{m}_{cn}"]))
            if err > PRED_REL:
                raise OracleMismatch(f"point {k}: group effects of {coef!r} of {g!r} differ from brms r_{m}_{cn} by {err}")
        for (g, _), cor in zip(cors, out[1 + len(groups):1 + len(groups) + len(cors)]):
            m, perm = b.gmap.brms_id[g], adapter.correlation_order(b.model, b.gmap, g)
            ref_cor = np.atleast_2d(p.pars[f"Cor_{m}"])[np.ix_(perm, perm)]
            err = rel_err(np.asarray(cor, dtype=float), ref_cor)
            if err > PRED_REL:
                raise OracleMismatch(f"point {k}: the correlation matrix of {g!r} differs from brms Cor_{m} by {err}")
        start = 1 + len(groups) + len(cors)
        for (name, _), value in zip(distributional, out[start:start + len(distributional)]):   # B01: exp of brms's own predictor
            X = b.fixture.standata[f"X_{name}"]
            dmap = b.cmap.distributional[name]
            bd = adapter.coefficients(p.pars, f"b_{name}")
            if dmap.centred_intercept:                              # brms: Intercept_<name> + Xc_<name> b_<name>, Xc centred on its means
                Xc = np.delete(X, dmap.intercept_column, axis=1)
                eta = float(p.pars[f"Intercept_{name}"]) + (Xc - Xc.mean(0)) @ bd
            else:
                eta = X @ bd
            err = rel_err(np.asarray(value, dtype=float).ravel(), np.exp(eta))
            if err > PRED_REL:
                raise OracleMismatch(f"point {k}: the graph's {name} differs from exp of brms's linear predictor of {name} by {err}")
            if f"b_{name}_Intercept" in p.pars:                      # the raw-scale intercept of name's formula
                nat = adapter.natural_point(b.model, p, b.cmap, b.gmap)[name]
                raw = nat["intercept"] - float(b.model.design.dpars[name].x_means @ nat["b"])
                err = rel_err(raw, float(p.pars[f"b_{name}_Intercept"]))
                if err > PRED_REL:
                    raise OracleMismatch(f"point {k}: raw-scale intercept of {name} {raw!r} vs brms b_{name}_Intercept "
                                         f"{float(p.pars[f'b_{name}_Intercept'])!r}")
        if b.cmap.thresholds:                                        # B05: b_Intercept = Intercept + sum_p mean(x_p) b_p
            nat = adapter.natural_point(b.model, p, b.cmap, b.gmap)
            brms = np.atleast_1d(p.pars["b_Intercept"])
            for what, raw in (("raw-scale thresholds", nat["intercept"] + float(d.x_means @ nat["b"])),
                              (f"the graph's {names['thresholds']!r}", np.asarray(out[-1], dtype=float))):
                err = rel_err(raw, brms)
                if err > PRED_REL:
                    raise OracleMismatch(f"point {k}: {what} {raw!r} vs brms b_Intercept {brms!r}")
        elif b.cmap.dpars:                                           # B05: each category's raw-scale intercept
            nat = adapter.natural_point(b.model, p, b.cmap, b.gmap)
            for c, dp in enumerate(b.cmap.dpars):
                if f"b_{dp}_Intercept" in p.pars:
                    raw = nat["intercept"][c] - float(d.x_means @ nat["b"][:, c])
                    err = rel_err(raw, float(p.pars[f"b_{dp}_Intercept"]))
                    if err > PRED_REL:
                        raise OracleMismatch(f"point {k}: raw-scale intercept of {dp} {raw!r} vs brms b_{dp}_Intercept "
                                             f"{float(p.pars[f'b_{dp}_Intercept'])!r}")
        elif "b_Intercept" in p.pars:
            nat = adapter.natural_point(b.model, p, b.cmap, b.gmap)
            raw = nat["intercept"] - float(d.x_means @ nat["b"])
            err = rel_err(raw, float(p.pars["b_Intercept"]))
            if err > PRED_REL:
                raise OracleMismatch(f"point {k}: raw-scale intercept {raw!r} vs brms b_Intercept {float(p.pars['b_Intercept'])!r}")


# -- G4 -----------------------------------------------------------------------------------------------------
def check_predictions(b: Bundle) -> None:
    if b.reference is not None and "epred" in b.reference.fallback:         # P6: brms where it is accurate, the reference where not
        return two_reference.check_predictions(b)
    idata = adapter.draws(b.fixture, b.model, b.cmap, b.gmap)
    mean_var = "expected" if b.model.family == "binomial" else b.model.names["mean"]
    for block, nd in (("fitted", None), ("newdata", b.newdata)):
        if block == "newdata" and nd is None:
            continue
        for include, suffix in ((True, ""), (False, "_nogroup")):
            ref_lin = b.fixture.prediction(block, "linpred" + suffix)
            ref_ep = b.fixture.prediction(block, "epred" + suffix)
            if ref_lin is None or ref_ep is None:
                if not include and b.model.spec.groups:
                    raise OracleMismatch(f"fixture lacks the {block} predictions without group effects")
                continue
            lin = b.model.predict(idata, newdata=nd, kind="linear", include_group_effects=include)["eta"].to_numpy()[0]
            err = rel_err(lin, ref_lin)
            if err > PRED_REL:
                raise OracleMismatch(f"{block}{suffix}: linear predictor differs from brms posterior_linpred by {err}")
            mean = b.model.predict(idata, newdata=nd, kind="mean", include_group_effects=include)[mean_var].to_numpy()[0]
            err = rel_err(mean, ref_ep)
            if err > PRED_REL:
                raise OracleMismatch(f"{block}{suffix}: {mean_var!r} differs from brms posterior_epred by {err}")


# -- G5 -----------------------------------------------------------------------------------------------------
def check_jax(b: Bundle) -> None:
    from pymc.sampling.jax import get_jaxified_logp

    jf = get_jaxified_logp(b.model.pymc_model)
    names = [v.name for v in b.model.pymc_model.value_vars]
    for k, p, vp in b.points():
        a, c = float(jf([vp[n] for n in names])), float(b.logp(vp))
        if rel_err(a, c) > LOGP_REL:
            raise OracleMismatch(f"point {k}: JAX log density {a!r} vs PyTensor {c!r}")


# -- G6 -----------------------------------------------------------------------------------------------------
_NUMBER = re.compile(r"(?<![A-Za-z_0-9.])[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def normalise_prior(text: str) -> str:
    """A prior string or bound with the whitespace removed and every number written as the double it denotes, so that
    ``student_t(3, 0.9, 2.5)`` and ``student_t(3,0.9,2.5)``, ``3`` and ``3.0``, ``1e+05`` and ``100000`` compare equal and
    nothing else does (brms prints 15 significant digits; ``ab.default_priors`` reproduces them, so equal strings are
    equal doubles)."""
    return _NUMBER.sub(lambda m: repr(float(m.group())), re.sub(r"\s+", "", text))


def default_prior_table(b: Bundle) -> list[dict[str, str]]:
    """``ab.default_priors`` for the case's formula, data, family and keyword arguments, written as brms's table for the
    case's model (``Priors.table``). brms's ``default_prior()`` is computed from the formula, data and family alone, so
    it describes brms's default model, which learns a Student-t ``nu``: a case that fixes nu (brms ``constant(4)``) is
    compared under ``nu=None``."""
    spec = b.case["autobayes"]
    kwargs = dict(spec.get("kwargs", {}))
    if "nu" in b.model.family_spec.learnable_parameters:
        kwargs["nu"] = None
    priors = ab.default_priors(spec["formula"], b.data, family=b.model.family, **kwargs)
    return priors.table(b.model.spec, b.model.family_spec, center_predictors=b.model.effective_center_predictors,
                        categories=getattr(b.model, "categories", ()) or None)   # the ordinal and categorical tables need them (B05)


def check_default_prior(b: Bundle) -> None:
    classes = {row["class"] for row in b.fixture.default_prior}
    expected = set()
    if b.model.spec.intercept and b.cmap.centred_intercept:
        expected.add("Intercept")
    if b.model.spec.coefficient_labels or (b.model.spec.intercept and not b.cmap.centred_intercept):
        expected.add("b")                                  # a design column or a monotonic term's coefficient (B03)
    if b.model.spec.mo_terms:                              # a monotonic term's simplex (B03)
        expected.add("simo")
    if b.model.spec.groups:
        expected.add("sd")
    if b.model.correlation_names:                          # a correlated group term's LKJ prior (A02)
        expected.add("cor")
    distributional = b.model.spec.dpar_names               # an auxiliary parameter with its own formula has no class row (B01)
    if "sigma" in b.model.names and "sigma" not in distributional:
        expected.add("sigma")
    if "shape" in b.model.names and "shape" not in distributional:
        expected.add("shape")
    if "phi" in b.model.names and "phi" not in distributional:
        expected.add("phi")
    dpar_rows = {(row["class"], row.get("dpar", "")) for row in b.fixture.default_prior}
    for name in distributional:                            # its formula's Intercept and b rows, dpar = name
        ds = b.model.spec.dpar(name)
        wanted = ({("Intercept", name)} if ds.intercept else set()) | ({("b", name)} if ds.columns else set())
        if not wanted <= dpar_rows:
            raise OracleMismatch(f"brms default_prior lacks the rows {sorted(wanted - dpar_rows)} of {name}'s formula")
    if "nu" in b.model.names:
        expected.add("nu")
    expected |= {par for par in ("zi", "hu") if par in b.model.names}     # B06
    mixture = getattr(b.model, "mixture_names", {})        # C05: each component's sigma<k> and theta (Intercept, b above)
    expected |= {*mixture["sigma"], "theta"} if mixture else set()
    if not expected <= classes:
        raise OracleMismatch(f"brms default_prior classes {sorted(classes)} lack {sorted(expected - classes)}")
    if not b.fixture.prior_summary:
        raise OracleMismatch("fixture has no prior_summary table")
    # ab.default_priors() reproduces brms's table: every row, in brms's order, every column (A07)
    ours, theirs = default_prior_table(b), b.fixture.default_prior

    def key(row):
        if row is None:
            return None
        return tuple(normalise_prior(row.get(k, "")) if k in ("prior", "lb", "ub") else row.get(k, "")
                     for k in ("prior", "class", "coef", "group", "resp", "dpar", "nlpar", "lb", "ub", "tag", "source"))
    diff = [f"  row {i}: AutoBayes {key(a)} vs brms {key(t)}"
            for i, (a, t) in enumerate(zip_longest(ours, theirs)) if key(a) != key(t)]
    if diff:
        raise OracleMismatch(f"ab.default_priors() written as brms's table differs from brms's default_prior() "
                             f"({len(ours)} rows vs {len(theirs)}):\n" + "\n".join(diff[:12]))


ALL = (check_design, check_logp, check_gradient, check_derived, check_predictions, check_jax, check_default_prior)


# -- C07: post-processing -----------------------------------------------------------------------------------------
def fixed_parameter_draws(b: Bundle):
    """The fixture's points as brms's fixed-parameter fit holds them (``gen_fixture.R``'s ``Fixed_param`` run): one chain
    per point with one draw each, so that the relative efficiencies brms hands loo, and ``ab.loo``'s default ones, are
    those of one-draw chains (loo's NA, used as 1)."""
    import arviz as az

    post = adapter.draws(b.fixture, b.model, b.cmap, b.gmap).posterior
    S = post.sizes["draw"]
    chains = post.isel(chain=0, drop=True).rename({"draw": "chain"}).assign_coords(chain=np.arange(S))
    return az.InferenceData(posterior=chains.expand_dims(draw=[0], axis=1))


def compare_post(what: str, ours, theirs) -> None:
    """``ours`` equals brms's ``theirs`` to ``PRED_REL`` where brms's is finite, and a non-finite entry of either is the
    same non-finite value in the other (Inf with Inf, -Inf with -Inf, NaN with NaN)."""
    ours, theirs = np.asarray(ours, dtype=float), np.asarray(theirs, dtype=float)
    if ours.shape != theirs.shape:
        raise OracleMismatch(f"{what}: AutoBayes has shape {ours.shape}, brms {theirs.shape}")
    finite_o, finite_t = np.isfinite(ours), np.isfinite(theirs)
    same = (ours == theirs) | (np.isnan(ours) & np.isnan(theirs))
    differ = (finite_o != finite_t) | (~finite_o & ~finite_t & ~same)
    if differ.any():
        where = [tuple(int(i) for i in ix) for ix in np.argwhere(differ)[:5]]
        raise OracleMismatch(f"{what}: non-finite entries differ at {where}: AutoBayes {ours[differ][:5].tolist()} vs brms "
                             f"{theirs[differ][:5].tolist()}")
    err = rel_err(ours[finite_o], theirs[finite_t])
    if err > PRED_REL:
        raise OracleMismatch(f"{what} differs from brms by {err}")


def check_post(b: Bundle) -> None:
    """C07: each post-processing function the case lists (``b.post``) on ``fixed_parameter_draws`` equals the fixture's
    ``post`` block: ``log_lik`` brms's ``log_lik()`` (draws x rows); ``bayes_R2`` brms's ``bayes_R2(summary = FALSE)``
    (draws x 1); ``loo`` brms's ``loo()``: the pointwise table (its five columns, by name), the estimates (rows elpd_loo,
    p_loo, looic; columns Estimate, SE), ``pareto_k`` and ``n_eff``. A case that lists none has nothing to compare."""
    if not b.post:
        return
    post = b.fixture.raw.get("post")
    if not post:
        raise OracleMismatch(f"the case lists {sorted(b.post)} but the fixture has no post block")
    idata = fixed_parameter_draws(b)
    S = len(b.fixture.points)
    for name, fn in b.post.items():
        if name not in post:
            raise OracleMismatch(f"the case lists {name} but the fixture's post block has {sorted(post)}")
        if name == "log_lik":
            ours = fn(b.model, idata).transpose("chain", "draw", "__obs__").to_numpy().reshape(S, -1)
            compare_post("log_lik", ours, fx.decode(post["log_lik"]))
        elif name == "bayes_R2":
            ours = fn(b.model, idata).transpose("chain", "draw").to_numpy().reshape(S, 1)
            compare_post("bayes_R2", ours, fx.decode(post["bayes_R2"]))
        elif name == "loo":
            res, ref = fn(b.model, idata), post["loo"]
            columns = list(ref["pointwise"].get("colnames") or [])
            if list(res.pointwise.columns) != columns:
                raise OracleMismatch(f"loo pointwise columns {list(res.pointwise.columns)} vs brms {columns}")
            table = fx.decode(ref["pointwise"])
            if res.pointwise.shape != table.shape:
                raise OracleMismatch(f"loo pointwise: AutoBayes has shape {res.pointwise.shape}, brms {table.shape}")
            for i, column in enumerate(columns):
                compare_post(f"loo pointwise {column}", res.pointwise[column].to_numpy(), table[:, i])
            if list(res.estimates.index) != list(ref["estimate_rows"]) or list(res.estimates.columns) != list(ref["estimates"]["colnames"]):
                raise OracleMismatch(f"loo estimates {list(res.estimates.index)} x {list(res.estimates.columns)} vs brms "
                                     f"{list(ref['estimate_rows'])} x {list(ref['estimates']['colnames'])}")
            compare_post("loo estimates", res.estimates.to_numpy(), fx.decode(ref["estimates"]))
            compare_post("loo pareto_k", res.pareto_k, fx.decode(ref["pareto_k"]))
            compare_post("loo n_eff", res.n_eff, fx.decode(ref["n_eff"]))
        else:
            raise OracleMismatch(f"no comparison for the post-processing function {name!r}")
