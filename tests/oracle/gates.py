"""The oracle gates, as functions of a `Bundle` (case, fixture, data, AutoBayes model and the two maps).
Each raises `OracleMismatch` naming the point and the quantities that differ; a passing gate returns None.

G1 `check_design`        standata equals the model's design, response, trials and group coding
G2 `check_logp`          log density with and without the Jacobian, and its prior part, at every point
G2 `check_gradient`      every gradient coordinate, mapped through the probed parameter table
G3 `check_derived`       the graph's linear predictor and group effects, and the raw-scale intercept
G4 `check_predictions`   predict() on the fixture's draws equals posterior_linpred/posterior_epred
G5 `check_jax`           the JAX-compiled density equals the PyTensor one at every point
G6 `check_default_prior` brms's default-prior table is present with the classes this model has
"""
from __future__ import annotations

from functools import cached_property

import numpy as np

from tests.oracle import adapter
from tests.oracle import fixture as fx
from tests.oracle.adapter import OracleMismatch
from tests.oracle.tolerances import DESIGN_ABS, GRAD_REL, LOGP_REL, PRED_REL, abs_err, rel_err


def prior_names(model) -> list[str]:
    """The free variables whose densities brms sums into ``lprior``: intercept, coefficients, sigma or shape,
    the SD of every group-level coefficient (not the standardised offsets, which brms adds in the model block)."""
    names = model.names
    out = [names["intercept"]] if model.spec.intercept else []
    out += list(model.spec.terms)
    if "sigma" in names:
        out.append(names["sigma"])
    if "shape" in names:                                   # negative binomial / Gamma shape (PyMC's alpha)
        out.append(names["shape"])
    out += [roles["sd"] for roles in model.group_coefficient_names.values()]
    return out


class Bundle:
    def __init__(self, case_id: str, fixture: fx.Fixture | None = None, model=None):
        self.case = fx.load_case(case_id)
        self.fixture = fixture or fx.Fixture.load(case_id)
        d = fx.fixture_dir(case_id)
        self.data = fx.read_frame(self.case, d / "data.csv")
        self.newdata = fx.read_frame(self.case, d / "newdata.csv") if (d / "newdata.csv").exists() else None
        self.model = model or adapter.build_model(self.case, self.data)
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


# -- G1 -----------------------------------------------------------------------------------------------------
def check_design(b: Bundle) -> None:
    sd, d = b.fixture.standata, b.model.design
    if int(sd["N"]) != d.n:
        raise OracleMismatch(f"brms N = {int(sd['N'])}, AutoBayes n = {d.n}")
    if sd["X"].shape != (d.n, d.X_raw.shape[1] + int(b.model.spec.intercept)):
        raise OracleMismatch(f"brms X is {sd['X'].shape}; AutoBayes has {d.X_raw.shape[1]} design columns and intercept={b.model.spec.intercept}")
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
    # the design columns and the group partitions were matched by content when the maps were built
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
    dets = [pmm[names["linpred"]]] + [pmm[roles["group_effect"]] for _, roles in groups]
    fn = pmm.compile_fn(pmm.replace_rvs_by_values(dets), inputs=pmm.value_vars, point_fn=True, on_unused_input="ignore")
    ref = b.fixture.prediction("fitted", "epred" if b.model.link == "log" else "linpred")
    for k, p, vp in b.points():
        out = fn(vp)
        lin = np.asarray(out[0], dtype=float).ravel()
        err = rel_err(lin, ref[k])
        if err > PRED_REL:
            raise OracleMismatch(f"point {k}: the graph's {names['linpred']!r} differs from brms by {err}")
        for ((g, coef), _), eff in zip(groups, out[1:]):
            m, cn = b.gmap.brms_id[g], b.gmap.coef_cn[(g, coef)]
            r = np.asarray(eff, dtype=float).ravel()[b.gmap.level_of[g]]
            err = rel_err(r, np.atleast_1d(p.pars[f"r_{m}_{cn}"]))
            if err > PRED_REL:
                raise OracleMismatch(f"point {k}: group effects of {coef!r} of {g!r} differ from brms r_{m}_{cn} by {err}")
        if "b_Intercept" in p.pars:
            nat = adapter.natural_point(b.model, p, b.cmap, b.gmap)
            raw = nat["intercept"] - float(d.x_means @ nat["b"])
            err = rel_err(raw, float(p.pars["b_Intercept"]))
            if err > PRED_REL:
                raise OracleMismatch(f"point {k}: raw-scale intercept {raw!r} vs brms b_Intercept {float(p.pars['b_Intercept'])!r}")


# -- G4 -----------------------------------------------------------------------------------------------------
def check_predictions(b: Bundle) -> None:
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
def check_default_prior(b: Bundle) -> None:
    classes = {row["class"] for row in b.fixture.default_prior}
    expected = set()
    if b.model.spec.intercept and b.cmap.centred_intercept:
        expected.add("Intercept")
    if b.model.spec.columns or (b.model.spec.intercept and not b.cmap.centred_intercept):
        expected.add("b")
    if b.model.spec.groups:
        expected.add("sd")
    if "sigma" in b.model.names:
        expected.add("sigma")
    if "shape" in b.model.names:
        expected.add("shape")
    if not expected <= classes:
        raise OracleMismatch(f"brms default_prior classes {sorted(classes)} lack {sorted(expected - classes)}")
    if not b.fixture.prior_summary:
        raise OracleMismatch("fixture has no prior_summary table")


ALL = (check_design, check_logp, check_gradient, check_derived, check_predictions, check_jax, check_default_prior)
