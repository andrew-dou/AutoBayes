"""Shrinkage priors on b, brms's horseshoe() and R2D2() (brms-parity D07, docs/shrinkage-priors.md): the constructors (brms's
defaults and argument order, the refused arguments), the joint log density of the graph against one written out with SciPy
from the papers' definitions (with and without the Jacobian of brms's coordinates), b = zb * sdb, value_point, the refusals of
every construct no frozen case verifies, describe(), the prior table, the target identity, prior-predictive draws, the fit
record, and the oracle's extension (a corrupted fixture fails the gates). No MCMC; the brms comparison is the oracle cases
gauss_horseshoe and gauss_r2d2."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import autobayes as ab
from autobayes.monotonic import stick_breaking_constrain, stick_breaking_free
from autobayes.priors import PriorError, brms_prior

FORMULA = "y ~ x + w + z"
COLUMNS = ["x", "w", "z[b]", "z[c]"]


def frame(n=30, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"x": rng.normal(size=n), "w": rng.normal(size=n), "z": np.array(["a", "b", "c"])[np.arange(n) % 3],
                         "g": [f"l{k}" for k in np.arange(n) % 4], "y": rng.normal(1.0, 1.5, n), "o": rng.uniform(1, 2, n),
                         "wt": rng.uniform(1, 2, n), "ord": np.arange(n) % 4, "t": np.arange(n), "y2": rng.normal(size=n)})


def priors(b, **extra):
    return ab.Priors(**{"intercept": ab.Normal(0, 5), "sigma": ab.HalfNormal(1), "b": b, **extra})


def build(b, formula=FORMULA, data=None, **kwargs):
    return ab.gaussian(formula, frame() if data is None else data, priors(b), categorical={"z": "a"}, **kwargs)


HS = ab.Horseshoe(3, 0.5, 1, 2, 4)
R2 = ab.R2D2(0.4, 3, 0.7)
HS_VALUES = {"zb": np.array([0.4, -1.1, 0.3, 0.9]), "hs_global": 0.35, "hs_slab": 1.7, "hs_local": np.array([0.8, 2.5, 0.1, 1.3])}
R2_VALUES = {"zb": np.array([0.4, -1.1, 0.3, 0.9]), "R2D2_R2": 0.3, "R2D2_phi": np.array([0.1, 0.4, 0.2, 0.3])}


# -- the constructors -----------------------------------------------------------------------------------------------------
def test_constructors_take_brms_defaults_and_argument_order():
    assert ab.Horseshoe().params == {"df": 1.0, "scale_global": 1.0, "df_global": 1.0, "scale_slab": 2.0, "df_slab": 4.0}
    assert ab.R2D2().params == {"mean_R2": 0.5, "prec_R2": 2.0, "cons_D2": 0.5}
    assert ab.Horseshoe(3, 0.5, 1, 2, 4) == ab.Horseshoe(df=3, scale_global=0.5, df_global=1, scale_slab=2, df_slab=4)
    assert ab.R2D2(0.4, 3, 0.7) == ab.R2D2(mean_R2=0.4, prec_R2=3, cons_D2=0.7)
    assert brms_prior(HS) == "horseshoe(df = 3, scale_global = 0.5, df_global = 1, scale_slab = 2, df_slab = 4)"
    assert brms_prior(R2) == "R2D2(mean_R2 = 0.4, prec_R2 = 3, cons_D2 = 0.7)"
    assert brms_prior(ab.R2D2(cons_D2=[0.5, 1, 2])) == "R2D2(mean_R2 = 0.5, prec_R2 = 2, cons_D2 = c(0.5, 1, 2))"
    assert str(HS) == "Horseshoe(df=3, scale_global=0.5, df_global=1, scale_slab=2, df_slab=4)"


@pytest.mark.parametrize("make, match", [
    (lambda: ab.Horseshoe(par_ratio=0.1), r"par_ratio = \.\.\.\) is not offered"),
    (lambda: ab.Horseshoe(autoscale=False), r"autoscale = FALSE\) is not offered"),
    (lambda: ab.R2D2(autoscale=False), r"autoscale = FALSE\) is not offered"),
    (lambda: ab.Horseshoe(main=True), r"main = TRUE\) is not offered"),
    (lambda: ab.R2D2(main=True), r"main = TRUE\) is not offered"),
    (lambda: ab.Horseshoe(df=0), "df must be a positive finite number"),
    (lambda: ab.Horseshoe(scale_slab=float("inf")), "scale_slab must be a positive finite number"),
    (lambda: ab.Horseshoe(df_slab=True), "df_slab must be a positive finite number"),
    (lambda: ab.R2D2(mean_R2=1), r"must lie in \(0, 1\)"),
    (lambda: ab.R2D2(prec_R2=-1), "prec_R2 must be a positive finite number"),
    (lambda: ab.R2D2(cons_D2=[1, 0]), "cons_D2 must be a positive finite number"),
    (lambda: ab.R2D2(cons_D2=[]), "at least one concentration"),
])
def test_constructor_refusals(make, match):
    with pytest.raises(PriorError, match=match):
        make()


# -- the density ----------------------------------------------------------------------------------------------------------
def _point(m, values, intercept=0.7, sigma=1.3):
    return m.value_point(intercept=intercept, sigma=sigma, shrinkage=values)


def _by_hand(m, values, intercept=0.7, sigma=1.3):
    """The joint log density written out from the papers' definitions with SciPy (brms's parameterisation), and its log
    Jacobian on brms's coordinates (logs of the scales, the logit of R2, Stan's stick-breaking map of phi)."""
    d, prior = m.design, m.priors.b
    p = {k: float(v) for k, v in prior.params.items()}
    zb = values["zb"]
    if prior.name == "Horseshoe":
        tau, lam, c2 = values["hs_global"], values["hs_local"], p["scale_slab"] ** 2 * values["hs_slab"]
        sdb = tau * np.sqrt(c2 * lam ** 2 / (c2 + tau ** 2 * lam ** 2))
        lp = (np.log(2) + stats.t.logpdf(tau, p["df_global"], 0, p["scale_global"] * sigma)
              + stats.invgamma.logpdf(values["hs_slab"], p["df_slab"] / 2, scale=p["df_slab"] / 2)
              + np.sum(np.log(2) + stats.t.logpdf(lam, p["df"])))
        jac = np.log(tau) + np.log(values["hs_slab"]) + np.sum(np.log(lam))
    else:
        R2, phi = values["R2D2_R2"], values["R2D2_phi"]
        sdb = np.sqrt(phi * sigma ** 2 * R2 / (1 - R2))
        lp = (stats.beta.logpdf(R2, p["mean_R2"] * p["prec_R2"], (1 - p["mean_R2"]) * p["prec_R2"])
              + stats.dirichlet.logpdf(phi, np.full(phi.size, p["cons_D2"])))
        jac = np.log(R2 * (1 - R2)) + stick_breaking_constrain(stick_breaking_free(phi), phi.size)[1]
    b = zb * sdb
    mu = intercept + d.X_model @ b
    lp += (np.sum(stats.norm.logpdf(d.y, mu, sigma)) + stats.norm.logpdf(intercept, 0, 5) + np.log(2)
           + stats.norm.logpdf(sigma, 0, 1) + np.sum(stats.norm.logpdf(zb)))
    return lp, lp + jac + np.log(sigma), b, sdb


@pytest.mark.parametrize("prior, values", [(HS, HS_VALUES), (R2, R2_VALUES)], ids=["horseshoe", "r2d2"])
def test_the_log_density_is_the_papers_with_brms_coordinates(prior, values):
    m = build(prior)
    assert list(m.spec.columns) == COLUMNS and np.allclose(m.design.X_model, m.design.X_raw - m.design.X_raw.mean(0))
    vp = _point(m, values)
    lp_nojac, lp, b, sdb = _by_hand(m, values)
    pmm = m.pymc_model
    # PyMC's HalfStudentT computes its log 2 in float32 (pt.log(2) of an integer, #45): 1.9e-9 per half-Student-t variable,
    # hs_global and the four hs_local, as for every HalfStudentT prior of the library; the oracle's tolerance is 1e-8 relative
    offset = (len(values["zb"]) + 1) * (float(np.float32(np.log(2))) - np.log(2)) if prior.name == "Horseshoe" else 0.0
    assert float(pmm.compile_logp(jacobian=False)(vp)) == pytest.approx(lp_nojac + offset, rel=1e-13)
    assert float(pmm.compile_logp()(vp)) == pytest.approx(lp + offset, rel=1e-13)
    fn = pmm.compile_fn(pmm.replace_rvs_by_values([pmm["sdb"], pmm["x"], pmm["w"], pmm["z"]]), inputs=pmm.value_vars,
                        point_fn=True, on_unused_input="ignore")
    got_sdb, x, w, z = (np.atleast_1d(np.asarray(v, dtype=float)) for v in fn(vp))
    np.testing.assert_allclose(got_sdb, sdb, rtol=1e-13)
    np.testing.assert_allclose(np.concatenate([x, w, z]), b, rtol=1e-13)


def test_the_graph_names_brms_parameters_on_brms_coordinates():
    hs, r2 = build(HS), build(R2)
    assert [v.name for v in hs.pymc_model.value_vars] == ["sigma_log__", "Intercept", "zb", "hs_global_log__", "hs_slab_log__",
                                                          "hs_local_log__"]
    assert [v.name for v in r2.pymc_model.value_vars] == ["sigma_log__", "Intercept", "zb", "R2D2_R2_logodds__",
                                                          "R2D2_phi_stickbreaking__"]
    assert {d.name for d in hs.pymc_model.deterministics} >= {"x", "w", "z", "sdb"}
    assert {d.name for d in r2.pymc_model.deterministics} >= {"x", "w", "z", "sdb", "R2D2_tau2"}
    assert [str(c) for c in hs.pymc_model.coords["zb_dim"]] == COLUMNS
    assert hs.shrinkage_names["lprior"] == ("hs_global", "hs_slab") and r2.shrinkage_names["lprior"] == ("R2D2_R2",)
    assert build(ab.Normal(0, 1)).shrinkage_names == {}


def test_value_point_takes_zb_and_the_scales_not_b():
    m, r = build(HS), build(R2)
    vp = _point(m, HS_VALUES)
    np.testing.assert_allclose(vp["hs_local_log__"], np.log(HS_VALUES["hs_local"]))
    np.testing.assert_allclose(_point(r, R2_VALUES)["R2D2_phi_stickbreaking__"], stick_breaking_free(R2_VALUES["R2D2_phi"]))
    assert float(_point(r, R2_VALUES)["R2D2_R2_logodds__"]) == pytest.approx(np.log(0.3 / 0.7))
    with pytest.raises(ValueError, match="b is a transformed parameter"):
        m.value_point(intercept=0.7, sigma=1.3, b=np.zeros(4), shrinkage=HS_VALUES)
    with pytest.raises(ValueError, match="shrinkage must be a mapping over exactly"):
        m.value_point(intercept=0.7, sigma=1.3)
    with pytest.raises(ValueError, match="shrinkage must be a mapping over exactly"):
        m.value_point(intercept=0.7, sigma=1.3, shrinkage=R2_VALUES)
    with pytest.raises(ValueError, match="hs_local must have 4 finite positive entries"):
        m.value_point(intercept=0.7, sigma=1.3, shrinkage={**HS_VALUES, "hs_local": -HS_VALUES["hs_local"]})
    with pytest.raises(ValueError, match=r"R2D2_R2 must be a finite scalar in \(0, 1\)"):
        r.value_point(intercept=0.7, sigma=1.3, shrinkage={**R2_VALUES, "R2D2_R2": 1.0})
    with pytest.raises(ValueError, match="R2D2_phi must sum to 1"):
        r.value_point(intercept=0.7, sigma=1.3, shrinkage={**R2_VALUES, "R2D2_phi": np.array([0.1, 0.4, 0.2, 0.4])})
    plain = build(ab.Normal(0, 1))
    with pytest.raises(ValueError, match="no shrinkage prior on b"):
        plain.value_point(intercept=0.7, sigma=1.3, b=np.zeros(4), shrinkage=HS_VALUES)
    with pytest.raises(ValueError, match="b is required"):
        plain.value_point(intercept=0.7, sigma=1.3)


def test_terms_that_are_columns_of_the_design_compose():
    """Interactions, transforms and a single coefficient take the prior; the coefficients stay zb * sdb column by column (the
    model checks it when it is built), in design-column order, on the coordinate zb_dim."""
    m = ab.gaussian("y ~ x * z + log(o)", frame(), priors(HS), categorical={"z": "a"})
    assert [str(c) for c in m.pymc_model.coords["zb_dim"]] == list(m.spec.columns) and len(m.spec.columns) == 6
    one = ab.gaussian("y ~ x", frame(), priors(HS))
    assert [v.name for v in one.pymc_model.value_vars][-1] == "hs_local_log__"
    vector = ab.gaussian("y ~ x + w", frame(), priors(ab.R2D2(cons_D2=[1, 3])))
    assert "Dirichlet(1, 3)" in vector.describe()


# -- what is refused ---------------------------------------------------------------------------------------------------------
H = ab.Horseshoe()
REFUSED = {
    "student": (lambda d: ab.student("y ~ x", d, priors(H), nu=4), "Gaussian family only"),
    "poisson": (lambda d: ab.poisson("ord ~ x", d, ab.Priors(intercept=ab.Normal(0, 5), b=H)), "the poisson family is refused"),
    "group": (lambda d: ab.gaussian("y ~ x + (1 | g)", d, priors(H, sd=ab.HalfNormal(1))), r"group-level terms \['\(1 \| g\)'\]"),
    "offset": (lambda d: ab.gaussian("y ~ x + offset(o)", d, priors(H)), "an offset"),
    "weights": (lambda d: ab.gaussian("y | weights(wt) ~ x", d, priors(H)), "response addition terms"),
    "distributional": (lambda d: ab.gaussian(["y ~ x", "sigma ~ w"], d, ab.Priors(intercept=ab.Normal(0, 5), b=H,
                                                                              sigma={"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 1)})),
                       "distributional formulas"),
    "mo": (lambda d: ab.gaussian("y ~ x + mo(ord)", d, priors(H, simo=ab.Dirichlet(1))), "special terms"),
    "gp": (lambda d: ab.gaussian("y ~ x + gp(w)", d, priors(H, sdgp=ab.HalfNormal(1), lscale=ab.InverseGamma(3, 1))),
           "Gaussian processes"),
    "ar": (lambda d: ab.gaussian("y ~ x + ar(time = t)", d, priors(H, ar=ab.Flat())), "an autocorrelation term"),
    "no intercept": (lambda d: ab.gaussian("y ~ 0 + x", d, ab.Priors(b=H, sigma=ab.HalfNormal(1))), "without an intercept"),
    "not centred": (lambda d: ab.gaussian("y ~ x", d, priors(H), center_predictors=False), "center_predictors=False"),
    "intercept only": (lambda d: ab.gaussian("y ~ 1", d, priors(H)), "no population-level coefficient"),
    "one coefficient's": (lambda d: ab.gaussian("y ~ x + w", d, priors({"x": H, "w": ab.Normal(0, 1)})), r"found at b\['x'\]"),
    "sd": (lambda d: ab.gaussian("y ~ x + (1 | g)", d, priors(ab.Normal(0, 1), sd=H)), "found at sd"),
    "sigma": (lambda d: ab.gaussian("y ~ x", d, ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 1), sigma=H)), "found at sigma"),
    "intercept": (lambda d: ab.gaussian("y ~ x", d, ab.Priors(intercept=ab.R2D2(), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1))),
                  "found at intercept"),
    "sigma's formula": (lambda d: ab.gaussian(["y ~ x", "sigma ~ w"], d, ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 1),
                                                                               sigma={"intercept": ab.Normal(0, 1), "b": H})),
                        r"found at sigma\['b'\]"),
    "mixture": (lambda d: ab.mixture("y ~ x", d, ab.Priors(components=[priors(H), priors(H)], theta=ab.Dirichlet(1)),
                                     families=["gaussian", "gaussian"]), r"found at components\[0\]\.b"),
    "multivariate": (lambda d: ab.gaussian("mvbind(y, y2) ~ x", d, ab.Priors(resp={"y": priors(H), "y2": priors(ab.Normal(0, 1))}),
                                           rescor=False), r"found at resp\['y'\]\.b"),
    "non-linear": (lambda d: ab.gaussian(["y ~ a * exp(b * x)", "a ~ 1", "b ~ 1"], d,
                                         ab.Priors(nlpars={"a": {"b": H}, "b": {"b": ab.Normal(0, 1)}}, sigma=ab.HalfNormal(1)),
                                         nl=True), r"found at nlpars\['a'\]\['b'\]"),
    "R2D2 of one coefficient": (lambda d: ab.gaussian("y ~ x", d, priors(ab.R2D2())), "a single coefficient"),
    "cons_D2 of another length": (lambda d: ab.gaussian("y ~ x + w", d, priors(ab.R2D2(cons_D2=[1, 2, 3]))),
                                  "gives 3 concentrations cons_D2 for a model of 2 coefficients"),
}


@pytest.mark.parametrize("what", list(REFUSED))
def test_every_unverified_construct_is_refused_by_name(what):
    make, match = REFUSED[what]
    with pytest.raises(PriorError, match=match):
        make(frame())


# -- describe(), the prior table, the target identity -----------------------------------------------------------------------
def test_describe():
    text = build(HS).describe()
    assert "with categorical predictor(s) and a shrinkage prior on b (horseshoe)" in text
    for line in ["  b ~ Horseshoe(df=3, scale_global=0.5, df_global=1, scale_slab=2, df_slab=4)   (brms horseshoe(df = 3, "
                 "scale_global = 0.5, df_global = 1, scale_slab = 2, df_slab = 4) on the whole class b, autoscale = TRUE",
                 "    b_k = zb[k] * sdb[k] for each coefficient k = x, w, z[b], z[c]",
                 "    hs_global ~ HalfStudentT(nu=1, sigma=0.5 * sigma)", "    hs_slab ~ InverseGamma(alpha=2, beta=2)",
                 "    hs_local[k] ~ HalfStudentT(nu=3, sigma=1)", "c2 = scale_slab^2 * hs_slab = 4 * hs_slab",
                 "  hs_local_log__           log(hs_local[x, w, z[b], z[c]])", "x, w, z = b = zb * sdb (brms's b)"]:
        assert line in text
    assert "b_x ~" not in text
    text = build(R2).describe()
    for line in ["  b ~ R2D2(mean_R2=0.4, prec_R2=3, cons_D2=0.7)", "R2D2_tau2 = sigma^2 * R2D2_R2 / (1 - R2D2_R2)",
                 "    R2D2_R2 ~ Beta(alpha=1.2, beta=1.8)", "    R2D2_phi ~ Dirichlet(0.7, 0.7, 0.7, 0.7)",
                 "  R2D2_phi_stickbreaking__ R2D2_phi[x, w, z[b], z[c]] on Stan's stick-breaking simplex coordinates (3 entries"]:
        assert line in text


@pytest.mark.parametrize("case", ["gauss_horseshoe", "gauss_r2d2"])
def test_the_prior_table_writes_the_prior_on_the_class_row_as_brms_does(case):
    """brms's prior_summary of the frozen case: the prior's string on the class row of b, the coefficient rows empty."""
    from tests.oracle import gates

    b = gates.Bundle(case)
    ours = [(r["class"], r["coef"], r["prior"]) for r in b.model.prior_table()]
    theirs = [(r["class"], r["coef"], r["prior"]) for r in b.fixture.prior_summary]
    assert ours == theirs


def test_target_identity():
    m = build(HS)
    assert build(HS).target_identity == m.target_identity
    assert build(ab.Horseshoe(3, 0.5, 1, 2, 5)).target_identity != m.target_identity
    assert build(R2).target_identity not in (m.target_identity, build(ab.R2D2(0.4, 3, 0.8)).target_identity)
    assert build(ab.Normal(0, 1)).target_identity != m.target_identity


# -- forward draws, the fit record -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("prior", [HS, R2], ids=["horseshoe", "r2d2"])
def test_prior_predictive(prior):
    m = build(prior)
    out = m.prior_predictive(draws=40, seed=3)
    pr = out.prior
    b = np.concatenate([pr["x"].values[..., None], pr["w"].values[..., None], pr["z"].values], axis=-1)
    np.testing.assert_allclose(b, pr["zb"].values * pr["sdb"].values, rtol=1e-12)
    assert np.isfinite(out.prior_predictive["y"].values).all() and out.prior_predictive["y"].shape == (1, 40, 30)
    if prior.name == "Horseshoe":
        assert (pr["hs_global"].values > 0).all() and (pr["hs_local"].values > 0).all() and (pr["hs_slab"].values > 0).all()
    else:
        assert np.allclose(pr["R2D2_phi"].values.sum(-1), 1.0) and ((pr["R2D2_R2"].values > 0) & (pr["R2D2_R2"].values < 1)).all()
        np.testing.assert_allclose(pr["R2D2_tau2"].values, pr["sigma"].values ** 2 * pr["R2D2_R2"].values / (1 - pr["R2D2_R2"].values))


def test_default_priors_are_brms_and_take_a_shrinkage_prior_on_b():
    pri = ab.default_priors(FORMULA, frame(), family="gaussian", categorical={"z": "a"})
    assert pri.b == ab.Flat()                                            # brms's default for b is flat, never a shrinkage prior
    from dataclasses import replace
    m = ab.gaussian(FORMULA, frame(), replace(pri, b=HS), categorical={"z": "a"})
    assert m.shrinkage_names["prior"] == "Horseshoe" and m.default_prior_labels == ["intercept", "sigma"]


def test_fit_record_on_the_frozen_points(monkeypatch):
    from tests.oracle import adapter, gates
    import autobayes.session as session

    b = gates.Bundle("gauss_horseshoe")
    post = adapter.draws(b.fixture, b.model, b.cmap, b.gmap)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rows = session._summary_rows(b.model, post)
    assert {"x", "w", "z[q]", "z[r]", "xpos", "sigma"} <= set(rows)
    monkeypatch.setattr(session, "_summary_rows", lambda *args: {})
    monkeypatch.setattr(session, "environment", lambda: {})
    fit = SimpleNamespace(target_identity=b.model.target_identity, idata=post, backend="synthetic", device_requested="cpu",
                          device_actual="cpu", platform="cpu", dtype="float64", chain_method="none", chains=1, draws=8,
                          tune=0, target_accept=0.9, seed=9, wall_seconds=0, notes=["the oracle's frozen points; not a fit"])
    rec = ab.fit_record(b.model, fit, panel={"values": {}, "status": {}, "overall": "not evaluated"})
    assert ab.compare_records(rec, json.loads(json.dumps(rec)))["agree"]
    pred = b.model.predict(post, kind="linear")["eta"].to_numpy()[0]
    np.testing.assert_allclose(pred, b.fixture.prediction("fitted", "linpred"), rtol=1e-9)


# -- the oracle's extension ---------------------------------------------------------------------------------------------------
def _corrupted(bundle, mutate):
    from tests.oracle import fixture as fx

    raw = copy.deepcopy(bundle.fixture.raw)
    mutate(raw)
    return bundle.with_fixture(fx.Fixture(raw))


@pytest.mark.parametrize("case", ["gauss_horseshoe", "gauss_r2d2"])
def test_the_gates_see_a_corrupted_shrinkage_fixture(case):
    from tests.oracle import gates
    from tests.oracle.adapter import OracleMismatch

    b = gates.Bundle(case)
    for gate in gates.ALL:
        gate(b)
    key = "hs_scale_global" if case == "gauss_horseshoe" else "R2D2_cons_D2"

    def data(raw):                                                      # another hyperparameter of the prior
        raw["standata"][key]["values"][0] *= 1.0 + 1e-9
    with pytest.raises(OracleMismatch, match=rf"brms {key} is"):
        gates.check_design(_corrupted(b, data))

    def scales(raw):                                                    # brms's transformed scales at a point
        raw["points"][2]["pars"]["sdb"]["values"][1] *= 1.0 + 1e-6
    with pytest.raises(OracleMismatch, match=r"point 2: the graph's sdb differs"):
        gates.check_derived(_corrupted(b, scales))

    def lprior(raw):                                                    # brms's prior part at a point
        raw["points"][4]["pars"]["lprior"]["values"][0] += 1e-6
    with pytest.raises(OracleMismatch, match=r"point 4: log density"):
        gates.check_logp(_corrupted(b, lprior))
    plain = ab.gaussian(b.case["autobayes"]["formula"], b.data, priors(ab.Normal(0, 1)), categorical={"z": "p"})
    with pytest.raises(OracleMismatch, match="the AutoBayes model has none"):
        gates.check_design(gates.Bundle(case, model=plain))


def test_the_adapter_checks_the_simplex_coordinates_of_r2d2():
    from tests.oracle import adapter, gates
    from tests.oracle.adapter import OracleMismatch

    b = gates.Bundle("gauss_r2d2")

    def phi(raw):                                                       # a simplex whose stick-breaking coordinates are not u
        values = raw["points"][5]["pars"]["R2D2_phi"]["values"]
        values[0], values[1] = values[1], values[0]
    bad = _corrupted(b, phi)
    with pytest.raises(OracleMismatch, match=r"point 5: Stan's stick-breaking coordinates of R2D2_phi"):
        adapter.coordinate_map(bad.fixture, bad.model, bad.cmap, bad.gmap)
