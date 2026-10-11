"""brms's default priors, opt-in (brms-parity A07): ``ab.default_priors()`` reproduces brms 2.23's ``default_prior()``
table for a formula, data and family; its priors are the constructors' own (``Flat`` for brms's flat ``b``,
``HalfStudentT`` on the bounded classes, ``InverseGamma`` for the negative binomial ``shape``, ``LKJ(1)``, the truncated
``Gamma(2, 0.1)`` nu); ``describe()`` and fit records mark them; nothing is filled in without being asked for.

The expected tables were printed by brms 2.23.0's ``default_prior()`` (R 4.5.2, C collation) on ``FRAME`` below; the
oracle checks the same arithmetic against brms on every frozen fixture (gate G6) and the densities under the defaults on
five fixtures frozen without priors (G2). Nothing here samples."""
from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pandas as pd
import pymc as pm
import pytest
from scipy import stats

import autobayes as ab
from autobayes.default_priors import BRMS_VERSION, location_scale, r_number, r_round
from autobayes.families import FAMILIES, get_family
from autobayes.formula import parse_formula

FRAME = pd.DataFrame({
    "y": [12.31, -40.2, 7.75, 103.4, -0.5, 66.0, 18.25, -71.3, 2.2, 35.0, 49.9, -12.6],      # MAD far above 2.5
    "x": [0.12, -1.3, 0.85, 1.7, -0.44, 0.03, -2.1, 0.9, 1.15, -0.6, 0.31, -0.05],
    "w": [1.4, 0.2, -0.7, 2.3, 0.95, -1.6, 0.4, 1.1, -0.25, 0.6, 1.8, -0.9],
    "e": [1.5, 2.25, 0.8, 3.1, 1.05, 0.6, 2.7, 1.9, 0.45, 1.35, 2.05, 4.2],
    "z": ["p", "q", "r", "p", "q", "r", "p", "q", "r", "p", "q", "r"],
    "g": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d"],
    "h": ["h1", "h2", "h1", "h2", "h1", "h2", "h1", "h2", "h1", "h2", "h1", "h2"],
    "cnt": [0, 3, 0, 1, 7, 2, 0, 0, 12, 4, 1, 0],                                             # zeros: brms reads them as 0.1
    "yb": [0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0],
    "k": [2, 5, 0, 7, 3, 1, 4, 6, 2, 0, 5, 3],
    "size": [6, 9, 4, 10, 8, 3, 7, 9, 5, 2, 6, 8],
    "ypos": [0.82, 3.5, 1.27, 0.06, 12.4, 2.2, 0.95, 5.75, 0.31, 1.6, 8.9, 0.47],
    "prop": [0.12, 0.55, 0.31, 0.08, 0.77, 0.43, 0.26, 0.64, 0.19, 0.38, 0.91, 0.5],
})
Z = {"z": "p"}

# (formula, family, keyword arguments, brms's rows as (class, coef, group, prior, lb)); every row's ub, resp, dpar, nlpar
# and tag are empty and its source "default"
BRMS = {
    "gaussian, correlated slope": ("y ~ x + z + (1 + x | g)", "gaussian", {"categorical": Z}, [
        ("Intercept", "", "", "student_t(3, 10, 35.3)", ""), ("b", "", "", "", ""), ("b", "x", "", "", ""),
        ("b", "zq", "", "", ""), ("b", "zr", "", "", ""), ("cor", "", "", "lkj(1)", ""), ("cor", "", "g", "", ""),
        ("sd", "", "", "student_t(3, 0, 35.3)", "0"), ("sd", "", "g", "", ""), ("sd", "Intercept", "g", "", ""),
        ("sd", "x", "g", "", ""), ("sigma", "", "", "student_t(3, 0, 35.3)", "0")]),
    "gaussian, uncentred, independent slope": ("y ~ x * w + (1 + x || g)", "gaussian", {"center_predictors": False}, [
        ("b", "", "", "", ""), ("b", "Intercept", "", "", ""), ("b", "w", "", "", ""), ("b", "x", "", "", ""),
        ("b", "x:w", "", "", ""), ("sd", "", "", "student_t(3, 0, 35.3)", "0"), ("sd", "", "g", "", ""),
        ("sd", "Intercept", "g", "", ""), ("sd", "x", "g", "", ""), ("sigma", "", "", "student_t(3, 0, 35.3)", "0")]),
    "poisson, offset and zeros": ("cnt ~ x + offset(log(e)) + (1 | g)", "poisson", {}, [
        ("Intercept", "", "", "student_t(3, -0.412687108229976, 3.1)", ""), ("b", "", "", "", ""), ("b", "x", "", "", ""),
        ("sd", "", "", "student_t(3, 0, 3.1)", "0"), ("sd", "", "g", "", ""), ("sd", "Intercept", "g", "", "")]),
    "negative binomial, categorical first": ("cnt ~ z * x", "negbinomial", {"categorical": Z}, [
        ("Intercept", "", "", "student_t(3, 0, 3.1)", ""), ("b", "", "", "", ""), ("b", "x", "", "", ""),
        ("b", "zq", "", "", ""), ("b", "zq:x", "", "", ""), ("b", "zr", "", "", ""), ("b", "zr:x", "", "", ""),
        ("shape", "", "", "inv_gamma(0.4, 0.3)", "0")]),
    "bernoulli, probit": ("yb ~ x + (1 | g)", "bernoulli", {"link": "probit"}, [
        ("Intercept", "", "", "student_t(3, 0, 2.5)", ""), ("b", "", "", "", ""), ("b", "x", "", "", ""),
        ("sd", "", "", "student_t(3, 0, 2.5)", "0"), ("sd", "", "g", "", ""), ("sd", "Intercept", "g", "", "")]),
    "binomial": ("k | trials(size) ~ x + z", "binomial", {"categorical": Z}, [
        ("Intercept", "", "", "student_t(3, 0, 2.5)", ""), ("b", "", "", "", ""), ("b", "x", "", "", ""),
        ("b", "zq", "", "", ""), ("b", "zr", "", "", "")]),
    "student, intercept only, learned nu": ("y ~ 1 + (1 | g)", "student", {}, [
        ("Intercept", "", "", "student_t(3, 10, 35.3)", ""), ("nu", "", "", "gamma(2, 0.1)", "1"),
        ("sd", "", "", "student_t(3, 0, 35.3)", "0"), ("sd", "", "g", "", ""), ("sd", "Intercept", "g", "", ""),
        ("sigma", "", "", "student_t(3, 0, 35.3)", "0")]),
    "lognormal, derived factor": ("ypos ~ x + (1 | g:h)", "lognormal", {}, [
        ("Intercept", "", "", "student_t(3, 0.4, 2.5)", ""), ("b", "", "", "", ""), ("b", "x", "", "", ""),
        ("sd", "", "", "student_t(3, 0, 2.5)", "0"), ("sd", "", "g:h", "", ""), ("sd", "Intercept", "g:h", "", ""),
        ("sigma", "", "", "student_t(3, 0, 2.5)", "0")]),
    "gamma, transforms": ("ypos ~ log(e) + I(x^2)", "gamma", {}, [
        ("Intercept", "", "", "student_t(3, 0.4, 2.5)", ""), ("b", "", "", "", ""), ("b", "IxE2", "", "", ""),
        ("b", "loge", "", "", ""), ("shape", "", "", "gamma(0.01, 0.01)", "0")]),
    "beta, no intercept": ("prop ~ 0 + x + z", "beta", {"categorical": Z}, [
        ("b", "", "", "", ""), ("b", "x", "", "", ""), ("b", "zp", "", "", ""), ("b", "zq", "", "", ""),
        ("b", "zr", "", "", ""), ("phi", "", "", "gamma(0.01, 0.01)", "0")]),
}


def _table(formula, family, kwargs, priors=None):
    priors = priors or ab.default_priors(formula, FRAME, family=family, **kwargs)
    spec = parse_formula(formula, FRAME, family, kwargs.get("categorical"))
    return priors.table(spec, get_family(family, kwargs.get("link")), center_predictors=kwargs.get("center_predictors", True))


@pytest.mark.parametrize("label", sorted(BRMS))
def test_the_table_is_brms_default_prior(label):
    formula, family, kwargs, rows = BRMS[label]
    table = _table(formula, family, kwargs)
    assert [(r["class"], r["coef"], r["group"], r["prior"], r["lb"]) for r in table] == rows
    assert all(r["ub"] == r["resp"] == r["dpar"] == r["nlpar"] == r["tag"] == "" and r["source"] == "default" for r in table)
    assert all(list(r) == ["prior", "class", "coef", "group", "resp", "dpar", "nlpar", "lb", "ub", "tag", "source"] for r in table)


def test_location_and_scale_are_brms_arithmetic():
    """brms's def_scale_prior: round(median(y_link), 1) and max(2.5, round(mad(y_link), 1)) with R's round(), median()
    and mad(); the values are R's."""
    assert location_scale([0.25, 0.25, 1], "identity") == (0.2, 2.5)            # R 4's round(): a tie goes to the even digit
    assert location_scale([0.35, 0.35, 2], "identity") == (0.3, 2.5)            # 0.35 is 0.34999999999999998 in binary
    assert location_scale([2.45, 2.45, 2.45, 9], "identity") == (2.5, 2.5)      # 2.4500000000000002 rounds up
    assert location_scale([-0.25, 1, -0.25], "identity") == (-0.2, 2.5)
    loc, scale = location_scale([-0.05, -0.05, 3], "identity")                 # R's round(-0.05, 1) is -0, printed 0
    assert (loc, scale) == (0.0, 2.5) and np.copysign(1.0, loc) == 1.0
    assert location_scale(FRAME["y"], "identity") == (10.0, 35.3)              # a large MAD: 1.4826 * 23.8 = 35.29 -> 35.3
    assert location_scale(FRAME["y"], None) == (0.0, 2.5)                       # logit, probit: brms's constants
    assert location_scale([1, 2], "log") == (0.3, 2.5)                          # the median of the logs (0.35), not log(1.5) = 0.41
    assert location_scale(FRAME["cnt"], "log") == (0.0, 3.1)                    # zeros read as 0.1: log(0.1) = -2.3 in the MAD
    loc, _ = location_scale(FRAME["cnt"], "log", offset=np.log(FRAME["e"]))
    assert r_number(loc) == "-0.412687108229976"                                # minus R's mean of the offset, not rounded
    # R 4.5.2's round(c(0.15, 0.25, 0.45, 1.05, -0.15, 2.45, 0.35), 1): the nearer candidate in double arithmetic, not
    # the decimal reading (0.45 and 1.05 go down, 2.45 up)
    assert [r_round(v, 1) for v in (0.15, 0.25, 0.45, 1.05, -0.15, 2.45, 0.35)] == [0.1, 0.2, 0.4, 1.0, -0.1, 2.5, 0.3]


def test_numbers_print_as_r_prints_them():
    expected = {1e5: "1e+05", 0.0001: "1e-04", 123456.7: "123456.7", -0.978004218445523: "-0.978004218445523",
                1 / 3: "0.333333333333333", 2.5: "2.5", 3.0: "3", 1e15: "1e+15", 1e-15: "1e-15", 100.0: "100", -0.0: "0",
                4.5512891607359052: "4.5512891607359"}                          # R's 15-digit scaling rounds twice here
    assert {x: r_number(x) for x in expected} == expected


def test_the_defaults_are_the_constructors_priors():
    formula, family, kwargs, _ = BRMS["gaussian, correlated slope"]
    pri = ab.default_priors(formula, FRAME, family=family, **kwargs)
    hst = ab.HalfStudentT(3, 35.3)
    assert pri.intercept == ab.StudentT(3, 10, 35.3) and pri.b == ab.Flat() and pri.sigma == hst
    assert pri.sd == {"g": {"Intercept": hst, "x": hst}} and pri.cor == {"g": ab.LKJ(1)}
    assert pri.shape is None and pri.phi is None and pri.nu is None
    every = [pri.intercept, pri.b, pri.sigma, *pri.sd["g"].values(), pri.cor["g"]]
    assert {p.source for p in every} == {"default"}
    model = ab.gaussian(formula, FRAME, pri, categorical=Z)
    assert model.default_prior_labels == ["intercept", "b:x", "b:z[q]", "b:z[r]", "sd:g", "sd:x|g", "cor:g", "sigma"]
    assert model.prior_table() == _table(formula, family, kwargs)
    assert ab.default_priors("cnt ~ x", FRAME, family="negbinomial").shape == ab.InverseGamma(0.4, 0.3)
    assert ab.default_priors("ypos ~ x", FRAME, family="gamma").shape == ab.Gamma(0.01, 0.01)
    assert ab.default_priors("prop ~ x", FRAME, family="beta").phi == ab.Gamma(0.01, 0.01)
    assert ab.default_priors("y ~ x", FRAME, family="student").nu == ab.Gamma(2, 0.1)           # nu=None: learned
    assert ab.default_priors("y ~ x", FRAME, family="student", nu=4).nu is None                 # fixed: no prior
    assert ab.default_priors("y ~ x", FRAME, center_predictors=False).intercept == ab.Flat()    # brms's flat b_Intercept
    assert ab.default_priors("y ~ 1 + (1 | g)", FRAME).b is None                                # no coefficient (D2)
    two = ab.default_priors("y ~ x + (1 | g) + (1 + x || h)", FRAME)
    assert two.sd == {"g": ab.HalfStudentT(3, 35.3), "h": {"Intercept": ab.HalfStudentT(3, 35.3), "x": ab.HalfStudentT(3, 35.3)}}
    assert two.cor is None


def test_flat_b_and_the_default_densities_in_the_graph():
    formula = "y ~ x + z + (1 | g)"
    model = ab.gaussian(formula, FRAME, ab.default_priors(formula, FRAME, categorical=Z), categorical=Z)
    pmm = model.pymc_model
    for name in ("x", "z"):
        assert isinstance(pmm[name].owner.op, pm.distributions.continuous.FlatRV)
    point = model.value_point(intercept=4.0, b=[1.5, -2.0, 0.5], sd=3.0, z=np.linspace(-1, 1, 4), sigma=20.0)
    prior = pmm.compile_logp(vars=[pmm[n] for n in ("Intercept", "x", "z", "1|g_sigma", "sigma")], jacobian=False)
    expected = (stats.t.logpdf(4.0, 3, loc=10, scale=35.3)                    # the Intercept: student_t(3, 10, 35.3)
                + 2 * np.log(2) + stats.t.logpdf(3.0, 3, scale=35.3) + stats.t.logpdf(20.0, 3, scale=35.3))   # half-t sd, sigma
    # the flat b adds nothing; PyMC's HalfStudentT adds its log 2 in single precision (pt.log of the integer 2), 1.9e-9
    # per half-t prior, far inside the oracle's 1e-8 relative tolerance
    assert abs(float(prior(point)) - expected) < 1e-8
    flat = pmm.compile_logp(vars=[pmm["x"], pmm["z"]], jacobian=False)
    assert float(flat(point)) == 0.0


def test_inverse_gamma_is_stan_inv_gamma():
    model = ab.negbinomial("cnt ~ x", FRAME, ab.default_priors("cnt ~ x", FRAME, family="negbinomial"))
    pmm = model.pymc_model
    shape = pmm.compile_logp(vars=[pmm["alpha"]], jacobian=False)
    for value in (0.05, 0.7, 3.0, 40.0):
        point = model.value_point(intercept=0.1, b=[0.2], shape=value)
        assert abs(float(shape(point)) - stats.invgamma.logpdf(value, 0.4, scale=0.3)) < 1e-10
    assert ab.InverseGamma(2, 1.5) == ab.Prior("InverseGamma", {"alpha": 2.0, "beta": 1.5})
    with pytest.raises(ab.PriorError, match="beta must be positive"):
        ab.InverseGamma(2, 0)


def test_refusals():
    with pytest.raises(ValueError, match=r"family must be one of \[.*\], got 'sratio'"):    # brms's; cumulative is B05's
        ab.default_priors("y ~ x", FRAME, family="sratio")
    with pytest.raises(ValueError, match=r"link must be one of \['cloglog', 'logit', 'probit'\] for the bernoulli family, got 'cauchit'"):
        ab.default_priors("yb ~ x", FRAME, family="bernoulli", link="cauchit")    # the cloglog link is offered since P6 (G6 checks it)
    with pytest.raises(ValueError, match="nu applies only to the student family"):
        ab.default_priors("y ~ x", FRAME, nu=4)
    with pytest.raises(ValueError, match="student requires a fixed finite scalar nu > 1"):
        ab.default_priors("y ~ x", FRAME, family="student", nu=0.5)
    with pytest.raises(ab.FormulaError):
        ab.default_priors("y ~ s(x)", FRAME)
    with pytest.raises(ab.FormulaError, match="columns not in data"):
        ab.default_priors("y ~ nope", FRAME)
    with pytest.raises(ab.PriorError, match="must be a positive-support prior"):        # flat is for location classes only
        ab.gaussian("y ~ x", FRAME, ab.Priors(intercept=ab.Flat(), b=ab.Flat(), sigma=ab.Flat()))
    with pytest.raises(ab.PriorError, match="cannot be InverseGamma: its truncation to nu > 1 is not one this library renormalises"):
        ab.student("y ~ x", FRAME, ab.Priors(intercept=ab.Flat(), b=ab.Flat(), sigma=ab.HalfNormal(1), nu=ab.InverseGamma(2, 1)))
    with pytest.raises(ab.PriorError, match="prior for 'nu' must be a positive-support prior"):
        ab.student("y ~ x", FRAME, ab.Priors(intercept=ab.Flat(), b=ab.Flat(), sigma=ab.HalfNormal(1), nu=ab.Flat()))
    flat = ab.gaussian("y ~ x", FRAME, ab.default_priors("y ~ x", FRAME))
    with pytest.raises(ValueError, match=r"the prior of \['b:x'\] is flat .* no prior predictive distribution"):
        flat.prior_predictive(draws=5, seed=1)


def test_a_family_without_a_recorded_default_is_refused(monkeypatch):
    import autobayes.families as families

    bare = replace(FAMILIES["gaussian"], default_scale_prior=None)
    monkeypatch.setattr(families, "FAMILIES", MappingProxyType({**FAMILIES, "gaussian": bare}))
    with pytest.raises(ab.PriorError, match="no brms default prior is recorded for the gaussian family's sigma"):
        ab.default_priors("y ~ x", FRAME)
    unlearned = replace(FAMILIES["student"], default_learned_priors=MappingProxyType({}))
    monkeypatch.setattr(families, "FAMILIES", MappingProxyType({**FAMILIES, "student": unlearned}))
    with pytest.raises(ab.PriorError, match="no brms default prior is recorded for the student family's learned nu"):
        ab.default_priors("y ~ x", FRAME, family="student")


def test_describe_marks_the_defaults_and_nothing_else():
    formula = "y ~ x + (1 + x | g)"
    pri = ab.default_priors(formula, FRAME)
    text = ab.gaussian(formula, FRAME, pri).describe()
    for line in ("a_c ~ StudentT(nu=3, mu=10, sigma=35.3)   (brms default student_t(3, 10, 35.3))",
                 "b_x ~ Flat()   (brms default: flat, improper)",
                 "sd_g ~ HalfStudentT(nu=3, sigma=35.3)   (brms default student_t(3, 0, 35.3), lb 0)",
                 "sd_x|g ~ HalfStudentT(nu=3, sigma=35.3)   (brms default student_t(3, 0, 35.3), lb 0)",
                 "sigma ~ HalfStudentT(nu=3, sigma=35.3)   (brms default student_t(3, 0, 35.3), lb 0)",
                 "on L_g as lkj_corr_cholesky)   (brms default)",
                 "Default priors (ab.default_priors(); docs/default-priors.md)",
                 f"intercept, b:x, sd:g, sd:x|g, cor:g, sigma: brms {BRMS_VERSION}'s default_prior()"):
        assert line in text, line
    # the same priors written by the user: no marks, and the same target (who chose a prior is not part of the model)
    hst = ab.HalfStudentT(3, 35.3)
    mine = ab.Priors(intercept=ab.StudentT(3, 10, 35.3), b=ab.Flat(), sd={"g": {"Intercept": hst, "x": hst}},
                     cor={"g": ab.LKJ(1)}, sigma=hst)
    assert mine == pri
    user = ab.gaussian(formula, FRAME, mine)
    assert "brms default" not in user.describe() and "Default priors" not in user.describe()
    assert user.target_identity == ab.gaussian(formula, FRAME, pri).target_identity
    learned = ab.student("y ~ x", FRAME, ab.default_priors("y ~ x", FRAME, family="student")).describe()
    assert "nu ~ Gamma(alpha=2, beta=0.1)   (brms default gamma(2, 0.1), lb 1), truncated to nu > 1" in learned
    nb = ab.negbinomial("cnt ~ x", FRAME, ab.default_priors("cnt ~ x", FRAME, family="negbinomial")).describe()
    assert "shape ~ InverseGamma(alpha=0.4, beta=0.3)   (brms default inv_gamma(0.4, 0.3), lb 0)" in nb


def test_fit_records_say_which_priors_were_defaults(monkeypatch):
    import autobayes.session as session

    monkeypatch.setattr(session, "environment", lambda: {})
    monkeypatch.setattr(session, "_summary_rows", lambda *args: {})
    panel = {"values": {}, "status": {}, "overall": "not evaluated"}

    def record(model):
        fit = SimpleNamespace(target_identity=model.target_identity, idata=None, backend="synthetic", device_requested="cpu",
                              device_actual="cpu", platform="cpu", dtype="float64", chain_method="none", chains=0, draws=0,
                              tune=0, target_accept=0.9, seed=1, wall_seconds=0, notes=["synthetic record; not a fit"])
        return ab.fit_record(model, fit, panel=panel)["model"]

    pri = ab.default_priors("y ~ x + (1 | g)", FRAME)
    rec = record(ab.gaussian("y ~ x + (1 | g)", FRAME, pri))
    assert rec["default_priors"] == {"brms": BRMS_VERSION, "parameters": ["intercept", "b:x", "sd:g", "sigma"]}
    assert rec["priors"] == {"intercept": "StudentT(nu=3, mu=10, sigma=35.3)", "b:x": "Flat()", "sd": "HalfStudentT(nu=3, sigma=35.3)",
                             "sigma": "HalfStudentT(nu=3, sigma=35.3)"}
    mixed = record(ab.gaussian("y ~ x + (1 | g)", FRAME, replace(pri, b=ab.Normal(0, 5))))   # one class replaced by the user
    assert mixed["default_priors"]["parameters"] == ["intercept", "sd:g", "sigma"]
    user = record(ab.gaussian("y ~ x + (1 | g)", FRAME, ab.Priors(ab.Normal(0, 5), ab.Normal(0, 2), ab.HalfNormal(1), ab.HalfNormal(1))))
    assert "default_priors" not in user


def test_an_intercept_only_model_needs_no_b_prior():
    """D2: brms has no coefficient prior without a coefficient; neither does AutoBayes now. A formula with a
    coefficient still needs one."""
    pri = ab.Priors(intercept=ab.Normal(0, 5), sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1))
    without = ab.gaussian("y ~ 1 + (1 | g)", FRAME, pri)
    with_b = ab.gaussian("y ~ 1 + (1 | g)", FRAME, replace(pri, b=ab.Normal(0, 2)))
    assert without.target_identity == with_b.target_identity and without.describe() == with_b.describe()
    with pytest.raises(ab.PriorError, match="b must be a Prior or a mapping"):
        ab.gaussian("y ~ x + (1 | g)", FRAME, pri)


def test_the_table_of_user_priors_puts_each_prior_at_its_most_general_level():
    pri = ab.Priors(intercept=ab.Normal(0, 5), b={"x": ab.Normal(0, 1), "z": ab.Normal(0, 2)},
                    sd={"g": ab.HalfNormal(1), "h": {"Intercept": ab.HalfNormal(2), "x": ab.Exponential(0.5)}},
                    sigma=ab.HalfCauchy(2))
    rows = _table("y ~ x + z + (1 | g) + (1 + x || h)", "gaussian", {"categorical": Z}, priors=pri)
    assert [(r["class"], r["coef"], r["group"], r["prior"], r["lb"], r["source"]) for r in rows] == [
        ("Intercept", "", "", "normal(0, 5)", "", "user"),
        ("b", "", "", "", "", "default"), ("b", "x", "", "normal(0, 1)", "", "user"),
        ("b", "zq", "", "normal(0, 2)", "", "user"), ("b", "zr", "", "normal(0, 2)", "", "user"),
        ("sd", "", "", "", "0", "default"), ("sd", "", "g", "normal(0, 1)", "", "user"),
        ("sd", "Intercept", "g", "", "", "user"), ("sd", "", "h", "", "", "default"),
        ("sd", "Intercept", "h", "normal(0, 2)", "", "user"), ("sd", "x", "h", "exponential(0.5)", "", "user"),
        ("sigma", "", "", "cauchy(0, 2)", "0", "user")]
    spelled = [(ab.Normal(0, 5), "normal(0, 5)"), (ab.StudentT(4, 0.2, 1.2), "student_t(4, 0.2, 1.2)"),
               (ab.HalfNormal(1), "normal(0, 1)"), (ab.HalfStudentT(3, 2.5), "student_t(3, 0, 2.5)"), (ab.HalfCauchy(2), "cauchy(0, 2)"),
               (ab.Exponential(0.5), "exponential(0.5)"), (ab.Gamma(2, 0.1), "gamma(2, 0.1)"),
               (ab.InverseGamma(0.4, 0.3), "inv_gamma(0.4, 0.3)"), (ab.LKJ(2), "lkj(2)"), (ab.Flat(), "")]
    from autobayes.priors import brms_prior

    assert [brms_prior(p) for p, _ in spelled] == [text for _, text in spelled]


def test_who_chose_a_prior_is_not_part_of_it():
    mine, default = ab.HalfStudentT(3, 2.5), replace(ab.HalfStudentT(3, 2.5), source="default")
    assert mine == default and str(mine) == str(default) and repr(mine) == repr(default)
    assert mine.source == "user" and default.source == "default"
