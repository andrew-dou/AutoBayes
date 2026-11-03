"""Group-level terms, offsets and more parameters in distributional formulas (brms-parity G06; docs/distributional-groups.md).

The term model and every refusal, the priors and brms's defaults, the design blocks and PyMC names, ``value_point``, the
graph's per-row parameters and log density against hand computations (scipy), predictions by hand with and without group
effects, ``log_lik``, ``describe()``, the target identity and the summaries of sigma's own varying intercept. Build-only;
nothing samples. The numbers against brms are the oracle's (tests/oracle: gauss_sigma_varint, gauss_sigma_shared_id(_slopes),
gauss_sigma_offset, student_nu_formula, zip_zi_formula, cumulative_disc, gauss_sigma_varint_post)."""
import re
import warnings

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy import stats
from scipy.special import expit

import autobayes as ab
from autobayes.families import get_family
from autobayes.families.spec import DISTRIBUTIONAL_INVERSE_LINKS
from autobayes.formula import FormulaError, GroupTerm, ModelSpec, check_dpar_terms, parse_formula
from autobayes.likelihoods import centred_log_link, discriminated_logistic_random, logm1_link
from autobayes.model import _has_dpar_formula
from autobayes.priors import PriorError

LEVELS = ("a", "b", "c", "d")


def frame(seed=5, n=24) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"g": np.repeat(LEVELS, n // 4), "h": np.tile(["u", "v", "w"], n // 3), "x": rng.normal(size=n),
                       "w": rng.normal(size=n), "z": np.tile(["p", "q", "r"], n // 3), "y": rng.normal(size=n),
                       "soff": rng.uniform(-0.3, 0.3, n), "cnt": rng.poisson(1.5, n), "g2": np.roll(np.repeat(LEVELS, n // 4), 3)})
    df.loc[[0, 5, 9, 14], "cnt"] = 0                              # zeros for the zero-inflated Poisson
    df["ord"] = np.tile([1, 2, 3, 4], n // 4)
    df["gby"] = np.where(np.isin(df.g, ["a", "b"]), "s", "t")
    return df


DF = frame()
Z = {"z": "p"}
J = np.array([LEVELS.index(v) for v in DF.g])                     # each row's level
HALF = np.log(2.0)


def half_normal(x, s):
    return float(np.sum(stats.norm.logpdf(x, 0, s)) + np.size(x) * HALF)


def graph_fn(m, names):
    pmm = m.pymc_model
    return pmm.compile_fn(pmm.replace_rvs_by_values([pmm[n] for n in names]), inputs=pmm.value_vars, point_fn=True,
                          on_unused_input="ignore")


def logp_nojac(m, vp):
    return float(m.pymc_model.compile_logp(jacobian=False)(vp))


def zdesign(df=DF):
    return np.column_stack([(df.z == "q").to_numpy(float), (df.z == "r").to_numpy(float)])


# -- the term model and what stays refused ---------------------------------------------------------------------------
def test_formulas_parse_with_their_own_groups_slopes_and_offsets():
    s = parse_formula(["y ~ x + (1 | g)", "sigma ~ z + (1 | g)"], DF, "gaussian", Z)
    assert s.dpar("sigma").group_terms == (GroupTerm("g"),) and s.shared_ids == ()
    s = parse_formula(["y ~ x + (1 + x |p| g)", "sigma ~ x + (1 + x |p| g)"], DF, "gaussian")
    assert s.shared_ids[0].labels == ("Intercept", "x", "sigma_Intercept", "sigma_x")
    s = parse_formula(["y ~ x", "sigma ~ x + offset(soff)"], DF, "gaussian")
    assert s.dpar("sigma").offset.term == "offset(soff)"
    assert parse_formula(["cnt ~ x", "zi ~ z + (1 | g)"], DF, "zero_inflated_poisson", Z).dpar("zi").group_terms == (GroupTerm("g"),)
    assert parse_formula(["y ~ x", "nu ~ z"], DF, "student", Z).dpar_names == ("nu",)
    assert parse_formula(["ord ~ x", "disc ~ z"], DF, "cumulative", Z).dpar_names == ("disc",)


REFUSED = [
    (["y ~ x", "sigma ~ x + (1 | g)"], "student",
     "group-level terms in a distributional formula ((1 | g)) are not supported for sigma of the student family"),
    (["y ~ x", "sigma ~ x + offset(soff)"], "student",
     "'offset(soff)' in a distributional formula is not supported for sigma of the student family"),
    (["cnt ~ x", "zi ~ x + offset(soff)"], "zero_inflated_poisson",
     "'offset(soff)' in a distributional formula is not supported for zi of the zero_inflated_poisson family"),
    (["y ~ x", "sigma ~ x + offset(soff) + offset(x)"], "gaussian", "several offset() terms or an expression inside one are summed in "
                                                                     "the response formula only"),
    (["y ~ x", "sigma ~ x + offset(2 * soff)"], "gaussian", "'offset(2 * soff)' in a distributional formula is not supported"),
    (["y ~ x", "sigma ~ x + (1 + x | g)"], "gaussian", "varying slopes in a group-level term of a distributional formula's own"),
    (["y ~ x", "sigma ~ x + (1 | g:h)"], "gaussian", "'(1 | g:h)' (a grouping factor of two columns) in a distributional formula"),
    (["y ~ x", "sigma ~ x + (1 | mm(g, g2))"], "gaussian", "(mm()) in a distributional formula is not supported"),
    (["y ~ x", "sigma ~ x + (1 | gr(g, by = gby))"], "gaussian", "(gr(by = )) in a distributional formula is not supported"),
    # and G05b's gr(cov = ), gr(dist = ) and mm(scale = FALSE)
    (["y ~ x", "sigma ~ x + (1 | gr(g, cov = A))"], "gaussian", "(gr(cov = )) in a distributional formula is not supported"),
    (["y ~ x", "sigma ~ x + (1 | gr(g, dist = 'student'))"], "gaussian", "(gr(dist = )) in a distributional formula is not supported"),
    (["y ~ x", "sigma ~ x + (1 | mm(g, g2, scale = FALSE))"], "gaussian", "(mm(), mm()'s scale = FALSE) in a distributional formula"),
    # the group-term forms the response formula takes since G05a stay refused in a distributional formula
    (["y ~ x + (1 + x |p| g)", "sigma ~ z + (1 + x |p| g)"], "gaussian",
     "'(1 + x |p| g)' (a varying slope without its population-level term) in a distributional formula is not supported"),
    (["y ~ x", "sigma ~ x + (1 | g) + (0 + x | g)"], "gaussian",
     "'(1 | g) + (0 + x | g)' (its factor in several terms) in a distributional formula is not supported"),
    (["y ~ x", "sigma ~ x + (0 + x | g)"], "gaussian", "'(0 + x | g)' has no varying intercept"),
    (["y ~ x * w + (1 + x:w |p| g)", "sigma ~ x * w + (1 + x:w |p| g)"], "gaussian",
     "'(1 + x:w |p| g)': the varying slope 'x:w' in a distributional formula is not supported"),
    (["cnt ~ x + (1 |p| g)", "zi ~ x + (1 |p| g)"], "zero_inflated_poisson",
     "group-level terms in the formula of zi of the zero_inflated_poisson family ((1 |p| g)) are not supported"),
    (["cnt ~ x + (1 |p| g)", "zi ~ x + (1 |p| g)"], "zero_inflated_poisson", "and a group of its own, '(1 | g)'"),
    (["cnt ~ x", "zi ~ x + (1 | g)"], "zero_inflated_negbinomial",
     "group-level terms in a distributional formula ((1 | g)) are not supported for zi of the zero_inflated_negbinomial family"),
    (["y ~ x", "nu ~ x + (1 | g)"], "student",
     "group-level terms in a distributional formula ((1 | g)) are not supported for nu of the student family"),
    (["ord ~ x", "disc ~ x + (1 | g)"], "cumulative",
     "group-level terms in a distributional formula ((1 | g)) are not supported for disc of the cumulative family"),
    (["y ~ x + (1 + x |p| g)", "sigma ~ x + (1 + x |p| g)"], "student",
     "varying slopes in a distributional formula are not supported for sigma of the student family"),
    (["y ~ x + (1 | g)", "sigma ~ 1 + (1 |p| g)"], "gaussian", "names no group-level term of the response formula"),
    (["y ~ x + (1 + x |p| g)", "sigma ~ 1 + (1 |p| h)"], "gaussian", "two grouping factors; brms refuses it too"),
    (["ord | thres(gr = h) ~ x", "disc ~ z"], "cumulative", "'thres(gr = h)' next to the distributional formula 'disc ~ z'"),
    (["ord ~ x", "disc ~ z"], "sratio", "brms's disc ~ ... (the discrimination of an ordinal family"),
    (["ypos ~ x", "nu ~ x"], "frechet", "brms's nu ~ ... is not offered here"),
]


@pytest.mark.parametrize("formula,family,message", REFUSED, ids=[f"{r[1]}:{r[0][1]}" for r in REFUSED])
def test_what_is_not_verified_is_refused_naming_it(formula, family, message):
    with pytest.raises(FormulaError, match=re.escape(message)):
        parse_formula(formula, None, family, Z if any(re.search(r"\bz\b", f) for f in formula) else None)


def test_a_term_without_its_varying_intercept_is_refused():
    # the helper refuses a slope-only term (offered in the response formula since G05a) handed to it in a distributional formula
    d = ModelSpec(formula="sigma ~ x", response="sigma", predictors=("x",), group=None, group_terms=(GroupTerm("g", ("x",)),))
    with pytest.raises(FormulaError, match=re.escape("'(0 + x | g)' has no varying intercept; a distributional formula's group-level term")):
        check_dpar_terms(d, "sigma", "sigma ~ x + (0 + x | g)", get_family("gaussian"), parse_formula("y ~ x"), "y ~ x")


def test_the_cumulative_disc_is_the_logit_link_s_only():
    pr = ab.Priors(intercept=ab.Normal(0, 3), b=ab.Normal(0, 1), disc={"intercept": ab.Normal(0, 0.5), "b": ab.Normal(0, 0.5)})
    with pytest.raises(FormulaError, match=re.escape("brms's disc ~ ... is offered for the cumulative family's logit link only; no frozen "
                                                     "brms case verifies it under the probit link")):
        ab.cumulative(["ord ~ x", "disc ~ z"], DF, pr, link="probit", categorical=Z)
    with pytest.raises(FormulaError, match=re.escape("the distributional formula 'disc ~ z' is not available under the cloglog link")):
        ab.cumulative(["ord ~ x", "disc ~ z"], DF, pr, link="cloglog", categorical=Z)


def test_has_dpar_formula_reads_the_list():
    assert _has_dpar_formula(["y ~ x", "nu ~ z"], "nu") and not _has_dpar_formula(["y ~ x", "sigma ~ z"], "nu")
    assert not _has_dpar_formula("y ~ x", "nu") and not _has_dpar_formula(None, "nu")


# -- sigma ~ z + (1 | g): a group of its own ---------------------------------------------------------------------------
OWN = ["y ~ x + (1 | g)", "sigma ~ z + (1 | g)"]
OWN_PRIORS = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=ab.HalfNormal(1),
                       sigma={"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 0.5), "sd": ab.HalfNormal(0.5)})


def own_model():
    return ab.gaussian(OWN, DF, OWN_PRIORS, categorical=Z)


def test_own_group_blocks_names_and_density():
    m = own_model()
    roles = m.dpar_group_names[("sigma", "g", "Intercept")]
    assert roles == {"sd": "sigma_1|g_sigma", "offset": "sigma_1|g_offset", "group_effect": "sigma_1|g", "coordinate": "g__factor_dim"}
    block = m.design.dpars["sigma"].group_blocks[("g", "Intercept")]
    assert block.levels == LEVELS and np.array_equal(block.Z, np.eye(4)[J]) and m.correlation_names == {}
    zu, zs = np.linspace(-1, 1, 4), np.array([0.3, -0.8, 0.5, 0.1])
    vp = m.value_point(intercept=0.2, b=[0.4], sd=0.7, z=zu, sigma={"intercept": -0.2, "b": [0.3, -0.1], "sd": 0.6, "z": zs})
    X, Xs = DF.x.to_numpy() - DF.x.mean(), zdesign() - zdesign().mean(0)
    mu = 0.2 + 0.4 * X + 0.7 * zu[J]
    sigma = np.exp(-0.2 + Xs @ [0.3, -0.1] + 0.6 * zs[J])
    got_mu, got_sigma, got_u = graph_fn(m, ["mu", "sigma", "sigma_1|g"])(vp)
    assert np.allclose(got_mu, mu, rtol=1e-12) and np.allclose(got_sigma, sigma, rtol=1e-12) and np.allclose(got_u, 0.6 * zs)
    expected = (stats.norm.logpdf(0.2, 0, 5) + stats.norm.logpdf(0.4, 0, 2) + half_normal(0.7, 1) + stats.norm.logpdf(zu).sum()
                + stats.norm.logpdf(-0.2, 0, 1) + stats.norm.logpdf([0.3, -0.1], 0, 0.5).sum() + half_normal(0.6, 0.5)
                + stats.norm.logpdf(zs).sum() + stats.norm.logpdf(DF.y, mu, sigma).sum())
    assert logp_nojac(m, vp) == pytest.approx(expected, rel=1e-12)


def test_own_group_value_point_refuses_malformed_entries():
    m = own_model()
    base = dict(intercept=0.2, b=[0.4], sd=0.7, z=np.zeros(4))
    with pytest.raises(ValueError, match=re.escape("sigma['sd'] for 'g' must be a finite positive scalar")):
        m.value_point(**base, sigma={"intercept": 0.0, "b": [0, 0], "sd": -1.0, "z": np.zeros(4)})
    with pytest.raises(ValueError, match=re.escape("sigma['z'] for 'g' must have 4 finite entries")):
        m.value_point(**base, sigma={"intercept": 0.0, "b": [0, 0], "sd": 0.5, "z": np.zeros(3)})
    with pytest.raises(ValueError, match=re.escape("sigma['sd'] for 'g' must cover exactly its coefficients ['Intercept']; got ['x']")):
        m.value_point(**base, sigma={"intercept": 0.0, "b": [0, 0], "sd": {"g": {"x": 0.5}}, "z": np.zeros(4)})


def own_draws(m, include_sigma_group=True):
    rng = np.random.default_rng(1)
    coords = {"chain": [0], "draw": [0, 1], "g__factor_dim": list(LEVELS), "sigma_z_dim": ["q", "r"]}
    data = {"Intercept": (("chain", "draw"), [[0.1, -0.2]]), "x": (("chain", "draw"), [[0.5, 0.3]]),
            "1|g": (("chain", "draw", "g__factor_dim"), rng.normal(size=(1, 2, 4))),
            "1|g_sigma": (("chain", "draw"), [[0.8, 0.6]]),
            "sigma_1|g": (("chain", "draw", "g__factor_dim"), rng.normal(0, 0.3, size=(1, 2, 4))),
            "sigma_1|g_sigma": (("chain", "draw"), [[0.4, 0.3]]),
            "sigma_Intercept": (("chain", "draw"), [[-0.2, 0.1]]), "sigma_z": (("chain", "draw", "sigma_z_dim"), [[[0.2, -0.1], [0.05, 0.3]]])}
    return az.InferenceData(posterior=xr.Dataset(data, coords=coords))


def test_own_group_predictions_and_log_lik_by_hand():
    m, idata = own_model(), own_draws(own_model())
    post = idata.posterior
    u = post["sigma_1|g"].to_numpy()[0]
    eta = (post["sigma_Intercept"].to_numpy()[0][:, None] + post["sigma_z"].to_numpy()[0] @ zdesign().T + u[:, J])
    got = m.predict(idata, kind="linear", dpar="sigma")["eta"].to_numpy()[0]
    assert np.allclose(got, eta, rtol=1e-12)
    no_groups = m.predict(idata, kind="mean", dpar="sigma", include_group_effects=False)["sigma"].to_numpy()[0]
    assert np.allclose(no_groups, np.exp(eta - u[:, J]), rtol=1e-12)
    mu = m.predict(idata, kind="linear")["eta"].to_numpy()[0]
    ll = ab.log_lik(m, idata).transpose("chain", "draw", "__obs__").to_numpy()[0]
    assert np.allclose(ll, stats.norm.logpdf(DF.y.to_numpy(), mu, np.exp(eta)), rtol=1e-12)
    new = DF.iloc[:2].assign(g=["a", "zz"])
    with pytest.raises(FormulaError, match=re.escape("levels ['zz'] of 'g' were not seen in training; sigma's formula has no effect for "
                                                     "them (new levels of a distributional formula's own group-level term are not offered")):
        m.predict(idata, newdata=new, kind="response", new_groups="population", seed=1)


def test_a_factor_only_sigma_s_formula_reads():
    m = ab.gaussian(["y ~ x", "sigma ~ x + (1 | g)"], DF, ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2),
                                                                    sigma={"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 0.5),
                                                                           "sd": {"g": ab.HalfNormal(0.5)}}))
    assert m.spec.group_terms == () and m.design.dpars["sigma"].group_blocks[("g", "Intercept")].levels == LEVELS
    zs = np.array([0.3, -0.8, 0.5, 0.1])
    vp = m.value_point(intercept=0.2, b=[0.4], sigma={"intercept": -0.2, "b": [0.3], "sd": {"g": 0.6}, "z": {"g": zs}})
    sigma = np.exp(-0.2 + 0.3 * (DF.x.to_numpy() - DF.x.mean()) + 0.6 * zs[J])
    assert np.allclose(graph_fn(m, ["sigma"])(vp)[0], sigma, rtol=1e-12)


def test_own_group_defaults_describe_and_identity():
    pr = ab.default_priors(OWN, DF, categorical=Z)
    log_scale = ab.default_priors(["y ~ x", "sigma ~ 1"], DF).sigma["intercept"].params["sigma"]
    assert pr.sigma["sd"] == {"g": ab.HalfStudentT(3, log_scale)}
    rows = {(r["class"], r["coef"], r["group"], r["dpar"]): r["prior"] for r in own_model().prior_table()}
    assert rows[("sd", "", "", "sigma")] == "normal(0, 0.5)" and ("sd", "Intercept", "g", "sigma") in rows
    text = own_model().describe()
    assert "log(sigma_i) = eta_sigma,i = a_sigma_c + " in text and "+ u_sigma,g[g_i]" in text
    assert "sigma's own group (brms's r_<id>_sigma_1), uncorrelated with every other" in text
    assert "sd_sigma,g ~ HalfNormal(sigma=0.5)" in text and "sigma_1|g_sigma_log__" in text
    identity = own_model().target_identity
    assert identity == own_model().target_identity
    other = ab.gaussian(OWN, DF, ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=ab.HalfNormal(1),
                                           sigma={"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 0.5), "sd": ab.HalfNormal(0.4)}),
                        categorical=Z).target_identity
    assert identity != other                                            # sigma's group SD prior is part of the target


def test_own_group_summaries_merge_into_the_factor():
    m = own_model()
    idata = own_draws(m)
    re_g = ab.ranef(m, idata)["g"]
    assert list(dict.fromkeys(c for c, _ in re_g.columns)) == ["Intercept", "sigma_Intercept"]
    assert list(ab.VarCorr(m, idata)["g"]["sd"].index) == ["Intercept", "sigma_Intercept"]
    names = list(ab.posterior_summary(m, idata).index)
    assert names[:7] == ["b_Intercept", "b_sigma_Intercept", "b_x", "b_sigma_zq", "b_sigma_zr", "sd_g__Intercept", "sd_g__sigma_Intercept"]
    assert "r_g__sigma[a,Intercept]" in names and "r_g[a,Intercept]" in names
    lprior = ab.posterior_summary(m, idata).loc["lprior", "Estimate"]
    assert np.isfinite(lprior)
    shared = ab.gaussian(["y ~ x + (1 |p| g)", "sigma ~ 1 + (1 |p| g)"], DF,
                         ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd={"g": ab.HalfNormal(1)}, cor={"g": ab.LKJ(2)},
                                   sigma={"intercept": ab.Normal(0, 1), "sd": ab.HalfNormal(0.5)}))
    with pytest.raises(ValueError, match=re.escape("and a varying intercept of its own, '(1 | g)', of the response formula's one")):
        ab.fixef(shared, idata)
    two = ab.gaussian(["y ~ x + (1 | g) + (1 | h)", "sigma ~ 1 + (1 | g)"], DF,
                      ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd={"g": ab.HalfNormal(1), "h": ab.HalfNormal(1)},
                                sigma={"intercept": ab.Normal(0, 1), "sd": ab.HalfNormal(0.5)}))
    with pytest.raises(ValueError, match="the distributional formula 'sigma ~ 1 \\+ \\(1 \\| g\\)'"):
        ab.fixef(two, idata)


# -- (1 + x |p| g) in both formulas: a shared ID with varying slopes ------------------------------------------------------
SLOPES = ["y ~ x + (1 + x |p| g)", "sigma ~ x + (1 + x |p| g)"]
SLOPE_PRIORS = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd={"g": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(1)}},
                         cor={"g": ab.LKJ(2)},
                         sigma={"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 0.5),
                                "sd": {"g": {"Intercept": ab.HalfNormal(0.5), "x": ab.HalfNormal(0.5)}}})


def test_shared_slopes_correlate_four_coefficients():
    m = ab.gaussian(SLOPES, DF, SLOPE_PRIORS)
    assert m.correlation_coefs("g") == ("Intercept", "x", "sigma_Intercept", "sigma_x")
    assert set(m.dpar_group_names) == {("sigma", "g", "Intercept"), ("sigma", "g", "x")}
    block = m.design.dpars["sigma"].group_blocks[("g", "x")]
    assert np.array_equal(block.Z, np.eye(4)[J] * DF.x.to_numpy()[:, None])
    C = np.array([[1.0, 0.3, -0.2, 0.1], [0.3, 1.0, 0.1, 0.0], [-0.2, 0.1, 1.0, 0.2], [0.1, 0.0, 0.2, 1.0]])
    L = np.linalg.cholesky(C)
    z = {k: np.linspace(-1, 1, 4) * (i + 1) / 4 for i, k in enumerate(("m0", "m1", "s0", "s1"))}
    sds = [0.8, 0.5, 0.6, 0.3]
    vp = m.value_point(intercept=0.2, b=[0.4], L={"g": L}, sd={"g": {"Intercept": 0.8, "x": 0.5}}, z={"g": {"Intercept": z["m0"], "x": z["m1"]}},
                       sigma={"intercept": -0.3, "b": [0.2], "sd": {"g": {"Intercept": 0.6, "x": 0.3}},
                              "z": {"g": {"Intercept": z["s0"], "x": z["s1"]}}})
    r = (np.diag(sds) @ L @ np.vstack(list(z.values()))).T           # brms's r = (diag(sd) L z)^T
    x = DF.x.to_numpy()
    mu = 0.2 + 0.4 * (x - x.mean()) + r[J, 0] + x * r[J, 1]
    sigma = np.exp(-0.3 + 0.2 * (x - x.mean()) + r[J, 2] + x * r[J, 3])
    got_mu, got_sigma = graph_fn(m, ["mu", "sigma"])(vp)
    assert np.allclose(got_mu, mu, rtol=1e-12) and np.allclose(got_sigma, sigma, rtol=1e-12)
    with pytest.raises(ValueError, match=re.escape("sigma['sd'] for 'g' must be a mapping over its coefficients ['Intercept', 'x']")):
        m.value_point(intercept=0.2, b=[0.4], L={"g": L}, sd={"g": {"Intercept": 0.8, "x": 0.5}}, z={"g": {"Intercept": z["m0"], "x": z["m1"]}},
                      sigma={"intercept": -0.3, "b": [0.2], "sd": {"g": 0.6}, "z": {"g": {"Intercept": z["s0"], "x": z["s1"]}}})
    with pytest.raises(ValueError, match=re.escape("sigma['z'] for the slope 'x' of 'g' must have 4 finite entries")):
        m.value_point(intercept=0.2, b=[0.4], L={"g": L}, sd={"g": {"Intercept": 0.8, "x": 0.5}}, z={"g": {"Intercept": z["m0"], "x": z["m1"]}},
                      sigma={"intercept": -0.3, "b": [0.2], "sd": {"g": {"Intercept": 0.6, "x": 0.3}},
                             "z": {"g": {"Intercept": z["s0"], "x": z["s1"][:3]}}})
    with pytest.raises(ValueError, match=re.escape("a term with varying slopes takes a mapping over its coefficients")):
        m.value_point(intercept=0.2, b=[0.4], L={"g": L}, sd={"g": {"Intercept": 0.8, "x": 0.5}}, z={"g": {"Intercept": z["m0"], "x": z["m1"]}},
                      sigma=[0.1])
    text = m.describe()
    assert "v_sigma,x,g[g_i] * x_i" in text and "is the varying slope of x of log sigma by g" in text
    assert "sd_sigma,x|g ~ HalfNormal(sigma=0.5)" in text
    pr = ab.default_priors(SLOPES, DF)
    log_scale = ab.default_priors(["y ~ x", "sigma ~ 1"], DF).sigma["intercept"].params["sigma"]
    assert pr.sigma["sd"] == {"g": {"Intercept": ab.HalfStudentT(3, log_scale), "x": ab.HalfStudentT(3, log_scale)}}


def test_shared_slopes_predict_by_hand():
    m = ab.gaussian(SLOPES, DF, SLOPE_PRIORS)
    rng = np.random.default_rng(2)
    coords = {"chain": [0], "draw": [0], "g__factor_dim": list(LEVELS)}
    effects = {name: rng.normal(0, 0.3, size=(1, 1, 4)) for name in ("1|g", "x|g", "sigma_1|g", "sigma_x|g")}
    data = {"Intercept": (("chain", "draw"), [[0.1]]), "x": (("chain", "draw"), [[0.5]]), "sigma_Intercept": (("chain", "draw"), [[-0.2]]),
            "sigma_x": (("chain", "draw"), [[0.3]]), **{k: (("chain", "draw", "g__factor_dim"), v) for k, v in effects.items()}}
    idata = az.InferenceData(posterior=xr.Dataset(data, coords=coords))
    x = DF.x.to_numpy()
    eta = -0.2 + 0.3 * x + effects["sigma_1|g"][0, 0][J] + x * effects["sigma_x|g"][0, 0][J]
    assert np.allclose(m.predict(idata, kind="linear", dpar="sigma")["eta"].to_numpy()[0, 0], eta, rtol=1e-12)


# -- sigma ~ x + offset(soff) -----------------------------------------------------------------------------------------
OFFSET = ["y ~ x", "sigma ~ x + offset(soff)"]
OFFSET_PRIORS = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma={"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 0.5)})


def test_an_offset_of_sigma_s_formula():
    m = ab.gaussian(OFFSET, DF, OFFSET_PRIORS)
    dd = m.design.dpars["sigma"]
    assert np.array_equal(dd.offset, DF.soff.to_numpy()) and dd.columns == ("x",) and dd.X_raw.shape == (24, 1)
    vp = m.value_point(intercept=0.2, b=[0.4], sigma={"intercept": -0.2, "b": [0.3]})
    x = DF.x.to_numpy()
    sigma = np.exp(-0.2 + 0.3 * (x - x.mean()) + DF.soff.to_numpy())
    assert np.allclose(graph_fn(m, ["sigma"])(vp)[0], sigma, rtol=1e-12)
    idata = az.InferenceData(posterior=xr.Dataset({"Intercept": (("chain", "draw"), [[0.1]]), "x": (("chain", "draw"), [[0.5]]),
                                                   "sigma_Intercept": (("chain", "draw"), [[-0.2]]), "sigma_x": (("chain", "draw"), [[0.3]])},
                                                  coords={"chain": [0], "draw": [0]}))
    new = DF.iloc[:3].assign(soff=[0.1, -0.2, 0.0])
    got = m.predict(idata, newdata=new, kind="mean", dpar="sigma")["sigma"].to_numpy()[0, 0]
    assert np.allclose(got, np.exp(-0.2 + 0.3 * new.x.to_numpy() + new.soff.to_numpy()), rtol=1e-12)
    with pytest.raises(FormulaError, match=re.escape("newdata lacks columns ['soff']")):
        m.predict(idata, newdata=DF.iloc[:3].drop(columns="soff"), kind="mean", dpar="sigma")
    assert "+ offset_i (offset(soff))" in m.describe() and "offset_i is soff at row i (brms's offsets_sigma)" in m.describe()
    assert m.target_identity != ab.gaussian(["y ~ x", "sigma ~ x"], DF, OFFSET_PRIORS).target_identity


def test_dpar_offset_checks_bambi_s_build():
    m = ab.gaussian(OFFSET, DF, OFFSET_PRIORS)
    comp, d = m.bambi_model.components["sigma"], m.spec.dpar("sigma")
    Xfull = np.asarray(comp.design.common.design_matrix, dtype=float)
    offset, X = m._dpar_offset(d, comp, Xfull)
    assert np.array_equal(offset, DF.soff.to_numpy()) and X.shape == (24, 2)
    with pytest.raises(RuntimeError, match=re.escape("does not equal soff computed from the data")):
        m._dpar_offset(d, comp, Xfull * 2.0)

    class Fake:
        offset_terms = {}
        design = comp.design
    with pytest.raises(RuntimeError, match=re.escape("Bambi did not build 'offset(soff)' as the single offset term")):
        m._dpar_offset(d, Fake, Xfull)


# -- nu ~ z (Student-t, brms's logm1) -----------------------------------------------------------------------------------
NU = ["y ~ x", "nu ~ z"]
NU_PRIORS = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1),
                      nu={"intercept": ab.Normal(2, 0.5), "b": ab.Normal(0, 0.5)})


def nu_model():
    return ab.student(NU, DF, NU_PRIORS, categorical=Z)


def test_nu_formula_density_names_and_predictions():
    m = nu_model()
    assert m.learned_parameters == {} and m.names["nu"] == "nu" and m.design.dpars["nu"].link == "logm1"
    assert m.bambi_model.family.link["nu"].name == "logm1"
    vp = m.value_point(intercept=0.2, b=[0.4], sigma=0.9, nu={"intercept": 1.5, "b": [0.3, -0.4]})
    x, Xs = DF.x.to_numpy(), zdesign() - zdesign().mean(0)
    nu = 1.0 + np.exp(1.5 + Xs @ [0.3, -0.4])
    mu = 0.2 + 0.4 * (x - x.mean())
    assert np.allclose(graph_fn(m, ["nu"])(vp)[0], nu, rtol=1e-12)
    expected = (stats.norm.logpdf(0.2, 0, 5) + stats.norm.logpdf(0.4, 0, 2) + half_normal(0.9, 1) + stats.norm.logpdf(1.5, 2, 0.5)
                + stats.norm.logpdf([0.3, -0.4], 0, 0.5).sum() + stats.t.logpdf(DF.y, nu, mu, 0.9).sum())
    assert logp_nojac(m, vp) == pytest.approx(expected, rel=1e-12)
    idata = az.InferenceData(posterior=xr.Dataset({"Intercept": (("chain", "draw"), [[0.1]]), "x": (("chain", "draw"), [[0.5]]),
                                                   "sigma": (("chain", "draw"), [[0.8]]), "nu_Intercept": (("chain", "draw"), [[1.2]]),
                                                   "nu_z": (("chain", "draw", "nu_z_dim"), [[[0.3, -0.2]]])},
                                                  coords={"chain": [0], "draw": [0], "nu_z_dim": ["q", "r"]}))
    nu_post = 1.0 + np.exp(1.2 + zdesign() @ [0.3, -0.2])
    assert np.allclose(m.predict(idata, kind="mean", dpar="nu")["nu"].to_numpy()[0, 0], nu_post, rtol=1e-12)
    ll = ab.log_lik(m, idata).transpose("chain", "draw", "__obs__").to_numpy()[0, 0]
    assert np.allclose(ll, stats.t.logpdf(DF.y.to_numpy(), nu_post, 0.1 + 0.5 * x, 0.8), rtol=1e-12)
    draws = m.predict(idata, kind="response", seed=3)["y"].to_numpy()
    assert draws.shape == (1, 1, 24) and np.isfinite(draws).all()
    text = m.describe()
    assert "nu_i = 1 + exp(eta_nu,i), one per row: the formula 'nu ~ z'" in text and "y_i ~ StudentT(nu_i, mu_i, sigma)" in text


def test_nu_formula_refuses_a_fixed_nu_and_needs_its_priors():
    with pytest.raises(ValueError, match=re.escape("nu=4 fixes the degrees of freedom and the formula 'nu ~ ...' gives them a linear predictor")):
        ab.student(NU, DF, NU_PRIORS, categorical=Z, nu=4)
    with pytest.raises(PriorError, match=re.escape("the formula 'nu ~ z' gives nu its own linear predictor, which needs priors")):
        ab.student(NU, DF, ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1)), categorical=Z)
    with pytest.raises(ValueError, match=re.escape("nu has its own formula 'nu ~ z': give nu={'intercept': float, 'b': array}")):
        nu_model().value_point(intercept=0.2, b=[0.4], sigma=0.9, nu=4.0)


def test_nu_formula_defaults_and_table():
    pr = ab.default_priors(NU, DF, family="student", categorical=Z)
    assert pr.nu == {"intercept": ab.Normal(2.7, 0.8), "b": ab.Flat()}
    rows = [(r["class"], r["dpar"]) for r in ab.student(NU, DF, pr, categorical=Z).prior_table()]
    assert ("nu", "") not in rows and ("Intercept", "nu") in rows and ("b", "nu") in rows


def test_logm1_inverse_and_bambi_link():
    assert np.allclose(DISTRIBUTIONAL_INVERSE_LINKS["logm1"](np.array([0.0, 1.0]), "nu"), [2.0, 1.0 + np.e])
    with pytest.raises(ValueError, match=re.escape("nu = 1 + exp(linear predictor) overflows")):
        DISTRIBUTIONAL_INVERSE_LINKS["logm1"](np.array([1e4]), "nu")
    link = logm1_link()
    assert link.name == "logm1" and np.allclose(link.linkinv(np.log(2.0)), 3.0) and np.allclose(link.link(3.0), np.log(2.0))


# -- zi ~ z + (1 | g) (zero-inflated Poisson) -------------------------------------------------------------------------
ZI = ["cnt ~ x", "zi ~ z + (1 | g)"]
ZI_PRIORS = ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1),
                      zi={"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 0.5), "sd": ab.HalfNormal(0.5)})


def zip_pmf(y, lam, zi):
    return np.where(y == 0, np.log(zi + (1 - zi) * np.exp(-lam)), np.log1p(-zi) + stats.poisson.logpmf(y, lam))


def test_zi_formula_with_its_own_group():
    m = ab.zero_inflated_poisson(ZI, DF, ZI_PRIORS, categorical=Z)
    assert m.design.dpars["zi"].link == "logit" and set(m.dpar_group_names) == {("zi", "g", "Intercept")}
    zs = np.array([0.3, -0.8, 0.5, 0.1])
    vp = m.value_point(intercept=0.2, b=[0.4], zi={"intercept": -0.5, "b": [0.3, -0.1], "sd": 0.6, "z": zs})
    x, Xs = DF.x.to_numpy(), zdesign() - zdesign().mean(0)
    lam = np.exp(0.2 + 0.4 * (x - x.mean()))
    zi = expit(-0.5 + Xs @ [0.3, -0.1] + 0.6 * zs[J])
    assert np.allclose(graph_fn(m, ["zi"])(vp)[0], zi, rtol=1e-12)
    expected = (stats.norm.logpdf(0.2, 0, 2) + stats.norm.logpdf(0.4, 0, 1) + stats.norm.logpdf(-0.5, 0, 1)
                + stats.norm.logpdf([0.3, -0.1], 0, 0.5).sum() + half_normal(0.6, 0.5) + stats.norm.logpdf(zs).sum()
                + zip_pmf(DF.cnt.to_numpy(), lam, zi).sum())
    assert logp_nojac(m, vp) == pytest.approx(expected, rel=1e-12)
    rng = np.random.default_rng(4)
    u = rng.normal(0, 0.4, size=(1, 1, 4))
    idata = az.InferenceData(posterior=xr.Dataset({"Intercept": (("chain", "draw"), [[0.1]]), "x": (("chain", "draw"), [[0.5]]),
                                                   "zi_Intercept": (("chain", "draw"), [[-0.4]]),
                                                   "zi_z": (("chain", "draw", "zi_z_dim"), [[[0.3, -0.2]]]),
                                                   "zi_1|g": (("chain", "draw", "g__factor_dim"), u)},
                                                  coords={"chain": [0], "draw": [0], "zi_z_dim": ["q", "r"], "g__factor_dim": list(LEVELS)}))
    lam_post = np.exp(0.1 + 0.5 * x)
    eta_zi = -0.4 + zdesign() @ [0.3, -0.2]
    mean = m.predict(idata, kind="mean")["mean"].to_numpy()[0, 0]
    assert np.allclose(mean, (1 - expit(eta_zi + u[0, 0][J])) * lam_post, rtol=1e-12)
    no_groups = m.predict(idata, kind="mean", include_group_effects=False)["mean"].to_numpy()[0, 0]
    assert np.allclose(no_groups, (1 - expit(eta_zi)) * lam_post, rtol=1e-12)    # re_formula = NA drops zi's effects too
    ll = ab.log_lik(m, idata).transpose("chain", "draw", "__obs__").to_numpy()[0, 0]
    assert np.allclose(ll, zip_pmf(DF.cnt.to_numpy(), lam_post, expit(eta_zi + u[0, 0][J])), rtol=1e-12)
    draws = m.predict(idata, kind="response", seed=2)["y"].to_numpy()
    assert draws.shape == (1, 1, 24) and (draws >= 0).all()
    text = m.describe()
    assert "y_i ~ ZeroInflatedPoisson(mu_i, zi_i)" in text and "zi_i = inv_logit(eta_zi,i), one per row" in text
    assert "E[y_i] = (1 - zi_i) mu_i" in text
    pr = ab.default_priors(ZI, DF, family="zero_inflated_poisson", categorical=Z)
    assert pr.zi == {"intercept": ab.Logistic(0, 1), "b": ab.Flat(), "sd": {"g": ab.HalfStudentT(3, 2.5)}}
    with pytest.raises(PriorError, match=re.escape("prior for 'zi:sd:g' must be a positive-support prior")):
        ab.zero_inflated_poisson(ZI, DF, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1),
                                                   zi={"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 0.5), "sd": ab.Normal(0, 1)}),
                                 categorical=Z)


# -- disc ~ z (cumulative logit) -------------------------------------------------------------------------------------
DISC = ["ord ~ x", "disc ~ z"]
DISC_PRIORS = ab.Priors(intercept=ab.Normal(0, 3), b=ab.Normal(0, 1.5), disc={"intercept": ab.Normal(0, 0.5), "b": ab.Normal(0, 0.5)})


def disc_model(formula=DISC, priors=DISC_PRIORS):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ab.cumulative(formula, DF, priors, categorical=Z if any(re.search(r"\bz\b", f) for f in formula) else None)


def cumulative_log_lik(y, mu, disc, tau):
    F = expit(disc[:, None] * (tau[None, :] - mu[:, None]))
    probs = np.diff(np.column_stack([np.zeros(len(y)), F, np.ones(len(y))]), axis=1)
    return np.log(probs[np.arange(len(y)), y - 1])


def test_disc_formula_density_centring_and_warning():
    m = disc_model()
    dd = m.design.dpars["disc"]
    assert dd.centered and dd.centred_draws and np.allclose(dd.x_means, zdesign().mean(0))
    assert m.bambi_model.family.link["disc"].name == "log_centred" and m.names["disc"] == "disc"
    assert any("identified by the priors alone" in w and "D188" in w for w in m.warnings)
    tau_c = np.array([-1.0, 0.2, 1.1])
    vp = m.value_point(intercept=tau_c, b=[0.4], disc={"intercept": 0.3, "b": [0.2, -0.5]})
    x, Xd = DF.x.to_numpy(), zdesign() - zdesign().mean(0)
    disc = np.exp(0.3 + Xd @ [0.2, -0.5])                       # brms: exp(Intercept_disc + Xc_disc b_disc)
    tau = tau_c + x.mean() * 0.4                                 # b_Intercept, the thresholds of the raw design
    got_disc, got_tau = graph_fn(m, ["disc", "b_Intercept"])(vp)
    assert np.allclose(got_disc, disc, rtol=1e-12) and np.allclose(got_tau, tau, rtol=1e-12)
    expected = (stats.norm.logpdf(tau_c, 0, 3).sum() + stats.norm.logpdf(0.4, 0, 1.5) + stats.norm.logpdf(0.3, 0, 0.5)
                + stats.norm.logpdf([0.2, -0.5], 0, 0.5).sum() + cumulative_log_lik(DF.ord.to_numpy(), 0.4 * x, disc, tau).sum())
    assert logp_nojac(m, vp) == pytest.approx(expected, rel=1e-12)
    # a fit's posterior holds the centred intercept (Bambi does not centre this model): predictions read it with the centred design
    idata = az.InferenceData(posterior=xr.Dataset({"Intercept": (("chain", "draw", "Intercept_dim"), [[tau_c]]),
                                                   "b_Intercept": (("chain", "draw", "Intercept_dim"), [[tau]]),
                                                   "x": (("chain", "draw"), [[0.4]]), "disc_Intercept": (("chain", "draw"), [[0.3]]),
                                                   "disc_z": (("chain", "draw", "disc_z_dim"), [[[0.2, -0.5]]])},
                                                  coords={"chain": [0], "draw": [0], "Intercept_dim": ["1", "2", "3"], "disc_z_dim": ["q", "r"]}))
    assert np.allclose(m.predict(idata, kind="mean", dpar="disc")["disc"].to_numpy()[0, 0], disc, rtol=1e-12)
    p = m.predict(idata, kind="mean")["p"].to_numpy()[0, 0]
    assert np.allclose(np.log(p[np.arange(24), DF.ord.to_numpy() - 1]), cumulative_log_lik(DF.ord.to_numpy(), 0.4 * x, disc, tau), rtol=1e-12)
    ll = ab.log_lik(m, idata).transpose("chain", "draw", "__obs__").to_numpy()[0, 0]
    assert np.allclose(ll, cumulative_log_lik(DF.ord.to_numpy(), 0.4 * x, disc, tau), rtol=1e-12)
    text = m.describe()
    assert "P(y_i <= k) = expit(disc_i (b_Intercept[k] - mu_i))" in text and "OrderedLogistic(disc_i mu_i, disc_i b_Intercept)" in text
    assert "disc_i = exp(eta_disc,i), one per row: the formula 'disc ~ z'" in text
    with pytest.raises(ValueError, match=re.escape("disc is the discrimination's formula ('disc ~ ...') of a cumulative model")):
        disc_model(["ord ~ x"], ab.Priors(intercept=ab.Normal(0, 3), b=ab.Normal(0, 1.5))).value_point(
            intercept=tau_c, b=[0.4], disc={"intercept": 0.3, "b": []})


def test_disc_formulas_that_fix_the_scale_do_not_warn():
    identified = disc_model(["ord ~ x", "disc ~ 0 + x"], ab.Priors(intercept=ab.Normal(0, 3), b=ab.Normal(0, 1.5),
                                                                   disc={"b": ab.Normal(0, 0.5)}))
    assert identified.warnings == [] and not identified.design.dpars["disc"].centered
    vp = identified.value_point(intercept=np.array([-1.0, 0.2, 1.1]), b=[0.4], disc={"b": [0.7]})
    assert np.allclose(graph_fn(identified, ["disc"])(vp)[0], np.exp(0.7 * DF.x.to_numpy()), rtol=1e-12)
    full = disc_model(["ord ~ x", "disc ~ 0 + z"], ab.Priors(intercept=ab.Normal(0, 3), b=ab.Normal(0, 1.5), disc={"b": ab.Normal(0, 0.5)}))
    assert any("identified by the priors alone" in w for w in full.warnings)   # a factor coded by all its levels
    alone = disc_model(["ord ~ x", "disc ~ 1"], ab.Priors(intercept=ab.Normal(0, 3), b=ab.Normal(0, 1.5), disc={"intercept": ab.Normal(0, 1)}))
    vp = alone.value_point(intercept=np.array([-1.0, 0.2, 1.1]), b=[0.4], disc={"intercept": 0.25})
    assert np.allclose(graph_fn(alone, ["disc"])(vp)[0], np.exp(0.25), rtol=1e-12)
    assert ab.cumulative("ord ~ x", DF, ab.Priors(intercept=ab.Normal(0, 3), b=ab.Normal(0, 1.5))).warnings == []


def test_disc_priors_defaults_and_draws():
    pr = ab.default_priors(DISC, DF, family="cumulative", categorical=Z)
    assert pr.disc == {"intercept": ab.Normal(0, 1), "b": ab.Flat()}
    with pytest.raises(PriorError, match=re.escape("disc gives the priors of a cumulative model's discrimination formula")):
        ab.cumulative("ord ~ x", DF, ab.Priors(intercept=ab.Normal(0, 3), b=ab.Normal(0, 1.5), disc={"intercept": ab.Normal(0, 1)}))
    rng = np.random.default_rng(6)
    draws = discriminated_logistic_random(np.zeros(5), np.array([-1.0, 0.0, 1.0]), np.full(5, 2.0), rng=rng)
    assert draws.shape == (5,) and set(draws.tolist()) <= {1, 2, 3, 4}
    link = centred_log_link([], np.zeros(0))
    assert link.name == "log_centred" and np.allclose(link.linkinv(0.0), 1.0) and np.allclose(link.link(1.0), 0.0)


# -- the checks of Bambi's build, and the paths every other model takes -------------------------------------------------
def test_dpar_design_checks_bambi_s_link_and_group_build(monkeypatch):
    from types import SimpleNamespace

    m = own_model()
    d, comp = m.spec.dpar("sigma"), m.bambi_model.components["sigma"]
    group = comp.design.group
    Zb = group.design_matrix.toarray() if hasattr(group.design_matrix, "toarray") else group.design_matrix

    def fake(terms=None, matrix=None):
        return SimpleNamespace(design=SimpleNamespace(group=SimpleNamespace(terms=terms or group.terms, slices=group.slices,
                                                                            design_matrix=Zb if matrix is None else matrix)))
    assert set(m._dpar_group_blocks(d, fake())) == {("g", "Intercept")}
    with pytest.raises(RuntimeError, match=re.escape("Bambi's level order of 'g' in the formula 'sigma ~ z + (1 | g)' differs")):
        m._dpar_group_blocks(d, fake(terms={"1|g": SimpleNamespace(groups=list(reversed(LEVELS)))}))
    with pytest.raises(RuntimeError, match=re.escape("Bambi's group design of '1|g' in the formula 'sigma ~ z + (1 | g)' is not the level "
                                                     "indicator of 'g'")):
        m._dpar_group_blocks(d, fake(matrix=np.zeros_like(np.asarray(Zb, dtype=float))))
    roles = m._shared_roles("g", "sigma", "Intercept")
    with monkeypatch.context() as mp:
        mp.setitem(m.pymc_model.named_vars_to_dims, roles["offset"], ("elsewhere",))
        with pytest.raises(RuntimeError, match=re.escape("PyMC dims of 'sigma_1|g_offset' are not ('g__factor_dim',)")):
            m._dpar_group_blocks(d, fake())
    with monkeypatch.context() as mp:
        mp.setattr(m, "_shared_roles", lambda factor, dpar, coef: {**roles, "sd": "sigma"})
        with pytest.raises(RuntimeError, match=re.escape("'sigma' is not a free variable of the graph")):
            m._dpar_group_blocks(d, fake())
    monkeypatch.setattr(m.bambi_model.family.link["sigma"], "name", "identity")
    with pytest.raises(RuntimeError, match=re.escape("Bambi's link of 'sigma' is identity, expected log")):
        m._dpar_design(d)


def test_the_models_without_these_formulas_keep_their_paths():
    base = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1))
    fixed = ab.student("y ~ x", DF, base, nu=4)                      # a fixed nu: Bambi's prior dict holds it
    assert fixed.nu == 4.0 and "nu" not in fixed.names and fixed.spec.dpar_names == ()
    learned = ab.student("y ~ x", DF, ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1),
                                                nu=ab.Gamma(2, 0.1)))
    assert learned.learned_parameters == {"nu": 1.0} and learned.names["nu"] == "nu"
    grouped = ab.cumulative("ord | thres(gr = h) ~ x", DF, ab.Priors(intercept=ab.Normal(0, 3), b=ab.Normal(0, 1.5)))
    assert grouped.family_spec.distributional_formulas == () and grouped.warnings == []
    m = ab.gaussian(["y ~ x + (1 | g)", "sigma ~ z"], DF, ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=ab.HalfNormal(1),
                                                                     sigma={"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 0.5)}),
                    categorical=Z)
    idata = own_draws(m)
    del idata.posterior["sigma_1|g"], idata.posterior["sigma_1|g_sigma"]
    assert list(ab.fixef(m, idata).index) == ["Intercept", "sigma_Intercept", "x", "sigma_zq", "sigma_zr"]   # no group terms in it


# -- with G02's class-wide priors and dropna -----------------------------------------------------------------------------
def test_class_wide_priors_inside_the_formulas_priors():
    m = ab.gaussian(["y ~ x + (1 | g)", "sigma ~ z + x + (1 | g)"], DF,
                    ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=ab.HalfNormal(1),
                              sigma={"intercept": ab.Normal(0, 1), "b": {"*": ab.Normal(0, 0.5), "x": ab.Normal(0, 1)},
                                     "sd": {"*": ab.HalfNormal(0.5)}}), categorical=Z)
    sub = m.priors.dpar_priors(m.spec, "sigma")
    assert sub.coefficient_priors(m.spec.dpar("sigma")) == {"z[q]": ab.Normal(0, 0.5), "z[r]": ab.Normal(0, 0.5), "x": ab.Normal(0, 1)}
    assert sub.sd_priors(m.spec.dpar("sigma")) == {("g", "Intercept"): ab.HalfNormal(0.5)}
    shared = ab.gaussian(SLOPES, DF, ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd={"g": {"Intercept": ab.HalfNormal(1),
                                                                                                   "x": ab.HalfNormal(1)}},
                                               cor={"g": ab.LKJ(2)}, sigma={"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 0.5),
                                                                            "sd": {"*": ab.HalfNormal(0.5), "g": {"x": ab.HalfNormal(0.2)}}}))
    assert shared.priors.dpar_priors(shared.spec, "sigma").sd_priors(shared.spec.dpar("sigma")) == {
        ("g", "Intercept"): ab.HalfNormal(0.5), ("g", "x"): ab.HalfNormal(0.2)}
    nu = ab.student(NU, DF, ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1),
                                      nu={"intercept": ab.Normal(2, 0.5), "b": {"*": ab.Normal(0, 0.5), "z": {"r": ab.Normal(0, 1)}}}),
                    categorical=Z)
    assert nu.priors.dpar_priors(nu.spec, "nu").coefficient_priors(nu.spec.dpar("nu")) == {"z[q]": ab.Normal(0, 0.5), "z[r]": ab.Normal(0, 1)}
    zi = ab.zero_inflated_poisson(ZI, DF, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1),
                                                    zi={"intercept": ab.Normal(0, 1), "b": {"*": ab.Normal(0, 0.5)}, "sd": {"*": ab.HalfNormal(0.5)}}),
                                  categorical=Z)
    assert zi.priors.dpar_priors(zi.spec, "zi").sd_priors(zi.spec.dpar("zi")) == {("g", "Intercept"): ab.HalfNormal(0.5)}
    disc = disc_model(DISC, ab.Priors(intercept=ab.Normal(0, 3), b=ab.Normal(0, 1.5),
                                      disc={"intercept": ab.Normal(0, 0.5), "b": {"*": ab.Normal(0, 0.5), "z": ab.Normal(0, 0.3)}}))
    rows = {(r["class"], r["coef"], r["dpar"]): r["prior"] for r in disc.prior_table()}
    assert rows[("b", "", "disc")] == "normal(0, 0.5)" and rows[("b", "zq", "disc")] == "normal(0, 0.3)"
    with pytest.raises(PriorError, match=re.escape("the override(s) ['h'] of sd={'*': ...} name no grouping factor of the formula")):
        ab.gaussian(OWN, DF, ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=ab.HalfNormal(1),
                                       sigma={"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 0.5),
                                              "sd": {"*": ab.HalfNormal(0.5), "h": ab.HalfNormal(1)}}), categorical=Z)


def test_dropna_counts_the_columns_only_sigma_s_formula_reads():
    df = DF.copy()
    df.loc[[3, 7], "w"] = np.nan                                       # read by sigma's formula only
    df.loc[[11], "soff"] = np.nan                                      # sigma's offset
    df["g"] = df["g"].astype(object)
    df.loc[[2], "g"] = None                                            # sigma's own group's factor, which the response formula lacks
    with pytest.warns(UserWarning, match="Rows containing NAs were excluded from the model"):
        m = ab.gaussian(["y ~ x", "sigma ~ w + offset(soff) + (1 | g)"], df,
                        ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2),
                                  sigma={"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 0.5), "sd": ab.HalfNormal(0.5)}), dropna=True)
    assert m.dropped_na_rows == (2, 3, 7, 11) and m.design.n == 20
    assert np.array_equal(m.design.dpars["sigma"].offset, df.soff.drop(index=[2, 3, 7, 11]).to_numpy())
