"""brms's special terms in interactions and with ids, ``me(x, sdx, gr = g)``, ``cs()`` on an expression and in a group-level term
(docs/special-term-ext.md, divergences D89, D197, D198): the term model and its parsing, every refusal, priors, names,
``value_point``, the densities against independent computations, predictions, ``describe()``, records and the oracle's
extension. The brms comparison is the oracle cases ``gauss_mo_interaction``, ``gauss_mo_mo_interaction``, ``gauss_mo_id``,
``gauss_mo_id_sigma``, ``gauss_mo_colon_factor``, ``gauss_me_gr``, ``gauss_me_interaction``, ``acat_cs_expression`` and
``acat_cs_group``. Nothing here samples."""
import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import autobayes as ab
from autobayes import special_terms
from autobayes.formula import (FormulaError, GroupTerm, PopTerm, category_specific_level, cs_label, cs_values, group_covariate,
                               measurement_error_columns, measurement_error_group, parse_formula)
from autobayes.monotonic import monotonic_values, stick_breaking_free
from autobayes.priors import PriorError
from tests.oracle import adapter, gates
from tests.oracle import fixture as fx


def _case(cid):
    case = fx.load_case(cid)
    return case, fx.read_frame(case, fx.fixture_dir(cid) / "data.csv")


@pytest.fixture(scope="module")
def frame():
    return _case("gauss_mo_interaction")[1]


@pytest.fixture(scope="module")
def colon():
    return gates.Bundle("gauss_mo_colon_factor")


@pytest.fixture(scope="module")
def interaction():
    return gates.Bundle("gauss_mo_interaction")


def _priors(**kw):
    base = dict(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1))
    return ab.Priors(**{**base, **kw})


# -- parsing ---------------------------------------------------------------------------------------------------------------
def test_the_term_model_of_special_products(frame):
    s = parse_formula("y ~ x + mo(ord) + mo(ord):x + (1 | g)", frame)
    assert [(t.name, t.kind) for t in s.pop_terms] == [("x", "numeric"), ("mo(ord)", "monotonic"), ("mo(ord):x", "special_product")]
    sp = s.sp_terms[0].special
    assert (sp.parts, sp.numeric, sp.operands, sp.has_covariate, sp.written) == (("mo(ord)",), ("x",), ("ord", "x"), True, "mo(ord):x")
    assert s.coefficient_labels == ("x", "mo(ord)", "mo(ord):x") and s.sp_terms[0].is_special and s.bambi_terms[-1].name == "x"
    assert sp.simplex_key(0) == "mo(ord):x" and sp.brms_simplex_coef(0) == "moord:x1" and sp.coding_of("ord").D == 3
    mm = parse_formula("y ~ mo(ord) * mo(o2)", frame).sp_terms[0].special
    assert [mm.simplex_key(j) for j in range(2)] == ["mo(ord):mo(o2)[ord]", "mo(ord):mo(o2)[o2]"]
    assert [mm.brms_simplex_coef(j) for j in range(2)] == ["moord:moo21", "moord:moo22"] and mm.covariate(frame) is None
    ids = parse_formula('y ~ mo(ord, id = "a") + mo(ord, id = \'a\'):x', frame)
    assert [t.name for t in ids.pop_terms] == ['mo(ord, id = "a")', 'mo(ord, id = "a"):x']
    assert {t.special.simplex_key(0) for t in ids.sp_terms} == {"a"} and ids.sp_terms[0].special.brms_simplex_coef(0) == "a"
    assert special_terms.simplexes(ids)["a"][2] == 'mo(ord, id = "a")'            # first use
    me = parse_formula("y ~ me(xobs, sdx) * w", frame)
    assert [t.name for t in me.pop_terms] == ["me(xobs, sdx)", "w", "me(xobs, sdx):w"]
    assert me.sp_terms[0].special.me_parts == ("me(xobs, sdx)",) and me.sp_terms[0].special.operands == ("xobs", "sdx", "w")
    gr = parse_formula("y ~ me(xg, sdxg, gr = g) + (1 | h)", frame)
    assert gr.pop_terms[0].operands == ("xg", "sdxg", "g") and measurement_error_group(gr.pop_terms[0].name) == "g"
    assert measurement_error_columns("me(x, s)") == ("x", "s") and measurement_error_group("me(x, s)") is None
    assert measurement_error_group("mo(x)") is None
    assert special_terms.me_name("x", "s", "g") == "me(x, s, gr = g)"
    assert special_terms.parse_product(["me(x, s, gr = g)", "w"], "p", "y", None).sides == ("me(x, s, gr = g)", "w")
    as_mapping = PopTerm("mo(ord):x", "special_product", (), special={"sides": ("mo(ord)", "x")})   # rebuilt from asdict()
    assert as_mapping.special == sp.__class__(("mo(ord)", "x"))


def test_a_factor_in_a_special_product_follows_r_s_coding(frame):
    """D197: an indicator per level without the term that drops the factor, the non-reference levels' with it."""
    alone = parse_formula("y ~ x + mo(ord):z", frame, categorical={"z": "p"})
    assert [t.name for t in alone.sp_terms] == ["mo(ord):zp", "mo(ord):zq", "mo(ord):zr"]
    assert all(t.special.coding == "indicator" and t.special.written == "mo(ord):z" for t in alone.sp_terms)
    np.testing.assert_array_equal(alone.sp_terms[1].special.covariate(frame), (frame.z == "q").to_numpy(dtype=float))
    unseen = parse_formula("y ~ x + mo(ord):z", None, categorical={"z": "p"})
    assert [t.name for t in unseen.sp_terms] == ["mo(ord):z"] and unseen.sp_terms[0].special.level is None
    main = parse_formula("y ~ mo(ord) * z", frame, categorical={"z": "p"})
    assert [t.name for t in main.sp_terms] == ["mo(ord):zq", "mo(ord):zr"] and main.sp_terms[0].special.coding == "contrast"
    with pytest.raises(FormulaError, match=r"has levels \['s'\] the model was not fitted with"):
        main.sp_terms[0].special.covariate(frame.assign(z="s"), ("p", "q", "r"), where="newdata column")


def test_category_specific_expressions_and_group_terms(frame):
    s = parse_formula("ord ~ cs(x + z) + (1 | g)", frame, "acat", categorical={"z": "p"})
    assert [t.name for t in s.cs_terms] == ["cs(x)", "cs(z)[q]", "cs(z)[r]"] and [cs_label(t) for t in s.cs_terms] == ["x", "zq", "zr"]
    assert category_specific_level("cs(z)[q]") == "q" and category_specific_level("cs(x)") is None
    np.testing.assert_array_equal(cs_values(s.cs_terms[1], frame), (frame.z == "q").to_numpy(dtype=float))
    with pytest.raises(FormulaError, match=r"has levels \['s'\] the model was not fitted with"):
        cs_values(s.cs_terms[1], frame.assign(z="s"), ("p", "q", "r"), where="newdata column")
    t = parse_formula("ord ~ x + (cs(1) | g)", frame, "acat").group_terms[0]
    assert t.thresholds and t.coefs == ("Intercept[1]", "Intercept[2]", "Intercept[3]") and t.custom and t.has_correlation
    assert t.canonical == "(cs(1) | g)" and t.covariate_columns("Intercept[2]") == (None,)
    unseen = parse_formula("ord ~ x + (cs( 1 ) | g)", None, "acat").group_terms[0]
    assert unseen.coefs == ("Intercept",) and unseen.has_correlation
    np.testing.assert_array_equal(group_covariate(s, "Intercept[1]", frame), np.ones(len(frame)))


@pytest.mark.parametrize("formula, kwargs, message", [
    ("y ~ mo(ord, 2)", {}, r"mo\(\) takes a column and optionally brms's id.*D198"),
    ("y ~ mo(ord, id = a)", {}, r"mo\(\) takes a column and optionally brms's id"),
    ("y ~ mo(log(ord)):x", {}, r"mo\(\) takes a column name here, not an expression"),
    ("y ~ mo(y, id = 'a')", {}, r"the response 'y' cannot be a monotonic predictor"),
    ("y ~ mo(ord):log(x)", {}, r"'log\(x\)' in an interaction with a special term .* is not supported.*D198"),
    ("y ~ mo(ord):y", {}, r"the response or trials column cannot enter an interaction"),
    ("y ~ mo(ord):mo(ord, id = 'a')", {}, r"a variable written twice in one interaction"),
    ("y ~ me(xobs, sdx) + mo(ord):me(xobs, sdx)", {}, r"mo\(\) and me\(\) in one term are not supported"),
    ("y ~ mo(ord):x:z:h", {"categorical": {"z": "p", "h": "h1"}}, r"multiplies the factors \['z', 'h'\].*D198"),
    ("y ~ me(xobs, sdx, 1)", {}, r"me\(\) takes the noisy column, the column of its known standard deviations and optionally"),
    ("y ~ me(log(xobs), sdx):w", {}, r"me\(\) takes column names here, not expressions"),
    ("y ~ me(xobs, sdx, gr = xobs)", {}, r"gr's column 'xobs' is the noisy column or its noise column"),
    ("y ~ mo(ord, id = 'a') + mo(o2, id = 'a')", {}, r"the id 'a' is given to mo\(ord\) and mo\(o2\).*D198"),
    ("y ~ mo(ord) + mo(ord):x + me(xobs, sdx)", {}, r"mo\(\) and me\(\) terms in one formula are not supported"),
    ("y ~ mo(ord):x + gp(w)", {}, r"the special term 'mo\(ord\):x' next to the Gaussian process 'gp\(w\)'.*D198"),
    ("y | weights(w) ~ mo(ord):x", {}, r"next to the addition term 'weights\(w\)'"),
    ("y ~ mo(ord):x + (1 | gr(g, by = h))", {}, r"next to the group term '\(1 \| gr\(g, by = h\)\)'"),
])
def test_refusals_name_the_construct(frame, formula, kwargs, message):
    with pytest.raises(FormulaError, match=message):
        parse_formula(formula, frame, **kwargs)


@pytest.mark.parametrize("family, formula, kwargs, message", [
    ("categorical", "cat3 ~ mo(ord):x", {}, r"the monotonic term 'mo\(ord\):x' is not available for the categorical family"),
    ("poisson", "cnt ~ me(xobs, sdx) + me(xobs, sdx):w", {}, r"the measurement-error term 'me\(xobs, sdx\)' is not available"),
    ("acat", "ord ~ x + (cs(x) | g)", {}, r"a category-specific group-level term is offered as \(cs\(1\) \| g\).*D89"),
    ("acat", "ord ~ x + (cs(1) || g)", {}, r"a category-specific group-level term is offered as"),
    ("acat", "ord ~ x + (cs(1) | g) + (1 | g)", {}, r"next to another group-level term of its factor.*D89"),
    ("acat", "ord ~ x + (cs(1) | g) + (1 | mm(h, g2))", {}, r"the category-specific group-level term next to"),
    ("cumulative", "ord ~ x + (cs(1) | g)", {}, r"'\(cs\(1\) \| g\)' is not available for the cumulative family.*D89"),
    ("acat", "ord ~ x + (cs(1) | g) + gp(w)", {}, r"the category-specific term '\(cs\(1\) \| g\)' next to the Gaussian process"),
])
def test_family_and_group_refusals(frame, family, formula, kwargs, message):
    with pytest.raises(FormulaError, match=message):
        parse_formula(formula, frame, family, **kwargs)


def test_the_special_term_of_an_me_part_needs_its_main_effect(frame):
    with pytest.raises(FormulaError, match=r"multiplies the latent values of 'me\(xobs, sdx\)', which is no main effect"):
        parse_formula("y ~ w + me(xobs, sdx):w", frame)


def test_contexts_that_refuse_special_products(frame):
    pri = ab.Priors(resp={r: _priors() for r in ("y", "y2")})
    with pytest.raises(FormulaError, match=r"the special term 'mo\(ord\):x' in a multivariate model is not supported"):
        ab.gaussian(["y ~ x + mo(ord):x", "y2 ~ x"], frame, pri, rescor=False)
    with pytest.raises(FormulaError, match=r"in a non-linear parameter's formula is not supported"):
        ab.gaussian(["y ~ a * exp(b * x)", "a ~ 1", "b ~ mo(ord):w"], frame, ab.Priors(), nl=True)
    with pytest.raises(FormulaError, match=r"the monotonic term 'mo\(ord\)' in the formula of zi \(logit link\) is not supported"):
        ab.zero_inflated_poisson(["cnt ~ x", "zi ~ mo(ord)"], frame, ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1),
                                 zi={"intercept": ab.Logistic(0, 1), "b": ab.Normal(0, 1), "simo": ab.Dirichlet(1)}))
    with pytest.raises(FormulaError, match=r"the special term 'mo\(ord\):x'"):
        special_terms.refuse_in(parse_formula("y ~ mo(ord):x", frame), "somewhere")
    special_terms.refuse_in(parse_formula("y ~ x", frame), "somewhere")                     # nothing to refuse
    from autobayes.families.links import check_link_formula
    from autobayes.families import get_family
    with pytest.raises(FormulaError, match=r"the special term 'mo\(ord\):x'"):
        check_link_formula(parse_formula("cnt ~ mo(ord):x", frame, "poisson"), get_family("poisson", "sqrt"))


def test_multinomial_refuses_a_special_product(frame):
    from autobayes.multinomial import _check_category_formula

    with pytest.raises(FormulaError, match=r"the special term 'mo\(ord\):x'"):
        _check_category_formula("mu2 ~ mo(ord):x", parse_formula("mu2 ~ mo(ord):x"))


def test_data_checks(frame):
    with pytest.raises(FormulaError, match=r"column 'z' of 'mo\(ord\):z' must be numeric and finite"):
        parse_formula("y ~ mo(ord):z", frame)
    with pytest.raises(FormulaError, match=r"column 'one' of 'mo\(ord\):one' is constant"):
        parse_formula("y ~ mo(ord):one", frame.assign(one=1.0))
    bad = frame.assign(xg=frame.xg.where(frame.index != 0, 99.0))
    with pytest.raises(FormulaError, match=r"they vary within the level 'f' \(brms: 'Measured values and measurement error should be "
                                           r"unique for each group\. Occured for level 'f' of group 'g'\.'\)"):
        parse_formula("y ~ me(xg, sdxg, gr = g)", bad)
    levels, index, Xn, noise = special_terms.measurement_levels("me(xg, sdxg, gr = g)", frame)
    assert levels == tuple(sorted(frame.g.unique())) and Xn[index[3]] == frame.xg[3] and noise[index[5]] == frame.sdxg[5]


# -- priors ------------------------------------------------------------------------------------------------------------------
def test_simplex_priors_are_keyed_by_id_term_and_column(frame):
    s = parse_formula("y ~ mo(ord) * mo(o2)", frame)
    nested = {"ord": ab.Dirichlet(2), "o2": ab.Dirichlet(3), "mo(ord):mo(o2)": {"ord": ab.Dirichlet(4), "o2": ab.Dirichlet(5)}}
    pr = _priors(simo=nested)
    assert pr.simo_priors(s) == {"ord": ab.Dirichlet(2), "o2": ab.Dirichlet(3)}
    assert pr.special_simo_priors(s) == {"mo(ord):mo(o2)[ord]": ab.Dirichlet(4), "mo(ord):mo(o2)[o2]": ab.Dirichlet(5)}
    flat = {"ord": ab.Dirichlet(2), "o2": ab.Dirichlet(3), "mo(ord):mo(o2)[ord]": ab.Dirichlet(4), "mo(ord):mo(o2)[o2]": ab.Dirichlet(5)}
    assert _priors(simo=flat).special_simo_priors(s) == pr.special_simo_priors(s)
    for simo, message in ((None, r"the monotonic simplexes \['ord', 'o2', 'mo\(ord\):mo\(o2\)\[ord\]'"),
                          (ab.Dirichlet(1), r"several simplexes"),
                          ({**nested, "mo(ord):mo(o2)": ab.Dirichlet(1)}, r"must map each mo\(\) column of the term"),
                          ({"ord": ab.Dirichlet(2)}, r"simo must cover exactly the simplexes"),
                          ({**nested, "o2": ab.Normal(0, 1)}, r"prior for 'simo:o2' must be a prior on a simplex"),
                          ({**nested, "o2": ab.Dirichlet(1, 2, 3)}, r"prior for 'simo:o2' \(2 steps\)")):
        with pytest.raises(PriorError, match=message):
            _priors(simo=simo).special_simo_priors(s)
    one = parse_formula('y ~ mo(ord, id = "a")', frame)
    assert _priors(simo=ab.Dirichlet(2)).special_simo_priors(one) == {"a": ab.Dirichlet(2)}
    assert _priors().special_simo_priors(parse_formula("y ~ me(xobs, sdx) * w", frame)) == {}
    assert _priors(simo={"mo(ord):z": ab.Dirichlet(1)}).special_simo_priors(parse_formula("y ~ mo(ord):z", None,
                                                                                          categorical={"z": "p"})) == {}


def test_default_priors_cover_the_special_products(frame):
    d = ab.default_priors("y ~ mo(ord) * mo(o2)", frame)
    assert d.simo == {"ord": ab.Dirichlet(1), "o2": ab.Dirichlet(1), "mo(ord):mo(o2)": {"ord": ab.Dirichlet(1), "o2": ab.Dirichlet(1)}}
    sigma = ab.default_priors(["y ~ x", "sigma ~ mo(ord, id = 'a')"], frame)
    assert sigma.sigma["simo"] == {"a": ab.Dirichlet(1)}


# -- the models --------------------------------------------------------------------------------------------------------------
def test_the_colon_factor_density_is_the_interaction_as_written(colon):
    """D197: at every frozen point the model's log density (without the Jacobian) is the as-written model's, computed here from
    the fixture's parameters with scipy: each level's term multiplied by its own indicator; brms's own (every level's term times
    the first level's indicator) is brms's lp_nojac, which the model's differs from."""
    b = colon
    sd, frame = b.fixture.standata, b.data
    codes = np.asarray(sd["Xmo_1"]).astype(int)
    for k, p, vp in b.points():
        pars = p.pars
        bsp, simos = np.atleast_1d(pars["bsp"]), [np.atleast_1d(pars[f"simo_{j}"]) for j in (1, 2, 3)]
        mo = [monotonic_values(s, codes) for s in simos]
        X = np.asarray(sd["X"])[:, 1:]
        a, sd1, sigma = (float(np.ravel(pars[k])[0]) for k in ("Intercept", "sd_1", "sigma"))
        base = a + (X - X.mean(0)) @ np.atleast_1d(pars["b"])
        r = np.atleast_1d(pars["r_1_1"])[np.asarray(sd["J_1"]).astype(int) - 1]
        prior = (stats.norm.logpdf(a, 0, 5) + stats.norm.logpdf(pars["b"], 0, 2).sum()
                 + stats.norm.logpdf(bsp, 0, 2).sum() + 2 * np.log(2) + stats.norm.logpdf(sd1, 0, 1)
                 + stats.norm.logpdf(sigma, 0, 1)
                 + sum(stats.dirichlet.logpdf(s, [2, 2, 2]) for s in simos) + stats.norm.logpdf(np.atleast_2d(pars["z_1"]), 0, 1).sum())
        y = np.asarray(sd["Y"])
        written = base + r + sum(bsp[j] * mo[j] * np.asarray(sd[f"Csp_{j + 1}"]) for j in range(3))
        brms = base + r + sum(bsp[j] * mo[j] for j in range(3)) * np.asarray(sd["Csp_1"])
        mine = float(b.logp_nojac(vp))
        assert mine == pytest.approx(prior + stats.norm.logpdf(y, written, sigma).sum(), rel=1e-10)
        assert p.lp_nojac == pytest.approx(prior + stats.norm.logpdf(y, brms, sigma).sum(), rel=1e-10)
        assert abs(mine - p.lp_nojac) > 1.0
        np.testing.assert_array_equal(np.asarray(sd["Csp_2"]), (frame.z == "q").to_numpy(dtype=float))


def test_names_value_point_and_its_refusals(interaction):
    m = interaction.model
    assert m.sp_names == {"b": {"mo(ord):x": "mo(ord):x"}, "simplex": {"mo(ord):x": "simo_mo(ord):x"},
                          "coordinate": {"mo(ord):x": "simo_mo(ord):x_dim"}}
    assert m.names["simo:mo(ord):x"] == "simo_mo(ord):x" and m.names["b:mo(ord):x"] == "mo(ord):x"
    good = dict(intercept=0.1, b=[0.2], sd=0.5, z=np.zeros(6), sigma=1.0)
    simo = {"ord": [0.2, 0.3, 0.5], "mo(ord):x": [0.5, 0.25, 0.25]}
    point = m.value_point(**good, bsp={"ord": 0.3, "mo(ord):x": -0.4}, simo=simo)
    assert point["mo(ord):x"] == -0.4
    np.testing.assert_allclose(point["simo_mo(ord):x_stickbreaking__"], stick_breaking_free([0.5, 0.25, 0.25]))
    for bsp, simo_, message in ((0.3, simo, r"bsp must map each special term"),
                                ({"ord": 0.3}, simo, r"bsp has no coefficient for the special term\(s\) \['mo\(ord\):x'\]"),
                                ({"ord": 0.3, "mo(ord):x": np.inf}, simo, r"bsp for 'mo\(ord\):x' must be a finite scalar"),
                                ({"ord": 0.3, "mo(ord):x": 0.1}, None, r"simo is required for the simplexes"),
                                ({"ord": 0.3, "mo(ord):x": 0.1}, {"ord": simo["ord"]}, r"simo must cover exactly")):
        with pytest.raises(ValueError, match=message):
            m.value_point(**good, bsp=bsp, simo=simo_)
    assert m._special_point(parse_formula("y ~ x", interaction.data), "", 1.0, None) == ({}, 1.0, None)


def test_a_single_special_term_takes_bare_values(frame):
    m = ab.gaussian("y ~ x:mo(ord)", frame, _priors(sd=None, simo=ab.Dirichlet(2)))
    point = m.value_point(intercept=0.0, b=[], sigma=1.0, bsp=0.5, simo=[0.2, 0.3, 0.5])
    assert point["x:mo(ord)"] == 0.5 and "x:mo(ord)" in m.describe()
    from autobayes.default_priors import brms_coefficient_names
    assert brms_coefficient_names(m.spec)["x:mo(ord)"] == "x:moord"


def test_a_factor_with_its_main_effect_and_mo_in_sigma_build(frame):
    """D197's other half (mo(ord) * z, which brms 2.23.0 refuses with an R error) and a plain mo() in sigma's formula: both build,
    their graph checked against the data at build (special_terms.check_effects)."""
    m = ab.gaussian("y ~ mo(ord) * z + (1 | g)", frame, _priors(simo={"ord": ab.Dirichlet(1), "mo(ord):zq": ab.Dirichlet(1),
                                                                      "mo(ord):zr": ab.Dirichlet(1)}), categorical={"z": "p"})
    assert list(m.design.special) == ["mo(ord):zq", "mo(ord):zr"]
    np.testing.assert_array_equal(m.design.special["mo(ord):zr"].covariate, (frame.z == "r").to_numpy(dtype=float))
    assert "1[z_i = q]" in m.describe()
    s = ab.gaussian(["y ~ x + (1 | g)", "sigma ~ mo(ord)"], frame,
                    ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=ab.HalfNormal(1),
                              sigma={"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 0.5), "simo": ab.Dirichlet(2)}))
    assert s.dpar_names["sigma"]["simo:mo(ord)"] == "sigma_simo_mo(ord)" and list(s.design.dpars["sigma"].special) == ["mo(ord)"]
    text = s.describe()
    assert "mo(sigma_simo_mo(ord), ord_i)" in text and "sigma_simo_mo(ord) ~ Dirichlet" in text
    assert "sigma_simo_mo(ord)_stickbreaking__ sigma_simo_mo(ord) on Stan's stick-breaking" in text


def test_the_graph_check_refuses_another_link(interaction):
    with pytest.raises(RuntimeError, match=r"the special terms are checked under the identity and log links"):
        special_terms.check_effects(interaction.model, interaction.model.spec, "mu", "probit", interaction.data)


def test_predictions_and_records(interaction, colon):
    """The fitted and new rows' linear predictor is the hand-written one at the frozen points (also gate G4), and a new level of
    a special term's factor is refused; the target identity and the fit record hold the special products."""
    b = interaction
    idata = adapter.draws(b.fixture, b.model, b.cmap, b.gmap)
    lin = b.model.predict(idata, newdata=b.newdata, kind="linear", include_group_effects=False)["eta"].to_numpy()[0]
    sd = b.fixture.standata
    for k, p in enumerate(b.fixture.points):
        pars = p.pars
        X = b.newdata
        mo = [monotonic_values(np.atleast_1d(pars[f"simo_{j}"]), (X.ord - 1).to_numpy()) for j in (1, 2)]
        want = (float(pars["b_Intercept"]) + np.atleast_1d(pars["b"])[0] * X.x + np.atleast_1d(pars["bsp"])[0] * mo[0]
                + np.atleast_1d(pars["bsp"])[1] * mo[1] * X.x)
        np.testing.assert_allclose(lin[k], want, rtol=1e-12)
    cdraws = adapter.draws(colon.fixture, colon.model, colon.cmap, colon.gmap)
    with pytest.raises(FormulaError, match=r"newdata has levels of the categorical predictor 'z' that were not in the fitted rows"):
        colon.model.predict(cdraws, newdata=colon.newdata.assign(z="s"), kind="linear", include_group_effects=False)
    ident = b.model.target_identity
    assert ident and json.dumps(ident)
    other = ab.gaussian(b.case["autobayes"]["formula"], b.data.copy(), adapter.priors_from_spec(b.case["autobayes"]["priors"]))
    assert other.target_identity == ident
    fit = ab.FitResult(idata, "pymc", "cpu", "cpu", "cpu", "float64", None, 1, 8, 0, 0.9, 0, seed=0,
                       _target_identity_json=json.dumps(b.model.target_identity))
    rec = ab.fit_record(b.model, fit, panel={"values": {}, "status": {}, "overall": "not evaluated"})
    assert rec["model"]["special_products"]["mo(ord):x"] == {"parts": ["mo(ord)"], "covariates": ["x"], "factor": None,
                                                              "level": None, "simplexes": ["mo(ord):x"]}


def test_post_processing_refuses_special_products(interaction):
    idata = adapter.draws(interaction.fixture, interaction.model, interaction.cmap, interaction.gmap)
    fit = ab.FitResult(idata, "pymc", "cpu", "cpu", "cpu", "float64", None, 1, 8, 0, 0.9, 0, seed=0,
                       _target_identity_json=json.dumps(interaction.model.target_identity))
    with pytest.raises(ValueError, match=r"conditional_effects\(\) of a model with the monotonic term 'mo\(ord\)'"):
        ab.conditional_effects(interaction.model, fit, "x")
    m = ab.gaussian("y ~ x + mo(ord):x", interaction.data, _priors(sd=None, simo=ab.Dirichlet(1)))
    with pytest.raises(ValueError, match=r"the special term 'mo\(ord\):x'"):
        ab.conditional_effects(m, fit, "x")
    with pytest.raises(Exception, match=r"the special term 'mo\(ord\):x'"):
        ab.prior_draws(m, draws=10, seed=1)


def test_the_category_specific_group_term_and_expression(frame):
    """(cs(1) | g) with a cs() of a factor: the per-threshold intercepts and the factor's contrast columns both enter mu_cs, as
    the build-time check reads off the graph; predictions keep the threshold dimension with the group effects."""
    pri = ab.Priors(intercept=ab.Normal(0, 3), b=ab.Normal(0, 1.5), sd={"g": ab.HalfNormal(1)}, cor={"g": ab.LKJ(2)})
    m = ab.acat("ord ~ x + cs(z) + (cs(1) | g)", frame, pri, categorical={"z": "p"})
    assert list(m.design.category_specific) == ["zq", "zr"]
    assert [c for (g, c) in m.design.group_blocks] == ["Intercept[1]", "Intercept[2]", "Intercept[3]"]
    text = m.describe()
    assert "r_g[g_i, k]" in text and "x_zq,i * bcs_zq[k]" in text
    with pytest.raises(ValueError, match=r"bcs must cover exactly the category-specific columns \['zq', 'zr'\]"):
        m._category_specific_point({"z": [0.0] * 3})
    assert set(m._category_specific_point({"zq": [0.1, 0.2, 0.3], "zr": [0.0, 0.0, 0.0]})) == {"cs(z)[q]", "cs(z)[r]"}


def test_brms_special_predictor_reads_brms_s_structure(interaction):
    b = interaction
    p = b.fixture.points[0]
    sd = b.fixture.standata
    mo = [monotonic_values(np.atleast_1d(p.pars[f"simo_{j}"]), np.asarray(sd[f"Xmo_{j}"]).astype(int)) for j in (1, 2)]
    want = np.atleast_1d(p.pars["bsp"])[0] * mo[0] + np.atleast_1d(p.pars["bsp"])[1] * mo[1] * np.asarray(sd["Csp_1"])
    np.testing.assert_allclose(adapter.brms_special_predictor(b.fixture, b.cmap, p.pars), want, rtol=1e-14)
    assert adapter.brms_special_predictor(b.fixture, b.cmap, p.pars, "_sigma") == 0.0


def test_more_refusals_and_the_term_s_invariant(frame):
    with pytest.raises(ValueError, match=r"a special product has its SpecialProduct, its name and no design column"):
        PopTerm("mo(ord):w", "special_product", (), special=special_terms.SpecialProduct(("mo(ord)", "x")))
    with pytest.raises(FormulaError, match=r"cs\(\) takes column names joined by '\+' here, not 'x \* w'"):
        parse_formula("ord ~ cs(x * w)", frame, "acat")
    with pytest.raises(FormulaError, match=r"me\(\) takes the noisy column, the column of its known standard deviations and optionally"):
        parse_formula("y ~ me(xobs, sdx) + me(xobs, sdx, foo):w", frame)


def test_me_gr_in_a_special_product(frame, monkeypatch):
    """me(xg, sdxg, gr = g) * w: the product reads each row's level's latent value (Model._measurement_rows), checked at build;
    describe(), value_point's per-level zme and the build-time checks of the per-level variables."""
    pri = _priors(meanme=ab.Normal(0, 2), sdme=ab.HalfNormal(1))
    m = ab.gaussian("y ~ me(xg, sdxg, gr = g) * w + (1 | h)", frame, pri)
    rows = m._measurement_rows("xg")
    np.testing.assert_array_equal(rows, m.design.measurement_error["xg"].index)
    assert m._measurement_rows("xg") is not None and list(m.design.special) == ["me(xg, sdxg, gr = g):w"]
    text = m.describe()
    assert "the latent (true) value of xg in level j of g" in text and "Xme_xg,i * w_i" in text
    good = dict(intercept=0.0, b=[0.1], sd=0.5, z=np.zeros(4), sigma=1.0, bsp={"xg": 0.2, "me(xg, sdxg, gr = g):w": 0.1},
                meanme=0.0, sdme=1.0)
    assert m.value_point(**good, zme=np.zeros(6))["zme_xg"].shape == (6,)
    with pytest.raises(ValueError, match=r"zme for 'xg' must have 6 finite entries, one per level of the gr column"):
        m.value_point(**good, zme=np.zeros(36))
    roles = m.me_names["xg"]
    pmm = m.pymc_model
    monkeypatch.setitem(pmm.named_vars_to_dims, roles["z"], ("other",))
    with pytest.raises(RuntimeError, match=r"PyMC dims of 'zme_xg' are not \('Xme_xg_dim',\)"):
        m._measurement_design()
    monkeypatch.setitem(pmm._coords, roles["coordinate"], ("x",) * 6)
    with pytest.raises(RuntimeError, match=r"is not the sorted levels of the gr column"):
        m._measurement_design()


def test_graph_checks_refuse_a_graph_the_hook_did_not_build(interaction, monkeypatch):
    m, spec = interaction.model, interaction.model.spec
    with pytest.raises(RuntimeError, match=r"the graph's 'sigma' does not move by the values of 'mo\(ord\):x'"):
        special_terms.check_effects(m, spec, "sigma", "identity", interaction.data)
    real = special_terms.names(spec)
    monkeypatch.setattr(special_terms, "names", lambda s, prefix="": {**real, "simplex": {"mo(ord):x": "sigma"}})
    with pytest.raises(RuntimeError, match=r"'sigma' is not a free simplex of the graph"):
        special_terms.check_effects(m, spec, "mu", "identity", interaction.data)
    monkeypatch.setattr(special_terms, "names", lambda s, prefix="": {**real, "b": {"mo(ord):x": "simo_ord"}})
    with pytest.raises(RuntimeError, match=r"'simo_ord' is not a free scalar of the graph"):
        special_terms.check_effects(m, spec, "mu", "identity", interaction.data)


def test_sigma_s_linear_predictor_has_its_special_term():
    """gauss_mo_id_sigma: predict(dpar = "sigma") is brms's sigma linear predictor, computed here from the fixture's
    Intercept_sigma, bsp_sigma and simo_sigma_1 at sigma's codes Xmo_sigma_1 (brms's own centring: none for mo())."""
    b = gates.Bundle("gauss_mo_id_sigma")
    idata = adapter.draws(b.fixture, b.model, b.cmap, b.gmap)
    eta = b.model.predict(idata, dpar="sigma", kind="linear")["eta"].to_numpy()[0]
    codes = np.asarray(b.fixture.standata["Xmo_sigma_1"]).astype(int)
    for k, p in enumerate(b.fixture.points):
        want = (float(np.ravel(p.pars["Intercept_sigma"])[0])
                + float(np.ravel(p.pars["bsp_sigma"])[0]) * monotonic_values(np.atleast_1d(p.pars["simo_sigma_1"]), codes))
        np.testing.assert_allclose(eta[k], want, rtol=1e-12)


def test_cs_group_under_brms_default_priors_and_bcs_checks(frame):
    pri = ab.default_priors("ord ~ x + (cs(1) | g)", frame, family="acat")
    m = ab.acat("ord ~ x + (cs(1) | g)", frame, pri)
    assert m.spec.has_category_specific and not m.spec.cs_terms
    c = ab.acat("ord ~ cs(x + z) + (1 | g)", frame, _priors(sigma=None, intercept=ab.Normal(0, 3)), categorical={"z": "p"})
    with pytest.raises(ValueError, match=r"bcs for 'zq' must be 3 finite numbers, one per threshold"):
        c._category_specific_point({"x": [0.0] * 3, "zq": [0.0], "zr": [0.0] * 3})


def test_prior_predictive_draws_the_special_products(frame):
    m = ab.gaussian("y ~ x + mo(ord):x", frame, _priors(sd=None, simo=ab.Dirichlet(2)))
    prior = m.prior_predictive(draws=5, seed=1).prior
    assert prior["simo_mo(ord):x"].shape == (1, 5, 3) and prior["mo(ord):x"].shape == (1, 5)


def test_a_three_way_product_and_a_covariate_written_first(frame):
    s = parse_formula("y ~ mo(ord):x:w", frame)
    np.testing.assert_allclose(s.sp_terms[0].special.covariate(frame), frame.x * frame.w)
    me = parse_formula("y ~ w * me(xobs, sdx)", frame)
    assert [t.name for t in me.pop_terms] == ["w", "me(xobs, sdx)", "w:me(xobs, sdx)"]
    assert me.sp_terms[0].special.numeric == ("w",) and me.sp_terms[0].special.parts == ("me(xobs, sdx)",)


# -- the oracle's extension fails where it should --------------------------------------------------------------------------
def _corrupted(bundle, mutate):
    import copy

    raw = copy.deepcopy(bundle.fixture.raw)
    mutate(raw)
    return bundle.with_fixture(fx.Fixture(raw))


def _scale(raw, key, k, by):
    raw["standata"][key]["values"][k] += by


def test_g1_fails_on_a_corrupted_covariate_code_count_or_concentration(interaction):
    gates.check_design(interaction)                                        # the frozen fixture passes
    for key, by, message in (("Csp_1", 1e-6, r"brms's Csp_1 is not the covariate of 'mo\(ord\):x'"),
                             ("Xmo_2", 1, r"brms's Xmo_2 is not the codes of the monotonic part"),
                             ("con_simo_2", 0.5, r"brms con_simo_2 is .*the model's prior gives"),
                             ("Ksp", 1, r"brms has Ksp = 3(\.0)? special terms; the AutoBayes formula has 2")):
        with pytest.raises(adapter.OracleMismatch, match=message):
            gates.check_design(_corrupted(interaction, lambda raw: _scale(raw, key, 0, by)))
    def extra(raw):
        raw["standata"]["Csp_2"] = raw["standata"]["Csp_1"]
    with pytest.raises(adapter.OracleMismatch, match=r"brms has \['Csp_2'\], which no special term of the AutoBayes model holds"):
        gates.check_design(_corrupted(interaction, extra))


def test_g1_fails_on_a_corrupted_level_index():
    b = gates.Bundle("gauss_me_gr")
    gates.check_me_groups(b)
    with pytest.raises(adapter.OracleMismatch, match=r"brms's Jme_1 is not each row's level of the gr column"):
        gates.check_me_groups(_corrupted(b, lambda raw: _scale(raw, "Jme_1", 0, 1)))
    with pytest.raises(adapter.OracleMismatch, match=r"brms's Nme_1 is 7"):
        gates.check_me_groups(_corrupted(b, lambda raw: _scale(raw, "Nme_1", 0, 1)))


def test_the_new_checks_refuse_brms_data_without_a_model_term(interaction):
    m = gates.Bundle("gauss_monotonic")                                    # no special product, no me(gr = )
    def add(raw):
        raw["standata"]["Csp_1"] = raw["standata"]["Xmo_1"]
        raw["standata"]["Jme_1"] = raw["standata"]["Xmo_1"]
    bad = _corrupted(m, add)
    with pytest.raises(adapter.OracleMismatch, match=r"brms has \['Csp_1'\]; the AutoBayes model has no special term"):
        gates.check_special_covariates(bad)
    with pytest.raises(adapter.OracleMismatch, match=r"brms has \['Jme_1'\]; the AutoBayes model has no me\(x, sdx, gr = g\)"):
        gates.check_me_groups(bad)


def test_sigma_s_linear_predictor_with_population_terms_and_a_special_term(frame):
    """predict(dpar = "sigma") of sigma ~ x + mo(ord): its intercept, x's coefficient and the special term at each row, from
    prior draws standing in for a posterior."""
    import arviz as az

    m = ab.gaussian(["y ~ x", "sigma ~ x + mo(ord)"], frame,
                    ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2),
                              sigma={"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 0.5), "simo": ab.Dirichlet(2)}))
    post = m.prior_predictive(draws=3, seed=2).prior
    eta = m.predict(az.InferenceData(posterior=post), dpar="sigma", kind="linear")["eta"].to_numpy()[0]
    codes = (frame.ord - frame.ord.min()).to_numpy()
    for d in range(3):
        want = (float(post["sigma_Intercept"][0, d]) + float(post["sigma_x"][0, d]) * frame.x.to_numpy()
                + float(post["sigma_mo(ord)"][0, d]) * monotonic_values(post["sigma_simo_mo(ord)"][0, d].to_numpy(), codes))
        np.testing.assert_allclose(eta[d], want, rtol=1e-12)
    with pytest.raises(FormulaError, match=r"the response 'ord' cannot be a category-specific predictor"):
        parse_formula("ord ~ x + cs(w + ord)", frame, "acat")
