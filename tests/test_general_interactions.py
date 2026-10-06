"""General two-way interactions, ``x:w`` (numeric by numeric) and ``z:h`` (categorical by categorical) (brms-parity
feature A04; docs/interactions.md).

Refusals with their messages, the term model and columns, the design against hand-built columns, the graph's linear
predictor and density at a point, by-hand and Bambi-path predictions on fitted and new rows, per-cell priors, the
empty-cell warning, the constant-product refusal, the fit record, the target identity and ``describe()``. Build-only;
nothing samples. The numbers against brms are the oracle's business (tests/oracle, cases
``gauss_interaction_numeric_numeric`` and ``gauss_interaction_categorical_categorical``)."""
import copy
import json
import re
from dataclasses import asdict

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.stats import halfnorm, norm

import autobayes as ab
from autobayes.formula import Categorical, PopTerm, interaction_cells, pop_term, pop_term_design
from tests.test_population_only import PANEL_STUB

LEVELS_G = ("g0", "g1", "g2", "g3")
Z, H = ("p", "q", "r"), ("h1", "h2", "h3")
CELLS = ("q, h2", "q, h3", "r, h2", "r, h3")                     # non-reference levels of z by those of h, z slowest
CELL_COLUMNS = ("z[q]:h[h2]", "z[q]:h[h3]", "z[r]:h[h2]", "z[r]:h[h3]")


def frame(n=24, seed=5, drop_cell=None) -> pd.DataFrame:
    """Every (z, h) cell observed unless ``drop_cell`` names one; x and w correlated so that mean(x * w) differs
    from mean(x) * mean(w)."""
    rng = np.random.default_rng(seed)
    z = np.array([Z[i % 3] for i in range(n)]); h = np.array([H[(i // 3) % 3] for i in range(n)])
    x = rng.normal(0.3, 1.0, n)
    df = pd.DataFrame({"y": rng.normal(size=n), "x": x, "w": 0.6 * x + rng.normal(size=n), "z": z, "h": h,
                       "g": [LEVELS_G[i % 4] for i in range(n)], "k": rng.integers(0, 4, n), "n": np.full(n, 5)})
    if drop_cell is not None:
        a, b = drop_cell
        df.loc[(df.z == a) & (df.h == b), "h"] = H[0] if b != H[0] else H[1]
    return df


PRIORS = ab.Priors(intercept=ab.Normal(0.2, 5), b=ab.Normal(0.1, 2), sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1.5))
POP_PRIORS = ab.Priors(intercept=ab.Normal(0.2, 5), b=ab.Normal(0.1, 2), sigma=ab.HalfNormal(1.5))
CATS = {"z": "p", "h": "h1"}


def indicator(labels, levels):
    return np.column_stack([np.asarray(labels) == l for l in levels]).astype(float)


def hand_design(df):
    """x, w, z[q], z[r], h[h2], h[h3], x * w, then one indicator product per cell, written without the library."""
    x, w = df.x.to_numpy(dtype=float), df.w.to_numpy(dtype=float)
    cols = [x, w, *indicator(df.z, Z[1:]).T, *indicator(df.h, H[1:]).T, x * w]
    cols += [((df.z == a) & (df.h == b)).to_numpy(dtype=float) for a in Z[1:] for b in H[1:]]
    return np.column_stack(cols)


@pytest.fixture(scope="module")
def both():
    """``y ~ x * w + z * h + (1 | g)``: a numeric-by-numeric and a categorical-by-categorical interaction together."""
    return ab.gaussian("y ~ x * w + z * h + (1 | g)", frame(), PRIORS, categorical=CATS)


# -- 1. the grammar: what is accepted and every refusal by message ----------------------------------------------
REFUSALS = [
    ("y ~ x:w:z", {}, "only a two-way interaction is supported in this slice ('x:w:z' names 3 terms); three-way and higher-order interactions are not"),
    ("y ~ x * w * z", {}, "only a two-way interaction is supported in this slice ('x * w * z' names 3 terms)"),
    ("y ~ x + x:w", {}, "the main effect 'w' is missing"),
    ("y ~ w + x:w", {}, "the main effect 'x' is missing"),
    ("y ~ z + z:h", {"categorical": {"z": "p", "h": "h1"}}, "the main effect 'h' is missing"),
    ("y ~ x + w + x:w + w:x", {}, "the interaction 'x:w' appears more than once after expansion"),
    ("y ~ z + h + z:h + h:z", {"categorical": {"z": "p", "h": "h1"}}, "the interaction 'z:h' appears more than once after expansion"),
    ("y ~ x * w + w:x", {}, "the interaction 'x:w' appears more than once after expansion"),
    ("y ~ x + w + x:x", {}, "a term cannot interact with itself"),
    ("y ~ x + x:g + (1 | g)", {}, "the grouping factor 'g' cannot enter a fixed interaction"),
    ("y ~ x + y:x", {}, "the response or trials column cannot enter an interaction"),
    ("k | trials(n) ~ x + x:n", {"family": "binomial"}, "the response or trials column cannot enter an interaction"),
    ("y ~ x + w + offset(w):x", {}, "offset() cannot enter an interaction; an offset is a known column added to the linear predictor with no coefficient"),
    ("y ~ x + z + log(x):z", {"categorical": {"z": "p"}}, "function calls and transformations inside the formula are not supported in an interaction"),
    ("y ~ x + w + x*w:z", {}, "mixing '*' and ':' in one term is not supported"),
    ("y ~ x + w + x:w + (1 + x:w || g)", {}, "an interaction cannot be a varying slope"),
    ("y ~ x + w + x:w + (1 | x:w)", {}, "'x': the grouping factor cannot also be a predictor or the response"),
]


@pytest.mark.parametrize("formula,kwargs,message", REFUSALS, ids=[r[0] for r in REFUSALS])
def test_refusals_name_the_construct(formula, kwargs, message):
    with pytest.raises(ab.FormulaError, match=re.escape(message)):
        ab.parse_formula(formula, **kwargs)


def test_data_refusals_and_warnings():
    df = frame()
    with pytest.raises(ab.FormulaError, match=re.escape("interaction 'x:w' is constant (x * w takes a single value in the fitted rows)")):
        ab.parse_formula("y ~ x * w", df.assign(w=1.0 / df.x))                 # x and w vary, x * w does not
    # a cell with no fitted rows is a zero column, kept as brms keeps it, with a warning
    m = ab.gaussian("y ~ z * h", frame(drop_cell=("r", "h3")), POP_PRIORS, categorical=CATS)
    assert m.warnings == ["cell(s) ['r, h3'] of 'z:h' have no fitted rows: their coefficients are informed by the prior alone (the design column is zero, as in brms)."]
    assert m.spec.columns[-4:] == CELL_COLUMNS and not m.design.X_raw[:, -1].any() and m.design.x_means[-1] == 0.0
    assert "WARNING: cell(s) ['r, h3'] of 'z:h' have no fitted rows" in m.describe()
    # cell labels that would coincide are refused rather than merged (Bambi joins the levels with ', ')
    bad = Categorical("z", "p", levels=("p", "q, h2", "q")), Categorical("h", "h1", levels=("h1", "h2", "h2, h2"))
    with pytest.raises(ab.FormulaError, match=re.escape("interaction 'z:h': two cells both label as 'q, h2, h2'")):
        pop_term("z:h", "interaction", bad)


# -- 2. the term model ----------------------------------------------------------------------------------------
def test_term_model_and_canonical_order():
    s = ab.parse_formula("y ~ x * w + (1 | g)")
    t = s.pop_terms[-1]
    assert s.pop_terms == (PopTerm("x", "numeric", ("x",), ()), PopTerm("w", "numeric", ("w",), ()),
                           PopTerm("x:w", "interaction", ("x:w",), (), ()))
    assert t.operands == ("x", "w") and t.categorical == () and not t.is_vector and t.levels == () and t.columns == ("x:w",)
    assert s.rhs == "x + w + x:w + (1 | g)" and s.bambi_formula == "y ~ x + w + x:w + (1 | g)" and s.columns == ("x", "w", "x:w")
    assert s.interactions == (("x", "w"),) and s.interaction_names == ("x:w",) and s.interaction_of("x:w") == ("x", "w")
    assert s.pop_columns == ("x", "w") and s.predictors == ("x", "w") and s.column_terms[-1] == ("x:w", "x:w", None)
    assert not s.is_vector_term("x:w") and s.levels_of("x:w") == () and s.term_columns("x:w") == ("x:w",)
    # the operands keep main-effect order, as R orders the variables of a term; 'x:z' stays numeric first
    assert ab.parse_formula("y ~ w + x + x:w").rhs == "w + x + w:x" and ab.parse_formula("y ~ w + x + w:x").rhs == "w + x + w:x"
    assert ab.parse_formula("y ~ h + z + z:h", categorical=CATS).rhs == "h + z + h:z"
    assert ab.parse_formula("y ~ z + x + z:x", categorical={"z": "p"}).rhs == "z + x + x:z"
    # categorical by categorical: a vector term whose levels are the cells, known once the data are seen
    before = ab.parse_formula("y ~ z * h", categorical=CATS)
    zh = before.pop_terms[-1]
    assert zh == PopTerm("z:h", "interaction", ("z:h",), (), ("z", "h")) and zh.is_vector and zh.operands == ("z", "h")
    after = ab.parse_formula("y ~ z * h", frame(), categorical=CATS)
    zh = after.pop_terms[-1]
    assert zh.levels == CELLS and zh.columns == CELL_COLUMNS and zh.categorical == ("z", "h")
    assert after.columns == ("z[q]", "z[r]", "h[h2]", "h[h3]", *CELL_COLUMNS) and after.levels_of("z:h") == CELLS
    assert after.column_terms[-4:] == tuple((c, "z:h", j) for j, c in enumerate(CELL_COLUMNS))
    hz = ab.parse_formula("y ~ h + z + h:z", frame(), categorical=CATS).pop_terms[-1]          # the first operand is slowest
    assert hz.columns == ("h[h2]:z[q]", "h[h2]:z[r]", "h[h3]:z[q]", "h[h3]:z[r]") and hz.levels == ("h2, q", "h2, r", "h3, q", "h3, r")
    cats = (Categorical("z", "p", levels=Z), Categorical("h", "h1", levels=H))
    assert interaction_cells(*cats) == (("q", "h2"), ("q", "h3"), ("r", "h2"), ("r", "h3"))
    assert pop_term("z:h", "interaction", cats) == PopTerm("z:h", "interaction", CELL_COLUMNS, CELLS, ("z", "h"))
    assert pop_term("x:w", "interaction", cats) == PopTerm("x:w", "interaction", ("x:w",), (), ())
    # the legacy constructor still reads the last operand of an interaction as its categorical side
    assert PopTerm("x:z", "interaction", ("x:z[q]",), ("q",)).categorical == ("z",) and PopTerm("x", "numeric", ("x",)).categorical == ()
    with pytest.raises(ValueError, match="categorical operands \\['v'\\] are not among its operands"):
        PopTerm("x:w", "interaction", ("x:w",), (), ("v",))
    for replica in (copy.deepcopy(after), ab.ModelSpec(**asdict(after))):
        assert replica == after and hash(replica) == hash(after) and replica.pop_terms == after.pop_terms


# -- 3. the design, the graph and the names ----------------------------------------------------------------------
def test_design_matches_hand_built_columns(both):
    m, df, d = both, both.data, both.design
    assert m.spec.columns == ("x", "w", "z[q]", "z[r]", "h[h2]", "h[h3]", "x:w", *CELL_COLUMNS)
    X = hand_design(frame())
    assert np.array_equal(d.X_raw, X) and np.allclose(d.X_model, X - X.mean(0)) and np.allclose(d.x_means, X.mean(0)) and d.centered
    xw = m.spec.columns.index("x:w")
    assert np.isclose(d.x_means[xw], np.mean(df.x * df.w)) and abs(d.x_means[xw] - df.x.mean() * df.w.mean()) > 0.05
    for j, (a, b) in enumerate(interaction_cells(m.spec.categorical("z"), m.spec.categorical("h"))):
        assert np.isclose(d.x_means[xw + 1 + j], np.mean((frame().z == a) & (frame().h == b)))
    assert m.coefficients == (("x", "x", None), ("w", "w", None), ("z[q]", "z", 0), ("z[r]", "z", 1), ("h[h2]", "h", 0), ("h[h3]", "h", 1),
                              ("x:w", "x:w", None), *((c, "z:h", j) for j, c in enumerate(CELL_COLUMNS)))
    # Bambi coded the terms as the term model says, and PyMC's coordinates are the cells
    comp = m.bambi_model.components["mu"].design.common
    assert comp.terms["x:w"].kind == "interaction" and list(comp.terms["x:w"].labels) == ["x:w"]
    assert comp.terms["z:h"].kind == "interaction" and list(comp.terms["z:h"].labels) == list(CELL_COLUMNS) and list(comp.terms["z:h"].levels) == list(CELLS)
    assert list(m.pymc_model.coords["z:h_dim"]) == list(CELLS) and "x:w_dim" not in m.pymc_model.coords
    assert {v.name for v in m.pymc_model.free_RVs} == {"Intercept", "x", "w", "z", "h", "x:w", "z:h", "sigma", "1|g_sigma", "1|g_offset"}
    assert m.names["b:x:w"] == "x:w" and m.names["b:z:h"] == "z:h"
    # each term's own block, and the whole raw design, from pop_term_design
    blocks = [pop_term_design(t, df, m.spec.categoricals) for t in m.spec.pop_terms]
    assert np.array_equal(np.concatenate(blocks, axis=1), d.X_raw) and blocks[-2].shape == (24, 1) and blocks[-1].shape == (24, 4)
    with pytest.raises(ValueError, match="needs the declared Categorical terms to build the cells of 'z:h'"):
        pop_term_design(m.spec.pop_term("z:h"), df)
    # the identity differs from the main-effects-only model on the same data and from either interaction alone
    plain = ab.gaussian("y ~ x + w + z + h + (1 | g)", frame(), PRIORS, categorical=CATS)
    only_xw = ab.gaussian("y ~ x * w + z + h + (1 | g)", frame(), PRIORS, categorical=CATS)
    assert len({m.target_identity["sha256"], plain.target_identity["sha256"], only_xw.target_identity["sha256"]}) == 3


def natural(k=0):
    return dict(intercept=0.3 + 0.1 * k, b=np.linspace(-0.6, 0.8, 11) + 0.05 * k, sd=0.8 - 0.1 * k,
                z=np.array([-0.4, 0.2, 0.9, -0.1]) + 0.1 * k, sigma=0.9 + 0.1 * k)


def test_graph_linear_predictor_and_density_at_a_point(both):
    m, d = both, both.design
    pmm = m.pymc_model
    fn = pmm.compile_fn(pmm.replace_rvs_by_values([pmm[m.names["linpred"]]]), inputs=pmm.value_vars, point_fn=True, on_unused_input="ignore")
    logp, nojac = pmm.compile_logp(), pmm.compile_logp(jacobian=False)
    for k in range(3):
        nat = natural(k)
        pt = m.value_point(**nat)
        assert float(pt["x:w"]) == nat["b"][6] and np.array_equal(pt["z:h"], nat["b"][7:])
        eta = nat["intercept"] + d.X_model @ nat["b"] + d.group_designs["g"].Z @ (nat["sd"] * nat["z"])
        assert np.allclose(np.asarray(fn(pt)[0]).ravel(), eta, atol=1e-12)
        ref = (norm.logpdf(nat["intercept"], 0.2, 5) + norm.logpdf(nat["b"], 0.1, 2).sum() + norm.logpdf(nat["z"]).sum()
               + halfnorm.logpdf(nat["sd"], scale=1) + halfnorm.logpdf(nat["sigma"], scale=1.5)
               + norm.logpdf(d.y, eta, nat["sigma"]).sum())
        assert abs(float(nojac(pt)) - ref) < 1e-8 and abs(float(logp(pt)) - ref - np.log(nat["sd"]) - np.log(nat["sigma"])) < 1e-8


# -- 4. predictions on fitted and new rows ----------------------------------------------------------------------
def synthetic(draws=4, seed=8):
    rng = np.random.default_rng(seed)
    shape = (1, draws)
    var = {"Intercept": (("chain", "draw"), rng.normal(0.2, 0.1, shape)), "x": (("chain", "draw"), rng.normal(0.4, 0.1, shape)),
           "w": (("chain", "draw"), rng.normal(-0.3, 0.1, shape)), "x:w": (("chain", "draw"), rng.normal(0.25, 0.1, shape)),
           "z": (("chain", "draw", "z_dim"), rng.normal(0, 0.5, shape + (2,))), "h": (("chain", "draw", "h_dim"), rng.normal(0, 0.5, shape + (2,))),
           "z:h": (("chain", "draw", "z:h_dim"), rng.normal(0, 0.5, shape + (4,))),
           "1|g_sigma": (("chain", "draw"), rng.uniform(0.3, 0.7, shape)), "1|g": (("chain", "draw", "g__factor_dim"), rng.normal(0, 0.5, shape + (4,))),
           "sigma": (("chain", "draw"), rng.uniform(0.3, 0.5, shape))}
    coords = {"chain": [0], "draw": np.arange(draws), "z_dim": list(Z[1:]), "h_dim": list(H[1:]), "z:h_dim": list(CELLS), "g__factor_dim": list(LEVELS_G)}
    return az.InferenceData(posterior=xr.Dataset(var, coords=coords))


def hand_eta(post, df, with_groups=True):
    """Intercept + every coefficient times its hand-built column + the row's group effect, per draw."""
    X = hand_design(df)
    B = np.concatenate([post["x"].to_numpy()[..., None], post["w"].to_numpy()[..., None], post["z"].to_numpy(), post["h"].to_numpy(),
                        post["x:w"].to_numpy()[..., None], post["z:h"].to_numpy()], axis=-1)
    eta = post["Intercept"].to_numpy()[..., None] + np.einsum("cdp,np->cdn", B, X)
    if with_groups:
        eta = eta + post["1|g"].to_numpy()[..., [LEVELS_G.index(g) for g in df.g]]
    return eta


def test_predictions_on_fitted_and_new_rows(both):
    m, idata = both, synthetic()
    P = idata.posterior
    eta = hand_eta(P, frame())
    assert np.allclose(m.predict(idata, kind="linear")["eta"].to_numpy(), eta, atol=1e-12)
    assert np.allclose(m.predict(idata, kind="mean")["mu"].to_numpy(), eta, atol=1e-10)                      # Bambi's path agrees
    assert np.allclose(m.predict(idata, kind="linear", include_group_effects=False)["eta"].to_numpy(), hand_eta(P, frame(), False), atol=1e-12)
    # new rows: products and cells recomputed from the raw columns; a fitted cell unseen in a row's level pair is just another cell
    new = pd.DataFrame({"x": [0.0, 1.5, -2.0, 0.5], "w": [1.0, -0.5, 2.0, 0.0], "z": ["p", "q", "r", "r"], "h": ["h1", "h3", "h2", "h1"],
                        "g": ["g2", "g0", "g3", "g1"]})
    eta_new = hand_eta(P, new)
    out = m.predict(idata, newdata=new, kind="linear")
    assert np.allclose(out["eta"].to_numpy(), eta_new, atol=1e-12) and out.attrs["new_levels"] == []
    assert np.allclose(m.predict(idata, newdata=new, kind="mean")["mu"].to_numpy(), eta_new, atol=1e-10)
    # reference rows carry no interaction coefficient: row 0 is (p, h1) with w = 1, so eta = a + b_x * 0 + b_w * 1 + b_x:w * 0 + u
    row0 = P["Intercept"].to_numpy() + P["w"].to_numpy() + P["1|g"].to_numpy()[..., 2]
    assert np.allclose(out["eta"].to_numpy()[..., 0], row0, atol=1e-12)
    # the slope in x at w = -0.5 in cell (q, h3): rows 1 and a shifted copy differ by b_x + w * b_x:w
    shifted = new.iloc[[1]].assign(x=2.5)
    diff = m.predict(idata, newdata=shifted, kind="linear")["eta"].to_numpy()[..., 0] - out["eta"].to_numpy()[..., 1]
    assert np.allclose(diff, P["x"].to_numpy() + (-0.5) * P["x:w"].to_numpy(), atol=1e-12)
    fixed = m.predict(idata, newdata=new.drop(columns="g"), kind="linear", include_group_effects=False)["eta"].to_numpy()
    assert np.allclose(fixed, hand_eta(P, new, False), atol=1e-12)
    with pytest.raises(ab.FormulaError, match=re.escape("newdata lacks columns ['w']")):
        m.predict(idata, newdata=new.drop(columns="w"), kind="linear")
    with pytest.raises(ab.FormulaError, match="not in the fitted rows: \\['h9'\\]"):
        m.predict(idata, newdata=new.assign(h=["h1", "h9", "h2", "h1"]), kind="linear")
    with pytest.raises(ab.FormulaError, match="newdata column 'w' must be numeric"):
        m.predict(idata, newdata=new.assign(w=["a", "b", "c", "d"]), kind="linear")


# -- 5. priors --------------------------------------------------------------------------------------------------
def test_prior_forms_and_refusals():
    df = frame()
    def pri(b):
        return ab.Priors(intercept=ab.Normal(0.2, 5), b=b, sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1.5))
    per_term = pri({"x": ab.Normal(0, 2), "w": ab.Normal(0, 2), "z": ab.Normal(0, 2), "h": ab.Normal(0, 2), "x:w": ab.Normal(0, 1),
                    "z:h": {"q, h2": ab.Normal(0, 1), "q, h3": ab.Normal(0.5, 1), "r, h2": ab.Normal(-0.5, 2), "r, h3": ab.Normal(0, 3)}})
    m = ab.gaussian("y ~ x * w + z * h + (1 | g)", df, per_term, categorical=CATS)
    coef = per_term.coefficient_priors(m.spec)
    assert coef["x:w"] == ab.Normal(0, 1) and coef["z[q]:h[h3]"] == ab.Normal(0.5, 1) and coef["z[r]:h[h2]"] == ab.Normal(-0.5, 2)
    assert list(coef) == list(m.spec.columns)
    bambi_prior = per_term.to_bambi(m.spec, "gaussian")["z:h"]
    assert np.array_equal(bambi_prior.args["mu"], [0.0, 0.5, -0.5, 0.0]) and np.array_equal(bambi_prior.args["sigma"], [1.0, 1.0, 2.0, 3.0])
    assert "b_z[q]:h[h3] ~ Normal(mu=0.5, sigma=1)" in m.describe() and "b_x:w ~ Normal(mu=0, sigma=1)" in m.describe()
    # the single prior per term keeps working, and the density agrees when the per-cell mapping spells it out
    same = pri({"x": ab.Normal(0.1, 2), "w": ab.Normal(0.1, 2), "z": ab.Normal(0.1, 2), "h": ab.Normal(0.1, 2), "x:w": ab.Normal(0.1, 2),
                "z:h": {c: ab.Normal(0.1, 2) for c in CELLS}})
    m1, m2 = ab.gaussian("y ~ x * w + z * h + (1 | g)", df, PRIORS, categorical=CATS), ab.gaussian("y ~ x * w + z * h + (1 | g)", df, same, categorical=CATS)
    pt = m1.value_point(**natural())
    assert abs(float(m1.pymc_model.compile_logp()(pt)) - float(m2.pymc_model.compile_logp()(pt))) < 1e-12
    base = {"x": ab.Normal(0, 2), "w": ab.Normal(0, 2), "z": ab.Normal(0, 2), "h": ab.Normal(0, 2)}
    for b, msg in [
        ({**base, "x:w": {"q": ab.Normal(0, 1)}, "z:h": ab.Normal(0, 1)}, "predictor 'x:w' takes one prior, got dict"),
        ({**base, "x:w": ab.Normal(0, 1), "z:h": {"q, h2": ab.Normal(0, 1)}},
         "per-cell priors for 'z:h' must cover exactly its cells ['q, h2', 'q, h3', 'r, h2', 'r, h3'] (each 'l, m' pairs a non-reference level of 'z' with one of 'h'; "
         "the references 'p' and 'h1' have no cell coefficient); missing ['q, h3', 'r, h2', 'r, h3'], unexpected []"),
        ({**base, "x:w": ab.Normal(0, 1), "z:h": {**{c: ab.Normal(0, 1) for c in CELLS}, "p, h1": ab.Normal(0, 1)}}, "unexpected ['p, h1']"),
        ({**base, "x:w": ab.Normal(0, 1), "z:h": {**{c: ab.Normal(0, 1) for c in CELLS}, "q": ab.Normal(0, 1)}}, "unexpected ['q']"),
        ({**base, "w:x": ab.Normal(0, 1), "z:h": ab.Normal(0, 1)}, "write the interaction key in its canonical spelling 'x:w' (the main effects' order)"),
        ({**base, "x:w": ab.Normal(0, 1), "h:z": ab.Normal(0, 1)}, "write the interaction key in its canonical spelling 'z:h' (the main effects' order)"),
        ({**base, "x:w": ab.Normal(0, 1), "z:h": ab.Normal(0, 1), "z[q]:h[h2]": ab.Normal(0, 1)}, "'z[q]:h[h2]' looks like a design-column label: give per-level priors as b={'z:h': {level: prior}}"),
        ({**base, "x:w": ab.Normal(0, 1), "z:h": {**{c: ab.Normal(0, 1) for c in CELLS[:-1]}, "r, h3": ab.StudentT(3, 0, 1)}}, "share one distribution family"),
        ({**base, "x:w": ab.HalfNormal(1), "z:h": ab.Normal(0, 1)}, "must be a location prior"),
    ]:
        with pytest.raises(ab.PriorError, match=re.escape(msg)):
            ab.gaussian("y ~ x * w + z * h + (1 | g)", df, pri(b), categorical=CATS)


# -- 6. intercept removal, records and describe() ----------------------------------------------------------------
def test_no_intercept_codes_the_first_categorical_full_rank_and_the_cells_by_contrasts():
    pr = ab.Priors(intercept=None, b=ab.Normal(0, 2), sigma=ab.HalfNormal(1))
    m = ab.gaussian("y ~ 0 + z * h", frame(), pr, categorical=CATS)
    assert m.spec.columns == ("z[p]", "z[q]", "z[r]", "h[h2]", "h[h3]", *CELL_COLUMNS) and not m.design.centered
    X = m.design.X_raw
    assert np.array_equal(X[:, :3], indicator(m.data.z.astype(str), Z)) and np.array_equal(X[:, 5:], hand_design(frame())[:, 7:])
    m2 = ab.gaussian("y ~ 0 + x * w", frame(), pr)
    assert m2.spec.columns == ("x", "w", "x:w") and np.array_equal(m2.design.X_raw[:, 2], frame().x.to_numpy() * frame().w.to_numpy())


def test_fit_record_and_describe(both):
    m = both
    idata = synthetic(draws=50)
    fit = ab.FitResult(idata, "pymc", "cpu", "cpu", "cpu", "float64", None, 1, 50, 0, 0.9, 0, seed=0,
                       _target_identity_json=json.dumps(m.target_identity))
    rec = ab.fit_record(m, fit, panel=PANEL_STUB)
    json.dumps(rec)
    assert rec["model"]["interactions"] == [["x", "w"], ["z", "h"]] and rec["model"]["terms"] == ["x", "w", "z", "h", "x:w", "z:h"]
    assert rec["model"]["design_columns"] == list(m.spec.columns) and rec["model"]["canonical_formula"] == "y ~ x + w + z + h + x:w + z:h + (1 | g)"
    assert rec["model"]["priors"]["b:x:w"] == "Normal(mu=0.1, sigma=2)" and rec["model"]["priors"]["b:z[r]:h[h3]"] == "Normal(mu=0.1, sigma=2)"
    assert {"x:w", "z:h[q, h2]", "z:h[r, h3]"} <= set(rec["posterior"])                 # ArviZ names a row variable[coordinate]
    assert np.isclose(rec["posterior"]["z:h[q, h3]"]["mean"], idata.posterior["z:h"].to_numpy()[..., 1].mean())
    text = m.describe()
    for s in ["AutoBayes M0 model with categorical predictor(s) and interaction(s)",
              "expanded  : y ~ x + w + z + h + x:w + z:h + (1 | g)",
              "(x_x * x_w - q_x:w) * b_x:w", "(1[z_i = q] * 1[h_i = h2] - p_z:h[q, h2]) * b_z:h[q, h2]", "(1[z_i = r] * 1[h_i = h3] - p_z:h[r, h3]) * b_z:h[r, h3] + u[g_i]",
              "x:w: the slope in x on the response scale is b_x + w * b_x:w and the slope in w is b_w + x * b_x:w; b_x and b_w are the slopes at w = 0 and at x = 0",
              "z:h: a row in level l of z and level m of h adds the cell's own b_z:h[l, m] to the main-effect coefficients of z and h",
              "the cell coefficient is zero when either level is the reference ('p' for z, 'h1' for h)",
              "The product column x * w is centred at its OWN mean q_x:w = ", "not at mean(x) * mean(w) = ",
              "Each cell column 1[z = l] * 1[h = m] is centred at the cell's proportion of the fitted rows: p_z:h[q, h2] = ",
              "at the fitted-row means of every expanded design column (x, 1[z = l], x * w, 1[z = l] * 1[h = m])",
              "intercept a = a_c - sum_p mean(x_p) * b_p - sum_l p_z[l] * b_z[l] - q_x:w * b_x:w - sum_(l,m) p_z:h[l, m] * b_z:h[l, m], computed after sampling",
              "b_z[q]:h[h2] ~ Normal(mu=0.1, sigma=2)",
              f"  {'z:h':<24} b_z:h[(q, h2), (q, h3), (r, h2), (r, h3)] (vector; dims z:h_dim, posterior rows z:h[q, h2], z:h[q, h3], z:h[r, h2], z:h[r, h3])",
              f"  {'x:w':<24} b_x:w"]:
        assert s in text, s
    assert "b_z[l] is the difference between level l and the reference 'p' on the response scale at the same predictors and group" in text
    # a numeric-by-categorical model keeps its wording: the new shapes add lines, they do not rewrite the old ones
    old = ab.gaussian("y ~ x * z + (1 | g)", frame(), PRIORS, categorical={"z": "p"}).describe()
    assert "x:z: the reference level 'p' has the raw intercept a and slope b_x" in old and "q_x:w" not in old and "cell" not in old
    assert "at the fitted-row means of every expanded design column (x, 1[z = l], x * 1[z = l])" in old
    uncentred = ab.gaussian("y ~ x * w + (1 | g)", frame(), PRIORS, center_predictors=False).describe()
    assert "mu_i = a + x_x * b_x + x_w * b_w + x_x * x_w * b_x:w + u[g_i]" in uncentred
