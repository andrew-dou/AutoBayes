"""Grouping syntax ``(1 | g:h)`` and ``(1 | g/h)`` (brms-parity feature A03; docs/grouping-syntax.md).

The expansion of ``g/h``, the derived factor ``g:h`` with brms's ``g_h`` labels over the observed combinations,
refusals with their messages, the design blocks and PyMC names, SD priors keyed by the factor as written,
``value_point`` and the density, predictions on fitted rows and on new rows (fitted pairs, unseen combinations
under both policies), the fit record and ``describe()``. Build-only; nothing samples. The numbers against brms
are the oracle's business (tests/oracle, cases ``gauss_nested_colon`` and ``gauss_nested_slash``)."""
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
from autobayes.formula import GroupTerm, bambi_group_name, factor_column, group_labels, sd_tag
from tests.test_population_only import PANEL_STUB

LEVELS_G, LEVELS_H = ("g0", "g1", "g2", "g3"), ("h0", "h1", "h2")
UNOBSERVED = {("g0", "h1"), ("g3", "h2")}                 # two combinations never seen, so g:h has 10 levels, not 12


def frame(seed=4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [(LEVELS_G[i % 4], LEVELS_H[(i // 4) % 3]) for i in range(24)]
    rows = [r for r in rows if r not in UNOBSERVED]
    n = len(rows)
    df = pd.DataFrame({"y": rng.normal(size=n), "x": rng.normal(0.4, 1.0, n), "w": rng.normal(size=n),
                       "z": rng.choice(["p", "q", "r"], size=n), "g": [g for g, _ in rows], "h": [h for _, h in rows],
                       "k": rng.integers(0, 4, n), "n": np.full(n, 5)})
    assert n == 20 and set(df.z) == {"p", "q", "r"}
    return df


PAIRS = tuple(sorted({(g, h) for g, h in zip(frame().g, frame().h)}))
LEVELS_GH = tuple(f"{g}_{h}" for g, h in PAIRS)
PRIORS = ab.Priors(intercept=ab.Normal(0.2, 5), b=ab.Normal(0.1, 2), sigma=ab.HalfNormal(1.5),
                   sd={"g": ab.HalfNormal(1), "g:h": ab.HalfNormal(0.5)})


def indicator(labels, levels):
    return np.column_stack([np.asarray(labels) == l for l in levels]).astype(float)


@pytest.fixture(scope="module")
def colon():
    return ab.gaussian("y ~ x + (1 | g) + (1 | g:h)", frame(), PRIORS)


@pytest.fixture(scope="module")
def slash():
    return ab.gaussian("y ~ x + (1 | g/h)", frame(), PRIORS)


# -- 1. the grammar: expansion, the term model and every refusal by message -------------------------------------
REFUSALS = [
    ("y ~ x + (1 | g:h:k)", {}, "only two factors can be combined in a grouping factor in this slice ('g:h:k' names 3 columns)"),
    ("y ~ x + (1 | g/h/k)", {}, "only two factors can be combined in a grouping factor in this slice ('g/h/k' names 3 columns)"),
    ("y ~ x + (1 | g/h:k)", {}, "mixing '/' and ':' in a grouping factor ('g/h:k') is not supported in this slice"),
    ("y ~ x + (1 | g:g)", {}, "a grouping factor cannot be combined with itself ('g:g')"),
    ("y ~ x + (1 | g/g)", {}, "a grouping factor cannot be combined with itself ('g/g')"),
    ("y ~ x + (1 | g:h) + (1 | g:h)", {}, "duplicate grouping factor 'g:h'; a factor appears in one group term, '(1 | g:h)' or '(1 + x || g:h)'"),
    ("y ~ x + (1 | g:h) + (1 | h:g)", {}, "'(1 | h:g)' groups by the same columns as '(1 | g:h)' ('h:g' and 'g:h' induce the same partition of the rows); keep one of them"),
    ("y ~ x + (1 | g) + (1 | g/h)", {}, "duplicate grouping factor 'g'; a factor appears in one group term, '(1 | g)' or '(1 + x || g)' ('(1 | g/h)' expands to '(1 | g) + (1 | g:h)')"),
    ("y ~ x + (1 | g/h) + (1 | g:h)", {}, "duplicate grouping factor 'g:h'"),
    ("y ~ x + (1 | y:h)", {}, "'y': the grouping factor cannot also be a predictor or the response"),
    ("y ~ x + (1 | x:h)", {}, "'x': the grouping factor cannot also be a predictor or the response"),
    ("y ~ x + (1 | g/y)", {}, "'y': the grouping factor cannot also be a predictor or the response"),
    ("k | trials(n) ~ x + (1 | g:n)", {"family": "binomial"}, "'n': the trials column cannot also be a predictor or the grouping factor"),
    ("y ~ x + g_h + (1 | g:h)", {}, "the derived grouping factor 'g:h' is handed to Bambi as the column 'g_h', which is also a column of the formula; rename that column"),
    ("y ~ x + (1 | g_h) + (1 | g:h)", {}, "the derived grouping factor 'g:h' is handed to Bambi as the column 'g_h', which is also a column of the formula"),
    ("y ~ x + (1 | a_b:c) + (1 | a:b_c)", {}, "the derived grouping factors 'a_b:c' and 'a:b_c' would both be handed to Bambi as the column 'a_b_c'"),
    ("y ~ x + z + (1 | z:h)", {"categorical": {"z": "p"}}, "'z': the grouping factor cannot also be a predictor or the response"),
    ("y ~ x + (1 + g || g:h)", {}, "the grouping factor 'g' cannot also be a varying slope"),
    ("y ~ x + (1 + h || g/h)", {}, "the grouping factor 'h' cannot also be a varying slope"),
    ("y ~ x + (1 + x | g:h)", {}, "brms's '|' gives the varying intercept and slope(s) a correlation matrix"),
    ("y ~ x + x:h + (1 | g:h)", {}, "the grouping factor 'h' cannot enter a fixed interaction"),
    ("y ~ x + (1 | g:1h)", {}, "'1h' is not a column name"),
    ("y ~ x + (1 | g:)", {}, "'' is not a column name"),
    ("y ~ x + (1 | log(g):h)", {}, "transformed grouping factors are not supported"),
    ("y ~ x + (1 | g:h | id)", {}, "correlation IDs are not supported"),
    ("y ~ x + (1 | mm(g, h):k)", {}, "multimembership grouping is not supported"),
]


@pytest.mark.parametrize("formula,kwargs,message", REFUSALS, ids=[r[0] for r in REFUSALS])
def test_refusals_name_the_construct(formula, kwargs, message):
    with pytest.raises(ab.FormulaError, match=re.escape(message)):
        ab.parse_formula(formula, **kwargs)


def test_expansion_and_the_term_model():
    s = ab.parse_formula("y ~ x + (1 | g) + (1 | g:h)")
    t = s.group_terms[1]
    assert s.group_terms == (GroupTerm("g", ("Intercept",), True), GroupTerm("g:h", ("Intercept",), True))
    assert s.groups == ("g", "g:h") and s.group_coefficients == (("g", "Intercept"), ("g:h", "Intercept"))
    assert t.operands == ("g", "h") and t.derived and t.column == "g_h" and t.canonical == "(1 | g:h)" and t.bambi_canonical == "(1 | g_h)"
    assert not s.group_terms[0].derived and s.group_terms[0].column == "g" and s.group_terms[0].operands == ("g",)
    assert s.rhs == "x + (1 | g) + (1 | g:h)" and s.bambi_formula == "y ~ x + (1 | g) + (1 | g_h)"
    # brms: (1 | g/h) is (1 | g) + (1 | g:h), in that order
    e = ab.parse_formula("y ~ x + (1 | g/h)")
    assert e.group_terms == s.group_terms and e.rhs == s.rhs and e.canonical_formula == "y ~ x + (1 | g) + (1 | g:h)" and e.bambi_formula == s.bambi_formula
    assert ab.parse_formula("y ~ x + (1 | g / h)").group_terms == ab.parse_formula("y ~ x + (1 | g : h) + (1|g)").group_terms[::-1] == s.group_terms
    # the derived factor alone, and with independent varying slopes
    alone = ab.parse_formula("y ~ x + (1 | g:h)")
    assert alone.groups == ("g:h",) and alone.group == "g:h" and alone.group_term == "1|g:h"
    sl = ab.parse_formula("y ~ x + (1 + x || g:h)")
    assert sl.group_terms == (GroupTerm("g:h", ("Intercept", "x"), False),) and sl.bambi_formula == "y ~ x + (1 + x | g_h)"
    assert ab.parse_formula("y ~ x + (1 + x || g/h)").group_terms == (GroupTerm("g", ("Intercept", "x"), False), GroupTerm("g:h", ("Intercept", "x"), False))
    # names: Bambi's are built from the column, the user-facing ones from the factor as written
    assert bambi_group_name("g:h", "Intercept") == "1|g_h" and bambi_group_name("g:h", "x") == "x|g_h" and factor_column("g:h") == "g_h"
    assert sd_tag("g:h", "Intercept") == "g:h" and sd_tag("g:h", "x") == "x|g:h"
    # the specification round-trips
    for replica in (copy.deepcopy(s), ab.ModelSpec(**asdict(s))):
        assert replica == s and hash(replica) == hash(s)


def test_derived_labels_are_the_observed_combinations():
    df = frame()
    labels = group_labels("g:h", df)
    assert list(labels) == [f"{g}_{h}" for g, h in zip(df.g, df.h)] and list(group_labels("g", df)) == list(df.g.astype(str))
    s = ab.parse_formula("y ~ x + (1 | g/h)", df)
    assert s.levels_by_group["g"] == LEVELS_G and s.levels_by_group["g:h"] == LEVELS_GH and len(LEVELS_GH) == 10
    assert {("g0", "h1"), ("g3", "h2")} == UNOBSERVED and not ({"g0_h1", "g3_h2"} & set(LEVELS_GH))
    # labels that would collide after joining with '_' are refused rather than merged (brms merges them silently)
    bad = df.copy()
    bad.loc[0, ["g", "h"]] = ["g0_h0", "x"]
    bad.loc[1, ["g", "h"]] = ["g0", "h0_x"]
    with pytest.raises(ab.FormulaError, match=re.escape("grouping factor 'g:h': the combinations ('g0_h0', 'x') and ('g0', 'h0_x') both label as 'g0_h0_x'")):
        ab.parse_formula("y ~ x + (1 | g:h)", bad)
    # the operand columns must exist and every derived level needs company
    with pytest.raises(ab.FormulaError, match=re.escape("columns not in data: ['h']")):
        ab.parse_formula("y ~ x + (1 | g:h)", df.drop(columns="h"))
    with pytest.raises(ab.FormulaError, match="grouping factor 'g:h' has a single level"):
        ab.parse_formula("y ~ x + (1 | g:h)", df.assign(g="a", h="b"))


# -- 2. the design blocks, the graph and the names -----------------------------------------------------------------
def test_design_blocks_names_and_priors(colon, slash):
    m, df = colon, colon.data
    d = m.design
    gh = [f"{g}_{h}" for g, h in zip(df.g, df.h)]
    assert list(df["g_h"]) == gh and list(df.g) == list(frame().g)                       # the derived column lives in the fitted frame
    assert list(d.group_blocks) == [("g", "Intercept"), ("g:h", "Intercept")]
    bg, bgh = d.group_blocks[("g", "Intercept")], d.group_blocks[("g:h", "Intercept")]
    assert bg.levels == LEVELS_G and bgh.levels == LEVELS_GH and bgh.J == 10
    assert np.array_equal(bg.Z, indicator(df.g, LEVELS_G)) and np.array_equal(bgh.Z, indicator(gh, LEVELS_GH))
    assert bgh.group_index.tolist() == [LEVELS_GH.index(l) for l in gh] and bgh.column_slice == slice(4, 14)
    assert np.array_equal(d.Z, np.column_stack([bg.Z, bgh.Z])) and d.group_slices == {"g": slice(0, 4), "g:h": slice(4, 14)}
    assert d.columns == ("x",) and d.centered
    # PyMC: the derived factor's variables are named after the column handed to Bambi
    pmm = m.pymc_model
    assert [v.name for v in pmm.value_vars if "|" in v.name] == ["1|g_sigma_log__", "1|g_offset", "1|g_h_sigma_log__", "1|g_h_offset"]
    assert list(pmm.coords["g_h__factor_dim"]) == list(LEVELS_GH) and tuple(pmm.named_vars_to_dims["1|g_h"]) == ("g_h__factor_dim",)
    assert m.group_names == {"g": {"sd": "1|g_sigma", "offset": "1|g_offset", "group_effect": "1|g", "coordinate": "g__factor_dim"},
                             "g:h": {"sd": "1|g_h_sigma", "offset": "1|g_h_offset", "group_effect": "1|g_h", "coordinate": "g_h__factor_dim"}}
    assert m.group_coefficient_names[("g:h", "Intercept")] == m.group_names["g:h"]
    # priors are keyed by the factor as written
    assert m.priors.sd_priors(m.spec) == {("g", "Intercept"): ab.HalfNormal(1), ("g:h", "Intercept"): ab.HalfNormal(0.5)}
    assert m.priors.group_priors(m.spec) == {"g": ab.HalfNormal(1), "g:h": ab.HalfNormal(0.5)}
    assert {"1|g", "1|g_h"} <= set(m.priors.to_bambi(m.spec, "gaussian"))
    with pytest.raises(ab.PriorError, match=re.escape("sd must cover exactly the grouping factors ['g', 'g:h']; got ['g', 'g_h']")):
        ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1), sd={"g": ab.HalfNormal(1), "g_h": ab.HalfNormal(0.5)}).validate(m.spec)
    with pytest.raises(ab.PriorError, match="each varying intercept needs a prior on the SD: pass sd for \\['g', 'g:h'\\]"):
        ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1)).validate(m.spec)
    # (1 | g/h) builds the same target; only the formula line differs
    assert slash.target_identity == m.target_identity and slash.spec.canonical_formula == m.spec.canonical_formula
    assert slash.describe().replace("y ~ x + (1 | g/h)", "y ~ x + (1 | g) + (1 | g:h)").replace("  expanded  : y ~ x + (1 | g) + (1 | g:h)   (canonical form handed to Bambi)\n", "") == m.describe()
    # a model with the plain factor g only, or with g and h crossed, is a different target
    plain = ab.gaussian("y ~ x + (1 | g) + (1 | h)", frame(), ab.Priors(intercept=ab.Normal(0.2, 5), b=ab.Normal(0.1, 2), sigma=ab.HalfNormal(1.5), sd={"g": ab.HalfNormal(1), "h": ab.HalfNormal(0.5)}))
    assert plain.target_identity != m.target_identity


def test_slope_on_the_derived_factor_falls_out_naturally():
    df = frame()
    m = ab.gaussian("y ~ x + (1 + x || g:h)", df, ab.Priors(intercept=ab.Normal(0.2, 5), b=ab.Normal(0.1, 2), sigma=ab.HalfNormal(1.5),
                                                          sd={"g:h": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(0.5)}}))
    gh = [f"{g}_{h}" for g, h in zip(df.g, df.h)]
    ind = indicator(gh, LEVELS_GH)
    assert list(m.design.group_blocks) == [("g:h", "Intercept"), ("g:h", "x")]
    assert np.array_equal(m.design.group_blocks[("g:h", "x")].Z, ind * df.x.to_numpy()[:, None])
    assert m.group_names["g:h"]["sd:x"] == "x|g_h_sigma" and m.group_names["g:h"]["group_effect:x"] == "x|g_h"
    assert "x_x,i * v_x|g:h[g:h_i]" in m.describe() and "sd_x|g:h ~ HalfNormal(sigma=0.5)" in m.describe()


def natural(k=0):
    return dict(intercept=0.3 + 0.1 * k, b=np.array([0.5 - 0.05 * k]),
                sd={"g": 0.8 - 0.1 * k, "g:h": 0.4 + 0.05 * k},
                z={"g": np.array([-0.4, 0.2, 0.9, -0.1]) + 0.1 * k, "g:h": np.linspace(-1.0, 1.0, 10) * (1 + 0.1 * k)},
                sigma=0.9 + 0.1 * k)


def test_value_point_and_the_density(colon):
    m, df, d = colon, colon.data, colon.design
    pmm = m.pymc_model
    logp, nojac = pmm.compile_logp(), pmm.compile_logp(jacobian=False)
    linpred = pmm.compile_fn(pmm.replace_rvs_by_values([pmm[m.names["linpred"]]]), inputs=pmm.value_vars, point_fn=True, on_unused_input="ignore")
    for k in range(3):
        nat = natural(k)
        pt = m.value_point(**nat)
        assert float(pt["1|g_h_sigma_log__"]) == np.log(nat["sd"]["g:h"]) and np.array_equal(pt["1|g_h_offset"], nat["z"]["g:h"])
        eta = (nat["intercept"] + d.X_model @ nat["b"] + d.group_designs["g"].Z @ (nat["sd"]["g"] * nat["z"]["g"])
               + d.group_designs["g:h"].Z @ (nat["sd"]["g:h"] * nat["z"]["g:h"]))
        ref = (norm.logpdf(nat["intercept"], 0.2, 5) + norm.logpdf(nat["b"], 0.1, 2).sum() + norm.logpdf(nat["z"]["g"]).sum()
               + norm.logpdf(nat["z"]["g:h"]).sum() + halfnorm.logpdf(nat["sd"]["g"], scale=1) + halfnorm.logpdf(nat["sd"]["g:h"], scale=0.5)
               + halfnorm.logpdf(nat["sigma"], scale=1.5) + norm.logpdf(df.y.to_numpy(), eta, nat["sigma"]).sum())
        assert abs(float(nojac(pt)) - ref) < 1e-8
        assert abs(float(logp(pt)) - ref - np.log(nat["sd"]["g"]) - np.log(nat["sd"]["g:h"]) - np.log(nat["sigma"])) < 1e-8
        assert np.allclose(np.asarray(linpred(pt)[0]).ravel(), eta, atol=1e-12)
    with pytest.raises(ValueError, match=re.escape("sd must cover exactly the grouping factors ['g', 'g:h']")):
        m.value_point(**{**natural(), "sd": {"g": 0.8, "g_h": 0.4}})
    with pytest.raises(ValueError, match="z for 'g:h' must have 10 finite entries"):
        m.value_point(**{**natural(), "z": {"g": np.zeros(4), "g:h": np.zeros(12)}})


# -- 3. predictions on fitted and new rows ----------------------------------------------------------------------
def synthetic(draws=4, seed=8):
    rng = np.random.default_rng(seed)
    shape = (1, draws)
    var = {"Intercept": (("chain", "draw"), rng.normal(0.2, 0.1, shape)), "x": (("chain", "draw"), rng.normal(0.4, 0.1, shape)),
           "1|g_sigma": (("chain", "draw"), rng.uniform(0.3, 0.7, shape)), "1|g_h_sigma": (("chain", "draw"), rng.uniform(0.2, 0.4, shape)),
           "1|g": (("chain", "draw", "g__factor_dim"), rng.normal(0, 0.5, shape + (4,))),
           "1|g_h": (("chain", "draw", "g_h__factor_dim"), rng.normal(0, 0.3, shape + (10,))),
           "sigma": (("chain", "draw"), rng.uniform(0.3, 0.5, shape))}
    return az.InferenceData(posterior=xr.Dataset(var, coords={"chain": [0], "draw": np.arange(draws), "g__factor_dim": list(LEVELS_G), "g_h__factor_dim": list(LEVELS_GH)}))


def test_predictions_on_fitted_rows_and_fitted_pairs(colon):
    m, idata = colon, synthetic()
    P = idata.posterior
    a, b, Ug, Ugh = P["Intercept"].to_numpy(), P["x"].to_numpy(), P["1|g"].to_numpy(), P["1|g_h"].to_numpy()
    x, gi, ghi = m.data.x.to_numpy(dtype=float), m.design.group_designs["g"].group_index, m.design.group_designs["g:h"].group_index
    eta = a[..., None] + b[..., None] * x + Ug[..., gi] + Ugh[..., ghi]
    assert np.allclose(m.predict(idata, kind="linear")["eta"].to_numpy(), eta, atol=1e-12)
    assert np.allclose(m.predict(idata, kind="mean")["mu"].to_numpy(), eta, atol=1e-10)                   # the Bambi path agrees
    assert np.allclose(m.predict(idata, kind="linear", include_group_effects=False)["eta"].to_numpy(), a[..., None] + b[..., None] * x, atol=1e-12)
    # new rows with fitted (g, h) pairs: newdata carries g and h, the derived labels are built here
    new = pd.DataFrame({"x": [0.0, 1.0, -2.0], "g": ["g1", "g2", "g0"], "h": ["h1", "h0", "h2"]})
    assert all((g, h) in PAIRS for g, h in zip(new.g, new.h))
    eta_new = a[..., None] + b[..., None] * new.x.to_numpy() + Ug[..., [LEVELS_G.index(g) for g in new.g]] + Ugh[..., [LEVELS_GH.index(f"{g}_{h}") for g, h in zip(new.g, new.h)]]
    out = m.predict(idata, newdata=new, kind="linear")
    assert np.allclose(out["eta"].to_numpy(), eta_new, atol=1e-12) and out.attrs["new_groups"] == "per_factor"
    assert json.loads(out.attrs["new_levels_by_factor_json"]) == {"g": [], "g:h": []}
    assert np.allclose(m.predict(idata, newdata=new, kind="mean")["mu"].to_numpy(), eta_new, atol=1e-10)
    with pytest.raises(ab.FormulaError, match=re.escape("newdata lacks columns ['h']")):
        m.predict(idata, newdata=new.drop(columns="h"), kind="linear")
    fixed = m.predict(idata, newdata=new.drop(columns=["g", "h"]), kind="linear", include_group_effects=False)["eta"].to_numpy()
    assert np.allclose(fixed, a[..., None] + b[..., None] * new.x.to_numpy(), atol=1e-12)


@pytest.mark.parametrize("policy", ["population", "resample_observed"])
def test_unseen_combinations_follow_the_factor_policy(colon, policy):
    """A pair of fitted levels never seen together is an unseen level of g:h, while g itself is a fitted level."""
    m, idata = colon, synthetic()
    P = idata.posterior
    a, b, Ug, Ugh, sd_gh = P["Intercept"].to_numpy(), P["x"].to_numpy(), P["1|g"].to_numpy(), P["1|g_h"].to_numpy(), P["1|g_h_sigma"].to_numpy()
    new = pd.DataFrame({"x": [0.5, 0.5, -1.0], "g": ["g0", "g0", "g1"], "h": ["h1", "h1", "h0"]})     # (g0, h1) was never observed
    with pytest.raises(ab.FormulaError, match=re.escape("newdata has levels of 'g:h' not seen in training: ['g0_h1']")):
        m.predict(idata, newdata=new, kind="linear")
    with pytest.raises(ab.FormulaError, match="not seen in training"):
        m.predict(idata, newdata=new, kind="linear", new_groups={"g": policy, "g:h": "error"})
    seed = 13
    out = m.predict(idata, newdata=new, kind="linear", new_groups={"g": "error", "g:h": policy}, seed=seed)
    eta = out["eta"].to_numpy()
    assert json.loads(out.attrs["new_levels_by_factor_json"]) == {"g": [], "g:h": ["g0_h1"]}
    rng = np.random.default_rng(seed)                                      # factor order g, then g:h; one effect per unseen label
    if policy == "population":
        u_star = rng.normal(0.0, 1.0, size=sd_gh.shape) * sd_gh
    else:
        chosen = rng.integers(0, len(LEVELS_GH), size=sd_gh.shape)
        u_star = np.take_along_axis(Ugh, chosen[..., None], axis=-1)[..., 0]
    base = a[..., None] + b[..., None] * new.x.to_numpy() + Ug[..., [0, 0, 1]]
    assert np.allclose(eta[..., 0], base[..., 0] + u_star, atol=1e-12) and np.allclose(eta[..., 1], base[..., 1] + u_star, atol=1e-12)
    assert np.allclose(eta[..., 2], base[..., 2] + Ugh[..., LEVELS_GH.index("g1_h0")], atol=1e-12)
    # a label unseen in g is unseen in g:h as well, and each factor applies its own policy
    both = pd.DataFrame({"x": [0.0], "g": ["new"], "h": ["h0"]})
    out = m.predict(idata, newdata=both, kind="linear", new_groups=policy, seed=seed)
    assert json.loads(out.attrs["new_levels_by_factor_json"]) == {"g": ["new"], "g:h": ["new_h0"]}


# -- 4. records and describe() ----------------------------------------------------------------------------------
def test_fit_record_and_describe(colon, slash):
    m = colon
    idata = synthetic(draws=50)
    fit = ab.FitResult(idata, "pymc", "cpu", "cpu", "cpu", "float64", None, 1, 50, 0, 0.9, 0, seed=0,
                       _target_identity_json=json.dumps(m.target_identity))
    rec = ab.fit_record(m, fit, panel=PANEL_STUB)
    json.dumps(rec)
    assert [g["factor"] for g in rec["model"]["groups"]] == ["g", "g:h"]
    gh = rec["model"]["groups"][1]
    assert gh["term"] == "1|g:h" and gh["levels"] == list(LEVELS_GH) and gh["J"] == 10 and gh["names"] == m.group_names["g:h"]
    assert rec["model"]["priors"]["sd:g:h"] == "HalfNormal(sigma=0.5)" and rec["model"]["priors"]["sd:g"] == "HalfNormal(sigma=1)"
    assert {"1|g_sigma", "1|g_h_sigma"} <= set(rec["posterior"])
    text = m.describe()
    for s in ["AutoBayes M0 model with derived grouping factor(s)",
              "n = 20 observations, J = 4 levels of g; J = 10 levels of g:h",
              "levels g:h: 0=g0_h0, 1=g0_h2, 2=g1_h0",
              "g:h       : derived grouping factor, one level per observed (g, h) combination, labelled g_h as brms does (paste(g, h, sep = \"_\")); "
              "unobserved combinations have no level; handed to Bambi as the column 'g_h', so its PyMC variables are 1|g_h, 1|g_h_sigma, 1|g_h_offset",
              "mu_i = a_c + (x_x - mean(x_x)) * b_x + u_g[g_i] + u_g:h[g:h_i]",
              "u_g:h[j] = sd_g:h * z_g:h[j], z_g:h[j] ~ Normal(0, 1), j = 0..9",
              "sd_g:h ~ HalfNormal(sigma=0.5)",
              "1|g_h_sigma_log__        log(sd_g:h)", "1|g_h_offset             z_g:h",
              "Deterministic: 1|g = u_g, 1|g_h = u_g:h, mu = linear predictor"]:
        assert s in text, s
    text2 = slash.describe()
    assert "  formula   : y ~ x + (1 | g/h)" in text2 and "  expanded  : y ~ x + (1 | g) + (1 | g:h)   (canonical form handed to Bambi)" in text2
