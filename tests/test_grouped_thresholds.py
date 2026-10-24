"""Grouped thresholds of the cumulative family, brms's ``thres(gr = z)`` and ``thres(K, gr = z)`` (brms-parity F06b,
docs/grouped-thresholds.md): the term's parsing and its refusals, the levels and their numbers of thresholds, the block-ordered
transform, the row density against the 50-digit reference of ``tests/test_ordinal_categorical.py`` on both paths (PyTensor's
gradient and JAX's autodiff), the model's density and gradient by hand, ``value_point``, predictions with the zero
probabilities of the categories a level does not reach, ``log_lik()`` and ``bayes_R2()`` by hand, prior-predictive draws, the
prior table (brms's printed rows), ``describe()`` and the target identity. No MCMC; the brms comparison is the oracle cases
``cumulative_thres_gr`` and ``cumulative_thres_k_gr``."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
import pytest
import xarray as xr
import arviz as az
from scipy.special import expit
from scipy.stats import halfnorm, norm

import autobayes as ab
from autobayes import grouped_thresholds as gt
from autobayes.formula import AdditionTerm, FormulaError, parse_formula
from autobayes.priors import PriorError
from tests.test_ordinal_categorical import exact_ordinal_row

# -- data: levels a and b reach category 4 (three thresholds), c only 3 (two) -----------------------------------------
X = [-1.0, 0.5, 2.0, 0.3, -0.2, 1.1, 0.0, -0.7, 0.9, -1.4, 0.2, 1.6]
Y = [1, 2, 4, 3, 2, 1, 4, 2, 3, 1, 3, 2]
Z = list("aaaaabbbbccc")


def frame(**over):
    df = pd.DataFrame({"x": X, "y": Y, "z": Z, "g": ["g1", "g2", "g3"] * 4, "w": [0.4, 1.0, -0.3, 0.8, 1.5, -1.1, 0.6, 0.0, 2.1, -0.4, 0.9, 1.2],
                       "e": [1.5, 2.0, 0.5, 1.0, 3.0, 1.2, 0.8, 2.2, 1.1, 0.6, 1.9, 2.4], "h": list("pqpqpqpqpqpq")})
    for k, v in over.items():
        df[k] = v
    return df


def priors(group=True):
    return ab.Priors(intercept=ab.Normal(0, 3), b=ab.Normal(0, 1.5), **({"sd": ab.HalfNormal(1)} if group else {}))


def model(formula="y | thres(gr = z) ~ x + (1 | g)", df=None, **kw):
    return ab.cumulative(formula, frame() if df is None else df, priors("|" in formula.split("~")[1]), **kw)


TAU = {"a": np.array([-0.8, 0.4, 1.9]), "b": np.array([-1.3, -0.2, 0.7]), "c": np.array([-0.5, 1.1])}


# -- the term -----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("text, count, column", [
    ("thres(gr = z)", None, "z"), ("thres(3, gr = z)", 3, "z"), ("thres(3, z)", 3, "z"), ("thres(gr = z, 3)", 3, "z"),
    ("thres(x = 3, gr = z)", 3, "z"), ("thres(gr=z)", None, "z")])
def test_the_term_is_matched_as_r_matches_brms_s_arguments(text, count, column):
    spec = parse_formula(f"y | {text} ~ x", family="cumulative")
    a = spec.addition("thres")
    assert (a.kind, a.count, a.column) == ("thres", count, column)
    assert a.term == ("thres(gr = z)" if count is None else "thres(3, gr = z)") and a.columns == ("z",)
    assert spec.canonical_formula == f"y | {a.term} ~ x"


@pytest.mark.parametrize("text, message", [
    ("thres(3)", r"'thres\(3\)': thres\(\) without gr = .*divergence D158"),
    ("thres()", r"thres\(\) without gr = "),
    ("thres(K = 3, gr = z)", r"takes the number of thresholds x and the grouping column gr.*got 'K'"),
    ("thres(gr = z, gr = h)", r"each at most once.*got 'gr'"),
    ("thres(3, z, h)", "at most two arguments"),
    ("thres(gr = z + h)", r"gr must be a column name, got 'z \+ h'"),
    ("thres(2.5, gr = z)", r"one positive whole number for every level of 'z'.*got '2.5'.*Number of thresholds should be unique"),
    ("thres(c(a = 3, b = 2), gr = z)", "one positive whole number"),
    ("thres(0, gr = z)", "one positive whole number"),
])
def test_malformed_terms_are_refused(text, message):
    with pytest.raises(FormulaError, match=message):
        parse_formula(f"y | {text} ~ x", family="cumulative")


@pytest.mark.parametrize("family", ["sratio", "cratio", "acat", "hurdle_cumulative", "gaussian", "poisson"])
def test_other_families_are_refused(family):
    with pytest.raises(FormulaError, match=rf"'thres\(gr = z\)': grouped thresholds are offered for the cumulative family's logit "
                                           rf"link only; brms's thres\(\) in the {family} family is not verified here.*D158"):
        parse_formula("y | thres(gr = z) ~ x", family=family)


@pytest.mark.parametrize("link", ["probit", "cloglog"])
def test_other_links_are_refused(link):
    with pytest.raises(FormulaError, match=rf"'thres\(gr = z\)' with the cumulative family's {link} link is not offered.*D158"):
        model("y | thres(gr = z) ~ x", link=link)
    with pytest.raises(FormulaError, match=rf"{link} link is not offered"):
        ab.default_priors("y | thres(gr = z) ~ x", frame(), family="cumulative", link=link)
    assert ab.cumulative("y ~ x", frame(), priors(False), link=link).design.thresholds is None   # without the term: as before


@pytest.mark.parametrize("formula, message", [
    ("y | thres(gr = z) ~ mo(x)", r"next to the term 'mo\(x\)' is not offered"),
    ("y | thres(gr = z) ~ gp(x)", r"next to the Gaussian process 'gp\(x\)' is not offered"),
    ("y | thres(gr = z) ~ x + z", "'z' is the grouping column of 'thres\\(gr = z\\)' and cannot also be"),
    ("y | thres(gr = z) ~ x + (1 | z)", "cannot also be the response, a predictor, a grouping factor"),
    ("y | thres(gr = z) ~ x + offset(z)", "cannot also be"),
])
def test_terms_not_verified_with_grouped_thresholds_are_refused(formula, message):
    with pytest.raises(FormulaError, match=message + ".*D158" if "next to" in message else message):
        parse_formula(formula, family="cumulative", categorical={"z": "a"} if formula.endswith("+ z") else None)


def test_check_thres_names_every_construct_it_refuses():
    """The constructs the cumulative family refuses before the term is checked (autocorrelation, SAR, me(), weights(), a
    distributional formula, mm()) are refused by this check too, naming the construct."""
    spec = parse_formula("y | thres(gr = z) ~ x", family="cumulative")
    cases = [
        (dict(additions=spec.additions + (AdditionTerm("weights", column="e"),)), "the addition term 'weights\\(e\\)'"),
        (dict(autocor=__import__("autobayes.autocor", fromlist=["Autocor"]).Autocor("ar", "x", None, 1, 0)), "the autocorrelation term"),
        (dict(sar=SimpleSar()), "the SAR term 'sar\\(M\\)'"),
        (dict(dpars=(parse_formula("sigma ~ w", family="gaussian", _dpar=True),)), "the distributional formula 'sigma ~ w'"),
    ]
    for override, message in cases:
        fake = _Spec(spec, **override)
        with pytest.raises(FormulaError, match=message):
            gt.check_thres(fake, "f")


class SimpleSar:
    term = "sar(M)"


class _Spec:
    """``spec`` with some of its fields replaced (``check_thres`` reads them by name)."""

    def __init__(self, spec, **override):
        self._spec, self._over = spec, override

    def __getattr__(self, name):
        return self._over[name] if name in self._over else getattr(self._spec, name)

    def addition(self, kind):
        return next((a for a in self.additions if a.kind == kind), None)


# -- the levels and their numbers of thresholds -------------------------------------------------------------------------
def test_levels_and_numbers_of_thresholds():
    a = parse_formula("y | thres(gr = z) ~ x", frame(), family="cumulative").addition("thres")
    assert (a.levels, a.nthres) == (("a", "b", "c"), (3, 3, 2))       # each level's largest category less one (brms's rule)
    a3 = parse_formula("y | thres(3, gr = z) ~ x", frame(), family="cumulative").addition("thres")
    assert (a3.levels, a3.nthres) == (("a", "b", "c"), (3, 3, 3))
    # a level without category 1 keeps its largest less one (brms's standata, probed); a Categorical column reads as its labels
    df = frame(y=[2, 2, 4, 3, 2, 1, 4, 2, 3, 2, 3, 2], z=pd.Categorical(Z, categories=["c", "b", "a", "zz"]))
    a = parse_formula("y | thres(gr = z) ~ x", df, family="cumulative").addition("thres")
    assert (a.levels, a.nthres) == (("a", "b", "c"), (3, 3, 2))


@pytest.mark.parametrize("over, formula, message", [
    (dict(y=[1, 2, 4, 3, 2, 1, 4, 2, 3, 1, 1, 1]), "thres(gr = z)",
     r"the rows of level\(s\) \['c'\] of 'z' all take the first category '1'.*Could not extract the number of thresholds"),
    (dict(y=pd.Categorical(Y, categories=[1, 2, 3, 4, 5], ordered=True)), "thres(gr = z)",
     r"no level of 'z' reaches the response's categories \['5'\].*D158"),
    (dict(), "thres(2, gr = z)", r"2 thresholds leave no place for the response's category '4'.*smaller than required by the response"),
    (dict(), "thres(5, gr = z)", r"5 thresholds make 6 categories, more than the response's 4.*D158"),
    (dict(z=[1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3]), "thres(gr = z)", r"has dtype int64; brms refuses a numeric grouping column too"),
    (dict(z=[True] * 6 + [False] * 6), "thres(gr = z)", r"is boolean; brms reads it as a factor.*D158"),
    (dict(z=[1j] * 12), "thres(gr = z)", "complex-valued"),
])
def test_data_brms_or_this_slice_refuses(over, formula, message):
    with pytest.raises(FormulaError, match=message):
        parse_formula(f"y | {formula} ~ x", frame(**over), family="cumulative")


def test_threshold_groups():
    m = model()
    g = m.design.thresholds
    assert (g.term, g.column, g.levels, g.nthres) == ("thres(gr = z)", "z", ("a", "b", "c"), (3, 3, 2))
    assert g.index.tolist() == [0] * 5 + [1] * 4 + [2] * 3
    assert (g.starts, g.total, g.width) == ((0, 3, 6), 8, 3)
    assert g.labels == ("a[1]", "a[2]", "a[3]", "b[1]", "b[2]", "b[3]", "c[1]", "c[2]")
    assert g.jthres().tolist() == [[1, 3]] * 5 + [[4, 6]] * 4 + [[7, 8]] * 3        # brms's Jthres of these rows
    assert g.jthres(np.array([2, 0])).tolist() == [[7, 8], [1, 3]]
    assert g.level_index(pd.Series(["c", "a", "b"])).tolist() == [2, 0, 1]
    with pytest.raises(FormulaError, match=r"newdata column 'z' has the level\(s\) \['d'\].*fitted levels \['a', 'b', 'c'\].*D159"):
        g.level_index(pd.Series(["a", "d"]), where="newdata column")
    tau = np.arange(16.0).reshape(2, 8)
    assert {k: v.tolist() for k, v in g.blocks(tau[0]).items()} == {"a": [0, 1, 2], "b": [3, 4, 5], "c": [6, 7]}
    rows = g.rows(tau, np.array([2, 0]))
    assert rows.shape == (2, 2, 3) and rows[1].tolist() == [[14, 15, 15], [8, 9, 10]]   # c's block padded with its last threshold


# -- the transform ------------------------------------------------------------------------------------------------------
def test_block_ordered_transform_is_stan_s_ordered_transform_within_each_block():
    tr = gt.BlockOrdered((3, 1, 2))
    u = np.array([0.3, -0.5, 1.2, 2.0, -1.0, 0.4])
    v = pt.dvector("v")
    back = pytensor.function([v], tr.backward(v))(u)
    expected = np.r_[np.cumsum([0.3, np.exp(-0.5), np.exp(1.2)]), 2.0, np.cumsum([-1.0, np.exp(0.4)])]
    assert np.allclose(back, expected, rtol=0, atol=1e-15)
    assert np.allclose(pytensor.function([v], tr.forward(v))(back), u, rtol=0, atol=1e-14)
    assert np.isclose(pytensor.function([v], tr.log_jac_det(v))(u), -0.5 + 1.2 + 0.4, rtol=0, atol=1e-15)
    batch = pt.dmatrix("b")                                     # a batch of points, as PyMC hands draws
    out = pytensor.function([batch], tr.backward(batch))(np.stack([u, u + 1]))
    assert np.allclose(out[0], expected) and out.shape == (2, 6)
    big = u.copy()
    big[0] = 800.0                                              # a first entry whose exponential overflows: never taken
    gradient = pytensor.function([v], pt.grad(tr.backward(v).sum(), v))(big)
    assert np.isfinite(gradient).all()


# -- the row density: the 50-digit reference, on both paths ---------------------------------------------------------------
ETAS = [-1000.0, -40.0, -0.5, 0.0, 3.0, 40.0, 1000.0]
WIDE = np.array([-1.2, -1.2 + 1e-9, 0.3])


def _rows(mode, y, eta, thr, k):
    yv, ev, tv, kv = pt.lvector("y"), pt.dvector("eta"), pt.dmatrix("tau"), pt.lvector("k")
    rows = gt.grouped_ordered_logistic_logp(yv, ev, tv, kv)
    if mode == "pytensor":
        f = pytensor.function([yv, ev, tv, kv], [rows, pt.grad(rows.sum(), ev), pt.grad(rows.sum(), tv)])
        return [np.asarray(a, dtype=float) for a in f(y, eta, thr, k)]
    jax = pytest.importorskip("jax")
    from pymc.sampling.jax import get_jaxified_graph

    jf = get_jaxified_graph(inputs=[yv, ev, tv, kv], outputs=[rows])
    return [np.asarray(a, dtype=float) for a in (jf(y, eta, thr, k)[0], jax.grad(lambda e: jf(y, e, thr, k)[0].sum())(eta),
                                                 jax.grad(lambda t: jf(y, eta, t, k)[0].sum())(thr))]


@pytest.mark.parametrize("mode", ["pytensor", "jax"])
def test_row_density_and_gradient_against_the_50_digit_reference(mode):
    """Rows with three thresholds (categories 1..4) and with two (categories 1..3, the third threshold a padding entry no row
    reads), at linear predictors from -1000 to 1000 and a threshold gap of 1e-9."""
    cases = [(y, e, k) for k in (2, 3) for y in range(1, k + 2) for e in ETAS]
    y = np.array([c[0] for c in cases])
    eta = np.array([c[1] for c in cases])
    k = np.array([c[2] for c in cases])
    thr = np.tile(WIDE, (len(cases), 1))
    value, d_eta, d_tau = _rows(mode, y, eta, thr, k)
    for i, (yi, ei, ki) in enumerate(cases):
        v, de, dt = exact_ordinal_row(int(yi), ei, WIDE[:ki])
        assert np.isfinite(value[i]) and np.isfinite(d_eta[i]) and np.isfinite(d_tau[i]).all()
        assert abs(value[i] - v) <= 1e-12 * max(1.0, abs(v)), (yi, ei, ki)
        assert abs(d_eta[i] - de) <= 1e-7 * max(1.0, abs(de)), (yi, ei, ki)
        assert np.allclose(d_tau[i][:ki], dt, rtol=1e-7, atol=1e-12) and (d_tau[i][ki:] == 0).all(), (yi, ei, ki)


def test_row_density_is_minus_infinity_outside_the_row_s_categories():
    value = pytensor.function([], gt.grouped_ordered_logistic_logp(np.array([0, 4, 3]), np.zeros(3), np.tile(WIDE, (3, 1)),
                                                                 np.array([3, 2, 2])))()
    assert value[0] == -np.inf and value[1] == -np.inf and np.isfinite(value[2])


def test_probabilities_are_zero_above_a_row_s_categories():
    eta = np.array([[0.3, -1.0], [2.0, 0.5]])
    thr = np.array([[-1.0, 0.0, 1.0], [-0.5, 0.8, 9.9]])                      # the second row's third entry is padding
    p = gt.grouped_cumulative_probabilities(eta, thr, np.array([3, 2]))
    assert p.shape == (2, 2, 4) and (p[:, 1, 3] == 0).all() and np.allclose(p.sum(-1), 1.0, rtol=0, atol=1e-15)
    for d in range(2):
        cdf3 = np.r_[0.0, expit(thr[0] - eta[d, 0]), 1.0]
        cdf2 = np.r_[0.0, expit(thr[1, :2] - eta[d, 1]), 1.0]
        assert np.allclose(p[d, 0], np.diff(cdf3), rtol=1e-13) and np.allclose(p[d, 1, :3], np.diff(cdf2), rtol=1e-13)
    assert gt.grouped_cumulative_probabilities(eta, thr, np.array([3, 2]), categories=5).shape == (2, 2, 5)
    draws = gt.grouped_ordered_logistic_random(np.zeros(200), np.tile(thr[1], (200, 1)), np.full(200, 2), rng=np.random.default_rng(1))
    assert set(np.unique(draws)) <= {1, 2, 3}


# -- the model ----------------------------------------------------------------------------------------------------------
def _point(m, tau=None, b=0.4, sd=0.7, zg=(0.5, -1.0, 0.3)):
    kw = dict(intercept=TAU if tau is None else tau, b=np.array([b]))
    if m.spec.groups:
        kw.update(sd=sd, z=np.array(zg))
    return kw


def by_hand(m, tau, b, sd=None, zg=None, jacobian=True):
    """The model's log density from the mathematics: each row's log P(y) from its level's thresholds, the priors (normal(0, 3)
    on every threshold, normal(0, 1.5) on b, half-normal(1) on sd, standard normal z) and the Jacobians (the log gaps of each
    level's thresholds, log sd)."""
    df = frame()
    eta = b * df.x.to_numpy()
    if sd is not None:
        eta = eta + sd * np.asarray(zg)[df.g.map({"g1": 0, "g2": 1, "g3": 2}).to_numpy()]
    total = 0.0
    for yi, zi, ei in zip(df.y, df.z, eta):
        cdf = np.r_[0.0, expit(tau[zi] - ei), 1.0]
        total += np.log(cdf[yi] - cdf[yi - 1])
    total += sum(norm.logpdf(t, 0, 3).sum() for t in tau.values()) + norm.logpdf(b, 0, 1.5)
    if sd is not None:
        total += halfnorm.logpdf(sd, scale=1) + norm.logpdf(zg).sum()
    if jacobian:
        total += sum(np.log(np.diff(t)).sum() for t in tau.values()) + (np.log(sd) if sd is not None else 0.0)
    return total


def test_density_and_gradient_by_hand_on_both_paths():
    from pymc.sampling.jax import get_jaxified_logp

    jax = pytest.importorskip("jax")
    m = model()
    assert not m.design.centered and m.effective_center_predictors is False and np.all(m.design.x_means == 0)
    vp = m.value_point(**_point(m))
    pm_model = m.pymc_model
    assert [v.name for v in pm_model.value_vars] == ["Intercept_ordered__", "x", "1|g_sigma_log__", "1|g_offset"]
    assert np.isclose(float(pm_model.compile_logp()(vp)), by_hand(m, TAU, 0.4, 0.7, [0.5, -1.0, 0.3]), rtol=1e-12)
    assert np.isclose(float(pm_model.compile_logp(jacobian=False)(vp)), by_hand(m, TAU, 0.4, 0.7, [0.5, -1.0, 0.3], False), rtol=1e-12)
    names = [v.name for v in pm_model.value_vars]
    jf = get_jaxified_logp(pm_model)
    assert np.isclose(float(jf([vp[n] for n in names])), float(pm_model.compile_logp()(vp)), rtol=1e-13)
    grad = np.asarray(pm_model.compile_dlogp()(vp))
    jgrad = np.concatenate([np.ravel(g) for g in jax.grad(lambda *v: jf(list(v)), argnums=tuple(range(len(names))))(*[vp[n] for n in names])])
    assert np.allclose(grad, jgrad, rtol=1e-11, atol=1e-12)
    # against a central difference of the by-hand density in the thresholds' unconstrained coordinates
    u = vp["Intercept_ordered__"]
    for j in range(u.size):
        def f(step):
            uu = u.copy()
            uu[j] += step
            tau = {lvl: np.cumsum(np.r_[uu[s], np.exp(uu[s + 1:s + m_])]) for lvl, s, m_ in
                   zip(m.design.thresholds.levels, m.design.thresholds.starts, m.design.thresholds.nthres)}
            return by_hand(m, tau, 0.4, 0.7, [0.5, -1.0, 0.3])
        assert np.isclose(grad[j], (f(1e-6) - f(-1e-6)) / 2e-6, rtol=1e-6, atol=1e-7)
    # at 300 rows under JAX: finite, the same density
    big = pd.concat([frame()] * 25, ignore_index=True)
    mb = model(df=big)
    vb = mb.value_point(**_point(mb))
    jb = get_jaxified_logp(mb.pymc_model)
    assert np.isclose(float(jb([vb[v.name] for v in mb.pymc_model.value_vars])), float(mb.pymc_model.compile_logp()(vb)), rtol=1e-12)


def test_graph_thresholds_are_the_sampled_ones():
    m = model()
    pmm = m.pymc_model
    assert tuple(pmm.coords[m.names["threshold_dim"]]) == m.design.thresholds.labels
    fn = pmm.compile_fn(pmm.replace_rvs_by_values([pmm["b_Intercept"], pmm["Intercept"]]), inputs=pmm.value_vars, point_fn=True,
                        on_unused_input="ignore")
    tau, raw = fn(m.value_point(**_point(m)))
    merged = np.concatenate([TAU[k] for k in "abc"])
    assert np.allclose(tau, merged, rtol=0, atol=1e-15) and np.allclose(raw, merged, rtol=0, atol=1e-15)
    start = m.pymc_model.initial_point()["Intercept_ordered__"]               # the support point: ordered within each level, at 0
    back = pytensor.function([], gt.BlockOrdered(m.design.thresholds.nthres).backward(pt.as_tensor(start)))()
    assert np.allclose(back, [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -0.5, 0.5], rtol=0, atol=1e-12)


def test_value_point_refusals():
    m = model()
    with pytest.raises(ValueError, match=r"intercept must map each level of 'z' \['a', 'b', 'c'\] to its thresholds"):
        m.value_point(**{**_point(m), "intercept": np.r_[TAU["a"], TAU["b"], TAU["c"]]})
    with pytest.raises(ValueError, match="intercept must map each level"):
        m.value_point(**{**_point(m), "intercept": {"a": TAU["a"], "b": TAU["b"]}})
    with pytest.raises(ValueError, match=r"intercept\['c'\] must be the 2 thresholds of level 'c', finite and strictly increasing"):
        m.value_point(**{**_point(m), "intercept": {**TAU, "c": np.array([0.0, 1.0, 2.0])}})
    with pytest.raises(ValueError, match=r"intercept\['a'\] must be the 3 thresholds"):
        m.value_point(**{**_point(m), "intercept": {**TAU, "a": np.array([0.0, 1.0, 0.5])}})
    # the ungrouped family keeps its vector of thresholds
    plain = ab.cumulative("y ~ x", frame(), priors(False))
    assert plain.design.thresholds is None
    assert np.allclose(plain.value_point(intercept=[-1.0, 0.0, 1.0], b=np.array([0.2]))["Intercept_ordered__"], [-1.0, 0.0, 0.0])


def test_combinations_offered_enter_mu_uncentred():
    """A categorical predictor, an interaction, a transform, an offset and a correlated slope enter mu as for the ungrouped
    family, uncentred: the graph's mu is the raw design times b plus the offset and the group effects."""
    df = frame()
    m = ab.cumulative("y | thres(3, gr = z) ~ x * h + log(e) + offset(w) + (1 + x | g)", df,
                      ab.Priors(intercept=ab.Normal(0, 3), b=ab.Normal(0, 1.5), sd={"g": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(1)}},
                                cor={"g": ab.LKJ(2)}),
                      categorical={"h": "p"})
    assert not m.design.centered and m.design.thresholds.nthres == (3, 3, 3)
    pmm = m.pymc_model
    b = np.array([0.4, -0.3, 0.2, 0.5])
    vp = m.value_point(intercept={"a": TAU["a"], "b": TAU["b"], "c": np.r_[TAU["c"], 2.5]}, b=b,
                       sd={"g": {"Intercept": 0.6, "x": 0.3}}, z={"g": {"Intercept": np.array([0.2, -0.4, 0.1]), "x": np.array([1.0, 0.0, -0.5])}},
                       L={"g": np.array([[1.0, 0.0], [0.6, 0.8]])})
    fn = pmm.compile_fn(pmm.replace_rvs_by_values([pmm["mu"], pmm["1|g"], pmm["x|g"]]), inputs=pmm.value_vars, point_fn=True,
                        on_unused_input="ignore")
    mu, u, v = (np.asarray(a) for a in fn(vp))
    gi = df.g.map({"g1": 0, "g2": 1, "g3": 2}).to_numpy()
    expected = m.design.X_raw @ b + df.w.to_numpy() + u[gi] + df.x.to_numpy() * v[gi]
    assert np.allclose(mu, expected, rtol=1e-13) and np.allclose(m.design.X_raw[:, m.spec.columns.index("log(e)")], np.log(df.e))


# -- predictions, log_lik, bayes_R2 ------------------------------------------------------------------------------------
def _idata(m, draws=3):
    rng = np.random.default_rng(4)
    taus = [{k: np.sort(v + rng.normal(0, 0.1, v.size)) for k, v in TAU.items()} for _ in range(draws)]
    merged = np.stack([np.concatenate([t[k] for k in "abc"]) for t in taus])
    post = {"Intercept": (("chain", "draw", "Intercept_dim"), merged[None]), "b_Intercept": (("chain", "draw", "Intercept_dim"), merged[None]),
            "x": (("chain", "draw"), rng.normal(0.4, 0.1, (1, draws))), "1|g_sigma": (("chain", "draw"), np.full((1, draws), 0.7)),
            "1|g": (("chain", "draw", m.group_names["g"]["coordinate"]), rng.normal(0, 0.5, (1, draws, 3)))}
    coords = {"chain": [0], "draw": np.arange(draws), "Intercept_dim": list(m.design.thresholds.labels),
              m.group_names["g"]["coordinate"]: ["g1", "g2", "g3"]}
    return az.InferenceData(posterior=xr.Dataset(post, coords=coords)), taus


def _probabilities(taus, idata, df):
    post = idata.posterior
    out = np.zeros((len(taus), len(df), 4))
    for d, t in enumerate(taus):
        eta = float(post["x"][0, d]) * df.x.to_numpy() + post["1|g"].values[0, d][df.g.map({"g1": 0, "g2": 1, "g3": 2}).to_numpy()]
        for i, (zi, ei) in enumerate(zip(df.z, eta)):
            p = np.diff(np.r_[0.0, expit(t[zi] - ei), 1.0])
            out[d, i, :p.size] = p
    return out


def test_predictions_by_hand():
    m = model()
    idata, taus = _idata(m)
    mean = m.predict(idata, kind="mean")
    expected = _probabilities(taus, idata, frame())
    assert mean["p"].dims == ("chain", "draw", "__obs__", "category") and np.allclose(mean["p"].values[0], expected, rtol=1e-12)
    assert (mean["p"].values[0][:, 9:, 3] == 0).all()                          # level c never reaches category 4: exactly 0
    nd = frame().iloc[[11, 0, 6]].reset_index(drop=True)
    new = m.predict(idata, newdata=nd, kind="mean")["p"].values[0]
    assert np.allclose(new, expected[:, [11, 0, 6]], rtol=1e-12) and (new[:, 0, 3] == 0).all()
    lin = m.predict(idata, newdata=nd, kind="linear")["eta"].values[0]
    assert lin.shape == (3, 3)
    y = m.predict(idata, kind="response", seed=3)["y"].values[0]
    assert y.shape == (3, 12) and y.min() >= 1 and y[:, 9:].max() <= 3
    with pytest.raises(FormulaError, match=r"newdata column 'z' has the level\(s\) \['d'\].*D159"):
        m.predict(idata, newdata=nd.assign(z=["a", "b", "d"]), kind="mean")
    with pytest.raises(FormulaError, match=r"newdata lacks columns \['z'\]"):
        m.predict(idata, newdata=nd.drop(columns="z"), kind="linear")


def test_expected_response_needs_each_row_s_level():
    m = model()
    idata, _ = _idata(m)
    with pytest.raises(ValueError, match="5 rows but no level of 'z' for them"):
        gt.expected_response(m, idata.posterior, np.zeros((1, 3, 5)))
    p = gt.expected_response(m, idata.posterior, np.zeros((1, 3, 12)))         # the fitted rows
    assert p.shape == (1, 3, 12, 4)


def test_log_lik_and_bayes_R2_by_hand():
    m = model()
    idata, taus = _idata(m)
    p = _probabilities(taus, idata, frame())
    y = np.array(Y)
    expected = np.log(np.take_along_axis(p, (y - 1)[None, :, None], axis=-1)[..., 0])
    ll = ab.log_lik(m, idata)
    assert ll.shape == (1, 3, 12) and np.allclose(ll.values[0], expected, rtol=1e-12)
    yhat = (p * np.arange(1, 5)).sum(-1)
    r2 = np.var(yhat, axis=-1, ddof=1) / (np.var(yhat, axis=-1, ddof=1) + np.var(y - yhat, axis=-1, ddof=1))
    assert np.allclose(ab.bayes_R2(m, idata).values[0], r2, rtol=1e-12)


def test_prior_predictive_thresholds_are_ordered_within_each_level():
    m = model()
    out = m.prior_predictive(draws=50, seed=5)
    tau = out.prior["Intercept"].values[0]
    g = m.design.thresholds
    assert tau.shape == (50, 8) and all((np.diff(tau[:, s:s + k], axis=-1) > 0).all() for s, k in zip(g.starts, g.nthres))
    y = out.prior_predictive["y"].values[0]
    assert y.min() >= 1 and y.max() <= 4 and y[:, 9:].max() <= 3               # level c's rows take categories 1..3 only


def test_thresholds_under_a_prior_without_forward_draws():
    import pymc as pm

    groups = model().design.thresholds
    with pm.Model():
        rv = gt.grouped_ordered_thresholds("t", "Cauchy", {"alpha": 0.0, "beta": 1.0}, groups, "t_dim")
        with pytest.raises(NotImplementedError, match="a Cauchy prior on the thresholds has no forward draws"):
            pm.draw(rv)


# -- priors, describe(), identity ---------------------------------------------------------------------------------------
def test_prior_table_lists_each_level_and_threshold_as_brms_does():
    m = model()
    rows = [(r["prior"], r["class"], r["coef"], r["group"]) for r in m.prior_table()]
    assert rows[:9] == [("normal(0, 3)", "Intercept", "", ""), ("", "Intercept", "", "a"), ("", "Intercept", "1", "a"),
                        ("", "Intercept", "2", "a"), ("", "Intercept", "3", "a"), ("", "Intercept", "", "b"), ("", "Intercept", "1", "b"),
                        ("", "Intercept", "2", "b"), ("", "Intercept", "3", "b")]
    assert rows[9:12] == [("", "Intercept", "", "c"), ("", "Intercept", "1", "c"), ("", "Intercept", "2", "c")]
    with pytest.raises(PriorError, match=r"the prior table of 'thres\(gr = z\)' needs its levels"):
        priors().table(parse_formula("y | thres(gr = z) ~ x + (1 | g)", family="cumulative"), m.family_spec, categories=m.categories)
    # the ungrouped family keeps one row per threshold under the class row
    plain = ab.cumulative("y ~ x", frame(), priors(False)).prior_table()
    assert [(r["class"], r["coef"], r["group"]) for r in plain[:4]] == [("Intercept", "", ""), ("Intercept", "1", ""),
                                                                       ("Intercept", "2", ""), ("Intercept", "3", "")]


def test_default_prior_table_brms_printed_with_eleven_thresholds():
    """brms 2.23's default_prior(y | thres(gr = z) ~ x, family = cumulative()) on a level with eleven thresholds (probed): its
    coefficient rows sort as text, 1, 10, 11, 2, ..."""
    df = pd.DataFrame({"y": [*range(1, 13), 1, 2, 3], "z": ["a"] * 12 + ["b"] * 3, "x": np.linspace(-1, 1, 15)})
    pri = ab.default_priors("y | thres(gr = z) ~ x", df, family="cumulative")
    m = ab.cumulative("y | thres(gr = z) ~ x", df, pri)
    rows = [(r["prior"], r["class"], r["coef"], r["group"]) for r in m.prior_table()]
    brms = [("student_t(3, 0, 2.5)", "Intercept", "", ""), ("", "Intercept", "", "a")] + \
        [("", "Intercept", c, "a") for c in ("1", "10", "11", "2", "3", "4", "5", "6", "7", "8", "9")] + \
        [("", "Intercept", "", "b"), ("", "Intercept", "1", "b"), ("", "Intercept", "2", "b"), ("", "b", "", ""), ("", "b", "x", "")]
    assert rows == brms


def test_describe_and_identity():
    m = model()
    text = m.describe()
    for needle in ("thresholds: thres(gr = z): one ordered set of thresholds per level of z (a 5 row(s), 3 threshold(s), b 4 row(s), 3 "
                   "threshold(s), c 3 row(s), 2 threshold(s))",
                   "P(y_i <= k) = expit(b_Intercept[z_i, k] - mu_i), k = 1..nthres[z_i]",
                   "Predictors are not centred: brms does not centre the design when the thresholds are grouped",
                   "Intercept[l, k] ~ Normal(mu=0, sigma=3), for each of the 8 thresholds of the 3 levels of z",
                   "y_i ~ OrderedLogistic(mu_i, b_Intercept[z_i])",
                   "Intercept_ordered__      Intercept[a, b, c], the thresholds of each level of z (8 entries",
                   "b_Intercept = Intercept, the thresholds of every level (not centred)"):
        assert needle in text, needle
    assert "truncation" not in text and "trunc()" not in text
    plain_text = ab.cumulative("y ~ x", frame(), priors(False)).describe()      # the ungrouped family's lines are unchanged
    assert "Intercept[k] ~ Normal(mu=0, sigma=3), for each of the 3 thresholds k (brms's class Intercept)" in plain_text
    assert "thres(" not in plain_text and "Intercept[1..3], the thresholds of the centred design" in plain_text
    same = model()
    other = model("y | thres(3, gr = z) ~ x + (1 | g)")
    plain = ab.cumulative("y ~ x + (1 | g)", frame(), priors(), center_predictors=False)
    ids = {m.target_identity["sha256"], other.target_identity["sha256"], plain.target_identity["sha256"]}
    assert same.target_identity == m.target_identity and len(ids) == 3
    shuffled = model(df=frame(z=list("bbbbbaaaaccc")))                  # the same data with the levels' rows changed
    assert shuffled.target_identity != m.target_identity


def test_the_family_registry_offers_the_term_for_the_cumulative_logit_family_only():
    from autobayes.families import FAMILIES

    assert {name for name, spec in FAMILIES.items() if "thres" in spec.addition_terms} == {"cumulative"}
    assert {name for name, spec in FAMILIES.items() if spec.grouped_thresholds is not None} == {"cumulative"}
    grouped = FAMILIES["cumulative"].grouped_thresholds()
    assert grouped.grouped_thresholds is None and grouped.name == "cumulative" and grouped.thresholds and not grouped.monotonic
    m = model()
    assert m.family_spec is grouped and m.family == "cumulative"


def test_new_groups_policies_and_no_group_effects():
    """A new level of the grouping factor g takes either unseen-group policy; the thresholds stay the row's level of z's."""
    m = model()
    idata, taus = _idata(m)
    nd = frame().iloc[[0, 9]].reset_index(drop=True)
    nd["g"] = ["new", "g1"]
    for policy in ("population", "resample_observed"):
        p = m.predict(idata, newdata=nd, kind="mean", new_groups=policy, seed=2)["p"].values[0]
        assert np.allclose(p.sum(-1), 1.0) and (p[:, 1, 3] == 0).all() and (p[:, 0] > 0).all()
    plain = m.predict(idata, kind="mean", include_group_effects=False)["p"].values[0]
    post = idata.posterior
    for d, t in enumerate(taus):
        eta = float(post["x"][0, d]) * frame().x.to_numpy()
        assert np.allclose(plain[d, 0], np.diff(np.r_[0.0, expit(t["a"] - eta[0]), 1.0]), rtol=1e-12)


def test_summaries_refuse_the_model_as_they_refuse_the_cumulative_family():
    m = model()
    idata, _ = _idata(m)
    with pytest.raises(ValueError, match="hypothesis\\(\\) of a model with the cumulative family's thresholds is not offered"):
        ab.hypothesis(m, idata, "x > 0")
    with pytest.raises(ValueError, match="conditional_effects\\(\\) of a model with the cumulative family's thresholds"):
        ab.conditional_effects(m, idata, "x")
    with pytest.raises(ValueError, match="thresholds"):
        ab.prior_draws(m, draws=10, seed=1)
