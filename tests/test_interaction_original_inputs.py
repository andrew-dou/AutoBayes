"""M3a-2 repair record: checks from the ORIGINAL supplied inputs and explicit mathematics (adapted from the
coordinator's independent file, artifacts/2026-09-10-autobayes-m3a2-independent/test_iqss_m3a2_independent.py),
multiple-interaction compatibility across families, and the conditional categorical contrast."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import arviz as az
from scipy import stats
from scipy.special import expit, gammaln

import autobayes as ab


def inputs(unsigned):
    levels = [0, 2**63, 2**64 - 1] if unsigned else [2**53, 2**53 + 1, 2**53 + 2]
    ix = np.array([0, 1, 2, 0, 2, 1, 0, 1, 2, 0, 1, 0])
    x = np.array([-2., .2, 1., -.3, 2., 1.3, .4, .8, 3., 1.2, -.7, .9])
    df = pd.DataFrame({"x": x, "z": pd.Series([levels[i] for i in ix], dtype="UInt64" if unsigned else "int64"),
                       "g": ["g0", "g1"] * 6, "y": np.linspace(-.7, 1.3, 12), "y01": np.arange(12) % 2,
                       "k": np.array([1, 2, 0, 1, 2, 3, 0, 1, 2, 1, 0, 2]), "n": np.array([3, 4, 0, 3, 5, 6, 2, 4, 5, 3, 2, 4])})
    return df, levels


LHS = {"gaussian": "y", "bernoulli": "y01", "binomial": "k | trials(n)"}


@pytest.mark.parametrize("family", ["gaussian", "bernoulli", "binomial"])
@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("unsigned", [False, True])
def test_original_label_design_density_gradients_and_predictions(family, center, unsigned):
    """Every expected value below is computed from the ORIGINAL exact scalar inputs, never from model.data."""
    jax = pytest.importorskip("jax")
    from pymc.sampling.jax import get_jaxified_logp

    original, levels = inputs(unsigned)
    ref = levels[1]; other = [levels[0], levels[2]]
    bmu = np.array([.2, -.4, .8, .25, -.6]); bscale = np.array([1.3, .7, 1.1, .9, 1.4])
    priors = ab.Priors(intercept=ab.Normal(.1, 1.5),
                       b={"x": ab.Normal(.2, 1.3), "z": {other[0]: ab.Normal(-.4, .7), other[1]: ab.Normal(.8, 1.1)},
                          "x:z": {other[0]: ab.Normal(.25, .9), other[1]: ab.Normal(-.6, 1.4)}},
                       sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1) if family == "gaussian" else None)
    model = getattr(ab, family)(LHS[family] + " ~ x + z + x:z + (1 | g)", original, priors, categorical={"z": ref}, center_predictors=center)
    data = original.loc[original.n > 0].copy() if family == "binomial" else original.copy()   # fitted rows chosen here
    xv = data.x.to_numpy(); labels = [int(v) for v in data.z.tolist()]
    inds = np.array([[int(v == l) for l in other] for v in labels], dtype=float)
    X = np.column_stack([xv, inds, xv[:, None] * inds]); means = X.mean(0) if center else np.zeros(5)
    assert np.array_equal(model.design.X_raw, X)
    assert np.allclose(model.design.X_model, X - means)
    assert model.original_labels["z"].tolist() == [str(v) for v in labels]
    gi = np.array([0 if v == "g0" else 1 for v in data.g])
    val = np.array([.3, .4, -.2, .7, .5, -.3, np.log(.6), -.3, .4] + ([np.log(.8)] if family == "gaussian" else []))

    def point(v):
        return model.value_point(intercept=v[0], b=v[1:6], sd=np.exp(v[6]), z=v[7:9], sigma=np.exp(v[9]) if family == "gaussian" else None)

    def reference(v, jac=True):
        eta = v[0] + (X - means) @ v[1:6] + np.exp(v[6]) * v[7:9][gi]
        lp = stats.norm.logpdf(v[0], .1, 1.5) + stats.norm.logpdf(v[1:6], bmu, bscale).sum()
        lp += stats.norm.logpdf(v[7:9]).sum() + stats.halfnorm.logpdf(np.exp(v[6]))
        if jac:
            lp += v[6]
        if family == "gaussian":
            lp += stats.norm.logpdf(data.y.to_numpy(), eta, np.exp(v[9])).sum() + stats.halfnorm.logpdf(np.exp(v[9]))
            if jac:
                lp += v[9]
        elif family == "bernoulli":
            y = data.y01.to_numpy(); lp += (y * eta - np.logaddexp(0, eta)).sum()
        else:
            k = data.k.to_numpy(); n = data.n.to_numpy()
            lp += (gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1) - k * np.logaddexp(0, -eta) - (n - k) * np.logaddexp(0, eta)).sum()
        return float(lp)

    p = point(val)
    assert abs(float(model.pymc_model.compile_logp()(p)) - reference(val)) < 1e-8
    assert abs(float(model.pymc_model.compile_logp(jacobian=False)(p)) - reference(val, False)) < 1e-8
    names = [v.name for v in model.pymc_model.value_vars]
    jf = get_jaxified_logp(model.pymc_model)
    assert abs(float(jf([p[n] for n in names])) - reference(val)) < 1e-8
    grad = jax.grad(lambda *args: jf(list(args)), argnums=tuple(range(len(names))))(*[p[n] for n in names])
    gmap = {n: np.asarray(v) for n, v in zip(names, grad)}
    gvec = np.r_[gmap["Intercept"], gmap["x"], gmap["z"], gmap["x:z"], gmap["1|g_sigma_log__"], gmap["1|g_offset"], gmap["sigma_log__"] if family == "gaussian" else []]
    h = 1e-5; fd = np.array([(reference(val + h * e) - reference(val - h * e)) / (2 * h) for e in np.eye(len(val))])
    assert np.allclose(gvec, fd, atol=1e-5, rtol=1e-6)
    # synthetic posterior draws: the prediction mapping without another sampling run
    var = {"Intercept": (("chain", "draw"), np.full((1, 3), .2)), "x": (("chain", "draw"), np.full((1, 3), .4)),
           "z": (("chain", "draw", "z_dim"), np.tile([-.2, .7], (1, 3, 1))), "x:z": (("chain", "draw", "x:z_dim"), np.tile([.5, -.3], (1, 3, 1))),
           "1|g_sigma": (("chain", "draw"), np.full((1, 3), .6)), "1|g": (("chain", "draw", "g__factor_dim"), np.tile([-.1, .3], (1, 3, 1)))}
    if family == "gaussian":
        var["sigma"] = (("chain", "draw"), np.full((1, 3), .8))
    coords = {"chain": [0], "draw": [0, 1, 2], "z_dim": [str(v) for v in other], "x:z_dim": [str(v) for v in other], "g__factor_dim": ["g0", "g1"]}
    post = az.InferenceData(posterior=xr.Dataset(var, coords=coords))
    new = original.iloc[[8, 0, 1, 4]].copy(); new["n"] = [0, 4, 7, 3]
    expected = []
    for xv_, lvl, group in zip(new.x, new.z, new.g):
        k = other.index(int(lvl)) if int(lvl) in other else None
        expected.append(.2 + .4 * xv_ + ([-.2, .7][k] + [.5, -.3][k] * xv_ if k is not None else 0) + (-.1 if group == "g0" else .3))
    expected = np.array(expected); prob = expected if family == "gaussian" else expit(expected)
    out = model.predict(post, newdata=new, kind="mean")
    assert np.allclose(out["mu" if family == "gaussian" else "p"], prob[None, None, :], atol=1e-10)
    if family == "binomial":
        assert np.allclose(out["expected"], (prob * new.n.to_numpy())[None, None, :])


@pytest.mark.parametrize("family", ["gaussian", "bernoulli", "binomial"])
def test_multiple_interactions_are_accepted(family):
    data, _ = inputs(False); data["z"] = ["a", "b", "c"] * 4; data["w"] = np.linspace(-1, 2, len(data)); data["v"] = ["p", "q"] * 6
    priors = ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1) if family == "gaussian" else None)
    for rhs, cat in [("x + w + z + x:z + w:z", {"z": "a"}),                       # shared categorical side
                     ("x + z + v + x:z + x:v", {"z": "a", "v": "p"}),               # shared numeric side
                     ("x + w + z + v + x:z + w:v", {"z": "a", "v": "p"}),           # disjoint sides
                     ("x*z + w*v", {"z": "a", "v": "p"})]:                          # shorthand twice
        model = getattr(ab, family)(LHS[family] + f" ~ {rhs} + (1 | g)", data, priors, categorical=cat)
        assert len(model.spec.interactions) == 2  # The original refused cases are now explicit M3a-3 coverage.
    m = getattr(ab, family)(LHS[family] + " ~ x + w + z + x:z + (1 | g)", data, priors, categorical={"z": "a"})   # one interaction plus another numeric term
    assert m.spec.terms == ("x", "w", "z", "x:z") and m.spec.interactions == (("x", "z"),)


def test_conditional_contrast_is_stated_and_numerically_right():
    """b_z[l] is the contrast at x = 0; at x = 2 with b_z = -0.2 and b_x:z = 0.5 the contrast is +0.8, not -0.2."""
    data, _ = inputs(False); data["z"] = ["a", "b", "c"] * 4
    priors = ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd=ab.HalfNormal(1))
    m = ab.bernoulli("y01 ~ x + z + x:z + (1 | g)", data, priors, categorical={"z": "a"})
    text = m.describe()
    assert "b_z[l] is the difference between level l and the reference 'a' on the log-odds scale at x = 0 (same group)" in text
    assert "at another value of x the difference is b_z[l] + x * b_x:z[l]" in text
    assert "at the same predictors and group; the reference has no coefficient" not in text
    var = {"Intercept": (("chain", "draw"), np.full((1, 2), .2)), "x": (("chain", "draw"), np.full((1, 2), .4)),
           "z": (("chain", "draw", "z_dim"), np.tile([-.2, .7], (1, 2, 1))), "x:z": (("chain", "draw", "x:z_dim"), np.tile([.5, -.3], (1, 2, 1))),
           "1|g_sigma": (("chain", "draw"), np.full((1, 2), .6)), "1|g": (("chain", "draw", "g__factor_dim"), np.tile([-.1, .3], (1, 2, 1)))}
    post = az.InferenceData(posterior=xr.Dataset(var, coords={"chain": [0], "draw": [0, 1], "z_dim": ["b", "c"], "x:z_dim": ["b", "c"], "g__factor_dim": ["g0", "g1"]}))
    new = pd.DataFrame({"x": [0.0, 0.0, 2.0, 2.0], "z": ["b", "a", "b", "a"], "g": ["g0"] * 4})
    eta = m.predict(post, newdata=new, kind="mean")["p"].to_numpy()
    logit = np.log(eta / (1 - eta))
    assert np.allclose(logit[..., 0] - logit[..., 1], -0.2, atol=1e-10)                    # contrast at x = 0 is b_z[b]
    assert np.allclose(logit[..., 2] - logit[..., 3], -0.2 + 2 * 0.5, atol=1e-10)          # at x = 2 it is b_z[b] + 2 b_x:z[b] = +0.8
    m2 = ab.gaussian("y ~ x + z + (1 | g)", data, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1)), categorical={"z": "a"})
    assert "at the same predictors and group; the reference has no coefficient" in m2.describe()   # constant contrast without an interaction
