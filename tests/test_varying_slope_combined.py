"""M3b-1 repair record (2026-09-12): value_point presence validation; the supported combination of a numeric
varying slope with categorical main effects and a numeric-by-categorical interaction, verified from the ORIGINAL
supplied rows and labels with explicit mathematics (adapted from the coordinator's probe,
artifacts/2026-09-12-autobayes-m3b1-review/probe.py); the analytic conditional-Gaussian posterior with both
group-effect blocks at fixed SDs against the graph's gradient and Hessian on both compilation paths."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import arviz as az
from scipy import stats
from scipy.special import expit

import autobayes as ab

LHS = {"gaussian": "y", "bernoulli": "binary", "binomial": "k | trials(n)"}


CROSSED_SITES = ["g3", "g1", "g2", "g2", "g3", "g1", "g1", "g2", "g3", "g3", "g1", "g2", "g2", "g3", "g1", "g1", "g2", "g3"]


def original_frame(family, crossed=False):
    """``crossed=True`` uses the coordinator's site assignment (artifacts/2026-09-12-autobayes-pr8-repair-independent/
    test_crossed.py) so that category and site are not aliased."""
    df = pd.DataFrame({"dose": np.array([-.7, .2, 1.4, 2.5, -1.2, .8] * 3) + 2.3, "category": ["c", "a", "b"] * 6,
                       "site": CROSSED_SITES if crossed else ["g1", "g2", "g3"] * 6, "y": np.linspace(-.8, 2.1, 18),
                       "binary": [0, 1] * 9, "k": [0, 1, 2] * 6, "n": [3] * 18})
    if family == "binomial":
        df.loc[0, ["k", "n"]] = 0                                   # one zero-trial row the model must drop
    return df


def priors_for(family):
    return ab.Priors(intercept=ab.Normal(.4, 2.1), b=ab.Normal(.1, 1.3), sd={"Intercept": ab.HalfNormal(.9), "dose": ab.Exponential(1.7)},
                     sigma=ab.HalfNormal(1.2) if family == "gaussian" else None)


def hand_design(original):
    """Raw fixed design [dose, 1[a], 1[c], dose 1[a], dose 1[c]], one-hot Z over sorted sites, group index."""
    x = original.dose.to_numpy()
    dum = np.array([[float(v == l) for l in ["a", "c"]] for v in original.category])
    X = np.column_stack([x, dum, x[:, None] * dum])
    gi = np.array([["g1", "g2", "g3"].index(v) for v in original.site])
    return x, X, np.eye(3)[gi], gi


# -- value_point presence validation (all four combinations, both model forms) ---------------------------
def test_value_point_presence_validation():
    df = original_frame("gaussian")
    J = 3
    base = ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1))
    m0 = ab.gaussian("y ~ dose + (1 | site)", df, base)
    slope_priors = ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd={"Intercept": ab.HalfNormal(1), "dose": ab.HalfNormal(1)}, sigma=ab.HalfNormal(1))
    m1 = ab.gaussian("y ~ dose + (1 + dose || site)", df, slope_priors)
    common = dict(intercept=0.3, b=[0.2], sd=0.5, z=np.zeros(J), sigma=0.6)
    combos = {"neither": {}, "sd_only": {"sd_slope": 0.3}, "z_only": {"z_slope": np.zeros(J)}, "both": {"sd_slope": 0.3, "z_slope": np.zeros(J)}}
    # varying-intercept-only model: only "neither" is valid; every extra is refused by name
    pt = m0.value_point(**common)
    assert set(pt) == {"Intercept", "dose", "1|site_sigma_log__", "1|site_offset", "sigma_log__"} and not any("dose|" in k for k in pt)
    for name in ("sd_only", "z_only", "both"):
        with pytest.raises(ValueError, match="given, but this model has no varying slope"):
            m0.value_point(**common, **combos[name])
    with pytest.raises(ValueError, match="sd_slope given"):
        m0.value_point(**common, sd_slope=0.3)
    with pytest.raises(ValueError, match="z_slope given"):
        m0.value_point(**common, z_slope=np.zeros(J))
    # varying-slope model: only "both" is valid; every missing argument is refused by name
    pt1 = m1.value_point(**common, **combos["both"])
    assert set(pt1) == set(pt) | {"dose|site_sigma_log__", "dose|site_offset"} and float(pt1["dose|site_sigma_log__"]) == np.log(0.3)
    with pytest.raises(ValueError, match="sd_slope and z_slope missing"):
        m1.value_point(**common)
    with pytest.raises(ValueError, match="z_slope missing"):
        m1.value_point(**common, sd_slope=0.3)
    with pytest.raises(ValueError, match="sd_slope missing"):
        m1.value_point(**common, z_slope=np.zeros(J))
    with pytest.raises(ValueError, match="z_slope must have 3 entries"):
        m1.value_point(**common, sd_slope=0.3, z_slope=np.zeros(J + 1))
    with pytest.raises(ValueError, match="z must have 3 entries"):
        m1.value_point(intercept=0.3, b=[0.2], sd=0.5, z=np.zeros(J + 1), sigma=0.6, sd_slope=0.3, z_slope=np.zeros(J))
    # valid numerical targets are unchanged: the point evaluates and the slope offsets enter the linear predictor
    lp = m1.pymc_model.compile_logp()
    a = float(lp(m1.value_point(**common, sd_slope=0.3, z_slope=np.zeros(J))))
    b = float(lp(m1.value_point(**common, sd_slope=0.3, z_slope=np.ones(J))))
    assert np.isfinite(a) and np.isfinite(b) and a != b


# -- the supported combination, from original inputs, both paths --------------------------------------------
@pytest.mark.parametrize("family", ["gaussian", "bernoulli", "binomial"])
@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("crossed", [False, True])
def test_slope_with_categorical_and_interaction_from_original_inputs(family, center, crossed):
    jax = pytest.importorskip("jax")
    from pymc.sampling.jax import get_jaxified_logp

    df = original_frame(family, crossed)
    m = getattr(ab, family)(LHS[family] + " ~ dose*category + (1 + dose || site)", df, priors_for(family),
                            categorical={"category": "b"}, center_predictors=center)
    original = df[df.n > 0] if family == "binomial" else df                    # positive-trial rows chosen here
    x, X, Z, gi = hand_design(original)
    assert np.array_equal(m.design.X_raw, X) and np.array_equal(m.design.Z, Z) and np.array_equal(m.design.Z_slope, x[:, None] * Z)
    assert m.original_labels["category"].tolist() == list(original.category) and list(m.design.levels) == ["g1", "g2", "g3"]
    assert m.spec.columns == ("dose", "category[a]", "category[c]", "dose:category[a]", "dose:category[c]")
    Xc = X - X.mean(0) if center else X

    def reference(a, beta, sd, z, sd1, z1, sigma=None, jac=True):
        eta = a + Xc @ beta + sd * z[gi] + x * sd1 * z1[gi]
        lp = (stats.norm.logpdf(a, .4, 2.1) + stats.norm.logpdf(beta, .1, 1.3).sum() + stats.norm.logpdf(z).sum() + stats.norm.logpdf(z1).sum()
              + stats.halfnorm.logpdf(sd, scale=.9) + stats.expon.logpdf(sd1, scale=1 / 1.7) + (np.log(sd) + np.log(sd1) if jac else 0.0))
        if family == "gaussian":
            lp += stats.halfnorm.logpdf(sigma, scale=1.2) + (np.log(sigma) if jac else 0.0) + stats.norm.logpdf(original.y, eta, sigma).sum()
        elif family == "bernoulli":
            lp += stats.bernoulli.logpmf(original.binary, expit(eta)).sum()
        else:
            lp += stats.binom.logpmf(original.k, original.n, expit(eta)).sum()
        return float(lp)

    names = [v.name for v in m.pymc_model.value_vars]
    logp, logp_nojac, dlogp = m.pymc_model.compile_logp(), m.pymc_model.compile_logp(jacobian=False), m.pymc_model.compile_dlogp()
    jf = get_jaxified_logp(m.pymc_model)
    jgrad = jax.grad(lambda *args: jf(list(args)), argnums=tuple(range(len(names))))
    rng = np.random.default_rng(12)
    for k in range(3):
        nat = dict(intercept=.7 + .2 * k, b=np.array([.3, .5, -.2, .1, -.1]) + .1 * k, sd=.8 - .1 * k, z=np.array([-.2, .6, -.7]) + .1 * k,
                   sd_slope=.4 + .05 * k, z_slope=np.array([.8, -.3, .5]) - .1 * k)
        if family == "gaussian":
            nat["sigma"] = .75 + .05 * k
        pt = m.value_point(**nat)
        ref = reference(nat["intercept"], nat["b"], nat["sd"], nat["z"], nat["sd_slope"], nat["z_slope"], nat.get("sigma"))
        ref_nojac = reference(nat["intercept"], nat["b"], nat["sd"], nat["z"], nat["sd_slope"], nat["z_slope"], nat.get("sigma"), jac=False)
        assert abs(float(logp(pt)) - ref) < 1e-8 and abs(float(logp_nojac(pt)) - ref_nojac) < 1e-8
        assert abs(float(jf([pt[n] for n in names])) - ref) < 1e-8
        # gradients: PyMC analytic vs central differences of the explicit reference in value space; JAX vs PyMC
        flat = np.concatenate([np.atleast_1d(np.asarray(pt[n], dtype=float)).ravel() for n in names])

        def f(vec):
            q, i = {}, 0
            for n_ in names:
                size = int(np.prod(np.shape(pt[n_]))); q[n_] = vec[i:i + size] if np.ndim(pt[n_]) else vec[i]; i += size
            beta = np.concatenate([np.atleast_1d(q["dose"]), q["category"], q["dose:category"]])
            return reference(float(q["Intercept"]), beta, float(np.exp(q["1|site_sigma_log__"])), np.asarray(q["1|site_offset"]),
                             float(np.exp(q["dose|site_sigma_log__"])), np.asarray(q["dose|site_offset"]),
                             float(np.exp(q["sigma_log__"])) if family == "gaussian" else None)
        h = 1e-5
        fd = np.array([(f(flat + h * e) - f(flat - h * e)) / (2 * h) for e in np.eye(len(flat))])
        grad_pt = np.asarray(dlogp(pt), dtype=float)
        grad_jx = np.concatenate([np.atleast_1d(np.asarray(g, dtype=float)).ravel() for g in jgrad(*[pt[n] for n in names])])
        assert np.allclose(grad_pt, fd, rtol=1e-6, atol=1e-5) and np.allclose(grad_jx, grad_pt, rtol=1e-9, atol=1e-9)

    # synthetic-posterior predictions: known sites and two unseen sites, (u, v) pairing in both modes
    D = 3
    var = {"Intercept": (("chain", "draw"), np.array([[.2, .3, .1]])), "dose": (("chain", "draw"), np.array([[.4, .5, .3]])),
           "category": (("chain", "draw", "category_dim"), np.array([[[-.2, .7], [-.1, .6], [-.3, .8]]])),
           "dose:category": (("chain", "draw", "dose:category_dim"), np.array([[[.5, -.3], [.4, -.2], [.6, -.4]]])),
           "1|site_sigma": (("chain", "draw"), np.array([[.6, .5, .7]])), "dose|site_sigma": (("chain", "draw"), np.array([[.3, .2, .4]])),
           "1|site": (("chain", "draw", "g__factor_dim"), np.array([[[-.1, .3, .2], [.0, .4, -.2], [-.3, .1, .5]]])),
           "dose|site": (("chain", "draw", "g__factor_dim"), np.array([[[.2, -.1, .3], [.1, .0, -.2], [.3, -.2, .1]]]))}
    if family == "gaussian":
        var["sigma"] = (("chain", "draw"), np.array([[.8, .7, .9]]))
    post = az.InferenceData(posterior=xr.Dataset(var, coords={"chain": [0], "draw": [0, 1, 2], "category_dim": ["a", "c"], "dose:category_dim": ["a", "c"], "g__factor_dim": ["g1", "g2", "g3"]}))
    new = pd.DataFrame({"dose": [0., 1., 2., 1., 1., 3., 1.], "category": ["a", "b", "c", "c", "c", "c", "a"], "site": ["g2", "g3", "g1", "new1", "new1", "new1", "new2"], "n": [4, 0, 5, 3, 3, 3, 6]})
    P = post.posterior
    a, bd = P["Intercept"].to_numpy(), P["dose"].to_numpy()
    U, V = P["1|site"].to_numpy(), P["dose|site"].to_numpy()

    def fixed_part(xv, cat):
        eff = 0.0
        if cat != "b":
            j = ["a", "c"].index(cat); eff = P["category"].to_numpy()[..., j] + P["dose:category"].to_numpy()[..., j] * xv
        return a + bd * xv + eff
    link = (lambda e: e) if family == "gaussian" else expit
    mean_name = "mu" if family == "gaussian" else "p"
    for mode in ("population", "resample_observed"):
        out = m.predict(post, newdata=new, kind="mean", new_groups=mode, seed=5)
        val = out[mean_name].to_numpy(); eta = val if family == "gaussian" else np.log(val / (1 - val))
        for i in range(3):                                                             # known sites, original labels
            j = ["g1", "g2", "g3"].index(new.site[i])
            assert np.allclose(eta[..., i], fixed_part(new.dose[i], new.category[i]) + U[..., j] + new.dose[i] * V[..., j], atol=1e-10)
        r = eta[..., 3:6] - np.stack([fixed_part(new.dose[i], new.category[i]) for i in (3, 4, 5)], -1)   # new1 at x = 1, 1, 3
        assert np.allclose(r[..., 0], r[..., 1])                                                          # repeated rows share the pair
        v_star = (r[..., 2] - r[..., 0]) / 2.0; u_star = r[..., 0] - v_star
        pairs = [any(np.isclose(u_star[0, d], U[0, d, j]) and np.isclose(v_star[0, d], V[0, d, j]) for j in range(3)) for d in range(D)]
        assert all(pairs) if mode == "resample_observed" else not any(pairs)
        assert np.isfinite(eta).all() and not np.allclose(eta[..., 6] - fixed_part(1., "a"), r[..., 0])   # new2 has its own pair
        if family == "binomial":
            assert np.allclose(out["expected"].to_numpy(), val * new.n.to_numpy(), atol=1e-10)


# -- analytic conditional-Gaussian posterior with both group-effect blocks at fixed SDs ----------------------
@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("crossed", [False, True])
def test_conditional_gaussian_with_both_group_blocks_gradient_and_hessian(center, crossed):
    jax = pytest.importorskip("jax")
    from pymc.sampling.jax import get_jaxified_logp

    df = original_frame("gaussian", crossed)
    m = ab.gaussian("y ~ dose*category + (1 + dose || site)", df, priors_for("gaussian"), categorical={"category": "b"}, center_predictors=center)
    x, X, Z, gi = hand_design(df)
    Xc = X - X.mean(0) if center else X
    sigma, sd0, sd1 = .75, .8, .4
    y = df.y.to_numpy()
    W = np.column_stack([np.ones(len(y)), Xc, sd0 * Z, sd1 * x[:, None] * Z])            # theta = (a_c, b[5], z0[3], z1[3])
    prior_prec = np.concatenate([[1 / 2.1 ** 2], np.full(5, 1 / 1.3 ** 2), np.ones(6)])
    prior_mean = np.concatenate([[.4], np.full(5, .1), np.zeros(6)])
    Pm = np.diag(prior_prec) + W.T @ W / sigma ** 2
    mean = np.linalg.solve(Pm, prior_prec * prior_mean + W.T @ y / sigma ** 2)
    order = ["Intercept", "dose", "category", "dose:category", "1|site_offset", "dose|site_offset"]   # the same theta order
    names = [v.name for v in m.pymc_model.value_vars]
    dlogp = m.pymc_model.compile_dlogp()
    d2 = m.pymc_model.compile_d2logp(vars=[m.pymc_model[n] for n in order], negate_output=False)
    jf = get_jaxified_logp(m.pymc_model)

    def split(theta):
        return {"Intercept": theta[0], "dose": theta[1], "category": theta[2:4], "dose:category": theta[4:6], "1|site_offset": theta[6:9], "dose|site_offset": theta[9:12]}

    def point(theta):
        q = split(theta)
        return m.value_point(intercept=q["Intercept"], b=theta[1:6], sd=sd0, z=q["1|site_offset"], sigma=sigma, sd_slope=sd1, z_slope=q["dose|site_offset"])

    def jax_logp_theta(theta_vec, base):
        q = split(theta_vec); args = []
        for n in names:
            args.append(q[n] if n in q else base[n])
        return jf(args)

    rng = np.random.default_rng(4)
    for _ in range(3):
        theta = mean + rng.normal(0, .5, 12)
        pt = point(theta)
        g = np.asarray(dlogp(pt), dtype=float)
        idx, i = {}, 0
        for n in names:
            size = int(np.prod(np.shape(pt[n]))); idx[n] = slice(i, i + size); i += size
        g_theta = np.concatenate([np.atleast_1d(g[idx[n]]) for n in order])
        assert np.allclose(g_theta, -Pm @ (theta - mean), rtol=1e-8, atol=1e-8)          # exact conditional gradient
        H = np.asarray(d2(pt), dtype=float)
        assert H.shape == (12, 12) and np.allclose(H, -Pm, rtol=1e-8, atol=1e-8)          # exact conditional Hessian (PyTensor)
        base = {n: pt[n] for n in names}
        gj = np.asarray(jax.grad(jax_logp_theta)(theta, base)); Hj = np.asarray(jax.hessian(jax_logp_theta)(theta, base))
        assert np.allclose(gj, -Pm @ (theta - mean), rtol=1e-8, atol=1e-8) and np.allclose(Hj, -Pm, rtol=1e-8, atol=1e-8)   # JAX path
    # the conditional mode is the conditional mean: the gradient vanishes there on both paths
    assert np.allclose(np.concatenate([np.atleast_1d(np.asarray(dlogp(point(mean)), dtype=float)[idx[n]]) for n in order]), 0.0, atol=1e-8)
