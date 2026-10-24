"""A8: what the predictive draws mean."""
import numpy as np
import pandas as pd
import pytest

import autobayes as ab


def test_in_sample_predictive_meaning(model, small_fit, data):
    ds = model.predict(small_fit, kind="response", seed=0)
    assert set(ds.data_vars) == {"mu", "y"} and ds.sizes["__obs__"] == len(data)
    post = small_fit.idata.posterior
    # mu from Bambi's predict equals the hand-computed linear predictor from the trace
    a = post["Intercept"].to_numpy(); b = post["x"].to_numpy(); u = post["1|g"].to_numpy()
    mu_hand = a[..., None] + b[..., None] * data.x.to_numpy() + u[..., model.design.group_index]
    assert np.allclose(ds["mu"].to_numpy(), mu_hand, atol=1e-10)
    # y draws scatter around mu with sd sigma: pooled residual sd matches posterior mean sigma
    resid = (ds["y"] - ds["mu"]).to_numpy()
    assert abs(resid.mean()) < 0.05
    assert abs(resid.std() - float(post["sigma"].mean())) < 0.05
    dm = model.predict(small_fit, kind="mean")
    assert set(dm.data_vars) == {"mu"}


def test_out_of_sample_observed_groups(model, small_fit):
    new = pd.DataFrame({"x": [0.0, 1.0, -1.0], "g": ["lvl00", "lvl03", "lvl00"]})
    ds = model.predict(small_fit, newdata=new, kind="mean")
    post = small_fit.idata.posterior
    lv = list(model.design.levels)
    a = post["Intercept"].to_numpy(); b = post["x"].to_numpy(); u = post["1|g"].to_numpy()
    for i, (xv, gv) in enumerate(zip(new.x, new.g)):
        assert np.allclose(ds["mu"][..., i].to_numpy(), a + b * xv + u[..., lv.index(gv)], atol=1e-10)


def test_new_groups_semantics(model, small_fit):
    new = pd.DataFrame({"x": [0.0, 0.0, 0.0], "g": ["lvl00", "NEW_A", "NEW_B"]})
    with pytest.raises(ab.FormulaError, match="not seen in training"):
        model.predict(small_fit, newdata=new)
    post = small_fit.idata.posterior
    a = post["Intercept"].to_numpy(); u = post["1|g"].to_numpy(); sd = post["1|g_sigma"].to_numpy()

    pop = model.predict(small_fit, newdata=new, kind="mean", new_groups="population", seed=1)
    assert pop.attrs["new_levels"] == ["NEW_A", "NEW_B"]
    assert np.allclose(pop["mu"][..., 0].to_numpy(), a + u[..., 0], atol=1e-10)   # observed level untouched
    dev_a = (pop["mu"][..., 1].to_numpy() - a)
    dev_b = (pop["mu"][..., 2].to_numpy() - a)
    # fresh Normal(0, sd) effects: spread ~ E[sd^2]^(1/2), independent across new levels
    assert abs(dev_a.std() - np.sqrt(np.mean(sd**2))) < 0.15 * np.sqrt(np.mean(sd**2))
    assert abs(np.corrcoef(dev_a.ravel(), dev_b.ravel())[0, 1]) < 0.15

    res = model.predict(small_fit, newdata=new, kind="mean", new_groups="resample_observed", seed=1)
    dev = res["mu"][..., 1].to_numpy() - a
    # brms 'uncertainty': every draw's effect is one of the observed levels' effects for that draw
    is_obs = np.isclose(dev[..., None], u).any(-1)
    assert is_obs.all()


# -- U1-U4 (docs/history/m1-decision.md, follow-up unit): unseen-label identity, Gaussian ------
def synthetic_posterior(model, *, draws=1500, effects=None, sigma=0.3):
    """A posterior whose group effects are distinct constants across draws, so that identity
    and sharing of unseen-label effects can be checked exactly."""
    import arviz as az
    import xarray as xr

    J = model.design.J
    effects = np.arange(J, dtype=float) if effects is None else np.asarray(effects, dtype=float)
    shape = (1, draws)
    data = {"Intercept": (("chain", "draw"), np.full(shape, 1.0)),
            "1|g_sigma": (("chain", "draw"), np.full(shape, 0.5)),
            "1|g": (("chain", "draw", "g__factor_dim"), np.broadcast_to(effects, shape + (J,)).copy())}
    for pname in model.spec.predictors:
        data[pname] = (("chain", "draw"), np.full(shape, 0.5))
    if model.family == "gaussian":
        data["sigma"] = (("chain", "draw"), np.full(shape, sigma))
    post = xr.Dataset(data, coords={"chain": [0], "draw": np.arange(draws), "g__factor_dim": list(model.design.levels)})
    return az.InferenceData(posterior=post)


def check_resample_observed_identity(model, idata, mean_name, transform, kind="mean"):
    """Shared by the Gaussian, Bernoulli and binomial tests: U1 (distinct labels), U2 (repeated
    label), U3 (observed label), U4 (values are the hand-computed linear predictor)."""
    J = model.design.J
    new = pd.DataFrame({"x": [0.0, 0.0, 0.0, 1.0, 0.0], "g": ["lvl00", "NEW_A", "NEW_B", "NEW_A", "lvl02"]})
    ds = model.predict(idata, newdata=new, kind=kind, new_groups="resample_observed", seed=11)
    assert ds.attrs["new_groups"] == "resample_observed" and ds.attrs["new_levels"] == ["NEW_A", "NEW_B"]
    post = idata.posterior
    a = post["Intercept"].to_numpy(); b = post["x"].to_numpy(); u = post["1|g"].to_numpy()
    v = ds[mean_name].to_numpy()
    # U3: observed labels use their own effect
    assert np.allclose(v[..., 0], transform(a + u[..., 0]), atol=1e-10)
    assert np.allclose(v[..., 4], transform(a + u[..., 2]), atol=1e-10)
    # U4/U1: each unseen label's effect is one of that draw's observed effects
    cand = transform(a[..., None] + u)                                  # (chain, draw, J) possible values
    for lab, col in [("NEW_A", 1), ("NEW_B", 2)]:
        hit = np.isclose(v[..., col][..., None], cand, atol=1e-10).any(-1)
        assert hit.all(), lab
    # U2: the repeated label (rows 1 and 3) shares its draw, once x b is removed
    eff_row1 = transform_inv(transform, v[..., 1]) - a
    eff_row3 = transform_inv(transform, v[..., 3]) - a - b * 1.0
    assert np.allclose(eff_row1, eff_row3, atol=1e-8)
    # U1: distinct labels coincide only by chance, near 1/J, never in every draw
    same = np.isclose(v[..., 1], v[..., 2], atol=1e-10).mean()
    n_draws = v.shape[0] * v.shape[1]
    se = np.sqrt((1 / J) * (1 - 1 / J) / n_draws)
    assert same < 1.0 and abs(same - 1 / J) < 5 * se + 0.02, (same, 1 / J)
    # reproducible with the seed, different without it
    again = model.predict(idata, newdata=new, kind=kind, new_groups="resample_observed", seed=11)
    assert np.array_equal(again[mean_name].to_numpy(), v)
    other = model.predict(idata, newdata=new, kind=kind, new_groups="resample_observed", seed=12)
    assert not np.array_equal(other[mean_name].to_numpy(), v)


def transform_inv(transform, values):
    from scipy.special import logit
    return values if transform is _identity else logit(values)


def _identity(x):
    return x


def test_resample_observed_keeps_distinct_unseen_labels(model):
    idata = synthetic_posterior(model)
    check_resample_observed_identity(model, idata, "mu", _identity)
    # U4 for kind="response": y scatters around mu with the posterior sigma
    new = pd.DataFrame({"x": [0.0, 0.0], "g": ["NEW_A", "NEW_B"]})
    ds = model.predict(idata, newdata=new, kind="response", new_groups="resample_observed", seed=1)
    resid = (ds["y"] - ds["mu"]).to_numpy()
    assert abs(resid.std() - 0.3) < 0.03 and abs(resid.mean()) < 0.03


def test_resample_observed_on_a_real_fit_is_not_collapsed(model, small_fit):
    """The erroneous behaviour gave every unseen label the same effect in every draw."""
    new = pd.DataFrame({"x": [0.0, 0.0, 0.0], "g": ["NEW_A", "NEW_B", "NEW_A"]})
    ds = model.predict(small_fit, newdata=new, kind="mean", new_groups="resample_observed", seed=2)
    v = ds["mu"].to_numpy()
    assert np.allclose(v[..., 0], v[..., 2])                              # same label, same effect
    assert np.isclose(v[..., 0], v[..., 1]).mean() < 0.5                  # distinct labels, mostly different
    u = small_fit.idata.posterior["1|g"].to_numpy(); a = small_fit.idata.posterior["Intercept"].to_numpy()
    assert np.isclose((v[..., 1] - a)[..., None], u, atol=1e-8).any(-1).all()
