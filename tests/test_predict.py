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
