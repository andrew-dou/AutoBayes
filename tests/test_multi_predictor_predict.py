"""Predictions with two numeric predictors, both families, in-sample and for unseen groups
(the hand-computed path must use every slope on the raw predictor scale)."""
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

import autobayes as ab
from tests.conftest import PRIORS, simulate
from tests.test_bernoulli import PRIORS as BPRIORS, simulate_binary


@pytest.mark.parametrize("family", ["gaussian", "bernoulli"])
def test_two_predictor_predictions(family):
    df = simulate(extra=True) if family == "gaussian" else simulate_binary(extra=True)
    priors = PRIORS if family == "gaussian" else BPRIORS
    m = ab.Model("y ~ x + w + (1 | g)", df, priors, family=family)
    assert m.spec.predictors == ("x", "w") and m.design.X_raw.shape == (len(df), 2)
    fit = m.fit(backend="pymc", chains=2, draws=60, tune=60, seed=3, cores=1)
    post = fit.idata.posterior
    a, bx, bw, u = (post[k].to_numpy() for k in ("Intercept", "x", "w", "1|g"))
    link = (lambda e: e) if family == "gaussian" else expit
    mean_name = "mu" if family == "gaussian" else "p"

    ins = m.predict(fit, kind="mean")
    eta = a[..., None] + bx[..., None] * df.x.to_numpy() + bw[..., None] * df.w.to_numpy() + u[..., m.design.group_index]
    assert np.allclose(ins[mean_name].to_numpy(), link(eta), atol=1e-10)

    new = pd.DataFrame({"x": [0.3, -1.2, 0.3], "w": [2.0, 0.5, 2.0], "g": ["lvl00", "NEW", "NEW"]})
    pop = m.predict(fit, newdata=new, kind="mean", new_groups="population", seed=4)
    eta_obs = a + bx * 0.3 + bw * 2.0 + u[..., 0]
    assert np.allclose(pop[mean_name][..., 0].to_numpy(), link(eta_obs), atol=1e-10)
    inv = (lambda v: v) if family == "gaussian" else (lambda v: np.log(v / (1 - v)))
    eff1 = inv(pop[mean_name][..., 1].to_numpy()) - (a + bx * -1.2 + bw * 0.5)
    eff2 = inv(pop[mean_name][..., 2].to_numpy()) - (a + bx * 0.3 + bw * 2.0)
    assert np.allclose(eff1, eff2, atol=1e-8)                       # same unseen label, same effect
    res = m.predict(fit, newdata=new, kind="mean", new_groups="resample_observed", seed=4)
    eff = inv(res[mean_name][..., 1].to_numpy()) - (a + bx * -1.2 + bw * 0.5)
    assert np.isclose(eff[..., None], u, atol=1e-8).any(-1).all()
    rec = ab.fit_record(m, fit)
    assert set(rec["model"]["priors"]) >= {"b:x", "b:w"} and set(rec["posterior"]) >= {"x", "w"}
