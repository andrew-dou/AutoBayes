"""Public prediction semantics with fixed draws; no sampler fixtures."""
import arviz as az
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
import autobayes as ab


@pytest.mark.parametrize("family", ["gaussian", "bernoulli", "binomial"])
def test_family_names_and_expected_values_with_fixed_draws(family):
    data = pd.DataFrame({"x": [-1., 0., 1., 2.], "y": [0., 1., 0., 1.], "n": [2, 4, 7, 9]})
    prior = ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1),
                      **({"sigma": ab.HalfNormal(1)} if family == "gaussian" else {}))
    formula = "y | trials(n) ~ x" if family == "binomial" else "y ~ x"
    model = getattr(ab, family)(formula, data, prior, center_predictors=False)
    intercept = np.array([[.1, -.2, .3], [-.1, .2, -.3]])
    slope = np.array([[.4, .5, -.6], [.7, -.8, .9]])
    posterior = {"Intercept": intercept, "x": slope}
    if family == "gaussian": posterior["sigma"] = np.ones_like(intercept)
    draws = az.from_dict(posterior=posterior)
    assert model.link == ("identity" if family == "gaussian" else "logit")
    assert model.names["linpred"] == ("mu" if family == "gaussian" else "logit_p")
    assert model.names["mean"] == ("mu" if family == "gaussian" else "p")
    assert ("sigma" in model.names) == (family == "gaussian")
    modified = pd.DataFrame({"x": [-2., .5], "n": [3, 11]})
    for frame, argument in [(data, None), (modified, modified)]:
        eta = intercept[..., None] + slope[..., None] * frame.x.to_numpy()
        expected = eta if family == "gaussian" else expit(eta)
        actual = model.predict(draws, newdata=argument, kind="mean")
        key = "mu" if family == "gaussian" else "p"
        np.testing.assert_allclose(actual[key].transpose("chain", "draw", "__obs__"), expected, atol=1e-12)
        assert set(actual.data_vars) == ({"p", "expected"} if family == "binomial" else {key})
        if family == "binomial":
            np.testing.assert_allclose(actual.expected, expected * frame.n.to_numpy(), atol=1e-12)
