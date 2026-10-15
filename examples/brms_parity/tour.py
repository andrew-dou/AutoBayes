#!/usr/bin/env python
"""A tour of what AutoBayes fits today, one brms model class per demo, on simulated data with known parameters.

    python examples/brms_parity/tour.py <out_dir>

Each demo builds the model as brms would write it, samples it with NumPyro on the CPU (four chains, 1000 warm-up and
1000 kept draws), and records the posterior summaries of its scalar parameters next to the values the data were
simulated from, the worst R-hat, the smallest bulk ESS and the wall time. Writes <out_dir>/tour.json.
"""
import json
import sys
import time
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import autobayes as ab  # noqa: E402

FIT = dict(backend="numpyro", chains=4, draws=1000, tune=1000, target_accept=0.9, seed=7)
N, H = ab.Normal, ab.HalfNormal


def std(v):
    return (v - v.mean()) / v.std()


def summarise(idata):
    """Mean and 90% interval of every scalar or short-vector parameter (at most six entries), flattened; a correlation
    matrix contributes its first off-diagonal entry. Group effects, latent values and per-row quantities are left out."""
    out = {}
    post = idata.posterior
    for name, v in post.data_vars.items():
        extra = [d for d in v.dims if d not in ("chain", "draw")]
        if len(extra) == 2 and name.startswith(("cor", "rescor")):
            v = v.isel({extra[0]: 0, extra[1]: 1})
            extra = []
        if len(extra) > 1 or (extra and v.sizes[extra[0]] > 6):
            continue
        flat = v.stack(s=("chain", "draw")).transpose(..., "s").values
        flat = flat.reshape(-1, flat.shape[-1]) if extra else flat.reshape(1, -1)
        for j, row in enumerate(flat):
            key = name if not extra else f"{name}[{j + 1}]"
            out[key] = dict(mean=float(row.mean()), q05=float(np.quantile(row, 0.05)), q95=float(np.quantile(row, 0.95)))
    return out


def diagnostics(idata):
    s = az.summary(idata, kind="diagnostics")
    return dict(max_rhat=float(s["r_hat"].max()), min_ess_bulk=float(s["ess_bulk"].min()))


def demo_slopes(rng):
    n, J = 1500, 40
    g = rng.integers(0, J, n); x = std(rng.normal(size=n))
    u0, u1 = rng.normal(0, 0.8, J), rng.normal(0, 0.4, J)
    y = rng.binomial(1, expit(-0.4 + 0.9 * x + u0[g] + u1[g] * x))
    df = pd.DataFrame({"y": y, "x": x, "g": [f"g{k:02d}" for k in g]})
    pr = ab.Priors(intercept=N(0, 2.5), b=N(0, 1), sd={"g": {"Intercept": H(1), "x": H(1)}}, cor={"g": ab.LKJ(2)})
    m = ab.bernoulli("y ~ x + (1 + x | g)", df, pr)
    code = ('ab.bernoulli("y ~ x + (1 + x | g)", df, ab.Priors(intercept=ab.Normal(0, 2.5), b=ab.Normal(0, 1),\n'
            '    sd={"g": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(1)}}, cor={"g": ab.LKJ(2)}))')
    truth = {"Intercept": -0.4, "x": 0.9, "1|g_sigma": 0.8, "x|g_sigma": 0.4, "cor|g": 0.0}
    return m, code, truth, "Logistic regression with correlated varying intercepts and slopes"


def demo_ordinal(rng):
    n, J = 1200, 25
    g = rng.integers(0, J, n); x = std(rng.normal(size=n))
    latent = 0.8 * x + rng.normal(0, 0.7, J)[g] + rng.logistic(size=n)
    y = 1 + np.searchsorted([-1.5, -0.3, 0.6, 1.8], latent)
    df = pd.DataFrame({"y": y, "x": x, "g": [f"g{k:02d}" for k in g]})
    m = ab.cumulative("y ~ x + (1 | g)", df, ab.Priors(intercept=N(0, 2.5), b=N(0, 1), sd=H(1)))
    code = 'ab.cumulative("y ~ x + (1 | g)", df, ab.Priors(intercept=ab.Normal(0, 2.5), b=ab.Normal(0, 1), sd=ab.HalfNormal(1)))'
    truth = {"x": 0.8, "1|g_sigma": 0.7, "Intercept[1]": -1.5, "Intercept[2]": -0.3, "Intercept[3]": 0.6, "Intercept[4]": 1.8}
    return m, code, truth, "Ordinal (cumulative logit) regression with a varying intercept"


def demo_zip(rng):
    n, J = 1500, 30
    g = rng.integers(0, J, n); x = std(rng.normal(size=n))
    mu = np.exp(0.6 + 0.5 * x + rng.normal(0, 0.5, J)[g])
    y = np.where(rng.random(n) < 0.3, 0, rng.poisson(mu))
    df = pd.DataFrame({"y": y, "x": x, "g": [f"g{k:02d}" for k in g]})
    m = ab.zero_inflated_poisson("y ~ x + (1 | g)", df, ab.Priors(intercept=N(0, 2.5), b=N(0, 1), sd=H(1), zi=ab.Beta(2, 2)))
    code = ('ab.zero_inflated_poisson("y ~ x + (1 | g)", df,\n'
            '    ab.Priors(intercept=ab.Normal(0, 2.5), b=ab.Normal(0, 1), sd=ab.HalfNormal(1), zi=ab.Beta(2, 2)))')
    truth = {"Intercept": 0.6, "x": 0.5, "1|g_sigma": 0.5, "zi": 0.3}
    return m, code, truth, "Zero-inflated Poisson counts"


def demo_distributional(rng):
    n = 800
    x, z = std(rng.normal(size=n)), std(rng.normal(size=n))
    y = 1.0 + 0.5 * x + rng.normal(size=n) * np.exp(-0.2 + 0.6 * z)
    df = pd.DataFrame({"y": y, "x": x, "z": z})
    pr = ab.Priors(intercept=N(0, 5), b=N(0, 1), sigma={"intercept": N(0, 1), "b": N(0, 1)})
    m = ab.gaussian(["y ~ x", "sigma ~ z"], df, pr)
    code = ('ab.gaussian(["y ~ x", "sigma ~ z"], df,\n'
            '    ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 1), sigma={"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 1)}))')
    truth = {"Intercept": 1.0, "x": 0.5, "sigma_Intercept": -0.2, "sigma_z": 0.6}
    return m, code, truth, "Distributional regression: the residual SD has its own formula"


def demo_multivariate(rng):
    n = 600
    x = std(rng.normal(size=n))
    e = rng.multivariate_normal([0, 0], [[1.0, 0.6], [0.6, 1.0]], n)
    df = pd.DataFrame({"y1": 0.5 + 1.0 * x + e[:, 0], "y2": -0.3 + 0.4 * x + e[:, 1], "x": x})
    resp = {"intercept": N(0, 5), "b": N(0, 1), "sigma": H(1)}
    m = ab.gaussian("mvbind(y1, y2) ~ x", df, ab.Priors(resp={"y1": resp, "y2": resp}, rescor=ab.LKJ(2)), rescor=True)
    code = ('resp = {"intercept": ab.Normal(0, 5), "b": ab.Normal(0, 1), "sigma": ab.HalfNormal(1)}\n'
            'ab.gaussian("mvbind(y1, y2) ~ x", df, ab.Priors(resp={"y1": resp, "y2": resp}, rescor=ab.LKJ(2)), rescor=True)')
    truth = {"y1_x": 1.0, "y2_x": 0.4, "rescor": 0.6}
    return m, code, truth, "Multivariate regression with a residual correlation"


def demo_nonlinear(rng):
    n, J = 600, 12
    g = rng.integers(0, J, n); x = rng.uniform(0, 3, n)
    a = 2.0 * np.exp(rng.normal(0, 0.15, J))[g]
    y = a * np.exp(0.5 * x) + rng.normal(0, 0.4, n)
    df = pd.DataFrame({"y": y, "x": x, "g": [f"g{k:02d}" for k in g]})
    pr = ab.Priors(nlpars={"a": {"b": N(2, 1), "sd": H(1)}, "b": {"b": N(0, 1)}}, sigma=H(1))
    m = ab.gaussian(["y ~ a * exp(b * x)", "a ~ 1 + (1 | g)", "b ~ 1"], df, pr, nl=True)
    code = ('ab.gaussian(["y ~ a * exp(b * x)", "a ~ 1 + (1 | g)", "b ~ 1"], df,\n'
            '    ab.Priors(nlpars={"a": {"b": ab.Normal(2, 1), "sd": ab.HalfNormal(1)}, "b": {"b": ab.Normal(0, 1)}},\n'
            '              sigma=ab.HalfNormal(1)), nl=True)')
    truth = {"b_b": 0.5, "a_Intercept": 2.0, "sigma": 0.4}
    return m, code, truth, "Non-linear growth curve with a varying amplitude"


def demo_gp(rng):
    n = 300
    x = rng.uniform(-3, 3, n)
    y = np.sin(1.3 * x) + 0.3 * x + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"y": y, "x": x})
    pr = ab.Priors(intercept=N(0, 2), sigma=H(1), sdgp=H(1), lscale=ab.InverseGamma(3, 1))
    m = ab.gaussian("y ~ gp(x, k = 20, c = 5/4)", df, pr)
    code = ('ab.gaussian("y ~ gp(x, k = 20, c = 5/4)", df, ab.Priors(intercept=ab.Normal(0, 2), sigma=ab.HalfNormal(1),\n'
            '    sdgp=ab.HalfNormal(1), lscale=ab.InverseGamma(3, 1)))')
    truth = {"sigma": 0.3}
    return m, code, truth, "Approximate Gaussian process (Hilbert-space basis, 20 functions)"


def demo_ar(rng):
    J, T = 20, 40
    rows = []
    for j in range(J):
        e = 0.0
        for t in range(T):
            x = rng.normal()
            e = 0.6 * e + rng.normal(0, 0.5)
            rows.append(dict(y=1.0 + 0.8 * x + e, x=x, t=t, g=f"s{j:02d}"))
    df = pd.DataFrame(rows)
    df["x"] = std(df["x"])
    pr = ab.Priors(intercept=N(0, 5), b=N(0, 1), sigma=H(1), ar=N(0, 0.5))
    m = ab.gaussian("y ~ x + ar(time = t, gr = g, p = 1)", df, pr)
    code = ('ab.gaussian("y ~ x + ar(time = t, gr = g, p = 1)", df,\n'
            '    ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1), ar=ab.Normal(0, 0.5)))')
    truth = {"ar[1]": 0.6, "sigma": 0.5}
    return m, code, truth, "Autoregressive residuals within each of 20 series"


def demo_mixture(rng):
    n = 800
    comp = rng.random(n) < 0.35
    y = np.where(comp, rng.normal(-2.0, 0.7, n), rng.normal(1.5, 1.0, n))
    df = pd.DataFrame({"y": y})
    c = lambda m0: ab.Priors(intercept=N(m0, 2), sigma=H(2))  # noqa: E731
    m = ab.mixture("y ~ 1", df, ab.Priors(components=[c(-1), c(1)], theta=ab.Dirichlet(2)), families=["gaussian", "gaussian"])
    code = ('comp = lambda m0: ab.Priors(intercept=ab.Normal(m0, 2), sigma=ab.HalfNormal(2))\n'
            'ab.mixture("y ~ 1", df, ab.Priors(components=[comp(-1), comp(1)], theta=ab.Dirichlet(2)),\n'
            '           families=["gaussian", "gaussian"])')
    truth = {"Intercept[1]": -2.0, "Intercept[2]": 1.5, "sigma1": 0.7, "sigma2": 1.0, "theta[1]": 0.35}
    return m, code, truth, "Two-component Gaussian mixture"


def demo_me(rng):
    n = 500
    xtrue = rng.normal(0, 1, n)
    sdx = rng.uniform(0.3, 0.8, n)
    xobs = xtrue + rng.normal(0, sdx)
    y = 0.5 + 1.2 * xtrue + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"y": y, "xobs": xobs, "sdx": sdx})
    pr = ab.Priors(intercept=N(0, 5), b=N(0, 2), sigma=H(1), meanme=N(0, 2), sdme=H(1))
    m = ab.gaussian("y ~ me(xobs, sdx)", df, pr)
    code = ('ab.gaussian("y ~ me(xobs, sdx)", df, ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2),\n'
            '    sigma=ab.HalfNormal(1), meanme=ab.Normal(0, 2), sdme=ab.HalfNormal(1)))')
    naive = np.polyfit(xobs, y, 1)[0]
    truth = {"slope_true": 1.2, "slope_naive_least_squares": float(naive)}
    return m, code, truth, "Measurement error in a predictor (the naive slope is attenuated)"


DEMOS = [demo_slopes, demo_ordinal, demo_zip, demo_distributional, demo_multivariate, demo_nonlinear, demo_gp, demo_ar,
         demo_mixture, demo_me]


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for k, demo in enumerate(DEMOS):
        rng = np.random.default_rng(100 + k)
        t0 = time.perf_counter()
        try:
            model, code, truth, title = demo(rng)
            t_build = time.perf_counter() - t0
            fit = ab.fit(model, **FIT)
            rec = dict(name=demo.__name__, title=title, code=code, truth=truth, build_seconds=t_build,
                       fit_seconds=fit.wall_seconds, summary=summarise(fit.idata), **diagnostics(fit.idata))
            if demo is demo_slopes:
                rec["loo"] = ab.loo(model, fit).estimates.to_dict()
                rec["bayes_R2"] = float(ab.bayes_R2(model, fit).mean())
            if demo is demo_ordinal:
                new = pd.DataFrame({"x": [-1.0, 0.0, 1.0], "g": ["g00"] * 3})
                p = model.predict(fit.idata, newdata=new, kind="mean", include_group_effects=False)
                var = list(p.data_vars)[0]
                rec["category_probabilities"] = p[var].mean(("chain", "draw")).values.round(3).tolist()
        except Exception as e:                                # record the failure and keep going
            rec = dict(name=demo.__name__, error=f"{type(e).__name__}: {e}")
        results.append(rec)
        print(demo.__name__, {k2: v for k2, v in rec.items() if k2 in ("fit_seconds", "max_rhat", "min_ess_bulk", "error")}, flush=True)
    (out / "tour.json").write_text(json.dumps(results, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
