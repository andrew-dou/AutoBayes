#!/usr/bin/env python
"""The AutoBayes side of the brms benchmark: fit each data set with NumPyro on the CPU and record times and draws.

    python examples/brms_parity/benchmark_autobayes.py <data_dir> <out_dir>

Same models, priors and sampler settings as benchmark_brms.R: four chains run in parallel, 1000 warm-up and 1000
kept draws each, target acceptance 0.9. Writes <out_dir>/autobayes_<model>.json with the timings, the number of
divergent transitions, the draws of the parameters both libraries report (named as brms names them), and the times of
the post-processing functions on the fit (POST below, each the first call and the median of three more), for compare.py.
"""
import json
import statistics
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import autobayes as ab  # noqa: E402

SETTINGS = dict(chains=4, draws=1000, tune=1000, target_accept=0.9, seed=2026)
N01, H1 = ab.Normal(0, 1), ab.HalfNormal(1)


def b1(df):
    pr = ab.Priors(intercept=ab.Normal(0, 2.5), b=N01, sd={"g": {"Intercept": H1, "x": H1}}, cor={"g": ab.LKJ(2)})
    return ab.bernoulli("y ~ x + (1 + x | g)", df, pr)


def b2(df):
    pr = ab.Priors(intercept=ab.Normal(0, 2.5), b=N01, sd={"g": {"Intercept": H1, "x": H1}}, cor={"g": ab.LKJ(2)}, sigma=H1)
    return ab.gaussian("y ~ x + z + (1 + x | g)", df, pr)


def b3(df):
    return ab.cumulative("y ~ x + (1 | g)", df, ab.Priors(intercept=ab.Normal(0, 2.5), b=N01, sd=H1))


def b4(df):
    return ab.zero_inflated_poisson("y ~ x + (1 | g)", df,
                                    ab.Priors(intercept=ab.Normal(0, 2.5), b=N01, sd=H1, zi=ab.Beta(2, 2)))


def b5(df):
    return ab.weibull("y ~ x + (1 | g)", df, ab.Priors(intercept=ab.Normal(0, 2.5), b=N01, sd=H1, shape=ab.Gamma(2, 0.5)))


def b6(df):
    return ab.zero_inflated_negbinomial("y ~ x + (1 | g)", df, ab.Priors(intercept=ab.Normal(0, 2.5), b=N01, sd=H1,
                                                                         shape=ab.Gamma(2, 0.5), zi=ab.Beta(2, 2)))


def b7(df):
    return ab.sratio("y ~ x + (1 | g)", df, ab.Priors(intercept=ab.Normal(0, 2.5), b=N01, sd=H1))


def b8(df):
    formula = "y ~ " + " + ".join(f"x{j}" for j in range(1, 21))
    return ab.gaussian(formula, df, ab.Priors(intercept=ab.Normal(0, 2.5), b=ab.Horseshoe(), sigma=H1))


def b9(df):
    return ab.beta_binomial("k | trials(size) ~ x + (1 | g)", df, ab.Priors(intercept=ab.Normal(0, 2.5), b=N01, sd=H1,
                                                                             phi=ab.Gamma(2, 0.1)))


def b10(df):
    return ab.gaussian("y ~ x + (1 | mm(g1, g2))", df, ab.Priors(intercept=ab.Normal(0, 2.5), b=N01, sd=H1, sigma=H1))


BUILD = {"B1_logistic_glmm": b1, "B2_gaussian_glmm": b2, "B3_ordinal_glmm": b3, "B4_zip_glmm": b4,
         "B5_weibull_glmm": b5, "B6_zinb_glmm": b6, "B7_sratio_glmm": b7, "B8_horseshoe_regression": b8,
         "B9_betabinomial_glmm": b9, "B10_multimembership": b10}
# The post-processing functions timed on each fit, as brms's: posterior_epred(), log_lik(), loo(), waic(), bayes_R2(),
# hypothesis(fit, "x > 0") and conditional_effects(fit, "x"). A function that refuses the model is recorded as such.
POST = {
    "posterior_epred": lambda m, f: m.predict(f, kind="mean"),
    "log_lik": lambda m, f: ab.log_lik(m, f),
    "loo": lambda m, f: ab.loo(m, f),
    "waic": lambda m, f: ab.waic(m, f),
    "bayes_R2": lambda m, f: ab.bayes_R2(m, f),
    "hypothesis": lambda m, f: ab.hypothesis(m, f, f"{_first_coef(m)} > 0"),
    "conditional_effects": lambda m, f: ab.conditional_effects(m, f, [_first_coef(m)]),
}


def _first_coef(model):
    return "x" if "x" in model.spec.columns else model.spec.columns[0]


def time_post(model, fit) -> dict:
    out = {}
    for name, call in POST.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t0 = time.perf_counter()
                call(model, fit)
                first = time.perf_counter() - t0
                again = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    call(model, fit)
                    again.append(time.perf_counter() - t0)
            out[name] = dict(first=first, median=statistics.median(again))
        except Exception as e:                                 # a refusal: recorded, not timed
            out[name] = dict(refused=f"{type(e).__name__}: {str(e)[:160]}")
    return out


def draws_by_brms_name(model, post) -> dict[str, list[float]]:
    """The draws (chain-major, flattened) of the parameters brms also reports, under brms's names."""
    out = {}

    def put(name, arr):
        out[name] = np.asarray(arr, dtype=float).reshape(-1).tolist()

    names = model.names
    if "intercept" in names and names["intercept"] in post:
        v = post[names["intercept"]]
        if v.ndim == 2:
            put("b_Intercept", v)
        else:                                                  # ordered thresholds: b_Intercept[1], ...
            for j in range(v.shape[-1]):
                put(f"b_Intercept[{j + 1}]", v.isel({v.dims[-1]: j}))
    for key, var in names.items():                             # every population-level coefficient, b_<column>
        if key.startswith("b:") and var in post and post[var].ndim == 2:
            put(f"b_{key[2:]}", post[var])
    for var in post.data_vars:                                 # group-level SDs: 1|g_sigma -> sd_g__Intercept
        if var.endswith("_sigma") and "|" in var:
            coef, group = var[:-len("_sigma")].split("|", 1)
            put(f"sd_{group}__{'Intercept' if coef == '1' else coef}", post[var])
    if "cor|g" in post:
        c = post["cor|g"]
        put("cor_g__Intercept__x", c.isel({c.dims[2]: 0, c.dims[3]: 1}))
    for par in ("sigma", "zi", "shape", "phi"):
        if isinstance(names.get(par), str) and names[par] in post:
            put(par, post[names[par]])
    return out


def main(data_dir: str, out_dir: str, only=None) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, build in BUILD.items():
        if only and name not in only:
            continue
        df = pd.read_csv(Path(data_dir) / f"{name}.csv", dtype={"g": str, "g1": str, "g2": str})
        t0 = time.perf_counter()
        model = build(df)
        t_build = time.perf_counter() - t0
        warm = ab.fit(model, backend="numpyro", chains=4, draws=1, tune=1, seed=1)   # JIT compilation and almost no sampling
        fit = ab.fit(model, backend="numpyro", **SETTINGS)
        divergent = int(np.asarray(fit.idata.sample_stats["diverging"]).sum()) if "diverging" in fit.idata.sample_stats else None
        record = dict(model=name, n=len(df), library="autobayes", backend="numpyro", device=fit.device_actual,
                      chain_method=fit.chain_method, build_seconds=t_build, compile_estimate_seconds=warm.wall_seconds,
                      fit_seconds=fit.wall_seconds, divergent=divergent, post=time_post(model, fit),
                      draws=draws_by_brms_name(model, fit.idata.posterior))
        (out / f"autobayes_{name}.json").write_text(json.dumps(record))
        print(f"{name}: build {t_build:.1f} s, compile ~{warm.wall_seconds:.1f} s, fit {fit.wall_seconds:.1f} s, "
              f"{divergent} divergent", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], set(sys.argv[3:]) or None)
