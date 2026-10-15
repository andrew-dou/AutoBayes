#!/usr/bin/env python
"""The AutoBayes side of the brms benchmark: fit each data set with NumPyro on the CPU and record times and draws.

    python examples/brms_parity/benchmark_autobayes.py <data_dir> <out_dir>

Same models, priors and sampler settings as benchmark_brms.R: four chains run in parallel, 1000 warm-up and 1000
kept draws each, target acceptance 0.9. Writes <out_dir>/autobayes_<model>.json with the timings and the draws of the
parameters both libraries report (named as brms names them), for compare.py.
"""
import json
import sys
import time
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


BUILD = {"B1_logistic_glmm": b1, "B2_gaussian_glmm": b2, "B3_ordinal_glmm": b3, "B4_zip_glmm": b4}


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
    for col in ("x", "z"):
        if col in post:
            put(f"b_{col}", post[col])
    for coef in ("Intercept", "x"):
        key = f"{'1' if coef == 'Intercept' else coef}|g_sigma"
        if key in post:
            put(f"sd_g__{coef}", post[key])
    if "cor|g" in post:
        c = post["cor|g"]
        put("cor_g__Intercept__x", c.isel({c.dims[2]: 0, c.dims[3]: 1}))
    for par in ("sigma", "zi"):
        if names.get(par) in post:
            put(par, post[names[par]])
    return out


def main(data_dir: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, build in BUILD.items():
        df = pd.read_csv(Path(data_dir) / f"{name}.csv", dtype={"g": str})
        t0 = time.perf_counter()
        model = build(df)
        t_build = time.perf_counter() - t0
        warm = ab.fit(model, backend="numpyro", chains=4, draws=1, tune=1, seed=1)   # JIT compilation and almost no sampling
        fit = ab.fit(model, backend="numpyro", **SETTINGS)
        record = dict(model=name, n=len(df), library="autobayes", backend="numpyro", device=fit.device_actual,
                      chain_method=fit.chain_method, build_seconds=t_build, compile_estimate_seconds=warm.wall_seconds,
                      fit_seconds=fit.wall_seconds, draws=draws_by_brms_name(model, fit.idata.posterior))
        (out / f"autobayes_{name}.json").write_text(json.dumps(record))
        print(f"{name}: build {t_build:.1f} s, compile ~{warm.wall_seconds:.1f} s, fit {fit.wall_seconds:.1f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
