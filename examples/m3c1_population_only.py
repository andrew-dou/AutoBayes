#!/usr/bin/env python
"""Researcher example for milestone M3c-1: a population-only model, no group-level term.

    y ~ dose + arm + dose:arm, Gaussian; arm in {control, low, high}, reference "control"; no site term.

    uv run --no-sync python examples/m3c1_population_only.py                    # PyMC NUTS on CPU
    uv run --no-sync python examples/m3c1_population_only.py --backend numpyro  # JAX/NumPyro on CPU

The frozen model was sampled on both CPU backends in the bounded validation recorded in
docs/history/m3c1-decision.md; its independent review and explicit gate correction are recorded there.
The priors have no ``sd`` (there is no varying intercept) and predictions take rows without any grouping column.
"""
import argparse
import json
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import autobayes as ab
from autobayes.diagnostics import format_panel, panel

TRUTH = dict(a=2.0, b=0.5, low=-0.6, high=0.9, dlow=0.3, dhigh=-0.8, sigma=0.4)
DOSE_MEAN = {"control": 0.0, "low": 1.0, "high": -0.5}


def simulate(rows=240, seed=29):
    rng = np.random.default_rng(seed)
    arm = rng.choice(["control", "low", "high"], size=rows, p=[0.5, 0.3, 0.2])
    dose = rng.normal(size=rows) + np.array([DOSE_MEAN[a] for a in arm])
    inter = np.select([arm == "low", arm == "high"], [TRUTH["low"], TRUTH["high"]], 0.0)
    slope = TRUTH["b"] + np.select([arm == "low", arm == "high"], [TRUTH["dlow"], TRUTH["dhigh"]], 0.0)
    y = TRUTH["a"] + inter + slope * dose + rng.normal(0, TRUTH["sigma"], rows)
    return pd.DataFrame({"y": y, "dose": dose, "arm": arm})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="pymc", choices=["pymc", "numpyro"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--record", type=Path, default=None)
    args = ap.parse_args()

    print(ab.format_environment(ab.environment()))
    df = simulate()
    priors = ab.Priors(
        intercept=ab.Normal(0, 5),                  # a_c: expected y at the means of every expanded design column
        b={"dose": ab.Normal(0, 2), "arm": ab.Normal(0, 2), "dose:arm": ab.Normal(0, 1)},
        sigma=ab.HalfNormal(1),                     # no sd: there is no group-level term
    )
    model = ab.gaussian("y ~ dose + arm + dose:arm", df, priors, categorical={"arm": "control"})
    print(model.describe())
    fit = model.fit(backend=args.backend, device=args.device, chains=args.chains, draws=args.draws, tune=args.tune, seed=args.seed)
    print(fit.report()); print(format_panel(panel(fit.idata)))
    print(az.summary(fit.idata, var_names=["Intercept", "dose", "arm", "dose:arm", "sigma"]).to_string())
    print(f"truth: Intercept {TRUTH['a']} (raw: dose = 0, control), dose {TRUTH['b']}, arm[low] {TRUTH['low']}, arm[high] {TRUTH['high']}, "
          f"dose:arm[low] {TRUTH['dlow']}, dose:arm[high] {TRUTH['dhigh']}, sigma {TRUTH['sigma']}   (one recovery, not calibration)")
    new = pd.DataFrame({"dose": [0.0, 1.0, 0.0, 1.0], "arm": ["high", "high", "control", "low"]})      # no grouping column
    out = model.predict(fit, newdata=new, kind="mean")
    print("expected y:", np.round(out["mu"].mean(("chain", "draw")).to_numpy(), 3), "| new_groups:", out.attrs["new_groups"])
    record = ab.fit_record(model, fit)
    if args.record is not None:
        if args.record.exists():
            sys.exit(f"refusing to overwrite {args.record}")
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(record, indent=1))
        print(f"fit record written to {args.record}")


if __name__ == "__main__":
    main()
