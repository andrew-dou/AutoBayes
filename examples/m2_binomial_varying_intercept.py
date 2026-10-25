#!/usr/bin/env python
"""Researcher example for milestone M2: grouped binomial counts.

    deaths | trials(patients) ~ x + (1 | clinic), binomial likelihood, logit link, logit-scale priors.

    uv run --no-sync python examples/m2_binomial_varying_intercept.py                    # PyMC NUTS on CPU
    uv run --no-sync python examples/m2_binomial_varying_intercept.py --backend numpyro  # JAX/NumPyro on CPU
    uv run --no-sync python examples/m2_binomial_varying_intercept.py --backend numpyro --device gpu   # not yet run on a GPU

The scenario: clinics (rows) each report how many of their patients had the outcome; ``x`` is
a standardised clinic-level risk score. The clinic intercepts absorb between-clinic differences
in the baseline log-odds. Three quantities come out of ``predict``: the probability ``p``, the
expected count ``n * p``, and integer count draws. ``--record`` writes the fit record;
``--compare`` checks agreement with an earlier record within four combined MCSE.
"""
import argparse
import json
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import autobayes as ab
from autobayes.diagnostics import format_panel, panel
from autobayes.session import compare_records, environment, fit_record, format_comparison, format_environment

TRUTH = dict(a=-0.5, b=0.8, sd=0.7)


def simulate(rows=300, J=30, seed=7):
    rng = np.random.default_rng(seed)
    g = np.array([i % J for i in range(rows)]); rng.shuffle(g)          # every clinic present
    x = rng.normal(size=rows)
    u = rng.normal(0, TRUTH["sd"], J)
    patients = rng.integers(5, 51, rows)
    patients[:2] = 0                                                      # two clinics reported no patients
    deaths = rng.binomial(patients, expit(TRUTH["a"] + TRUTH["b"] * x + u[g]))
    df = pd.DataFrame({"deaths": deaths, "patients": patients, "x": x, "clinic": [f"c{k:02d}" for k in g]})
    df.attrs["mean_clinic_effect"] = float(u.mean())      # the intercept is identified relative to this
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="pymc", choices=["pymc", "numpyro"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--record", type=Path, default=None, help="write the fit record (JSON) here")
    ap.add_argument("--compare", type=Path, default=None, help="an earlier fit record to agree with")
    args = ap.parse_args()

    print(format_environment(environment()))
    print()
    df = simulate()
    print(f"{len(df)} clinic rows, {int(df.patients.sum())} patients, {int(df.deaths.sum())} deaths, "
          f"{int((df.patients == 0).sum())} rows with no patients")
    priors = ab.Priors(
        intercept=ab.Normal(0, 1.5),      # log-odds of the outcome at the mean risk score (centred intercept)
        b=ab.Normal(0, 1),                # log-odds ratio per unit of x
        sd=ab.HalfNormal(1),              # SD of the clinic intercepts, logit scale
    )                                     # no sigma: a count likelihood has no residual scale
    model = ab.binomial("deaths | trials(patients) ~ x + (1 | clinic)", df, priors)
    print(model.describe())
    print()

    fit = model.fit(backend=args.backend, device=args.device, chains=args.chains,
                    draws=args.draws, tune=args.tune, seed=args.seed)
    print(fit.report())
    print()
    print(format_panel(panel(fit.idata)))
    print()
    print(az.summary(fit.idata, var_names=["Intercept", "x", "1|clinic_sigma"]).to_string())
    ubar = df.attrs["mean_clinic_effect"]
    print(f"truth: Intercept {TRUTH['a']}, x {TRUTH['b']}, sd {TRUTH['sd']}   (one recovery, not calibration)")
    print(f"the realised mean of the {df.clinic.nunique()} simulated clinic effects is {ubar:+.3f}: in one data set the "
          f"intercept is identified relative to it, so compare Intercept with {TRUTH['a'] + ubar:.3f}")

    # Three quantities: probability, expected count, count draws.
    resp = model.predict(fit, kind="response", seed=1)
    fitted = model.data
    print()
    print(f"in-sample: mean predicted probability {float(resp['p'].mean()):.3f}; "
          f"expected deaths summed over clinics {float(resp['expected'].sum(dim='__obs__').mean()):.1f} "
          f"(observed {int(fitted.deaths.sum())}); mean of count draws {float(resp['y'].sum(dim='__obs__').mean()):.1f}")
    new = pd.DataFrame({"x": [0.0, 0.0, 0.0], "patients": [40, 40, 0], "clinic": ["c00", "new_clinic", "new_clinic"]})
    pop = model.predict(fit, newdata=new, kind="response", new_groups="population", seed=1)
    print(f"a 40-patient clinic at x = 0: observed clinic c00 expected deaths {float(pop['expected'][..., 0].mean()):.1f} "
          f"(SD {float(pop['expected'][..., 0].std()):.1f}); a new clinic {float(pop['expected'][..., 1].mean()):.1f} "
          f"(SD {float(pop['expected'][..., 1].std()):.1f}); with no patients: expected {float(pop['expected'][..., 2].max()):.0f}, draws {int(pop['y'][..., 2].max())}")
    prob = model.predict(fit, newdata=new.drop(columns="patients"), kind="probability", new_groups="population", seed=1)
    print(f"probabilities need no trials column: P(outcome) for c00 at x = 0 is {float(prob['p'][..., 0].mean()):.3f}")

    record = fit_record(model, fit)
    if args.record is not None:
        if args.record.exists():
            sys.exit(f"refusing to overwrite {args.record}")
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(record, indent=1))
        print(f"fit record written to {args.record}")
    if args.compare is not None:
        earlier = json.loads(args.compare.read_text())
        print()
        print(format_comparison(compare_records(record, earlier)))


if __name__ == "__main__":
    main()
