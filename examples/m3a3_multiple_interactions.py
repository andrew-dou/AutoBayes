"""Build-only: shared numeric/factor interactions plus two varying intercepts."""
import pandas as pd
import autobayes as ab


def build_example():
    data = pd.DataFrame({
        "dose": [-1., .5, 1.5, 0., 2., -.5, .2, 1.2, -.8, 1.8, .7, -1.2],
        "time": [.3, 1.4, -.2, 1.1, .7, -.6, 1.6, -.3, .8, -.7, .4, 1.3],
        "arm": ["control", "low", "high"] * 4,
        "region": ["rural", "rural", "urban", "urban"] * 3,
        "clinic": ["A", "A", "B", "B", "C", "C"] * 2,
        "year": ["2024", "2025"] * 6,
        "y": [-.2, .8, 1.7, .5, 2.1, -.1, .6, 1.4, -.3, 1.9, 1., -.5],
    })
    priors = ab.Priors(
        intercept=ab.Normal(0, 2),
        b={"dose": ab.Normal(0, 1), "time": ab.Normal(0, 1),
           "arm": ab.Normal(0, 1), "region": ab.Normal(0, 1),
           "dose:arm": {"high": ab.Normal(0, .8), "low": ab.Normal(0, .4)},
           "time:arm": ab.Normal(0, .5), "dose:region": ab.StudentT(4, 0, .5)},
        sd={"clinic": ab.HalfNormal(.8), "year": ab.Exponential(2)},
        sigma=ab.HalfNormal(1),
    )
    return ab.gaussian(
        "y ~ dose + time + arm + region + dose:arm + time:arm + dose:region + (1 | clinic) + (1 | year)",
        data, priors, categorical={"arm": "control", "region": "rural"},
    )


if __name__ == "__main__":
    model = build_example()
    print(model.describe())
    print("Coefficient order:", model.spec.columns)
    print("Build only. This tiny API example is not sampling or calibration evidence.")
