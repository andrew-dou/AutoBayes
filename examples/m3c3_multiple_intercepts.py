"""Build-only example: additive county/year intercepts. No sampling on execution."""
import pandas as pd
import autobayes as ab


def build_example():
    data = pd.DataFrame({
        "county": ["A", "A", "B", "B", "C", "C"],
        "year": ["2024", "2025"] * 3,
        "x": [-1., 1., 0., 2., -2., .5],
        "y": [-.4, 1.2, .3, 1.7, -.6, .8],
    })
    priors = ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1),
                       sd={"county": ab.HalfNormal(1), "year": ab.Exponential(2)},
                       sigma=ab.HalfNormal(1))
    return ab.gaussian("y ~ x + (1 | county) + (1 | year)", data, priors)


if __name__ == "__main__":
    model = build_example()
    print(model.describe())
    print("Group slices:", model.design.group_slices)
    print("Build only. This tiny table is not a sampling or calibration validation.")
