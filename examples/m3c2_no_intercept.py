"""Build and inspect cell-specific baselines; no sampling or hardware allocation."""
import pandas as pd
import autobayes as ab


def build_example():
    df = pd.DataFrame({"y": [.2, 1.1, .8, 2., 1.5, 2.2],
                       "dose": [0., 1., 2., 0., 1., 2.],
                       "arm": ["control"] * 3 + ["treated"] * 3})
    priors = ab.Priors(b={"dose": ab.Normal(0, 1),
                         "arm": {"control": ab.Normal(0, 2), "treated": ab.Normal(0, 2)},
                         "dose:arm": {"treated": ab.Normal(0, 1)}}, sigma=ab.HalfNormal(1))
    return ab.gaussian("y ~ 0 + dose + arm + dose:arm", df, priors,
                       categorical={"arm": "control"})


if __name__ == "__main__":
    model = build_example()
    print(model.describe())
    print("Coefficient order:", model.spec.columns)
    print("No feature sampling/GPU acceptance has run for this increment.")
