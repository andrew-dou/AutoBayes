import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import autobayes as ab

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def evidence_dir() -> Path:
    """A fresh, unique directory for this run's candidate evidence (see ``tests/evidence.py``).

    Default parent: ``evidence-runs/`` under the repository (gitignored). ``AUTOBAYES_EVIDENCE_DIR``
    names another parent (CI passes one). Anything resolving into ``docs/`` or into an existing
    run is refused before any write; ``run.json`` (environment, git head, argv, pid) is written
    first, with exclusive create. The committed ``docs/m*-evidence/`` files are history.
    """
    from tests.evidence import allocate_run_dir, git_head

    return allocate_run_dir(REPO, os.environ.get("AUTOBAYES_EVIDENCE_DIR"), head=git_head(REPO),
                            environment=ab.environment())


def simulate(n=60, J=6, seed=0, a=1.0, b=0.5, sd=0.5, sigma=0.3, extra=False):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, J, n)
    x = rng.normal(size=n)
    u = rng.normal(0, sd, J)
    y = a + b * x + u[g] + rng.normal(0, sigma, n)
    df = pd.DataFrame({"y": y, "x": x, "g": [f"lvl{k:02d}" for k in g]})
    if extra:
        w = rng.normal(size=n)
        df["w"] = w
        df["y"] = df["y"] - 0.3 * w
    return df


PRIORS = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1))

PRIORS_DICT = {
    "intercept": ("Normal", {"mu": 0.0, "sigma": 5.0}),
    "b": {"x": ("Normal", {"mu": 0.0, "sigma": 2.0}), "w": ("Normal", {"mu": 0.0, "sigma": 2.0})},
    "sd": ("HalfNormal", {"sigma": 1.0}),
    "sigma": ("HalfNormal", {"sigma": 1.0}),
}


@pytest.fixture(scope="session")
def data():
    return simulate()


@pytest.fixture(scope="session")
def model(data):
    return ab.gaussian("y ~ x + (1 | g)", data, PRIORS)


@pytest.fixture(scope="session")
def small_fit(model):
    return model.fit(backend="pymc", chains=2, draws=150, tune=150, seed=1, cores=1)
