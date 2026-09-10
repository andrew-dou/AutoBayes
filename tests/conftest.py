import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import autobayes as ab

REPO = Path(__file__).resolve().parent.parent


def _git_head() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=REPO, capture_output=True,
                             text=True, timeout=10, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True,
                               timeout=10).stdout.strip() != ""
        return out + ("-dirty" if dirty else "")
    except Exception:
        return "nogit"


@pytest.fixture(scope="session")
def evidence_dir() -> Path:
    """Where a test run writes candidate evidence. Never a committed path.

    ``AUTOBAYES_EVIDENCE_DIR`` overrides it (CI passes a run-id path); otherwise
    ``evidence-runs/<UTC timestamp>-<git head>/`` under the repository, which is gitignored.
    The committed records under ``docs/m*-evidence/`` are history and are not rewritten by
    running the tests. A ``run.json`` with the environment and git head is written first so
    that every JSON in the directory can be traced to a stack and a commit.
    """
    override = os.environ.get("AUTOBAYES_EVIDENCE_DIR")
    if override:
        d = Path(override)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        d = REPO / "evidence-runs" / f"{stamp}-{_git_head()}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.json").write_text(json.dumps({
        "git_head": _git_head(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable, "argv": sys.argv, "environment": ab.environment(),
    }, indent=1, default=str))
    return d


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
