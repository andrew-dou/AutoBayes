import inspect
import os
import re
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


# -- the `sampling` marker (brms-parity loop, P1) --------------------------------------------------------
# Any test that runs a sampler is marked automatically, so the per-iteration local run can deselect it
# (`-m "not slow and not sampling"`) while hosted CI still runs everything (`-m "not slow"`). Detection is by
# source text and deliberately conservative: a test is `sampling` when its own source, the source of any fixture
# in its closure, or the source of a same-module helper it calls contains a `.fit(` or `pm.sample(` call. A script
# held in a string and run in a fresh process counts, which is why this is a text match and not an AST walk.
# `@pytest.mark.nosampling` opts out a test whose match does not sample (a stubbed sampler, a refused device).
_SAMPLER_CALL = re.compile(r"\.fit\(|\bpm\.sample\(")


def _source(func) -> str:
    try:
        return inspect.getsource(func)
    except (OSError, TypeError):
        return ""


def calls_sampler(func, module=None) -> bool:
    """True when the source of ``func`` holds a sampler call, directly or through a same-module helper function."""
    src = _source(func)
    if _SAMPLER_CALL.search(src):
        return True
    if module is None:
        return False
    helpers = {name: _source(obj) for name, obj in vars(module).items()
               if inspect.isfunction(obj) and getattr(obj, "__module__", None) == module.__name__ and obj is not func}
    fitting = {name for name, text in helpers.items() if _SAMPLER_CALL.search(text)}
    changed = True
    while changed:                                         # helpers that call sampling helpers
        changed = False
        for name, text in helpers.items():
            if name not in fitting and any(re.search(rf"\b{re.escape(h)}\(", text) for h in fitting):
                fitting.add(name); changed = True
    return any(re.search(rf"\b{re.escape(h)}\(", src) for h in fitting)


def pytest_collection_modifyitems(config, items):
    marked = set()
    for item in items:
        if item.get_closest_marker("nosampling") is not None:
            continue
        funcs = [(getattr(item, "function", None), getattr(item, "module", None))]
        info = getattr(item, "_fixtureinfo", None)
        if info is not None:
            for defs in info.name2fixturedefs.values():
                funcs.extend((d.func, inspect.getmodule(d.func)) for d in defs)
        if any(f is not None and calls_sampler(f, m) for f, m in funcs):
            item.add_marker(pytest.mark.sampling)
            marked.add(item.nodeid)
    config.sampling_nodeids = frozenset(marked)            # recorded before `-m` deselects anything
    config.collected_files = frozenset(item.nodeid.split("::")[0] for item in items)
