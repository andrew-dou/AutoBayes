"""Behaviour that depends on process state, checked in fresh interpreters.

Each test starts ``sys.executable`` with a small script and reads a JSON line back. They
cover what an in-process test cannot: the order of ``import jax`` and ``import autobayes``,
the ``AUTOBAYES_HOST_DEVICES`` switch, and a process where JAX is not importable.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

MODEL = """
import json, numpy as np, pandas as pd
rng = np.random.default_rng(0); n, J = 40, 4
g = rng.integers(0, J, n); x = rng.normal(size=n)
df = pd.DataFrame({"y": 1 + 0.5 * x + rng.normal(0, 0.5, J)[g] + rng.normal(0, 0.3, n), "x": x, "g": [f"l{k}" for k in g]})
priors = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1))
m = ab.gaussian("y ~ x + (1 | g)", df, priors)
"""


def run(script: str, **env_updates) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "XLA_FLAGS"}   # a preset host count would mask the tests
    env.update(env_updates)
    env.setdefault("PYTENSOR_FLAGS", os.environ.get("PYTENSOR_FLAGS", ""))
    proc = subprocess.run([sys.executable, "-c", script], cwd=REPO, env=env, capture_output=True, text=True,
                          timeout=600)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    assert proc.returncode == 0 and lines, (proc.returncode, proc.stdout[-2000:], proc.stderr[-3000:])
    return json.loads(lines[-1])


def test_autobayes_first_gives_parallel_cpu_chains():
    pytest.importorskip("numpyro")
    out = run("import autobayes as ab\nimport jax\n" + MODEL + """
r = m.fit(backend="numpyro", device="cpu", chains=2, draws=30, tune=30, seed=1)
print(json.dumps({"cpu_devices": len(jax.devices("cpu")), "chain_method": r.chain_method, "notes": r.notes, "x64": bool(jax.config.jax_enable_x64)}))
""")
    assert out["cpu_devices"] >= 2 and out["chain_method"] == "parallel" and out["notes"] == [] and out["x64"]


def test_jax_touched_first_falls_back_to_sequential_with_a_note():
    pytest.importorskip("numpyro")
    out = run("import jax\njax.devices()\nimport autobayes as ab\n" + MODEL + """
r = m.fit(backend="numpyro", device="cpu", chains=2, draws=30, tune=30, seed=1)
print(json.dumps({"cpu_devices": len(jax.devices("cpu")), "chain_method": r.chain_method, "notes": r.notes, "report": r.report()}))
""")
    assert out["cpu_devices"] == 1 and out["chain_method"] == "sequential"
    assert any("sequentially" in n for n in out["notes"]) and "chain method     : sequential" in out["report"]


def test_host_devices_env_zero_disables_the_import_time_setting():
    """With the switch off, importing autobayes no longer exposes several CPU devices, so a
    script that initialises JAX before its first fit gets sequential chains. (fit() itself
    still sets the count when JAX is untouched at that point, so the switch is only
    observable if JAX is initialised first, which is what this script does.)"""
    pytest.importorskip("numpyro")
    out = run("import autobayes as ab\nimport jax\njax.devices()\n" + MODEL + """
r = m.fit(backend="numpyro", device="cpu", chains=2, draws=30, tune=30, seed=1)
print(json.dumps({"cpu_devices": len(jax.devices("cpu")), "chain_method": r.chain_method}))
""", AUTOBAYES_HOST_DEVICES="0")
    assert out["cpu_devices"] == 1 and out["chain_method"] == "sequential"


def test_fit_sets_the_host_device_count_when_jax_is_still_untouched():
    """The import-time setting is a convenience; fit() also sets the count if JAX has not
    been initialised, so the switch alone does not force sequential chains."""
    pytest.importorskip("numpyro")
    out = run("import autobayes as ab\nimport jax\n" + MODEL + """
r = m.fit(backend="numpyro", device="cpu", chains=2, draws=30, tune=30, seed=1)
print(json.dumps({"cpu_devices": len(jax.devices("cpu")), "chain_method": r.chain_method}))
""", AUTOBAYES_HOST_DEVICES="0")
    assert out["cpu_devices"] == 2 and out["chain_method"] == "parallel"


def test_environment_and_numpyro_backend_without_jax():
    out = run("""
import sys
for name in ("jax", "jaxlib", "numpyro"):
    sys.modules[name] = None          # makes 'import jax' raise ImportError in this process
import json
import autobayes as ab
""" + MODEL + """
env = ab.environment()
try:
    m.fit(backend="numpyro", device="cpu", chains=1, draws=5, tune=5)
    err = None
except ab.DeviceError as e:
    err = str(e)
r = m.fit(backend="pymc", chains=1, draws=10, tune=10, seed=1, cores=1)
print(json.dumps({"jax_devices": env["jax_devices"], "jax_error": env.get("jax_error"), "err": err, "pymc_backend": r.backend}))
""")
    assert out["jax_devices"] is None and out["jax_error"] and "jax" in out["jax_error"].lower()
    assert out["err"] and "needs jax" in out["err"] and "backend='pymc'" in out["err"]
    assert out["pymc_backend"] == "pymc"
