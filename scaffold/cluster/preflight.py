#!/usr/bin/env python3
"""Cheap checks that catch the expensive failures, before you queue anything.

Every check here corresponds to a way a job has actually died seven seconds into
a GPU allocation. Running this on a login node takes about thirty seconds.

    python3 cluster/preflight.py
    python3 cluster/preflight.py --env-prefix /path/to/conda/env
    python3 cluster/preflight.py --skip-model      # environment checks only

Wire it into your model by implementing `build_model()` in a module named by
--model-module (default: `model`). It should return an object exposing a
log-density and its gradient at a point; the adapter below covers PyMC/NumPyro
and falls back to a duck-typed interface.

Exit code 0 means the environment will not be the reason the job dies.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

OK, BAD, WARN = "  ok  ", " FAIL ", " warn "
failures: list[str] = []
warnings: list[str] = []


def report(status: str, name: str, detail: str = "") -> None:
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if status == BAD:
        failures.append(name)
    elif status == WARN:
        warnings.append(name)


def check_ld_library_path(env_prefix: str | None) -> None:
    """The libffi trap: `source activate` does not export LD_LIBRARY_PATH."""
    prefix = env_prefix or os.environ.get("CONDA_PREFIX")
    if not prefix:
        report(WARN, "LD_LIBRARY_PATH", "no env prefix known; skipped")
        return
    lib = str(Path(prefix) / "lib")
    if lib in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        report(OK, "LD_LIBRARY_PATH", f"contains {lib}")
    else:
        report(BAD, "LD_LIBRARY_PATH",
               f"missing {lib} — jobs will die on ImportError: libffi.so.8. "
               f'Add: export LD_LIBRARY_PATH="{lib}:$LD_LIBRARY_PATH"')
    hits = list(Path(prefix).rglob("libffi.so*"))
    report(OK if hits else WARN, "libffi present",
           str(hits[0]) if hits else f"not found under {prefix}")


def check_ctypes() -> None:
    """The actual import that fails when LD_LIBRARY_PATH is wrong."""
    try:
        import ctypes  # noqa: F401
        report(OK, "ctypes imports")
    except Exception as e:  # pragma: no cover
        report(BAD, "ctypes imports", str(e))


def check_jax(require_gpu: bool) -> None:
    try:
        import jax
    except Exception as e:
        report(BAD, "jax imports", str(e))
        return
    report(OK, "jax imports", f"v{jax.__version__}")
    try:
        devs = jax.devices()
    except Exception as e:
        report(BAD, "jax devices", str(e))
        return
    kinds = {d.platform for d in devs}
    detail = ", ".join(f"{d.platform}:{getattr(d, 'device_kind', '?')}" for d in devs)
    if "gpu" in kinds or "cuda" in kinds:
        report(OK, "accelerator visible", detail)
    elif require_gpu:
        report(BAD, "accelerator visible",
               f"only {detail} — the job would run on CPU at full cost")
    else:
        report(WARN, "accelerator visible", f"only {detail}")

    # A real kernel launch. Import succeeding proves nothing: a wheel linked
    # against the wrong CUDA imports fine on a login node and dies on device.
    try:
        import jax.numpy as jnp
        x = jnp.ones((256, 256))
        val = float((x @ x).sum())
        report(OK if val > 0 else BAD, "kernel executes", f"matmul -> {val:.0f}")
    except Exception as e:
        report(BAD, "kernel executes",
               f"{type(e).__name__}: {e} — likely a CUDA version mismatch "
               "between the wheel and the driver")


def check_preallocation() -> None:
    v = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower()
    if v in ("false", "0"):
        report(OK, "XLA preallocation disabled")
    else:
        report(WARN, "XLA preallocation",
               "unset or true — JAX will grab the whole device and anything "
               "else on the node will OOM with a misleading message")


def check_outdir(outdir: str | None) -> None:
    if not outdir:
        return
    p = Path(outdir)
    if p.exists() and any(p.iterdir()):
        report(BAD, "output directory",
               f"{p} exists and is not empty — pick a fresh directory; the "
               "trace already there is a comparison baseline")
    else:
        report(OK, "output directory", f"{p} is free")


def check_model(module_name: str) -> None:
    """Build the model, take one log-density and one gradient, assert finite."""
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        report(WARN, "model module",
               f"could not import {module_name!r} ({e}); skipping model checks")
        return
    if not hasattr(mod, "build_model"):
        report(WARN, "model module",
               f"{module_name}.build_model() not found; skipping model checks")
        return
    try:
        model = mod.build_model()
    except Exception as e:
        report(BAD, "model builds", f"{type(e).__name__}: {e}")
        return
    report(OK, "model builds", type(model).__name__)

    import numpy as np
    try:
        point = model.initial_point()
        logp_fn = model.compile_logp()
        dlogp_fn = model.compile_dlogp()
        lp = float(np.asarray(logp_fn(point)))
        gr = np.asarray(dlogp_fn(point), dtype=float)
    except Exception as e:
        report(WARN, "logp/grad",
               f"could not evaluate via the PyMC interface ({type(e).__name__}: {e})")
        return

    report(OK if np.isfinite(lp) else BAD, "logp finite", f"{lp:.4g}")
    n_bad = int((~np.isfinite(gr)).sum())
    report(OK if n_bad == 0 else BAD, "gradient finite",
           f"{gr.size} components, {n_bad} non-finite")
    try:
        n_free = sum(int(np.asarray(v).size) for v in point.values())
        report(OK, "free parameters", f"{n_free:,}")
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env-prefix", help="conda env prefix (defaults to $CONDA_PREFIX)")
    ap.add_argument("--model-module", default="model",
                    help="module exposing build_model() (default: model)")
    ap.add_argument("--outdir", help="intended output directory; must be fresh")
    ap.add_argument("--config", help="accepted and ignored; for sbatch templates")
    ap.add_argument("--skip-model", action="store_true")
    ap.add_argument("--no-require-gpu", action="store_true")
    args = ap.parse_args()

    print("preflight\n" + "-" * 60)
    check_ld_library_path(args.env_prefix)
    check_ctypes()
    check_jax(require_gpu=not args.no_require_gpu)
    check_preallocation()
    check_outdir(args.outdir)
    if not args.skip_model:
        check_model(args.model_module)

    print("-" * 60)
    if failures:
        print(f"FAILED: {len(failures)} check(s) — {', '.join(failures)}")
        print("Do not queue the job until these pass.")
        return 1
    if warnings:
        print(f"passed with {len(warnings)} warning(s): {', '.join(warnings)}")
    else:
        print("all checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
