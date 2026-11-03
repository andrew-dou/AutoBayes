#!/usr/bin/env python
"""Model manifest: a JSON record of what models build, so that a refactor of the model machinery can be shown to leave every
model unchanged, bit for bit.

    python tools/model_manifest.py oracle --out PATH [--cases a,b,...]    # every frozen oracle case that builds
    python tools/model_manifest.py compare A B [--mean-rtol 1e-12]          # two manifests, field by field

Run it from a checkout with ``PYTHONPATH=<checkout>`` (the ``autobayes`` and ``tests`` it imports are that checkout's; the
first line printed is ``autobayes.__file__``). ``oracle`` writes one JSON line per case (``record_model`` of the case's model,
plus ``oracle_record``); a case that does not build records its error. ``compare`` reads two such files (or the JSON lines a
pytest plugin writes with ``record_model``) and reports, per key, how many fields it compared and how many are equal, and
every difference; it exits 1 when any field differs beyond ``--mean-rtol`` (which applies to ``predict(kind="mean")``
arrays only; everything else must be equal exactly).

What ``record_model`` records: ``target_identity``, ``describe()``, ``names``, ``group_names``, ``correlation_names``,
``coefficients``; the PyMC graph's free variables and value variables in order (with their types), its deterministics, dims
and coordinates; and a sha256 of every array of the model's design (``Design`` and every block it holds, each group block's
``Z`` included, by dtype, shape and bytes, so that a sign bit counts). ``oracle_record`` adds, for an oracle case: the log
density and its gradient as float hex at the fixture's points and at seeded points, with and without the Jacobian, on
PyTensor and on the JAX backend; and ``predict(kind="linear")``, ``predict(kind="mean")`` and ``log_lik`` on a seeded synthetic
posterior (draws of the graph's own variables at seeded points), on the fitted rows and on the case's new rows.
"""
from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import json
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path

import numpy as np

GROUP_BLOCK_ATTRIBUTES = ("levels", "Z", "group_index", "column_slice", "memberships", "weights", "by_levels", "by_index",
                          "monotonic", "covariates")   # the group block's attributes as they were before index-coded blocks
DRAWS = (2, 2)                                       # chains, draws of the synthetic posterior
SCALE = 0.3                                          # the spread of the seeded points around the graph's initial point


def _error(e: BaseException) -> str:
    return f"{type(e).__name__}: {str(e)[:600]}"


def array_hash(a) -> str:
    """sha256 of an array's dtype, shape and bytes (an object array by its values' JSON)."""
    a = np.asarray(a)
    if a.dtype == object or a.dtype.kind in "US":
        return "obj:" + hashlib.sha256(json.dumps(a.tolist(), default=str).encode()).hexdigest()
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(f"{a.dtype.str}{list(a.shape)}".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def encode_array(a) -> dict:
    """An array as dtype, shape, sha256 and its bytes in base64 (for a tolerance comparison)."""
    a = np.ascontiguousarray(np.asarray(a))
    if a.dtype == object or a.dtype.kind in "US":
        return {"dtype": "object", "values": json.loads(json.dumps(a.tolist(), default=str))}
    return {"dtype": a.dtype.str, "shape": list(a.shape), "sha256": array_hash(a), "b64": base64.b64encode(a.tobytes()).decode()}


def decode_array(entry: dict) -> np.ndarray:
    return np.frombuffer(base64.b64decode(entry["b64"]), dtype=np.dtype(entry["dtype"])).reshape(entry["shape"])


def _plain(v):
    """A JSON-able rendering of a small value (names, levels, slices)."""
    if v is None or isinstance(v, (str, bool, int, float)):
        return v
    if isinstance(v, slice):
        return f"slice({v.start}, {v.stop}, {v.step})"
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, Mapping):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return repr(v)


def _walk(value, path: str, out: dict, depth: int = 0) -> None:
    """Every array reachable from ``value`` (a design, its blocks and mappings), hashed under its attribute path."""
    if depth > 6:
        return
    if isinstance(value, np.ndarray):
        out[path] = array_hash(value)
    elif type(value).__name__ == "GroupBlock":
        for name in GROUP_BLOCK_ATTRIBUTES:
            try:
                _walk(getattr(value, name, None), f"{path}.{name}", out, depth + 1)
            except Exception as e:                   # noqa: BLE001 - a lazy array above its cap, recorded as such
                out[f"{path}.{name}"] = "error " + _error(e)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        names = list(dict.fromkeys([*(["Z"] if hasattr(type(value), "Z") or "Z" in getattr(value, "__dict__", {}) else []),
                                    *(f.name for f in dataclasses.fields(value))]))
        for name in names:
            try:
                _walk(getattr(value, name), f"{path}.{name}", out, depth + 1)
            except Exception as e:                   # noqa: BLE001
                out[f"{path}.{name}"] = "error " + _error(e)
    elif isinstance(value, Mapping):
        for k, x in value.items():
            _walk(x, f"{path}[{k!r}]", out, depth + 1)
    elif isinstance(value, (list, tuple)) and any(isinstance(x, np.ndarray) or dataclasses.is_dataclass(x) for x in value):
        for k, x in enumerate(value):
            _walk(x, f"{path}[{k}]", out, depth + 1)
    elif isinstance(value, (str, int, float, bool, type(None), slice, list, tuple, np.generic)):
        out[path] = _plain(value)


def graph_record(pmm) -> dict:
    coords = {}
    for k, v in pmm.coords.items():
        values = None if v is None else [str(x) for x in v]
        coords[k] = None if values is None else {"n": len(values), "sha256": hashlib.sha256(json.dumps(values).encode()).hexdigest(),
                                                 "head": values[:5]}
    return {"free_RVs": [[v.name, str(v.type)] for v in pmm.free_RVs],
            "value_vars": [[v.name, str(v.type)] for v in pmm.value_vars],
            "deterministics": [[v.name, str(v.type)] for v in pmm.deterministics],
            "observed_RVs": [[v.name, str(v.type)] for v in pmm.observed_RVs],
            "potentials": [v.name for v in pmm.potentials],
            "dims": {k: None if v is None else list(v) for k, v in pmm.named_vars_to_dims.items()},
            "coords": coords}


def record_model(model) -> dict:
    """The cheap fields of one built model (no compilation)."""
    out = {"class": type(model).__name__}
    for name in ("target_identity", "names", "group_names", "correlation_names", "coefficients"):
        try:
            out[name] = _plain(getattr(model, name))
        except Exception as e:                       # noqa: BLE001
            out[name] = "error " + _error(e)
    try:
        out["describe"] = model.describe()
    except Exception as e:                           # noqa: BLE001
        out["describe"] = "error " + _error(e)
    try:
        out["graph"] = graph_record(model.pymc_model)
    except Exception as e:                           # noqa: BLE001
        out["graph"] = "error " + _error(e)
    design = {}
    for owner in (model, *getattr(model, "submodels", {}).values()) if isinstance(getattr(model, "submodels", None), Mapping) else (model,):
        prefix = "design" if owner is model else f"submodel[{getattr(owner.spec, 'response', '?')!r}].design"
        if getattr(owner, "design", None) is not None:
            _walk(owner.design, prefix, design)
    out["design"] = design
    return out


# -- oracle cases --------------------------------------------------------------------------------------------------
def _seed(case_id: str) -> int:
    return int(hashlib.sha256(case_id.encode()).hexdigest()[:8], 16)


def _seeded_points(pmm, seed: int, k: int) -> list[dict]:
    ip = pmm.initial_point()
    rng = np.random.default_rng(seed)
    return [{n: np.asarray(v, dtype=float) + SCALE * rng.normal(size=np.shape(v)) for n, v in ip.items()} for _ in range(k)]


def _hexes(v) -> list[str] | str:
    a = np.asarray(v, dtype=float)
    return float(a).hex() if a.ndim == 0 else [float(x).hex() for x in a.ravel()]


def _densities(pmm, points: list[dict]) -> dict:
    """logp and dlogp at each point, with and without the Jacobian, on PyTensor and JAX, as float hex."""
    from pymc.sampling.jax import get_jaxified_graph

    out = {}
    names = [v.name for v in pmm.value_vars]
    for jac in (True, False):
        tag = "jac" if jac else "nojac"
        try:
            lp, dlp = pmm.compile_logp(jacobian=jac), pmm.compile_dlogp(jacobian=jac)
            out[f"pytensor_{tag}"] = [[_hexes(lp(p)), _hexes(dlp(p))] for p in points]
        except Exception as e:                       # noqa: BLE001
            out[f"pytensor_{tag}"] = "error " + _error(e)
        try:
            fn = get_jaxified_graph(inputs=pmm.value_vars, outputs=[pmm.logp(jacobian=jac), pmm.dlogp(jacobian=jac)])
            vals = []
            for p in points:
                a, g = fn(*[np.asarray(p[n]) for n in names])
                vals.append([_hexes(np.asarray(a)), _hexes(np.asarray(g))])
            out[f"jax_{tag}"] = vals
        except Exception as e:                       # noqa: BLE001
            out[f"jax_{tag}"] = "error " + _error(e)
    return out


def synthetic_posterior(model, seed: int):
    """Draws of every unobserved variable of the graph that has no observation dimension, at seeded points of the value
    space, as an InferenceData in the model's own names and dims (chains x draws = ``DRAWS``)."""
    import arviz as az
    import xarray as xr

    pmm = model.pymc_model
    dims_of = pmm.named_vars_to_dims
    keep = [v for v in pmm.unobserved_RVs if not any(str(d).startswith("__obs__") for d in (dims_of.get(v.name) or ()))]
    fn = pmm.compile_fn(pmm.replace_rvs_by_values(keep), inputs=pmm.value_vars, on_unused_input="ignore", point_fn=True)
    chains, draws = DRAWS
    points = _seeded_points(pmm, seed + 1, chains * draws)
    values = [fn(p) for p in points]
    data, used = {}, set()
    for i, v in enumerate(keep):
        arr = np.stack([np.asarray(values[k][i]) for k in range(len(points))]).reshape(chains, draws, *np.shape(values[0][i]))
        dims = tuple(dims_of.get(v.name) or ()) or tuple(f"{v.name}_dim_{j}" for j in range(arr.ndim - 2))
        data[v.name] = (("chain", "draw", *dims), arr)
        used.update(dims)
    coords = {k: list(c) for k, c in pmm.coords.items() if k in used and c is not None}
    return az.InferenceData(posterior=xr.Dataset(data, coords={**coords, "chain": np.arange(chains), "draw": np.arange(draws)}))


def _prediction(call) -> dict:
    try:
        ds = call()
    except Exception as e:                           # noqa: BLE001
        return {"error": _error(e)}
    if hasattr(ds, "data_vars"):
        return {"vars": {k: encode_array(ds[k].to_numpy()) for k in ds.data_vars},
                "attrs": {k: _plain(v) for k, v in ds.attrs.items()}}
    return {"vars": {"log_lik": encode_array(np.asarray(ds))}}


def _predictions(model, idata, newdata) -> dict:
    import autobayes as ab

    out = {}
    blocks = [("fitted", None)] + ([("newdata", newdata)] if newdata is not None else [])
    for block, nd in blocks:
        for kind in ("linear", "mean"):
            res = _prediction(lambda: ab.predict(model, idata, newdata=nd, kind=kind))
            out[f"{block}.{kind}"] = res
            if "not seen in training" in res.get("error", ""):
                for policy in ("population", "resample_observed"):
                    out[f"{block}.{kind}.{policy}"] = _prediction(
                        lambda: ab.predict(model, idata, newdata=nd, kind=kind, new_groups=policy, seed=11))
        out[f"{block}.linear.nogroup"] = _prediction(lambda: ab.predict(model, idata, newdata=nd, kind="linear",
                                                                         include_group_effects=False))
    out["fitted.log_lik"] = _prediction(lambda: ab.log_lik(model, idata))
    return out


def oracle_record(case_id: str) -> dict:
    from tests.oracle import adapter, gates
    from tests.oracle import fixture as fx

    case = fx.load_case(case_id)
    d = fx.fixture_dir(case_id)
    data = fx.read_frame(case, d / "data.csv")
    newdata = fx.read_frame(case, d / "newdata.csv") if (d / "newdata.csv").exists() else None
    data2 = fx.read_data2(case, d)
    build = case
    if data2:
        spec = case["autobayes"]
        build = {**case, "autobayes": {**spec, "kwargs": {**spec.get("kwargs", {}), "data2": data2}}}
    out = {"case": case_id, "gated": "gate" in case}
    try:
        model = adapter.build_model(build, data)
    except Exception as e:                           # noqa: BLE001
        out["build_error"] = _error(e)
        return out
    out["model"] = record_model(model)
    pmm = model.pymc_model
    try:
        bundle = gates.Bundle(case_id, model=model)
        out["fixture_points"] = _densities(pmm, [vp for _, _, vp in bundle.points()])
    except Exception as e:                           # noqa: BLE001
        out["fixture_points"] = "error " + _error(e)
    try:
        out["seeded_points"] = _densities(pmm, _seeded_points(pmm, _seed(case_id), 3))
    except Exception as e:                           # noqa: BLE001
        out["seeded_points"] = "error " + _error(e)
    try:
        idata = synthetic_posterior(model, _seed(case_id))
        out["predictions"] = _predictions(model, idata, newdata)
    except Exception as e:                           # noqa: BLE001
        out["predictions"] = "error " + _error(e) + traceback.format_exc()[-800:]
    return out


# -- comparison ------------------------------------------------------------------------------------------------------
def _rel(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return np.inf
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.maximum(np.abs(a), np.abs(b))
        err = np.where(scale > 0, np.abs(a - b) / np.where(scale > 0, scale, 1.0), 0.0)
    return float(np.max(err)) if err.size else 0.0


def compare_values(a, b, path: str, diffs: list, counts: list, mean_rtol: float) -> None:
    if isinstance(a, dict) and isinstance(b, dict) and "b64" in a and "b64" in b:
        counts[0] += 1
        if a["sha256"] == b["sha256"]:
            counts[1] += 1
            return
        tolerant = ".mean" in path
        try:
            err = _rel(decode_array(a), decode_array(b))
        except Exception:                            # noqa: BLE001
            err = np.inf
        if tolerant and err <= mean_rtol:
            counts[2] += 1
            diffs.append((path, f"within tolerance: max relative difference {err:.3g}"))
        else:
            diffs.append((path, f"arrays differ: max relative difference {err:.3g} (dtype {a['dtype']} vs {b['dtype']}, "
                                f"shape {a['shape']} vs {b['shape']})"))
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                counts[0] += 1
                diffs.append((f"{path}.{k}", "only in " + ("B" if k not in a else "A")))
                continue
            compare_values(a[k], b[k], f"{path}.{k}", diffs, counts, mean_rtol)
        return
    counts[0] += 1
    if a == b:
        counts[1] += 1
    else:
        sa, sb = json.dumps(a)[:300], json.dumps(b)[:300]
        diffs.append((path, f"A {sa} | B {sb}"))


def read_manifest(path) -> dict:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            rec = json.loads(line)
            out[rec.get("key") or rec["case"]] = rec
    return out


def compare(path_a, path_b, mean_rtol: float, *, limit: int = 400, exclude: tuple[str, ...] = ()) -> int:
    a, b = read_manifest(path_a), read_manifest(path_b)
    keys = sorted(set(a) | set(b))
    only_a = [k for k in keys if k not in b and not k.startswith(exclude)]
    only_b = [k for k in keys if k not in a and not k.startswith(exclude)]
    counts, diffs = [0, 0, 0], []
    for k in keys:
        if k in a and k in b:
            compare_values(a[k], b[k], k, diffs, counts, mean_rtol)
    hard = [d for d in diffs if not d[1].startswith("within tolerance")]
    print(f"records: A {len(a)}, B {len(b)}, both {len(set(a) & set(b))}, only in A {len(only_a)}, only in B {len(only_b)}")
    print(f"fields compared {counts[0]}, equal {counts[1]}, within the mean tolerance {counts[2]}, different {len(hard)}")
    for k in only_a[:50]:
        print(f"ONLY IN A {k}")
    for k in only_b[:50]:
        print(f"ONLY IN B {k}")
    for path, what in diffs[:limit]:
        print(f"{'TOL ' if what.startswith('within') else 'DIFF'} {path}: {what}")
    return 1 if hard or only_a or only_b else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("oracle")
    o.add_argument("--out", required=True)
    o.add_argument("--cases", default="")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--mean-rtol", type=float, default=1e-12)
    c.add_argument("--limit", type=int, default=400)
    c.add_argument("--exclude", default="", help="comma-separated key prefixes left out of the only-in counts")
    args = ap.parse_args(argv)
    if args.cmd == "compare":
        return compare(args.a, args.b, args.mean_rtol, limit=args.limit,
                       exclude=tuple(x for x in args.exclude.split(",") if x))
    import autobayes
    print(autobayes.__file__, flush=True)
    from tests.oracle import fixture as fx

    ids = [c for c in (args.cases.split(",") if args.cases else fx.case_ids())
           if fx.has_fixture(c) and "autobayes" in fx.load_case(c)]
    done = set()
    out = Path(args.out)
    if out.exists():                                 # resume: skip the cases already recorded
        done = set(read_manifest(out))
    with open(out, "a", encoding="utf-8") as f:
        for k, cid in enumerate(ids):
            if cid in done:
                continue
            rec = oracle_record(cid)
            f.write(json.dumps(rec, sort_keys=True) + "\n")
            f.flush()
            print(f"[{k + 1}/{len(ids)}] {cid} {'build error' if 'build_error' in rec else 'ok'}", flush=True)
            import gc
            gc.collect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
