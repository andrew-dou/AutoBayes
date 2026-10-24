"""Before/after evidence for the correction record: the exposed RV's mass over its support and
off-support values, for n = 2 at eta = 0, by observed outcome; both paths.  Usage: PKG_DIR OUT_JSON"""
import json, subprocess, sys
from pathlib import Path
pkg, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
sys.path.insert(0, str(pkg))
import numpy as np, pandas as pd, pymc as pm, pytensor
import autobayes as ab
from pymc.sampling.jax import get_jaxified_graph
assert Path(ab.__file__).resolve().is_relative_to(pkg), ab.__file__
head = subprocess.run(["git", "-C", str(pkg), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or "exported tree"
priors = ab.Priors(intercept=ab.Normal(0, 1.5), b=ab.Normal(0, 1), sd=ab.HalfNormal(1))
res = {"package": str(pkg), "git_head_of_package_dir": head, "n": 2, "eta": 0.0, "true_pmf": [0.25, 0.5, 0.25], "by_observed_k": [], "off_support": []}
for k_obs in (0, 1, 2):
    df = pd.DataFrame({"k": [k_obs, k_obs], "n": [2, 2], "g": ["a", "b"]})
    m = ab.binomial("k | trials(n) ~ 1 + (1 | g)", df, priors)
    pmm = m.pymc_model; rv = pmm[m.names["y"]]
    point = m.value_point(intercept=0.0, b=[], sd=1.0, z=np.zeros(2))
    args = [point[x.name] for x in pmm.value_vars]
    row = {"observed_k": k_obs}
    for dtype, name in (("int64", "int"), ("float64", "float")):
        v = pytensor.tensor.vector("v", dtype=dtype)
        lp = pmm.replace_rvs_by_values([pm.logp(rv, v)])[0]
        f_pt = pmm.compile_fn(lp, inputs=[v, *pmm.value_vars], point_fn=False, on_unused_input="ignore")
        f_jx = get_jaxified_graph(inputs=[v, *pmm.value_vars], outputs=[lp])
        pt_ = [float(np.exp(f_pt(np.array([k, k], dtype=dtype), *args)[0])) for k in (0, 1, 2)]
        jx_ = [float(np.exp(np.asarray(f_jx(np.array([k, k], dtype=dtype), *args)[0])[0])) for k in (0, 1, 2)]
        row[f"pmf_pytensor_{name}_value"] = pt_; row[f"pmf_jax_{name}_value"] = jx_
        row[f"mass_pytensor_{name}_value"] = sum(pt_); row[f"mass_jax_{name}_value"] = sum(jx_)
        if k_obs == 1:
            bad = [-1.0, 3.0, 1.5, 0.5] if dtype == "float64" else [-1, 3, 5, -7]
            for b in bad:
                res["off_support"].append({"value_dtype": dtype, "value": b,
                                           "logp_pytensor": float(f_pt(np.array([b, b], dtype=dtype), *args)[0]),
                                           "logp_jax": float(np.asarray(f_jx(np.array([b, b], dtype=dtype), *args)[0])[0])})
    res["by_observed_k"].append(row)
    print(f"observed k={k_obs}: pmf int-valued pytensor {np.round(row['pmf_pytensor_int_value'], 6)} (mass {row['mass_pytensor_int_value']:.6f}) "
          f"jax {np.round(row['pmf_jax_int_value'], 6)} (mass {row['mass_jax_int_value']:.6f})")
for o in res["off_support"]:
    print(f"off-support value {o['value']!r} ({o['value_dtype']}): logp pytensor {o['logp_pytensor']}, jax {o['logp_jax']}")
out.write_text(json.dumps(res, indent=1) + "\n")
print("wrote", out)
