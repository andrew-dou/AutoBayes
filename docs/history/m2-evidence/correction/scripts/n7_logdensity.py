"""N7: the observed-data log density (likelihood term and total model logp) of the M2 example's
data at fixed parameter points, for a given package tree.  Usage: PKG_DIR OUT_JSON"""
import json, subprocess, sys
from pathlib import Path
pkg, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
sys.path.insert(0, str(pkg)); sys.path.insert(1, str(pkg / "examples"))
import numpy as np, pymc as pm
import autobayes as ab
from pymc.sampling.jax import get_jaxified_logp, get_jaxified_graph
assert Path(ab.__file__).resolve().is_relative_to(pkg), ab.__file__
from m2_binomial_varying_intercept import simulate
head = subprocess.run(["git", "-C", str(pkg), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or "exported tree"
df = simulate()
priors = ab.Priors(intercept=ab.Normal(0, 1.5), b=ab.Normal(0, 1), sd=ab.HalfNormal(1))
model = ab.binomial("deaths | trials(patients) ~ x + (1 | clinic)", df, priors)
pmm = model.pymc_model
names = [v.name for v in pmm.value_vars]
logp = pmm.compile_logp()
rows_term = pmm.replace_rvs_by_values([pmm.logp(vars=[pmm[model.names["y"]]], sum=False)[0]])[0]
f_rows = pmm.compile_fn(rows_term, inputs=pmm.value_vars, point_fn=True, on_unused_input="ignore")
jf = get_jaxified_logp(pmm)
f_rows_jx = get_jaxified_graph(inputs=pmm.value_vars, outputs=[rows_term])
rng = np.random.default_rng(2026)
res = {"package": str(pkg), "git_head_of_package_dir": head, "rows": int(len(model.design.y)), "dropped_rows": [int(i) for i in model.dropped_rows], "points": []}
for i in range(6):
    nat = dict(intercept=float(rng.normal(-0.5, 0.5)), b=rng.normal(0.8, 0.3, 1), sd=float(np.exp(rng.normal(-0.4, 0.3))), z=rng.normal(size=model.design.J))
    if i == 0:
        nat = dict(intercept=-0.5, b=np.array([0.8]), sd=0.7, z=np.zeros(model.design.J))
    point = model.value_point(**nat)
    rows_pt = np.asarray(f_rows(point)); rows_jx = np.asarray(f_rows_jx(*[point[n] for n in names])[0])
    res["points"].append({"intercept": nat["intercept"], "b": [float(x) for x in nat["b"]], "sd": nat["sd"], "z_first3": [float(x) for x in nat["z"][:3]],
                          "total_logp_pytensor": float(logp(point)), "total_logp_jax": float(jf([point[n] for n in names])),
                          "likelihood_sum_pytensor": float(rows_pt.sum()), "likelihood_sum_jax": float(rows_jx.sum()),
                          "likelihood_rows_pytensor": rows_pt.tolist(), "likelihood_rows_jax": rows_jx.tolist()})
    print(i, res["points"][-1]["total_logp_pytensor"], res["points"][-1]["likelihood_sum_pytensor"])
out.write_text(json.dumps(res, indent=1) + "\n"); print("wrote", out)
