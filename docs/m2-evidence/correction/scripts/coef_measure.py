"""Accuracy of the symbolic coefficient log C(n, v) against the 60-digit decimal reference, both paths."""
import math, sys, json
from pathlib import Path
import numpy as np
import pytensor, pytensor.tensor as pt
from pymc.sampling.jax import get_jaxified_graph
sys.path.insert(0, ".")
from autobayes.likelihoods import log_binomial_coefficient_sym
from tests.reference_binomial import log_binom_dec
from tests.test_binomial import GRID_N, grid_ks

n_s = pt.vector("n", dtype="int64"); v_s = pt.vector("v", dtype="int64")
out = log_binomial_coefficient_sym(n_s, v_s)
f_pt = pytensor.function([n_s, v_s], out)
f_jx = get_jaxified_graph(inputs=[n_s, v_s], outputs=[out])
worst = {}
for n in GRID_N + [10**7]:
    ks = grid_ks(n) + [16, 15, 17, n - 16, n - 15, n - 17]
    ks = sorted({k for k in ks if 0 <= k <= n})
    ref = np.array([float(log_binom_dec(n, k)) for k in ks])
    a = f_pt(np.full(len(ks), n), np.array(ks)); b = np.asarray(f_jx(np.full(len(ks), n), np.array(ks))[0])
    ea, eb = np.abs(a - ref), np.abs(b - ref)
    i, j = int(ea.argmax()), int(eb.argmax())
    interior = [t for t, k in enumerate(ks) if 0 < k < n and k not in (1, n - 1, 2, n - 2)]
    worst[n] = dict(pt_max=float(ea.max()), pt_k=ks[i], jx_max=float(eb.max()), jx_k=ks[j],
                    interior_max=float(max(ea[interior].max(), eb[interior].max())) if interior else None,
                    rel_max=float(max((ea / np.maximum(np.abs(ref), 1)).max(), (eb / np.maximum(np.abs(ref), 1)).max())))
    print(n, worst[n])
Path(sys.argv[1]).write_text(json.dumps({"reference": "tests/reference_binomial.log_binom_dec (60-digit decimal)", "grid": "GRID_N + [1e7], grid_ks(n) plus 15,16,17 and n-15,n-16,n-17", "max_abs_error_by_n": worst}, indent=1) + "\n")
