"""Investigate the failed F7 gate: is the x|g_sigma posterior wrong, or is the realised spread of the 30 simulated slopes small?"""
import warnings; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
import autobayes as ab
rng = np.random.default_rng(17)
J, per = 30, 20
g = np.repeat([f"g{j:02d}" for j in range(J)], per)
x = rng.normal(size=J * per)
u = rng.normal(0, 0.5, J); v = rng.normal(0, 0.3, J)
gi = np.repeat(np.arange(J), per)
y = 1.0 + 0.5 * x + u[gi] + x * v[gi] + rng.normal(0, 0.4, J * per)
df = pd.DataFrame({"y": y, "x": x, "g": g})
print(f"realised SD of the 30 simulated intercepts u: {u.std(ddof=1):.4f} (population 0.5); of the 30 slopes v: {v.std(ddof=1):.4f} (population 0.3)")
print(f"  chi-square scale: 29 * s^2 / sigma^2 = {29 * v.var(ddof=1) / 0.09:.1f} for v (expected 29), {29 * u.var(ddof=1) / 0.25:.1f} for u")
# independent per-group OLS (no library): spread of the raw slope estimates minus their sampling noise
slopes, ses, icpts = [], [], []
for j in range(J):
    m = gi == j; X = np.column_stack([np.ones(m.sum()), x[m]])
    beta, res, *_ = np.linalg.lstsq(X, y[m], rcond=None)
    resid = y[m] - X @ beta; s2 = resid @ resid / (m.sum() - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    slopes.append(beta[1]); ses.append(np.sqrt(cov[1, 1])); icpts.append(beta[0])
slopes, ses, icpts = np.array(slopes), np.array(ses), np.array(icpts)
print(f"  OLS slopes across groups: SD {slopes.std(ddof=1):.4f}, mean se {ses.mean():.4f}; noise-corrected spread sqrt(SD^2 - mean se^2) = {np.sqrt(max(slopes.var(ddof=1) - (ses**2).mean(), 0)):.4f}")
print(f"  correlation of OLS slopes with the true v: {np.corrcoef(slopes, 0.5 + v)[0,1]:.3f}; of OLS intercepts with the true u: {np.corrcoef(icpts, 1.0 + u)[0,1]:.3f}")
priors = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd={"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(1)}, sigma=ab.HalfNormal(1))
m = ab.gaussian("y ~ x + (1 + x || g)", df, priors)
fit = m.fit(backend="pymc", chains=2, draws=500, tune=500, seed=17, cores=1)
post = fit.idata.posterior
vm = post["x|g"].mean(("chain", "draw")).to_numpy(); um = post["1|g"].mean(("chain", "draw")).to_numpy()
print(f"  posterior-mean v_j vs true v_j: correlation {np.corrcoef(vm, v)[0,1]:.3f}, regression slope {np.polyfit(v, vm, 1)[0]:.3f} (shrinkage < 1 expected); SD of posterior means {vm.std(ddof=1):.4f}")
print(f"  posterior-mean u_j vs true u_j: correlation {np.corrcoef(um, u)[0,1]:.3f}, regression slope {np.polyfit(u, um, 1)[0]:.3f}; SD of posterior means {um.std(ddof=1):.4f}")
sd1 = post["x|g_sigma"].to_numpy().ravel(); sd0 = post["1|g_sigma"].to_numpy().ravel()
print(f"  x|g_sigma posterior: mean {sd1.mean():.3f}, 99% interval [{np.quantile(sd1, .005):.3f}, {np.quantile(sd1, .995):.3f}]; realised v SD {v.std(ddof=1):.3f} inside: {np.quantile(sd1, .005) <= v.std(ddof=1) <= np.quantile(sd1, .995)}")
print(f"  1|g_sigma posterior: mean {sd0.mean():.3f}, 99% interval [{np.quantile(sd0, .005):.3f}, {np.quantile(sd0, .995):.3f}]; realised u SD {u.std(ddof=1):.3f}")
# how unusual is this seed? realised slope SDs over 2000 fresh draws of 30 slopes
sds = np.array([np.random.default_rng(s).normal(0, 0.3, 30).std(ddof=1) for s in range(1000, 3000)])
print(f"  fraction of 30-draw samples with realised SD <= {v.std(ddof=1):.3f}: {(sds <= v.std(ddof=1)).mean():.4f}")
