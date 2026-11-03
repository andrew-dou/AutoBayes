"""The Tier G part 3 fixtures (G08 multivariate extensions, G09 gp() extensions, G10 special-term extensions) are what their notes
say, checked on the frozen points against the mathematics written here, independently of autobayes:

- brms's Stan data is the frame's: each response and its weights (one weights column per response, brms's weights_<resp>), a
  Gaussian process's distinct inputs divided by dmax (the largest Euclidean distance between them; none with scale = FALSE) in
  first-appearance order with each row's index, per level of a by factor (its rows, Igp_k_l) and with a numeric by's values
  (Cgp_k), a monotonic term's codes (the ordinal minus 1) and the covariate of its interaction (Csp_k), me(gr = g)'s noisy value
  and noise per level of g with each row's level (Jme_k), and cs()'s design;
- brms's linear predictors are their documented construction: the population terms, the group-level effects (one correlated
  block across the two responses for (1 |p| g), per-threshold intercepts for (cs(1) | g)), bsp times the product of the monotonic
  transforms D * sum(simo[1..x]) and the interaction's covariate, the latent me() values, and each Gaussian process: chol(K +
  1e-12 I) zgp with the squared-exponential, Matern 3/2, Matern 5/2 or exponential kernel of the Stan Functions Reference
  (isotropic in the Euclidean distance, or with iso = FALSE the product of one-dimensional kernels with a length-scale each), or
  the Hilbert-space basis with the Matern spectral density of Rasmussen and Williams (2006, eq. 4.15);
- brms's log density without the Jacobian is its lprior, the standard normal latent variables (z, zgp, zme), me()'s
  normal(Xn | Xme, noise) and the likelihood: a multivariate normal with Sigma = diag(sigma) Rescor diag(sigma) per row under a
  residual correlation (weighted by the first response's weights), independent normals otherwise (each response's weights), the
  adjacent-category or cumulative logit with thresholds b_Intercept minus the category-specific terms;
- brms's posterior_linpred and posterior_epred on the fitted rows, with and without group-level effects, are those linear
  predictors and expected values, row by row (and the approximate GP's on the new rows, its basis at their inputs).
"""
import math
import re

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, stats
from scipy.special import expit, gammainc, gammainccinv, gammaln

from tests.oracle import adapter, datasets, gates
from tests.oracle import fixture as fx
from tests.oracle.adapter import OracleMismatch
from tests.oracle.tolerances import rel_err

FEATURES = ("G08_multivariate_ext", "G09_gp_ext", "G10_special_term_ext")
CASES = [cid for cid in fx.case_ids() if fx.has_fixture(cid) and fx.load_case(cid).get("feature") in FEATURES]
MV = [cid for cid in CASES if "mvbind" in str(fx.load_case(cid)["brms"].get("formula")) or "mvbf" in fx.load_case(cid)["brms"]]
GP = [cid for cid in CASES if fx.load_case(cid)["feature"] == "G09_gp_ext"]

# brms's special terms of each case, as its Stan program writes them per formula part ("" the mean, "_sigma" sigma's): each bsp
# coefficient multiplies the product of the monotonic transforms of its simplexes (numbers k of simo_k and Xmo_k) or the latent
# me() values ("me"), times its interaction's covariate Csp_k where it has one
SPECIAL = {
    "gauss_mo_interaction": {"": [((1,), None), ((2,), "Csp_1")]},
    "gauss_mo_mo_interaction": {"": [((1,), None), ((2,), None), ((3, 4), None)]},
    "gauss_mo_id": {"": [((1,), None), ((2,), "Csp_1")]},
    "gauss_mo_id_sigma": {"": [((1,), None)], "_sigma": [((1,), None)]},
    "gauss_mo_colon_factor": {"": [((1,), "Csp_1"), ((2,), "Csp_1"), ((3,), "Csp_1")]},     # brms 2.23.0: Csp_1 for each level
    "gauss_me_gr": {"": [("me", None)]},
    "gauss_me_interaction": {"": [("me", None), ("me", "Csp_1")]},
}
AS_WRITTEN = {"gauss_mo_colon_factor": {"": [((1,), "Csp_1"), ((2,), "Csp_2"), ((3,), "Csp_3")]}}   # mo(ord):z, one indicator each


def _load(case_id):
    case = fx.load_case(case_id)
    return case, fx.Fixture.load(case_id), fx.read_frame(case, fx.fixture_dir(case_id) / "data.csv")


def _vec(p, name) -> np.ndarray:
    return np.atleast_1d(np.asarray(p.pars[name], dtype=float))


def _formulas(case) -> list[str]:
    form = case["brms"].get("mvbf") or case["brms"]["formula"]
    return [form] if isinstance(form, str) else list(form)


def _responses(case) -> list[str]:
    """The responses of a multivariate case, in brms's order (y, y2); [] for a univariate one."""
    if case["id"] not in MV:
        return []
    return ["y", "y2"]


def _ordinal(case) -> bool:
    return case["brms"]["family"]["family"] in ("acat", "cumulative")


# -- the linear predictors ---------------------------------------------------------------------------------------------------
def _mo(sd, p, k: int, part: str = "") -> np.ndarray:
    """brms's mo(simo_k, Xmo_k) per row: D times the sum of the simplex's first x entries (0 for x = 0)."""
    simo = _vec(p, f"simo{part}_{k}")
    return len(simo) * np.concatenate([[0.0], np.cumsum(simo)])[sd[f"Xmo{part}_{k}"].astype(int)]


def _latent_me(sd, p) -> np.ndarray:
    """me()'s latent value of each row: Xme_1, per level of gr (Jme_1) where there is one."""
    xme = _vec(p, "Xme_1")
    return xme[sd["Jme_1"].astype(int) - 1] if "Jme_1" in sd else xme


def _special(case_id, sd, p, part: str = "", terms=None) -> np.ndarray:
    total = np.zeros(int(sd["N"]))
    for k, (factors, csp) in enumerate((terms or SPECIAL).get(case_id, {}).get(part, ()), start=1):
        value = _latent_me(sd, p) if factors == "me" else np.prod([_mo(sd, p, j, part) for j in factors], axis=0)
        total = total + float(_vec(p, f"bsp{part}")[k - 1]) * value * (sd[csp] if csp else 1.0)
    return total


_K1 = {   # the one-dimensional kernels of the Stan Functions Reference at r = distance / length-scale
    "exp_quad": lambda r: np.exp(-0.5 * r ** 2),
    "matern32": lambda r: (1 + math.sqrt(3) * r) * np.exp(-math.sqrt(3) * r),
    "matern52": lambda r: (1 + math.sqrt(5) * r + 5 * r ** 2 / 3) * np.exp(-math.sqrt(5) * r),
    "exponential": lambda r: np.exp(-r),
}
_NU = {"matern32": 1.5, "matern52": 2.5, "exponential": 0.5}


def _kernel_name(case) -> str:
    m = re.search(r'cov\s*=\s*"(\w+)"', " ".join(_formulas(case)))
    return m.group(1) if m else "exp_quad"


def _gp_cov(X, sdgp: float, lscale: np.ndarray, kernel: str) -> np.ndarray:
    """K + 1e-12 I at the inputs X (rows): isotropic in the Euclidean distance (one length-scale), or with a length-scale per
    dimension the product of one-dimensional kernels (iso = FALSE)."""
    X = np.atleast_2d(X)
    if len(lscale) == 1:
        d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
        K = sdgp ** 2 * _K1[kernel](d / lscale[0])
    else:
        K = sdgp ** 2 * np.prod([_K1[kernel](np.abs(X[:, None, j] - X[None, :, j]) / lscale[j]) for j in range(X.shape[1])], axis=0)
    return K + 1e-12 * np.eye(len(X))


def _spd(w: np.ndarray, sdgp: float, lscale: float, kernel: str) -> np.ndarray:
    """The spectral density at the frequencies w (rows, D columns) of the isotropic kernel (Rasmussen and Williams 2006, eqs.
    4.15 and 4.20): the squared exponential sdgp^2 (2 pi l^2)^(D/2) exp(-l^2 |w|^2 / 2), and the Matern of order nu
    sdgp^2 2^D pi^(D/2) Gamma(nu + D/2) (2 nu)^nu / (Gamma(nu) l^(2 nu)) (2 nu / l^2 + |w|^2)^(-(nu + D/2))."""
    w2 = (np.atleast_2d(w) ** 2).sum(1)
    D = np.atleast_2d(w).shape[1]
    if kernel == "exp_quad":
        return sdgp ** 2 * (2 * math.pi * lscale ** 2) ** (D / 2) * np.exp(-0.5 * lscale ** 2 * w2)
    nu = _NU[kernel]
    log_c = (D * math.log(2) + D / 2 * math.log(math.pi) + gammaln(nu + D / 2) + nu * math.log(2 * nu) - gammaln(nu)
             - 2 * nu * math.log(lscale))
    return sdgp ** 2 * np.exp(log_c - (nu + D / 2) * np.log(2 * nu / lscale ** 2 + w2))


def _gp_ids(sd) -> list[int]:
    return sorted({int(m.group(1)) for k in sd if (m := re.fullmatch(r"Xgp_(\d+)(?:_\d+)?", k))})


def _gp(case, sd, p) -> np.ndarray:
    """Every Gaussian process's contribution to the mean on the fitted rows."""
    total, kernel = np.zeros(int(sd["N"])), _kernel_name(case)
    for k in _gp_ids(sd):
        sdgp, lscale = _vec(p, f"sdgp_{k}"), np.atleast_2d(np.asarray(p.pars[f"lscale_{k}"], dtype=float))
        if f"Igp_{k}_1" in sd:                                        # by = <factor>: one process per level, on its rows
            for lv in range(1, len(sdgp) + 1):
                L = np.linalg.cholesky(_gp_cov(sd[f"Xgp_{k}_{lv}"], sdgp[lv - 1], lscale[lv - 1], kernel))
                f = L @ _vec(p, f"zgp_{k}_{lv}")
                rows = sd[f"Igp_{k}_{lv}"].astype(int) - 1
                total[rows] += sd[f"Cgp_{k}_{lv}"] * f[sd[f"Jgp_{k}_{lv}"].astype(int) - 1]
            continue
        if f"NBgp_{k}" in sd:                                          # the approximate GP: its basis times the scaled weights
            f = np.atleast_2d(sd[f"Xgp_{k}"]) @ (np.sqrt(_spd(sd[f"slambda_{k}"], sdgp[0], lscale[0][0], kernel)) * _vec(p, f"zgp_{k}"))
        else:
            f = np.linalg.cholesky(_gp_cov(sd[f"Xgp_{k}"], sdgp[0], lscale[0], kernel)) @ _vec(p, f"zgp_{k}")
        total = total + (sd[f"Cgp_{k}"] if f"Cgp_{k}" in sd else 1.0) * f[sd[f"Jgp_{k}"].astype(int) - 1]
    return total


_R = re.compile(r"r_(\d+)_(?:(y2?|sigma)_)?(\d+)$")


def _group_terms(case, sd, p, part: str = "") -> np.ndarray:
    """The part's group-level effects per row, r[J] Z for each of its coefficients ("" the mean or "_y", "_y2" a response's);
    the per-threshold intercepts of (cs(1) | g) are not the mean's (``_cs_terms``)."""
    total = np.zeros(int(sd["N"]))
    if case["id"] == "acat_cs_group":
        return total
    owner = part[1:]
    for name in p.pars:
        m = _R.match(name)
        if not m or (m.group(2) or "") != owner:
            continue
        g, c = m.group(1), m.group(3)
        J = sd[f"J_{g}{part if owner in ('y', 'y2') else ''}"].astype(int) - 1
        total = total + _vec(p, name)[J] * sd[f"Z_{g}{part}_{c}"]
    return total


def _eta(case, sd, p, part: str = "", groups: bool = True, special=None) -> np.ndarray:
    """The part's linear predictor on the fitted rows, written uncentred: b_Intercept (a response's b_<resp>_Intercept, sigma's
    b_sigma_Intercept) plus X b, the special terms, the Gaussian processes and the group-level effects (an ordinal model's mean
    has no intercept: its thresholds are b_Intercept)."""
    X = np.atleast_2d(sd[f"X{part}"]).reshape(int(sd["N"]), -1)
    eta = np.zeros(int(sd["N"]))
    if not (part == "" and _ordinal(case)) and f"Intercept{part}" in p.pars:
        eta = eta + float(p.pars[f"b{part}_Intercept"])
        X = X[:, 1:]
    if X.shape[1]:
        eta = eta + X @ _vec(p, f"b{part}")
    eta = eta + _special(case["id"], sd, p, part, special)
    if part == "":
        eta = eta + _gp(case, sd, p)
    return eta + (_group_terms(case, sd, p, part) if groups else 0.0)


def _cs_terms(case, sd, p, groups: bool = True) -> np.ndarray:
    """(rows, thresholds): the category-specific terms, Xcs bcs, plus (cs(1) | g)'s per-threshold intercepts r_1_k[J_1] Z_1_k."""
    out = np.zeros((int(sd["N"]), int(sd["nthres"])))
    if "Xcs" in sd:
        out = out + np.atleast_2d(sd["Xcs"]).reshape(int(sd["N"]), -1) @ np.atleast_2d(p.pars["bcs"])
    if case["id"] == "acat_cs_group" and groups:
        for k in range(1, int(sd["nthres"]) + 1):
            out[:, k - 1] += _vec(p, f"r_1_{k}")[sd["J_1"].astype(int) - 1] * sd[f"Z_1_{k}"]
    return out


def _thresholds(case, sd, p, groups: bool = True) -> np.ndarray:
    return _vec(p, "b_Intercept")[None, :] - _cs_terms(case, sd, p, groups)


def _ordinal_probabilities(case, sd, p, groups: bool = True) -> np.ndarray:
    """(rows, categories): the adjacent-category logit, P(k) proportional to exp(sum_{j < k} (mu - tau_j)), or the cumulative logit,
    P(k) = F(tau_k - mu) - F(tau_{k-1} - mu), with each row's thresholds tau (disc = 1)."""
    mu, tau = _eta(case, sd, p, groups=groups), _thresholds(case, sd, p, groups)
    if case["brms"]["family"]["family"] == "acat":
        logits = np.column_stack([np.zeros(len(mu)), np.cumsum(mu[:, None] - tau, axis=1)])
        return np.exp(logits - np.logaddexp.reduce(logits, axis=1)[:, None])
    F = expit(tau - mu[:, None])
    return np.diff(np.column_stack([np.zeros(len(mu)), F, np.ones(len(mu))]), axis=1)


def _sigma(case, sd, p, part: str = "") -> np.ndarray:
    if "X_sigma" in sd:
        return np.exp(_eta(case, sd, p, "_sigma"))
    return np.full(int(sd["N"]), float(p.pars[f"sigma{part}"]))


def _log_lik(case, sd, p, special=None) -> np.ndarray:
    """Each row's log likelihood at a frozen point, from the mathematics (a multivariate model's summed over its responses)."""
    resp = _responses(case)
    if resp:
        mus = np.column_stack([_eta(case, sd, p, f"_{r}") for r in resp])
        sig = np.array([float(p.pars[f"sigma_{r}"]) for r in resp])
        Y = np.column_stack([sd[f"Y_{r}"] for r in resp])
        if "Rescor" in p.pars:
            S = np.diag(sig) @ np.atleast_2d(p.pars["Rescor"]) @ np.diag(sig)
            ll = np.array([stats.multivariate_normal.logpdf(Y[n], mus[n], S) for n in range(len(Y))])
            return ll * (sd["weights_y"] if "weights_y" in sd else 1.0)
        per = stats.norm.logpdf(Y, mus, sig[None, :])
        if "weights_y" in sd:
            per = per * np.column_stack([sd[f"weights_{r}"] for r in resp])
        return per.sum(1)
    y = sd["Y"]
    if _ordinal(case):
        return np.log(_ordinal_probabilities(case, sd, p)[np.arange(len(y)), y.astype(int) - 1])
    mu = _eta(case, sd, p, special=special)
    return stats.norm.logpdf(y, mu, _sigma(case, sd, p))


def _latent_terms(sd, p) -> float:
    """The standard normal z, zgp and zme, and me()'s normal(Xn | Xme, noise)."""
    out = sum(float(stats.norm.logpdf(np.ravel(np.asarray(v, dtype=float))).sum())
              for k, v in p.pars.items() if re.fullmatch(r"z_\d+|zgp_\d+(_\d+)?|zme_\d+", k))
    if "Xn_1" in sd:
        out += float(stats.norm.logpdf(sd["Xn_1"], _vec(p, "Xme_1"), sd["noise_1"]).sum())
    return out


def _gp_tolerance(case_id) -> float:
    """An exact Gaussian process's Cholesky factor in NumPy and in Stan differ by about cond(K) times the double precision: these
    cases' K have cond(K) up to 9e6 at the frozen points (``test_exact_gp_kernels_are_well_conditioned``), and the log density and
    linear predictors agree within 3e-13 here."""
    return 1e-11 if case_id in GP else 1e-12


# -- the tests -------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("case_id", CASES)
def test_log_density_is_lprior_latent_terms_and_the_likelihood(case_id):
    """At every frozen point brms's log density without the Jacobian is lprior, the latent standard normals (and me()'s noisy
    values) and ``_log_lik`` summed over the rows."""
    case, f, _ = _load(case_id)
    for p in f.points:
        expected = float(p.pars["lprior"]) + _latent_terms(f.standata, p) + float(_log_lik(case, f.standata, p).sum())
        assert rel_err(expected, p.lp_nojac) <= _gp_tolerance(case_id), (case_id, expected, p.lp_nojac)


@pytest.mark.parametrize("case_id", CASES)
def test_fitted_predictions_are_the_linear_predictors(case_id):
    """posterior_linpred on the fitted rows is each response's linear predictor (draws, rows, responses), an ordinal model's mean
    plus each threshold's category-specific terms (draws, rows, thresholds); posterior_epred the mean, or the category
    probabilities; both without the group-level effects under re_formula = NA (also (cs(1) | g)'s, after which a model without
    other category-specific terms has posterior_linpred (draws, rows), the mean alone)."""
    case, f, _ = _load(case_id)
    sd, resp = f.standata, _responses(case)
    tol = max(_gp_tolerance(case_id), 1e-12)
    for kind, groups in (("", True), ("_nogroup", False)):
        lin, ep = f.prediction("fitted", "linpred" + kind), f.prediction("fitted", "epred" + kind)
        if lin is None:
            assert kind == "_nogroup" and not f.groups
            continue
        if resp:
            want = np.stack([np.column_stack([_eta(case, sd, p, f"_{r}", groups) for r in resp]) for p in f.points])
            np.testing.assert_allclose(lin, want, rtol=tol, atol=tol)
            np.testing.assert_allclose(ep, want, rtol=tol, atol=tol)
        elif _ordinal(case):
            if "Xcs" in sd or groups:                               # with category-specific terms: one column per threshold
                want = np.stack([_eta(case, sd, p, groups=groups)[:, None] + _cs_terms(case, sd, p, groups) for p in f.points])
            else:                                                   # (cs(1) | g) dropped by re_formula = NA: the mean alone
                want = np.stack([_eta(case, sd, p, groups=False) for p in f.points])
            np.testing.assert_allclose(lin, want, rtol=tol, atol=tol)
            np.testing.assert_allclose(ep, np.stack([_ordinal_probabilities(case, sd, p, groups) for p in f.points]), rtol=tol, atol=tol)
        else:
            want = np.stack([_eta(case, sd, p, groups=groups) for p in f.points])
            np.testing.assert_allclose(lin, want, rtol=tol, atol=tol)
            np.testing.assert_allclose(ep, want, rtol=tol, atol=tol)


def _distinct_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The distinct rows of ``values`` in first-appearance order and each row's index into them."""
    keys = [tuple(v) for v in values]
    order = list(dict.fromkeys(keys))
    return np.array(order, dtype=float), np.array([order.index(k) for k in keys])


def _gp_inputs(case) -> tuple[list[str], str | None, bool]:
    text = " ".join(_formulas(case))
    args = [a.strip() for a in re.search(r"gp\(([^)]*)\)", text).group(1).split(",")]
    cols = [a for a in args if "=" not in a]
    by = next((a.split("=")[1].strip() for a in args if a.startswith("by")), None)
    return cols, by, "scale = FALSE" not in text


@pytest.mark.parametrize("case_id", GP)
def test_gp_stan_data_is_the_frame_s(case_id):
    """A Gaussian process's Stan data: its distinct inputs (the rows of its columns) divided by dmax, the largest Euclidean distance
    between them (the inputs as they are with scale = FALSE, and no dmax), in first-appearance order, and each row's index Jgp; with
    by = z one process per level of z on that level's rows Igp_k_l (Cgp_k_l = 1) whose inputs are scaled by the level's own dmax;
    with a numeric by its values Cgp_k; the approximate GP's basis sin(pi j (u - cmean + L) / (2 L)) / sqrt(L) at the scaled
    inputs u, cmean their mean over the distinct inputs, L = c and slambda_j = pi j / (2 L)."""
    case, f, frame = _load(case_id)
    sd = f.standata
    cols, by, scaled = _gp_inputs(case)
    raw = frame[cols].to_numpy(dtype=float)

    def check(rows, suffix):
        inputs, index = _distinct_rows(raw[rows])
        dist = np.sqrt(((inputs[:, None, :] - inputs[None, :, :]) ** 2).sum(-1))
        dmax = dist.max() if scaled else 1.0
        assert (f"dmax_{suffix}" in sd) == scaled
        if scaled:
            assert abs(float(sd[f"dmax_{suffix}"]) - dmax) <= 1e-14 * dmax
        np.testing.assert_allclose(np.atleast_2d(sd[f"Xgp_prior_{suffix}"]).reshape(len(inputs), -1), inputs / dmax, rtol=1e-15, atol=0)
        np.testing.assert_array_equal(sd[f"Jgp_{suffix}"].astype(int) - 1, index)
        u = inputs / dmax
        if f"NBgp_{suffix}" in sd:
            L, nb = float(sd[f"Lgp_{suffix}"]), int(sd[f"NBgp_{suffix}"])
            assert L == 1.25 and nb == 10
            cmean = u.mean(0)
            np.testing.assert_allclose(sd[f"cmeans_{suffix}"], cmean, rtol=1e-14, atol=1e-15)
            j = np.arange(1, nb + 1)
            np.testing.assert_allclose(np.atleast_2d(sd[f"slambda_{suffix}"]).ravel(), np.pi * j / (2 * L), rtol=1e-15)
            basis = np.sin(np.pi * j[None, :] * (u - cmean + L) / (2 * L)) / np.sqrt(L)
            np.testing.assert_allclose(sd[f"Xgp_{suffix}"], basis, rtol=0, atol=1e-14)
        else:
            np.testing.assert_allclose(np.atleast_2d(sd[f"Xgp_{suffix}"]).reshape(len(inputs), -1), u, rtol=1e-15, atol=0)

    if by is not None and by in case["brms"]["factors"]:
        levels = case["brms"]["factors"][by]
        assert int(sd["Kgp_1"]) == len(levels) and not any(k == "Xgp_1" for k in sd)
        for lv, level in enumerate(levels, start=1):
            rows = np.flatnonzero(frame[by].to_numpy() == level)
            np.testing.assert_array_equal(sd[f"Igp_1_{lv}"].astype(int) - 1, rows)
            np.testing.assert_array_equal(sd[f"Cgp_1_{lv}"], np.ones(len(rows)))
            check(rows, f"1_{lv}")
        return
    check(np.arange(len(frame)), "1")
    if by is not None:
        np.testing.assert_array_equal(sd["Cgp_1"], frame[by].to_numpy(dtype=float))
    assert int(sd["Dgp_1"]) == len(cols) and int(sd["Kgp_1"]) == 1


@pytest.mark.parametrize("case_id", [c for c in GP if "k = " not in " ".join(_formulas(fx.load_case(c)))])
def test_exact_gp_kernels_are_well_conditioned(case_id):
    """The exact processes' kernel matrices at the frozen points have cond(K) below 1e7 (at most 8.9e6, gauss_gp_by_numeric), so
    that a Cholesky factor computed in double precision by another implementation agrees with Stan's to about 1e-12
    (gauss_gp_exact's note: at cond(K) of 1e13 they differ by 4.5e-6): the inputs are few (four values of ord, the six pairs of (o2, b2), twelve of size) or the kernel rough
    (the exponential's on 36 values of x)."""
    case, f, _ = _load(case_id)
    sd, kernel = f.standata, _kernel_name(case)
    worst = 0.0
    for p in f.points:
        sdgp, lscale = _vec(p, "sdgp_1"), np.atleast_2d(np.asarray(p.pars["lscale_1"], dtype=float))
        keys = [k for k in sd if re.fullmatch(r"Xgp_1(_\d+)?", k)]
        for lv, key in enumerate(sorted(keys)):
            worst = max(worst, np.linalg.cond(_gp_cov(sd[key], sdgp[lv], lscale[lv], kernel)))
    assert worst < 1e7, worst


def _lscale_rule(distances: np.ndarray) -> tuple[float, float]:
    """brms's default lscale prior (D67): the inverse gamma with 1 % of its mass below the smallest distance between distinct inputs
    (floored at 0.01) and 1 % above the largest, solved here to full precision: InvGamma(a, b)'s cdf at x is Q(a, b / x) (the
    upper regularised gamma), so the lower condition gives b = lo Q^-1(a, 0.01) and the upper one is one equation in a."""
    lo, hi = max(float(distances[distances > 0].min()), 0.01), float(distances.max())
    scale = lambda a: lo * gammainccinv(a, 0.01)
    a = optimize.brentq(lambda a: gammainc(a, scale(a) / hi) - 0.01, 1e-3, 1e4, xtol=1e-14, rtol=1e-15)
    return float(a), float(scale(a))


@pytest.mark.parametrize("case_id", GP)
def test_default_lscale_prior_is_the_same_rule_for_every_kernel(case_id):
    """brms's default lscale prior (each fixture's default_prior table, row lscale | gp<cols>[<level or dimension>]) is the
    inverse gamma of D67 on the distances between the process's distinct scaled inputs, whatever the kernel (squared exponential,
    Matern 3/2 and 5/2, exponential): Euclidean distances for gp(o2, b2), one dimension's distances each with iso = FALSE, each
    level's inputs with by = z, the unscaled inputs with scale = FALSE; printed to 6 decimals, within 2e-6 of the exact solution
    (brms's solver stops within about 1e-7 of it, D67); normal(0, 0.5) where there are two distinct inputs (iso = FALSE's binary
    b2 dimension)."""
    case, f, frame = _load(case_id)
    sd = f.standata
    cols, by, _ = _gp_inputs(case)
    rows = {r["coef"]: r["prior"] for r in f.default_prior if r["class"] == "lscale" and r["coef"]}
    blocks = {}
    if by is not None and by in case["brms"]["factors"]:
        for lv, level in enumerate(case["brms"]["factors"][by], start=1):
            blocks[f"gp{''.join(cols)}{by}{level}"] = np.atleast_2d(sd[f"Xgp_prior_1_{lv}"])
    elif "iso = FALSE" in " ".join(_formulas(case)):
        X = np.atleast_2d(sd["Xgp_prior_1"])
        inputs, _ = _distinct_rows(X)
        for j, c in enumerate(cols):
            blocks[f"gp{''.join(cols)}{c}"] = np.unique(inputs[:, j])[:, None]
    else:
        blocks[f"gp{''.join(cols)}{by or ''}"] = np.atleast_2d(sd["Xgp_prior_1"]).reshape(len(_distinct_rows(frame[cols].to_numpy())[0]), -1)
    assert sorted(rows) == sorted(blocks)
    for coef, X in blocks.items():
        X = np.unique(np.atleast_2d(X).reshape(len(X), -1), axis=0)
        if len(X) == 2:                                                # one distance: brms's normal(0, 0.5) (D67)
            assert rows[coef] == "normal(0, 0.5)", (coef, rows[coef])
            continue
        dist = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
        a, b = _lscale_rule(dist)
        printed = [float(v) for v in re.fullmatch(r"inv_gamma\(([\d.]+), ([\d.]+)\)", rows[coef]).groups()]
        assert abs(printed[0] - a) < 2e-6 and abs(printed[1] - b) < 2e-6, (coef, rows[coef], a, b)


def test_matern32_default_priors_enter_lprior_as_printed():
    """gauss_gp_matern32_default_prior leaves sdgp and lscale at brms's defaults: lprior is normal(0, 5) on the intercept, the half
    student_t(3, 0, 2.5) of sdgp, the printed inverse gamma of lscale (the same rule as the squared exponential's), and the half
    normal(0, 1) priors of sd_1 and sigma."""
    case, f, _ = _load("gauss_gp_matern32_default_prior")
    row = next(r["prior"] for r in f.prior_summary if r["class"] == "lscale" and r["coef"])
    a, b = (float(v) for v in re.fullmatch(r"inv_gamma\(([\d.]+), ([\d.]+)\)", row).groups())
    sdgp_row = next(r["prior"] for r in f.prior_summary if r["class"] == "sdgp" and not r["coef"])
    assert sdgp_row == "student_t(3, 0, 2.5)"
    half = lambda x, s: float(stats.norm.logpdf(x, 0, s)) + math.log(2.0)
    for p in f.points:
        expected = (stats.norm.logpdf(float(p.pars["Intercept"]), 0, 5)
                    + float(stats.t.logpdf(_vec(p, "sdgp_1")[0], 3, 0, 2.5)) + math.log(2.0)
                    + float(stats.invgamma.logpdf(float(np.ravel(p.pars["lscale_1"])[0]), a, scale=b))
                    + half(_vec(p, "sd_1")[0], 1.0) + half(float(p.pars["sigma"]), 1.0))
        assert rel_err(float(p.pars["lprior"]), float(expected)) <= 1e-12


def test_approximate_matern_gp_predicts_new_rows_from_its_basis():
    """gauss_gp_approx_matern32's posterior_linpred on the new rows is b_Intercept + the basis at the new inputs (x / dmax - cmean,
    the fitted dmax and cmean) times sqrt(S(slambda)) zgp, plus the group effect of each row's level of g."""
    case, f, frame = _load("gauss_gp_approx_matern32")
    sd = f.standata
    nd = fx.read_frame(case, fx.fixture_dir(case["id"]) / "newdata.csv")
    L, nb = float(sd["Lgp_1"]), int(sd["NBgp_1"])
    u = nd["x"].to_numpy() / float(sd["dmax_1"]) - float(np.ravel(sd["cmeans_1"])[0])
    assert np.abs(u).max() < L
    j = np.arange(1, nb + 1)
    basis = np.sin(np.pi * j[None, :] * (u[:, None] + L) / (2 * L)) / np.sqrt(L)
    levels = next(g["levels"] for g in f.groups if g["group"] == "g")
    J = np.array([levels.index(v) for v in nd["g"]])
    for kind, groups in (("linpred", True), ("linpred_nogroup", False)):
        want = []
        for p in f.points:
            w = np.sqrt(_spd(sd["slambda_1"], _vec(p, "sdgp_1")[0], float(np.ravel(p.pars["lscale_1"])[0]), "matern32")) * _vec(p, "zgp_1")
            want.append(float(p.pars["b_Intercept"]) + basis @ w + (_vec(p, "r_1_1")[J] if groups else 0.0))
        np.testing.assert_allclose(f.prediction("newdata", kind), np.stack(want), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("case_id", [c for c in CASES if c in SPECIAL])
def test_special_term_stan_data_is_the_frame_s(case_id):
    """A monotonic term's codes are its ordinal minus 1 with Jmo its number of steps; an interaction's covariate Csp is the
    numeric column (mo(ord):x: x, me(xobs, sdx):w: w) or, for mo(ord):z, one indicator column per level of z in level order;
    me(xg, sdxg, gr = g)'s noisy values and noise are one per level of g (Nme = 6) and Jme each row's level, its Xn and noise
    those of xobs and sdx row by row without gr."""
    case, f, frame = _load(case_id)
    sd, text = f.standata, " ".join(_formulas(case))
    for part, terms in SPECIAL[case_id].items():
        for factors, csp in terms:
            if factors == "me":
                continue
            for k in factors:
                col = "o2" if (case_id == "gauss_mo_mo_interaction" and k in (2, 4)) else "ord"
                np.testing.assert_array_equal(sd[f"Xmo{part}_{k}"], frame[col].to_numpy() - 1)
                assert int(np.atleast_1d(sd[f"Jmo{part}"])[k - 1]) == frame[col].max() - 1
    if case_id == "gauss_mo_colon_factor":
        for k, level in enumerate(case["brms"]["factors"]["z"], start=1):
            np.testing.assert_array_equal(sd[f"Csp_{k}"], (frame["z"] == level).to_numpy(dtype=float))
    elif "Csp_1" in sd:
        col = re.search(r"(?:\):|\) \* )(\w+)", text).group(1)
        np.testing.assert_array_equal(sd["Csp_1"], frame[col].to_numpy(dtype=float))
    if "Xn_1" in sd and "Jme_1" in sd:
        levels = case["brms"]["factors"]["g"]
        np.testing.assert_array_equal(sd["Jme_1"].astype(int) - 1, [levels.index(v) for v in frame["g"]])
        per_level = frame.drop_duplicates("g").set_index("g").loc[levels]
        assert int(sd["Nme_1"]) == len(levels)
        np.testing.assert_array_equal(sd["Xn_1"], per_level["xg"].to_numpy())
        np.testing.assert_array_equal(sd["noise_1"], per_level["sdxg"].to_numpy())
    elif "Xn_1" in sd:
        np.testing.assert_array_equal(sd["Xn_1"], frame["xobs"].to_numpy())
        np.testing.assert_array_equal(sd["noise_1"], frame["sdx"].to_numpy())


def test_mo_id_shares_one_simplex_within_a_formula_only():
    """mo(ord, id = "a") twice in one formula (gauss_mo_id): one simplex, simo_2 = simo_1 at every point (a transformed parameter,
    not a declared one), its one prior dirichlet(2, 2, 2) (coef "a") once in lprior. The same id in mu's and sigma's
    formulas (gauss_mo_id_sigma) shares nothing: simo_1 and simo_sigma_1 are separate simplexes, each with its own coordinates and
    prior (dirichlet(2, 2, 2) and dirichlet(1.5) in the table, coef "a" with and without dpar sigma); a simplex is declared in brms's
    parameters block (its unconstrained coordinates are a joint block the probe does not reach)."""
    _, f, _ = _load("gauss_mo_id")
    assert "simo_1" in f.declarations and "simo_2" not in f.declarations
    for p in f.points:
        np.testing.assert_array_equal(_vec(p, "simo_2"), _vec(p, "simo_1"))
    rows = [(r["coef"], r["prior"]) for r in f.prior_summary if r["class"] == "simo" and r["prior"]]
    assert rows == [("a", "dirichlet(2, 2, 2)")]
    _, g, _ = _load("gauss_mo_id_sigma")
    assert {"simo_1", "simo_sigma_1"} <= set(g.declarations)
    assert all(np.abs(_vec(p, "simo_1") - _vec(p, "simo_sigma_1")).max() > 1e-3 for p in g.points)
    rows = {(r["coef"], r["dpar"]): r["prior"] for r in g.prior_summary if r["class"] == "simo" and r["prior"]}
    assert rows == {("a", ""): "dirichlet(2, 2, 2)", ("a", "sigma"): "dirichlet(1.5)"}


def test_mo_colon_factor_multiplies_every_level_by_one_indicator():
    """Upstream: brms 2.23.0 builds y ~ x + mo(ord):z with one simplex and one coefficient per level of z and the Stan data of each
    level's indicator (Csp_1, Csp_2, Csp_3), but its Stan program multiplies all three terms by Csp_1: the log density at every
    frozen point is that model's (``SPECIAL``), and differs from the interaction as written (each level's term times its own
    indicator, ``AS_WRITTEN``) by 1.03 to 872 at the frozen points."""
    case, f, _ = _load("gauss_mo_colon_factor")
    sd = f.standata
    assert {"Csp_1", "Csp_2", "Csp_3"} <= set(sd)
    for p in f.points:
        rest = float(p.pars["lprior"]) + _latent_terms(sd, p)
        as_written = rest + float(_log_lik(case, sd, p, special=AS_WRITTEN).sum())
        assert abs(as_written - p.lp_nojac) > 1.0


def test_me_gr_has_one_latent_value_per_level():
    """me(xg, sdxg, gr = g): six latent values Xme_1 = meanme_1 + sdme_1 zme_1, one per level of g, each row taking its level's."""
    _, f, _ = _load("gauss_me_gr")
    for p in f.points:
        assert _vec(p, "zme_1").shape == (6,)
        np.testing.assert_allclose(_vec(p, "Xme_1"), _vec(p, "meanme_1")[0] + _vec(p, "sdme_1")[0] * _vec(p, "zme_1"), rtol=1e-15)


@pytest.mark.parametrize("case_id", MV)
def test_mv_stan_data_is_the_frame_s(case_id):
    """Each response's Y and design (bf(y ~ x + (1 | g)) + bf(y2 ~ z): X_y has x, X_y2 the treatment coding of z; mvbind() one
    design for both), weights(w) as weights_y and weights_y2, and nrescor exactly with a residual correlation."""
    case, f, frame = _load(case_id)
    sd = f.standata
    for r in ("y", "y2"):
        np.testing.assert_array_equal(sd[f"Y_{r}"], frame[r].to_numpy())
        if "weights(" in " ".join(_formulas(case)):
            np.testing.assert_array_equal(sd[f"weights_{r}"], frame["w"].to_numpy())
        else:
            assert f"weights_{r}" not in sd
    assert f.colnames("X_y") == ["Intercept", "x"]
    assert f.colnames("X_y2") == (["Intercept", "zq", "zr"] if case_id == "gauss_mv_separate_rescor" else ["Intercept", "x"])
    assert ("nrescor" in sd) == bool(case["brms"]["rescor"])


def test_mv_weights_with_rescor_weight_the_joint_density_by_the_first_response_s_weights():
    """brms takes weights in a multivariate model with a residual correlation (it requires every response's weights to be the same
    column) and weights each row's joint multivariate normal density once, by weights_y: at every frozen point the log density is
    that, and not the joint density weighted by the product or the sum of both responses' weights."""
    case, f, frame = _load("gauss_mv_weights_rescor")
    sd = f.standata
    np.testing.assert_array_equal(sd["weights_y"], frame["w"].to_numpy())
    np.testing.assert_array_equal(sd["weights_y2"], frame["w"].to_numpy())
    for p in f.points:
        rest = float(p.pars["lprior"]) + _latent_terms(sd, p)
        joint = _log_lik(case, sd, p) / sd["weights_y"]
        assert rel_err(rest + float((sd["weights_y"] * joint).sum()), p.lp_nojac) <= 1e-12
        assert abs(rest + float((2 * sd["weights_y"] * joint).sum()) - p.lp_nojac) > 1e-3


def test_mv_id_across_responses_is_one_correlated_block():
    """(1 |p| g) in mvbind(y, y2) ~ x: one group term of g with two coefficients, y's intercept and y2's (sd_1 of length 2, one
    cholesky_factor_corr[2] L_1), r_1_y_1 and r_1_y2_2 the two columns of (diag(sd_1) L_1 z_1)'."""
    for cid in ("gauss_mv_id_across", "gauss_mv_id_across_rescor"):
        _, f, _ = _load(cid)
        assert sorted((g["resp"], g["coef"], g["id"], g["cn"]) for g in f.groups) == [("y", "Intercept", 1, 1), ("y2", "Intercept", 1, 2)]
        for p in f.points:
            r = (np.diag(_vec(p, "sd_1")) @ np.atleast_2d(p.pars["L_1"]) @ np.atleast_2d(p.pars["z_1"])).T
            np.testing.assert_allclose(np.column_stack([_vec(p, "r_1_y_1"), _vec(p, "r_1_y2_2")]), r, rtol=1e-13, atol=1e-15)


def test_mv_id_across_sd_priors_follow_each_response():
    """With one ID across the two responses, brms applies each response's sd | g | resp prior to that response's entry of sd_1
    (normal(0, 1) on sd_1[1], y's; normal(0, 0.5) on sd_1[2], y2's), unlike one ID across mu and sigma, where the mean's prior
    covers the whole vector (gauss_sigma_shared_id, D187): lprior is the sum below at every frozen point, the 2 by 2 LKJ(2) of L_1
    (and of Lrescor) log(3/4) + 2 log L[2, 2]."""
    half = lambda x, s: float(stats.norm.logpdf(x, 0, s)) + math.log(2.0)
    lkj2 = lambda L: math.log(0.75) + 2.0 * math.log(np.atleast_2d(L)[1, 1])
    for cid in ("gauss_mv_id_across", "gauss_mv_id_across_rescor"):
        _, f, _ = _load(cid)
        for p in f.points:
            sd1 = _vec(p, "sd_1")
            expected = half(sd1[0], 1.0) + half(sd1[1], 0.5) + lkj2(p.pars["L_1"])
            for r in ("y", "y2"):
                expected += (float(stats.norm.logpdf(float(p.pars[f"Intercept_{r}"]), 0, 5)) + float(stats.norm.logpdf(_vec(p, f"b_{r}"), 0, 2).sum())
                             + half(float(p.pars[f"sigma_{r}"]), 1.0))
            if "Lrescor" in p.pars:
                expected += lkj2(p.pars["Lrescor"])
            assert rel_err(float(p.pars["lprior"]), expected) <= 1e-12, (cid, float(p.pars["lprior"]), expected)


def test_cumulative_cs_thresholds_cross_and_probabilities_go_negative():
    """cs(x) in the cumulative family (brms: "experimental"): the ordered Intercept minus x bcs is not ordered at every row; at the
    frozen points (drawn from (-0.6, 0.6), where brms's log density is finite; on (-1.2, 1.2) it is NaN at most points) some rows
    have crossed thresholds, and brms's posterior_epred there has negative category probabilities (rows sum to 1)."""
    case, f, _ = _load("cumulative_cs")
    sd = f.standata
    crossed = [int((np.diff(_thresholds(case, sd, p), axis=1) < 0).any(1).sum()) for p in f.points]
    assert sum(c > 0 for c in crossed) >= 2, crossed
    ep = f.prediction("fitted", "epred")
    assert ep.min() < 0
    np.testing.assert_allclose(ep.sum(-1), 1.0, rtol=0, atol=1e-12)


@pytest.mark.parametrize("case_id, comparison", [
    ("gauss_mo_interaction", "check_special_covariates"), ("gauss_me_interaction", "check_special_covariates"),
    ("gauss_me_gr", "check_me_groups"), ("gauss_gp_by_factor", "check_gp_by"), ("gauss_gp_by_numeric", "check_gp_by"),
])
def test_new_stan_data_is_not_left_unchecked(case_id, comparison, monkeypatch):
    """Until gates.py compares brms's Csp_k (a special term's interaction covariate), Jme_k (me(gr = )'s level index) and Cgp_k,
    Igp_k_l (gp(by = )) in G1, a bundle of a case that records them fails with the oracle's mismatch naming the missing
    comparison, whatever model it is built with."""
    if hasattr(gates, comparison):
        pytest.skip(f"{comparison} has landed")
    monkeypatch.setattr(adapter, "in_data_order", lambda case, fixture, model, data: fixture)
    with pytest.raises(OracleMismatch, match=comparison):
        gates.Bundle(case_id, model=object())


def test_tier_g3_extends_tier_g2_column_for_column():
    spec = {"seed": 3201, "n": 36, "newdata": {"seed": 3301, "n": 5}}
    data, nd = datasets.make({"generator": "tier_g3", **spec})
    base, base_nd = datasets.make({"generator": "tier_g2", **spec})
    added = ["o2", "b2", "xg", "sdxg"]
    assert list(data.columns) == [*base.columns, *added]
    assert list(nd.columns) == [*base_nd.columns, "o2", "b2", "xg", "sdxg"]
    pd.testing.assert_frame_equal(data[base.columns], base)
    pd.testing.assert_frame_equal(nd[base_nd.columns], base_nd)
    assert sorted(set(zip(data["o2"], data["b2"]))) == [(a, b) for a in (1, 2, 3) for b in (0, 1)]
    assert (data.groupby("g")[["xg", "sdxg"]].nunique() == 1).all().all() and data["sdxg"].between(0.2, 0.6).all()
    per_level = data.drop_duplicates("g").set_index("g")
    np.testing.assert_array_equal(nd["xg"].to_numpy(), per_level.loc[nd["g"], "xg"].to_numpy())
    assert not data[added].isna().any().any() and not nd[added].isna().any().any()
