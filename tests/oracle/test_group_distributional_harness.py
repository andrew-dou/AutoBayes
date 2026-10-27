"""The Tier G part 2 fixtures (G05a group-term syntax, G05b gr(cov = ), gr(dist = "student"), mm() slopes, mmc() and scale =
FALSE, G06 group-level terms, shared IDs and offsets in distributional formulas) are what their notes say, checked on the frozen
points against the mathematics written here, independently of autobayes:

- brms's Stan data is the frame's: each group-level coefficient's covariate (1 for an intercept, the design column of a factor,
  transform or interaction slope, the raw column of a slope without its population term), the levels of g:h:f2 as brms labels
  them, mm()'s memberships, weights (1/2 each, or the raw columns with scale = FALSE) and per-member covariates (mmc()), a
  distributional formula's offset, and gr(cov = A)'s Lcov, the lower Cholesky factor of A on the factor's levels;
- brms's group-level effects are its documented construction from sd, z, the correlation factor L, the known covariance's
  Cholesky factor (Kronecker product) and gr(dist = "student")'s scale mixture sqrt(df udf), with the one exception of brms
  2.23.0's ``scale_r_cor_cov``, which drops the negative entries of Lcov (``gauss_gr_cov_slope_signed``);
- brms's log density without the Jacobian is its lprior, the standard normal z, gr(dist = "student")'s inverse chi-square udf and
  the likelihood of every row from each parameter's linear predictor (the population terms, the offset, the group-level effects
  of that parameter) under its link: sigma and disc exp, nu 1 + exp (brms's logm1), zi inverse logit;
- brms's posterior_linpred and posterior_epred on the fitted rows, with and without group-level effects (re_formula = NA drops
  them from every parameter), and a post block's log_lik, are those linear predictors and likelihoods row by row.
"""
import re

import numpy as np
import pytest
from scipy import stats
from scipy.special import expit

from tests.oracle import adapter, gates
from tests.oracle import fixture as fx
from tests.oracle.adapter import OracleMismatch
from tests.oracle.tolerances import rel_err

FEATURES = ("G05a_group_term_syntax", "G05b_gr_cov_dist", "G06_distributional_groups")
CASES = [cid for cid in fx.case_ids() if fx.has_fixture(cid) and fx.load_case(cid).get("feature") in FEATURES]
DPARS = ("sigma", "nu", "zi", "disc")
_R = re.compile(rf"r_(\d+)_(?:({'|'.join(DPARS)})_)?(\d+)$")


def _load(case_id):
    case = fx.load_case(case_id)
    return case, fx.Fixture.load(case_id), fx.read_frame(case, fx.fixture_dir(case_id) / "data.csv")


def _vec(p, name) -> np.ndarray:
    return np.atleast_1d(np.asarray(p.pars[name], dtype=float))


def _mo(sd, p) -> np.ndarray:
    """brms's mo(simo_1, Xmo_1) per row: D times the sum of the simplex's first i entries (0 for i = 0); one monotonic term."""
    assert int(sd["Imo"]) == 1
    simo = _vec(p, "simo_1")
    return len(simo) * np.concatenate([[0.0], np.cumsum(simo)])[sd["Xmo_1"].astype(int)]


def _coefficients(p, dpar=""):
    """brms's group-level coefficients of a parameter ("" the mean), as (id, cn, the r_ variable's name)."""
    out = []
    for name in p.pars:
        m = _R.match(name)
        if m and (m.group(2) or "") == dpar:
            out.append((int(m.group(1)), int(m.group(3)), name))
    return sorted(out)


def _group_terms(sd, p, dpar="") -> np.ndarray:
    """The parameter's group-level effects per row: r[J] Z for each coefficient, summed over the members of an mm() term with its
    weights W; a monotonic varying slope has no Z, brms multiplies its r by the population term's mo(simo, Xmo)."""
    total = np.zeros(int(sd["N"]))
    pre = f"{dpar}_" if dpar else ""
    for g, c, name in _coefficients(p, dpar):
        r = _vec(p, name)
        if f"J_{g}" in sd:
            Z = sd[f"Z_{g}_{pre}{c}"] if f"Z_{g}_{pre}{c}" in sd else _mo(sd, p)
            total = total + r[sd[f"J_{g}"].astype(int) - 1] * Z
            continue
        k = 1
        while f"J_{g}_{k}" in sd:
            total = total + sd[f"W_{g}_{k}"] * r[sd[f"J_{g}_{k}"].astype(int) - 1] * sd[f"Z_{g}_{pre}{c}_{k}"]
            k += 1
        assert k > 2, f"{name}: neither J_{g} nor the members J_{g}_k"
    return total


def _eta(sd, p, dpar="", groups=True) -> np.ndarray:
    """brms's linear predictor of a parameter on the fitted rows, written uncentred: the raw-scale intercept b_Intercept plus X b,
    the offset, the monotonic term and the group-level effects (a cumulative model's mean has no intercept: its thresholds are
    b_Intercept)."""
    sfx = f"_{dpar}" if dpar else ""
    X = sd[f"X{sfx}"]
    eta = np.zeros(int(sd["N"]))
    if f"Intercept{sfx}" in p.pars and not (dpar == "" and "nthres" in sd):
        eta = eta + float(p.pars[f"b{sfx}_Intercept" if dpar else "b_Intercept"])
        X = X[:, 1:]
    if X.shape[1]:
        eta = eta + X @ _vec(p, f"b{sfx}")
    if f"offsets{sfx}" in sd:
        eta = eta + sd[f"offsets{sfx}"]
    if dpar == "" and "bsp" in p.pars:
        eta = eta + float(_vec(p, "bsp")[0]) * _mo(sd, p)
    return eta + (_group_terms(sd, p, dpar) if groups else 0.0)


def _parameters(case, sd, p, groups=True) -> dict:
    """Every parameter of the family per row, on its own scale: a formula's under its link, else the point's constant."""
    out = {"mu": _eta(sd, p, groups=groups)}
    link = {"sigma": np.exp, "nu": lambda e: 1.0 + np.exp(e), "zi": expit, "disc": np.exp}
    for dpar in DPARS:
        if f"X_{dpar}" in sd:
            out[dpar] = link[dpar](_eta(sd, p, dpar, groups=groups))
        elif dpar in p.pars:
            out[dpar] = np.full(int(sd["N"]), float(p.pars[dpar]))
    return out


def _cumulative_probabilities(mu, disc, thresholds) -> np.ndarray:
    """(rows, categories): brms's cumulative logit with discrimination, P(Y <= k) = inv_logit(disc (theta_k - mu))."""
    F = expit(disc[:, None] * (thresholds[None, :] - mu[:, None]))
    cdf = np.column_stack([np.zeros(len(mu)), F, np.ones(len(mu))])
    return np.diff(cdf, axis=1)


def _log_lik(case, sd, p) -> np.ndarray:
    """Each row's log likelihood at a frozen point, from the mathematics."""
    family = case["brms"]["family"]["family"]
    y, par = sd["Y"], _parameters(case, sd, p)
    if family == "gaussian":
        return stats.norm.logpdf(y, par["mu"], par["sigma"])
    if family == "student":
        return stats.t.logpdf(y, par["nu"], par["mu"], par["sigma"])
    if family == "zero_inflated_poisson":
        lam, zi = np.exp(par["mu"]), par["zi"]
        return np.where(y == 0, np.log(zi + (1.0 - zi) * np.exp(-lam)), np.log1p(-zi) + stats.poisson.logpmf(y, lam))
    if family == "cumulative":
        probs = _cumulative_probabilities(par["mu"], par.get("disc", np.ones(len(y))), _vec(p, "b_Intercept"))
        return np.log(probs[np.arange(len(y)), y.astype(int) - 1])
    raise AssertionError(f"no hand-written likelihood of {family}")


def _epred(case, sd, p, groups=True) -> np.ndarray:
    family = case["brms"]["family"]["family"]
    par = _parameters(case, sd, p, groups=groups)
    if family in ("gaussian", "student"):
        return par["mu"]
    if family == "zero_inflated_poisson":
        return (1.0 - par["zi"]) * np.exp(par["mu"])
    return _cumulative_probabilities(par["mu"], par.get("disc", np.ones(int(sd["N"]))), _vec(p, "b_Intercept"))


def _r_of(sd, p, g: int, *, drop_negative: bool = False) -> np.ndarray:
    """Group ``g``'s effects (levels, coefficients) from the mathematics: diag(sd) L z per level (L the identity without
    correlations), times sqrt(df udf) per level for gr(dist = "student"); with gr(cov = A), Lcov z for one coefficient and
    (Lcov kron diag(sd) L) vec(z) for several (A kron Sigma), Lcov's negative entries dropped when ``drop_negative`` (brms
    2.23.0's scale_r_cor_cov)."""
    s, z = _vec(p, f"sd_{g}"), np.atleast_2d(np.asarray(p.pars[f"z_{g}"], dtype=float))      # z: (coefficients, levels)
    L = np.asarray(p.pars[f"L_{g}"], dtype=float) if f"L_{g}" in p.pars else np.eye(len(s))
    LC = np.diag(s) @ L
    if f"Lcov_{g}" in sd:
        Lcov = sd[f"Lcov_{g}"]
        if len(s) == 1:                                                # a dense product, every entry of Lcov
            return (s[0] * (Lcov @ z[0]))[:, None]
        if drop_negative:
            Lcov = np.where(Lcov > 1e-10, Lcov, 0.0)
        return (np.kron(Lcov, LC) @ z.flatten(order="F")).reshape(z.shape[1], z.shape[0])
    r = (LC @ z).T
    if f"df_{g}" in p.pars:
        r = r * np.sqrt(float(p.pars[f"df_{g}"]) * _vec(p, f"udf_{g}"))[:, None]
    return r


def _r_brms(p, g: int) -> np.ndarray:
    """brms's effects of group ``g`` (levels, coefficients), every parameter's coefficients in brms's order (cn)."""
    cols = sorted((int(m.group(3)), name) for name in p.pars if (m := _R.match(name)) and int(m.group(1)) == g)
    return np.column_stack([_vec(p, name) for _, name in cols])


@pytest.mark.parametrize("case_id", CASES)
def test_group_effects_are_brms_s_construction(case_id):
    """Every group's effects at every frozen point are ``_r_of``'s, to rounding, as brms 2.23.0 builds them (a known covariance's
    negative Cholesky entries dropped with several coefficients); brms's transformed dfm (gr(dist = "student")) is sqrt(df udf)."""
    case, f, _ = _load(case_id)
    sd = f.standata
    ids = sorted({int(m.group(1)) for name in f.points[0].pars if (m := _R.match(name))})
    assert ids
    for p in f.points:
        for g in ids:
            np.testing.assert_allclose(_r_brms(p, g), _r_of(sd, p, g, drop_negative=True), rtol=0, atol=1e-12)
            if f"df_{g}" in p.pars:
                np.testing.assert_allclose(_vec(p, f"dfm_{g}"), np.sqrt(float(p.pars[f"df_{g}"]) * _vec(p, f"udf_{g}")), rtol=1e-14)


def test_gr_cov_slopes_drop_the_negative_entries_of_lcov():
    """Upstream: with gr(g, cov = A) and more than one coefficient brms 2.23.0 builds the effects in ``scale_r_cor_cov``, which adds
    Lcov[i, j] LC[k, l] z only where Lcov[i, j] > 1e-10, so a negative entry of A's Cholesky factor (A = ``g_relationship``, five
    of them) is dropped: the effects at the frozen points differ from (Lcov kron diag(sd) L) vec(z) by up to 0.65, and their
    implied covariance is Lcov+ Lcov+' kron Sigma, not A kron Sigma (Lcov+ Lcov+'[b, c] = 0.25 for A[b, c] = 0.1). With A's
    Cholesky factor non-negative (``g_covariance``, gauss_gr_cov_slope) and with one coefficient (gauss_gr_cov, a dense
    product) brms's effects are the Kronecker product's."""
    case, f, _ = _load("gauss_gr_cov_slope_signed")
    sd = f.standata
    Lcov = sd["Lcov_1"]
    assert int((Lcov < -1e-10).sum()) == 5
    worst = max(np.max(np.abs(_r_brms(p, 1) - _r_of(sd, p, 1))) for p in f.points)
    assert 0.1 < worst < 1.0
    plus = np.where(Lcov > 1e-10, Lcov, 0.0)
    A = fx.read_data2(case, fx.fixture_dir(case["id"]))["A"].to_numpy()
    assert abs((plus @ plus.T)[1, 2] - 0.25) < 1e-12 and A[1, 2] == 0.1
    for other in ("gauss_gr_cov_slope", "gauss_gr_cov"):
        _, g, _ = _load(other)
        for p in g.points:
            np.testing.assert_allclose(_r_brms(p, 1), _r_of(g.standata, p, 1), rtol=0, atol=1e-12)


@pytest.mark.parametrize("case_id", CASES)
def test_log_density_is_lprior_z_udf_and_the_likelihood(case_id):
    """At every frozen point brms's log density without the Jacobian is lprior, the standard normal of every z, gr(dist =
    "student")'s inv_chi_square(udf | df) and ``_log_lik`` summed over the rows."""
    case, f, _ = _load(case_id)
    for p in f.points:
        z = sum(float(stats.norm.logpdf(np.ravel(v)).sum()) for k, v in p.pars.items() if re.fullmatch(r"z_\d+", k))
        udf = sum(float(stats.invgamma.logpdf(_vec(p, k), float(p.pars["df_" + k[4:]]) / 2.0, scale=0.5).sum())
                  for k in p.pars if re.fullmatch(r"udf_\d+", k))
        expected = float(p.pars["lprior"]) + z + udf + float(_log_lik(case, f.standata, p).sum())
        assert rel_err(expected, p.lp_nojac) <= 1e-12, (case_id, expected, p.lp_nojac)


@pytest.mark.parametrize("case_id", CASES)
def test_fitted_predictions_are_the_linear_predictors(case_id):
    """posterior_linpred on the fitted rows is the mean's linear predictor (``_eta``), posterior_epred the family's mean (the
    cumulative's category probabilities with disc), each without every parameter's group-level effects under re_formula = NA."""
    case, f, _ = _load(case_id)
    sd = f.standata
    for kind, groups in (("", True), ("_nogroup", False)):
        lin, ep = f.prediction("fitted", "linpred" + kind), f.prediction("fitted", "epred" + kind)
        np.testing.assert_allclose(lin, np.stack([_eta(sd, p, groups=groups) for p in f.points]), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(ep, np.stack([_epred(case, sd, p, groups=groups) for p in f.points]), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("case_id", [c for c in CASES if "log_lik" in fx.load_case(c)["brms"].get("extras", [])])
def test_log_lik_is_each_row_s_likelihood(case_id):
    """A post block's log_lik (draws, rows) is ``_log_lik`` at each frozen point: a distributional parameter's group-level effects
    enter it."""
    case, f, _ = _load(case_id)
    np.testing.assert_allclose(fx.decode(f.raw["post"]["log_lik"]), np.stack([_log_lik(case, f.standata, p) for p in f.points]),
                               rtol=1e-12, atol=1e-12)


def _formula_parts(case) -> list[str]:
    form = case["brms"]["formula"]
    return [form] if isinstance(form, str) else list(form)


@pytest.mark.parametrize("case_id", CASES)
def test_stan_data_is_the_frame_s(case_id):
    """brms's group-level data is the frame's: a coefficient's covariate Z is 1 for an intercept, the design column of the same
    name (a factor, transform or interaction slope) or the raw column (a slope without its population term); J indexes brms's
    levels, which for g:h:f2 are the labels joined by underscores; mm()'s weights are 1/2 each without weights and the raw
    columns with scale = FALSE, its per-member Z of mmc(x1, x2) the columns x1 and x2; a distributional offset is its column;
    Lcov is the lower Cholesky factor of A on the factor's levels."""
    case, f, frame = _load(case_id)
    sd, text = f.standata, " ".join(_formula_parts(case))
    xcols = {dp: f.colnames(f"X_{dp}" if dp else "X") for dp in ("", *DPARS) if (f"X_{dp}" if dp else "X") in sd}
    for grp in f.groups:
        g, c, cn = grp["id"], grp["coef"], grp["cn"]
        dpar = grp.get("dpar") or next((dp for dp in DPARS if f"Z_{g}_{dp}_{cn}" in sd), "")
        pre = f"{dpar}_" if dpar else ""
        if f"J_{g}" not in sd:                                         # mm(g, g2, ...)
            weights = re.search(r"weights\s*=\s*cbind\((\w+),\s*(\w+)\)", text)
            for k, member in enumerate(("g", "g2"), start=1):
                assert np.array_equal(sd[f"J_{g}_{k}"], [grp["levels"].index(v) + 1 for v in frame[member]])
                W = frame[weights.group(k)].to_numpy() if weights else np.full(len(frame), 0.5)
                assert "scale = FALSE" in text or not weights
                np.testing.assert_array_equal(sd[f"W_{g}_{k}"], W)
                Z = sd[f"Z_{g}_{pre}{cn}_{k}"]
                mmc = re.search(r"mmc\((\w+),\s*(\w+)\)", text)
                expected = 1.0 if c == "Intercept" else frame[mmc.group(k)] if c.startswith("mmc") else frame[c]
                np.testing.assert_array_equal(Z, np.broadcast_to(np.asarray(expected, dtype=float), Z.shape))
            continue
        parts = grp["group"].split(":")
        labels = ["_".join(row) for row in frame[parts].astype(str).to_numpy()]
        assert np.array_equal(sd[f"J_{g}"], [grp["levels"].index(v) + 1 for v in labels])
        key = f"Z_{g}_{pre}{cn}"
        if c.startswith("mo"):
            assert key not in sd                                       # mo(): r multiplies mo(simo, Xmo) (_group_terms)
        elif c == "Intercept":
            assert np.array_equal(sd[key], np.ones(len(frame)))
        elif c in xcols[dpar]:
            assert np.array_equal(sd[key], sd[f"X_{dpar}" if dpar else "X"][:, xcols[dpar].index(c)])
        else:
            assert np.array_equal(sd[key], frame[c].to_numpy(dtype=float))   # a slope without its population term
    offsets = {}
    for part in _formula_parts(case)[1:]:                              # a distributional formula's offset(col)
        dpar, rhs = re.match(r"\s*(\w+)\s*~(.*)", part).groups()
        if (off := re.search(r"\boffset\((\w+)\)", rhs)):
            offsets[dpar] = off.group(1)
    assert sorted(offsets) == sorted(k[len("offsets_"):] for k in sd if k.startswith("offsets_"))
    for dpar, col in offsets.items():
        assert np.array_equal(sd[f"offsets_{dpar}"], frame[col].to_numpy())
    for g in {grp["id"] for grp in f.groups}:
        if f"Lcov_{g}" in sd:
            A = fx.read_data2(case, fx.fixture_dir(case_id))["A"]
            levels = next(grp["levels"] for grp in f.groups if grp["id"] == g)
            np.testing.assert_allclose(sd[f"Lcov_{g}"], np.linalg.cholesky(A.loc[levels, levels].to_numpy()), rtol=0, atol=1e-14)


def test_student_group_effects_carry_df_s_truncated_gamma_prior():
    """gr(g, dist = "student"): lprior holds df_1's gamma(2, 0.1) truncated to df > 1 (its normalising constant
    -log(1 - Gamma(1 | 2, 0.1)) included) beside the other priors; udf_1's inv_chi_square(df_1) is in the model block
    (``test_log_density_is_lprior_z_udf_and_the_likelihood``)."""
    _, f, _ = _load("gauss_gr_student")
    half = np.log(2.0)                                                 # normal(0, s) on a positive parameter, truncated at 0
    gamma = stats.gamma(2.0, scale=10.0)
    for p in f.points:
        expected = (stats.norm.logpdf(float(p.pars["Intercept"]), 0, 5) + stats.norm.logpdf(_vec(p, "b"), 0, 2).sum()
                    + stats.norm.logpdf(float(p.pars["sigma"]), 0, 1) + half + stats.norm.logpdf(_vec(p, "sd_1"), 0, 1).sum() + half
                    + gamma.logpdf(float(p.pars["df_1"])) - gamma.logsf(1.0))
        assert rel_err(float(p.pars["lprior"]), float(expected)) <= 1e-12


def test_shared_id_group_sd_prior_also_covers_sigma_s_sd():
    """Upstream: in ``bf(y ~ x + (1 |p| g), sigma ~ z + (1 |p| g))`` with ``set_prior("normal(0, 1)", class = "sd", group = "g")``
    (the mean's SDs of g) and ``set_prior("normal(0, 0.5)", class = "sd", dpar = "sigma")`` (sigma's SDs), brms 2.23.0's prior
    table says sigma's SD of g inherits normal(0, 0.5) (its row ``sd | group g | dpar sigma`` is empty), but its lprior applies
    normal(0, 1) to the whole sd_1 vector, sigma's coefficient included: lprior is the sum below with normal(0, 1) on both SDs,
    and not with normal(0, 0.5) on sd_1[2], at every frozen point (2x2 LKJ(2): log(3/4) + 2 log L[2, 2]). Written with the group,
    ``set_prior("normal(0, 0.5)", class = "sd", group = "g", dpar = "sigma")``, brms applies it to sigma's SDs
    (gauss_sigma_shared_id_slopes: ``normal_lpdf(sd_1[3] | 0, 0.5)``)."""
    case, f, _ = _load("gauss_sigma_shared_id")
    rows = {(r["class"], r["group"], r["dpar"]): r["prior"] for r in f.prior_summary if r["class"] == "sd" and not r["coef"]}
    assert rows[("sd", "g", "")] == "normal(0, 1)" and rows[("sd", "", "sigma")] == "normal(0, 0.5)" and rows[("sd", "g", "sigma")] == ""
    half = lambda x, s: float(stats.norm.logpdf(x, 0, s)) + np.log(2.0)
    for p in f.points:
        s = _vec(p, "sd_1")
        rest = (stats.norm.logpdf(float(p.pars["Intercept"]), 0, 5) + stats.norm.logpdf(_vec(p, "b"), 0, 2).sum()
                + stats.norm.logpdf(float(p.pars["Intercept_sigma"]), 0, 1) + stats.norm.logpdf(_vec(p, "b_sigma"), 0, 0.5).sum()
                + half(s[0], 1.0) + np.log(0.75) + 2.0 * np.log(np.asarray(p.pars["L_1"])[1, 1]))
        assert rel_err(float(p.pars["lprior"]), float(rest + half(s[1], 1.0))) <= 1e-12
        assert abs(float(p.pars["lprior"]) - float(rest + half(s[1], 0.5))) > 0.1
    _, g, _ = _load("gauss_sigma_shared_id_slopes")
    rows = {(r["class"], r["group"], r["dpar"]): r["prior"] for r in g.prior_summary if r["class"] == "sd" and not r["coef"]}
    assert rows[("sd", "g", "")] == "normal(0, 1)" and rows[("sd", "g", "sigma")] == "normal(0, 0.5)"


@pytest.mark.parametrize("case_id, sds", [
    ("gauss_split_same_factor", (("sd_1", 1.0), ("sd_2", 0.5))),      # sd | g, and sd | g | x for the second term of g
    ("gauss_slope_only", (("sd_1", 0.5), ("sd_2", 1.0))),             # sd | g (its one coefficient, x), sd | h
    ("gauss_three_level", (("sd_1", 1.0), ("sd_2", 0.5), ("sd_3", 0.3))),   # sd | g, sd | g:h, sd | g:h:f2
])
def test_group_sd_priors_follow_the_group_and_the_coefficient(case_id, sds):
    """brms puts each group term's SD prior by its group and coefficient: two terms of one factor take the factor's prior and
    the coefficient's (``(1 | g) + (0 + x | g)``: normal(0, 1) on ID 1, normal(0, 0.5) on ID 2's x), and g/h/f2 one prior per
    expanded group; lprior is these half normals beside normal(0, 5), normal(0, 2) and sigma's half normal(0, 1)."""
    _, f, _ = _load(case_id)
    half = lambda x, s: float((stats.norm.logpdf(np.ravel(x), 0, s) + np.log(2.0)).sum())
    for p in f.points:
        expected = (stats.norm.logpdf(float(p.pars["Intercept"]), 0, 5) + stats.norm.logpdf(_vec(p, "b"), 0, 2).sum()
                    + half(p.pars["sigma"], 1.0) + sum(half(p.pars[k], s) for k, s in sds))
        assert rel_err(float(p.pars["lprior"]), float(expected)) <= 1e-12


@pytest.mark.parametrize("case_id, comparison", [
    ("gauss_gr_cov", "check_group_covariance"), ("gauss_gr_cov_slope", "check_group_covariance"),
    ("gauss_gr_cov_slope_signed", "check_group_covariance"), ("gauss_sigma_offset", "check_dpar_offsets"),
])
def test_known_covariance_and_distributional_offsets_are_not_left_unchecked(case_id, comparison, monkeypatch):
    """Until gates.py compares brms's Lcov_k (gr(cov = ), G05b) and offsets_<dpar> (a distributional formula's offset, G06) in G1,
    a bundle of a case that records them fails with the oracle's mismatch naming the missing comparison, whatever model it is
    built with."""
    if hasattr(gates, comparison):
        pytest.skip(f"{comparison} has landed")
    monkeypatch.setattr(adapter, "in_data_order", lambda case, fixture, model, data: fixture)
    with pytest.raises(OracleMismatch, match=comparison):
        gates.Bundle(case_id, model=object())
