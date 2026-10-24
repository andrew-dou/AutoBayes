"""The oracle gates, as functions of a `Bundle` (case, fixture, data, AutoBayes model and the two maps).
Each raises `OracleMismatch` naming the point and the quantities that differ; a passing gate returns None.

G1 `check_design`        standata equals the model's design, response, trials, offsets, group coding and addition-term
                         data (weights, cens, lb, ub; B02; rcens, E04; dec, F04; se, denom and se()'s sigma = 0, F06a), each
                         distributional formula's X_<dpar> (B01), each
                         monotonic term's codes, number of steps and Dirichlet concentrations (Xmo_k, Jmo, con_simo_k; B03),
                         and a mixture's component designs X_mu<k> and theta's concentrations (con_theta; C05),
                         and each Gaussian process's row index, inputs or basis and scaling (Jgp_k, Xgp_k, Xgp_prior_k,
                         dmax_k, cmeans_k; an approximate GP's slambda_k, Lgp_k, NBgp_k; C03);
                         an autocorrelation term's Kar, Kma and J_lag, with brms's rows sorted by series and time
                         compared through the permutation derived from the data (C04, adapter.in_data_order);
                         a SAR term's Msar, eigenMsar, N_tg and its parameter's bounds (F02b, adapter.check_sar_design);
                         an fcor() term's Mfcor and N_tg (F03a, adapter.check_fcor_design);
                         a residual correlation term's groups (N_tg, begin_tg, end_tg, nobs_tg; unstr's n_unique_t,
                         n_unique_cortime, Jtime_tg) through the same permutation (F03b, adapter.check_timecor_design)
G2 `check_logp`          log density with and without the Jacobian, and its prior part, at every point
G2 `check_gradient`      every gradient coordinate, mapped through the probed parameter table
G3 `check_derived`       the graph's linear predictor, group effects and correlation matrices, and the raw-scale intercept
                         (F03b: unstr's Cortime and cosy's Cholesky factor, check_timecor)
                         (the cumulative thresholds and a categorical model's per-category intercepts, B05); an
                         auxiliary parameter's own formula: its per-row values and raw-scale intercept (B01); E03: a zi, hu or
                         alpha formula's values, the inverse of brms's default link of that parameter (_BRMS_DPAR_LINK)
G4 `check_predictions`   predict() on the fixture's draws equals posterior_linpred/posterior_epred
G5 `check_jax`           the JAX-compiled density equals the PyTensor one at every point
G6 `check_default_prior` brms's default-prior table has the classes this model has, and `ab.default_priors()` for the
                         same formula, data and family, written as brms's table, equals it row by row (A07)
C07 `check_post`         for a case that lists post-processing functions: `ab.log_lik`, `ab.bayes_R2` and `ab.loo` on the
                         fixture's draws, held as brms's fixed-parameter fit holds them (one chain per point), equal the
                         fixture's `post` block (brms's log_lik(), bayes_R2() and loo()), every non-finite entry exactly
D05 `check_waic`, `check_loo_compare`   in the same `post` gate: `ab.waic` against brms's waic(); `ab.loo_compare` of the case's
                         model with the case it names (`compare_with`), by loo and by waic, and `ab.model_weights` ("loo",
                         "pseudobma" with BB=False) against brms's loo_compare() and model_weights() of the two fits
D06 `check_hypothesis`, `check_conditional_effects`   in the same `post` gate: `ab.hypothesis` of the case's hypotheses
                         (`post_args`) against brms's hypothesis() table, column by column (labels and stars exactly), and
                         its draws; `ab.conditional_effects` of the case's effects against brms's conditional_effects()
                         table of each, every column (the grid, the values held, cond__, effect1__ and the summaries)
E05 `check_prior_draws`, `check_loo_model_weights`   in the same `post` gate: `ab.prior_draws` has brms's prior_draws() columns
                         (brms's draws are Stan's RNG's: they are compared through what is computed from them, the hypothesis
                         table's Savage-Dickey evidence ratios, `check_hypothesis` handing `ab.hypothesis` the fixture's prior
                         draws); an interaction effect's table (effect2__) is checked by `check_conditional_effects` as any
                         other; `ab.loo_model_weights` and `ab.model_weights(..., "stacking")` against brms's
                         loo_model_weights() of the two fits

A case that opts in to two-reference acceptance (P6, its gate field names a committed high-precision reference and the
quantity kinds that may fall back on it: "gradient", "epred") has G2's gradient and G4's expected values checked by
`two_reference`: brms where brms is accurate, the reference where it is not; every other check is unchanged.

C06: G1 checks brms's coding of a response's missing rows (`check_response`: Inf in `Y`, `Jmi`, `Nmi`, `lbmi`, `ubmi`) and, through
the adapter, `Ksp`, `Xn_1`, `noise_1` and `Mme_1`; G2 maps `Ymi`, `bsp`, `meanme`, `sdme` and `zme` (`meanme` and `sdme` in the
prior part); G3 checks an `me()` term's latent values against brms's `Xme_1` (`check_latent`); G4 and G6 read them as every
other quantity.

A multivariate model (C01) passes each gate response by response where the gate is about one response (G1's design, G3's
linear predictor, group effects and raw-scale intercept, each response's view of the fixture, adapter.ResponseView) and as one
model where it is about the joint density (G2, G5: one log density and gradient, the residual correlation's included; G3's
rescor; G4: the predictions' response axis; G6: brms's table with every response's rows and rescor).
F06c, `subset()`: G1 checks each response's rows (`N_<resp>`, `Y_<resp>`, `_check_design_subset`), G3 each response's linear
predictor against brms's own on its rows, and G4 (`check_response_predictions`, for a case with `prediction_resp`) that
`predict()` refuses the joint predictions brms refuses and that `predict(resp=)` equals brms's per-response predictions.

D04: G1 matches brms's Xcs and Kcs to the model's cs() columns (adapter.match_cs), G2 maps the unordered thresholds and bcs, and
G3 compares the graph's per-threshold predictor mu_cs of a model with cs() terms with brms's (rows, thresholds) posterior_linpred.

F06b: grouped thresholds (thres(gr = z)): G1 matches brms's ngrthres, nthres (one per level) and Jthres to the model's levels
(adapter.match_threshold_groups); G2 maps each Intercept_k to its level's block; G3 compares the graph's b_Intercept and the
raw-scale thresholds with brms's b_Intercept_k of every level (and its merged_Intercept), in the model's merged order.

D08: G2's prior part includes a shared correlation ID's distributional SDs and the potential that carries brms's normalisation
constant of gr(by = )'s SDs; G3 adds a shared ID's distributional effects (r_m_<dpar>_c) to brms's linear predictor of the
parameter and checks the graph's effects against them.

A non-linear model (C02, brms's nl = TRUE) is checked by the same gates through the adapter's non-linear maps: G1 its
covariates C_k, each parameter's X_<nlpar> (K_<nlpar>, never centred) and group designs (`check_nonlinear_design`); G2 every
b_<nlpar> coordinate; G3 the graph's non-linear predictor, each parameter's values against brms's own X_<nlpar> b_<nlpar> plus
its group effects, the group effects and correlation matrices (`check_nonlinear_derived`); G4 predict(); G6 the nlpar rows.

F01a, a count-matrix model (brms's multinomial and dirichlet_multinomial): G1 checks brms's Y (rows x K), ncat, trials and each
category's X_<mu> (K_<mu>, Kc_<mu>) and groups by content (`check_multinomial_design`); G2 maps Intercept_<mu>, b_<mu>, each category's
sd_<id> and z_<id>, and phi; G3 the graph's mu, each category's values against brms's own linear predictor of it, the raw-scale
intercepts b_<mu>_Intercept and the group effects (`check_multinomial_derived`); G4 compares predict()'s expected counts with
posterior_epred; G6 the categories' dpar rows.

E02: G1 checks a parameter brms's formula fixes (`bf(quantile = p)`, in its Stan data) against the model's fixed value
(`check_fixed_parameters`); G2 maps kappa, quantile (Stan's logit) and the GEV's tmp_xi; G3 compares the graph's derived xi with
the fixture's record of brms's `posterior_epred(dpar = "xi")` (`check_derived_parameters`), which G4's draws also carry.

D07, a shrinkage prior on b (brms's horseshoe() or R2D2()): G1 checks brms's Kscales and the prior's data (hs_df, hs_df_global,
hs_df_slab, hs_scale_global, hs_scale_slab; R2D2_mean_R2, R2D2_prec_R2, R2D2_cons_D2) against the model's prior
(`check_shrinkage_data`); G2's prior part is brms's lprior, which holds hs_global and hs_slab (or R2D2_R2) and not the coefficients,
now transformed parameters (zb, hs_local and R2D2_phi are in brms's model block); G3 checks the graph's coefficients, sdb and
R2D2_tau2 against brms's b, sdb and R2D2_tau2 (`check_shrinkage`).

F02a: G1 checks a CAR term's Stan data (`adapter.check_car_design`: Nloc, Jloc, Nedges, edges1, edges2, Nneigh, eigenMcar,
car_scale), G2 maps sdcar, car, rhocar, rcar, zcar and nszcar (sdcar, car and rhocar in the prior part), G3 checks the graph's rcar
against brms's (`check_car_effects`), G4 reads rcar from the draws and G6 the classes sdcar, car and rhocar.
"""
from __future__ import annotations

import re
from functools import cached_property
from itertools import zip_longest
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.special import expit

import autobayes as ab
from autobayes.priors import dirichlet_concentration

from tests.oracle import adapter
from tests.oracle import fixture as fx
from tests.oracle import two_reference
from tests.oracle.adapter import OracleMismatch
from tests.oracle.tolerances import DESIGN_ABS, GRAD_REL, LOGP_REL, PRED_REL, abs_err, rel_err


def prior_names(model) -> list[str]:
    """The free variables whose densities brms sums into ``lprior``: intercept, coefficients, sigma, shape or phi, a learned nu,
    zi or hu (B06), a family's alpha, beta or ndt (D01; ndt's prior truncated to (0, min_Y), its normalising constant in
    lprior as in brms's), the SD of every group-level coefficient (not the standardised offsets, which brms adds in the model block) and the
    correlation Cholesky factor of every correlated group term (``lkj_corr_cholesky_lpdf(L_m | eta)``); for an auxiliary
    parameter with its own formula (B01), that formula's intercept and coefficients (brms's ``Intercept_sigma``, ``b_sigma``);
    the simplex of every monotonic term (``dirichlet_lpdf(simo_k | con_simo_k)``; its coefficient is among the terms, B03);
    a mixture's ``sigma<k>`` and its mixing proportions (``dirichlet_lpdf(theta | con_theta)``; its ordered intercepts are the
    intercept, each component's density as brms adds it, C05);
    the marginal SD and length-scale of every Gaussian process (not its standardised ``zgp``, which brms adds in the model
    block, C03); for a non-linear model (C02), every parameter's coefficients, group SDs and correlation factors, and the
    auxiliary parameter; an autocorrelation term's ``ar`` and ``ma`` (C04); a residual correlation term's ``Lcortime``
    (``lkj_corr_cholesky_lpdf``) or ``cosy`` (F03b)."""
    if adapter.is_multivariate(model):                       # C01: each response's, under its names, and the residual correlation's
        return [model.variable(r, name) for r, sub in model.submodels.items() for name in prior_names(sub)] + (["Lrescor"] if model.rescor else [])
    if adapter.is_nonlinear(model):
        out = [v for roles in model.nlpar_names.values() for k, v in roles.items() if k != "value"]
        out += [roles["sd"] for roles in model.nlpar_group_names.values()]
        out += [roles["L"] for roles in model.nlpar_correlation_names.values()]
        scale = model.family_spec.scale_parameter
        return out + ([model.names[scale]] if scale is not None else [])
    if adapter.is_multinomial(model):                          # F01a: each category's intercept, coefficients and group SDs; phi
        out = [v for roles in model.mu_names.values() for k, v in roles.items() if k not in ("value", "raw_intercept")]
        out += [roles["sd"] for roles in model.mu_group_names.values()]
        return out + (["phi"] if "phi" in model.names else [])
    names = model.names
    out = [names["intercept"]] if model.spec.intercept else []
    shrinkage = getattr(model, "shrinkage_names", {})       # D07: b is then zb * sdb, a transformed parameter without a prior
    out += [] if shrinkage else list(model.spec.terms)
    out += list(shrinkage.get("lprior", ()))                # D07: brms's lprior holds hs_global and hs_slab, or R2D2_R2
    distributional = model.spec.dpar_names                  # the auxiliary parameters with a formula of their own (B01)
    if "sigma" in names and "sigma" not in distributional:
        out.append(names["sigma"])
    if "shape" in names and "shape" not in distributional:   # negative binomial / Gamma shape (PyMC's alpha)
        out.append(names["shape"])
    if "phi" in names and "phi" not in distributional:       # Beta precision
        out.append(names["phi"])
    out += [v for name in distributional for k, v in model.dpar_names[name].items() if k != "value"]   # their formulas (B01)
    if "nu" in names:                                      # a learned Student-t nu, its prior truncated to nu > 1 (A13)
        out.append(names["nu"])
    out += [names[par] for par in ("zi", "hu") if par in names and par not in distributional]   # the zero-inflation / hurdle
    out += [names[e.variable] for e in model.family_spec.extra_parameters                      # probability (B06); D01: alpha,
            if e.name not in distributional]                    # beta, ndt (ndt's truncation constant included); E02: kappa, quantile,
                                                                # the GEV's tmp_xi (brms's prior of class xi); E03: with a formula,
                                                                # its intercept and coefficients, above
    out += [roles["sd"] for roles in model.group_coefficient_names.values()]
    out += [roles["sd"] for roles in getattr(model, "dpar_group_names", {}).values()]   # D08: a shared ID's distributional SDs
    # D08: brms's normalisation constant of gr(by = )'s SDs (a potential of the graph, where brms's lprior has it)
    out += [roles["by_normalisation"] for roles in model.group_coefficient_names.values()
            if roles.get("by_normalisation") in model.pymc_model.named_vars]
    out += [roles["L"] for roles in model.correlation_names.values()]
    out += [roles["simplex"] for roles in model.mo_names.values()]
    mixture = getattr(model, "mixture_names", {})          # C05: each component's sigma<k> and theta
    out += [*mixture["sigma"], mixture["theta"]] if mixture else []
    out += [roles[key] for roles in getattr(model, "gp_names", {}).values() for key in ("sdgp", "lscale")]   # C03
    out += [v for k, v in getattr(model, "autocor_names", {}).items() if not k.endswith("_coordinate")]   # an ARMA term's ar and ma (C04)
    # F02a: a CAR term's sdcar, car and rhocar (brms's lprior; its latent vectors' densities are in brms's model block)
    out += [v for k, v in getattr(model, "car_names", {}).items() if k in ("sdcar", "car", "rhocar")]
    out += [v for k, v in getattr(model, "timecor_names", {}).items() if k in ("L", "cosy")]   # F03b: Lcortime or cosy
    # C06: an me() term's meanme and sdme (brms's lprior; its zme and the noisy values' density are in the model block, and the
    # missing values Ymi have no prior); an me() / mi() term's coefficient is among the terms
    out += [roles[k] for roles in getattr(model, "me_names", {}).values() for k in ("mean", "sd")]
    # F01b: the logistic_normal family's sigma<k> and Llncor (lkj_corr_cholesky_lpdf(Llncor | eta)); the dirichlet phi is above
    simplex = getattr(model, "simplex_names", {})
    out += [*simplex.get("sigma", ()), *([simplex["L"]] if "L" in simplex else [])]
    return out


class Bundle:
    def __init__(self, case_id: str, fixture: fx.Fixture | None = None, model=None):
        self.case = fx.load_case(case_id)
        self.fixture = fixture or fx.Fixture.load(case_id)
        d = fx.fixture_dir(case_id)
        self.data = fx.read_frame(self.case, d / "data.csv")
        self.newdata = fx.read_frame(self.case, d / "newdata.csv") if (d / "newdata.csv").exists() else None
        # Tier F: brms's data2 matrices by name ({} for a case without), handed to the case's constructor as data2= (car(W), sar(M),
        # fcor(M) name them in the formula)
        self.data2 = fx.read_data2(self.case, d)
        build = self.case
        if self.data2:
            spec = self.case["autobayes"]
            build = {**self.case, "autobayes": {**spec, "kwargs": {**spec.get("kwargs", {}), "data2": self.data2}}}
        self.model = model or adapter.build_model(build, self.data)
        # C04: brms sorts an autocorrelation model's Stan rows by series and time; its per-row data are compared in data order
        self.fixture = adapter.in_data_order(self.case, self.fixture, self.model, self.data)
        # C07: the post-processing functions a case names (``"post": ["log_lik", "bayes_R2", "loo"]`` in its autobayes
        # section, each proposed as ``ab.<name>(model, idata)``) must exist, so that such a case fails until they do; the
        # gate that compares them with the fixture's ``post`` block (brms's log_lik(), bayes_R2(), loo()) lands with them; the
        # same holds for the names Tier E cases propose (``prior_draws``, ``loo_model_weights``)
        self.post = {name: getattr(ab, name) for name in self.case.get("autobayes", {}).get("post", ())}
        # Tier F part 2: F07's summaries (fixef, ranef, coef, VarCorr, posterior_summary, summary, residuals) are names there too,
        # compared on the fixture's summary draws; an F06 fixture with each response's own predictions (``prediction_resp``,
        # subset(): brms refuses the joint ones) needs G4's comparison of them, ``check_response_predictions``, which lands with
        # subset(): until then such a case fails here rather than leaving its predictions unchecked
        if self.case["brms"].get("prediction_resp") and "check_response_predictions" not in globals():
            raise OracleMismatch("the fixture records each response's predictions (prediction_resp); gates.py has no "
                                 "check_response_predictions to compare them")
        self.cmap = adapter.match_columns(self.fixture, self.model)
        self.gmap = adapter.match_groups(self.fixture, self.model)

    def with_fixture(self, fixture: fx.Fixture) -> "Bundle":
        """The same case and model against another fixture (used to show that a corrupted fixture fails)."""
        return Bundle(self.case["id"], fixture=fixture, model=self.model)

    def points(self):
        for k, p in enumerate(self.fixture.points):
            yield k, p, adapter.value_point(self.model, p, self.cmap, self.gmap)

    @cached_property
    def logp(self):
        return self.model.pymc_model.compile_logp()

    @cached_property
    def logp_nojac(self):
        return self.model.pymc_model.compile_logp(jacobian=False)

    @cached_property
    def dlogp(self):
        return self.model.pymc_model.compile_dlogp()

    @cached_property
    def dlogp_nojac(self):
        return self.model.pymc_model.compile_dlogp(jacobian=False)

    @cached_property
    def lprior(self):
        rvs = [self.model.pymc_model[name] for name in prior_names(self.model)]
        return self.model.pymc_model.compile_logp(vars=rvs, jacobian=False)

    @cached_property
    def reference(self):
        """P6: the committed high-precision reference of a case that opts in to two-reference acceptance (its gate field
        names one, ``two_reference.load`` validates the opt-in), else None."""
        return two_reference.load(self.case)


# -- G1 -----------------------------------------------------------------------------------------------------
def check_design(b: Bundle) -> None:
    if adapter.is_multivariate(b.model):                      # C01
        return _check_design_multivariate(b)
    sd, d = b.fixture.standata, b.model.design
    if int(sd["N"]) != d.n:
        raise OracleMismatch(f"brms N = {int(sd['N'])}, AutoBayes n = {d.n}")
    if adapter.is_nonlinear(b.model):                              # C02: no X; the covariates and each parameter's X_<nlpar>
        return check_nonlinear_design(b)
    if adapter.is_multinomial(b.model):                            # F01a: the count matrix, trials and each category's X_<mu>
        return check_multinomial_design(b)
    # B05: the cumulative model's X has no intercept column (its thresholds take its place); the categorical model has one
    # design per non-reference category
    designs = [f"X_{dp}" for dp in b.cmap.dpars] or ["X"]
    intercept_column = int(b.model.spec.intercept and not b.cmap.thresholds)
    for name in designs:
        if sd[name].shape != (d.n, d.X_raw.shape[1] + intercept_column):
            raise OracleMismatch(f"brms {name} is {sd[name].shape}; AutoBayes has {d.X_raw.shape[1]} design columns and intercept={b.model.spec.intercept}")
    if b.cmap.thres_levels:                                      # F06b: one nthres per level, Jthres, ngrthres
        adapter.match_threshold_groups(b.fixture, b.model)
        if int(np.max(sd["nthres"])) != len(b.model.categories) - 1:
            raise OracleMismatch(f"brms's largest nthres is {int(np.max(sd['nthres']))}; AutoBayes has {len(b.model.categories)} categories")
    elif b.cmap.thresholds and int(sd["nthres"]) != len(b.model.categories) - 1:
        raise OracleMismatch(f"brms has nthres = {int(sd['nthres'])}; AutoBayes has {len(b.model.categories)} categories")
    if b.cmap.dpars and not b.cmap.mixture and int(sd["ncat"]) != len(b.model.categories):   # a mixture's dpars are components
        raise OracleMismatch(f"brms has ncat = {int(sd['ncat'])}; AutoBayes has {len(b.model.categories)} categories")
    if b.cmap.mixture:                                           # C05: K component designs (matched by content) and con_theta
        K = len(b.model.components)
        if len(b.cmap.dpars) != K:
            raise OracleMismatch(f"brms has {len(b.cmap.dpars)} component designs; the AutoBayes mixture has {K} components")
        con = dirichlet_concentration(b.model.priors.theta, K)
        theirs = np.atleast_1d(sd["con_theta"])
        if theirs.shape != con.shape or abs_err(theirs, con) > DESIGN_ABS:
            raise OracleMismatch(f"brms con_theta is {theirs!r}; the prior of theta gives {con!r}")
    check_response(sd, b.model)                                  # C06: with brms's Inf at the missing rows (Jmi, Nmi), if any
    if d.trials is not None:
        if "trials" not in sd:
            raise OracleMismatch("AutoBayes has trials but brms standata has none")
        err = abs_err(sd["trials"], d.trials)
        if err > DESIGN_ABS:
            raise OracleMismatch(f"trials differ by {err}")
    elif "trials" in sd:
        raise OracleMismatch("brms has trials but AutoBayes does not")
    if d.offset is not None:                                     # brms's offsets: the offset term's values per row
        if "offsets" not in sd:
            raise OracleMismatch("AutoBayes has an offset term but brms standata has no offsets")
        err = abs_err(sd["offsets"], d.offset)
        if err > DESIGN_ABS:
            raise OracleMismatch(f"offsets differ by {err}")
    elif "offsets" in sd:
        raise OracleMismatch("brms has offsets but AutoBayes has no offset term")
    # brms's addition-term data (B02): weights(), cens(), trunc() bounds; E04: cens(c, y2)'s rcens; F06a: se()'s se, rate()'s denom
    for key in ("weights", "cens", "rcens", "lb", "ub", "se", "denom"):
        mine = getattr(d, key, None)
        if mine is None:
            if key in sd:
                raise OracleMismatch(f"brms has {key} but AutoBayes does not")
            continue
        if key not in sd:
            raise OracleMismatch(f"AutoBayes has {key} but brms standata has none")
        theirs, mine = np.asarray(sd[key], dtype=float), np.asarray(mine, dtype=float)
        if theirs.shape != mine.shape:
            raise OracleMismatch(f"brms {key} has shape {theirs.shape}, AutoBayes {mine.shape}")
        inf = np.isinf(theirs)                                   # an open truncation bound: brms's Inf where the model has inf, same sign
        if not np.array_equal(theirs[inf], mine[inf]) or np.isinf(mine[~inf]).any():
            raise OracleMismatch(f"{key}: brms's infinite entries (rows {np.flatnonzero(inf)[:5].tolist()}...) differ from AutoBayes's")
        err = abs_err(theirs[~inf], mine[~inf])
        if not err <= DESIGN_ABS:
            raise OracleMismatch(f"{key} differ by {err}")
    check_fixed_sigma(b)                                         # F06a: se(col) without sigma = TRUE, brms's data sigma = 0
    mine = getattr(d, "dec", None)                              # F04: brms's dec of dec(col), each wiener row's decision
    if (mine is None) != ("dec" not in sd):
        raise OracleMismatch("brms has dec but AutoBayes does not" if mine is None else "AutoBayes has dec but brms standata has none")
    if mine is not None and (np.shape(sd["dec"]) != np.shape(mine) or abs_err(sd["dec"], mine) > DESIGN_ABS):
        raise OracleMismatch(f"dec differs: brms {np.asarray(sd['dec'])[:8]}..., AutoBayes {np.asarray(mine)[:8]}...")
    for name, dd in d.dpars.items():                            # an auxiliary parameter's own formula (B01): X_sigma
        key = f"X_{name}"
        if key not in sd:
            raise OracleMismatch(f"AutoBayes gives {name} its own formula but brms standata has no {key}")
        if sd[key].shape != (d.n, dd.X_raw.shape[1] + int(b.model.spec.dpar(name).intercept)):
            raise OracleMismatch(f"brms {key} is {sd[key].shape}; AutoBayes has {dd.X_raw.shape[1]} design columns for {name} "
                                 f"and intercept={b.model.spec.dpar(name).intercept}")
    for column, k in (b.cmap.monotonic or {}).items():          # B03: Xmo_k and Jmo[k] were matched by content; the prior's
        block = d.monotonic[column]                              # concentrations are brms's con_simo_k
        con = dirichlet_concentration(b.model.priors.simo_priors(b.model.spec)[column], block.D)
        theirs = np.atleast_1d(sd[f"con_simo_{k}"])
        if theirs.shape != con.shape or abs_err(theirs, con) > DESIGN_ABS:
            raise OracleMismatch(f"brms con_simo_{k} is {theirs!r}; the prior of {block.term!r} gives {con!r}")
    for column, k in (b.cmap.gp or {}).items():                 # C03: Jgp_k and Xgp_k were matched by content; the scaling
        g, block = b.model.spec.gp(column), d.gp[column]         # and the approximate GP's boundary and frequencies here
        mine = {f"dmax_{k}": np.float64(g.dmax), f"cmeans_{k}": np.float64(g.cmean), f"Xgp_prior_{k}": block.inputs[:, None]}
        if g.approximate:
            mine.update({f"Lgp_{k}": np.float64(g.L), f"NBgp_{k}": np.float64(g.k), f"slambda_{k}": block.slambda[:, None]})
        elif any(f"{key}_{k}" in sd for key in ("Lgp", "NBgp", "slambda")):
            raise OracleMismatch(f"brms's Gaussian process {k} is approximate (Lgp_{k}, NBgp_{k}); {g.term!r} is exact")
        for key, value in mine.items():
            if key not in sd or np.shape(sd[key]) != np.shape(value) or abs_err(sd[key], value) > DESIGN_ABS:
                raise OracleMismatch(f"Gaussian process {g.term!r}: brms {key} is {sd.get(key)!r}; AutoBayes has {value!r}")
    adapter.check_autocor_design(b.fixture, b.model)             # C04: Kar, Kma, J_lag and each sorted row's series and position
    adapter.check_sar_design(b.fixture, b.model)                 # F02b: Msar, eigenMsar, N_tg and the parameter's bounds
    adapter.check_fcor_design(b.fixture, b.model)                # F03a: Mfcor and N_tg
    adapter.check_car_design(b.fixture, b.model)                 # F02a: a CAR term's locations, edges, Nneigh, eigenMcar, car_scale
    adapter.check_timecor_design(b.fixture, b.model)             # F03b: N_tg, begin_tg, end_tg, nobs_tg, Jtime_tg and the groups
    check_shrinkage_data(b)                                      # D07: Kscales and a shrinkage prior's data
    check_fixed_parameters(b)                                    # E02: a parameter the formula fixes (bf(quantile = p))
    # the design columns (every formula's), the group partitions and the monotonic codes were matched by content when the
    # maps were built
    adapter.match_columns(b.fixture, b.model)
    adapter.match_groups(b.fixture, b.model)


def check_fixed_sigma(b: Bundle) -> None:
    """G1 of ``se(col)`` without ``sigma = TRUE`` (F06a): brms writes ``sigma = 0`` into its Stan data exactly when the model has
    the term and no sigma parameter (no ``sigma`` among its names), and nowhere else."""
    sd, se = b.fixture.standata, b.model.spec.addition("se")
    mine = se is not None and not se.sigma and "sigma" not in b.model.names
    if "sigma" in sd and not mine:
        raise OracleMismatch(f"brms's Stan data holds sigma = {float(sd['sigma'])!r} (se() without sigma = TRUE); the AutoBayes model "
                             + ("learns sigma" if "sigma" in b.model.names else "has no such se() term"))
    if mine and "sigma" not in sd:
        raise OracleMismatch("the AutoBayes model fixes sigma at 0 under se() without sigma = TRUE; brms's Stan data has no sigma")
    if mine and float(sd["sigma"]) != 0.0:
        raise OracleMismatch(f"brms fixes sigma at {float(sd['sigma'])!r} in its Stan data; se() without sigma = TRUE fixes it at 0")


def check_fixed_parameters(b: Bundle) -> None:
    """G1 of a parameter brms's formula may fix (E02, ``FamilySpec.fixable``: ``bf(quantile = p)``, which brms writes into its
    Stan data under the parameter's name): fixed in the model exactly when brms's data holds it, to the same value."""
    sd, fam = b.fixture.standata, b.model.family_spec
    for name in getattr(fam, "fixable", {}):
        fixed = name in fam.fixed_parameters
        if fixed != (name in sd):
            raise OracleMismatch(f"brms's Stan data {'lacks' if fixed else 'holds'} {name}"
                                 + (f" = {float(sd[name])!r}" if name in sd else "") + f"; the AutoBayes model "
                                 + (f"fixes it at {getattr(b.model, name)!r}" if fixed else f"learns {name}"))
        if fixed and abs_err(sd[name], getattr(b.model, name)) > DESIGN_ABS:
            raise OracleMismatch(f"brms fixes {name} at {float(sd[name])!r}; the AutoBayes model at {getattr(b.model, name)!r}")


def check_response(sd: dict, model) -> None:
    """G1's response check: brms's ``Y`` equals the model's response on every row. A response whose missing values are
    parameters (C06, ``y | mi() ~ ...``, ``Design.missing[response]``) is compared on its observed rows, and brms's own coding of
    the missing rows must be the model's: ``Y`` infinite exactly there (brms writes Inf), ``Jmi`` those rows (1-based), ``Nmi``
    their number and ``lbmi`` / ``ubmi`` brms's open bounds of the missing values."""
    y_brms, y = np.asarray(sd["Y"], dtype=float), np.asarray(model.design.y, dtype=float)
    block = getattr(model.design, "missing", {}).get(model.spec.response)
    if block is None:
        if not np.isfinite(y_brms).all() or "Jmi" in sd or "Nmi" in sd:
            raise OracleMismatch("brms's response has missing values (Inf in Y, Jmi, Nmi); the AutoBayes model holds none")
        observed = np.ones(y.shape, dtype=bool)
    else:
        rows = np.flatnonzero(~np.isfinite(y_brms))
        if rows.tolist() != block.index.tolist() or not np.isinf(y_brms[rows]).all():
            raise OracleMismatch(f"brms's Y is infinite at rows {rows.tolist()}; the AutoBayes model's missing rows are {block.index.tolist()}")
        if (np.atleast_1d(sd.get("Jmi", np.zeros(0))).astype(int) - 1).tolist() != block.index.tolist() or int(sd.get("Nmi", -1)) != block.index.size:
            raise OracleMismatch(f"brms's Jmi {sd.get('Jmi')} / Nmi {sd.get('Nmi')} are not the model's missing rows {block.index.tolist()}")
        if "lbmi" in sd and not (np.isneginf(sd["lbmi"]) and np.isposinf(sd["ubmi"])):
            raise OracleMismatch(f"brms bounds the missing values by lbmi {sd['lbmi']}, ubmi {sd['ubmi']}; the AutoBayes model does not")
        observed = np.isfinite(y_brms)
        if not np.isnan(y[~observed]).all():
            raise OracleMismatch("the AutoBayes response is not missing (NaN) at brms's missing rows")
    err = abs_err(y_brms[observed], y[observed])
    if not err <= DESIGN_ABS:
        raise OracleMismatch(f"response differs by {err}")


# -- G2 -----------------------------------------------------------------------------------------------------
def check_logp(b: Bundle) -> None:
    for k, p, vp in b.points():
        a, an, pri = float(b.logp(vp)), float(b.logp_nojac(vp)), float(b.lprior(vp))
        pri_b = float(p.pars["lprior"]) if "lprior" in p.pars else None
        bad = rel_err(a, p.lp) > LOGP_REL or rel_err(an, p.lp_nojac) > LOGP_REL
        bad = bad or (pri_b is not None and rel_err(pri, pri_b) > LOGP_REL)
        if bad:
            raise OracleMismatch(
                f"point {k}: log density AutoBayes {a!r} vs brms {p.lp!r}; without Jacobian {an!r} vs {p.lp_nojac!r}; "
                f"prior part {pri!r} vs {pri_b!r}; remainder {an - pri!r} vs {None if pri_b is None else p.lp_nojac - pri_b!r}")


def check_gradient(b: Bundle) -> None:
    if b.reference is not None and "gradient" in b.reference.fallback:      # P6: brms where it is accurate, the reference where not
        return two_reference.check_gradient(b)
    coords = adapter.coordinate_map(b.fixture, b.model, b.cmap, b.gmap)
    for k, p, vp in b.points():
        layout = adapter.value_layout(b.model, vp)
        total = sum(s.stop - s.start for s in layout.values())
        if len(coords) != total or p.grad.size != total:
            raise OracleMismatch(f"brms has {len(coords)} unconstrained coordinates, AutoBayes {total}")
        ga, gan = np.asarray(b.dlogp(vp), dtype=float), np.asarray(b.dlogp_nojac(vp), dtype=float)
        hit = np.zeros(total, dtype=bool)
        for i, var, idx in coords:
            pos = layout[var].start + idx
            hit[pos] = True
            if rel_err(ga[pos], p.grad[i]) > GRAD_REL or rel_err(gan[pos], p.grad_nojac[i]) > GRAD_REL:
                raise OracleMismatch(
                    f"point {k}: gradient of {var}[{idx}] AutoBayes {ga[pos]!r} vs brms {p.grad[i]!r} "
                    f"(without Jacobian {gan[pos]!r} vs {p.grad_nojac[i]!r})")
        if not hit.all():
            missing = [v for v, s in layout.items() if not hit[s].all()]
            raise OracleMismatch(f"AutoBayes coordinates without a brms counterpart: {missing}")


# -- G3 -----------------------------------------------------------------------------------------------------
# brms's default link of each distributional parameter a frozen case gives a formula (brmsfamily()'s link_sigma, link_shape,
# link_phi: log, B01; link_zi, link_hu: logit and link_alpha: identity, E03), with its inverse, which G3 applies to brms's own
# linear predictor of the parameter; the model's DparDesign.link must be the same
_BRMS_DPAR_LINK = {"sigma": ("log", np.exp), "shape": ("log", np.exp), "phi": ("log", np.exp),
                   "zi": ("logit", expit), "hu": ("logit", expit),
                   "alpha": ("identity", lambda eta: eta),
                   "bs": ("log", np.exp)}                      # F04: wiener()'s link_bs


def check_derived(b: Bundle) -> None:
    if adapter.is_multivariate(b.model):                      # C01
        return _check_derived_multivariate(b)
    if adapter.is_nonlinear(b.model):                              # C02: mu, each parameter's values, group effects, Cor
        return check_nonlinear_derived(b)
    if adapter.is_multinomial(b.model):                            # F01a: mu, each category's values, group effects, raw intercepts
        return check_multinomial_derived(b)
    pmm, names, d = b.model.pymc_model, b.model.names, b.model.design
    groups = list(b.model.group_coefficient_names.items())          # every group-level coefficient: intercepts and slopes
    cors = list(b.model.correlation_names.items())                  # the correlation matrix of every correlated term (A02)
    distributional = list(b.model.dpar_names.items())               # each auxiliary parameter's own formula (B01): its per-row values
    dpar_groups = list(getattr(b.model, "dpar_group_names", {}).items())   # D08: a shared ID's distributional coefficients
    dets = ([pmm[names["linpred"]]] + [pmm[roles["group_effect"]] for _, roles in groups] + [pmm[roles["cor"]] for _, roles in cors]
            + [pmm[roles["value"]] for _, roles in distributional] + [pmm[roles["group_effect"]] for _, roles in dpar_groups])
    if b.cmap.thresholds:                                           # B05: the graph's b_Intercept, brms's raw thresholds (the last)
        dets.append(pmm[names["thresholds"]])
    fn = pmm.compile_fn(pmm.replace_rvs_by_values(dets), inputs=pmm.value_vars, point_fn=True, on_unused_input="ignore")
    # Under a log link Bambi's parent deterministic holds the mean exp(eta), which is brms's epred when the parent is the
    # expected response (Poisson, negative binomial, Gamma: names["linpred"] == names["mean"] == "mu"); the zero-inflated
    # and hurdle families (B06) keep eta itself as their parent log_mu, whose brms counterpart is posterior_linpred.
    holds_mean = b.model.link == "log" and names["linpred"] == names["mean"]
    ref = b.fixture.prediction("fitted", "epred" if holds_mean else "linpred")
    if holds_mean and b.model.spec.addition("trunc") is not None:   # E04: brms's epred of a truncated model is the truncated
        ref = np.exp(b.fixture.prediction("fitted", "linpred"))    # mean; the parent holds the untruncated mean exp(eta)
    for k, p, vp in b.points():
        out = fn(vp)
        # B05: a categorical model's linear predictor has one column per non-reference category, brms's (rows, categories); D04:
        # an ordinal model with cs() terms one per threshold, brms's (rows, thresholds)
        lin = np.asarray(out[0], dtype=float) if (b.cmap.dpars or b.cmap.cs) else np.asarray(out[0], dtype=float).ravel()
        err = rel_err(lin, ref[k])
        if err > PRED_REL:
            raise OracleMismatch(f"point {k}: the graph's {names['linpred']!r} differs from brms by {err}")
        for ((g, coef), _), eff in zip(groups, out[1:1 + len(groups)]):
            if b.gmap.category:                                     # F01b: each category's column against its own group term
                check_category_effects(b, k, p, g, coef, eff)
                continue
            m, cn = b.gmap.brms_id[g], b.gmap.coef_cn[(g, coef)]
            r = np.asarray(eff, dtype=float).ravel()[b.gmap.level_of[g]]
            err = rel_err(r, np.atleast_1d(p.pars[f"r_{m}_{cn}"]))
            if err > PRED_REL:
                raise OracleMismatch(f"point {k}: group effects of {coef!r} of {g!r} differ from brms r_{m}_{cn} by {err}")
        for (g, _), cor in zip(cors, out[1 + len(groups):1 + len(groups) + len(cors)]):
            m, perm = b.gmap.brms_id[g], adapter.correlation_order(b.model, b.gmap, g)
            ref_cor = np.atleast_2d(p.pars[f"Cor_{m}"])[np.ix_(perm, perm)]
            err = rel_err(np.asarray(cor, dtype=float), ref_cor)
            if err > PRED_REL:
                raise OracleMismatch(f"point {k}: the correlation matrix of {g!r} differs from brms Cor_{m} by {err}")
        start = 1 + len(groups) + len(cors)
        for (name, _), value in zip(distributional, out[start:start + len(distributional)]):   # B01: exp of brms's own predictor
            if name not in _BRMS_DPAR_LINK:                         # E03: inv_logit for zi and hu, the identity for alpha
                raise OracleMismatch(f"no brms link of {name}'s formula is recorded here")
            link, inverse = _BRMS_DPAR_LINK[name]
            if b.model.design.dpars[name].link != link:
                raise OracleMismatch(f"the model's link of {name}'s formula is {b.model.design.dpars[name].link}; brms's default is {link}")
            X = b.fixture.standata[f"X_{name}"]
            dmap = b.cmap.distributional[name]
            bd = adapter.coefficients(p.pars, f"b_{name}")
            if dmap.centred_intercept:                              # brms: Intercept_<name> + Xc_<name> b_<name>, Xc centred on its means
                Xc = np.delete(X, dmap.intercept_column, axis=1)
                eta = float(p.pars[f"Intercept_{name}"]) + (Xc - Xc.mean(0)) @ bd
            else:
                eta = X @ bd
            for (dname, g, coef), _ in dpar_groups:                # D08: plus brms's r_m_<name>_c[J_m] Z_m_<name>_c of a shared ID
                if dname == name:
                    m, cn = b.gmap.brms_id[g], b.gmap.dpar_cn[(name, g, coef)]
                    J = b.fixture.standata[f"J_{m}"].astype(int) - 1
                    eta = eta + np.atleast_1d(p.pars[f"r_{m}_{name}_{cn}"])[J] * b.fixture.standata[f"Z_{m}_{name}_{cn}"]
            err = rel_err(np.asarray(value, dtype=float).ravel(), inverse(eta))
            if err > PRED_REL:
                raise OracleMismatch(f"point {k}: the graph's {name} differs from the inverse {link} link of brms's linear predictor "
                                     f"of {name} by {err}")
            if f"b_{name}_Intercept" in p.pars:                      # the raw-scale intercept of name's formula
                nat = adapter.natural_point(b.model, p, b.cmap, b.gmap)[name]
                raw = nat["intercept"] - float(b.model.design.dpars[name].x_means @ nat["b"])
                err = rel_err(raw, float(p.pars[f"b_{name}_Intercept"]))
                if err > PRED_REL:
                    raise OracleMismatch(f"point {k}: raw-scale intercept of {name} {raw!r} vs brms b_{name}_Intercept "
                                         f"{float(p.pars[f'b_{name}_Intercept'])!r}")
        start = 1 + len(groups) + len(cors) + len(distributional)
        for ((name, g, coef), _), eff in zip(dpar_groups, out[start:start + len(dpar_groups)]):   # D08: brms's r_m_<name>_c
            m, cn = b.gmap.brms_id[g], b.gmap.dpar_cn[(name, g, coef)]
            err = rel_err(np.asarray(eff, dtype=float).ravel()[b.gmap.level_of[g]], np.atleast_1d(p.pars[f"r_{m}_{name}_{cn}"]))
            if err > PRED_REL:
                raise OracleMismatch(f"point {k}: group effects of {coef!r} of {g!r} in {name}'s formula differ from brms r_{m}_{name}_{cn} "
                                     f"by {err}")
        if b.cmap.thres_levels:                                      # F06b: every level's b_Intercept_k = Intercept_k (no centring)
            nat = adapter.natural_point(b.model, p, b.cmap, b.gmap)
            brms = adapter.merged_thresholds(b.model, b.cmap, p.pars, "b_Intercept")
            merged = np.concatenate([np.atleast_1d(p.pars[f"Intercept_{j}"]) for j in range(1, len(b.cmap.thres_levels) + 1)])
            if rel_err(np.atleast_1d(p.pars["merged_Intercept"]), merged) > PRED_REL:
                raise OracleMismatch(f"point {k}: brms's merged_Intercept is not its Intercept_k in turn")
            raw = np.concatenate([nat["intercept"][level] for level in b.model.design.thresholds.levels]) + float(d.x_means @ nat["b"])
            for what, mine in (("raw-scale thresholds", raw), (f"the graph's {names['thresholds']!r}", np.asarray(out[-1], dtype=float))):
                err = rel_err(mine, brms)
                if err > PRED_REL:
                    raise OracleMismatch(f"point {k}: {what} {mine!r} vs brms b_Intercept_k {brms!r}")
        elif b.cmap.thresholds:                                      # B05: b_Intercept = Intercept + sum_p mean(x_p) b_p
            nat = adapter.natural_point(b.model, p, b.cmap, b.gmap)
            brms = np.atleast_1d(p.pars["b_Intercept"])
            for what, raw in (("raw-scale thresholds", nat["intercept"] + float(d.x_means @ nat["b"])),
                              (f"the graph's {names['thresholds']!r}", np.asarray(out[-1], dtype=float))):
                err = rel_err(raw, brms)
                if err > PRED_REL:
                    raise OracleMismatch(f"point {k}: {what} {raw!r} vs brms b_Intercept {brms!r}")
        elif b.cmap.dpars:                                           # B05: each category's raw-scale intercept
            nat = adapter.natural_point(b.model, p, b.cmap, b.gmap)
            for c, dp in enumerate(b.cmap.dpars):
                if f"b_{dp}_Intercept" in p.pars:
                    raw = nat["intercept"][c] - float(d.x_means @ nat["b"][:, c])
                    err = rel_err(raw, float(p.pars[f"b_{dp}_Intercept"]))
                    if err > PRED_REL:
                        raise OracleMismatch(f"point {k}: raw-scale intercept of {dp} {raw!r} vs brms b_{dp}_Intercept "
                                             f"{float(p.pars[f'b_{dp}_Intercept'])!r}")
        elif "b_Intercept" in p.pars:
            nat = adapter.natural_point(b.model, p, b.cmap, b.gmap)
            raw = nat["intercept"] - float(d.x_means @ nat["b"])
            err = rel_err(raw, float(p.pars["b_Intercept"]))
            if err > PRED_REL:
                raise OracleMismatch(f"point {k}: raw-scale intercept {raw!r} vs brms b_Intercept {float(p.pars['b_Intercept'])!r}")
    check_latent(b)                                                  # C06: an me() term's latent values, brms's Xme_k
    check_shrinkage(b)                                               # D07: b, sdb and R2D2_tau2 of a shrinkage prior on b
    check_derived_parameters(b)                                      # E02: the GEV's xi against brms's dpar_epred record
    check_car_effects(b)                                             # F02a: a CAR term's rcar against brms's
    check_timecor(b)                                                 # F03b: unstr's Cortime, cosy's Cholesky factor


def check_car_effects(b: Bundle) -> None:
    """G3 of a CAR term (F02a): at every point the graph's ``rcar`` (a deterministic of the latent vectors and parameters, the
    free vector of escar) equals brms's ``rcar``, location by location (the locations checked to be brms's)."""
    names = getattr(b.model, "car_names", {})
    if not names:
        return
    adapter.check_car_locations(b.fixture, b.model)
    pmm = b.model.pymc_model
    fn = pmm.compile_fn(pmm.replace_rvs_by_values([pmm[names["rcar"]]]), inputs=pmm.value_vars, point_fn=True, on_unused_input="ignore")
    for k, p, vp in b.points():
        err = rel_err(np.asarray(fn(vp)[0], dtype=float), np.atleast_1d(p.pars["rcar"]))
        if err > PRED_REL:
            raise OracleMismatch(f"point {k}: the graph's rcar differs from brms's rcar by {err}")


def check_category_effects(b: Bundle, k: int, p, g: str, coef: str, eff) -> None:
    """F01b, G3 of a per-category family's group term at point ``k``: the graph's effects of ``(g, coef)``, ``(levels, K - 1)``,
    column by column against brms's ``r_m_<dpar>_c`` of each non-reference category's own group (``GroupMap.category``)."""
    from autobayes.default_priors import brms_dpar_names

    eff = np.asarray(eff, dtype=float)
    for c, ((m, cn), dp) in enumerate(zip(b.gmap.category[(g, coef)], brms_dpar_names(b.model.categories[1:]))):
        err = rel_err(eff[b.gmap.level_of[g], c], np.atleast_1d(p.pars[f"r_{m}_{dp}_{cn}"]))
        if err > PRED_REL:
            raise OracleMismatch(f"point {k}: group effects of {coef!r} of {g!r} in {dp} differ from brms r_{m}_{dp}_{cn} by {err}")


def check_timecor(b: Bundle) -> None:
    """G3 of a residual correlation term (F03b): ``unstr``'s graph deterministic ``Cortime`` and its ``Lcortime`` equal brms's
    ``Cortime`` and ``Lcortime`` at every point; for ``cosy``, the Cholesky factor the graph builds from its ``cosy`` (the
    compound-symmetric matrix of the largest group) equals brms's ``Lcortime``. Nothing for a model without the term."""
    from autobayes.timecor import cosy_cholesky

    roles = getattr(b.model, "timecor_names", {})
    if not roles:
        return
    pmm = b.model.pymc_model
    if "L" in roles:
        outs, refs = [pmm[roles["cor"]], pmm[roles["L"]]], ("Cortime", "Lcortime")
    else:
        outs, refs = [cosy_cholesky(pmm[roles["cosy"]], b.model.design.timecor.dim)], ("Lcortime",)
    fn = pmm.compile_fn(pmm.replace_rvs_by_values(outs), inputs=pmm.value_vars, point_fn=True, on_unused_input="ignore")
    for k, p, vp in b.points():
        for name, value in zip(refs, fn(vp)):
            err = rel_err(np.asarray(value, dtype=float), np.atleast_2d(p.pars[name]))
            if err > PRED_REL:
                raise OracleMismatch(f"point {k}: the graph's {name} differs from brms's by {err}")


_SHRINKAGE_DATA = ("Kscales", "hs_df", "hs_df_global", "hs_df_slab", "hs_scale_global", "hs_scale_slab", "R2D2_mean_R2",
                   "R2D2_prec_R2", "R2D2_cons_D2")


def check_shrinkage_data(b: Bundle) -> None:
    """G1 of a shrinkage prior on b (D07): brms's ``Kscales`` is the number of coefficients and the prior's data are the
    prior's arguments (the horseshoe's ``hs_df``, ``hs_df_global``, ``hs_df_slab``, ``hs_scale_global``, ``hs_scale_slab``;
    R2D2's ``R2D2_mean_R2``, ``R2D2_prec_R2`` and ``R2D2_cons_D2``, one concentration per coefficient in brms's order of b);
    a model without a shrinkage prior has none of them."""
    from autobayes.shrinkage import concentration

    sd, roles = b.fixture.standata, getattr(b.model, "shrinkage_names", {})
    present = sorted(k for k in _SHRINKAGE_DATA if k in sd)
    if not roles:
        if present:
            raise OracleMismatch(f"brms has a shrinkage prior's data {present}; the AutoBayes model has none")
        return
    prior, K = b.model.priors.b, len(b.model.spec.columns)
    p = {k: float(v) for k, v in prior.params.items()}
    if prior.name == "Horseshoe":
        mine = {"Kscales": K, "hs_df": p["df"], "hs_df_global": p["df_global"], "hs_df_slab": p["df_slab"],
                "hs_scale_global": p["scale_global"], "hs_scale_slab": p["scale_slab"]}
    else:
        cons = np.empty(K)
        cons[[b.cmap.b_index(k) for k in b.cmap.x_column]] = concentration(prior, K)   # in brms's order of b
        mine = {"Kscales": K, "R2D2_mean_R2": p["mean_R2"], "R2D2_prec_R2": p["prec_R2"], "R2D2_cons_D2": cons}
    if present != sorted(mine):
        raise OracleMismatch(f"brms has the shrinkage data {present}; the AutoBayes model's {prior.name} prior has {sorted(mine)}")
    for key, value in mine.items():
        theirs = np.asarray(sd[key], dtype=float)
        if theirs.shape != np.shape(value) or abs_err(theirs, value) > DESIGN_ABS:
            raise OracleMismatch(f"brms {key} is {sd[key]!r}; the AutoBayes model's {prior} gives {value!r}")


def check_shrinkage(b: Bundle) -> None:
    """G3 of a shrinkage prior on b (D07): at every point the graph's coefficients (each term's deterministic, in design-column
    order), their scales ``sdb`` and R2D2's ``R2D2_tau2`` equal brms's transformed parameters ``b``, ``sdb`` and ``R2D2_tau2``
    (brms's vectors in its order of b, mapped through the column map)."""
    roles = getattr(b.model, "shrinkage_names", {})
    if not roles:
        return
    pmm = b.model.pymc_model
    terms = [t.name for t in b.model.spec.bambi_terms]
    tau2 = [roles["R2D2_tau2"]] if "R2D2_tau2" in roles else []
    fn = pmm.compile_fn(pmm.replace_rvs_by_values([pmm[v] for v in (roles["sdb"], *terms, *tau2)]), inputs=pmm.value_vars,
                        point_fn=True, on_unused_input="ignore")
    order = [b.cmap.b_index(k) for k in b.cmap.x_column]
    for k, p, vp in b.points():
        sdb, *rest = (np.atleast_1d(np.asarray(v, dtype=float)) for v in fn(vp))
        coefs = np.concatenate(rest[:len(terms)])
        pairs = [("sdb", sdb, np.atleast_1d(p.pars["sdb"])[order]), ("b", coefs, adapter.coefficients(p.pars)[order])]
        pairs += [("R2D2_tau2", rest[-1], np.atleast_1d(p.pars["R2D2_tau2"]))] if tau2 else []
        for what, mine, theirs in pairs:
            err = rel_err(mine, theirs.astype(float))
            if err > PRED_REL:
                raise OracleMismatch(f"point {k}: the graph's {what} differs from brms's {what} by {err}")


def check_derived_parameters(b: Bundle) -> None:
    """G3 of a family parameter brms derives in its model block from a sampled one (E02, ``ExtraParameter.derived``: the GEV's
    ``xi`` from ``tmp_xi``, the data, ``mu`` and ``sigma``): at every point the graph's deterministic of it equals the fixture's
    record of brms's ``posterior_epred(dpar = name)`` (``adapter.derived_draws``: one value per point, the same on every row); a
    record of a parameter the model does not derive is a mismatch."""
    derived = [e for e in b.model.family_spec.extra_parameters if e.derived]
    recorded = sorted(((b.fixture.raw.get("post") or {}).get("dpar_epred") or {}))
    if sorted(e.name for e in derived) != recorded:
        raise OracleMismatch(f"brms's dpar_epred records {recorded}; the AutoBayes model derives {sorted(e.name for e in derived)}")
    if not derived:
        return
    pmm = b.model.pymc_model
    fn = pmm.compile_fn(pmm.replace_rvs_by_values([pmm[b.model.names[e.name]] for e in derived]), inputs=pmm.value_vars,
                        point_fn=True, on_unused_input="ignore")
    brms = {e.name: adapter.derived_draws(b.fixture, e.name) for e in derived}
    for k, p, vp in b.points():
        for e, value in zip(derived, fn(vp)):
            err = rel_err(np.asarray(value, dtype=float), brms[e.name][k])
            if err > PRED_REL:
                raise OracleMismatch(f"point {k}: the graph's {e.name} {float(value)!r} differs from brms's posterior_epred(dpar = "
                                     f"{e.name!r}) {brms[e.name][k]!r} by {err}")


def check_latent(b: Bundle) -> None:
    """G3 of an ``me()`` term (C06): the graph's latent values ``Xme_<x>`` equal brms's ``Xme_k`` at every point."""
    measurement = b.cmap.measurement or {}
    if not measurement:
        return
    pmm = b.model.pymc_model
    names = [b.model.me_names[x]["latent"] for x in measurement]
    fn = pmm.compile_fn(pmm.replace_rvs_by_values([pmm[v] for v in names]), inputs=pmm.value_vars, point_fn=True, on_unused_input="ignore")
    for k, p, vp in b.points():
        for (x, g), value in zip(measurement.items(), fn(vp)):
            err = rel_err(np.asarray(value, dtype=float), np.atleast_1d(p.pars[f"Xme_{g}"]))
            if err > PRED_REL:
                raise OracleMismatch(f"point {k}: the graph's latent values of {x!r} differ from brms Xme_{g} by {err}")


# -- G4 -----------------------------------------------------------------------------------------------------
def check_predictions(b: Bundle) -> None:
    if b.reference is not None and "epred" in b.reference.fallback:         # P6: brms where it is accurate, the reference where not
        return two_reference.check_predictions(b)
    if b.case["brms"].get("prediction_resp"):                               # F06c: each response's own predictions (subset())
        return check_response_predictions(b)
    idata = adapter.draws(b.fixture, b.model, b.cmap, b.gmap)
    # brms's posterior_epred of a family with trials is trials * mu, predict()'s "expected" (D5): the binomial's and, since D02,
    # the beta-binomial's
    mean_var = "expected" if b.model.family_spec.has_trials else b.model.names["mean"]
    for block, nd in (("fitted", None), ("newdata", b.newdata)):
        if block == "newdata" and nd is None:
            continue
        for include, suffix in ((True, ""), (False, "_nogroup")):
            ref_lin = b.fixture.prediction(block, "linpred" + suffix)
            ref_ep = b.fixture.prediction(block, "epred" + suffix)
            # Tier F: brms fails to compute posterior_epred for some cases (logistic_normal; sar type "error"), recorded as
            # epred_refused; the linear predictor is still checked
            epred_refused = ref_ep is None and b.fixture.prediction_refusal(block, "epred" + suffix) is not None
            if ref_lin is None or (ref_ep is None and not epred_refused):
                if not include and b.model.spec.groups:
                    raise OracleMismatch(f"fixture lacks the {block} predictions without group effects")
                continue
            lin = b.model.predict(idata, newdata=nd, kind="linear", include_group_effects=include)["eta"].to_numpy()[0]
            err = rel_err(lin, ref_lin)
            if err > PRED_REL:
                raise OracleMismatch(f"{block}{suffix}: linear predictor differs from brms posterior_linpred by {err}")
            if epred_refused:
                continue
            mean = b.model.predict(idata, newdata=nd, kind="mean", include_group_effects=include)[mean_var].to_numpy()[0]
            err = rel_err(mean, ref_ep)
            if err > PRED_REL:
                raise OracleMismatch(f"{block}{suffix}: {mean_var!r} differs from brms posterior_epred by {err}")


def check_response_predictions(b: Bundle) -> None:
    """G4 of a multivariate model whose joint predictions brms refuses and records response by response (F06c, a case's
    ``prediction_resp``: ``subset()``): ``predict()`` without ``resp`` must refuse each joint prediction brms refuses (a
    ``ValueError``), and ``predict(resp=r)``, on the fitted rows and on ``newdata``, with and without group effects, must equal
    brms's ``posterior_linpred`` / ``posterior_epred(resp = r)`` on the rows of ``r``'s subset (``Fixture.response_prediction``),
    row for row."""
    idata = adapter.draws(b.fixture, b.model, b.cmap, b.gmap)
    mean_var = b.model.names["mean"]
    for block, nd in (("fitted", None), ("newdata", b.newdata)):
        if block == "newdata" and nd is None:
            continue
        for brms_kind, kind in (("linpred", "linear"), ("epred", "mean")):
            if b.fixture.prediction_refusal(block, brms_kind) is None:
                continue
            try:
                b.model.predict(idata, newdata=nd, kind=kind)
            except ValueError as e:
                if "resp" not in str(e):                                    # refused, but not for brms's reason
                    raise OracleMismatch(f"{block}: the joint posterior_{brms_kind} is refused for another reason than brms's: {e}") from None
                continue
            raise OracleMismatch(f"{block}: brms refuses the joint posterior_{brms_kind} ({b.fixture.prediction_refusal(block, brms_kind)!r}); "
                                 "the AutoBayes model returns it")
        for r in b.case["brms"]["prediction_resp"]:
            for include, suffix in ((True, ""), (False, "_nogroup")):
                for brms_kind, kind, var in (("linpred", "linear", "eta"), ("epred", "mean", mean_var)):
                    ref = b.fixture.response_prediction(block, r, brms_kind + suffix)
                    if ref is None:
                        raise OracleMismatch(f"fixture lacks the {block} {brms_kind}{suffix} of the response {r!r}")
                    ours = b.model.predict(idata, newdata=nd, kind=kind, resp=r, include_group_effects=include)[var].to_numpy()[0]
                    if ours.shape != ref.shape:
                        raise OracleMismatch(f"{block}{suffix}: the response {r!r} is predicted on {ours.shape[-1]} rows; brms's "
                                             f"posterior_{brms_kind}(resp = {r!r}) on {ref.shape[-1]}")
                    err = rel_err(ours, ref)
                    if err > PRED_REL:
                        raise OracleMismatch(f"{block}{suffix}: the response {r!r}'s {var!r} differs from brms posterior_{brms_kind}"
                                             f"(resp = {r!r}) by {err}")


# -- G5 -----------------------------------------------------------------------------------------------------
def check_jax(b: Bundle) -> None:
    from pymc.sampling.jax import get_jaxified_logp

    jf = get_jaxified_logp(b.model.pymc_model)
    names = [v.name for v in b.model.pymc_model.value_vars]
    for k, p, vp in b.points():
        a, c = float(jf([vp[n] for n in names])), float(b.logp(vp))
        if rel_err(a, c) > LOGP_REL:
            raise OracleMismatch(f"point {k}: JAX log density {a!r} vs PyTensor {c!r}")


# -- G6 -----------------------------------------------------------------------------------------------------
_NUMBER = re.compile(r"(?<![A-Za-z_0-9.])[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def normalise_prior(text: str) -> str:
    """A prior string or bound with the whitespace removed and every number written as the double it denotes, so that
    ``student_t(3, 0.9, 2.5)`` and ``student_t(3,0.9,2.5)``, ``3`` and ``3.0``, ``1e+05`` and ``100000`` compare equal and
    nothing else does (brms prints 15 significant digits; ``ab.default_priors`` reproduces them, so equal strings are
    equal doubles)."""
    return _NUMBER.sub(lambda m: repr(float(m.group())), re.sub(r"\s+", "", text))


def default_prior_table(b: Bundle) -> list[dict[str, str]]:
    """``ab.default_priors`` for the case's formula, data, family and keyword arguments, written as brms's table for the
    case's model (``Priors.table``). brms's ``default_prior()`` is computed from the formula, data and family alone, so
    it describes brms's default model, which learns a Student-t ``nu``: a case that fixes nu (brms ``constant(4)``) is
    compared under ``nu=None``."""
    spec = b.case["autobayes"]
    kwargs = dict(spec.get("kwargs", {}))
    if "nu" in b.model.family_spec.learnable_parameters:
        kwargs["nu"] = None
    if b.data2:                                               # Tier F: brms's default_prior() reads data2 (car, sar, fcor)
        kwargs["data2"] = b.data2
    priors = ab.default_priors(spec["formula"], b.data, family=b.model.family, **kwargs)
    return priors.table(b.model.spec, b.model.family_spec, center_predictors=b.model.effective_center_predictors,
                        categories=getattr(b.model, "categories", ()) or None)   # the ordinal and categorical tables need them (B05)


def check_default_prior(b: Bundle) -> None:
    if adapter.is_multivariate(b.model):                      # C01
        return _check_default_prior_multivariate(b)
    classes = {row["class"] for row in b.fixture.default_prior}
    expected = set()
    if b.model.spec.intercept and b.cmap.centred_intercept:
        expected.add("Intercept")
    if b.model.spec.coefficient_labels or (b.model.spec.intercept and not b.cmap.centred_intercept):
        expected.add("b")                                  # a design column or a monotonic term's coefficient (B03)
    if b.model.spec.mo_terms:                              # a monotonic term's simplex (B03)
        expected.add("simo")
    if getattr(b.model.spec, "gp_terms", ()):              # a Gaussian process's marginal SD and length-scale (C03)
        expected |= {"sdgp", "lscale"}
    if b.model.spec.groups:
        expected.add("sd")
    if b.model.correlation_names:                          # a correlated group term's LKJ prior (A02)
        expected.add("cor")
    distributional = b.model.spec.dpar_names               # an auxiliary parameter with its own formula has no class row (B01)
    if "sigma" in b.model.names and "sigma" not in distributional:
        expected.add("sigma")
    if "shape" in b.model.names and "shape" not in distributional:
        expected.add("shape")
    if "phi" in b.model.names and "phi" not in distributional:
        expected.add("phi")
    dpar_rows = {(row["class"], row.get("dpar", "")) for row in b.fixture.default_prior}
    for name in distributional:                            # its formula's Intercept and b rows, dpar = name
        ds = b.model.spec.dpar(name)
        wanted = ({("Intercept", name)} if ds.intercept else set()) | ({("b", name)} if ds.columns else set())
        if not wanted <= dpar_rows:
            raise OracleMismatch(f"brms default_prior lacks the rows {sorted(wanted - dpar_rows)} of {name}'s formula")
    if "nu" in b.model.names:
        expected.add("nu")
    expected |= {par for par in ("zi", "hu") if par in b.model.names and par not in distributional}     # B06; E03: its dpar rows
    expected |= {e.name for e in b.model.family_spec.extra_parameters if e.name not in distributional}     # D01: alpha, beta, ndt
    mixture = getattr(b.model, "mixture_names", {})        # C05: each component's sigma<k> and theta (Intercept, b above)
    expected |= {*mixture["sigma"], "theta"} if mixture else set()
    expected |= {part for part in ("ar", "ma") if part in getattr(b.model, "autocor_names", {})}   # C04: an ARMA term's ar and ma
    expected |= set(getattr(b.model, "sar_names", {}))     # F02b: a SAR term's lagsar or errorsar
    timecor = getattr(b.model.spec, "timecor", None)       # F03b: unstr()'s cortime or cosy()'s cosy
    expected |= {timecor.prior_class} if timecor is not None else set()
    expected |= {"meanme", "sdme"} if b.model.spec.me_terms else set()     # C06: an me() term's latent values
    expected |= set(b.model.spec.car.classes) if getattr(b.model.spec, "car", None) is not None else set()   # F02a: sdcar, car, rhocar
    if not expected <= classes:
        raise OracleMismatch(f"brms default_prior classes {sorted(classes)} lack {sorted(expected - classes)}")
    if adapter.is_nonlinear(b.model):                              # C02: each parameter's b (and sd) rows, cor by group
        nl_rows = {(row["class"], row.get("nlpar", "")) for row in b.fixture.default_prior}
        wanted = {("b", d.response) for d in b.model.spec.nlpars} | {("sd", d.response) for d in b.model.spec.nlpars if d.group_terms}
        wanted |= {("cor", "")} if b.model.spec.correlated_factors else set()
        if not wanted <= nl_rows:
            raise OracleMismatch(f"brms default_prior lacks the non-linear parameters' rows {sorted(wanted - nl_rows)}")
    if adapter.is_multinomial(b.model):                            # F01a: each category's Intercept, b and sd rows by dpar
        dp_rows = {(row["class"], row.get("dpar", "")) for row in b.fixture.default_prior}
        wanted = {(cls, d.response) for d in b.model.spec.mus
                  for cls, has in (("Intercept", True), ("b", bool(d.columns)), ("sd", bool(d.group_terms))) if has}
        if not wanted <= dp_rows:
            raise OracleMismatch(f"brms default_prior lacks the categories' rows {sorted(wanted - dp_rows)}")
    if not b.fixture.prior_summary:
        raise OracleMismatch("fixture has no prior_summary table")
    # ab.default_priors() reproduces brms's table: every row, in brms's order, every column (A07)
    ours, theirs = default_prior_table(b), b.fixture.default_prior

    def key(row):
        if row is None:
            return None
        return tuple(normalise_prior(row.get(k, "")) if k in ("prior", "lb", "ub") else row.get(k, "")
                     for k in ("prior", "class", "coef", "group", "resp", "dpar", "nlpar", "lb", "ub", "tag", "source"))
    diff = [f"  row {i}: AutoBayes {key(a)} vs brms {key(t)}"
            for i, (a, t) in enumerate(zip_longest(ours, theirs)) if key(a) != key(t)]
    if diff:
        raise OracleMismatch(f"ab.default_priors() written as brms's table differs from brms's default_prior() "
                             f"({len(ours)} rows vs {len(theirs)}):\n" + "\n".join(diff[:12]))


# -- non-linear models (C02) ---------------------------------------------------------------------------------------
def check_nonlinear_design(b: Bundle) -> None:
    """G1 of a non-linear model: the response; no linear design, trials, offsets or addition-term data on either side; brms's
    covariates C_k, each parameter's X_<nlpar> (with K_<nlpar>; never centred) and its group designs matched by content (the
    adapter's maps, rebuilt here) and of the shapes the model has."""
    sd, d = b.fixture.standata, b.model.design
    err = abs_err(sd["Y"], d.y)
    if err > DESIGN_ABS:
        raise OracleMismatch(f"response differs by {err}")
    for key in ("X", "trials", "offsets", "weights", "cens", "lb", "ub"):
        if key in sd:
            raise OracleMismatch(f"brms has {key}, which a non-linear model of this slice has not")
    covariates = sorted(k for k in sd if k.startswith("C_") and k[2:].isdigit())
    if len(covariates) != len(b.model.spec.covariates):
        raise OracleMismatch(f"brms has the covariates {covariates}; AutoBayes's non-linear formula reads {list(b.model.spec.covariates)}")
    for name, nd in d.nlpars.items():
        if sd[f"X_{name}"].shape != nd.X.shape or int(sd[f"K_{name}"]) != nd.X.shape[1]:
            raise OracleMismatch(f"brms X_{name} is {sd[f'X_{name}'].shape} with K_{name} = {int(sd[f'K_{name}'])}; AutoBayes's design of "
                                 f"{name!r} is {nd.X.shape}")
    adapter.match_nonlinear_columns(b.fixture, b.model)
    adapter.match_nonlinear_groups(b.fixture, b.model)


def check_nonlinear_derived(b: Bundle) -> None:
    """G3 of a non-linear model at every point: the graph's non-linear predictor (``mu``) against brms's posterior_linpred,
    each parameter's values per row against brms's own linear predictor of it (``adapter.brms_nlpar_values``: X_<nlpar>
    b_<nlpar> plus its group effects, from the fixture alone), every group-level coefficient's effects against r_<id>_<nlpar>_<k>,
    and every correlation matrix against Cor_<id>."""
    pmm, model = b.model.pymc_model, b.model
    nlpars = list(model.nlpar_names.items())
    groups = list(model.nlpar_group_names.items())
    cors = list(model.nlpar_correlation_names.items())
    dets = ([pmm[model.names["linpred"]]] + [pmm[roles["value"]] for _, roles in nlpars] + [pmm[roles["group_effect"]] for _, roles in groups]
            + [pmm[roles["cor"]] for _, roles in cors])
    fn = pmm.compile_fn(pmm.replace_rvs_by_values(dets), inputs=pmm.value_vars, point_fn=True, on_unused_input="ignore")
    ref = b.fixture.prediction("fitted", "linpred")
    for k, p, vp in b.points():
        out = [np.asarray(v, dtype=float) for v in fn(vp)]
        err = rel_err(out[0].ravel(), ref[k])
        if err > PRED_REL:
            raise OracleMismatch(f"point {k}: the graph's non-linear predictor {model.names['linpred']!r} differs from brms by {err}")
        for (name, _), value in zip(nlpars, out[1:1 + len(nlpars)]):
            err = rel_err(value.ravel(), adapter.brms_nlpar_values(b.fixture, p, name))
            if err > PRED_REL:
                raise OracleMismatch(f"point {k}: the graph's values of the non-linear parameter {name!r} differ from brms's linear "
                                     f"predictor of {name} by {err}")
        start = 1 + len(nlpars)
        for ((name, g, coef), _), eff in zip(groups, out[start:start + len(groups)]):
            m, cn = b.gmap.brms_id[(name, g)], b.gmap.coef_cn[(name, g, coef)]
            err = rel_err(eff.ravel()[b.gmap.level_of[(name, g)]], np.atleast_1d(p.pars[f"r_{m}_{name}_{cn}"]))
            if err > PRED_REL:
                raise OracleMismatch(f"point {k}: group effects of {coef!r} of {g!r} in {name!r} differ from brms r_{m}_{name}_{cn} by {err}")
        start += len(groups)
        for ((name, g), _), cor in zip(cors, out[start:]):
            m = b.gmap.brms_id[(name, g)]
            perm = [b.gmap.coef_cn[(name, g, c)] - 1 for c in model.spec.nlpar(name).group_term_of(g).coefs]
            err = rel_err(cor, np.atleast_2d(p.pars[f"Cor_{m}"])[np.ix_(perm, perm)])
            if err > PRED_REL:
                raise OracleMismatch(f"point {k}: the correlation matrix of {g!r} in {name!r} differs from brms Cor_{m} by {err}")


# -- count-matrix responses (F01a) ---------------------------------------------------------------------------------
def check_multinomial_design(b: Bundle) -> None:
    """G1 of a count-matrix model: brms's ``Y`` is the model's (rows, K) count matrix, ``ncat`` its number of categories and
    ``trials`` its trials; no linear design, offsets or addition-term data on either side; each category's ``X_<mu>`` (with
    ``K_<mu>``, ``Kc_<mu>``) and group designs matched by content (the adapter's maps, rebuilt here)."""
    sd, d = b.fixture.standata, b.model.design
    Y = np.asarray(sd["Y"], dtype=float).reshape(d.n, -1)
    if Y.shape != d.y.shape or abs_err(Y, d.y) > DESIGN_ABS:
        raise OracleMismatch(f"brms's Y {Y.shape} differs from the AutoBayes count matrix {d.y.shape}")
    if int(sd["ncat"]) != len(b.model.categories):
        raise OracleMismatch(f"brms has ncat = {int(sd['ncat'])}; AutoBayes has the categories {list(b.model.categories)}")
    if "trials" not in sd or abs_err(sd["trials"], d.trials) > DESIGN_ABS:
        raise OracleMismatch("brms's trials differ from the AutoBayes model's")
    for key in ("X", "offsets", "weights", "cens", "lb", "ub"):
        if key in sd:
            raise OracleMismatch(f"brms has {key}, which a count-matrix model of this slice has not")
    for name, md in d.mus.items():
        k = len(md.columns)
        if int(sd[f"K_{name}"]) != k + 1 or int(sd[f"Kc_{name}"]) != k:
            raise OracleMismatch(f"brms has K_{name} = {int(sd[f'K_{name}'])}, Kc_{name} = {int(sd[f'Kc_{name}'])}; AutoBayes's design of "
                                 f"{name!r} has {k} columns and an intercept")
    adapter.match_multinomial_columns(b.fixture, b.model)
    adapter.match_multinomial_groups(b.fixture, b.model)


def check_multinomial_derived(b: Bundle) -> None:
    """G3 of a count-matrix model at every point: the graph's ``mu`` (rows, K - 1) against brms's posterior_linpred, each
    category's values against brms's own linear predictor of it (``adapter.brms_mu_values``, from the fixture alone), its
    raw-scale intercept against ``b_<mu>_Intercept`` and its group effects against ``r_<id>_<mu>_1``."""
    pmm, model = b.model.pymc_model, b.model
    mus = list(model.mu_names.items())
    groups = list(model.mu_group_names.items())
    dets = ([pmm[model.names["linpred"]]] + [pmm[roles["value"]] for _, roles in mus] + [pmm[roles["raw_intercept"]] for _, roles in mus]
            + [pmm[roles["group_effect"]] for _, roles in groups])
    fn = pmm.compile_fn(pmm.replace_rvs_by_values(dets), inputs=pmm.value_vars, point_fn=True, on_unused_input="ignore")
    ref = b.fixture.prediction("fitted", "linpred")
    for k, p, vp in b.points():
        out = [np.asarray(v, dtype=float) for v in fn(vp)]
        err = rel_err(out[0], ref[k])
        if err > PRED_REL:
            raise OracleMismatch(f"point {k}: the graph's {model.names['linpred']!r} differs from brms's posterior_linpred by {err}")
        for (name, _), value, raw in zip(mus, out[1:1 + len(mus)], out[1 + len(mus):1 + 2 * len(mus)]):
            err = rel_err(value, adapter.brms_mu_values(b.fixture, p, name))
            if err > PRED_REL:
                raise OracleMismatch(f"point {k}: the graph's values of {name!r} differ from brms's linear predictor of {name} by {err}")
            err = rel_err(raw, p.pars[f"b_{name}_Intercept"])
            if err > PRED_REL:
                raise OracleMismatch(f"point {k}: the raw-scale intercept of {name!r} differs from brms's b_{name}_Intercept by {err}")
        for ((name, g, _), _), eff in zip(groups, out[1 + 2 * len(mus):]):
            m = b.gmap.brms_id[(name, g)]
            err = rel_err(eff[b.gmap.level_of[(name, g)]], np.atleast_1d(p.pars[f"r_{m}_{name}_1"]))
            if err > PRED_REL:
                raise OracleMismatch(f"point {k}: group effects of {g!r} in {name!r} differ from brms r_{m}_{name}_1 by {err}")


ALL = (check_design, check_logp, check_gradient, check_derived, check_predictions, check_jax, check_default_prior)


# -- C07: post-processing -----------------------------------------------------------------------------------------
def fixed_parameter_draws(b: Bundle):
    """The fixture's points as brms's fixed-parameter fit holds them (``gen_fixture.R``'s ``Fixed_param`` run): one chain
    per point with one draw each, so that the relative efficiencies brms hands loo, and ``ab.loo``'s default ones, are
    those of one-draw chains (loo's NA, used as 1)."""
    import arviz as az

    post = adapter.draws(b.fixture, b.model, b.cmap, b.gmap).posterior
    S = post.sizes["draw"]
    chains = post.isel(chain=0, drop=True).rename({"draw": "chain"}).assign_coords(chain=np.arange(S))
    return az.InferenceData(posterior=chains.expand_dims(draw=[0], axis=1))


def compare_post(what: str, ours, theirs) -> None:
    """``ours`` equals brms's ``theirs`` to ``PRED_REL`` where brms's is finite, and a non-finite entry of either is the
    same non-finite value in the other (Inf with Inf, -Inf with -Inf, NaN with NaN)."""
    ours, theirs = np.asarray(ours, dtype=float), np.asarray(theirs, dtype=float)
    if ours.shape != theirs.shape:
        raise OracleMismatch(f"{what}: AutoBayes has shape {ours.shape}, brms {theirs.shape}")
    finite_o, finite_t = np.isfinite(ours), np.isfinite(theirs)
    same = (ours == theirs) | (np.isnan(ours) & np.isnan(theirs))
    differ = (finite_o != finite_t) | (~finite_o & ~finite_t & ~same)
    if differ.any():
        where = [tuple(int(i) for i in ix) for ix in np.argwhere(differ)[:5]]
        raise OracleMismatch(f"{what}: non-finite entries differ at {where}: AutoBayes {ours[differ][:5].tolist()} vs brms "
                             f"{theirs[differ][:5].tolist()}")
    err = rel_err(ours[finite_o], theirs[finite_t])
    if err > PRED_REL:
        raise OracleMismatch(f"{what} differs from brms by {err}")


def check_post(b: Bundle) -> None:
    """C07: each post-processing function the case lists (``b.post``) on ``fixed_parameter_draws`` equals the fixture's
    ``post`` block: ``log_lik`` brms's ``log_lik()`` (draws x rows); ``bayes_R2`` brms's ``bayes_R2(summary = FALSE)``
    (draws x 1); ``loo`` brms's ``loo()``: the pointwise table (its five columns, by name), the estimates (rows elpd_loo,
    p_loo, looic; columns Estimate, SE), ``pareto_k`` and ``n_eff``; ``waic`` brms's ``waic()`` (D05, ``check_waic``);
    ``loo_compare`` brms's ``loo_compare()`` and ``model_weights()`` of this model and the case it names (D05,
    ``check_loo_compare``); ``hypothesis`` and ``conditional_effects`` brms's ``hypothesis()`` and ``conditional_effects()``
    of the case's ``post_args`` (D06, ``check_hypothesis``, ``check_conditional_effects``); ``prior_draws`` brms's
    ``prior_draws()`` columns (E05, ``check_prior_draws``; with it, the hypotheses' evidence ratios come from the fixture's
    prior draws) and ``loo_model_weights`` brms's stacking weights (E05, ``check_loo_model_weights``). A case that lists none
    has nothing to compare."""
    if not b.post:
        return
    post = b.fixture.raw.get("post")
    if not post:
        raise OracleMismatch(f"the case lists {sorted(b.post)} but the fixture has no post block")
    idata = fixed_parameter_draws(b)
    S = len(b.fixture.points)
    for name, fn in b.post.items():
        if name not in post:
            raise OracleMismatch(f"the case lists {name} but the fixture's post block has {sorted(post)}")
        if name == "log_lik":
            ours = fn(b.model, idata).transpose("chain", "draw", "__obs__").to_numpy().reshape(S, -1)
            compare_post("log_lik", ours, fx.decode(post["log_lik"]))
        elif name == "bayes_R2":
            ours = fn(b.model, idata).transpose("chain", "draw").to_numpy().reshape(S, 1)
            compare_post("bayes_R2", ours, fx.decode(post["bayes_R2"]))
        elif name == "loo":
            res, ref = fn(b.model, idata), post["loo"]
            columns = list(ref["pointwise"].get("colnames") or [])
            if list(res.pointwise.columns) != columns:
                raise OracleMismatch(f"loo pointwise columns {list(res.pointwise.columns)} vs brms {columns}")
            table = fx.decode(ref["pointwise"])
            if res.pointwise.shape != table.shape:
                raise OracleMismatch(f"loo pointwise: AutoBayes has shape {res.pointwise.shape}, brms {table.shape}")
            for i, column in enumerate(columns):
                compare_post(f"loo pointwise {column}", res.pointwise[column].to_numpy(), table[:, i])
            if list(res.estimates.index) != list(ref["estimate_rows"]) or list(res.estimates.columns) != list(ref["estimates"]["colnames"]):
                raise OracleMismatch(f"loo estimates {list(res.estimates.index)} x {list(res.estimates.columns)} vs brms "
                                     f"{list(ref['estimate_rows'])} x {list(ref['estimates']['colnames'])}")
            compare_post("loo estimates", res.estimates.to_numpy(), fx.decode(ref["estimates"]))
            compare_post("loo pareto_k", res.pareto_k, fx.decode(ref["pareto_k"]))
            compare_post("loo n_eff", res.n_eff, fx.decode(ref["n_eff"]))
        elif name == "waic":                                         # D05
            check_waic(fn(b.model, idata), post["waic"])
        elif name == "loo_compare":                                  # D05
            check_loo_compare(b, idata, post["loo_compare"])
        elif name == "hypothesis":                                   # D06
            check_hypothesis(b, idata, post["hypothesis"])
        elif name == "conditional_effects":                          # D06
            check_conditional_effects(b, idata, post["conditional_effects"])
        elif name == "prior_draws":                                  # E05
            check_prior_draws(b, post["prior_draws"])
        elif name == "loo_model_weights":                            # E05
            check_loo_model_weights(b, idata, post["loo_model_weights"])
        else:
            raise OracleMismatch(f"no comparison for the post-processing function {name!r}")


# -- D05: waic, loo_compare and model_weights ---------------------------------------------------------------------
def check_waic(res, ref) -> None:
    """D05: ``ab.waic`` against brms's ``waic()``: the pointwise table (its three columns, by name) and the estimates (rows
    elpd_waic, p_waic, waic; columns Estimate, SE)."""
    columns = list(ref["pointwise"].get("colnames") or [])
    if list(res.pointwise.columns) != columns:
        raise OracleMismatch(f"waic pointwise columns {list(res.pointwise.columns)} vs brms {columns}")
    table = fx.decode(ref["pointwise"])
    if res.pointwise.shape != table.shape:
        raise OracleMismatch(f"waic pointwise: AutoBayes has shape {res.pointwise.shape}, brms {table.shape}")
    for i, column in enumerate(columns):
        compare_post(f"waic pointwise {column}", res.pointwise[column].to_numpy(), table[:, i])
    if list(res.estimates.index) != list(ref["estimate_rows"]) or list(res.estimates.columns) != list(ref["estimates"]["colnames"]):
        raise OracleMismatch(f"waic estimates {list(res.estimates.index)} x {list(res.estimates.columns)} vs brms "
                             f"{list(ref['estimate_rows'])} x {list(ref['estimates']['colnames'])}")
    compare_post("waic estimates", res.estimates.to_numpy(), fx.decode(ref["estimates"]))


def check_loo_compare(b: Bundle, idata, ref) -> None:
    """D05: brms's ``loo_compare()`` of this case's model with the case its autobayes section names (``compare_with``), on the
    same data, each on its own frozen points held as brms's fixed-parameter fit holds them. The other model must be the target
    brms evaluated (its log density at its points equals the fixture's ``other_lp`` to ``LOGP_REL``); then both models'
    pointwise elpd_loo and elpd_waic, ``ab.loo_compare`` by loo and by waic (rows best first, by name; loo's eight columns, by
    name) and ``ab.model_weights`` by ``"loo"`` and by ``"pseudobma"`` with ``BB=False`` (in the order given) equal brms's.
    ``compare_post``'s ``max(1, |a|, |b|)`` scale compares a weight far below one absolutely, so the weights' logarithms are
    compared too: a weight of 1e-115 must be right to ``PRED_REL`` in its exponent."""
    other_id = b.case.get("autobayes", {}).get("compare_with")
    models = list(ref["models"])
    if not other_id or models != [b.case["id"], other_id]:
        raise OracleMismatch(f"loo_compare: the fixture compares {models}; the case compares {[b.case['id'], other_id]}")
    other = Bundle(other_id)
    lp = np.array([other.logp(vp) for _, _, vp in other.points()])
    err = rel_err(lp, fx.decode(ref["other_lp"]))
    if err > LOGP_REL:
        raise OracleMismatch(f"loo_compare: {other_id}'s log density at its points differs from brms's other_lp by {err}")
    fits = ((b.model, idata), (other.model, fixed_parameter_draws(other)))
    results = {"loo": {m: ab.loo(model, d) for m, (model, d) in zip(models, fits)},
               "waic": {m: ab.waic(model, d) for m, (model, d) in zip(models, fits)}}
    for crit in ("loo", "waic"):
        elpd = ref[f"elpd_{crit}"]
        if list(elpd.get("colnames") or []) != models:
            raise OracleMismatch(f"loo_compare elpd_{crit}: brms's columns {elpd.get('colnames')} are not {models}")
        ours = np.column_stack([results[crit][m].pointwise[f"elpd_{crit}"].to_numpy() for m in models])
        compare_post(f"loo_compare pointwise elpd_{crit}", ours, fx.decode(elpd))
        table, theirs = b.post["loo_compare"](results[crit]), ref[crit]
        if list(table.index) != list(theirs["rows"]):
            raise OracleMismatch(f"loo_compare by {crit}: rows {list(table.index)} vs brms {list(theirs['rows'])}")
        if list(table.columns) != list(theirs["table"].get("colnames") or []):
            raise OracleMismatch(f"loo_compare by {crit}: columns {list(table.columns)} vs brms {theirs['table'].get('colnames')}")
        compare_post(f"loo_compare by {crit}", table.to_numpy(), fx.decode(theirs["table"]))
    for kind, kwargs in (("loo", {}), ("pseudobma", {"BB": False})):
        w = ab.model_weights(results["loo"], kind, **kwargs)
        if list(w.index) != list(ref["weights_models"]):
            raise OracleMismatch(f"model_weights {kind}: models {list(w.index)} vs brms {list(ref['weights_models'])}")
        theirs = fx.decode(ref[f"weights_{kind}"])
        compare_post(f"model_weights {kind}", w.to_numpy(), theirs)
        with np.errstate(divide="ignore"):
            compare_post(f"model_weights {kind} (log)", np.log(w.to_numpy()), np.log(theirs))


# -- D06: hypothesis and conditional_effects ---------------------------------------------------------------------
def _post_args(b: Bundle, name: str) -> list:
    """The arguments the case hands ``name`` (its autobayes section's ``post_args``), which must be what brms was given (the
    frozen brms section's list of that name)."""
    args = list((b.case.get("autobayes", {}).get("post_args") or {}).get(name) or [])
    frozen = list(b.case["brms"].get(name) or [])
    if not args or args != frozen:
        raise OracleMismatch(f"{name}: the case's post_args {args} are not brms's {frozen}")
    return args


def _compare_table_column(what: str, ours, theirs) -> None:
    """One column of a brms table from the fixture (``{"dims", "values"}`` numbers, or a list of text with ``None`` for
    brms's NA) against the same column of an AutoBayes table: numbers with ``compare_post``, text exactly."""
    if isinstance(theirs, dict):
        try:
            values = np.asarray(ours, dtype=float)
        except (TypeError, ValueError):
            raise OracleMismatch(f"{what}: AutoBayes's column is not numeric ({list(ours)[:3]}...); brms's is") from None
        compare_post(what, values, fx.decode(theirs))
        return
    text = [None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v) for v in ours]
    pairs = list(zip_longest(text, list(theirs)))
    diff = [i for i, (a, t) in enumerate(pairs) if a != t]
    if diff:
        raise OracleMismatch(f"{what} differs from brms at rows {diff[:5]}: AutoBayes {[pairs[i][0] for i in diff[:3]]} vs brms "
                             f"{[pairs[i][1] for i in diff[:3]]}")


def check_hypothesis(b: Bundle, idata, ref) -> None:
    """D06: ``ab.hypothesis`` of the case's hypotheses on the fixture's draws against brms's ``hypothesis()``: the table's
    columns by name and in order, each column (``Hypothesis`` and ``Star`` exactly, the numbers to ``PRED_REL`` with brms's
    NA, stored NaN, exactly), and the draws of each hypothesis, ``(draws, hypotheses)`` with brms's names ``H1``, ..."""
    hypotheses = _post_args(b, "hypothesis")
    prior = recorded_prior_draws(b)                                 # E05: brms's prior draws, for the Savage-Dickey ratios
    res = b.post["hypothesis"](b.model, idata, hypotheses, **({} if prior is None else {"prior": prior}))
    table = ref["table"]
    columns = list(table["columns"])
    if list(res.table.columns) != columns:
        raise OracleMismatch(f"hypothesis columns {list(res.table.columns)} vs brms {columns}")
    for column in columns:
        _compare_table_column(f"hypothesis {column}", res.table[column].tolist(), table["values"][column])
    names = list(ref["samples"].get("colnames") or [])
    if list(res.samples.columns) != names:
        raise OracleMismatch(f"hypothesis samples columns {list(res.samples.columns)} vs brms {names}")
    compare_post("hypothesis samples", res.samples.to_numpy(), fx.decode(ref["samples"]))


def check_conditional_effects(b: Bundle, idata, ref) -> None:
    """D06: ``ab.conditional_effects`` of the case's effects on the fixture's draws against brms's ``conditional_effects()``:
    the effects in brms's order, and for each its table's columns by name and in order and every column (the grid of the
    effect, the response and the other variables as held, the grouping factors' NA as None, ``cond__``, ``effect1__``,
    ``estimate__``, ``se__``, ``lower__``, ``upper__``), text exactly and numbers to ``PRED_REL``."""
    effects = _post_args(b, "conditional_effects")
    if effects != list(ref):
        raise OracleMismatch(f"conditional_effects: the case asks for {effects}; brms's tables are {list(ref)}")
    res = b.post["conditional_effects"](b.model, idata, effects)
    if list(res) != effects:
        raise OracleMismatch(f"conditional_effects returned {list(res)} for the effects {effects}")
    for effect in effects:
        table, theirs = res[effect], ref[effect]
        columns = list(theirs["columns"])
        if list(table.columns) != columns:
            raise OracleMismatch(f"conditional_effects {effect}: columns {list(table.columns)} vs brms {columns}")
        for column in columns:
            _compare_table_column(f"conditional_effects {effect} {column}", table[column].tolist(), theirs["values"][column])


# -- E05: prior draws, Savage-Dickey ratios and stacking weights ------------------------------------------------------
def recorded_prior_draws(b: Bundle):
    """E05: the fixture's ``post.prior_draws`` (brms's ``prior_draws()`` of the fixed-parameter fit with ``sample_prior =
    "yes"``) as a ``DataFrame`` with brms's column names, when the case lists ``prior_draws``; else None."""
    if "prior_draws" not in b.post:
        return None
    ref = (b.fixture.raw.get("post") or {}).get("prior_draws")
    if ref is None:
        raise OracleMismatch("the case lists prior_draws but the fixture's post block has none")
    return pd.DataFrame(fx.decode(ref), columns=list(ref.get("colnames") or []))


def check_prior_draws(b: Bundle, ref) -> None:
    """E05: ``ab.prior_draws`` of the case's model, as many draws as brms's, has brms's ``prior_draws()`` columns by name and
    in order (``Intercept``, ``b`` or one ``b_<coef>`` per coefficient, the family's parameters, ``sd_<group>`` ...), every
    draw finite, the SDs and the family's scale parameters positive, ``zi`` and ``hu`` in (0, 1), a correlation in (-1, 1).
    The draws themselves are random (brms's are Stan's RNG's) and are compared through what is computed from them:
    ``check_hypothesis`` hands the fixture's draws to ``ab.hypothesis``, whose evidence ratios must be brms's."""
    names = list(ref.get("colnames") or [])
    S = int(ref["dims"][0])
    ours = b.post["prior_draws"](b.model, draws=S, seed=1)
    if list(ours.columns) != names:
        raise OracleMismatch(f"prior_draws columns {list(ours.columns)} vs brms {names}")
    values = ours.to_numpy(dtype=float)
    if values.shape != (S, len(names)) or not np.isfinite(values).all():
        raise OracleMismatch(f"prior_draws: shape {values.shape} (brms ({S}, {len(names)})) or non-finite draws")
    for j, name in enumerate(names):
        v = values[:, j]
        if (name.startswith("sd_") or name in ("sigma", "shape", "phi", "nu")) and not (v > 0).all():
            raise OracleMismatch(f"prior_draws {name}: a draw is not positive")
        if name in ("zi", "hu") and not ((v > 0) & (v < 1)).all():
            raise OracleMismatch(f"prior_draws {name}: a draw is outside (0, 1)")
        if name.startswith("cor_") and not (np.abs(v) < 1).all():
            raise OracleMismatch(f"prior_draws {name}: a correlation outside (-1, 1)")


def check_loo_model_weights(b: Bundle, idata, ref) -> None:
    """E05: loo's ``loo_model_weights()`` (stacking, its default) of this model and the case it names (``compare_with``),
    which brms's ``model_weights(weights = "stacking")`` equals: the other model must be the target brms evaluated (its log
    density at its points equals the fixture's ``loo_compare.other_lp``), then ``ab.loo_model_weights`` and
    ``ab.model_weights(..., "stacking")`` of the two ``ab.loo()`` results, in that order, equal brms's weights, as numbers and
    as logarithms (as ``check_loo_compare`` compares weights)."""
    other_id = b.case.get("autobayes", {}).get("compare_with")
    models = list(ref["models"])
    if ref.get("method") != "stacking" or not other_id or models != [b.case["id"], other_id]:
        raise OracleMismatch(f"loo_model_weights: the fixture weighs {models} by {ref.get('method')!r}; the case compares "
                             f"{[b.case['id'], other_id]} by stacking")
    other = Bundle(other_id)
    lp = np.array([other.logp(vp) for _, _, vp in other.points()])
    other_lp = ((b.fixture.raw.get("post") or {}).get("loo_compare") or {}).get("other_lp")
    if other_lp is None or rel_err(lp, fx.decode(other_lp)) > LOGP_REL:
        raise OracleMismatch(f"loo_model_weights: {other_id}'s log density at its points is not the fixture's other_lp")
    results = {models[0]: ab.loo(b.model, idata), models[1]: ab.loo(other.model, fixed_parameter_draws(other))}
    theirs = fx.decode(ref["weights"])
    for what, w in (("loo_model_weights", b.post["loo_model_weights"](results)),
                    ("model_weights stacking", ab.model_weights(results, "stacking"))):
        if list(w.index) != models:
            raise OracleMismatch(f"{what}: models {list(w.index)} vs brms {models}")
        compare_post(what, w.to_numpy(), theirs)
        with np.errstate(divide="ignore"):
            compare_post(f"{what} (log)", np.log(w.to_numpy()), np.log(theirs))


# -- multivariate models (C01) ------------------------------------------------------------------------------
def _response_bundle(b: Bundle, response: str):
    """One response of a multivariate model as the univariate gates see a bundle: its view of the fixture, its own model and
    its maps."""
    return SimpleNamespace(fixture=b.cmap.views[response], model=b.model.submodels[response], cmap=b.cmap.maps[response],
                           gmap=b.gmap.maps[response])


def _check_design_multivariate(b: Bundle) -> None:
    """G1 of a multivariate model: brms's ``N``, ``nresp`` and ``nrescor`` (present exactly with a residual correlation), the
    responses side by side (brms's ``Y_y``, ``Y_y2``, ... as the model's ``Y``), then every response's design, response and
    group coding as G1 checks a univariate model's, on its view of the fixture."""
    sd, m = b.fixture.standata, b.model
    K = len(m.responses)
    if int(sd["N"]) != m.design.n:
        raise OracleMismatch(f"brms N = {int(sd['N'])}, AutoBayes n = {m.design.n}")
    if m.rescor and (int(sd.get("nresp", -1)) != K or int(sd.get("nrescor", -1)) != K * (K - 1) // 2):
        raise OracleMismatch(f"brms has nresp = {sd.get('nresp')}, nrescor = {sd.get('nrescor')}; AutoBayes has {K} correlated responses")
    if not m.rescor and "nrescor" in sd:
        raise OracleMismatch("brms has a residual correlation (nrescor) but the AutoBayes model has rescor=False")
    if m.design.rows is not None:                                   # F06c: subset(), each response on its rows (N_<resp>, Y_<resp>)
        return _check_design_subset(b)
    Y_brms = np.column_stack([sd[f"Y_{r}"] for r in m.responses])
    observed = np.isfinite(Y_brms)                                   # C06: brms writes Inf at a response's missing rows
    if not np.array_equal(observed, ~np.isnan(m.design.Y)):
        raise OracleMismatch("brms's responses are infinite (missing) at other rows than the AutoBayes model's missing ones")
    err = abs_err(Y_brms[observed], m.design.Y[observed])
    if not err <= DESIGN_ABS:
        raise OracleMismatch(f"the responses differ by {err}")
    for r, sub in m.submodels.items():                              # C06: an mi(col) term reads the imputing response's column
        for col, block in sub.design.missing.items():
            owner = m.submodels[col].design.missing[col] if col in m.submodels else None
            if owner is None or block.index.tolist() != owner.index.tolist() or not np.array_equal(block.filled, owner.filled):
                raise OracleMismatch(f"response {r!r}: the column {col!r} it reads through mi() is not {col!r}'s own, with its missing rows")
    for r in m.responses:
        try:
            check_design(_response_bundle(b, r))
        except OracleMismatch as e:
            raise OracleMismatch(f"response {r!r}: {e}") from None


def _check_design_subset(b: Bundle) -> None:
    """G1 of a multivariate model with ``subset()`` (F06c): each response's ``N_<resp>`` and ``Y_<resp>`` are its rows of the data
    (the model's ``design.rows``, whose response values ``Y`` holds there and nowhere else), the rows in data order whose indicator is
    1 in the case's data, then each response's design, response and group coding as G1 checks a univariate model's, on its view."""
    sd, m = b.fixture.standata, b.model
    for k, r in enumerate(m.responses):
        rows, column = m.design.rows[r], m.spec.subsets.get(r)
        expected = np.arange(len(b.data)) if column is None else np.flatnonzero(b.data[column].to_numpy() == 1)
        if rows.tolist() != expected.tolist() or int(sd[f"N_{r}"]) != rows.size:
            raise OracleMismatch(f"response {r!r}: brms N_{r} = {int(sd[f'N_{r}'])} rows, the model's rows {rows.tolist()[:12]} "
                                 f"({rows.size}); subset({column}) holds {expected.size}")
        on = np.zeros(m.design.n, dtype=bool)
        on[rows] = True
        if np.isnan(m.design.Y[on, k]).any() or not np.isnan(m.design.Y[~on, k]).all():
            raise OracleMismatch(f"response {r!r}: the model's Y is not its response on its rows and NaN elsewhere")
        err = abs_err(np.asarray(sd[f"Y_{r}"], dtype=float), m.design.Y[on, k])
        if not err <= DESIGN_ABS:
            raise OracleMismatch(f"response {r!r}: Y_{r} differs from the response on its rows by {err}")
        try:
            check_design(_response_bundle(b, r))
        except OracleMismatch as e:
            raise OracleMismatch(f"response {r!r}: {e}") from None


def _check_derived_multivariate(b: Bundle) -> None:
    """G3 of a multivariate model, on its graph: every response's linear predictor (brms's linpred on that response's slice),
    group effects (``r_m_<resp>_c``) and correlation matrices (``Cor_m``), its raw-scale intercept (``b_<resp>_Intercept``),
    and the residual correlation matrix ``rescor`` (brms's ``Rescor``)."""
    m, pmm = b.model, b.model.pymc_model
    wanted = []
    for r, sub in m.submodels.items():
        wanted += [m.variable(r, sub.names["linpred"])] + [m.variable(r, roles["group_effect"]) for roles in sub.group_coefficient_names.values()]
        wanted += [m.variable(r, roles["cor"]) for roles in sub.correlation_names.values()]
    wanted += ["rescor"] if m.rescor else []
    fn = pmm.compile_fn(pmm.replace_rvs_by_values([pmm[v] for v in wanted]), inputs=pmm.value_vars, point_fn=True, on_unused_input="ignore")
    ref = b.fixture.prediction("fitted", "linpred")
    # F06c: brms refuses a subset() model's joint predictions; each response's own, on its rows, are its linear predictor
    own = {r: b.fixture.response_prediction("fitted", r, "linpred") for r in m.responses} if ref is None else {}
    if ref is None and any(v is None for v in own.values()):
        raise OracleMismatch("the fixture has neither the joint linear predictor nor each response's")
    for k, p, vp in b.points():
        out = dict(zip(wanted, fn(vp)))
        nat = adapter.natural_point(m, p, b.cmap, b.gmap)["resp"]
        for i, (r, sub) in enumerate(m.submodels.items()):
            view, gmap = b.cmap.views[r], b.gmap.maps[r]
            pars = view.point(p).pars
            brms = ref[k][:, i] if ref is not None else own[r][k]
            err = rel_err(np.asarray(out[m.variable(r, sub.names["linpred"])], dtype=float).ravel(), brms)
            if err > PRED_REL:
                raise OracleMismatch(f"point {k}: the graph's linear predictor of {r!r} differs from brms by {err}")
            for (g, coef), roles in sub.group_coefficient_names.items():
                mm, cn = gmap.brms_id[g], gmap.coef_cn[(g, coef)]
                eff = np.asarray(out[m.variable(r, roles["group_effect"])], dtype=float).ravel()[gmap.level_of[g]]
                err = rel_err(eff, np.atleast_1d(pars[f"r_{mm}_{cn}"]))
                if err > PRED_REL:
                    raise OracleMismatch(f"point {k}: group effects of {coef!r} of {g!r} for {r!r} differ from brms r_{mm}_{r}_{cn} by {err}")
            for g, roles in sub.correlation_names.items():
                mm, perm = gmap.brms_id[g], adapter.correlation_order(sub, gmap, g)
                err = rel_err(np.asarray(out[m.variable(r, roles["cor"])], dtype=float), np.atleast_2d(pars[f"Cor_{mm}"])[np.ix_(perm, perm)])
                if err > PRED_REL:
                    raise OracleMismatch(f"point {k}: the correlation matrix of {g!r} for {r!r} differs from brms Cor_{mm} by {err}")
            if "b_Intercept" in pars:
                raw = nat[r]["intercept"] - float(sub.design.x_means @ nat[r]["b"])
                err = rel_err(raw, float(pars["b_Intercept"]))
                if err > PRED_REL:
                    raise OracleMismatch(f"point {k}: raw-scale intercept of {r!r} {raw!r} vs brms b_{r}_Intercept {float(pars['b_Intercept'])!r}")
        if m.rescor:
            err = rel_err(np.asarray(out["rescor"], dtype=float), np.atleast_2d(p.pars["Rescor"]))
            if err > PRED_REL:
                raise OracleMismatch(f"point {k}: the residual correlation matrix differs from brms Rescor by {err}")


def _check_default_prior_multivariate(b: Bundle) -> None:
    """G6 of a multivariate model: brms's default-prior table has every response's classes under its ``resp`` (``Intercept``,
    ``b``, ``sd``, the family's auxiliary parameter), a correlated term's ``cor`` and ``rescor`` without one, and
    ``ab.default_priors()`` for the same formula, data, family and ``rescor``, written as brms's table, equals it row by row."""
    m = b.model
    rows = {(row.get("resp", ""), row["class"]) for row in b.fixture.default_prior}
    expected = {("", "rescor")} if m.rescor else set()
    for r, sub in m.submodels.items():
        centred = b.cmap.maps[r].centred_intercept
        expected |= {(r, "Intercept")} if sub.spec.intercept and centred else set()
        expected |= {(r, "b")} if sub.spec.coefficient_labels or (sub.spec.intercept and not centred) else set()
        expected |= {(r, "sd")} if sub.spec.groups else set()
        expected |= {(r, sub.family_spec.scale_parameter)} if sub.family_spec.scale_parameter is not None else set()
        expected |= {("", "cor")} if sub.correlation_names else set()
    if not expected <= rows:
        raise OracleMismatch(f"brms default_prior (resp, class) rows {sorted(rows)} lack {sorted(expected - rows)}")
    if not b.fixture.prior_summary:
        raise OracleMismatch("fixture has no prior_summary table")
    ours, theirs = default_prior_table(b), b.fixture.default_prior

    def key(row):
        if row is None:
            return None
        return tuple(normalise_prior(row.get(k, "")) if k in ("prior", "lb", "ub") else row.get(k, "")
                     for k in ("prior", "class", "coef", "group", "resp", "dpar", "nlpar", "lb", "ub", "tag", "source"))
    diff = [f"  row {i}: AutoBayes {key(a)} vs brms {key(t)}" for i, (a, t) in enumerate(zip_longest(ours, theirs)) if key(a) != key(t)]
    if diff:
        raise OracleMismatch(f"ab.default_priors() written as brms's table differs from brms's default_prior() "
                             f"({len(ours)} rows vs {len(theirs)}):\n" + "\n".join(diff[:12]))
