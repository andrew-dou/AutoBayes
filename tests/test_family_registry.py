"""The family registry (brms-parity loop, P3): no family-name branch survives outside autobayes/families/, the
registry holds exactly the registered families with their parent, link and mean names, and every constructor
builds the model that ``get_family`` describes."""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import autobayes as ab
from autobayes.families import FAMILIES, FamilySpec, get_family

PACKAGE = Path(ab.__file__).resolve().parent


# -- (a) the lint ---------------------------------------------------------------------------------------------
def _names_family(node) -> bool:
    return (isinstance(node, ast.Name) and node.id == "family") or (isinstance(node, ast.Attribute) and node.attr == "family")


def _holds_string(node) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return any(_holds_string(e) for e in node.elts)
    return False


def family_name_branches(path: Path) -> list[int]:
    """Lines where something named ``family`` is compared with, or tested for membership in, string constants,
    or is the subject of a ``match`` statement."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            if any(_names_family(o) for o in operands) and any(_holds_string(o) for o in operands):
                lines.append(node.lineno)
        elif isinstance(node, ast.Match) and _names_family(node.subject):
            lines.append(node.lineno)
    return sorted(lines)


def test_no_family_name_branches_outside_the_registry():
    modules = sorted(p for p in PACKAGE.rglob("*.py") if PACKAGE / "families" not in p.parents)
    assert {p.name for p in modules} >= {"model.py", "predict.py", "priors.py", "formula.py", "prior_predictive.py",
                                         "identity.py", "session.py", "likelihoods.py"}
    offenders = {str(p.relative_to(PACKAGE)): lines for p in modules if (lines := family_name_branches(p))}
    assert not offenders, f"family-name branches outside autobayes/families/: {offenders}"


def test_the_lint_sees_comparisons_membership_and_match(tmp_path):
    src = "\n".join([
        "def f(model, family, other):",
        "    if model.family == 'gaussian': pass",                 # 2
        "    if family in ('bernoulli', 'binomial'): pass",       # 3
        "    if 'student' != family: pass",                        # 4
        "    if family not in {'poisson'}: pass",                  # 5
        "    if family == other: pass",                            # not a string constant
        "    if model.kind == 'gaussian': pass",                   # not named family
        "    match model.family:",                                 # 8
        "        case 'poisson': pass",
    ])
    path = tmp_path / "m.py"
    path.write_text(src + "\n", encoding="utf-8")
    assert family_name_branches(path) == [2, 3, 4, 5, 8]


# -- (b) the registry -----------------------------------------------------------------------------------------
EXPECTED = {                               # family -> (Bambi parent, link, predict()'s expected-response name)
    "gaussian": ("mu", "identity", "mu"),
    "student": ("mu", "identity", "mu"),
    "poisson": ("mu", "log", "mu"),
    "negbinomial": ("mu", "log", "mu"),
    "gamma": ("mu", "log", "mu"),
    "lognormal": ("mu", "identity", "mean"),
    "beta": ("logit_mu", "logit", "mu"),
    "bernoulli": ("logit_p", "logit", "p"),
    "binomial": ("logit_p", "logit", "p"),
    "zero_inflated_poisson": ("log_mu", "log", "mean"),
    "hurdle_poisson": ("log_mu", "log", "mean"),
    "cumulative": ("mu", "logit", "p"),
    "categorical": ("mu", "logit", "p"),
    "mixture": ("mu", "identity", "mean"),
    "weibull": ("mu", "log", "mu"),                       # D01
    "exponential": ("mu", "log", "mu"),
    "skew_normal": ("mu", "identity", "mu"),
    "shifted_lognormal": ("mu", "identity", "mean"),
    "exgaussian": ("mu", "identity", "mu"),
    "beta_binomial": ("logit_mu", "logit", "mu"),                 # D02
    "geometric": ("mu", "log", "mu"),
    "negbinomial2": ("mu", "log", "mu"),
    "sratio": ("mu", "logit", "p"),                       # D04
    "cratio": ("mu", "logit", "p"),
    "acat": ("mu", "logit", "p"),
    "zero_inflated_negbinomial": ("log_mu", "log", "mean"),   # D03
    "hurdle_negbinomial": ("log_mu", "log", "mean"),
    "zero_inflated_binomial": ("logit_p", "logit", "expected"),
    "zero_inflated_beta": ("logit_mu", "logit", "mean"),
    "hurdle_gamma": ("log_mu", "log", "mean"),
    "hurdle_lognormal": ("mu", "identity", "mean"),
    "zero_one_inflated_beta": ("logit_mu", "logit", "mean"),    # E01
    "zero_inflated_beta_binomial": ("logit_mu", "logit", "expected"),
    "hurdle_cumulative": ("mu", "logit", "p"),
    "xbeta": ("logit_mu", "logit", "mean"),
    "inverse_gaussian": ("inverse_squared_mu", "1/mu^2", "mu"),   # E02: the linear predictor eta = 1 / mu^2 is the parent
    "von_mises": ("tan_half_mu", "tan_half", "mu"),
    "asym_laplace": ("mu", "identity", "mean"),
    "frechet": ("log_mu", "log", "mu"),
    "gen_extreme_value": ("mu", "identity", "mean"),
    "wiener": ("mu", "identity", "mean"),                 # F04: the drift rate mu; the mean response time is "mean"
    "multinomial": ("mu", "logit", "p"),                  # F01a: mu = the non-reference categories' linear predictors
    "dirichlet_multinomial": ("mu", "logit", "p"),
    "dirichlet": ("mu", "logit", "p"),                    # F01b: the K - 1 linear predictors are the parent
    "logistic_normal": ("mu", "identity", "mean"),        # (its mean is refused, as brms refuses posterior_epred)
}


def test_registry_lists_exactly_the_registered_families():
    assert set(FAMILIES) == set(EXPECTED) and len(FAMILIES) == 45
    for name, (parent, link, mean) in EXPECTED.items():
        spec = get_family(name)
        assert isinstance(spec, FamilySpec) and spec is FAMILIES[name]
        assert (spec.name, spec.parent, spec.link, spec.mean_name) == (name, parent, link, mean)
    assert {name for name, spec in FAMILIES.items() if spec.scale_parameter == "sigma"} == {"gaussian", "student", "lognormal", "skew_normal",
                                                                                            "shifted_lognormal", "exgaussian", "negbinomial2",
                                                                                            "hurdle_lognormal", "asym_laplace", "gen_extreme_value"}
    assert {name for name, spec in FAMILIES.items() if spec.scale_parameter == "shape"} == {"negbinomial", "gamma", "weibull",
                                                                                            "zero_inflated_negbinomial", "hurdle_negbinomial",
                                                                                            "hurdle_gamma", "inverse_gaussian"}
    assert {name for name, spec in FAMILIES.items() if spec.scale_parameter == "phi"} == {"beta", "beta_binomial", "zero_inflated_beta",
                                                                                          "zero_one_inflated_beta",   # E01
                                                                                          "zero_inflated_beta_binomial", "xbeta",
                                                                                          "dirichlet_multinomial",                 # F01a
                                                                                          "dirichlet"}                # F01b
    assert {name for name, spec in FAMILIES.items() if spec.continuous} == {"gaussian", "student", "gamma", "lognormal", "beta", "mixture",
                                                                            "weibull", "exponential", "skew_normal", "shifted_lognormal",
                                                                            "exgaussian", "zero_inflated_beta", "hurdle_gamma", "hurdle_lognormal",
                                                                            "zero_one_inflated_beta", "xbeta",          # E01
                                                                            "inverse_gaussian", "von_mises", "asym_laplace", "frechet", "gen_extreme_value",   # E02
                                                                            "wiener",                                    # F04
                                                                            "dirichlet", "logistic_normal"}             # F01b
    assert {spec.bambi_scale_parameter for spec in FAMILIES.values() if spec.scale_parameter == "shape"} == {"alpha"}
    assert {spec.bambi_scale_parameter for spec in FAMILIES.values() if spec.scale_parameter == "phi"} == {"phi"}
    assert all((spec.scale_parameter is None) == (spec.bambi_scale_parameter is None) for spec in FAMILIES.values())
    assert {name for name, spec in FAMILIES.items() if spec.fixed_parameters} == {"student", "frechet"}   # E02: the Frechet nu
    assert {name: set(spec.learnable_parameters) for name, spec in FAMILIES.items() if spec.learnable_parameters} == {
        "student": {"nu"}, "frechet": {"nu"}}
    assert {name: set(spec.fixable) for name, spec in FAMILIES.items() if spec.fixable} == {"asym_laplace": {"quantile"}}   # E02
    assert {name for name, spec in FAMILIES.items() if spec.has_trials} == {"binomial", "beta_binomial", "zero_inflated_binomial",
                                                                            "zero_inflated_beta_binomial",             # E01
                                                                            "multinomial", "dirichlet_multinomial"}    # F01a
    assert {name for name, spec in FAMILIES.items() if "probability" in spec.prediction_kinds} == {"binomial", "beta_binomial"}
    assert {name for name, spec in FAMILIES.items() if "response" not in spec.prediction_kinds} == {"wiener"}   # F04: no draws (D149)
    assert {name for name, spec in FAMILIES.items() if "dec" in spec.addition_terms} == {"wiener"}             # F04: brms's dec()
    assert {name for name, spec in FAMILIES.items() if spec.expected_response is not None} == {"lognormal", "zero_inflated_poisson", "hurdle_poisson",
                                                                                               "cumulative", "mixture", "shifted_lognormal",
                                                                                               "sratio", "cratio", "acat",   # D04
                                                                                               "zero_inflated_negbinomial", "hurdle_negbinomial",   # D03
                                                                                               "zero_inflated_binomial", "zero_inflated_beta",
                                                                                               "hurdle_gamma", "hurdle_lognormal",
                                                                                               "zero_one_inflated_beta",   # E01
                                                                                               "zero_inflated_beta_binomial",
                                                                                               "hurdle_cumulative", "xbeta",
                                                                                               "asym_laplace", "gen_extreme_value",   # E02
                                                                                               "wiener"}                              # F04
    assert {name for name, spec in FAMILIES.items() if spec.mixture} == {"mixture"}
    assert {name for name, spec in FAMILIES.items() if spec.response_categories is not None} == {"cumulative", "categorical", "sratio",
                                                                                                 "cratio", "acat", "hurdle_cumulative"}
    assert {name for name, spec in FAMILIES.items() if spec.thresholds} == {"cumulative", "sratio", "cratio", "acat", "hurdle_cumulative"}
    assert {name for name, spec in FAMILIES.items() if spec.hurdle_category is not None} == {"hurdle_cumulative"}        # E01
    assert {name for name, spec in FAMILIES.items() if spec.thresholds and not spec.ordered_thresholds} == {"sratio", "cratio", "acat"}   # D04
    assert {name for name, spec in FAMILIES.items() if spec.category_specific} == {"sratio", "cratio", "acat"}
    assert {name for name, spec in FAMILIES.items() if spec.per_category} == {"categorical", "multinomial", "dirichlet_multinomial",
                                                                              "dirichlet", "logistic_normal"}   # F01a, F01b
    assert {name for name, spec in FAMILIES.items() if spec.matrix_response} == {"multinomial", "dirichlet_multinomial"}   # F01a
    assert {name for name, spec in FAMILIES.items() if spec.simplex is not None} == {"dirichlet", "logistic_normal"}   # F01b
    assert {name: spec.zero_parameter for name, spec in FAMILIES.items() if spec.zero_parameter} == {"zero_inflated_poisson": "zi", "hurdle_poisson": "hu",
        "zero_inflated_negbinomial": "zi", "hurdle_negbinomial": "hu", "zero_inflated_binomial": "zi", "zero_inflated_beta": "zi",
        "hurdle_gamma": "hu", "hurdle_lognormal": "hu", "zero_inflated_beta_binomial": "zi", "hurdle_cumulative": "hu"}   # E01
    assert {name: tuple(e.name for e in spec.extra_parameters) for name, spec in FAMILIES.items() if spec.extra_parameters} == {
        "skew_normal": ("alpha",), "exgaussian": ("beta",), "shifted_lognormal": ("ndt",),          # D01
        "zero_one_inflated_beta": ("zoi", "coi"), "xbeta": ("kappa",),                            # E01
        "von_mises": ("kappa",), "asym_laplace": ("quantile",), "gen_extreme_value": ("xi",),       # E02
        "wiener": ("bs", "ndt", "bias")}                                                            # F04
    assert {name: e.variable for name, spec in FAMILIES.items() for e in spec.extra_parameters if e.derived} == {
        "gen_extreme_value": "tmp_xi"}                                                               # E02: brms's tmp_xi
    with pytest.raises(ValueError, match=r"family must be one of \['acat', 'asym_laplace', 'bernoulli', 'beta', 'beta_binomial', 'binomial', 'categorical', 'cratio', 'cumulative', 'dirichlet', 'dirichlet_multinomial', 'exgaussian', 'exponential', 'frechet', 'gamma', 'gaussian', 'gen_extreme_value', 'geometric', 'hurdle_cumulative', 'hurdle_gamma', 'hurdle_lognormal', 'hurdle_negbinomial', 'hurdle_poisson', 'inverse_gaussian', 'logistic_normal', 'lognormal', 'mixture', 'multinomial', 'negbinomial', 'negbinomial2', 'poisson', 'shifted_lognormal', 'skew_normal', 'sratio', 'student', 'von_mises', 'weibull', 'wiener', 'xbeta', 'zero_inflated_beta', 'zero_inflated_beta_binomial', 'zero_inflated_binomial', 'zero_inflated_negbinomial', 'zero_inflated_poisson', 'zero_one_inflated_beta'\], got 'tweedie'"):
        get_family("tweedie")
    with pytest.raises(TypeError):
        FAMILIES["tweedie"] = None


# -- (c) the constructors ---------------------------------------------------------------------------------------
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 12
    return pd.DataFrame({"x": rng.normal(size=n), "y": rng.normal(size=n), "yb": [0, 1] * (n // 2),
                         "cnt": rng.poisson(2.0, n), "k": rng.integers(0, 6, n), "n": np.full(n, 5),
                         "ypos": rng.gamma(2.0, 1.0, n), "prop": rng.uniform(0.05, 0.95, n),
                         "ord": [1, 2, 3, 4] * (n // 4), "cat": ["a", "b", "c"] * (n // 3),
                         "zprop": np.r_[0.0, rng.uniform(0.05, 0.95, n - 1)], "zpos": np.r_[0.0, rng.gamma(2.0, 1.0, n - 1)],
                         "zoprop": np.r_[0.0, 1.0, rng.uniform(0.05, 0.95, n - 2)], "ordh": [0, 1, 2, 3] * (n // 4)}   # E01
                        ).assign(ang=np.linspace(-3.0, 3.0, n),   # E02: von Mises angles in [-pi, pi]
                                 rt=np.linspace(0.4, 1.5, n), dec=[1, 0, 0] * (n // 3),   # F04: response times and decisions
                                 c1=[1, 0, 2] * (n // 3), c2=[2, 3, 0] * (n // 3), c3=[2, 2, 3] * (n // 3), n5=np.full(n, 5),   # F01a
                                 s1=np.linspace(0.1, 0.3, n), s2=np.linspace(0.5, 0.2, n),   # F01b: a simplex response
                                 s3=lambda d: 1.0 - d["s1"] - d["s2"])


EXAMPLES = {                               # family -> (formula, extra prior classes, constructor keywords)
    "gaussian": ("y ~ x", {"sigma": ab.HalfNormal(1)}, {}),
    "student": ("y ~ x", {"sigma": ab.HalfNormal(1)}, {"nu": 4}),
    "poisson": ("cnt ~ x", {}, {}),
    "negbinomial": ("cnt ~ x", {"shape": ab.Gamma(2, 0.5)}, {}),
    "gamma": ("ypos ~ x", {"shape": ab.Gamma(2, 0.5)}, {}),
    "lognormal": ("ypos ~ x", {"sigma": ab.HalfNormal(1)}, {}),
    "beta": ("prop ~ x", {"phi": ab.Gamma(2, 0.5)}, {}),
    "bernoulli": ("yb ~ x", {}, {}),
    "binomial": ("k | trials(n) ~ x", {}, {}),
    "zero_inflated_poisson": ("cnt ~ x", {"zi": ab.Beta(2, 2)}, {}),
    "hurdle_poisson": ("cnt ~ x", {"hu": ab.Beta(2, 2)}, {}),
    "cumulative": ("ord ~ x", {}, {}),
    "categorical": ("cat ~ x", {}, {}),
    "mixture": ("y ~ x", {"components": [ab.Priors(intercept=ab.Normal(-1, 1), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1)),
                                         ab.Priors(intercept=ab.Normal(1, 1), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1))],
                          "theta": ab.Dirichlet(2)}, {"families": ["gaussian", "gaussian"]}),
    "weibull": ("ypos ~ x", {"shape": ab.Gamma(2, 0.5)}, {}),                                      # D01
    "exponential": ("ypos ~ x", {}, {}),
    "skew_normal": ("y ~ x", {"sigma": ab.HalfNormal(1), "alpha": ab.Normal(0, 4)}, {}),
    "shifted_lognormal": ("ypos ~ x", {"sigma": ab.HalfNormal(1), "ndt": ab.Uniform(0, "min_Y")}, {}),
    "exgaussian": ("y ~ x", {"sigma": ab.HalfNormal(1), "beta": ab.Gamma(1, 0.1)}, {}),
    "beta_binomial": ("k | trials(n) ~ x", {"phi": ab.Gamma(2, 0.5)}, {}),                       # D02
    "geometric": ("cnt ~ x", {}, {}),
    "negbinomial2": ("cnt ~ x", {"sigma": ab.Gamma(0.4, 0.3)}, {}),
    "sratio": ("ord ~ x", {}, {}),                                                                  # D04
    "cratio": ("ord ~ x", {}, {}),
    "acat": ("ord ~ x", {}, {}),
    "zero_inflated_negbinomial": ("cnt ~ x", {"shape": ab.Gamma(2, 0.5), "zi": ab.Beta(2, 2)}, {}),   # D03
    "hurdle_negbinomial": ("cnt ~ x", {"shape": ab.Gamma(2, 0.5), "hu": ab.Beta(2, 2)}, {}),
    "zero_inflated_binomial": ("k | trials(n) ~ x", {"zi": ab.Beta(2, 2)}, {}),
    "zero_inflated_beta": ("zprop ~ x", {"phi": ab.Gamma(2, 0.5), "zi": ab.Beta(2, 2)}, {}),
    "hurdle_gamma": ("zpos ~ x", {"shape": ab.Gamma(2, 0.5), "hu": ab.Beta(2, 2)}, {}),
    "hurdle_lognormal": ("zpos ~ x", {"sigma": ab.HalfNormal(1), "hu": ab.Beta(2, 2)}, {}),
    "zero_one_inflated_beta": ("zoprop ~ x", {"phi": ab.Gamma(2, 0.5), "zoi": ab.Beta(2, 2), "coi": ab.Beta(2, 2)}, {}),   # E01
    "zero_inflated_beta_binomial": ("k | trials(n) ~ x", {"phi": ab.Gamma(2, 0.5), "zi": ab.Beta(2, 2)}, {}),
    "hurdle_cumulative": ("ordh ~ x", {"hu": ab.Beta(2, 2)}, {}),
    "xbeta": ("zoprop ~ x", {"phi": ab.Gamma(2, 0.5), "kappa": ab.Gamma(2, 2)}, {}),
    "inverse_gaussian": ("ypos ~ x + offset(n)", {"shape": ab.Gamma(2, 0.5)}, {}),                  # E02 (eta = 1 / mu^2 > 0)
    "von_mises": ("ang ~ x", {"kappa": ab.Gamma(2, 0.01)}, {}),
    "asym_laplace": ("y ~ x", {"sigma": ab.HalfNormal(1), "quantile": ab.Beta(2, 2)}, {}),
    "frechet": ("ypos ~ x", {"nu": ab.Gamma(2, 0.1)}, {}),
    "gen_extreme_value": ("y ~ x", {"sigma": ab.HalfNormal(1), "xi": ab.Normal(0, 0.5)}, {}),
    "wiener": ("rt | dec(dec) ~ x", {"bs": ab.Gamma(1, 1), "ndt": ab.Uniform(0, "min_Y"), "bias": ab.Beta(2, 2)}, {}),   # F04
    "multinomial": ("cbind(c1, c2, c3) | trials(n5) ~ x", {}, {}),                                   # F01a
    "dirichlet_multinomial": ("cbind(c1, c2, c3) | trials(n5) ~ x", {"phi": ab.Gamma(0.01, 0.01)}, {}),
    "dirichlet": ("cbind(s1, s2, s3) ~ x", {"phi": ab.Gamma(2, 0.5)}, {}),                          # F01b
    "logistic_normal": ("cbind(s1, s2, s3) ~ x", {"sigma": ab.HalfNormal(1), "lncor": ab.LKJ(2)}, {}),
}


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_constructors_agree_with_the_registry(family):
    formula, extra, kwargs = EXAMPLES[family]
    shared = {} if "components" in extra else {"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 1)}   # a mixture's are its components' (C05)
    priors = ab.Priors(**shared, **extra)
    model = getattr(ab, family)(formula, frame(), priors, **kwargs)
    spec = get_family(family)
    assert model.family == family and model.family_spec is spec
    assert model.link == spec.link
    assert model.names["linpred"] == spec.parent and model.names["mean"] == spec.mean_name
    assert ("sigma" in model.names) == (spec.scale_parameter == "sigma")
    assert ("shape" in model.names) == (spec.scale_parameter == "shape")
    if spec.scale_parameter is not None:                   # the auxiliary parameter is exposed under its brms name, valued by Bambi's
        assert model.names[spec.scale_parameter] == spec.bambi_scale_parameter
        assert f"{spec.bambi_scale_parameter}_log__" in {v.name for v in model.pymc_model.value_vars}
    if spec.zero_parameter is not None:                    # zi / hu keeps brms's name and is sampled on the log-odds scale
        assert model.names[spec.zero_parameter] == spec.zero_parameter
        assert f"{spec.zero_parameter}_logodds__" in {v.name for v in model.pymc_model.value_vars}
    for extra in spec.extra_parameters:                    # D01: alpha, beta, ndt keep brms's names, on their own coordinates
        assert model.names[extra.name] == extra.name and extra.coordinate in {v.name for v in model.pymc_model.value_vars}
    # a fixed parameter is the one given (the Student-t nu); a learnable one left None is learned (E02: the Frechet nu)
    assert all(getattr(model, name) == kwargs.get(name) for name in spec.fixed_parameters)
    assert model.components == tuple(kwargs.get("families", ()))
    same = ab.Model(formula, frame(), priors, family=family, **kwargs)
    assert same.target_identity == model.target_identity and same.describe() == model.describe()
