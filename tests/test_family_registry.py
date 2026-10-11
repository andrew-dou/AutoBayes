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
}


def test_registry_lists_exactly_the_registered_families():
    assert set(FAMILIES) == set(EXPECTED) and len(FAMILIES) == 14
    for name, (parent, link, mean) in EXPECTED.items():
        spec = get_family(name)
        assert isinstance(spec, FamilySpec) and spec is FAMILIES[name]
        assert (spec.name, spec.parent, spec.link, spec.mean_name) == (name, parent, link, mean)
    assert {name for name, spec in FAMILIES.items() if spec.scale_parameter == "sigma"} == {"gaussian", "student", "lognormal"}
    assert {name for name, spec in FAMILIES.items() if spec.scale_parameter == "shape"} == {"negbinomial", "gamma"}
    assert {name for name, spec in FAMILIES.items() if spec.scale_parameter == "phi"} == {"beta"}
    assert {name for name, spec in FAMILIES.items() if spec.continuous} == {"gaussian", "student", "gamma", "lognormal", "beta", "mixture"}
    assert {spec.bambi_scale_parameter for spec in FAMILIES.values() if spec.scale_parameter == "shape"} == {"alpha"}
    assert {spec.bambi_scale_parameter for spec in FAMILIES.values() if spec.scale_parameter == "phi"} == {"phi"}
    assert all((spec.scale_parameter is None) == (spec.bambi_scale_parameter is None) for spec in FAMILIES.values())
    assert {name for name, spec in FAMILIES.items() if spec.fixed_parameters} == {"student"}
    assert {name for name, spec in FAMILIES.items() if spec.has_trials} == {"binomial"}
    assert {name for name, spec in FAMILIES.items() if "probability" in spec.prediction_kinds} == {"binomial"}
    assert {name for name, spec in FAMILIES.items() if spec.expected_response is not None} == {"lognormal", "zero_inflated_poisson", "hurdle_poisson",
                                                                                               "cumulative", "mixture"}
    assert {name for name, spec in FAMILIES.items() if spec.mixture} == {"mixture"}
    assert {name for name, spec in FAMILIES.items() if spec.response_categories is not None} == {"cumulative", "categorical"}
    assert {name for name, spec in FAMILIES.items() if spec.thresholds} == {"cumulative"}
    assert {name for name, spec in FAMILIES.items() if spec.per_category} == {"categorical"}
    assert {name: spec.zero_parameter for name, spec in FAMILIES.items() if spec.zero_parameter} == {"zero_inflated_poisson": "zi", "hurdle_poisson": "hu"}
    with pytest.raises(ValueError, match=r"family must be one of \['bernoulli', 'beta', 'binomial', 'categorical', 'cumulative', 'gamma', 'gaussian', 'hurdle_poisson', 'lognormal', 'mixture', 'negbinomial', 'poisson', 'student', 'zero_inflated_poisson'\], got 'weibull'"):
        get_family("weibull")
    with pytest.raises(TypeError):
        FAMILIES["weibull"] = None


# -- (c) the constructors ---------------------------------------------------------------------------------------
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 12
    return pd.DataFrame({"x": rng.normal(size=n), "y": rng.normal(size=n), "yb": [0, 1] * (n // 2),
                         "cnt": rng.poisson(2.0, n), "k": rng.integers(0, 6, n), "n": np.full(n, 5),
                         "ypos": rng.gamma(2.0, 1.0, n), "prop": rng.uniform(0.05, 0.95, n),
                         "ord": [1, 2, 3, 4] * (n // 4), "cat": ["a", "b", "c"] * (n // 3)})


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
    assert all(getattr(model, name) == kwargs[name] for name in spec.fixed_parameters)
    assert model.components == tuple(kwargs.get("families", ()))
    same = ab.Model(formula, frame(), priors, family=family, **kwargs)
    assert same.target_identity == model.target_identity and same.describe() == model.describe()
