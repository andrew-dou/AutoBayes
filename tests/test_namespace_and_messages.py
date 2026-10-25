"""F09 (docs/namespace-and-messages.md): the scaffold era's names leave the top-level namespace with a deprecation, the
library's messages and ``describe()`` lose the project's milestone jargon, and the order of a model's free variables no longer
depends on ``PYTHONHASHSEED``."""
from __future__ import annotations

import ast
import importlib
import json
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import autobayes as ab
from autobayes import likelihoods
from autobayes.formula import check_counts, parse_formula
from tests import test_mixture, test_multinomial
from tests.brms_syntax import audit

REPO = Path(__file__).resolve().parent.parent
MOVED = {"compare_records": "autobayes.session", "environment": "autobayes.session", "fit_record": "autobayes.session",
         "format_comparison": "autobayes.session", "format_environment": "autobayes.session", "panel": "autobayes.diagnostics",
         "Categorical": "autobayes.formula", "GroupTerm": "autobayes.formula", "ModelSpec": "autobayes.formula",
         "PopTerm": "autobayes.formula", "parse_formula": "autobayes.formula", "NonlinearSpec": "autobayes.nonlinear"}
# what the library's user-facing text no longer says: the scaffold's slice and subset, milestone ids and decision records
JARGON = re.compile(r"this slice|this subset|verified subset|docs/history/|decision\.md|(?-i:\bM[0-3][a-c]?(-[0-9])?\b)|milestone", re.I)


def run(script: str, **env_updates) -> dict:
    """``script`` in a fresh interpreter at the repository root; its last JSON line."""
    env = {**os.environ, **env_updates}
    proc = subprocess.run([sys.executable, "-c", script], cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    assert proc.returncode == 0 and lines, (proc.returncode, proc.stdout[-2000:], proc.stderr[-3000:])
    return json.loads(lines[-1])


# -- 1. the top-level namespace ------------------------------------------------------------------------------------------

def test_moved_names_left_all_and_kept_their_modules():
    assert not set(MOVED) & set(ab.__all__) and ab._MOVED == MOVED
    # what a user of the brms-like surface needs stays, the errors fit() and the constructors raise among them
    assert {"DeviceError", "FormulaError", "PriorError", "FitResult", "Model", "Priors", "fit", "gaussian", "predict"} <= set(ab.__all__)
    assert all(hasattr(ab, name) for name in ab.__all__)
    for name, module in MOVED.items():
        assert callable(getattr(importlib.import_module(module), name))
    assert ab.session is importlib.import_module("autobayes.session")    # the modules stay reachable, without a warning
    assert ab.diagnostics is importlib.import_module("autobayes.diagnostics")


def test_top_level_access_warns_once_per_name_naming_its_module(monkeypatch):
    for name, module in MOVED.items():
        monkeypatch.delitem(vars(ab), name, raising=False)               # as if no one had reached it in this process
        with pytest.warns(DeprecationWarning) as record:
            value = getattr(ab, name)
        assert len(record) == 1 and record[0].filename == __file__        # attributed to the caller (stacklevel=2)
        assert str(record[0].message) == (f"autobayes.{name} is deprecated and will be removed from the top-level namespace; "
                                          f"import it from its module: from {module} import {name}")
        assert value is getattr(importlib.import_module(module), name)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert getattr(ab, name) is value                             # once per name: the second access is silent


def test_unknown_top_level_name_is_an_attribute_error():
    with pytest.raises(AttributeError, match=r"^module 'autobayes' has no attribute 'fit_records'$"):
        ab.fit_records
    assert not hasattr(ab, "Categoricals")


def test_fresh_interpreter_star_import_and_warnings():
    out = run("""
import json, warnings
with warnings.catch_warnings(record=True) as on_import:
    warnings.simplefilter("always")
    from autobayes import *
star = sorted(k for k in globals() if not k.startswith("_"))
import autobayes as ab
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    from autobayes import fit_record, parse_formula
    ab.environment; ab.environment; ab.panel
mine = [str(x.message) for x in on_import if "top-level namespace" in str(x.message)]
print(json.dumps({"star": star, "on_import": mine, "messages": [str(x.message) for x in w],
                  "categories": [x.category.__name__ for x in w], "same": fit_record is ab.session.fit_record}))
""")
    assert not set(MOVED) & set(out["star"]) and {"gaussian", "fit", "DeviceError", "FormulaError"} <= set(out["star"])
    assert "session" not in out["star"] and out["on_import"] == [] and out["same"]
    assert out["categories"] == ["DeprecationWarning"] * 4
    assert [m.split(" is deprecated")[0] for m in out["messages"]] == ["autobayes.fit_record", "autobayes.parse_formula",
                                                                        "autobayes.environment", "autobayes.panel"]
    assert out["messages"][3].endswith("from autobayes.diagnostics import panel")


def test_user_facing_text_imports_the_moved_names_from_their_modules():
    """The README, the current documentation, the examples and the CI workflows do not reach a moved name at the top level
    (docs/history/ keeps the milestone record as it was)."""
    names = "|".join(MOVED)
    pattern = re.compile(rf"\b(?:ab|autobayes)\.(?:{names})\b|from autobayes import [^\n]*\b(?:{names})\b")
    files = [REPO / "README.md", *(REPO / "docs").glob("*.md"), *(REPO / "docs" / "brms-parity").glob("*.md"),
             *(REPO / "examples").rglob("*.py"), *(REPO / ".github" / "workflows").glob("*.yml")]
    hits = [f"{f.relative_to(REPO)}: {m.group(0)}" for f in files if f.is_file() for m in pattern.finditer(f.read_text())]
    assert hits == []


# -- 2. milestone jargon -------------------------------------------------------------------------------------------------

def _library_strings():
    """Every string literal under autobayes/ outside an expression statement (docstrings are documentation)."""
    for path in sorted((REPO / "autobayes").rglob("*.py")):
        tree = ast.parse(path.read_text())
        documentation = {id(c) for n in ast.walk(tree) if isinstance(n, ast.Expr) for c in ast.walk(n.value)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in documentation:
                yield f"{path.relative_to(REPO)}:{node.lineno}", node.value


def test_library_messages_carry_no_milestone_jargon():
    hits = [(where, m.group(0)) for where, text in _library_strings() for m in JARGON.finditer(text)]
    assert hits == []
    assert JARGON.search("is not supported in this slice") and JARGON.search("M0 does not handle missing data")   # the lint bites


CORPUS_REFUSALS = [e for e in audit.load_json(audit.CORPUS)["entries"] if e["result"]["outcome"] == "refused_named"]


def test_corpus_refusals_name_their_construct_without_jargon():
    """Every refusal of the syntax audit's corpus (tests/brms_syntax/), as AutoBayes raises it today and as corpus.json
    records it."""
    frame, store = audit.python_frame(), audit.load_json(audit.STANDATA)
    for entry in CORPUS_REFUSALS:
        result = audit.run_entry(entry, frame, store)
        assert result["status"] == "error", entry["id"]
        assert not JARGON.search(result["message"]) and not JARGON.search(entry["result"]["message"]), entry["id"]
        assert result["message"] == entry["result"]["message"], f"{entry['id']}: its recorded message is stale"


def _frame(n=24, seed=3):
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n)
    return pd.DataFrame({"y": y, "y2": 0.5 * y + rng.normal(size=n), "ypos": np.exp(y / 4), "x": rng.normal(size=n),
                         "g": [f"g{k}" for k in rng.integers(0, 4, n)], "yb": rng.integers(0, 2, n), "k": rng.integers(0, 6, n),
                         "n": np.full(n, 6)})


PRI = dict(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd=ab.HalfNormal(1))


@pytest.mark.parametrize("build, heading", [
    (lambda d: ab.gaussian("y ~ x + (1 | g)", d, ab.Priors(**PRI, sigma=ab.HalfNormal(1))), "AutoBayes Gaussian model"),
    (lambda d: ab.bernoulli("yb ~ x + (1 | g)", d, ab.Priors(**PRI)), "AutoBayes Bernoulli model"),
    (lambda d: ab.binomial("k | trials(n) ~ x + (1 | g)", d, ab.Priors(**PRI)), "AutoBayes binomial model"),
], ids=["gaussian", "bernoulli", "binomial"])
def test_describe_names_the_family(build, heading):
    text = build(_frame()).describe()
    assert text.splitlines()[0] == heading and not JARGON.search(text)


def test_describe_of_multivariate_and_nonlinear_models_names_the_family():
    d = _frame()
    response = {"intercept": ab.Normal(0, 2), "b": ab.Normal(0, 1), "sigma": ab.HalfNormal(1)}
    mv = ab.gaussian("mvbind(y, y2) ~ x", d, ab.Priors(resp={"y": response, "y2": response}), rescor=False)
    nl = ab.gaussian(["ypos ~ a * exp(b * x)", "a ~ 1", "b ~ 1"], d,
                     ab.Priors(nlpars={"a": {"b": ab.Normal(1, 1)}, "b": {"b": ab.Normal(0, 1)}}, sigma=ab.HalfNormal(1)), nl=True)
    assert mv.describe().splitlines()[0].startswith("AutoBayes multivariate Gaussian model of 2 responses")
    assert nl.describe().splitlines()[0].startswith("AutoBayes Gaussian model with a non-linear formula")
    assert not JARGON.search(mv.describe()) and not JARGON.search(nl.describe())


def _refusal_frame():
    return pd.DataFrame({"y": np.linspace(0, 1, 10), "x": np.linspace(-1, 1, 10), "w": np.linspace(0, 2, 10) ** 2,
                         "v": np.linspace(3, 1, 10), "s": list("ababababab"), "z": list("pqpqpqpqpq"),
                         "xm": [np.nan, 1.0, 2, 0.5, 1, 2, 3, 1, 0.2, 0.4], "sdx": np.full(10, 0.3), "k": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2],
                         "yo": [1, 2, 3, 1, 2, 3, 1, 2, 3, 2]})


def _multinomial(formula=test_multinomial.F, **kw):
    return ab.Model(formula, test_multinomial.frame(), test_multinomial.pri(), family="multinomial", **kw)


def _mixture(formula):
    return ab.mixture(formula, test_mixture.data(), test_mixture.mixture_priors(), families=["gaussian", "gaussian"])


REWORDED = [   # refusals whose wording F09 changed and that no other changed test reaches: each still refuses, in plain words
    (lambda: _mixture("y ~ x + (1 | g)"), "'y ~ x + (1 | g)': a mixture takes population-level terms only; a group-level term"),
    (lambda: _mixture("y ~ x + offset(e)"), "'y ~ x + offset(e)': a mixture takes no offset term (brms adds it"),
    (lambda: parse_formula('y ~ x + (1 | gr(g, by = gby, id = "p"))'), "a correlation ID with gr(by = ) is not supported (divergence D106)"),
    (lambda: parse_formula("y ~ x + (1 |p| g:h)"), "a correlation ID with a combined grouping factor ('g:h') is not supported (docs/"),
    (lambda: parse_formula("y ~ x + z + cs(x):z", family="acat", categorical={"z": "p"}),
     "an interaction with a category-specific term ('cs(x)') is not supported: brms's cs(x):z"),
    (lambda: parse_formula("y ~ x + cs(z)", family="acat", categorical={"z": "p"}), "'z' is declared categorical; cs() takes a numeric column (brms's"),
    (lambda: check_counts("y", pd.Series([2e6, 1.0]), domain="poisson"),
     "y exceeds the Poisson input-count limit 1000000; see docs/poisson-responses.md"),
    (lambda: ab.default_priors("yo ~ x + cs(s)", _refusal_frame(), family="acat"),
     "column 's' of 'cs(s)' has dtype str; cs() takes a numeric column (divergence D89)"),
    (lambda: parse_formula("y | thres(gr = z) ~ x + z", family="cumulative", categorical={"z": "p"}),
     "cannot also be the response, a predictor, a grouping factor or the offset (docs/grouped-thresholds.md, divergence D158)"),
    (lambda: _multinomial("cbind(m1, m2, m3) | trials(size) ~ x + offset(xpos)"),
     "the offset term 'offset(xpos)' is not supported for the multinomial families (docs/multinomial.md, divergence D136)"),
    (lambda: _multinomial(data2={"M": np.eye(12)}), "which a multinomial model does not take (docs/multinomial.md, divergence D136)"),
    (lambda: _multinomial(center_predictors=False), "center_predictors=False with the multinomial family is not supported: brms's"),
    (lambda: ab.default_priors(test_multinomial.F, test_multinomial.frame(), family="multinomial", center_predictors=False),
     "center_predictors=False with the multinomial family is not supported (docs/multinomial.md, divergence D136)"),
    (lambda: ab.gaussian(["y ~ mi(xm) + x", "mvbind(xm, w) ~ x"], _refusal_frame(), ab.Priors(), rescor=False),
     "mvbind() inside a list of formulas is not supported; a list of formulas"),
    (lambda: ab.gaussian(["y ~ mi(xm)", "xm | mi() ~ x", "v ~ me(w, sdx)"], _refusal_frame(), ab.Priors(), rescor=False),
     "'v ~ me(w, sdx)': a measurement-error term in a multivariate model is not supported; me() belongs"),
    (lambda: ab.gaussian(["y ~ mi(xm)", "xm | mi() ~ mo(k)"], _refusal_frame(), ab.Priors(), rescor=False),
     "'xm | mi() ~ mo(k)': a monotonic term in a multivariate model is not supported (docs/multivariate.md, divergence D56)"),
]


@pytest.mark.parametrize("call, message", REWORDED, ids=[m[:40] for _, m in REWORDED])
def test_reworded_refusals(call, message):
    with pytest.raises(ab.FormulaError) as info:
        call()
    assert message in str(info.value) and not JARGON.search(str(info.value))


# -- 3. the order of the free variables ------------------------------------------------------------------------------------

def test_ordered_family_lists_auxiliary_parameters_in_the_likelihoods_order():
    import bambi as bmb

    student = likelihoods.ordered_family("t")                            # Bambi's built-in family by name
    assert list(student.auxiliary_parameters) == ["sigma", "nu"]
    assert isinstance(student, bmb.Family) and type(student).__name__ == "StudentT"
    aux = student.auxiliary_parameters                                    # the set operations Bambi's set had
    assert "nu" in aux and "mu" not in aux and set(aux - {"nu"}) == {"sigma"} and set(aux & {"nu", "kappa"}) == {"nu"}
    base = likelihoods.hurdle_poisson_family()
    ordered = likelihoods.ordered_family(base)
    assert ordered is not base and type(base) is bmb.Family and isinstance(ordered, bmb.Family)    # a copy; the base is unchanged
    assert list(ordered.auxiliary_parameters) == ["hu"] and isinstance(base.auxiliary_parameters, set)
    assert type(likelihoods.ordered_family(likelihoods.hurdle_poisson_family())) is type(ordered)   # one subclass per class


ORDER_PROBE = """
import json
from tests.oracle import adapter, fixture as fx
out = {}
for cid in ("hurdle_cumulative_varint", "student_learned_nu", "wiener_varint"):
    case = fx.load_case(cid)
    m = adapter.build_model(case, fx.read_frame(case, fx.fixture_dir(cid) / "data.csv"))
    out[cid] = {"value_vars": [v.name for v in m.pymc_model.value_vars], "free": [v.name for v in m.pymc_model.free_RVs],
                "describe": m.describe()}
print(json.dumps(out))
"""


def test_free_variables_and_describe_do_not_depend_on_the_hash_seed():
    """Before F09 Bambi created a model's auxiliary parameters in the iteration order of a set of their names: under
    PYTHONHASHSEED 0 the hurdle cumulative model's hu came before its thresholds, under 2 after them; the Student-t model's
    sigma and nu and the Wiener model's three parameters changed places too, and describe() of the hurdle cumulative model
    listed them in either order."""
    first, second = run(ORDER_PROBE, PYTHONHASHSEED="0"), run(ORDER_PROBE, PYTHONHASHSEED="2")
    assert first == second
    assert first["hurdle_cumulative_varint"]["value_vars"][:2] == ["Intercept_ordered__", "hu_logodds__"]
    assert first["student_learned_nu"]["value_vars"][:2] == ["sigma_log__", "nu_interval__"]
    assert first["wiener_varint"]["value_vars"][:3] == ["bs_log__", "ndt_interval__", "bias_logodds__"]
    listed = first["hurdle_cumulative_varint"]["describe"].split("PyMC variables sampled (value space)\n")[1].splitlines()
    assert [line.split()[0] for line in listed[:2]] == ["Intercept_ordered__", "hu_logodds__"]
