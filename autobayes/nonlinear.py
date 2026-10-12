"""Non-linear formulas, brms's ``bf(y ~ a * exp(b * x), a ~ 1 + (1 | g), b ~ 1, nl = TRUE)`` (brms-parity C02,
docs/nonlinear.md).

A non-linear model is given as a list of formulas with ``nl=True``: the non-linear formula first, ``<response> ~
<expression>``, then one linear formula per non-linear parameter, ``["ypos ~ a * exp(b * x)", "a ~ 1 + (1 | g)", "b ~ 1"]``::

    mu_i    = a_i * exp(b_i * x_i)                    (the expression, row by row; brms's mu)
    a_i     = X_a,i b_a + u_a,g[g_i]                  (a's own linear predictor, brms's nlp_a; never centred)
    b_i     = X_b,i b_b
    y_i     ~ Normal(mu_i, sigma)

The expression is parsed here, by a small recursive-descent parser of its own (``parse_expression``), into a tree of
``Number``, ``Name``, ``Negate``, ``Binary`` and ``Call`` nodes; nothing a user writes is ever evaluated as code. The
grammar is R's arithmetic on this subset: ``+ - * / ^`` (R's ``**`` is ``^``), parentheses, unary minus, numeric
constants and the functions ``exp``, ``log``, ``sqrt``, ``inv_logit`` (``plogis``, R's name for it, is read as
``inv_logit``) and ``pow(x, y)``, with R's precedence (``^`` binds tightest and to the right, ``-a^2`` is ``-(a^2)``
and ``a^-b`` is ``a^(-b)``; unary minus binds tighter than ``*`` and ``/``). Every name is a non-linear parameter
(it has a formula) or a numeric data column, a covariate; the covariates are brms's ``C_1, C_2, ...`` in order of first
appearance. The tree is turned into PyTensor (the graph) and NumPy (``predict()``) by walking it with a fixed table of
functions.

Each non-linear parameter's formula is a linear formula of the term model (``formula.parse_formula``; docs/term-model.md)
whose left-hand side names the parameter: population-level terms of every kind but ``mo()`` (numeric, declared
categorical, interactions, transforms; ``0 +`` removes its intercept) and the group-level terms the term model takes
(``(1 | g)``, ``(1 + x || g)``, ``(1 + x | g)``, ``g:h``, ``g/h``). brms never centres such a formula: its design
``X_<nlpar>`` keeps the intercept column, and the intercept is the coefficient ``Intercept`` of brms's class ``b``. Each
group-level term is a group of its own (brms gives the same factor in two parameters' formulas two group ids).

The graph is built here, not by Bambi (which has no non-linear formulas): per parameter one PyMC variable per term
(``a_Intercept``, ``a_x``, ``a_z`` over ``a_z_dim``), Bambi's non-centred group effects under the parameter's prefix
(``a_1|g = a_1|g_sigma * a_1|g_offset``, a slope's ``a_x|g``; a correlated term's Cholesky factor ``a_L|g`` on Stan's
``cholesky_corr`` coordinates and ``a_cor|g``), the parameter's values per row as a deterministic named after it (``a``),
the expression as the deterministic ``mu`` and the family's likelihood (``FamilySpec.nonlinear_likelihood``, the
Gaussian's ``Normal(mu, sigma)``). Priors are per parameter, ``Priors(nlpars={"a": {"b": ..., "sd": ...}})`` (brms's
``nlpar =``), and ``Priors(cor={factor: LKJ(eta)})`` for a correlated term (brms's class ``cor`` has no ``nlpar``).

Refused, naming the construct (docs/brms-parity/divergences.md, D60-D63): brms's ``nlf()`` and ``lf()`` and a non-linear
parameter inside another's formula (D60); functions and operators outside the grammar, an integer-literal division brms's
Stan code would truncate is real division here, the response as a covariate, ``gp()`` and ``mo()`` in the expression or in
a parameter's formula, and ``offset()`` or addition terms with a non-linear formula (D61); distributional formulas with
``nl``, non-linear models in multivariate formulas (``mvbind()`` or a list of response formulas) and in mixtures
(``ab.mixture``), and families without a verified non-linear likelihood (D62); ``loop = FALSE`` (D63).
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
import pandas as pd

from autobayes.formula import (_FORMULA_RE, _IDENT, FormulaError, ModelSpec, _is_numeric, bambi_group_name, categorical_labels,
                               factor_column, group_labels, parse_formula, pop_term_design, reject_complex, sd_tag,
                               validate_data)
from autobayes.model import Design, GroupBlock, Model
from autobayes.priors import AUXILIARY, Prior, PriorError, Priors, describe_prior

# -- the expression grammar ----------------------------------------------------------------------------------------------
FUNCTIONS = MappingProxyType({"exp": 1, "log": 1, "sqrt": 1, "inv_logit": 1, "plogis": 1, "pow": 2})   # name -> arguments
_ALIASES = MappingProxyType({"plogis": "inv_logit"})          # R's name of the logistic CDF, which Stan does not have
_SPECIAL_TERMS = MappingProxyType({"gp": "Gaussian process (C03)", "mo": "monotonic term (B03)"})   # brms's terms, not functions
GRAMMAR = ("the non-linear expression takes non-linear parameters, numeric data columns and numeric constants, combined with "
           "+ - * / ^ (R's ** is ^), parentheses, unary minus and the functions exp(), log(), sqrt(), inv_logit() (or plogis()) "
           "and pow(x, y) (docs/nonlinear.md, divergence D61)")
NL_FORM = ("a list of formulas with nl=True: the non-linear formula first, '<response> ~ <expression>', then one formula "
           "per non-linear parameter, ['y ~ a * exp(b * x)', 'a ~ 1 + (1 | g)', 'b ~ 1'] (brms's bf(y ~ a * exp(b * x), "
           "a ~ 1 + (1 | g), b ~ 1, nl = TRUE); docs/nonlinear.md")


@dataclass(frozen=True)
class Number:
    """A numeric constant, a double (R reads every literal of this grammar as one)."""
    value: float


@dataclass(frozen=True)
class Name:
    """A non-linear parameter or a covariate (a data column), resolved by ``parse_nonlinear``."""
    name: str


@dataclass(frozen=True)
class Negate:
    """Unary minus."""
    operand: object


@dataclass(frozen=True)
class Binary:
    """``left op right`` with ``op`` one of ``+ - * / ^``."""
    op: str
    left: object
    right: object


@dataclass(frozen=True)
class Call:
    """A function of the grammar (a key of ``FUNCTIONS``; ``plogis`` is stored as ``inv_logit``)."""
    function: str
    args: tuple


_TOKEN_RE = re.compile(rf"(?P<number>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)|(?P<name>{_IDENT})|(?P<op>\*\*|[-+*/^(),])")


def tokenize(text: str) -> list[tuple[str, str, int]]:
    """``(kind, text, position)`` per token: ``number``, ``name`` or ``op`` (``**`` is returned as ``^``). Anything else
    is refused with the character and its position."""
    tokens, pos = [], 0
    while True:
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos == len(text):
            return tokens
        m = _TOKEN_RE.match(text, pos)
        if not m:
            raise FormulaError(f"{text!r}: {text[pos]!r} at position {pos} is outside the grammar; {GRAMMAR}")
        kind, value = m.lastgroup, m.group(m.lastgroup)
        if kind == "number" and m.end() < len(text) and (text[m.end()].isalpha() or text[m.end()] in "_."):
            raise FormulaError(f"{text!r}: the literal at position {pos} ({text[pos:m.end() + 1]!r}...) is not a decimal number; R's "
                               "integer, hexadecimal and complex literals (2L, 0x10, 1i) are outside the grammar: write the "
                               "number as a decimal")
        if kind == "number" and not np.isfinite(float(value)):
            raise FormulaError(f"{text!r}: the constant {value!r} is not a finite double")
        tokens.append((kind, "^" if value == "**" else value, pos))
        pos = m.end()


class _Parser:
    """Recursive descent over R's precedence for the grammar's operators::

        expr    := term (('+' | '-') term)*
        term    := unary (('*' | '/') unary)*
        unary   := '-' unary | power
        power   := primary ('^' unary)?
        primary := number | name | name '(' expr (',' expr)* ')' | '(' expr ')'
    """

    def __init__(self, text: str):
        self.text, self.tokens, self.i = text, tokenize(text), 0

    def _peek(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def _at(self, *ops) -> bool:
        tok = self._peek()
        return tok is not None and tok[0] == "op" and tok[1] in ops

    def _error(self, what: str):
        tok = self._peek()
        where = f"at position {tok[2]} ({tok[1]!r})" if tok is not None else "at the end"
        return FormulaError(f"{self.text!r}: {what} {where}; {GRAMMAR}")

    def _expect(self, op: str) -> None:
        if not self._at(op):
            raise self._error(f"expected {op!r}")
        self.i += 1

    def parse(self):
        if not self.tokens:
            raise FormulaError(f"{self.text!r}: the non-linear expression is empty; {GRAMMAR}")
        node = self._expr()
        if self._peek() is not None:
            raise self._error("unexpected token")
        return node

    def _expr(self):
        node = self._term()
        while self._at("+", "-"):
            op = self.tokens[self.i][1]
            self.i += 1
            node = Binary(op, node, self._term())
        return node

    def _term(self):
        node = self._unary()
        while self._at("*", "/"):
            op = self.tokens[self.i][1]
            self.i += 1
            node = Binary(op, node, self._unary())
        return node

    def _unary(self):
        if self._at("-"):
            self.i += 1
            return Negate(self._unary())
        if self._at("+"):
            raise self._error("unary plus is outside the grammar (write the term without it)")
        return self._power()

    def _power(self):
        base = self._primary()
        if self._at("^"):
            self.i += 1
            return Binary("^", base, self._unary())
        return base

    def _primary(self):
        tok = self._peek()
        if tok is None:
            raise self._error("the expression ends early")
        kind, value, pos = tok
        self.i += 1
        if kind == "number":
            return Number(float(value))
        if kind == "name":
            if self._at("("):
                return self._call(value, pos)
            return Name(value)
        if value == "(":
            node = self._expr()
            self._expect(")")
            return node
        self.i -= 1
        raise self._error("expected a number, a name, a function or '('")

    def _call(self, name: str, pos: int):
        if name in ("nlf", "lf"):
            raise FormulaError(f"{self.text!r}: brms's {name}() is not supported here (divergence D60): give each non-linear "
                               "parameter its own linear formula in the list, and write the non-linear expression once")
        if name in _SPECIAL_TERMS:
            raise FormulaError(f"{self.text!r}: {name}() at position {pos} is brms's {_SPECIAL_TERMS[name]}, a term of a linear "
                               "formula, not a function of the non-linear expression (brms writes it into its Stan program as a "
                               "call of a function Stan does not have); a non-linear model takes no such term in this slice, in a "
                               f"parameter's formula either (divergence D61); {GRAMMAR}")
        if name not in FUNCTIONS:
            raise FormulaError(f"{self.text!r}: {name}() at position {pos} is outside the grammar; {GRAMMAR}")
        self._expect("(")
        args = [self._expr()]
        while self._at(","):
            self.i += 1
            args.append(self._expr())
        self._expect(")")
        if len(args) != FUNCTIONS[name]:
            raise FormulaError(f"{self.text!r}: {name}() takes {FUNCTIONS[name]} argument{'s' if FUNCTIONS[name] > 1 else ''}, "
                               f"got {len(args)}; {GRAMMAR}")
        return Call(_ALIASES.get(name, name), tuple(args))


def parse_expression(text: str):
    """The tree of a non-linear expression (no names resolved yet). Never evaluates ``text``: it is tokenized and parsed
    by ``_Parser``, and anything outside the grammar is refused."""
    if not isinstance(text, str):
        raise FormulaError(f"a non-linear expression is a string, got {type(text).__name__}")
    return _Parser(text).parse()


_PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2}


def number_text(v: float) -> str:
    """A constant as the canonical expression writes it: the shortest text that reads back as the same double (``2``,
    ``0.25``, ``1e-05``)."""
    text = repr(float(v))
    return text[:-2] if text.endswith(".0") else text


def render(node) -> str:
    """The expression as text with the parentheses its tree needs (``parse_expression(render(t)) == t``)."""
    if isinstance(node, Number):
        return number_text(node.value)
    if isinstance(node, Name):
        return node.name
    if isinstance(node, Call):
        return f"{node.function}({', '.join(render(a) for a in node.args)})"
    if isinstance(node, Negate):
        inner = render(node.operand)
        simple = isinstance(node.operand, (Number, Name, Call)) or (isinstance(node.operand, Binary) and node.operand.op == "^")
        return f"-{inner}" if simple else f"-({inner})"
    if node.op == "^":
        left = render(node.left) if isinstance(node.left, (Number, Name, Call)) else f"({render(node.left)})"
        right = render(node.right) if isinstance(node.right, (Number, Name, Call)) else f"({render(node.right)})"
        return f"{left}^{right}"
    p = _PRECEDENCE[node.op]
    left = render(node.left)
    if isinstance(node.left, Binary) and node.left.op != "^" and _PRECEDENCE[node.left.op] < p:
        left = f"({left})"
    right = render(node.right)
    if (isinstance(node.right, Negate)
            or (isinstance(node.right, Binary) and node.right.op != "^" and _PRECEDENCE[node.right.op] <= p)):
        right = f"({right})"
    return f"{left} {node.op} {right}"


def names_in_order(node) -> tuple[str, ...]:
    """The names of the expression in order of first appearance, left to right (R's ``all.vars``, brms's order of the
    covariates ``C_1, C_2, ...``)."""
    out: list[str] = []

    def walk(n):
        if isinstance(n, Name):
            if n.name not in out:
                out.append(n.name)
        elif isinstance(n, Negate):
            walk(n.operand)
        elif isinstance(n, Binary):
            walk(n.left)
            walk(n.right)
        elif isinstance(n, Call):
            for a in n.args:
                walk(a)
    walk(node)
    return tuple(out)


def functions_in(node) -> tuple[str, ...]:
    """The functions the expression calls, each once, in order of first appearance."""
    out: list[str] = []

    def walk(n):
        if isinstance(n, Negate):
            walk(n.operand)
        elif isinstance(n, Binary):
            walk(n.left)
            walk(n.right)
        elif isinstance(n, Call):
            if n.function not in out:
                out.append(n.function)
            for a in n.args:
                walk(a)
    walk(node)
    return tuple(out)


def to_pytensor(node, env: Mapping):
    """The expression in PyTensor: ``env`` maps every name to a tensor (a parameter's values per row, a covariate's column);
    constants are float64 (PyTensor would make a Python 2.0 a float32 constant)."""
    import pytensor.tensor as pt

    if isinstance(node, Number):
        return pt.as_tensor_variable(np.float64(node.value))
    if isinstance(node, Name):
        return env[node.name]
    if isinstance(node, Negate):
        return -to_pytensor(node.operand, env)
    if isinstance(node, Call):
        args = [to_pytensor(a, env) for a in node.args]
        return {"exp": pt.exp, "log": pt.log, "sqrt": pt.sqrt, "inv_logit": pt.sigmoid, "pow": pt.pow}[node.function](*args)
    left, right = to_pytensor(node.left, env), to_pytensor(node.right, env)
    return {"+": lambda: left + right, "-": lambda: left - right, "*": lambda: left * right, "/": lambda: left / right,
            "^": lambda: pt.pow(left, right)}[node.op]()


def to_numpy(node, env: Mapping) -> np.ndarray:
    """The expression in NumPy on arrays that broadcast (``predict()``: draws by row); ``env`` as for ``to_pytensor``.
    Floating-point warnings are silenced: the caller refuses a non-finite result."""
    from scipy.special import expit

    with np.errstate(all="ignore"):
        if isinstance(node, Number):
            return np.float64(node.value)
        if isinstance(node, Name):
            return env[node.name]
        if isinstance(node, Negate):
            return -to_numpy(node.operand, env)
        if isinstance(node, Call):
            args = [to_numpy(a, env) for a in node.args]
            return {"exp": np.exp, "log": np.log, "sqrt": np.sqrt, "inv_logit": expit, "pow": np.power}[node.function](*args)
        left, right = to_numpy(node.left, env), to_numpy(node.right, env)
        return {"+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide, "^": np.power}[node.op](left, right)


# -- the formulas -----------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class NonlinearSpec:
    """A parsed non-linear model (C02): ``formula`` the non-linear formula as written, ``response`` its left-hand side,
    ``expression`` the parsed right-hand side, ``nlpars`` one ``ModelSpec`` per non-linear parameter in the order given
    (its ``response`` is the parameter's name; levels and categorical codings known once the data are seen), ``covariates``
    the data columns of the expression in order of first appearance (brms's ``C_1, C_2, ...``) and ``formulas`` every
    formula as written.

    It stands where a ``ModelSpec`` stands on a model: the views of the response formula's linear terms that the rest of the
    library reads (``intercept``, ``terms``, ``columns``, ``group_terms``, ``dpars``, ...) are empty here, because a
    non-linear formula has none; the parameters' terms are in ``nlpars``."""

    formula: str
    response: str
    expression: object
    nlpars: tuple
    covariates: tuple
    formulas: tuple

    is_nonlinear = True                   # what the shared code dispatches on (getattr(spec, "is_nonlinear", False))
    # the response formula's linear terms, of which a non-linear formula has none
    intercept = False
    pop_terms = ()
    group_terms = ()
    terms = ()
    columns = ()
    column_terms = ()
    coefficient_labels = ()
    predictors = ()
    interactions = ()
    interaction_names = ()
    categorical_names = ()
    categoricals = ()
    transform_terms = ()
    bambi_terms = ()
    mo_terms = ()
    monotonics = ()
    gp_terms = ()                         # Gaussian processes (C03): none, as a non-linear model takes none (D61)
    autocor = None                        # an autocorrelation term (C04): none, as a non-linear model takes none (D70)
    groups = ()
    group_coefficients = ()
    correlated_terms = ()
    has_group = False
    has_slopes = False
    levels_by_group = MappingProxyType({})
    dpars = ()
    dpar_names = ()
    offset = None
    additions = ()
    trials = None

    def addition(self, kind: str):
        """A non-linear formula takes no response addition term (D61)."""
        return None

    @property
    def rhs(self) -> str:
        """The expression as the canonical formula writes it (``render``)."""
        return render(self.expression)

    @property
    def canonical_formula(self) -> str:
        return f"{self.response} ~ {self.rhs}"

    @property
    def nlpar_names(self) -> tuple[str, ...]:
        return tuple(d.response for d in self.nlpars)

    def nlpar(self, name: str) -> ModelSpec:
        """The formula of non-linear parameter ``name`` (``KeyError`` when it has none)."""
        for d in self.nlpars:
            if d.response == name:
                return d
        raise KeyError(name)

    @property
    def nlpar_group_terms(self) -> tuple:
        """``(parameter, GroupTerm)`` for every group-level term of every parameter's formula, in formula order."""
        return tuple((d.response, t) for d in self.nlpars for t in d.group_terms)

    @property
    def factors(self) -> tuple[str, ...]:
        """The grouping factors of every parameter's formula, each once, in formula order."""
        return tuple(dict.fromkeys(t.factor for _, t in self.nlpar_group_terms))

    @property
    def correlated_factors(self) -> tuple[str, ...]:
        """The factors of the correlated terms ``(1 + x | g)`` of every parameter's formula, each once (brms's class ``cor``
        is keyed by group alone)."""
        return tuple(dict.fromkeys(t.factor for _, t in self.nlpar_group_terms if t.has_correlation))

    @property
    def all_categoricals(self) -> tuple:
        """The declared categorical columns of every parameter's formula, each once (the first formula's coding)."""
        out = []
        for d in self.nlpars:
            out += [c for c in d.categoricals if all(c.name != o.name for o in out)]
        return tuple(out)

    @property
    def model_columns(self) -> tuple[str, ...]:
        """The data columns the expression and the parameters' population-level terms read, each once: the covariates, then
        each formula's ``pop_columns``."""
        return tuple(dict.fromkeys([*self.covariates, *(c for d in self.nlpars for c in d.pop_columns)]))

    @property
    def data_columns(self) -> tuple[str, ...]:
        """Every column the model reads: the response, ``model_columns`` and the columns of the grouping factors."""
        operands = [c for _, t in self.nlpar_group_terms for c in t.operands]
        return tuple(dict.fromkeys([self.response, *self.model_columns, *operands]))


_MULTIVARIATE = ("a non-linear model in a multivariate formula is not supported (divergence D62; brms gives each response its "
                 "own non-linear parameters, X_<response>_<nlpar>): give one response's non-linear model, " + NL_FORM)


def _check_formula_list(formulas) -> None:
    from autobayes.formula import MVBIND_RE

    first = formulas[0] if isinstance(formulas, (list, tuple)) and formulas else formulas
    if isinstance(first, str) and MVBIND_RE.match(first):              # brms's mvbind(y, y2) ~ ... (C01)
        raise FormulaError(f"{first!r}: mvbind() with nl=True: {_MULTIVARIATE}")
    if isinstance(formulas, str) or not isinstance(formulas, (list, tuple)):
        raise FormulaError(f"nl=True takes {NL_FORM}; got {formulas!r}")
    if any(isinstance(f, (list, tuple)) for f in formulas):            # a formula per response, brms's bf(...) + bf(...)
        raise FormulaError(f"a list of response formulas with nl=True: {_MULTIVARIATE}")
    if not all(isinstance(f, str) for f in formulas):
        raise FormulaError(f"nl=True takes {NL_FORM}; got {formulas!r}")
    for text in formulas:
        m = re.match(r"^\s*(nlf|lf)\s*\(", text)
        if m:
            raise FormulaError(f"{text!r}: brms's {m.group(1)}() is not supported here (divergence D60): give each non-linear "
                               "parameter its own linear formula '<parameter> ~ <terms>' in the list")
    if len(formulas) < 2:
        raise FormulaError(f"nl=True takes {NL_FORM}; {list(formulas)!r} has no formula of a non-linear parameter")


def parse_nonlinear(formulas, data: pd.DataFrame | None = None, family: str = "gaussian",
                    categorical: Mapping[str, object] | None = None) -> NonlinearSpec:
    """Parse and validate a non-linear model (``NL_FORM``), optionally against a data frame.

    The first formula's right-hand side is the expression (``parse_expression``); every other formula is
    ``<parameter> ~ <terms>``, parsed by ``formula.parse_formula`` as a formula of its own and checked against the data as
    a distributional formula is (``validate_data(..., dpar=True)``). ``categorical`` is shared: a declared column is
    categorical in every parameter's formula that names it as a main effect, and must be one. Refused, naming the
    construct: a parameter the expression does not use, or a name of the expression that is neither a parameter nor a
    column; a parameter in another parameter's formula; the response in the expression or in a parameter's formula;
    a formula of a distributional parameter (``sigma ~ ...``), or of ``mu``; addition terms, ``offset()``, ``mo()`` and
    ``gp()``; ``mvbind()`` or a list of response formulas (a multivariate model); a covariate that is not numeric, finite
    and observed on every row."""
    from autobayes.families import get_family

    fam = get_family(family)
    _check_formula_list(formulas)
    main, *rest = formulas
    if "~" in main and "|" in main.split("~", 1)[0]:
        raise FormulaError(f"{main!r}: response addition terms (trials(), weights(), cens(), trunc()) with a non-linear formula are "
                           "not supported in this slice (divergence D61); the left-hand side is the response column alone")
    head = _FORMULA_RE.match(main)
    if not head:
        raise FormulaError(f"{main!r}: the non-linear formula is '<response> ~ <expression>'; {NL_FORM}")
    response, rhs = head.group(1), head.group(2)
    expression = parse_expression(rhs)
    dpars = {p for p in (fam.scale_parameter, fam.zero_parameter, *fam.fixed_parameters, *fam.learnable_parameters) if p}
    parsed: list[tuple[str, str, str]] = []                     # (parameter, right-hand side, formula as written)
    for text in rest:
        m = _FORMULA_RE.match(text)
        if not m or "|" in text.split("~", 1)[0]:
            raise FormulaError(f"{text!r}: a non-linear parameter's formula is '<parameter> ~ <terms>' with the parameter's name on "
                               f"the left; {NL_FORM}")
        name, prhs = m.group(1), m.group(2)
        if name == response:
            raise FormulaError(f"{text!r}: {name!r} is the response; the non-linear formula comes first ({main!r})")
        if name in dpars or name in ("mu", fam.parent):
            raise FormulaError(f"{text!r}: a formula of the distributional parameter {name!r} with a non-linear formula is not "
                               "supported in this slice (divergence D62): a non-linear model's parameters are the names of its "
                               f"expression, and {name!r} keeps its single prior")
        if name in (p for p, _, _ in parsed):
            raise FormulaError(f"{text!r}: {name!r} has two formulas; give each non-linear parameter one")
        parsed.append((name, prhs, text.strip()))
    used = names_in_order(expression)
    nl_names = [p for p, _, _ in parsed]
    for name, _, text in parsed:
        if name not in used:
            raise FormulaError(f"{text!r}: {name!r} does not appear in the non-linear formula {main.strip()!r}; every formula after "
                               "the first is a non-linear parameter's (brms: 'The parameter ... is not a valid distributional or "
                               "non-linear parameter')")
    covariates = tuple(n for n in used if n not in nl_names)
    if response in covariates:
        raise FormulaError(f"{main!r}: the response {response!r} cannot be a covariate of its own non-linear predictor (divergence "
                           "D61)")

    # Route the declared categorical columns: each parameter's formula receives the declarations of its own main effects.
    spec0 = [parse_formula(f"{name} ~ {prhs}", None, family, None) for name, prhs, _ in parsed]
    mains = [{t.name for t in s.pop_terms if t.kind == "numeric"} for s in spec0]
    if isinstance(categorical, Mapping):
        for col in categorical:
            if col in covariates:
                raise FormulaError(f"{col!r} is declared categorical, but it is a covariate of the non-linear formula, which reads "
                                   "it as a number; one column cannot be both")
        stray = [col for col in categorical if not any(col in m for m in mains)]
        if stray:
            raise FormulaError(f"{stray} declared categorical, but no non-linear parameter's formula names "
                               f"{'it' if len(stray) == 1 else 'them'} as a main effect")
        routed = [{k: v for k, v in categorical.items() if k in m} or None for m in mains]
    elif categorical is not None:
        raise FormulaError(f"categorical must be a mapping {{column: reference level}}, got {type(categorical).__name__}")
    else:
        routed = [None] * len(parsed)
    nlpars = []
    for (name, prhs, text), cat in zip(parsed, routed):
        d = parse_formula(f"{name} ~ {prhs}", None, family, cat)
        if d.offset is not None:
            raise FormulaError(f"{text!r}: {d.offset.term!r} in a non-linear parameter's formula is not supported in this slice "
                               "(divergence D61)")
        if d.mo_terms:
            raise FormulaError(f"{text!r}: the monotonic term {d.mo_terms[0].name!r} in a non-linear parameter's formula is not "
                               "supported in this slice (divergence D61)")
        if d.gp_terms:                                                 # C03: brms adds gp_pred_<nlpar>_1 to the parameter
            raise FormulaError(f"{text!r}: the Gaussian process {d.gp_terms[0].term!r} in a non-linear parameter's formula is not "
                               "supported in this slice (divergence D61)")
        reads = [*d.pop_columns, *(c for t in d.group_terms for c in t.operands)]
        nested = [c for c in reads if c in nl_names]
        if nested:
            raise FormulaError(f"{text!r}: {nested[0]!r} is a non-linear parameter; a non-linear parameter inside another's formula "
                               "is not supported (divergence D60): write the relation in the non-linear formula")
        if response in reads:
            raise FormulaError(f"{text!r}: the response {response!r} cannot be a predictor of a non-linear parameter")
        if data is not None:
            d = validate_data(d, data, family, dpar=True)
        nlpars.append(d)
    spec = NonlinearSpec(formula=main.strip(), response=response, expression=expression, nlpars=tuple(nlpars),
                         covariates=covariates, formulas=tuple(f.strip() for f in formulas))
    if data is not None:
        _validate_nonlinear_data(spec, data, fam)
    return spec


def _validate_nonlinear_data(spec: NonlinearSpec, data: pd.DataFrame, fam) -> None:
    """The response and the covariates on ``data`` (each parameter's formula was checked by ``validate_data``): present,
    the response in the family's support, the covariates numeric and finite, nothing missing, two rows at least."""
    missing = [c for c in (spec.response, *spec.covariates) if c not in data.columns]
    if missing:
        raise FormulaError(f"columns not in data: {missing}"
                           + (f" ({[c for c in missing if c != spec.response]} appear in the non-linear formula, where every name "
                              "is a non-linear parameter with its own formula or a data column)" if set(missing) - {spec.response} else ""))
    reject_complex(f"response {spec.response!r}", data[spec.response])
    fam.validate_response(spec, data)
    for c in spec.covariates:
        reject_complex(f"covariate {c!r}", data[c])
        if not _is_numeric(data[c]):
            raise FormulaError(f"covariate {c!r} of the non-linear formula has dtype {data[c].dtype}; a covariate is a numeric "
                               "column (a categorical predictor belongs in a non-linear parameter's formula)")
    na = data[[spec.response, *spec.covariates]].isna().any()
    if na.any():
        raise FormulaError(f"missing values in columns {list(na[na].index)}; drop or impute them explicitly before fitting.")
    for c in spec.covariates:
        if not np.isfinite(data[c].to_numpy(dtype=float)).all():
            raise FormulaError(f"covariate {c!r} has non-finite values")
    if len(data) < 2:
        raise FormulaError("need at least two observations")


# -- priors -------------------------------------------------------------------------------------------------------------
_NLPAR_KEYS = ("b", "sd")


def nlpar_priors(priors: Priors, spec: NonlinearSpec, name: str) -> Priors:
    """The priors of parameter ``name``'s formula as a ``Priors`` of that formula (``intercept``, ``b``, ``sd``, ``cor``), so
    that the response formula's rules apply to it unchanged (``coefficient_priors``, ``sd_priors``, ``cor_priors``), read from
    ``Priors(nlpars={name: {"b": ..., "sd": ...}}, cor=...)``. ``"b"`` is brms's class ``b`` with ``nlpar = name``: one prior
    for every coefficient, the intercept included (brms's coefficient ``Intercept``; the formula is never centred), or a
    mapping by term with an ``"Intercept"`` entry exactly when the formula has an intercept. ``"sd"`` is read as the response
    formula's ``sd`` (by factor, then by coefficient for a term with slopes) and is required exactly when the formula has a
    group-level term. ``cor`` is the model's, by factor (brms's class ``cor`` has no ``nlpar``)."""
    d = spec.nlpar(name)
    entry = (priors.nlpars or {}).get(name)
    brms = f"brms's class b and sd with nlpar = {name}"
    if not isinstance(entry, Mapping):
        raise PriorError(f"the non-linear parameter {name!r} ('{d.formula}') needs its priors as a mapping {{'b': ...}}"
                         + (", {'sd': ...}" if d.group_terms else "") + f" ({brms}), got {type(entry).__name__}")
    extra = [k for k in entry if k not in _NLPAR_KEYS]
    if extra:
        hint = (" (the LKJ prior of a correlated term is Priors(cor={factor: ab.LKJ(eta)}): brms's class cor has no nlpar)"
                if "cor" in extra else "")
        raise PriorError(f"the priors of {name!r} take the keys {list(_NLPAR_KEYS)} ({brms}); unexpected {extra}{hint}")
    b = entry.get("b")
    if b is None:
        raise PriorError(f"'{d.formula}' has coefficients, which need priors: pass nlpars={{{name!r}: {{'b': ...}}}} ({brms}; the "
                         "intercept is the coefficient 'Intercept' of class b, as the formula is never centred)")
    intercept = rest = None
    if isinstance(b, Prior):
        intercept, rest = (b if d.intercept else None), (b if d.pop_terms else None)
    elif isinstance(b, Mapping):
        if d.intercept and "Intercept" not in b:
            raise PriorError(f"the coefficient priors of {name!r} must cover its intercept: '{d.formula}' has one, the coefficient "
                             "'Intercept' of brms's class b; add 'Intercept' to the mapping")
        if not d.intercept and "Intercept" in b:
            raise PriorError(f"'{d.formula}' has no intercept; omit 'Intercept' from the coefficient priors of {name!r}")
        intercept = b.get("Intercept")
        others = {k: v for k, v in b.items() if k != "Intercept"}
        rest = others if (others or d.pop_terms) else None
    else:
        raise PriorError(f"the coefficient priors of {name!r} are one Prior or a mapping by term, got {type(b).__name__}")
    sd = entry.get("sd")
    if d.group_terms and sd is None:
        raise PriorError(f"'{d.formula}' has the group-level term(s) {[t.canonical for t in d.group_terms]}, whose SDs need priors: "
                         f"pass nlpars={{{name!r}: {{..., 'sd': ...}}}}")
    if not d.group_terms and sd is not None:
        raise PriorError(f"'{d.formula}' has no group-level term; omit 'sd' from the priors of {name!r}")
    cor = None
    factors = tuple(dict.fromkeys(t.factor for t in d.correlated_terms))
    if factors:
        everywhere = spec.correlated_factors
        if isinstance(priors.cor, Mapping):
            cor = {g: priors.cor[g] for g in factors if g in priors.cor}
        elif priors.cor is not None and len(everywhere) == 1:
            cor = {factors[0]: priors.cor}
        cor = cor or None
    return Priors(intercept=intercept, b=rest, sd=sd, cor=cor)


def check_nonlinear_priors(priors: Priors, spec: NonlinearSpec, fam) -> None:
    """``Priors.validate`` for a non-linear model: the classes of a linear formula are refused (its coefficients are its
    parameters'); ``nlpars`` covers exactly the parameters, each read by ``nlpar_priors`` and checked by the response
    formula's rules (location priors for ``b``, positive-support priors for ``sd``, exact coverage); ``cor`` covers
    exactly the factors of the correlated terms; the family's auxiliary parameter takes its one prior."""
    for field in ("intercept", "b", "sd", "simo", "nu"):
        if getattr(priors, field) is not None:
            raise PriorError(f"a non-linear model has no {field} prior of its own: its coefficients and group SDs are its non-linear "
                             f"parameters', given as nlpars={{'<parameter>': {{'b': ..., 'sd': ...}}}} (docs/nonlinear.md)")
    for fields, owner, why in ((("components", "theta"), "a mixture (ab.mixture, docs/mixtures.md)",
                                "a non-linear model in a mixture is not supported (divergence D62)"),
                               (("resp", "rescor"), "a multivariate model (docs/multivariate.md)",
                                "a non-linear model in a multivariate formula is not supported (divergence D62)"),
                               (("sdgp", "lscale"), "a Gaussian process gp(x) (docs/gaussian-processes.md)",
                                "a non-linear model takes no Gaussian process (divergence D61)")):
        given = [f for f in fields if getattr(priors, f) is not None]
        if given:
            raise PriorError(f"{' and '.join(given)} {'is' if len(given) == 1 else 'are'} the prior{'s' if len(given) > 1 else ''} "
                             f"of {owner}; {why}")
    scale, zero = fam.scale_parameter, fam.zero_parameter
    for name in AUXILIARY:
        if getattr(priors, name) is not None and name not in (scale, zero):
            raise PriorError(f"the {fam.name} family has no {AUXILIARY[name]}; do not pass {name}")
    if scale is not None:
        prior = getattr(priors, scale)
        if prior is None:
            raise PriorError(fam.scale_missing_error)
        if isinstance(prior, Mapping):
            raise PriorError(f"prior for {scale!r} is a mapping, the form of a distributional formula's priors; a non-linear model "
                             f"gives {scale} one positive-support prior (divergence D62)")
    if not isinstance(priors.nlpars, Mapping):
        names = list(spec.nlpar_names)
        raise PriorError(f"the non-linear parameters {names} need priors: pass nlpars={{{', '.join(f'{n!r}: {{...}}' for n in names)}}} "
                         "(brms's class b and sd with nlpar =)" + ("" if priors.nlpars is None else f"; got {type(priors.nlpars).__name__}"))
    missing = [n for n in spec.nlpar_names if n not in priors.nlpars]
    extra = [n for n in priors.nlpars if n not in spec.nlpar_names]
    if missing or extra:
        raise PriorError(f"nlpars must cover exactly the non-linear parameters {list(spec.nlpar_names)}; missing {missing}, "
                         f"unexpected {extra}")
    factors = spec.correlated_factors
    if not factors and priors.cor is not None:
        raise PriorError("no non-linear parameter's formula has a correlated group term '(1 + x | g)'; do not pass cor")
    if factors:
        if priors.cor is None:
            raise PriorError(f"the correlated group term(s) on {list(factors)} need an LKJ prior on their correlation matrix: pass "
                             f"cor={{{', '.join(f'{g!r}: ab.LKJ(eta)' for g in factors)}}} (brms's class cor, keyed by group)")
        if isinstance(priors.cor, Mapping) and set(priors.cor) != set(factors):
            raise PriorError(f"cor must cover exactly the factors of the correlated group terms {list(factors)}; got {list(priors.cor)}")
        if not isinstance(priors.cor, Mapping) and len(factors) != 1:
            raise PriorError(f"several factors have correlated group terms {list(factors)}: cor must be a mapping over them")
    for d in spec.nlpars:
        sub = nlpar_priors(priors, spec, d.response)
        try:
            # the response formula's rules on the parameter's formula; the family's scale prior is handed through only so that
            # validate() finds it (it is checked as the model's own scale prior)
            Priors(intercept=sub.intercept, b=sub.b, sd=sub.sd, cor=sub.cor,
                   **({scale: getattr(priors, scale)} if scale is not None else {})).validate(d, fam)
        except PriorError as e:
            raise PriorError(f"non-linear parameter {d.response!r} ('{d.formula}'): {e}") from None


def nonlinear_improper(priors: Priors, spec: NonlinearSpec) -> list[str]:
    """``Priors.improper`` for a non-linear model: the coefficients under a flat prior, labelled ``a:b:Intercept``,
    ``a:b:x``."""
    out = []
    for d in spec.nlpars:
        sub = nlpar_priors(priors, spec, d.response)
        labelled = ([("Intercept", sub.intercept)] if d.intercept else []) + list(sub.coefficient_priors(d).items())
        out += [f"{d.response}:b:{col}" for col, p in labelled if isinstance(p, Prior) and p.name == "Flat"]
    return out


def nonlinear_prior_labels(priors: Priors, spec: NonlinearSpec, fam) -> list[tuple[str, Prior]]:
    """Every prior of a non-linear model with the label messages and fit records give it: ``a:b:Intercept``, ``a:b:z[q]``,
    ``a:sd:g``, ``c:sd:w|h``, ``cor:h``, ``sigma``, in that order."""
    out = []
    for d in spec.nlpars:
        sub = nlpar_priors(priors, spec, d.response)
        out += [(f"{d.response}:b:Intercept", sub.intercept)] if d.intercept else []
        out += [(f"{d.response}:b:{col}", p) for col, p in sub.coefficient_priors(d).items()]
        out += [(f"{d.response}:sd:{sd_tag(g, c)}", p) for (g, c), p in sub.sd_priors(d).items()]
    factors = spec.correlated_factors
    if factors:
        cor = priors.cor if isinstance(priors.cor, Mapping) else {factors[0]: priors.cor}
        out += [(f"cor:{g}", cor[g]) for g in factors]
    if fam.scale_parameter is not None:
        out.append((fam.scale_parameter, getattr(priors, fam.scale_parameter)))
    return out


# -- the model ----------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class NlparDesign:
    """The design of one non-linear parameter's formula: ``X`` is brms's ``X_<name>`` (the intercept column first when the
    formula has one, never centred), ``columns`` its labels (``Intercept``, then the term model's ``x``, ``z[q]``, ...),
    ``group_blocks`` one ``GroupBlock`` per group-level coefficient keyed ``(factor, coefficient)`` (brms's ``Z_<id>_<name>_<k>``
    spread over ``J_<id>``) and ``Z`` the blocks side by side."""
    name: str
    columns: tuple
    X: np.ndarray
    intercept: bool
    group_blocks: Mapping
    Z: np.ndarray


@dataclass(frozen=True)
class NonlinearDesign(Design):
    """What a non-linear model was fit to: the response ``y``, each covariate's column by name in brms's ``C`` order
    (``covariates``) and each parameter's design (``nlpars``). The linear design fields of ``Design`` are empty: the
    non-linear formula has no linear terms of its own."""
    covariates: Mapping = field(default_factory=lambda: MappingProxyType({}))
    nlpars: Mapping = field(default_factory=lambda: MappingProxyType({}))


def nlpar_design(d: ModelSpec, frame: pd.DataFrame) -> np.ndarray:
    """The raw design of parameter formula ``d`` on ``frame`` (the fitted frame, or checked new rows): the intercept column
    when the formula has one, then each population-level term's block (``pop_term_design``)."""
    n = len(frame)
    blocks = ([np.ones((n, 1))] if d.intercept else []) + [pop_term_design(t, frame, d.categoricals) for t in d.pop_terms]
    return np.concatenate(blocks, axis=1) if blocks else np.zeros((n, 0))


def _location_rv(name: str, priors, dims=None):
    """A free variable under one location prior, or a vector under one prior per entry (the same distribution, its
    parameters per entry); parameters as float64."""
    import pymc as pm

    if isinstance(priors, Prior):
        return getattr(pm, priors.name)(name, **{k: np.float64(v) for k, v in priors.params.items()}, dims=dims)
    first = priors[0]
    params = {k: np.array([float(p.params[k]) for p in priors], dtype=np.float64) for k in first.params}
    return getattr(pm, first.name)(name, **params, dims=dims)


def _positive_rv(name: str, prior: Prior):
    """A free positive variable under its positive-support prior (PyMC samples it on the log scale, as Stan does)."""
    import pymc as pm

    return getattr(pm, prior.name)(name, **{k: np.float64(v) for k, v in prior.params.items()})


def _lkj_cholesky_rv(name: str, eta: float, M: int, dims):
    """brms's ``L`` under ``lkj_corr_cholesky(eta)`` (``correlation.lkj_cholesky_logp``), sampled on Stan's ``cholesky_corr``
    coordinates (``<name>_cholesky_corr__``), registered in the model being built; its initial point is the identity."""
    import pymc as pm
    import pytensor.tensor as pt

    from autobayes.correlation import CholeskyCorrTransform, lkj_cholesky_logp, lkj_cholesky_rng

    eta = float(eta)

    def logp(value, eta_):
        return lkj_cholesky_logp(value, eta, M)

    def random(eta_, rng=None, size=None):
        return lkj_cholesky_rng(eta, M, rng if rng is not None else np.random.default_rng(), size)

    def support_point(rv, size, eta_):
        return pt.eye(M)

    return pm.CustomDist(name, np.float64(eta), logp=logp, random=random, support_point=support_point,
                         signature=f"()->({M},{M})", default_transform=CholeskyCorrTransform(M), dims=dims,
                         class_name=f"LKJCholesky_{M}")


class NonlinearModel(Model):
    """A non-linear model (C02, docs/nonlinear.md), built by ``ab.gaussian(formulas, data, priors, nl=True)``: ``spec`` is a
    ``NonlinearSpec``, ``design`` a ``NonlinearDesign``, ``pymc_model`` the graph this class builds itself (there is no
    Bambi model: ``bambi_model`` is None, and ``fit()`` samples ``pymc_model`` through ``sample_posterior``). Everything a
    ``Model`` offers is offered: ``describe()``, ``value_point()``, ``predict()`` (with ``nlpar=``), ``prior_table()``,
    ``prior_predictive()``, ``fit()``, the target identity and fit records."""

    def __init__(self, formula, data: pd.DataFrame, priors: Priors, *, family: str = "gaussian", center_predictors: bool = True,
                 categorical: Mapping[str, object] | None = None, loop: bool = True):
        from autobayes.families import FAMILIES, get_family

        fam = get_family(family)
        if fam.nonlinear_likelihood is None:
            offering = [name for name, other in FAMILIES.items() if other.nonlinear_likelihood is not None]
            raise FormulaError(f"non-linear formulas are verified for the {', '.join(offering)} famil{'y' if len(offering) == 1 else 'ies'} "
                               f"only; the {fam.name} family is not offered with nl=True (divergence D62)")
        if loop is False:
            raise FormulaError("loop=False is not supported (divergence D63): brms then hands the expression to Stan as vector "
                               "arithmetic, where '*' between two vectors is not elementwise; the expression is evaluated row by "
                               "row here, as brms's default loop = TRUE")
        if loop is not True:
            raise ValueError(f"loop must be True (brms's default) or False, got {loop!r}")
        if not isinstance(priors, Priors):
            raise PriorError(f"priors must be an ab.Priors, got {type(priors).__name__}")
        self.family = family
        self.family_spec = fam
        self.priors = priors
        self.center_predictors = bool(center_predictors)
        self.effective_center_predictors = False          # brms never centres a non-linear parameter's formula
        self.dropped_rows: tuple[int, ...] = ()
        self.warnings: list[str] = []
        self.categories: tuple[str, ...] = ()
        self.loop = True
        self.bambi_model = None
        spec0 = parse_nonlinear(formula, None, family, categorical)
        priors.validate(spec0, fam)                       # structure now; per-level coverage once the levels are known
        missing = [c for c in spec0.data_columns if c not in data.columns]
        if missing:
            raise FormulaError(f"columns not in data: {missing}")
        df = data[list(spec0.data_columns)].copy().reset_index(drop=True)
        self.spec: NonlinearSpec = parse_nonlinear(formula, df, family, categorical)
        priors.validate(self.spec, fam)
        for d in self.spec.nlpars:                        # a cell of z:h with no fitted rows (brms keeps the zero column too)
            for t in d.pop_terms:
                if t.kind == "interaction" and len(t.categorical) == 2:
                    empty = [lvl for lvl, col in zip(t.levels, pop_term_design(t, df, d.categoricals).T) if not col.any()]
                    if empty:
                        self.warnings.append(f"cell(s) {empty} of {t.name!r} in the formula of {d.response} have no fitted rows: their "
                                             "coefficients are informed by the prior alone (the design column is zero, as in brms).")
        self.original_labels: dict[str, np.ndarray] = {}
        for cat in self.spec.all_categoricals:            # each declared column as its labels (brms's levels), once
            labels, unused = categorical_labels(f"categorical predictor {cat.name!r}", df[cat.name])
            self.original_labels[cat.name] = labels
            df[cat.name] = labels
            if unused:
                self.warnings.append(f"declared categories {list(unused)} of {cat.name!r} do not occur in the fitted rows: they get no "
                                     "coefficient, and predictions for them are refused.")
        for _, t in self.spec.nlpar_group_terms:         # labels as text; a derived factor g:h gets its own column g_h
            df[t.column] = group_labels(t.factor, df)
        self.warnings.extend(fam.response_warnings(self.spec, df))
        self.data = df
        self._check_names()
        self.design = self._build_design()
        self.pymc_model = self._build_graph()
        self._target_identity = MappingProxyType(freeze_nonlinear_target(self))

    # -- names ------------------------------------------------------------------------------------------------------------
    @property
    def names(self) -> dict[str, str]:
        """PyMC names of the model's roles: ``"linpred"`` and ``"mean"`` the deterministic of the non-linear predictor
        (``mu``; brms's ``mu``, on the link scale), ``"y"`` the response, the family's auxiliary parameter by its brms name
        (``"sigma"``) and ``"nlpar:<name>"`` each parameter's values per row (the deterministic ``<name>``). The variables of
        each parameter's formula are in ``nlpar_names``, ``nlpar_group_names`` and ``nlpar_correlation_names``."""
        fam = self.family_spec
        out = {"linpred": fam.parent, "mean": fam.mean_name, "y": self.spec.response}
        if fam.scale_parameter is not None:
            out[fam.scale_parameter] = fam.bambi_scale_parameter
        if fam.continuous and fam.parent == "mu":
            out["mu"] = "mu"
        out.update({f"nlpar:{name}": name for name in self.spec.nlpar_names})
        return out

    @property
    def nlpar_names(self) -> dict[str, dict[str, str]]:
        """PyMC names of each parameter's formula, by parameter in formula order: ``"value"`` its values per row (the
        deterministic named after it, brms's ``nlp_<name>``), ``"b:Intercept"`` its intercept (``a_Intercept``) when it has
        one and ``"b:<term>"`` each population-level term's coefficient(s) (``a_x``; a vector over ``a_z_dim`` for a
        categorical or interaction term): brms's ``b_<name>``, coefficient by coefficient."""
        return {d.response: {"value": d.response, **({"b:Intercept": f"{d.response}_Intercept"} if d.intercept else {}),
                             **{f"b:{t.name}": f"{d.response}_{t.name}" for t in d.pop_terms}}
                for d in self.spec.nlpars}

    @property
    def nlpar_coefficients(self) -> dict[str, tuple[tuple[str, str, int | None], ...]]:
        """``(design column, PyMC variable, index into it or None)`` per column of each parameter's design ``X_<name>``, in
        column order (``("Intercept", "a_Intercept", None)``, ``("z[q]", "a_z", 0)``): where brms's ``b_<name>[k]`` lives."""
        out = {}
        for d in self.spec.nlpars:
            roles = self.nlpar_names[d.response]
            rows = [("Intercept", roles["b:Intercept"], None)] if d.intercept else []
            rows += [(col, roles[f"b:{term}"], j) for col, term, j in d.column_terms]
            out[d.response] = tuple(rows)
        return out

    @property
    def nlpar_group_names(self) -> dict[tuple[str, str, str], dict[str, str]]:
        """PyMC roles of every group-level coefficient of every parameter's formula, keyed ``(parameter, factor,
        coefficient)``: ``sd`` (brms's ``sd_<id>[k]``), ``offset`` (``z_<id>[k]``) and ``group_effect`` (``r_<id>_<name>_<k>``),
        Bambi's names under the parameter's prefix (``a_1|g_sigma``, ``a_1|g_offset``, ``a_1|g``; ``a_x|g`` for a slope),
        and ``coordinate``, the factor's level dimension (``g__factor_dim``, shared by every parameter)."""
        out = {}
        for name, t in self.spec.nlpar_group_terms:
            for coef in t.coefs:
                base = f"{name}_{bambi_group_name(t.factor, coef)}"
                out[(name, t.factor, coef)] = {"sd": f"{base}_sigma", "offset": f"{base}_offset", "group_effect": base,
                                               "coordinate": f"{factor_column(t.factor)}__factor_dim"}
        return out

    @property
    def nlpar_correlation_names(self) -> dict[tuple[str, str], dict[str, str]]:
        """PyMC names of each correlated term ``(1 + x | g)`` of a parameter's formula, keyed ``(parameter, factor)``: ``L``
        the Cholesky factor (brms's ``L_<id>``, sampled as ``<L>_cholesky_corr__``), ``cor`` the correlation matrix
        (``Cor_<id>``), and their two coefficient dimensions."""
        out = {}
        for name, t in self.spec.nlpar_group_terms:
            if t.has_correlation:
                column = factor_column(t.factor)
                out[(name, t.factor)] = {"L": f"{name}_L|{column}", "cor": f"{name}_cor|{column}",
                                         "coordinate": f"{name}_{column}__cor_dim", "coordinate_T": f"{name}_{column}__cor_dim_T"}
        return out

    def _check_names(self) -> None:
        """Refuse generated names that would alias one another (a variable, a coordinate or the observed response), and
        retained names ending in ``_offset`` (the sampler drops those from the posterior, as Bambi does)."""
        variables = [self.names["linpred"], self.spec.response, *self.spec.nlpar_names]
        if self.family_spec.scale_parameter is not None:
            variables.append(self.family_spec.bambi_scale_parameter)
        coefficients = [v for roles in self.nlpar_names.values() for k, v in roles.items() if k != "value"]
        effects = [roles["group_effect"] for roles in self.nlpar_group_names.values()]
        variables += coefficients + effects + [roles[k] for roles in self.nlpar_group_names.values() for k in ("sd", "offset")]
        variables += [roles[k] for roles in self.nlpar_correlation_names.values() for k in ("L", "cor")]
        coords = ["__obs__", *dict.fromkeys(roles["coordinate"] for roles in self.nlpar_group_names.values())]
        coords += [roles[k] for roles in self.nlpar_correlation_names.values() for k in ("coordinate", "coordinate_T")]
        coords += [f"{self.nlpar_names[d.response][f'b:{t.name}']}_dim" for d in self.spec.nlpars for t in d.pop_terms if t.is_vector]
        names = variables + coords
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise FormulaError(f"generated variable/coordinate name collision: {duplicates}; rename a non-linear parameter or a column")
        if any(n.endswith("_offset") for n in coefficients + effects):
            raise FormulaError("generated name collision: the suffix '_offset' is reserved for the standardised group effects; rename "
                               "the column or the non-linear parameter")

    # -- design and graph -----------------------------------------------------------------------------------------------------
    def _build_design(self) -> NonlinearDesign:
        s, df = self.spec, self.data
        n = len(df)
        nlpars = {}
        for d in s.nlpars:
            X = nlpar_design(d, df)
            group_blocks, start = {}, 0
            for t in d.group_terms:
                levels = d.levels_by_group[t.factor]
                labels = df[t.column].to_numpy()
                indicator = np.column_stack([labels == level for level in levels]).astype(float)
                for coef in t.coefs:
                    covariate = np.ones(n) if coef == "Intercept" else df[coef].to_numpy(dtype=float)
                    group_blocks[(t.factor, coef)] = GroupBlock(levels, indicator * covariate[:, None], indicator.argmax(1),
                                                                slice(start, start + len(levels)))
                    start += len(levels)
            Z = np.concatenate([b.Z for b in group_blocks.values()], axis=1) if group_blocks else np.zeros((n, 0))
            nlpars[d.response] = NlparDesign(name=d.response, columns=(("Intercept",) if d.intercept else ()) + d.columns, X=X,
                                             intercept=d.intercept, group_blocks=MappingProxyType(group_blocks), Z=Z)
        return NonlinearDesign(y=df[s.response].to_numpy(dtype=float), X_raw=np.zeros((n, 0)), X_model=np.zeros((n, 0)),
                               x_means=np.zeros(0), predictors=(), Z=np.zeros((n, 0)), group_blocks=MappingProxyType({}),
                               centered=False, covariates=MappingProxyType({c: df[c].to_numpy(dtype=float) for c in s.covariates}),
                               nlpars=MappingProxyType(nlpars))

    def _build_graph(self):
        """The PyMC graph: each parameter's coefficients, group effects and values per row, the non-linear predictor ``mu``
        and the family's likelihood (``FamilySpec.nonlinear_likelihood``)."""
        import pymc as pm
        import pytensor.tensor as pt

        s, design, fam = self.spec, self.design, self.family_spec
        n = design.n
        env = {}
        with pm.Model(coords={"__obs__": np.arange(n)}) as pmm:
            for d in s.nlpars:
                name, nd, roles = d.response, design.nlpars[d.response], self.nlpar_names[d.response]
                sub = nlpar_priors(self.priors, s, name)
                per_column = sub.coefficient_priors(d)
                parts, col = [], 0
                if d.intercept:
                    parts.append(_location_rv(roles["b:Intercept"], sub.intercept) * np.ones(n))
                    col = 1
                for t in d.pop_terms:
                    var, k = roles[f"b:{t.name}"], len(t.columns)
                    block = np.asarray(nd.X[:, col:col + k], dtype=np.float64)
                    if t.is_vector:
                        pmm.add_coord(f"{var}_dim", list(t.levels))
                        coefficient = _location_rv(var, [per_column[c] for c in t.columns], dims=f"{var}_dim")
                        parts.append(pm.math.dot(block, coefficient))
                    else:
                        parts.append(_location_rv(var, per_column[t.name]) * block[:, 0])
                    col += k
                sd_priors, cor_priors = sub.sd_priors(d), sub.cor_priors(d)
                for t in d.group_terms:
                    levels = d.levels_by_group[t.factor]
                    groles = [self.nlpar_group_names[(name, t.factor, c)] for c in t.coefs]
                    coord = groles[0]["coordinate"]
                    if coord not in pmm.coords:
                        pmm.add_coord(coord, list(levels))
                    sds = [_positive_rv(r["sd"], sd_priors[(t.factor, c)]) for r, c in zip(groles, t.coefs)]
                    zs = [pm.Normal(r["offset"], np.float64(0.0), np.float64(1.0), dims=coord) for r in groles]
                    if t.has_correlation:                 # brms's scale_r_cor: r = (diag(sd) L z)^T
                        croles, M = self.nlpar_correlation_names[(name, t.factor)], len(t.coefs)
                        for dim in (croles["coordinate"], croles["coordinate_T"]):
                            pmm.add_coord(dim, list(t.coefs))
                        L = _lkj_cholesky_rv(croles["L"], cor_priors[t.factor].params["eta"], M,
                                             (croles["coordinate"], croles["coordinate_T"]))
                        pm.Deterministic(croles["cor"], L @ L.T, dims=(croles["coordinate"], croles["coordinate_T"]))
                        effects = [sds[0] * zs[0]] + [sds[c] * sum(L[c, k] * zs[k] for k in range(c + 1)) for c in range(1, M)]
                    else:
                        effects = [sd * z for sd, z in zip(sds, zs)]
                    for coef, r, effect in zip(t.coefs, groles, effects):
                        block = nd.group_blocks[(t.factor, coef)]
                        u = pm.Deterministic(r["group_effect"], effect, dims=coord)
                        covariate = np.ones(n) if coef == "Intercept" else self.data[coef].to_numpy(dtype=np.float64)
                        parts.append(u[np.asarray(block.group_index, dtype=np.int64)] * covariate)
                eta = parts[0]
                for p in parts[1:]:
                    eta = eta + p
                env[name] = pm.Deterministic(roles["value"], eta, dims="__obs__")
            for c in s.covariates:
                env[c] = pt.as_tensor_variable(np.asarray(design.covariates[c], dtype=np.float64))
            mu = pm.Deterministic(self.names["linpred"], to_pytensor(s.expression, env), dims="__obs__")
            scales = {}
            if fam.scale_parameter is not None:
                scales[fam.scale_parameter] = _positive_rv(fam.bambi_scale_parameter, getattr(self.priors, fam.scale_parameter))
            fam.nonlinear_likelihood(self, mu, scales)
        return pmm

    # -- points ---------------------------------------------------------------------------------------------------------------
    def value_point(self, *, nlpars=None, sigma=None, shape=None, phi=None, **unexpected) -> dict[str, np.ndarray]:
        """A point in PyMC's sampled space from natural-scale values: ``nlpars={name: {"b": ..., "sd": ..., "z": ..., "L":
        ...}}`` per parameter, ``b`` the coefficients in the order of its design's columns (``design.nlpars[name].columns``,
        the intercept first; brms's ``b_<name>``), ``sd`` and ``z`` as ``Model.value_point`` takes them for the formula's
        group-level terms (by factor; a bare value for a single intercept-only factor; by coefficient for a term with
        slopes), ``L`` the Cholesky factor of each correlated term by factor (a bare matrix for a single one); and the
        family's auxiliary parameter (``sigma``). Scales are stored on the log scale, ``L`` on Stan's ``cholesky_corr``
        coordinates."""
        from autobayes.correlation import check_correlation_cholesky, cholesky_corr_free

        if unexpected:
            raise ValueError(f"a non-linear model's point takes nlpars={{...}} and its auxiliary parameter; unexpected {sorted(unexpected)}")
        s, fam = self.spec, self.family_spec
        if not isinstance(nlpars, Mapping) or set(nlpars) != set(s.nlpar_names):
            raise ValueError(f"nlpars must be a mapping over exactly the non-linear parameters {list(s.nlpar_names)}")
        point: dict[str, np.ndarray] = {}

        def log_scale(value, label):
            arr = np.asarray(value, dtype=float)
            if arr.shape != () or not np.isfinite(arr) or arr <= 0:
                raise ValueError(f"{label} must be a finite positive scalar")
            return np.asarray(np.log(arr))

        for d in s.nlpars:
            name, entry = d.response, nlpars[d.response]
            if not isinstance(entry, Mapping) or set(entry) - {"b", "sd", "z", "L"}:
                raise ValueError(f"nlpars[{name!r}] is a mapping with the keys 'b', and 'sd', 'z' ('L') for its group-level terms")
            nd, roles = self.design.nlpars[name], self.nlpar_names[name]
            b = np.atleast_1d(np.asarray(entry.get("b", ()), dtype=float))
            if b.shape != (len(nd.columns),) or not np.isfinite(b).all():
                raise ValueError(f"nlpars[{name!r}]['b'] must have {len(nd.columns)} finite entries, one per design column "
                                 f"{list(nd.columns)}")
            i = 0
            if d.intercept:
                point[roles["b:Intercept"]] = np.asarray(b[0])
                i = 1
            for t in d.pop_terms:                         # b in design-column order: one entry per term column
                k = len(t.columns)
                point[roles[f"b:{t.name}"]] = np.array(b[i:i + k]) if t.is_vector else np.asarray(b[i])
                i += k
            if not d.group_terms:
                if any(entry.get(k) is not None for k in ("sd", "z", "L")):
                    raise ValueError(f"'{d.formula}' has no group-level term; sd, z and L do not apply to {name!r}")
                continue
            sds, zs = self._by_factor(d, entry.get("sd"), "sd", name), self._by_factor(d, entry.get("z"), "z", name)
            for t in d.group_terms:
                for coef in t.coefs:
                    groles, block = self.nlpar_group_names[(name, t.factor, coef)], nd.group_blocks[(t.factor, coef)]
                    z = np.asarray(zs[t.factor][coef], dtype=float)
                    if z.shape != (block.J,) or not np.isfinite(z).all():
                        raise ValueError(f"z of {coef!r} of {t.factor!r} in {name!r} must have {block.J} finite entries")
                    point[f"{groles['sd']}_log__"] = log_scale(sds[t.factor][coef], f"sd:{sd_tag(t.factor, coef)} of {name!r}")
                    point[groles["offset"]] = z
            correlated = [t for t in d.group_terms if t.has_correlation]
            L = entry.get("L")
            if not correlated:
                if L is not None:
                    raise ValueError(f"'{d.formula}' has no correlated group term; L does not apply to {name!r}")
            else:
                if L is None:
                    raise ValueError(f"L is required for the correlated term(s) of {name!r}: {[t.canonical for t in correlated]}")
                by = L if isinstance(L, Mapping) else ({correlated[0].factor: L} if len(correlated) == 1 else None)
                if by is None or set(by) != {t.factor for t in correlated}:
                    raise ValueError(f"L of {name!r} must cover exactly the factors {[t.factor for t in correlated]}")
                for t in correlated:
                    croles = self.nlpar_correlation_names[(name, t.factor)]
                    point[f"{croles['L']}_cholesky_corr__"] = cholesky_corr_free(
                        check_correlation_cholesky(by[t.factor], len(t.coefs), f"L for {t.factor!r} of {name!r}"))
        scale = fam.scale_parameter
        auxiliary = {"sigma": sigma, "shape": shape, "phi": phi}
        if {k for k, v in auxiliary.items() if v is not None} != ({scale} if scale is not None else set()):
            raise ValueError(fam.scale_rule)
        if scale is not None:
            point[f"{fam.bambi_scale_parameter}_log__"] = log_scale(auxiliary[scale], scale)
        expected = {v.name for v in self.pymc_model.value_vars}
        if set(point) != expected:
            raise RuntimeError(f"value names {sorted(point)} do not match PyMC's {sorted(expected)}")
        return point

    @staticmethod
    def _by_factor(d: ModelSpec, value, label: str, name: str) -> dict[str, dict[str, object]]:
        """``value_point``'s ``sd`` / ``z`` of one parameter's formula, ``{factor: {coefficient: value}}``."""
        terms = d.group_terms
        if value is None:
            raise ValueError(f"{label} is required for the group-level terms of {name!r}: {[t.canonical for t in terms]}")
        if isinstance(value, Mapping) and set(value) == {t.factor for t in terms}:
            by = dict(value)
        elif len(terms) == 1:
            by = {terms[0].factor: value}
        else:
            raise ValueError(f"{label} of {name!r} must be a mapping over exactly its grouping factors {[t.factor for t in terms]}")
        out = {}
        for t in terms:
            entry = by[t.factor]
            if isinstance(entry, Mapping):
                if set(entry) != set(t.coefs):
                    raise ValueError(f"{label} of {t.factor!r} in {name!r} must cover exactly its coefficients {list(t.coefs)}")
                out[t.factor] = dict(entry)
            elif t.slopes:
                raise ValueError(f"{label} of {t.factor!r} in {name!r} must be a mapping over its coefficients {list(t.coefs)}")
            else:
                out[t.factor] = {"Intercept": entry}
        return out

    # -- description ----------------------------------------------------------------------------------------------------------
    @property
    def default_prior_labels(self) -> list[str]:
        """The priors that came from ``ab.default_priors`` (brms's defaults), labelled as ``nonlinear_prior_labels`` labels them."""
        return [label for label, p in nonlinear_prior_labels(self.priors, self.spec, self.family_spec)
                if getattr(p, "source", "user") == "default"]

    def describe(self) -> str:
        return describe_nonlinear(self)

    def sample_posterior(self, *, inference_method: str = "pymc", draws: int = 1000, tune: int = 1000, chains=None, cores=None,
                         random_seed=None, init: str = "auto", n_init: int = 50000, discard_tuned_samples: bool = True,
                         omit_offsets: bool = True, **kwargs):
        """Sample ``pymc_model`` with PyMC's ``sample`` (``nuts_sampler`` = ``inference_method``, ``"pymc"`` or ``"numpyro"``), as
        Bambi's ``fit()`` samples a model it built: the per-row deterministics (the parameters' values and ``mu``) are not
        kept, and the standardised group effects (``_offset``) are dropped from the posterior unless ``omit_offsets=False``.
        ``ab.fit`` calls it with the same arguments it gives Bambi's ``fit()``."""
        import pymc as pm
        from pymc.util import get_default_varnames

        per_row = {self.names["linpred"], *self.spec.nlpar_names}
        var_names = [v.name for v in get_default_varnames(self.pymc_model.unobserved_value_vars, include_transformed=False)
                     if v.name not in per_row]
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores, random_seed=random_seed, init=init, n_init=n_init,
                          discard_tuned_samples=discard_tuned_samples, var_names=var_names, nuts_sampler=inference_method,
                          model=self.pymc_model, **kwargs)
        if omit_offsets and hasattr(idata, "posterior"):
            idata.posterior = idata.posterior.drop_vars([v for v in idata.posterior.data_vars if v.endswith("_offset")])
        return idata


# -- describe() ----------------------------------------------------------------------------------------------------------------
def _effect_symbol(name: str, factor: str, coef: str) -> str:
    return f"u_{name}_{factor}" if coef == "Intercept" else f"v_{name}_{coef}|{factor}"


def _column_text(t, level) -> str:
    """One design column of a term on row i, as ``describe()`` writes it."""
    if t.kind == "numeric":
        return f"{t.name}_i"
    if t.kind == "transform":
        return t.transform.expression(f"{t.transform.column}_i")
    if t.kind == "categorical":
        return f"1[{t.name}_i = {level}]"
    if not t.categorical:                                  # x:w
        return " * ".join(f"{c}_i" for c in t.operands)
    if len(t.categorical) == 1:                            # x:z
        z = t.categorical[0]
        x = next(c for c in t.operands if c != z)
        return f"{x}_i * 1[{z}_i = {level}]"
    first, second = level.split(", ", 1)                   # z:h, the cell "l, m"
    return f"1[{t.operands[0]}_i = {first}] * 1[{t.operands[1]}_i = {second}]"


def _linear_parts(d: ModelSpec, name: str) -> list[str]:
    """The population-level part of parameter ``name``'s linear predictor, column by column (never centred), each column
    times its coefficient under brms's name (``b_a[zq]``)."""
    from autobayes.default_priors import brms_coefficient_names

    names = brms_coefficient_names(d)
    parts = [f"b_{name}[Intercept]"] if d.intercept else []
    for t in d.pop_terms:
        for col, level in zip(t.columns, t.levels or (None,)):
            parts.append(f"{_column_text(t, level)} * b_{name}[{names[col]}]")
    return parts


def describe_nonlinear(model: NonlinearModel) -> str:
    """``describe()`` of a non-linear model: the formulas, data and levels; the non-linear predictor written out; each
    parameter's linear predictor (never centred), group effects and correlations; the priors under brms's coefficient names;
    the likelihood; the sampled variables and the deterministics."""
    from autobayes.default_priors import brms_coefficient_names

    s, d, pr, fam = model.spec, model.design, model.priors, model.family_spec
    covariates = ", ".join(f"{c} (brms's C_{k})" for k, c in enumerate(s.covariates, start=1)) or "none"
    lines = [f"AutoBayes {fam.tag} model with a non-linear formula: {len(s.nlpars)} non-linear parameter(s) "
             f"({', '.join(s.nlpar_names)}), covariate(s) {covariates}",
             f"  formula   : {s.formula}   (non-linear, brms's nl = TRUE)"]
    if s.canonical_formula != s.formula:
        lines += [f"  expanded  : {s.canonical_formula}"]
    for ds in s.nlpars:
        lines += [f"  {ds.response:<10}: {ds.formula}   (non-linear parameter: its own linear predictor, never centred)"]
    groups = "; ".join(f"{name}: J = {len(s.nlpar(name).levels_by_group[t.factor])} levels of {t.factor}"
                       for name, t in s.nlpar_group_terms)
    lines += [f"  family    : {model.family}, {model.link} link",
              f"  data      : n = {d.n} observations" + (f"; {groups}" if groups else "; no group-level term")]
    for factor in s.factors:
        levels = next(nd.group_blocks[k].levels for nd in d.nlpars.values() for k in nd.group_blocks if k[0] == factor)
        lines += [f"  {'levels ' + factor:<10}: {', '.join(f'{j}={l}' for j, l in enumerate(levels))}"]
    for cat in s.all_categoricals:
        owners = [ds.response for ds in s.nlpars if cat.name in ds.categorical_names]
        labels = model.data[cat.name].astype(str).to_numpy()
        lines += [f"  {cat.name:<10}: categorical in the formula(s) of {', '.join(owners)}, reference {cat.reference!r}; levels "
                  f"(sorted) {', '.join(cat.levels)}; proportions among fitted rows "
                  + ", ".join(f"{l} {np.mean(labels == l):.3f}" for l in cat.levels)]
    lines += ["", "Non-linear predictor",
              f"  {fam.linpred_symbol}_i = {render(_with_suffix(s.expression, set(s.nlpar_names) | set(s.covariates)))}",
              "  evaluated row by row as written, each non-linear parameter at its value on the row ("
              + ", ".join(f"{name}_i" for name in s.nlpar_names) + ")"
              + (f" and each covariate at the row's value ({', '.join(f'{c}_i' for c in s.covariates)})" if s.covariates else "")
              + "; brms's mu with nl = TRUE."]
    if "inv_logit" in functions_in(s.expression):
        lines += ["  inv_logit(x) = 1 / (1 + exp(-x)) (R's plogis)."]
    for ds in s.nlpars:
        name, nd = ds.response, d.nlpars[ds.response]
        body = _linear_parts(ds, name) + [f"{_effect_symbol(name, t.factor, c)}[{t.factor}_i]" + ("" if c == "Intercept" else f" * {c}_i")
                                          for t in ds.group_terms for c in t.coefs]
        lines += ["", f"Non-linear parameter {name}: {ds.formula}", f"  {name}_i = {' + '.join(body)}",
                  f"  The design X_{name} (columns {', '.join(nd.columns)}) is not centred: brms never centres a non-linear parameter's "
                  "formula" + (f", so b_{name}[Intercept] is {name}'s value with every predictor at zero" if ds.intercept else "")
                  + f"; the coefficients are brms's b_{name}."]
        for t in ds.group_terms:
            block = nd.group_blocks[(t.factor, t.coefs[0])]
            for c in t.coefs:
                tag = sd_tag(t.factor, c)
                lines += [f"  {_effect_symbol(name, t.factor, c)}[j] = sd_{name}_{tag} * z_{name}_{tag}[j], z_{name}_{tag}[j] ~ Normal(0, 1), "
                          f"j = 0..{block.J - 1}" + ("" if c == "Intercept" else
                                                     f": the slope of {c} varies by {t.factor} ('{'|' if t.correlated else '||'}')")]
            if t.has_correlation:
                effects = ", ".join(f"{_effect_symbol(name, t.factor, c)}[j]" for c in t.coefs)
                lines += [f"  ({effects}) = diag(sd) L_{name}_{t.factor} (z[j]): L_{name}_{t.factor} is the Cholesky factor of their "
                          f"correlation matrix Cor_{name}_{t.factor} (brms's scale_r_cor)"]
        if ds.group_terms:
            lines += [f"  Each group-level term of {name} is a group of its own (brms gives the same factor in two parameters' "
                      "formulas two group ids); group effects are not mean-centered."]
    lines += ["", "Priors"]
    for ds in s.nlpars:
        name, sub = ds.response, nlpar_priors(pr, s, ds.response)
        names = brms_coefficient_names(ds)
        if ds.intercept:
            lines += [f"  b_{name}[Intercept] ~ {describe_prior(sub.intercept)}"]
        lines += [f"  b_{name}[{names[col]}] ~ {describe_prior(p)}" for col, p in sub.coefficient_priors(ds).items()]
        lines += [f"  sd_{name}_{sd_tag(g, c)} ~ {describe_prior(p, lb=0)}" for (g, c), p in sub.sd_priors(ds).items()]
    for label, prior in nonlinear_prior_labels(pr, s, fam):
        if label.startswith("cor:"):
            g = label[4:]
            lines += [f"  Cor of every correlated term on {g} ~ {prior} (brms lkj({prior.params['eta']:g}), class cor, group {g}, "
                      "placed on each Cholesky factor as lkj_corr_cholesky)" + ("   (brms default)" if prior.source == "default" else "")]
    lines += fam.likelihood_lines(model)
    if model.default_prior_labels:
        from autobayes.default_priors import BRMS_VERSION
        lines += ["", "Default priors (ab.default_priors(..., nl=True); docs/default-priors.md, docs/nonlinear.md)",
                  f"  {', '.join(model.default_prior_labels)}: brms {BRMS_VERSION}'s default_prior() for this model, marked (brms default)",
                  "  above. A non-linear parameter's coefficients are flat (improper: they add nothing to the log density, and",
                  "  prior_predictive() is refused); its group SDs and sigma are student_t(3, 0, scale) with brms's scale of the response."]
    for w in model.warnings:
        lines += ["", "WARNING: " + w]
    lines += ["", "PyMC variables sampled (value space)"]
    roles = {}
    for ds in s.nlpars:
        name, nroles, names = ds.response, model.nlpar_names[ds.response], brms_coefficient_names(ds)
        if ds.intercept:
            roles[nroles["b:Intercept"]] = f"b_{name}[Intercept]"
        for t in ds.pop_terms:
            var = nroles[f"b:{t.name}"]
            roles[var] = (f"b_{name}[{', '.join(names[c] for c in t.columns)}] (vector; dims {var}_dim, coordinates {', '.join(t.levels)})"
                          if t.is_vector else f"b_{name}[{names[t.columns[0]]}]")
    for (name, g, c), r in model.nlpar_group_names.items():
        tag = sd_tag(g, c)
        roles[f"{r['sd']}_log__"] = f"log(sd_{name}_{tag})"
        roles[r["offset"]] = f"z_{name}_{tag}"
    for (name, g), r in model.nlpar_correlation_names.items():
        roles[f"{r['L']}_cholesky_corr__"] = f"L_{name}_{g} on Stan's cholesky_corr coordinates (canonical partial correlations, arctanh scale)"
    if fam.scale_parameter is not None:
        roles[f"{fam.bambi_scale_parameter}_log__"] = f"log({fam.scale_parameter})"
    for v in model.pymc_model.value_vars:
        lines.append(f"  {v.name:<24} {roles.get(v.name, v.name)}")
    lines.append("Deterministic: " + ", ".join(
        [f"{name} = {name}_i per row (brms's nlp_{name})" for name in s.nlpar_names]
        + [f"{r['group_effect']} = {_effect_symbol(name, g, c)}" for (name, g, c), r in model.nlpar_group_names.items()]
        + [f"{r['cor']} = Cor_{name}_{g} = L L^T" for (name, g), r in model.nlpar_correlation_names.items()]
        + [f"{model.names['linpred']} = the non-linear predictor, per row"]))
    return "\n".join(lines)


def _with_suffix(node, names: set):
    """The expression with every parameter and covariate written ``name_i`` (``describe()``)."""
    if isinstance(node, Name):
        return Name(f"{node.name}_i") if node.name in names else node
    if isinstance(node, Negate):
        return Negate(_with_suffix(node.operand, names))
    if isinstance(node, Binary):
        return Binary(node.op, _with_suffix(node.left, names), _with_suffix(node.right, names))
    if isinstance(node, Call):
        return Call(node.function, tuple(_with_suffix(a, names) for a in node.args))
    return node


# -- predictions ----------------------------------------------------------------------------------------------------------------
_NEW_GROUPS = ("error", "resample_observed", "population")


def _policies(factors: tuple[str, ...], new_groups) -> dict[str, str]:
    if not factors and new_groups != "error":
        raise ValueError("this model has no group-level term; new_groups does not apply, so leave it at its default")
    if isinstance(new_groups, Mapping):
        if set(new_groups) != set(factors):
            raise ValueError(f"new_groups must cover exactly the grouping factors {list(factors)}")
        policies = {g: new_groups[g] for g in factors}
    else:
        policies = {g: new_groups for g in factors}
    values = policies.values() if factors else [new_groups]
    if any(not isinstance(v, str) or v not in _NEW_GROUPS for v in values):
        raise ValueError(f"new_groups must be one of {sorted(_NEW_GROUPS)} or an exact-key factor mapping")
    return policies


def check_nonlinear_newdata(model: NonlinearModel, newdata: pd.DataFrame, *, include_group_effects: bool = True) -> pd.DataFrame:
    """The rows of ``newdata`` a non-linear model is predicted on: the covariates and every parameter's columns (and the
    grouping factors' unless group effects are excluded), numeric where they are numbers, the transforms in their domain, the
    categorical labels fitted ones; each grouping factor's labels in its column (a derived factor's ``g_h`` built here)."""
    s = model.spec
    operands = [c for _, t in s.nlpar_group_terms for c in t.operands] if include_group_effects else []
    cols = list(dict.fromkeys([*s.model_columns, *operands]))
    missing = [c for c in cols if c not in newdata.columns]
    if missing:
        hint = (" (the covariates of the non-linear formula are needed on every row to predict)"
                if set(missing) & set(s.covariates) else "")
        raise FormulaError(f"newdata lacks columns {missing}{hint}")
    df = newdata[cols].copy()
    if df.isna().any().any():
        raise FormulaError("newdata contains missing values")
    categorical = {c.name for c in s.all_categoricals}
    for c in s.model_columns:
        if c in categorical:
            continue
        reject_complex(f"newdata column {c!r}", df[c])
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise FormulaError(f"newdata column {c!r} must be numeric")
    for c in s.covariates:
        if not np.isfinite(df[c].to_numpy(dtype=float)).all():
            raise FormulaError(f"newdata covariate {c!r} has non-finite values")
    for d in s.nlpars:
        for t in d.transform_terms:
            t.transform.values(df, where="newdata column")     # numeric, finite, in the function's domain, or refused
    for cat in s.all_categoricals:
        labels, _ = categorical_labels(f"newdata column {cat.name!r}", df[cat.name])
        unseen = sorted(set(labels.tolist()) - set(cat.levels))
        if unseen:
            raise FormulaError(
                f"newdata has levels of the categorical predictor {cat.name!r} that were not in the fitted rows: {unseen}; fitted "
                f"levels {list(cat.levels)} (reference {cat.reference!r}). A categorical predictor has a coefficient only for its "
                "fitted levels; it is not a grouping factor, so new_groups does not apply to it.")
        df[cat.name] = labels
    for _, t in (s.nlpar_group_terms if include_group_effects else ()):
        df[t.column] = group_labels(t.factor, df)
    return df.reset_index(drop=True)


def _nlpar_draws(model: NonlinearModel, post, frame: pd.DataFrame, name: str, *, include_group_effects: bool,
                 policies: dict[str, str], rng) -> np.ndarray:
    """(chain, draw, row) draws of parameter ``name``'s values on ``frame``: its design times its coefficient draws, plus each
    group-level coefficient's effect at the row's level times its covariate (an unseen level by its factor's policy, as
    ``predict._predict_by_hand`` draws it: ``population`` a fresh ``Normal(0, sd)`` per coefficient, ``diag(sd) L z`` for a
    correlated term; ``resample_observed`` one observed level's whole effect vector per label and draw)."""
    d, nd = model.spec.nlpar(name), model.design.nlpars[name]
    X = nlpar_design(d, frame)
    B = np.stack([post[var].to_numpy() if j is None else post[var].to_numpy()[..., j]
                  for _, var, j in model.nlpar_coefficients[name]], axis=-1)
    eta = np.einsum("cdk,nk->cdn", B, X)
    if not include_group_effects:
        return eta
    for t in d.group_terms:
        levels = list(nd.group_blocks[(t.factor, t.coefs[0])].levels)
        labels = frame[t.column].to_numpy()
        coefs = []
        for coef in t.coefs:
            r = model.nlpar_group_names[(name, t.factor, coef)]
            covariate = np.ones(len(frame)) if coef == "Intercept" else frame[coef].to_numpy(dtype=float)
            coefs.append((post[r["sd"]].to_numpy(), post[r["group_effect"]].transpose("chain", "draw", r["coordinate"]).to_numpy(),
                          covariate))
        index = {lvl: i for i, lvl in enumerate(levels)}
        position = np.array([index.get(lvl, -1) for lvl in labels], dtype=np.int64)
        seen = position >= 0
        for _, u, covariate in coefs:
            eta[..., seen] += u[..., position[seen]] * covariate[seen]
        unseen = sorted(set(labels[~seen].tolist()))
        if not unseen:
            continue
        L = None
        if t.has_correlation and policies[t.factor] == "population":
            c = model.nlpar_correlation_names[(name, t.factor)]
            rows = {c["coordinate"]: list(t.coefs), c["coordinate_T"]: list(t.coefs)}
            L = post[c["L"]].sel(rows).transpose("chain", "draw", c["coordinate"], c["coordinate_T"]).to_numpy()
        for lvl in unseen:
            if policies[t.factor] == "population":
                zs = [rng.normal(0.0, 1.0, size=sd.shape) for sd, _, _ in coefs]
                effects = ([z * sd for z, (sd, _, _) in zip(zs, coefs)] if L is None else
                           [sd * sum(L[..., k, m] * zs[m] for m in range(k + 1)) for k, (sd, _, _) in enumerate(coefs)])
            else:
                chosen = rng.integers(0, len(levels), size=coefs[0][0].shape)
                effects = [np.take_along_axis(u, chosen[..., None], axis=-1)[..., 0] for _, u, _ in coefs]
            rows_of = np.flatnonzero(labels == lvl)
            for effect, (_, _, covariate) in zip(effects, coefs):
                eta[..., rows_of] += effect[..., None] * covariate[rows_of]
    return eta


def predict_nonlinear(model: NonlinearModel, fit_result, *, newdata: pd.DataFrame | None = None, kind: str = "response",
                      new_groups="error", seed: int | None = None, include_group_effects: bool = True, dpar: str | None = None,
                      nlpar: str | None = None):
    """``predict()`` of a non-linear model, by hand from the posterior: ``kind="linear"`` the non-linear predictor (brms's
    ``posterior_linpred``, on the link scale, as ``eta``), ``kind="mean"`` the expected response (``posterior_epred``) and
    ``kind="response"`` new observations too; ``nlpar=<name>`` a parameter's values instead (``posterior_linpred(nlpar = )``
    under ``kind="linear"`` as ``eta``, ``posterior_epred(nlpar = )`` under ``kind="mean"`` under its name; the same numbers,
    a parameter has no link). Group-level policies (``new_groups``, ``include_group_effects``) are ``predict()``'s, by grouping
    factor across every parameter's formula; the random numbers of unseen levels follow the parameters in formula order, then
    their group-level terms, then the sorted unseen labels, and a factor in two parameters' formulas draws each term's
    effects independently."""
    import json

    import xarray as xr

    from autobayes.families import family_of
    from autobayes.identity import check_fit_binding
    from autobayes.predict import expected_response

    fam = family_of(model)
    if kind not in fam.prediction_kinds:
        raise ValueError(fam.kind_error)
    if dpar is not None:
        raise ValueError(f"a non-linear model has no distributional formula; dpar does not apply (got {dpar!r}): a non-linear "
                         "parameter's values are predicted with nlpar=")
    s = model.spec
    if nlpar is not None:
        if nlpar not in s.nlpar_names:
            raise ValueError(f"nlpar must name a non-linear parameter, {list(s.nlpar_names)}; got {nlpar!r}")
        if kind not in ("linear", "mean"):
            raise ValueError(f"nlpar={nlpar!r} takes kind='linear' or kind='mean' (its values; brms's posterior_linpred and "
                             "posterior_epred with nlpar =); a new observation is drawn for the response only")
    if not isinstance(include_group_effects, (bool, np.bool_)):
        raise ValueError("include_group_effects must be boolean")
    if not include_group_effects and new_groups != "error":
        raise ValueError("new_groups must stay at its default when group effects are excluded")
    factors = s.factors
    policies = _policies(factors, new_groups)
    binding = check_fit_binding(model, fit_result, allow_unverified=True)
    idata = getattr(fit_result, "idata", fit_result)
    df = None if newdata is None else check_nonlinear_newdata(model, newdata, include_group_effects=include_group_effects)
    frame = model.data if df is None else df
    new_levels: dict[str, list] = {}
    if include_group_effects:
        for name, t in s.nlpar_group_terms:
            fitted = set(model.design.nlpars[name].group_blocks[(t.factor, t.coefs[0])].levels)
            new_levels[t.factor] = sorted(set(new_levels.get(t.factor, [])) | (set(frame[t.column]) - fitted))
    for g, unseen in new_levels.items():
        if unseen and policies[g] == "error":
            raise FormulaError(
                f"newdata has levels of {g!r} not seen in training: {unseen}. Pass new_groups='population' to draw their effects "
                "from Normal(0, sd), or new_groups='resample_observed' for brms's default 'uncertainty' behaviour; an exact-key "
                "mapping can set the policy separately for each factor.")
    post = idata.posterior
    rng = np.random.default_rng(seed)
    values = {name: _nlpar_draws(model, post, frame, name, include_group_effects=include_group_effects, policies=policies, rng=rng)
              for name in s.nlpar_names}
    dims = ("chain", "draw", "__obs__")
    coords = {"chain": post["chain"], "draw": post["draw"], "__obs__": np.arange(len(frame))}
    if nlpar is not None:
        eta = values[nlpar]
        if not np.isfinite(eta).all():
            raise ValueError(f"the values of {nlpar} contain nonfinite draws")
        out = {("eta" if kind == "linear" else nlpar): (dims, eta)}
    else:
        env = {**values, **{c: frame[c].to_numpy(dtype=float) for c in s.covariates}}
        mu = np.broadcast_to(to_numpy(s.expression, env), values[s.nlpar_names[0]].shape).astype(float)
        if not np.isfinite(mu).all():
            raise ValueError("the non-linear predictor contains nonfinite draws: a function of the expression left its domain for some "
                             "draws and rows (log or sqrt of a negative number, a division by zero, an overflow)")
        if kind == "linear":
            out = {"eta": (dims, mu)}
        else:
            mean = expected_response(fam, model, post, mu)
            out = {fam.mean_name: (dims, mean)}
            if kind == "response":
                out["y"] = (dims, fam.draw_response(model, post, mu, mean, frame, rng))
    ds = xr.Dataset(out, coords=coords)
    ds.attrs["include_group_effects"] = int(bool(include_group_effects))
    if not factors or not include_group_effects:
        ds.attrs.update(new_groups="not_applicable", new_levels=[])
    elif len(factors) == 1:
        ds.attrs["new_groups"] = "in_sample" if df is None else policies[factors[0]]
        ds.attrs["new_levels"] = new_levels.get(factors[0], [])
    else:
        ds.attrs["new_groups"] = "in_sample" if df is None else "per_factor"
        ds.attrs["new_groups_by_factor_json"] = json.dumps(policies)
        ds.attrs["new_levels_by_factor_json"] = json.dumps(new_levels)
    ds.attrs["fit_binding_status"] = binding
    if nlpar is not None:
        ds.attrs["nlpar"] = nlpar
    if kind == "linear":
        ds.attrs.update(prediction_scale="linear", link=model.link if nlpar is None else "identity")
    return ds


# -- brms's default priors and prior table ----------------------------------------------------------------------------------
def nonlinear_default_priors(formula, data: pd.DataFrame, *, family: str = "gaussian", categorical=None, nu=None, link=None,
                             families=None, order=None, rescor=None) -> Priors:
    """brms's ``default_prior()`` of a non-linear model (brms 2.23, checked against R): every parameter's coefficients flat
    (class ``b``, ``Flat()``), its group SDs ``student_t(3, 0, scale)`` (``HalfStudentT(3, scale)``) and the family's auxiliary
    parameter its default (the Gaussian ``sigma``: the same), with brms's data-dependent ``scale`` of the response
    (``default_priors.location_scale``: ``max(2.5, round(mad(y), 1))``), and ``lkj(1)`` on each correlated term's factor.
    ``ab.default_priors(..., nl=True)`` hands its keywords through: a mixture's ``families`` and ``order``, a multivariate
    model's ``rescor`` and a family without a non-linear likelihood are refused (divergence D62), and so is ``nu``."""
    from autobayes.default_priors import DF, _default, location_scale
    from autobayes.families import FAMILIES, get_family
    from autobayes.priors import LKJ, Flat, HalfStudentT

    if families is not None or order is not None:
        raise FormulaError("families= and order= are a mixture's (ab.mixture, docs/mixtures.md); a non-linear model in a mixture "
                           "is not supported (divergence D62)")
    if rescor is not None:
        raise FormulaError("rescor= belongs to a multivariate model, 'mvbind(y, y2) ~ ...'; a non-linear model in a multivariate "
                           "formula is not supported (docs/nonlinear.md, divergence D62)")
    fam = get_family(family, link)
    if fam.mixture or fam.nonlinear_likelihood is None:
        raise FormulaError(f"the {fam.name} family is not offered with nl=True (divergence D62)")
    if nu is not None:
        accepting = [name for name, other in FAMILIES.items() if "nu" in other.fixed_parameters]
        raise ValueError(f"nu applies only to the {' or '.join(accepting)} family, which is not offered with nl=True (divergence D62)")
    spec = parse_nonlinear(formula, data, fam.name, categorical)
    _, scale = location_scale(np.asarray(data[spec.response], dtype=float), fam.default_prior_link)
    nlpars = {}
    for d in spec.nlpars:
        entry = {"b": _default(Flat())}
        if d.group_terms:
            entry["sd"] = {t.factor: (_default(HalfStudentT(DF, scale)) if not t.slopes else
                                      {c: _default(HalfStudentT(DF, scale)) for c in t.coefs}) for t in d.group_terms}
        nlpars[d.response] = entry
    auxiliary = {}
    if fam.scale_parameter is not None:
        if fam.default_scale_prior is None:
            raise PriorError(f"no brms default prior is recorded for the {fam.name} family's {fam.scale_parameter}; give the priors "
                             "explicitly")
        auxiliary[fam.scale_parameter] = _default(fam.default_scale_prior(scale))
    cor = {g: _default(LKJ(1)) for g in spec.correlated_factors}
    priors = Priors(nlpars=nlpars, cor=cor or None, **auxiliary)
    priors.validate(spec, fam)
    return priors


def nonlinear_prior_table(priors: Priors, spec: NonlinearSpec, fam) -> list[dict[str, str]]:
    """A non-linear model's priors in brms's table form (``default_priors.prior_table``'s rows and order): each parameter's
    class ``b`` rows (the intercept is the coefficient ``Intercept``) and class ``sd`` rows under its ``nlpar``, the class
    ``cor`` rows by group with no ``nlpar``, and the auxiliary parameter's row. A prior shared by a class sits on the class
    row, as brms writes its defaults."""
    from autobayes.default_priors import COLUMNS, brms_coefficient_names, brms_rename
    from autobayes.priors import brms_prior

    priors.validate(spec, fam)
    rows: list[dict[str, str]] = []

    def add(prior, cls, *, coef="", group="", nlpar="", lb="", source=None):
        text = brms_prior(prior) if isinstance(prior, Prior) else ""
        src = source if source is not None else (prior.source if isinstance(prior, Prior) else "default")
        rows.append(dict(zip(COLUMNS, (text, cls, coef, group, "", "", nlpar, lb, "", "", src))))

    def same(ps):
        return all(p == ps[0] for p in ps)

    for d in spec.nlpars:
        name, sub = d.response, nlpar_priors(priors, spec, d.response)
        names = brms_coefficient_names(d)
        coefs = ([("Intercept", sub.intercept)] if d.intercept else []) + [(names[col], p) for col, p in sub.coefficient_priors(d).items()]
        ps = [p for _, p in coefs]
        if same(ps):                                      # one prior for the class: on the class row, the members inherit it
            add(ps[0], "b", nlpar=name)
            for coef, p in coefs:
                add(None, "b", coef=coef, nlpar=name, source=p.source)
        else:
            add(None, "b", nlpar=name)
            for coef, p in coefs:
                add(p, "b", coef=coef, nlpar=name)
        sd = sub.sd_priors(d)
        if sd:
            ps = list(sd.values())
            add(ps[0] if same(ps) else None, "sd", nlpar=name, lb="0")
            for t in d.group_terms:
                mine = [sd[(t.factor, c)] for c in t.coefs]
                if same(ps):
                    add(None, "sd", group=t.factor, nlpar=name, source=mine[0].source)
                    for c, p in zip(t.coefs, mine):
                        add(None, "sd", coef=brms_rename(c), group=t.factor, nlpar=name, source=p.source)
                elif same(mine):
                    add(mine[0], "sd", group=t.factor, nlpar=name)
                    for c, p in zip(t.coefs, mine):
                        add(None, "sd", coef=brms_rename(c), group=t.factor, nlpar=name, source=p.source)
                else:
                    add(None, "sd", group=t.factor, nlpar=name)
                    for c, p in zip(t.coefs, mine):
                        add(p, "sd", coef=brms_rename(c), group=t.factor, nlpar=name)
    factors = spec.correlated_factors
    if factors:                                           # brms's class cor, by group with no nlpar
        cor = priors.cor if isinstance(priors.cor, Mapping) else {factors[0]: priors.cor}
        ps = [cor[g] for g in factors]
        add(ps[0] if same(ps) else None, "cor")
        for g in factors:
            if same(ps):
                add(None, "cor", group=g, source=cor[g].source)
            else:
                add(cor[g], "cor", group=g)
    if fam.scale_parameter is not None:
        add(getattr(priors, fam.scale_parameter), fam.scale_parameter, lb="0")
    rows.sort(key=lambda r: (r["resp"], r["dpar"], r["nlpar"], r["class"], r["group"], r["coef"], r["tag"]))
    return rows


# -- the target identity and fit records -------------------------------------------------------------------------------------
def freeze_nonlinear_target(model: NonlinearModel) -> dict[str, str]:
    """The target identity of a non-linear model (``identity.freeze_target``'s schema and encoding): the family and link, the
    canonical non-linear formula, the response, each covariate's column, each parameter's canonical formula, design, coding,
    group blocks and priors, and the auxiliary and correlation priors. Only the digest is retained."""
    from autobayes.identity import SCHEMA, _canonical, _prior

    s, d, fam = model.spec, model.design, model.family_spec
    nlpars = []
    for ds in s.nlpars:
        nd = d.nlpars[ds.response]
        nlpars.append({"name": ds.response, "formula": ds.canonical_formula, "intercept": ds.intercept, "columns": list(nd.columns),
                       "X": nd.X,
                       "categorical": [{"name": c.name, "reference": c.reference, "levels": c.levels, "coefficients": c.coefficients,
                                        "full_rank": c.full_rank} for c in ds.categoricals],
                       "groups": [{"factor": g, "coef": c, "levels": b.levels, "index": b.group_index, "Z": b.Z,
                                   "correlated": ds.group_term_of(g).correlated, "noncentered": True}
                                  for (g, c), b in nd.group_blocks.items()]})
    payload = {"schema": SCHEMA, "family": model.family, "link": model.link,
               "nonlinear": {"formula": s.canonical_formula, "response": s.response, "y": d.y,
                             "covariates": [{"name": c, "values": d.covariates[c]} for c in s.covariates], "nlpars": nlpars},
               "priors": {label: _prior(p) for label, p in nonlinear_prior_labels(model.priors, s, fam)}}
    encoded = json.dumps(_canonical(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return {"schema": SCHEMA, "sha256": hashlib.sha256(encoded).hexdigest()}


def nonlinear_summary_rows(model: NonlinearModel, idata) -> dict:
    """The fit record's posterior summary of a non-linear model: every parameter's coefficients, its group SDs and the
    auxiliary parameter."""
    import arviz as az

    names = list(dict.fromkeys(var for rows in model.nlpar_coefficients.values() for _, var, _ in rows))
    names += [roles["sd"] for roles in model.nlpar_group_names.values()]
    scale = model.family_spec.scale_parameter
    names += [model.names[scale]] if scale is not None else []
    s = az.summary(idata, var_names=names, hdi_prob=0.94, round_to="none")
    return {name: {k: float(r[k]) for k in ("mean", "sd", "hdi_3%", "hdi_97%", "mcse_mean", "ess_bulk", "ess_tail", "r_hat")}
            for name, r in s.iterrows()}


def nonlinear_fit_record(model: NonlinearModel, fit, panel: dict, binding: str) -> dict:
    """``session.fit_record`` of a non-linear model: its own model section (every formula, the covariates, each parameter's
    formula, design columns, groups and PyMC names, every prior by label) and posterior summary; the execution, environment
    and diagnostics sections are ``fit_record``'s."""
    from autobayes import __version__
    from autobayes import session
    from autobayes.default_priors import BRMS_VERSION

    s, d = model.spec, model.design
    defaults = model.default_prior_labels

    def groups(ds):
        nd = d.nlpars[ds.response]
        return [{"factor": t.factor, "coefs": list(t.coefs), "correlated": t.has_correlation,
                 "levels": list(nd.group_blocks[(t.factor, t.coefs[0])].levels),
                 "names": {c: model.nlpar_group_names[(ds.response, t.factor, c)] for c in t.coefs},
                 **({"correlation_names": model.nlpar_correlation_names[(ds.response, t.factor)]} if t.has_correlation else {})}
                for t in ds.group_terms]

    return {
        "autobayes_version": __version__,
        "model": {
            "formula": list(s.formulas),
            "family": model.family,
            **({"target_identity": model.target_identity} if binding == "matched" else
               {"requested_target_identity": model.target_identity}),
            "fit_binding_status": binding,
            "link": model.link,
            "n": int(d.n),
            "nonlinear": {"formula": s.formula, "canonical_formula": s.canonical_formula, "covariates": list(s.covariates),
                          "names": dict(model.names),
                          "nlpars": {ds.response: {"formula": ds.formula, "canonical_formula": ds.canonical_formula,
                                                   "design_columns": list(d.nlpars[ds.response].columns),
                                                   "names": model.nlpar_names[ds.response], "groups": groups(ds),
                                                   "categorical": {c.name: {"reference": c.reference, "levels": list(c.levels),
                                                                            "coefficients": list(c.coefficients),
                                                                            "coding": "full_level" if c.full_rank else "treatment"}
                                                                   for c in ds.categoricals}}
                                     for ds in s.nlpars}},
            "priors": {label: str(p) for label, p in nonlinear_prior_labels(model.priors, s, model.family_spec)},
            **({"default_priors": {"brms": BRMS_VERSION, "parameters": list(defaults)}} if defaults else {}),
            "center_predictors": False,
            "center_predictors_requested": model.center_predictors,
            "noncentered": True if s.factors else None,
            "warnings": list(model.warnings),
        },
        "execution": {
            "backend": fit.backend, "device_requested": fit.device_requested,
            "device_actual": fit.device_actual, "platform": fit.platform, "dtype": fit.dtype,
            "chain_method": fit.chain_method, "chains": fit.chains, "draws": fit.draws,
            "tune": fit.tune, "target_accept": fit.target_accept, "seed": fit.seed,
            "wall_seconds": round(float(fit.wall_seconds), 3), "notes": list(fit.notes),
        },
        "environment": session.environment(),
        "diagnostics": {"values": {k: (float(v) if v is not None else None) for k, v in panel["values"].items()},
                        "status": dict(panel["status"]), "overall": panel["overall"]},
        "posterior": nonlinear_summary_rows(model, fit.idata),
    }
