"""brms's ``hypothesis()`` on the population-level coefficients of a fit (brms-parity D06, docs/post-processing.md).
Accepted against brms 2.23 on the frozen oracle cases ``gauss_hypothesis_effects`` and ``bernoulli_hypothesis_effects`` and
against brms's outputs on recorded draws (``tests/hypothesis_probes``); none of brms's source is used.

A hypothesis is ``left (= OR < OR >) right`` over brms's names of the class ``b`` coefficients (``x``, ``zq``, ``x:zq``,
``Intercept``, the raw-scale intercept). Its draws are those of ``left - right`` (of ``left`` when ``right`` is ``0``), and
brms's table summarises them (settled on brms's outputs): ``Estimate`` the mean and ``Est.Error`` the SD (with ``robust``,
the median and the MAD, ``1.4826 median |v - median v|``); ``CI.Lower`` and ``CI.Upper`` the ``alpha / 2`` and ``1 - alpha
/ 2`` quantiles (R's type 7) of a point hypothesis (``=``) and the ``alpha`` and ``1 - alpha`` quantiles of a one-sided one;
for a one-sided hypothesis ``Post.Prob`` the share of draws strictly on its side of zero and ``Evid.Ratio`` the draws on that
side over the others (zero draws count against it; ``Inf`` when every draw is on its side); for a point hypothesis both are
NaN, brms's NA without prior draws (divergence D96); ``Star`` is ``*`` when ``Post.Prob > 1 - alpha`` (one-sided) or the
interval excludes zero (point). The ``Hypothesis`` column is brms's text: whitespace removed, ``(left) op 0`` when ``right``
is the text ``0``, else ``(left)-(right) op 0``.

The text is read by the small parser below and never evaluated as code. It takes what a hypothesis on coefficients needs:
numbers in decimal notation, names, parentheses, ``+ - * / ^`` (``^`` right-associative and above the unary signs, as in
R) and ``exp``, ``log``, ``sqrt`` and ``abs`` of one argument; anything else is refused (divergence D97).
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HYPOTHESIS_COLUMNS = ("Hypothesis", "Estimate", "Est.Error", "CI.Lower", "CI.Upper", "Evid.Ratio", "Post.Prob", "Star")
FUNCTIONS = {"abs": np.abs, "exp": np.exp, "log": np.log, "sqrt": np.sqrt}
MAD_CONSTANT = 1.4826                       # R's mad(): the MAD scaled to the SD of a normal distribution
_FORM = "every hypothesis must be of the form 'left (= OR < OR >) right'"
_NUMBER = re.compile(r"\d+(?:\.\d+)?|\.\d+")
_NAME = re.compile(r"[A-Za-z][A-Za-z0-9._]*(?::[A-Za-z][A-Za-z0-9._]*)*")


# -- the parser ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ParsedHypothesis:
    """One hypothesis read by ``parse_hypothesis``: ``text`` as given, ``label`` brms's ``Hypothesis`` text, ``sign`` one of
    ``=``, ``<``, ``>``, ``left`` and ``right`` the two sides as expression trees (``right`` is None when it is the text
    ``0``) and ``names`` the parameter names read, in order of appearance."""

    text: str
    label: str
    sign: str
    left: tuple
    right: tuple | None
    names: tuple[str, ...]

    @property
    def one_sided(self) -> bool:
        return self.sign != "="


class _Parser:
    """A recursive-descent parser of one side of a hypothesis (whitespace already removed) into a tree of tuples:
    ``("num", value)``, ``("name", name)``, ``("neg", x)``, ``("pos", x)``, ``("bin", op, a, b)`` and ``("call", fn, x)``.
    The grammar is R's for these operators: sums of products of signed powers, ``^`` binding tighter than a sign on its
    left and taking a signed power on its right (``-2^2`` is ``-4``, ``2^-1`` is ``0.5``, ``2^3^2`` is ``2^9``)."""

    def __init__(self, hypothesis: str, text: str):
        self.hypothesis, self.text, self.pos, self.names = hypothesis, text, 0, []

    def fail(self, why: str):
        raise ValueError(f"hypothesis {self.hypothesis!r}: {why}")

    def peek(self) -> str:
        return self.text[self.pos] if self.pos < len(self.text) else ""

    def parse(self) -> tuple:
        node = self.sum()
        if self.pos != len(self.text):
            self.unexpected()
        return node

    def unexpected(self):
        if self.pos >= len(self.text):
            self.fail(f"{self.text!r} ends too early")
        self.fail(f"unexpected {self.text[self.pos]!r} at {self.text[self.pos:]!r} (a hypothesis takes numbers, coefficient "
                  f"names, parentheses, + - * / ^ and {', '.join(sorted(FUNCTIONS))}; divergence D97)")

    def sum(self) -> tuple:
        node = self.product()
        while self.peek() and self.peek() in "+-":
            op = self.peek()
            self.pos += 1
            node = ("bin", op, node, self.product())
        return node

    def product(self) -> tuple:
        node = self.signed()
        while self.peek() and self.peek() in "*/":
            op = self.peek()
            self.pos += 1
            node = ("bin", op, node, self.signed())
        return node

    def signed(self) -> tuple:
        if self.peek() and self.peek() in "+-":
            op = self.peek()
            self.pos += 1
            return ("neg" if op == "-" else "pos", self.signed())
        return self.power()

    def power(self) -> tuple:
        base = self.primary()
        if self.peek() == "^":
            self.pos += 1
            return ("bin", "^", base, self.signed())
        return base

    def primary(self) -> tuple:
        c = self.peek()
        if c == "(":
            self.pos += 1
            node = self.sum()
            if self.peek() != ")":
                self.unexpected()
            self.pos += 1
            return node
        if m := _NUMBER.match(self.text, self.pos):
            end = m.end()
            if end < len(self.text) and (self.text[end].isalnum() or self.text[end] in "._"):
                self.fail(f"{self.text[self.pos:end + 1]!r}...: write numbers in decimal notation, like 0.001 (brms does not "
                          "take exponent notation, R's 1L or hexadecimal either; divergence D97)")
            self.pos = end
            return ("num", float(m.group()))
        if m := _NAME.match(self.text, self.pos):
            name, self.pos = m.group(), m.end()
            if self.peek() == "(":
                if name not in FUNCTIONS:
                    self.fail(f"the function {name}() is not offered in a hypothesis; offered: "
                              f"{', '.join(f'{f}()' for f in sorted(FUNCTIONS))} (divergence D97)")
                self.pos += 1
                arg = self.sum()
                if self.peek() != ")":
                    self.unexpected()
                self.pos += 1
                return ("call", name, arg)
            self.names.append(name)
            return ("name", name)
        self.unexpected()


def parse_hypothesis(text: str) -> ParsedHypothesis:
    """Read one hypothesis (module docstring); refuses anything outside its grammar with a message naming the problem."""
    if not isinstance(text, str):
        raise ValueError(f"a hypothesis is a string like 'x > 0', got {type(text).__name__}")
    compact = re.sub(r"\s+", "", text)                      # brms removes all whitespace first
    signs = [c for c in compact if c in "=<>"]
    if len(signs) != 1:
        raise ValueError(f"hypothesis {text!r}: {_FORM} with exactly one of them (brms: \"Every hypothesis must be of the "
                         "form 'left (= OR < OR >) right'\"; <=, >=, == and != are not taken)")
    sign = signs[0]
    left, right = compact.split(sign)
    if not left or not right:
        raise ValueError(f"hypothesis {text!r}: {_FORM}, with both sides given")
    lp, rp = _Parser(text, left), _Parser(text, right)
    ltree, rtree = lp.parse(), rp.parse()
    names = tuple(dict.fromkeys(lp.names + rp.names))
    if not names:
        raise ValueError(f"hypothesis {text!r} names no parameter (brms refuses it too)")
    zero = right == "0"
    label = f"({left}) {sign} 0" if zero else f"({left})-({right}) {sign} 0"
    return ParsedHypothesis(text=text, label=label, sign=sign, left=ltree, right=None if zero else rtree, names=names)


def _evaluate(node: tuple, draws: Mapping[str, np.ndarray]):
    kind = node[0]
    if kind == "num":
        return node[1]
    if kind == "name":
        return draws[node[1]]
    if kind == "neg":
        return -_evaluate(node[1], draws)
    if kind == "pos":
        return +_evaluate(node[1], draws)
    if kind == "call":
        return FUNCTIONS[node[1]](_evaluate(node[2], draws))
    a, b = _evaluate(node[2], draws), _evaluate(node[3], draws)
    op = node[1]
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        return np.true_divide(a, b)
    return np.power(a, b)


def hypothesis_values(parsed: ParsedHypothesis, draws: Mapping[str, np.ndarray]) -> np.ndarray:
    """The draws of ``left - right`` (of ``left`` when ``right`` is ``0``), with ``draws`` the parameters' draws by name."""
    S = len(next(iter(draws.values())))
    with np.errstate(all="ignore"):
        value = _evaluate(parsed.left, draws)
        if parsed.right is not None:
            value = value - _evaluate(parsed.right, draws)
    return np.broadcast_to(np.asarray(value, dtype=float), (S,)).copy()


# -- the summaries -------------------------------------------------------------------------------------------------
def _mean(v: np.ndarray) -> float:
    return math.fsum(v) / v.size if np.isfinite(v).all() else float(np.mean(v))


def _summary(parsed: ParsedHypothesis, v: np.ndarray, alpha: float, robust: bool) -> list:
    S = v.size
    if robust:
        estimate = float(np.median(v))
        error = MAD_CONSTANT * float(np.median(np.abs(v - estimate)))
    else:
        estimate = _mean(v)
        error = float(np.std(v, ddof=1)) if S > 1 else math.nan
    probs = (alpha, 1.0 - alpha) if parsed.one_sided else (alpha / 2.0, 1.0 - alpha / 2.0)
    lower, upper = (float(q) for q in np.quantile(v, probs))
    if parsed.one_sided:
        k = int(np.sum(v < 0) if parsed.sign == "<" else np.sum(v > 0))
        prob = k / S
        ratio = math.inf if k == S else k / (S - k)
        star = "*" if prob > 1.0 - alpha else ""
    else:
        prob = ratio = math.nan
        star = "*" if lower > 0 or upper < 0 else ""
    return [parsed.label, estimate, error, lower, upper, ratio, prob, star]


@dataclass(frozen=True)
class Hypothesis:
    """brms's ``hypothesis()`` result: ``table`` brms's columns (``HYPOTHESIS_COLUMNS``), one row per hypothesis in the order
    given; ``samples`` the draws of each hypothesis, ``(draws, hypotheses)`` with brms's column names ``H1``, ``H2``, ...
    (the chains one after another); ``hypotheses`` the hypotheses as given; ``alpha`` and ``robust`` as used;
    ``fit_binding_status`` whether the fit was checked against the model (``"matched"``) or carried no identity
    (``"unverified"``; ``""`` for draws given by name)."""

    table: pd.DataFrame
    samples: pd.DataFrame
    hypotheses: tuple[str, ...]
    alpha: float
    robust: bool
    fit_binding_status: str = field(default="")

    def __str__(self) -> str:
        level = 1.0 - self.alpha
        shown = self.table.copy()
        for c in HYPOTHESIS_COLUMNS[1:-1]:
            shown[c] = shown[c].map(lambda v: f"{v:.2f}" if np.isfinite(v) else ("NA" if np.isnan(v) else f"{v}".title()))
        centre = "median and MAD" if self.robust else "mean and SD"
        return "\n".join([
            f"Hypothesis tests on the population-level coefficients (class b) from {len(self.samples)} posterior draws",
            shown.to_string(index=False),
            f"Estimate, Est.Error: the {centre} of each hypothesis's draws. CI: the {100 * (1 - 2 * self.alpha):g}% interval "
            f"of a one-sided hypothesis, the {100 * level:g}% interval of a point hypothesis.",
            f"'*': a one-sided hypothesis's posterior probability exceeds {level:g}, or a point hypothesis's interval excludes 0.",
            "A point hypothesis has no evidence ratio here (no prior draws; divergence D96)."])


def _check_options(hypotheses, alpha, robust) -> tuple[str, ...]:
    if isinstance(hypotheses, str):
        hypotheses = (hypotheses,)
    if not isinstance(hypotheses, Sequence) or not hypotheses:
        raise ValueError("hypotheses must be a string or a non-empty list of strings, like ['x > 0', 'x + w = 0']")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.floating)) or not 0 < float(alpha) < 1:
        raise ValueError(f"alpha must be a number strictly between 0 and 1, got {alpha!r}")
    if not isinstance(robust, (bool, np.bool_)):
        raise ValueError(f"robust must be True or False, got {robust!r}")
    return tuple(hypotheses)


def hypothesis_from_draws(draws: Mapping[str, np.ndarray], hypotheses, *, alpha: float = 0.05, robust: bool = False,
                          unavailable: Mapping[str, str] | None = None, what: str = "parameter") -> Hypothesis:
    """brms's ``hypothesis()`` of named draws (its default method on a data frame of draws): ``draws`` maps each parameter
    name to its ``(S,)`` draws; ``unavailable`` maps names that exist but cannot be tested to the reason, quoted when a
    hypothesis names one; ``what`` names the parameters in messages."""
    hypotheses = _check_options(hypotheses, alpha, robust)
    arrays = {k: np.asarray(v, dtype=float).ravel() for k, v in draws.items()}
    sizes = {v.size for v in arrays.values()}
    if len(sizes) != 1 or 0 in sizes:
        raise ValueError(f"the draws must have the same positive number of draws for every {what}, got {sorted(sizes)}")
    S = sizes.pop()
    rows, samples = [], []
    for text in hypotheses:
        parsed = parse_hypothesis(text)
        for name in parsed.names:
            if name in arrays:
                continue
            if unavailable and name in unavailable:
                raise ValueError(f"hypothesis {text!r}: {name!r} {unavailable[name]}")
            raise ValueError(f"hypothesis {text!r}: no {what} {name!r}; the {what}s are {sorted(arrays)}")
        v = hypothesis_values(parsed, {k: arrays[k] for k in parsed.names})
        if np.isnan(v).any():
            raise ValueError(f"hypothesis {text!r} is NaN at {int(np.isnan(v).sum())} of {S} draws (the log or square root of a "
                             "negative value?); brms stops there too")
        rows.append(_summary(parsed, v, float(alpha), bool(robust)))
        samples.append(v)
    table = pd.DataFrame(rows, columns=list(HYPOTHESIS_COLUMNS))
    for c in HYPOTHESIS_COLUMNS[1:-1]:
        table[c] = table[c].astype(float)
    sample_frame = pd.DataFrame(np.column_stack(samples), columns=[f"H{k}" for k in range(1, len(samples) + 1)])
    return Hypothesis(table=table, samples=sample_frame, hypotheses=hypotheses, alpha=float(alpha), robust=bool(robust))


# -- on a model's fit ----------------------------------------------------------------------------------------------
def refuse_unsupported(model, what: str) -> None:
    """``what`` (a post-processing function of D06) is not offered for a multivariate or non-linear model, a mixture or
    the cumulative and categorical families, whose coefficients brms names per response, parameter, component or category
    (divergence D99)."""
    if getattr(model, "responses", None):
        raise ValueError(f"{what} of a multivariate model is not offered (brms names its coefficients per response; "
                         "docs/post-processing.md, divergence D99)")
    if getattr(getattr(model, "spec", None), "is_nonlinear", False):
        raise ValueError(f"{what} of a non-linear model (nl=True) is not offered (brms names its coefficients per non-linear "
                         "parameter; docs/post-processing.md, divergence D99)")
    fam = model.family_spec
    for flag, construct in ((fam.mixture, "a mixture"), (fam.per_category, f"the {model.family} family's categories"),
                            (fam.thresholds, f"the {model.family} family's thresholds")):
        if flag:
            raise ValueError(f"{what} of a model with {construct} is not offered (docs/post-processing.md, divergence D99)")


def coefficient_draws(model, posterior) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """The draws of brms's class ``b`` of a univariate model by brms's names (``Intercept``, the raw-scale intercept, and
    each design column's name, ``default_priors.brms_coefficient_names``), flattened chain by chain; and the brms names that
    cannot be tested here, with the reason: the special terms' (``mo()``, ``me()``, ``mi()``; brms's class ``bsp``) and the
    coefficients of a distributional formula (``sigma_Intercept``, ``sigma_z``; brms's ``b_sigma_...``)."""
    from autobayes.default_priors import brms_coefficient_names

    spec = model.spec
    brms = brms_coefficient_names(spec)
    out = {}
    if spec.intercept:
        out["Intercept"] = posterior[model.names["intercept"]].transpose("chain", "draw").to_numpy().reshape(-1)
    for column, variable, j in model.coefficients:
        arr = posterior[variable].transpose("chain", "draw", ...).to_numpy()
        out[brms[column]] = (arr if j is None else arr[..., j]).reshape(-1)
    unavailable = {brms[t.name]: f"is the coefficient of {t.name}, brms's class bsp, not b (divergence D97)"
                   for t in spec.pop_terms if t.is_special}
    for ds in spec.dpars:                                     # B01: brms's b_<dpar>_..., not offered here
        names = ([f"{ds.response}_Intercept"] if ds.intercept else []) + [f"{ds.response}_{n}" for n in brms_coefficient_names(ds).values()]
        unavailable.update({n: f"is a coefficient of the distributional formula for {ds.response}, which hypothesis() does not take "
                               "here (divergence D97)" for n in names})
    return out, unavailable


def hypothesis(model, fit_result, hypotheses, *, alpha: float = 0.05, robust: bool = False) -> Hypothesis:
    """brms's ``hypothesis(fit, hypotheses)`` with its defaults ``class = "b"``, ``alpha = 0.05``, ``robust = FALSE``: each
    hypothesis (``"x > 0"``, ``"x + w = 0"``, ``"zr < zq"``, ``"Intercept < 0"``; module docstring) over brms's names of the
    population-level coefficients, on the posterior draws of ``fit_result`` (a ``FitResult`` or its ``InferenceData``,
    checked against the model as ``predict()`` checks it). Returns an ``ab.Hypothesis``: brms's table and the draws."""
    from autobayes.identity import check_fit_binding

    refuse_unsupported(model, "hypothesis()")
    hypotheses = _check_options(hypotheses, alpha, robust)
    binding = check_fit_binding(model, fit_result, allow_unverified=True)
    posterior = getattr(fit_result, "idata", fit_result).posterior
    draws, unavailable = coefficient_draws(model, posterior)
    result = hypothesis_from_draws(draws, hypotheses, alpha=alpha, robust=robust, unavailable=unavailable,
                                   what="population-level coefficient (class b)")
    return Hypothesis(table=result.table, samples=result.samples, hypotheses=result.hypotheses, alpha=result.alpha,
                      robust=result.robust, fit_binding_status=binding)
