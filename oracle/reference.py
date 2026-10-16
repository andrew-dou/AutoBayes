#!/usr/bin/env python
"""High-precision references for the brms oracle cases whose brms side is inexact at some frozen points (P6).

    python oracle/reference.py <case-id> [<case-id> ...]          # write tests/oracle/references/<case-id>.json
    python oracle/reference.py --check <case-id> [...] | --all    # recompute and compare with the committed file

A case that opts in to two-reference acceptance (its ``gate`` field ``{"expect": "pass", "reference": "<case-id>.json",
"divergence": ..., "fallback": [...]}``, docs/brms-parity/LOOP.md) is compared with brms wherever brms is accurate and
with this reference where brms is not. The reference is an independent third implementation: it uses the standard
library only (nothing from autobayes or tests/), reads the frozen fixture's ``case_brms.json``, ``data.csv``,
``newdata.csv`` and the seeded points' unconstrained values in ``fixture.json``, applies brms's constraining transforms
and their Jacobians itself (identity for ``b``, ``Intercept`` and ``z_1``; ``exp`` with log-Jacobian ``u`` for ``sigma``
and ``sd_1``; Stan's ``ordered`` transform of the cumulative model's thresholds, ``Intercept[1] = u_1`` and
``Intercept[k] = Intercept[k - 1] + exp(u_k)`` with log-Jacobian ``u_2 + ... + u_K``; as the fixture's probed ``upars`` table
says), and evaluates the model's log density in Python's decimal arithmetic at 50 significant digits. The models, in
brms's parameterisation (``eta = Intercept + b (x - mean(x)) + sd_1 z_1[g]``, the separate centred intercept and
non-centred group effects of a varying intercept on ``g``; the cumulative model's ``eta`` has no intercept, its
thresholds ``c_1 < ... < c_K`` are brms's ``Intercept``, and ``c_0 = -inf``, ``c_{K+1} = inf``):

    bernoulli, cloglog   yb ~ x + (1 | g)                     yb log(1 - exp(-exp(eta))) - (1 - yb) exp(eta)
    binomial, cloglog    k | trials(size) ~ x + (1 | g)       log C(size, k) + k log(1 - exp(-exp(eta))) - (size - k) exp(eta)
    gaussian, censored   y | cens(cens_ind) ~ x + (1 | g)     by code 0 / 1 / -1: log of the density, of 1 - Phi or of Phi
                                                              of (y - eta) / sigma (the density's -log sigma included)
    gaussian, truncated  ypos | trunc(lb = 0) ~ x + (1 | g)   the density minus log(1 - Phi((lb - eta) / sigma))
    cumulative, probit   ord ~ x + (1 | g)                    log(Phi(c_ord - eta) - Phi(c_{ord-1} - eta)), ord in 1..K+1,
                                                              the difference taken in the tail it lies in (``log_normal_interval``)

with brms's priors: ``normal(m, s)`` on ``Intercept`` and ``b`` (``normal_lpdf``), ``normal(m, s)`` on ``sd_1`` and ``sigma``,
which are declared ``<lower=0>`` (``normal_lpdf(. | m, s) - normal_lccdf(0 | m, s)``), and ``std_normal`` on ``z_1``. Every
normalising constant is included, as ``rstan::log_prob`` includes them. The inputs are the doubles brms and AutoBayes
evaluate, converted to decimal exactly: the CSV's numbers parsed as binary64 (the generator rounds them to four decimals
so that R and Python parse the same double) and the fixture's ``u``; everything from there on, the centring of ``x``
included, is exact to 50 digits.

Per seeded point the file holds the log density with and without the Jacobian, the gradient with and without it in
brms's unconstrained order (without: a fourth-order central difference of the 50-digit density with step 1e-10, whose
truncation and rounding errors are both below 1e-35; with: plus 1 in each log-transformed coordinate and in each
threshold coordinate after the first, the derivative of its log-Jacobian ``u``), and, for the truncated model, every expected value gate G4 compares: ``posterior_epred``'s
truncated mean ``eta + sigma phi(a) / (1 - Phi(a))``, ``a = (lb - eta) / sigma``, on the fitted rows and on ``newdata``
(``x`` centred at the fitted mean), with and without the group effects. Values are written as strings with 30
significant digits, with the sha256 of every file read. Erf and erfc: the series of positive terms for ``|x| < 3`` (with
15 guard digits) and the Laplace continued fraction above; the gradient's step and the guard digits are constants here.
Re-running the script reproduces a committed file byte for byte (``--check``); a committed reference is never
overwritten (a changed fixture would be a new case id).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from decimal import ROUND_HALF_EVEN, Context, Decimal, getcontext, localcontext
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "oracle" / "fixtures"
REFERENCES = REPO / "tests" / "oracle" / "references"

PREC = 50                       # significant digits of the arithmetic
GUARD = 15                      # further digits inside erf, erfc, pi and the constants
DIGITS = 30                     # significant digits written
STEP = Decimal("1e-10")         # step of the fourth-order central difference
CF_FROM = Decimal(3)            # erfc by the continued fraction from here on, by the series below
CONTEXT = Context(prec=PREC, rounding=ROUND_HALF_EVEN, Emin=-999999, Emax=999999)

MODELS = {   # the brms formulas this script implements -> (family, link); Case.row_term is the likelihood of each
    "yb ~ x + (1 | g)": ("bernoulli", "cloglog"),
    "k | trials(size) ~ x + (1 | g)": ("binomial", "cloglog"),
    "y | cens(cens_ind) ~ x + (1 | g)": ("gaussian", "identity"),
    "ypos | trunc(lb = 0) ~ x + (1 | g)": ("gaussian", "identity"),
    "ord ~ x + (1 | g)": ("cumulative", "probit"),
}
_NORMAL = re.compile(r"normal\(\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\)")


# -- 50-digit special functions ---------------------------------------------------------------------------------------
def _guarded():
    ctx = CONTEXT.copy()
    ctx.prec = PREC + GUARD
    return localcontext(ctx)


@lru_cache(maxsize=None)
def _pi() -> Decimal:
    """pi from Machin's formula 16 atan(1/5) - 4 atan(1/239), at PREC + 2 GUARD digits."""
    ctx = CONTEXT.copy()
    ctx.prec = PREC + 2 * GUARD
    with localcontext(ctx):
        eps = Decimal(10) ** -(ctx.prec + 2)

        def atan_inv(n: int) -> Decimal:
            power, total, k = Decimal(1) / n, Decimal(0), 0
            while power > eps:
                total += power / (2 * k + 1) * (-1 if k % 2 else 1)
                power /= n * n
                k += 1
            return total
        return 16 * atan_inv(5) - 4 * atan_inv(239)


def _erf_series(x: Decimal) -> Decimal:
    """erf(x) for 0 <= x < 3 in the caller's context: 2 / sqrt(pi) exp(-x^2) sum_n x (2 x^2)^n / (1 3 5 ... (2n + 1)), a sum
    of positive terms (no cancellation inside it)."""
    two_x2 = 2 * x * x
    term = total = x
    n = 0
    eps = Decimal(10) ** -(getcontext().prec + 2)
    while True:
        n += 1
        term = term * two_x2 / (2 * n + 1)
        total += term
        if n > two_x2 and term <= eps * total:
            break
    return 2 / _pi().sqrt() * (-(x * x)).exp() * total


def _erfc_cf(x: Decimal) -> Decimal:
    """erfc(x) for x >= 3 in the caller's context: exp(-x^2) / (sqrt(pi) K) with Laplace's continued fraction K = x + (1/2) /
    (x + (2/2) / (x + (3/2) / (x + ...))) (Abramowitz and Stegun 7.1.14), by the modified Lentz method; every partial
    numerator and denominator is positive."""
    eps = Decimal(10) ** -(getcontext().prec + 2)
    f = c = x
    d = Decimal(0)
    n = 0
    while True:
        n += 1
        a = Decimal(n) / 2
        d = 1 / (x + a * d)
        c = x + a / c
        delta = c * d
        f *= delta
        if abs(delta - 1) <= eps:
            break
    return (-(x * x)).exp() / (_pi().sqrt() * f)


def _erfc(x: Decimal) -> Decimal:
    """erfc(x) for every x, relative to its value, in the caller's (guarded) context."""
    if x >= CF_FROM:
        return _erfc_cf(x)
    if x >= 0:
        return 1 - _erf_series(x)              # erfc(3) = 2.2e-5: at most 5 of the guard digits are lost
    if x > -CF_FROM:
        return 1 + _erf_series(-x)
    return 2 - _erfc_cf(-x)


def log_cdf(t: Decimal) -> Decimal:
    """log Phi(t), the standard Normal log-CDF, as log(erfc(-t / sqrt 2) / 2)."""
    with _guarded():
        r = (_erfc(-t / Decimal(2).sqrt()) / 2).ln()
    with localcontext(CONTEXT):
        return +r


def log_ccdf(t: Decimal) -> Decimal:
    """log(1 - Phi(t)) as log(erfc(t / sqrt 2) / 2)."""
    with _guarded():
        r = (_erfc(t / Decimal(2).sqrt()) / 2).ln()
    with localcontext(CONTEXT):
        return +r


def log_normal_interval(a: Decimal | None, b: Decimal | None) -> Decimal:
    """log(Phi(b) - Phi(a)) for a < b, ``None`` standing for -inf (a) or inf (b): in the upper tail (a >= 0) as
    (erfc(a / sqrt 2) - erfc(b / sqrt 2)) / 2, in the lower tail (b <= 0) as (erfc(-b / sqrt 2) - erfc(-a / sqrt 2)) / 2,
    and across 0 as 1 - erfc(b / sqrt 2) / 2 - erfc(-a / sqrt 2) / 2, each a difference of terms of the size of the result
    or smaller; when the two terms of a difference agree in their leading digits, it is recomputed with as many further
    digits as they cancel."""
    if a is None and b is None:
        return Decimal(0)
    if a is None:
        return log_cdf(b)
    if b is None:
        return log_ccdf(a)
    if not a < b:
        raise ValueError(f"log_normal_interval needs a < b, got {a} and {b}")
    extra = 0
    while True:
        ctx = CONTEXT.copy()
        ctx.prec = PREC + GUARD + extra
        with localcontext(ctx):
            root2 = Decimal(2).sqrt()
            if a >= 0:
                hi, lo = _erfc(a / root2) / 2, _erfc(b / root2) / 2
            elif b <= 0:
                hi, lo = _erfc(-b / root2) / 2, _erfc(-a / root2) / 2
            else:
                hi, lo = 1 - _erfc(-a / root2) / 2, _erfc(b / root2) / 2
            diff = hi - lo
            lost = max(0, hi.adjusted() - diff.adjusted())
            if lost <= GUARD // 2 + extra:
                r = diff.ln()
                break
        extra = lost + GUARD // 2
    with localcontext(CONTEXT):
        return +r


def inverse_mills(t: Decimal) -> Decimal:
    """phi(t) / (1 - Phi(t)), the standard Normal density over its upper tail mass."""
    with _guarded():
        r = (-(t * t) / 2).exp() / (2 * _pi()).sqrt() / (_erfc(t / Decimal(2).sqrt()) / 2)
    with localcontext(CONTEXT):
        return +r


@lru_cache(maxsize=None)
def _half_log_two_pi() -> Decimal:
    with _guarded():
        r = (2 * _pi()).ln() / 2
    with localcontext(CONTEXT):
        return +r


def normal_lpdf(v: Decimal, m: Decimal, s: Decimal) -> Decimal:
    z = (v - m) / s
    return -s.ln() - _half_log_two_pi() - z * z / 2


def log1m_exp_neg(e: Decimal) -> Decimal:
    """log(1 - exp(-e)) for e > 0, with as many further digits as 1 - exp(-e) cancels (the cloglog link's log P(y = 1))."""
    ctx = CONTEXT.copy()
    ctx.prec = PREC + GUARD + max(0, -e.adjusted())
    with localcontext(ctx):
        r = (1 - (-e).exp()).ln()
    with localcontext(CONTEXT):
        return +r


# -- the case: data, priors, parameters ----------------------------------------------------------------------------
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(text: str) -> Decimal:
    """A CSV number as the double R and Python parse it to, exactly."""
    return Decimal(float(text))


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class Case:
    """The frozen fixture of ``case_id`` as this script needs it: the model, the rows (doubles, exactly), the priors and the
    unconstrained points, each checked against what the script implements. Build it, and use it, inside
    ``localcontext(CONTEXT)``."""

    def __init__(self, case_id: str):
        self.id = case_id
        self.dir = FIXTURES / case_id
        self.sources = {name: self.dir / name for name in ("case_brms.json", "data.csv", "newdata.csv", "fixture.json")
                        if (self.dir / name).exists()}
        brms = json.loads((self.dir / "case_brms.json").read_text(encoding="utf-8"))
        self.fixture = json.loads((self.dir / "fixture.json").read_text(encoding="utf-8"))
        self.formula = brms["formula"]
        if self.formula not in MODELS:
            raise SystemExit(f"{case_id}: no reference model for brms formula {self.formula!r}; this script implements {sorted(MODELS)}")
        family, link = MODELS[self.formula]
        if (brms["family"].get("family"), brms["family"].get("link", "identity")) != (family, link):
            raise SystemExit(f"{case_id}: the formula {self.formula!r} is implemented for {family}({link}), not {brms['family']}")
        self.family, self.link = family, link
        self.lb = Decimal(0) if "trunc(lb = 0)" in self.formula else None
        self.levels = list(brms["factors"]["g"])
        if [grp["levels"] for grp in self.fixture["groups"]] != [self.levels]:
            raise SystemExit(f"{case_id}: brms's group table {self.fixture['groups']} is not one varying intercept on g with the levels {self.levels}")
        self.priors = self._priors(brms["priors"])
        self.upars = [(e["par"], int(e["elem"]), e["transform"]) for e in self.fixture["upars"]]
        if family == "cumulative":     # the ordered thresholds are one transform of the whole vector, which the probe calls "joint"
            self.nthres = int(self.fixture["standata"]["nthres"]["values"][0])
            expected = [("b", "identity")] + [("Intercept", "joint")] * self.nthres + [("sd_1", "log")] \
                + [("z_1", "identity")] * len(self.levels)
            if [(par, transform) for par, _, transform in self.upars] != expected:
                raise SystemExit(f"{case_id}: brms's unconstrained coordinates are {self.upars}; this script implements {expected}")
        else:
            expected = [("b", 1, "identity"), ("Intercept", 1, "identity")] + ([("sigma", 1, "log")] if family == "gaussian" else []) \
                + [("sd_1", 1, "log")] + [("z_1", j, "identity") for j in range(1, len(self.levels) + 1)]
            if self.upars != expected:
                raise SystemExit(f"{case_id}: brms's unconstrained coordinates are {self.upars}; this script implements {expected}")
        fitted = read_csv(self.dir / "data.csv")
        response = {"bernoulli": "yb", "binomial": "k", "cumulative": "ord"}.get(family, "ypos" if self.lb is not None else "y")
        X, Y = self.fixture["standata"]["X"], self.fixture["standata"]["Y"]
        j, n = X["colnames"].index("x"), len(fitted)
        if [float(r["x"]) for r in fitted] != X["values"][j * n:(j + 1) * n] or [float(r[response]) for r in fitted] != Y["values"]:
            raise SystemExit(f"{case_id}: data.csv's x or {response} is not, double for double, brms's standata X or Y")
        self.x_mean = sum(number(r["x"]) for r in fitted) / len(fitted)
        self.rows = [self._row(r) for r in fitted]
        self.new_rows = [self._row(r, response=False) for r in read_csv(self.dir / "newdata.csv")] \
            if "newdata.csv" in self.sources else []
        self.points = [[Decimal(float(v)) for v in p["u"]["values"]] for p in self.fixture["points"]]
        for k, (u, p) in enumerate(zip(self.points, self.fixture["points"])):     # the transforms are brms's, value by value
            mine = self.constrain(u)[0]
            for name, theirs in p["pars"].items():
                for a, b in zip(mine.get(name, ()), theirs["values"]):
                    if abs(float(a) - b) > 1e-14 * max(1.0, abs(b)):
                        raise SystemExit(f"{case_id}: point {k}: {name} from u is {float(a)!r}, brms's constrained value {b!r}")

    def _priors(self, rows: list[dict]) -> dict[str, tuple[Decimal, Decimal]]:
        out = {}
        for p in rows:
            m = _NORMAL.fullmatch(p["prior"].strip())
            key = p["class"]
            if m is None or key in out or p.get("coef") or (key == "sd") != ("group" in p) or p.get("group", "g") != "g":
                raise SystemExit(f"{self.id}: prior {p} is not a normal(m, s) prior of one class this script implements")
            out[key] = (Decimal(m.group(1)), Decimal(m.group(2)))
        wanted = {"Intercept", "b", "sd"} | ({"sigma"} if self.family == "gaussian" else set())
        if set(out) != wanted:
            raise SystemExit(f"{self.id}: priors on {sorted(out)}; the model needs exactly {sorted(wanted)}")
        return out

    def _row(self, r: dict[str, str], response: bool = True) -> dict:
        row = {"xc": number(r["x"]) - self.x_mean, "level": self.levels.index(r["g"])}
        if not response:
            return row
        if self.formula.startswith("yb "):
            row["y"] = int(r["yb"])
        elif self.family == "cumulative":
            row["y"] = int(r["ord"])
            if not 1 <= row["y"] <= self.nthres + 1:
                raise SystemExit(f"{self.id}: category {row['y']} outside 1..{self.nthres + 1}")
        elif self.formula.startswith("k "):
            row["k"], row["n"] = int(r["k"]), int(r["size"])
            row["log_choose"] = Decimal(math.comb(row["n"], row["k"])).ln()
        elif self.lb is None:
            row["y"], row["cens"] = number(r["y"]), int(r["cens_ind"])
            if row["cens"] not in (-1, 0, 1):
                raise SystemExit(f"{self.id}: censoring code {row['cens']}")
        else:
            row["y"] = number(r["ypos"])
        return row

    # -- the target ------------------------------------------------------------------------------------------------
    def constrain(self, u: list[Decimal]) -> tuple[dict[str, list[Decimal]], Decimal]:
        """brms's constrained parameters at the unconstrained point ``u`` and the log-Jacobian of the transforms."""
        pars, jacobian = {}, Decimal(0)
        for (par, _, transform), v in zip(self.upars, u):
            if transform == "log":
                pars.setdefault(par, []).append(v.exp())
                jacobian += v
            elif transform == "joint" and par in pars:          # Stan's ordered: each threshold the previous plus exp(u)
                pars[par].append(pars[par][-1] + v.exp())
                jacobian += v
            else:
                pars.setdefault(par, []).append(v)
        return pars, jacobian

    def jacobian_slopes(self) -> list[int]:
        """d (log-Jacobian) / d u_i: 1 in each log-transformed coordinate and each threshold coordinate after the first."""
        out, seen = [], set()
        for par, _, transform in self.upars:
            out.append(1 if transform == "log" or (transform == "joint" and par in seen) else 0)
            seen.add(par)
        return out

    def eta(self, pars: dict, row: dict, groups: bool = True) -> Decimal:
        if self.family == "cumulative":                     # the thresholds take the intercept's place
            eta = pars["b"][0] * row["xc"]
        else:
            eta = pars["Intercept"][0] + pars["b"][0] * row["xc"]
        return eta + pars["sd_1"][0] * pars["z_1"][row["level"]] if groups else eta

    def row_term(self, pars: dict, row: dict) -> Decimal:
        eta = self.eta(pars, row)
        if self.family == "cumulative":                     # probit: Phi(c_y - eta) - Phi(c_{y-1} - eta)
            c, y = pars["Intercept"], row["y"]
            return log_normal_interval(c[y - 2] - eta if y > 1 else None, c[y - 1] - eta if y <= self.nthres else None)
        if self.link == "cloglog":
            e = eta.exp()
            if "k" in row:
                return row["log_choose"] + row["k"] * log1m_exp_neg(e) - (row["n"] - row["k"]) * e
            return log1m_exp_neg(e) if row["y"] == 1 else -e
        sigma = pars["sigma"][0]
        if self.lb is not None:
            return normal_lpdf(row["y"], eta, sigma) - log_ccdf((self.lb - eta) / sigma)
        if row["cens"] == 0:
            return normal_lpdf(row["y"], eta, sigma)
        t = (row["y"] - eta) / sigma
        return log_ccdf(t) if row["cens"] == 1 else log_cdf(t)

    def lp_nojac(self, u: list[Decimal]) -> Decimal:
        """brms's target without the Jacobian: the likelihood, lprior and std_normal(z_1)."""
        pars, _ = self.constrain(u)
        lp = Decimal(0)
        for name, cls in (("Intercept", "Intercept"), ("b", "b")):
            m, s = self.priors[cls]
            lp += sum(normal_lpdf(v, m, s) for v in pars[name])
        for name, cls in (("sd_1", "sd"), ("sigma", "sigma")):
            if name in pars:                       # <lower=0>: the normal truncated to (0, inf), brms's - normal_lccdf(0 | m, s)
                m, s = self.priors[cls]
                lp += sum(normal_lpdf(v, m, s) - log_ccdf((0 - m) / s) for v in pars[name])
        lp += sum(normal_lpdf(v, Decimal(0), Decimal(1)) for v in pars["z_1"])
        lp += sum(self.row_term(pars, row) for row in self.rows)
        return lp

    def gradient_nojac(self, u: list[Decimal]) -> list[Decimal]:
        """d lp_nojac / d u_i by the fourth-order central difference (f(-2h) - 8 f(-h) + 8 f(h) - f(2h)) / (12 h)."""
        out = []
        for i in range(len(u)):
            def at(step: Decimal) -> Decimal:
                v = list(u)
                v[i] = u[i] + step
                return self.lp_nojac(v)
            out.append((at(-2 * STEP) - 8 * at(-STEP) + 8 * at(STEP) - at(2 * STEP)) / (12 * STEP))
        return out

    def truncated_mean(self, pars: dict, row: dict, groups: bool) -> Decimal:
        """brms's posterior_epred of the truncated Gaussian: eta + sigma phi(a) / (1 - Phi(a)), a = (lb - eta) / sigma."""
        eta, sigma = self.eta(pars, row, groups), pars["sigma"][0]
        return eta + sigma * inverse_mills((self.lb - eta) / sigma)


# -- the file ----------------------------------------------------------------------------------------------------------
def text(v: Decimal) -> str:
    return format(v, f".{DIGITS - 1}E")


def build(case_id: str) -> str:
    """The reference file of ``case_id`` as text."""
    with localcontext(CONTEXT):
        case = Case(case_id)
        points = []
        for u in case.points:
            _, jacobian = case.constrain(u)
            lp_nojac = case.lp_nojac(u)
            grad_nojac = case.gradient_nojac(u)
            grad = [g + slope for g, slope in zip(grad_nojac, case.jacobian_slopes())]
            points.append({"lp": text(lp_nojac + jacobian), "lp_nojac": text(lp_nojac), "grad": [text(g) for g in grad],
                           "grad_nojac": [text(g) for g in grad_nojac]})
        out = {
            "case": case_id,
            "generator": "oracle/reference.py",
            "model": f"{case.formula}, brms {case.family}({case.link}); priors "
                     + ", ".join(f"{cls} normal({m}, {s})" for cls, (m, s) in case.priors.items()) + ", z_1 std_normal",
            "arithmetic": f"decimal, {PREC} significant digits ({GUARD} guard digits in erf, erfc and the constants); values "
                          f"written with {DIGITS}; the gradient without the Jacobian is a fourth-order central difference of the "
                          f"{PREC}-digit density with step {STEP}, with it plus 1 in each log-transformed coordinate",
            "sources": {str(path.relative_to(REPO)): sha256(path) for path in case.sources.values()},
            "upars": [f"{par}[{elem}]" for par, elem, _ in case.upars],
            "points": points,
        }
        if case.lb is not None:                         # G4's expected values: posterior_epred, the truncated mean
            pars = [case.constrain(u)[0] for u in case.points]
            out["predictions"] = {
                block: {kind: [[text(case.truncated_mean(p, row, groups)) for row in rows] for p in pars]
                        for kind, groups in (("epred", True), ("epred_nogroup", False))}
                for block, rows in (("fitted", case.rows), ("newdata", case.new_rows)) if rows}
    return json.dumps(out, indent=1, ensure_ascii=False) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--check", action="store_true", help="recompute and compare with the committed file byte for byte")
    ap.add_argument("--all", action="store_true", help="with --check: every committed reference")
    args = ap.parse_args(argv)
    ids = list(args.ids)
    if args.all:
        if not args.check:
            ap.error("--all goes with --check")
        ids = sorted(p.stem for p in REFERENCES.glob("*.json"))
    if not ids:
        ap.error("give case ids (or --check --all)")
    failed = False
    for cid in ids:
        path = REFERENCES / f"{cid}.json"
        if args.check:
            if not path.exists():
                print(f"{cid}: no committed reference")
                failed = True
                continue
            same = build(cid) == path.read_text(encoding="utf-8")
            print(f"{cid}: {'identical' if same else 'DIFFERS from the committed file'}")
            failed = failed or not same
        elif path.exists():
            print(f"{cid}: already written (a committed reference is never overwritten; --check compares)")
        else:
            REFERENCES.mkdir(parents=True, exist_ok=True)
            path.write_text(build(cid), encoding="utf-8")
            print(f"{cid}: written to {path.relative_to(REPO)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
