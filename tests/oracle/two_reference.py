"""Two-reference acceptance (P6; docs/brms-parity/LOOP.md, "Two references").

A case whose brms side is inexact at some frozen points opts in through its gate field, for example
``{"expect": "pass", "reference": "gauss_censored.json", "divergence": "D24", "fallback": ["gradient"]}``: ``reference`` is
a committed high-precision reference in ``tests/oracle/references/`` (``oracle/reference.py``, 50 digits), ``divergence``
the row of docs/brms-parity/divergences.md that says why brms is inexact there, and ``fallback`` the quantity kinds that
may use the reference: ``"gradient"`` (every gradient coordinate, with and without the Jacobian, of gate G2) and/or
``"epred"`` (gate G4's expected values). Log densities (``lp``, ``lp_nojac``, ``lprior``) never fall back. For each frozen
point and each entry of a listed kind, with ``tol`` the usual tolerance (``tolerances.py``, unchanged):

1. the entry passes if AutoBayes agrees with brms within ``tol``;
2. otherwise it passes only if brms differs from the reference by more than ``tol`` and AutoBayes agrees with the
   reference within ``tol``;
3. wherever AutoBayes and brms agree within ``tol``, the reference must agree with brms within ``tol`` too; otherwise the
   reference is wrong and the gate fails, whatever AutoBayes does.

Every other gate (G1, G3, G5, G6, the log densities of G2, and G4's linear predictor and, unless listed, its expected
values) is unchanged and compares with brms only: ``gates.check_gradient`` and ``gates.check_predictions`` hand an
opted-in case to ``check_gradient`` and ``check_predictions`` here and keep their own path for every other case. These
return a ``FallbackReport`` (how many entries passed on the reference, and where) and emit it as a ``ReferenceFallback``
warning, which pytest's summary shows.
"""
from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from tests.oracle import adapter
from tests.oracle import fixture as fx
from tests.oracle.adapter import OracleMismatch
from tests.oracle.tolerances import GRAD_REL, PRED_REL, rel_err

REFERENCES = fx.ROOT / "references"
REPO = fx.ROOT.parents[1]
FALLBACK_KINDS = ("gradient", "epred")       # G2's gradient coordinates (with and without the Jacobian); G4's expected values
LOG_DENSITIES = ("lp", "lp_nojac", "lprior")  # never fall back


class ReferenceFallback(Warning):
    """How many entries of an opted-in case passed on its reference rather than on brms, and where. A ``Warning`` and not a
    ``UserWarning``, which the suite's pytest configuration ignores, so that the summary of every run shows it."""


@dataclass(frozen=True)
class FallbackReport:
    case: str
    gate: str                     # "G2 gradient" or "G4 epred"
    entries: int                  # the entries of the listed kind compared
    where: tuple[str, ...]        # each entry that passed on the reference (rule 2), not on brms

    @property
    def count(self) -> int:
        return len(self.where)

    def __str__(self) -> str:
        text = (f"{self.case} {self.gate}: {self.count} of {self.entries} entries passed on the reference, where brms is off "
                f"it by more than the tolerance")
        return text + (": " + "; ".join(_grouped(self.where)) if self.where else "")


def _grouped(where: tuple[str, ...]) -> list[str]:
    """The entries, a gradient coordinate whose value with and without the Jacobian both fell back written once."""
    pair = (" (with the Jacobian)", " (without the Jacobian)")
    out = []
    for w in where:
        for this, other in (pair, pair[::-1]):
            if w.endswith(this) and w[: -len(this)] + other in where:
                if this == pair[0]:
                    out.append(w[: -len(this)] + " (with and without the Jacobian)")
                break
        else:
            out.append(w)
    return out


class Reference:
    """A committed reference file (``oracle/reference.py``) of a case that opts in, with its opt-in."""

    def __init__(self, case_id: str, file: str, divergence: str, fallback: frozenset[str], raw: dict):
        self.case_id, self.file, self.divergence, self.fallback, self.raw = case_id, file, divergence, fallback, raw

    @property
    def upars(self) -> list[str]:
        return list(self.raw["upars"])

    @cached_property
    def points(self) -> list[dict]:
        """Per frozen point: ``lp``, ``lp_nojac`` and the gradient ``grad`` / ``grad_nojac`` in brms's unconstrained order,
        each decimal string read as the nearest double."""
        return [{"lp": float(p["lp"]), "lp_nojac": float(p["lp_nojac"]),
                 "grad": np.array([float(v) for v in p["grad"]]), "grad_nojac": np.array([float(v) for v in p["grad_nojac"]])}
                for p in self.raw["points"]]

    def prediction(self, block: str, kind: str) -> np.ndarray | None:
        """(points, rows) expected values of ``block`` ("fitted", "newdata") and ``kind`` ("epred", "epred_nogroup")."""
        entry = self.raw.get("predictions", {}).get(block, {}).get(kind)
        return None if entry is None else np.asarray([[float(v) for v in row] for row in entry], dtype=float)


def load(case: dict) -> Reference | None:
    """The reference of a case that opts in to two-reference acceptance, with the opt-in validated; None for every other
    case (no file is read)."""
    gate = case.get("gate") or {}
    if "reference" not in gate:
        return None
    cid = case["id"]
    if gate.get("expect") != "pass":
        raise OracleMismatch(f"{cid}: a gate field that names a reference expects 'pass', not {gate.get('expect')!r}")
    fallback = gate.get("fallback")
    if not isinstance(fallback, list) or not fallback or not all(isinstance(k, str) for k in fallback):
        raise OracleMismatch(f"{cid}: the gate's fallback must be a non-empty list of quantity kinds from {list(FALLBACK_KINDS)}, "
                             f"got {fallback!r}")
    if set(fallback) & set(LOG_DENSITIES):
        raise OracleMismatch(f"{cid}: log densities never fall back on a reference ({sorted(set(fallback) & set(LOG_DENSITIES))} "
                             f"in the gate's fallback): G2's log densities compare with brms only")
    if not set(fallback) <= set(FALLBACK_KINDS):
        raise OracleMismatch(f"{cid}: the gate's fallback {sorted(set(fallback) - set(FALLBACK_KINDS))} is not a quantity kind "
                             f"that may fall back; those are {list(FALLBACK_KINDS)}")
    divergence = gate.get("divergence")
    if not isinstance(divergence, str) or not re.fullmatch(r"D\d+", divergence):
        raise OracleMismatch(f"{cid}: the gate names no divergence (docs/brms-parity/divergences.md) for why brms is inexact")
    path = REFERENCES / str(gate["reference"])
    if path.parent != REFERENCES or not path.is_file():
        raise OracleMismatch(f"{cid}: no reference file {gate['reference']!r} in {REFERENCES.relative_to(REPO)}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("case") != cid:
        raise OracleMismatch(f"{cid}: {path.name} is the reference of {raw.get('case')!r}")
    stale = [name for name, digest in raw["sources"].items() if not (REPO / name).is_file() or fx.sha256(REPO / name) != digest]
    if stale:
        raise OracleMismatch(f"{cid}: {path.name} was computed from other versions of {stale} (their sha256 differ)")
    return Reference(cid, path.name, divergence, frozenset(fallback), raw)


def entry(a: float, brms: float, ref: float, tol: float, what: str) -> bool:
    """One entry under the three rules: False if it passes on brms (rule 1), True if it passes on the reference (rule 2);
    raises if it fails, or if AutoBayes and brms agree and the reference does not (rule 3)."""
    ab, br, ar = rel_err(a, brms), rel_err(brms, ref), rel_err(a, ref)
    if ab <= tol:
        if br > tol:
            raise OracleMismatch(f"{what}: the reference is wrong: AutoBayes {a!r} and brms {brms!r} agree ({ab:.2e}) but the "
                                 f"reference {ref!r} is {br:.2e} off brms (tolerance {tol:g})")
        return False
    if br > tol and ar <= tol:
        return True
    if br > tol:
        raise OracleMismatch(f"{what}: AutoBayes {a!r} is off both brms {brms!r} ({ab:.2e}) and the reference {ref!r} "
                             f"({ar:.2e}) (tolerance {tol:g})")
    raise OracleMismatch(f"{what}: AutoBayes {a!r} is {ab:.2e} off brms {brms!r}, which agrees with the reference {ref!r} "
                         f"({br:.2e}; tolerance {tol:g}), so nothing falls back")


def _report(b, gate: str, entries: int, where: list[str]) -> FallbackReport:
    report = FallbackReport(b.case["id"], gate, entries, tuple(where))
    warnings.warn(ReferenceFallback(str(report)), stacklevel=3)
    return report


def check_gradient(b) -> FallbackReport:
    """Gate G2's gradient under the three rules: every coordinate at every point, with and without the Jacobian, mapped
    through the probed parameter table as ``gates.check_gradient`` maps it."""
    ref = b.reference
    if ref.upars != [f"{e['par']}[{e['elem']}]" for e in b.fixture.upars] or len(ref.points) != len(b.fixture.points):
        raise OracleMismatch(f"{ref.case_id}: the reference's points and coordinates {ref.upars} are not the fixture's")
    coords = adapter.coordinate_map(b.fixture, b.model, b.cmap, b.gmap)
    labels = {i: name for i, name in enumerate(ref.upars)}
    where, entries = [], 0
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
            for jacobian, a, theirs, high in ((True, ga[pos], p.grad[i], ref.points[k]["grad"][i]),
                                              (False, gan[pos], p.grad_nojac[i], ref.points[k]["grad_nojac"][i])):
                entries += 1
                what = (f"point {k}: gradient of {var}[{idx}] (brms {labels[i]}) {'with' if jacobian else 'without'} "
                        f"the Jacobian")
                if entry(float(a), float(theirs), float(high), GRAD_REL, what):
                    where.append(f"point {k} {labels[i]} ({'with' if jacobian else 'without'} the Jacobian)")
        if not hit.all():
            missing = [v for v, s in layout.items() if not hit[s].all()]
            raise OracleMismatch(f"AutoBayes coordinates without a brms counterpart: {missing}")
    return _report(b, "G2 gradient", entries, where)


def check_predictions(b) -> FallbackReport:
    """Gate G4 with its expected values under the three rules; its linear predictor compares with brms only, as
    ``gates.check_predictions`` does."""
    ref = b.reference
    idata = adapter.draws(b.fixture, b.model, b.cmap, b.gmap)
    mean_var = "expected" if b.model.family == "binomial" else b.model.names["mean"]
    where, entries = [], 0
    for block, nd in (("fitted", None), ("newdata", b.newdata)):
        if block == "newdata" and nd is None:
            continue
        for include, suffix in ((True, ""), (False, "_nogroup")):
            ref_lin = b.fixture.prediction(block, "linpred" + suffix)
            ref_ep = b.fixture.prediction(block, "epred" + suffix)
            if ref_lin is None or ref_ep is None:
                if not include and b.model.spec.groups:
                    raise OracleMismatch(f"fixture lacks the {block} predictions without group effects")
                continue
            lin = b.model.predict(idata, newdata=nd, kind="linear", include_group_effects=include)["eta"].to_numpy()[0]
            err = rel_err(lin, ref_lin)
            if err > PRED_REL:
                raise OracleMismatch(f"{block}{suffix}: linear predictor differs from brms posterior_linpred by {err}")
            mean = b.model.predict(idata, newdata=nd, kind="mean", include_group_effects=include)[mean_var].to_numpy()[0]
            high = ref.prediction(block, "epred" + suffix)
            if high is None or high.shape != ref_ep.shape or mean.shape != ref_ep.shape:
                raise OracleMismatch(f"{ref.case_id}: brms's {block} epred{suffix} is {ref_ep.shape}, AutoBayes's {mean.shape}, "
                                     f"the reference's {None if high is None else high.shape}")
            for (k, row), a in np.ndenumerate(mean):
                entries += 1
                if entry(float(a), float(ref_ep[k, row]), float(high[k, row]), PRED_REL,
                         f"{block}{suffix}: point {k}, row {row}: {mean_var!r} vs brms posterior_epred"):
                    where.append(f"{block} epred{suffix} point {k} row {row}")
    return _report(b, "G4 epred", entries, where)
