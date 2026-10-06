"""Cases and frozen fixtures.

A case is `tests/oracle/cases/<id>.json` with a `brms` section (formula, family, factor levels, priors,
optional `bf_args`/`brm_args`), a `data` section (generator and seeds) and an `autobayes` section (the call
that is expected to build the same model). `oracle/regen.py new <id>` freezes the brms side into
`tests/oracle/fixtures/<id>/`: `case_brms.json` (the canonical brms section, which `gen_fixture.R` reads),
`data.csv`, `newdata.csv` and `fixture.json`. Fixtures are immutable once committed; a changed case gets a
new id. Arrays in `fixture.json` are `{"dims": [...], "values": [...]}` with column-major values.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CASES = ROOT / "cases"
FIXTURES = ROOT / "fixtures"


def decode(obj) -> np.ndarray:
    """A dims/values array from the fixture as a NumPy array (R stores column-major)."""
    dims = [int(d) for d in obj["dims"]]
    values = np.asarray(obj["values"], dtype=float)
    return values.reshape(dims, order="F") if dims else values.reshape(())


def case_ids() -> list[str]:
    return sorted(p.stem for p in CASES.glob("*.json"))


def load_case(case_id: str) -> dict:
    with open(CASES / f"{case_id}.json", encoding="utf-8") as f:
        case = json.load(f)
    if case.get("id") != case_id:
        raise ValueError(f"{case_id}.json declares id {case.get('id')!r}")
    return case


def frozen_section(case: dict) -> dict:
    """What `gen_fixture.R` sees: the brms section plus the id, the point seed and the number of points."""
    return {"id": case["id"], "seed": int(case["seed"]), "n_points": int(case.get("n_points", 8)), **case["brms"]}


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=False) + "\n"


def fixture_dir(case_id: str) -> Path:
    return FIXTURES / case_id


def has_fixture(case_id: str) -> bool:
    return (fixture_dir(case_id) / "fixture.json").exists()


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_frame(case: dict, path) -> pd.DataFrame:
    """The committed CSV, factor columns as strings (a label like "1" must stay a label)."""
    factors = list(case["brms"].get("factors", {}))
    return pd.read_csv(path, dtype={c: str for c in factors})


@dataclass(frozen=True)
class Point:
    u: np.ndarray
    pars: dict[str, np.ndarray]
    lp: float
    grad: np.ndarray
    lp_nojac: float
    grad_nojac: np.ndarray


class Fixture:
    def __init__(self, raw: dict):
        self.raw = raw

    @classmethod
    def load(cls, case_id: str) -> "Fixture":
        with open(fixture_dir(case_id) / "fixture.json", encoding="utf-8") as f:
            return cls(json.load(f))

    @property
    def meta(self) -> dict:
        return self.raw["meta"]

    @cached_property
    def standata(self) -> dict[str, np.ndarray]:
        return {k: decode(v) for k, v in self.raw["standata"].items()}

    def colnames(self, name: str) -> list[str] | None:
        return self.raw["standata"][name].get("colnames")

    @cached_property
    def points(self) -> list[Point]:
        return [Point(u=decode(p["u"]), pars={k: decode(v) for k, v in p["pars"].items()}, lp=float(p["lp"]),
                      grad=decode(p["grad"]), lp_nojac=float(p["lp_nojac"]), grad_nojac=decode(p["grad_nojac"]))
                for p in self.raw["points"]]

    @property
    def upars(self) -> list[dict]:
        """One entry per unconstrained coordinate: index (1-based), par, elem (1-based, column-major), transform."""
        return self.raw["upars"]

    @property
    def groups(self) -> list[dict]:
        return self.raw["groups"]

    def prediction(self, block: str, kind: str) -> np.ndarray | None:
        """(n_points, rows) matrix for block "fitted" or "newdata" and kind "linpred", "epred",
        "linpred_nogroup" or "epred_nogroup"; None when the fixture has no such entry."""
        entry = self.raw["predictions"].get(block, {}).get(kind)
        return None if entry is None else decode(entry)

    @property
    def prior_summary(self) -> list[dict]:
        return self.raw["prior_summary"]

    @property
    def default_prior(self) -> list[dict]:
        return self.raw["default_prior"]

    @property
    def declarations(self) -> dict[str, str]:
        return self.raw["declarations"]
