"""Cases and frozen fixtures.

A case is `tests/oracle/cases/<id>.json` with a `brms` section (formula, family, factor levels, priors,
optional `bf_args`/`brm_args`), a `data` section (generator and seeds) and an `autobayes` section (the call
that is expected to build the same model). `oracle/regen.py new <id>` freezes the brms side into
`tests/oracle/fixtures/<id>/`: `case_brms.json` (the canonical brms section, which `gen_fixture.R` reads),
`data.csv`, `newdata.csv` and `fixture.json`. Fixtures are immutable once committed; a changed case gets a
new id. Arrays in `fixture.json` are `{"dims": [...], "values": [...]}` with column-major values.

Tier C cases may add opt-in keys to the brms section (documented in `oracle/R/gen_fixture.R`): `na_columns`
(columns with missing values, read here as NaN), `mvbf` and `rescor` (multivariate models), a `family` with a
`mixture` list, `point_halfwidth` (the box the seeded points are drawn from), and `extras`, which add
`groups[*].resp/dpar/nlpar`, a top-level `responses` list (the last axis of a multivariate model's predictions)
and a `post` block (brms's `log_lik`, `bayes_R2` and `loo` of the fixed-parameter fit) to `fixture.json`.

Tier D cases may add to that `post` block: `extras` "waic" (brms's `waic`), `loo_compare` (another case's frozen brms
section on the same data: brms's `loo_compare` of the two fits by loo and by waic, both models' pointwise elpd and
`model_weights`), `hypothesis` (a list of hypotheses: brms's table and draws) and `conditional_effects` (a list of
effects: one table each). A table is `{"columns": [...], "values": {column: array or list of strings}}`; an array entry
"NaN" is brms's NA (a point hypothesis's evidence ratio without prior draws), and a missing text value is `null` (the
grouping factor's column of a conditional_effects table, which leaves out the group-level effects).

Tier E cases may add: `prior_draws` (with `brm_args` sample_prior "yes"; brms's `prior_draws()` of the fixed-parameter fit as
`post.prior_draws`, (draws, variables) with brms's names; the seeded points' `pars` then leave out the prior_* generated quantities,
which Stan draws anew on every constrain_pars() call; a point hypothesis of such a fit has brms's Savage-Dickey evidence ratio),
`hypothesis_seed` (hypothesis()'s seed, which fixes the last bit of a Savage-Dickey ratio), `dpar_epred` (brms's
`posterior_epred(dpar = )` of distributional parameters the points do not carry, as `post.dpar_epred`, (draws, rows) each),
`extras` "loo_model_weights" (with `loo_compare`: brms's stacking weights of the two fits as `post.loo_model_weights`) and
interaction effects ("x:z") in `conditional_effects`, whose tables have each variable of the effect and `effect2__`.

Tier F cases may add `data2` (a list of names) to the brms section, with the matrix each name stands for in the data section
(``"data2": {name: kind}``, a kind of `datasets.DATA2`): `oracle/regen.py` writes each as `data2_<name>.csv` next to `data.csv`,
`gen_fixture.R` hands them to brms as ``data2 = list(<name> = <matrix>)`` (car(W), sar(M), fcor(M)) and the fixture's meta records
their sha256 (`data2_sha256`); `read_data2` reads them back, labelled, and `gates.Bundle` hands them to the case's constructor as
``data2=``. A matrix response (brms's multinomial, dirichlet, dirichlet_multinomial and logistic_normal) is written in the formula
as ``cbind(m1, m2, m3)`` over ordinary columns, which brms evaluates into the same matrix column it would take from the data.
`prediction_refused` (a list of kinds, "linpred" or "epred") records a prediction brms fails to compute for the model as its error
message, `predictions.<block>.<kind>_refused` (and `<kind>_nogroup_refused`), in place of the array (`Fixture.prediction_refusal`).
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
    values = np.asarray(obj["values"], dtype=float)         # strings "Inf"/"-Inf" (truncation bounds) parse as infinities
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
    """The committed CSV, factor columns as strings (a label like "1" must stay a label). A case that declares
    columns with missing values (``brms.na_columns``, for mi()) reads an empty field of those columns as NaN and
    nothing else as missing, as ``gen_fixture.R`` reads them as NA."""
    factors = list(case["brms"].get("factors", {}))
    na_columns = list(case["brms"].get("na_columns", []))
    if not na_columns:
        return pd.read_csv(path, dtype={c: str for c in factors})
    return pd.read_csv(path, dtype={c: str for c in factors}, keep_default_na=False, na_values={c: [""] for c in na_columns})


def read_matrix(path) -> pd.DataFrame:
    """A data2 matrix as `datasets.write_matrix` writes it: labels as text, every entry a double."""
    frame = pd.read_csv(path, index_col=0, dtype=str, keep_default_na=False)
    frame.index.name = None
    return frame.astype(float)


def data2_path(directory, name: str) -> Path:
    return Path(directory) / f"data2_{name}.csv"


def read_data2(case: dict, directory) -> dict[str, pd.DataFrame]:
    """The matrices a Tier F case hands brms as ``data2`` (its brms section's ``data2`` names), each as a frame labelled as brms's
    matrix is (car()'s W by site), read from ``directory``; {} for a case without."""
    return {name: read_matrix(data2_path(directory, name)) for name in case["brms"].get("data2", [])}


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

    def prediction_refusal(self, block: str, kind: str) -> str | None:
        """brms's error message for a prediction it fails to compute (Tier F, a case's ``prediction_refused``), else None."""
        return self.raw["predictions"].get(block, {}).get(f"{kind}_refused")

    @property
    def prior_summary(self) -> list[dict]:
        return self.raw["prior_summary"]

    @property
    def default_prior(self) -> list[dict]:
        return self.raw["default_prior"]

    @property
    def declarations(self) -> dict[str, str]:
        return self.raw["declarations"]
