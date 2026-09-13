"""Stubbed output-assembly tests for docs/m3b1-evidence/scripts/f7_marginal_reference.py (second budget-guard
review, 2026-09-13): the driver module is imported without running it; no Gaussian reference, optimisation or
integration is invoked; stage results are stubs."""
import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
DRIVER = ROOT / "docs" / "m3b1-evidence" / "scripts" / "f7_marginal_reference.py"


@pytest.fixture(scope="module")
def drv():
    spec = importlib.util.spec_from_file_location("f7_marginal_reference_driver", DRIVER)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)          # import only: main() is guarded
    return mod


def records():
    return json.load(open(ROOT / "docs" / "m3b1-evidence" / "F7_varying_slope_recovery_records.json"))


def stub_marginal(center_theta, width=0.02):
    grid = np.linspace(center_theta - 1.0, center_theta + 1.0, 141)
    dens = np.exp(-0.5 * ((grid - center_theta) / width) ** 2); dens /= np.trapezoid(dens, grid)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(grid))]); cdf /= cdf[-1]
    return {"theta_grid": grid.tolist(), "density_theta": dens.tolist(), "cdf": cdf.tolist(), "mean": float(np.trapezoid(np.exp(grid) * dens, grid)), "log_mass_minus_logZ": 0.0}


def complete_stages(drv, face_lo_sd1=-30.0):
    centers = {"1|g_sigma": np.log(0.355), "x|g_sigma": np.log(0.1758), "sigma": np.log(0.4245)}
    margs = {k: stub_marginal(c) for k, c in centers.items()}
    means = [margs[k]["mean"] for k in centers]                    # the 3-D means agree with the stub marginals by construction
    stages = {"mode": {"theta_star": list(centers.values()), "ell_star": -407.0, "scales": [0.14, 0.18, 0.03], "lowrank_minus_dense": -1e-12},
              "faces": {"1|g_sigma:lo": -63.6, "1|g_sigma:hi": -27.1, "x|g_sigma:lo": face_lo_sd1, "x|g_sigma:hi": -55.6, "sigma:lo": -37.9, "sigma:hi": -27.4},
              "box8": [[0, 1]] * 3, "box10": [[0, 1]] * 3, "expansions": [0, 0, 0],
              "levels": [{"n": 48, "logZ": -411.3857019, "means": means, "evaluations": 48 ** 3}, {"n": 64, "logZ": -411.3857019, "means": means, "evaluations": 64 ** 3}],
              "expanded_B10_n64": {"n": 64, "logZ": -411.3857019, "means": means, "evaluations": 64 ** 3},
              "marginals": margs,
              "cdf_variants": {f"{k}:{v}": stub_marginal(c) for k, c in centers.items() for v in drv.CDF_VARIANTS}}
    return stages


def ledger_view(**kw):
    base = dict(reason="complete", requested=1500, reserved_uncertain=0, confirmed=1500, remaining=500, seconds_spent=3.0, cap=2000, unit_deadline_seconds=900.0)
    base.update(kw); return base


def test_inventory_lists_six_faces_and_every_cdf_point_twice(drv):
    inv = drv.condition_inventory(drv.reference_points(records()))
    assert [n for n in inv if n.startswith("face:")] == [f"face:{f}" for f in drv.FACES] and len(drv.FACES) == 6
    cdf = [n for n in inv if n.startswith("cdf_stability:")]
    assert len(cdf) == 7 * 2 and "cdf_stability:x|g_sigma@population_0.30:resolution_B8_48" in cdf and "cdf_stability:sigma@mcmc_mean_numpyro:domain_B10_40" in cdf
    assert len(inv) == 6 + 3 + 3 + 2 + 3 + 14 == len(set(inv))


def test_complete_stub_run_is_accepted_and_compared(drv, tmp_path):
    out = drv.assemble_output(complete_stages(drv), ledger_view(), records(), "stub", {"evaluations": 2000, "seconds": 900})
    assert out["acceptance"]["accepted"] and out["status"] == "complete" and out["exit_code"] == 0 and set(out["conditions"]) == set(out["required_conditions"])
    assert out["acceptance"]["run_valid"] and out["acceptance"]["protocol_accepted"]
    assert all(v is True for v in out["conditions"].values()) and out["budget"]["confirmed"] == 1500 and out["budget"]["reserved_uncertain"] == 0
    assert set(out["comparison"]) == {"pymc", "numpyro"} and all(c["within"] for be in out["comparison"].values() for c in be.values())
    drv.write_summary(out, tmp_path / "s.txt"); assert "confirmed 1500" in (tmp_path / "s.txt").read_text()


def test_unmet_face_blocks_acceptance_even_when_everything_else_is_met(drv):
    out = drv.assemble_output(complete_stages(drv, face_lo_sd1=-20.04), ledger_view(), records(), "stub", {})
    assert not out["acceptance"]["accepted"] and out["acceptance"]["failed"] == ["face:x|g_sigma:lo"] and out["comparison"] == {}
    assert out["status"].startswith("incomplete") and out["conditions"]["face:x|g_sigma:lo"] is False and out["exit_code"] == 4


def test_stopped_or_over_budget_unit_is_never_accepted_even_with_all_conditions_met(drv):
    """The third review's reproduction: complete stub stages plus stopped='unit deadline exceeded' and 6 s against 5."""
    stages = complete_stages(drv); stages["stopped"] = "unit time allowance 5 s exceeded by 1.0 s after the last batch"
    out = drv.assemble_output(stages, ledger_view(reason="stopped", seconds_spent=6.0, unit_deadline_seconds=5.0), records(), "stub", {"evaluations": 2000, "seconds": 5})
    a = out["acceptance"]
    assert a["protocol_accepted"] and not a["run_valid"] and not a["accepted"] and out["exit_code"] == 3 and out["comparison"] == {}
    assert any("run stopped" in p for p in a["run_problems"]) and any("exceeds allowance" in p for p in a["run_problems"]) and out["status"].startswith("incomplete")
    over = drv.assemble_output(complete_stages(drv), ledger_view(confirmed=2001, remaining=0), records(), "stub", {})
    assert not over["acceptance"]["accepted"] and over["exit_code"] == 3 and "exceeds cap" in over["acceptance"]["run_problems"][0]
    unconf = drv.assemble_output(complete_stages(drv), ledger_view(reserved_uncertain=2), records(), "stub", {})
    assert not unconf["acceptance"]["accepted"] and unconf["exit_code"] == 3
    exact = drv.assemble_output(complete_stages(drv), ledger_view(confirmed=2000, requested=2000, remaining=0, seconds_spent=900.0), records(), "stub", {})
    assert exact["acceptance"]["accepted"] and exact["exit_code"] == 0                # a valid exact-cap, exact-allowance completion passes
    assert drv.exit_code({"accepted": False, "run_valid": True}) == 4 and drv.exit_code({"accepted": False, "run_valid": False}) == 3


def test_stopped_after_faces_reports_everything_else_unchecked(drv):
    stages = {k: v for k, v in complete_stages(drv, face_lo_sd1=-20.04).items() if k in ("mode", "faces", "box8", "box10", "expansions")}
    stages["stopped"] = "cap 2000000 exhausted: ..."
    out = drv.assemble_output(stages, ledger_view(reason="stopped", reserved_uncertain=40, confirmed=1960, remaining=0), records(), "stub", {})
    a = out["acceptance"]
    assert not a["accepted"] and a["failed"] == ["face:x|g_sigma:lo"] and len(a["unchecked"]) == 31 - 6 and a["omitted"] == [] and out["exit_code"] == 3
    assert all(f"face:{f}" in out["conditions"] for f in drv.FACES) and out["stopped"].startswith("cap") and out["intervals"] == {} and out["comparison"] == {}
    assert out["budget"]["reserved_uncertain"] == 40 and out["budget"]["confirmed"] == 1960


def test_exhausted_during_cdf_variants_leaves_those_points_unchecked(drv):
    stages = complete_stages(drv)
    stages["cdf_variants"] = {k: v for k, v in stages["cdf_variants"].items() if not k.endswith("domain_B10_40")}   # domain variant never computed
    stages["stopped"] = "cap exhausted during cdf variants"
    out = drv.assemble_output(stages, ledger_view(reason="stopped"), records(), "stub", {})
    a = out["acceptance"]
    assert not a["accepted"] and a["failed"] == [] and a["unchecked"] == sorted(n for n in out["required_conditions"] if n.endswith("domain_B10_40")) and len(a["unchecked"]) == 7
    assert out["comparison"] == {} and out["intervals"] and all(n in out["conditions"] for n in out["required_conditions"])


def test_driver_source_has_no_attempt_shadowing_and_guards_main():
    src = DRIVER.read_text(); tree = ast.parse(src)
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "attempt"]
    assert 'if __name__ == "__main__"' in src and "rid = evaluator.charge(2" in src and "fail_charge(rid" in src   # diagnostic calls gated, confirmed or failed
