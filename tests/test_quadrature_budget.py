"""Controller tests for the evaluation-budget rules (docs/m3b1-decision.md, cap correction 2026-09-13). The
evaluator is a fake that records batch sizes; no posterior density is evaluated anywhere in this module."""
import numpy as np
import pytest

from tests.quadrature_budget import BudgetExhausted, BudgetedEvaluator, acceptance_report


class FakeEvaluator:
    def __init__(self):
        self.batches = []

    def __call__(self, pts):
        self.batches.append(len(pts))
        return np.zeros(len(pts))


def test_batches_never_exceed_the_remaining_budget_and_counts_are_distinct(tmp_path):
    fake = FakeEvaluator(); ledger = tmp_path / "ledger.json"
    ev = BudgetedEvaluator(fake, cap_evaluations=1000, deadline_seconds=60, ledger_path=ledger, attempt="a1", batch_size=300)
    out = ev(np.zeros((650, 3)))
    assert out.shape == (650,) and fake.batches == [300, 300, 50]
    assert ev.completed_total == 650 and ev.attempted_total == 650 and ev.remaining == 350
    with pytest.raises(BudgetExhausted, match="cap 1000 exhausted: 1000 completed, 1150 attempted"):
        ev(np.zeros((500, 3)))                                   # only the 350 that fit are evaluated; the rest never is
    assert fake.batches[3:] == [300, 50] and ev.completed_total == 1000 and ev.attempted_total == 1150
    assert sum(fake.batches) == 1000                             # no evaluation beyond the cap, ever


def test_usage_persists_across_retries_and_is_never_reset(tmp_path):
    ledger = tmp_path / "ledger.json"
    f1 = FakeEvaluator()
    e1 = BudgetedEvaluator(f1, cap_evaluations=500, deadline_seconds=60, ledger_path=ledger, attempt="attempt1", batch_size=100)
    with pytest.raises(BudgetExhausted):
        e1(np.zeros((900, 3)))
    assert sum(f1.batches) == 500
    f2 = FakeEvaluator()
    e2 = BudgetedEvaluator(f2, cap_evaluations=500, deadline_seconds=60, ledger_path=ledger, attempt="attempt2", batch_size=100)   # a retry
    assert e2.completed_total == 500 and e2.remaining == 0        # the cap is cumulative; a retry does not reset it
    with pytest.raises(BudgetExhausted, match="exhausted"):
        e2(np.zeros((10, 3)))
    assert f2.batches == [] and e2.attempted_total == 910 and e2.completed_total == 500
    assert e2.ledger["attempts"]["attempt1"]["completed"] == 500 and e2.ledger["attempts"]["attempt2"]["completed"] == 0
    assert e2.ledger["attempts"]["attempt2"]["refused_batches"] == 1 and "exhausted" in e2.ledger["attempts"]["attempt2"]["stopped"]
    with pytest.raises(ValueError, match="a new cap needs a new ledger"):
        BudgetedEvaluator(f2, cap_evaluations=900, deadline_seconds=60, ledger_path=ledger, attempt="attempt3")


def test_deadline_stops_before_the_next_batch_with_a_checkpoint(tmp_path):
    clock = {"t": 0.0}
    fake = FakeEvaluator(); seen = []
    ev = BudgetedEvaluator(fake, cap_evaluations=10_000, deadline_seconds=5.0, ledger_path=tmp_path / "l.json", attempt="a",
                           batch_size=100, checkpoint=seen.append, clock=lambda: clock["t"])
    def ticking(pts):
        clock["t"] += 3.0
        return fake(pts)
    ev.evaluate = ticking
    with pytest.raises(BudgetExhausted, match="deadline 5 s exceeded after 200 completed"):
        ev(np.zeros((1000, 3)))
    assert fake.batches == [100, 100] and seen[-1]["reason"].startswith("deadline") and seen[-1]["completed_total"] == 200
    assert [c["reason"] for c in seen[:-1]] == ["batch", "batch"]   # partial checkpoints after every completed batch


def test_acceptance_requires_every_prespecified_condition():
    ok = acceptance_report({"faces": True, "refinement": True, "domain": True, "cdf_resolution": True, "cdf_domain": True})
    assert ok["accepted"] and ok["failed"] == [] and ok["missing"] == [] and ok["n_conditions"] == 5
    bad = acceptance_report({"faces": False, "refinement": True, "domain": True, "cdf_resolution": None, "cdf_domain": True})
    assert not bad["accepted"] and bad["failed"] == ["faces"] and bad["missing"] == ["cdf_resolution"]
    partial = acceptance_report({"faces": True, "refinement": True, "domain": None})
    assert not partial["accepted"] and partial["missing"] == ["domain"]


def test_seeded_f7_ledger_refuses_any_evaluation_under_the_exhausted_cap(tmp_path):
    """The committed ledger carries the 4,125,554 evaluations already spent against the 2,000,000 cap; a copy of it
    must refuse the very first batch (fake evaluator: nothing is evaluated)."""
    import json, shutil
    from pathlib import Path
    src = Path(__file__).resolve().parent.parent / "docs" / "m3b1-evidence" / "F7_marginal_reference_ledger.json"
    ledger = tmp_path / "ledger.json"; shutil.copy(src, ledger)
    seeded = json.loads(ledger.read_text())
    assert seeded["cap_evaluations"] == 2_000_000 and seeded["completed_total"] == 4_125_554
    assert sum(a["completed"] for a in seeded["attempts"].values()) == 4_125_554
    fake = FakeEvaluator()
    ev = BudgetedEvaluator(fake, cap_evaluations=2_000_000, deadline_seconds=900, ledger_path=ledger, attempt="dry")
    assert ev.remaining == 0
    with pytest.raises(BudgetExhausted, match="exhausted: 4125554 completed"):
        ev(np.zeros((1, 3)))
    assert fake.batches == [] and ev.completed_total == 4_125_554
