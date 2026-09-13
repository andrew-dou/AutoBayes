"""Controller tests (docs/m3b1-decision.md, cap corrections of 2026-09-13). Every evaluator here is a fake that
records batch sizes; clocks are fake; no posterior density is evaluated anywhere in this module."""
import json
from pathlib import Path

import numpy as np
import pytest

from tests.quadrature_budget import BudgetExhausted, BudgetedEvaluator, LedgerBusy, StaleLedger, acceptance_report


class Fake:
    def __init__(self, fail_after=None):
        self.batches, self.fail_after = [], fail_after

    def __call__(self, pts):
        self.batches.append(len(pts))
        if self.fail_after is not None and len(self.batches) > self.fail_after:
            raise RuntimeError("fake evaluator consumed the batch then failed")
        return np.zeros(len(pts))


def make(fake, ledger, attempt, **kw):
    kw.setdefault("cap_evaluations", 1000); kw.setdefault("unit_deadline_seconds", 60); kw.setdefault("batch_size", 100)
    return BudgetedEvaluator(fake, ledger_path=ledger, attempt=attempt, **kw)


def test_batches_never_exceed_the_remainder_and_the_three_counts_are_distinct(tmp_path):
    fake = Fake(); ev = make(fake, tmp_path / "l.json", "a1", batch_size=300)
    out = ev(np.zeros((650, 3)))
    assert out.shape == (650,) and fake.batches == [300, 300, 50]
    assert (ev.requested_total, ev.reserved_uncertain_total, ev.confirmed_total, ev.remaining) == (650, 0, 650, 350)
    with pytest.raises(BudgetExhausted, match="cap 1000 exhausted: 1000 confirmed \\+ 0 reserved-uncertain, 1150 requested"):
        ev(np.zeros((500, 3)))
    assert fake.batches[3:] == [300, 50] and sum(fake.batches) == 1000


def test_reservation_persists_before_work_and_a_failed_batch_stays_counted(tmp_path):
    """The independent review's reproduction: a batch that raises after consuming work must not be refunded, and a
    retry must not re-spend it (ledger cap 10, batch 10)."""
    ledger = tmp_path / "l.json"; spent = []
    fail = Fake(fail_after=0); ev = make(fail, ledger, "a", cap_evaluations=10, batch_size=10)
    with pytest.raises(RuntimeError, match="consumed the batch then failed"):
        ev(np.zeros((10, 3)))
    on_disk = json.loads(ledger.read_text())
    assert on_disk["reserved_uncertain_total"] == 10 and on_disk["confirmed_completed_total"] == 0 and on_disk["requested_total"] == 10
    assert on_disk["attempts"]["a"]["failed_reservations"][0]["size"] == 10 and "RuntimeError" in on_disk["attempts"]["a"]["failed_reservations"][0]["error"]
    ok = Fake(); retry = make(ok, ledger, "b", cap_evaluations=10, batch_size=10)      # released by the failure -> no takeover needed
    assert retry.spent == 10 and retry.remaining == 0
    with pytest.raises(BudgetExhausted, match="0 confirmed \\+ 10 reserved-uncertain"):
        retry(np.zeros((10, 3)))
    assert ok.batches == [] and sum(fail.batches) + sum(ok.batches) == 10               # total fake work never exceeds the cap


def test_reservation_survives_a_simulated_crash_without_release(tmp_path):
    """A process that dies mid-batch never confirms or releases: the next owner must take over explicitly, and the
    unconfirmed reservation remains counted."""
    ledger = tmp_path / "l.json"; fake = Fake()
    ev = make(fake, ledger, "crashed", cap_evaluations=100, batch_size=40)
    ev._reserve(40, "batch@0")                                    # reserved, then the process "dies" here
    with pytest.raises(LedgerBusy, match="held by owner"):
        make(Fake(), ledger, "next", cap_evaluations=100, batch_size=40)
    nxt = make(Fake(), ledger, "next", cap_evaluations=100, batch_size=40, takeover=True)
    assert nxt.spent == 40 and nxt.remaining == 60 and nxt.ledger["takeovers"][0]["from_attempt"] == "crashed"
    nxt(np.zeros((60, 3)))
    assert nxt.confirmed_total == 60 and nxt.reserved_uncertain_total == 40 and nxt.remaining == 0
    with pytest.raises(StaleLedger):                              # the dead owner's in-memory copy is stale: its writes are refused
        ev._persist()


def test_stale_write_is_rejected_when_the_file_moved_underneath(tmp_path):
    ledger = tmp_path / "l.json"; ev = make(Fake(), ledger, "a")
    d = json.loads(ledger.read_text()); d["version"] += 5; ledger.write_text(json.dumps(d))
    with pytest.raises(StaleLedger, match="write refused"):
        ev(np.zeros((1, 3)))


def test_new_cap_or_deadline_needs_a_new_ledger(tmp_path):
    ledger = tmp_path / "l.json"; make(Fake(), ledger, "a").release()
    with pytest.raises(ValueError, match="new cap or deadline needs a new ledger"):
        make(Fake(), ledger, "b", cap_evaluations=5000)
    with pytest.raises(ValueError, match="new cap or deadline needs a new ledger"):
        make(Fake(), ledger, "b", unit_deadline_seconds=120)


def test_unit_deadline_is_cumulative_across_attempts_and_checked_after_the_last_batch(tmp_path):
    """The review's reproduction: two attempts each starting a 6 s batch under a 5 s UNIT deadline. The first batch
    overruns by one batch (recorded); the retry is refused before any work."""
    ledger = tmp_path / "l.json"; clk = {"t": 0.0}; fake = Fake()
    def slow(pts):
        clk["t"] += 6.0; return fake(pts)
    ev = make(slow, ledger, "a", cap_evaluations=100, unit_deadline_seconds=5.0, batch_size=1, clock=lambda: clk["t"])
    with pytest.raises(BudgetExhausted, match="exceeded by 1.0 s after the last batch \\(granularity: one batch\\)"):
        ev(np.zeros((3, 3)))
    assert fake.batches == [1] and ev.ledger["attempts"]["a"]["deadline_overrun_seconds"] == 1.0 and ev.time_spent() == 6.0
    with pytest.raises(BudgetExhausted, match="already consumed"):
        make(slow, ledger, "b", cap_evaluations=100, unit_deadline_seconds=5.0, batch_size=1, clock=lambda: clk["t"])
    assert fake.batches == [1]                                                     # the retry evaluated nothing
    # an interrupted batch's time reservation also counts: reserve with an estimate, never confirm
    ledger2 = tmp_path / "l2.json"; ev2 = make(Fake(), ledger2, "a", cap_evaluations=100, unit_deadline_seconds=5.0, reserve_seconds_per_batch=4.0)
    ev2._reserve(10, "batch@0")
    nxt = make(Fake(), ledger2, "b", cap_evaluations=100, unit_deadline_seconds=5.0, takeover=True, reserve_seconds_per_batch=4.0)
    assert nxt.time_spent() == 4.0                                                # the dead attempt's estimate stays counted
    nxt(np.zeros((30, 3)))                                                        # 4.0 < 5.0: allowed; its own estimate is replaced by the measured time
    assert nxt.confirmed_total == 30 and abs(nxt.time_spent() - 4.0) < 0.5 and nxt.ledger["seconds_reserved_uncertain_total"] == 4.0
    nxt._reserve(1, "batch@x")                                                    # a second unconfirmed reservation pushes the unit over 5 s
    with pytest.raises(BudgetExhausted, match="already consumed \\(8.0 s incl. unconfirmed reservations\\); retry refused"):
        make(Fake(), ledger2, "c", cap_evaluations=100, unit_deadline_seconds=5.0, takeover=True, reserve_seconds_per_batch=4.0)   # refused at construction


def test_external_work_is_charged_under_the_same_rules(tmp_path):
    ev = make(Fake(), tmp_path / "l.json", "a", cap_evaluations=5)
    ev.charge(2, "dense diagnostic"); assert ev.reserved_uncertain_total == 2 and ev.remaining == 3
    ev.confirm_charge(2, 0.5); assert ev.confirmed_total == 2 and ev.reserved_uncertain_total == 0
    ev.charge(3, "lowrank diagnostic")
    with pytest.raises(BudgetExhausted, match="exhausted: 2 confirmed \\+ 3 reserved-uncertain"):
        ev(np.zeros((1, 3)))


def test_acceptance_fails_closed_on_omitted_unchecked_false_numpy_false_malformed_and_unexpected():
    req = ["faces", "refinement", "domain", "cdf_resolution", "cdf_domain"]
    assert acceptance_report({k: True for k in req}, req)["accepted"]
    assert acceptance_report({k: np.bool_(True) for k in req}, req)["accepted"]
    with pytest.raises(ValueError):
        acceptance_report({}, [])
    r = acceptance_report({}, req); assert not r["accepted"] and r["omitted"] == sorted(req)
    r = acceptance_report({"faces": True}, req); assert not r["accepted"] and r["omitted"] == sorted(req[1:]) and r["met"] == ["faces"]
    r = acceptance_report({**{k: True for k in req}, "faces": np.bool_(False)}, req); assert not r["accepted"] and r["failed"] == ["faces"]
    r = acceptance_report({**{k: True for k in req}, "domain": None}, req); assert not r["accepted"] and r["unchecked"] == ["domain"]
    for bad in (float("nan"), float("inf"), 1, 0, 1.0, "True", "yes", [True]):
        r = acceptance_report({**{k: True for k in req}, "refinement": bad}, req); assert not r["accepted"] and r["malformed"] == ["refinement"], bad
    r = acceptance_report({**{k: True for k in req}, "extra_typo": True}, req); assert not r["accepted"] and r["unexpected"] == ["extra_typo"]


def test_seeded_f7_ledger_refuses_any_evaluation_under_the_exhausted_cap(tmp_path):
    """A copy of the committed ledger (the spent evaluations of attempts 1, 2 and the tail addendum plus the four
    historically uncounted diagnostic calls) refuses the first batch; the fake evaluator is never called."""
    import shutil
    src = Path(__file__).resolve().parent.parent / "docs" / "m3b1-evidence" / "F7_marginal_reference_ledger.json"
    ledger = tmp_path / "ledger.json"; shutil.copy(src, ledger)
    seeded = json.loads(ledger.read_text())
    assert seeded["cap_evaluations"] == 2_000_000 and seeded["confirmed_completed_total"] == 4_125_558 and seeded["reserved_uncertain_total"] == 0
    assert sum(a["confirmed"] for a in seeded["attempts"].values()) == 4_125_558 and seeded["released"] is True
    fake = Fake(); ev = BudgetedEvaluator(fake, cap_evaluations=2_000_000, unit_deadline_seconds=900, ledger_path=ledger, attempt="dry")
    assert ev.remaining == 0 and ev.spent == 4_125_558
    with pytest.raises(BudgetExhausted, match="exhausted: 4125558 confirmed"):
        ev(np.zeros((1, 3)))
    assert fake.batches == []
