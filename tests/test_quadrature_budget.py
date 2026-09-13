"""Controller tests (docs/m3b1-decision.md, cap corrections of 2026-09-13, third guard review). Every evaluator here
is a fake that records batch sizes; clocks are fake; no posterior density is evaluated anywhere in this module."""
import json
from pathlib import Path

import numpy as np
import pytest

from tests.quadrature_budget import BudgetExhausted, BudgetedEvaluator, LedgerBusy, StaleLedger, acceptance_report, run_is_valid


class Fake:
    def __init__(self, fail_after=None, clock=None, seconds=0.0):
        self.batches, self.fail_after, self.clock, self.seconds = [], fail_after, clock, seconds

    def __call__(self, pts):
        self.batches.append(len(pts))
        if self.clock is not None:
            self.clock["t"] += self.seconds
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
    with pytest.raises(BudgetExhausted, match="cap 1000 would be exceeded by 'batch@350'"):
        ev(np.zeros((500, 3)))
    assert fake.batches[3:] == [300, 50] and sum(fake.batches) == 1000
    assert ev.ledger["released"] is True


def test_failed_batch_keeps_its_reservation_and_a_retry_cannot_respend_it(tmp_path):
    ledger = tmp_path / "l.json"
    fail = Fake(fail_after=0); ev = make(fail, ledger, "a", cap_evaluations=10, batch_size=10)
    with pytest.raises(RuntimeError, match="consumed the batch then failed"):
        ev(np.zeros((10, 3)))
    d = json.loads(ledger.read_text())
    assert d["reserved_uncertain_total"] == 10 and d["confirmed_completed_total"] == 0 and d["requested_total"] == 10 and d["released"] is True
    assert d["attempts"]["a"]["failed_reservations"][0]["size"] == 10 and "RuntimeError" in d["attempts"]["a"]["failed_reservations"][0]["error"]
    ok = Fake(); retry = make(ok, ledger, "b", cap_evaluations=10, batch_size=10)
    assert retry.spent == 10 and retry.remaining == 0
    with pytest.raises(BudgetExhausted, match="0 confirmed \\+ 10 reserved-uncertain"):
        retry(np.zeros((10, 3)))
    assert ok.batches == [] and sum(fail.batches) == 10


def test_external_charge_passes_the_same_gate_and_confirmations_are_validated(tmp_path):
    """The third review's reproduction: cap 1, charge(2) must be refused before any work."""
    ev = make(Fake(), tmp_path / "c.json", "a", cap_evaluations=1)
    with pytest.raises(BudgetExhausted, match="cap 1 would be exceeded by 'two diagnostics' \\(2 requested, 1 remaining"):
        ev.charge(2, "two diagnostics")
    assert ev.spent == 0 and ev.ledger["released"] is True
    ev = make(Fake(), tmp_path / "c2.json", "a", cap_evaluations=5)
    for bad in (0, -1, 1.5, True, "2"):
        with pytest.raises(ValueError, match="integer >= 1"):
            ev.charge(bad, "bad")
    rid = ev.charge(2, "dense diagnostic"); assert ev.reserved_uncertain_total == 2 and ev.remaining == 3
    with pytest.raises(StaleLedger, match="does not name the open reservation"):
        ev.confirm_charge("bogus", 2, 0.1)
    with pytest.raises(StaleLedger, match="confirmation size 3 != reserved size 2"):
        ev.confirm_charge(rid, 3, 0.1)
    with pytest.raises(ValueError, match="finite non-negative"):
        ev.confirm_charge(rid, 2, float("nan"))
    with pytest.raises(ValueError, match="finite non-negative"):
        ev.confirm_charge(rid, 2, -1.0)
    ev.confirm_charge(rid, 2, 0.5); assert ev.confirmed_total == 2 and ev.reserved_uncertain_total == 0
    with pytest.raises(StaleLedger, match="does not name the open reservation"):
        ev.confirm_charge(rid, 2, 0.1)                                              # a second confirmation of the same id is stale
    rid2 = ev.charge(3, "lowrank diagnostic")
    with pytest.raises(BudgetExhausted, match="would be exceeded"):
        ev(np.zeros((1, 3)))                                                        # 2 confirmed + 3 reserved = cap
    ev2 = make(Fake(), tmp_path / "c3.json", "a", cap_evaluations=5)
    rid3 = ev2.charge(2, "diag"); ev2.fail_charge(rid3, "RuntimeError: boom", 0.7)
    assert ev2.reserved_uncertain_total == 2 and ev2.ledger["attempts"]["a"]["failed_reservations"][0]["elapsed_measured"] == 0.7


def test_failed_batch_time_is_measured_and_the_expired_retry_is_refused(tmp_path):
    """The third review's reproduction: a fake batch takes 6 s then raises under a 5 s unit allowance."""
    ledger = tmp_path / "t.json"; clk = {"t": 0.0}
    fail = Fake(fail_after=0, clock=clk, seconds=6.0)
    ev = make(fail, ledger, "a", cap_evaluations=100, unit_deadline_seconds=5.0, clock=lambda: clk["t"])
    with pytest.raises(RuntimeError):
        ev(np.zeros((1, 3)))
    d = json.loads(ledger.read_text())
    assert d["attempts"]["a"]["failed_reservations"][0]["elapsed_measured"] == 6.0 and d["attempts"]["a"]["wall_seconds"] >= 6.0
    assert d["time_spent_at_last_write"] >= 6.0
    good = Fake(clock=clk)
    with pytest.raises(BudgetExhausted, match="already consumed \\(6.0 s incl. unconfirmed work\\); attempt refused"):
        make(good, ledger, "b", cap_evaluations=100, unit_deadline_seconds=5.0, clock=lambda: clk["t"])
    assert good.batches == [] and sum(fail.batches) == 1


def test_unknown_crash_duration_conservatively_exhausts_the_allowance(tmp_path):
    """An attempt that reserved work and died without releasing: on takeover its unknown duration is charged as the
    allowance that remained when it last wrote, so the retry is refused; the evaluation reservation stays counted."""
    ledger = tmp_path / "k.json"; clk = {"t": 0.0}
    dead = make(Fake(), ledger, "dead", cap_evaluations=100, unit_deadline_seconds=50.0, clock=lambda: clk["t"])
    clk["t"] = 2.0; dead._reserve(10, "batch@0")                                    # wrote at 2.0 s, then "dies" here
    dead._release_lock()                                                             # the OS releases a dead process's lock
    with pytest.raises(LedgerBusy, match="never released"):
        make(Fake(), ledger, "next", cap_evaluations=100, unit_deadline_seconds=50.0, clock=lambda: clk["t"])
    with pytest.raises(BudgetExhausted, match="already consumed \\(5[0-9]\\.[0-9] s incl. unconfirmed work\\)"):
        make(Fake(), ledger, "next", cap_evaluations=100, unit_deadline_seconds=50.0, clock=lambda: clk["t"], takeover=True)
    d = json.loads(ledger.read_text())
    assert d["attempts"]["dead"]["crash_time_penalty_seconds"] == 48.0 and d["reserved_uncertain_total"] == 10 and d["takeovers"][0]["from_attempt"] == "dead"


def test_driver_overhead_counts_towards_the_unit_allowance_and_the_final_batch_is_checked(tmp_path):
    ledger = tmp_path / "o.json"; clk = {"t": 0.0}; fake = Fake(clock=clk, seconds=1.0)
    ev = make(fake, ledger, "a", cap_evaluations=100, unit_deadline_seconds=5.0, batch_size=1, clock=lambda: clk["t"])
    clk["t"] += 3.0                                                                  # driver overhead outside the evaluator
    ev(np.zeros((1, 3)))                                                             # 3 + 1 = 4 s: allowed
    clk["t"] += 0.5
    with pytest.raises(BudgetExhausted, match="exceeded by 0.5 s after the last batch \\(granularity: one batch\\)"):
        ev(np.zeros((1, 3)))                                                         # starts at 4.5 s (< 5), ends at 5.5 s: recorded overrun, stop
    assert fake.batches == [1, 1] and ev.ledger["attempts"]["a"]["deadline_overrun_seconds"] == 0.5
    with pytest.raises(BudgetExhausted, match="already consumed"):
        make(Fake(), ledger, "b", cap_evaluations=100, unit_deadline_seconds=5.0, clock=lambda: clk["t"])


def test_simultaneous_owners_are_rejected_by_the_os_lock(tmp_path):
    ledger = tmp_path / "lock.json"
    first = make(Fake(), ledger, "a")
    with pytest.raises(LedgerBusy, match="another live process holds the lock"):
        make(Fake(), ledger, "b")
    first.release()
    second = make(Fake(), ledger, "b"); second.release()                             # allowed once released
    first_again = make(Fake(), ledger, "c")
    with pytest.raises(StaleLedger, match="closed"):
        first(np.zeros((1, 3)))                                                      # a released evaluator cannot work again
    first_again.release()


def test_stale_write_is_rejected_and_a_new_cap_needs_a_new_ledger(tmp_path):
    ledger = tmp_path / "l.json"; ev = make(Fake(), ledger, "a")
    d = json.loads(ledger.read_text()); d["version"] += 5; ledger.write_text(json.dumps(d))
    with pytest.raises(StaleLedger, match="write refused"):
        ev(np.zeros((1, 3)))
    ev._closed = True; ev._release_lock()
    with pytest.raises(ValueError, match="new cap or allowance needs a new ledger"):
        make(Fake(), ledger, "b", cap_evaluations=5000)


def test_run_validity_is_independent_of_the_scientific_conditions():
    ok = dict(reason="complete", requested=2000, reserved_uncertain=0, confirmed=2000, remaining=0, seconds_spent=4.0, cap=2000, unit_deadline_seconds=5.0)
    assert run_is_valid(ok, None) == {"valid": True, "problems": []}                  # exact-cap completion is valid
    assert not run_is_valid(ok, "unit time allowance exceeded")["valid"]
    assert "exceeds allowance" in run_is_valid({**ok, "seconds_spent": 6.0}, None)["problems"][0]
    assert "exceeds cap" in run_is_valid({**ok, "confirmed": 2001}, None)["problems"][0]
    assert "unconfirmed reservations" in run_is_valid({**ok, "reserved_uncertain": 3}, None)["problems"][0]
    assert "invalid count" in run_is_valid({**ok, "confirmed": -1}, None)["problems"][0]


def test_acceptance_fails_closed_on_omitted_unchecked_false_numpy_false_malformed_and_unexpected():
    req = ["faces", "refinement", "domain", "cdf_resolution", "cdf_domain"]
    assert acceptance_report({k: True for k in req}, req)["accepted"]
    assert acceptance_report({k: np.bool_(True) for k in req}, req)["accepted"]
    with pytest.raises(ValueError):
        acceptance_report({}, [])
    r = acceptance_report({}, req); assert not r["accepted"] and r["omitted"] == sorted(req)
    r = acceptance_report({"faces": True}, req); assert not r["accepted"] and r["omitted"] == sorted(req[1:])
    r = acceptance_report({**{k: True for k in req}, "faces": np.bool_(False)}, req); assert not r["accepted"] and r["failed"] == ["faces"]
    r = acceptance_report({**{k: True for k in req}, "domain": None}, req); assert not r["accepted"] and r["unchecked"] == ["domain"]
    for bad in (float("nan"), float("inf"), 1, 0, 1.0, "True", "yes", [True]):
        r = acceptance_report({**{k: True for k in req}, "refinement": bad}, req); assert not r["accepted"] and r["malformed"] == ["refinement"], bad
    r = acceptance_report({**{k: True for k in req}, "extra_typo": True}, req); assert not r["accepted"] and r["unexpected"] == ["extra_typo"]


def test_seeded_f7_ledger_refuses_any_evaluation_under_the_exhausted_cap(tmp_path):
    import shutil
    src = Path(__file__).resolve().parent.parent / "docs" / "m3b1-evidence" / "F7_marginal_reference_ledger.json"
    ledger = tmp_path / "ledger.json"; shutil.copy(src, ledger)
    seeded = json.loads(ledger.read_text())
    assert seeded["cap_evaluations"] == 2_000_000 and seeded["confirmed_completed_total"] == 4_125_558 and seeded["released"] is True
    fake = Fake(); ev = BudgetedEvaluator(fake, cap_evaluations=2_000_000, unit_deadline_seconds=900, ledger_path=ledger, attempt="dry")
    assert ev.remaining == 0 and ev.spent == 4_125_558
    with pytest.raises(BudgetExhausted, match="would be exceeded by 'batch@0' \\(1 requested, 0 remaining: 4125558 confirmed"):
        ev(np.zeros((1, 3)))
    assert fake.batches == []
