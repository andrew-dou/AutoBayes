"""A budget controller for capped deterministic evaluations (docs/m3b1-decision.md, marginal-reference cap
corrections of 2026-09-13; second independent review the same day).

Rules: the cap and the wall deadline belong to the UNIT and live in a ledger file that is never reset. Every batch
is RESERVED in the ledger before any work starts; a reservation whose completion is never confirmed (exception,
crash, interruption) stays counted against the cap and the deadline for ever ("reserved-uncertain"); confirmed work
moves from reserved to completed. Requested, reserved-uncertain and confirmed-completed counts are kept distinct.
The remaining budget is checked before every batch and a batch is never larger than the remainder; the deadline is
checked before every batch and again after the last one (an overrun is bounded by one batch and is recorded, not
hidden). Ledger ownership is explicit: a second live owner is rejected unless a takeover is declared and recorded;
a write whose on-disk version moved underneath it is rejected as stale. ``acceptance_report`` accepts only when an
explicit, complete, prespecified inventory of condition names is present with each value a genuine boolean True;
omitted, unchecked (None), false (Python or NumPy), malformed and unexpected entries all fail closed.
No numerical evaluation happens in this module."""
from __future__ import annotations

import json
import math
import time
import uuid
from pathlib import Path

import numpy as np


class BudgetExhausted(RuntimeError):
    """The unit cap or the unit deadline would be exceeded; nothing more is evaluated."""


class LedgerBusy(RuntimeError):
    """Another owner holds the ledger and no takeover was declared."""


class StaleLedger(RuntimeError):
    """The ledger on disk changed underneath this owner; the write was refused."""


class BudgetedEvaluator:
    def __init__(self, evaluate, *, cap_evaluations: int, unit_deadline_seconds: float, ledger_path, attempt: str,
                 batch_size: int = 32768, checkpoint=None, clock=time.perf_counter, takeover: bool = False,
                 reserve_seconds_per_batch: float | None = None):
        self.evaluate = evaluate
        self.cap = int(cap_evaluations)
        self.unit_deadline = float(unit_deadline_seconds)
        self.ledger_path = Path(ledger_path)
        self.attempt = str(attempt)
        self.batch_size = int(batch_size)
        self.checkpoint = checkpoint
        self.clock = clock
        self.owner = uuid.uuid4().hex
        self.reserve_seconds_per_batch = reserve_seconds_per_batch
        self.ledger = self._load()
        self._check_caps()
        self._disk_owner = self.ledger.get("owner")                     # the owner this instance last saw on disk
        if self.ledger.get("owner") and not self.ledger.get("released", True) and not takeover:
            raise LedgerBusy(
                f"ledger {self.ledger_path} is held by owner {self.ledger['owner']} (attempt {self.ledger.get('owner_attempt')}); "
                "declare takeover=True to record a takeover (its unconfirmed reservations stay counted)"
            )
        if self.ledger.get("owner") and not self.ledger.get("released", True):
            self.ledger.setdefault("takeovers", []).append({"from": self.ledger["owner"], "from_attempt": self.ledger.get("owner_attempt"),
                                                            "to": self.owner, "to_attempt": self.attempt})
        self.ledger["owner"], self.ledger["owner_attempt"], self.ledger["released"] = self.owner, self.attempt, False
        self.ledger["attempts"].setdefault(self.attempt, {"requested": 0, "reserved_uncertain": 0, "confirmed": 0, "seconds_confirmed": 0.0,
                                                          "seconds_reserved_uncertain": 0.0, "refused_batches": 0, "stopped": None,
                                                          "deadline_overrun_seconds": 0.0, "failed_reservations": []})
        self._persist()
        if self.time_spent() >= self.unit_deadline:
            self._stop(f"unit deadline {self.unit_deadline:.0f} s already consumed ({self.time_spent():.1f} s incl. unconfirmed reservations); retry refused")

    # -- ledger --------------------------------------------------------------------------------------------
    def _load(self) -> dict:
        if self.ledger_path.exists():
            return json.loads(self.ledger_path.read_text())
        return {"cap_evaluations": self.cap, "unit_deadline_seconds": self.unit_deadline, "version": 0,
                "requested_total": 0, "reserved_uncertain_total": 0, "confirmed_completed_total": 0,
                "seconds_confirmed_total": 0.0, "seconds_reserved_uncertain_total": 0.0, "attempts": {}, "owner": None, "released": True}

    def _check_caps(self) -> None:
        if int(self.ledger["cap_evaluations"]) != self.cap or float(self.ledger["unit_deadline_seconds"]) != self.unit_deadline:
            raise ValueError(
                f"ledger cap/deadline {self.ledger['cap_evaluations']}/{self.ledger['unit_deadline_seconds']} differ from the "
                f"requested {self.cap}/{self.unit_deadline}; a new cap or deadline needs a new ledger under a new authorisation"
            )
        for k in ("requested_total", "reserved_uncertain_total", "confirmed_completed_total", "version"):
            self.ledger[k] = int(self.ledger.get(k, 0))

    def _persist(self) -> None:
        """Optimistic-concurrency write: refuse if the on-disk version is not the one this owner last saw."""
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        if self.ledger_path.exists():
            on_disk = json.loads(self.ledger_path.read_text())
            if int(on_disk.get("version", 0)) != int(self.ledger["version"]):
                raise StaleLedger(f"ledger version on disk {on_disk.get('version')} != {self.ledger['version']} held by owner {self.owner}; write refused")
            if on_disk.get("owner") != self._disk_owner:
                raise StaleLedger("ledger owner changed on disk; write refused")
        self.ledger["version"] += 1
        tmp = self.ledger_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.ledger, indent=1) + "\n")
        tmp.replace(self.ledger_path)
        self._disk_owner = self.ledger.get("owner")

    # -- accounting views ----------------------------------------------------------------------------------
    @property
    def confirmed_total(self) -> int:
        return int(self.ledger["confirmed_completed_total"])

    @property
    def reserved_uncertain_total(self) -> int:
        return int(self.ledger["reserved_uncertain_total"])

    @property
    def requested_total(self) -> int:
        return int(self.ledger["requested_total"])

    @property
    def spent(self) -> int:
        """Counted against the cap: confirmed work plus every unconfirmed reservation."""
        return self.confirmed_total + self.reserved_uncertain_total

    @property
    def remaining(self) -> int:
        return max(self.cap - self.spent, 0)

    def time_spent(self) -> float:
        return float(self.ledger["seconds_confirmed_total"]) + float(self.ledger["seconds_reserved_uncertain_total"])

    def _stop(self, reason: str) -> None:
        self.ledger["attempts"][self.attempt]["stopped"] = reason
        self.release()
        if self.checkpoint is not None:
            self.checkpoint(self.progress(reason))
        raise BudgetExhausted(reason)

    def progress(self, reason: str) -> dict:
        return dict(reason=reason, requested=self.requested_total, reserved_uncertain=self.reserved_uncertain_total,
                    confirmed=self.confirmed_total, remaining=self.remaining, seconds_spent=self.time_spent())

    def release(self) -> None:
        self.ledger["released"] = True
        self._persist()

    def charge(self, n: int, label: str) -> None:
        """Count external real work (for example a dense-covariance diagnostic evaluation) under the same rules:
        reserved before it starts, confirmed by the caller afterwards via ``confirm_charge``."""
        self._reserve(int(n), label)

    def confirm_charge(self, n: int, seconds: float = 0.0) -> None:
        self._confirm(int(n), float(seconds))

    # -- reservation / confirmation -------------------------------------------------------------------------
    def _reserve(self, size: int, label: str) -> float:
        est = self.reserve_seconds_per_batch if self.reserve_seconds_per_batch is not None else self._max_batch_seconds()
        a = self.ledger["attempts"][self.attempt]
        self.ledger["reserved_uncertain_total"] += size; a["reserved_uncertain"] += size
        self.ledger["seconds_reserved_uncertain_total"] += est; a["seconds_reserved_uncertain"] += est
        a["failed_reservations"].append({"label": label, "size": size, "seconds_estimate": est})   # removed on confirmation
        self._persist()
        return est

    def _confirm(self, size: int, seconds: float) -> None:
        a = self.ledger["attempts"][self.attempt]
        rec = a["failed_reservations"].pop()                                   # the reservation just made
        self.ledger["reserved_uncertain_total"] -= size; a["reserved_uncertain"] -= size
        self.ledger["seconds_reserved_uncertain_total"] -= rec["seconds_estimate"]; a["seconds_reserved_uncertain"] -= rec["seconds_estimate"]
        self.ledger["confirmed_completed_total"] += size; a["confirmed"] += size
        self.ledger["seconds_confirmed_total"] += seconds; a["seconds_confirmed"] += seconds
        a["max_batch_seconds"] = max(a.get("max_batch_seconds", 0.0), seconds)
        self._persist()

    def _max_batch_seconds(self) -> float:
        return max([at.get("max_batch_seconds", 0.0) for at in self.ledger["attempts"].values()] + [0.0])

    # -- evaluation ----------------------------------------------------------------------------------------
    def __call__(self, points):
        """Evaluate ``points`` in batches no larger than the remaining budget. Each batch is reserved in the
        ledger before it runs and confirmed after; a batch that raises leaves its reservation counted."""
        points = np.asarray(points)
        n = len(points)
        self.ledger["requested_total"] += n
        self.ledger["attempts"][self.attempt]["requested"] += n
        self._persist()
        out = np.empty(n)
        i = 0
        while i < n:
            if self.time_spent() >= self.unit_deadline:
                self._stop(f"unit deadline {self.unit_deadline:.0f} s reached ({self.time_spent():.1f} s spent incl. reservations) after {self.confirmed_total} confirmed evaluations")
            remaining = self.remaining
            if remaining == 0:
                self.ledger["attempts"][self.attempt]["refused_batches"] += 1
                self._stop(f"cap {self.cap} exhausted: {self.confirmed_total} confirmed + {self.reserved_uncertain_total} reserved-uncertain, {self.requested_total} requested")
            size = min(self.batch_size, remaining, n - i)                 # never larger than what remains
            self._reserve(size, f"batch@{i}")
            t0 = self.clock()
            try:
                out[i:i + size] = self.evaluate(points[i:i + size])
            except BaseException as e:                                       # the reservation stays counted
                a = self.ledger["attempts"][self.attempt]
                a["failed_reservations"][-1]["error"] = f"{type(e).__name__}: {e}"
                a["stopped"] = f"evaluate raised during batch@{i} of {size}; reservation kept as uncertain"
                self.release()
                raise
            seconds = self.clock() - t0
            self._confirm(size, seconds)
            i += size
            if self.checkpoint is not None:
                self.checkpoint(self.progress("batch"))
            if self.time_spent() > self.unit_deadline:                      # checked after the batch as well
                over = self.time_spent() - self.unit_deadline
                self.ledger["attempts"][self.attempt]["deadline_overrun_seconds"] = over
                self._stop(f"unit deadline {self.unit_deadline:.0f} s exceeded by {over:.1f} s after the last batch (granularity: one batch)")
        return out


def _is_true(v) -> bool:
    return isinstance(v, (bool, np.bool_)) and bool(v) is True


def acceptance_report(conditions: dict, required: list | tuple) -> dict:
    """Fail-closed acceptance. ``required`` is the explicit, complete, prespecified inventory of condition names;
    ``conditions`` maps names to True (met), False / numpy False (failed) or None (not checked). Accepted only when
    every required name is present and genuinely True; omitted, unchecked, failed, malformed (non-boolean, NaN/inf,
    strings) and unexpected entries are listed and each blocks acceptance."""
    required = list(required)
    if not required or len(set(required)) != len(required):
        raise ValueError("required must be a non-empty list of distinct prespecified condition names")
    omitted = sorted(k for k in required if k not in conditions)
    unchecked = sorted(k for k in required if k in conditions and conditions[k] is None)
    failed, malformed = [], []
    for k in required:
        if k not in conditions or conditions[k] is None:
            continue
        v = conditions[k]
        if isinstance(v, (bool, np.bool_)):
            if not bool(v):
                failed.append(k)
        elif isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_)):
            malformed.append(k)                                              # numbers (incl. NaN/inf) are not verdicts
        else:
            malformed.append(k)
    unexpected = sorted(k for k in conditions if k not in required)
    met = sorted(k for k in required if k in conditions and _is_true(conditions[k]))
    accepted = not omitted and not unchecked and not failed and not malformed and not unexpected and len(met) == len(required)
    return {"accepted": accepted, "n_required": len(required), "met": met, "failed": sorted(failed), "unchecked": unchecked,
            "omitted": omitted, "malformed": sorted(malformed), "unexpected": unexpected}
