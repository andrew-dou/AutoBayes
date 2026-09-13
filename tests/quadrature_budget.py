"""A budget controller for capped deterministic evaluations (docs/m3b1-decision.md, marginal-reference cap
corrections of 2026-09-13; second and third independent guard reviews the same day).

Rules. The evaluation cap and the time allowance belong to the UNIT and live in a ledger file that is never reset.
Every piece of real work (a batch of evaluations or an external diagnostic charge) passes ONE central gate before it
starts: ownership, integer size >= 1, remaining cap, remaining time. It is written to the ledger as a reservation
before it runs and is confirmed only afterwards by a confirmation that must name the open reservation and its exact
size; a reservation that is never confirmed (exception, crash, interruption) stays counted for ever. Requested,
reserved-uncertain and confirmed counts are kept distinct. Time is whole-unit wall time measured by the injected
clock per attempt (driver overhead included, not only evaluator calls); the measured duration of a batch that raised
is recorded; an attempt that died without releasing is treated, on takeover, as having exhausted the allowance that
remained when it last wrote the ledger (its true duration is unknown); an expired unit refuses further attempts at
construction; the allowance is checked before every gate and again after every batch, so an overrun is bounded by
one batch and is recorded. Ownership is enforced with an OS-level exclusive lock on a sidecar file for the life of
the evaluator (a simultaneous owner is rejected at once); the ledger's owner/version fields add a takeover record
and reject stale writes. ``acceptance_report`` accepts only an explicit, complete inventory of genuine booleans.
No numerical evaluation happens in this module."""
from __future__ import annotations

import fcntl
import json
import math
import time
import uuid
from pathlib import Path

import numpy as np


class BudgetExhausted(RuntimeError):
    """The unit cap or the unit time allowance would be exceeded; nothing more is evaluated."""


class LedgerBusy(RuntimeError):
    """Another owner holds the ledger (lock held, or an unreleased owner without a declared takeover)."""


class StaleLedger(RuntimeError):
    """The ledger on disk changed underneath this owner, or a confirmation does not match an open reservation."""


def _finite_nonneg(x) -> float:
    x = float(x)
    if not math.isfinite(x) or x < 0:
        raise ValueError(f"expected a finite non-negative number, got {x!r}")
    return x


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
        self.reserve_seconds_per_batch = None if reserve_seconds_per_batch is None else _finite_nonneg(reserve_seconds_per_batch)
        self._closed = False
        self._acquire_lock()
        self.ledger = self._load()
        self._check_caps()
        self._disk_owner = self.ledger.get("owner")
        if self.ledger.get("owner") and not self.ledger.get("released", True):
            if not takeover:
                self._release_lock()
                raise LedgerBusy(
                    f"ledger {self.ledger_path} is held by owner {self.ledger['owner']} (attempt {self.ledger.get('owner_attempt')}) that never "
                    "released it; declare takeover=True to record a takeover (its unconfirmed reservations and its unknown duration stay counted)"
                )
            dead = self.ledger.get("owner_attempt")
            entry = self.ledger["attempts"].get(dead)
            if entry is not None and entry.get("crash_time_penalty_seconds") is None:
                entry["crash_time_penalty_seconds"] = max(self.unit_deadline - float(self.ledger.get("time_spent_at_last_write", 0.0)), 0.0)
            self.ledger.setdefault("takeovers", []).append({"from": self.ledger["owner"], "from_attempt": dead, "to": self.owner, "to_attempt": self.attempt})
        self.started_at = self.clock()
        self.ledger["owner"], self.ledger["owner_attempt"], self.ledger["released"] = self.owner, self.attempt, False
        self.ledger["attempts"].setdefault(self.attempt, {"requested": 0, "reserved_uncertain": 0, "confirmed": 0, "wall_seconds": 0.0,
                                                          "max_batch_seconds": 0.0, "refused": 0, "stopped": None, "deadline_overrun_seconds": 0.0,
                                                          "open_reservations": [], "failed_reservations": [], "crash_time_penalty_seconds": None})
        self._persist()
        if self.time_spent() >= self.unit_deadline:
            self._stop(f"unit time allowance {self.unit_deadline:.0f} s already consumed ({self.time_spent():.1f} s incl. unconfirmed work); attempt refused")

    # -- lock, ledger --------------------------------------------------------------------------------------
    def _acquire_lock(self) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_fd = open(self.ledger_path.with_suffix(".lock"), "a+")
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            self._lock_fd.close()
            raise LedgerBusy(f"another live process holds the lock {self.ledger_path.with_suffix('.lock')}; simultaneous owners are rejected")

    def _release_lock(self) -> None:
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        finally:
            self._lock_fd.close()

    def _load(self) -> dict:
        if self.ledger_path.exists():
            return json.loads(self.ledger_path.read_text())
        return {"cap_evaluations": self.cap, "unit_deadline_seconds": self.unit_deadline, "version": 0,
                "requested_total": 0, "reserved_uncertain_total": 0, "confirmed_completed_total": 0,
                "time_spent_at_last_write": 0.0, "attempts": {}, "owner": None, "released": True}

    def _check_caps(self) -> None:
        if int(self.ledger["cap_evaluations"]) != self.cap or float(self.ledger["unit_deadline_seconds"]) != self.unit_deadline:
            self._release_lock()
            raise ValueError(
                f"ledger cap/allowance {self.ledger['cap_evaluations']}/{self.ledger['unit_deadline_seconds']} differ from the "
                f"requested {self.cap}/{self.unit_deadline}; a new cap or allowance needs a new ledger under a new authorisation"
            )
        for k in ("requested_total", "reserved_uncertain_total", "confirmed_completed_total", "version"):
            self.ledger[k] = int(self.ledger.get(k, 0))

    def _persist(self) -> None:
        if self._closed:
            raise StaleLedger("this evaluator was closed; writes refused")
        if self.ledger_path.exists():
            on_disk = json.loads(self.ledger_path.read_text())
            if int(on_disk.get("version", 0)) != int(self.ledger["version"]) or on_disk.get("owner") != self._disk_owner:
                raise StaleLedger(f"ledger changed on disk underneath owner {self.owner}; write refused")
        a = self.ledger["attempts"].get(self.attempt)
        if a is not None:
            a["wall_seconds"] = max(a["wall_seconds"], self.clock() - self.started_at)   # driver overhead included
        self.ledger["time_spent_at_last_write"] = self.time_spent()
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
        return self.confirmed_total + self.reserved_uncertain_total

    @property
    def remaining(self) -> int:
        return max(self.cap - self.spent, 0)

    def time_spent(self) -> float:
        """Whole-unit time: every attempt's wall time (including overhead and measured failed batches), plus the
        conservative penalty of any attempt that died without releasing, plus estimates of open reservations."""
        total = 0.0
        for name, a in self.ledger["attempts"].items():
            wall = float(a.get("wall_seconds", 0.0))
            if name == self.attempt:
                wall = max(wall, self.clock() - self.started_at)
            total += wall + float(a.get("crash_time_penalty_seconds") or 0.0)
            total += sum(float(r.get("seconds_estimate", 0.0)) for r in a.get("open_reservations", []))
        return total

    def progress(self, reason: str) -> dict:
        return dict(reason=reason, requested=self.requested_total, reserved_uncertain=self.reserved_uncertain_total,
                    confirmed=self.confirmed_total, remaining=self.remaining, seconds_spent=self.time_spent(),
                    cap=self.cap, unit_deadline_seconds=self.unit_deadline)

    def _stop(self, reason: str) -> None:
        self.ledger["attempts"][self.attempt]["stopped"] = reason
        self.release()
        if self.checkpoint is not None:
            self.checkpoint(self.progress(reason))
        raise BudgetExhausted(reason)

    def release(self) -> None:
        if self._closed:
            return
        self.ledger["released"] = True
        self._persist()
        self._closed = True
        self._release_lock()

    close = release

    # -- the single gate ------------------------------------------------------------------------------------
    def _gate(self, size, label: str) -> None:
        if self._closed:
            raise StaleLedger("evaluator closed; no further work may be reserved")
        if not isinstance(size, (int, np.integer)) or isinstance(size, (bool, np.bool_)) or int(size) < 1:
            raise ValueError(f"reservation size must be an integer >= 1, got {size!r} for {label!r}")
        if self.time_spent() >= self.unit_deadline:
            self._stop(f"unit time allowance {self.unit_deadline:.0f} s reached ({self.time_spent():.1f} s) before {label!r}; {self.confirmed_total} confirmed evaluations")
        if int(size) > self.remaining:
            self.ledger["attempts"][self.attempt]["refused"] += 1
            self._stop(f"cap {self.cap} would be exceeded by {label!r} ({size} requested, {self.remaining} remaining: {self.confirmed_total} confirmed + {self.reserved_uncertain_total} reserved-uncertain)")

    def _reserve(self, size: int, label: str) -> str:
        self._gate(size, label)
        est = self.reserve_seconds_per_batch if self.reserve_seconds_per_batch is not None else self._max_batch_seconds()
        rid = uuid.uuid4().hex[:12]
        a = self.ledger["attempts"][self.attempt]
        self.ledger["reserved_uncertain_total"] += int(size); a["reserved_uncertain"] += int(size)
        a["open_reservations"].append({"id": rid, "label": label, "size": int(size), "seconds_estimate": est, "reserved_at": self.clock() - self.started_at})
        self._persist()
        return rid

    def _confirm(self, rid: str, size: int, seconds: float) -> None:
        a = self.ledger["attempts"][self.attempt]
        if not a["open_reservations"] or a["open_reservations"][-1]["id"] != rid:
            raise StaleLedger(f"confirmation {rid!r} does not name the open reservation of attempt {self.attempt!r}")
        rec = a["open_reservations"][-1]
        if int(size) != rec["size"]:
            raise StaleLedger(f"confirmation size {size} != reserved size {rec['size']} for {rid!r}")
        seconds = _finite_nonneg(seconds)
        a["open_reservations"].pop()
        self.ledger["reserved_uncertain_total"] -= rec["size"]; a["reserved_uncertain"] -= rec["size"]
        self.ledger["confirmed_completed_total"] += rec["size"]; a["confirmed"] += rec["size"]
        a["max_batch_seconds"] = max(a["max_batch_seconds"], seconds)
        self._persist()

    def _fail(self, rid: str, error: str, elapsed: float) -> None:
        a = self.ledger["attempts"][self.attempt]
        rec = a["open_reservations"].pop()
        assert rec["id"] == rid
        rec.update(error=error, elapsed_measured=float(elapsed))
        a["failed_reservations"].append(rec)                                    # its size stays in reserved_uncertain for ever
        a["max_batch_seconds"] = max(a["max_batch_seconds"], float(elapsed))
        a["stopped"] = f"work raised during {rec['label']!r} after {elapsed:.1f} s; reservation of {rec['size']} kept as uncertain"
        self.release()

    def _max_batch_seconds(self) -> float:
        return max([float(at.get("max_batch_seconds", 0.0)) for at in self.ledger["attempts"].values()] + [0.0])

    # -- external real work (diagnostics) through the same gate ----------------------------------------------
    def charge(self, n: int, label: str) -> str:
        """Reserve ``n`` evaluation-equivalents of external real work; returns the reservation id to confirm."""
        return self._reserve(n, label)

    def confirm_charge(self, rid: str, n: int, seconds: float) -> None:
        self._confirm(rid, n, seconds)

    def fail_charge(self, rid: str, error: str, elapsed: float) -> None:
        self._fail(rid, error, elapsed)

    # -- evaluation ----------------------------------------------------------------------------------------
    def __call__(self, points):
        points = np.asarray(points)
        n = len(points)
        if n < 1:
            raise ValueError("nothing to evaluate")
        self.ledger["requested_total"] += n
        self.ledger["attempts"][self.attempt]["requested"] += n
        self._persist()
        out = np.empty(n)
        i = 0
        while i < n:
            size = min(self.batch_size, self.remaining, n - i)
            if size < 1:
                size = n - i                                                     # let the gate refuse and record it
            rid = self._reserve(size, f"batch@{i}")
            t0 = self.clock()
            try:
                out[i:i + size] = self.evaluate(points[i:i + size])
            except BaseException as e:
                self._fail(rid, f"{type(e).__name__}: {e}", self.clock() - t0)
                raise
            self._confirm(rid, size, self.clock() - t0)
            i += size
            if self.checkpoint is not None:
                self.checkpoint(self.progress("batch"))
            if self.time_spent() > self.unit_deadline:                          # checked after the batch as well
                over = self.time_spent() - self.unit_deadline
                self.ledger["attempts"][self.attempt]["deadline_overrun_seconds"] = over
                self._stop(f"unit time allowance {self.unit_deadline:.0f} s exceeded by {over:.1f} s after the last batch (granularity: one batch)")
        return out


def run_is_valid(view: dict, stopped) -> dict:
    """Budget/run validity of a unit, independent of the scientific conditions: not stopped, no unconfirmed
    reservations, confirmed work within the cap (an exact-cap completion is valid), time within the allowance."""
    problems = []
    if stopped:
        problems.append(f"run stopped: {stopped}")
    if int(view.get("reserved_uncertain", 0)) != 0:
        problems.append(f"unconfirmed reservations: {view.get('reserved_uncertain')}")
    if int(view.get("confirmed", 0)) > int(view.get("cap", 0)):
        problems.append(f"confirmed {view.get('confirmed')} exceeds cap {view.get('cap')}")
    if float(view.get("seconds_spent", 0.0)) > float(view.get("unit_deadline_seconds", 0.0)):
        problems.append(f"time {view.get('seconds_spent')} s exceeds allowance {view.get('unit_deadline_seconds')} s")
    for k in ("requested", "reserved_uncertain", "confirmed", "cap"):
        if int(view.get(k, -1)) < 0:
            problems.append(f"invalid count {k}={view.get(k)}")
    return {"valid": not problems, "problems": problems}


def _is_true(v) -> bool:
    return isinstance(v, (bool, np.bool_)) and bool(v) is True


def acceptance_report(conditions: dict, required: list | tuple) -> dict:
    """Fail-closed acceptance of the scientific conditions against an explicit, complete inventory of names."""
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
        else:
            malformed.append(k)
    unexpected = sorted(k for k in conditions if k not in required)
    met = sorted(k for k in required if k in conditions and _is_true(conditions[k]))
    accepted = not omitted and not unchecked and not failed and not malformed and not unexpected and len(met) == len(required)
    return {"accepted": accepted, "n_required": len(required), "met": met, "failed": sorted(failed), "unchecked": unchecked,
            "omitted": omitted, "malformed": sorted(malformed), "unexpected": unexpected}
