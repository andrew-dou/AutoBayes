"""A budget controller for capped deterministic evaluations (docs/m3b1-decision.md, marginal-reference cap
correction, 2026-09-13). Written after the F7 marginal-reference driver overran its 2,000,000-evaluation unit
cap: the old driver evaluated a whole batch and checked the counter afterwards, and a retry reset the counter.

Rules enforced here: the remaining CUMULATIVE budget is checked before every bounded batch; a batch is never
larger than what remains; completed and attempted counts are kept distinct; usage is persisted in a ledger
file across processes and retries and is never reset; a wall-clock deadline is checked before every batch;
a checkpoint callback receives partial progress; an exhausted cap stops the run with ``BudgetExhausted``.
``acceptance_report`` accepts only when every prespecified condition is present and true, listing failed and
missing conditions explicitly. No numerical evaluation happens in this module."""
from __future__ import annotations

import json
import time
from pathlib import Path


class BudgetExhausted(RuntimeError):
    """The cumulative evaluation cap (or the wall deadline) would be exceeded; nothing more is evaluated."""


class BudgetedEvaluator:
    def __init__(self, evaluate, *, cap_evaluations: int, deadline_seconds: float, ledger_path, attempt: str,
                 batch_size: int = 32768, checkpoint=None, clock=time.perf_counter):
        self.evaluate = evaluate
        self.cap = int(cap_evaluations)
        self.deadline_seconds = float(deadline_seconds)
        self.ledger_path = Path(ledger_path)
        self.attempt = attempt
        self.batch_size = int(batch_size)
        self.checkpoint = checkpoint
        self.clock = clock
        self.t0 = clock()
        self.ledger = self._load()
        self.ledger.setdefault("cap_evaluations", self.cap)
        if self.ledger["cap_evaluations"] != self.cap:
            raise ValueError(f"ledger cap {self.ledger['cap_evaluations']} differs from the requested cap {self.cap}; a new cap needs a new ledger")
        self.ledger.setdefault("attempts", {})
        self.ledger["attempts"].setdefault(attempt, {"completed": 0, "attempted": 0, "refused_batches": 0, "stopped": None})
        self._persist()

    # -- ledger --------------------------------------------------------------------------------------------
    def _load(self) -> dict:
        if self.ledger_path.exists():
            return json.loads(self.ledger_path.read_text())
        return {"completed_total": 0, "attempted_total": 0}

    def _persist(self) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.ledger_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.ledger, indent=1) + "\n")
        tmp.replace(self.ledger_path)

    @property
    def completed_total(self) -> int:
        return int(self.ledger["completed_total"])

    @property
    def attempted_total(self) -> int:
        return int(self.ledger["attempted_total"])

    @property
    def remaining(self) -> int:
        return max(self.cap - self.completed_total, 0)

    def elapsed(self) -> float:
        return self.clock() - self.t0

    def _stop(self, reason: str) -> None:
        self.ledger["attempts"][self.attempt]["stopped"] = reason
        self._persist()
        if self.checkpoint is not None:
            self.checkpoint(dict(reason=reason, completed_total=self.completed_total, attempted_total=self.attempted_total, elapsed=self.elapsed()))
        raise BudgetExhausted(reason)

    # -- evaluation ----------------------------------------------------------------------------------------
    def __call__(self, points):
        """Evaluate ``points`` in batches no larger than the remaining budget; returns the full result only if
        every batch fits; otherwise stops after the last batch that fit (those evaluations are counted)."""
        import numpy as np

        points = np.asarray(points)
        n = len(points)
        self.ledger["attempted_total"] += n
        self.ledger["attempts"][self.attempt]["attempted"] += n
        self._persist()
        out = np.empty(n)
        i = 0
        while i < n:
            if self.elapsed() > self.deadline_seconds:
                self._stop(f"deadline {self.deadline_seconds:.0f} s exceeded after {self.completed_total} completed evaluations")
            remaining = self.remaining
            if remaining == 0:
                self.ledger["attempts"][self.attempt]["refused_batches"] += 1
                self._stop(f"cap {self.cap} exhausted: {self.completed_total} completed, {self.attempted_total} attempted")
            size = min(self.batch_size, remaining, n - i)         # never larger than what remains
            out[i:i + size] = self.evaluate(points[i:i + size])
            i += size
            self.ledger["completed_total"] += size
            self.ledger["attempts"][self.attempt]["completed"] += size
            self._persist()
            if self.checkpoint is not None:
                self.checkpoint(dict(reason="batch", completed_total=self.completed_total, attempted_total=self.attempted_total, elapsed=self.elapsed()))
        return out


def acceptance_report(conditions: dict) -> dict:
    """``conditions`` maps a prespecified condition name to True (met), False (not met) or None (not checked).
    Accepted only when every condition is present and True; failed and missing conditions are listed."""
    failed = sorted(k for k, v in conditions.items() if v is False)
    missing = sorted(k for k, v in conditions.items() if v is None)
    return {"accepted": not failed and not missing, "failed": failed, "missing": missing,
            "met": sorted(k for k, v in conditions.items() if v is True), "n_conditions": len(conditions)}
