"""Tolerances of the brms oracle gates.

A change here changes what "parity with brms" means for every feature, so it is reviewed as a change of
contract, never made inside a feature branch. Relative errors are measured against max(1, |a|, |b|): the
log density of a small model is O(10-100), so these are also roughly absolute tolerances there.
"""
from __future__ import annotations

import numpy as np

DESIGN_ABS = 1e-12      # G1: design matrices, group coding, response and trials (exact in theory)
LOGP_REL = 1e-8         # G2, G5: log density and its prior part
GRAD_REL = 1e-7         # G2: gradient entries
PRED_REL = 1e-9         # G3, G4: derived quantities, linear predictors and expected values


def rel_err(a, b) -> float:
    """Largest elementwise |a - b| / max(1, |a|, |b|)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape {a.shape} vs {b.shape}")
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b) / np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))))


def abs_err(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape {a.shape} vs {b.shape}")
    return 0.0 if a.size == 0 else float(np.max(np.abs(a - b)))
