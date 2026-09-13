"""Injected-evaluator quadrature for a proposed reference; imports no F7 target.

This module has no CLI and starts no run at import. A caller must explicitly
supply a budgeted evaluator and a reviewed protocol. Agreement between rules,
domains and sampled faces is a diagnostic, not a certified error/tail bound.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np
from scipy.special import logsumexp

from tests.quadrature_budget import acceptance_report, run_is_valid

NAMES = ("1|g_sigma", "x|g_sigma", "sigma")
LIMITS = {"mean": 1e-4, "logZ": 1e-4, "cdf": 1e-3, "face": -25.0}


def positive_int(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError("expected a positive integer")
    return int(value)


def box_array(box):
    box = np.asarray(box, dtype=float)
    if box.shape != (3, 2) or not np.isfinite(box).all() or not np.all(box[:, 0] < box[:, 1]):
        raise ValueError("expected three finite, increasing intervals")
    return box


def rule_arrays(box, nodes):
    box = box_array(box)
    if len(nodes) != 3:
        raise ValueError("expected three axis resolutions")
    xs, ws = [], []
    for (lo, hi), n in zip(box, nodes):
        x, w = np.polynomial.legendre.leggauss(positive_int(n))
        half = (hi - lo) / 2
        xs.append(lo + half * (1 + x))
        ws.append(np.log(w) + np.log(half))
    return xs, ws


def batches(xs, logweights, batch_size):
    """Generate at most batch_size points without materializing a full tensor."""
    shape = tuple(len(x) for x in xs)
    for start in range(0, math.prod(shape), positive_int(batch_size)):
        ids = np.unravel_index(np.arange(start, min(start + batch_size, math.prod(shape))), shape)
        yield np.column_stack([x[i] for x, i in zip(xs, ids)]), sum(w[i] for w, i in zip(logweights, ids))


def values(evaluate, points):
    y = np.asarray(evaluate(points), dtype=float)
    if y.shape != (len(points),) or not np.isfinite(y).all():
        raise ValueError("evaluator must return one finite log density per point")
    return y


def integrate_box(evaluate, box, nodes, *, batch_size=32768):
    xs, ws = rule_arrays(box, nodes)
    log_mass = -np.inf
    log_moments = np.full(3, -np.inf)
    count = 0
    for points, logweights in batches(xs, ws, batch_size):
        weighted = values(evaluate, points) + logweights
        log_mass = np.logaddexp(log_mass, logsumexp(weighted))
        log_moments = np.logaddexp(log_moments, logsumexp(weighted[:, None] + points, axis=0))
        count += len(points)
    means = np.exp(log_moments - log_mass)
    if not np.isfinite(log_mass) or not np.isfinite(means).all():
        raise ValueError("nonfinite integral or scale moment")
    return {"logZ": float(log_mass), "means": means.tolist(), "evaluations": count}


def sample_face(evaluate, box, axis, side, n, anchor, ell_reference, *, batch_size=32768):
    box = box_array(box)
    n = positive_int(n)
    if axis not in (0, 1, 2) or side not in (0, 1):
        raise ValueError("invalid face")
    anchor = np.asarray(anchor, dtype=float)
    if anchor.shape != (3,) or not np.isfinite(anchor).all() or not math.isfinite(ell_reference):
        raise ValueError("invalid fixed face reference")
    if not np.all((anchor >= box[:, 0]) & (anchor <= box[:, 1])):
        raise ValueError("anchor must be inside the box before projection")
    xs, ws = rule_arrays(box, [n] * 3)
    xs[axis], ws[axis] = np.array([box[axis, side]]), np.array([0.0])
    maximum = -np.inf
    count = 0
    for points, _ in batches(xs, ws, batch_size):
        maximum = max(maximum, float(values(evaluate, points).max()))
        count += len(points)
    projected = anchor.copy()
    projected[axis] = box[axis, side]
    maximum = max(maximum, float(values(evaluate, projected[None, :])[0]))
    return {"maximum_log_density": maximum, "deficit": maximum - ell_reference, "evaluations": count + 1}


def evaluate_anchor(evaluate, anchor, ell_reference):
    """One counted evaluation at the UNPROJECTED anchor, compared with the saved reference value. This checks that
    the supplied evaluator is consistent with the stored reference at one point (it does not establish target
    identity or any tail property); the face deficits are only meaningful when it holds."""
    anchor = np.asarray(anchor, dtype=float)
    if anchor.shape != (3,) or not np.isfinite(anchor).all() or not math.isfinite(ell_reference):
        raise ValueError("invalid anchor reference")
    ell_anchor = float(values(evaluate, anchor[None, :])[0])
    return {"ell_anchor": ell_anchor, "abs_error": abs(ell_anchor - ell_reference), "evaluations": 1}


def inventory(plan):
    """Build a complete fixed stage inventory before any evaluator call; the first stage is the anchor
    reference-consistency evaluation (revision v3r1)."""
    if not isinstance(plan.get("protocol_id"), str) or not plan["protocol_id"]:
        raise ValueError("protocol_id required")
    tol = plan.get("anchor_reference_tolerance")
    if isinstance(tol, bool) or not isinstance(tol, (int, float)) or not math.isfinite(tol) or tol <= 0:
        raise ValueError("anchor_reference_tolerance (finite, positive) required: a plan without the anchor reference gate is refused")
    if set(plan["domains"]) != {"B0", "B1"} or len(plan["nodes_per_axis"]) != 2:
        raise ValueError("two named nested domains and two rules required")
    boxes = {k: box_array(plan["domains"][k]) for k in ("B0", "B1")}
    inner, outer = boxes.values()
    if not np.all((outer[:, 0] <= inner[:, 0]) & (inner[:, 1] <= outer[:, 1])):
        raise ValueError("domains must be nested")
    rules = plan["nodes_per_axis"]
    for rule in rules:
        if len(rule) != 3:
            raise ValueError("three nodes per rule required")
        for n in rule:
            positive_int(n)
    if not all(hi > lo for lo, hi in zip(*rules)):
        raise ValueError("refined rule must increase every axis resolution")
    points = plan["cdf_points"]
    if not points or len({(p['scale'], p['label']) for p in points}) != len(points):
        raise ValueError("CDF points must be nonempty and uniquely identified")
    nface = positive_int(plan["face_nodes"])
    anchor = np.asarray(plan["face_mode_anchor"], dtype=float)
    if anchor.shape != (3,) or not np.isfinite(anchor).all() or not math.isfinite(plan['fixed_ell_reference']):
        raise ValueError("invalid face reference")
    if not np.all((anchor >= inner[:, 0]) & (anchor <= inner[:, 1])):
        raise ValueError("anchor must lie in the inner box")
    out = {"anchor:reference": dict(kind="anchor", point=anchor.tolist(), ell_reference=float(plan["fixed_ell_reference"]),
                                    tolerance=float(tol), evaluations=1)}
    for name, box in boxes.items():
        for level, nodes in enumerate(rules):
            base = f"{name}:r{level}"
            out[base + ":full"] = dict(kind="integral", box=box.tolist(), nodes=list(nodes), evaluations=math.prod(nodes))
            for index, point in enumerate(points):
                axis = NAMES.index(point['scale'])
                value = point['value']
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError("CDF value must be positive and finite")
                upper = math.log(value)
                if not box[axis, 0] < upper < box[axis, 1]:
                    raise ValueError("CDF point must be strictly inside both domains")
                truncated = box.copy()
                truncated[axis, 1] = upper
                out[f"{base}:cdf:{index}"] = dict(kind="integral", box=truncated.tolist(), nodes=list(nodes), evaluations=math.prod(nodes))
        for axis in range(3):
            for side in (0, 1):
                out[f"{name}:face:{axis}:{side}"] = dict(kind="face", box=box.tolist(), axis=axis, side=side, nodes=nface, evaluations=nface**2 + 1)
    return out


def plan_digest(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True, allow_nan=False).encode()).hexdigest()


def assemble(plan, stages, budget, *, stopped=None):
    expected = inventory(plan)
    conditions = {f"stage:{key}": None for key in expected}
    parsed = {}
    for key, spec in expected.items():
        if key not in stages:
            continue
        value = stages[key]
        try:
            n = value['evaluations']
            valid = not isinstance(n, (bool, np.bool_)) and isinstance(n, (int, np.integer)) and n == spec['evaluations']
            valid = valid and value['spec'] == spec and value['plan_sha256'] == plan_digest(plan)
            if spec['kind'] == 'integral':
                means = np.asarray(value['means'], dtype=float)
                valid = valid and np.isfinite(value['logZ']) and means.shape == (3,) and np.isfinite(means).all() and (means > 0).all()
            elif spec['kind'] == 'anchor':
                ell_anchor = value['ell_anchor']
                valid = valid and isinstance(ell_anchor, (float, int)) and not isinstance(ell_anchor, bool) and math.isfinite(ell_anchor)
                valid = valid and math.isfinite(value['abs_error']) and value['abs_error'] == abs(ell_anchor - plan['fixed_ell_reference'])
            else:
                valid = valid and np.isfinite(value['deficit']) and np.isfinite(value['maximum_log_density'])
                valid = valid and value['deficit'] == value['maximum_log_density'] - plan['fixed_ell_reference']
            conditions[f"stage:{key}"] = bool(valid)
            if valid:
                parsed[key] = value
        except (TypeError, ValueError, KeyError):
            conditions[f"stage:{key}"] = False
    conditions['no_unexpected_stages'] = not bool(stages.keys() - expected.keys())
    cdfs = {}
    for domain in ('B0', 'B1'):
        for level in (0, 1):
            base = f'{domain}:r{level}'
            for i in range(len(plan['cdf_points'])):
                key = f'{base}:cdf:{i}'
                conditions['cdf_range:' + key] = None
                if key in parsed and base + ':full' in parsed:
                    log_ratio = parsed[key]['logZ'] - parsed[base + ':full']['logZ']
                    valid = math.isfinite(log_ratio) and log_ratio <= 0
                    conditions['cdf_range:' + key] = valid
                    if valid:
                        cdfs[key] = math.exp(log_ratio)
    differences = {}
    pairs = [('resolution:B0', 'B0:r0', 'B0:r1'), ('resolution:B1', 'B1:r0', 'B1:r1'),
             ('domain:r0', 'B0:r0', 'B1:r0'), ('domain:r1', 'B0:r1', 'B1:r1')]
    for label, left, right in pairs:
        for metric, tolerance in [('logZ', LIMITS['logZ'])] + [(f'mean:{k}', LIMITS['mean']) for k in range(3)] + [(f'cdf:{k}', LIMITS['cdf']) for k in range(len(plan['cdf_points']))]:
            name = label + ':' + metric
            conditions[name] = None
            if metric.startswith('cdf:'):
                if left + ':' + metric not in cdfs or right + ':' + metric not in cdfs:
                    continue
                x, y = cdfs[left + ':' + metric], cdfs[right + ':' + metric]
            else:
                if left + ':full' not in parsed or right + ':full' not in parsed:
                    continue
                a, b = parsed[left + ':full'], parsed[right + ':full']
                x, y = (a['logZ'], b['logZ']) if metric == 'logZ' else (a['means'][int(metric[-1])], b['means'][int(metric[-1])])
            differences[name] = float(abs(x - y))
            conditions[name] = differences[name] <= tolerance
    for key, spec in expected.items():
        if spec['kind'] == 'face':
            conditions['face:' + key] = parsed[key]['deficit'] <= LIMITS['face'] if key in parsed else None
    a = expected['anchor:reference']
    conditions['anchor_reference_consistency'] = (parsed['anchor:reference']['abs_error'] <= a['tolerance']) if 'anchor:reference' in parsed else None
    count = sum(v['evaluations'] for v in expected.values())
    conditions['exact_complete_count'] = budget.get('confirmed') == count and budget.get('requested') == count
    report = acceptance_report(conditions, list(conditions))
    validity = run_is_valid(budget, stopped)
    accepted = bool(report['accepted'] and validity['valid'])
    return dict(protocol_id=plan['protocol_id'], plan_sha256=plan_digest(plan), stages=stages, budget=budget,
                required_conditions=list(conditions), conditions=conditions, stage_acceptance=report, run_validity=validity,
                accepted=accepted, status='diagnostics_passed' if accepted else 'inconclusive',
                exit_code=0 if accepted else (3 if not validity['valid'] else 4), stopped=stopped,
                planned_evaluations=count, differences=differences, cdfs=cdfs,
                comparison={}, interpretation='Numerical diagnostics only; the anchor check establishes reference consistency at one point, not target identity; no tail-mass bound, F7 acceptance, convergence or calibration claim.')


def run_protocol(evaluator, plan, *, checkpoint=None, batch_size=32768):
    """Execute only an explicitly supplied generic evaluator; never select a target.

The caller owns authorization and output-file handling. Partial stages and the
cumulative ledger survive an interruption. There is no restart/retry loop.
Preflight (binding implementation, 2026-09-13): if the single counted anchor
evaluation disagrees with the saved reference, nothing else is evaluated in
this unit; the recorded stage and its checkpoint survive, every remaining
condition stays unchecked, assembly still runs under the active guard, the
evaluator is released afterwards and the frozen final ledger is re-validated.
There is no second probe, retry or allowance.
"""
    try:
        plan = copy.deepcopy(plan)
        expected = inventory(plan)
        digest = plan_digest(plan)
        if evaluator.requested_total or evaluator.spent:
            raise ValueError('a fresh explicit unit is required; no reuse of partial numerical work')
    except BaseException:
        evaluator.release()
        raise
    stages, stopped = {}, None
    try:
        for key, spec in expected.items():
            if spec['kind'] == 'integral':
                result = integrate_box(evaluator, spec['box'], spec['nodes'], batch_size=batch_size)
            elif spec['kind'] == 'anchor':
                result = evaluate_anchor(evaluator, spec['point'], spec['ell_reference'])
            else:
                result = sample_face(evaluator, spec['box'], spec['axis'], spec['side'], spec['nodes'],
                                     plan['face_mode_anchor'], plan['fixed_ell_reference'], batch_size=batch_size)
            stages[key] = dict(result, spec=spec, plan_sha256=digest)
            if checkpoint is not None:
                checkpoint(copy.deepcopy(stages), evaluator.progress('in_progress'))
            if spec['kind'] == 'anchor' and not (math.isfinite(result['abs_error']) and result['abs_error'] <= spec['tolerance']):
                stopped = (f"preflight: anchor reference mismatch abs_error={result['abs_error']!r} > tolerance={spec['tolerance']!r}; "
                           "no further evaluation in this unit")
                break
    except (Exception, KeyboardInterrupt) as error:
        stopped = f'{type(error).__name__}: {error}'
    reason = 'stopped' if stopped else 'complete'
    try:
        output = assemble(plan, stages, evaluator.progress(reason), stopped=stopped)
    finally:
        evaluator.release()
    # Assembly time is part of this unit; use the final frozen ledger view.
    output['budget'] = evaluator.progress(reason)
    output['run_validity'] = run_is_valid(output['budget'], stopped)
    if not output['run_validity']['valid']:
        output.update(accepted=False, status='inconclusive', exit_code=3, comparison={})
    return output
