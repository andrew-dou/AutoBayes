"""Analytic/stub integration tests only: no F7 target, model library or sampler."""
import copy
import math

import numpy as np
import pytest

from tests import reference_direct_cdf as q
from tests.quadrature_budget import BudgetedEvaluator


def plan():
    return dict(protocol_id='toy-only', domains={'B0': [[-1, 1]] * 3, 'B1': [[-1.1, 1.1]] * 3},
                nodes_per_axis=[[16, 20, 16], [24, 28, 24]], cdf_points=[dict(scale='sigma', label='zero', value=1.0)],
                face_nodes=5, face_mode_anchor=[0, 0, 0], fixed_ell_reference=0.0)


def evaluator(tmp_path, fun, cap, *, size=32768, clock=None):
    kw = dict(cap_evaluations=cap, unit_deadline_seconds=60, ledger_path=tmp_path/'unit.json', attempt='toy', batch_size=size)
    if clock is not None:
        kw['clock'] = clock
    return BudgetedEvaluator(fun, **kw)


def stub_stages(p):
    out = {}
    for key, spec in q.inventory(p).items():
        value = dict(evaluations=spec['evaluations'], spec=spec, plan_sha256=q.plan_digest(p))
        if spec['kind'] == 'face':
            value.update(maximum_log_density=-30., deficit=-30.)
        else:
            value.update(logZ=math.log(.5) if ':cdf:' in key else 0., means=[1.,1.,1.])
        out[key] = value
    return out


def valid_budget(p):
    count = sum(s['evaluations'] for s in q.inventory(p).values())
    return dict(reason='complete', requested=count, confirmed=count, reserved_uncertain=0, remaining=1,
                cap=count+1, seconds_spent=1., unit_deadline_seconds=60.)


def test_uniform_shifted_box_scale_means_and_large_log_offset():
    box = np.array([[-.7, .4], [-1., .6], [.2, .9]])
    calls = []
    def fun(x):
        calls.append(len(x))
        return np.full(len(x), 1200.)
    actual = q.integrate_box(fun, box, [8, 9, 10], batch_size=17)
    widths = box[:,1]-box[:,0]
    expected = (np.exp(box[:,1])-np.exp(box[:,0]))/widths
    assert actual['logZ'] == pytest.approx(1200.+np.log(widths).sum(), abs=1e-11)
    np.testing.assert_allclose(actual['means'], expected, rtol=1e-11)
    assert sum(calls) == actual['evaluations'] == 720 and max(calls) <= 17


def test_nonseparable_polynomial_truncated_cdf_and_remapped_weights():
    # p(x,y,z)=1+.2xy+.1z on [0,1]^3; exact mass 1.1.
    def fun(x): return np.log(1.+.2*x[:,0]*x[:,1]+.1*x[:,2])
    full = q.integrate_box(fun, [[0,1]]*3, [3,4,5], batch_size=11)
    t=.37
    trunc = q.integrate_box(fun, [[0,t],[0,1],[0,1]], [3,4,5], batch_size=11)
    assert math.exp(full['logZ']) == pytest.approx(1.1, abs=1e-14)
    assert math.exp(trunc['logZ']-full['logZ']) == pytest.approx((1.05*t+.05*t*t)/1.1, abs=1e-14)


def test_projected_face_anchor_is_counted_and_prevents_even_grid_missing_peak():
    count=0
    def fun(x):
        nonlocal count
        count += len(x)
        return -np.square(x/.1).sum(axis=1)/2
    actual=q.sample_face(fun,[[-1,1]]*3,1,0,4,[0,0,0],0.,batch_size=5)
    assert actual['deficit'] == pytest.approx(-50.)
    assert count == actual['evaluations'] == 17


def test_small_tensor_protocol_exact_count_and_analytic_cdf(tmp_path):
    p=plan(); count=sum(s['evaluations'] for s in q.inventory(p).values())
    ev=evaluator(tmp_path,lambda x:-np.square(x/.2).sum(axis=1)/2,count+1)
    checkpoints=[]
    out=q.run_protocol(ev,p,checkpoint=lambda stages,b:checkpoints.append((len(stages),b['confirmed'])),batch_size=4096)
    assert out['budget']['confirmed'] == out['planned_evaluations'] == count
    assert len(checkpoints) == len(q.inventory(p)) == 20
    assert max(abs(c-.5) for c in out['cdfs'].values()) < 1e-4
    # Finite-domain normalizer/CDF calculations are accurate, but these toy faces
    # deliberately fail the separate tail diagnostic. Never report full acceptance.
    assert not out['accepted'] and any(k.startswith('face:') for k in out['stage_acceptance']['failed'])
    assert out['run_validity']['valid'] and out['comparison']=={}
    assert out['status'] == ('diagnostics_passed' if out['accepted'] else 'inconclusive')
    assert ev._closed


def test_cap_stops_partial_stage_before_any_extra_call(tmp_path):
    seen=[]
    ev=evaluator(tmp_path,lambda x:(seen.append(len(x)) or np.zeros(len(x))),7,size=4)
    out=q.run_protocol(ev,plan(),batch_size=5)
    assert sum(seen)==out['budget']['confirmed']==7
    assert out['budget']['reserved_uncertain']==0 and out['exit_code']==3 and not out['accepted']
    assert out['comparison']=={} and not out['stages']
    assert out['stage_acceptance']['unchecked']


def test_interrupt_keeps_reservation_and_returns_inconclusive(tmp_path):
    def fail(x): raise KeyboardInterrupt('fixture interruption')
    ev=evaluator(tmp_path,fail,100,size=4)
    out=q.run_protocol(ev,plan(),batch_size=5)
    assert out['budget']['confirmed']==0 and out['budget']['reserved_uncertain']==4
    assert out['exit_code']==3 and not out['accepted'] and ev._closed
    assert 'KeyboardInterrupt' in out['stopped'] and out['comparison']=={}


def test_complete_stubs_pass_all_explicit_conditions():
    p=plan();out=q.assemble(p,stub_stages(p),valid_budget(p))
    assert out['accepted'] and out['exit_code']==0
    assert len(out['differences']) == 4*(1+3+1)
    assert set(out['conditions'])==set(out['required_conditions'])
    assert all(v is True for v in out['conditions'].values())
    assert out['comparison']=={}


@pytest.mark.parametrize('fault',['missing','nan','metadata','count','cdf_over_one','tail','extra','stopped','budget'])
def test_fail_closed_assembly(fault):
    p=plan();stages=stub_stages(p);budget=valid_budget(p);stopped=None
    key='B0:r0:full'
    if fault=='missing': del stages[key]
    if fault=='nan': stages[key]['logZ']=float('nan')
    if fault=='metadata': stages[key]['spec']=dict(stages[key]['spec'],nodes=[2,2,2])
    if fault=='count': stages[key]['evaluations']-=1
    if fault=='cdf_over_one': stages['B0:r0:cdf:0']['logZ']=1.
    if fault=='tail': stages['B0:face:0:0'].update(maximum_log_density=-20.,deficit=-20.)
    if fault=='extra': stages['unplanned']=copy.deepcopy(stages[key])
    if fault=='stopped': stopped='fixture stop'
    if fault=='budget': budget['confirmed']-=1;budget['remaining']+=1
    out=q.assemble(p,stages,budget,stopped=stopped)
    assert not out['accepted'] and out['exit_code'] in (3,4) and out['comparison']=={}


@pytest.mark.parametrize('bad',[0,True,1.5])
def test_invalid_resolution_refused_before_evaluation(bad):
    p=plan();p['nodes_per_axis'][0][0]=bad
    with pytest.raises(ValueError):q.inventory(p)


def test_wrong_output_shape_or_nonfinite_values_refused():
    for fun in [lambda x:np.ones((len(x),1)),lambda x:np.full(len(x),np.nan)]:
        with pytest.raises(ValueError):q.integrate_box(fun,[[0,1]]*3,[2,2,2])


def test_changed_plan_cannot_reuse_stage_flags():
    p=plan();stages=stub_stages(p);p['cdf_points'][0]['value']=1.01
    out=q.assemble(p,stages,valid_budget(p))
    assert not out['accepted'] and all(not v for k,v in out['conditions'].items() if k.startswith('stage:'))


def test_assembly_time_is_included_in_final_deadline(tmp_path, monkeypatch):
    p=plan();p['nodes_per_axis']=[[2,3,2],[3,4,3]];p['face_nodes']=2
    count=sum(x['evaluations'] for x in q.inventory(p).values());clock=[0.]
    ev=evaluator(tmp_path,lambda x:np.zeros(len(x)),count+1,clock=lambda:clock[0])
    original=q.assemble
    def delayed(*args,**kwargs):
        out=original(*args,**kwargs)
        clock[0]=61.
        return out
    monkeypatch.setattr(q,'assemble',delayed)
    out=q.run_protocol(ev,p)
    assert out['budget']['seconds_spent']==61. and not out['run_validity']['valid']
    assert out['exit_code']==3 and not out['accepted'] and ev._closed


def test_frozen_v3_plan_has_complete_inventory_without_evaluating_target():
    import json
    from pathlib import Path
    p=json.loads((Path(__file__).resolve().parent.parent/'docs/m3b1-evidence/direct_cdf_v3_protocol.json').read_text())
    inv=q.inventory(p)
    assert len(inv)==44 and sum(s['evaluations'] for s in inv.values())==12_038_156
    assert sum(s['evaluations'] for s in inv.values() if s['kind']=='face')==110_604
    assert len(p['cdf_points'])==7 and p['actual_posterior_evaluations']==0
