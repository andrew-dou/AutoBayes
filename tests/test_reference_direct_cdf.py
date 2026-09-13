"""Analytic/stub integration tests only: no F7 target, model library or sampler."""
import copy
import json
import math

import numpy as np
import pytest

from tests import reference_direct_cdf as q
from tests.quadrature_budget import BudgetedEvaluator


def plan():
    return dict(protocol_id='toy-only', domains={'B0': [[-1, 1]] * 3, 'B1': [[-1.1, 1.1]] * 3},
                nodes_per_axis=[[16, 20, 16], [24, 28, 24]], cdf_points=[dict(scale='sigma', label='zero', value=1.0)],
                face_nodes=5, face_mode_anchor=[0, 0, 0], fixed_ell_reference=0.0, anchor_reference_tolerance=1e-6)


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
        elif spec['kind'] == 'anchor':
            value.update(ell_anchor=0., abs_error=0.)
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
    assert len(checkpoints) == len(q.inventory(p)) == 21 and checkpoints[0] == (1, 1)     # the anchor stage runs first: one point
    assert out['stages']['anchor:reference']['evaluations'] == 1 and out['conditions']['anchor_reference_consistency'] is True
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
    assert out['comparison']=={} and list(out['stages']) == ['anchor:reference']          # the one-point anchor fitted; the first integral did not
    assert out['stage_acceptance']['unchecked']


def test_interrupt_keeps_reservation_and_returns_inconclusive(tmp_path):
    def fail(x): raise KeyboardInterrupt('fixture interruption')
    ev=evaluator(tmp_path,fail,100,size=4)
    out=q.run_protocol(ev,plan(),batch_size=5)
    assert out['budget']['confirmed']==0 and out['budget']['reserved_uncertain']==1      # the interrupted work is the one-point anchor stage
    assert 'anchor:reference' not in out['stages'] and out['conditions']['anchor_reference_consistency'] is None
    assert out['exit_code']==3 and not out['accepted'] and ev._closed
    assert 'KeyboardInterrupt' in out['stopped'] and out['comparison']=={}


def test_complete_stubs_pass_all_explicit_conditions():
    p=plan();out=q.assemble(p,stub_stages(p),valid_budget(p))
    assert out['accepted'] and out['exit_code']==0
    assert len(out['differences']) == 4*(1+3+1)
    assert set(out['conditions'])==set(out['required_conditions'])
    assert all(v is True for v in out['conditions'].values())
    assert out['comparison']=={}


@pytest.mark.parametrize('fault',['missing','nan','metadata','count','cdf_over_one','tail','extra','stopped','budget',
                                   'anchor_missing','anchor_nan','anchor_offset','anchor_count','anchor_inconsistent','anchor_string'])
def test_fail_closed_assembly(fault):
    p=plan();stages=stub_stages(p);budget=valid_budget(p);stopped=None
    key='B0:r0:full'
    if fault=='anchor_missing': del stages['anchor:reference']
    if fault=='anchor_nan': stages['anchor:reference'].update(ell_anchor=float('nan'),abs_error=float('nan'))
    if fault=='anchor_offset': stages['anchor:reference'].update(ell_anchor=-40.,abs_error=40.)           # shifted evaluator: exceeds 1e-6
    if fault=='anchor_count': stages['anchor:reference']['evaluations']=2
    if fault=='anchor_inconsistent': stages['anchor:reference'].update(ell_anchor=-40.,abs_error=0.)       # abs_error not recomputable from ell_anchor
    if fault=='anchor_string': stages['anchor:reference'].update(ell_anchor='0.0',abs_error=0.)
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
    if fault.startswith('anchor'):
        assert out['conditions']['anchor_reference_consistency'] is not True


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


def test_constant_offset_target_is_rejected_by_the_anchor_gate(tmp_path):
    """The reviewed F1 defect: a narrow Gaussian whose faces truly fail, and the same density shifted by -40. Before
    the anchor gate the shifted run was accepted with identical CDFs; the gate fails closed, and since the binding
    implementation the failed counted preflight also stops the unit after that single evaluation."""
    p=plan();p['nodes_per_axis']=[[8,10,8],[12,14,12]];p['face_nodes']=3
    count=sum(s['evaluations'] for s in q.inventory(p).values())
    true=lambda x:-np.square(x/.15).sum(axis=1)/2
    shifted=lambda x:true(x)-40.
    seen={'true':[],'shifted':[]};outs={}
    for name,fun in (('true',true),('shifted',shifted)):
        ev=BudgetedEvaluator(lambda x,f=fun,n=name:(seen[n].append(len(x)) or f(x)),cap_evaluations=count+1,unit_deadline_seconds=60,ledger_path=tmp_path/f'{name}.json',attempt='toy',batch_size=4096)
        outs[name]=q.run_protocol(ev,p,batch_size=4096)
    assert outs['true']['conditions']['anchor_reference_consistency'] is True and outs['true']['stages']['anchor:reference']['abs_error']==0.
    assert sum(seen['true'])==count and outs['true']['cdfs'] and outs['true']['run_validity']['valid']        # the consistent target runs every stage
    s=outs['shifted']
    assert s['conditions']['anchor_reference_consistency'] is False and s['stages']['anchor:reference']['abs_error']==40.
    assert seen['shifted']==[1] and s['budget']['requested']==s['budget']['confirmed']==1 and s['budget']['reserved_uncertain']==0
    assert list(s['stages'])==['anchor:reference'] and s['stopped'].startswith('preflight: anchor reference mismatch')
    assert s['exit_code']==3 and not s['accepted'] and not s['run_validity']['valid'] and s['cdfs']=={} and s['comparison']=={}
    assert 'anchor_reference_consistency' in s['stage_acceptance']['failed']
    unchecked=set(s['stage_acceptance']['unchecked'])
    assert all(k in unchecked for k in s['conditions'] if k.startswith(('stage:B','cdf_range:','resolution:','domain:','face:')))


def test_pure_assembly_offset_counterexample_still_fails_on_the_anchor_condition_alone():
    """Retained evidence for the reviewed counterexample without the early stop: a complete run whose ordinary stages
    all came from a constant-shifted target has identical ratios, and only the anchor condition rejects it."""
    p=plan();good=stub_stages(p);bad=copy.deepcopy(good);bad['anchor:reference'].update(ell_anchor=-40.,abs_error=40.)
    ok=q.assemble(p,good,valid_budget(p));out=q.assemble(p,bad,valid_budget(p))
    assert ok['accepted'] and ok['exit_code']==0 and not out['accepted'] and out['exit_code']==4 and out['run_validity']['valid']
    assert out['stage_acceptance']['failed']==['anchor_reference_consistency'] and out['cdfs']==ok['cdfs'] and out['cdfs']


def test_failed_anchor_preflight_stops_after_exactly_one_counted_evaluation(tmp_path):
    p=plan();p['nodes_per_axis']=[[2,3,2],[3,4,3]];p['face_nodes']=2
    seen=[];checkpoints=[]
    ev=BudgetedEvaluator(lambda x:(seen.append(len(x)) or np.full(len(x),-40.)),cap_evaluations=10_000,unit_deadline_seconds=60,ledger_path=tmp_path/'u.json',attempt='toy')
    out=q.run_protocol(ev,p,checkpoint=lambda stages,b:checkpoints.append((list(stages),dict(b))))
    assert seen==[1] and out['budget']['requested']==1 and out['budget']['confirmed']==1 and out['budget']['reserved_uncertain']==0
    assert len(checkpoints)==1 and checkpoints[0][0]==['anchor:reference'] and checkpoints[0][1]['confirmed']==1     # the preflight checkpoint survives
    assert out['stopped'].startswith('preflight: anchor reference mismatch') and out['exit_code']==3 and not out['accepted']
    assert out['conditions']['anchor_reference_consistency'] is False and out['conditions']['stage:anchor:reference'] is True
    assert out['stage_acceptance']['failed']==['anchor_reference_consistency','exact_complete_count']
    assert all(out['conditions'][k] is None for k in out['conditions'] if k.startswith(('stage:B','cdf_range:','resolution:','domain:','face:')))
    assert out['comparison']=={} and ev._closed and json.loads((tmp_path/'u.json').read_text())['released']
    # no second probe or allowance in the same unit: another attempt on this ledger is refused before any evaluation
    ev2=BudgetedEvaluator(lambda x:(seen.append(len(x)) or np.zeros(len(x))),cap_evaluations=10_000,unit_deadline_seconds=60,ledger_path=tmp_path/'u.json',attempt='toy-2')
    with pytest.raises(ValueError,match='fresh explicit unit'):q.run_protocol(ev2,p)
    assert seen==[1] and ev2._closed


def test_nonfinite_or_malformed_anchor_value_stops_with_one_reservation(tmp_path):
    nan=lambda x:np.full(len(x),np.nan);wide=lambda x:np.ones((len(x),2))
    for name,bad,confirmed,uncertain in (('nan',nan,1,0),('shape',wide,0,1)):
        seen=[]
        ev=BudgetedEvaluator(lambda x,f=bad:(seen.append(len(x)) or f(x)),cap_evaluations=10_000,unit_deadline_seconds=60,ledger_path=tmp_path/f'{name}.json',attempt='toy')
        out=q.run_protocol(ev,plan())
        assert seen==[1] and out['budget']['requested']==1 and out['budget']['confirmed']==confirmed and out['budget']['reserved_uncertain']==uncertain
        assert out['stages']=={} and 'ValueError' in out['stopped'] and out['conditions']['anchor_reference_consistency'] is None
        assert out['exit_code']==3 and not out['accepted'] and ev._closed


def test_anchor_tolerance_boundary_is_inclusive(tmp_path):
    p=plan();p['nodes_per_axis']=[[2,3,2],[3,4,3]];p['face_nodes']=2
    count=sum(s['evaluations'] for s in q.inventory(p).values())
    for offset,expected in ((1e-6,count),(1.5e-6,1)):
        seen=[]
        ev=BudgetedEvaluator(lambda x,o=offset:(seen.append(len(x)) or np.full(len(x),o)),cap_evaluations=count+1,unit_deadline_seconds=60,ledger_path=tmp_path/f'{offset}.json',attempt='toy')
        out=q.run_protocol(ev,p)
        assert sum(seen)==out['budget']['confirmed']==expected and (out['stopped'] is None)==(expected==count)


def test_anchor_is_the_first_stage_and_exactly_one_point(tmp_path):
    seen=[]
    p=plan();p['nodes_per_axis']=[[2,3,2],[3,4,3]];p['face_nodes']=2
    ev=BudgetedEvaluator(lambda x:(seen.append(len(x)) or np.zeros(len(x))),cap_evaluations=10_000,unit_deadline_seconds=60,ledger_path=tmp_path/'u.json',attempt='toy')
    out=q.run_protocol(ev,p)
    assert seen[0]==1 and list(q.inventory(p))[0]=='anchor:reference' and out['stages']['anchor:reference']['evaluations']==1
    assert out['planned_evaluations']==sum(seen)==out['budget']['confirmed']


def test_cap_of_one_evaluates_only_the_anchor_then_stops(tmp_path):
    seen=[]
    ev=BudgetedEvaluator(lambda x:(seen.append(len(x)) or np.zeros(len(x))),cap_evaluations=1,unit_deadline_seconds=60,ledger_path=tmp_path/'u.json',attempt='toy',batch_size=4)
    out=q.run_protocol(ev,plan(),batch_size=5)
    assert seen==[1] and out['budget']['confirmed']==1 and list(out['stages'])==['anchor:reference']
    assert out['conditions']['anchor_reference_consistency'] is True and not out['accepted'] and out['exit_code']==3


def test_plan_without_anchor_tolerance_or_with_invalid_anchor_is_refused():
    p=plan();del p['anchor_reference_tolerance']
    with pytest.raises(ValueError,match='anchor_reference_tolerance'):q.inventory(p)
    for bad in (0,-1e-6,float('nan'),True):
        p=plan();p['anchor_reference_tolerance']=bad
        with pytest.raises(ValueError):q.inventory(p)
    with pytest.raises(ValueError):q.evaluate_anchor(lambda x:np.zeros(len(x)),[0,float('nan'),0],0.)
    with pytest.raises(ValueError):q.evaluate_anchor(lambda x:np.full(len(x),np.nan),[0,0,0],0.)


def test_frozen_plans_v3_history_and_v3r1_inventory_without_evaluating_target():
    import hashlib, json
    from pathlib import Path
    root=Path(__file__).resolve().parent.parent/'docs/m3b1-evidence'
    v3=json.loads((root/'direct_cdf_v3_protocol.json').read_text())
    with pytest.raises(ValueError,match='anchor_reference_tolerance'):q.inventory(v3)            # v3 (history) has no anchor gate: refused
    v3r1=json.loads((root/'direct_cdf_v3r1_protocol.json').read_text())
    inv=q.inventory(v3r1)
    assert len(inv)==45 and sum(s['evaluations'] for s in inv.values())==12_038_157==v3r1['planned_evaluations']
    assert sum(s['evaluations'] for s in inv.values() if s['kind']=='face')==110_604 and inv['anchor:reference']['evaluations']==1
    assert v3r1['revised_from_protocol_sha256']==hashlib.sha256((root/'direct_cdf_v3_protocol.json').read_bytes()).hexdigest()
    assert v3r1['proposed_unit_cap']==12_050_000 and v3r1['actual_posterior_evaluations']==0 and len(v3r1['cdf_points'])==7
    for k in ('domains','nodes_per_axis','cdf_points','face_nodes','face_mode_anchor','fixed_ell_reference'):
        assert v3r1[k]==v3[k]
