"""Target identity and comparison guards, using synthetic records; no MCMC."""
import copy
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import autobayes as ab
from autobayes.identity import _canonical


def frame():
    return pd.DataFrame({'y':[.2,1.1,-.3,2.], 'x':[0.,1.,2.,3.], 'z':['a','b','a','b'], 'g':['u','u','v','v']})


def build(df=None, *, center=True, reference='a', scale=1):
    return ab.gaussian('y ~ x + z + (1 | g)', frame() if df is None else df,
                      ab.Priors(intercept=ab.Normal(0,2),b=ab.Normal(0,scale),sd=ab.HalfNormal(1),sigma=ab.HalfNormal(1)),
                      center_predictors=center,categorical={'z':reference})


def record(m):
    return {'model':{'formula':m.spec.formula,'family':m.family,'priors':{},'target_identity':m.target_identity},
            'posterior':{'x':{'mean':.1,'mcse_mean':.05}}, 'execution':{'device_actual':'cpu'}}


def test_identical_construction_dtype_and_json():
    a=build();b=build(frame().astype({'x':'int64'}))
    assert a.target_identity==b.target_identity
    rec=record(a);out=ab.compare_records(rec,json.loads(json.dumps(record(b))))
    assert out['identity_status']=='matched' and out['agree']
    assert set(a.target_identity)=={'schema','sha256'} and len(a.target_identity['sha256'])==64


@pytest.mark.parametrize('change',['response','predictor','order','group','reference','centering','prior'])
def test_changed_targets_refused(change):
    a=build();df=frame();kw={}
    if change=='response':df.loc[0,'y']+=1
    elif change=='predictor':df.loc[0,'x']+=1
    elif change=='order':df=df.iloc[::-1].reset_index(drop=True)
    elif change=='group':df.loc[0,'g']='v'
    elif change=='reference':kw['reference']='b'
    elif change=='centering':kw['center']=False
    elif change=='prior':kw['scale']=2
    b=build(df,**kw)
    with pytest.raises(ValueError,match='different models or data'):
        ab.compare_records(record(a),record(b))


def test_fixed_nu_and_effective_binomial_rows():
    pr=ab.Priors(intercept=ab.Normal(0,2),b=ab.Normal(0,1),sigma=ab.HalfNormal(1))
    a=ab.student('y ~ x',frame(),pr,nu=4);b=ab.student('y ~ x',frame(),pr,nu=8)
    assert a.target_identity != b.target_identity
    df=pd.DataFrame({'k':[0,1,2],'n':[0,2,3],'x':[99.,1.,2.]})
    bp=ab.Priors(intercept=ab.Normal(0,2),b=ab.Normal(0,1))
    a=ab.binomial('k | trials(n) ~ x',df,bp)
    b=ab.binomial('k | trials(n) ~ x',df.iloc[1:].copy(),bp)
    assert a.target_identity==b.target_identity and a.dropped_rows==(0,) and b.dropped_rows==()
    c=ab.binomial('k | trials(n) ~ x',df.assign(n=[0,3,3]),bp)
    assert a.target_identity!=c.target_identity


def test_identity_frozen_after_build_and_defensive_copy():
    df=frame();m=build(df);before=m.target_identity
    df.loc[0,'y']=123
    m.data.loc[0,'y']=456
    exposed=m.target_identity;exposed['sha256']='0'*64
    assert m.target_identity==before


def test_execution_not_target_and_record_export(monkeypatch):
    import autobayes.session as session
    monkeypatch.setattr(session,'environment',lambda:{})
    monkeypatch.setattr(session,'_summary_rows',lambda *args:{'x':{'mean':.1,'mcse_mean':.05}})
    m=build();fit=SimpleNamespace(target_identity=m.target_identity,idata=None,backend='synthetic',device_requested='cpu',device_actual='cpu',platform='cpu',dtype='float64',chain_method='none',chains=0,draws=0,tune=0,target_accept=.9,seed=1,wall_seconds=0,notes=['synthetic record; not a fit'])
    panel={'values':{},'status':{},'overall':'not evaluated'}
    a=ab.fit_record(m,fit,panel=panel)
    fit.backend='synthetic_other';fit.seed=9
    b=ab.fit_record(m,fit,panel=panel)
    assert a['model']['target_identity']==m.target_identity==b['model']['target_identity']
    assert ab.compare_records(a,b)['identity_status']=='matched'
    assert 'X_model' not in json.dumps(a) and 'values' not in a['model']['target_identity']


def test_legacy_requires_opt_in_and_warns():
    a=record(build());b=copy.deepcopy(a);del b['model']['target_identity']
    with pytest.raises(ValueError,match='missing'):ab.compare_records(a,b)
    c=ab.compare_records(a,b,allow_legacy=True)
    assert c['identity_status']=='unverified_legacy' and 'WARNING' in ab.format_comparison(c)
    del a['model']['target_identity']
    with pytest.raises(ValueError,match='missing'):ab.compare_records(a,b)
    assert ab.compare_records(a,b,allow_legacy=True)['identity_status']=='unverified_legacy'


@pytest.mark.parametrize('bad',[{}, {'schema':'future','sha256':'0'*64},{'schema':'autobayes-target-v1','sha256':'bad'}])
def test_malformed_identity_never_bypassed(bad):
    a=record(build());b=copy.deepcopy(a);b['model']['target_identity']=bad
    with pytest.raises(ValueError,match='invalid or unsupported'):ab.compare_records(a,b,allow_legacy=True)


def test_canonical_numeric_contract():
    assert _canonical(1)==_canonical(1.)==_canonical(np.int64(1))
    assert _canonical(-0.)==_canonical(0.)
    assert _canonical(np.ones((2,1)))!=_canonical(np.ones((1,2)))
    for v in [np.inf,np.nan]:
        with pytest.raises(ValueError,match='finite'):_canonical(v)
