from types import SimpleNamespace
import arviz as az
import numpy as np
import pytest
import pymc as pm
import autobayes as ab


def fixture():
    model=SimpleNamespace(target_identity={'test':'snapshot'},pymc_model=object(),names={'y':'y'},
                          design=SimpleNamespace(n=2,y=np.array([0,1]),trials=np.array([2,2])),family='binomial')
    out=az.from_dict(prior={'a':np.ones((1,2))},prior_predictive={'y':np.ones((1,2,2))},observed_data={'y':np.array([0,1])})
    return model,out


def test_dispatch_and_snapshot(monkeypatch):
    m,out=fixture()
    def sample(**kw):
        assert kw==dict(draws=2,random_seed=3,model=m.pymc_model,return_inferencedata=True)
        m.target_identity={'changed':'during-call'}
        return out
    monkeypatch.setattr(pm,'sample_prior_predictive',sample)
    result=ab.prior_predictive(m,draws=np.int64(2),seed=np.int64(3))
    assert result.attrs['autobayes_built_target_identity_json']=='{"test": "snapshot"}'


@pytest.mark.parametrize('error',['nonfinite','support','rows','dimensions','missing'])
def test_invalid_output_refused(monkeypatch,error):
    m,out=fixture()
    if error=='nonfinite':out.prior.a.values[0,0]=np.inf
    if error=='support':out.prior_predictive.y.values[0,0,0]=3
    if error=='rows':out.observed_data.y.values[0]=1
    if error=='dimensions':m.design.n=3
    if error=='missing':del out.prior
    monkeypatch.setattr(pm,'sample_prior_predictive',lambda **kw:out)
    with pytest.raises(ValueError):ab.prior_predictive(m,draws=2)
    assert 'autobayes_generation_kind' not in out.attrs


def test_failure_is_not_retried(monkeypatch):
    m,_=fixture();calls=[]
    def fail(**kw):calls.append(kw);raise ValueError('rate too large')
    monkeypatch.setattr(pm,'sample_prior_predictive',fail)
    with pytest.raises(ValueError,match='no retry'):ab.prior_predictive(m,draws=2)
    assert len(calls)==1
