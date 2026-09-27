"""Fit provenance with stubbed samplers and fixed draws; no MCMC."""
import json
import arviz as az
import numpy as np
import pandas as pd
import pytest
import autobayes as ab
from autobayes.identity import FIT_IDENTITY_ATTR, check_fit_binding


def model(delta=0):
    return ab.gaussian('y ~ x',pd.DataFrame({'y':[.1+delta,.2,.4],'x':[0.,1.,2.]}),
                       ab.Priors(intercept=ab.Normal(0,2),b=ab.Normal(0,1),sigma=ab.HalfNormal(1)))


def draws():
    return az.from_dict(posterior={'Intercept':np.full((2,4),.2),'x':np.full((2,4),.3),'sigma':np.ones((2,4))})


def stub_fit(m,monkeypatch,backend='pymc'):
    monkeypatch.setattr(m.bambi_model,'fit',lambda **kwargs:draws())
    if backend=='numpyro':
        import jax
        monkeypatch.setattr('autobayes.inference.resolve_jax_device',lambda *args:(jax.devices('cpu')[0],'sequential'))
    return m.fit(backend=backend,chains=2,draws=4,tune=0,seed=12)


@pytest.mark.parametrize('backend',['pymc','numpyro'])
def test_dispatch_capture_and_wrong_model_refusal(monkeypatch,backend):
    m=model();fit=stub_fit(m,monkeypatch,backend)
    assert fit.target_identity==m.target_identity
    assert json.loads(fit.idata.attrs[FIT_IDENTITY_ATTR])==m.target_identity
    assert check_fit_binding(model(),fit)=='matched'
    assert model().predict(fit,kind='mean').attrs['fit_binding_status']=='matched'
    with pytest.raises(ValueError,match='different models or data'):model(1).predict(fit,kind='mean')
    with pytest.raises(ValueError,match='different models or data'):ab.fit_record(model(1),fit)
    exposed=fit.target_identity;exposed['sha256']='0'*64
    assert fit.target_identity==m.target_identity


def test_capture_precedes_sampler(monkeypatch):
    m=model();before=m.target_identity
    def run(**kwargs):
        m._target_identity={'schema':before['schema'],'sha256':'0'*64}
        return draws()
    monkeypatch.setattr(m.bambi_model,'fit',run)
    fit=m.fit(chains=2,draws=4,tune=0)
    assert fit.target_identity==before
    with pytest.raises(ValueError,match='different models or data'):check_fit_binding(m,fit)


def test_saved_draws_and_conflicting_metadata(monkeypatch,tmp_path):
    m=model();fit=stub_fit(m,monkeypatch)
    path=tmp_path/'draws.nc';fit.idata.to_netcdf(path)
    loaded=az.from_netcdf(path)
    try:
        assert check_fit_binding(m,loaded)=='matched'
        assert m.predict(loaded,kind='mean').attrs['fit_binding_status']=='matched'
    finally:loaded.close()
    fit.idata.attrs[FIT_IDENTITY_ATTR]=json.dumps(model(1).target_identity)
    with pytest.raises(ValueError,match='different models or data'):check_fit_binding(m,fit)


@pytest.mark.parametrize('bad',['not json','null','{}',{'schema':'not JSON text'}])
def test_malformed_metadata_refused(bad):
    m=model();idata=draws();idata.attrs[FIT_IDENTITY_ATTR]=bad
    with pytest.raises(ValueError):m.predict(idata,kind='mean')


def test_unbound_draws_and_explicit_legacy_records(monkeypatch):
    m=model();idata=draws()
    assert m.predict(idata,kind='mean').attrs['fit_binding_status']=='unverified'
    fit=ab.FitResult(idata,'synthetic','cpu','cpu','cpu','float64',None,2,4,0,.9,0)
    import autobayes.session as session
    monkeypatch.setattr(session,'environment',lambda:{})
    monkeypatch.setattr(session,'_summary_rows',lambda *args:{})
    panel={'values':{},'status':{},'overall':'not evaluated'}
    with pytest.raises(ValueError,match='binding is missing'):ab.fit_record(m,fit,panel=panel)
    rec=ab.fit_record(m,fit,panel=panel,allow_unverified=True)
    assert 'target_identity' not in rec['model']
    assert rec['model']['fit_binding_status']=='unverified'
    with pytest.raises(ValueError,match='missing'):ab.compare_records(rec,rec)
    assert ab.compare_records(rec,rec,allow_legacy=True)['identity_status']=='unverified_legacy'


def test_explicit_null_result_binding_is_not_legacy(monkeypatch):
    m=model();fit=stub_fit(m,monkeypatch)
    fit._target_identity_json='null'
    with pytest.raises(ValueError,match='invalid fit-result'):m.predict(fit,kind='mean')
