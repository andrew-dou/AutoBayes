"""Poisson original-input numerical and prediction checks, no MCMC."""
from types import SimpleNamespace
import json
import arviz as az
import numpy as np
import pandas as pd
import pytest
from scipy.special import gammaln
from scipy.stats import norm, halfnorm
import autobayes as ab


def data():
    return pd.DataFrame({'x':[-1.,.5,2.,3.], 'y':[0,1,2,4], 'g':['a','a','b','b']})


def priors(group=False):
    return ab.Priors(intercept=ab.Normal(0,2),b=ab.Normal(0,1),**({'sd':ab.HalfNormal(1)} if group else {}))


@pytest.mark.parametrize('values', [[-1,1],[.5,1],[True,False],[np.inf,1],[np.nan,1],[1+1j,1],['1','2'],[1000001,1]])
def test_invalid_counts(values):
    with pytest.raises(ab.FormulaError):
        ab.poisson('y ~ 1',pd.DataFrame({'y':values}),ab.Priors(intercept=ab.Normal(0,2),b={}))


def test_zeros_and_irrelevant_priors():
    df=data();df.y=0
    m=ab.poisson('y ~ x',df,priors())
    assert m.design.n==4 and np.array_equal(m.design.y,[0,0,0,0]) and not m.dropped_rows
    assert 'all observed counts are zero' in m.warnings[0]
    with pytest.raises(ab.PriorError,match='sigma'):
        ab.poisson('y ~ x',data(),ab.Priors(intercept=ab.Normal(0,2),b=ab.Normal(0,1),sigma=ab.HalfNormal(1)))
    with pytest.raises(ValueError,match='nu'):
        ab.Model('y ~ x',data(),priors(),family='poisson',nu=4)
    with pytest.raises(ab.FormulaError,match='binomial'):
        ab.poisson('y | trials(n) ~ x',data().assign(n=5),priors())


@pytest.mark.parametrize('eta',[-30.,0.,np.log(1e6),30.])
def test_count_boundaries_density_gradient_jax(eta):
    y=np.array([0,1,1000000]);m=ab.poisson('y ~ 1',pd.DataFrame({'y':y}),ab.Priors(intercept=ab.Normal(0,2),b={}))
    p=m.value_point(intercept=eta,b=[])
    ref=(y*eta-np.exp(eta)-gammaln(y+1)).sum()+norm.logpdf(eta,0,2)
    grad=(y-np.exp(eta)).sum()-eta/4
    np.testing.assert_allclose(m.pymc_model.compile_logp()(p),ref,rtol=1e-10,atol=1e-7)
    np.testing.assert_allclose(m.pymc_model.compile_dlogp()(p),[grad],rtol=1e-10,atol=1e-7)
    from pymc.sampling.jax import get_jaxified_logp
    import jax
    j=get_jaxified_logp(m.pymc_model)
    args=[p[v.name] for v in m.pymc_model.value_vars]
    np.testing.assert_allclose(j(args),ref,rtol=1e-10,atol=1e-7)
    np.testing.assert_allclose(jax.grad(lambda q:j(q))(args),[grad],rtol=1e-10,atol=1e-7)


@pytest.mark.parametrize('intercept',[True,False])
def test_original_hierarchical_interaction_density(intercept):
    df=data().assign(z=['a','b','a','b'],h=['u','v','u','v'])
    pr=ab.Priors(intercept=ab.Normal(0,2) if intercept else None,b=ab.Normal(0,1),sd={'g':ab.HalfNormal(1),'h':ab.HalfNormal(1)})
    m=ab.poisson('y ~ '+('' if intercept else '0 + ')+'x*z + (1 | g) + (1 | h)',df,pr,categorical={'z':'a'})
    p=m.pymc_model.initial_point();p['x']=np.array(-.2);p['z']=np.array([.4]) if intercept else np.array([.1,.4]);p['x:z']=np.array([.3])
    x=df.x.to_numpy();z=(df.z=='b').to_numpy().astype(float);xz=x*z
    eta=-.2*x+np.where(z,.4,0 if intercept else .1)+.3*xz
    ref=norm.logpdf(-.2)+norm.logpdf(p['z']).sum()+norm.logpdf(.3)
    if intercept:
        p['Intercept']=np.array(.3);eta+=.3-(-.2*x.mean()+.4*z.mean()+.3*xz.mean());ref+=norm.logpdf(.3,0,2)
    for g,sd,offset in [('g',.5,np.array([.2,-.2])),('h',.7,np.array([-.1,.1]))]:
        p[f'1|{g}_sigma_log__']=np.array(np.log(sd));p[f'1|{g}_offset']=offset
        levels=sorted(df[g].unique());eta+=sd*np.array([offset[levels.index(v)] for v in df[g]])
        ref+=halfnorm.logpdf(sd)+np.log(sd)+norm.logpdf(offset).sum()
    ref+=(df.y*eta-np.exp(eta)-gammaln(df.y+1)).sum()
    np.testing.assert_allclose(m.pymc_model.compile_logp()(p),ref,atol=1e-9)
    assert m.design.centered==intercept and m.link=='log'
    assert 'P(y_i = 1)' not in m.describe() and 'Poisson(mu_i)' in m.describe()


def draws(group=False,a=.3):
    p={'Intercept':np.full((2,4),a),'x':np.full((2,4),-.2)}
    if group:p.update({'1|g_sigma':np.full((2,4),.5),'1|g':np.broadcast_to([.1,-.1],(2,4,2))})
    return az.from_dict(posterior=p,coords={'g__factor_dim':['a','b']},dims={'1|g':['g__factor_dim']})


@pytest.mark.parametrize('policy',['population','resample_observed'])
def test_seeded_new_group_predictions(policy):
    m=ab.poisson('y ~ x + (1 | g)',data(),priors(True));new=pd.DataFrame({'x':[2.,-1.],'g':['new','new']})
    rng=np.random.default_rng(8)
    effect=rng.normal(size=(2,4))*.5 if policy=='population' else np.take([.1,-.1],rng.integers(0,2,size=(2,4)))
    expected=np.exp(.3-.2*new.x.to_numpy()+effect[...,None])
    out=m.predict(draws(True),newdata=new,kind='response',new_groups=policy,seed=8)
    np.testing.assert_allclose(out.mu,expected)
    np.testing.assert_array_equal(out.y,rng.poisson(expected))
    assert out.y.dtype.kind in 'iu'


def test_ordinary_predictions_limits_and_records(monkeypatch):
    m=ab.poisson('y ~ x',data(),priors());post=draws()
    expected=np.broadcast_to(np.exp(.3-.2*data().x.to_numpy()),(2,4,4))
    out=m.predict(post,kind='response',seed=9)
    np.testing.assert_allclose(out.mu,expected)
    np.testing.assert_array_equal(out.y,np.random.default_rng(9).poisson(expected))
    with pytest.raises(ValueError,match='overflow'):m.predict(draws(a=1000.),kind='mean')
    with pytest.raises(ValueError,match='random-generator'):m.predict(draws(a=50.),kind='response')
    assert np.isfinite(m.predict(draws(a=50.),kind='mean').mu).all()
    with pytest.raises(ValueError,match='kind'):m.predict(post,kind='probability')
    import autobayes.session as session
    monkeypatch.setattr(session,'_summary_rows',lambda *args:{})
    monkeypatch.setattr(session,'environment',lambda:{})
    fit=SimpleNamespace(target_identity=m.target_identity,idata=post,backend='synthetic',device_requested='cpu',device_actual='cpu',platform='cpu',dtype='float64',chain_method='none',chains=2,draws=4,tune=0,target_accept=.9,seed=9,wall_seconds=0,notes=['fixed synthetic draws; not a fit'])
    rec=ab.fit_record(m,fit,panel={'values':{},'status':{},'overall':'not evaluated'})
    assert rec['model']['family']=='poisson' and rec['model']['link']=='log'
    assert 'log scale' in rec['model']['intercept_prior_applies_to']
    assert 'sigma' not in rec['model']['priors']
    assert ab.compare_records(rec,json.loads(json.dumps(rec)))['agree']
