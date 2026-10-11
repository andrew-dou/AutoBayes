"""Fixed-nu response checks against original-input references; no MCMC."""
import copy
import json
from types import SimpleNamespace
import arviz as az
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, halfnorm, t
import autobayes as ab


def data():
    return pd.DataFrame({'x': [-1., .5, 2., 3.], 'y': [-2., .2, 1.5, 4.], 'g': ['a','a','b','b']})


def priors(group=False):
    return ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1),
                     **({'sd': ab.HalfNormal(1)} if group else {}))


@pytest.mark.parametrize('nu', [None, True, 0, 1, -2, np.nan, np.inf, [4], '4'])
def test_invalid_nu(nu):
    with pytest.raises(ValueError, match='nu'):
        ab.student('y ~ x', data(), priors(), nu=nu)


def test_scale_and_family_refusals():
    with pytest.raises(ab.PriorError, match='sigma'):
        ab.student('y ~ x', data(), ab.Priors(intercept=ab.Normal(0,2), b=ab.Normal(0,1)), nu=4)
    with pytest.raises(ValueError, match='nu'):
        ab.Model('y ~ x', data(), priors(), nu=4)
    for name in ['sigma', 'nu']:
        df = data().rename(columns={'x':name})
        with pytest.raises(ab.FormulaError, match='collision'):
            ab.student(f'y ~ {name}', df, priors(), nu=4)


@pytest.mark.parametrize('nu', [1.1, 2., 7.])
@pytest.mark.parametrize('center', [False, True])
def test_original_input_density_gradient_and_jax(nu, center):
    df=data(); m=ab.student('y ~ x', df, priors(), nu=nu, center_predictors=center)
    p=m.value_point(intercept=.3,b=[-.2],sigma=.8)
    names=[v.name for v in m.pymc_model.value_vars]
    assert set(names)=={'Intercept','x','sigma_log__'}
    x=df.x.to_numpy(); x=x-x.mean() if center else x
    def reference(q):
        a,b,ls=q['Intercept'],q['x'],q['sigma_log__']; sigma=np.exp(ls)
        return (t.logpdf(df.y.to_numpy(),df=nu,loc=a+b*x,scale=sigma).sum()
                +norm.logpdf(a,0,2)+norm.logpdf(b,0,1)+halfnorm.logpdf(sigma)+ls)
    actual=m.pymc_model.compile_logp()(p)
    np.testing.assert_allclose(actual,reference(p),atol=1e-9)
    g=m.pymc_model.compile_dlogp()(p); fd=[]
    for name in names:
        hi=dict(p);lo=dict(p);hi[name]=p[name]+1e-5;lo[name]=p[name]-1e-5
        fd.append((reference(hi)-reference(lo))/2e-5)
    np.testing.assert_allclose(g,fd,rtol=1e-5,atol=1e-6)
    from pymc.sampling.jax import get_jaxified_logp
    j=get_jaxified_logp(m.pymc_model)
    np.testing.assert_allclose(j([p[n] for n in names]),reference(p),atol=1e-9)
    import jax
    jg=jax.grad(lambda q:j(q))([p[n] for n in names])
    np.testing.assert_allclose(jg,fd,rtol=1e-5,atol=1e-6)
    desc=m.describe()
    assert 'fixed, not sampled' in desc and 'scale, not residual SD' in desc
    assert 'log-odds' not in desc and 'logit' not in desc


@pytest.mark.parametrize('nu', [3.7, 4, 2.3])
def test_fixed_nu_reaches_pymc_in_float64(nu):
    # PyTensor makes a Python float that float32 represents exactly (4.0, 7.0, 2.5) a float32 constant, and the Student-t
    # normalising constant was then computed in float32: at nu = 4 every row's log density was 5.6e-10 off the exact
    # value; 3.7 and 2.3 stayed float64 (kept here so that both kinds of value stay covered)
    rng=np.random.default_rng(3); df=pd.DataFrame({'x':rng.normal(size=200)}); df['y']=.4+.7*df.x+rng.standard_t(nu,size=200)
    m=ab.student('y ~ x',df,priors(),nu=nu)
    import pytensor
    obs=m.pymc_model.observed_RVs[0]; nu_in=obs.owner.inputs[2]
    assert nu_in.dtype=='float64' and float(np.asarray(pytensor.function([],nu_in)()).ravel()[0])==float(nu)
    p=m.value_point(intercept=.3,b=[.6],sigma=.9); x=df.x.to_numpy()-df.x.mean()
    lik=m.pymc_model.compile_logp(vars=[obs])(p)
    np.testing.assert_allclose(lik,t.logpdf(df.y.to_numpy(),df=nu,loc=.3+.6*x,scale=.9).sum(),rtol=1e-14)


def draws(group=False):
    p={'Intercept':np.full((2,4),.3),'x':np.full((2,4),-.2),'sigma':np.full((2,4),.8)}
    if group:
        p.update({'1|g_sigma':np.full((2,4),.5),'1|g':np.broadcast_to([.1,-.1],(2,4,2))})
    return az.from_dict(posterior=p,coords={'g__factor_dim':['a','b']},dims={'1|g':['g__factor_dim']})


@pytest.mark.parametrize('policy',['population','resample_observed'])
def test_new_group_predictions_use_student_scale(policy):
    m=ab.student('y ~ x + (1 | g)',data(),priors(True),nu=1.5)
    new=pd.DataFrame({'x':[2.,-1.],'g':['new','new']}); post=draws(True)
    out=m.predict(post,newdata=new,kind='response',new_groups=policy,seed=8)
    rng=np.random.default_rng(8)
    effect=rng.normal(size=(2,4))*.5 if policy=='population' else np.take([.1,-.1],rng.integers(0,2,size=(2,4)))
    mu=.3-.2*new.x.to_numpy()+effect[...,None]
    np.testing.assert_allclose(out.mu,mu)
    np.testing.assert_allclose(out.y,mu+.8*rng.standard_t(1.5,size=mu.shape))
    assert set(out.data_vars)=={'mu','y'}
    mean=m.predict(post,newdata=new,kind='mean',new_groups=policy,seed=8)
    np.testing.assert_allclose(mean.mu,mu)


def test_ordinary_predictions_and_record_identity(monkeypatch):
    m=ab.student('y ~ x',data(),priors(),nu=4)
    post=draws();out=m.predict(post,kind='mean')
    np.testing.assert_allclose(out.mu,np.broadcast_to(.3-.2*data().x.to_numpy(),(2,4,4)))
    response=m.predict(post,kind='response',seed=9)
    assert response.y.shape == (2,4,4) and np.isfinite(response.y).all()
    np.testing.assert_array_equal(response.y,m.predict(post,kind='response',seed=9).y)
    import autobayes.session as session
    monkeypatch.setattr(session,'_summary_rows',lambda *args:{})
    monkeypatch.setattr(session,'environment',lambda:{})
    fit=SimpleNamespace(target_identity=m.target_identity,idata=post,backend='synthetic',device_requested='cpu',device_actual='cpu',platform='cpu',dtype='float64',chain_method='none',chains=2,draws=4,tune=0,target_accept=.9,seed=8,wall_seconds=0,notes=['fixed synthetic draws; not a fit'])
    a=ab.fit_record(m,fit,panel={'values':{},'status':{},'overall':'not evaluated'})
    assert a['model']['family_parameters']=={'nu':4.}
    assert 'logit' not in a['model']['intercept_prior_applies_to']
    a=json.loads(json.dumps(a));assert ab.compare_records(a,copy.deepcopy(a))['agree']
    b=copy.deepcopy(a);b['model']['family_parameters']['nu']=8
    with pytest.raises(ValueError,match='different models'):ab.compare_records(a,b)
    b=copy.deepcopy(a);del b['model']['family_parameters']
    with pytest.raises(ValueError,match='different models'):ab.compare_records(a,b)
    a['model']['family']='gaussian';a['model'].pop('family_parameters');b=copy.deepcopy(a)
    b['model']['family_parameters']={}
    assert ab.compare_records(a,b)['agree']


@pytest.mark.parametrize('intercept', [True, False])
def test_categorical_interaction_multiple_groups_original_density(intercept):
    df=data().assign(z=['a','b','a','b'],h=['u','v','u','v'])
    pr=ab.Priors(intercept=ab.Normal(0,2) if intercept else None,b=ab.Normal(0,1),
                 sd={'g':ab.HalfNormal(1),'h':ab.HalfNormal(1)},sigma=ab.HalfNormal(1))
    formula='y ~ '+('' if intercept else '0 + ')+'x * z + (1 | g) + (1 | h)'
    m=ab.student(formula,df,pr,nu=4,categorical={'z':'a'})
    # Reference uses original labels/products and declared fixed parameters, not model.design.
    p=m.pymc_model.initial_point()
    p['sigma_log__']=np.array(np.log(.8));p['x']=np.array(-.2)
    p['z']=np.array([.4]) if intercept else np.array([.1,.4])
    p['x:z']=np.array([.3])
    x=df.x.to_numpy();z=(df.z=='b').to_numpy().astype(float);xz=x*z
    mu=-.2*x+np.where(z,.4,0 if intercept else .1)+.3*xz
    ref=norm.logpdf(-.2)+norm.logpdf(p['z']).sum()+norm.logpdf(.3)+halfnorm.logpdf(.8)+np.log(.8)
    if intercept:
        p['Intercept']=np.array(.3)
        mu+=.3-(-.2*x.mean()+.4*z.mean()+.3*xz.mean())
        ref+=norm.logpdf(.3,0,2)
    for g,sd,offset in [('g',.5,np.array([.2,-.2])),('h',.7,np.array([-.1,.1]))]:
        p[f'1|{g}_sigma_log__']=np.array(np.log(sd));p[f'1|{g}_offset']=offset
        levels=sorted(df[g].unique());mu+=sd*np.array([offset[levels.index(v)] for v in df[g]])
        ref+=halfnorm.logpdf(sd)+np.log(sd)+norm.logpdf(offset).sum()
    ref+=t.logpdf(df.y,df=4,loc=mu,scale=.8).sum()
    np.testing.assert_allclose(m.pymc_model.compile_logp()(p),ref,atol=1e-9)
    assert m.design.centered == intercept
