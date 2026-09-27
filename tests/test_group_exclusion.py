from types import SimpleNamespace
import numpy as np
import pytest
from scipy.special import expit
import autobayes as ab
from tests.test_multiple_groups import make,posterior,common


@pytest.mark.parametrize('family',['gaussian','bernoulli','binomial','student','poisson'])
@pytest.mark.parametrize('intercept',[True,False])
def test_fixed_only(family,intercept):
    base,df=make('gaussian',intercept)
    if family in ('gaussian','bernoulli','binomial'):m,df=make(family,intercept)
    else:
        pr=ab.Priors(ab.Normal(0,1) if intercept else None,ab.Normal(0,1),sd={'g':ab.HalfNormal(1),'h':ab.HalfNormal(1)},**({'sigma':ab.HalfNormal(1)} if family=='student' else {}))
        m=getattr(ab,family)('y ~ '+('' if intercept else '0 + ')+'x + z + x:z + (1 | g) + (1 | h)',df.assign(y=[0,1]*6),pr,categorical={'z':'b'},**({'nu':4} if family=='student' else {}))
    post=posterior('gaussian',intercept,draws=4)
    # Absence of group arrays proves excluded effects need not be accessed.
    post.posterior=post.posterior.drop_vars(['1|g','1|g_sigma','1|h','1|h_sigma'])
    fit=SimpleNamespace(idata=post,target_identity=m.target_identity)
    for frame in [None,df.iloc[[3,1,0]].drop(columns=['g','h'])]:
        src=df if frame is None else frame;p=post.posterior
        beta=np.concatenate([p.x.values[...,None],p.z.values,p['x:z'].values],axis=-1)
        eta=np.einsum('cdp,np->cdn',beta,common(src,intercept))+(p.Intercept.values[...,None] if intercept else 0)
        linear=m.predict(fit,kind='linear',newdata=frame,include_group_effects=False)
        np.testing.assert_allclose(linear.eta,eta)
        assert linear.attrs['new_groups']=='not_applicable' and not linear.attrs['include_group_effects']
        out=m.predict(fit,kind='mean',newdata=frame,include_group_effects=False)
        expected=expit(eta) if family in ('bernoulli','binomial') else np.exp(eta) if family=='poisson' else eta
        np.testing.assert_allclose(out['p' if family in ('bernoulli','binomial') else 'mu'],expected)
        if family=='binomial':np.testing.assert_allclose(out.expected,expected*src.n.to_numpy())
    assert m.spec.groups==('g','h')


def test_default_true_and_rng():
    m,df=make();p=posterior('gaussian',True,draws=4)
    a=m.predict(p,kind='linear');b=m.predict(p,kind='linear',include_group_effects=True)
    np.testing.assert_array_equal(a.eta,b.eta)
    mean=m.predict(p,kind='mean',include_group_effects=False).mu.values
    response=m.predict(p,kind='response',include_group_effects=False,seed=13).y.values
    np.testing.assert_array_equal(response,np.random.default_rng(13).normal(mean,p.posterior.sigma.values[...,None]))


def test_refusals_and_nonlinear_semantics():
    m,_=make();p=posterior('gaussian',True,draws=4)
    for flag in [0,1,'false',None]:
        with pytest.raises(ValueError,match='boolean'):m.predict(p,include_group_effects=flag)
    for policy in ['population',{'g':'error','h':'error'}]:
        with pytest.raises(ValueError,match='default'):m.predict(p,include_group_effects=False,new_groups=policy)
    p.attrs['autobayes_generation_kind']='prior_predictive'
    with pytest.raises(ValueError,match='not a posterior'):m.predict(p,include_group_effects=False)
    assert not np.isclose(expit(1),np.mean(expit(np.array([-1,3]))))


def test_fixed_column_required_and_shared_role_still_refused():
    m,df=make()
    with pytest.raises(ab.FormulaError,match='lacks columns'):
        m.predict(posterior('gaussian',True,draws=4),newdata=df.drop(columns=['z']),include_group_effects=False)
    with pytest.raises(ab.FormulaError,match='cannot also be a predictor'):
        ab.gaussian('y ~ x + z + (1 | z)',df,ab.Priors(ab.Normal(0,1),ab.Normal(0,1),sd=ab.HalfNormal(1),sigma=ab.HalfNormal(1)),categorical={'z':'b'})
