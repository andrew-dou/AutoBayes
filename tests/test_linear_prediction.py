from types import SimpleNamespace
import numpy as np
import pytest
from scipy.special import expit
import autobayes as ab
from tests.test_multiple_groups import make, posterior, hand_mean
from tests.test_poisson import data, priors, draws


@pytest.mark.parametrize('family',['gaussian','bernoulli','binomial'])
@pytest.mark.parametrize('intercept',[False,True])
def test_composed_linear(family,intercept):
    m,df=make(family,intercept);post=posterior(family,intercept,draws=4)
    fit=SimpleNamespace(idata=post,target_identity=m.target_identity)
    new=df.iloc[[4,1,0]].copy()
    if family=='binomial':new=new.drop(columns=['n'])
    for frame in [None,new]:
        src=df if frame is None else frame
        expected=hand_mean(post.posterior,src,intercept,'gaussian')
        out=m.predict(fit,kind='linear',newdata=frame)
        assert list(out.data_vars)==['eta']
        np.testing.assert_allclose(out.eta,expected,atol=1e-12)
        assert out.attrs['link']==m.link and out.attrs['prediction_scale']=='linear'
        assert out.attrs['fit_binding_status']=='matched'
        mean=m.predict(fit,kind='probability' if family=='binomial' else 'mean',newdata=frame)
        np.testing.assert_allclose(mean['mu' if family=='gaussian' else 'p'],expected if family=='gaussian' else expit(expected),atol=1e-12)


@pytest.mark.parametrize('family',['student','poisson'])
def test_remaining_families(family):
    pr=priors() if family=='poisson' else ab.Priors(ab.Normal(0,2),ab.Normal(0,1),sigma=ab.HalfNormal(1))
    m=getattr(ab,family)('y ~ x',data(),pr,**({'nu':4} if family=='student' else {}))
    post=draws();fit=SimpleNamespace(idata=post,target_identity=m.target_identity)
    out=m.predict(fit,kind='linear');expected=np.broadcast_to(.3-.2*data().x.to_numpy(),(2,4,4))
    np.testing.assert_allclose(out.eta,expected)
    if family=='poisson':np.testing.assert_allclose(m.predict(fit,kind='mean').mu,np.exp(expected))


def test_extreme_and_nonfinite():
    m=ab.poisson('y ~ x',data(),priors());post=draws(a=1000.)
    assert np.isfinite(m.predict(post,kind='linear').eta).all()
    with pytest.raises(ValueError,match='overflow'):m.predict(post,kind='mean')
    post.posterior.Intercept.values[:]=np.inf
    with pytest.raises(ValueError,match='nonfinite'):m.predict(post,kind='linear')


@pytest.mark.parametrize('policy',['population','resample_observed'])
def test_new_levels(policy):
    m,df=make();post=posterior('gaussian',True,draws=4);new=df.iloc[[0,0]].copy();new['g']='new'
    a=m.predict(post,kind='linear',newdata=new,new_groups=policy,seed=12)
    b=m.predict(post,kind='mean',newdata=new,new_groups=policy,seed=12)
    np.testing.assert_array_equal(a.eta,b.mu)
    np.testing.assert_array_equal(a.eta[...,0],a.eta[...,1])


def test_binding_and_prior_refused():
    m=ab.poisson('y ~ x',data(),priors());other=ab.poisson('y ~ x',data().assign(x=lambda df:df.x+1),priors());p=draws()
    with pytest.raises(ValueError,match='identities differ'):m.predict(SimpleNamespace(idata=p,target_identity=other.target_identity),kind='linear')
    p.attrs['autobayes_generation_kind']='prior_predictive'
    with pytest.raises(ValueError,match='not a posterior'):m.predict(p,kind='linear')
