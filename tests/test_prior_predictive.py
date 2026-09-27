import json
import numpy as np
import pandas as pd
import pytest
import autobayes as ab


def make(family='gaussian', grouped=False, changed=False):
    df=pd.DataFrame({'y':[0,1,0,1,1,0], 'x':[2.,3.,4.,5.,6.,7.], 'n':[2]*6,
                     'g':['a','b','a','b','a','b'], 'z':['a','b','a','b','a','b']})
    if changed: df.y=1-df.y
    priors=ab.Priors(intercept=ab.Normal(0,.3),b=ab.Normal(0,.2),
        **({'sd':ab.HalfNormal(.2)} if grouped else {}),
        **({'sigma':ab.HalfNormal(.3)} if family in ('gaussian','student') else {}))
    formula=('y | trials(n)' if family=='binomial' else 'y')+' ~ x'+(' + z + x:z + (1 | g)' if grouped else '')
    return getattr(ab,family)(formula,df,priors,**({'nu':4} if family=='student' else {}),
                              **({'categorical':{'z':'a'}} if grouped else {}))


@pytest.mark.parametrize('family',['gaussian','student','bernoulli','binomial','poisson'])
def test_forward_family(family):
    m=make(family);before=m.target_identity
    out=m.prior_predictive(draws=8,seed=17)
    assert out.prior_predictive[m.names['y']].shape==(1,8,6)
    assert out.attrs['autobayes_generation_kind']=='prior_predictive'
    assert json.loads(out.attrs['autobayes_built_target_identity_json'])==before==m.target_identity
    assert 'posterior' not in out.groups()
    with pytest.raises(ValueError,match='not a posterior fit'):m.predict(out)
    with pytest.raises(ValueError,match='not a posterior fit'):ab.fit_record(m,out,allow_unverified=True)


def test_seed_and_response_invariance():
    m=make();a=m.prior_predictive(draws=8,seed=91);b=m.prior_predictive(draws=8,seed=91)
    c=make(changed=True).prior_predictive(draws=8,seed=91)
    for group in ['prior','prior_predictive']:
        for name in getattr(a,group):
            np.testing.assert_array_equal(getattr(a,group)[name],getattr(b,group)[name])
            np.testing.assert_array_equal(getattr(a,group)[name],getattr(c,group)[name])
    assert not np.array_equal(a.observed_data.y,c.observed_data.y)


def test_grouped_centering_arithmetic():
    m=make(grouped=True);out=m.prior_predictive(draws=8,seed=23);p=out.prior
    # Graph intercept applies at column means. Reconstruct independently from source.
    x=m.data.x.to_numpy();z=(m.data.z.astype(str)=='b').to_numpy().astype(float)
    X=np.column_stack([x,z,x*z]);X=X-X.mean(axis=0)
    beta=np.stack([p.x.values,p.z.values[...,0],p['x:z'].values[...,0]],axis=-1)
    expected=p.Intercept.values[...,None]+np.einsum('cdp,np->cdn',beta,X)
    expected+=p['1|g'].values[:,:,[0,1,0,1,0,1]]
    np.testing.assert_allclose(p.mu.values,expected,atol=1e-12)
    assert list(p['g__factor_dim'].values)==['a','b']


@pytest.mark.parametrize('draws',[0,-1,True,1.2])
def test_bad_draws(draws):
    with pytest.raises(ValueError,match='positive integer'):ab.prior_predictive(None,draws=draws)


@pytest.mark.parametrize('seed',[-1,True,1.2])
def test_bad_seed(seed):
    with pytest.raises(ValueError,match='nonnegative integer'):ab.prior_predictive(None,seed=seed)
