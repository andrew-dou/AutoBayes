"""M3a-3: original-input oracles and synthetic retained draws; no sampler fits."""
import copy
import json

import arviz as az
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, gammaln
from scipy.stats import expon, halfnorm, multivariate_normal, norm, t

import autobayes as ab
from tests.test_population_only import PANEL_STUB

LHS = {'gaussian': 'y', 'bernoulli': 'yb', 'binomial': 'k | trials(n)'}
REFS = {'z': 'middle', 'v': 'right'}
PATTERNS = {
    'shared_factor': (('x', 'w', 'z'), (('x', 'z'), ('w', 'z'))),
    'shared_numeric': (('x', 'z', 'v'), (('x', 'z'), ('x', 'v'))),
    'disjoint': (('x', 'w', 'z', 'v'), (('x', 'z'), ('w', 'v'))),
    'three': (('x', 'w', 'z', 'v'), (('x', 'z'), ('w', 'z'), ('x', 'v'))),
}
GROUP_LEVELS = {'g': ('A', 'B', 'C'), 'h': ('A', 'B')}
SD = {'g': .6, 'h': .35}
OFFSETS = {'g': np.array([-.4, .2, .9]), 'h': np.array([.7, -.5])}


def inputs():
    rows = []
    for zi, z in enumerate(('alpha', 'middle', 'omega')):
        for vi, v in enumerate(('left', 'right')):
            for repeat in range(zi + 1):
                for j in range(4 if vi == 0 else 2):
                    i = len(rows)
                    x = (-1.1, .3, .8, 1.7)[j] + .4*zi + .2*vi
                    w = (-.8, 1.4, .2, .9)[j] - .3*zi + .1*repeat
                    n = 2 + i % 5
                    rows.append(dict(x=x, w=w, z=z, v=v, g=GROUP_LEVELS['g'][i % 3],
                                     h=GROUP_LEVELS['h'][(i // 2) % 2], y=.2+.4*x-.2*w+.13*(i % 5)-.1*zi,
                                     yb=i % 3 == 0, k=i % (n+1), n=n))
    df = pd.DataFrame(rows)
    df.yb = df.yb.astype(int)
    return df


def declarations(pattern, first='z'):
    mains, pairs = PATTERNS[pattern]
    cats = [n for n in mains if n in REFS]
    if first in cats:
        cats = [first] + [n for n in cats if n != first]
    return tuple(n for n in mains if n not in REFS) + tuple(cats), pairs


def columns(df, mains, pairs, refs, intercept, fitted_levels=None):
    """Only caller rows and declarations determine expected labels/coding/order."""
    cat_order = [n for n in mains if n in refs]
    levels = ({n: sorted({str(v) for v in df[n]}) for n in cat_order}
              if fitted_levels is None else fitted_levels)
    parts, labels, term_levels = [], [], {}
    for name in mains:
        if name not in refs:
            parts.append(df[name].to_numpy(float)); labels.append(name)
            term_levels[name] = ()
        else:
            selected = [l for l in levels[name] if l != refs[name]]
            if not intercept and name == cat_order[0]: selected = [refs[name]] + selected
            term_levels[name] = tuple(selected)
            for level in selected:
                parts.append(np.array([str(v) == level for v in df[name]], float))
                labels.append(f'{name}[{level}]')
    for x, z in pairs:
        term = f'{x}:{z}'
        selected = [l for l in levels[z] if l != refs[z]]
        term_levels[term] = tuple(selected)
        for level in selected:
            parts.append(df[x].to_numpy(float) * np.array([str(v) == level for v in df[z]]))
            labels.append(f'{term}[{level}]')
    return np.column_stack(parts), labels, term_levels


def prior_definitions(term_levels, normal_only=False):
    public, per_column = {}, {}
    for i, (term, levels) in enumerate(term_levels.items()):
        defs = []
        for j in range(len(levels) or 1):
            mu, scale = -.23 + .09*i + .07*j, .7 + .13*i + .11*j
            # Different term and level parameters expose block/coordinate cross-wiring.
            kind = 'student' if ':' in term and i % 2 == 0 and not normal_only else 'normal'
            spec = (kind, 4. + j, mu, scale)
            defs.append(spec)
            label = f'{term}[{levels[j]}]' if levels else term
            per_column[label] = spec
        def make(spec):
            kind, nu, mu, scale = spec
            return ab.StudentT(nu, mu, scale) if kind == 'student' else ab.Normal(mu, scale)
        public[term] = {l: make(d) for l, d in zip(levels, defs)} if levels else make(defs[0])
    return public, per_column


def build(family='gaussian', pattern='three', intercept=True, center=True, grouped=True,
          first='z', refs=None, df=None, normal_only=False):
    df = inputs() if df is None else df
    refs = REFS if refs is None else refs
    mains, pairs = declarations(pattern, first)
    refs = {n: refs[n] for n in mains if n in refs}
    fitted = df.loc[df.n > 0].copy() if family == 'binomial' else df.copy()
    X, labels, term_levels = columns(fitted, mains, pairs, refs, intercept)
    b, priors = prior_definitions(term_levels, normal_only)
    sd = {'g': ab.HalfNormal(.8), 'h': ab.Exponential(1.4)} if grouped else None
    pr = ab.Priors(ab.Normal(-.2, 1.7) if intercept else None, b, sd,
                   ab.HalfNormal(1.1) if family == 'gaussian' else None)
    rhs = ('0 + ' if not intercept else '') + ' + '.join([*mains, *(f'{x}:{z}' for x,z in pairs)])
    if grouped: rhs += ' + (1 | g) + (1 | h)'
    model = getattr(ab, family)(f'{LHS[family]} ~ {rhs}', df, pr, categorical=refs, center_predictors=center)
    return model, dict(original=df, fitted=fitted, X=X, labels=labels, term_levels=term_levels,
                       mains=mains, pairs=pairs, refs=refs, priors=priors, intercept=intercept,
                       centered=intercept and center, grouped=grouped, family=family,
                       levels={n:tuple(sorted({str(v) for v in fitted[n]})) for n in refs})


def point(model, case):
    b = np.linspace(-.4, .65, len(case['labels']))
    natural = dict(b=b)
    if case['intercept']: natural['intercept'] = .3
    if case['grouped']: natural.update(sd=SD, z=OFFSETS)
    if case['family'] == 'gaussian': natural['sigma'] = .9
    return model.value_point(**natural), b


def coefficient_vector(q, case):
    return np.concatenate([np.atleast_1d(q[n]) for n in case['term_levels']])


def reference(q, case, jacobian=True):
    df = case['fitted']; X = case['X']
    if case['centered']: X = X - X.mean(0)
    beta = coefficient_vector(q, case)
    eta = X @ beta
    lp = 0.
    if case['intercept']:
        eta += q['Intercept']; lp += norm.logpdf(q['Intercept'], -.2, 1.7)
    for label, value in zip(case['labels'], beta):
        kind, nu, mu, scale = case['priors'][label]
        lp += t.logpdf(value, nu, mu, scale) if kind == 'student' else norm.logpdf(value, mu, scale)
    if case['grouped']:
        for g, levels in GROUP_LEVELS.items():
            log_sd = q[f'1|{g}_sigma_log__']; sd = np.exp(log_sd); offsets = q[f'1|{g}_offset']
            idx = np.array([levels.index(str(v)) for v in df[g]])
            eta += sd * offsets[idx]
            lp += norm.logpdf(offsets).sum()
            lp += halfnorm.logpdf(sd, scale=.8) if g == 'g' else expon.logpdf(sd, scale=1/1.4)
            if jacobian: lp += log_sd
    if case['family'] == 'gaussian':
        log_sigma = q['sigma_log__']; sigma = np.exp(log_sigma)
        lp += norm.logpdf(df.y.to_numpy(), eta, sigma).sum() + halfnorm.logpdf(sigma, scale=1.1)
        if jacobian: lp += log_sigma
    else:
        k = df.k.to_numpy() if case['family'] == 'binomial' else df.yb.to_numpy()
        n = df.n.to_numpy() if case['family'] == 'binomial' else np.ones(len(df))
        lp += (gammaln(n+1)-gammaln(k+1)-gammaln(n-k+1)-k*np.logaddexp(0,-eta)-(n-k)*np.logaddexp(0,eta)).sum()
    return float(lp)


CASES = [(family, pattern, (i+j) % 2 == 0, (i+j) % 4 < 2, (i+j) % 3 != 0,
          'v' if (i+j) % 2 else 'z')
         for i,family in enumerate(LHS) for j,pattern in enumerate(PATTERNS)]


@pytest.mark.parametrize('family,pattern,intercept,center,grouped,first', CASES)
def test_original_columns_coordinates_graph(family,pattern,intercept,center,grouped,first):
    m,c = build(family,pattern,intercept,center,grouped,first)
    X = c['X']; Xm = X-X.mean(0) if c['centered'] else X
    assert list(m.spec.columns) == c['labels']
    assert m.spec.interactions == c['pairs'] and list(m.spec.terms) == list(c['term_levels'])
    assert np.array_equal(m.design.X_raw,X) and np.array_equal(m.design.X_model,Xm)
    assert m.design.centered == c['centered']
    assert ('Intercept' in m.pymc_model.named_vars) is intercept
    for term, levels in c['term_levels'].items():
        if levels: assert tuple(m.pymc_model.coords[f'{term}_dim']) == levels
    q,b = point(m,c)
    assert np.array_equal(coefficient_vector(q,c),b)
    eta = Xm@b + (.3 if intercept else 0)
    if grouped:
        for g,levels in GROUP_LEVELS.items(): eta += SD[g]*OFFSETS[g][[levels.index(v) for v in c['fitted'][g]]]
    pmm=m.pymc_model
    fn=pmm.compile_fn(pmm.replace_rvs_by_values([pmm[m.names['linpred']]]),inputs=pmm.value_vars,point_fn=True,on_unused_input='ignore')
    assert np.allclose(fn(q)[0],eta,rtol=1e-12,atol=1e-12)
    # A shared product column's actual mean is not generally the product of means.
    x,z=c['pairs'][0];level=next(l for l in sorted(set(c['fitted'][z])) if l!=c['refs'][z])
    assert not np.isclose(np.mean(c['fitted'][x]*(c['fitted'][z]==level)),c['fitted'][x].mean()*(c['fitted'][z]==level).mean())


@pytest.mark.parametrize('family',LHS)
@pytest.mark.parametrize('intercept',[False,True])
def test_normalized_density_and_independent_gradients(family,intercept):
    import jax
    from pymc.sampling.jax import get_jaxified_logp
    m,c=build(family,intercept=intercept,first='v' if not intercept else 'z',grouped=intercept)
    q,_=point(m,c);pmm=m.pymc_model;names=[v.name for v in pmm.value_vars]
    lp=pmm.compile_logp();nojac=pmm.compile_logp(jacobian=False);grad=pmm.compile_dlogp()
    jf=get_jaxified_logp(pmm);jg=jax.grad(lambda *v:jf(list(v)),argnums=tuple(range(len(names))))
    for shift in (0.,.13):
        shifted={k:np.asarray(v+shift) for k,v in q.items()}
        assert abs(float(lp(shifted))-reference(shifted,c))<1e-8
        assert abs(float(nojac(shifted))-reference(shifted,c,False))<1e-8
        assert abs(float(jf([shifted[n] for n in names]))-reference(shifted,c))<1e-8
        flat=np.concatenate([np.atleast_1d(shifted[n]).ravel() for n in names])
        def ref_flat(v):
            out={};i=0
            for name in names:
                size=shifted[name].size;out[name]=v[i:i+size].reshape(shifted[name].shape);i+=size
            return reference(out,c)
        eps=1e-5
        fd=np.array([(ref_flat(flat+eps*e)-ref_flat(flat-eps*e))/(2*eps) for e in np.eye(len(flat))])
        assert np.allclose(grad(shifted),fd,rtol=1e-6,atol=1e-5)
        jgrad=np.concatenate([np.atleast_1d(v).ravel() for v in jg(*[shifted[n] for n in names])])
        assert np.allclose(jgrad,grad(shifted),rtol=1e-9,atol=1e-9)


def reference_map(theta, old, new):
    """Explicit raw coefficient reparameterization; no least-squares/backend design solve."""
    intercept=old['intercept'];old_labels=(['Intercept'] if intercept else [])+old['labels']
    values=dict(zip(old_labels,theta));out={}
    cats=[n for n in old['mains'] if n in old['refs']]
    first=None if intercept else cats[0]
    def main(z,l): return values.get(f'{z}[{l}]',0.)
    def delta(x,z,l): return values.get(f'{x}:{z}[{l}]',0.)
    shift=sum(main(z,new['refs'][z]) for z in cats if z!=first)
    if intercept:out['Intercept']=values['Intercept']+shift
    for n in old['mains']:
        if n not in cats:
            out[n]=values[n]+sum(delta(x,z,new['refs'][z]) for x,z in old['pairs'] if x==n)
        else:
            for l in new['term_levels'][n]:out[f'{n}[{l}]']=main(n,l)+shift if n==first else main(n,l)-main(n,new['refs'][n])
    for x,z in old['pairs']:
        for l in new['term_levels'][f'{x}:{z}']:out[f'{x}:{z}[{l}]']=delta(x,z,l)-delta(x,z,new['refs'][z])
    return np.array([out[n] for n in ((['Intercept'] if intercept else [])+new['labels'])])


@pytest.mark.parametrize('pattern,intercept,first',[
    ('shared_factor',True,'z'),('shared_factor',False,'z'),
    ('shared_numeric',True,'z'),('shared_numeric',False,'z'),('shared_numeric',False,'v')])
def test_explicit_reference_maps_and_transported_density(pattern,intercept,first):
    old,oc=build(pattern=pattern,intercept=intercept,grouped=False,first=first,normal_only=True)
    new,nc=build(pattern=pattern,intercept=intercept,grouped=False,first=first,normal_only=True,
                 refs={'z':'omega','v':'left'})
    d=len(oc['labels'])+int(intercept)
    T=np.column_stack([reference_map(e,oc,nc) for e in np.eye(d)])
    C0=np.eye(d);C1=np.eye(d)
    if intercept:C0[0,1:]=oc['X'].mean(0);C1[0,1:]=nc['X'].mean(0)
    M=C1@T@np.linalg.inv(C0)
    raw=np.linspace(-.6,.8,d);transformed=T@raw
    A0=np.column_stack([np.ones(len(oc['X'])),oc['X']]) if intercept else oc['X']
    A1=np.column_stack([np.ones(len(nc['X'])),nc['X']]) if intercept else nc['X']
    assert np.allclose(A0@raw,A1@transformed,rtol=1e-12,atol=1e-12)
    sign,logdet=np.linalg.slogdet(M);assert sign!=0 and abs(logdet)<1e-12
    model0=C0@raw;model1=C1@transformed
    def lp(model,c,theta):
        kwargs=dict(b=theta[int(intercept):],sigma=.9)
        if intercept:kwargs['intercept']=theta[0]
        return float(model.pymc_model.compile_logp()(model.value_point(**kwargs)))
    def parameters(c):
        specs=[c['priors'][l] for l in c['labels']]
        mus=[s[2] for s in specs];sds=[s[3] for s in specs]
        return (np.r_[-.2,mus],np.r_[1.7,sds]) if intercept else (np.array(mus),np.array(sds))
    mu0,sd0=parameters(oc);mu1,sd1=parameters(nc)
    pr0=norm.logpdf(model0,mu0,sd0).sum();pr1=norm.logpdf(model1,mu1,sd1).sum()
    lp0=lp(old,oc,model0);lp1=lp(new,nc,model1)
    assert abs((lp0-pr0)-(lp1-pr1))<1e-8
    assert abs(pr0-pr1)>1e-4  # Same independent-prior recipe in different coding is different.
    transported=multivariate_normal.logpdf(model1,mean=M@mu0,cov=M@np.diag(sd0**2)@M.T)
    assert abs(transported-(pr0-logdet))<1e-10
    assert abs((lp1-pr1+transported)-(lp0-logdet))<1e-8
    if intercept:assert abs(model0[0]-model1[0])<1e-12


def posterior(c,draws=20):
    rng=np.random.default_rng(881);shape=(2,draws);values={};dims={};coords={}
    for i,(term,levels) in enumerate(c['term_levels'].items()):
        values[term]=rng.normal(.07*i,.2,shape+((len(levels),) if levels else ()))
        if levels:dims[term]=[f'{term}_dim'];coords[f'{term}_dim']=list(levels)
    if c['intercept']:values['Intercept']=rng.normal(.2,.1,shape)
    if c['family']=='gaussian':values['sigma']=np.full(shape,.9)
    if c['grouped']:
        for g,levels in GROUP_LEVELS.items():
            values[f'1|{g}_sigma']=np.full(shape,SD[g])
            values[f'1|{g}']=np.broadcast_to(SD[g]*OFFSETS[g],shape+(len(levels),)).copy()
            dims[f'1|{g}']=[f'{g}__factor_dim'];coords[f'{g}__factor_dim']=list(levels)
    return az.from_dict(posterior=values,dims=dims,coords=coords)


def expected_prediction(post,frame,c,unseen=None):
    X,_,_=columns(frame,c['mains'],c['pairs'],c['refs'],c['intercept'],c['levels'])
    beta=np.concatenate([post[n].values if levels else post[n].values[...,None] for n,levels in c['term_levels'].items()],axis=-1)
    eta=np.einsum('cdp,np->cdn',beta,X)
    if c['intercept']:eta+=post.Intercept.values[...,None]
    if c['grouped']:
        for g,levels in GROUP_LEVELS.items():
            for i,label in enumerate(frame[g]):
                eta[...,i]+=(unseen[g,label] if unseen and (g,label) in unseen else post[f'1|{g}'].values[...,levels.index(label)])
    return eta if c['family']=='gaussian' else expit(eta)


@pytest.mark.parametrize('family',LHS)
@pytest.mark.parametrize('grouped,intercept',[(False,False),(True,True)])
def test_synthetic_predictions_and_complete_records(family,grouped,intercept,tmp_path):
    m,c=build(family,grouped=grouped,intercept=intercept)
    idata=posterior(c);post=idata.posterior
    new=c['original'].iloc[[15,2,31,8,20,1]].copy().reset_index(drop=True)
    new['x']+=.37;new['w']-=.21
    for frame,arg in [(c['fitted'],None),(new,new),(new.iloc[::-1],new.iloc[::-1])]:
        ds=m.predict(idata,newdata=arg,kind='mean')
        assert np.allclose(ds[m.names['mean']],expected_prediction(post,frame,c),rtol=1e-10,atol=1e-10)
        if family=='binomial':
            assert np.allclose(ds.expected,ds.p*frame.n.to_numpy())
            prob=m.predict(idata,newdata=frame.drop(columns='n'),kind='probability')
            assert np.allclose(prob.p,ds.p)
    if grouped:
        new['g']=['NEW','NEW','A','NEXT','NEW','A'];new['h']=['NEW','A','NEW','NEXT','NEW','A']
        policies={'g':'population','h':'resample_observed'};effects={};rng=np.random.default_rng(43)
        for g in ('g','h'):
            for label in ('NEW','NEXT'):
                effects[g,label]=(rng.normal(size=(2,20))*SD[g] if g=='g' else
                                  np.take_along_axis(post[f'1|{g}'].values,rng.integers(0,2,size=(2,20))[...,None],axis=-1)[...,0])
        ds=m.predict(idata,newdata=new,kind='mean',new_groups=policies,seed=43)
        assert np.allclose(ds[m.names['mean']],expected_prediction(post,new,c,effects),rtol=1e-10,atol=1e-10)
        ds.to_netcdf(tmp_path/'prediction.nc')
    for cat in c['refs']:
        bad=c['original'].iloc[:2].copy();bad[cat]='unfitted'
        with pytest.raises(ab.FormulaError,match='not in the fitted rows'):
            m.predict(idata,newdata=bad,kind='mean',new_groups='population' if grouped else 'error')
    fit=ab.FitResult(idata,'pymc','cpu','cpu','cpu','float64',None,2,20,0,.9,0,seed=0,_target_identity_json=json.dumps(m.target_identity))
    record=ab.fit_record(m,fit,panel=PANEL_STUB);json.dumps(record)
    assert record['model']['interactions']==[list(p) for p in c['pairs']]
    assert record['model']['design_columns']==c['labels']
    for term,levels in c['term_levels'].items():
        for j,level in enumerate(levels or [None]):
            label=f'{term}[{level}]' if level is not None else term
            assert label in record['posterior'] and f'b:{label}' in record['model']['priors']
            actual=post[term].values[...,j] if levels else post[term].values
            assert np.isclose(record['posterior'][label]['mean'],actual.mean())
    assert ab.compare_records(record,copy.deepcopy(record))['agree']


def test_composed_descriptions_and_nonzero_slopes():
    m,c=build('bernoulli');text=m.describe();idata=posterior(c)
    assert 'at x = 0 and w = 0 (same group effects)' in text
    assert 'b_z[l] + x * b_x:z[l] + w * b_w:z[l]' in text
    assert 'slope in x on the log-odds scale is b_x + b_x:z[z_i] + b_x:v[v_i]' in text
    frame=c['original'].iloc[[0]].copy();frame['x']=1.7;frame['w']=-.8;frame['z']='omega';frame['v']='left'
    raised=frame.copy();raised.x+=1
    post=idata.posterior
    expected=post.x+post['x:z'].sel({'x:z_dim':'omega'})+post['x:v'].sel({'x:v_dim':'left'})
    # Compare linear predictors, not probability-scale differences.
    a=expected_prediction(post,frame,{**c,'family':'gaussian'})
    b=expected_prediction(post,raised,{**c,'family':'gaussian'})
    assert np.allclose((b-a)[...,0],expected)
    reference_row=frame.copy();reference_row.z='middle'
    contrast=post.z.sel(z_dim='omega')+1.7*post['x:z'].sel({'x:z_dim':'omega'})-.8*post['w:z'].sel({'w:z_dim':'omega'})
    assert np.allclose((a-expected_prediction(post,reference_row,{**c,'family':'gaussian'}))[...,0],contrast)
    noint,_=build(intercept=False)
    assert 'b_z[l] - b_z[middle] + x * b_x:z[l] + w * b_w:z[l]' in noint.describe()


def test_grammar_prior_refusals_and_exact_identity():
    df=inputs();m,c=build()
    for rhs in ['x*z+w*z','x*z+x*v','x+w+z+x:z+z:x','x+z+x:w','x+w+z+x:w',
                'z+v+z:v','x+w+z+x:z:w','x+z+x:x','x+z+log(x):z','x+z+x:z+(x|g)']:
        with pytest.raises(ab.FormulaError):ab.parse_formula('y~'+rhs,df,categorical={'z':'middle','v':'right'} if 'v' in rhs else {'z':'middle'})
    for bad in [dict(m.priors.b,**{'z:x':ab.Normal(0,1)}),
                {k:v for k,v in m.priors.b.items() if k!='w:z'},
                dict(m.priors.b,**{'x:z':{'alpha':ab.Normal(0,1),'omega':ab.StudentT(4,0,1)}}),
                dict(m.priors.b,**{'x:z':{'alpha':ab.Normal(0,1),'omega':ab.Normal(0,1),'middle':ab.Normal(0,1)}})]:
        with pytest.raises(ab.PriorError):ab.gaussian(m.spec.formula,df,ab.Priors(m.priors.intercept,bad,m.priors.sd,m.priors.sigma),categorical=REFS)
    with pytest.raises(ValueError,match='entries'):m.value_point(intercept=.1,b=[.2],sd=SD,z=OFFSETS,sigma=.9)
    big=2**63+41;labels={'alpha':big,'middle':big+1,'omega':big+2}
    df['z']=pd.Series([labels[v] for v in df.z],dtype=object)
    exact,ec=build(df=df,refs={'z':str(big+1),'v':'right'})
    assert np.array_equal(exact.design.X_raw,ec['X'])
    assert list(exact.pymc_model.coords['x:z_dim'])==[str(big),str(big+2)]
    assert list(exact.pymc_model.coords['w:z_dim'])==[str(big),str(big+2)]
    with pytest.raises(ab.FormulaError,match='collision'):
        ab.gaussian(m.spec.formula.replace(' ~ ',' ~ z_dim + '),inputs().assign(z_dim=np.arange(36)),
                    ab.Priors(ab.Normal(0,1),ab.Normal(0,1),m.priors.sd,ab.HalfNormal(1)),categorical=REFS)


def test_binomial_filtering_and_removed_levels():
    df=inputs();extra=df.iloc[[0]].copy();extra['z']='removed';extra['v']='removed';extra['n']=0;extra['k']=0
    original=pd.concat([extra,df],ignore_index=True)
    m,c=build('binomial',df=original)
    assert m.dropped_rows==(0,) and len(c['fitted'])==36
    assert np.array_equal(m.design.X_raw,c['X'])
    assert np.allclose(m.design.x_means,c['X'].mean(0))
    with pytest.raises(ab.FormulaError,match='not in the fitted rows'):
        m.predict(posterior(c),newdata=extra,kind='probability',new_groups='population')
    new=df.iloc[:3].copy();new['n']=0
    ds=m.predict(posterior(c,draws=3),newdata=new,kind='response',seed=61)
    assert np.array_equal(ds.expected,np.zeros((2,3,3))) and np.array_equal(ds.y,np.zeros((2,3,3)))


def test_documented_example_builds():
    from examples.m3a3_multiple_interactions import build_example

    model = build_example()
    assert model.spec.interactions == (("dose", "arm"), ("time", "arm"), ("dose", "region"))
    assert model.spec.groups == ("clinic", "year")
    assert model.spec.columns == ("dose", "time", "arm[high]", "arm[low]", "region[urban]",
                                  "dose:arm[high]", "dose:arm[low]", "time:arm[high]",
                                  "time:arm[low]", "dose:region[urban]")
    assert "b_dose + b_dose:arm[arm_i] + b_dose:region[region_i]" in model.describe()
    assert "b_arm[l] + dose * b_dose:arm[l] + time * b_time:arm[l]" in model.describe()
