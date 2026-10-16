"""M3c-3 deterministic acceptance; synthetic parameters, never sampler fits."""
import copy
import json
from dataclasses import replace

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
from scipy.special import expit, gammaln
from scipy.stats import norm, halfnorm, expon

import autobayes as ab
from tests.test_intercept_controls import source
from tests.test_population_only import PANEL_STUB


LHS = {"gaussian": "y", "bernoulli": "yb", "binomial": "k | trials(n)"}
LEVELS = {"g": ("A", "B", "C"), "h": ("A", "B"), "j": ("A", "B", "C", "D")}
SD = {"g": .6, "h": .35, "j": .2}
OFFSETS = {"g": np.array([-.4, .2, .9]), "h": np.array([.7, -.5]), "j": np.array([.1, .8, -.2, -.6])}


def inputs():
    return source().assign(h=list("ABBABAABABAB"), j=list("ABCDABCDABCD"))


def common(df, intercept):
    # Independent original-input oracle: never consult ModelSpec, Design or raw_design.
    levels = ("a", "c") if intercept else ("b", "a", "c")
    return np.column_stack([df.x.to_numpy(), *[(df.z == l).to_numpy(float) for l in levels],
                            *[df.x.to_numpy() * (df.z == l).to_numpy() for l in ("a", "c")]])


def block(df, g):
    return np.column_stack([(df[g] == l).to_numpy(float) for l in LEVELS[g]])


def make(family="gaussian", intercept=True, center=True, groups=("g", "h"), df=None, sd=None):
    if df is None:
        df = inputs()
    scales = {"g": ab.HalfNormal(.8), "h": ab.Exponential(1.4), "j": ab.HalfNormal(.4)}
    pr = ab.Priors(ab.Normal(-.3, 1.8) if intercept else None, ab.Normal(.2, 1.3),
                   {g: scales[g] for g in groups} if sd is None else sd,
                   ab.HalfNormal(1.1) if family == "gaussian" else None)
    formula = f"{LHS[family]} ~ " + ("" if intercept else "0 + ") + "x + z + x:z"
    formula += "".join(f" + (1 | {g})" for g in groups)
    return getattr(ab, family)(formula, df, pr, categorical={"z": "b"}, center_predictors=center), df


def natural(family, intercept, groups=("g", "h")):
    kw = dict(b=np.linspace(-.4, .6, 5 if intercept else 6),
              sd={g: SD[g] for g in groups}, z={g: OFFSETS[g] for g in groups})
    if intercept:
        kw["intercept"] = .3
    if family == "gaussian":
        kw["sigma"] = .9
    return kw


@pytest.mark.parametrize("family", LHS)
@pytest.mark.parametrize("intercept", [False, True])
@pytest.mark.parametrize("center", [False, True])
def test_original_design_graph_and_legacy_access(family, intercept, center):
    m, df = make(family, intercept, center)
    X = common(df, intercept)
    Xm = X - X.mean(0) if intercept and center else X
    assert m.spec.groups == ("g", "h") and m.spec.has_group
    assert m.spec.levels_by_group == {g: LEVELS[g] for g in ("g", "h")}
    assert np.array_equal(m.design.X_raw, X) and np.array_equal(m.design.X_model, Xm)
    assert np.array_equal(m.design.Z, np.column_stack([block(df, "g"), block(df, "h")]))
    assert m.design.group_slices == {"g": slice(0, 3), "h": slice(3, 5)}
    for g in ("g", "h"):
        b = m.design.group_designs[g]
        assert np.array_equal(b.Z, block(df, g)) and b.levels == LEVELS[g]
        assert b.group_index.tolist() == [LEVELS[g].index(l) for l in df[g]]
        names = m.group_names[g]
        assert names == {"sd": f"1|{g}_sigma", "offset": f"1|{g}_offset", "group_effect": f"1|{g}", "coordinate": f"{g}__factor_dim"}
        assert tuple(m.pymc_model.coords[names["coordinate"]]) == LEVELS[g]
    assert not {"sd", "offset", "group_effect"} & set(m.names)
    for obj, attrs in [(m.spec, ["group", "levels", "group_term"]), (m.design, ["J", "levels", "group_index"])]:
        for attr in attrs:
            with pytest.raises(ValueError, match="ambiguous"):
                getattr(obj, attr)
    nat = natural(family, intercept)
    eta = (.3 if intercept else 0) + Xm @ nat["b"]
    eta += sum(block(df, g) @ (SD[g] * OFFSETS[g]) for g in ("g", "h"))
    pmm = m.pymc_model
    fn = pmm.compile_fn(pmm.replace_rvs_by_values([pmm[m.names["linpred"]]]),
                        inputs=pmm.value_vars, point_fn=True, on_unused_input="ignore")
    assert np.allclose(fn(m.value_point(**nat))[0], eta, atol=1e-12)
    assert m.design.centered == (intercept and center)
    assert "sd_g ~ HalfNormal(sigma=0.8)" in m.describe()
    assert "sd_h ~ Exponential(lam=1.4)" in m.describe()


def reference(q, df, family, intercept, center, jacobian=True):
    X = common(df, intercept)
    if intercept and center:
        X = X - X.mean(0)
    beta = np.r_[q["x"], q["z"], q["x:z"]]
    eta = X @ beta + (q["Intercept"] if intercept else 0)
    lp = norm.logpdf(beta, .2, 1.3).sum()
    if intercept:
        lp += norm.logpdf(q["Intercept"], -.3, 1.8)
    for g in ("g", "h"):
        scale = np.exp(q[f"1|{g}_sigma_log__"])
        z = q[f"1|{g}_offset"]
        eta += block(df, g) @ (scale*z)
        lp += norm.logpdf(z).sum()
        lp += halfnorm.logpdf(scale, scale=.8) if g == "g" else expon.logpdf(scale, scale=1/1.4)
        if jacobian:
            lp += q[f"1|{g}_sigma_log__"]
    if family == "gaussian":
        sigma = np.exp(q["sigma_log__"])
        lp += norm.logpdf(df.y.to_numpy(), eta, sigma).sum() + halfnorm.logpdf(sigma, scale=1.1)
        if jacobian:
            lp += q["sigma_log__"]
    else:
        k = df.k.to_numpy() if family == "binomial" else df.yb.to_numpy()
        n = df.n.to_numpy() if family == "binomial" else np.ones(len(df))
        lp += (gammaln(n+1)-gammaln(k+1)-gammaln(n-k+1)-k*np.logaddexp(0, -eta)-(n-k)*np.logaddexp(0, eta)).sum()
    return float(lp)


@pytest.mark.parametrize("family", LHS)
@pytest.mark.parametrize("intercept,center", [(True, True), (False, True), (True, False)])
def test_normalized_density_jacobian_and_gradients(family, intercept, center):
    import jax
    from pymc.sampling.jax import get_jaxified_logp
    m, df = make(family, intercept, center)
    pmm = m.pymc_model
    names = [v.name for v in pmm.value_vars]
    logp, nojac, grad = pmm.compile_logp(), pmm.compile_logp(jacobian=False), pmm.compile_dlogp()
    jf = get_jaxified_logp(pmm)
    jg = jax.grad(lambda *v: jf(list(v)), argnums=tuple(range(len(names))))
    q = m.value_point(**natural(family, intercept))
    for shift in (0., .12):
        q = {k: np.asarray(v + shift) for k, v in q.items()}
        expected = reference(q, df, family, intercept, center)
        assert abs(float(logp(q))-expected) < 1e-8
        assert abs(float(nojac(q))-reference(q, df, family, intercept, center, False)) < 1e-8
        assert abs(float(jf([q[k] for k in names]))-expected) < 1e-8
        flat = np.concatenate([np.atleast_1d(q[k]).ravel() for k in names])
        def ref_flat(v):
            out, i = {}, 0
            for k in names:
                size = q[k].size
                out[k] = v[i:i+size].reshape(q[k].shape)
                i += size
            return reference(out, df, family, intercept, center)
        eps = 1e-5
        fd = np.array([(ref_flat(flat+eps*e)-ref_flat(flat-eps*e))/(2*eps) for e in np.eye(len(flat))])
        ptgrad = grad(q)
        jgrad = np.concatenate([np.atleast_1d(v).ravel() for v in jg(*[q[k] for k in names])])
        assert np.allclose(ptgrad, fd, rtol=1e-6, atol=1e-5)
        assert np.allclose(jgrad, ptgrad, rtol=1e-9, atol=1e-9)


def test_fixed_scale_gaussian_precision_and_mean():
    m, df = make()
    X = common(df, True); X -= X.mean(0)
    A = np.column_stack([np.ones(len(df)), X, .6*block(df, "g"), .35*block(df, "h")])
    Dinv = np.diag(np.r_[1/1.8**2, np.full(5, 1/1.3**2), np.ones(5)])
    prior_mean = np.r_[-.3, np.full(5, .2), np.zeros(5)]
    precision = Dinv + A.T @ A / .9**2
    mean = np.linalg.solve(precision, Dinv @ prior_mean + A.T @ df.y.to_numpy() / .9**2)
    pmm = m.pymc_model
    fixed = pm.do(pmm, {pmm['sigma']: .9, pmm['1|g_sigma']: .6, pmm['1|h_sigma']: .35})
    indices = {'Intercept': [0], 'x': [1], 'z': [2, 3], 'x:z': [4, 5], '1|g_offset': [6, 7, 8], '1|h_offset': [9, 10]}
    names = [v.name for v in fixed.value_vars]
    order = np.concatenate([indices[n] for n in names])
    point = {n: np.asarray(mean[indices[n]][0] if len(indices[n]) == 1 else mean[indices[n]]) for n in names}
    assert np.allclose(fixed.compile_d2logp()(point), precision[np.ix_(order, order)], rtol=1e-10, atol=1e-10)
    assert np.allclose(fixed.compile_dlogp()(point), 0, atol=1e-9)


def posterior(family, intercept, groups=("g", "h"), draws=400):
    rng = np.random.default_rng(709)
    shape = (2, draws)
    values = {"x": rng.normal(.1, .1, shape), "z": rng.normal(.2, .1, shape+(2 if intercept else 3,)),
              "x:z": rng.normal(0, .1, shape+(2,))}
    dims = {"z": ["z_dim"], "x:z": ["x:z_dim"]}
    coords = {"z_dim": ["a", "c"] if intercept else ["b", "a", "c"], "x:z_dim": ["a", "c"]}
    if intercept:
        values["Intercept"] = rng.normal(.3, .1, shape)  # raw intercept, as Bambi reports
    if family == 'gaussian':
        values['sigma'] = rng.uniform(.5, .6, shape)
    for g in groups:
        values[f'1|{g}_sigma'] = rng.uniform(SD[g], SD[g]+.1, shape)
        values[f'1|{g}'] = np.broadcast_to(SD[g]*OFFSETS[g], shape+(len(LEVELS[g]),)).copy()
        dims[f'1|{g}'] = [f'{g}__factor_dim']; coords[f'{g}__factor_dim'] = list(LEVELS[g])
    return az.from_dict(posterior=values, dims=dims, coords=coords)


def hand_mean(post, df, intercept, family, effects=None):
    beta = np.concatenate([post.x.values[..., None], post.z.values, post['x:z'].values], axis=-1)
    eta = np.einsum('cdp,np->cdn', beta, common(df, intercept))
    if intercept:
        eta += post.Intercept.values[..., None]
    for g in ('g', 'h'):
        for i, label in enumerate(df[g]):
            eta[..., i] += (effects[(g, label)] if effects and (g, label) in effects
                           else post[f'1|{g}'].values[..., LEVELS[g].index(label)])
    return eta if family == 'gaussian' else expit(eta)


@pytest.mark.parametrize('family', LHS)
@pytest.mark.parametrize('intercept', [False, True])
def test_prediction_arithmetic_records_and_mixed_policies(family, intercept, tmp_path):
    m, df = make(family, intercept)
    idata = posterior(family, intercept); post = idata.posterior
    for new in (None, df.iloc[[8, 1, 10, 3]].reset_index(drop=True)):
        ds = m.predict(idata, newdata=new, kind='mean')
        d = df if new is None else new
        assert np.allclose(ds[m.names['mean']], hand_mean(post, d, intercept, family), atol=1e-10, rtol=1e-10)
        if family == 'binomial':
            assert np.allclose(ds.expected, ds.p*d.n.to_numpy())
    new = df.iloc[:5].copy().reset_index(drop=True)
    new['g'] = ['NEW', 'NEW', 'A', 'NEXT', 'NEW']
    new['h'] = ['NEW', 'A', 'NEW', 'NEXT', 'NEW']
    policies = {'g': 'population', 'h': 'resample_observed'}
    ds = m.predict(idata, newdata=new, kind='mean', new_groups=policies, seed=43)
    rng = np.random.default_rng(43); effects = {}
    for g in ('g', 'h'):
        shape = post[f'1|{g}_sigma'].shape
        for label in ('NEW', 'NEXT'):
            if g == 'g':
                effects[g, label] = rng.normal(size=shape)*post[f'1|{g}_sigma'].values
            else:
                idx = rng.integers(0, 2, size=shape)
                effects[g, label] = np.take_along_axis(post[f'1|{g}'].values, idx[..., None], axis=-1)[..., 0]
    assert np.allclose(ds[m.names['mean']], hand_mean(post, new, intercept, family, effects), atol=1e-10, rtol=1e-10)
    assert abs(np.corrcoef(effects['g','NEW'].ravel(), effects['h','NEW'].ravel())[0,1]) < .12
    assert json.loads(ds.attrs['new_groups_by_factor_json']) == policies
    assert json.loads(ds.attrs['new_levels_by_factor_json']) == {'g':['NEW','NEXT'], 'h':['NEW','NEXT']}
    ds.to_netcdf(tmp_path/'predictions.nc')
    # Row order changes no RNG traversal: factors and unseen labels, never rows, determine it.
    rev = m.predict(idata, newdata=new.iloc[::-1], kind='mean', new_groups=policies, seed=43)
    assert np.array_equal(rev[m.names['mean']].values[..., ::-1], ds[m.names['mean']].values)
    for bad in ('error', {'g':'population','h':'error'}):
        with pytest.raises(ab.FormulaError, match='not seen in training'):
            m.predict(idata, newdata=new, kind='mean', new_groups=bad)
    for bad in ({'g':'population'}, {'g':'population','h':'wrong'}, {'g':'population','h':'error','j':'error'}):
        with pytest.raises(ValueError):
            m.predict(idata, newdata=new, kind='mean', new_groups=bad)
    fit = ab.FitResult(idata, 'pymc', 'cpu', 'cpu', 'cpu', 'float64', None, 2, 400, 0, .9, 0, seed=0, _target_identity_json=json.dumps(m.target_identity))
    rec = ab.fit_record(m, fit, panel=PANEL_STUB)
    json.dumps(rec)
    assert 'group' not in rec['model'] and 'J' not in rec['model']
    assert [g['factor'] for g in rec['model']['groups']] == ['g','h']
    assert rec['model']['priors']['sd:g'] == 'HalfNormal(sigma=0.8)'
    assert rec['model']['priors']['sd:h'] == 'Exponential(lam=1.4)'
    assert rec['model']['center_predictors_requested'] is True
    assert rec['model']['center_predictors'] is intercept
    assert {'1|g_sigma','1|h_sigma'} <= set(rec['posterior'])
    assert ('Intercept' in rec['posterior']) is intercept
    assert ab.compare_records(rec, copy.deepcopy(rec))['agree']
    extra = copy.deepcopy(rec); extra['posterior']['extra'] = extra['posterior']['x']
    for a, b in ((rec, extra), (extra, rec)):
        with pytest.raises(ValueError, match='identical posterior parameter sets'):
            ab.compare_records(a, b)


def test_order_three_groups_and_legacy_construction():
    m, df = make(groups=('h','g','j'))
    assert m.spec.groups == ('h','g','j') and list(m.group_names) == ['h','g','j']
    assert np.array_equal(m.design.Z, np.column_stack([block(df,g) for g in ('h','g','j')]))
    point = m.value_point(**natural('gaussian', True, ('h','g','j')))
    assert {f'1|{g}_sigma_log__' for g in ('h','g','j')} <= set(point)
    a,_=make(); b,_=make(groups=('h','g')); post=posterior('gaussian', True)
    assert np.allclose(a.predict(post, kind='mean').mu, b.predict(post, kind='mean').mu, atol=1e-12)
    old = ab.ModelSpec('y~x+(1|g)', 'y', ('x',), 'g', True, ('A','B'))
    assert old.group == 'g' and old.levels == ('A','B') and old.group_term == '1|g'
    assert replace(old, intercept=False).groups == ('g',)
    assert ab.ModelSpec('y~x','y',('x',),None).group is None
    single,_=make(groups=('g',), sd=ab.HalfNormal(.8))
    mapped,_=make(groups=('g',))
    nat=natural('gaussian',True,('g',)); scalar={**nat,'sd':.6,'z':OFFSETS['g']}
    qa, qb=single.value_point(**scalar), mapped.value_point(**nat)
    assert all(np.array_equal(qa[k],qb[k]) for k in qa)
    new=df.iloc[:3].copy(); new['g']=['NEW','NEXT','NEW']
    post=posterior('gaussian',True,('g',))
    da=single.predict(post,newdata=new,kind='mean',new_groups='population',seed=5)
    db=mapped.predict(post,newdata=new,kind='mean',new_groups={'g':'population'},seed=5)
    assert np.array_equal(da.mu,db.mu) and da.attrs == db.attrs


def test_prior_value_refusals_and_collisions():
    for bad in (ab.HalfNormal(1), {'g':ab.HalfNormal(1)}, {'g':ab.HalfNormal(1),'h':ab.Normal(0,1)},
                {'g':ab.HalfNormal(1),'h':ab.HalfNormal(1),'j':ab.HalfNormal(1)}):
        with pytest.raises(ab.PriorError):
            make(sd=bad)
    m,_=make(); nat=natural('gaussian',True)
    for key, value in [('sd',.6), ('z',np.zeros(3)), ('sd',{'g':.6}), ('z',{'g':OFFSETS['g'],'h':np.zeros(3)}),
                       ('sd',{'g':0.,'h':.2}), ('sd',{'g':np.inf,'h':.2}), ('sigma',-1.),
                       ('z',{'g':np.array([0.,np.nan,1.]),'h':OFFSETS['h']})]:
        with pytest.raises(ValueError):
            m.value_point(**{**nat,key:value})
    pr=ab.Priors(ab.Normal(0,1),ab.Normal(0,1),{'g':ab.HalfNormal(1),'h':ab.HalfNormal(1)},ab.HalfNormal(1))
    df=inputs().assign(g__factor=inputs().z)
    with pytest.raises(ab.FormulaError,match='collision'):
        ab.gaussian('y~x+g__factor+(1|g)+(1|h)',df,pr,categorical={'g__factor':'b'})


@pytest.mark.parametrize('rhs', ['x+(1|g)+(1|g)','x+(1|g/h/j)','x+(1|g:h:j)','x+(0+x||g)',   # A03: two-factor '/' and ':' are accepted
    'x+(1|id|g)','x+(1|log(g))','x+(1+x|mm(g,h))','x+(x|g)','x+(0|g)',   # A02 accepts 'x+(1+x|g)'; D08 '(1|mm(g,h))'
    '(1|g)+(1|h)','0+(1|g)+(1|h)','x+(1|g)+(1|x)','x+(1|g)+(1|y)'])
def test_scoped_formula_refusals(rhs):
    with pytest.raises(ab.FormulaError):
        ab.parse_formula('y~'+rhs, inputs())


def test_binomial_filtered_levels_counts_and_factor_validation():
    df=inputs(); df.loc[:1,'h']='DROPPED'; df.loc[:1,['k','n']]=0
    m,_=make('binomial',df=df)
    assert m.spec.levels_by_group['h'] == ('A','B') and m.dropped_rows == (0,1)
    assert np.array_equal(m.design.group_designs['h'].Z,block(df.iloc[2:],'h'))
    post=posterior('binomial',True); new=inputs().iloc[:3].copy(); new['n']=[0,1,7]
    ds=m.predict(post,newdata=new,kind='response',seed=2)
    assert np.all(ds.y.values[...,0]==0) and np.all(ds.expected.values[...,0]==0)
    assert np.issubdtype(ds.y.dtype,np.integer) and np.all(ds.y.values<=new.n.to_numpy())
    assert np.all(ds.y.values>=0)
    assert 'p' in m.predict(post,newdata=new.drop(columns='n'),kind='probability')
    single=inputs(); single['h']='A'; single.loc[0,'h']='B'; single.loc[0,['k','n']]=0
    with pytest.raises(ab.FormulaError,match="'h' has a single level"):
        make('binomial',df=single)
    for bad in (inputs().assign(h=None), inputs().drop(columns='h')):
        with pytest.raises(ab.FormulaError):
            make(df=bad)
    with pytest.raises(ab.FormulaError,match='trials column'):
        ab.parse_formula('k | trials(n) ~ x + (1|g) + (1|n)', inputs(), 'binomial')


def test_generated_group_names_and_reserved_offsets():
    df=inputs()
    for second in ('g_sigma', 'g_offset', 'county_offset'):
        data=df.assign(**{second:df.h})
        pr=ab.Priors(ab.Normal(0,1), ab.Normal(0,1),
                     {'g':ab.HalfNormal(1),second:ab.HalfNormal(.5)},ab.HalfNormal(1))
        with pytest.raises(ab.FormulaError,match='collision'):
            ab.gaussian(f'y~x+(1|g)+(1|{second})',data,pr)


def test_build_only_example():
    from examples.m3c3_multiple_intercepts import build_example
    m=build_example()
    assert m.spec.groups == ('county','year')
    assert m.design.group_slices == {'county':slice(0,3),'year':slice(3,5)}
    assert np.array_equal(m.design.group_designs['county'].Z,
                          np.array([[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1]]))


@pytest.mark.parametrize('groups', [(), ('g',), ('g','h')])
def test_spec_dataclass_copy_and_serialization(groups):
    import pickle
    from dataclasses import asdict
    formula='y~x'+''.join(f'+(1|{g})' for g in groups)
    spec=ab.parse_formula(formula,inputs())
    for replica in (copy.deepcopy(spec),pickle.loads(pickle.dumps(spec)),ab.ModelSpec(**asdict(spec))):
        assert replica == spec and hash(replica) == hash(spec)
        assert replica.groups == groups
        assert replica.levels_by_group == {g:LEVELS[g] for g in groups}
    assert replace(spec,intercept=False).groups == groups


@pytest.mark.parametrize('groups', [(), ('g',), ('g','h')])
def test_spec_group_metadata_is_immutable(groups):
    source = {g: list(LEVELS[g]) for g in groups}
    spec = ab.ModelSpec('y~x', 'y', ('x',), groups=groups, levels_by_group=source)
    original_hash = hash(spec)
    lookup = {spec: 'original'}
    expected = {g: LEVELS[g] for g in groups}
    with pytest.raises(TypeError):
        spec.levels_by_group['g'] = ('replacement-a', 'replacement-b')
    for levels in source.values():
        levels[:] = ['changed-source']
    source.clear()
    assert spec.levels_by_group == expected
    assert hash(spec) == original_hash and lookup[spec] == 'original'
    assert lookup[copy.deepcopy(spec)] == 'original'
