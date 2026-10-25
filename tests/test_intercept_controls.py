"""No-intercept acceptance: original-input references, no sampling or recovery fits."""
import json
from types import SimpleNamespace

import arviz as az
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, gammaln
from scipy.stats import norm, halfnorm, t
import autobayes as ab
from tests.test_population_only import PANEL_STUB


def source():
    return pd.DataFrame({"x": [-2., 0., 1., 3., 4., 2., -1., 5., 2., 1., 6., -3.],
                         "z": list("bacbcabaccab"), "w": list("uvvuvuvuuvvu"),
                         "g": ["B", "A", "B", "A", "C", "C"] * 2,
                         "y": [1., -1., 2., 4., 3., 2., -2., 6., 3., 0., 5., -1.],
                         "yb": [0, 1, 0, 1, 1, 0] * 2, "k": [0, 1, 3, 4, 2, 1] * 2, "n": [4] * 12})


LHS = {"gaussian": "y", "bernoulli": "yb", "binomial": "k | trials(n)"}
RHS = {"numeric": "x", "categorical": "x + z", "interaction": "x + z + x:z"}


def make(family="gaussian", kind="interaction", group=False, center=True):
    df = source()
    pr = ab.Priors(b=ab.Normal(.2, 1.3), sd=ab.HalfNormal(.8) if group else None,
                   sigma=ab.HalfNormal(1.1) if family == "gaussian" else None)
    m = getattr(ab, family)(f"{LHS[family]} ~ 0 + {RHS[kind]}" + (" + (1|g)" if group else ""), df, pr,
                            categorical={"z": "b"} if kind != "numeric" else None, center_predictors=center)
    return m, df


def independent_design(df, kind):
    # Deliberately uses original caller data and explicit promised labels, never ModelSpec/design.
    cols, labels = [df.x.to_numpy()], ["x"]
    if kind != "numeric":
        cols += [(df.z == level).to_numpy(dtype=float) for level in ("b", "a", "c")]
        labels += ["z[b]", "z[a]", "z[c]"]
    if kind == "interaction":
        cols += [df.x.to_numpy() * (df.z == level).to_numpy() for level in ("a", "c")]
        labels += ["x:z[a]", "x:z[c]"]
    return np.column_stack(cols), labels


@pytest.mark.parametrize("rhs", ["0 + x", "x - 1", "-1 + x", "x + 0", "0+x-1"])
@pytest.mark.parametrize("group", [False, True])
def test_spellings(rhs, group):
    s = ab.parse_formula("y ~ " + rhs + (" + (1|g)" if group else ""), source())
    assert not s.intercept and s.rhs == "0 + x" + (" + (1 | g)" if group else "")


@pytest.mark.parametrize("rhs,match", [("0", "at least one"), ("0 + (1|g)", "at least one"),
    ("x - 2", "only subtracting"), ("x - z", "only subtracting"), ("x - 1.0", "only subtracting"),
    ("x - 1*x", "only subtracting"), ("1 + 0 + x", "conflicting"), ("0 + x + 1", "conflicting"),
    ("x + (0|g)", "varying slopes"), ("0 + x:z", "main effect"), ("0 + x + (0 + x|g)", "varying slopes")])   # A02 accepts '(1+x|g)', F08 '(x|g)'
def test_scoped_refusals(rhs, match):
    with pytest.raises(ab.FormulaError, match=match):
        ab.parse_formula("y ~ " + rhs, categorical={"z": "b"} if "x:z" in rhs else None)


@pytest.mark.parametrize("family", LHS)
@pytest.mark.parametrize("kind", RHS)
@pytest.mark.parametrize("group,center", [(False, True), (False, False), (True, True), (True, False)])
def test_design_graph_and_prior_contract(family, kind, group, center):
    m, df = make(family, kind, group, center)
    X, labels = independent_design(df, kind)
    assert list(m.spec.columns) == labels
    assert np.array_equal(m.design.X_raw, X) and np.array_equal(m.design.X_model, X)
    assert np.array_equal(m.design.x_means, np.zeros(len(labels))) and not m.design.centered
    assert m.center_predictors is center and not m.effective_center_predictors
    assert "intercept" not in m.names and "Intercept" not in m.pymc_model.named_vars
    assert "Intercept" not in m.priors.to_bambi(m.spec, family)
    b = np.linspace(-.4, .7, X.shape[1])
    kw = dict(b=b, sigma=.9) if family == "gaussian" else dict(b=b)
    if group:
        kw.update(sd=.6, z=np.array([-.2, .4, .8]))
    pt = m.value_point(**kw)
    pm = m.pymc_model
    fn = pm.compile_fn(pm.replace_rvs_by_values([pm[m.names["linpred"]]]), inputs=pm.value_vars, point_fn=True, on_unused_input="ignore")
    expected = X @ b + (np.array([-.12, .24, .48])[["ABC".index(g) for g in df.g]] if group else 0)
    assert np.allclose(fn(pt)[0], expected, atol=1e-12)
    with pytest.raises(ValueError, match="no population intercept"):
        m.value_point(**kw, intercept=0.)
    assert "No population intercept or intercept prior" in m.describe()
    assert "~ None" not in m.describe() and "effective=False" in m.describe()


def reference(q, df, family, kind, group, jacobian=True):
    # Independent normalized density, includes change-of-variable terms explicitly.
    X, _ = independent_design(df, kind)
    b = np.concatenate([np.atleast_1d(q[k]) for k in (["x"] + (["z"] if kind != "numeric" else []) + (["x:z"] if kind == "interaction" else []))])
    eta = X @ b
    logp = norm.logpdf(b, .2, 1.3).sum()
    if group:
        sd = np.exp(q["1|g_sigma_log__"]); offsets = q["1|g_offset"]
        eta += sd * offsets[["ABC".index(g) for g in df.g]]
        logp += halfnorm.logpdf(sd, scale=.8) + norm.logpdf(offsets).sum()
        if jacobian: logp += q["1|g_sigma_log__"]
    if family == "gaussian":
        sigma = np.exp(q["sigma_log__"])
        logp += halfnorm.logpdf(sigma, scale=1.1) + norm.logpdf(df.y.to_numpy(), eta, sigma).sum()
        if jacobian: logp += q["sigma_log__"]
    else:
        n = df.n.to_numpy() if family == "binomial" else np.ones(len(df))
        k = df.k.to_numpy() if family == "binomial" else df.yb.to_numpy()
        logp += (gammaln(n+1) - gammaln(k+1) - gammaln(n-k+1) - k*np.logaddexp(0., -eta) - (n-k)*np.logaddexp(0., eta)).sum()
    return float(logp)


@pytest.mark.parametrize("family", LHS)
@pytest.mark.parametrize("kind", RHS)
@pytest.mark.parametrize("group", [False, True])
def test_density_jacobian_and_gradients(family, kind, group):
    import jax
    from pymc.sampling.jax import get_jaxified_logp
    m, df = make(family, kind, group, center=not group)
    pm = m.pymc_model; names = [v.name for v in pm.value_vars]
    lp, lp0, grad = pm.compile_logp(), pm.compile_logp(jacobian=False), pm.compile_dlogp()
    jf = get_jaxified_logp(pm)
    jg = jax.grad(lambda *v: jf(list(v)), argnums=tuple(range(len(names))))
    rng = np.random.default_rng(407)
    for _ in range(2):
        q = {k: np.asarray(rng.normal(size=np.shape(v)) * .3) for k, v in pm.initial_point().items()}
        ref = reference(q, df, family, kind, group)
        assert abs(lp(q)-ref) < 1e-8 and abs(lp0(q)-reference(q, df, family, kind, group, False)) < 1e-8
        assert abs(float(jf([q[k] for k in names])) - ref) < 1e-8
        flat = np.concatenate([np.atleast_1d(q[k]).ravel() for k in names])
        def f(v):
            out, i = {}, 0
            for k in names:
                size = q[k].size;out[k] = v[i:i+size].reshape(q[k].shape);i += size
            return reference(out, df, family, kind, group)
        h=1e-5
        fd=np.array([(f(flat+h*e)-f(flat-h*e))/(2*h) for e in np.eye(len(flat))])
        ptgrad=np.asarray(grad(q))
        jgrad=np.concatenate([np.atleast_1d(v).ravel() for v in jg(*[q[k] for k in names])])
        assert np.allclose(ptgrad, fd, rtol=1e-6, atol=1e-5)
        assert np.allclose(jgrad, ptgrad, rtol=1e-9, atol=1e-9)


def test_prior_coverage_and_default_intercept_compatibility():
    df=source()
    normal=ab.Normal(0,1)
    with pytest.raises(ab.PriorError, match="no population intercept"):
        ab.gaussian("y ~ 0+x",df,ab.Priors(normal,normal,sigma=ab.HalfNormal(1)))
    with pytest.raises(ab.PriorError, match="intercept prior is required"):
        ab.gaussian("y ~ x",df,ab.Priors(b=normal,sigma=ab.HalfNormal(1)))
    with pytest.raises(ab.PriorError, match="b must be"):
        ab.gaussian("y ~ 0+x",df,ab.Priors(sigma=ab.HalfNormal(1)))
    pr=ab.Priors(b={"x":normal, "z":{"b":ab.Normal(.7,2),"a":normal,"c":normal},
                    "x:z":{"a":ab.StudentT(4,.2,1.2),"c":ab.StudentT(6,-.3,2)}},sigma=ab.HalfNormal(1))
    m=ab.gaussian("y ~ x*z -1",df,pr,categorical={"z":"b"})
    assert set(pr.coefficient_priors(m.spec))=={"x","z[b]","z[a]","z[c]","x:z[a]","x:z[c]"}
    q=m.value_point(b=[.1,.2,.3,.4,.5,.6],sigma=.8)
    lp = (norm.logpdf(df.y, independent_design(df,"interaction")[0] @ np.array([.1,.2,.3,.4,.5,.6]),.8).sum()
          + norm.logpdf([.1,.3,.4]).sum()+norm.logpdf(.2,.7,2)+t.logpdf(.5,4,loc=.2,scale=1.2)
          +t.logpdf(.6,6,loc=-.3,scale=2)+halfnorm.logpdf(.8)+np.log(.8))
    assert abs(m.pymc_model.compile_logp()(q)-lp)<1e-8
    for term in ("z","x:z"):
        bad=dict(pr.b);bad[term] = dict(bad[term])
        if term=="z":del bad[term]["b"]
        else:bad[term]["b"]=normal
        with pytest.raises(ab.PriorError,match="must cover exactly"):
            ab.gaussian("y ~ 0+x*z",df,ab.Priors(b=bad,sigma=ab.HalfNormal(1)),categorical={"z":"b"})
    old=ab.gaussian("y~x",df,ab.Priors(normal,normal,sigma=ab.HalfNormal(1)))
    assert old.design.centered and old.spec.intercept
    with pytest.raises(ValueError,match="intercept is required"):
        old.value_point(b=[.1],sigma=1.)
    assert "Intercept" in old.value_point(intercept=0.,b=[.1],sigma=1.)


@pytest.mark.parametrize("first", ["z","w"])
def test_multiple_factors_only_first_spans_intercept(first):
    df=source();other="w" if first=="z" else "z"
    m=ab.gaussian(f"y~0+x+{first}+{other}+x:z",df,ab.Priors(b=ab.Normal(0,1),sigma=ab.HalfNormal(1)),categorical={"z":"b","w":"v"})
    expected=[df.x.to_numpy()];labels=["x"]
    for cat in (first,other):
        levels=(["b","a","c"] if cat=="z" else ["v","u"])
        if cat != first:levels=levels[1:]
        expected.extend((df[cat]==l).to_numpy(dtype=float) for l in levels);labels.extend(f"{cat}[{l}]" for l in levels)
    expected.extend(df.x.to_numpy()*(df.z==l) for l in ("a","c"));labels.extend(["x:z[a]","x:z[c]"])
    assert list(m.spec.columns)==labels
    assert np.array_equal(m.design.X_raw,np.column_stack(expected))
    assert "Reordering factors changes independent prior meaning" in m.describe()


def fixed_draws(group, gaussian):
    rng=np.random.default_rng(16)
    vals={"x":rng.normal(size=(2,80)),"z":rng.normal(size=(2,80,3)),"x:z":rng.normal(size=(2,80,2))}
    dims={"z":["z_dim"],"x:z":["x:z_dim"]};coords={"z_dim":["b","a","c"],"x:z_dim":["a","c"]}
    if gaussian:vals["sigma"]=np.full((2,80),.8)
    if group:
        vals["1|g_sigma"]=np.full((2,80),.6);vals["1|g"]=rng.normal(size=(2,80,3))*.6
        dims["1|g"]=["g__factor_dim"];coords["g__factor_dim"]=["A","B","C"]
    return az.from_dict(posterior=vals,dims=dims,coords=coords)


@pytest.mark.parametrize("family", LHS)
@pytest.mark.parametrize("group", [False,True])
def test_fixed_draw_predictions_and_records(family, group):
    m,df=make(family,group=group);id=fixed_draws(group,family=="gaussian");p=id.posterior
    b=np.concatenate([p.x.values[...,None],p.z.values,p["x:z"].values],axis=-1)
    for data in [None,pd.DataFrame({"x":[0.,-2.,4.,0.],"z":["b","a","c","b"],"g":["A","B","C","A"],"n":[4,0,2,4]})]:
        d=df if data is None else data
        expected=np.einsum("cdp,np->cdn",b,independent_design(d,"interaction")[0])
        if group:expected+=p["1|g"].values[:,:,["ABC".index(g) for g in d.g]]
        expected=expected if family=="gaussian" else expit(expected)
        out=m.predict(id,newdata=data,kind="mean")
        assert np.allclose(out["mu" if family=="gaussian" else "p"],expected,atol=1e-12)
        if family=="binomial":assert np.allclose(out.expected,expected*d.n.to_numpy())
    if group:
        new=pd.DataFrame({"x":[1.,1.,1.],"z":["b"]*3,"g":["NEW","NEW","NEXT"],"n":[2]*3})
        for policy in ["population","resample_observed"]:
            out=m.predict(id,newdata=new,new_groups=policy,kind="mean",seed=20)
            a=out["mu" if family=="gaussian" else "p"].values
            assert np.array_equal(a[...,0],a[...,1])
            # One sampled group effect per label, reconstructed independently from the seed.
            rng=np.random.default_rng(20);effects={}
            for label in ["NEW","NEXT"]:
                if policy=="population":effects[label]=rng.normal(size=(2,80))*.6
                else:
                    indexes=rng.integers(0,3,size=(2,80));effects[label]=np.take_along_axis(p["1|g"].values,indexes[...,None],axis=-1)[...,0]
            eta=p.x.values+p.z.values[...,0]
            expected=np.stack([eta+effects[l] for l in new.g],axis=-1)
            assert np.allclose(a,expected if family=="gaussian" else expit(expected))
    fit=SimpleNamespace(target_identity=m.target_identity,idata=id,backend="pymc",device_requested="cpu",device_actual="cpu",platform="cpu",dtype="float64",chain_method=None,
                        chains=2,draws=80,tune=0,target_accept=.9,seed=16,wall_seconds=0.,notes=["synthetic draws; no sampling"])
    record=ab.fit_record(m,fit,panel=PANEL_STUB);json.dumps(record)
    assert record["model"]["intercept"] is False and record["model"]["intercept_prior_applies_to"] is None
    assert not record["model"]["center_predictors"] and record["model"]["center_predictors_requested"] is True
    assert "intercept" not in record["model"]["priors"] and "Intercept" not in record["posterior"]
    assert record["model"]["categorical"]["z"]["coding"]=="full_level"


def test_default_intercept_categorical_record_retains_semantics():
    from tests.test_population_only import synthetic_fit
    m=ab.gaussian("y ~ x + z",source(),ab.Priors(ab.Normal(0,1),ab.Normal(0,1),sigma=ab.HalfNormal(1)),categorical={"z":"b"})
    record=ab.fit_record(m,synthetic_fit(m),panel=PANEL_STUB)
    assert record["model"]["categorical"]=={"z":{"reference":"b","levels":["a","b","c"],"coefficients":["a","c"],"unused_levels":[],"coding":"treatment"}}
    assert record["model"]["intercept"] and record["model"]["center_predictors"]
    assert record["model"]["center_predictors_requested"]
    assert "Intercept" in record["posterior"] and "intercept" in record["model"]["priors"]
