"""Group-level terms from integer level codes (docs/scaling.md): the lazy dense blocks against formulae's dense design and the
model's own construction (sign bits included), the size cap, the streamed target identity against the v1 JSON, the gathers in
predict() and log_lik() against a dense reference (new levels included), the checks of Bambi's terms, the guard on Bambi's
private layout, and a memory test at a size whose dense indicator could not be built. Build-only; nothing samples."""
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import arviz as az
import formulae
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.special import expit

import autobayes as ab
from autobayes import group_index as gi
from autobayes import identity
from autobayes.model import GroupBlock

REPO = Path(__file__).resolve().parent.parent
HN = ab.HalfNormal(1)
BASE = dict(intercept=ab.Normal(0, 5), b=ab.Normal(0, 1), sigma=HN)


def frame(n=48, seed=3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x1[:4] = [0.0, -0.0, -1.5, 2.0]                                   # a zero, a negative zero and a negative slope value
    g = np.array(list("abcde"))[np.arange(n) % 5]
    g2 = np.where(np.arange(n) % 4 == 0, g, rng.choice(list("abcde"), n))   # every fourth row has the same level twice
    df = pd.DataFrame({"y": rng.normal(size=n), "yb": rng.integers(0, 2, n), "x1": x1, "x2": rng.normal(size=n),
                       "k": rng.integers(0, 4, n), "z": rng.normal(size=n), "g": g, "g2": g2,
                       "h": np.array(list("pqr"))[np.arange(n) % 3], "w1": rng.uniform(0.2, 1.2, n), "w2": rng.uniform(0.2, 1.2, n),
                       "ord": rng.integers(1, 4, n)})
    df["gby"] = np.where(np.isin(df.g, ["a", "b"]), "u", "v")
    return df


DF = frame()
A = pd.DataFrame(np.eye(5) + 0.2 * (np.ones((5, 5)) - np.eye(5)), index=list("abcde"), columns=list("abcde"))


def bern():
    return ab.bernoulli("yb ~ x1 + x2 + (1 + x1 | g) + (1 | h)", DF, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1),
                        sd={"g": {"Intercept": HN, "x1": HN}, "h": HN}, cor={"g": ab.LKJ(2)}))


MODELS = {
    "bern": bern,
    "int_slope": lambda: ab.gaussian("y ~ x1 + (0 + k || g)", DF, ab.Priors(**BASE, sd={"*": HN})),
    "derived": lambda: ab.gaussian("y ~ x1 + (1 | g:h)", DF, ab.Priors(**BASE, sd=HN)),
    "mm": lambda: ab.gaussian("y ~ x1 + (1 + x1 | mm(g, g2, weights = cbind(w1, w2))) + (1 | h)", DF,
                              ab.Priors(**BASE, sd={"mmgg2": {"Intercept": HN, "x1": HN}, "h": HN}, cor={"mmgg2": ab.LKJ(2)})),
    "by": lambda: ab.gaussian("y ~ x1 + (1 | gr(g, by = gby))", DF, ab.Priors(**BASE, sd=HN)),
    "cov": lambda: ab.gaussian("y ~ x1 + (1 + x1 | gr(g, cov = A))", DF, ab.Priors(**BASE, sd={"g": {"Intercept": HN, "x1": HN}},
                                                                                     cor={"g": ab.LKJ(2)}), data2={"A": A}),
    "student": lambda: ab.gaussian('y ~ x1 + (1 + x1 || gr(g, dist = "student"))', DF,
                                   ab.Priors(**BASE, sd={"g": {"Intercept": HN, "x1": HN}}, df={"g": ab.Gamma(2, 0.1)})),
    "mo": lambda: ab.gaussian("y ~ mo(ord) + (1 + mo(ord) | g)", DF, ab.Priors(**BASE, sd={"g": {"Intercept": HN, "mo(ord)": HN}},
                                                                            cor={"g": ab.LKJ(2)}, simo={"ord": ab.Dirichlet(2, 2)})),
    "dpar": lambda: ab.gaussian(["y ~ x1 + (1 + x1 |p| g)", "sigma ~ z + x1 + (1 + x1 |p| g) + (1 | h)"], DF, ab.Priors(
        intercept=ab.Normal(0, 5), b=ab.Normal(0, 1), sd={"g": {"Intercept": HN, "x1": HN}}, cor={"g": ab.LKJ(2)},
        sigma={"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 1), "sd": {"g": {"Intercept": HN, "x1": HN}, "h": HN}})),
}


@pytest.fixture(scope="module")
def models():
    return {name: build() for name, build in MODELS.items()}


def same_bits(a, b) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b) and np.array_equal(np.signbit(a), np.signbit(b))


def formulae_z(formula: str, frame: pd.DataFrame) -> np.ndarray:
    """formulae's dense group design of ``formula``: what Bambi held, and what the blocks were cut from, before G11."""
    return np.asarray(formulae.design_matrices(formula, frame, na_action="error").group.design_matrix.toarray(), dtype=float)


def added_z(block) -> np.ndarray:
    """The model's own construction of a block (mm(), gr(by = , cov = , dist = )): zeros, then each member's weight (times the
    covariate) added at its level, np.add.at, as the blocks were built before G11."""
    n, J = len(block.group_index), block.J
    Z = np.zeros((n, J))
    members = block.memberships or (block.group_index,)
    weights = block.weights or (np.ones(n),)
    covariates = block.covariates or ((block.covariate,) if block.covariate is not None else (None,) * len(members))
    for index, weight, cov in zip(members, weights, covariates):
        np.add.at(Z, (np.arange(n), index), weight if cov is None else weight * cov)
    return Z


# -- the lazy dense views ----------------------------------------------------------------------------------------------------
def test_lazy_z_equals_formulae_s_dense_design_bit_for_bit(models):
    for name in ("bern", "int_slope", "derived"):
        m = models[name]
        ref = formulae_z(m.spec.bambi_formula, m.data)
        assert same_bits(m.design.Z, ref), name
        for key, block in m.design.group_blocks.items():
            assert same_bits(block.Z, ref[:, block.column_slice]) and block.dense is None, (name, key)
    block = models["bern"].design.group_blocks[("g", "x1")]
    assert np.signbit(DF.x1.to_numpy()[1]) and not np.signbit(block.Z[1]).any()        # -0.0 is a 0.0 in Z, as formulae gave it
    assert block.Z[2, block.group_index[2]] == -1.5 and block.covariate.dtype == np.float64
    d = models["dpar"].design.dpars["sigma"]
    ref = formulae_z("y ~ (1 + x1 | g) + (1 | h)", models["dpar"].data)
    assert list(d.group_blocks) == [("g", "Intercept"), ("g", "x1"), ("h", "Intercept")]
    for key, b in d.group_blocks.items():
        assert same_bits(b.Z, ref[:, b.column_slice]), key


def test_lazy_z_equals_the_models_own_construction(models):
    for name in ("mm", "by", "cov", "student"):
        m = models[name]
        for key, block in m.design.group_blocks.items():
            ref = formulae_z("y ~ (1 | h)", m.data) if key[0] == "h" else added_z(block)
            assert same_bits(block.Z, ref), (name, key)
        assert same_bits(m.design.Z, np.concatenate([b.Z for b in m.design.group_blocks.values()], axis=1)), name
    mm = models["mm"].design.group_blocks[("mmgg2", "Intercept")]
    twice = np.flatnonzero(mm.memberships[0] == mm.memberships[1])
    assert twice.size and np.allclose(mm.Z[twice, mm.memberships[0][twice]], mm.weights[0][twice] + mm.weights[1][twice])
    mo = models["mo"].design.group_blocks[("g", "mo(ord)")]
    assert mo.monotonic == "ord" and same_bits(mo.Z, np.eye(5)[mo.group_index])


def test_a_dense_block_stays_dense_and_replace_still_takes_z():
    import dataclasses

    Z = np.array([[1.0, 0.0], [0.0, -0.0]])
    b = GroupBlock(("a", "b"), Z, np.array([0, 1]), slice(0, 2))
    assert b.Z is Z and b.dense is Z and b.n == 2 and b.dot(np.array([2.0, 3.0])).tolist() == [2.0, 0.0]
    c = dataclasses.replace(b, Z=2 * Z)
    assert same_bits(c.Z, 2 * Z) and c.column_slice == slice(0, 2)
    lazy = GroupBlock(("a", "b"), None, np.array([1, 0, 1]), slice(0, 2), covariate=np.array([-0.0, 2.0, -3.0]))
    moved = dataclasses.replace(lazy, column_slice=slice(4, 6))
    assert moved.dense is None and same_bits(moved.Z, lazy.Z) and moved.column_slice == slice(4, 6)
    out = np.full((3, 2), np.nan)
    b2 = GroupBlock(("a", "b"), Z[[0, 1, 1]], np.array([0, 1, 1]), slice(0, 2))
    b2.fill(out)
    assert same_bits(out, Z[[0, 1, 1]])
    with pytest.raises(ValueError, match="a dense GroupBlock has no row values"):
        b.row_values()
    with pytest.raises(TypeError, match="GroupBlock needs group_index and column_slice"):
        GroupBlock(("a",), None)


def test_row_values_and_dot_reproduce_z(models):
    rng = np.random.default_rng(0)
    for name, m in models.items():
        for key, block in m.design.group_blocks.items():
            Z = block.Z
            cols, vals = block.row_values()
            rebuilt = np.zeros_like(Z)
            for i in range(Z.shape[0]):
                for c, v in zip(cols[i], vals[i]):
                    if c >= 0:
                        rebuilt[i, c] = v
            assert np.array_equal(rebuilt, Z) and (np.diff(np.where(cols < 0, 10**9, cols), axis=1) >= 0).all(), (name, key)
            v = rng.normal(size=block.J)
            assert np.allclose(block.dot(v), Z @ v, rtol=1e-13, atol=1e-13), (name, key)


def test_the_dense_views_refuse_above_their_cap(models, monkeypatch):
    m = models["bern"]
    monkeypatch.setattr(gi, "DENSE_Z_MAX_ENTRIES", 48 * 5 - 1)
    with pytest.raises(ValueError, match=re.escape("GroupBlock.Z would be a dense 48 x 5 array (240 entries, more than "
                                                   "group_index.DENSE_Z_MAX_ENTRIES = 239)")):
        m.design.group_blocks[("g", "Intercept")].Z
    with pytest.raises(ValueError, match=re.escape("Design.Z would be a dense 48 x 13 array")):
        m.design.Z
    assert m.design.group_blocks[("h", "Intercept")].Z.shape == (48, 3)            # 144 entries: under the cap
    assert m.design.group_blocks[("g", "x1")].dot(np.ones(5)).shape == (48,)       # the gathers need no dense view


# -- the streamed target identity ------------------------------------------------------------------------------------------
def v1_digest(payload) -> str:
    """The schema-v1 digest as it was computed before G11: the whole canonical JSON, then its sha256."""
    text = json.dumps(identity._canonical(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dense_payload(value):
    """``value`` with every LazyZ replaced by its block's dense Z."""
    if isinstance(value, identity.LazyZ):
        return value.block.Z
    if isinstance(value, dict):
        return {k: dense_payload(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(dense_payload(v) for v in value)
    return value


EDGE = [0.0, -0.0, 1.0, -1.5, 5e-324, -5e-324, 2.2250738585072014e-308, 1.7976931348623157e308, -1.7976931348623157e308,
        1e-300, 0.1, 1 / 3, 2.0**52 + 1]


def random_payload(rng) -> dict:
    return {
        "floats": np.array(EDGE + rng.normal(size=7).tolist()), "ints": np.arange(-3, 4), "uints": np.arange(4, dtype=np.uint8),
        "bools": np.array([True, False, True]), "matrix": rng.normal(size=(3, 4)), "empty": np.zeros((0, 3)), "scalar": np.float64(-0.0),
        "zero_d": np.array(2.5), "objects": np.array(["a", None, 1.5], dtype=object), "text": np.array(["é", "a\"b"]),
        "nested": [{"b": (1, 2.5, None), "a": [True, "x y", -0.0]}, ()], "number": -7, "flag": False, "none": None,
        "float32": np.array([0.1, -0.0], dtype=np.float32), "unicode": "zßé\U0001f600",
    }


def lazy_blocks(rng) -> list[GroupBlock]:
    n = 7
    return [
        GroupBlock(("a",), None, np.zeros(n, dtype=np.int64), slice(0, 1)),                                       # J = 1
        GroupBlock(("a", "b", "c"), None, rng.integers(0, 3, n), slice(0, 3), covariate=np.array([0.0, -0.0, -1.5, 2.0, 0.25, -7.0, 1e-300])),
        GroupBlock(("a", "b", "c", "d"), None, np.array([0, 3, 1, 3, 2, 0, 3]), slice(0, 4),                      # duplicate members
                   memberships=(np.array([0, 3, 1, 3, 2, 0, 3]), np.array([0, 1, 1, 3, 0, 2, 3]), np.array([0, 1, 2, 3, 0, 2, 0])),
                   weights=(rng.uniform(size=n), rng.uniform(size=n), rng.uniform(size=n)),
                   covariates=(rng.normal(size=n), -rng.uniform(size=n), np.zeros(n))),
        GroupBlock(("a", "b"), None, np.array([1, 1, 0, 0, 1, 0, 1]), slice(0, 2),
                   memberships=(np.array([1, 1, 0, 0, 1, 0, 1]), np.array([1, 0, 0, 1, 1, 0, 1])), weights=(np.full(n, 0.5), np.full(n, 0.5))),
        GroupBlock(("a", "b", "c"), None, np.array([2, 2, 2, 2, 2, 2, 2]), slice(0, 3)),                           # last column only
    ]


def test_the_streamed_digest_equals_the_v1_json_on_random_payloads():
    rng = np.random.default_rng(42)
    for _ in range(4):
        payload = random_payload(rng)
        assert identity.digest(payload) == v1_digest(payload)
    for block in lazy_blocks(rng):
        payload = {"Z": identity.LazyZ(block), "index": block.group_index, "levels": block.levels}
        assert identity.digest(payload) == v1_digest(dense_payload(payload)), block
        assert identity.z_identity(block).block is block
    dense = GroupBlock(("a", "b"), np.eye(2), np.array([0, 1]), slice(0, 2))
    assert identity.z_identity(dense) is dense.Z
    long = rng.normal(size=70_000)                                   # more than one chunk of elements, and more than one buffer
    assert identity.digest({"long": long, "bools": long > 0}) == v1_digest({"long": long, "bools": long > 0})
    wide = GroupBlock(tuple(f"l{j}" for j in range(50_000)), None, np.array([0, 49_999, 25_000]), slice(0, 50_000))
    assert identity.digest({"Z": identity.LazyZ(wide)}) == v1_digest({"Z": wide.Z})
    tall = GroupBlock(("a", "b", "c"), None, rng.integers(0, 3, 50_000), slice(0, 3), covariate=rng.normal(size=50_000))
    assert identity.digest({"Z": identity.LazyZ(tall)}) == v1_digest({"Z": tall.Z})   # many short pieces through the buffer
    empty = GroupBlock(("a", "b"), None, np.zeros(0, dtype=np.int64), slice(0, 2))
    assert identity.digest([identity.LazyZ(empty)]) == v1_digest([empty.Z])


def test_the_streamed_digest_refuses_what_the_canonical_form_refuses():
    for bad, error, message in [({"x": np.array([1.0, np.nan])}, ValueError, "finite"), ({"x": math.inf}, ValueError, "finite"),
                                ({1: 2}, TypeError, "string keys"), ({"x": {1, 2}}, TypeError, "unsupported target identity type: set"),
                                ({"x": np.array([1j])}, TypeError, "unsupported target identity type: complex")]:
        with pytest.raises(error, match=message):
            identity._canonical(bad)
        with pytest.raises(error, match=message):
            identity.digest(bad)


def test_every_model_s_identity_is_the_v1_digest(models, monkeypatch):
    """Each identity function hands its payload to ``identity.digest``; the digest of every model equals the sha256 of the whole
    v1 JSON of the same payload with dense blocks (the response formula's, a distributional formula's, a non-linear, count-matrix
    and multivariate model's)."""
    seen = []
    streamed = identity.digest

    def both(payload):
        out = streamed(payload)
        assert out == v1_digest(dense_payload(payload))
        seen.append(out)
        return out
    monkeypatch.setattr(identity, "digest", both)
    for name, build in MODELS.items():
        assert build().target_identity["sha256"] == models[name].target_identity["sha256"] == seen[-1], name
    nl = ab.gaussian(["y ~ a * exp(b * x1)", "a ~ 1 + (1 | g)", "b ~ 1"], DF,
                     ab.Priors(nlpars={"a": {"b": ab.Normal(1, 1), "sd": HN}, "b": {"b": ab.Normal(0, 1)}}, sigma=HN), nl=True)
    counts = DF.assign(c1=np.arange(48) % 3 + 1, c2=(np.arange(48) // 3) % 4, c3=np.arange(48) % 2)
    counts["size"] = counts.c1 + counts.c2 + counts.c3
    mn = ab.multinomial("cbind(c1, c2, c3) | trials(size) ~ x1 + (1 | g)", counts,
                        ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd=HN))
    mv = ab.gaussian("mvbind(y, z) ~ x1 + (1 | g)", DF, ab.Priors(resp={r: {**BASE, "sd": HN} for r in ("y", "z")}), rescor=False)
    assert len({nl.target_identity["sha256"], mn.target_identity["sha256"], mv.target_identity["sha256"]}) == 3
    assert seen[-1] == mv.target_identity["sha256"]


# -- predictions and log_lik: one gather per coefficient -----------------------------------------------------------------
def synthetic_posterior(m, chains=2, draws=3, seed=1):
    """Draws of the graph's own variables (no per-row ones) at seeded points of the value space."""
    pmm = m.pymc_model
    dims_of = pmm.named_vars_to_dims
    keep = [v for v in pmm.unobserved_RVs if "__obs__" not in (dims_of.get(v.name) or ())]
    fn = pmm.compile_fn(pmm.replace_rvs_by_values(keep), inputs=pmm.value_vars, on_unused_input="ignore", point_fn=True)
    rng, ip = np.random.default_rng(seed), pmm.initial_point()
    values = [fn({k: np.asarray(v, dtype=float) + 0.3 * rng.normal(size=np.shape(v)) for k, v in ip.items()})
              for _ in range(chains * draws)]
    data, used = {}, set()
    for i, v in enumerate(keep):
        arr = np.stack([np.asarray(x[i]) for x in values]).reshape(chains, draws, *np.shape(values[0][i]))
        dims = tuple(dims_of.get(v.name) or ()) or tuple(f"{v.name}_dim_{j}" for j in range(arr.ndim - 2))
        data[v.name] = (("chain", "draw", *dims), arr)
        used.update(dims)
    coords = {k: list(c) for k, c in pmm.coords.items() if k in used}
    return az.InferenceData(posterior=xr.Dataset(data, coords={**coords, "chain": np.arange(chains), "draw": np.arange(draws)}))


def dense_group_part(m, post, frame, extra=None):
    """sum over coefficients of Z_c @ u_c on ``frame``, Z_c dense from the labels (``extra``: each factor's unseen labels' effects
    per coefficient, appended to the fitted levels')."""
    out = 0.0
    for t in m.spec.group_terms:
        levels = list(m.design.factor_blocks[t.factor].levels)
        new = sorted(set(frame[t.column]) - set(levels))
        ind = (frame[t.column].to_numpy()[:, None] == np.array(levels + new, dtype=object)[None, :]).astype(float)
        for c in t.coefs:
            u = post[m.group_coefficient_names[(t.factor, c)]["group_effect"]].to_numpy()
            if new:
                u = np.concatenate([u, np.stack(extra[t.factor][c], axis=-1)], axis=-1)
            cov = np.ones(len(frame)) if c == "Intercept" else frame[c].to_numpy(dtype=float)
            out = out + np.einsum("nj,cdj->cdn", ind * cov[:, None], u)
    return out


def test_predictions_and_log_lik_equal_a_dense_reference(models):
    m = models["bern"]
    idata = synthetic_posterior(m)
    post = idata.posterior
    pop = ab.predict(m, idata, kind="linear", include_group_effects=False)["eta"].to_numpy()
    lin = ab.predict(m, idata, kind="linear")["eta"].to_numpy()
    assert np.allclose(lin, pop + dense_group_part(m, post, m.data), rtol=1e-12, atol=1e-12)
    y = m.design.y
    expected = y * np.log(expit(lin)) + (1 - y) * np.log(expit(-lin))
    assert np.allclose(ab.log_lik(m, idata).to_numpy(), expected, rtol=1e-11, atol=1e-12)
    mean = ab.predict(m, idata, kind="mean")
    assert np.allclose(mean["p"].to_numpy(), expit(lin), rtol=1e-12) and mean.attrs["new_groups"] == "in_sample"
    nd = DF.iloc[:12].assign(g=["new2", "a", "new1", "b", "new2", "c", "new1", "d", "e", "a", "new2", "b"])
    # brms's sample_new_levels = "gaussian": per sorted unseen label a z per coefficient, diag(sd) L z
    sd = [post[m.group_coefficient_names[("g", c)]["sd"]].to_numpy() for c in ("Intercept", "x1")]
    L = post["L|g"].transpose("chain", "draw", ...).to_numpy()
    rng, extra = np.random.default_rng(5), {"g": {"Intercept": [], "x1": []}}
    for _ in ("new1", "new2"):
        z = [rng.normal(0.0, 1.0, size=sd[0].shape) for _ in range(2)]
        extra["g"]["Intercept"].append(sd[0] * L[..., 0, 0] * z[0])
        extra["g"]["x1"].append(sd[1] * (L[..., 1, 0] * z[0] + L[..., 1, 1] * z[1]))
    got = ab.predict(m, idata, newdata=nd, kind="linear", new_groups="population", seed=5)
    nd_frame = nd.assign(g=nd.g.astype(str), h=nd.h.astype(str))
    pop_new = ab.predict(m, idata, newdata=nd, kind="linear", include_group_effects=False)["eta"].to_numpy()
    assert np.allclose(got["eta"].to_numpy(), pop_new + dense_group_part(m, post, nd_frame, extra), rtol=1e-12, atol=1e-12)
    assert got.attrs["new_levels_by_factor_json"] == json.dumps({"g": ["new1", "new2"], "h": []})
    # brms's default "uncertainty": one fitted level per label and draw, its whole effect vector
    rng, extra = np.random.default_rng(9), {"g": {"Intercept": [], "x1": []}}
    for _ in ("new1", "new2"):
        chosen = rng.integers(0, 5, size=sd[0].shape)
        for c in ("Intercept", "x1"):
            u = post[m.group_coefficient_names[("g", c)]["group_effect"]].to_numpy()
            extra["g"][c].append(np.take_along_axis(u, chosen[..., None], axis=-1)[..., 0])
    got = ab.predict(m, idata, newdata=nd, kind="linear", new_groups="resample_observed", seed=9)["eta"].to_numpy()
    assert np.allclose(got, pop_new + dense_group_part(m, post, nd_frame, extra), rtol=1e-12, atol=1e-12)


def test_multi_membership_per_category_and_distributional_gathers(models):
    m = models["mm"]
    idata = synthetic_posterior(m)
    post = idata.posterior
    pop = ab.predict(m, idata, kind="linear", include_group_effects=False)["eta"].to_numpy()
    group = sum(np.einsum("nj,cdj->cdn", b.Z, post[m.group_coefficient_names[k]["group_effect"]].to_numpy())
                for k, b in m.design.group_blocks.items())
    assert np.allclose(ab.predict(m, idata, kind="linear")["eta"].to_numpy(), pop + group, rtol=1e-12, atol=1e-12)
    d = models["dpar"]
    idata = synthetic_posterior(d)
    post = idata.posterior
    eta = ab.predict(d, idata, kind="linear", dpar="sigma")["eta"].to_numpy()
    eta0 = ab.predict(d, idata, kind="linear", dpar="sigma", include_group_effects=False)["eta"].to_numpy()
    group = sum(np.einsum("nj,cdj->cdn", b.Z, post[d.dpar_group_names[("sigma", *k)]["group_effect"]].to_numpy())
                for k, b in d.design.dpars["sigma"].group_blocks.items())
    assert np.allclose(eta, eta0 + group, rtol=1e-12, atol=1e-12)
    rng = np.random.default_rng(1)
    p = rng.dirichlet([2.0, 3.0, 4.0], size=24)
    simplex = pd.DataFrame({"a": p[:, 0], "b": p[:, 1], "c": 1.0 - p[:, 0] - p[:, 1], "x": rng.normal(size=24),
                            "g": np.array(list("uvwxyz"))[np.arange(24) % 6]})
    c = ab.dirichlet("cbind(a, b, c) ~ x + (1 | g)", simplex, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd=HN,
                                                                          phi=ab.Gamma(2, 0.5)))
    idata = synthetic_posterior(c)
    post = idata.posterior
    eta = ab.predict(c, idata, kind="linear")["eta"].to_numpy()
    eta0 = ab.predict(c, idata, kind="linear", include_group_effects=False)["eta"].to_numpy()
    u = post[c.group_coefficient_names[("g", "Intercept")]["group_effect"]].to_numpy()
    assert np.allclose(eta, eta0 + np.einsum("nj,cdjk->cdnk", c.design.group_blocks[("g", "Intercept")].Z, u), rtol=1e-12, atol=1e-12)


# -- the checks of Bambi's terms -------------------------------------------------------------------------------------------
def test_the_index_group_terms_bambi_builds_from(models):
    m = models["bern"]
    comp = m.bambi_model.components[m.family_spec.parent]
    assert comp.design.group is None and list(comp.group_specific_terms) == ["1|g", "x1|g", "1|h"]
    t = comp.group_specific_terms["x1|g"]
    assert t.name == "x1|g" and t.kind == "numeric" and t.shape == (48, 5) and not t.categorical and t.levels == t.groups == list("abcde")
    assert t.predictor is not None and t.group_index.dtype == np.int64 and t.prior.auto_scale is False and t.coords == {"g__factor_dim": list("abcde")}
    assert comp.group_specific_terms["1|g"].predictor.dtype == np.dtype(int)
    with pytest.raises(AttributeError, match="has no formulae term"):
        t.term
    with pytest.raises(AttributeError, match="not formulae's dense indicator"):
        t.grouper
    import pickle

    back = pickle.loads(pickle.dumps(t))
    assert type(back) is type(t) is gi.IndexGroupTerm and back.name == "x1|g" and np.array_equal(back.group_index, t.group_index)
    with pytest.raises(AttributeError, match="has no attribute 'NoSuchTerm'"):
        gi.NoSuchTerm
    sigma = models["dpar"].bambi_model.components["sigma"].group_specific_terms["x1|g"]
    assert sigma.name == "sigma_x1|g" and sigma.prefix == "sigma"
    assert models["int_slope"].bambi_model.components["mu"].group_specific_terms["k|g"].predictor.dtype == DF.k.dtype   # formulae's
    assert m.spec.bambi_formula.endswith("+ (1 + x1 | g) + (1 | h)") and m.spec.bambi_population_formula == "yb ~ x1 + x2"


def test_the_population_formula_bambi_is_handed():
    from autobayes.formula import parse_formula

    assert parse_formula("k | trials(n) ~ x + (1 | g)", None, "binomial").bambi_population_formula == "p(k, n) ~ x"
    assert parse_formula("y ~ 0 + x + (1 | g)").bambi_population_rhs == "0 + x"
    assert parse_formula("y ~ 1 + (1 | g)").bambi_population_rhs == "1"
    assert parse_formula("y ~ x + offset(log(e)) + (1 | g)").bambi_population_rhs == "x + offset(log(e))"


def test_extract_design_refuses_terms_other_than_its_own(models):
    m = bern()
    comp = m.bambi_model.components[m.family_spec.parent]
    terms = dict(comp.terms)
    t1, tx = terms["1|g"], terms["x1|g"]
    cases = [
        (lambda mp: mp.setattr(comp.design, "group", object()), "Bambi built a group design from the formula"),
        (lambda mp: mp.setattr(comp, "terms", {k: v for k, v in terms.items() if k != "1|h"}), "Bambi group terms differ from formula order"),
        (lambda mp: mp.setattr(t1, "_groups", list("edcba")), "Bambi level order for 'g' differs from sorted fitted labels"),
        (lambda mp: mp.setattr(t1, "group_index", (t1.group_index + 1) % 5), "group design for 'g' does not encode original rows one-hot"),
        (lambda mp: mp.setattr(t1, "_predictor", 2 * t1.predictor), "group design for 'g' does not encode original rows one-hot"),
        (lambda mp: mp.setattr(tx, "_kind", "categoric"), "Bambi coded the varying slope 'x1|g' as 'categoric'; expected a numeric slope"),
        (lambda mp: mp.setattr(tx, "_predictor", tx.predictor + 1),
         "Bambi's block for 'x1|g' is not the level indicator of 'g' times the 'x1' column"),
    ]
    for change, message in cases:
        with pytest.MonkeyPatch.context() as mp:
            change(mp)
            with pytest.raises(RuntimeError, match=re.escape(message)):
                m._extract_design()
    assert m._extract_design().group_blocks.keys() == m.design.group_blocks.keys()
    pop = ab.gaussian("y ~ x1", DF, ab.Priors(**BASE))
    pcomp = pop.bambi_model.components["mu"]
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(pcomp.terms, "1|g", t1)
        with pytest.raises(RuntimeError, match="Bambi built a group design for a formula without a group-level term"):
            pop._extract_design()
    d = models["dpar"]
    dcomp = d.bambi_model.components["sigma"]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(dcomp.design, "group", object())
        with pytest.raises(RuntimeError, match=re.escape("Bambi's group terms of the formula 'sigma ~ z + x1 + (1 + x1 |p| g) + (1 | h)' are")):
            d._dpar_group_blocks(d.spec.dpar("sigma"), dcomp)
    nog = ab.gaussian(["y ~ x1 + (1 | g)", "sigma ~ z"], DF, ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 1), sd={"g": HN},
                                                                         sigma={"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 1)}))
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(nog.bambi_model.components["sigma"].terms, "1|g", t1)
        with pytest.raises(RuntimeError, match=re.escape("Bambi built group-level terms for the formula 'sigma ~ z', which has none")):
            nog._dpar_design(nog.spec.dpar("sigma"))


def test_level_codes_and_helpers():
    assert gi.level_codes(np.array(["b", "a", "b"], dtype=object), ("a", "b")).tolist() == [1, 0, 1]
    assert gi.level_codes(pd.Series(["b"]), ("a", "b")).dtype == np.int64 and gi.level_codes([], ("a",)).shape == (0,)
    with pytest.raises(RuntimeError, match=re.escape("labels ['z'] are not among the fitted levels")):
        gi.level_codes(["a", "z"], ("a", "b"))
    gi.check_dense_size(10, 5, "x")
    frame = pd.DataFrame({"k": np.array([1, 2], dtype=np.int64)})
    assert gi.bambi_formula_value(frame, None).dtype == np.dtype(int) and gi.bambi_formula_value(frame, "k").dtype == np.int64


# -- the guard on Bambi's private layout -----------------------------------------------------------------------------------
def test_bambi_s_private_layout_is_the_one_this_design_relies_on(monkeypatch):
    problems = gi.bambi_layout_problems()
    assert not problems, ("the index-coded group terms (G11, docs/scaling.md, autobayes/group_index.py: IndexGroupTerm handed to "
                          "Bambi's backend in place of formulae's dense design) rely on Bambi 0.17.2's private layout, which "
                          f"has changed: {problems}; re-verify the design (tools/model_manifest.py) before updating the hashes")
    import bambi
    from bambi.config import config

    monkeypatch.setattr(bambi, "__version__", "0.18.0")
    monkeypatch.setitem(gi._BAMBI_SOURCES, "bambi.terms.group_specific.GroupSpecificTerm", "0" * 64)
    monkeypatch.setitem(gi._BAMBI_SOURCES, "bambi.no_such_module.Thing", "0" * 64)
    config["SPARSE_DOT"] = True
    try:
        problems = gi.bambi_layout_problems()
    finally:
        config["SPARSE_DOT"] = False
    assert problems[0] == "Bambi is 0.18.0, not 0.17.2" and "bambi.terms.group_specific.GroupSpecificTerm changed" in problems[1]
    assert problems[2].startswith("bambi.no_such_module.Thing: AttributeError") and "SPARSE_DOT" in problems[3]
    with pytest.raises(ImportError):
        gi.source_hash("no_such_package_at_all.Thing")


# -- memory: a size whose dense indicator could not be built ---------------------------------------------------------------
MEMORY_SCRIPT = r'''
import json, sys, time, tracemalloc
import numpy as np, pandas as pd
N, J, REPO = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
rng = np.random.default_rng(1)
codes = np.r_[np.arange(J), rng.integers(0, J, N - J)]
rng.shuffle(codes)
labels = np.array([f"g{j:05d}" for j in range(J)], dtype=object)[codes]
df = pd.DataFrame({"y": (rng.random(N) < 0.5).astype(int), "x1": rng.normal(size=N), "x2": rng.normal(size=N), "g": labels})
import autobayes as ab
assert ab.__file__.startswith(REPO), ab.__file__
import jax
from pymc.sampling.jax import get_jaxified_logp
from pymc.blocking import DictToArrayBijection
import arviz as az, xarray as xr
HN = ab.HalfNormal(1)
tracemalloc.start()
t0 = time.perf_counter()
m = ab.bernoulli("y ~ x1 + x2 + (1 + x1 | g)", df, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1),
                 sd={"g": {"Intercept": HN, "x1": HN}}, cor={"g": ab.LKJ(2)}))
out = {"build_s": time.perf_counter() - t0}
pmm = m.pymc_model
ip = pmm.initial_point()
fn = pmm.logp_dlogp_function(ravel_inputs=True)
fn.set_extra_values({})
lp, grad = fn(DictToArrayBijection.map({v.name: ip[v.name] for v in fn._grad_vars}).data)
jf = get_jaxified_logp(pmm, negative_logp=False)
vg = jax.jit(jax.value_and_grad(lambda xs: jf(xs)))
xs = [np.asarray(ip[v.name]) for v in pmm.value_vars]
val, _ = vg(xs)
hlo = vg.lower(xs).as_text()
out["logp"] = [float(lp), float(val)]
out["hlo_NxJ"] = sum(f"{N}x{J}" in line or f"{J}x{N}" in line for line in hlo.splitlines())
keep = [v for v in pmm.unobserved_RVs if "__obs__" not in (pmm.named_vars_to_dims.get(v.name) or ())]
f = pmm.compile_fn(pmm.replace_rvs_by_values(keep), inputs=pmm.value_vars, on_unused_input="ignore", point_fn=True)
draws = [f({k: np.asarray(v, dtype=float) + 0.3 * rng.normal(size=np.shape(v)) for k, v in ip.items()}) for _ in range(4)]
data = {v.name: (("chain", "draw", *(pmm.named_vars_to_dims.get(v.name) or ())), np.stack([d[i] for d in draws])[None])
        for i, v in enumerate(keep)}
coords = {k: list(c) for k, c in pmm.coords.items() if k != "__obs__"}
idata = az.InferenceData(posterior=xr.Dataset(data, coords={**coords, "chain": [0], "draw": np.arange(4)}))
lin = ab.predict(m, idata, kind="linear")["eta"]
ll = ab.log_lik(m, idata)
nd = df.iloc[:2000].assign(g=np.where(np.arange(2000) % 3 == 0, "new_level", df.g.iloc[:2000]))
new = ab.predict(m, idata, newdata=nd, kind="linear", new_groups="population", seed=1)["eta"]
out["shapes"] = [list(lin.shape), list(ll.shape), list(new.shape)]
raised = []
for what in (lambda: m.design.Z, lambda: m.design.group_blocks[("g", "x1")].Z):
    try:
        what()
    except ValueError as e:
        raised.append("DENSE_Z_MAX_ENTRIES" in str(e))
out["Z_refused"] = raised
out["peak_GB"] = tracemalloc.get_traced_memory()[1] / 2**30
out["total_s"] = time.perf_counter() - t0
print("RESULT " + json.dumps(out))
'''


def run_memory_test(N: int, J: int) -> dict:
    res = subprocess.run([sys.executable, "-c", MEMORY_SCRIPT, str(N), str(J), str(REPO)], cwd=REPO, capture_output=True, text=True,
                         timeout=1800)
    line = next((ln for ln in res.stdout.splitlines() if ln.startswith("RESULT ")), None)
    assert res.returncode == 0 and line is not None, res.stderr[-3000:]
    out = json.loads(line[7:])
    assert abs(out["logp"][0]) == pytest.approx(abs(out["logp"][1]), rel=1e-10), out   # PyMC's jaxified logp is the negative
    assert out["hlo_NxJ"] == 0 and out["Z_refused"] == [True, True], out
    assert out["shapes"] == [[1, 4, N], [1, 4, N], [1, 4, 2000]], out
    assert out["peak_GB"] < 0.5, out
    return out


def test_memory_two_hundred_thousand_rows_two_thousand_levels():
    """N = 2e5, J = 2e3: formulae's dense indicator alone was 3.2 GB; build, a log density and gradient on PyTensor and JAX,
    predict(kind="linear"), log_lik and new rows with a new level stay under 0.5 GB of traced Python memory."""
    run_memory_test(200_000, 2_000)


@pytest.mark.slow
def test_memory_two_hundred_thousand_rows_twenty_thousand_levels():
    """N = 2e5, J = 2e4 (formulae's indicator alone was 32 GB): as above. Slow: the target identity hashes the v1 bytes of two
    dense 2e5 x 2e4 blocks, about 190 GB at about 2 GB/s (docs/scaling.md)."""
    run_memory_test(200_000, 20_000)
