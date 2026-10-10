"""``s(x)``: the port of mgcv's thin-plate regression spline (autobayes/smooths.py) against mgcv's own output
(brms-parity B04, phase 1: the basis alone, not yet in the formula or the model).

mgcv's matrices for seeded one-dimensional data sets are frozen in ``tests/oracle/smooth_bases/<id>.json`` by
``oracle/R/gen_smooth_basis.R`` (additions only); nothing here needs R. On the ``standard`` data sets the port is held
to the parity tolerances: brms's ``Xs`` exactly (1e-12), ``Zs`` up to an orthogonal ``R`` estimated by least squares
(``R'R = I`` to 1e-10, ``Zs_port R = Zs_mgcv`` to 1e-12, and ``R`` rounds to a signed permutation that maps to 1e-10),
and the same on new values (``PredictMat`` / ``brms:::s2rPred``) and on the fitted rows (what ``standata(fit)``
returns). On the ``conditioning`` data sets (outlying values, ``k`` near the number of unique values) mgcv's own
output moves under exact symmetries of the construction (``x -> a x + b``) by more than those tolerances; there the
port is held to ten times that movement (``meta.mgcv_rounding``, measured by the generator). The ``gauss_spline``
oracle fixture's ``Xs`` / ``Zs_1_1`` (brms's own ``standata``) are checked directly. Nothing samples.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autobayes import smooths
from autobayes.smooths import IndeterminateBasisError, lanczos_start, mgcv_lanczos, smooth_basis, tprs_construct
from tests.oracle.fixture import decode

ROOT = Path(__file__).resolve().parent / "oracle"
BASES = ROOT / "smooth_bases"
FIXTURE = ROOT / "fixtures" / "gauss_spline"
STANDARD = ("exponential_n80_k12", "gauss_spline", "k3_n30", "k4_n30", "normal_n150", "normal_n50", "offset_mean_n40",
            "small_n12_k5", "symmetric_full_basis_n50_k10", "ties_gapped_integers_n70_k6", "ties_rounded_n60",
            "tiny_scale_n40", "uniform_n100_k8", "unique_equals_k_n20_k6")
CONDITIONING = ("k_near_unique_n20_k15", "outliers_n40", "unique_equals_k_n25_k10")

XS_ABS = 1e-12          # brms's Xs, exactly
R_ORTH = 1e-10          # |R'R - I| for the least-squares R between the Zs blocks
ZS_ABS = 1e-12          # |Zs_port R - Zs_mgcv| with that R
PERM_ABS = 1e-10        # |Zs_port P - Zs_mgcv| with P = round(R), an exactly orthogonal signed permutation
CONSTRUCT_REL = 1e-10   # intermediate stages (the raw tprs construction, smoothCon's pieces): diagnostics only
ROUNDING_FACTOR = 10.0  # conditioning sets: agreement within ten times mgcv's own movement


def load(case_id: str) -> dict:
    with open(BASES / f"{case_id}.json", encoding="utf-8") as f:
        return json.load(f)


def k_of(c: dict) -> int | None:
    return None if c["meta"]["k"] == -1 else c["meta"]["k"]


def rel_err(a, b) -> float:
    """Largest |a - b| / max(1, |a|, |b|), as tests/oracle/tolerances.py measures predictions."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape, (a.shape, b.shape)
    return float(np.max(np.abs(a - b) / np.maximum(1.0, np.maximum(np.abs(a), np.abs(b))))) if a.size else 0.0


def column_signs(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    s = np.sign(np.sum(A * B, axis=0))
    s[s == 0] = 1.0
    return s


def align(A: np.ndarray, B: np.ndarray) -> dict:
    """The least-squares ``R`` with ``A R ~ B``, how far it is from orthogonal, its residual, and the signed
    permutation ``P = round(R)`` with its (exactly orthogonal) residual."""
    R = np.linalg.lstsq(A, B, rcond=None)[0]
    P = np.round(R)
    is_perm = bool(np.all(np.abs(P).sum(axis=0) == 1) and np.all(np.abs(P).sum(axis=1) == 1))
    return {"R": R, "orth": float(np.max(np.abs(R.T @ R - np.eye(R.shape[1])))), "resid": float(np.max(np.abs(A @ R - B))),
            "P": P, "is_perm": is_perm, "perm_resid": float(np.max(np.abs(A @ P - B)))}


@pytest.fixture(scope="module")
def built():
    """Every frozen data set with the port's basis built on it."""
    out = {}
    for cid in STANDARD + CONDITIONING:
        c = load(cid)
        out[cid] = (c, smooth_basis(decode(c["x"]), k_of(c)))
    return out


def test_frozen_set_is_complete():
    on_disk = sorted(p.stem for p in BASES.glob("*.json"))
    assert on_disk == sorted(STANDARD + CONDITIONING)
    for cid in on_disk:
        meta = load(cid)["meta"]
        assert meta["id"] == cid and meta["generator"] == "oracle/R/gen_smooth_basis.R"
        assert meta["role"] == ("standard" if cid in STANDARD else "conditioning")
        assert (meta["mgcv"], meta["brms"]) == ("1.9.3", "2.23.0")


# -- the raw construction: smooth.construct.tp.smooth.spec --------------------------------------------------------
@pytest.mark.parametrize("cid", STANDARD + CONDITIONING)
def test_tprs_bookkeeping(cid, built):
    c, b = built[cid]
    tp, ref = b.tprs, c["tprs"]
    assert tp.k == ref["bs_dim"] and tp.rank == ref["rank"] and ref["null_space_dim"] == smooths.NULL_DIM
    assert tp.shift == decode(ref["shift"])[0]                 # R's mean(), bit for bit
    np.testing.assert_array_equal(tp.Xu, decode(ref["Xu"]))    # the knots: sorted unique centred values
    assert tp.UZ.shape == decode(ref["UZ"]).shape and tp.X.shape == decode(ref["X"]).shape


@pytest.mark.parametrize("cid", STANDARD)
def test_tprs_construction(cid, built):
    c, b = built[cid]
    tp, ref = b.tprs, c["tprs"]
    X = decode(ref["X"])
    s = column_signs(tp.X, X)                                  # eigenvector signs are LAPACK's business
    assert s[-2:].tolist() == [1.0, 1.0]                       # the null-space columns 1 and x are not flipped
    assert np.allclose(np.sqrt(np.mean(tp.X ** 2, axis=0)), 1.0, rtol=1e-14, atol=0)
    assert rel_err(tp.X * s, X) <= CONSTRUCT_REL
    assert rel_err(tp.UZ * s, decode(ref["UZ"])) <= CONSTRUCT_REL
    S = decode(ref["S"])
    assert np.max(np.abs(tp.S * np.outer(s, s) - S)) <= CONSTRUCT_REL * np.max(np.abs(S))


# -- smoothCon(absorb.cons = TRUE, diagonal.penalty = TRUE) ----------------------------------------------------------
@pytest.mark.parametrize("cid", STANDARD)
def test_smoothcon(cid, built):
    c, b = built[cid]
    ref = c["smooth"]
    s = column_signs(b.tprs.X, decode(c["tprs"]["X"]))
    assert ref["rank"] == b.rank and ref["null_space_dim"] == 1 and ref["df"] == b.k - 1
    assert abs(b.penalty_scale / decode(ref["S_scale"])[0] - 1.0) <= CONSTRUCT_REL
    # the LINPACK reflector of the constraint colMeans(X): qraux = u[0], qr = (-sign(C[0]) |C|, u[1:]); with the raw
    # columns flipped by s, mgcv's reflector is s[0] s[i] u[i]
    u = b.reflector
    assert abs(decode(ref["qraux"])[0] - u[0]) <= CONSTRUCT_REL
    qr = decode(ref["qr"])[:, 0]
    assert rel_err(qr[1:], s[0] * s[1:] * u[1:]) <= CONSTRUCT_REL
    if b.k == 3:                                               # the penalty is already diagonal: nat.param skipped
        assert b.reparam is None and ref["diagRP"] is None
    else:
        P_ref = decode(ref["diagRP"])
        P = s[1:, None] * b.reparam                            # absorbed coordinates follow the raw column signs
        assert rel_err(P * column_signs(P, P_ref), P_ref) <= CONSTRUCT_REL
        np.testing.assert_array_equal(decode(ref["S"]), np.diag(np.r_[np.ones(b.rank), 0.0]))


# -- brms's Xs and Zs ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("cid", STANDARD)
def test_xs_exact_and_zs_up_to_orthogonal(cid, built):
    c, b = built[cid]
    Xf, Zs = decode(c["random"]["Xf"]), decode(c["random"]["Zs"])
    assert b.Xs.shape == Xf.shape == (decode(c["x"]).size, 1) and b.Zs.shape == Zs.shape
    assert np.max(np.abs(b.Xs - Xf)) <= XS_ABS
    a = align(b.Zs, Zs)
    assert a["orth"] <= R_ORTH and a["resid"] <= ZS_ABS
    assert a["is_perm"] and a["perm_resid"] <= PERM_ABS


@pytest.mark.parametrize("cid", STANDARD)
def test_prediction_new_rows(cid, built):
    c, b = built[cid]
    ref = c["predict_new"]
    xnew = decode(c["xnew"])
    R = align(b.Zs, decode(c["random"]["Zs"]))["R"]           # R is a property of the two constructions
    Xs, Zs = b.predict(xnew)
    assert rel_err(Xs, decode(ref["Xf"])) <= XS_ABS
    assert rel_err(Zs @ R, decode(ref["Zs"])) <= ZS_ABS
    PM, PM_ref = b.predict_matrix(xnew), decode(ref["PredictMat"])
    assert rel_err(PM * column_signs(PM, PM_ref), PM_ref) <= ZS_ABS
    assert np.all(column_signs(PM, PM_ref)[b.rank:] == 1.0)  # the null column exactly, sign included


@pytest.mark.parametrize("cid", STANDARD)
def test_prediction_fitted_rows(cid, built):
    """What ``standata(fit)`` hands back for the fitted rows: PredictMat on the stored construction."""
    c, b = built[cid]
    ref = c["predict_fitted"]
    R = align(b.Zs, decode(c["random"]["Zs"]))["R"]
    Xs, Zs = b.predict(decode(c["x"]))
    assert np.max(np.abs(Xs - decode(ref["Xf"]))) <= XS_ABS
    assert np.max(np.abs(Zs @ R - decode(ref["Zs"]))) <= ZS_ABS


@pytest.mark.parametrize("cid", CONDITIONING)
def test_conditioning_sets_agree_to_mgcv_rounding(cid, built):
    """mgcv's own output moves by ``meta.mgcv_rounding`` under exact symmetries of the construction; the port agrees
    with it to within ten times that, through an exactly orthogonal signed permutation of the Zs columns."""
    c, b = built[cid]
    rnd = c["meta"]["mgcv_rounding"]
    tol_x, tol_z = ROUNDING_FACTOR * max(rnd["Xf"], XS_ABS / 10), ROUNDING_FACTOR * max(rnd["Zs"], XS_ABS / 10)
    assert b.tprs.ritz_ratio > 1e5                            # the diagnostic that flags these inputs
    assert np.max(np.abs(b.Xs - decode(c["random"]["Xf"]))) <= tol_x
    a = align(b.Zs, decode(c["random"]["Zs"]))
    assert a["is_perm"] and a["perm_resid"] <= tol_z
    Xs, Zs = b.predict(decode(c["xnew"]))
    assert rel_err(Xs, decode(c["predict_new"]["Xf"])) <= tol_x
    assert rel_err(Zs @ a["P"], decode(c["predict_new"]["Zs"])) <= tol_z
    Xs, Zs = b.predict(decode(c["x"]))
    assert np.max(np.abs(Xs - decode(c["predict_fitted"]["Xf"]))) <= tol_x
    assert np.max(np.abs(Zs @ a["P"] - decode(c["predict_fitted"]["Zs"]))) <= tol_z


@pytest.mark.parametrize("cid", STANDARD)
def test_standard_sets_are_well_conditioned(cid, built):
    assert built[cid][1].tprs.ritz_ratio < 1e5


# -- the gauss_spline oracle fixture (brms's standata) -----------------------------------------------------------------
def test_gauss_spline_fixture_standata():
    sd = json.loads((FIXTURE / "fixture.json").read_text(encoding="utf-8"))["standata"]
    x = pd.read_csv(FIXTURE / "data.csv")["x"].to_numpy()
    b = smooth_basis(x)
    Xs_f, Zs_f = decode(sd["Xs"]), decode(sd["Zs_1_1"])
    assert decode(sd["knots_1"]).tolist() == [b.rank] == [8] and decode(sd["Ks"]) == b.Xs.shape[1] == 1
    Xs, Zs = b.predict(x)                                      # standata(fit) evaluates the stored basis
    assert np.max(np.abs(Xs - Xs_f)) <= XS_ABS
    a = align(Zs, Zs_f)
    assert a["orth"] <= R_ORTH and a["resid"] <= ZS_ABS and a["is_perm"] and a["perm_resid"] <= PERM_ABS
    # the basis as brm() builds it for Stan (U D Z, Ritz values) differs from standata's (E U Z) by mgcv's Lanczos
    # residual; both are mgcv's, and the port reproduces each
    assert np.max(np.abs(b.Xs - Xs_f)) <= XS_ABS
    gap = align(b.Zs, Zs_f)["perm_resid"]
    assert 1e-12 < gap < 1e-9


# -- refusals and edge cases ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("x, k", [(np.repeat(np.arange(1.0, 16.0), 4), 12), (np.arange(1.0, 51.0), None),
                                  (np.linspace(-1.0, 1.0, 41), None), (np.arange(-5.0, 6.0)[np.arange(11) != 5] * 1.5, 8)])
def test_symmetric_knots_refused(x, k):
    """Symmetric knots make the first QT pivot zero in exact arithmetic: mgcv's basis then follows rounding noise.
    (With exactly k knots mgcv skips the eigen-decomposition and nothing is ambiguous: symmetric_full_basis_n50_k10.)"""
    with pytest.raises(IndeterminateBasisError, match="zero up to rounding"):
        smooth_basis(x, k)


def test_refusals():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="fewer unique covariate combinations"):
        smooth_basis(rng.normal(size=9))
    with pytest.raises(NotImplementedError, match="subsamples the knots"):
        smooth_basis(rng.normal(size=smooths.MAX_KNOTS + 1))
    with pytest.raises(ValueError, match="non-finite"):
        smooth_basis(np.r_[rng.normal(size=20), np.nan])
    with pytest.raises(ValueError, match="one-dimensional"):
        smooth_basis(rng.normal(size=(20, 2)))
    with pytest.raises(ValueError, match="numeric"):
        smooth_basis(np.array(list("abcdefghijkl")))
    with pytest.raises(ValueError, match="integer"):
        smooth_basis(rng.normal(size=20), 4.5)


def test_minimum_k_warns_and_rises_to_three():
    x = np.random.default_rng(4).normal(size=25)
    with pytest.warns(UserWarning, match="increased to minimum possible"):
        b = smooth_basis(x, 2)
    assert b.k == 3 and b.Zs.shape == (25, 1) and b.Xs.shape == (25, 1)


def test_ties_share_rows():
    x = np.round(np.random.default_rng(5).normal(size=50), 1)
    b = smooth_basis(x)
    _, first = np.unique(x, return_index=True)
    for v, i in zip(np.unique(x), first):
        rows = np.flatnonzero(x == v)
        assert np.array_equal(b.Zs[rows], np.broadcast_to(b.Zs[i], (rows.size, b.rank)))


# -- the Lanczos iteration ----------------------------------------------------------------------------------------------
def test_lanczos_start_vector():
    """``Rlanczos``'s start vector: jran = (106 jran + 1283) mod 6075 from 1, mapped to jran / 6075 - 0.5."""
    raw = np.array([1389 / 6075 - 0.5, 2717 / 6075 - 0.5])
    v = lanczos_start(2)
    np.testing.assert_allclose(v, raw / np.linalg.norm(raw), rtol=0, atol=1e-16)
    assert abs(np.linalg.norm(lanczos_start(37)) - 1.0) < 1e-15


def test_lanczos_runs_to_exact_eigenpairs():
    """Iterated to the end (n = 12), the Ritz pairs are eigenpairs, in mgcv's order: positive values downwards, then
    the negative ones with the most negative last."""
    rng = np.random.default_rng(6)
    A = rng.normal(size=(12, 12))
    A = A + A.T
    res = mgcv_lanczos(A, 5)
    w = np.linalg.eigvalsh(A)
    top = sorted(w, key=abs, reverse=True)[:5]
    expect = sorted([v for v in top if v > 0], reverse=True) + sorted([v for v in top if v < 0], reverse=True)
    np.testing.assert_allclose(res.values, expect, rtol=1e-12)
    assert np.max(np.abs(A @ res.vectors - res.vectors * res.values)) < 1e-10 * np.max(np.abs(w))
    np.testing.assert_allclose(res.vectors.T @ res.vectors, np.eye(5), atol=1e-12)


def test_tprs_shapes_and_null_space():
    x = np.random.default_rng(7).uniform(-1, 3, size=40)
    tp = tprs_construct(x)
    assert tp.X.shape == (40, 10) and tp.S.shape == (10, 10) and tp.UZ.shape == (42, 10)
    np.testing.assert_array_equal(tp.S[-2:], 0.0)
    np.testing.assert_allclose(tp.X[:, -1] * np.sqrt(np.mean((x - tp.shift) ** 2)), x - tp.shift, rtol=1e-13)
    np.testing.assert_allclose(smooths.tprs_predict(tp, x), tp.X, atol=1e-9)   # E U Z vs U D Z: Lanczos residual
