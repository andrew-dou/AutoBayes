"""M3b-1 marginal-reference protocol (docs/m3b1-decision.md): the low-rank evaluation of the analytically
integrated Gaussian varying-slope marginal equals a dense multivariate-normal evaluation, on a small crossed
fixture over many random scale points and on the frozen F7 data at fixed points; the log-scale target carries
the HalfNormal priors and Jacobians. Deterministic; no fits."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tests.reference_marginal import GaussianSlopeMarginal, f7_data


def crossed_fixture():
    rng = np.random.default_rng(3)
    groups = np.array(["b", "c", "a", "a", "c", "b", "b", "a", "c", "c", "a", "b", "a", "c", "b"])   # interleaved, unequal sizes
    x = rng.normal(size=len(groups)) + 0.4
    y = rng.normal(size=len(groups))
    return y, x, groups


@pytest.mark.parametrize("center", [True, False])
def test_low_rank_equals_dense_on_crossed_fixture(center):
    y, x, groups = crossed_fixture()
    ref = GaussianSlopeMarginal(y, x, groups, prior_mean=(0.3, -0.2), prior_sd=(2.5, 1.5), center=center)
    assert ref.J == 3 and ref.levels == ["a", "b", "c"] and ref.n_j.tolist() == [5.0, 5.0, 5.0]
    rng = np.random.default_rng(11)
    pts = np.exp(rng.uniform(-3.5, 1.0, size=(200, 3)))                       # scales over four decades
    pts = np.vstack([pts, [[1e-6, 1e-6, 0.7], [2.0, 1e-7, 0.3], [1e-7, 1.5, 0.05], [0.5, 0.3, 0.4]]])   # near-degenerate blocks
    lr = ref.log_marginal(pts[:, 0], pts[:, 1], pts[:, 2])
    dense = np.array([ref.dense_log_marginal(*p) for p in pts])
    assert np.allclose(lr, dense, rtol=1e-10, atol=1e-8), np.abs(lr - dense).max()


def test_low_rank_equals_dense_on_frozen_f7_data():
    y, x, g = f7_data()
    assert len(y) == 600 and len(set(g)) == 30 and abs(y.mean() - 1.0) < 0.5
    ref = GaussianSlopeMarginal(y, x, g, prior_mean=(0.0, 0.0), prior_sd=(5.0, 2.0), center=True)
    pts = np.array([[0.5, 0.3, 0.4], [0.354, 0.175, 0.424], [0.05, 0.05, 1.0], [1.5, 0.8, 0.2], [0.354, 0.30, 0.424]])
    lr = ref.log_marginal(pts[:, 0], pts[:, 1], pts[:, 2])
    dense = np.array([ref.dense_log_marginal(*p) for p in pts])              # 600 x 600 dense covariance, Cholesky
    assert np.allclose(lr, dense, rtol=1e-10, atol=1e-6), np.abs(lr - dense).max()
    # the log-scale target: marginal + HalfNormal(1) priors + Jacobians, checked against the explicit formula
    theta = np.log(pts)
    lp = ref.log_posterior_theta(theta)
    expected = dense + sum(stats.halfnorm.logpdf(pts[:, k], scale=1.0) + theta[:, k] for k in range(3))
    assert np.allclose(lp, expected, rtol=1e-10, atol=1e-6)


def test_marginal_equals_explicit_integration_on_one_tiny_group():
    """One group, two rows, all effects present: compare with the covariance written out by hand."""
    y = np.array([0.3, -0.4]); x = np.array([1.0, -2.0]); g = np.array(["k", "k"])
    ref = GaussianSlopeMarginal(y, x, g, prior_mean=(0.0, 0.0), prior_sd=(5.0, 2.0), center=True)
    sd0, sd1, sigma = 0.6, 0.25, 0.5
    xc = x - x.mean(); A = np.column_stack([np.ones(2), xc])
    cov = sigma ** 2 * np.eye(2) + A @ np.diag([25.0, 4.0]) @ A.T + sd0 ** 2 * np.ones((2, 2)) + sd1 ** 2 * np.outer(x, x)
    hand = stats.multivariate_normal(mean=np.zeros(2), cov=cov).logpdf(y)
    assert abs(float(ref.log_marginal(sd0, sd1, sigma)[0]) - hand) < 1e-10
