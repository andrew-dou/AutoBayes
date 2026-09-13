"""An independent deterministic reference for the FULL Gaussian varying-slope posterior of the scales.

Written from the mathematics; imports nothing from ``autobayes`` and nothing from PyMC or Bambi. For the
model  y_i = a_c + b (x_i - xbar) + u[g_i] + x_i v[g_i] + eps_i  with  beta = (a_c, b) ~ N(m, V),
u_j ~ N(0, sd0^2), v_j ~ N(0, sd1^2), eps ~ N(0, sigma^2), the Gaussian coefficients and both
group-effect blocks integrate analytically:

    y ~ Normal(A m,  sigma^2 I + A V A^T + sd0^2 Z Z^T + sd1^2 Zx Zx^T),
    A = [1, x - xbar],  Z = one-hot groups,  Zx = x (raw) times Z,

leaving the three positive scales (sd0, sd1, sigma) with their HalfNormal priors. On the log scales
theta = (log sd0, log sd1, log sigma) the target carries the three Jacobians e^theta_k.

The covariance is evaluated with group-block, low-rank algebra (each group block is sigma^2 I plus a
rank-2 update; the fixed-effect term is a global rank-2 update, Woodbury and the matrix determinant
lemma), and can be cross-checked against a dense multivariate-normal evaluation (``dense_log_marginal``).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

LOG_2PI = np.log(2.0 * np.pi)


class GaussianSlopeMarginal:
    """Precomputes the data-only quantities from the ORIGINAL rows (y, x, group labels)."""

    def __init__(self, y, x, groups, *, prior_mean=(0.0, 0.0), prior_sd=(5.0, 2.0), center=True):
        y = np.asarray(y, dtype=float); x = np.asarray(x, dtype=float); groups = np.asarray(groups).astype(str)
        self.n = len(y); self.y = y; self.x = x
        self.levels = sorted(set(groups.tolist()))
        self.gidx = np.array([self.levels.index(g) for g in groups])
        self.J = len(self.levels)
        self.xbar = x.mean() if center else 0.0
        self.A = np.column_stack([np.ones(self.n), x - self.xbar])              # fixed-effect design (centred, as Bambi)
        self.m = np.asarray(prior_mean, dtype=float)
        self.V = np.diag(np.asarray(prior_sd, dtype=float) ** 2)
        self.r = y - self.A @ self.m                                            # residual about the prior mean
        # per-group data summaries with U_j = [1, x_j] (raw x: the group slope multiplies raw x)
        self.n_j = np.zeros(self.J); self.rr = np.zeros(self.J)
        self.s = np.zeros((self.J, 2)); self.t = np.zeros((self.J, 2, 2)); self.G = np.zeros((self.J, 2, 2))
        self.Ar = np.zeros((self.J, 2)); self.AA = np.zeros((self.J, 2, 2))
        for j in range(self.J):
            mask = self.gidx == j
            U = np.column_stack([np.ones(mask.sum()), x[mask]]); Aj = self.A[mask]; rj = self.r[mask]
            self.n_j[j] = mask.sum(); self.rr[j] = rj @ rj
            self.s[j] = U.T @ rj; self.t[j] = U.T @ Aj; self.G[j] = U.T @ U
            self.Ar[j] = Aj.T @ rj; self.AA[j] = Aj.T @ Aj

    # -- low-rank evaluation, vectorised over P parameter points --------------------------------------
    def log_marginal(self, sd0, sd1, sigma):
        """log N(y; A m, Sigma(sd0, sd1, sigma)) for arrays of scales (shape (P,))."""
        sd0 = np.atleast_1d(np.asarray(sd0, dtype=float)); sd1 = np.atleast_1d(np.asarray(sd1, dtype=float)); sigma = np.atleast_1d(np.asarray(sigma, dtype=float))
        P = len(sd0); s2 = sigma ** 2
        D = np.zeros((P, 1, 2, 2)); D[:, 0, 0, 0] = sd0 ** 2; D[:, 0, 1, 1] = sd1 ** 2
        G = self.G[None]                                                        # (1, J, 2, 2)
        C = np.eye(2)[None, None] + G @ D / s2[:, None, None, None]             # I + sigma^-2 G D, (P, J, 2, 2)
        det_C = C[..., 0, 0] * C[..., 1, 1] - C[..., 0, 1] * C[..., 1, 0]
        inv_C = np.empty_like(C)
        inv_C[..., 0, 0] = C[..., 1, 1]; inv_C[..., 1, 1] = C[..., 0, 0]; inv_C[..., 0, 1] = -C[..., 0, 1]; inv_C[..., 1, 0] = -C[..., 1, 0]
        inv_C /= det_C[..., None, None]
        W = D @ inv_C                                                           # (D^-1 + sigma^-2 G)^-1 without forming D^-1
        s = self.s[None]; t = self.t[None]                                      # (1, J, 2), (1, J, 2, 2)
        Ws = np.einsum("pjab,pjb->pja", W, np.broadcast_to(s, (P,) + s.shape[1:]))
        quad = self.rr[None] / s2[:, None] - np.einsum("pja,pja->pj", np.broadcast_to(s, Ws.shape), Ws) / s2[:, None] ** 2
        Wt = np.einsum("pjab,pjbc->pjac", W, np.broadcast_to(t, (P,) + t.shape[1:]))                # (P, J, 2, 2)
        b = self.Ar[None] / s2[:, None, None] - np.einsum("pjba,pjb->pja", np.broadcast_to(t, Wt.shape), Ws) / s2[:, None, None] ** 2
        M = self.AA[None] / s2[:, None, None, None] - np.einsum("pjba,pjbc->pjac", np.broadcast_to(t, Wt.shape), Wt) / s2[:, None, None, None] ** 2
        logdet_g = (self.n_j[None] * np.log(s2)[:, None] + np.log(det_C)).sum(1)
        q_g, w, Msum = quad.sum(1), b.sum(1), M.sum(1)                          # (P,), (P, 2), (P, 2, 2)
        # global rank-2 update: Sigma = Sigma_g + A V A^T
        VM = self.V[None] @ Msum
        I_VM = np.eye(2)[None] + VM
        det_IVM = I_VM[:, 0, 0] * I_VM[:, 1, 1] - I_VM[:, 0, 1] * I_VM[:, 1, 0]
        inv_IVM = np.empty_like(I_VM)
        inv_IVM[:, 0, 0] = I_VM[:, 1, 1]; inv_IVM[:, 1, 1] = I_VM[:, 0, 0]; inv_IVM[:, 0, 1] = -I_VM[:, 0, 1]; inv_IVM[:, 1, 0] = -I_VM[:, 1, 0]
        inv_IVM /= det_IVM[:, None, None]
        Kinv = inv_IVM @ self.V[None]                                           # (V^-1 + M)^-1 = (I + V M)^-1 V
        quad_total = q_g - np.einsum("pa,pab,pb->p", w, Kinv, w)
        logdet = logdet_g + np.log(det_IVM)
        return -0.5 * (self.n * LOG_2PI + logdet + quad_total)

    def dense_log_marginal(self, sd0, sd1, sigma):
        """The same quantity from the explicit n x n covariance (cross-check only)."""
        Z = np.eye(self.J)[self.gidx]; Zx = self.x[:, None] * Z
        cov = sigma ** 2 * np.eye(self.n) + self.A @ self.V @ self.A.T + sd0 ** 2 * Z @ Z.T + sd1 ** 2 * Zx @ Zx.T
        return float(stats.multivariate_normal(mean=self.A @ self.m, cov=cov).logpdf(self.y))

    def log_posterior_theta(self, theta, prior_scales=(1.0, 1.0, 1.0)):
        """Unnormalised log posterior density of theta = (log sd0, log sd1, log sigma): the marginal likelihood,
        the three HalfNormal(prior_scales) priors on the natural scales, and the three Jacobians e^theta."""
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        sd0, sd1, sigma = np.exp(theta[:, 0]), np.exp(theta[:, 1]), np.exp(theta[:, 2])
        lp = self.log_marginal(sd0, sd1, sigma)
        for k, (val, sc) in enumerate(((sd0, prior_scales[0]), (sd1, prior_scales[1]), (sigma, prior_scales[2]))):
            lp = lp + stats.halfnorm.logpdf(val, scale=sc) + theta[:, k]
        return lp


def f7_data(seed=17, J=30, per=20):
    """The frozen F7 generator (tests/test_sampling.py::test_varying_slope_recovery_both_backends), reproduced exactly."""
    rng = np.random.default_rng(seed)
    g = np.repeat([f"g{j:02d}" for j in range(J)], per)
    x = rng.normal(size=J * per)
    u = rng.normal(0, 0.5, J); v = rng.normal(0, 0.3, J)
    gi = np.repeat(np.arange(J), per)
    y = 1.0 + 0.5 * x + u[gi] + x * v[gi] + rng.normal(0, 0.4, J * per)
    return y, x, g
