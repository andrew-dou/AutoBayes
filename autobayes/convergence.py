"""The convergence diagnostics brms's ``summary()`` reports (brms-parity F07, docs/summaries.md): ``Rhat``, ``Bulk_ESS`` and
``Tail_ESS``, the rank-normalised split R-hat and the bulk and tail effective sample sizes of Vehtari, Gelman, Simpson,
Carpenter and Buerkner (2021), "Rank-normalization, folding, and localization: an improved R-hat for assessing convergence of
MCMC", Bayesian Analysis 16(2), 667-718, which brms takes from the posterior package (1.6.1 here: ``rhat()``, ``ess_bulk()``,
``ess_tail()``). Written from the paper; what it leaves open was settled on posterior's outputs on probe draws
(``tests/summary_probes/posterior-1.6.1.json``), none of posterior's code used. Each function takes the draws of one quantity
as ``(chains, draws)`` and returns a float, NaN where posterior returns NA.

- Split chains: each chain's first ``floor(n / 2)`` and last ``floor(n / 2)`` draws are two chains; an odd chain's middle draw
  is left out (probed: dropping its first or last draw instead misses).
- Rank normalisation: ranks over all the split draws, ties averaged, ``z = Phi^-1((r - 3/8) / (S + 1/4))`` (Blom's offsets).
- R-hat of split chains (``M`` chains of ``N``): ``sqrt(var+ / W)``, ``W`` the mean of the chains' variances, ``var+ = (N - 1) / N W
  + B / N``, ``B / N`` the variance of the chain means; ``rhat`` is the larger of the bulk R-hat (of the rank-normalised split
  draws) and the tail R-hat (of the rank-normalised folded split draws, ``|theta - median(theta)|`` with the median of all the
  draws, an odd chain's middle draw included: probed).
- ESS of split chains: ``S / tau``, ``S = M N``, ``rho_t = 1 - (W - mean_m c_m(t)) / var+`` with ``c_m`` chain ``m``'s
  autocovariance (divisor ``N``) and ``rho_0 = 1``; ``tau = -1 + 2 sum_k P_k + rho_2K`` over Geyer's initial monotone sequence of
  the pair sums ``P_k = rho_2k + rho_2k+1``: from ``k = 1``, pairs are formed while the previous one is positive and ``k <=
  floor((N - 4) / 2)``; the pair ``K`` that stops it (the first that is not positive, or the last formed) is left out of the sum and
  its even lag added (``max(rho_2K, 0)`` when ``P_K < 0``: Geyer's truncation and the paper's correction for antithetic chains);
  each pair is at most the one before. ``tau`` is at least ``1 / log10(S)`` (the ESS at most ``S log10 S``). With fewer than six
  draws per split chain there are no pairs and ``tau = 2``; with fewer than three, or constant draws, no ESS: posterior's values
  on the probes (the limit ``N - 4`` is posterior's, one lag shorter than ArviZ 0.23's ``_ess``, which otherwise computes the same and
  whose bulk ESS on the frozen F07 cases is up to 8 % below brms's).
- ``ess_bulk``: the ESS of the rank-normalised split draws; ``ess_tail``: the smaller ESS of the indicators ``theta <= q`` at the
  5 % and 95 % quantiles (R's type 7, of all the draws), split; NaN when a draw is not finite or the draws are constant (their
  range below the double's epsilon, posterior's tolerance on the probes).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata

EPS = float(np.finfo(float).eps)


def _chains(draws) -> np.ndarray:
    x = np.asarray(draws, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"the draws of one quantity are (chains, draws); got the shape {x.shape}")
    return x


def split_chains(draws) -> np.ndarray:
    """``(2 chains, floor(draws / 2))``: each chain's first and last halves, an odd chain's middle draw left out."""
    x = _chains(draws)
    half = x.shape[1] // 2
    return np.concatenate([x[:, :half], x[:, x.shape[1] - half:]], axis=0)


def z_scale(x) -> np.ndarray:
    """The rank-normalised draws: ``Phi^-1((r - 3/8) / (S + 1/4))``, ``r`` the average rank of each among all ``S``."""
    x = np.asarray(x, dtype=float)
    ranks = rankdata(x, method="average").reshape(x.shape)
    return ndtri((ranks - 0.375) / (x.size + 0.25))


def _rhat_split(x: np.ndarray) -> float:
    """BDA3's R-hat of chains already split: ``sqrt(var+ / W)``."""
    M, N = x.shape
    W = float(np.mean(np.var(x, axis=1, ddof=1)))
    B_over_N = float(np.var(np.mean(x, axis=1), ddof=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return math.sqrt(np.float64((N - 1) / N * W + B_over_N) / W)


def rhat(draws) -> float:
    """posterior's ``rhat()``: the larger of the bulk and the tail rank-normalised split R-hat (module docstring); NaN for a
    draw that is NaN, constant draws and chains of fewer than four draws."""
    x = _chains(draws)
    s = split_chains(x)
    if np.isnan(x).any() or s.shape[1] < 2:
        return math.nan
    bulk = _rhat_split(z_scale(s))
    tail = _rhat_split(z_scale(np.abs(s - np.median(x))))
    return math.nan if math.isnan(bulk) or math.isnan(tail) else max(bulk, tail)


def _autocovariance(x: np.ndarray) -> np.ndarray:
    """Each chain's autocovariance at every lag, divisor ``N`` (by FFT, zero-padded past ``2 N``)."""
    N = x.shape[1]
    centred = x - x.mean(axis=1, keepdims=True)
    size = 1 << (2 * N - 1).bit_length()
    f = np.fft.rfft(centred, n=size, axis=1)
    return np.fft.irfft(f * np.conjugate(f), n=size, axis=1)[:, :N] / N


def ess_split(x) -> float:
    """The ESS of chains already split (``(chains, draws)``), from their autocorrelations (module docstring); NaN for constant
    draws or fewer than three draws per chain."""
    x = np.asarray(x, dtype=float)
    M, N = x.shape
    S = M * N
    if N < 3 or np.ptp(x) < EPS:
        return math.nan
    if N < 6:                                             # no pair of lags is formed (posterior's value on the probes)
        return S / 2.0
    acov = _autocovariance(x)
    W = float(np.mean(acov[:, 0])) * N / (N - 1)
    var_plus = W * (N - 1) / N + float(np.var(np.mean(x, axis=1), ddof=1))
    rho = 1.0 - (W - acov.mean(axis=0)) / var_plus
    rho[0] = 1.0
    pairs = [rho[0] + rho[1]]
    K = (N - 4) // 2
    for k in range(1, (N - 4) // 2 + 1):
        if not pairs[-1] > 0:
            K = k - 1
            break
        pairs.append(rho[2 * k] + rho[2 * k + 1])
    stop = pairs.pop(K)                                   # the pair that stops the sequence is not summed
    extra = rho[2 * K] if stop >= 0 else max(rho[2 * K], 0.0)
    tau = -1.0 + 2.0 * float(np.sum(np.minimum.accumulate(pairs))) + extra
    return S / max(tau, 1.0 / math.log10(S))


def ess_bulk(draws) -> float:
    """posterior's ``ess_bulk()``: the ESS of the rank-normalised split draws; NaN for a draw that is NaN."""
    x = _chains(draws)
    if np.isnan(x).any():
        return math.nan
    return ess_split(z_scale(split_chains(x)))


def ess_tail(draws) -> float:
    """posterior's ``ess_tail()``: the smaller ESS of the indicators of the draws at or below their 5 % and 95 % quantiles
    (type 7), split; NaN for a draw that is not finite, constant draws, or an indicator without an ESS."""
    x = _chains(draws)
    if not np.isfinite(x).all() or np.ptp(x) < EPS:
        return math.nan
    out = [ess_split(split_chains((x <= q).astype(float))) for q in np.quantile(x, [0.05, 0.95])]
    return math.nan if any(math.isnan(v) for v in out) else min(out)
