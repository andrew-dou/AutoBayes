"""The kernel density at a point behind brms's Savage-Dickey evidence ratio (brms-parity E05, docs/post-processing.md):
brms's ``density_ratio(x, y, point)``, the density of draws ``x`` at ``point`` over that of draws ``y``, each a Gaussian
kernel density estimate with R's default bandwidth on R's binned grid, read at the point by a cubic spline.

Written from R's documentation of ``density()`` and ``bw.nrd0()`` (the estimate on a regular grid by the fast Fourier
transform after linear binning: Silverman 1982, "Algorithm AS 176", Applied Statistics 31(1); Jones and Lotwick 1984,
"Remark AS R50", Applied Statistics 33(1)), of ``spline()`` (the Forsythe, Malcolm and Moler cubic spline: Forsythe,
Malcolm and Moler 1977, "Computer Methods for Mathematical Computations", section 4.4) and from brms 2.23's and R 4.5's
outputs on probe inputs (``tests/hypothesis_probes/brms-2.23.0-density-ratio.json``), which settle what the documentation
leaves open; none of brms's or R's source is used. Settled on the probes:

- the bandwidth is ``bw.nrd0``: ``0.9 min(sd, IQR / 1.34) S^(-1/5)`` (the SD with divisor ``S - 1``, the quartiles R's type
  7, ``(1 - h) x_j + h x_(j+1)``), where ``min(...)`` is 0 replaced by the SD, else by ``|x_1|``, else by 1;
- the estimate is evaluated on 4096 points from the smallest to the largest draw, the range moved to ``point - sd / 4`` (or
  ``point + sd / 4``) when the point lies below (above) the draws; R bins the draws linearly on 4096 cells from 4
  bandwidths below that range to 4 above it, convolves the bins with the Gaussian kernel at the cells' own spacing
  (R 4.4's grid), clips at 0 and interpolates linearly onto the 4096 points;
- the density at the point is the FMM cubic spline through those 4096 values (not-a-knot end conditions differ next to the
  ends; linear interpolation by about 1e-8).

Agreement with brms is to round-off of the transform: ``1e-13`` relative plus about ``1e-16`` of the estimate's peak
absolute. Far in the draws' tails, where the estimate falls to that round-off (below about ``1e-12`` of its peak), brms's
value and this one are both round-off and differ; ``density_ratio`` warns there.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.linalg import solve_banded

DENSITY_POINTS = 4096          # brms's density_ratio(n = 4096), the n of every evaluation here
_EXTENSION = 4.0               # R bins on a range 4 bandwidths wider than the evaluation range on each side
TAIL_WARNING = 1e-10           # a density at the point below this share of the estimate's peak is round-off territory


def _draws(x, what: str) -> np.ndarray:
    v = np.asarray(x, dtype=float).ravel()
    if v.size < 2:
        raise ValueError(f"the density of {what} needs at least two draws (R's bandwidth rule does), got {v.size}")
    if not np.isfinite(v).all():
        raise ValueError(f"the {what} must be finite: {int((~np.isfinite(v)).sum())} of {v.size} are not")
    return v


def _quantile7(sorted_v: np.ndarray, p: float) -> float:
    """R's type 7 quantile, formed as R's outputs show (the probes' 1e6-scale draws settle it): ``(1 - h) x_j + h x_(j+1)``
    with ``j = floor((n - 1) p)`` and ``h`` the fraction left over."""
    pos = (sorted_v.size - 1) * p
    j = int(math.floor(pos))
    h = pos - j
    if h == 0.0:
        return float(sorted_v[j])
    return float((1.0 - h) * sorted_v[j] + h * sorted_v[j + 1])


def bw_nrd0(x) -> float:
    """R's ``bw.nrd0(x)``: Silverman's rule of thumb, ``0.9 min(sd, IQR / 1.34) n^(-1/5)``, with the documented guarantee of
    a positive result where the quartiles coincide (the minimum replaced by the SD, else by ``|x_1|``, else by 1)."""
    v = _draws(x, "draws")
    sd = float(np.std(v, ddof=1))
    ordered = np.sort(v)
    q1, q3 = _quantile7(ordered, 0.25), _quantile7(ordered, 0.75)
    lo = min(sd, (q3 - q1) / 1.34)
    if not lo > 0:
        lo = sd if sd > 0 else (abs(float(v[0])) if v[0] != 0 else 1.0)
    return 0.9 * lo * v.size ** -0.2


def _binned_density(v: np.ndarray, bw: float, lower: float, upper: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """R's ``density(v, bw, from = lower, to = upper, n = n)`` for ``n`` a power of two at least 512: the draws binned
    linearly on ``n`` cells spanning ``[lower - 4 bw, upper + 4 bw]``, convolved (circularly, on twice as many cells, so
    nothing wraps) with the Gaussian kernel at the cells' spacing, clipped at 0 and interpolated linearly onto ``n`` evenly
    spaced points from ``lower`` to ``upper``. Returns those points and the density there."""
    lo, up = lower - _EXTENSION * bw, upper + _EXTENSION * bw
    delta = (up - lo) / (n - 1)
    pos = (v - lo) / delta
    cell = np.floor(pos).astype(np.int64)
    frac = pos - cell
    mass = np.zeros(2 * n)
    weight = 1.0 / v.size
    inside = (cell >= 0) & (cell <= n - 2)
    np.add.at(mass, cell[inside], weight * (1.0 - frac[inside]))
    np.add.at(mass, cell[inside] + 1, weight * frac[inside])
    below, top = cell == -1, cell == n - 1        # a draw within one cell of either end keeps its share on the grid
    np.add.at(mass, np.zeros(int(below.sum()), dtype=np.int64), weight * frac[below])
    np.add.at(mass, np.full(int(top.sum()), n - 1, dtype=np.int64), weight * (1.0 - frac[top]))
    offsets = np.arange(2 * n) * delta
    offsets[n + 1:] = -offsets[n - 1:0:-1]          # the kernel at circular distances 0, delta, ..., n delta, -(n-1) delta, ...
    kernel = np.exp(-0.5 * (offsets / bw) ** 2) / (bw * math.sqrt(2.0 * math.pi))
    smooth = np.fft.ifft(np.fft.fft(mass) * np.conj(np.fft.fft(kernel))).real[:n]
    smooth = np.maximum(smooth, 0.0)
    grid = np.linspace(lo, up, n)
    points = np.linspace(lower, upper, n)
    return points, np.interp(points, grid, smooth)


def _fmm_spline_at(xk: np.ndarray, yk: np.ndarray, x0: float) -> float:
    """R's ``spline(xk, yk, xout = x0)$y`` with its default method ``"fmm"``: the cubic spline through the knots whose third
    derivative at each end is that of the cubic through the four knots there (Forsythe, Malcolm and Moler 1977)."""
    n = xk.size
    h = np.diff(xk)
    s = np.diff(yk) / h

    def third(xs, ys):                               # the third divided difference of four points
        d1 = np.diff(ys) / np.diff(xs)
        d2 = np.diff(d1) / (xs[2:] - xs[:-2])
        return (d2[1] - d2[0]) / (xs[3] - xs[0])

    # second derivatives M: h[i-1] M[i-1] + 2 (h[i-1] + h[i]) M[i] + h[i] M[i+1] = 6 (s[i] - s[i-1]) inside, and at the
    # ends (M[1] - M[0]) / h[0] = 6 f[x0..x3], (M[n-1] - M[n-2]) / h[n-2] = 6 f[x_{n-4}..x_{n-1}]
    ab = np.zeros((3, n))
    rhs = np.zeros(n)
    ab[1, 1:-1] = 2.0 * (h[:-1] + h[1:])
    ab[0, 2:] = h[1:]
    ab[2, :-2] = h[:-1]
    rhs[1:-1] = 6.0 * (s[1:] - s[:-1])
    ab[1, 0], ab[0, 1], rhs[0] = -1.0, 1.0, 6.0 * h[0] * third(xk[:4], yk[:4])
    ab[2, n - 2], ab[1, n - 1], rhs[n - 1] = -1.0, 1.0, 6.0 * h[-1] * third(xk[-4:], yk[-4:])
    m = solve_banded((1, 1), ab, rhs)
    i = int(np.clip(np.searchsorted(xk, x0, side="right") - 1, 0, n - 2))
    t, hi = x0 - xk[i], h[i]
    slope = s[i] - hi * (2.0 * m[i] + m[i + 1]) / 6.0
    return float(yk[i] + t * (slope + t * (m[i] / 2.0 + t * (m[i + 1] - m[i]) / (6.0 * hi))))


def density_at(x, point: float, *, what: str = "draws") -> tuple[float, float]:
    """The kernel density of the draws ``x`` at ``point`` as brms's ``density_ratio(x, point = point)`` computes it (module
    docstring), and the peak of that estimate on its grid (the scale of its round-off)."""
    v = _draws(x, what)
    point = float(point)
    if not math.isfinite(point):
        raise ValueError(f"the point must be finite, got {point}")
    bw = bw_nrd0(v)
    lower, upper = float(v.min()), float(v.max())
    if point < lower:
        lower = point - float(np.std(v, ddof=1)) / 4.0
    elif point > upper:
        upper = point + float(np.std(v, ddof=1)) / 4.0
    points, dens = _binned_density(v, bw, lower, upper, DENSITY_POINTS)
    peak = float(dens.max())
    if points[0] == points[-1]:                     # every draw at the point: R's spline collapses the tied knots to one
        return float(dens.mean()), peak
    return _fmm_spline_at(points, dens, point), peak


def density_ratio(x, y=None, point: float = 0.0, *, warn: bool = True) -> float:
    """brms's ``density_ratio(x, y, point)``: the density of the draws ``x`` at ``point`` over the density of the draws ``y``
    there (only the former without ``y``), each estimated as ``density_at`` does. With ``warn``, a density below ``1e-10`` of
    its estimate's peak (the point far in the draws' tail, where the estimate is round-off, brms's too) is warned about."""
    dx, peak_x = density_at(x, point, what="draws of x")
    if warn and dx < TAIL_WARNING * peak_x:
        warnings.warn(f"the density of x at {point:g} is {dx:.3g}, below {TAIL_WARNING:g} of its peak: the point is far in the "
                      "draws' tail, where the kernel estimate is round-off (brms's too); the value is not reliable", stacklevel=2)
    if y is None:
        return dx
    dy, peak_y = density_at(y, point, what="draws of y")
    if warn and dy < TAIL_WARNING * peak_y:
        warnings.warn(f"the density of y at {point:g} is {dy:.3g}, below {TAIL_WARNING:g} of its peak: the point is far in the "
                      "draws' tail, where the kernel estimate is round-off (brms's too); the ratio is not reliable", stacklevel=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.float64(dx) / np.float64(dy))
