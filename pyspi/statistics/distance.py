import functools
import os
import warnings
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import pairwise_distances, pairwise_kernels
import tslearn.metrics
from tslearn.barycenters import (
    euclidean_barycenter,
    dtw_barycenter_averaging,
    dtw_barycenter_averaging_subgradient,
    softdtw_barycenter,
)
from hyppo.independence import (
    MGC,
    Dcorr,
)

from pyspi.base import (
    Directed,
    Undirected,
    Unsigned,
    Signed,
    parse_bivariate,
    parse_multivariate,
)
from pyspi.utils import fmt_param, require_int, require_positive_float


# ---------------------------------------------------------------------------
# DTW Sakoe-Chiba auto-radius configuration (env-var tunable)
# ---------------------------------------------------------------------------
_DTW_SAKOE_LINEAR_FRAC = float(os.getenv("PYSPI_DTW_SAKOE_LINEAR_FRAC", "0.10"))
_DTW_SAKOE_SQRT_COEFF = float(os.getenv("PYSPI_DTW_SAKOE_SQRT_COEFF", "1.5"))
_DTW_SAKOE_MIN_RADIUS = int(os.getenv("PYSPI_DTW_SAKOE_MIN_RADIUS", "10"))


def _auto_sakoe_radius(length):
    if length <= 1:
        return 1
    linear = int(np.ceil(max(0.0, _DTW_SAKOE_LINEAR_FRAC) * length))
    sqrt_scaled = int(np.ceil(max(0.0, _DTW_SAKOE_SQRT_COEFF) * np.sqrt(length)))
    radius = min(linear, sqrt_scaled)
    radius = max(1, _DTW_SAKOE_MIN_RADIUS, radius)
    return min(radius, length - 1)


class _RetainedHalf:
    """Evaluate only the half of the pairs that `Undirected` keeps.

    `Undirected.multivariate` evaluates both orders of every pair and keeps
    bivariate(i, j) with i < j for both entries. For a statistic that holds no
    state between calls, evaluating only the kept half, in the order the
    inherited loop reaches it, gives the same matrix bit for bit at half the
    cost. List it before `Undirected` in the bases.
    """

    @parse_multivariate
    def multivariate(self, data):
        M = data.n_processes
        A = np.full((M, M), np.nan)
        for j in range(M):
            for i in range(j):
                A[i, j] = A[j, i] = self.bivariate(data, i=i, j=j)
        return A


class PairwiseDistance(Undirected, Unsigned):
    """``metric`` distance between each pair of processes, optionally / sqrt(T).

    ``normalise=True`` divides by ``sqrt(T)`` and adds ``_rmse`` to the
    identifier. For ``metric="euclidean"`` that is exactly the root mean square
    difference, ``sqrt(mean((x - y)**2))``, and the name is literal. For the
    other metrics -- cityblock, cosine, chebyshev, canberra, braycurtis -- the
    suffix is pyspi's house label for "divided by sqrt(T)" and nothing more.
    Those quantities are not root *mean* squares, and dividing them by sqrt(T)
    does **not** make them comparable across record lengths either -- only the
    Euclidean norm grows as sqrt(T) under i.i.d. differences, so only for
    `euclidean` is the division the right power. Read `_rmse` as the name of the
    normalisation that was applied, not as a claim about the metric.
    """

    name = "Pairwise distance"
    identifier = "pdist"
    labels = ["unsigned", "distance", "unordered", "nonlinear", "undirected"]

    def __init__(self, metric="euclidean", normalise=False, **kwargs):
        self._metric = metric
        self._normalise = normalise
        self.identifier += f"_{metric}"
        if normalise:
            self.identifier += "_rmse"

    @parse_multivariate
    def multivariate(self, data):
        Z = data.to_numpy(squeeze=True)
        D = pairwise_distances(Z, metric=self._metric)
        if self._normalise:
            T = Z.shape[1]
            D = D / np.sqrt(T)
        return D


""" TODO: include optional kernels in each method
"""


def _median_gaussian_distance(v):
    """``1 - K / max(K)`` for a Gaussian kernel ``K`` of median-heuristic width.

    hyppo's ``Hsic`` statistic is dcorr of these induced distances. Its
    ``compute_kern(x, y)`` takes the width from ``x`` alone and applies it to
    ``y`` as well, so ``Hsic().statistic(x, y) != Hsic().statistic(y, x)``.
    Gretton et al. (2005, 2008) give each variable the median distance between
    its own samples; this is hyppo's arithmetic with that width.
    """
    l2 = pairwise_distances(v, metric="l2")
    n = l2.shape[0]
    med = np.median(l2[~np.eye(n, dtype=bool)])
    # As hyppo: a zero median (mostly tied samples) falls back to width 1.
    med = med if med else 1
    K = pairwise_kernels(v, metric="rbf", gamma=1.0 / (2 * (med**2)))
    return 1 - K / np.max(K)


class HilbertSchmidtIndependenceCriterion(_RetainedHalf, Undirected, Unsigned):
    """Normalised Hilbert-Schmidt Independence Criterion (HSIC).

    hyppo's statistic: dcorr of the distances ``1 - K / max(K)`` induced by a
    Gaussian kernel, each variable with its own median-heuristic width (see
    `_median_gaussian_distance`). Like `DistanceCorrelation`, the unbiased
    variant is not square-rooted and the biased one is.
    """

    name = "Hilbert-Schmidt Independence Criterion"
    identifier = "hsic"
    labels = ["unsigned", "distance", "unordered", "nonlinear", "undirected"]

    def __init__(self, biased=False):
        self._biased = biased
        if biased:
            self.identifier += "_biased"

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        x, y = data.to_numpy()[[i, j]]
        dx = _median_gaussian_distance(x)
        dy = _median_gaussian_distance(y)
        return Dcorr(bias=self._biased, compute_distance=None).statistic(dx, dy)


def _row_dominance_counts(rx, ry):
    """``c[i, j] = #{k : rx[i, k] <= rx[i, j] and ry[i, k] <= ry[i, j]}``.

    ``rx`` and ``ry`` are row-wise max-ranks (1..n). Offline 2-D dominance
    counting with one Fenwick tree per row, all rows at once: because ranks
    are max-ranks, the ``k`` with ``rx[i, k] <= rx[i, j]`` are exactly the first
    ``rx[i, j]`` entries of row ``i`` in ``rx`` order, so ``j``'s query is
    answered once that many entries have been inserted. O(n log n) per row.
    """
    n_rows, n = rx.shape
    n_bits = n.bit_length()
    rows = np.arange(n_rows)
    ry_in_rx_order = np.take_along_axis(
        ry, np.argsort(rx, axis=1, kind="stable"), axis=1)
    # Queries grouped by the prefix length p = rx at which they are answered.
    q_order = np.argsort(rx, axis=None, kind="stable")
    bounds = np.searchsorted(rx.ravel()[q_order], np.arange(1, n + 2))
    q_rows = q_order // n
    q_vals = ry.ravel()[q_order]
    # Column n + 1 absorbs updates that run past the end of the tree.
    tree = np.zeros((n_rows, n + 2), dtype=np.int64)
    out = np.zeros(n_rows * n, dtype=np.int64)
    for p in range(1, n + 1):
        idx = ry_in_rx_order[:, p - 1].copy()
        for _ in range(n_bits):
            tree[rows, idx] += 1
            idx += idx & -idx
            np.minimum(idx, n + 1, out=idx)
        lo, hi = bounds[p - 1], bounds[p]
        if hi > lo:
            q_r = q_rows[lo:hi]
            idx = q_vals[lo:hi].copy()
            acc = np.zeros(hi - lo, dtype=np.int64)
            for _ in range(n_bits):
                acc += tree[q_r, idx]  # tree[:, 0] stays 0
                idx -= idx & -idx
            out[q_order[lo:hi]] = acc
    return out.reshape(n_rows, n)


_HHG_CHUNK_ENTRIES = 2**22


def _hhg_statistic(x, y):
    """hyppo's ``HHG().statistic(x, y)``, bit for bit, in O(n^2 log n).

    The sum over ``i != j`` of Pearson's chi-square for the 2x2 table that
    splits the other samples by ``d_x(i, .) <= d_x(i, j)`` and
    ``d_y(i, .) <= d_y(i, j)`` (Heller, Heller & Gorfine 2013). hyppo forms each
    table by scanning all ``k``, O(n^3); its counts are row-wise ranks and 2-D
    dominance counts, so they are computed that way here, as integers, then
    combined with hyppo's int64 arithmetic and summed as hyppo sums them.
    """
    distx = pairwise_distances(x, metric="euclidean")
    disty = pairwise_distances(y, metric="euclidean")
    n = distx.shape[0]
    S = np.zeros((n, n))
    # Rows are independent; bound the int64 work arrays' size.
    step = max(1, _HHG_CHUNK_ENTRIES // n)
    for start in range(0, n, step):
        stop = min(n, start + step)
        # #{k : d[i, k] <= d[i, j]}, which counts k = i and k = j themselves.
        na = rankdata(distx[start:stop], method="max", axis=1).astype(np.int64)
        nb = rankdata(disty[start:stop], method="max", axis=1).astype(np.int64)
        nab = _row_dominance_counts(na, nb)
        t11 = nab - 2
        t12 = na - nab
        t21 = nb - nab
        t22 = n - na - nb + nab
        denom = (t11 + t12) * (t21 + t22) * (t11 + t21) * (t12 + t22)
        num = (n - 2) * (t12 * t21 - t11 * t22) ** 2
        block = np.zeros((stop - start, n))
        ok = denom > 0
        block[ok] = num[ok] / denom[ok]
        S[start:stop] = block
    np.fill_diagonal(S, 0)
    mask = np.ones(S.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    return np.sum(S[mask])


class HellerHellerGorfine(_RetainedHalf, Undirected, Unsigned):
    """Heller-Heller-Gorfine independence criterion (see `_hhg_statistic`)."""

    name = "Heller-Heller-Gorfine Independence Criterion"
    identifier = "hhg"
    # Symmetric by construction: hhg(x, y) == hhg(y, x) exactly.
    labels = ["unsigned", "distance", "unordered", "nonlinear", "undirected"]

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        x, y = data.to_numpy()[[i, j]]
        return _hhg_statistic(x, y)


class DistanceCorrelation(_RetainedHalf, Undirected, Unsigned):
    """Distance correlation"""

    name = "Distance correlation"
    identifier = "dcorr"
    labels = ["unsigned", "distance", "unordered", "nonlinear", "undirected"]

    def __init__(self, biased=False):
        self._biased = biased
        if biased:
            self.identifier += "_biased"

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        """ """
        x, y = data.to_numpy()[[i, j]]
        stat = Dcorr(bias=self._biased).statistic(x, y)
        return stat


class MultiscaleGraphCorrelation(_RetainedHalf, Undirected, Unsigned):
    """Multiscale graph correlation"""

    name = "Multiscale graph correlation"
    identifier = "mgc"
    labels = ["distance", "unsigned", "unordered", "nonlinear", "undirected"]

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        x, y = data.to_numpy()[[i, j]]
        stat = MGC().statistic(x, y)
        return stat


def _lagged_sum(test, x, y, max_lag):
    """``sum_{l=0}^{L} (n - l)/n * S(x[l:], y[:n - l])`` (Mehta et al. 2020).

    The cross-distance statistics of hyppo's ``DcorrX`` and ``MGCX``, as their
    docstrings define them: the present of ``x`` against the past of ``y``,
    i.e. the dependence of ``x`` on the history of ``y``. hyppo 0.5.2 computes
    the ``l >= 1`` terms by passing the sliced distance matrices to an ``S``
    built to compute Euclidean distances, so every row of a distance matrix is
    treated as a point in ``R^(n-l)`` and only the ``l = 0`` term is right.
    Here ``S`` is given the distance matrices as such: the distance matrix of
    ``x[l:]`` is the block ``distx[l:, l:]`` of that of ``x``.
    """
    distx = pairwise_distances(x, metric="euclidean")
    disty = pairwise_distances(y, metric="euclidean")
    n = distx.shape[0]
    dep_lag = [test.statistic(distx, disty)]
    for lag in range(1, max_lag + 1):
        sx = np.ascontiguousarray(distx[lag:, lag:])
        sy = np.ascontiguousarray(disty[: n - lag, : n - lag])
        dep_lag.append((n - lag) * test.statistic(sx, sy) / n)
    return np.sum(dep_lag)


class CrossDistanceCorrelation(Directed, Unsigned):
    """Cross-distance correlation (DcorrX; Mehta et al. 2020).

    ``A[i, j] = sum_{l=0}^{max_lag} (T - l)/T * dcorr(z_j[l:], z_i[:T - l])``,
    the unbiased distance correlation of target ``j`` with the lagged source
    ``i``, summed over lags (see `_lagged_sum`).
    """

    name = "Cross-distance correlation"
    identifier = "dcorrx"
    labels = ["distance", "unsigned", "temporal", "directed", "nonlinear"]

    def __init__(self, max_lag=1):
        self._max_lag = max_lag
        self.identifier += f"_maxlag-{max_lag}"

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        # `_lagged_sum(x, y)`, like hyppo's DcorrX(x, y), pairs the present
        # of x with the past of y, so it measures y -> x: source i goes second.
        z = data.to_numpy()
        x = z[j]
        y = z[i]
        return _lagged_sum(Dcorr(compute_distance=None), x, y, self._max_lag)


class CrossMultiscaleGraphCorrelation(Directed, Unsigned):
    """Cross-multiscale graph correlation (MGCX; Mehta et al. 2020).

    As `CrossDistanceCorrelation`, with MGC in place of dcorr.
    """

    name = "Cross-multiscale graph correlation"
    identifier = "mgcx"
    labels = ["unsigned", "distance", "temporal", "directed", "nonlinear"]

    def __init__(self, max_lag=1):
        self._max_lag = max_lag
        self.identifier += f"_maxlag-{max_lag}"

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        # As in CrossDistanceCorrelation: source i goes second.
        z = data.to_numpy()
        x = z[j]
        y = z[i]
        return _lagged_sum(MGC(compute_distance=None), x, y, self._max_lag)


class TimeWarping(_RetainedHalf, Undirected, Unsigned):

    labels = ["unsigned", "distance", "temporal", "undirected", "nonlinear"]

    def __init__(self, global_constraint=None):
        gcstr = global_constraint
        if gcstr is not None:
            gcstr = gcstr.replace("_", "-")
            self.identifier += f"_constraint-{gcstr}"
        self._global_constraint = global_constraint

    @property
    def simfn(self):
        try:
            return self._simfn
        except AttributeError:
            raise NotImplementedError(
                f"Add the similarity function for {self.identifier}"
            )

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        z = data.to_numpy(squeeze=True)
        return self._simfn(z[i], z[j], global_constraint=self._global_constraint)


class DynamicTimeWarping(TimeWarping):
    """DTW with optional dtaidistance C backend and Sakoe-Chiba band constraint.

    Falls back to tslearn if dtaidistance is not available or for itakura constraint.
    """

    name = "Dynamic time warping"
    identifier = "dtw"

    def __init__(
        self,
        global_constraint=None,
        sakoe_chiba_radius=None,
        sakoe_chiba_ratio=None,
        normalise=False,
        **kwargs,
    ):
        if sakoe_chiba_radius is not None and sakoe_chiba_ratio is not None:
            raise ValueError("Set only one of sakoe_chiba_radius or sakoe_chiba_ratio.")
        # Validated, not coerced: `int(2.7)` silently becomes a band of 2 and
        # `float(True)` a ratio of 1.0, both of which look like the caller's
        # intent and are not.
        if sakoe_chiba_radius is not None:
            sakoe_chiba_radius = require_int("sakoe_chiba_radius",
                                             sakoe_chiba_radius, minimum=1)
        if sakoe_chiba_ratio is not None:
            sakoe_chiba_ratio = require_positive_float("sakoe_chiba_ratio",
                                                       sakoe_chiba_ratio)

        super().__init__(global_constraint=global_constraint, **kwargs)
        self._simfn = tslearn.metrics.dtw
        self._sakoe_chiba_radius = sakoe_chiba_radius
        self._sakoe_chiba_ratio = sakoe_chiba_ratio
        self._normalise = normalise
        self._warned_itakura_fallback = False
        self._warned_c_fallback = False

        if global_constraint == "sakoe_chiba":
            if sakoe_chiba_radius is not None:
                self.identifier += f"_radius-{sakoe_chiba_radius}"
            elif sakoe_chiba_ratio is not None:
                self.identifier += f"_ratio-{fmt_param(sakoe_chiba_ratio)}"
            else:
                self.identifier += "_radius-auto"
        if normalise:
            self.identifier += "_rmse"

    def _resolve_radius(self, n):
        if self._sakoe_chiba_radius is not None:
            return min(self._sakoe_chiba_radius, max(1, n - 1))
        if self._sakoe_chiba_ratio is not None:
            ratio_radius = int(np.ceil(self._sakoe_chiba_ratio * n))
            return min(max(1, ratio_radius), max(1, n - 1))
        return _auto_sakoe_radius(n)

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        z = data.to_numpy(squeeze=True)
        x = np.ascontiguousarray(z[i], dtype=np.double)
        y = np.ascontiguousarray(z[j], dtype=np.double)
        constraint = self._global_constraint
        n = min(len(x), len(y))

        if constraint == "itakura":
            if not self._warned_itakura_fallback:
                warnings.warn(
                    "DynamicTimeWarping(itakura) falls back to tslearn; "
                    "dtaidistance C backend does not support itakura.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_itakura_fallback = True
            d = tslearn.metrics.dtw(x, y, global_constraint="itakura")
            return d / np.sqrt(n) if self._normalise else d

        radius = None
        window = None
        if constraint == "sakoe_chiba":
            radius = self._resolve_radius(n)
            window = radius + 1

        try:
            from dtaidistance import dtw as _dtw_c
            from dtaidistance.exceptions import CythonException
            kwargs = {"use_c": True}
            if window is not None:
                kwargs["window"] = window
            d = _dtw_c.distance(x, y, **kwargs)
        except ImportError:
            # dtaidistance not installed — use tslearn
            if constraint == "sakoe_chiba":
                d = tslearn.metrics.dtw(
                    x, y, global_constraint="sakoe_chiba", sakoe_chiba_radius=radius,
                )
            else:
                d = tslearn.metrics.dtw(x, y, global_constraint=constraint)
        except (CythonException, ValueError):
            if not self._warned_c_fallback:
                warnings.warn(
                    "dtaidistance C backend unavailable for DynamicTimeWarping; "
                    "falling back to tslearn.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_c_fallback = True
            if constraint == "sakoe_chiba":
                d = tslearn.metrics.dtw(
                    x, y, global_constraint="sakoe_chiba", sakoe_chiba_radius=radius,
                )
            else:
                d = tslearn.metrics.dtw(x, y)
        return d / np.sqrt(n) if self._normalise else d

    @parse_multivariate
    def multivariate(self, data):
        """Batch DTW via dtaidistance.dtw.distance_matrix when available."""
        Z = data.to_numpy(squeeze=True)  # (M, T)
        M = Z.shape[0]
        series = [np.ascontiguousarray(Z[i], dtype=np.double) for i in range(M)]

        constraint = self._global_constraint

        if constraint == "itakura":
            # dtaidistance doesn't support itakura; fall back to bivariate loop
            A = np.full((M, M), np.nan)
            for i in range(M):
                for j in range(i + 1, M):
                    d = tslearn.metrics.dtw(
                        series[i], series[j], global_constraint="itakura"
                    )
                    A[i, j] = d
                    A[j, i] = d
            # The bivariate path and the dtaidistance path both divide by
            # sqrt(T) under `normalise`; this branch did not, so
            # `bivariate(data, i, j)` and `multivariate(data)[i, j]` differed by
            # a factor of sqrt(T) for the itakura constraint alone.
            if self._normalise:
                A = A / np.sqrt(Z.shape[1])
            return A

        try:
            from dtaidistance import dtw as _dtw_c

            kwargs = {"use_c": True, "compact": False}
            if constraint == "sakoe_chiba":
                n = min(len(s) for s in series)
                radius = self._resolve_radius(n)
                kwargs["window"] = radius + 1

            try:
                dm = _dtw_c.distance_matrix(series, **kwargs)
            except Exception:
                kwargs["use_c"] = False
                dm = _dtw_c.distance_matrix(series, **kwargs)

            dm = np.array(dm)
            mask_upper = np.triu(np.ones((M, M), dtype=bool), k=1)
            dm_sym = np.where(mask_upper, dm, dm.T)
            np.fill_diagonal(dm_sym, np.nan)
            if self._normalise:
                T = Z.shape[1]
                dm_sym = dm_sym / np.sqrt(T)
            return dm_sym

        except ImportError:
            # Fall back to bivariate loop with tslearn
            A = np.full((M, M), np.nan)
            for i in range(M):
                for j in range(i + 1, M):
                    d = self.bivariate(data, i=i, j=j)
                    A[i, j] = d
                    A[j, i] = d
            return A


def _banded_lcss(x, y, global_constraint=None, eps=1.0):
    """LCSS similarity in which only pairs inside the global band may match.

    The band (tslearn's own Sakoe-Chiba radius-1 or Itakura slope-2 mask)
    restricts which pairs count as matches, not where the recursion may pass
    (Vlachos et al. 2002). tslearn < 0.10 ignored the band altogether, so both
    constrained SPIs equalled `lcss`. tslearn 0.10 applies it by holding every
    masked cell at zero, which is the same thing for the Sakoe-Chiba band but
    not for the Itakura parallelogram: that pinches to a single cell at each
    end, so any pair whose last samples differ by more than `eps` scores
    exactly 0 (43% of random pairs at T=100), and the rest can read low.

    A row of the LCSS table is non-decreasing, and a match cell's
    `A[i-1, j-1] + 1` is never below its left or upper neighbour, so each row
    is the running maximum of its candidates; that is one vectorised pass per
    row instead of a Python loop per cell.
    """
    if global_constraint is None:
        return tslearn.metrics.lcss(x, y, eps=eps)
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n, m = len(x), len(y)
    if global_constraint == "sakoe_chiba":
        band = tslearn.metrics.sakoe_chiba_mask(n, m)
    elif global_constraint == "itakura":
        band = tslearn.metrics.itakura_mask(n, m)
    else:
        raise ValueError(f"Unknown LCSS global constraint: {global_constraint}")
    match = np.asarray(band, dtype=bool) & (np.abs(x[:, None] - y[None, :]) <= eps)
    prev = np.zeros(m + 1)
    for i in range(n):
        cand = np.where(match[i], prev[:-1] + 1, prev[1:])
        prev = np.concatenate(([0.0], np.maximum.accumulate(cand)))
    return prev[-1] / min(n, m)


class LongestCommonSubsequence(TimeWarping):

    name = "Longest common subsequence"
    identifier = "lcss"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._simfn = _banded_lcss


class SoftDynamicTimeWarping(TimeWarping):
    """Soft-DTW (Cuturi & Blondel 2017) at tslearn's default ``gamma = 1``.

    This is the raw soft-minimum alignment cost, which is **not a distance**:
    it can be negative, and ``softdtw(x, x)`` is not 0. The non-negative
    soft-DTW *divergence* of Blondel, Mensch & Vert (2021),
    ``D(x, y) - (D(x, x) + D(y, y)) / 2``, is a different quantity and is not
    what pyspi ships under this identifier.
    """

    name = "Soft dynamic time warping"
    identifier = "softdtw"

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        # The constraint used to be dropped here, so both constrained SPIs
        # equalled `softdtw`. tslearn holds cells outside the band at +inf,
        # which for a soft minimum over paths is exactly "no admissible path
        # passes through them" (unlike LCSS; see `_banded_lcss`). Its defaults
        # apply: Sakoe-Chiba radius 1, Itakura slope 2.
        z = data.to_numpy(squeeze=True)
        return tslearn.metrics.soft_dtw(
            z[i], z[j], global_constraint=self._global_constraint)


class Barycenter(Undirected, Signed):
    """A statistic of the barycenter of the pair under the Euclidean, DTW
    (DBA), subgradient-DBA or soft-DTW geometry. The barycenter of {x, y} is a
    function of the unordered pair, so the measure is undirected.

    ``statistic="corr"`` is the alignment-adjusted correlation
    ``2 * mean(b**2) / mean_energy - 1`` of the barycenter ``b``, where
    ``mean_energy`` is the average of ``mean(x**2)`` and ``mean(y**2)``. For
    the Euclidean barycenter ``(x + y) / 2`` of two zero-mean series this is
    exactly Pearson's r (``mean(b**2) = (var_x + var_y + 2 cov) / 4``); for
    the DTW geometries it is the share of the pair's amplitude that survives
    averaging after optimal alignment, on the same [-1, 1] scale (exactly 1
    for identical series under DBA; soft-DTW's barycenter is smoothed even
    then, so its value for an identical pair is below 1, about 0.87 at
    ``gamma=1`` on a random walk). It is the
    only statistic of the barycenter that carries pairwise information: the
    barycenter of two zero-mean series has mean zero by construction, so
    ``statistic="mean"`` without ``squared`` reports the residue of warping
    (it tracked marginal skewness, not coupling, in the 3.0.0 make-up audit),
    and its maximum tracks the larger of the two series' own maxima. The
    squared mean is a monotone function of ``corr`` and is not bundled; the
    ``mean`` and ``max`` statistics remain constructible.
    """

    name = "Barycenter"
    identifier = "bary"
    labels = ["distance", "signed", "undirected", "temporal", "nonlinear"]
    _cache_namespace = "barycenter"

    @property
    def _cache_subkey(self):
        # Actual cache is keyed (mode, pair); statistic/squared are post-lookup
        # transforms. Sub-bucket amortization by mode so 4 separate caches don't
        # get lumped into one and mis-cost the cheap modes.
        return (self._mode,)

    def __init__(self, mode="euclidean", squared=False, statistic="mean"):
        if mode == "euclidean":
            self._fn = euclidean_barycenter
        elif mode == "dtw":
            self._fn = dtw_barycenter_averaging
        elif mode == "sgddtw":
            # A generator of its own for every fit. Left to its default,
            # tslearn draws the update order from the global NumPy stream, so
            # a pair's barycenter depended on how many had been fitted before
            # it: on process order, on which SPIs ran earlier, on serial
            # versus parallel, and on which other pairs were computed at all.
            self._fn = functools.partial(
                dtw_barycenter_averaging_subgradient, random_state=42)
        elif mode == "softdtw":
            self._fn = softdtw_barycenter
        else:
            raise NameError(f"Unknown Barycenter mode: {mode}")
        self._mode = mode

        self._squared = squared
        self._preproc = lambda x: x
        if squared:
            self._preproc = lambda x: x**2
            self.identifier += f"-sq"

        self._statistic = statistic
        if statistic == "corr":
            if squared:
                raise ValueError("statistic='corr' already squares the barycenter; "
                                 "squared=True is not meaningful with it.")
            # 2 E[b^2] / E[energy] - 1, with the pair's energy supplied by
            # `bivariate`: Pearson's r for the Euclidean barycenter, an
            # alignment-adjusted correlation otherwise.
            self._statfn = lambda x, e: 2.0 * np.nanmean(x ** 2) / e - 1.0
        elif statistic == "mean":
            self._statfn = lambda x: np.nanmean(self._preproc(x))
        elif statistic == "max":
            self._statfn = lambda x: np.nanmax(self._preproc(x))
        elif statistic == "max_time":
            self._statfn = lambda x: np.argmax(self._preproc(x))
        else:
            raise NameError(f"Unknown statistic: {statistic}")

        self.identifier += f"_{mode}_{statistic}"

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        # One barycenter per unordered pair. The subgradient (sgddtw) fit is
        # not symmetric in its inputs -- tslearn visits the series in a seeded
        # random order -- so the pair is always fitted as [z[max], z[min]],
        # the orientation the inherited loop reaches first. Caching whichever
        # orientation was asked for first made `bivariate(i=0, j=1)` disagree
        # with `multivariate()[0, 1]`.
        key = (max(i, j), min(i, j))
        try:
            bc = data.barycenter[self._mode][key]
        except (AttributeError, KeyError):
            z = data.to_numpy(squeeze=True)
            bc = self._fn(z[list(key)])
            try:
                data.barycenter[self._mode][key] = bc
            except AttributeError:
                data.barycenter = {self._mode: {key: bc}}
            except KeyError:
                data.barycenter[self._mode] = {key: bc}

        if self._statistic == "corr":
            z = data.to_numpy(squeeze=True)
            energy = 0.5 * (np.mean(z[i] ** 2) + np.mean(z[j] ** 2))
            return self._statfn(bc, energy)
        return self._statfn(bc)


class GromovWasserstainTau(_RetainedHalf, Undirected, Unsigned):
    """GW_tau (Kravtsova, McGee & Dawes 2023, Bull. Math. Biol. 85, 71).

    Each series becomes the planar trajectory ``(t, x_t)``; ``vec_geo_dist``
    is the vector of intrinsic (arc-length) distances of every point from the
    trajectory's start, and ``wass_sorted`` the closed-form 2-Wasserstein
    distance between the two sorted vectors (Proposition 1(e) of the paper:
    ``sqrt(mean((sorted v_i - sorted v_j)^2))`` at equal lengths).

    The distance vector has one entry per point, the first being 0 for the
    start itself (the paper's ``d_X(r_X, x_k)`` for ``k = 1..n``). The
    authors' reference MATLAB (``vec_geo_dist.m`` in github.com/kravtsova2/
    GWtau) omits that entry and returns ``n - 1`` cumulative sums, and so did
    pyspi until 3.0.0; at equal lengths the two differ by the constant factor
    ``sqrt(n / (n - 1))``, so no comparison between pairs is affected.

    The class name keeps its historical misspelling because configs and saved
    results refer to it.
    """

    name = "Gromov-Wasserstein distance (GW_tau)"
    identifier = "gwtau"
    labels = ["unsigned", "distance", "unordered", "nonlinear", "undirected"]

    @staticmethod
    def vec_geo_dist(x):
        diffs = np.diff(x, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        # The start is at distance 0 from itself: one entry per point.
        return np.concatenate([[0.0], np.cumsum(distances)])

    @staticmethod
    def wass_sorted(x1, x2):
        x1 = np.sort(x1)[::-1] # sort in descending order
        x2 = np.sort(x2)[::-1]

        if len(x1) == len(x2):
            res = np.sqrt(np.mean((x1 - x2) ** 2))
        else:
            N, M = len(x1), len(x2)
            i_ratios = np.arange(1, N + 1) / N
            j_ratios = np.arange(1, M + 1) / M


            min_values = np.minimum.outer(i_ratios, j_ratios)
            max_values = np.maximum.outer(i_ratios - 1/N, j_ratios - 1/M)

            lam = np.where(min_values > max_values, min_values - max_values, 0)

            diffs_squared = (x1[:, None] - x2) ** 2
            my_sum = np.sum(lam * diffs_squared)

            res = np.sqrt(my_sum)

        return res

    @staticmethod
    def gwtau(xi, xj):
        timei = np.arange(len(xi))
        timej = np.arange(len(xj))
        traji = np.column_stack([timei, xi])
        trajj = np.column_stack([timej, xj])

        vi = GromovWasserstainTau.vec_geo_dist(traji)
        vj = GromovWasserstainTau.vec_geo_dist(trajj)
        gw = GromovWasserstainTau.wass_sorted(vi, vj)

        return gw

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        x, y = data.to_numpy()[[i, j]]
        stat = self.gwtau(x, y)
        return stat


# ---------------------------------------------------------------------------
# CrossPairwiseDistance: lagged Euclidean distance
# ---------------------------------------------------------------------------

class CrossPairwiseDistance(Undirected, Unsigned):
    """Root-T-normalised cost of a lag-shifted alignment path, over lags 0..tau.

    At each lag ``t`` the two series are paired with an offset of ``t``, the
    ``t`` samples that fall off each end are stuttered against the opposite
    series' boundary sample, and the cost is
    ``sqrt(sum(differences**2) / T)``. The forward and backward offsets are
    minimised over, so the result is symmetric in the pair; ``min`` or ``mean``
    then reduces over ``t``.

    The ``_rmse`` in the identifier is the same house label
    `PairwiseDistance` uses: divided by ``sqrt(T)``. It is **not** the
    conventional DTW "path-length RMSE", which divides by the number of steps
    in the path -- here the divisor is the record length, and the boundary
    stutter means the path has ``T + t`` steps rather than ``T``. The point of
    the stutter is that the alignment is a *valid* DTW path (monotone,
    continuous, matched endpoints), and DTW minimises over all such paths, so
    ``dtw_rmse <= xpdist`` holds by construction rather than empirically.

    At ``tau=0`` the path is the identity alignment and the value is exactly
    ``pdist_euclidean`` with ``normalise=True``.
    """
    name = "Cross pairwise distance"
    labels = ["distance", "nonlinear", "undirected", "temporal"]

    def __init__(self, metric="euclidean", tau=1, statistic="min"):
        if metric != "euclidean":
            raise ValueError(f"Unsupported metric: {metric!r}. Only 'euclidean' supported.")
        # `int(tau) < 0` accepted `tau=1.7` (truncated to 1), `tau=True`
        # (silently 1) and raised an opaque conversion error on nan/inf. tau
        # counts samples, so it must be an integer.
        tau = require_int("tau", tau, minimum=0)
        stat = str(statistic).lower()
        if stat not in {"min", "mean"}:
            raise ValueError(f"statistic must be 'min' or 'mean', got: {statistic!r}")
        self._metric = metric
        self._tau = tau
        self._statistic = stat
        # `_rmse` is the house label for the sqrt(T) normalisation, shared with
        # PairwiseDistance and DynamicTimeWarping(normalise=True). See the class
        # docstring for what it does and does not claim.
        self.identifier = f"xpdist_{metric}_tau-{self._tau}_{stat}_rmse"

    @staticmethod
    def _shifted_dist_rmse(x, y, s):
        # Realises one specific DTW path: pair x[k+s] with y[k] over the overlap,
        # then stutter the dropped boundary samples against the corner of the
        # opposite series. Since the result is the cost of a valid DTW path and
        # DTW minimises over all paths, the resulting xpdist satisfies
        # dtw_rmse <= xpdist by construction. Normalise by sqrt(T) to match
        # DynamicTimeWarping(normalise=True).
        T = len(x)
        if s == 0:
            diff = x - y
        elif s > 0:
            left  = x[:s]      - y[0]
            mid   = x[s:]      - y[: T - s]
            right = x[T - 1]   - y[T - s :]
            diff = np.concatenate([left, mid, right])
        else:
            sa = -s
            left  = y[:sa]     - x[0]
            mid   = y[sa:]     - x[: T - sa]
            right = y[T - 1]   - x[T - sa :]
            diff = np.concatenate([left, mid, right])
        return float(np.sqrt(np.sum(diff ** 2) / T))

    def _cross_dist(self, x, y):
        tau = self._tau
        T = len(x)
        per_lag = np.empty(tau + 1)
        per_lag[0] = self._shifted_dist_rmse(x, y, 0)
        for t in range(1, tau + 1):
            if t >= T:
                per_lag[t] = np.inf
                continue
            fwd = self._shifted_dist_rmse(x, y, +t)
            bwd = self._shifted_dist_rmse(x, y, -t)
            per_lag[t] = min(fwd, bwd)
        if self._statistic == "min":
            return float(np.min(per_lag))
        finite = per_lag[np.isfinite(per_lag)]
        return float(np.mean(finite)) if finite.size > 0 else np.nan

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        Z = data.to_numpy(squeeze=True)
        return self._cross_dist(Z[i], Z[j])

    @parse_multivariate
    def multivariate(self, data):
        Z = data.to_numpy(squeeze=True)  # (M, T)
        M = Z.shape[0]
        A = np.full((M, M), np.nan)
        for i in range(M):
            for j in range(i + 1, M):
                d = self._cross_dist(Z[i], Z[j])
                A[i, j] = d
                A[j, i] = d
        return A
