import functools
import os
import warnings

from hyppo.independence import MGC
# hyppo's compiled centring and sum; pyspi applies the final step itself
# (`_dcorr_centred`), so the value does not depend on the hyppo version.
from hyppo.independence.dcorr import _center_distmat, _dcov
import numba
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import pairwise_distances, pairwise_kernels
from tslearn.barycenters import (
    euclidean_barycenter,
    dtw_barycenter_averaging,
    dtw_barycenter_averaging_subgradient,
    softdtw_barycenter,
)
import tslearn.metrics

from pyspi.base import (
    Directed,
    NotAppropriate,
    Undirected,
    Unsigned,
    Signed,
    fill_pairwise,
    parse_bivariate,
    parse_multivariate,
)
from pyspi.utils import fmt_param, require_int, require_positive_float

# A guarded import (absent before scipy 1.14), so it follows the sorted groups.
try:
    # The steps of scipy's `multiscale_graphcorr`, which hyppo's MGC calls,
    # taken separately so that each series' share is computed once
    # (`_mgc_parts`). Private: if a scipy release moves them, MGC and MGCX
    # fall back to hyppo for every pair.
    from scipy.stats._mgc import _smooth_mgc_map, _threshold_mgc_map
    from scipy.stats._stats import _center_distance_matrix, _local_covariance
except ImportError:  # pragma: no cover
    _local_covariance = None


# ---------------------------------------------------------------------------
# DTW Sakoe-Chiba auto-radius configuration (env-var tunable)
# ---------------------------------------------------------------------------
_DTW_SAKOE_LINEAR_FRAC = float(os.getenv("PYSPI_DTW_SAKOE_LINEAR_FRAC", "0.10"))
_DTW_SAKOE_SQRT_COEFF = float(os.getenv("PYSPI_DTW_SAKOE_SQRT_COEFF", "1.5"))
_DTW_SAKOE_MIN_RADIUS = int(os.getenv("PYSPI_DTW_SAKOE_MIN_RADIUS", "10"))


def _auto_sakoe_radius(length):
    """``min(ceil(0.1 T), ceil(1.5 sqrt(T)))``, at least 10, at most ``T - 1``."""
    if length <= 1:
        return 1
    linear = int(np.ceil(max(0.0, _DTW_SAKOE_LINEAR_FRAC) * length))
    sqrt_scaled = int(np.ceil(max(0.0, _DTW_SAKOE_SQRT_COEFF) * np.sqrt(length)))
    radius = min(linear, sqrt_scaled)
    radius = max(1, _DTW_SAKOE_MIN_RADIUS, radius)
    return min(radius, length - 1)


class PairwiseDistance(Undirected, Unsigned):
    """
    Distance between two time series, treated as vectors of length T.

    ``A[i, j]`` is ``d(x_i, x_j)`` for the chosen ``metric``, computed with
    scikit-learn's ``pairwise_distances``. Undirected and unsigned (zero for
    identical series); a distance, so lower means more alike.

    ``normalise=True`` divides by ``sqrt(T)`` and appends ``_rmse`` to the
    identifier. For the Euclidean metric that is the root mean square
    difference, ``sqrt(mean((x_i - x_j)**2))``. For the other metrics
    ``_rmse`` only names the division by ``sqrt(T)``: the result is not a
    root mean square, and only the Euclidean distance grows as ``sqrt(T)``
    with independent differences, so only it becomes comparable across
    record lengths.

    Parameters
    ----------
    metric : str, optional
        Any metric that ``sklearn.metrics.pairwise_distances`` accepts, such
        as ``'euclidean'``, ``'cityblock'``, ``'chebyshev'``, ``'canberra'``,
        ``'braycurtis'`` or ``'cosine'``. Default: ``'euclidean'``.
    normalise : bool, optional
        Divide by ``sqrt(T)``. Default: False.
    """

    name = "Pairwise distance"
    identifier = "pdist"
    # A distance: lower means more alike.
    labels = ["distance", "unordered", "nonlinear", "undirected", "inverted"]

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


# TODO: include optional kernels in each method.

# ---------------------------------------------------------------------------
# Itakura-constrained DTW
# ---------------------------------------------------------------------------
def _itakura_mask(n, m, max_slope=2.0):
    """
    ``tslearn.metrics.itakura_mask(n, m, max_slope)``, vectorised.

    The same arithmetic, column by column, as tslearn's numba version, without
    its per-process compilation.
    """
    min_slope = 1 / float(max_slope)
    max_slope = max_slope * (float(n) / float(m))
    min_slope = min_slope * (float(n) / float(m))
    j = np.arange(m)
    lower = np.ceil(np.maximum(
        np.round(min_slope * j, decimals=2),
        np.round((n - 1) - max_slope * (m - 1) + max_slope * j, decimals=2)))
    upper = np.floor(np.minimum(
        np.round(max_slope * j, decimals=2),
        np.round((n - 1) - min_slope * (m - 1) + min_slope * j, decimals=2)) + 1)
    rows = np.arange(n)[:, None]
    mask = (rows >= lower) & (rows < upper)
    if not (mask.any(axis=1).all() and mask.any(axis=0).all()):
        raise RuntimeWarning(
            "'itakura_max_slope' constraint is unfeasible "
            "(ie. leads to no admissible path) for the "
            "provided time series sizes",
        )
    return mask


@functools.lru_cache(maxsize=8)
def _itakura_rows(n, m):
    """Each row's admitted columns ``lo[i]:hi[i]`` (the parallelogram is convex)."""
    band = _itakura_mask(n, m)
    lo = band.argmax(axis=1)
    hi = m - band[:, ::-1].argmax(axis=1)
    return lo, hi


@numba.njit(cache=True, nogil=True)
def _banded_dtw(x, y, lo, hi):
    """
    tslearn's DTW recursion over row ``i``'s columns ``lo[i]:hi[i]`` only.

    Each cell is ``(x_i - y_j)^2 + min(up, left, diagonal)`` in tslearn's
    order, so the value is bitwise tslearn's; cells outside the band stay at
    +inf as in tslearn's masked accumulated-cost matrix.
    """
    m = y.shape[0]
    prev = np.full(m + 1, np.inf)
    prev[0] = 0.0
    cur = np.empty(m + 1)
    for i in range(x.shape[0]):
        cur[:] = np.inf
        for j in range(lo[i], hi[i]):
            d = x[i] - y[j]
            cur[j + 1] = d * d + min(prev[j + 1], cur[j], prev[j])
        prev, cur = cur, prev
    return np.sqrt(prev[m])


def _itakura_dtw(x, y):
    """``tslearn.metrics.dtw(x, y, global_constraint="itakura")`` for 1-D series."""
    lo, hi = _itakura_rows(len(x), len(y))
    return _banded_dtw(x, y, lo, hi)


# ---------------------------------------------------------------------------
# Distance correlation from per-series centred distance matrices
# ---------------------------------------------------------------------------
# Upper bound on the centred matrices one SPI call holds at a time.
_DCORR_CACHE_BYTES = 2**28


def _centred(dist, bias):
    """hyppo's centred distance matrix and its sum of squares."""
    C = _center_distmat(np.ascontiguousarray(dist), bias)
    return C, _dcov(C, C, bias, False)


def _dcorr_centred(cx, cy, bias):
    """
    hyppo 0.5.2's ``_dcorr`` on two `_centred` results, bit for bit.

    0 when a variance is not positive, or (biased only) the covariance is
    not; the biased ratio is square-rooted (dCor), the unbiased one is the
    U-centred estimate of dCor^2 and can be negative. Other hyppo releases
    differ in this final step, so it is applied here rather than taken from
    the installed hyppo.
    """
    (x, vx), (y, vy) = cx, cy
    covar = _dcov(x, y, bias, False)
    if vx <= 0 or vy <= 0 or (covar <= 0 and bias):
        return 0.0
    stat = covar / np.sqrt(np.real(vx * vy))
    return np.sqrt(stat) if bias else stat


def _cached_pairs(identifier, M, prepare, combine, item_bytes, directed=False):
    """
    `fill_pairwise` of ``combine(prepare(i), prepare(j))`` over every pair.

    Each series is prepared once per block of series it is paired within,
    and at most two blocks (about `_DCORR_CACHE_BYTES`) are held at a time.
    When every series fits in one block, each is prepared once and the
    pairs are visited in `Undirected.multivariate`'s order. Undirected:
    ``i < j``, mirrored.
    """
    b = int(max(1, _DCORR_CACHE_BYTES // (2 * item_bytes)))
    blocks = [range(s, min(s + b, M)) for s in range(0, M, b)]
    pairs = []
    for p, P in enumerate(blocks):
        for Q in blocks[p:]:
            for j in Q:
                for i in P:
                    if i < j:
                        pairs += [(i, j), (j, i)] if directed else [(i, j)]
    cache = {}

    def get(s, keep):
        if s not in cache:
            for k in [k for k in cache if k // b not in keep]:
                del cache[k]
            cache[s] = prepare(s)
        return cache[s]

    def compute(i, j):
        keep = (i // b, j // b)
        return combine(get(i, keep), get(j, keep))

    A = np.full((M, M), np.nan)
    return fill_pairwise(identifier, A, pairs, compute, mirror=not directed)


def _dcorr_spi_matrix(spi, data, distance):
    """``dcorr(distance(z_i), distance(z_j))`` for every pair ``i < j``."""
    z = data.to_numpy()
    M, T = z.shape[0], z.shape[1]
    return _cached_pairs(
        spi.identifier, M,
        lambda s: _centred(distance(z[s]), spi._biased),
        lambda a, b: _dcorr_centred(a, b, spi._biased),
        item_bytes=8 * T * T)


def _euclidean_distance(v):
    return pairwise_distances(v, metric="euclidean")


def _median_gaussian_distance(v):
    """
    ``1 - K / max(K)`` for a Gaussian kernel ``K`` of median-heuristic width.

    hyppo's ``Hsic`` statistic is dcorr of these induced distances. Here each
    variable's width is the median distance between its own samples; hyppo
    applies ``x``'s width to ``y`` as well, which makes its statistic depend
    on argument order.
    """
    l2 = pairwise_distances(v, metric="l2")
    n = l2.shape[0]
    med = np.median(l2[~np.eye(n, dtype=bool)])
    # As hyppo: a zero median (mostly tied samples) falls back to width 1.
    med = med if med else 1
    K = pairwise_kernels(v, metric="rbf", gamma=1.0 / (2 * (med**2)))
    return 1 - K / np.max(K)


class HilbertSchmidtIndependenceCriterion(Undirected, Unsigned):
    """
    Normalised Hilbert-Schmidt Independence Criterion (HSIC) [1]_ [2]_.

    The statistic of hyppo's ``Hsic``: the distance correlation (as
    `DistanceCorrelation`) of the distances ``1 - K / max(K)`` induced by a
    Gaussian kernel ``K``, which equals a normalised HSIC [3]_. Each series is
    one sample of ``T`` scalar observations, so time order is ignored.

    Unlike hyppo, which applies the first variable's kernel width to both,
    each variable gets its own median-heuristic width [2]_ (the median
    distance between its own samples), so the value is symmetric in the
    pair. A zero median (mostly tied samples) falls back to width 1, as in
    hyppo.

    Undirected and unsigned. With ``biased=False`` the value is the U-centred
    estimate of the squared statistic and can be slightly negative under
    independence; with ``biased=True`` it is the square root of the
    V-statistic ratio, in ``[0, 1]``. The two are on different scales.

    References
    ----------
    .. [1] A. Gretton, O. Bousquet, A. Smola, B. Schölkopf, "Measuring
           statistical dependence with Hilbert-Schmidt norms", Algorithmic
           Learning Theory (ALT 2005), Lecture Notes in Computer Science
           3734, 63-77 (2005).
    .. [2] A. Gretton, K. Fukumizu, C. H. Teo, L. Song, B. Schölkopf,
           A. J. Smola, "A kernel statistical test of independence", Advances
           in Neural Information Processing Systems 20, 585-592 (2007).
    .. [3] D. Sejdinovic, B. Sriperumbudur, A. Gretton, K. Fukumizu,
           "Equivalence of distance-based and RKHS-based statistics in
           hypothesis testing", Annals of Statistics 41(5), 2263-2291 (2013).

    Parameters
    ----------
    biased : bool, optional
        Use the biased (V-statistic) estimator instead of the unbiased
        (U-centred) one, and append ``_biased`` to the identifier.
        Default: False.
    """

    name = "Hilbert-Schmidt Independence Criterion"
    identifier = "hsic"
    labels = ["distance", "unordered", "nonlinear", "undirected"]

    def __init__(self, biased=False):
        self._biased = biased
        if biased:
            self.identifier += "_biased"

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        x, y = data.to_numpy()[[i, j]]
        return _dcorr_centred(
            _centred(_median_gaussian_distance(x), self._biased),
            _centred(_median_gaussian_distance(y), self._biased), self._biased)

    @parse_multivariate
    def multivariate(self, data):
        return _dcorr_spi_matrix(self, data, _median_gaussian_distance)


@numba.njit(cache=True, nogil=True)
def _row_dominance_counts(rx, order, ry):
    """
    ``c[i, j] = #{k : rx[i, k] <= rx[i, j] and ry[i, k] <= ry[i, j]}``.

    ``rx`` and ``ry`` are row-wise max-ranks (1..n) and ``order[i]`` sorts
    row ``i`` of ``rx``. Offline 2-D dominance counting with a Fenwick tree
    per row: because ranks are max-ranks, the ``k`` with
    ``rx[i, k] <= rx[i, j]`` are exactly the entries of row ``i`` up to and
    including ``j``'s tie group in ``rx`` order, so ``j``'s query is answered
    once that group has been inserted. O(n log n) per row.
    """
    n_rows, n = rx.shape
    out = np.empty((n_rows, n), dtype=np.int64)
    tree = np.zeros(n + 1, dtype=np.int64)
    for i in range(n_rows):
        tree[:] = 0
        p = 0
        while p < n:
            q = p
            v = rx[i, order[i, p]]
            while q < n and rx[i, order[i, q]] == v:
                idx = ry[i, order[i, q]]
                while idx <= n:
                    tree[idx] += 1
                    idx += idx & -idx
                q += 1
            for t in range(p, q):
                j = order[i, t]
                idx = ry[i, j]
                acc = 0
                while idx > 0:
                    acc += tree[idx]
                    idx -= idx & -idx
                out[i, j] = acc
            p = q
    return out


_HHG_CHUNK_ENTRIES = 2**22


def _hhg_ranks(x):
    """
    The part of `_hhg_statistic` that depends on one series alone.

    Row-wise max-ranks of its Euclidean distance matrix,
    ``#{k : d[i, k] <= d[i, j]}`` (counting ``k = i`` and ``k = j``
    themselves), and the order that sorts each row of them.
    """
    d = pairwise_distances(x, metric="euclidean")
    n = d.shape[0]
    ranks = np.empty((n, n), dtype=np.int32)
    order = np.empty((n, n), dtype=np.int32)
    # Rows are independent; bound the float64 and int64 temporaries' size.
    step = max(1, _HHG_CHUNK_ENTRIES // n)
    for start in range(0, n, step):
        stop = min(n, start + step)
        ranks[start:stop] = rankdata(d[start:stop], method="max", axis=1)
        order[start:stop] = np.argsort(ranks[start:stop], axis=1,
                                       kind="stable")
    return ranks, order


def _hhg_from_ranks(rank_x, rank_y):
    """`_hhg_statistic` from the two series' `_hhg_ranks`."""
    (rx, order), (ry, _) = rank_x, rank_y
    n = rx.shape[0]
    S = np.zeros((n, n))
    # Rows are independent; bound the int64 work arrays' size.
    step = max(1, _HHG_CHUNK_ENTRIES // n)
    for start in range(0, n, step):
        stop = min(n, start + step)
        na = rx[start:stop].astype(np.int64)
        nb = ry[start:stop].astype(np.int64)
        nab = _row_dominance_counts(rx[start:stop], order[start:stop],
                                    ry[start:stop])
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


def _hhg_statistic(x, y):
    """
    hyppo's ``HHG().statistic(x, y)`` in O(n^2 log n) rather than O(n^3).

    The sum over ``i != j`` of Pearson's chi-square for the 2x2 table that
    splits the other samples by ``d_x(i, .) <= d_x(i, j)`` and
    ``d_y(i, .) <= d_y(i, j)`` (Heller, Heller & Gorfine 2013). The table
    counts are row-wise ranks (`_hhg_ranks`, one series at a time) and 2-D
    dominance counts.
    """
    return _hhg_from_ranks(_hhg_ranks(x), _hhg_ranks(y))


class HellerHellerGorfine(Undirected, Unsigned):
    """
    Heller-Heller-Gorfine independence statistic [1]_.

    For each ordered pair of time points ``(s, t)``, ``s != t``, the other
    samples are split by whether they lie within ``|x_s - x_t|`` of ``x_s``
    and within ``|y_s - y_t|`` of ``y_s``; the statistic is the sum over all
    ``(s, t)`` of Pearson's chi-square for that 2x2 table. Each series is one
    sample of ``T`` scalar observations, so time order is ignored. The value
    equals hyppo's ``HHG().statistic``, computed in ``O(T^2 log T)`` from
    row-wise ranks and dominance counts instead of hyppo's ``O(T^3)`` loop.
    Each series' ranks are computed once for all its pairs.

    Undirected and unsigned (``hhg(x, y) == hhg(y, x)`` exactly). The sum is
    not normalised, so it grows with ``T``.

    References
    ----------
    .. [1] R. Heller, Y. Heller, M. Gorfine, "A consistent multivariate test
           of association based on ranks of distances", Biometrika 100(2),
           503-510 (2013).
    """

    name = "Heller-Heller-Gorfine Independence Criterion"
    identifier = "hhg"
    labels = ["distance", "unordered", "nonlinear", "undirected"]

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        x, y = data.to_numpy()[[i, j]]
        return _hhg_statistic(x, y)

    @parse_multivariate
    def multivariate(self, data):
        # Each series' ranks (8 T^2 bytes) are computed once, not once per
        # pair. The cache holds at most 256 MB of them, or two series'
        # (16 T^2 bytes) when T > 4096.
        z = data.to_numpy()
        T = z.shape[1]
        return _cached_pairs(self.identifier, z.shape[0],
                             lambda s: _hhg_ranks(z[s]), _hhg_from_ranks,
                             item_bytes=8 * T * T)


class DistanceCorrelation(Undirected, Unsigned):
    """
    Distance correlation [1]_.

    The distance correlation of two series, each treated as one sample of
    ``T`` scalar observations (time order is ignored), from their ``T x T``
    Euclidean distance matrices. It is zero in the population if and only if
    the two are independent, so it detects nonlinear as well as linear
    dependence.

    With ``biased=False`` the value is the unbiased (U-centred) estimate of
    the squared distance correlation [2]_, which can be slightly negative
    under independence. With ``biased=True`` it is the square root of the
    double-centred (V-statistic) ratio of [1]_, in ``[0, 1]``. The two are
    on different scales. Values match hyppo's ``Dcorr`` (0.5.2) statistic.
    Undirected and unsigned.

    References
    ----------
    .. [1] G. J. Székely, M. L. Rizzo, N. K. Bakirov, "Measuring and testing
           dependence by correlation of distances", Annals of Statistics
           35(6), 2769-2794 (2007).
    .. [2] G. J. Székely, M. L. Rizzo, "Partial distance correlation with
           methods for dissimilarities", Annals of Statistics 42(6),
           2382-2412 (2014).

    Parameters
    ----------
    biased : bool, optional
        Use the biased (V-statistic) estimator instead of the unbiased one,
        and append ``_biased`` to the identifier. Default: False.
    """

    name = "Distance correlation"
    identifier = "dcorr"
    labels = ["distance", "unordered", "nonlinear", "undirected"]

    def __init__(self, biased=False):
        self._biased = biased
        if biased:
            self.identifier += "_biased"

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        x, y = data.to_numpy()[[i, j]]
        return _dcorr_centred(_centred(_euclidean_distance(x), self._biased),
                              _centred(_euclidean_distance(y), self._biased),
                              self._biased)

    @parse_multivariate
    def multivariate(self, data):
        return _dcorr_spi_matrix(self, data, _euclidean_distance)


class MultiscaleGraphCorrelation(Undirected, Unsigned):
    """
    Multiscale graph correlation (MGC) [1]_.

    hyppo's ``MGC().statistic`` on the two series, each treated as one sample
    of ``T`` scalar observations (time order is ignored), with Euclidean
    distances. MGC computes local distance correlations restricted to each
    sample's ``k`` and ``l`` nearest neighbours in the two variables, over
    all ``(k, l)``, and reports a smoothed maximum, so it is sensitive to
    nonlinear dependence that the global distance correlation dilutes.
    Undirected. MGC needs at least 5 samples; a shorter series is not
    appropriate (``NotAppropriate``, NaN in a Calculator table). The value is
    hyppo's, bit for bit; each series' share of the MGC map (its ranks,
    centring and local variances) is computed once, not once per pair. This
    needs scipy 1.14 or later; with an older scipy the SPI calls hyppo's
    ``MGC`` for every pair, with the same values.

    References
    ----------
    .. [1] C. Shen, C. E. Priebe, J. T. Vogelstein, "From distance
           correlation to multiscale graph correlation", Journal of the
           American Statistical Association 115(529), 280-291 (2020).
    """

    name = "Multiscale graph correlation"
    identifier = "mgc"
    labels = ["distance", "unordered", "nonlinear", "undirected"]

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        x, y = data.to_numpy()[[i, j]]
        _require_mgc_samples(x.shape[0])
        if _local_covariance is None:
            return MGC().statistic(x, y)
        return _mgc_from_parts(_mgc_parts(_euclidean_distance(x)),
                               _mgc_parts(_euclidean_distance(y)))

    @parse_multivariate
    def multivariate(self, data):
        z = data.to_numpy()
        _require_mgc_samples(z.shape[1])
        if _local_covariance is None:
            return super().multivariate(data)
        return _cached_pairs(
            self.identifier, z.shape[0],
            lambda s: _mgc_parts(_euclidean_distance(z[s])), _mgc_from_parts,
            item_bytes=16 * z.shape[1] ** 2)


def _require_mgc_samples(T):
    """Refuse unless ``T`` reaches MGC's minimum, ``_MGC_MIN_SAMPLES`` samples."""
    if T < _MGC_MIN_SAMPLES:
        raise NotAppropriate(
            f"Multiscale graph correlation needs T >= {_MGC_MIN_SAMPLES} "
            f"samples; got T = {T}.")


def _mgc_parts(dist):
    """
    The share of scipy's MGC map that depends on one distance matrix.

    Its MGC-centred distances, their column ranks and its local variances
    (the diagonal of its local covariance with itself), as scipy's
    ``_local_correlations`` computes them for each of the two inputs.
    """
    cent, rank = _center_distance_matrix(dist, "mgc", True)
    var = _local_covariance(cent, cent.T, rank, rank.T).diagonal()
    return cent, rank, var


def _mgc_from_parts(px, py):
    """
    hyppo's ``MGC().statistic`` from the `_mgc_parts` of its two inputs.

    scipy's ``_local_correlations`` (the local covariance of the pair over
    the local variances, clipped at 1 and zero where a variance is not
    positive) followed by ``_mgc_stat``'s thresholding and smoothing, with
    scipy's own functions in the same order, so the value is bit-identical.
    """
    (cx, rx, vx), (cy, ry, vy) = px, py
    cov = _local_covariance(cx, cy.T, rx, ry.T)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / np.sqrt(vx.reshape(-1, 1) @ vy.reshape(-1, 1).T).real
        corr[corr > 1] = 1
    corr[vx <= 0, :] = 0
    corr[:, vy <= 0] = 0
    n, m = corr.shape
    if m == 1 or n == 1:
        return corr[m - 1][n - 1]
    sig_connect = _threshold_mgc_map(corr, cx.shape[0] - 1)
    return _smooth_mgc_map(sig_connect, corr)[0]


# Fewest samples of one term: with 3 samples the U-centred distance matrices
# vanish, so the unbiased statistic is 0/0; scipy's MGC refuses fewer than 5.
_UNBIASED_DCOV_MIN_SAMPLES = 4
_MGC_MIN_SAMPLES = 5


def _require_lagged_samples(T, max_lag, need, measure):
    """Refuse unless the shortest lagged segment, ``T - max_lag`` samples, has ``need``."""
    if T - max_lag < need:
        raise NotAppropriate(
            f"{measure} needs at least {need} samples in each lagged term; "
            f"at max_lag = {max_lag} the shortest has T - max_lag = "
            f"{T - max_lag}, so it needs T >= {max_lag + need}; got T = {T}.")


def _lagged_sum(test, x, y, max_lag):
    """
    ``sum_{l=0}^{L} (n - l)/n * S(x[l:], y[:n - l])``, as in DcorrX and MGCX.

    The cross-distance statistic of hyppo's ``DcorrX`` and ``MGCX`` as their
    docstrings define it: the present of ``x`` against the past of ``y``, i.e.
    the dependence of ``x`` on the history of ``y``. The distance matrix of
    ``x[l:]`` is the block ``distx[l:, l:]``. hyppo 0.5.2's own ``l >= 1``
    terms treat rows of the distance matrices as points, so it is not used.
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
    """
    Cross-distance correlation (DcorrX) [1]_.

    ``A[i, j] = sum_{l=0}^{max_lag} (T - l)/T * dcorr(x_j[l:], x_i[:T - l])``:
    the unbiased distance correlation (as `DistanceCorrelation` with
    ``biased=False``) between the target's present and the source's value
    ``l`` samples earlier, weighted by the overlap and summed over lags.
    Row = source, column = target; unsigned.

    This is the statistic as hyppo's ``DcorrX`` docstring defines it, with
    the distance matrix of ``x_j[l:]`` taken as the block ``D[l:, l:]`` of
    the full one. hyppo 0.5.2's own lagged terms treat rows of the distance
    matrices as points, so its implementation is not used. Each series'
    centred distance matrices are computed once per lag, not once per pair.

    With 3 samples the U-centred distance matrices are identically zero, so
    the unbiased statistic is 0/0 and every lagged term needs ``n >= 4``
    samples: the SPI is not appropriate
    (``NotAppropriate``, NaN in a Calculator table) when
    ``T < max_lag + 4``.

    References
    ----------
    .. [1] C. Shen, J. Chung, R. Mehta, T. Xu, J. T. Vogelstein,
           "Independence testing for temporal data", Transactions on Machine
           Learning Research (2024).

    Parameters
    ----------
    max_lag : int, optional
        Largest lag ``l``, in samples. Default: 1.
    """

    name = "Cross-distance correlation"
    identifier = "dcorrx"
    labels = ["distance", "temporal", "directed", "nonlinear"]

    def __init__(self, max_lag=1):
        self._max_lag = max_lag
        self.identifier += f"_maxlag-{fmt_param(max_lag)}"

    def _prepare(self, z, lag):
        """
        Centred distances of ``z[lag:]`` (as target) and ``z[:n - lag]``
        (as source), the blocks `_lagged_sum` crops.
        """
        d = _euclidean_distance(z)
        if lag == 0:
            c = _centred(d, False)
            return c, c
        m = d.shape[0] - lag
        return _centred(d[lag:, lag:], False), _centred(d[:m, :m], False)

    @staticmethod
    def _weighted_sum(stats, n):
        # `_lagged_sum`'s terms and summation, so the value is bit-identical.
        return np.sum([stats[0]] + [(n - lag) * stats[lag] / n
                                    for lag in range(1, len(stats))])

    def _require_samples(self, T):
        _require_lagged_samples(T, self._max_lag, _UNBIASED_DCOV_MIN_SAMPLES,
                                "The unbiased cross-distance correlation")

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        # Source i, target j: the target's present against the source's past.
        z = data.to_numpy()
        self._require_samples(z.shape[1])
        stats = [_dcorr_centred(self._prepare(z[j], lag)[0],
                                self._prepare(z[i], lag)[1], False)
                 for lag in range(self._max_lag + 1)]
        return self._weighted_sum(stats, z.shape[1])

    @parse_multivariate
    def multivariate(self, data):
        z = data.to_numpy()
        M, n = z.shape[0], z.shape[1]
        self._require_samples(n)
        S = [_cached_pairs(self.identifier, M,
                           lambda s, lag=lag: self._prepare(z[s], lag),
                           lambda src, tgt: _dcorr_centred(tgt[0], src[1], False),
                           item_bytes=16 * (n - lag) ** 2, directed=True)
             for lag in range(self._max_lag + 1)]
        A = np.full((M, M), np.nan)
        for i in range(M):
            for j in range(M):
                if i != j:
                    A[i, j] = self._weighted_sum([s[i, j] for s in S], n)
        return A


class CrossMultiscaleGraphCorrelation(Directed, Unsigned):
    """
    Cross-multiscale graph correlation (MGCX) [1]_.

    As `CrossDistanceCorrelation`, with hyppo's MGC statistic (as
    `MultiscaleGraphCorrelation`) in place of the distance correlation:
    ``A[i, j] = sum_{l=0}^{max_lag} (T - l)/T * mgc(x_j[l:], x_i[:T - l])``.
    Row = source, column = target; hyppo's ``MGCX`` implementation is not
    used, for the same reason. MGC needs at least 5 samples, so every lagged
    term does, and the SPI is not appropriate (``NotAppropriate``, NaN in a
    Calculator table) when ``T < max_lag + 5``. Each series' share of each
    lagged MGC map is computed once per lag, not once per pair. This needs
    scipy 1.14 or later; with an older scipy the SPI calls hyppo's ``MGC``
    for every pair and lag, with the same values.

    References
    ----------
    .. [1] C. Shen, J. Chung, R. Mehta, T. Xu, J. T. Vogelstein,
           "Independence testing for temporal data", Transactions on Machine
           Learning Research (2024).

    Parameters
    ----------
    max_lag : int, optional
        Largest lag ``l``, in samples. Default: 1.
    """

    name = "Cross-multiscale graph correlation"
    identifier = "mgcx"
    labels = ["distance", "temporal", "directed", "nonlinear"]

    def __init__(self, max_lag=1):
        self._max_lag = max_lag
        self.identifier += f"_maxlag-{fmt_param(max_lag)}"

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        # As in CrossDistanceCorrelation: source i goes second.
        z = data.to_numpy()
        self._require_samples(z.shape[1])
        if _local_covariance is None:
            return _lagged_sum(MGC(compute_distance=None), z[j], z[i],
                               self._max_lag)
        stats = [_mgc_from_parts(self._prepare(z[j], lag)[0],
                                 self._prepare(z[i], lag)[1])
                 for lag in range(self._max_lag + 1)]
        return CrossDistanceCorrelation._weighted_sum(stats, z.shape[1])

    @parse_multivariate
    def multivariate(self, data):
        z = data.to_numpy()
        M, n = z.shape[0], z.shape[1]
        self._require_samples(n)
        if _local_covariance is None:
            return super().multivariate(data)
        S = [_cached_pairs(self.identifier, M,
                           lambda s, lag=lag: self._prepare(z[s], lag),
                           lambda src, tgt: _mgc_from_parts(tgt[0], src[1]),
                           item_bytes=32 * (n - lag) ** 2, directed=True)
             for lag in range(self._max_lag + 1)]
        A = np.full((M, M), np.nan)
        for i in range(M):
            for j in range(M):
                if i != j:
                    A[i, j] = CrossDistanceCorrelation._weighted_sum(
                        [s[i, j] for s in S], n)
        return A

    def _require_samples(self, T):
        _require_lagged_samples(T, self._max_lag, _MGC_MIN_SAMPLES,
                                "The cross-multiscale graph correlation")

    @staticmethod
    def _prepare(z, lag):
        """
        `_mgc_parts` of ``z[lag:]`` (as target) and ``z[:n - lag]`` (as
        source), the blocks of the distance matrix `_lagged_sum` crops.
        """
        d = _euclidean_distance(z)
        if lag == 0:
            p = _mgc_parts(d)
            return p, p
        m = d.shape[0] - lag
        return (_mgc_parts(np.ascontiguousarray(d[lag:, lag:])),
                _mgc_parts(np.ascontiguousarray(d[:m, :m])))


class TimeWarping(Undirected, Unsigned):
    """
    Base for the elastic alignment measures (DTW, soft-DTW, LCSS).

    A subclass sets ``_simfn(x, y, global_constraint=...)``. A
    ``global_constraint`` other than None is appended to the identifier as
    ``_constraint-sakoe-chiba`` or ``_constraint-itakura``.
    """

    labels =["distance", "temporal", "undirected", "nonlinear"]

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
    """
    Dynamic time warping (DTW) distance [1]_.

    ``A[i, j] = sqrt(min_pi sum_{(a, b) in pi} (x_i[a] - x_j[b])^2)``, the
    minimum over warping paths ``pi`` (monotone, continuous, matched
    endpoints) of the accumulated squared difference, square-rooted as in
    tslearn and dtaidistance. Undirected and unsigned; zero for identical
    series, and lower means more alike.

    ``global_constraint`` restricts the paths to a band around the diagonal:
    a Sakoe-Chiba band of fixed radius [1]_ or an Itakura parallelogram of
    maximum slope 2 [2]_. Without ``sakoe_chiba_radius`` or
    ``sakoe_chiba_ratio`` the Sakoe-Chiba radius is
    ``min(ceil(0.1 T), ceil(1.5 sqrt(T)))``, at least 10 and at most
    ``T - 1`` (identifier suffix ``_radius-auto``); the environment variables
    ``PYSPI_DTW_SAKOE_LINEAR_FRAC``, ``PYSPI_DTW_SAKOE_SQRT_COEFF`` and
    ``PYSPI_DTW_SAKOE_MIN_RADIUS`` change the 0.1, 1.5 and 10. tslearn's own
    default radius is 1.

    The distance is computed with dtaidistance's C backend when it is
    installed and with tslearn otherwise, with the same band in either. The
    Itakura constraint, which dtaidistance lacks, uses a banded version of
    tslearn's recursion that gives tslearn's value.

    References
    ----------
    .. [1] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization
           for spoken word recognition", IEEE Transactions on Acoustics,
           Speech, and Signal Processing 26(1), 43-49 (1978).
    .. [2] F. Itakura, "Minimum prediction residual principle applied to
           speech recognition", IEEE Transactions on Acoustics, Speech, and
           Signal Processing 23(1), 67-72 (1975).

    Parameters
    ----------
    global_constraint : str or None, optional
        ``None`` (no constraint), ``'sakoe_chiba'`` or ``'itakura'``.
        Default: None.
    sakoe_chiba_radius : int, optional
        Sakoe-Chiba radius in samples (at least 1, capped at ``T - 1``).
        Used only with ``global_constraint='sakoe_chiba'``; set at most one
        of this and ``sakoe_chiba_ratio``. Default: None (the automatic
        radius).
    sakoe_chiba_ratio : float, optional
        Sakoe-Chiba radius as a fraction of ``T``, rounded up (at least 1,
        capped at ``T - 1``). Used only with
        ``global_constraint='sakoe_chiba'``. Default: None.
    normalise : bool, optional
        Divide by ``sqrt(T)`` and append ``_rmse`` to the identifier, the
        same normalisation as `PairwiseDistance`. Default: False.
    """

    name = "Dynamic time warping"
    identifier = "dtw"
    labels = TimeWarping.labels + ["inverted"]

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
        # Validated rather than coerced, so 2.7 or True is rejected instead
        # of silently becoming a radius of 2 or a ratio of 1.0.
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
        """Sakoe-Chiba radius for series of length ``n``."""
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
            d = _itakura_dtw(x, y)
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
            # dtaidistance not installed: use tslearn.
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
        """Every pair, with dtaidistance's ``distance_matrix`` when installed."""
        Z = data.to_numpy(squeeze=True)  # (M, T)
        M = Z.shape[0]
        series = [np.ascontiguousarray(Z[i], dtype=np.double) for i in range(M)]

        constraint = self._global_constraint

        if constraint == "itakura":
            # dtaidistance doesn't support itakura.
            A = np.full((M, M), np.nan)
            for i in range(M):
                for j in range(i + 1, M):
                    d = _itakura_dtw(series[i], series[j])
                    A[i, j] = d
                    A[j, i] = d
            # The same sqrt(T) normalisation as the other paths.
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
            # dtaidistance not installed: tslearn, pair by pair.
            A = np.full((M, M), np.nan)
            for i in range(M):
                for j in range(i + 1, M):
                    d = self.bivariate(data, i=i, j=j)
                    A[i, j] = d
                    A[j, i] = d
            return A


def _banded_lcss(x, y, global_constraint=None, eps=1.0):
    """
    LCSS similarity in which only pairs inside the global band may match.

    The band (tslearn's own Sakoe-Chiba radius-1 or Itakura slope-2 mask)
    restricts which pairs count as matches, not where the recursion may pass
    (Vlachos et al. 2002). tslearn's own banded LCSS is not used: depending
    on the version it ignores the band, or holds every masked cell at zero,
    which for the Itakura parallelogram (a single cell at each end) scores
    any pair whose last samples differ by more than `eps` as exactly 0.

    A row of the LCSS table is non-decreasing, and a match cell's
    `A[i-1, j-1] + 1` is never below its left or upper neighbour, so each row
    is the running maximum of its candidates, computed in one vectorised pass.
    """
    if global_constraint is None:
        return tslearn.metrics.lcss(x, y, eps=eps)
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n, m = len(x), len(y)
    if global_constraint == "sakoe_chiba":
        band = tslearn.metrics.sakoe_chiba_mask(n, m)
    elif global_constraint == "itakura":
        band = _itakura_mask(n, m)
    else:
        raise ValueError(f"Unknown LCSS global constraint: {global_constraint}")
    match = np.asarray(band, dtype=bool) & (np.abs(x[:, None] - y[None, :]) <= eps)
    prev = np.zeros(m + 1)
    for i in range(n):
        cand = np.where(match[i], prev[:-1] + 1, prev[1:])
        prev = np.concatenate(([0.0], np.maximum.accumulate(cand)))
    return prev[-1] / min(n, m)


class LongestCommonSubsequence(TimeWarping):
    """
    Longest common subsequence (LCSS) similarity [1]_.

    Two samples match when ``|x_i[a] - x_j[b]| <= 1`` (tslearn's ``eps``,
    in the units of the data, so one standard deviation after z-scoring).
    The length of the longest subsequence of matching pairs, monotone in both
    series, is divided by ``min(T_i, T_j)``, so the value is in ``[0, 1]``,
    with 1 for identical series. Undirected; a similarity, so larger means
    more alike.

    Under a ``global_constraint`` only pairs inside the band (tslearn's
    Sakoe-Chiba radius 1 or Itakura slope 2) may match, as in [1]_; this is
    computed by pyspi, since tslearn's own banded LCSS does not apply the
    band this way. Without a constraint the value is tslearn's ``lcss``.

    References
    ----------
    .. [1] M. Vlachos, G. Kollios, D. Gunopulos, "Discovering similar
           multidimensional trajectories", Proceedings of the 18th
           International Conference on Data Engineering, 673-684 (2002).

    Parameters
    ----------
    global_constraint : str or None, optional
        ``None`` (no constraint), ``'sakoe_chiba'`` or ``'itakura'``.
        Default: None.
    """

    name = "Longest common subsequence"
    identifier = "lcss"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._simfn = _banded_lcss


class SoftDynamicTimeWarping(TimeWarping):
    """
    Soft dynamic time warping (soft-DTW) [1]_.

    tslearn's ``soft_dtw`` at its default ``gamma = 1``: the DTW recursion
    on squared differences with the minimum replaced by the soft minimum
    ``-gamma log sum exp(-a / gamma)``, so every warping path contributes.
    Undirected and unsigned: a dissimilarity, so lower means more alike.

    This is the raw soft-minimum alignment cost, which is not a distance: its
    value can be negative (it is for most pairs of z-scored series), and
    ``softdtw(x, x)`` is not 0. It is not the
    non-negative soft-DTW divergence ``D(x, y) - (D(x, x) + D(y, y)) / 2``
    of [2]_.

    A ``global_constraint`` uses tslearn's defaults: Sakoe-Chiba radius 1,
    Itakura maximum slope 2.

    References
    ----------
    .. [1] M. Cuturi, M. Blondel, "Soft-DTW: a differentiable loss function
           for time-series", Proceedings of the 34th International
           Conference on Machine Learning, PMLR 70, 894-903 (2017).
    .. [2] M. Blondel, A. Mensch, J.-P. Vert, "Differentiable divergences
           between time series", Proceedings of the 24th International
           Conference on Artificial Intelligence and Statistics, PMLR 130,
           3853-3861 (2021).

    Parameters
    ----------
    global_constraint : str or None, optional
        ``None`` (no constraint), ``'sakoe_chiba'`` or ``'itakura'``.
        Default: None.
    """

    name = "Soft dynamic time warping"
    identifier = "softdtw"
    labels = TimeWarping.labels + ["inverted"]

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        z =data.to_numpy(squeeze=True)
        return tslearn.metrics.soft_dtw(
            z[i], z[j], global_constraint=self._global_constraint)


class Barycenter(Undirected, Signed):
    """
    A statistic of the barycenter (average series) of the pair [1]_ [2]_ [3]_.

    The barycenter ``b`` of ``{x_i, x_j}`` is computed with tslearn under one
    of four geometries: Euclidean (``(x_i + x_j) / 2``), DTW (DTW barycenter
    averaging, DBA [1]_), subgradient DBA [2]_ or soft-DTW [3]_. It is a
    function of the unordered pair, so the measure is undirected.

    ``statistic='corr'`` is the alignment-adjusted correlation
    ``2 mean(b^2) / e - 1``, where ``e`` is the average of ``mean(x_i^2)``
    and ``mean(x_j^2)``. For the Euclidean barycenter of two zero-mean series
    this is ``2 cov / (var_i + var_j)``, which is Pearson's r when the two
    variances are equal, as after pyspi's default z-scoring, and lies in
    ``[-1, 1]``. For the DTW geometries it is the share of the pair's
    amplitude that survives averaging after optimal alignment (soft-DTW
    smooths the barycenter, so identical series give less than 1). It is at
    least -1 but not bounded above by 1: a warping path can repeat
    high-amplitude samples, so ``mean(b^2)`` can exceed ``e``, as for series
    with sparse large spikes. The barycenter of two zero-mean series has
    mean near 0, so ``corr`` is more informative than ``mean``.

    ``'mean'`` and ``'max'`` are the mean and maximum of ``b`` (of ``b^2``
    with ``squared=True``), and ``'max_time'`` is the index of the maximum.
    Since ``rms(b) = sqrt(e (1 + corr) / 2)``, with ``e`` close to 1 for
    z-scored inputs, ``max`` is mostly ``corr`` scaled by the barycenter's
    crest factor ``max(b) / rms(b)``. Squared barycenters and
    ``max_time`` are unsigned; the other statistics are signed.

    tslearn's DBA starts from the Euclidean mean and stays near it when the
    pair is out of phase, so under ``mode='dtw'`` a time-shifted copy of a
    smooth series can score as low as an independent pair. The subgradient
    fit uses a fixed seed (42), and each pair is fitted once, in a fixed
    order, and shared by every statistic of the same mode.

    References
    ----------
    .. [1] F. Petitjean, A. Ketterlin, P. Gançarski, "A global averaging
           method for dynamic time warping, with applications to
           clustering", Pattern Recognition 44(3), 678-693 (2011).
    .. [2] D. Schultz, B. J. Jain, "Nonsmooth analysis and subgradient methods
           for averaging in dynamic time warping spaces", Pattern Recognition
           74, 340-358 (2018).
    .. [3] M. Cuturi, M. Blondel, "Soft-DTW: a differentiable loss function
           for time-series", Proceedings of the 34th International
           Conference on Machine Learning, PMLR 70, 894-903 (2017).

    Parameters
    ----------
    mode : str, optional
        Barycenter geometry: ``'euclidean'``, ``'dtw'``, ``'sgddtw'``
        (subgradient DBA) or ``'softdtw'``. Default: ``'euclidean'``.
    squared : bool, optional
        Square the barycenter before ``'mean'``, ``'max'`` or ``'max_time'``,
        and append ``-sq`` to the identifier. Not allowed with ``'corr'``.
        Default: False.
    statistic : str, optional
        ``'mean'``, ``'max'``, ``'max_time'`` or ``'corr'``.
        Default: ``'mean'``.
    """

    name = "Barycenter"
    identifier = "bary"
    labels = ["distance", "signed", "undirected", "temporal", "nonlinear"]
    _cache_namespace = "barycenter"

    @property
    def _cache_subkey(self):
        # The cache is keyed by (mode, pair); statistic and squared act after
        # the lookup. Grouping by mode keeps the cheap modes' costs separate.
        return (self._mode,)

    def __init__(self, mode="euclidean", squared=False, statistic="mean"):
        if mode == "euclidean":
            self._fn = euclidean_barycenter
        elif mode == "dtw":
            self._fn = dtw_barycenter_averaging
        elif mode == "sgddtw":
            # A fixed seed per fit: tslearn's default draws from NumPy's
            # global stream, which would make the result depend on run order.
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
            # 2 E[b^2] / e - 1, with the pair's mean energy e from `bivariate`.
            self._statfn = lambda x, e: 2.0 * np.nanmean(x ** 2) / e - 1.0
        elif statistic == "mean":
            self._statfn = lambda x: np.nanmean(self._preproc(x))
        elif statistic == "max":
            self._statfn = lambda x: np.nanmax(self._preproc(x))
        elif statistic == "max_time":
            self._statfn = lambda x: np.argmax(self._preproc(x))
        else:
            raise NameError(f"Unknown statistic: {statistic}")

        # A squared barycenter and an argmax index are non-negative.
        if squared or statistic == "max_time":
            self.labels = [l for l in self.labels if l != "signed"]
            self.issigned = lambda: False
        # The time index of the maximum is an offset, not a strength.
        if statistic == "max_time":
            self.labels = self.labels + ["timing"]

        self.identifier += f"_{mode}_{statistic}"

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        # One barycenter per unordered pair, always fitted in the same order
        # because the subgradient (sgddtw) fit depends on input order.
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


class GromovWassersteinTau(Undirected, Unsigned):
    """
    Gromov-Wasserstein distance GW_tau between two time series [1]_.

    Each series becomes the planar trajectory ``(t, x_t)``; ``vec_geo_dist``
    is the vector of intrinsic (arc-length) distances of every point from the
    trajectory's start, and ``wass_sorted`` the closed-form 2-Wasserstein
    distance between the two sorted vectors (Proposition 1(e) of [1]_:
    ``sqrt(mean((sorted v_i - sorted v_j)^2))`` at equal lengths).
    Undirected and unsigned; zero for identical series, and lower means more
    alike.

    The distance vector has one entry per point, the first being 0 for the
    start itself (the paper's ``d_X(r_X, x_k)`` for ``k = 1..n``). The
    authors' MATLAB code omits it, which only rescales the value by a
    constant.

    The arc lengths are cumulative sums of the steps
    ``sqrt(1 + (x_{t+1} - x_t)^2)``, so GW_tau depends on each series only
    through its sequence of absolute increments (it is 0 for ``x_j = -x_i``).
    For z-scored series the mean squared increment is about ``2 (1 - r_1)``,
    with ``r_1`` the lag-1 autocorrelation, so the value largely reflects the
    difference between the two series' lag-1 autocorrelations.

    References
    ----------
    .. [1] N. Kravtsova, R. L. McGee II, A. T. Dawes, "Scalable
           Gromov-Wasserstein based comparison of biological time series",
           Bulletin of Mathematical Biology 85(8), 77 (2023).
    """

    name = "Gromov-Wasserstein distance (GW_tau)"
    identifier = "gwtau"
    labels = ["distance", "unordered", "nonlinear", "undirected", "inverted"]

    @staticmethod
    def vec_geo_dist(x):
        diffs = np.diff(x, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        # The start is at distance 0 from itself: one entry per point.
        return np.concatenate([[0.0], np.cumsum(distances)])

    @staticmethod
    def wass_sorted(x1, x2):
        x1 = np.sort(x1)[::-1]  # descending
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

        vi = GromovWassersteinTau.vec_geo_dist(traji)
        vj = GromovWassersteinTau.vec_geo_dist(trajj)
        gw = GromovWassersteinTau.wass_sorted(vi, vj)

        return gw

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        x, y = data.to_numpy()[[i, j]]
        stat = self.gwtau(x, y)
        return stat


class GromovWasserstainTau(GromovWassersteinTau):
    """
    Deprecated, misspelled alias of `GromovWassersteinTau`.

    The identifier (``gwtau``) and the values are the same; it warns on use
    and will be removed in pyspi 4.0.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "GromovWasserstainTau is deprecated and will be removed in pyspi "
            "4.0; use GromovWassersteinTau. The identifier and the computed "
            "values are identical.",
            DeprecationWarning, stacklevel=2,
        )
        super().__init__(*args, **kwargs)


# ---------------------------------------------------------------------------
# CrossPairwiseDistance: lagged Euclidean distance
# ---------------------------------------------------------------------------

class CrossPairwiseDistance(Undirected, Unsigned):
    """
    Smallest (or mean) root-T-normalised distance over lag shifts up to ``tau``.

    At each lag ``t`` the two series are paired with an offset of ``t``, the
    ``t`` samples that fall off each end are repeated against the opposite
    series' boundary sample, and the cost is ``sqrt(sum(differences^2) / T)``.
    The smaller of the forward and backward shifts is kept, so the value is
    symmetric in the pair, and ``statistic`` then takes the minimum or the
    mean over ``t = 0..tau`` (lags of ``T`` or more are skipped by the mean).
    Undirected and unsigned; a distance, so lower means more alike.

    Each shifted pairing is a valid DTW warping path, so the value is never
    below the unconstrained DTW distance with ``normalise=True``. At
    ``tau = 0`` it equals `PairwiseDistance` with the Euclidean metric and
    ``normalise=True``. The ``_rmse`` in the identifier names the division
    by ``sqrt(T)``, as there; the path has ``T + t`` steps, so it is not a
    per-step root mean square.

    Parameters
    ----------
    metric : str, optional
        Only ``'euclidean'`` is supported. Default: ``'euclidean'``.
    tau : int, optional
        Largest lag, in samples (at least 0). Default: 1.
    statistic : str, optional
        ``'min'`` or ``'mean'`` over lags (case-insensitive).
        Default: ``'min'``.
    """
    name = "Cross pairwise distance"
    labels = ["distance", "nonlinear", "undirected", "temporal", "inverted"]

    def __init__(self, metric="euclidean", tau=1, statistic="min"):
        if metric != "euclidean":
            raise ValueError(f"Unsupported metric: {metric!r}. Only 'euclidean' supported.")
        # Validated rather than coerced, so 1.7 or True is rejected.
        tau = require_int("tau", tau, minimum=0)
        stat = str(statistic).lower()
        if stat not in {"min", "mean"}:
            raise ValueError(f"statistic must be 'min' or 'mean', got: {statistic!r}")
        self._metric = metric
        self._tau = tau
        self._statistic = stat
        # `_rmse` names the sqrt(T) normalisation, as in PairwiseDistance.
        self.identifier = f"xpdist_{metric}_tau-{self._tau}_{stat}_rmse"

    @staticmethod
    def _shifted_dist_rmse(x, y, s):
        # One DTW path: x[k + s] against y[k] over the overlap, and the dropped
        # boundary samples against the opposite series' end sample.
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
