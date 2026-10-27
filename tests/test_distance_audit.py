"""Independent references for the SPIs in `pyspi/statistics/distance.py`.

Each test writes the statistic out from its published definition, without the
backend the SPI calls, so a backend that ignores a parameter or computes
something else under the same name is caught here rather than frozen into a
baseline.
"""
import numpy as np
import pytest
from scipy.special import logsumexp

from pyspi.data import Data


# --------------------------------------------------------------------------
# Soft-DTW
# --------------------------------------------------------------------------

def _soft_dtw_cell_by_cell(x, y, band, gamma=1.0):
    """Cuturi & Blondel (2017), Algorithm 1, over the alignments inside `band`.

    A cell outside the band is not on any admissible path: it stays at +inf
    and contributes exp(-inf) = 0 to every soft-min it enters.
    """
    n, m = len(x), len(y)
    R = np.full((n + 1, m + 1), np.inf)
    R[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if not band[i - 1, j - 1]:
                continue
            prev = np.array([R[i - 1, j - 1], R[i - 1, j], R[i, j - 1]])
            softmin = -gamma * logsumexp(-prev / gamma)
            R[i, j] = (x[i - 1] - y[j - 1]) ** 2 + softmin
    return R[n, m]


@pytest.mark.parametrize("constraint", [None, "sakoe_chiba", "itakura"])
def test_soft_dtw_applies_its_global_constraint(constraint):
    """`softdtw_constraint-*` must sum over the banded alignments only.

    The constraint used to be accepted, written into the identifier, and then
    never passed to tslearn, so both constrained SPIs equalled `softdtw`.
    The bands are tslearn's own masks at its defaults (Sakoe-Chiba radius 1,
    Itakura slope 2), as for `lcss_constraint-*`.
    """
    import tslearn.metrics
    from pyspi.statistics.distance import SoftDynamicTimeWarping

    rng = np.random.default_rng(3)
    T = 40
    data = Data(data=rng.standard_normal((2, T)), dim_order="ps")
    x, y = data.to_numpy(squeeze=True)
    if constraint is None:
        band = np.ones((T, T), dtype=bool)
    elif constraint == "sakoe_chiba":
        band = np.asarray(tslearn.metrics.sakoe_chiba_mask(T, T), dtype=bool)
    else:
        band = np.asarray(tslearn.metrics.itakura_mask(T, T), dtype=bool)

    value = SoftDynamicTimeWarping(global_constraint=constraint).bivariate(
        data, i=0, j=1)
    assert value == pytest.approx(_soft_dtw_cell_by_cell(x, y, band), rel=1e-10)
    if constraint is not None:
        # Restricting the alignments can only raise the soft minimum.
        plain = SoftDynamicTimeWarping().bivariate(data, i=0, j=1)
        assert value > plain + 1.0


# --------------------------------------------------------------------------
# Cross-distance correlation and cross-MGC (hyppo DcorrX / MGCX)
# --------------------------------------------------------------------------

def _driven_pair(T=300, seed=0):
    """Process 0 drives process 1 at lag 1, with no feedback."""
    rng = np.random.default_rng(seed)
    x = np.zeros(T)
    y = np.zeros(T)
    ex, ey = rng.standard_normal(T), rng.standard_normal(T)
    for t in range(1, T):
        x[t] = 0.5 * x[t - 1] + ex[t]
        y[t] = 0.5 * y[t - 1] + 0.8 * x[t - 1] + ey[t]
    return Data(np.vstack([x, y]))


@pytest.mark.parametrize("cls_name", ["CrossDistanceCorrelation",
                                      "CrossMultiscaleGraphCorrelation"])
@pytest.mark.parametrize("max_lag", [1, 10])
def test_cross_distance_statistics_are_source_by_target(cls_name, max_lag):
    """A[0, 1] (0 -> 1, the true direction) must exceed A[1, 0].

    hyppo's `DcorrX(x, y)` pairs the present of `x` with the past of `y`, so
    it measures y -> x; pyspi's convention (`base.Directed`) is row = source,
    column = target.
    """
    import pyspi.statistics.distance as distance

    spi = getattr(distance, cls_name)(max_lag=max_lag)
    A = np.asarray(spi.multivariate(_driven_pair()), dtype=float)
    assert A[0, 1] > A[1, 0] + 0.1, (
        f"{spi.identifier}: A[0,1]={A[0, 1]:.4f}, A[1,0]={A[1, 0]:.4f}")


def _unbiased_dcorr(a, b):
    """U-centred distance correlation (Szekely & Rizzo 2014), as `dcorr`."""
    def u_centred(v):
        n = len(v)
        d = np.abs(v[:, None] - v[None, :])
        u = (d - d.sum(0) / (n - 2) - d.sum(1)[:, None] / (n - 2)
             + d.sum() / ((n - 1) * (n - 2)))
        np.fill_diagonal(u, 0)
        return u
    A, B = u_centred(a), u_centred(b)
    return (A * B).sum() / np.sqrt((A * A).sum() * (B * B).sum())


def _mgc(a, b):
    import warnings
    from scipy.stats import multiscale_graphcorr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return multiscale_graphcorr(a, b, reps=0).statistic


@pytest.mark.parametrize("cls_name,stat", [
    ("CrossDistanceCorrelation", _unbiased_dcorr),
    ("CrossMultiscaleGraphCorrelation", _mgc),
])
@pytest.mark.parametrize("max_lag", [1, 3])
def test_cross_distance_statistics_sum_the_lagged_statistic(cls_name, stat, max_lag):
    """Mehta et al. (2020), as hyppo's own docstring states it:
    sum_{l=0}^{L} (n - l)/n * S(target[l:], source[:n - l]).

    hyppo 0.5.2 computes the l >= 1 terms by handing the sliced *distance
    matrices* to an S that was built to compute Euclidean distances, so each
    row of a distance matrix is treated as a point in R^(n-l) and S measures
    the dependence between distance profiles, not between the lagged series.
    Only the l = 0 term was right.
    """
    import pyspi.statistics.distance as distance

    rng = np.random.default_rng(11)
    n = 60
    src = rng.standard_normal(n)
    tgt = np.r_[0.0, np.tanh(2 * src[:-1])] + 0.5 * rng.standard_normal(n)
    data = Data(np.vstack([src, tgt]), dim_order="ps")
    z = data.to_numpy(squeeze=True)

    A = getattr(distance, cls_name)(max_lag=max_lag).multivariate(data)
    for i, j in [(0, 1), (1, 0)]:
        x, y = z[j], z[i]  # target, source
        ref = sum((n - l) / n * stat(x[l:], y[:n - l]) for l in range(max_lag + 1))
        assert A[i, j] == pytest.approx(ref, abs=1e-12), (i, j)


@pytest.mark.parametrize("cross,plain", [
    ("CrossDistanceCorrelation", "DistanceCorrelation"),
    ("CrossMultiscaleGraphCorrelation", "MultiscaleGraphCorrelation"),
])
def test_cross_distance_statistics_at_lag_zero_are_the_plain_statistic(cross, plain):
    import pyspi.statistics.distance as distance

    data = Data(np.random.default_rng(2).standard_normal((3, 50)), dim_order="ps")
    a = getattr(distance, cross)(max_lag=0).multivariate(data)
    b = getattr(distance, plain)().multivariate(data)
    # Not bit-identical: the cross statistic is evaluated as S(target, source)
    # and the plain one as S(i, j); MGC is symmetric only to rounding.
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-15)


# --------------------------------------------------------------------------
# HSIC
# --------------------------------------------------------------------------

def _normalised_hsic(a, b, biased):
    """Normalised HSIC with a Gaussian median-heuristic kernel *per variable*.

    Gretton et al. (2005, 2008) set each variable's kernel width to the median
    distance between that variable's samples. hyppo's `Hsic` statistic is
    the kernel analogue of its `Dcorr`: dcorr of the induced distances
    1 - K / max(K), so U-centred and unsquared when unbiased, double-centred
    and square-rooted when biased.
    """
    n = len(a)

    def induced_distance(v):
        d = np.abs(v[:, None] - v[None, :])
        med = np.median(d[~np.eye(n, dtype=bool)])
        K = np.exp(-(d ** 2) / (2 * med ** 2))
        return 1 - K / K.max()

    def centred(d):
        if biased:
            return d - d.mean(0) - d.mean(1)[:, None] + d.mean()
        u = (d - d.sum(0) / (n - 2) - d.sum(1)[:, None] / (n - 2)
             + d.sum() / ((n - 1) * (n - 2)))
        np.fill_diagonal(u, 0)
        return u

    A, B = centred(induced_distance(a)), centred(induced_distance(b))
    ratio = (A * B).sum() / np.sqrt((A * A).sum() * (B * B).sum())
    return np.sqrt(ratio) if biased else ratio


@pytest.mark.parametrize("biased", [False, True])
def test_hsic_uses_each_variables_own_kernel_width(biased):
    """hyppo 0.5.2 takes the median-heuristic width from `x` alone and uses it
    for `y` too, so hsic(x, y) != hsic(y, x) and the value pyspi reported for
    a pair depended on which process came first. Gaussian, exponential and
    uniform marginals keep different median distances after z-scoring."""
    from pyspi.statistics.distance import HilbertSchmidtIndependenceCriterion

    rng = np.random.default_rng(4)
    n = 80
    g = rng.standard_normal(n)
    series = np.stack([g, np.exp(g + 0.5 * rng.standard_normal(n)),
                       rng.uniform(size=n) + 0.3 * g])
    data = Data(series, dim_order="ps")
    z = data.to_numpy(squeeze=True)
    spi = HilbertSchmidtIndependenceCriterion(biased=biased)
    for i in range(3):
        for j in range(3):
            if i != j:
                assert spi.bivariate(data, i=i, j=j) == pytest.approx(
                    _normalised_hsic(z[i], z[j], biased), abs=1e-12), (i, j)


@pytest.mark.parametrize("biased", [False, True])
def test_hsic_of_a_rescaled_copy_is_one(biased):
    """With a width per variable the Gaussian kernel matrix is scale-free, so a
    series and a rescaled copy have identical kernels and HSIC exactly 1."""
    from pyspi.statistics.distance import HilbertSchmidtIndependenceCriterion

    x = np.random.default_rng(6).standard_normal(60)
    data = Data(np.stack([x, 3 * x]), dim_order="ps", zscore=False)
    A = HilbertSchmidtIndependenceCriterion(biased=biased).multivariate(data)
    assert A[0, 1] == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------
# Barycenter
# --------------------------------------------------------------------------

@pytest.mark.parametrize("statistic", ["mean", "max"])
def test_barycenter_value_does_not_depend_on_call_order(statistic):
    """One barycenter per unordered pair, whichever orientation is asked first.

    tslearn's subgradient DBA visits the series in a random order drawn from
    a fixed seed, so the barycenter of [x, y] differs from that of [y, x].
    The cache stored whichever was computed first under both orientations,
    so `bivariate(i=0, j=1)` on fresh data disagreed with
    `multivariate()[0, 1]`, which reaches the pair as (1, 0).
    """
    from pyspi.statistics.distance import Barycenter

    z = np.random.default_rng(8).standard_normal((3, 40))
    spi = Barycenter(mode="sgddtw", statistic=statistic)
    A = spi.multivariate(Data(z, dim_order="ps"))
    assert np.array_equal(A, A.T, equal_nan=True)
    for i, j in [(0, 1), (1, 0), (0, 2), (2, 1)]:
        fresh = Data(z, dim_order="ps")
        assert spi.bivariate(fresh, i=i, j=j) == A[i, j], (i, j)


# --------------------------------------------------------------------------
# HHG
# --------------------------------------------------------------------------

@pytest.mark.parametrize("chunk_rows", [None, 7])
def test_hhg_statistic_is_bit_identical_to_hyppo(monkeypatch, chunk_rows):
    """The O(n^2 log n) counts reproduce hyppo's O(n^3) scan exactly,
    including tied distances, a constant series and row-chunking."""
    import warnings
    import pyspi.statistics.distance as distance
    from hyppo.independence import HHG

    rng = np.random.default_rng(12)
    for n in [5, 6, 17, 64, 121]:
        if chunk_rows is not None:
            monkeypatch.setattr(distance, "_HHG_CHUNK_ENTRIES", chunk_rows * n)
        x = rng.standard_normal((n, 1))
        y = np.tanh(x) + rng.standard_normal((n, 1))
        cases = {
            "continuous": (x, y),
            "tied": (np.round(2 * x) / 2, np.round(y)),
            "binary": (np.round(x), (rng.uniform(size=(n, 1)) > 0.5) * 1.0),
            "constant": (np.ones((n, 1)), y),
        }
        for name, (a, b) in cases.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ref = HHG().statistic(a, b)
            assert distance._hhg_statistic(a, b) == ref, (n, name)
