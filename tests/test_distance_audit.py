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
