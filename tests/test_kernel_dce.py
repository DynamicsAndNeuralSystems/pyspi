"""Dynamic correlation exclusion (Theiler window) for the kernel estimators.

JIDT parity is pinned in `test_jidt_reference.py`; this module checks the
contract around it: no window and a window of 0 are one estimator, the
identifier and validation, an independent brute-force reference of the window
``|j - i| <= w`` (the kraskov window), and the effect on spurious dependence
between autocorrelated series.
"""

import pickle

import numpy as np
import pytest
from scipy.spatial import cKDTree

from pyspi.base import NotAppropriate
from pyspi.data import Data
from pyspi.statistics.infotheory import (
    ConditionalEntropy,
    DirectedInfo,
    JointEntropy,
    KernelMICalculator,
    MutualInfo,
    TimeLaggedMutualInfo,
    TransferEntropy,
)

KERNEL_CLASSES = [
    (MutualInfo, {}),
    (TimeLaggedMutualInfo, {}),
    (TransferEntropy, {"k_history": 1}),
]


def _ar1(rng, T, phi):
    x = np.zeros(T)
    e = rng.standard_normal(T)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + e[t]
    return x


@pytest.fixture(scope="module")
def ar_data():
    """Three independent AR(1) processes with phi = 0.95."""
    rng = np.random.default_rng(0)
    Z = np.vstack([_ar1(rng, 500, 0.95) for _ in range(3)])
    return Data(Z, dim_order="ps", zscore=True)


@pytest.mark.parametrize("cls, kw", KERNEL_CLASSES)
def test_no_window_is_the_existing_estimator(ar_data, cls, kw):
    """dyn_corr_excl None and 0 give the same identifier and identical values:
    as for kraskov, the point itself is excluded either way."""
    base = cls(estimator="kernel", kernel_width=0.25, **kw)
    zero = cls(estimator="kernel", kernel_width=0.25, dyn_corr_excl=0, **kw)
    assert zero.identifier == base.identifier
    assert np.array_equal(base.multivariate(ar_data), zero.multivariate(ar_data),
                          equal_nan=True)


@pytest.mark.parametrize("cls, kw", KERNEL_CLASSES)
def test_identifier_carries_the_window(cls, kw):
    stem = cls(estimator="kernel", kernel_width=0.25, **kw).identifier
    for v in (1, 5, "AUTO"):
        spi = cls(estimator="kernel", kernel_width=0.25, dyn_corr_excl=v, **kw)
        assert f"_W-0.25_DCE-{v}" in spi.identifier
        assert spi.identifier.replace(f"_DCE-{v}", "") == stem
        assert spi._getkey() == ("kernel", 0.25, v)


def test_bundled_identifiers():
    assert (MutualInfo(estimator="kernel", kernel_width=0.25,
                       dyn_corr_excl="AUTO").identifier
            == "mi_kernel_W-0.25_DCE-AUTO")
    assert (TimeLaggedMutualInfo(estimator="kernel", kernel_width=0.25,
                                 dyn_corr_excl="AUTO").identifier
            == "tlmi_kernel_W-0.25_DCE-AUTO")
    assert (TransferEntropy(estimator="kernel", kernel_width=0.25, k_history=1,
                            dyn_corr_excl="AUTO").identifier
            == "te_kernel_W-0.25_DCE-AUTO_k-1")


@pytest.mark.parametrize("bad", [-1, "auto", "10", 2.5])
def test_window_is_validated(bad):
    with pytest.raises((ValueError, TypeError)):
        MutualInfo(estimator="kernel", dyn_corr_excl=bad)


@pytest.mark.parametrize("cls", [JointEntropy, ConditionalEntropy, DirectedInfo])
def test_composed_kernel_measures_refuse_the_window(cls):
    """They are built from marginal entropies and would ignore it."""
    with pytest.raises(ValueError, match="dyn_corr_excl"):
        cls(estimator="kernel", dyn_corr_excl=5)


def test_other_estimators_still_refuse_the_window():
    with pytest.raises(ValueError, match="not used by estimator"):
        MutualInfo(estimator="gaussian", dyn_corr_excl=5)


def _brute_force_kernel_mi(x, y, width, w):
    """Box-kernel MI with Theiler window w, in O(N^2).

    Point i is compared with every t with |t - i| > w, each probability is a
    count over the number of points compared, and an empty joint box
    contributes 0, as in JIDT's KernelEstimatorMultiVariate.
    """
    x = x / np.std(x, ddof=1)
    y = y / np.std(y, ddof=1)
    N = len(x)
    total = 0.0
    for i in range(N):
        t = np.arange(N)
        keep = np.abs(t - i) > w
        nx = np.sum(keep & (np.abs(x - x[i]) <= width))
        ny = np.sum(keep & (np.abs(y - y[i]) <= width))
        nxy = np.sum(keep & (np.abs(x - x[i]) <= width) & (np.abs(y - y[i]) <= width))
        if nxy > 0:
            total += np.log(nxy * keep.sum() / (nx * ny))
    return total / N


@pytest.mark.parametrize("w", [0, 1, 3, 10, 19])
def test_kernel_mi_matches_a_brute_force_reference(w):
    """No window (the point itself excluded) and windows up to N / 2, where
    the edge pools differ most."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal(40)
    y = 0.6 * x + rng.standard_normal(40)
    data = Data(np.vstack([x, y]), dim_order="ps", zscore=False)
    spi = MutualInfo(estimator="kernel", kernel_width=0.5, dyn_corr_excl=w or None)
    assert spi.bivariate(data, i=0, j=1) == pytest.approx(
        _brute_force_kernel_mi(x, y, 0.5, w), abs=1e-12)


def test_window_too_wide_for_the_series_is_refused():
    rng = np.random.default_rng(0)
    data = Data(rng.standard_normal((2, 40)), dim_order="ps", zscore=False)
    MutualInfo(estimator="kernel", dyn_corr_excl=19).bivariate(data, i=0, j=1)
    with pytest.raises(NotAppropriate, match="leaves no comparison points"):
        MutualInfo(estimator="kernel", dyn_corr_excl=20).bivariate(data, i=0, j=1)


@pytest.mark.parametrize("cls, kw", KERNEL_CLASSES)
def test_window_lowers_spurious_dependence_between_autocorrelated_series(
        ar_data, cls, kw):
    """Independent AR(1) pairs: the true MI, TLMI and TE are all 0.

    With only the point itself excluded, temporally adjacent samples still
    inflate the estimates; a window of 50 lowers every MI and TLMI estimate.
    That is not asserted for TE, whose conditioning on the target past
    already absorbs most of it.
    """
    off = ~np.eye(3, dtype=bool)
    none = cls(estimator="kernel", kernel_width=0.25, **kw).multivariate(ar_data)[off]
    wide = cls(estimator="kernel", kernel_width=0.25, dyn_corr_excl=50,
               **kw).multivariate(ar_data)[off]
    assert np.all(np.isfinite(wide))
    if cls is not TransferEntropy:
        assert np.all(wide < none)


@pytest.mark.parametrize("cls, kw", KERNEL_CLASSES)
def test_auto_window_is_per_pair_and_survives_pickling(ar_data, cls, kw):
    """Each pair gets its own AUTO window, also on an unpickled instance."""
    spi = cls(estimator="kernel", kernel_width=0.25, dyn_corr_excl="AUTO", **kw)
    restored = pickle.loads(pickle.dumps(spi))
    A = np.asarray(spi.multivariate(ar_data))
    B = np.asarray(restored.multivariate(ar_data))
    assert np.array_equal(A, B, equal_nan=True)
    for i, j in [(0, 1), (2, 1), (1, 2)]:
        fresh = cls(estimator="kernel", kernel_width=0.25,
                    dyn_corr_excl=int(ar_data.theiler[i, j]), **kw)
        # MI is filled once per unordered pair and mirrored, so (2, 1) is
        # the (1, 2) computation: equal up to round-off.
        assert fresh.bivariate(ar_data, i=i, j=j) == pytest.approx(A[i, j],
                                                                   abs=1e-12)


@pytest.mark.parametrize("cls, kw", KERNEL_CLASSES)
def test_auto_window_of_zero_is_no_window(cls, kw):
    """Both exclude the point itself only, as for kraskov."""
    rng = np.random.default_rng(1)
    data = Data(np.cumsum(rng.standard_normal((2, 120)), axis=1), dim_order="ps")
    data.theiler = np.zeros((2, 2))
    auto = cls(estimator="kernel", kernel_width=0.5, dyn_corr_excl="AUTO", **kw)
    none = cls(estimator="kernel", kernel_width=0.5, **kw)
    assert auto.bivariate(data, i=0, j=1) == none.bivariate(data, i=0, j=1)


@pytest.mark.parametrize("w", [None, 1])
@pytest.mark.parametrize("cls, kw", KERNEL_CLASSES)
def test_no_neighbours_anywhere_is_not_appropriate(cls, kw, w):
    """Every joint box is empty once the point itself is excluded, with or
    without a window, so there is no estimate; 0 would read as independence."""
    x = np.vstack([np.arange(12.0), np.arange(12.0)[::-1] ** 2])
    data = Data(x, dim_order="ps", zscore=False)
    spi = cls(estimator="kernel", kernel_width=0.01, dyn_corr_excl=w, **kw)
    with pytest.raises(NotAppropriate, match="every joint count is zero"):
        spi.bivariate(data, i=0, j=1)


@pytest.mark.parametrize("cls, kw", KERNEL_CLASSES)
def test_negative_resolved_window_is_refused(ar_data, cls, kw):
    """As for kraskov; it must not fall back to the estimator without a window."""
    data = Data(ar_data.to_numpy(squeeze=True), dim_order="ps", zscore=False)
    data.theiler = -np.ones((3, 3))
    spi = cls(estimator="kernel", kernel_width=0.25, dyn_corr_excl="AUTO", **kw)
    with pytest.raises(ValueError, match="Theiler window must be >= 0"):
        spi.bivariate(data, i=0, j=1)


def test_auto_window_is_at_least_one_from_ten_samples():
    """2 * <acf_i, acf_j> >= 1 for any two series, so only the N // 10 cap
    brings the AUTO window to 0."""
    rng = np.random.default_rng(3)
    spi = MutualInfo(estimator="kernel", kernel_width=0.5, dyn_corr_excl="AUTO")
    for T in (10, 25, 200):
        t = np.arange(T)
        Z = np.vstack([rng.standard_normal(T), (-1.0) ** t + 0.1 * rng.standard_normal(T),
                       np.cumsum(rng.standard_normal(T))])
        data = Data(Z, dim_order="ps")
        assert min(spi._resolve_theiler(data, i, j)
                   for i in range(3) for j in range(3) if i != j) >= 1
    data = Data(rng.standard_normal((2, 9)), dim_order="ps")
    assert spi._resolve_theiler(data, 0, 1) == 0
