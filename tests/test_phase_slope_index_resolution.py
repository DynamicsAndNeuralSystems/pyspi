"""The multitaper phase slope index at a frequency resolution fixed in units of fs.

Its groups of frequencies keep a fixed width as the series lengthens, so each
group averages more spectral estimates: a lag gets easier to detect on a
longer series, the null spread falls, and short series (groups one multitaper
bandwidth wide) behave as described in the class docstring.
"""
import numpy as np
import pytest

from pyspi.base import NotAppropriate
from pyspi.data import Data
from pyspi.statistics.spectral import PhaseSlopeIndex

BANDS = ((0, 0.5), (0, 0.25), (0.25, 0.5))


def _system(T, lag, seed):
    """x is AR(1) and drives y at ``lag`` samples; z is independent."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((3, T + 50 + lag))
    x = np.zeros(T + 50 + lag)
    for t in range(1, x.size):
        x[t] = 0.5 * x[t - 1] + e[0, t]
    y = np.zeros_like(x)
    y[lag:] = 0.8 * x[:-lag] + 0.5 * e[1, lag:]
    return np.vstack([x, y, e[2]])[:, 50 + lag:]


def _signal_and_null(T, lag, band, reps):
    values = np.array([
        PhaseSlopeIndex(fmin=band[0], fmax=band[1]).multivariate(
            Data(_system(T, lag, seed), dim_order="ps"))[0, 1:]
        for seed in range(reps)])
    return values[:, 0], values[:, 1]


@pytest.mark.parametrize("band", BANDS)
@pytest.mark.parametrize("lag", [2, 10])
def test_a_longer_series_makes_a_lag_easier_to_detect(lag, band):
    """Signal-to-null ratio (mean A[x, y] over the sd of A[x, z]) rises with T.

    From T = 250 to T = 4000 the index of the coupled pair stays within a
    factor of two (groups one bandwidth, 0.024, wide become 0.02 wide) and
    the spread between independent series falls about as 1 / T, so
    the ratio grows several-fold.
    """
    sig_short, null_short = _signal_and_null(250, lag, band, reps=12)
    sig_long, null_long = _signal_and_null(4000, lag, band, reps=12)
    ratio_short = sig_short.mean() / null_short.std()
    ratio_long = sig_long.mean() / null_long.std()
    assert ratio_short > 2
    assert ratio_long > 3 * ratio_short
    assert null_long.std() < 0.4 * null_short.std()
    assert 0.5 < sig_long.mean() / sig_short.mean() < 2


def test_independent_series_give_an_index_centred_on_zero():
    _, null = _signal_and_null(1000, 2, (0, 0.5), reps=40)
    assert abs(null.mean()) < 4 * null.std() / np.sqrt(null.size)


@pytest.mark.parametrize("T", [100, 250, 1000])
def test_antisymmetric_and_positive_when_the_row_leads(T):
    data = Data(_system(T, 3, seed=0), dim_order="ps")
    for fmin, fmax in BANDS:
        A = PhaseSlopeIndex(fmin=fmin, fmax=fmax).multivariate(data)
        off = ~np.eye(3, dtype=bool)
        assert A[off] == pytest.approx(-A.T[off], rel=1e-9, abs=1e-15)
        assert A[0, 1] > 0 > A[1, 0]  # x leads y


def test_short_series_use_groups_one_bandwidth_wide():
    """Below 2 NW / resolution = 300 samples a finer resolution changes nothing.

    Each group is one multitaper bandwidth wide whatever the resolution, so
    0.02 and 0.001 agree at T = 250; at T = 1000 the default groups are 20
    bins wide and the two differ.
    """
    for T, same in ((100, True), (250, True), (1000, False)):
        data = Data(_system(T, 2, seed=1), dim_order="ps")
        default = PhaseSlopeIndex().multivariate(data)
        fine = PhaseSlopeIndex(resolution=0.001).multivariate(data)
        off = ~np.eye(3, dtype=bool)
        assert np.allclose(default[off], fine[off], rtol=1e-12, atol=0) == same, T


def test_a_band_without_two_bandwidths_is_not_appropriate():
    """Two groups of 2 NW = 6 bins: the band [0.25, 0.5] needs 12 bins of the grid.

    At T = 44 it holds 11 besides fs/2 (0.25 to 0.477 in steps of 1/44), at
    T = 48 it holds 12.
    """
    with pytest.raises(NotAppropriate, match=r"two groups of 6 bins .* holds 11 there"):
        PhaseSlopeIndex(fmin=0.25, fmax=0.5).multivariate(
            Data(_system(44, 1, seed=0), dim_order="ps"))
    A = PhaseSlopeIndex(fmin=0.25, fmax=0.5).multivariate(
        Data(_system(48, 1, seed=0), dim_order="ps"))
    assert np.isfinite(A[~np.eye(3, dtype=bool)]).all()


def test_a_narrow_band_on_a_long_series_keeps_two_groups():
    """A band narrower than twice the resolution is split in two, not refused."""
    A = PhaseSlopeIndex(fmin=0.1, fmax=0.13).multivariate(
        Data(_system(8000, 2, seed=0), dim_order="ps"))
    assert A[0, 1] > 0


@pytest.mark.parametrize("bad, error", [
    (0, ValueError), (-0.01, ValueError), (float("nan"), ValueError),
    (float("inf"), ValueError), (0.6, ValueError),
    ("0.02", TypeError), (True, TypeError)])
def test_resolution_is_validated_at_construction(bad, error):
    with pytest.raises(error, match="resolution"):
        PhaseSlopeIndex(resolution=bad)


def test_identifier_names_only_a_non_default_resolution():
    base = PhaseSlopeIndex().identifier
    assert base == "psi_multitaper_mean_fs-1_fmin-0_fmax-0-5"
    assert PhaseSlopeIndex(resolution=0.02).identifier == base
    assert PhaseSlopeIndex(resolution=0.01).identifier == base + "_res-0-01"
    assert (PhaseSlopeIndex(time_halfbandwidth_product=4, resolution=0.05).identifier
            == base + "_nw-4_res-0-05")
    # The default is fs / 50, so it scales with fs as the band does.
    hz = PhaseSlopeIndex(fs=250, fmin=8, fmax=12)
    assert "_res-" not in hz.identifier
    assert PhaseSlopeIndex(fs=250, fmin=8, fmax=12, resolution=5).identifier == hz.identifier
    assert PhaseSlopeIndex(fs=250, fmin=8, fmax=12, resolution=1).identifier.endswith("_res-1")


def test_resolution_is_part_of_the_cache_key():
    """Two resolutions on one Data must not share a cached result."""
    series = _system(2000, 2, seed=3)
    shared = Data(series, dim_order="ps")
    for resolution in (0.02, 0.005, 0.02, 0.005):
        fresh = PhaseSlopeIndex(resolution=resolution).multivariate(
            Data(series, dim_order="ps"))
        got = PhaseSlopeIndex(resolution=resolution).multivariate(shared)
        assert np.array_equal(got, fresh, equal_nan=True)
    assert not np.allclose(PhaseSlopeIndex(resolution=0.02).multivariate(shared)[0, 1],
                           PhaseSlopeIndex(resolution=0.005).multivariate(shared)[0, 1])


def test_sampling_frequency_only_relabels_the_axis():
    """fs = 250 with [0, 62.5] and resolution 5 equals fs = 1 with [0, 0.25] and 0.02."""
    data = Data(_system(1500, 4, seed=2), dim_order="ps")
    a = PhaseSlopeIndex(fmin=0, fmax=0.25).multivariate(data)
    b = PhaseSlopeIndex(fs=250, fmin=0, fmax=62.5).multivariate(data)
    off = ~np.eye(3, dtype=bool)
    assert b[off] == pytest.approx(a[off], rel=1e-9)
