"""The multitaper phase slope index with groups of frequencies that narrow as sqrt(T).

The default group width ``0.4 fs / sqrt(T)`` (never below one multitaper
bandwidth) makes a lag easier to detect on a longer series, extends the range
of lags whose sign is right as ``sqrt(T)``, reverses the sign between that
limit and twice it, and changes continuously with ``T``.
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


def _all_bands(series):
    data = Data(series, dim_order="ps")
    return np.array([PhaseSlopeIndex(fmin=lo, fmax=hi).multivariate(data) for lo, hi in BANDS])


def _ratios(T, lag, reps):
    """Per band: mean A[x, y] over the sd of A[x, z], and that sd."""
    values = np.array([_all_bands(_system(T, lag, seed))[:, 0, 1:] for seed in range(reps)])
    null_sd = values[:, :, 1].std(axis=0, ddof=1)
    return values[:, :, 0].mean(axis=0) / null_sd, null_sd


@pytest.mark.parametrize("lag", [2, 10])
def test_a_longer_series_makes_a_lag_easier_to_detect(lag):
    """In each bundled band the signal-to-null ratio grows from T = 250 to 8000.

    The groups narrow as ``1 / sqrt(T)`` while each averages about
    ``0.4 sqrt(T)`` bins, so the null spread falls faster than the index of a
    short lag changes: about twofold for a 2-sample lag and fourfold for a
    10-sample lag over this range.
    """
    short, null_short = _ratios(250, lag, reps=24)
    long, null_long = _ratios(8000, lag, reps=24)
    assert (short > 3).all()
    assert (long > 1.4 * short).all(), (short, long)
    assert (null_long < 0.85 * null_short).all()


def test_independent_series_give_an_index_centred_on_zero():
    values = np.array([_all_bands(_system(1000, 2, seed))[:, 0, 2] for seed in range(40)])
    assert (np.abs(values.mean(axis=0)) < 4 * values.std(axis=0, ddof=1) / np.sqrt(40)).all()


@pytest.mark.parametrize("T", [100, 250, 1000])
def test_antisymmetric_and_positive_when_the_row_leads(T):
    off = ~np.eye(3, dtype=bool)
    for A in _all_bands(_system(T, 3, seed=0)):
        assert A[off] == pytest.approx(-A.T[off], rel=1e-9, abs=1e-15)
        assert A[0, 1] > 0 > A[1, 0]  # x leads y


def test_the_sign_is_right_below_the_limit_and_reversed_above_it():
    """At T = 4000 the groups are 0.0063 (full band) and 0.0064 apart.

    The sign is right for lags below ``fs / (2 d)``, 78 samples, and reversed
    between that and twice it: x leading y by 65 samples gives a positive
    index, by 95 samples a negative one, in every band and realisation.
    """
    for seed in range(3):
        assert (_all_bands(_system(4000, 65, seed))[:, 0, 1] > 0).all()
        assert (_all_bands(_system(4000, 95, seed))[:, 0, 1] < 0).all()


def test_the_limit_grows_with_the_length_of_the_series():
    """A 40-sample lag is past the limit at T = 1000 (39) and inside it at 4000 (79)."""
    for seed in range(3):
        assert (_all_bands(_system(1000, 45, seed))[:, 0, 1] < 0).all()
        assert (_all_bands(_system(4000, 45, seed))[:, 0, 1] > 0).all()


def test_short_series_use_groups_one_bandwidth_wide():
    """Up to T = 225 the default width is one bandwidth, so a finer resolution changes nothing.

    At T = 1000 the default groups are 0.4 / sqrt(1000) = 0.0126 wide and the
    two differ.
    """
    off = ~np.eye(3, dtype=bool)
    for T, same in ((100, True), (200, True), (1000, False)):
        data = Data(_system(T, 2, seed=1), dim_order="ps")
        default = PhaseSlopeIndex().multivariate(data)
        fine = PhaseSlopeIndex(resolution=0.001).multivariate(data)
        assert np.allclose(default[off], fine[off], rtol=1e-12, atol=0) == same, T


@pytest.mark.parametrize("lag", [5, 20])
def test_the_index_changes_continuously_with_the_length(lag):
    """One more sample moves the index by less than the spread between independent series.

    Truncations of one realisation to T = 1000, ..., 1040 (across T = 1024,
    where the number of groups in [0, 0.5] reaches 40).
    """
    series = _system(1040, lag, seed=5)
    values = np.array([_all_bands(series[:, :T])[:, 0, 1] for T in range(1000, 1041)])
    null = np.array([_all_bands(_system(1020, 1, seed))[:, 0, 2] for seed in range(40)])
    steps = np.abs(np.diff(values, axis=0)).max(axis=0)
    assert (steps < null.std(axis=0, ddof=1)).all(), steps


@pytest.mark.parametrize("band, shortest", [((0, 0.5), 24), ((0, 0.25), 48), ((0.25, 0.5), 48)])
def test_a_band_narrower_than_two_bandwidths_is_not_appropriate(band, shortest):
    """The band must hold two bandwidths, ``T >= 4 NW fs / (fmax - fmin)``."""
    spi = PhaseSlopeIndex(fmin=band[0], fmax=band[1])
    with pytest.raises(NotAppropriate, match=rf"at least {shortest} observations; "
                                             rf"the series has {shortest - 1}\."):
        spi.multivariate(Data(_system(shortest - 1, 1, seed=0), dim_order="ps"))
    A = spi.multivariate(Data(_system(shortest, 1, seed=0), dim_order="ps"))
    assert np.isfinite(A[~np.eye(3, dtype=bool)]).all()


def test_a_narrow_band_on_a_long_series_is_split_rather_than_refused():
    """[0.1, 0.13] at T = 8000: 0.03 / (0.4 / sqrt(8000)) = 4.5, so 4 and 5 groups."""
    for seed in range(3):
        A = PhaseSlopeIndex(fmin=0.1, fmax=0.13).multivariate(
            Data(_system(8000, 10, seed), dim_order="ps"))
        assert A[0, 1] > 0


@pytest.mark.parametrize("bad, error", [
    (0, ValueError), (-0.01, ValueError), (float("nan"), ValueError),
    (float("inf"), ValueError), (0.6, ValueError),
    ("0.02", TypeError), (True, TypeError)])
def test_resolution_is_validated_at_construction(bad, error):
    with pytest.raises(error, match="resolution"):
        PhaseSlopeIndex(resolution=bad)


def test_identifier_names_an_explicit_resolution_only():
    """None, the length-dependent default, adds nothing; any value adds ``_res-``."""
    base = PhaseSlopeIndex().identifier
    assert base == "psi_multitaper_mean_fs-1_fmin-0_fmax-0-5"
    assert PhaseSlopeIndex(resolution=None).identifier == base
    assert PhaseSlopeIndex(resolution=0.02).identifier == base + "_res-0-02"
    assert PhaseSlopeIndex(resolution=np.float64(0.01)).identifier == base + "_res-0-01"
    assert (PhaseSlopeIndex(time_halfbandwidth_product=4, resolution=0.05).identifier
            == base + "_nw-4_res-0-05")
    hz = PhaseSlopeIndex(fs=250, fmin=8, fmax=12)
    assert "_res-" not in hz.identifier
    assert PhaseSlopeIndex(fs=250, fmin=8, fmax=12, resolution=5).identifier.endswith("_res-5")


def test_resolution_is_part_of_the_cache_key():
    """The default and a fixed resolution on one Data must not share a cached result."""
    series = _system(2000, 2, seed=3)
    shared = Data(series, dim_order="ps")
    for resolution in (None, 0.005, None, 0.005):
        fresh = PhaseSlopeIndex(resolution=resolution).multivariate(
            Data(series, dim_order="ps"))
        got = PhaseSlopeIndex(resolution=resolution).multivariate(shared)
        assert np.array_equal(got, fresh, equal_nan=True)
    assert not np.isclose(PhaseSlopeIndex().multivariate(shared)[0, 1],
                          PhaseSlopeIndex(resolution=0.005).multivariate(shared)[0, 1])


@pytest.mark.parametrize("resolution", [None, 0.02])
def test_sampling_frequency_only_relabels_the_axis(resolution):
    """fs = 250 with [0, 62.5] equals fs = 1 with [0, 0.25], the resolution scaled with fs."""
    data = Data(_system(1500, 4, seed=2), dim_order="ps")
    scaled = None if resolution is None else 250 * resolution
    a = PhaseSlopeIndex(fmin=0, fmax=0.25, resolution=resolution).multivariate(data)
    b = PhaseSlopeIndex(fs=250, fmin=0, fmax=62.5, resolution=scaled).multivariate(data)
    off = ~np.eye(3, dtype=bool)
    assert b[off] == pytest.approx(a[off], rel=1e-9)
