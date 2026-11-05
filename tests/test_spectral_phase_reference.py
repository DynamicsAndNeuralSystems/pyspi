"""Multitaper phase measures against their published definitions and band contracts.

Per-frequency references use ``scipy`` DPSS tapers and ``numpy.fft.rfft``; the
band-selection tests reduce the backend's own per-frequency array by hand.
"""
import os

import numpy as np
import pytest
from scipy.signal.windows import dpss

from pyspi.data import Data

# The backend's defaults: time-halfbandwidth product 3, 2 * 3 - 1 tapers.
NW, K = 3, 5


def _taper_coefficients(Z):
    """(K, F, M) tapered Fourier coefficients of the demeaned rows of ``Z``."""
    Z = np.asarray(Z, float)
    Z = Z - Z.mean(axis=1, keepdims=True)
    X = np.fft.rfft(dpss(Z.shape[1], NW, K)[:, None, :] * Z[None], axis=-1)
    return np.transpose(X, (0, 2, 1))


# --------------------------------------------------------------------------
# Debiased squared phase lag index (Vinck et al. 2011)
# --------------------------------------------------------------------------

def _dspli_reference(Z):
    """Vinck's estimator: the mean of sign(Im S_k) sign(Im S_l) over k != l.

    ``S_k = X_i,k conj(X_j,k)`` is the cross-spectrum of taper ``k``. Written
    as the pairwise-product sum, not as ``(K PLI^2 - 1)/(K - 1)``: the two
    agree only when no sign is zero.
    """
    X = _taper_coefficients(Z)
    s = np.sign(np.imag(X[..., :, None] * np.conj(X[..., None, :])))  # (K,F,M,M)
    pairs = np.einsum("kfij,lfij->fij", s, s) - np.einsum("kfij,kfij->fij", s, s)
    return pairs / (K * (K - 1))


@pytest.mark.parametrize("fmin, fmax", [(0, 0.5), (0, 0.25), (0.25, 0.5)])
def test_debiased_squared_pli_is_the_pairwise_sign_product_mean(fmin, fmax):
    """The per-frequency estimator and the shipped band means, against the definition.

    At 0 and at the Nyquist frequency every taper's Fourier coefficient of a
    real signal is real, so ``Im S_k = 0`` for every taper: the pairwise
    products are all zero and so is the estimator (the backend's shortcut
    ``(K PLI^2 - 1)/(K - 1)`` assumes ``sign^2 == 1`` and gives
    ``-1/(K - 1) = -0.25`` there). The band means leave those two bins out, as
    every phase measure does
    (test_phase_measures_reduce_over_the_band_without_the_real_valued_bins),
    so the spectrum itself is compared bin by bin, real-valued bins included.
    """
    from pyspi.statistics.spectral import DebiasedSquaredPhaseLagIndex

    rng = np.random.default_rng(11)
    Z = rng.standard_normal((4, 200))
    Z[1, 3:] += 0.7 * Z[0, :-3]
    freq = np.fft.rfftfreq(Z.shape[1])
    reference = _dspli_reference(Z)
    off = ~np.eye(4, dtype=bool)

    spi = DebiasedSquaredPhaseLagIndex(fmin=fmin, fmax=fmax)
    data = Data(Z, dim_order="ps", zscore=False)
    spectrum, _ = spi._get_cache(data)
    assert np.allclose(spectrum[0][:, off], reference[:, off], rtol=0, atol=1e-12)

    band = (freq >= fmin) & (freq <= fmax) & (freq != 0) & (freq != 0.5)
    got = spi.multivariate(data)
    expected = reference[band].mean(axis=0)
    assert np.allclose(got[off], expected[off], rtol=0, atol=1e-12)


def test_debiased_squared_pli_is_zero_where_the_cross_spectrum_is_real():
    """At 0 Hz and Nyquist no taper has a phase lag to count: exactly 0, not -0.25."""
    from pyspi.statistics.spectral import DebiasedSquaredPhaseLagIndex

    Z = np.random.default_rng(2).standard_normal((3, 100))
    spectrum, freq = DebiasedSquaredPhaseLagIndex()._get_cache(
        Data(Z, dim_order="ps", zscore=False))
    assert freq[0] == 0 and freq[-1] == 0.5
    off = ~np.eye(3, dtype=bool)
    for k in (0, -1):
        assert np.all(spectrum[0, k][off] == 0), spectrum[0, k][off]


# --------------------------------------------------------------------------
# A `statistic` an SPI cannot apply must be refused, not silently replaced
# --------------------------------------------------------------------------

def test_phase_slope_index_refuses_a_band_statistic():
    """PSI is a sum over the band; there is no spectrum to take a maximum of."""
    from pyspi.statistics.spectral import PhaseSlopeIndex

    for statistic in ("max", "delay"):
        with pytest.raises(ValueError, match="PhaseSlopeIndex"):
            PhaseSlopeIndex(statistic=statistic)


@pytest.mark.parametrize("cls_name", [
    "CoherenceMagnitude", "ImaginaryCoherence", "PhaseLockingValue",
    "PhaseLagIndex", "WeightedPhaseLagIndex", "PairwisePhaseConsistency",
    "DebiasedSquaredPhaseLagIndex", "DebiasedSquaredWeightedPhaseLagIndex"])
@pytest.mark.parametrize("statistic", ["delay", "slope", "rvalue"])
def test_regression_statistics_belong_to_group_delay_only(cls_name, statistic):
    """``delay``, ``slope`` and ``rvalue`` are GroupDelay's regression outputs.

    A spectral measure has no regression to report, so it must refuse them.
    """
    import pyspi.statistics.spectral as spectral

    with pytest.raises(ValueError, match=cls_name):
        getattr(spectral, cls_name)(statistic=statistic)


@pytest.mark.parametrize("statistic", ["mean", "max"])
def test_group_delay_refuses_a_band_statistic(statistic):
    """``mean`` and ``max`` are band statistics; GroupDelay refuses them."""
    from pyspi.statistics.spectral import GroupDelay

    with pytest.raises(ValueError, match="GroupDelay"):
        GroupDelay(statistic=statistic)


# --------------------------------------------------------------------------
# Group delay: independent frequencies are one multitaper bandwidth apart
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(6))
def test_group_delay_uses_every_independent_frequency_of_a_narrow_band(seed):
    """A 0.016-wide coupled band at T=20000 holds about 50 independent estimates.

    Multitaper estimates are approximately independent one bandwidth
    ``2 NW / T`` apart: six FFT bins at the backend's defaults, the spacing
    PhaseSlopeIndex uses. GroupDelay thins its significant run to that spacing
    (not to a fixed fraction of the band), so the fit uses all of them and
    recovers a 4-sample delay to within 0.5 samples.
    """
    from scipy import signal

    from pyspi.statistics.spectral import GroupDelay

    rng = np.random.default_rng(seed)
    T, lag = 20000, 4
    x = rng.standard_normal(T + lag)
    narrow = signal.filtfilt(
        *signal.butter(3, [0.192, 0.208], btype="band", fs=1), x)
    y = narrow[:-lag] / narrow.std() + 0.05 * rng.standard_normal(T)
    data = Data(np.vstack([x[lag:], y]), dim_order="ps", zscore=False)

    delay = GroupDelay(statistic="delay", fmin=0, fmax=0.5).multivariate(data)
    assert delay[0, 1] == pytest.approx(lag, abs=0.5)
    assert delay[1, 0] == pytest.approx(-lag, abs=0.5)


# --------------------------------------------------------------------------
# Phase measures: 0 Hz and the Nyquist frequency carry no phase
# --------------------------------------------------------------------------

_FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "data", "fixtures")

# Every multitaper class whose value is a function of the phase of the
# cross-spectrum, with the backend method that gives its per-frequency array.
_PHASE_MEASURES = {
    "ImaginaryCoherence": "imaginary_coherence",
    "PhaseLockingValue": "phase_locking_value",
    "PhaseLagIndex": "phase_lag_index",
    "WeightedPhaseLagIndex": "weighted_phase_lag_index",
    # The backend's dsPLI shortcut is wrong only where a sign is zero, i.e. at
    # the real-valued bins this reference leaves out; inside it is Vinck's
    # estimator (test_debiased_squared_pli_is_the_pairwise_sign_product_mean).
    "DebiasedSquaredPhaseLagIndex": "debiased_squared_phase_lag_index",
    "DebiasedSquaredWeightedPhaseLagIndex":
        "debiased_squared_weighted_phase_lag_index",
    "PairwisePhaseConsistency": "pairwise_phase_consistency",
}
# The class names whose configs ship a `max` variant.
_WITH_MAX = ("ImaginaryCoherence", "PhaseLockingValue", "PhaseLagIndex",
             "WeightedPhaseLagIndex", "PairwisePhaseConsistency")
_BANDS = [(0, 0.5), (0, 0.25), (0.25, 0.5)]


def _kuramoto():
    return Data(data=os.path.join(_FIXTURES, "kuramoto_M7_T100.npy"),
                dim_order="sp")


def _odd_fft():
    """T=101: the backend pads to 105 samples, so there is no Nyquist bin."""
    rng = np.random.default_rng(5)
    Z = rng.standard_normal((4, 101))
    Z[1, 2:] += 0.8 * Z[0, :-2]
    return Data(Z, dim_order="ps", zscore=False)


def _backend_spectrum(data, method):
    """The per-frequency array, straight from spectral_connectivity."""
    import spectral_connectivity as sc

    z = data.to_numpy(squeeze=True).T[:, np.newaxis, :]
    conn = sc.Connectivity.from_multitaper(
        sc.Multitaper(z, sampling_frequency=1))
    return np.asarray(getattr(conn, method)())[0], np.asarray(conn.frequencies)


def _interior(freq, fmin, fmax):
    """The closed band without 0 and fs/2 (exact at fs=1 on this grid)."""
    return (freq >= fmin) & (freq <= fmax) & (freq != 0) & (freq != 0.5)


@pytest.mark.parametrize("make", [_kuramoto, _odd_fft], ids=["T100", "T101"])
@pytest.mark.parametrize("fmin, fmax", _BANDS)
@pytest.mark.parametrize("statistic", ["mean", "max"])
@pytest.mark.parametrize("cls_name", sorted(_PHASE_MEASURES))
def test_phase_measures_reduce_over_the_band_without_the_real_valued_bins(
        cls_name, statistic, fmin, fmax, make):
    """At 0 Hz and at Nyquist the cross-spectrum of real signals is real.

    Every taper's Fourier coefficient of a real signal is real at those two
    frequencies, so each taper's cross-spectrum has phase exactly 0 or pi and
    none of these measures estimates anything there: PLI, dsPLI and imaginary
    coherence are 0, wPLI is 0/0 (the backend returns 0, dswPLI NaN), and
    PLV and PPC count how many of the five tapers agree in sign -- 1.0
    whenever all do. The shipped bands are closed and each contains one of the
    two, which would add a constant that measures nothing. Here the reference
    is the backend's own per-frequency array, reduced over the band without
    those bins.

    The T=101 case has an odd FFT length and therefore no Nyquist bin: its
    last bin (0.4952) carries phase and must stay in the band.
    """
    import pyspi.statistics.spectral as spectral

    if statistic == "max" and cls_name not in _WITH_MAX:
        pytest.skip("no shipped max variant")
    data = make()
    spectrum, freq = _backend_spectrum(data, _PHASE_MEASURES[cls_name])
    reduce = np.nanmean if statistic == "mean" else np.nanmax
    expected = reduce(spectrum[_interior(freq, fmin, fmax)], axis=0)

    got = getattr(spectral, cls_name)(
        statistic=statistic, fmin=fmin, fmax=fmax).multivariate(data)
    off = ~np.eye(got.shape[0], dtype=bool)
    assert np.allclose(got[off], expected[off], rtol=0, atol=1e-12,
                       equal_nan=True), np.max(np.abs(got[off] - expected[off]))


@pytest.mark.parametrize("make", [_kuramoto, _odd_fft], ids=["T100", "T101"])
@pytest.mark.parametrize("fmin, fmax", _BANDS)
def test_coherence_phase_is_the_circular_mean_without_the_real_valued_bins(
        fmin, fmax, make):
    """At 0 Hz and Nyquist the coherence phase is exactly 0 or pi.

    On kuramoto_M7_T100 that is 0 at some pairs and pi at others, so including
    those bins would pull each band mean toward an arbitrary axis.
    """
    from pyspi.statistics.spectral import CoherencePhase

    data = make()
    phase, freq = _backend_spectrum(data, "coherence_phase")
    inside = phase[_interior(freq, fmin, fmax)]
    resultant = np.mean(np.exp(1j * inside), axis=0)
    expected = np.angle(resultant)

    got = CoherencePhase(statistic="mean", fmin=fmin, fmax=fmax).multivariate(data)
    lower = np.tril_indices(got.shape[0], -1)
    # The class's own antipodal/zero-resultant rule may make an entry NaN.
    defined = ~np.isnan(got[lower])
    assert defined.mean() > 0.9
    assert np.allclose(got[lower][defined], expected[lower][defined],
                       rtol=0, atol=1e-12)
    assert np.allclose(got, -got.T, equal_nan=True)


@pytest.mark.parametrize("cls_name", ["PhaseLockingValue",
                                      "PairwisePhaseConsistency"])
@pytest.mark.parametrize("fmin, fmax", _BANDS)
def test_plv_and_ppc_maxima_are_not_set_by_a_real_valued_bin(cls_name, fmin, fmax):
    """``plv_*_max`` and ``ppc_*_max`` are below 1 on kuramoto_M7_T100.

    At 0 Hz or Nyquist both are exactly 1.0 whenever the five tapers' real
    cross-spectra share a sign. No interior bin of any pair reaches 1.0 (the
    largest is 0.99999).
    """
    import pyspi.statistics.spectral as spectral

    got = getattr(spectral, cls_name)(
        statistic="max", fmin=fmin, fmax=fmax).multivariate(_kuramoto())
    off = ~np.eye(got.shape[0], dtype=bool)
    assert np.all(got[off] < 1), f"{np.sum(got[off] == 1)} entries are exactly 1"


@pytest.mark.parametrize("statistic", ["mean", "max"])
@pytest.mark.parametrize("fmin, fmax", _BANDS + [(0, 0)])
def test_coherence_magnitude_keeps_the_closed_band(statistic, fmin, fmax):
    """An amplitude measure is defined at 0 Hz and Nyquist: nothing is dropped."""
    from pyspi.statistics.spectral import CoherenceMagnitude

    data = _kuramoto()
    spectrum, freq = _backend_spectrum(data, "coherence_magnitude")
    reduce = np.nanmean if statistic == "mean" else np.nanmax
    expected = reduce(spectrum[(freq >= fmin) & (freq <= fmax)], axis=0)

    got = CoherenceMagnitude(
        statistic=statistic, fmin=fmin, fmax=fmax).multivariate(data)
    off = ~np.eye(got.shape[0], dtype=bool)
    assert np.allclose(got[off], expected[off], rtol=0, atol=1e-12)


@pytest.mark.parametrize("fmin, fmax", [(0, 0), (0.5, 0.5), (0, 0.005)])
@pytest.mark.parametrize("cls_name", sorted(_PHASE_MEASURES) + ["CoherencePhase"])
def test_a_band_of_only_real_valued_bins_is_refused(cls_name, fmin, fmax):
    """No phase measure has a value on a band holding nothing but 0 Hz or Nyquist.

    ``(0, 0.005)`` at T=100 holds only the 0 Hz bin (the grid step is 0.01).
    """
    import pyspi.statistics.spectral as spectral

    spi = getattr(spectral, cls_name)(fmin=fmin, fmax=fmax)
    with pytest.raises(ValueError, match="real-valued"):
        spi.multivariate(_kuramoto())


# --------------------------------------------------------------------------
# Closed band with a tolerant edge; PSI and GroupDelay follow the phase rule
# --------------------------------------------------------------------------

def _backend_frequencies(data, fs):
    import spectral_connectivity as sc

    z = np.transpose(data.to_numpy(squeeze=True))[:, None, :]
    return np.asarray(sc.Connectivity.from_multitaper(
        sc.Multitaper(z, sampling_frequency=fs)).frequencies)


@pytest.mark.parametrize("fs", [3, 7, 250])
def test_closed_band_keeps_the_nyquist_bin_at_any_sampling_rate(fs):
    """``freq <= fs/2`` is false for the backend's Nyquist bin at fs=3, T=100.

    The grid is ``k * fs / nfft``, whose last bin is 1.5000000000000002, so an
    exact comparison drops the bin ``fmax`` names. The shared ``_closed_band``
    compares with a relative 1e-9 tolerance; at fs=1 it is the exact
    comparison.
    """
    from pyspi.statistics.spectral import CoherenceMagnitude, _closed_band

    rng = np.random.default_rng(0)
    data = Data(rng.standard_normal((2, 100)), dim_order="ps")
    freq = _backend_frequencies(data, fs)
    assert freq[-1] != fs / 2 or fs == 250, "premise: an inexact Nyquist bin"
    band = CoherenceMagnitude(fs=fs, fmin=0, fmax=fs / 2)._band(freq)
    assert band.size == freq.size
    exact = (freq >= 0) & (freq <= 0.5)
    freq1 = _backend_frequencies(data, 1)
    assert np.array_equal(_closed_band(freq1, 0, 0.5), (freq1 >= 0) & (freq1 <= 0.5))
    assert np.array_equal(_closed_band(freq1, 0, 0.25), (freq1 >= 0) & (freq1 <= 0.25))
    assert np.array_equal(_closed_band(freq1, 0.25, 0.5), (freq1 >= 0.25) & (freq1 <= 0.5))


@pytest.mark.parametrize("fmin, fmax", [(0, 0.5), (0, 0.25), (0.25, 0.5)])
def test_phase_slope_index_uses_the_closed_band_less_the_real_bins(fmin, fmax):
    """PSI keeps an interior band edge such as 0.25.

    It follows the phase rule: closed band less 0 Hz and fs/2, thinned to one
    bin per multitaper bandwidth, and equals Nolte's sum computed directly on
    that grid.
    """
    import spectral_connectivity as sc

    from pyspi.statistics.spectral import PhaseSlopeIndex, _real_valued_bins

    rng = np.random.default_rng(1)
    x = rng.standard_normal(300)
    y = np.r_[x[3:], rng.standard_normal(3)] + 0.3 * rng.standard_normal(300)
    data = Data(np.vstack([x, y]), dim_order="ps")
    z = np.transpose(data.to_numpy(squeeze=True))[:, None, :]
    C = sc.Connectivity.from_multitaper(sc.Multitaper(z, sampling_frequency=1))
    freq = np.asarray(C.frequencies)
    keep = np.flatnonzero((freq >= fmin) & (freq <= fmax) & ~_real_valued_bins(freq, 1))
    thinned = keep[::PhaseSlopeIndex._FREQUENCY_STEP]
    coh = np.asarray(C.coherency())[0, thinned, 0, 1]
    want = np.imag(np.sum(np.conj(coh[:-1]) * coh[1:]))

    got = PhaseSlopeIndex(fmin=fmin, fmax=fmax).multivariate(data)
    assert got[0, 1] == pytest.approx(want, rel=1e-12)
    assert got[1, 0] == pytest.approx(-want, rel=1e-12)
    assert 0.0 not in freq[keep] and 0.5 not in freq[keep]  # real-valued bins never kept
    if fmin == 0.25:
        assert 0.25 in freq[keep], "the interior edge the identifier names is kept"


def test_group_delay_fit_leaves_out_the_real_valued_bins():
    """0 Hz and fs/2 force the phase to 0 or pi whatever the delay.

    They are not observations of the phase slope. On a delayed pair the
    full-band fit must equal the fit on the band that stops one bin short of
    both edges, since the two see the same bins.
    """
    from pyspi.statistics.spectral import GroupDelay

    rng = np.random.default_rng(2)
    T, lag = 400, 3
    x = rng.standard_normal(T + lag)
    y = x[:-lag] + 0.2 * rng.standard_normal(T)
    data = Data(np.vstack([x[lag:], y]), dim_order="ps")
    freq = _backend_frequencies(data, 1)
    df = freq[1] - freq[0]
    full = GroupDelay(statistic="delay", fmin=0, fmax=0.5).multivariate(data)
    interior = GroupDelay(statistic="delay", fmin=df, fmax=0.5 - df).multivariate(data)
    assert np.isfinite(full[0, 1])
    assert full[0, 1] == pytest.approx(interior[0, 1], rel=1e-12)
    assert full[0, 1] == pytest.approx(lag, abs=0.5)
