"""Multitaper phase measures against independent references and their contracts.

The per-frequency references here do not go through spectral_connectivity:
tapered Fourier coefficients come from ``scipy.signal.windows.dpss`` and
``numpy.fft.rfft`` (which, like the backend, returns exactly real coefficients
at 0 and at the Nyquist frequency of a real signal), and each measure is
written out from its published definition. The band-selection tests are the
exception: what they check is which frequencies pyspi reduces over, so their
reference is the backend's own per-frequency array reduced by hand.
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
    products are all zero and so is the estimator. The backend's shortcut
    ``(K PLI^2 - 1)/(K - 1)`` assumes ``sign^2 == 1`` and returns
    ``-1/(K - 1) = -0.25`` there for every input, which pulled each
    ``dspli_multitaper_mean_*`` down by about 0.25 / 26 at T=100. The band
    means now leave those two bins out, as every phase measure does
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
    """PSI is a sum over the band; there is no spectrum to take a maximum of.

    ``PhaseSlopeIndex(statistic="max")`` used to return exactly the default
    variant's matrix under a ``psi_multitaper_max_*`` identifier, labelled
    ``asymmetric`` although the values were antisymmetric.
    """
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
    """``statistic="delay"`` on a spectrum used to return its 0 Hz bin.

    The band-statistic dispatcher fell through to GroupDelay's tuple branch on
    the ``TypeError`` from calling ``None``, which indexed the spectrum's first
    frequency, under an identifier such as ``cohmag_multitaper_delay_*``.
    """
    import pyspi.statistics.spectral as spectral

    with pytest.raises(ValueError, match=cls_name):
        getattr(spectral, cls_name)(statistic=statistic)


@pytest.mark.parametrize("statistic", ["mean", "max"])
def test_group_delay_refuses_a_band_statistic(statistic):
    """It reported an ``IndexError`` from inside ``multivariate`` instead."""
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
    PhaseSlopeIndex uses. GroupDelay instead thinned its significant run to
    one point per ``(fmax - fmin) / 50`` whatever ``T`` -- 200 bins here -- so
    the fit rested on three points and missed a 4-sample delay by 0.5 to 2.0
    samples on every one of these seeds (now 0.03 to 0.13); for a 0.005-wide
    band it returned NaN. (At T=100 the same rule is a 1-bin step, and three
    *adjacent*, strongly correlated bins passed the
    three-independent-points requirement.)
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
    two, so every value was a reduction over the band *plus* a constant that
    measures nothing. Here the reference is the backend's own per-frequency
    array, reduced over the band without those bins.

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

    On kuramoto_M7_T100 that is 0 at some pairs and pi at others, so each band
    mean was pulled toward an arbitrary axis by a bin with no phase in it.
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
    """``plv_*_max`` and ``ppc_*_max`` were exactly 1.0 on 16 of 42 entries.

    On kuramoto_M7_T100 (full band; 6 and 12 of 42 in the half bands) the
    maximum was exactly 1.0 for no reason but that the five tapers' real
    cross-spectra at 0 Hz or Nyquist had one sign. No interior bin of any pair
    reaches 1.0 (the largest is 0.99999).
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
    It used to return the measure at that bin -- PLV 0.2, 0.6 or 1.0 -- as if
    it were a band estimate.
    """
    import pyspi.statistics.spectral as spectral

    spi = getattr(spectral, cls_name)(fmin=fmin, fmax=fmax)
    with pytest.raises(ValueError, match="real-valued"):
        spi.multivariate(_kuramoto())
