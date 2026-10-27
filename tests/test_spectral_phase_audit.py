"""Multitaper phase measures against independent references and their contracts.

The references here do not go through spectral_connectivity: tapered Fourier
coefficients come from ``scipy.signal.windows.dpss`` and ``numpy.fft.rfft``
(which, like the backend, returns exactly real coefficients at 0 and at the
Nyquist frequency of a real signal), and each measure is written out from its
published definition.
"""
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
    """The shipped band means against the definition, bin by bin.

    At 0 and at the Nyquist frequency every taper's Fourier coefficient of a
    real signal is real, so ``Im S_k = 0`` for every taper: the pairwise
    products are all zero and so is the estimator. The backend's shortcut
    ``(K PLI^2 - 1)/(K - 1)`` assumes ``sign^2 == 1`` and returns
    ``-1/(K - 1) = -0.25`` there for every input, which pulled each
    ``dspli_multitaper_mean_*`` down by about 0.25 / 26 at T=100.
    """
    from pyspi.statistics.spectral import DebiasedSquaredPhaseLagIndex

    rng = np.random.default_rng(11)
    Z = rng.standard_normal((4, 200))
    Z[1, 3:] += 0.7 * Z[0, :-3]
    freq = np.fft.rfftfreq(Z.shape[1])
    band = (freq >= fmin) & (freq <= fmax)
    expected = _dspli_reference(Z)[band].mean(axis=0)

    got = DebiasedSquaredPhaseLagIndex(fmin=fmin, fmax=fmax).multivariate(
        Data(Z, dim_order="ps", zscore=False))
    off = ~np.eye(4, dtype=bool)
    assert np.allclose(got[off], expected[off], rtol=0, atol=1e-12)


def test_debiased_squared_pli_is_zero_where_the_cross_spectrum_is_real():
    """A band holding only the 0 Hz bin: no taper has a phase lag to count."""
    from pyspi.statistics.spectral import DebiasedSquaredPhaseLagIndex

    Z = np.random.default_rng(2).standard_normal((3, 100))
    got = DebiasedSquaredPhaseLagIndex(fmin=0, fmax=0).multivariate(
        Data(Z, dim_order="ps", zscore=False))
    off = ~np.eye(3, dtype=bool)
    assert np.all(got[off] == 0), got[off]


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
