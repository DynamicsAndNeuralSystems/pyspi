"""Multitaper phase measures and group delay against independent references.

The references here do not go through spectral_connectivity: tapered Fourier
coefficients come from ``scipy.signal.windows.dpss`` and ``numpy.fft.rfft``
(which, like the backend, returns exactly real coefficients at 0 and at the
Nyquist frequency of a real signal), and each measure is written out from its
published definition.
"""
import numpy as np
import pytest
from scipy import stats
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
