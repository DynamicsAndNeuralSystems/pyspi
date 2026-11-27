"""Wilson's factorisation starts from a square root of the zero-lag covariance.

spectral_connectivity 2.x starts Wilson's iteration from the transposed
Cholesky factor of the zero-lag covariance, which is not a square root of it
unless the covariance is diagonal. For a slowly varying pair whose innovations
are correlated at lag zero, the iteration then ends at a factor that is not
minimum phase, without a warning. pyspi factorises the whitened spectrum
instead (``spectral._minimum_phase_factor``). These tests feed exact VAR(1)
spectra through pyspi's own code paths, so every error is the factorisation's.
"""

import numpy as np
import pytest
from scipy.fft import ifft

from pyspi.statistics import spectral

# Process 0 drives process 1; both are slowly varying, and their innovations
# are correlated at lag zero.
A_SLOW = np.array([[0.99, 0.0], [0.05, 0.99]])
SIGMA = np.array([[1.0, 0.5], [0.5, 1.0]])
N_FFT = 1024


def _var1(A, Sigma, n):
    """Transfer function and two-sided spectrum of ``x_t = A x_{t-1} + e_t``."""
    w = 2 * np.pi * np.fft.fftfreq(n)
    H = np.linalg.inv(np.eye(2) - A * np.exp(-1j * w)[:, None, None])
    return H, H @ Sigma @ np.conj(np.swapaxes(H, -1, -2))


def _geweke(H, Sigma, S):
    """Geweke's spectral GC ``(0 -> 1, 1 -> 0)`` at every frequency."""
    s0, s1 = S[:, 0, 0].real, S[:, 1, 1].real
    p01 = Sigma[0, 0] - Sigma[0, 1] ** 2 / Sigma[1, 1]
    p10 = Sigma[1, 1] - Sigma[0, 1] ** 2 / Sigma[0, 0]
    return (np.log(s1 / (s1 - p01 * np.abs(H[:, 1, 0]) ** 2)),
            np.log(s0 / (s0 - p10 * np.abs(H[:, 0, 1]) ** 2)))


def _exact_connectivity(H, Sigma, cls):
    """A ``Connectivity`` whose cross-spectral matrix is exactly ``H Sigma H^H``.

    Two tapers carry the columns of ``H chol(Sigma)``, scaled by sqrt(2), so
    their mean outer product is the spectrum.
    """
    F = H @ np.linalg.cholesky(Sigma)
    coefficients = np.sqrt(2) * np.moveaxis(F, -1, 0)   # (taper, f, signal)
    return cls(coefficients[np.newaxis, np.newaxis],
               frequencies=np.fft.fftfreq(H.shape[0]))


def test_the_backend_factorises_a_spectrum_with_identity_zero_lag_covariance(
        monkeypatch):
    """The backend's own start is then the identity, a square root of it."""
    seen = []
    inner = spectral.minimum_phase_decomposition
    monkeypatch.setattr(spectral, "minimum_phase_decomposition",
                        lambda csm, **kw: seen.append(csm) or inner(csm, **kw))
    _, S = _var1(A_SLOW, SIGMA, N_FFT)
    spectral._minimum_phase_factor(S[np.newaxis])
    lag0 = ifft(seen[0], axis=-3)[..., 0, :, :].real
    np.testing.assert_allclose(lag0, np.eye(2)[np.newaxis], atol=1e-12)


@pytest.mark.parametrize("A, Sigma", [
    (A_SLOW, SIGMA),
    (np.array([[0.5, 0.0], [0.4, 0.5]]), SIGMA),
    (np.array([[0.5, 0.0], [0.4, 0.5]]), np.eye(2)),
], ids=["slow-correlated", "fast-correlated", "fast-independent"])
def test_spectral_gc_of_an_exact_spectrum_is_gewekes(A, Sigma):
    """Nonparametric spectral GC's computation recovers the analytic values.

    On the slow, correlated pair, spectral_connectivity 2.x's start gives a
    mean GC of 0.145 for 0 -> 1 and 0.034 for 1 -> 0, against 0.022 and 0.
    The remaining error, at most 2e-3 in the mean, is the grid's: the
    factor's lags wrap round 1024 bins.
    """
    H, S = _var1(A, Sigma, N_FFT)
    power = np.real(np.diagonal(S, axis1=-2, axis2=-1))[np.newaxis]
    F = spectral._pairwise_spectral_granger(power, S[np.newaxis], [(0, 1)])[0]
    # F[f, target, source]. The backend sets a GC of at most 0 to NaN.
    gc01, gc10 = np.nan_to_num(F[:, 1, 0]), np.nan_to_num(F[:, 0, 1])
    true01, true10 = _geweke(H, Sigma, S)
    half = slice(0, N_FFT // 2 + 1)
    assert abs(gc01.mean() - true01[half].mean()) < 3e-3
    assert abs(gc10.mean() - true10[half].mean()) < 3e-3
    assert gc01.mean() > 10 * gc10.mean()


def test_directed_coherence_reads_the_minimum_phase_factor():
    """``_FactoredConnectivity`` gives the true transfer function and innovations.

    These are all that ``DirectedCoherence`` and the other bivariate directed
    measures read. 4096 bins resolve this pair's factor to 3e-5; from
    spectral_connectivity 2.x's start the transfer function is 9e-4 off.
    """
    n = 4096
    H, _ = _var1(A_SLOW, SIGMA, n)
    conn = _exact_connectivity(H, SIGMA, spectral._FactoredConnectivity)
    half = slice(0, n // 2 + 1)
    np.testing.assert_allclose(conn._noise_covariance[0], SIGMA, atol=1e-6)
    err = np.abs(conn._transfer_function[0] - H[half]).max()
    assert err < 1e-4, f"transfer function off by {err:.3g}"
