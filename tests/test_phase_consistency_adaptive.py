"""Spectrally weighted Hilbert phase consistency (``ppc_hilbert_adaptive``).

The value is checked against an independent reference (a standalone function
written from the definition, with an explicit sampling frequency) and against
the properties any implementation must have: about 0 for independent pairs
whatever their spectra, also at short length with a drifting phase
difference, high for phase-locked oscillators, symmetric,
invariant to the sign of either process, free of a sampling-frequency
parameter, NaN where undefined, and multivariate equal to bivariate.
"""
import inspect

import numpy as np
import pytest
import spectral_connectivity as sc
from scipy import signal

from pyspi.data import Data
from pyspi.statistics.spectral import AdaptivePhaseConsistency


# Reference implementation, kept independent of the SPI: bands in Hz with an
# explicit sampling frequency, the backend's default tapers, and one pass per
# pair (no per-process reuse).
def _reference_flat_power(x, fs):
    m = sc.Multitaper(x[:, None, None], sampling_frequency=fs)
    S = np.mean(np.abs(m.fft()[0, 0, :, :, 0]) ** 2, axis=0)[: x.size // 2 + 1]
    f = np.arange(S.size) * fs / m.n_fft_samples
    keep = (f >= 3 * fs / x.size) & (f < fs / 2)
    a, b = np.polyfit(np.log(f[keep]), np.log(S[keep]), 1)
    return f[keep], S[keep] / np.exp(b + a * np.log(f[keep]))


def _reference_segment_ppc(dphi, L):
    M = dphi.size // L
    if M < 3:
        return np.nan
    s = np.exp(1j * dphi[: M * L]).reshape(M, L).sum(axis=1)
    num = np.abs(s.sum()) ** 2 - np.sum(np.abs(s) ** 2) - 2 * np.sum((s[1:] * np.conj(s[:-1])).real)
    return num / ((M * L) ** 2 - M * L * L - 2 * (M - 1) * L * L)


def _reference(x, y, fs=1.0):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    T = x.size
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    f, Gx = _reference_flat_power(x - x.mean(), fs)
    _, Gy = _reference_flat_power(y - y.mean(), fs)
    G = Gx * Gy
    trim = slice(T // 20, T - T // 20)
    num = den = 0.0
    fc = max(4 * fs / T, 0.005 * fs)
    while fc <= 0.4 * fs:
        lo, hi = fc / 2, min(1.5 * fc, 0.49 * fs)
        sos = signal.butter(2, [lo, hi], btype="band", fs=fs, output="sos")
        px = np.angle(signal.hilbert(signal.sosfiltfilt(sos, x)))
        py = np.angle(signal.hilbert(signal.sosfiltfilt(sos, y)))
        dphi = (px - py)[trim]
        L = int(np.ceil(2 * fs / (hi - lo)))
        p = _reference_segment_ppc(dphi, L)
        inb = (f >= lo) & (f <= hi)
        if dphi.size // L >= 6 and np.isfinite(p) and inb.any():
            w = G[inb].mean()
            num += w * p
            den += w
        fc *= np.sqrt(2)
    return num / den if den > 0 else np.nan


def _data(z):
    return Data(data=np.asarray(z), dim_order="ps", zscore=False)


def _off(a):
    return np.asarray(a, float)[~np.eye(a.shape[0], dtype=bool)]


def _white(rng, M, T):
    return rng.standard_normal((M, T))


def _red(rng, M, T):
    return signal.lfilter([1.0], [1.0, -0.9], rng.standard_normal((M, T)), axis=1)


def _pink(rng, M, T):
    f = np.fft.rfftfreq(T)
    f[0] = f[1]
    E = np.fft.rfft(rng.standard_normal((M, T)), axis=1)
    return np.fft.irfft(E / np.sqrt(f), T, axis=1)


def _resonant(rng, M, T, f0=0.1, r=0.95):
    """Independent AR(2) processes that share one spectral peak at ``f0``."""
    a = [1.0, -2 * r * np.cos(2 * np.pi * f0), r * r]
    return signal.lfilter([1.0], a, rng.standard_normal((M, T)), axis=1)


def _locked(rng, T, f0, lag=0.8, noise=0.5, diffusion=0.05):
    """Two oscillators sharing one diffusing phase at a fixed lag, on 1/f noise."""
    phi = 2 * np.pi * f0 * np.arange(T) + np.cumsum(diffusion * rng.standard_normal(T))
    background = _pink(rng, 2, T)
    background /= background.std(axis=1, keepdims=True)
    return np.stack([np.cos(phi), np.cos(phi + lag)]) + noise * background


def _mixed(seed, T=1000):
    """Two locked oscillators, an independent red process and white noise."""
    rng = np.random.default_rng(seed)
    return np.vstack([_locked(rng, T, 0.07), _red(rng, 1, T), _white(rng, 1, T)])


@pytest.mark.parametrize("generator", [_white, _red, _pink, _resonant])
def test_independent_pairs_are_near_zero(generator):
    z = generator(np.random.default_rng(7), 8, 2000)
    off = _off(AdaptivePhaseConsistency().multivariate(_data(z)))
    assert np.isfinite(off).all()
    assert abs(off.mean()) < 0.03
    assert np.abs(off).max() < 0.25


def test_detuned_oscillators_at_short_length_are_near_zero():
    """Independent oscillators a little apart in frequency, at T = 100: each
    pair's phase difference drifts by up to a few cycles across the record."""
    rng = np.random.default_rng(11)
    T = 100
    f0 = np.linspace(0.04, 0.08, 8)
    phi = (2 * np.pi * f0[:, None] * np.arange(T) + rng.uniform(0, 2 * np.pi, (8, 1))
           + np.cumsum(0.05 * rng.standard_normal((8, T)), axis=1))
    z = np.sin(phi) + 0.2 * rng.standard_normal((8, T))
    off = _off(AdaptivePhaseConsistency().multivariate(_data(z)))
    assert np.isfinite(off).all()
    assert abs(off.mean()) < 0.05
    assert off.min() > -0.45


@pytest.mark.parametrize("generator", [_white, _red, _pink])
def test_independent_pairs_at_short_length_are_near_zero(generator):
    z = generator(np.random.default_rng(13), 8, 100)
    off = _off(AdaptivePhaseConsistency().multivariate(_data(z)))
    assert np.isfinite(off).all()
    assert abs(off.mean()) < 0.05
    assert off.min() > -0.2


@pytest.mark.parametrize("f0", [0.05, 0.25])
def test_phase_locked_oscillators_score_high(f0):
    z = _locked(np.random.default_rng(int(100 * f0)), 2000, f0)
    assert AdaptivePhaseConsistency().bivariate(_data(z), i=0, j=1) > 0.7


def test_symmetric_and_invariant_to_sign_flips():
    z = _mixed(seed=1)
    spi = AdaptivePhaseConsistency()
    A = spi.multivariate(_data(z))
    assert np.array_equal(A, A.T, equal_nan=True)
    assert np.isnan(np.diag(A)).all()
    for i, j in ((0, 1), (0, 2), (1, 3)):
        assert spi.bivariate(_data(z), i=j, j=i) == pytest.approx(A[i, j], rel=1e-12, abs=1e-14)
    for k in range(z.shape[0]):
        flipped = z.copy()
        flipped[k] *= -1
        np.testing.assert_allclose(spi.multivariate(_data(flipped)), A,
                                   rtol=1e-10, atol=1e-12, equal_nan=True)


def test_multivariate_equals_bivariate():
    z = _mixed(seed=2)
    spi = AdaptivePhaseConsistency()
    A = spi.multivariate(_data(z))
    for i in range(z.shape[0]):
        for j in range(i + 1, z.shape[0]):
            assert spi.bivariate(_data(z), i=i, j=j) == A[i, j]


def test_matches_reference_implementation():
    z = _mixed(seed=3)
    A = AdaptivePhaseConsistency().multivariate(_data(z))
    expected = np.full_like(A, np.nan)
    for i in range(z.shape[0]):
        for j in range(z.shape[0]):
            if i != j:
                expected[i, j] = _reference(z[i], z[j])
    np.testing.assert_allclose(_off(A), _off(expected), rtol=1e-10, atol=1e-12)


def test_has_no_sampling_frequency():
    """Bands are in cycles per sample: there is no ``fs`` to set, and the
    reference evaluated at any sampling frequency gives the same value."""
    assert not inspect.signature(AdaptivePhaseConsistency).parameters
    with pytest.raises(TypeError):
        AdaptivePhaseConsistency(fs=250)
    z = _mixed(seed=4)
    value = AdaptivePhaseConsistency().bivariate(_data(z), i=0, j=1)
    for fs in (1.0, 250.0):
        assert _reference(z[0], z[1], fs=fs) == pytest.approx(value, rel=1e-10)


@pytest.mark.parametrize("level", [0.0, 0.1, 3.0])
def test_constant_process_gives_nan(level):
    z = _white(np.random.default_rng(5), 3, 500)
    z[1] = level
    A = AdaptivePhaseConsistency().multivariate(_data(z))
    assert np.isnan(A[1]).all() and np.isnan(A[:, 1]).all()
    assert np.isfinite(A[0, 2])


@pytest.mark.parametrize("T", [10, 23, 45])
def test_series_too_short_for_six_segments_gives_nan(T):
    z = _white(np.random.default_rng(T), 2, T)
    assert np.isnan(AdaptivePhaseConsistency().bivariate(_data(z), i=0, j=1))


def test_shortest_defined_series():
    z = _white(np.random.default_rng(46), 2, 46)
    assert np.isfinite(AdaptivePhaseConsistency().bivariate(_data(z), i=0, j=1))
