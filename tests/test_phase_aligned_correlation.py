"""Phase-aligned correlation P against an independent reference and its theory.

P is the supremum of the zero-lag correlation over all-pass filtering of one
process, so ``r <= max_h xcorr(h) <= P``, with equality on the right for a pure
delay. The oracle for the value is a separate einsum implementation (the one the
SPI was specified from); the rest are properties the estimate must have whatever
the implementation: symmetry, the delay identity, the null floor
``sqrt(pi / (4 K))`` for K Welch segments, and the ladder.
"""
import numpy as np
import pytest
from scipy import signal

from pyspi.data import Data
from pyspi.statistics.basic import CrossCorrelation
from pyspi.statistics.spectral import PhaseAlignedCorrelation


# Reference implementation, (time x processes), kept independent of the SPI's
# vectorisation: explicit segment loop, einsum cross-spectra, two-sided weights.
def _reference_weights(n_bins, n):
    w = np.full(n_bins, 2.0)
    w[0] = 1.0
    if n % 2 == 0:
        w[-1] = 1.0
    return w


def _reference_P(data, nperseg):
    x = np.asarray(data, float)
    starts = range(0, len(x) - nperseg + 1, nperseg // 2)
    segs = np.stack([x[s: s + nperseg] for s in starts])
    segs = segs - segs.mean(axis=1, keepdims=True)
    X = np.fft.rfft(segs * signal.get_window("hann", nperseg)[None, :, None], axis=1)
    S = np.einsum("sfj,sfk->fjk", X, X.conj()) / len(segs)
    w = _reference_weights(S.shape[0], nperseg)
    power = np.einsum("f,fjj->j", w, S).real
    return np.einsum("f,fjk->jk", w, np.abs(S)) / np.sqrt(np.outer(power, power))


def _data(z):
    return Data(data=z, dim_order="ps", zscore=False)


def _off(a):
    return np.asarray(a, float)[~np.eye(a.shape[0], dtype=bool)]


def _coupled(M, T, seed):
    """Correlated, coloured processes: mixed white noise through AR(1) filters."""
    rng = np.random.default_rng(seed)
    mixed = rng.standard_normal((M, M)) @ rng.standard_normal((M, T))
    return np.stack([signal.lfilter([1.0], [1.0, -a], row)
                     for a, row in zip(np.linspace(-0.6, 0.8, M), mixed)])


def _segments(T, nperseg):
    return (T - nperseg) // (nperseg // 2) + 1


@pytest.mark.parametrize("T,nperseg", [(20000, 256), (20000, None), (1000, None),
                                       (1000, 64), (3001, 101), (100, None)])
def test_matches_reference_implementation(T, nperseg):
    z = _coupled(5, T, seed=T)
    spi = PhaseAlignedCorrelation(nperseg=nperseg)
    n = spi._segment_length(T)
    got = spi.multivariate(_data(z))
    expected = _reference_P(z.T, n)
    np.testing.assert_allclose(_off(got), _off(expected), rtol=1e-12, atol=1e-14)


def test_automatic_segment_length():
    spi = PhaseAlignedCorrelation()
    assert [spi._segment_length(T) for T in (64, 100, 1000, 2047, 2048, 20000)] == \
        [8, 8, 64, 128, 256, 256]
    with pytest.raises(ValueError, match="T >= 64"):
        spi._segment_length(63)


def test_reference_settings_are_the_default_at_long_records():
    z = _coupled(4, 20000, seed=1)
    auto = PhaseAlignedCorrelation().multivariate(_data(z))
    fixed = PhaseAlignedCorrelation(nperseg=256).multivariate(_data(z))
    np.testing.assert_array_equal(auto, fixed)


def test_parameters_are_validated_and_reach_the_identifier():
    assert PhaseAlignedCorrelation().identifier == "phasealigned_welch_nperseg-auto"
    assert PhaseAlignedCorrelation(nperseg=128).identifier == \
        "phasealigned_welch_nperseg-128"
    for bad, err in ((True, TypeError), (2.5, TypeError), (1, ValueError)):
        with pytest.raises(err):
            PhaseAlignedCorrelation(nperseg=bad)
    with pytest.raises(ValueError, match="exceeds the series length"):
        PhaseAlignedCorrelation(nperseg=512).multivariate(_data(_coupled(2, 500, 0)))


def test_matrix_is_symmetric_bounded_and_finite_off_the_diagonal():
    z = _coupled(6, 3000, seed=3)
    spi = PhaseAlignedCorrelation()
    A = spi.multivariate(_data(z))
    assert np.array_equal(A, A.T, equal_nan=True)
    assert np.isnan(np.diag(A)).all()
    off = _off(A)
    assert np.isfinite(off).all()
    assert (off >= 0).all() and (off <= 1 + 1e-12).all()
    for i, j in ((0, 1), (4, 2), (5, 0)):
        assert spi.bivariate(_data(z), i=i, j=j) == pytest.approx(A[i, j], rel=1e-12)
    # Unsigned: negating a process leaves |S_jk| unchanged.
    flipped = z.copy()
    flipped[2] *= -1
    np.testing.assert_allclose(spi.multivariate(_data(flipped)), A,
                               rtol=1e-13, equal_nan=True)


@pytest.mark.parametrize("delay,gain", [(1, 0.7), (5, 0.6), (12, 0.5)])
def test_pure_delay_gives_the_maximum_cross_correlation(delay, gain):
    """A delay has linear phase, so one global shift already aligns every frequency.

    The 0.01 tolerance covers estimation error (|P - xcorr| <= 0.004 over three
    seeds) and the Hann resolution loss of a delay inside a 256-sample segment
    (P scales by 0.986 at d = 12).
    """
    rng = np.random.default_rng(delay)
    T = 20000
    e = rng.standard_normal(T + delay)
    z = np.stack([e[delay:],
                  gain * e[:-delay] + np.sqrt(1 - gain ** 2) * rng.standard_normal(T)])
    xcorr = CrossCorrelation(statistic="max", sigonly=False).multivariate(_data(z))[0, 1]
    P = PhaseAlignedCorrelation().multivariate(_data(z))[0, 1]
    r = np.corrcoef(z)[0, 1]
    assert abs(r) < 0.05 < gain - 0.05 < xcorr
    assert P == pytest.approx(xcorr, abs=0.01)


@pytest.mark.parametrize("T", [1000, 20000])
def test_independent_white_noise_sits_at_the_null_floor(T):
    """Zero coherence leaves a positive bias of about sqrt(pi / (4 K)).

    Measured over 200 replicates the mean is 3% (K=155) and 2% (K=30) above the
    formula, because half-overlapping segments are not quite independent.
    """
    z = np.random.default_rng(T).standard_normal((6, T))
    spi = PhaseAlignedCorrelation()
    floor = np.sqrt(np.pi / (4 * _segments(T, spi._segment_length(T))))
    off = _off(spi.multivariate(_data(z)))
    assert 0.9 < off.mean() / floor < 1.15
    assert off.max() < 2 * floor


def test_correlation_ladder_on_long_records():
    """r <= max xcorr <= P on every pair, and each rung is reached where it should be.

    Process 1 is synchronous with 0 (all three equal), 2 is a pure delay of 0
    (r ~ 0, xcorr = P) and 3 is dispersive: its low band is undelayed and its
    high band delayed by four samples, so no single shift aligns both and
    xcorr < P.
    """
    T = 2 ** 16
    rng = np.random.default_rng(0)
    e = rng.standard_normal(T)
    f = np.fft.rfftfreq(T)
    E = np.fft.rfft(e)
    low = np.fft.irfft(E * (f < 0.1), T)
    high = np.fft.irfft(E * (f > 0.3), T)
    noise = rng.standard_normal((3, T))
    z = np.stack([e,
                  0.8 * e + 0.6 * noise[0],
                  0.8 * np.roll(e, 3) + 0.6 * noise[1],
                  low + np.roll(high, 4) + 0.5 * noise[2]])

    r = np.corrcoef(z)
    xcorr = CrossCorrelation(statistic="max", sigonly=False).multivariate(_data(z))
    P = PhaseAlignedCorrelation(nperseg=256).multivariate(_data(z))

    iu = np.triu_indices(4, 1)
    # r is the lag-0 term of the same cross-correlation, so this rung is exact.
    assert (r[iu] <= xcorr[iu] + 1e-12).all()
    # Estimated by different estimators; the slack is a few times the measured
    # worst case (xcorr - P = 0.0009 over three seeds), which is mostly the
    # Hann resolution loss at the 3-sample delay (P scales by 0.9991).
    assert (xcorr[iu] <= P[iu] + 0.005).all()

    assert xcorr[0, 1] - r[0, 1] < 1e-12 and abs(P[0, 1] - xcorr[0, 1]) < 0.01
    assert xcorr[0, 2] - r[0, 2] > 0.7 and abs(P[0, 2] - xcorr[0, 2]) < 0.01
    assert P[0, 3] - xcorr[0, 3] > 0.1
