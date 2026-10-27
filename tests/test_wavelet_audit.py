"""Wavelet phase slope index against independent references.

`PhaseSlopeIndex` (statistics/wavelet.py) delegates to mne-connectivity's
`phase_slope_index` with Morlet wavelets. These tests pin what pyspi itself
decides around that call: the sign convention and matrix orientation, and the
units of the frequency band.
"""
import warnings

import numpy as np
import pytest

from pyspi.data import Data


def _lagged(seed=0, t=400, lag=4):
    """Rows: x, x delayed by `lag` plus noise (x leads), independent noise."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal(t + lag))
    lead = x[lag:]
    follow = x[:-lag] + 0.5 * rng.standard_normal(t)
    return np.vstack([lead, follow, rng.standard_normal(t)])


def _psi(arr, **kwargs):
    from pyspi.statistics.wavelet import PhaseSlopeIndex

    with warnings.catch_warnings():
        # fs != 1 triggers the base class's "Multiple sampling frequencies"
        # notice; it is not what these tests are about.
        warnings.simplefilter("ignore")
        spi = PhaseSlopeIndex(**kwargs)
        return np.asarray(spi.multivariate(Data(arr, dim_order="ps")), float)


def test_wavelet_psi_is_positive_from_the_leading_to_the_lagging_process():
    """Row = source: A[i, j] > 0 when i leads j, and A is antisymmetric."""
    A = _psi(_lagged(), fmin=0, fmax=0.25)
    assert A[0, 1] > 0, f"x leads y but psi[x, y] = {A[0, 1]:.4g}"
    assert A[1, 0] == pytest.approx(-A[0, 1], rel=1e-12)


@pytest.mark.parametrize("band", [(0, 0.25), (0.25, 0.5), (0, 0.5)])
def test_wavelet_psi_is_invariant_to_the_unit_of_frequency(band):
    """Changing fs and the band together only changes units, not the result.

    `fs` is in the identifier and the band is in Hz, so
    `PhaseSlopeIndex(fs=F, fmin=F*lo, fmax=F*hi)` asks for the same band of the
    same samples as `PhaseSlopeIndex(fs=1, fmin=lo, fmax=hi)`. The data-supported
    floor for `fmin` is five cycles over the record, `5 * fs / T` Hz; it used to
    be computed as `5 / T` whatever `fs` was, so at fs=250 it let `fmin=0` reach
    1/250 of the supported floor and summed phase slopes over wavelets far
    longer than the signal.
    """
    lo, hi = band
    arr = _lagged()
    ref = _psi(arr, fs=1, fmin=lo, fmax=hi)
    fs = 250.0
    got = _psi(arr, fs=fs, fmin=lo * fs, fmax=hi * fs)
    off = ~np.eye(3, dtype=bool)
    assert np.allclose(got[off], ref[off], rtol=1e-9, atol=1e-12), (
        f"band {band}: fs={fs:g} gives {np.round(got[off], 6)}, "
        f"fs=1 gives {np.round(ref[off], 6)}"
    )


# --------------------------------------------------------------------------
# What the single-epoch path computes, and where its sign can be trusted
# --------------------------------------------------------------------------

_BANDS = ((0, 0.5), (0, 0.25), (0.25, 0.5))


def _delayed(lag, seed, t=500):
    """White x, and y[t] = x[t - lag] plus 10% noise: x (row 0) leads."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(t + lag)
    return np.vstack([x[lag:], x[:t] + 0.1 * rng.standard_normal(t)])


def test_wavelet_psi_mean_is_the_time_average_of_the_instantaneous_phase_slope():
    """With one epoch the wavelet coherency is a unit phasor at every (f, t).

    mne-connectivity averages the cross-spectrum over epochs and nothing else,
    so for a single recording ``C(f, t) = exp(i dphi(f, t))``, ``|C| = 1``, and
    its PSI at time t is ``sum_k sin(dphi(f_{k+1}, t) - dphi(f_k, t))`` over the
    interior points of the 10-point grid (its band is open, so both ends drop
    out). The ``mean`` variant is the time average of that. Written out here
    from MNE's Morlet wavelets and numpy's convolution.
    """
    from mne.time_frequency import morlet

    T = 300
    arr = _delayed(3, 0, T)
    z = Data(arr, dim_order="ps").to_numpy(squeeze=True)
    for lo, hi in _BANDS:
        freqs = np.linspace(max(lo, 5.0 / T), hi, 10)
        n_cycles = np.clip(0.95 * T * freqs / 1.59, 1.0, 7.0)
        wavelets = morlet(1.0, freqs, n_cycles=n_cycles, zero_mean=True)
        coef = np.array([[np.convolve(row, w, mode="same") for w in wavelets]
                         for row in z])
        dphi = np.angle(coef[0] * coef[1].conj())[1:-1]
        expected = np.mean(np.sum(np.sin(np.diff(dphi, axis=0)), axis=0))
        got = _psi(arr, fmin=lo, fmax=hi)
        assert got[0, 1] == pytest.approx(expected, rel=1e-9, abs=1e-12), (lo, hi)
        assert got[1, 0] == pytest.approx(-expected, rel=1e-9, abs=1e-12), (lo, hi)


def test_wavelet_psi_mean_sign_holds_for_short_lags_and_reverses_beyond_half_a_grid_step():
    """The lag range the class docstring states.

    The grid step is ``df = (fmax - fmin) / 9`` whatever T, so neighbouring
    phases differ by ``2 pi lag df`` and the sign turns over near
    ``lag = 1 / (2 df)``: 9.2 samples for [0, 0.5] at T=500, 18.75 for
    [0, 0.25]. In the round-3 simulation (50 white-noise seeds at T=500 and
    2000) every seed had the right sign for lags 1-5 in all three bands, and
    at lag 12 every seed had the wrong sign on [0, 0.5].
    """
    for seed in range(3):
        for lag in (1, 2, 3, 5):
            for lo, hi in _BANDS:
                A = _psi(_delayed(lag, seed), fmin=lo, fmax=hi)
                assert A[0, 1] > 0 > A[1, 0], (seed, lag, lo, hi)
        assert _psi(_delayed(12, seed), fmin=0, fmax=0.5)[0, 1] < 0
        assert _psi(_delayed(12, seed), fmin=0, fmax=0.25)[0, 1] > 0
        assert _psi(_delayed(20, seed), fmin=0, fmax=0.25)[0, 1] < 0


def test_wavelet_psi_max_is_noise_which_is_why_it_is_not_bundled():
    """The maximum over time of the instantaneous slope tends to its bound.

    The bound is 7, one per neighbouring pair of the eight interior frequencies.
    On independent white noise the maximum is positive both ways round and
    grows with T (4.4, 5.3, 5.8 on [0, 0.5] at T = 100, 500, 2000), and in the
    round-3 simulation it exceeded the median for a pair at lag 2 in 85-100% of
    200 independent pairs, in every band. The three ``max`` variants were
    removed from every bundled config; the class still accepts
    ``statistic="max"``.
    """
    from pyspi.calculator import bundled_configs, load_spis_from_yaml, resolve_config

    T = 2000
    for lo, hi in _BANDS:
        null = [_psi(np.random.default_rng(100 + s).standard_normal((2, T)),
                     fmin=lo, fmax=hi, statistic="max") for s in range(4)]
        coupled = [_psi(_delayed(2, s, T), fmin=lo, fmax=hi, statistic="max")
                   for s in range(4)]
        for A in null:
            assert A[0, 1] > 0 and A[1, 0] > 0, (lo, hi)
        assert (np.mean([A[0, 1] for A in null])
                > np.mean([A[0, 1] for A in coupled])), (lo, hi)

    for name in bundled_configs():
        spis = load_spis_from_yaml(resolve_config(name), quiet=True)
        bundled_max = sorted(k for k in spis if k.startswith("psi_wavelet_max"))
        assert not bundled_max, f"{name} bundles {bundled_max}"
