"""Wavelet phase slope index (mne-connectivity, Morlet wavelets) against references.

Pins the single-epoch value, its sign convention and orientation, and the band's units.
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
    floor for `fmin` is five cycles over the record, `5 * fs / T` Hz.
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
    [0, 0.25].
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
    On independent white noise the maximum is positive both ways round, grows
    with T, and exceeds that of a pair coupled at lag 2. No bundled config
    includes the three ``max`` variants; the class still accepts
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


# --------------------------------------------------------------------------
# Epoch-averaged phase-lag measures on the shared 125-frequency grid
# --------------------------------------------------------------------------

def _morlet(f, n_cycles):
    """Zero-mean complex Morlet wavelet at fs = 1, out to five standard deviations.

    ``sigma_t = n_cycles / (2 pi f)``; the scale is irrelevant to the phase-lag
    measures, which depend on the cross-spectrum only through signs and ratios.
    """
    sigma_t = n_cycles / (2 * np.pi * f)
    t = np.arange(0.0, 5.0 * sigma_t)
    t = np.r_[-t[::-1], t[1:]]
    oscillation = np.exp(2j * np.pi * f * t) - np.exp(-2 * (np.pi * f * sigma_t) ** 2)
    return oscillation * np.exp(-t ** 2 / (2 * sigma_t ** 2))


def _imag_cross_spectrum(z, i, j, n_cycles=7.0):
    """``Im(W_i conj(W_j))`` per epoch, frequency and time on the shared grid."""
    freqs = np.linspace(0.2, 0.5, 125)

    def coef(x):
        return np.stack([np.convolve(x, _morlet(f, n_cycles), mode="same") for f in freqs])

    S = np.stack([coef(z[i, :, r]) * np.conj(coef(z[j, :, r])) for r in range(z.shape[2])])
    return freqs, S.imag


def _epochs(seed=0, n_epochs=8, t=150):
    """Three processes over several realisations: y follows x at lag 2, w is independent."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_epochs, t + 2))
    y = x[:, :-2] + 0.8 * rng.standard_normal((n_epochs, t))
    w = rng.standard_normal((n_epochs, t))
    return np.stack([x[:, 2:], y, w]).transpose(0, 2, 1)  # (process, time, epoch)


def _analysed(z):
    """The series the SPI sees: ``Data`` z-scores each process in each realisation."""
    return Data(z, dim_order="psr").to_numpy()


def _wavelet(cls_name, z, **kwargs):
    import pyspi.statistics.wavelet as wavelet

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        spi = getattr(wavelet, cls_name)(**kwargs)
        return spi.multivariate(Data(z, dim_order="psr"))


# The band stops short of fs/2: at the Nyquist frequency the Morlet coefficients
# of a real series are real, so Im S there is rounding noise.
_FMAX = 0.45


@pytest.mark.parametrize("cls_name, measure", [
    ("PhaseLagIndex", "pli"),
    ("DebiasedSquaredPhaseLagIndex", "dspli"),
    ("weighted_PhaseLagIndex", "wpli2"),
])
def test_wavelet_phase_lag_measures_match_the_definitions_on_seven_cycle_wavelets(cls_name, measure):
    """Vinck et al. (2011), from 7-cycle Morlet coefficients averaged over epochs.

    ``PLI = |mean sign(Im S)|``; the debiased squared PLI is
    ``(n PLI^2 - 1) / (n - 1)`` over the ``n`` epochs; the squared weighted PLI
    is ``(|sum Im S| / sum |Im S|)^2``. Each is averaged over time and the
    frequencies in the band.
    """
    z = _epochs()
    n = z.shape[2]
    A = _wavelet(cls_name, z, fmax=_FMAX)
    for i, j in [(1, 0), (2, 0), (2, 1)]:
        freqs, im = _imag_cross_spectrum(_analysed(z), i, j)
        band = freqs <= _FMAX
        im = im[:, band]
        pli = np.abs(np.mean(np.sign(im), axis=0))
        per_ft = {
            "pli": pli,
            "dspli": (n * pli ** 2 - 1) / (n - 1),
            "wpli2": (np.abs(im.sum(axis=0)) / np.abs(im).sum(axis=0)) ** 2,
        }[measure]
        expected = np.mean(per_ft)
        assert A[i, j] == pytest.approx(expected, rel=1e-9, abs=1e-12), (i, j)
        assert A[j, i] == A[i, j]


def test_the_weighted_wavelet_pli_is_the_square_of_the_unsquared_wpli():
    """``wspli`` squares mne-connectivity's ``wpli`` at each frequency and time."""
    from mne_connectivity import spectral_connectivity_epochs

    z = _epochs(seed=1)
    freqs = np.linspace(0.2, 0.5, 125)
    wpli = spectral_connectivity_epochs(
        np.moveaxis(_analysed(z), 2, 0), method="wpli", mode="cwt_morlet", sfreq=1,
        fmin=5 / z.shape[1], fmax=0.5, cwt_freqs=freqs,
        cwt_n_cycles=np.full(freqs.size, 7.0), verbose=False,
    ).get_data(output="dense")
    band = freqs <= _FMAX
    A = _wavelet("weighted_PhaseLagIndex", z, fmax=_FMAX)
    assert A[1, 0] == pytest.approx(np.mean(wpli[1, 0, band] ** 2), rel=1e-12)
    assert A[1, 0] < np.mean(wpli[1, 0, band])


def test_wavelet_cycle_count_is_capped_so_the_wavelet_fits_a_short_record():
    """A 7-cycle wavelet at 0.2 is 55 samples long; a 40-sample record gets fewer cycles.

    The cap is ``0.95 T f / 1.59`` cycles (MNE's Morlet spans about
    ``1.59 n_cycles / f`` samples), clipped to [1, 7], as for ``PhaseSlopeIndex``.
    """
    z = _epochs(t=40)
    freqs = np.linspace(0.2, 0.5, 125)
    n_cycles = np.clip(0.95 * 40 * freqs / 1.59, 1.0, 7.0)
    assert n_cycles[0] < 7 and n_cycles[-1] == 7

    def coef(x):
        return np.stack([np.convolve(x, _morlet(f, c), mode="same")
                         for f, c in zip(freqs, n_cycles)])

    zz = _analysed(z)
    im = np.stack([(coef(zz[1, :, r]) * np.conj(coef(zz[0, :, r]))).imag
                   for r in range(zz.shape[2])])[:, freqs <= _FMAX]
    expected = np.mean(np.abs(np.mean(np.sign(im), axis=0)))
    A = _wavelet("PhaseLagIndex", z, fmax=_FMAX)
    assert A[1, 0] == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_the_old_wavelet_dspli_class_name_warns_and_gives_identical_values():
    """``DebiasedSquaredWeightedPhaseLagIndex`` is a deprecated alias.

    The measure is the unweighted debiased squared PLI (identifier ``dspli``);
    the weighted one is ``debiased_weighted_squared_PhaseLagIndex``.
    """
    import pyspi.statistics.wavelet as wavelet

    with pytest.warns(DeprecationWarning, match="DebiasedSquaredPhaseLagIndex"):
        old = wavelet.DebiasedSquaredWeightedPhaseLagIndex()
    new = wavelet.DebiasedSquaredPhaseLagIndex()
    assert isinstance(old, wavelet.DebiasedSquaredPhaseLagIndex)
    assert old.identifier == new.identifier
    z = _epochs(seed=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        a = old.multivariate(Data(z, dim_order="psr"))
        b = new.multivariate(Data(z, dim_order="psr"))
    np.testing.assert_array_equal(a, b)


def test_no_bundled_config_uses_the_shared_wavelet_grid():
    """Only PhaseSlopeIndex, which builds its own grid, is bundled from this module."""
    import pyspi.statistics.wavelet as wavelet
    from pyspi.calculator import bundled_configs, load_spis_from_yaml, resolve_config

    for name in bundled_configs():
        for spi in load_spis_from_yaml(resolve_config(name), quiet=True).values():
            if isinstance(spi, wavelet.mne):
                assert isinstance(spi, wavelet.PhaseSlopeIndex), (name, spi.identifier)
