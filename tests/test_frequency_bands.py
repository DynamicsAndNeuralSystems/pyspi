"""Frequency bands of the spectral and wavelet SPIs: units, validation, empty bands.

The multitaper and wavelet classes take ``fs``, ``fmin`` and ``fmax``, with the
band in the units of ``fs``; ``fs`` only relabels the frequency axis. Each
checks ``0 <= fmin <= fmax <= fs / 2`` at construction and raises
``NotAppropriate`` when the band holds no frequency of the series' grid at
compute time.
"""
import re
import warnings

import numpy as np
import pytest

from pyspi.calculator import (
    Calculator,
    bundled_configs,
    load_spis_from_yaml,
    resolve_config,
)
from pyspi.base import NotAppropriate
from pyspi.data import Data
from pyspi.quality import Quality
from pyspi.statistics import spectral, wavelet


def _data(T=400, M=3, seed=0):
    """``M`` coupled processes: an AR(1) driver, a lagged copy and noise."""
    rng = np.random.default_rng(seed)
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = 0.6 * x[t - 1] + rng.standard_normal()
    rows = [x, np.roll(x, 2) + 0.5 * rng.standard_normal(T)]
    rows += [rng.standard_normal(T) for _ in range(M - 2)]
    return Data(np.array(rows), dim_order="ps")


# One constructor per family; each takes fs, fmin, fmax and statistic.
_CLASSES = {
    "cohmag": lambda **kw: spectral.CoherenceMagnitude(**kw),
    "sgc_nonparametric": lambda **kw: spectral.SpectralGrangerCausality(
        method="nonparametric", **kw),
    "sgc_parametric": lambda **kw: spectral.SpectralGrangerCausality(
        method="parametric", **kw),
    "gd": lambda **kw: spectral.GroupDelay(**{"statistic": "delay", **kw}),
    "psi_wavelet": lambda **kw: wavelet.PhaseSlopeIndex(**kw),
    "wtc": lambda **kw: wavelet.WaveletCoherence(**kw),
}

_INVALID = [
    (dict(fs=0), ValueError, "fs must be > 0"),
    (dict(fs=-250), ValueError, "fs must be > 0"),
    (dict(fs=np.inf), ValueError, "fs must be finite"),
    (dict(fs="250"), TypeError, "fs must be a positive real number"),
    (dict(fmin=-0.1), ValueError, r"fmin must be in \[0, fs / 2\] = \[0, 0.5\]"),
    (dict(fmin=np.nan), ValueError, "fmin must be finite"),
    (dict(fmax=0.9), ValueError, r"fmax must be in \[0, fs / 2\].*units of fs"),
    (dict(fmin=8, fmax=12), ValueError, "units of fs, which defaults to 1"),
    (dict(fs=250, fmax=200), ValueError, r"fmax must be in \[0, fs / 2\] = \[0, 125\]"),
    (dict(fmin=0.3, fmax=0.2), ValueError, "fmin must be <=? fmax"),
    (dict(fmax=True), TypeError, "fmax must be a real number"),
]


@pytest.mark.parametrize("make", _CLASSES.values(), ids=_CLASSES.keys())
@pytest.mark.parametrize("kwargs, error, message", _INVALID,
                         ids=[str(k) for k, _, _ in _INVALID])
def test_an_invalid_band_is_refused_at_construction(make, kwargs, error, message):
    with pytest.raises(error, match=message):
        make(**kwargs)


@pytest.mark.parametrize("make", _CLASSES.values(), ids=_CLASSES.keys())
def test_an_unknown_statistic_is_a_value_error_naming_the_valid_ones(make):
    with pytest.raises(ValueError, match="supports"):
        make(statistic="median")


@pytest.mark.parametrize("make", _CLASSES.values(), ids=_CLASSES.keys())
def test_nyquist_computed_in_floating_point_is_accepted(make):
    fs = 0.7
    fmax = 0.35 * (1 + 1e-12)
    assert fmax > fs / 2
    make(fs=fs, fmin=0.1, fmax=fmax)


def test_a_zero_width_band_is_refused_only_where_the_estimator_needs_width():
    # A single frequency is a valid closed band for the multitaper and
    # coherence classes (for wavelet coherence, one of its fixed grid
    # points, 65 / 256 here). The wavelet phase slope index spaces its
    # frequencies linearly from fmin to fmax, and the multitaper one needs two
    # frequencies a bandwidth apart, so both need fmin < fmax.
    spectral.CoherenceMagnitude(fmin=0.25, fmax=0.25)
    wavelet.WaveletCoherence(fmin=65 / 256, fmax=65 / 256)
    for cls in (wavelet.PhaseSlopeIndex, spectral.PhaseSlopeIndex):
        with pytest.raises(ValueError, match="fmin must be < fmax"):
            cls(fmin=0.25, fmax=0.25)


def test_spectral_gc_reads_fmin_zero_as_the_lowest_positive_frequency():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        zero = spectral.SpectralGrangerCausality(fmin=0, fmax=0.5)
    floor = spectral.SpectralGrangerCausality(fmin=1e-5, fmax=0.5)
    assert zero.identifier == floor.identifier == (
        "sgc_nonparametric_mean_fs-1_fmin-1e-05_fmax-0-5")
    d = _data()
    np.testing.assert_array_equal(zero.multivariate(d), floor.multivariate(d))
    with pytest.raises(ValueError, match="spectral GC is undefined at 0 Hz"):
        spectral.SpectralGrangerCausality(fmin=0, fmax=0)


# --------------------------------------------------------------------------
# A band with no frequency of the series' grid, at compute time
# --------------------------------------------------------------------------

@pytest.mark.parametrize("spi, detail", [
    # Grid step 1/T = 0.01 at T = 100: nothing in (0.101, 0.109).
    (spectral.CoherenceMagnitude(fmin=0.101, fmax=0.109), "no frequency of the series' grid"),
    (spectral.DirectedCoherence(fmin=0.101, fmax=0.109), "no frequency of the series' grid"),
    (spectral.SpectralGrangerCausality(fmin=0.101, fmax=0.109), "no frequency of the series' grid"),
    (spectral.PhaseLockingValue(fmin=0.101, fmax=0.109), "no frequency of the series' grid"),
    # At T = 100 these hold only 0 Hz and only fs / 2; longer series fill them.
    (spectral.PhaseLockingValue(fmin=0, fmax=0.004), "only the real-valued bins"),
    (spectral.GroupDelay(statistic="delay", fmin=0.495, fmax=0.5), "only the real-valued bins"),
    # 5 fs / T = 0.05 at T = 100.
    (wavelet.PhaseSlopeIndex(fmin=0, fmax=0.04), "no frequency of the series' grid"),
], ids=lambda x: getattr(x, "identifier", ""))
def test_an_empty_band_is_not_appropriate_when_computed(spi, detail):
    with pytest.raises(NotAppropriate, match=re.escape(
            f"{type(spi).__name__}: the band [{spi._fmin}, {spi._fmax}] holds {detail}")):
        spi.multivariate(_data(T=100))


def test_wavelet_coherence_empty_band_is_not_appropriate_when_computed():
    spi = wavelet.WaveletCoherence(fmin=0, fmax=0.01)
    with pytest.raises(NotAppropriate, match=re.escape(
            "WaveletCoherence: the band [0, 0.01] holds no frequency of the series' grid")):
        spi.multivariate(_data(T=100))


def test_calculator_records_an_empty_band_as_not_appropriate(tmp_path):
    config = tmp_path / "empty_band.yaml"
    config.write_text(
        ".statistics.spectral:\n"
        "  CoherenceMagnitude:\n"
        "    labels: [undirected]\n"
        "    configs:\n"
        "      - fmin: 0.101\n"
        "        fmax: 0.109\n"
        "      - fmin: 0\n"
        "        fmax: 0.5\n")
    d = _data(T=100)
    calc = Calculator(dataset=d.to_numpy(squeeze=True), config=str(config))
    with pytest.warns(UserWarning, match="is not appropriate for this dataset"):
        calc.compute()
    empty = "cohmag_multitaper_mean_fs-1_fmin-0-101_fmax-0-109"
    assert calc.not_appropriate[empty].startswith("CoherenceMagnitude: ")
    assert "holds no frequency of the series' grid" in calc.not_appropriate[empty]
    assert calc.errors == {}
    q = calc.quality[empty].to_numpy()
    off = ~np.eye(q.shape[0], dtype=bool)
    assert (q[off] == Quality.NAN).all()
    assert "cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-5" not in calc.not_appropriate


# --------------------------------------------------------------------------
# fs relabels the frequency axis and changes no value
# --------------------------------------------------------------------------

# Equal up to floating-point rounding; Wilson's factorisation, behind
# nonparametric spectral GC, iterates to a tolerance, so its agreement is looser.
@pytest.mark.parametrize("make, rtol", [
    (lambda **kw: spectral.CoherenceMagnitude(**kw), 1e-12),
    (lambda **kw: spectral.SpectralGrangerCausality(method="nonparametric", **kw), 1e-6),
    (lambda **kw: spectral.SpectralGrangerCausality(method="parametric", order=2, **kw), 1e-12),
    (lambda **kw: wavelet.PhaseSlopeIndex(**kw), 1e-12),
    (lambda **kw: wavelet.WaveletCoherence(**kw), 1e-12),
], ids=["cohmag", "sgc_nonparametric", "sgc_parametric", "psi_wavelet", "wtc"])
def test_a_band_in_hz_equals_the_band_in_cycles_per_sample(make, rtol):
    d = _data(T=1000)
    hz = make(fs=250, fmin=8, fmax=12).multivariate(d)
    normalised = make(fs=1, fmin=0.032, fmax=0.048).multivariate(d)
    assert np.isfinite(hz[~np.eye(3, dtype=bool)]).all()
    np.testing.assert_allclose(hz, normalised, rtol=rtol, atol=1e-15)


def test_a_band_in_hz_names_fs_in_the_identifier():
    assert (spectral.CoherenceMagnitude(fs=250, fmin=8, fmax=12).identifier
            == "cohmag_multitaper_mean_fs-250_fmin-8_fmax-12")


# --------------------------------------------------------------------------
# Bundled configs: no warning from a valid band
# --------------------------------------------------------------------------

_QUIET = re.compile(r"^(sgc_|cohmag_|plv_|dtf_|wtc_|psi_wavelet_)")


def test_bundled_spectral_spis_construct_and_compute_without_a_warning():
    spis = {}
    for name in bundled_configs():
        spis.update(load_spis_from_yaml(resolve_config(name), quiet=True))
    d = _data(T=200)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for name in bundled_configs():
            load_spis_from_yaml(resolve_config(name), quiet=True)
        chosen = [k for k in spis if _QUIET.match(k)]
        assert any(k.startswith("sgc_") for k in chosen)
        for key in chosen:
            spis[key].multivariate(d)
