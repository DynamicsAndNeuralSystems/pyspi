"""Spectral and wavelet SPIs refuse input that cannot support the estimate.

A band that holds no frequency of the series' grid, or a series too short for
the tapers, segments, wavelets, bands or VAR orders an SPI needs, raises
``pyspi.base.NotAppropriate``. The Calculator stores NaN with ``Quality.NAN``
and records the reason in ``calc.errors``, prefixed ``"NotAppropriate: "``.
A band that no series could fill is a bad parameter, refused with
``ValueError`` at construction instead.
"""
import warnings

import numpy as np
import pytest
import yaml

from pyspi.base import NotAppropriate
from pyspi.calculator import Calculator
from pyspi.data import Data
from pyspi.quality import Quality
from pyspi.statistics import spectral, wavelet

# (module, class, kwargs, T, reason): each a refusal on a series of length T.
CASES = [
    ("spectral", "CoherenceMagnitude", dict(fmin=0.101, fmax=0.109), 100,
     "holds no frequency of the series' grid"),
    ("spectral", "PhaseLockingValue", dict(fmin=0, fmax=0.004), 100,
     "holds only the real-valued bins"),
    ("spectral", "DirectedCoherence", dict(fmin=0.101, fmax=0.109), 100,
     "holds no frequency of the series' grid"),
    ("spectral", "SpectralGrangerCausality", dict(fmin=0.101, fmax=0.109), 100,
     "holds no frequency of the series' grid"),
    ("spectral", "GroupDelay", dict(statistic="delay", fmin=0.495, fmax=0.5), 100,
     "holds only the real-valued bins"),
    ("spectral", "PhaseSlopeIndex", dict(fmin=0.1, fmax=0.12), 100,
     "needs at least two independent frequencies"),
    ("spectral", "SpectralGrangerCausality", dict(method="parametric"), 3,
     "still improving at order 0, the largest that 3 observations support"),
    ("spectral", "SpectralGrangerCausality", dict(method="parametric", order=20), 12,
     "order 20 needs at least 23 observations, but the series has 12"),
    ("spectral", "PhaseAlignedCorrelation", dict(nperseg=128), 100,
     "nperseg=128 exceeds the series length T=100"),
    ("spectral", "PhaseAlignedCorrelation", {}, 12,
     "needs T >= 20 for 4 segments"),
    ("spectral", "AdaptivePhaseConsistency", {}, 40,
     "no band holds 6 segments in a series of 40 samples"),
    ("wavelet", "PhaseSlopeIndex", dict(fmin=0, fmax=0.04), 100,
     "holds no frequency of the series' grid (the lowest it resolves"),
    ("wavelet", "WaveletCoherence", dict(fmin=0, fmax=0.01), 100,
     "holds no frequency of the series' grid"),
    ("wavelet", "WaveletCoherence", {}, 8,
     "a series of 8 samples is too short for 6-cycle wavelets"),
]


def _data(T, M=3):
    return Data(np.random.default_rng(T).standard_normal((M, T)), dim_order="ps")


def _spi(module, cls, kwargs):
    return getattr({"spectral": spectral, "wavelet": wavelet}[module], cls)(**kwargs)


def _id(case):
    module, cls, kwargs, T, _ = case
    return f"{module}.{cls}({','.join(f'{k}={v}' for k, v in kwargs.items())})_T{T}"


@pytest.mark.parametrize("case", CASES, ids=[_id(c) for c in CASES])
def test_a_direct_call_raises_not_appropriate(case):
    module, cls, kwargs, T, reason = case
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(NotAppropriate, match=reason.replace("(", r"\(")):
            _spi(module, cls, kwargs).multivariate(_data(T))


@pytest.mark.parametrize("case", CASES, ids=[_id(c) for c in CASES])
def test_the_calculator_records_nan_with_the_reason(case, tmp_path):
    module, cls, kwargs, T, reason = case
    config = tmp_path / "refusal.yaml"
    config.write_text(yaml.safe_dump(
        {f".statistics.{module}": {cls: {"labels": [], "configs": [kwargs]}}}))
    calc = Calculator(dataset=_data(T), config=str(config))
    (key,) = calc.spis
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calc.compute(progress=False)

    assert calc.errors[key].startswith("NotAppropriate: ")
    assert reason in calc.errors[key]
    q = calc.quality[key].to_numpy()
    assert (q[~np.eye(q.shape[0], dtype=bool)] == Quality.NAN).all()
    assert np.isnan(calc.table[key].to_numpy(dtype=float)).all()


def test_order_selection_that_does_not_turn_over_is_not_appropriate(tmp_path):
    """An MA(1) pair has no finite AR order, so BIC improves up to max_order=3."""
    e = np.random.default_rng(0).standard_normal((2, 2001))
    X = e[:, 1:] + 0.9 * e[:, :-1]
    config = tmp_path / "sgc.yaml"
    config.write_text(yaml.safe_dump({".statistics.spectral": {
        "SpectralGrangerCausality": {"labels": [], "configs": [
            dict(method="parametric", max_order=3)]}}}))
    calc = Calculator(dataset=Data(X, dim_order="ps"), config=str(config))
    (key,) = calc.spis
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calc.compute(progress=False)
    assert calc.errors[key].startswith(
        "NotAppropriate: Parametric spectral Granger causality: automatic AR "
        "order selection did not converge at max_order=3 on 2000 observations")
    q = calc.quality[key].to_numpy()
    assert (q[~np.eye(2, dtype=bool)] == Quality.NAN).all()


def test_a_band_off_the_parametric_grid_is_refused_at_construction():
    """The parametric method's grid, k fs / 1026, does not depend on the series."""
    with pytest.raises(ValueError, match="holds no frequency of the parametric method's grid"):
        spectral.SpectralGrangerCausality(method="parametric", fmin=0.4995, fmax=0.4999)
    spectral.SpectralGrangerCausality(method="parametric", fmin=0.498, fmax=0.4995)
    spectral.SpectralGrangerCausality(fmin=0.4995, fmax=0.4999)


@pytest.mark.parametrize("make, message", [
    (lambda: spectral.PhaseLockingValue(fmin=0, fmax=0), r"holds only 0 Hz"),
    (lambda: spectral.GroupDelay(statistic="delay", fmin=0.5, fmax=0.5),
     r"holds only fs / 2"),
    (lambda: spectral.PhaseSlopeIndex(fmin=0.2, fmax=0.2), "fmin must be < fmax"),
    (lambda: wavelet.WaveletCoherence(fmin=0.098, fmax=0.105),
     "holds none of the wavelet grid"),
    (lambda: wavelet.WaveletCoherence(fmin=0, fmax=0.003),
     "holds none of the wavelet grid"),
], ids=["plv-0-0", "gd-nyquist", "psi-zero-width", "wtc-between-grid-points",
        "wtc-below-grid"])
def test_a_band_empty_at_every_length_is_refused_at_construction(make, message):
    with pytest.raises(ValueError, match=message):
        make()


def test_bands_that_longer_series_fill_are_computed_there():
    """The replacement cases above are refused at T = 100 only for want of length."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for spi in (spectral.PhaseLockingValue(fmin=0, fmax=0.004),
                    spectral.GroupDelay(statistic="delay", fmin=0.495, fmax=0.5)):
            spi.multivariate(_data(1000))


def test_a_long_enough_series_is_still_computed():
    """The length checks refuse only what the estimators cannot compute."""
    for spi, T in ((spectral.SpectralGrangerCausality(method="parametric"), 10),
                   (spectral.SpectralGrangerCausality(method="parametric", order=20), 23),
                   (spectral.AdaptivePhaseConsistency(), 46),
                   (spectral.PhaseAlignedCorrelation(), 20)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            A = spi.multivariate(_data(T))
        assert np.isfinite(A[~np.eye(3, dtype=bool)]).any(), (spi.identifier, T)
