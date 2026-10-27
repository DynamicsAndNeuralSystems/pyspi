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
