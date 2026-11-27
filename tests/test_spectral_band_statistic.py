"""
The band statistic of a cell that is NaN at every in-band frequency.

Such a cell is NaN. NumPy's NaN-ignoring mean and max warn on an all-NaN
slice ("Mean of empty slice", "All-NaN slice encountered"), so before this
check spectral Granger causality and the other band-reduced spectral SPIs
emitted a RuntimeWarning for a value they already label not appropriate.
"""

import warnings

import numpy as np
import pytest

from pyspi.data import Data
from pyspi.statistics.spectral import CoherenceMagnitude, SpectralGrangerCausality


@pytest.mark.parametrize("statistic", ["mean", "max"])
def test_all_nan_cell_is_nan_without_a_warning(statistic):
    spi = CoherenceMagnitude(statistic=statistic)
    rng = np.random.default_rng(0)
    spectra = rng.random((6, 3, 3))
    spectra[:, 2, 0] = np.nan       # every in-band frequency
    spectra[2, 1, 2] = np.nan       # one frequency only
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        adj = spi._band_statistic(spectra)
    fn = np.nanmean if statistic == "mean" else np.nanmax
    expected = fn(spectra, axis=0)
    np.fill_diagonal(expected, np.nan)
    expected[2, 0] = np.nan
    np.testing.assert_array_equal(adj, expected)


@pytest.mark.parametrize("method", ["nonparametric", "parametric"])
@pytest.mark.parametrize("statistic", ["mean", "max"])
def test_spectral_gc_all_nan_pair_has_a_reason_and_no_runtime_warning(
        method, statistic):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((3, 200))
    X[1, 1:] += 0.6 * X[0, :-1]
    data = Data(X, dim_order="ps")
    spi = SpectralGrangerCausality(method=method, fmin=0, fmax=0.5,
                                   statistic=statistic)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        before = spi.multivariate(data)
    cached = data.spectral_gc[spi._getkey()]
    cached["F"] = cached["F"].copy()
    cached["F"][0, :, 2, 0] = np.nan   # F[0, f, target, source]: 0 -> 2
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        after = spi.multivariate(data)
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert np.isnan(after[0, 2])
    others = ~np.eye(3, dtype=bool)
    others[0, 2] = False
    np.testing.assert_array_equal(after[others], before[others])
