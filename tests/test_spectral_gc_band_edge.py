"""Spectral GC's band from ``fmin=0`` starts at ``fs / 1e5``, whatever ``fs`` is.

Spectral GC is undefined at 0 Hz, so ``fmin=0`` sets the lower edge just above
it, at a fixed fraction of ``fs``. Rescaling ``fs`` (with the band) therefore
gives the same value, and at the bundled ``fs=1`` the edge is 1e-5, which the
bundled identifiers name.
"""
import warnings

import numpy as np
import pytest

from pyspi.data import Data
from pyspi.statistics.spectral import SpectralGrangerCausality


def _coupled_pair(T=500, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((2, T + 100))
    x = np.zeros((2, T + 100))
    for t in range(1, T + 100):
        x[0, t] = 0.5 * x[0, t - 1] + e[0, t]
        x[1, t] = 0.3 * x[1, t - 1] + 0.7 * x[0, t - 1] + e[1, t]
    return x[:, 100:]


# The parametric fit does not see fs. fs sets the scale of the multitaper
# spectrum, and Wilson's iteration, which stops at a tolerance of 1e-8, stops
# at slightly different factors for different scales.
RTOL = {"parametric": 1e-12, "nonparametric": 1e-8}


@pytest.mark.parametrize("method", ["parametric", "nonparametric"])
def test_rescaling_fs_leaves_the_whole_band_value_unchanged(method):
    data = Data(_coupled_pair(), dim_order="ps")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want = SpectralGrangerCausality(method=method, fmin=0).multivariate(data)
        # 1 / 86400 is daily sampling with fs in Hz.
        for fs in (1000, 1e-3, 1e-4, 4e-5, 1 / 86400):
            got = SpectralGrangerCausality(method=method, fs=fs,
                                           fmin=0).multivariate(data)
            np.testing.assert_allclose(got, want, rtol=RTOL[method],
                                       err_msg=f"fs={fs}")


@pytest.mark.parametrize("method", ["parametric", "nonparametric"])
def test_a_sub_band_rescaled_with_fs_gives_the_same_value(method):
    data = Data(_coupled_pair(), dim_order="ps")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want = SpectralGrangerCausality(method=method, fmin=0,
                                        fmax=0.25).multivariate(data)
        got = SpectralGrangerCausality(method=method, fs=1e-4, fmin=0,
                                       fmax=0.25e-4).multivariate(data)
    np.testing.assert_allclose(got, want, rtol=RTOL[method])


def test_the_edge_at_fs_1_is_1e_5_and_the_identifier_names_it():
    zero = SpectralGrangerCausality(fmin=0, fmax=0.5)
    assert zero._fmin == 1e-5
    assert zero.identifier == "sgc_nonparametric_mean_fs-1_fmin-1e-05_fmax-0-5"
    assert SpectralGrangerCausality().identifier == zero.identifier
    parametric = SpectralGrangerCausality(method="parametric", fmin=0,
                                          fmax=0.25, order=1)
    assert parametric.identifier == (
        "sgc_parametric_mean_fs-1_fmin-1e-05_fmax-0-25_order-1")


@pytest.mark.parametrize("fs, edge", [(1000, "0-01"), (250, "0-0025"),
                                      (1e-4, "1e-09")])
def test_the_identifier_names_the_edge_used_at_other_fs(fs, edge):
    spi = SpectralGrangerCausality(fs=fs, fmin=0)
    assert spi._fmin == fs / 1e5
    assert f"_fmin-{edge}_" in spi.identifier


def test_a_band_ending_below_the_edge_is_refused_with_the_numbers():
    with pytest.raises(ValueError, match=r"fs / 1e5 = 0\.01 .*got 0\.005"):
        SpectralGrangerCausality(fs=1000, fmin=0, fmax=0.005)
