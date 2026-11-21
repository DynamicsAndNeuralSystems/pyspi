"""Automatic VAR order selection of parametric spectral Granger causality.

The search tries orders up to ``min(max_order, T - 1)``, the largest lag at
which the sample autocovariance has data, and stops at the first order where
BIC increases. A series on which BIC is still improving at that limit, or an
explicit ``order`` above ``T``, is not appropriate.
"""
import warnings

import nitime.algorithms as nt_alg
import nitime.utils as nt_utils
import numpy as np
import pytest

from pyspi.base import NotAppropriate
from pyspi.data import Data
from pyspi.statistics import spectral
from pyspi.statistics.spectral import _fit_bivariate_var

SEEDS = range(5)


def _white(seed, T):
    return np.random.default_rng(seed).standard_normal((2, T))


def _coupled_var1(seed, T):
    """Process 0 drives process 1 at lag 1; 20 samples of burn-in."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((2, T + 20))
    X = np.zeros((2, T + 20))
    for t in range(1, T + 20):
        X[0, t] = 0.7 * X[0, t - 1] + e[0, t]
        X[1, t] = 0.4 * X[1, t - 1] + 0.6 * X[0, t - 1] + e[1, t]
    return X[:, 20:]


def _fit_with_every_lag_up_to_max_order(x, max_order=50):
    """The search with the autocovariance computed to ``max_order`` whatever ``T``."""
    x = x - x.mean(axis=-1, keepdims=True)
    n_obs = x.shape[-1]
    R = np.stack([x[:, k:] @ x[:, :n_obs - k].T
                  for k in range(max_order + 1)]) / n_obs
    best = None
    for p in range(max_order + 1):
        coef, ecov = nt_alg.lwr_recursion(R[: p + 1])
        crit = nt_utils.bayesian_information_criterion(ecov, 2, p, 2 * n_obs)
        if best is not None and crit > best[0]:
            return best[1:]
        best = (crit, coef, ecov)
    raise ValueError("did not converge")


@pytest.mark.parametrize("make", [_white, _coupled_var1], ids=["white", "coupled_var1"])
@pytest.mark.parametrize("T", [10, 20, 30, 40, 49])
def test_a_bundled_automatic_order_sgc_gives_values_on_a_short_series(make, T):
    spi = spectral.SpectralGrangerCausality(method="parametric", fmin=0, fmax=0.5)
    assert spi.identifier == "sgc_parametric_mean_fs-1_fmin-1e-05_fmax-0-5_order-None"
    for seed in SEEDS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            A = spi.multivariate(Data(make(seed, T), dim_order="ps"))
        assert np.isfinite(A[~np.eye(2, dtype=bool)]).all(), (seed, A)


@pytest.mark.parametrize("make", [_white, _coupled_var1], ids=["white", "coupled_var1"])
@pytest.mark.parametrize("T", [50, 51, 64, 100, 257])
def test_the_fit_is_unchanged_where_the_series_supports_max_order(make, T):
    """At T >= 51 the autocovariance is the same; at T = 50 the search ends at
    order 49 instead of 50, which matters only if BIC were still improving
    there."""
    for seed in SEEDS:
        z = Data(make(seed, T), dim_order="ps").to_numpy(squeeze=True)
        coef, ecov = _fit_bivariate_var(z, None, 50)
        coef_all, ecov_all = _fit_with_every_lag_up_to_max_order(z, 50)
        assert np.array_equal(coef, coef_all) and np.array_equal(ecov, ecov_all)


def test_bic_still_improving_at_the_longest_supported_order_is_not_appropriate():
    z = _white(0, 3)
    with pytest.raises(NotAppropriate,
                       match="still improving at order 2, the largest that 3 "
                             "observations support"):
        _fit_bivariate_var(z, None, 50)


def test_an_explicit_order_above_the_series_length_is_not_appropriate():
    with pytest.raises(NotAppropriate,
                       match="order 20 needs autocovariances up to lag 20, but "
                             "the series has 12 observations"):
        _fit_bivariate_var(_white(0, 12), 20)
    _fit_bivariate_var(_white(0, 20), 20)
