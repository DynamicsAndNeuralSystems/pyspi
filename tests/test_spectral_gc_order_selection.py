"""The series lengths and VAR orders of parametric spectral Granger causality.

An order ``p`` fit needs ``max(20, 3p + 1)`` observations: more observations
than each equation has coefficients, and at least 20. The automatic search
tries orders up to ``min(max_order, (T - 1) // 3)`` and stops at the first
order where BIC increases. A shorter series, or one on which BIC is still
improving at the last order tried, is not appropriate.
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
@pytest.mark.parametrize("T", [20, 30, 40, 49])
def test_a_bundled_automatic_order_sgc_gives_values_from_20_observations(make, T):
    spi = spectral.SpectralGrangerCausality(method="parametric", fmin=0, fmax=0.5)
    assert spi.identifier == "sgc_parametric_mean_fs-1_fmin-1e-05_fmax-0-5_order-None"
    for seed in SEEDS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            A = spi.multivariate(Data(make(seed, T), dim_order="ps"))
        assert np.isfinite(A[~np.eye(2, dtype=bool)]).all(), (seed, A)


@pytest.mark.parametrize("order", [None, 1, 6])
@pytest.mark.parametrize("T", [3, 10, 19])
def test_fewer_than_20_observations_is_not_appropriate(order, T):
    with pytest.raises(NotAppropriate,
                       match=f"at least 20 observations, but the series has {T}"):
        _fit_bivariate_var(_coupled_var1(0, T), order)


@pytest.mark.parametrize("order, shortest", [(1, 20), (6, 20), (7, 22), (20, 61)])
def test_an_explicit_order_needs_more_observations_than_coefficients(order, shortest):
    """``max(20, 3p + 1)`` observations: ``T - p`` of them fit ``2p`` coefficients."""
    assert spectral._var_min_observations(order) == shortest
    with pytest.raises(NotAppropriate,
                       match=f"order {order} needs at least {shortest} "
                             f"observations, but the series has {shortest - 1}"):
        _fit_bivariate_var(_white(0, shortest - 1), order)
    for seed in SEEDS:
        coef, ecov = _fit_bivariate_var(_white(seed, shortest), order)
        assert coef.shape[0] == order and np.isfinite(ecov).all()
        assert np.linalg.cond(ecov) < 1e6


@pytest.mark.parametrize("kwargs, error", [
    (dict(order=-1), ValueError), (dict(order=2.5), TypeError),
    (dict(max_order=0), ValueError), (dict(max_order=True), TypeError)])
def test_an_order_that_is_not_a_count_is_refused_at_construction(kwargs, error):
    with pytest.raises(error, match="order must be"):
        spectral.SpectralGrangerCausality(method="parametric", **kwargs)
    spectral.SpectralGrangerCausality(method="parametric", order=0, max_order=1)


@pytest.mark.parametrize("make", [_white, _coupled_var1], ids=["white", "coupled_var1"])
@pytest.mark.parametrize("T", [20, 31, 50, 64, 100, 150, 151, 257])
def test_the_search_selects_the_order_the_full_search_does(make, T):
    """The cap ends the search early only if BIC is still improving at it."""
    for seed in SEEDS:
        z = Data(make(seed, T), dim_order="ps").to_numpy(squeeze=True)
        coef, ecov = _fit_bivariate_var(z, None, 50)
        coef_all, ecov_all = _fit_with_every_lag_up_to_max_order(z, min(50, T - 1))
        assert np.array_equal(coef, coef_all) and np.array_equal(ecov, ecov_all)


@pytest.mark.parametrize("T, last", [(20, 6), (22, 7), (100, 33), (150, 49)])
def test_bic_still_improving_at_the_last_order_tried_is_not_appropriate(
        monkeypatch, T, last):
    monkeypatch.setattr(spectral.nt_utils, "bayesian_information_criterion",
                        lambda ecov, n_process, order, n_total: -order)
    with pytest.raises(NotAppropriate,
                       match=f"still improving at order {last}, the largest "
                             f"that {T} observations support"):
        _fit_bivariate_var(_white(0, T), None, 50)
    with pytest.raises(NotAppropriate, match="did not converge at max_order = 50"):
        _fit_bivariate_var(_white(0, 151), None, 50)
    with pytest.raises(NotAppropriate, match="did not converge at max_order = 4"):
        _fit_bivariate_var(_white(0, T), None, 4)
