"""statistics/misc.py against closed forms and independent references."""
import warnings

import numpy as np
import pytest

from pyspi.calculator import load_spis_from_yaml, resolve_config
from pyspi.data import Data


def _full_spis(prefix):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spis = load_spis_from_yaml(resolve_config("full"), quiet=True)
    return {k: v for k, v in spis.items() if k.startswith(prefix)}


# --------------------------------------------------------------------------
# LinearModel
# --------------------------------------------------------------------------

def _ridge_mse(x, y, alpha=1.0):
    """In-sample MSE of sklearn's Ridge(alpha) with an intercept, in closed form."""
    xc, yc = x - x.mean(), y - y.mean()
    w = (xc @ yc) / (xc @ xc + alpha)
    return np.mean((yc - w * xc) ** 2)


def test_lmfit_is_the_in_sample_mse_of_predicting_the_column_from_the_row():
    """`lmfit_Ridge[i, j]` regresses process j on process i (row = predictor).

    Unscaled data with unequal variances make the two orientations differ, so
    a transposed result cannot pass.
    """
    from pyspi.statistics.misc import LinearModel

    rng = np.random.default_rng(0)
    x = rng.standard_normal(300)
    y = 3.0 * x + 2.0 * rng.standard_normal(300)
    data = Data(np.vstack([x, y]), dim_order="ps", zscore=False)
    A = LinearModel("Ridge").multivariate(data)
    assert A[0, 1] == pytest.approx(_ridge_mse(x, y), rel=1e-10)
    assert A[1, 0] == pytest.approx(_ridge_mse(y, x), rel=1e-10)


@pytest.mark.xfail(strict=True, reason=(
    "lmfit_Lasso is data-independent under the default z-scoring: sklearn's "
    "Lasso(alpha=1) zeroes the slope whenever |x.y|/n < 1, which for z-scored "
    "x and y is |r|(T-1)/T < 1, always. Every entry is var(y) = (T-1)/T. "
    "Unsalvageable without choosing alpha; see the audit report."))
def test_lmfit_lasso_depends_on_the_data():
    from pyspi.statistics.misc import LinearModel

    rng = np.random.default_rng(1)
    x = rng.standard_normal(200)
    independent = Data(np.vstack([x, rng.standard_normal(200)]), dim_order="ps")
    nearly_equal = Data(np.vstack([x, x + 0.05 * rng.standard_normal(200)]),
                        dim_order="ps")
    a = LinearModel("Lasso").multivariate(independent)[0, 1]
    b = LinearModel("Lasso").multivariate(nearly_equal)[0, 1]
    assert not np.isclose(a, b, rtol=1e-9, atol=1e-12), (
        f"Lasso MSE is {a!r} for independent series and {b!r} for r~0.999")


# --------------------------------------------------------------------------
# Cointegration
# --------------------------------------------------------------------------

def _johansen_textbook(x, k_ar_diff):
    """Johansen's reduced-rank regression with an unrestricted constant.

    R0 and R1 are the residuals of dx_t and x_{t-1} on [1, dx_{t-1..t-k}];
    the eigenvalues solve |lambda S11 - S10 S00^-1 S01| = 0 (Johansen 1988;
    Luetkepohl 2005, sec. 8.2). Returns (trace, max-eig) for H0: rank 0.
    """
    dx = np.diff(x, axis=0)
    n = dx.shape[0] - k_ar_diff
    lags = [dx[k_ar_diff - l:k_ar_diff - l + n] for l in range(1, k_ar_diff + 1)]
    W = np.column_stack([np.ones(n)] + lags)
    resid = lambda Y: Y - W @ np.linalg.lstsq(W, Y, rcond=None)[0]
    R0 = resid(dx[k_ar_diff:])
    R1 = resid(x[k_ar_diff:-1])
    S00, S01, S11 = R0.T @ R0 / n, R0.T @ R1 / n, R1.T @ R1 / n
    lam = np.sort(np.linalg.eigvals(
        np.linalg.solve(S11, S01.T @ np.linalg.solve(S00, S01))).real)[::-1]
    return -n * np.sum(np.log(1 - lam)), -n * np.log(1 - lam[0])


def test_johansen_reports_the_rank_zero_trace_and_max_eigenvalue_statistics():
    """Against a textbook Johansen for det_order=0 (constant), both lag orders.

    That pins the reported numbers as the r=0 test statistics (not p-values or
    critical values) and shows `k_ar_diff` reaches statsmodels. det_order=1 is
    statsmodels' LeSage-style linear detrending of the levels, which has no
    textbook twin; there the check is that it differs from det_order=0.
    """
    from pyspi.statistics.misc import Cointegration

    rng = np.random.default_rng(2)
    common = rng.standard_normal(250).cumsum()
    arr = np.vstack([common + rng.standard_normal(250),
                     0.5 * common + rng.standard_normal(250),
                     rng.standard_normal(250).cumsum()])
    data = Data(arr, dim_order="ps")
    z = data.to_numpy(squeeze=True)
    for k_ar_diff in (1, 10):
        trace = Cointegration("johansen", "trace_stat", 0, k_ar_diff).multivariate(data)
        maxeig = Cointegration("johansen", "max_eig_stat", 0, k_ar_diff).multivariate(data)
        for i, j in ((0, 1), (0, 2), (1, 2)):
            ref_trace, ref_max = _johansen_textbook(z[[i, j]].T, k_ar_diff)
            assert trace[i, j] == pytest.approx(ref_trace, rel=1e-9)
            assert maxeig[i, j] == pytest.approx(ref_max, rel=1e-9)
            assert trace[j, i] == trace[i, j] and maxeig[j, i] == maxeig[i, j]
        lin = Cointegration("johansen", "trace_stat", 1, k_ar_diff).multivariate(data)
        off = ~np.eye(3, dtype=bool)
        assert not np.allclose(lin[off], trace[off])


def test_aeg_is_the_engle_granger_tstat_of_the_row_on_the_column():
    from statsmodels.tsa.stattools import coint

    from pyspi.statistics.misc import Cointegration

    rng = np.random.default_rng(3)
    arr = rng.standard_normal((2, 200)).cumsum(axis=1)
    data = Data(arr, dim_order="ps")
    z = data.to_numpy(squeeze=True)
    for trend, autolag in (("c", "aic"), ("ct", "aic"), ("ct", "bic")):
        spi = Cointegration("aeg", "tstat", autolag=autolag, maxlag=10, trend=trend)
        A = spi.multivariate(data)
        assert A[0, 1] == coint(z[0], z[1], trend=trend, autolag=autolag, maxlag=10)[0]
        assert A[1, 0] == coint(z[1], z[0], trend=trend, autolag=autolag, maxlag=10)[0]
