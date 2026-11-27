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


def test_lmfit_lasso_is_data_independent_which_is_why_it_is_not_bundled():
    """sklearn's ``Lasso(alpha=1)`` zeroes the slope whenever ``|x.y|/n < 1``.

    For z-scored ``x`` and ``y`` that is ``|r| (T-1)/T < 1``, always, so every
    entry is ``var(y) = (T-1)/T`` whatever the data. ``ElasticNet(alpha=1)``
    zeroes it unless ``|r| > 0.5``. Neither is in the bundled configs (the
    class still accepts them).
    """
    from pyspi.calculator import Calculator
    from pyspi.statistics.misc import LinearModel

    rng = np.random.default_rng(1)
    x = rng.standard_normal(200)
    independent = Data(np.vstack([x, rng.standard_normal(200)]), dim_order="ps")
    nearly_equal = Data(np.vstack([x, x + 0.05 * rng.standard_normal(200)]),
                        dim_order="ps")
    a = LinearModel("Lasso").multivariate(independent)[0, 1]
    b = LinearModel("Lasso").multivariate(nearly_equal)[0, 1]
    assert a == pytest.approx(b) == pytest.approx(199 / 200)

    bundled = set(Calculator(config="full").spis)
    assert not {"lmfit_Lasso", "lmfit_ElasticNet"} & bundled


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


def test_aeg_is_the_engle_granger_tstat_of_the_column_regressed_on_the_row():
    """``coint_aeg`` puts the regressor on the row, like ``lmfit_*``.

    ``statsmodels.coint(y0, y1)`` regresses ``y0`` on ``y1`` and unit-root-tests
    the residual, so ``table[i, j]`` must call ``coint(z[j], z[i])``.
    """
    from statsmodels.tsa.stattools import coint

    from pyspi.statistics.misc import Cointegration

    rng = np.random.default_rng(3)
    arr = rng.standard_normal((2, 200)).cumsum(axis=1)
    data = Data(arr, dim_order="ps")
    z = data.to_numpy(squeeze=True)
    for trend, autolag in (("c", "aic"), ("ct", "aic"), ("ct", "bic")):
        spi = Cointegration("aeg", "tstat", autolag=autolag, maxlag=10, trend=trend)
        A = spi.multivariate(data)
        assert A[0, 1] == coint(z[1], z[0], trend=trend, autolag=autolag, maxlag=10)[0]
        assert A[1, 0] == coint(z[0], z[1], trend=trend, autolag=autolag, maxlag=10)[0]
        assert A[0, 1] != A[1, 0]


def test_aeg_refuses_a_collinear_pair_that_statsmodels_scores_minus_inf():
    """statsmodels' ``coint`` returns ``-inf`` when the regression fits
    (almost) exactly; that pair is not appropriate, with a reason, and the
    others keep statsmodels' value."""
    from statsmodels.tsa.stattools import coint

    from pyspi.calculator import Calculator

    rng = np.random.default_rng(0)
    X = rng.standard_normal((3, 150))
    X[1] = X[0] + 1e-7 * rng.standard_normal(150)
    calc = Calculator(dataset=X, dim_order="ps", verbose=False, config="full")
    key = "coint_aeg_tstat_trend-c_autolag-aic_maxlag-10"
    calc._spis = {key: calc.spis[key]}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calc.compute()
    assert not [w for w in caught if "colinear" in str(w.message)]
    A = calc.table[key].to_numpy()
    assert np.isnan(A[0, 1]) and np.isnan(A[1, 0])
    refused = {(i, j) for i, j, _ in calc.pair_not_appropriate[key]}
    assert refused == {("proc-0", "proc-1"), ("proc-1", "proc-0")}
    assert all("collinear" in why for *_, why in calc.pair_not_appropriate[key])
    z = calc.dataset.to_numpy(squeeze=True)
    assert A[0, 2] == coint(z[2], z[0], trend="c", autolag="aic", maxlag=10)[0]
    assert key not in calc.errors


def test_aeg_and_lmfit_agree_on_which_index_is_the_regressor():
    """Both directed regression families regress the column on the row.

    Checked against the libraries directly rather than against each other:
    ``lmfit`` fits ``sklearn`` with ``X = z[i]`` and ``y = z[j]``, and ``aeg``
    calls ``coint(z[j], z[i])``, whose first argument is the regressand.
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error
    from statsmodels.tsa.stattools import coint

    from pyspi.statistics.misc import Cointegration, LinearModel

    rng = np.random.default_rng(5)
    x = rng.standard_normal(300).cumsum()
    y = 2.0 * x + rng.standard_normal(300)
    data = Data(np.vstack([x, y]), dim_order="ps")
    z = data.to_numpy(squeeze=True)
    cols = data.to_numpy()  # (M, T, 1): sklearn wants a 2-d design matrix

    lm = LinearModel("Ridge").multivariate(data)
    fitted = Ridge(random_state=42).fit(cols[0], cols[1].ravel())
    assert lm[0, 1] == pytest.approx(
        mean_squared_error(fitted.predict(cols[0]), cols[1].ravel()))

    aeg = Cointegration("aeg", "tstat", autolag="aic", maxlag=10, trend="c").multivariate(data)
    assert aeg[0, 1] == coint(z[1], z[0], trend="c", autolag="aic", maxlag=10)[0]


def test_cointegration_is_labelled_linear():
    """Johansen's VECM and Engle-Granger's OLS are linear models.

    Label merging lets the class's linearity win over the config's, so the
    class must say `linear` too for all eleven `coint_*` SPIs to be linear.
    """
    coint = _full_spis("coint_")
    assert len(coint) == 11
    wrong = sorted(k for k, v in coint.items()
                   if "linear" not in v.labels or "nonlinear" in v.labels)
    assert not wrong, f"cointegration SPIs not labelled linear: {wrong}"


# --------------------------------------------------------------------------
# PowerEnvelopeCorrelation
# --------------------------------------------------------------------------

def _anticorrelated_envelopes(t=2000):
    n = np.arange(t)
    env = 1.5 + np.sin(2 * np.pi * 0.004 * n)
    x = env * np.sin(2 * np.pi * 0.11 * n)
    y = (3.0 - env) * np.sin(2 * np.pi * 0.17 * n + 1.0)
    return np.vstack([x, y])


def test_pec_reports_the_envelope_correlation_with_its_sign():
    """Two carriers whose amplitude envelopes move in antiphase.

    The independent reference is the Pearson correlation of the Hilbert
    amplitude envelopes (of their logs for `log`), exactly -1 up to edge
    effects. A correlation coefficient carries its sign, so unless `absolute`
    is applied the SPI must report itself as signed (an unsigned SPI is
    min-shifted by `Calculator._rmmin` and passed through `abs()` by
    `set_group`, folding anticorrelated envelopes onto correlated ones).
    """
    from scipy.signal import hilbert

    from pyspi.statistics.misc import PowerEnvelopeCorrelation

    arr = _anticorrelated_envelopes()
    data = Data(arr, dim_order="ps")
    z = data.to_numpy(squeeze=True)
    amp = np.abs(hilbert(z, axis=-1))
    for log in (False, True):
        e = np.log(amp ** 2) if log else amp
        ref = np.corrcoef(e)[0, 1]
        got = PowerEnvelopeCorrelation(log=log).multivariate(data)[0, 1]
        assert got == pytest.approx(ref, abs=1e-9)
        assert got < -0.9

    for orth in (False, True):
        for log in (False, True):
            spi = PowerEnvelopeCorrelation(orth=orth, log=log)
            value = spi.multivariate(data)[0, 1]
            assert value < 0, (spi.identifier, value)
            assert spi.issigned(), f"{spi.identifier} = {value:.3f} reports unsigned"
    for log in (False, True):
        spi = PowerEnvelopeCorrelation(orth=True, log=log, absolute=True)
        assert spi.multivariate(data)[0, 1] >= 0
        assert not spi.issigned()


def test_bundled_pec_signedness_follows_absolute():
    pec = _full_spis("pec")
    assert len(pec) == 6
    for ident, spi in pec.items():
        expect_signed = not ident.endswith("_abs")
        assert spi.issigned() is expect_signed, ident
        assert ("signed" in spi.labels) is expect_signed, ident


# --------------------------------------------------------------------------
# GPModel
# --------------------------------------------------------------------------

def _gp_kernel(name):
    from sklearn.gaussian_process import kernels
    return kernels.ConstantKernel() + kernels.WhiteKernel() + getattr(kernels, name)()


@pytest.mark.parametrize("kernel", ["DotProduct", "RBF"])
def test_gpfit_is_the_leave_one_out_mse_of_the_fitted_gp(kernel):
    """`gpfit_*[i, j]` is the leave-one-out MSE of a GP predicting j from i.

    Reference by brute force: fit the GP once (hyperparameters by maximum
    marginal likelihood, sklearn defaults), then for every t refit the same
    kernel with the optimizer off on the other T-1 points and predict y_t.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from pyspi.statistics.misc import GPModel

    rng = np.random.default_rng(3)
    x = rng.standard_normal(80)
    y = np.sin(2 * x) + 0.5 * rng.standard_normal(80)
    data = Data(np.vstack([x, y]), dim_order="ps")
    z = data.to_numpy(squeeze=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp = GaussianProcessRegressor(kernel=_gp_kernel(kernel)).fit(z[0][:, None], z[1])
        resid = []
        for t in range(len(x)):
            keep = np.arange(len(x)) != t
            g = GaussianProcessRegressor(kernel=gp.kernel_, optimizer=None)
            g.fit(z[0][keep, None], z[1][keep])
            resid.append(z[1][t] - g.predict(z[0][[t], None])[0])
    assert GPModel(kernel).multivariate(data)[0, 1] == pytest.approx(
        np.mean(np.square(resid)), rel=1e-8)


def test_gpfit_dot_product_equals_sklearns_gp_fit():
    """The O(T) DotProduct fit equals `GaussianProcessRegressor`'s, to rounding.

    Reference: sklearn's fit of the same kernel and its leave-one-out
    residuals from the Cholesky factor, as `GPModel` computes them for
    other kernels. Covers linear, nonlinear, heavy-tailed, near-deterministic
    and unrelated pairs.
    """
    from scipy.linalg import solve_triangular
    from sklearn.gaussian_process import GaussianProcessRegressor
    from pyspi.statistics.misc import GPModel

    rng = np.random.default_rng(7)
    x = rng.standard_normal(200)
    e = rng.standard_normal(200)
    for y in (e, x + 0.6 * e, np.tanh(2 * x) + 0.3 * e, x ** 2,
              rng.standard_cauchy(200), x + 1e-3 * e):
        data = Data(np.vstack([x, y]), dim_order="ps")
        z = data.to_numpy(squeeze=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp = GaussianProcessRegressor(kernel=_gp_kernel("DotProduct")).fit(
                z[0][:, None], z[1])
        L_inv = solve_triangular(gp.L_, np.eye(200), lower=True)
        ref = np.mean((gp.alpha_ / np.einsum("ij,ij->j", L_inv, L_inv)) ** 2)
        assert GPModel("DotProduct").multivariate(data)[0, 1] == pytest.approx(
            ref, rel=1e-10, abs=1e-15)


def _exact_dot_product_gp(theta, x, y, alpha=1e-10):
    """Log marginal likelihood and leave-one-out MSE in exact arithmetic.

    ``K = c + sigma_0^2 + x x' + (noise + alpha) I`` at ``theta = log(c,
    noise, sigma_0)``, inverted by Gauss-Jordan elimination on `Fraction`s,
    so the only rounding is in the final conversion to float.
    """
    import math
    from fractions import Fraction

    c, noise, sigma_0 = (Fraction(float(v)) for v in np.exp(theta))
    xs = [Fraction(float(v)) for v in x]
    ys = [Fraction(float(v)) for v in y]
    T = len(ys)
    rows = [[c + sigma_0 ** 2 + xs[i] * xs[j]
             + (noise + Fraction(alpha) if i == j else 0) for j in range(T)]
            + [Fraction(int(i == j)) for j in range(T)] for i in range(T)]
    det = Fraction(1)
    for col in range(T):
        pivot = rows[col][col]
        det *= pivot
        rows[col] = [v / pivot for v in rows[col]]
        for r in range(T):
            if r != col and rows[r][col]:
                f = rows[r][col]
                rows[r] = [a - f * b for a, b in zip(rows[r], rows[col])]
    k_inv = [row[T:] for row in rows]
    a = [sum(k_inv[i][j] * ys[j] for j in range(T)) for i in range(T)]
    log_det = math.log(det.numerator) - math.log(det.denominator)
    lml = (-0.5 * float(sum(yi * ai for yi, ai in zip(ys, a)))
           - 0.5 * log_det - 0.5 * T * math.log(2 * math.pi))
    loo = float(sum((a[t] / k_inv[t][t]) ** 2 for t in range(T)) / T)
    return lml, loo


def test_gpfit_dot_product_is_exact_on_badly_scaled_data(monkeypatch):
    """Without z-scoring, the DotProduct fit is exact where sklearn's is not.

    With the predictor in units of 1e6 or 1e8, the T x T kernel matrix that
    sklearn factorises has entries of order 1e12 or more beside a noise term
    of order 1, so its Cholesky factor loses the noise term (or fails).
    Reference: exact rational arithmetic at the hyperparameters the fit
    returns. The value must be the exact leave-one-out MSE there, and those
    hyperparameters must reach an exact log marginal likelihood at least as
    high as that at sklearn's, wherever sklearn's fit succeeds.
    """
    from scipy import optimize
    from sklearn.gaussian_process import GaussianProcessRegressor
    from pyspi.statistics.misc import GPModel

    fitted = []
    minimize = optimize.minimize

    def recording_minimize(*args, **kwargs):
        res = minimize(*args, **kwargs)
        fitted.append(res.x.copy())
        return res

    monkeypatch.setattr(optimize, "minimize", recording_minimize)

    T = 12
    pairs = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        x, e = rng.standard_normal(T), rng.standard_normal(T)
        pairs.append((1e6 * (x + 0.5 * e), x))
    pairs.append((1e8 * x, 1e8 * (x + 0.5 * e)))
    for k, (u, v) in enumerate(pairs):
        data = Data(np.vstack([u, v]), dim_order="ps", zscore=False)
        z = data.to_numpy(squeeze=True)
        fitted.clear()
        value = GPModel("DotProduct").bivariate(data, i=0, j=1)
        lml, loo = _exact_dot_product_gp(fitted[0], z[0], z[1])
        assert value == pytest.approx(loo, rel=1e-10), k
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp = GaussianProcessRegressor(
                    kernel=_gp_kernel("DotProduct")).fit(z[0][:, None], z[1])
        except np.linalg.LinAlgError:
            continue
        lml_sklearn, _ = _exact_dot_product_gp(gp.kernel_.theta, z[0], z[1])
        assert lml >= lml_sklearn - 1e-9 * abs(lml_sklearn), k


def test_gpfit_rbf_equals_sklearns_gp_fit():
    """The RBF fit (sklearn's, with a cheaper gradient) equals sklearn's.

    Only the likelihood gradient is computed differently, so it differs from
    sklearn's in the last bits and the optimizer can stop elsewhere along a
    flat optimum. That matters only near interpolation (y a near-copy of a
    function of x, leave-one-out MSE ~1e-6), where sklearn's own value moves
    by up to ~1e-3 relative when the samples are merely reordered; those
    pairs are left out here. Covers linear, nonlinear, heavy-tailed,
    autocorrelated and unrelated pairs, and x in large units (`zscore=False`).
    """
    from scipy.linalg import solve_triangular
    from sklearn.gaussian_process import GaussianProcessRegressor
    from pyspi.statistics.misc import GPModel

    rng = np.random.default_rng(11)
    T = 150
    x = rng.standard_normal(T)
    e = rng.standard_normal(T)
    ar = np.zeros(T)
    for t in range(1, T):
        ar[t] = 0.9 * ar[t - 1] + e[t]
    pairs = [(x, e), (x, x + 0.6 * e), (x, np.tanh(2 * x) + 0.3 * e),
             (x, np.sin(3 * x) + 0.5 * e), (x, x ** 2 + 0.3 * e),
             (rng.standard_cauchy(T), e), (ar, np.roll(ar, 1) + 0.3 * e),
             (100.0 * x, e)]
    for k, (u, v) in enumerate(pairs):
        data = Data(np.vstack([u, v]), dim_order="ps", zscore=k < 7)
        z = data.to_numpy(squeeze=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp = GaussianProcessRegressor(kernel=_gp_kernel("RBF")).fit(
                z[0][:, None], z[1])
            value = GPModel("RBF").multivariate(data)[0, 1]
        L_inv = solve_triangular(gp.L_, np.eye(T), lower=True)
        ref = np.mean((gp.alpha_ / np.einsum("ij,ij->j", L_inv, L_inv)) ** 2)
        assert value == pytest.approx(ref, rel=1e-8), k


def test_gpfit_rbf_does_not_interpolate_an_unrelated_pair():
    """An independent pair must score ~var(y) (nothing to predict), not ~0.

    The kernel's RBF term has unit amplitude, so at a vanishing length scale it
    is white noise of variance 1 and the likelihood cannot tell it from the
    WhiteKernel. That solution interpolates y: its in-sample prediction is y
    itself, so its in-sample MSE is ~0. With x in large units (zscore=False)
    the optimizer starts inside it; its leave-one-out prediction is ~0, so the
    leave-one-out MSE is ~var(y).
    """
    from pyspi.statistics.misc import GPModel

    for seed in range(3):
        rng = np.random.default_rng(seed)
        x = 100.0 * rng.standard_normal(300)
        y = rng.standard_normal(300)
        data = Data(np.vstack([x, y]), dim_order="ps", zscore=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            value = GPModel("RBF").multivariate(data)[0, 1]
        assert value > 0.9 * y.var(), (seed, value, y.var())


@pytest.mark.parametrize("kernel", ["DotProduct", "RBF"])
def test_gpfit_recovers_the_noise_variance_and_the_null(kernel):
    """Known answers on z-scored data (T=300).

    y = x + 0.6 e: the best predictor's MSE is the noise variance after
    z-scoring, 0.36 / 1.36 = 0.265, for both kernels. Independent pair:
    (T-1)/T = 0.997.
    """
    from pyspi.statistics.misc import GPModel

    sig, nul = [], []
    for seed in range(5):
        rng = np.random.default_rng([seed, 11])
        x = rng.standard_normal(300)
        e = rng.standard_normal(300)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sig.append(GPModel(kernel).multivariate(
                Data(np.vstack([x, x + 0.6 * e]), dim_order="ps"))[0, 1])
            nul.append(GPModel(kernel).multivariate(
                Data(np.vstack([x, rng.standard_normal(300)]), dim_order="ps"))[0, 1])
    assert np.mean(sig) == pytest.approx(0.36 / 1.36, abs=0.03)
    assert np.mean(nul) == pytest.approx(299 / 300, abs=0.02)
