import inspect
import warnings
import sklearn.covariance as cov
from scipy import stats, signal
import numpy as np
import pandas as pd

from pyspi.base import (
    Directed,
    Undirected,
    Signed,
    Unsigned,
    parse_bivariate,
    parse_multivariate,
)
from pyspi.utils import require_int


def _spearman_matrix(Z):
    """Spearman's rho between the rows of ``Z``: Pearson on average ranks.

    Same values as ``stats.spearmanr(Z, axis=1)``, but always a matrix, and a
    constant row gives NaN only in its own row and column.
    """
    return np.corrcoef(np.apply_along_axis(stats.rankdata, 1, Z))


class Estimators(Undirected, Signed):
    """Base class for (functional) connectivity-based statistics

    Information on covariance estimators at: https://scikit-learn.org/stable/modules/covariance.html

    When cross-validation selects an ``alpha`` near 0, ``GraphicalLassoCV`` may
    not converge (silently, as fit warnings are suppressed); on data with a
    dense precision matrix, prefer ``GraphicalLasso`` or ``LedoitWolf``.
    """

    name = "Covariance"
    labels = ["basic", "unordered", "linear", "undirected"]
    _cache_namespace = "covariance"

    @property
    def _cache_subkey(self):
        # Cache is data.covariance[estimator]; (kind, squared) are post-lookup.
        return (self._estimator,)

    def __init__(self, kind, estimator="EmpiricalCovariance", squared=False):
        paramstr = f"_{estimator}"
        if squared:
            paramstr = "-sq" + paramstr
            self.labels = Estimators.labels + ["unsigned"]
            self.issigned = lambda: False
        else:
            self.labels = Estimators.labels + ["signed"]
        self.identifier = self.identifier + paramstr
        self._squared = squared
        self._estimator = estimator
        self._kind = kind

    def _from_cache(self, data):
        try:
            mycov = data.covariance[self._estimator]
        except (AttributeError, KeyError):
            z = data.to_numpy(squeeze=True).T

            estimator = getattr(cov, self._estimator)
            # MinCovDet and EllipticEnvelope search from random subsets of the
            # observations, so each is given its own fixed-seed generator:
            # drawn from the global NumPy stream, the fit (and all four SPIs
            # read from it) would change from run to run.
            params = inspect.signature(estimator).parameters
            kwargs = {"random_state": 42} if "random_state" in params else {}
            # GraphicalLasso(CV) are converged tightly so the result is the
            # solution, not the solver path: at the default tol=1e-4 the value
            # depends on the path, which differs between scikit-learn versions.
            if "enet_tol" in params:
                kwargs.update(tol=1e-8, enet_tol=1e-8)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mycov = estimator(**kwargs).fit(z)
            try:
                data.covariance[self._estimator] = mycov
            except AttributeError:
                data.covariance = {self._estimator: mycov}
        return mycov

    def _matrix(self, mycov):
        """Copy of the cached fit's ``M x M`` matrix, so the cache is not modified."""
        return getattr(mycov, self._kind + "_").copy()

    @parse_multivariate
    def multivariate(self, data):
        matrix = self._matrix(self._from_cache(data))
        np.fill_diagonal(matrix, np.nan)
        if self._squared:
            return np.square(matrix)
        else:
            return matrix


class Covariance(Estimators):

    name = "Covariance"
    identifier = "cov"

    def __init__(self, estimator="EmpiricalCovariance", squared=False):
        super().__init__(kind="covariance", squared=squared, estimator=estimator)


class Precision(Estimators):

    name = "Precision"
    identifier = "prec"

    def __init__(self, estimator="EmpiricalCovariance", squared=False):
        super().__init__(kind="precision", squared=squared, estimator=estimator)


class PartialCorrelation(Estimators):
    """Partial correlation (Marrelec et al. 2006, NeuroImage 32, 228-237).

    With ``P`` the precision (inverse covariance) matrix of the ``M`` processes
    as estimated by the scikit-learn covariance ``estimator``,
    ``pcorr_ij = -P_ij / sqrt(P_ii P_jj)``: the correlation between ``x_i``
    and ``x_j`` conditional on (after linearly regressing out) all other
    ``M - 2`` processes, at lag 0. For ``EmpiricalCovariance`` this is the
    Pearson correlation of the OLS residuals of ``x_i`` and ``x_j`` on the
    remaining processes; the other estimators shrink or sparsify ``P`` first.

    Undirected and signed, in ``[-1, 1]``, with the opposite sign and the
    same zero pattern as the off-diagonal of ``prec_<estimator>``.

    ``estimator`` is any ``sklearn.covariance`` estimator that ``Precision``
    accepts, and ``squared=True`` squares the result (unsigned). The
    covariance fit is shared with ``Covariance`` and ``Precision``
    (``data.covariance[estimator]``).
    """

    name = "Partial correlation"
    identifier = "pcorr"

    def __init__(self, estimator="EmpiricalCovariance", squared=False):
        super().__init__(
            kind="partial_correlation", squared=squared, estimator=estimator
        )

    def _matrix(self, mycov):
        prec = mycov.precision_
        with np.errstate(invalid="ignore", divide="ignore"):
            # A non-positive diagonal (a singular fit) gives NaN, not a value.
            d = np.sqrt(np.where(np.diag(prec) > 0, np.diag(prec), np.nan))
            return -prec / np.outer(d, d)

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        # Conditions on every other process, so needs the whole (cached) fit.
        return self.multivariate(data)[i, j]


class CrossCorrelation(Undirected, Signed):
    """Sample cross-correlation over lags in [-T//4, +T//4].

    ``r(l) = sum_t x[t+l] y[t] / (T sd_x sd_y)`` on the demeaned series (the
    biased estimator, so ``r(0)`` is Pearson's r). ``statistic`` reduces the
    retained lags: ``max`` is the largest signed value, ``mean`` their mean.

    ``sigonly`` is a historical misnomer kept for compatibility. It is a
    pointwise amplitude threshold -- keep the lags with
    ``|r(l)| > 1.96/sqrt(T)`` -- and not a significance test: the band is not
    inflated for the series' own autocorrelation, and it is not corrected for
    being applied at every lag in the window. When no lag clears it the
    statistic is 0.
    """

    name = "Cross correlation"
    labels = ["basic", "linear", "undirected", "temporal"]

    def __init__(self, squared=False, statistic="max", sigonly=True):
        self.identifier = "xcorr"
        self._squared = squared
        self._statistic = statistic
        self._sigonly = sigonly

        if self._squared:
            self.issigned = lambda: False
            self.identifier = self.identifier + "-sq"
            self.labels = CrossCorrelation.labels + ["unsigned"]
        else:
            self.labels = CrossCorrelation.labels + ["signed"]
        self.identifier += f"_{statistic}_sig-{sigonly}"

    # Lags are examined out to +/- T // 4. Beyond a quarter of the record the
    # sample cross-correlation is estimated from fewer than 3T/4 overlapping
    # points and its variance grows without bound under the biased
    # normalisation used here.
    _MAX_LAG_FRACTION = 4

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        T = data.n_observations
        try:
            r_ij = data.xcorr[(i, j)]
        except (KeyError, AttributeError):
            x, y = data.to_numpy(squeeze=True)[[i, j]]

            # Demeaned, and divided by T (the biased estimator) rather than
            # T - |l|: the result is a correlation, so it stays in [-1, 1], the
            # zero lag equals Pearson's r, and `max` is not dominated by the
            # high-variance tail the unbiased normalisation gives at large
            # lags. statsmodels' `ccf(adjusted=False)` and matplotlib's `xcorr`
            # make the same choice.
            x = x - x.mean()
            y = y - y.mean()
            scale = T * np.sqrt(np.mean(x ** 2) * np.mean(y ** 2))

            # Force FFT method: O(N log N) vs O(N^2) direct for short signals.
            r_full = signal.correlate(x, y, "full", method="fft") / scale

            # correlate(x, y, "full")[T - 1 + l] == sum_t x[t + l] * y[t], so
            # the zero lag sits at T - 1 and the window is centred there. An
            # off-centre window would make `max` depend on the order of the
            # pair, for a measure that is undirected.
            lag_max = T // self._MAX_LAG_FRACTION
            r_ij = r_full[T - 1 - lag_max: T + lag_max]

            try:
                data.xcorr[(i, j)] = r_ij
            except AttributeError:
                data.xcorr = {(i, j): r_ij}
            # r_yx(l) == r_xy(-l), so the opposite orientation is the reversed
            # sequence, not the same one.
            data.xcorr[(j, i)] = r_ij[::-1]

        # `sigonly` is a pointwise amplitude threshold, not an inferential
        # test. The name is historical and kept for compatibility: it keeps
        # only the lags whose sample cross-correlation exceeds
        # 1.96/sqrt(T) in magnitude. That cut is the two-sided 5% band for a
        # *single* correlation between two independent white series, and
        # neither of the two things that would make it a significance test is
        # done here -- the band is not inflated for the series'
        # autocorrelation, and it is
        # not corrected for having been applied at every one of the ~T/2 lags
        # in the window. Read it as "drop the small lags", not as "these lags
        # are significant".
        if getattr(self, "_sigonly", False):
            # Scaled by the sample size T, not by the width of the lag window.
            threshold = 1.96 / np.sqrt(T)
            above = np.abs(r_ij) > threshold
            if not above.any():
                # Nothing clears the cut, so the thresholded association is
                # zero, not the largest of ~T/2 null sample correlations.
                return 0.0
            # Selecting *the lags above the cut*, rather than the contiguous
            # run of them around lag zero: a set of lags is invariant under the
            # l -> -l reversal, so the two orientations of this undirected SPI
            # agree even when the peak is off lag 0.
            r_ij = r_ij[above]

        if self._squared:
            r_ij = r_ij ** 2
        if self._statistic == "max":
            return float(np.max(r_ij))
        elif self._statistic == "mean":
            return float(np.mean(r_ij))
        else:
            raise TypeError(f"Unknown statistic: {self._statistic}")


class SpearmanR(Undirected, Signed):

    name = "Spearman's correlation coefficient"
    identifier = "spearmanr"
    labels = ["basic", "unordered", "rank", "linear", "undirected"]

    def __init__(self, squared=False):
        self._squared = squared
        if squared:
            self.issigned = lambda: False
            self.identifier = self.identifier + "-sq"
            self.labels = self.labels + ["unsigned"]
        else:
            self.labels = self.labels + ["signed"]

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        x, y = data.to_numpy()[[i, j]]
        if self._squared:
            return stats.spearmanr(x, y).correlation ** 2
        else:
            return stats.spearmanr(x, y).correlation

    @parse_multivariate
    def multivariate(self, data):
        """Vectorized: Spearman's rho on the full (M, T) matrix at once."""
        Z = data.to_numpy(squeeze=True)  # (M, T)
        rho = _spearman_matrix(Z)
        if self._squared:
            rho = rho ** 2
        np.fill_diagonal(rho, np.nan)
        return rho


class KendallTau(Undirected, Signed):

    name = "Kendall's tau"
    identifier = "kendalltau"
    labels = ["basic", "unordered", "rank", "linear", "undirected"]

    def __init__(self, squared=False):
        self._squared = squared
        if squared:
            self.issigned = lambda: False
            self.identifier = self.identifier + "-sq"
            self.labels = self.labels + ["unsigned"]
        else:
            self.labels = self.labels + ["signed"]

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        x, y = data.to_numpy()[[i, j]]
        if self._squared:
            return stats.kendalltau(x, y).correlation ** 2
        else:
            return stats.kendalltau(x, y).correlation

    @parse_multivariate
    def multivariate(self, data):
        """Vectorized: pandas .corr(method='kendall') on (T, M) DataFrame."""
        Z = data.to_numpy(squeeze=True)  # (M, T)
        df = pd.DataFrame(Z.T)
        tau = np.asarray(df.corr(method="kendall").values, dtype=float).copy()
        if self._squared:
            tau = tau ** 2
        np.fill_diagonal(tau, np.nan)
        return tau


# Seed of the permutation that breaks ties in ChatterjeeXi's source reproducibly.
_XI_TIE_SEED = 0


def _xi_source_order(x):
    """Indices that sort ``x``, with ties broken by a fixed seeded permutation."""
    perm = np.random.default_rng(_XI_TIE_SEED).permutation(x.size)
    return perm[np.argsort(x[perm], kind="stable")]


def _xi_target_terms(y):
    """``(r, denominator)`` for the target ``y``.

    ``r[k] = #{j : y_j <= y_k}`` and the denominator is
    ``2 * sum_k l_k (n - l_k) / n`` with ``l_k = #{j : y_j >= y_k}``.
    The denominator is 0 (xi undefined) when ``y`` is constant.
    """
    n = y.size
    r = stats.rankdata(y, method="max").astype(np.int64)
    l = n - stats.rankdata(y, method="min").astype(np.int64) + 1
    return r, 2 * np.sum(l * (n - l)) / n


class ChatterjeeXi(Directed, Unsigned):
    """Chatterjee's rank correlation xi (Chatterjee 2021, JASA 116, 2009-2022).

    For a source ``X`` and a target ``Y`` of length ``n``, sort the pairs by
    ``X`` and let ``Y_(1), ..., Y_(n)`` be the targets in that order. With
    ``r_k = #{j : Y_j <= Y_(k)}`` and ``l_k = #{j : Y_j >= Y_(k)}``,
    ``xi_n(X, Y) = 1 - n sum_k |r_{k+1} - r_k| / (2 sum_k l_k (n - l_k))``,
    the first sum over ``k = 1..n-1`` and the second over ``k = 1..n``. This
    is the general form, valid with ties in ``Y``.

    It uses ranks only. As ``n -> infinity`` it tends to 0 if and only if
    ``Y`` is independent of ``X``, and to 1 if and only if ``Y`` is a
    measurable function of ``X``, so it also detects non-monotone dependence
    such as ``Y = X^2``. Under independence it can be slightly negative.

    Asymmetric and unsigned: ``A[i, j] = xi(X = x_i, Y = x_j)``, row = source,
    measures how nearly the column process is a function of the row process.

    ``tau`` is the lag in samples (default 0). For ``tau > 0`` the SPI is
    ``xi(x_i[:-tau], x_j[tau:])``: how much ``x_j[t]`` is a function of
    ``x_i[t - tau]``.

    Chatterjee breaks ties in ``X`` at random; here they are broken by a fixed
    seeded permutation, so values are reproducible. The value is NaN when the
    target is constant, when either series has a non-finite sample, or when
    fewer than two samples remain after the lag.
    """

    name = "Chatterjee's xi correlation"
    identifier = "xi"
    labels = ["basic", "rank", "nonlinear", "directed", "unsigned"]

    def __init__(self, tau=0):
        self._tau = require_int("tau", tau, minimum=0)
        self.identifier = f"xi_tau-{self._tau}"
        self.labels = ChatterjeeXi.labels + (
            ["unordered"] if self._tau == 0 else ["temporal"]
        )

    @staticmethod
    def _xi(order, r, denom):
        if not denom > 0:
            return np.nan
        steps = np.abs(np.diff(r[order])).sum()
        return float(1.0 - steps / denom)

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        z = data.to_numpy(squeeze=True)
        if z.shape[-1] - self._tau < 2:
            return np.nan
        tau = self._tau
        x, y = (z[i], z[j]) if tau == 0 else (z[i, :-tau], z[j, tau:])
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            return np.nan
        r, denom = _xi_target_terms(y)
        return self._xi(_xi_source_order(x), r, denom)

    @parse_multivariate
    def multivariate(self, data):
        """Sort each source once and rank each target once."""
        Z = data.to_numpy(squeeze=True)  # (M, T)
        M, T = Z.shape
        A = np.full((M, M), np.nan)
        if T - self._tau < 2:
            return A
        tau = self._tau
        X, Y = (Z, Z) if tau == 0 else (Z[:, :-tau], Z[:, tau:])
        finite_x = np.all(np.isfinite(X), axis=1)
        finite_y = np.all(np.isfinite(Y), axis=1)
        orders = [_xi_source_order(X[i]) if finite_x[i] else None for i in range(M)]
        R = np.zeros(Y.shape, dtype=np.int64)
        denoms = np.full(M, np.nan)
        for j in range(M):
            if finite_y[j]:
                R[j], denoms[j] = _xi_target_terms(Y[j])
        valid_y = finite_y & (denoms > 0)
        for i in range(M):
            if orders[i] is None:
                continue
            # steps[j] = sum_k |R[j, order_k+1] - R[j, order_k]| for every target
            steps = np.abs(np.diff(R[:, orders[i]], axis=1)).sum(axis=1)
            A[i, valid_y] = 1.0 - steps[valid_y] / denoms[valid_y]
        np.fill_diagonal(A, np.nan)
        return A


class LaggedCorrelation(Undirected, Signed):
    """Lagged correlation SPI.

    Computes symmetric lagged correlation: 0.5 * (corr(x[tau:], y[:-tau]) + corr(y[tau:], x[:-tau])).
    Supports Pearson, Spearman, and Kendall estimators.
    """
    name = "Lagged correlation"
    labels = ["basic", "linear", "undirected", "temporal"]

    def __init__(self, estimator="pearson", tau=None, max_tau=None, squared=False):
        est = str(estimator).lower()
        if est not in {"pearson", "spearman", "kendall"}:
            raise ValueError(f"Unknown estimator: {estimator}")
        if max_tau is not None:
            raise ValueError("max_tau is only supported in config expansion; use tau.")
        if tau is None:
            raise ValueError("LaggedCorrelation requires tau.")

        self._estimator = est
        self._squared = bool(squared)
        if self._squared:
            self.issigned = lambda: False
            self.labels = LaggedCorrelation.labels + ["unsigned"]
            suffix = "-sq"
        else:
            self.labels = LaggedCorrelation.labels + ["signed"]
            suffix = ""
        self._tau = require_int("tau", tau, minimum=0)
        self.identifier = f"corr_{est}_tau-{self._tau}{suffix}"

    def _corr(self, x, y):
        x = np.asarray(x).reshape(-1)
        y = np.asarray(y).reshape(-1)
        if x.size < 2 or y.size < 2:
            return np.nan
        if self._estimator == "pearson":
            return stats.pearsonr(x, y).correlation
        if self._estimator == "spearman":
            return stats.spearmanr(x, y).correlation
        if self._estimator == "kendall":
            return stats.kendalltau(x, y).correlation
        raise ValueError(f"Unknown estimator: {self._estimator}")

    def _lagged_corr(self, x, y, tau):
        if tau == 0:
            return self._corr(x, y)
        if tau >= x.size:
            return np.nan
        return self._corr(x[tau:], y[:-tau])

    def _symmetric_lagged_corr(self, x, y, tau):
        # NaN if either direction is undefined, as in `multivariate`.
        forward = self._lagged_corr(x, y, tau)
        backward = self._lagged_corr(y, x, tau)
        return 0.5 * (forward + backward)

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        x, y = data.to_numpy()[[i, j]]
        value = self._symmetric_lagged_corr(x, y, self._tau)
        return value**2 if self._squared else value

    @parse_multivariate
    def multivariate(self, data):
        """Vectorized multivariate lagged correlation."""
        Z = data.to_numpy(squeeze=True)  # (M, T)
        M, T = Z.shape
        tau = self._tau

        if tau == 0 or tau >= T:
            if tau >= T:
                return np.full((M, M), np.nan)
            if self._estimator == "pearson":
                C = np.corrcoef(Z)
            elif self._estimator == "spearman":
                C = _spearman_matrix(Z)
            elif self._estimator == "kendall":
                C = np.asarray(
                    pd.DataFrame(Z.T).corr(method="kendall").values, dtype=float
                ).copy()
            else:
                raise ValueError(f"Unknown estimator: {self._estimator}")
            if self._squared:
                C = C ** 2
            np.fill_diagonal(C, np.nan)
            return C

        Z_lead = Z[:, tau:]
        Z_lag = Z[:, :-tau]

        if self._estimator == "pearson":
            stacked = np.vstack([Z_lead, Z_lag])
            C_full = np.corrcoef(stacked)
            forward = C_full[:M, M:]
            backward = C_full[M:, :M]
        elif self._estimator == "spearman":
            stacked = np.vstack([Z_lead, Z_lag])
            rho = _spearman_matrix(stacked)
            forward = rho[:M, M:]
            backward = rho[M:, :M]
        elif self._estimator == "kendall":
            # Only the lead x lag block is needed; `backward` is its transpose.
            forward = np.full((M, M), np.nan)
            for a in range(M):
                for b in range(M):
                    if a != b:
                        forward[a, b] = stats.kendalltau(Z_lead[a], Z_lag[b])[0]
            backward = forward.T
        else:
            raise ValueError(f"Unknown estimator: {self._estimator}")

        C = 0.5 * (forward + backward)
        if self._squared:
            C = C ** 2
        np.fill_diagonal(C, np.nan)
        return C
