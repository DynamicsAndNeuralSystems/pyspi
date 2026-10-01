import warnings
import sklearn.covariance as cov
from scipy import stats, signal
import numpy as np
import pandas as pd

from pyspi.base import Undirected, Signed, parse_bivariate, parse_multivariate
from pyspi.utils import require_int


class Estimators(Undirected, Signed):
    """Base class for (functional) connectivity-based statistics

    Information on covariance estimators at: https://scikit-learn.org/stable/modules/covariance.html
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

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mycov = getattr(cov, self._estimator)().fit(z)
            try:
                data.covariance[self._estimator] = mycov
            except AttributeError:
                data.covariance = {self._estimator: mycov}
        return mycov

    @parse_multivariate
    def multivariate(self, data):
        mycov = self._from_cache(data)
        matrix = getattr(mycov, self._kind + "_")
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


class CrossCorrelation(Undirected, Signed):

    name = "Cross correlation"
    labels = ["basic", "linear", "undirected", "temporal"]

    """Sample cross-correlation over lags in [-T//4, +T//4].

    ``sigonly`` is a historical misnomer kept for compatibility. It is a
    pointwise amplitude threshold -- keep the lags with
    ``|r(l)| > 1.96/sqrt(T)`` -- and not a significance test: the band is not
    inflated for the series' own autocorrelation, and it is not corrected for
    being applied at every lag in the window. When no lag clears it the
    statistic is 0.
    """

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
    # normalisation used here; the quarter cut is the convention pyspi shipped
    # and is kept.
    _MAX_LAG_FRACTION = 4

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        T = data.n_observations
        try:
            r_ij = data.xcorr[(i, j)]
        except (KeyError, AttributeError):
            x, y = data.to_numpy(squeeze=True)[[i, j]]

            # Demeaned, and normalised by T rather than T-1. Three separate
            # problems with the previous line
            # `correlate(x, y) / x.std() / y.std() / (T - 1)`:
            #
            #   * `signal.correlate` was given the *raw* series while the
            #     divisor used `std()`, which demeans. The two halves of the
            #     ratio therefore described different quantities: on
            #     `arange(10)` against itself with zscore=False it returned
            #     3.8384 for a correlation.
            #   * The lag-l sum has T - |l| terms, not T - 1. Dividing by T - 1
            #     is neither the biased (T) nor the unbiased (T - |l|)
            #     normalisation, and it put the zero lag of a series with
            #     itself at T/(T-1): exactly 1.1111 for T = 10.
            #   * `std()` is the sample standard deviation (ddof=0 in numpy,
            #     so 1/T) while the divisor was T-1 -- mismatched conventions
            #     in the same expression.
            #
            # Biased (divide by T) rather than unbiased (divide by T - |l|):
            # the result is a *correlation*, so it must stay in [-1, 1], the
            # zero lag must equal Pearson's r, and the `max` statistic must not
            # be dominated by the high-variance tail that the unbiased
            # normalisation produces at large lags. This is the same choice
            # statsmodels' `ccf(adjusted=False)` and matplotlib's `xcorr` make.
            x = x - x.mean()
            y = y - y.mean()
            scale = T * np.sqrt(np.mean(x ** 2) * np.mean(y ** 2))

            # Force FFT method: O(N log N) vs O(N^2) direct for short signals.
            r_full = signal.correlate(x, y, "full", method="fft") / scale

            # correlate(x, y, "full")[T - 1 + l] == sum_t x[t + l] * y[t], so
            # the zero lag sits at T - 1 and the window must be centred there.
            # `r_full[T - T//4 : T + T//4]` was centred on T, i.e. on lag +1:
            # the lag window was asymmetric, which makes `max` over it depend
            # on the order of the pair for a measure declared undirected.
            lag_max = T // self._MAX_LAG_FRACTION
            r_ij = r_full[T - 1 - lag_max: T + lag_max]

            try:
                data.xcorr[(i, j)] = r_ij
            except AttributeError:
                data.xcorr = {(i, j): r_ij}
            # r_yx(l) == r_xy(-l), so the opposite orientation is the reversed
            # sequence, not the same one. Aliasing the two made `bivariate(j,i)`
            # return the lag profile of (i,j).
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
            # The previous threshold was `1.96/sqrt(len(r_ij)//2)` -- the
            # half-width of the lag *window*, T//4 -- so it was twice too wide
            # and moved with the lag cut rather than with the sample size.
            threshold = 1.96 / np.sqrt(T)
            above = np.abs(r_ij) > threshold
            if not above.any():
                # Nothing clears the cut, so the thresholded association is
                # zero. Falling back to the unfiltered window instead reported
                # the largest of ~T/2 sample correlations under the null: on
                # two independent length-400 series that is about 0.135 for the
                # `max` statistic, which is the opposite of what a threshold is
                # supposed to do at the null.
                return 0.0
            # Selecting *the lags above the cut*, rather than the contiguous
            # run of them around lag zero. The previous code walked outwards
            # from the centre, which is only right when the peak is at lag 0:
            # for a pair where i leads j by one sample, r(0) is already below
            # the cut, so the lobe extended one way and not the other and the
            # two orientations of an SPI declared *undirected* disagreed --
            # measured 0.9957 against -0.0202 on a lag-1 pair. A set of lags is
            # invariant under the l -> -l reversal; a one-sided run is not.
            # (Its slice was independently wrong: `r_ij[N - fzr : N + fzf]`
            # mirrored `fzr`, which was already an absolute index.)
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
        """Vectorized: scipy.stats.spearmanr on full (M, T) matrix at once."""
        Z = data.to_numpy(squeeze=True)  # (M, T)
        rho, _ = stats.spearmanr(Z, axis=1)
        if Z.shape[0] == 2:
            rho = np.array([[1.0, rho], [rho, 1.0]])
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
        forward = self._lagged_corr(x, y, tau)
        backward = self._lagged_corr(y, x, tau)
        if np.isnan(forward):
            return backward
        if np.isnan(backward):
            return forward
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
                C, _ = stats.spearmanr(Z, axis=1)
                if M == 2:
                    C = np.array([[1.0, C], [C, 1.0]])
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
            rho, _ = stats.spearmanr(stacked, axis=1)
            if stacked.shape[0] == 2:
                rho = np.array([[1.0, rho], [rho, 1.0]])
            forward = rho[:M, M:]
            backward = rho[M:, :M]
        elif self._estimator == "kendall":
            stacked = np.vstack([Z_lead, Z_lag])
            df = pd.DataFrame(stacked.T)
            C_full = df.corr(method="kendall").values
            forward = C_full[:M, M:]
            backward = C_full[M:, :M]
        else:
            raise ValueError(f"Unknown estimator: {self._estimator}")

        C = 0.5 * (forward + backward)
        if self._squared:
            C = C ** 2
        np.fill_diagonal(C, np.nan)
        return C
